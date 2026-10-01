"""PLUMBING TEST ONLY. Synthetic sine/noise wavs and a fake manifest exercise
manifest -> rights check -> split -> audio report. This is NOT evidence of voice quality or model performance."""
import json
import subprocess
import sys
import warnings

import numpy as np
import pytest
import soundfile as sf

from training import annotate, audio_report, data_rights, manifest, split

SR = 22050


def wav(kind: str, seconds=2.0, seed=0):
    t = np.arange(int(SR * seconds)) / SR
    if kind == "noise":
        return (np.random.default_rng(seed).normal(0, 0.2, len(t))).astype(np.float32)
    if kind == "clip":
        return np.clip(3 * np.sin(2 * np.pi * 200 * t), -1, 1).astype(np.float32)
    f = 150 + 40 * seed
    return (0.3 * np.sin(2 * np.pi * f * t) * (0.5 + 0.5 * (np.sin(2 * np.pi * 2 * t) > -0.2))).astype(np.float32)


@pytest.fixture
def ds(tmp_path):
    rows = []
    for spk in ("s1", "s2", "s3"):
        for i in range(12):
            name = f"{spk}_{i}.wav"
            sf.write(tmp_path / name, wav("tone", seed=i + 12 * int(spk[1])), SR)
            rows.append({"audio": name, "text": f"वाक्य {spk} संख्या {i}", "speaker_id": spk, "language": "hi",
                         "emotion": "happy" if i == 0 else "", "style": "calm" if i == 1 else "",
                         "label_source": "human_verified" if i == 0 else ("model_guess" if i == 1 else ""),
                         "rights_id": "r-" + spk})
    sf.write(tmp_path / "bad.wav", wav("clip"), SR)
    rows.append({"audio": "bad.wav", "text": "बुरा", "speaker_id": "s1", "language": "hi", "rights_id": "r-s1"})
    # duplicate transcript across speakers (leakage bait)
    rows[13]["text"] = rows[2]["text"]  # s2 clip duplicates an s1 transcript
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    return tmp_path


def rights_file(tmp_path, **over):
    ents = [{"rights_id": f"r-{s}", "source": "synthetic", "licence": "test", "consent_record_id": "c",
             "speaker_authorization": True, "speaker_ids": [s], "permitted_uses": ["tts_training"],
             "vendor_generated": False} for s in ("s1", "s2", "s3")]
    ents[0].update(over)
    p = tmp_path / "rights.jsonl"
    p.write_text("".join(json.dumps(e) + "\n" for e in ents), "utf-8")
    return p


def test_labels_only_human_verified(ds):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rows = manifest.load_manifest(ds / "manifest.jsonl")
    by = {(r["speaker_id"], r["text"]): r for r in rows}
    assert by[("s1", "वाक्य s1 संख्या 0")]["emotion"] == "happy"
    assert by[("s1", "वाक्य s1 संख्या 1")]["style"] is None  # unverified label dropped, not kept
    assert any("dropping unverified" in str(x.message) for x in w)


def test_rights_enforced(ds):
    rows = manifest.load_manifest(ds / "manifest.jsonl")
    data_rights.enforce(rows, rights_file(ds))
    for over in ({"speaker_authorization": False}, {"permitted_uses": ["research"]},
                 {"vendor_generated": True}, {"vendor_generated": True, "vendor_generation_permission": False}):
        with pytest.raises(data_rights.RightsError):
            data_rights.enforce(rows, rights_file(ds, **over))
    data_rights.enforce(rows, rights_file(ds, vendor_generated=True, vendor_generation_permission=True))
    with pytest.raises(data_rights.RightsError):
        data_rights.enforce(rows, None)
    partial = [e for e in data_rights.load_rights(rights_file(ds)) if e["rights_id"] != "r-s3"]
    assert data_rights.check_rows(rows, partial)


def _prep(ds):
    rows = manifest.load_manifest(ds / "manifest.jsonl")
    for r in rows:
        r["audio_sha256"] = manifest.sha256_file(r["audio"])
    return rows


def test_split_deterministic_no_leakage(ds):
    rows = _prep(ds)
    a, _ = split.make_splits(rows, seed=7, val_fraction=0.15, test_fraction=0.15)
    b, _ = split.make_splits(rows, seed=7, val_fraction=0.15, test_fraction=0.15)
    assert {k: [r["id"] for r in v] for k, v in a.items()} == {k: [r["id"] for r in v] for k, v in b.items()}
    assert a["test"] and a["val"] and a["train"]
    texts = {k: {split._key(r["text"]) for r in v} for k, v in a.items()}
    assert not (texts["train"] & texts["test"]) and not (texts["train"] & texts["val"]) and not (texts["val"] & texts["test"])
    c, _ = split.make_splits(rows, seed=8, val_fraction=0.15, test_fraction=0.15)
    assert [r["id"] for r in c["test"]] != [r["id"] for r in a["test"]]


def test_speaker_disjoint_and_heldout_guard(ds):
    rows = _prep(ds)
    s, dropped = split.make_splits(rows, seed=1, test_fraction=0.3, speaker_disjoint_test=True)
    spk = {k: {r["speaker_id"] for r in v} for k, v in s.items()}
    assert spk["test"] and not (spk["test"] & spk["train"]) and not (spk["test"] & spk["val"])
    tk = {split._key(r["text"]) for r in s["test"]}
    assert not any(split._key(r["text"]) in tk for r in s["train"] + s["val"])
    split.write_splits(s, ds / "splits", dropped)
    split.assert_not_heldout(ds / "splits" / "train.jsonl")
    with pytest.raises(split.HeldOutError):
        split.assert_not_heldout(ds / "splits" / "test.jsonl")


def test_train_refuses_test_set(ds):
    (ds / "test.csv").write_text("a.wav|x\n", "utf-8")
    with pytest.raises(split.HeldOutError):
        split.assert_not_heldout(ds / "test.csv")


def test_audio_report(ds):
    rows = [r for r in manifest.load_manifest(ds / "manifest.jsonl")]
    rep = audio_report.report(rows)
    assert rep["overall"]["files"] == len(rows)
    bad = next(f for f in rep["files"] if f["file"].endswith("bad.wav"))
    assert "clipping" in bad["rejects"]
    sf.write(ds / "n.wav", wav("noise"), SR)
    assert "low_snr" in audio_report.analyze(ds / "n.wav")["rejects"]
    assert {"s1", "s2", "s3"} <= set(rep["by_speaker"]) and "happy" in rep["by_emotion"]
    assert "| s1 |" in audio_report.to_markdown(rep)


def test_annotation_update(ds):
    manifest.write_jsonl(ds / "ann.jsonl", manifest.load_manifest(ds / "manifest.jsonl"))
    rows = manifest.read_rows(ds / "ann.jsonl")
    rid = rows[5]["id"]
    annotate.apply_update(rows, {"id": rid, "text": "सही पाठ", "emotion": "sad"}, "tester")
    row = next(x for x in rows if x["id"] == rid)
    assert row["text"] == "सही पाठ" and row["emotion"] == "sad"
    assert row["label_source"] == "human_verified" and row["verified_by"] == "tester"
    annotate.apply_update(rows, {"id": rid, "rejected": True, "reject_reason": "noisy"}, "tester")
    assert row["rejected"] and row["reject_reason"] == "noisy"


def test_end_to_end_cli(ds):
    """manifest -> rights -> split -> audio report via the real CLIs."""
    py = [sys.executable, "-m"]
    ok = subprocess.run(py + ["training.audio_report", "--manifest", str(ds / "manifest.jsonl"), "--out", str(ds / "rep")],
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert (ds / "rep" / "audio_report.json").exists() and (ds / "rep" / "audio_report.md").exists()
    rows = _prep(ds)
    manifest.write_jsonl(ds / "clean.jsonl", rows)
    r = subprocess.run(py + ["training.split", "--manifest", str(ds / "clean.jsonl"), "--out", str(ds / "sp"),
                             "--speaker-disjoint-test", "--test-fraction", "0.3"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (ds / "sp" / "test.jsonl.heldout").exists()
