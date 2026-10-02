"""Dataset quality gates, review sidecar, dataset report and HF parquet ingestion. Synthetic audio only (no network);
this checks the mechanics of the gates, not that they are tuned for real speech."""
import hashlib
import io
import json
import sys

import numpy as np
import pytest
import soundfile as sf

from training import audio_report, ingest_hf, prepare_dataset
from training import quality_gates as qg

SR = 22050
GOOD_TEXT = "नमस्ते आपका स्वागत है हम आपकी कैसे मदद कर सकते हैं"


def voice(seed: int, sec: float = 3.0, snr: float = 40.0, sr: int = SR) -> np.ndarray:
    """Harmonic 'speech-like' signal with a random syllable envelope, silent lead/tail and white noise at `snr` dB."""
    r = np.random.default_rng(seed)
    t = np.arange(int(sr * sec)) / sr
    f0 = r.uniform(100, 250)
    x = sum(r.uniform(0.2, 1) / h * np.sin(2 * np.pi * f0 * h * t + r.uniform(0, 6)) for h in range(1, 8))
    x *= np.interp(t, np.linspace(0, sec, int(sec * 5)), r.uniform(0.2, 1, int(sec * 5)))
    x[: int(0.4 * sr)] = 0
    x[-int(0.4 * sr):] = 0
    x *= 0.1 / np.sqrt(np.mean(x[x != 0] ** 2))
    n = r.normal(0, 1, len(t))
    n *= 0.1 / 10 ** (snr / 20) / n.std()
    return (x + n).astype(np.float32)


def rand_text(seed: int, words: int = 6) -> str:
    r = np.random.default_rng(1000 + seed)
    cons, vow = "कखगचजटडतदनपबमयरलवशसह", ["", "ा", "ि", "ी", "ु", "े", "ो"]
    return " ".join("".join(r.choice(list(cons)) + r.choice(vow) for _ in range(3)) for _ in range(words))


def record(cid: str, speaker: str, text: str, wav: np.ndarray, style: str | None = None) -> dict:
    pcm = (wav * 32767).astype(np.int16)
    return {"id": cid, "speaker": speaker, "text": text, "audio_sha": hashlib.sha1(pcm.tobytes()).hexdigest(), "fp": qg.fingerprint(wav, SR),
            "duration_s": len(wav) / SR, "m": qg.measure(wav, SR), "style": style}


def reasons_for(wav: np.ndarray, text: str = GOOD_TEXT, **th) -> list[str]:
    return qg.clip_reasons(qg.measure(wav, SR), text, qg.Thresholds(**th))


# ---- per-clip gates --------------------------------------------------------------------------------------------

def test_clean_clip_passes_and_snr_tracks_truth():
    assert reasons_for(voice(1)) == []
    for snr in (12, 20, 30):
        est = qg.measure(voice(2, snr=snr), SR)["snr_db"]
        assert abs(est - snr) < 3, (snr, est)
    assert "low_snr" in reasons_for(voice(3, snr=6))


def test_loudness_reference_and_levels():
    t = np.arange(SR * 3) / SR
    sine = (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)  # BS.1770: 0 dBFS 997 Hz sine = -3.01 LUFS
    assert abs(qg.lufs(sine, SR) - (-23.01)) < 0.5
    quiet = voice(4) * 0.002
    assert {"rms_low", "loudness_low"} <= set(reasons_for(quiet, min_snr_db=0))
    assert {"rms_high", "loudness_high"} <= set(reasons_for(np.clip(voice(4) * 12, -0.95, 0.95), max_clip_frac=1.0, max_clip_run=10**6))


def test_clipping_fraction_and_run_length():
    assert "clipping" in reasons_for(np.clip(voice(5) * 30, -1, 1))
    spike = voice(5).copy()
    spike[30000:30010] = 1.0  # 10 consecutive full-scale samples: tiny fraction, long run
    m = qg.measure(spike, SR)
    assert m["clip_frac"] < 0.001 and m["clip_run"] >= 10 and "clipping" in qg.clip_reasons(m, GOOD_TEXT, qg.Thresholds())


def test_silence_and_speech_ratio():
    x = np.random.default_rng(0).normal(0, 0.001, SR * 6).astype(np.float32)
    x[SR * 2: int(SR * 2.4)] += (0.2 * np.sin(2 * np.pi * 300 * np.arange(int(SR * 0.4)) / SR)).astype(np.float32)
    m = qg.measure(x, SR)
    assert m["silence_ratio"] > 0.8 and abs(m["speech_ratio"] + m["silence_ratio"] - 1) < 1e-9
    assert {"silence_ratio", "too_little_speech"} <= set(qg.clip_reasons(m, GOOD_TEXT, qg.Thresholds(min_snr_db=0)))


def test_duration_and_speech_rate():
    assert "too_short" in reasons_for(voice(7, sec=0.9), min_snr_db=0)
    assert "too_long" in reasons_for(voice(7, sec=16))
    assert "speech_rate_high" in reasons_for(voice(7), GOOD_TEXT * 20)
    assert "speech_rate_low" in reasons_for(voice(7), "हाँ")
    assert "empty_transcript" in reasons_for(voice(7), "।।")


def test_thresholds_overridable_from_cli():
    import argparse

    p = argparse.ArgumentParser()
    qg.add_args(p)
    th = qg.from_args(p.parse_args(["--min-snr-db", "35", "--allow-review", "--max-s", "20"]))
    assert th.min_snr_db == 35 and th.allow_review and th.max_s == 20 and th.max_cps == 30
    assert "low_snr" in reasons_for(voice(8, snr=25), min_snr_db=th.min_snr_db)


# ---- dataset-level gates ---------------------------------------------------------------------------------------

def test_exact_duplicates_and_speaker_leakage():
    a, b = voice(10), voice(11)
    recs = [record("a1", "s1", rand_text(1), a), record("a2", "s1", rand_text(2), a),  # same audio, same speaker
            record("a3", "s2", rand_text(3), a),  # same audio under another speaker: leakage
            record("t1", "s1", rand_text(4), voice(12)), record("t2", "s1", rand_text(4) + " ।", voice(13)),  # same text twice
            record("x1", "s1", rand_text(5), b), record("x2", "s2", rand_text(5), voice(14))]  # same text, different speakers
    why, info = qg.dataset_reasons(recs, qg.Thresholds())
    assert why["a2"] == ["duplicate_audio"] and why["a3"] == ["speaker_leak_audio"] and "a1" not in why
    assert why["t2"] == ["duplicate_text"] and "t1" not in why
    assert "x2" not in why and info["shared_text_across_speakers"] == 1  # reported, not rejected by default
    why2, _ = qg.dataset_reasons(recs, qg.Thresholds(reject_cross_speaker_text=True))
    assert why2["x2"] == ["speaker_leak_text"]


def test_near_duplicates():
    base = voice(20)
    noisy = base + np.random.default_rng(1).normal(0, 2e-4, len(base)).astype(np.float32)  # same take re-encoded
    other = [record(f"o{i}", "s1", rand_text(30 + i), voice(40 + i, sec=3.0 + 0.05 * i)) for i in range(8)]
    t1 = "आपका ऋण खाता सक्रिय कर दिया गया है कृपया प्रतीक्षा करें धन्यवाद"
    recs = [record("n1", "s1", rand_text(21), base), record("n2", "s1", rand_text(22), noisy),
            record("p1", "s1", t1, voice(60)), record("p2", "s1", t1.replace("करें", "करे"), voice(61))] + other
    why, _ = qg.dataset_reasons(recs, qg.Thresholds())
    assert why == {"n2": ["near_duplicate_audio"], "p2": ["near_duplicate_text"]}
    assert qg.dataset_reasons(recs, qg.Thresholds(no_near_duplicates=True))[0] == {}
    cross = [record("c1", "s1", rand_text(70), base), record("c2", "s2", rand_text(71), noisy)] + other
    assert qg.dataset_reasons(cross, qg.Thresholds())[0] == {"c2": ["speaker_leak_audio"]}


def test_condition_outlier_per_speaker():
    th = qg.Thresholds(no_near_duplicates=True)
    recs = [record(f"a{i}", "s1", rand_text(i), voice(100 + i, snr=40 + (i % 3))) for i in range(30)]
    recs += [record(f"b{i}", "s2", rand_text(50 + i), voice(200 + i, snr=18 + (i % 3))) for i in range(30)]  # s2 is just noisier: fine
    recs.append(record("odd", "s1", rand_text(99), voice(300, snr=22)))  # 20 dB noisier than the rest of s1
    why, _ = qg.dataset_reasons(recs, th)
    assert "condition_outlier_noise_db" in why["odd"] and set(why) == {"odd"}
    few, _ = qg.dataset_reasons(recs[:5] + [recs[-1]], th)  # too few clips to judge a speaker
    assert few == {}


def test_split_overlap():
    r = lambda i, t, h: {"id": i, "text": t, "audio_sha256": h}  # noqa: E731
    ov = qg.split_overlap({"train": [r("1", "क ख", "h1"), r("2", "ग", "h2")], "val": [r("3", "घ", "h3")], "test": [r("4", "क ख।", "h9"), r("5", "च", "h2")]})
    assert ov == {"text": 1, "audio": 1}


# ---- review sidecar -----------------------------------------------------------------------------------------------

def test_review_validation_and_filtering(tmp_path):
    f = tmp_path / "review.jsonl"
    f.write_text("\n".join(json.dumps(e) for e in [
        {"id": "a", "quality": "approved", "naturalness": 5, "pronunciation": 4, "noise": 5},
        {"id": "b", "quality": "rejected"}, {"id": "c", "quality": "review"},
        {"id": "d", "quality": "approved", "naturalness": 2, "noise": 1}]), "utf-8")
    rv = qg.load_review(f)
    th = qg.Thresholds(min_naturalness=3, min_noise=3)
    assert qg.review_reasons(["a"], rv, th) == [] and qg.review_reasons(["zzz", "a"], rv, th) == []
    assert qg.review_reasons(["b"], rv, th) == ["review_rejected"]
    assert qg.review_reasons(["c"], rv, th) == ["review_pending"]
    assert qg.review_reasons(["c"], rv, qg.Thresholds(allow_review=True)) == []
    assert qg.review_reasons(["d"], rv, th) == ["review_low_naturalness", "review_low_noise"]
    assert qg.review_reasons(["unreviewed"], rv, th) == []
    (tmp_path / "map.json").write_text(json.dumps({"x": {"quality": "approved", "naturalness": 3}}), "utf-8")
    assert qg.load_review(tmp_path / "map.json")["x"]["naturalness"] == 3
    for bad in ({"id": "q", "quality": "great"}, {"id": "q", "quality": "approved", "naturalness": 6}, {"id": "q", "quality": "approved", "naturalness": True},
                {"id": "q", "quality": "approved", "tone": 3}, {"quality": "approved"}):
        f.write_text(json.dumps([bad]), "utf-8")
        with pytest.raises(qg.ReviewError):
            qg.load_review(f)
    f.write_text(json.dumps([{"id": "q", "quality": "review"}, {"id": "q", "quality": "approved"}]), "utf-8")
    with pytest.raises(qg.ReviewError, match="duplicate"):
        qg.load_review(f)


# ---- report ---------------------------------------------------------------------------------------------------------

def test_dataset_report_fields():
    def r(i, spk, cat, dur, snr, lufs, reasons=()):
        return {"id": i, "speaker": spk, "category": cat, "duration_s": dur, "reasons": list(reasons), "m": {"snr_db": snr, "lufs": lufs}}
    recs = [r("1", "f", "calm", 3600.0, 30, -23), r("2", "f", "angry", 1800.0, 20, -20), r("3", "m", None, 1800.0, 10, -30, ["low_snr", "clipping"]),
            r("4", "m", "calm", 3600.0, 40, -25)]
    rep = audio_report.dataset_report(recs, qg.Thresholds())
    assert rep["clips"] == {"total": 4, "accepted": 3, "rejected": 1}
    assert rep["hours"] == {"total": 3.0, "accepted": 2.5, "rejected": 0.5}
    assert rep["rejection_reasons"] == {"low_snr": {"clips": 1, "hours": 0.5}, "clipping": {"clips": 1, "hours": 0.5}}
    assert rep["by_speaker"]["f"]["accepted_hours"] == 1.5 and rep["by_speaker"]["m"]["total_hours"] == 1.5
    assert rep["by_category"]["(none)"]["accepted_hours"] == 0 and rep["by_category"]["calm"]["accepted_hours"] == 2.0
    assert rep["average_duration_s"]["accepted"] == 3000.0
    assert set(rep["snr_db"]["accepted"]) == {"mean", "median", "p10", "p90"} and rep["snr_db"]["accepted"]["median"] == 30
    assert rep["loudness_lufs"]["percentiles"]["p50"] == -23 and sum(rep["loudness_lufs"]["histogram_2lu_bins"].values()) == 3
    md = audio_report.dataset_markdown(rep)
    assert "## Rejection reasons" in md and "| low_snr | 1 | 0.5 |" in md and "| calm |" in md


# ---- prepare_dataset with gates, review, workers ----------------------------------------------------------------------

def _rights(path, speakers):
    path.write_text(json.dumps({"rights_id": "r1", "source": "synthetic", "licence": "test", "consent_record_id": "c1", "speaker_authorization": True,
                                "speaker_ids": speakers, "permitted_uses": ["tts_training"], "vendor_generated": False}) + "\n", "utf-8")
    return path


def test_prepare_gates_review_and_worker_pool(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    for i in range(20):
        sf.write(raw / f"c{i:02d}.wav", voice(i), SR)
        (raw / f"c{i:02d}.txt").write_text(rand_text(i), "utf-8")
    sf.write(raw / "dup.wav", voice(0), SR)  # same recording as c00
    (raw / "dup.txt").write_text(rand_text(500), "utf-8")
    sf.write(raw / "noisy.wav", voice(77, snr=5), SR)
    (raw / "noisy.txt").write_text(rand_text(501), "utf-8")
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"c01": {"quality": "rejected"}, "c02": {"quality": "review"}, "c03.wav": {"quality": "approved", "naturalness": 1}}), "utf-8")
    argv = ["prep", "--input", str(raw), "--output", str(out), "--rights", str(_rights(tmp_path / "r.jsonl", ["speaker0"])),
            "--review", str(review), "--min-naturalness", "3", "--workers", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    prepare_dataset.main()
    rep = json.loads((out / "dataset_report.json").read_text("utf-8"))
    assert rep["clips"] == {"total": 22, "accepted": 17, "rejected": 5}
    got = {w for r in rep["rejection_reasons"] for w in [r]}
    assert {"review_rejected", "review_pending", "review_low_naturalness", "low_snr"} <= got
    assert ("duplicate_audio" in got)  # dup.wav vs c00 (identical samples after normalisation)
    assert len(list((out / "wavs").glob("*.wav"))) == 17  # rejected clips leave no wav behind
    rejected = {json.loads(line)["file"].split("/")[-1]: json.loads(line)["reasons"] for line in (out / "rejected.jsonl").read_text("utf-8").splitlines()}
    assert rejected["c01.wav"] == ["review_rejected"] and rejected["noisy.wav"][0] == "low_snr"
    # --allow-review keeps the pending clip
    monkeypatch.setattr(sys, "argv", argv + ["--allow-review", "--output", str(tmp_path / "out2")])
    prepare_dataset.main()
    assert json.loads((tmp_path / "out2" / "dataset_report.json").read_text("utf-8"))["clips"]["accepted"] == 18


def test_prepare_rejects_bad_review_file(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    sf.write(raw / "a.wav", voice(1), SR)
    (raw / "a.txt").write_text(GOOD_TEXT, "utf-8")
    bad = tmp_path / "rv.json"
    bad.write_text(json.dumps([{"id": "a", "quality": "maybe"}]), "utf-8")
    monkeypatch.setattr(sys, "argv", ["prep", "--input", str(raw), "--output", str(tmp_path / "o"), "--allow-unverified-rights", "--review", str(bad)])
    with pytest.raises(SystemExit, match="invalid --review"):
        prepare_dataset.main()


# ---- ingest_hf ------------------------------------------------------------------------------------------------------

def _wav_bytes(seed: int, sr: int = 48000) -> bytes:
    b = io.BytesIO()
    sf.write(b, voice(seed, sr=sr), sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


def _rasa_parquet(root, n=6):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rows = [{"filename": f"clip{i}.wav", "text": rand_text(i), "language": "hindi", "gender": "Female" if i < n // 2 else "Male",
             "style": ["conv", "read"][i % 2], "duration": 3.0, "wav_path": f"x/clip{i}.wav",
             "audio": {"bytes": _wav_bytes(i), "path": f"clip{i}.wav"}} for i in range(n)]
    (root / "Hindi").mkdir(parents=True)
    t = pa.Table.from_pylist(rows, schema=pa.schema([("filename", pa.string()), ("text", pa.string()), ("language", pa.string()),
                                                      ("gender", pa.string()), ("style", pa.string()), ("duration", pa.float64()),
                                                      ("wav_path", pa.string()), ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())]))]))
    pq.write_table(t, root / "Hindi" / "train-00000-of-00001.parquet", row_group_size=2)
    return rows


def _ingest(src, out, **kw):
    return ingest_hf.ingest(str(src), out, ingest_hf.PRESETS["rasa"]["map"], config="Hindi", licence="CC-BY-4.0", speaker_authorization=True, **kw)


def test_ingest_hf_rasa_layout_filters_resume_rights(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _rasa_parquet(src)
    info = _ingest(src, out, gender="female", batch_size=2)
    assert info["new_clips"] == 3 and info["speakers"] == ["rasa_hi_female"] and info["revision"] == "local"
    meta = list(__import__("csv").DictReader((out / "metadata.csv").open(encoding="utf-8")))
    assert [r["id"] for r in meta] == ["clip0", "clip1", "clip2"]
    r0 = meta[0]
    assert (r0["speaker_id"], r0["gender"], r0["style"], r0["language"], r0["license"], r0["source_row_id"], r0["rights_id"]) == \
        ("rasa_hi_female", "female", "conv", "hi", "CC-BY-4.0", "train-00000-of-00001:0", r0["rights_id"])
    assert r0["source_repo"] == str(src) and r0["source_revision"] == "local" and r0["source_duration"] == "3.0" and float(r0["duration"]) == pytest.approx(3.0, abs=0.01)
    x, sr = sf.read(out / "wavs" / "clip0.wav", dtype="float32")
    assert sr == SR and x.ndim == 1 and sf.info(out / "wavs" / "clip0.wav").subtype == "PCM_16"
    # resume: nothing new, nothing duplicated; then widen the filter and only the new clips are added
    again = _ingest(src, out, gender="female")
    assert again["new_clips"] == 0 and again["skipped_existing"] == 3
    more = _ingest(src, out)
    assert more["new_clips"] == 3 and more["clips"] == 6 and len(list((out / "wavs").glob("*.wav"))) == 6
    assert len(list(__import__("csv").DictReader((out / "metadata.csv").open(encoding="utf-8")))) == 6
    # rights entry validates with data_rights and covers both speakers
    from training import data_rights

    (e,) = data_rights.load_rights(out / "rights.jsonl")
    assert e["licence"] == "CC-BY-4.0" and e["speaker_ids"] == ["rasa_hi_female", "rasa_hi_male"] and "tts_training" in e["permitted_uses"]
    assert e["revision"] == "local" and e["attribution"] and e["source_url"]
    assert data_rights.violation(e) is None


def test_ingest_hf_style_filter_and_max_hours(tmp_path):
    src = tmp_path / "src"
    _rasa_parquet(src)
    assert _ingest(src, tmp_path / "o1", styles=["CONV"])["clips"] == 3
    capped = _ingest(src, tmp_path / "o2", max_hours=7 / 3600)  # 3 s clips: stops after 3 clips (9 s >= 7 s)
    assert capped["clips"] == 3 and capped["hours"] == pytest.approx(9 / 3600, abs=2e-4)
    with pytest.raises(SystemExit, match="nothing matched"):
        _ingest(src, tmp_path / "o3", gender="nonbinary")


def test_ingest_hf_custom_mapping_indicvoices_layout(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    (tmp_path / "src" / "data").mkdir(parents=True)
    rows = [{"text": rand_text(i), "speaker_id": f"spk{i % 2}", "gender": "female", "age_group": "18-30", "snr": 31.5, "duration": 3.0,
             "audio": {"bytes": _wav_bytes(i, sr=16000), "path": None}} for i in range(4)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "src" / "data" / "train-0.parquet")
    info = ingest_hf.ingest(str(tmp_path / "src"), tmp_path / "out", ingest_hf.PRESETS["indicvoices-r"]["map"], licence="CC-BY-4.0", split="train")
    meta = list(__import__("csv").DictReader((tmp_path / "out" / "metadata.csv").open(encoding="utf-8")))
    assert info["clips"] == 4 and {r["speaker_id"] for r in meta} == {"ivr_spk0", "ivr_spk1"}
    assert meta[0]["age_group"] == "18-30" and meta[0]["source_snr"] == "31.5"
    assert sf.info(tmp_path / "out" / "wavs" / f"{meta[0]['id']}.wav").samplerate == SR
    assert ingest_hf.parse_map(["a=b", "c={d}_x"]) == {"a": "b", "c": "{d}_x"}


def test_ingest_requires_licence_and_rights_gate_in_prepare(tmp_path, monkeypatch):
    src, out = tmp_path / "src", tmp_path / "out"
    _rasa_parquet(src)
    with pytest.raises(SystemExit, match="licence"):
        ingest_hf.ingest(str(src), out, ingest_hf.PRESETS["rasa"]["map"], config="Hindi")
    _ingest(src, out)
    # ingest -> prepare (manifest mode picks up out/rights.jsonl) -> categories, genders and speakers reach the report
    monkeypatch.setattr(sys, "argv", ["prep", "--manifest", str(out / "metadata.csv"), "--output", str(tmp_path / "prep"), "--workers", "1"])
    prepare_dataset.main()
    rep = json.loads((tmp_path / "prep" / "dataset_report.json").read_text("utf-8"))
    assert rep["clips"]["accepted"] == 6 and set(rep["by_speaker"]) == {"rasa_hi_female", "rasa_hi_male"} and set(rep["by_category"]) == {"conv", "read"}
    meta = (tmp_path / "prep" / "metadata.csv").read_text("utf-8") + (tmp_path / "prep" / "test.csv").read_text("utf-8")
    assert all(len(line.split("|")) == 3 for line in meta.splitlines())  # multi-speaker id|speaker|text
    rows = [json.loads(line) for line in (tmp_path / "prep" / "manifest.jsonl").read_text("utf-8").splitlines()]
    assert {r["gender"] for r in rows} == {"female", "male"} and all(r["source_repo"] == str(src) for r in rows)
    # no rights file -> refuses, and a speaker the rights entry does not cover -> refuses
    (out / "rights.jsonl").rename(out / "rights.bak")
    monkeypatch.setattr(sys, "argv", ["prep", "--manifest", str(out / "metadata.csv"), "--output", str(tmp_path / "p2")])
    with pytest.raises(SystemExit, match="refusing"):
        prepare_dataset.main()
    entry = json.loads((out / "rights.bak").read_text("utf-8"))
    entry["speaker_ids"] = ["rasa_hi_female"]
    (out / "rights.jsonl").write_text(json.dumps(entry) + "\n", "utf-8")
    with pytest.raises(SystemExit, match="refusing"):
        prepare_dataset.main()
    entry.update(speaker_ids=["rasa_hi_female", "rasa_hi_male"], speaker_authorization=False)  # presets assert it; a bare repo must too
    (out / "rights.jsonl").write_text(json.dumps(entry) + "\n", "utf-8")
    with pytest.raises(SystemExit, match="refusing"):
        prepare_dataset.main()
