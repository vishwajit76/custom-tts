import json
import sys

import numpy as np
import soundfile as sf

from training import prepare_dataset as pd

SR = pd.SR


def tone(seconds: float, amp: float = 0.3, seed: int = 0) -> np.ndarray:
    """Modulated tone with a quiet 0.3 s noise-floor lead-in/out (a gate-friendly stand-in for a clean recording)."""
    t = np.arange(int(SR * seconds)) / SR
    x = amp * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    x += np.random.default_rng(seed).normal(0, amp / 1000, len(t))
    x[: int(0.3 * SR)] *= 0.01 if seconds > 1 else 1
    x[-int(0.3 * SR):] *= 0.01 if seconds > 1 else 1
    return x.astype(np.float32)


def test_denoise_and_loudness():
    rng = np.random.default_rng(0)
    clean = np.concatenate([np.zeros(SR), tone(2)])
    noisy = clean + rng.normal(0, 0.02, len(clean)).astype(np.float32)
    out = pd.denoise(noisy)
    assert np.std(out[: SR // 2]) < 0.5 * np.std(noisy[: SR // 2])  # noise-only lead-in got quieter
    norm = pd.loudness_normalize(tone(1, 0.01))
    assert abs(20 * np.log10(np.sqrt(np.mean(norm**2))) + 20) < 0.5 and np.abs(norm).max() <= 0.95


def test_prepare_end_to_end(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "out"
    (raw / "asha").mkdir(parents=True)
    sf.write(raw / "good.wav", tone(2.5), SR)
    (raw / "good.txt").write_text("नमस्ते, आपका दिन शुभ हो।", "utf-8")
    sf.write(raw / "asha" / "a1.wav", tone(3), SR)
    (raw / "asha" / "metadata.csv").write_text("a1|आपकी EMI ₹500 है।\n", "utf-8")
    sf.write(raw / "short.wav", tone(0.4), SR)
    (raw / "short.txt").write_text("जी", "utf-8")
    sf.write(raw / "mismatch.wav", tone(1.5), SR)
    (raw / "mismatch.txt").write_text("यह बहुत लंबा वाक्य है " * 20, "utf-8")
    sf.write(raw / "clipped.wav", np.ones(SR * 2, np.float32), SR)
    (raw / "clipped.txt").write_text("नमस्ते", "utf-8")
    sf.write(raw / "untranscribed.wav", tone(2), SR)

    rights = tmp_path / "rights.jsonl"
    rights.write_text(json.dumps({
        "rights_id": "r1", "source": "synthetic test", "licence": "test", "consent_record_id": "c1",
        "speaker_authorization": True, "speaker_ids": ["main", "asha"], "permitted_uses": ["tts_training"],
        "vendor_generated": False}) + "\n", "utf-8")
    monkeypatch.setattr(sys, "argv", ["prep", "--input", str(raw), "--output", str(out), "--speaker", "main",
                                      "--rights", str(rights)])
    pd.main()
    report = json.loads((out / "report.json").read_text("utf-8"))
    rows = (out / "metadata.csv").read_text("utf-8").strip().splitlines() + [
        r for r in (out / "test.csv").read_text("utf-8").splitlines() if r]
    assert report["clips"] == 2 and report["multi_speaker"] and report["rejected"] == 4
    assert {r.split("|")[1] for r in rows} == {"main", "asha"}
    assert any("पाँच सौ रुपये" in r for r in rows)  # transcripts normalized like inference
    reasons = " ".join(r["reason"] for r in report["rejections"])
    for word in ("too_short", "speech_rate_high", "clipping", "no_transcript"):
        assert word in reasons
    assert report["rejected_by_reason"]["no_transcript"] == 1
    rep = json.loads((out / "dataset_report.json").read_text("utf-8"))
    assert rep["clips"] == {"total": 6, "accepted": 2, "rejected": 4} and set(rep["by_speaker"]) == {"main", "asha"}
    assert (out / "dataset_report.md").exists() and len((out / "rejected.jsonl").read_text("utf-8").splitlines()) == 4
    for r in rows:
        data, sr = sf.read(out / "wavs" / r.split("|")[0])
        assert sr == SR and data.ndim == 1


def test_prepare_refuses_without_rights(tmp_path, monkeypatch):
    import pytest

    raw = tmp_path / "raw"
    raw.mkdir()
    sf.write(raw / "a.wav", tone(2.5), SR)
    (raw / "a.txt").write_text("नमस्ते, आपका दिन शुभ हो।", "utf-8")
    monkeypatch.setattr(sys, "argv", ["prep", "--input", str(raw), "--output", str(tmp_path / "o")])
    with pytest.raises(SystemExit) as e:
        pd.main()
    assert "refusing" in str(e.value)
    assert not (tmp_path / "o" / "metadata.csv").exists()


def test_piper_text_is_safe_for_pipers_csv_reader():
    import csv
    import io

    from training.prepare_dataset import piper_text

    t = piper_text('"कोई तो जीतेगा ही|" मन में सोचा।')
    assert t == "कोई तो जीतेगा ही। मन में सोचा।"
    rows = list(csv.reader(io.StringIO(f"a.wav|f|{piper_text(chr(34) + 'x')}\nb.wav|f|y\n"), delimiter="|"))
    assert rows == [["a.wav", "f", "x"], ["b.wav", "f", "y"]]
