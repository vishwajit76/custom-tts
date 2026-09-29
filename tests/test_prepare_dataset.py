import json
import sys

import numpy as np
import soundfile as sf

from training import prepare_dataset as pd

SR = pd.SR


def tone(seconds: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


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

    monkeypatch.setattr(sys, "argv", ["prep", "--input", str(raw), "--output", str(out), "--speaker", "main"])
    pd.main()
    report = json.loads((out / "report.json").read_text("utf-8"))
    rows = (out / "metadata.csv").read_text("utf-8").strip().splitlines() + [
        r for r in (out / "test.csv").read_text("utf-8").splitlines() if r]
    assert report["clips"] == 2 and report["multi_speaker"] and report["rejected"] == 4
    assert {r.split("|")[1] for r in rows} == {"main", "asha"}
    assert any("पाँच सौ रुपये" in r for r in rows)  # transcripts normalized like inference
    reasons = " ".join(r["reason"] for r in report["rejections"])
    for word in ("too short", "speech rate", "clipped", "no transcript"):
        assert word in reasons
    for r in rows:
        data, sr = sf.read(out / "wavs" / r.split("|")[0])
        assert sr == SR and data.ndim == 1
