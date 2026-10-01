import numpy as np
import pytest

from app.services import speaker_encoder as se


def voice(f0, dur=3.0, sr=16000, seed=0):
    t = np.arange(int(dur * sr)) / sr
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    x += np.random.default_rng(seed).normal(0, 0.005, x.shape)
    return (0.3 * x / np.abs(x).max()).astype(np.float32), sr


def test_deterministic_normalized_and_cached(tmp_path):
    enc = se.SpeakerEncoder("mfcc", cache_dir=tmp_path)
    w, sr = voice(120)
    a, b = enc.embed(w, sr), enc.embed(w.copy(), sr)
    assert np.array_equal(a, b) and abs(np.linalg.norm(a) - 1) < 1e-5
    assert len(list((tmp_path / "mfcc").glob("*.npy"))) == 1  # persisted by audio sha256
    fresh = se.SpeakerEncoder("mfcc", cache_dir=tmp_path).embed(w, sr)  # loaded from .npy
    assert np.array_equal(a, fresh)
    assert enc.similarity(a, a) == pytest.approx(1.0, abs=1e-6)


def test_resamples_and_similarity_orders():
    enc = se.SpeakerEncoder("mfcc")
    lo, sr = voice(110)
    lo2, _ = voice(110, seed=1)
    hi, _ = voice(240)
    assert enc.similarity(enc.embed(lo, sr), enc.embed(lo2, sr)) > enc.similarity(enc.embed(lo, sr), enc.embed(hi, sr))
    import soxr
    assert enc.embed(soxr.resample(lo, sr, 44100), 44100).shape == enc.embed(lo, sr).shape


def test_quality_checks():
    enc = se.SpeakerEncoder("mfcc")
    with pytest.raises(se.EncoderAudioError, match="short"):
        enc.embed(np.ones(8000, np.float32) * 0.1, 16000)
    with pytest.raises(se.EncoderAudioError, match="silent"):
        enc.embed(np.zeros(32000, np.float32), 16000)
    with pytest.raises(se.EncoderAudioError, match="clipped"):
        enc.embed(np.sign(voice(120)[0]), 16000)


def test_consistency_and_labels():
    enc = se.SpeakerEncoder("mfcc")
    assert not enc.neural and enc.name == "mfcc"
    embs = [enc.embed(*voice(120, seed=s)) for s in range(3)]
    c = enc.consistency(embs)
    assert len(c["scores"]) == 3 and c["calibrated"] is False
    assert enc.consistency(embs[:1])["min"] is None


def test_unknown_and_missing_neural_backend():
    with pytest.raises(ValueError):
        se.SpeakerEncoder("nope")
    if not se.available_backends()["resemblyzer"]:
        with pytest.raises(se.BackendUnavailable):
            se.SpeakerEncoder("resemblyzer").embed(*voice(120))
