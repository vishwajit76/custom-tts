import json

import numpy as np
import pytest

from bench import compare_checkpoints as cc


def test_bootstrap_is_deterministic_and_covers_the_mean():
    rng = np.random.default_rng(1)
    mat = rng.normal(0.10, 0.05, size=(50, 3))
    a, b = cc.summarize(mat, 500, 7), cc.summarize(mat, 500, 7)
    assert a == b
    assert a["ci95"][0] < a["mean"] < a["ci95"][1] and a["n_sentences"] == 50 and a["n_repeats"] == 3
    assert len(a["repeat_means"]) == 3 and a["repeat_std"] >= 0
    assert cc.summarize(mat[:, :1], 200, 0)["repeat_std"] is None  # one repeat: variance not estimable, not faked


def test_ci_width_shrinks_with_more_sentences():
    rng = np.random.default_rng(2)
    small = cc.summarize(rng.normal(0.1, 0.05, (20, 3)), 500)["ci95"]
    big = cc.summarize(rng.normal(0.1, 0.05, (400, 3)), 500)["ci95"]
    assert (big[1] - big[0]) < (small[1] - small[0])


def test_paired_delta_detects_shift_not_noise():
    rng = np.random.default_rng(3)
    base = rng.normal(0.10, 0.05, (50, 1)) + rng.normal(0, 0.005, (50, 3))
    same = base + rng.normal(0, 0.005, base.shape)
    better = base - 0.03 + rng.normal(0, 0.005, base.shape)
    assert not cc.paired_delta(base, same, 500)["excludes_zero"]
    d = cc.paired_delta(base, better, 500)
    assert d["excludes_zero"] and d["delta_mean"] < 0 and d["ci95"][1] < 0
    with pytest.raises(AssertionError):
        cc.paired_delta(base, base[:10])


def test_paired_beats_unpaired_when_sentences_differ_in_difficulty():
    rng = np.random.default_rng(4)
    difficulty = rng.uniform(0, 0.4, (50, 1))  # hard sentences are hard for both models
    a = difficulty + rng.normal(0, 0.005, (50, 3))
    b = difficulty - 0.01 + rng.normal(0, 0.005, (50, 3))
    assert cc.paired_delta(a, b, 1000)["excludes_zero"]  # a -0.01 shift is invisible to unpaired CIs (sentence spread 0.4) but clear when paired


def test_mulaw_channels():
    sr = 22050
    t = np.arange(sr) / sr
    x = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    w8, s8 = cc.channel(x, sr, "8k")
    w16, s16 = cc.channel(x, sr, "16k")
    assert (s8, s16) == (8000, 16000) and len(w8) == 8000 and len(w16) == 16000
    ref8 = cc.soxr.resample(x, sr, 8000)
    assert float(np.sqrt(np.mean((w8 - ref8) ** 2))) < 0.02  # 8-bit companding error is small for speech-level signals
    assert float(np.sqrt(np.mean((w8 - ref8) ** 2))) > 1e-5  # ...but not zero: the channel really quantizes


def test_build_entry_has_the_documented_schema():
    rng = np.random.default_rng(5)
    M = {k: rng.uniform(0.05, 0.2, (6, 3)) for k in ("cer", "cer_legacy", "per", "utmos", "spk_neural", "spk_mfcc", "cer_8k", "cer_16k")}
    raw = {"M": M, "hyps": ["h"] * 6, "refs": ["r"] * 6, "durs": [1.0] * 6, "loo": 0.92}
    meta = {"path": "milestones/step_1", "global_step": 1, "onnx_sha256": "ab" * 32, "hf_commit": {"title": "t"}, "experiment_id": "unknown", "git_sha": "unknown"}
    e = cc.build_entry(meta, raw, {"noise_scale": 0.667}, 3, "small", "small", "utmos-note", 12, 200, 0)
    for k in ("checkpoint", "experiment_id", "git_sha", "dataset", "cer", "per", "utmos", "speaker_similarity", "telephony", "inference_params", "repeats", "ci"):
        assert k in e, k
    assert "PREDICTED MOS" in e["utmos"]["label"] and "NOT speaker verification" in e["speaker_similarity"]["neural"]["label"]
    assert set(e["speaker_similarity"]) == {"neural", "mfcc"} and set(e["telephony"]) == {"8k", "16k"}
    assert e["cer"]["ci95"][0] <= e["cer"]["mean"] <= e["cer"]["ci95"][1]
    json.dumps(e)  # serializable


def test_eval_set_info_reports_overlap_check():
    info = cc.eval_set_info(cc.me.DEFAULT_SENTENCES)
    assert info["n"] == 50 and len(info["sha256"]) == 64
    ov = info["overlap_with_training_and_test_text"]
    assert isinstance(ov, str) or ov["n_overlapping"] == 0
