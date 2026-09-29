"""bench/eval.py and bench/previews.py on synthetic audio (no models, no Whisper)."""
import json

import numpy as np
import pytest

from app.services.conditioning import EngineCapabilities
from app.services.speaker_encoder import SpeakerEncoder
from bench import eval as ev
from bench import previews

SR = 16000


def voiced(f0=150, secs=1.5, gap=None):
    t = np.arange(int(SR * secs)) / SR
    w = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3)) * 0.2
    if gap:
        a, b = int(gap[0] * SR), int(gap[1] * SR)
        w[a:b] = 0
    return w.astype(np.float32)


def test_signal_metrics_pause_f0_clipping():
    m = ev.signal_metrics(voiced(gap=(0.6, 0.9)), SR, "abc")
    assert m["pauses"]["count"] == 1 and 250 <= m["pauses"]["max_ms"] <= 330
    assert abs(m["f0"]["median_hz"] - 150) < 8 and m["clipping_ratio"] == 0 and not m["silent"]
    clip = ev.signal_metrics(np.clip(voiced() * 10, -1, 1), SR)
    assert clip["clipping_ratio"] > 0.05
    s = ev.signal_metrics(np.zeros(SR, np.float32), SR)
    assert s["silent"] and s["silence_ratio"] == 1.0


def test_roundtrip_8k_reports_loss_of_high_band():
    t = np.arange(SR * 2) / 24000
    bright = (0.3 * np.sin(2 * np.pi * 300 * t) + 0.3 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
    r = ev.roundtrip_8k(bright, 24000, SpeakerEncoder("mfcc"))
    assert r["energy_above_4khz_lost_pct"] > 40 and r["lsd_db_100_3400hz"] < 3.0
    assert r["speaker_similarity_after"] is not None


def test_evaluate_identical_inputs_and_unsupported_rows():
    calls = []

    def synth(text, voice, cond):
        calls.append((text, voice, json.dumps(cond, sort_keys=True)))
        if cond and cond.get("emotion"):
            raise ValueError("unsupported controls: emotion")
        f0 = 150 * (2 ** ((cond or {}).get("pitch", 0) / 12))
        return voiced(f0, 1.2), SR, ["pitch:dsp"] if cond and cond.get("pitch") else [], []

    conds = [{"name": "neutral", "condition": None}, {"name": "pitch+2", "condition": {"pitch": 2}}, {"name": "emo", "condition": {"emotion": "happy"}}]
    rep = ev.evaluate(synth, ["a", "b"], ["एक", "दो"], conds, SpeakerEncoder("mfcc"))
    assert len(calls) == 2 * 3 * 2  # same grid for every voice
    assert {(c[1]) for c in calls} == {"a", "b"}
    a = rep["voices"]["a"]
    assert a["emo"]["n_unsupported"] == 2 and a["emo"]["rows"][0]["status"] == "unsupported"
    assert a["pitch+2"]["rows"][0]["applied"] == ["pitch:dsp"]
    assert a["neutral"]["rows"][0]["speaker_similarity"] == pytest.approx(1.0, abs=1e-3)
    assert a["neutral"]["cer"] == {"skipped": "not requested"}  # CER never invented
    assert rep["encoder_neural"] is False and "same voice" in rep["reference"]
    json.dumps(rep)


def test_previews_only_supported_combos():
    plain = previews.supported_combos(EngineCapabilities(speed=True))
    assert {c["name"] for c in plain} == {"neutral", "speed-0.85", "speed-1.15"}
    dsp = {c["name"]: c for c in previews.supported_combos(EngineCapabilities(speed=True), dsp=True)}
    assert dsp["pitch+2"]["kind"] == "dsp" and dsp["pitch+2"]["applied"] == ["pitch:dsp"]
    assert not any(n.startswith(("emotion", "style", "role")) for n in dsp)
    native = {c["name"]: c for c in previews.supported_combos(EngineCapabilities(native_emotion=True, role=True))}
    assert native["emotion-calm"]["kind"] == "native" and "style-warm" not in native and "role-narrator" in native
    assert previews.label("v", dsp["pitch+2"]) == "v | pitch+2 [dsp]"
