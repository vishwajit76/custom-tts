"""Phase 5: ExpressiveEngine boundary (fake native engine), capability gating, DSP prosody (opt-in, labelled dsp)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services import dsp, expressive_engine, tts
from app.services.conditioning import EngineCapabilities, VoiceCondition, control_kinds, split_controls, validate_condition
from app.services.expressive_engine import ExpressiveEngine

SR = 24000


class FakeExpressive(ExpressiveEngine):
    """Records what reaches the model; output amplitude encodes nothing, we only check the call."""
    capabilities = EngineCapabilities(native_emotion=True, native_style=True, role=True, speed=True, streaming="sentence")
    ready = True
    max_workers = 2

    def __init__(self):
        self.calls = []

    def load(self):
        pass

    def voices(self):
        return [{"voice_id": "expr:v", "sample_rate": SR}]

    def sample_rate(self, voice):
        return SR

    def _wav(self):
        return np.concatenate([np.zeros(SR // 4, np.float32), 0.3 * np.sin(np.arange(SR // 2) / 6).astype(np.float32)])

    def synth(self, text, voice, speed, ref=None, ref_text=None):
        self.calls.append(("plain", text, {}))
        return self._wav()

    def synth_native(self, text, voice, speed, controls, ref=None, ref_text=None):
        self.calls.append(("native", text, dict(controls)))
        return self._wav()


class Plain(FakeExpressive):
    capabilities = EngineCapabilities(speed=True, streaming="sentence")


@pytest.fixture()
def make_client(monkeypatch):
    def make(engine, **cfg):
        tts._cache.clear()
        monkeypatch.setattr(tts, "engine", engine)
        monkeypatch.setattr(settings, "default_voice", "expr:v")
        monkeypatch.setattr(settings, "api_keys", "")
        monkeypatch.setattr(settings, "cache_size", 0)
        for k, v in cfg.items():
            monkeypatch.setattr(settings, k, v)
        from app.main import app
        return TestClient(app)
    return make


def test_native_conditioning_flows_end_to_end(make_client):
    eng = FakeExpressive()
    with make_client(eng) as c:
        r = c.post("/v1/audio/speech", json={"input": "नमस्ते।", "condition": {
            "emotion": "empathetic", "emotion_strength": 0.6, "style": "warm", "role": "customer_support"}})
    assert r.status_code == 200
    assert r.headers["x-tts-applied-controls"] == "emotion,emotion_strength,style,role"  # native: no ":steered"
    kind, _, ctl = eng.calls[0]
    assert kind == "native" and ctl == {"emotion": "empathetic", "emotion_strength": 0.6, "style": "warm", "role": "customer_support"}


def test_unconditioned_request_uses_plain_synth(make_client):
    eng = FakeExpressive()
    with make_client(eng) as c:
        assert c.post("/v1/audio/speech", json={"input": "नमस्ते।"}).status_code == 200
    assert eng.calls[0][0] == "plain"


def test_capabilities_gate_native_controls(make_client):
    eng = Plain()
    with make_client(eng) as c:
        r = c.post("/v1/audio/speech", json={"input": "नमस्ते।", "condition": {"emotion": "happy"}})
        assert r.status_code == 422 and r.json()["detail"]["unsupported"] == ["emotion"]
        r = c.post("/v1/audio/speech", json={"input": "नमस्ते।", "condition": {"emotion": "happy", "fallback": "ignore"}})
        assert r.status_code == 200 and r.headers["x-tts-ignored-controls"] == "emotion" and r.headers["x-tts-applied-controls"] == ""
    assert [k for k, *_ in eng.calls] == ["plain"]  # the engine never saw the control


def test_native_controls_reach_engine_via_ws_and_cache_key_differs(make_client):
    eng = FakeExpressive()
    with make_client(eng, cache_size=16) as c:
        for emo in ("calm", "happy", "calm"):
            assert c.post("/v1/audio/speech", json={"input": "नमस्ते।", "condition": {"emotion": emo}}).status_code == 200
    assert [x[2]["emotion"] for x in eng.calls] == ["calm", "happy"]  # 3rd is a cache hit; different emotion is not


def test_no_native_dsp_or_prompt_kinds():
    assert control_kinds(FakeExpressive.capabilities)["emotion"] == "native"
    assert control_kinds(EngineCapabilities(prompt_emotion=True))["emotion"] == "steered"
    k = control_kinds(EngineCapabilities(), dsp=True)
    assert k["pitch"] == "dsp" and k["energy"] == "dsp" and k["emotion"] == "none" and k["style"] == "none" and k["role"] == "none"
    assert control_kinds(EngineCapabilities(cloning=True))["cloning"] == "embedding"


def test_dsp_never_applies_emotion():
    caps = EngineCapabilities(speed=True)
    with pytest.raises(Exception):
        validate_condition(VoiceCondition(emotion="happy"), caps, dsp=True)
    _, applied, ignored = validate_condition(VoiceCondition(emotion="happy", pitch=2, energy=1.2, fallback="ignore"), caps, dsp=True)
    assert applied == ["pitch:dsp", "energy:dsp"] and ignored == ["emotion"]


def test_dsp_off_by_default_rejects_pitch(make_client):
    assert settings.dsp_prosody is False
    with make_client(Plain()) as c:
        r = c.post("/v1/audio/speech", json={"input": "नमस्ते।", "condition": {"pitch": 2}})
    assert r.status_code == 422 and r.json()["detail"]["unsupported"] == ["pitch"]


def _f0(wav, sr):
    w = wav[len(wav) // 4:3 * len(wav) // 4]
    sp = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    return np.argmax(sp) * sr / len(w)


def test_dsp_pitch_and_energy_end_to_end(make_client):
    import asyncio
    eng = Plain()
    with make_client(eng, dsp_prosody=True) as c:
        base = c.post("/v1/audio/speech", json={"input": "नमस्ते।", "response_format": "pcm", "sample_rate": SR})
        r = c.post("/v1/audio/speech", json={"input": "नमस्ते।", "response_format": "pcm", "sample_rate": SR,
                                             "condition": {"pitch": 4, "energy": 0.5}})
        bad = c.post("/v1/audio/speech", json={"input": "नमस्ते।", "condition": {"style": "warm"}})
    assert r.status_code == 200 and r.headers["x-tts-applied-controls"] == "pitch:dsp,energy:dsp"
    assert bad.status_code == 422  # DSP does not unlock style/emotion
    a, b = (np.frombuffer(x.content, "<i2").astype(np.float32) / 32767 for x in (base, r))
    assert len(a) == len(b)  # duration preserved
    assert abs(_f0(b, SR) / _f0(a, SR) - 2 ** (4 / 12)) < 0.06
    assert 0.35 < np.abs(b).max() / np.abs(a).max() < 0.65  # gain 0.5 (pitch shift keeps amplitude roughly)


def test_dsp_functions():
    w = 0.9 * np.sin(np.arange(SR) / 5).astype(np.float32)
    boosted = dsp.apply_prosody(w, SR, energy=2.0)
    assert np.abs(boosted).max() <= 1.0 and np.abs(boosted).max() > 0.95  # soft-limited, not hard-clipped
    assert np.array_equal(dsp.apply_prosody(w, SR), w)  # no controls: untouched
    half = dsp.apply_prosody(w, SR, energy=0.0, strength=0.5)  # strength scales toward neutral
    assert abs(np.abs(half).max() - 0.45) < 0.02
    short = np.ones(100, np.float32) * 0.1
    assert len(dsp.apply_prosody(short, SR, pitch=3)) == 100


def test_split_controls_and_capabilities_endpoint(make_client):
    eff = VoiceCondition(pitch=2, energy=1.2, prosody_strength=0.5, emotion="calm")
    n, p = split_controls(eff, FakeExpressive.capabilities, dsp=True)
    assert n == {"emotion": "calm"} and p == {"pitch": 2, "energy": 1.2, "strength": 0.5}
    with make_client(FakeExpressive(), dsp_prosody=True) as c:
        j = c.get("/v1/capabilities").json()
    assert j["dsp"]["enabled"] and j["engines"]["fakeexpressive"]["control_kinds"]["emotion"] == "native"
    assert j["engines"]["fakeexpressive"]["control_kinds"]["pitch"] == "dsp"


def test_expressive_disabled_unless_configured(monkeypatch):
    with pytest.raises(expressive_engine.ExpressiveNotConfigured):
        expressive_engine.build_configured("")
    with pytest.raises(ValueError):
        expressive_engine.build_configured("tests.test_expressive_dsp:np")
    assert type(expressive_engine.build_configured("tests.test_expressive_dsp:FakeExpressive")).__name__ == "FakeExpressive"
    monkeypatch.setattr(settings, "expressive_engine", "")
    with pytest.raises(expressive_engine.ExpressiveNotConfigured):
        tts._make_engine("expressive")


def test_base_class_refuses_silent_ignore():
    class Half(ExpressiveEngine):
        capabilities = EngineCapabilities(native_emotion=True)
    with pytest.raises(NotImplementedError):
        Half().synth_native("x", "v", 1.0, {"emotion": "calm"})


@pytest.fixture(autouse=True)
def _clear_cache_after():
    yield
    tts._cache.clear()
