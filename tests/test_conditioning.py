"""Conditioning schema, capability truthfulness, reject/ignore behaviour, and API backward compatibility (mock engines)."""
import base64
import importlib
import inspect

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import settings
from app.services import tts
from app.services.conditioning import (
    DEFAULT_CAPABILITIES,
    EngineCapabilities,
    UnsupportedControl,
    VoiceCondition,
    validate_condition,
)

SR = 22050
NONE_CAPS = EngineCapabilities(speed=True)


# ---------- schema ----------
def test_defaults_and_enums():
    c = VoiceCondition()
    assert c.speed == 1.0 and c.fallback == "reject" and c.emotion is None
    assert VoiceCondition(emotion="apologetic", style="storytelling", role="customer_support").role.value == "customer_support"


@pytest.mark.parametrize("bad", [
    {"emotion": "furious"}, {"style": "robotic"}, {"role": "pirate"}, {"emotion": "happy", "emotion_strength": 1.5},
    {"style": "soft", "style_strength": -0.1}, {"speed": 0.2}, {"speed": 3}, {"fallback": "maybe"},
    {"emotion_strength": 0.5}, {"style_strength": 0.5}, {"reference_text": "x"}, {"prosody_strength": 2},
])
def test_invalid_conditions(bad):
    with pytest.raises(ValidationError):
        VoiceCondition(**bad)


# ---------- validation ----------
def test_reject_unsupported_by_default():
    with pytest.raises(UnsupportedControl) as e:
        validate_condition(VoiceCondition(emotion="happy", emotion_strength=0.5, role="teacher"), NONE_CAPS, "piper")
    assert e.value.controls == ["emotion", "emotion_strength", "role"] and "piper" in str(e.value)


def test_ignore_records_and_never_claims_applied():
    eff, applied, ignored = validate_condition(
        VoiceCondition(emotion="sad", pitch=2, speed=1.25, fallback="ignore"), NONE_CAPS)
    assert applied == ["speed"] and ignored == ["emotion", "pitch"]
    assert eff.emotion is None and eff.pitch is None and eff.speed == 1.25


def test_unsupported_speed_and_clamping():
    no_speed = EngineCapabilities()
    _, applied, ignored = validate_condition(VoiceCondition(speed=1.5, fallback="ignore"), no_speed)
    assert ignored == ["speed"] and not applied
    eff, applied, _ = validate_condition(VoiceCondition(speed=0.5), EngineCapabilities(speed=True, speed_range=(0.7, 2.0)))
    assert applied == ["speed:clamped_to_0.7"] and eff.speed == 0.5  # engine clamps; we only report it


def test_native_vs_steered_and_cloning():
    _, applied, _ = validate_condition(VoiceCondition(emotion="calm", style="warm"), EngineCapabilities(native_emotion=True, prompt_emotion=True))
    assert applied == ["emotion", "style:steered"]
    _, applied, _ = validate_condition(VoiceCondition(emotion="calm"), EngineCapabilities(prompt_emotion=True))
    assert applied == ["emotion:steered"]
    _, applied, _ = validate_condition(VoiceCondition(reference_audio="AAAA", reference_text="hi"), EngineCapabilities(cloning=True))
    assert applied == ["reference_audio", "reference_text"]
    with pytest.raises(UnsupportedControl):
        validate_condition(VoiceCondition(reference_audio="AAAA"), NONE_CAPS)
    with pytest.raises(UnsupportedControl):  # speaker ids are not bound to any engine yet
        validate_condition(VoiceCondition(speaker_id="s1"), EngineCapabilities(cloning=True, speaker_embedding=True))


def test_empty_condition_is_noop():
    eff, applied, ignored = validate_condition(VoiceCondition(), EngineCapabilities())
    assert applied == [] and ignored == [] and eff == VoiceCondition()


# ---------- capability truthfulness per engine ----------
def _caps(module, cls):
    return getattr(importlib.import_module(f"app.services.{module}"), cls).capabilities


def test_piper_caps():
    c = _caps("piper_engine", "PiperEngine")
    assert c.speed and not (c.cloning or c.native_emotion or c.native_style or c.prompt_emotion or c.role or c.pitch or c.energy or c.speaker_embedding)
    with pytest.raises(UnsupportedControl):
        validate_condition(VoiceCondition(emotion="happy"), c, "piper")


def test_qwen_caps():
    c = _caps("qwen_engine", "QwenEngine")
    assert c.cloning and c.speed and not (c.native_emotion or c.prompt_emotion or c.native_style or c.speaker_embedding)


@pytest.mark.parametrize("module, cls", [("supertonic_engine", "SupertonicEngine"), ("kokoro_engine", "KokoroEngine")])
def test_optional_engine_caps(module, cls):
    pytest.importorskip({"supertonic_engine": "supertonic", "kokoro_engine": "kokoro_onnx"}[module])
    c = _caps(module, cls)
    assert c.speed and not (c.cloning or c.native_emotion or c.prompt_emotion or c.role or c.pitch or c.energy)
    if cls == "SupertonicEngine":
        assert c.speed_range == (0.7, 2.0)  # matches the clamp in SupertonicEngine.synth


@pytest.mark.parametrize("module, cls, mod_pkg", [("piper_engine", "PiperEngine", None), ("qwen_engine", "QwenEngine", None),
                                                  ("supertonic_engine", "SupertonicEngine", "supertonic"), ("kokoro_engine", "KokoroEngine", "kokoro_onnx")])
def test_claims_match_synth_signature(module, cls, mod_pkg):
    """No engine's synth() accepts emotion/pitch/energy/style parameters, so none may claim to consume them."""
    if mod_pkg:
        pytest.importorskip(mod_pkg)
    engine_cls = getattr(importlib.import_module(f"app.services.{module}"), cls)
    params = set(inspect.signature(engine_cls.synth).parameters)
    c = engine_cls.capabilities
    for claim, param in [("native_emotion", "emotion"), ("native_style", "style"), ("pitch", "pitch"), ("energy", "energy"), ("role", "role")]:
        assert not getattr(c, claim) or param in params
    assert c.cloning == engine_cls.supports_cloning
    assert c.streaming == "sentence"  # pipeline-level; no engine streams inside a model call


# ---------- API (mock engines) ----------
class FakeEngine:
    supports_cloning = False
    max_workers = 2
    ready = True
    capabilities = EngineCapabilities(speed=True, streaming="sentence", languages=("hi",))
    calls: list

    def __init__(self):
        self.calls = []

    def load(self):
        pass

    def voices(self):
        return [{"voice_id": "fake", "sample_rate": SR}]

    def has_voice(self, v):
        return v == "fake"

    def sample_rate(self, v):
        return SR

    def synth(self, text, voice, speed, ref=None, ref_text=None):
        self.calls.append((text, speed))
        return np.full(SR // 10, 0.5, np.float32)


class EmotiveEngine(FakeEngine):
    supports_cloning = True
    capabilities = EngineCapabilities(cloning=True, speed=True, prompt_emotion=True, streaming="sentence")


@pytest.fixture()
def eng(monkeypatch):
    e = FakeEngine()
    monkeypatch.setattr(tts, "engine", e)
    monkeypatch.setattr(settings, "default_voice", "fake")
    monkeypatch.setattr(settings, "api_keys", "")
    monkeypatch.setattr(settings, "cache_size", 0)
    return e


@pytest.fixture()
def client(eng):
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_old_bodies_unchanged(client):
    for body in ({"input": "नमस्ते।"}, {"input": "नमस्ते।", "voice": "fake", "speed": 1.2, "response_format": "pcm", "sample_rate": 8000},
                 {"input": "नमस्ते।", "model": "tts-1"}):
        r = client.post("/v1/audio/speech", json=body)
        assert r.status_code == 200 and "x-tts-applied-controls" not in r.headers
    r = client.post("/v1/audio/speech/stream", json={"input": "नमस्ते।"})
    assert r.status_code == 200 and "x-tts-ignored-controls" not in r.headers


def test_422_on_unsupported_emotion(client):
    for path in ("/v1/audio/speech", "/v1/audio/speech/stream"):
        r = client.post(path, json={"input": "नमस्ते।", "condition": {"emotion": "happy"}})
        assert r.status_code == 422
        assert r.json()["detail"]["unsupported"] == ["emotion"]


def test_invalid_enum_is_422(client):
    r = client.post("/v1/audio/speech", json={"input": "x", "condition": {"emotion": "furious"}})
    assert r.status_code == 422


def test_ignore_fallback_headers_and_speed(client, eng):
    body = {"input": "नमस्ते।", "condition": {"emotion": "sad", "speed": 1.5, "fallback": "ignore"}}
    for path in ("/v1/audio/speech", "/v1/audio/speech/stream"):
        r = client.post(path, json=body)
        assert r.status_code == 200
        assert r.headers["x-tts-applied-controls"] == "speed" and r.headers["x-tts-ignored-controls"] == "emotion"
    assert eng.calls[-1][1] == 1.5


def test_condition_speed_only_and_top_level_speed(client, eng):
    r = client.post("/v1/audio/speech", json={"input": "नमस्ते।", "speed": 1.3, "condition": {"style": "soft", "fallback": "ignore"}})
    assert r.status_code == 200 and eng.calls[-1][1] == 1.3
    assert r.headers["x-tts-ignored-controls"] == "style"


def test_steered_emotion_and_cloning_flow(client, monkeypatch):
    monkeypatch.setattr(tts, "engine", EmotiveEngine())
    r = client.post("/v1/audio/speech", json={"input": "नमस्ते।", "condition": {"emotion": "calm"}})
    assert r.status_code == 200 and r.headers["x-tts-applied-controls"] == "emotion:steered"
    import io

    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, np.float32), 16000, format="WAV")
    ref = base64.b64encode(buf.getvalue()).decode()
    r = client.post("/v1/audio/speech", json={"input": "नमस्ते।", "condition": {"reference_audio": ref}})
    assert r.status_code == 200 and r.headers["x-tts-applied-controls"] == "reference_audio"


def test_legacy_reference_audio_still_400_without_cloner(client):
    assert client.post("/v1/audio/speech", json={"input": "x", "reference_audio": "AAAA"}).status_code == 400


def test_voices_and_capabilities_endpoints(client):
    v = client.get("/v1/voices").json()["data"][0]
    assert v["voice_id"] == "fake" and v["sample_rate"] == SR  # old fields intact
    assert v["capabilities"]["speed"] is True and v["capabilities"]["native_emotion"] is False
    c = client.get("/v1/capabilities").json()
    assert c["engines"]["fake"]["voices"] == ["fake"] and "happy" in c["controls"]["emotion"]
    assert 8000 in c["output_sample_rates"]


def test_multi_engine_caps_per_voice(client, monkeypatch):
    class Other(EmotiveEngine):
        def voices(self):
            return [{"voice_id": "other:v", "sample_rate": SR}]

        def has_voice(self, v):
            return v == "other:v"

    monkeypatch.setattr(tts, "engine", tts.MultiEngine([FakeEngine(), Other()]))
    assert tts.capabilities_for("fake").prompt_emotion is False and tts.capabilities_for("other:v").prompt_emotion is True
    assert client.post("/v1/audio/speech", json={"input": "x", "voice": "fake", "condition": {"emotion": "calm"}}).status_code == 422
    assert client.post("/v1/audio/speech", json={"input": "x", "voice": "other:v", "condition": {"emotion": "calm"}}).status_code == 200


def test_engine_without_capabilities_attr_gets_default(monkeypatch):
    class Bare:
        ready = True

        def has_voice(self, v):
            return True

    monkeypatch.setattr(tts, "engine", Bare())
    assert tts.capabilities_for("fake") == DEFAULT_CAPABILITIES


def test_ws_conditioning(client):
    with client.websocket_connect("/v1/audio/ws") as ws:
        ws.send_json({"type": "speak", "id": "a", "text": "नमस्ते।", "sample_rate": 8000})
        m = ws.receive_json()
        assert m == {"type": "start", "id": "a", "sample_rate": 8000, "encoding": "pcm_s16le"}  # unchanged for old clients
    with client.websocket_connect("/v1/audio/ws") as ws:
        ws.send_json({"type": "speak", "id": "b", "text": "नमस्ते।", "condition": {"role": "teacher"}})
        m = ws.receive_json()
        assert m["type"] == "error" and m["code"] == "unsupported_control" and m["unsupported"] == ["role"]
        ws.send_json({"type": "speak", "id": "c", "text": "नमस्ते।", "condition": {"role": "teacher", "fallback": "ignore"}})
        m = ws.receive_json()
        assert m["type"] == "start" and m["ignored_controls"] == ["role"] and m["applied_controls"] == []
        ws.send_json({"type": "speak", "id": "d", "text": "x", "condition": {"emotion": "nope"}})
        while True:
            m = ws.receive()
            if m.get("text") and '"d"' in m["text"]:
                assert "bad_request" in m["text"]
                break
