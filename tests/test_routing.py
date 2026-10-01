"""Phase 10: routing_policy - schema, capability-based selection, explicit fallback, explicit voice wins, headers."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import settings
from app.services import routing, tts
from app.services.conditioning import EngineCapabilities, VoiceCondition

SR = 22050


class Base:
    supports_cloning = False
    max_workers = 2
    ready = True
    vid = "x"
    capabilities = EngineCapabilities(speed=True, streaming="sentence")

    def load(self):
        pass

    def voices(self):
        return [{"voice_id": self.vid, "sample_rate": SR}]

    def has_voice(self, v):
        return v == self.vid

    def sample_rate(self, v):
        return SR

    def synth(self, text, voice, speed, ref=None, ref_text=None):
        return np.concatenate([np.zeros(SR // 4, np.float32), 0.3 * np.sin(np.arange(SR // 2) / 6).astype(np.float32)])

    def synth_native(self, text, voice, speed, controls, ref=None, ref_text=None):
        return self.synth(text, voice, speed)


class QuickEngine(Base):
    vid = "quick:v"


class MidEngine(Base):
    vid = "mid:v"


class ExprEngine(Base):
    vid = "expr:v"
    capabilities = EngineCapabilities(speed=True, native_emotion=True, native_style=True, role=True, streaming="sentence")


TIERS = "quick:fast,mid:balanced,expr:slow"


@pytest.fixture()
def cfg(monkeypatch):
    tts._cache.clear()
    monkeypatch.setattr(settings, "engine_latency_tiers", TIERS)
    monkeypatch.setattr(settings, "default_voice", "quick:v")
    monkeypatch.setattr(settings, "api_keys", "")
    monkeypatch.setattr(settings, "cache_size", 0)


def client_for(monkeypatch, *engines):
    monkeypatch.setattr(tts, "engine", tts.MultiEngine(list(engines)) if len(engines) > 1 else engines[0])
    from app.main import app
    return TestClient(app)


def post(c, **kw):
    return c.post("/v1/audio/speech", json={"input": "नमस्ते।", "response_format": "pcm", **kw})


def test_schema():
    assert VoiceCondition(routing_policy="fast").routing_policy == "fast"
    with pytest.raises(ValidationError):
        VoiceCondition(routing_policy="turbo")
    from app.models.schemas import SpeechRequest, WsSpeak
    assert SpeechRequest(input="x", routing_policy="clone").routing_policy == "clone"
    with pytest.raises(ValidationError):
        WsSpeak(text="x", routing_policy="best")
    assert routing.parse_tiers(TIERS) == {"quick": "fast", "mid": "balanced", "expr": "slow"}
    with pytest.raises(ValueError):
        routing.parse_tiers("a:warp")


def test_fast_and_balanced_pick_by_tier(cfg, monkeypatch):
    with client_for(monkeypatch, ExprEngine(), MidEngine(), QuickEngine()) as c:
        r = post(c, routing_policy="fast")
        assert r.status_code == 200 and r.headers["x-tts-routing"] == "policy;engine=quick;voice=quick:v"
        r = post(c, routing_policy="balanced")
        assert r.headers["x-tts-routing"] == "policy;engine=mid;voice=mid:v"


def test_expressive_needs_capable_engine(cfg, monkeypatch):
    with client_for(monkeypatch, QuickEngine(), MidEngine()) as c:
        r = post(c, routing_policy="expressive")
        assert r.status_code == 422 and "expressive_engine" in r.json()["detail"]["unsupported"]
        r = post(c, condition={"emotion": "calm", "routing_policy": "fast"})  # capable of nothing: never silently routed
        assert r.status_code == 422 and "emotion" in r.json()["detail"]["unsupported"]
        r = post(c, condition={"emotion": "calm", "routing_policy": "expressive", "fallback": "ignore"})
        assert r.status_code == 200  # explicit fallback: routed balanced, and told
        assert r.headers["x-tts-ignored-controls"] == "emotion,routing_policy" and r.headers["x-tts-applied-controls"] == ""


def test_expressive_routes_emotion_to_capable_engine(cfg, monkeypatch):
    expr = ExprEngine()
    seen = []
    orig = expr.synth_native
    expr.synth_native = lambda *a, **k: (seen.append(a[3]), orig(*a, **k))[1]
    with client_for(monkeypatch, QuickEngine(), expr) as c:
        r = post(c, condition={"emotion": "happy", "routing_policy": "fast"})  # fast prefers quick but it can't do emotion
        assert r.status_code == 200 and r.headers["x-tts-routing"].startswith("policy;engine=expr")
        assert r.headers["x-tts-applied-controls"] == "emotion" and seen == [{"emotion": "happy"}]


def test_clone_policy(cfg, monkeypatch):
    class Cloner(Base):
        vid = "clone:v"
        supports_cloning = True
        capabilities = EngineCapabilities(cloning=True, speed=True)
    with client_for(monkeypatch, QuickEngine(), MidEngine()) as c:
        assert post(c, routing_policy="clone").status_code == 422
        r = post(c, condition={"routing_policy": "clone", "fallback": "ignore"})
        assert r.status_code == 200 and r.headers["x-tts-ignored-controls"] == "routing_policy"
    d = routing.route("clone", None, [("quick", QuickEngine()), ("cloner", Cloner())], {"quick": "fast"}, "quick:v")
    assert d.engine == "cloner" and not d.degraded


def test_explicit_voice_wins(cfg, monkeypatch):
    with client_for(monkeypatch, QuickEngine(), MidEngine(), ExprEngine()) as c:
        r = post(c, voice="mid:v", routing_policy="fast")
        assert r.status_code == 200 and r.headers["x-tts-routing"] == "explicit"
        r = post(c, voice="quick:v", condition={"routing_policy": "expressive"})  # would 422 if it were consulted... it is not
        assert r.status_code == 200 and r.headers["x-tts-routing"] == "explicit"


def test_no_policy_is_unchanged(cfg, monkeypatch):
    with client_for(monkeypatch, QuickEngine(), MidEngine()) as c:
        r = post(c)
        assert r.status_code == 200 and "x-tts-routing" not in r.headers


def test_single_engine_incapable_is_rejected(cfg, monkeypatch):
    with client_for(monkeypatch, QuickEngine()) as c:
        assert post(c, routing_policy="expressive").status_code == 422
        assert post(c, routing_policy="fast").status_code == 200


def test_ws_reports_routing(cfg, monkeypatch):
    with client_for(monkeypatch, QuickEngine(), MidEngine()) as c, c.websocket_connect("/v1/audio/ws") as ws:
        ws.send_json({"type": "speak", "id": "r", "text": "नमस्ते।", "routing_policy": "balanced"})
        start = json.loads(ws.receive()["text"])
        assert start["routing"] == "policy" and start["routed_engine"] == "mid" and start["routed_voice"] == "mid:v"
        ws.send_json({"type": "speak", "id": "q", "text": "नमस्ते।", "routing_policy": "expressive"})
        while True:
            m = ws.receive()
            if m.get("text") and json.loads(m["text"]).get("type") in ("error",):
                assert json.loads(m["text"])["code"] == "unsupported_control"
                break


def test_capabilities_lists_policies_and_tiers(cfg, monkeypatch):
    with client_for(monkeypatch, QuickEngine(), MidEngine()) as c:
        j = c.get("/v1/capabilities").json()
    assert j["controls"]["routing_policy"] == ["fast", "balanced", "expressive", "clone"]
    assert j["engines"]["quick"]["latency_tier"] == "fast" and j["engines"]["mid"]["latency_tier"] == "balanced"
