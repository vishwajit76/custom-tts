"""P6 production hardening: request-size caps on the JSON speech routes, traversal, and the auth surface as documented."""
import json

import pytest
from fastapi.testclient import TestClient

from app.core import limits
from app.core.config import settings
from app.services import tts
from tests.test_telephony import Tone


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(tts, "engine", Tone())
    monkeypatch.setattr(settings, "default_voice", "t")
    monkeypatch.setattr(settings, "api_keys", "")
    monkeypatch.setattr(settings, "cache_size", 0)
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_speech_json_body_cap_413_declared_length(client):
    """Only upload routes were capped: a POST of any size to /v1/audio/speech was buffered whole before validation."""
    cap = limits.body_limit("/v1/audio/speech")
    assert cap == settings.max_input_chars * 6 + 65536  # text-only engine: no room for a reference clip
    big = json.dumps({"input": "क" * 10, "pad": "x" * (cap + 1)})
    for path in ("/v1/audio/speech", "/v1/audio/speech/stream"):
        r = client.post(path, content=big, headers={"Content-Type": "application/json"})
        assert r.status_code == 413 and r.json() == {"error": "request body too large"}
    ok = client.post("/v1/audio/speech", json={"input": "नमस्ते।", "sample_rate": 8000})
    assert ok.status_code == 200


def test_speech_json_body_cap_413_chunked_without_content_length(client):
    cap = limits.body_limit("/v1/audio/speech")

    def chunks():
        for _ in range(cap // 50_000 + 3):
            yield b"x" * 50_000

    r = client.post("/v1/audio/speech", content=chunks(), headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_cloning_engine_gets_room_for_a_base64_reference(monkeypatch):
    monkeypatch.setattr(tts, "engine", type("E", (), {"supports_cloning": True})())
    cap = limits.body_limit("/v1/audio/speech")
    assert cap > settings.upload_max_bytes * 4 // 3  # a base64 clip of upload_max_bytes fits


def test_other_routes_are_not_capped_here():
    assert limits.body_limit("/v1/models") is None and limits.body_limit("/v1/audio/ws") is None
    assert limits.body_limit("/v1/audio/speech/../x") is None


@pytest.mark.parametrize("bad", ["../evil", "a/b", "..", ".hidden", "a\\b", "x" * 65, "a b"])
def test_legacy_voice_ids_cannot_traverse(bad):
    import re

    from app.models.schemas import VOICE_ID
    from app.services.qwen_engine import QwenEngine

    assert not re.fullmatch(VOICE_ID, bad)
    with pytest.raises(ValueError):
        QwenEngine._wav_path(bad)


def test_legacy_voice_upload_with_traversal_id_is_422_even_before_the_engine_check(client):
    r = client.post("/v1/voices", data={"voice_id": "../evil"}, files={"audio": ("a.wav", b"RIFF", "audio/wav")})
    assert r.status_code == 422


def test_health_and_metrics_need_no_key_but_v1_does(client, monkeypatch):
    """Documented surface: /health and /metrics are open (probes, Prometheus; keep them off the public listener), /v1/* needs a key."""
    monkeypatch.setattr(settings, "api_keys", "k1")
    assert client.get("/health").status_code == 200 and client.get("/metrics").status_code == 200
    assert client.get("/v1/voices").status_code == 401
    assert client.post("/v1/audio/speech", json={"input": "x"}).status_code == 401
    assert client.get("/v1/voices", headers={"Authorization": "Bearer k1"}).status_code == 200


def test_legacy_voice_delete_with_bad_id_is_422_not_500(client):
    """The unanchored pattern let '../evil'-like ids through validation; the engine's fullmatch then raised ValueError (500)."""
    assert client.delete("/v1/voices/a..%2Fb").status_code in (404, 422)
    assert client.delete("/v1/voices/evil%20x").status_code == 422


def test_retention_is_enforced_while_running_not_only_at_startup(monkeypatch, tmp_path):
    """An idle speaker kept its expired raw reference until the next restart or read."""
    import time
    from datetime import datetime, timedelta, timezone

    import numpy as np

    from app.main import app
    from app.services import speaker_registry as sr

    monkeypatch.setattr(settings, "speakers_dir", tmp_path / "spk")
    monkeypatch.setattr(tts, "engine", Tone())
    monkeypatch.setattr(settings, "retention_sweep_minutes", 0.0005)  # 30 ms
    reg = sr.get_registry()
    reg.create("o", "asha", "Asha", retention=sr.Retention(delete_after_days=1))
    m = {"sha256": "h", "duration_s": 4.0, "sample_rate": 16000, "rms_dbfs": -20.0, "clip_fraction": 0.0, "speech_fraction": 0.9, "quality": 0.8}
    with TestClient(app):  # lifespan running: the periodic sweep task is alive
        ref = reg.add_reference("asha", "o", np.full(64000, 0.01, np.float32), 16000, m)
        sp = reg._load("asha")
        sp.references[0].created_at = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        reg._save(sp)
        wav = tmp_path / "spk" / "asha" / ref.path
        assert wav.exists()
        deadline = time.time() + 5
        while wav.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert not wav.exists() and reg._load("asha").references[0].path is None


# ---------- WebSocket: a stale cancel must not poison a later request ----------
def _events_until(ws, kinds):
    seen = []
    while True:
        m = ws.receive()
        if m.get("bytes") is not None:
            continue
        ev = json.loads(m["text"])
        seen.append(ev)
        if ev["type"] in kinds:
            return seen


def test_cancel_of_a_finished_or_unknown_id_does_not_cancel_a_later_request_with_that_id(client):
    """cancel{id} for an id that was not queued was remembered forever: a later speak reusing the id (clients reuse "r1")
    was silently answered `cancelled`, and the remembered set grew without bound."""
    with client.websocket_connect("/v1/audio/ws") as ws:
        ws.send_json({"type": "speak", "id": "r1", "text": "नमस्ते।", "sample_rate": 8000})
        assert _events_until(ws, {"end"})[-1]["type"] == "end"
        ws.send_json({"type": "cancel", "id": "r1"})  # too late: already finished
        ws.send_json({"type": "cancel", "id": "never-sent"})
        ws.send_json({"type": "speak", "id": "r1", "text": "नमस्ते।", "sample_rate": 8000})
        ev = _events_until(ws, {"end", "cancelled", "error"})
        assert ev[-1]["type"] == "end" and ev[-1]["audio_ms"] > 0


def test_cancel_of_a_queued_id_still_works(client):
    with client.websocket_connect("/v1/audio/ws") as ws:
        ws.send_json({"type": "speak", "id": "first", "text": "यह एक लंबा वाक्य है। " * 10, "sample_rate": 8000})
        ws.send_json({"type": "speak", "id": "second", "text": "नमस्ते।", "sample_rate": 8000})
        ws.send_json({"type": "cancel", "id": "second"})
        ends = []
        while len(ends) < 2:
            m = ws.receive()
            if m.get("text"):
                ev = json.loads(m["text"])
                if ev["type"] in ("end", "cancelled"):
                    ends.append((ev["id"], ev["type"]))
        assert dict(ends) == {"first": "end", "second": "cancelled"}
