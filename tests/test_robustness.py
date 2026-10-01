"""Auth, overload, cancellation / barge-in, concurrency and backward compatibility (fake engine, no models)."""
import asyncio
import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import ws as ws_mod
from app.core import security
from app.core.config import settings
from app.services import tts
from app.services.scheduler import Scheduler

SR = 22050


class Slow:
    supports_cloning = False
    max_workers = 2
    ready = True

    def __init__(self, delay=0.05):
        self.delay, self.calls, self.lock = delay, 0, threading.Lock()

    def load(self):
        pass

    def voices(self):
        return [{"voice_id": "fake", "sample_rate": SR}]

    def has_voice(self, v):
        return v == "fake"

    def sample_rate(self, v):
        return SR

    def synth(self, text, voice, speed, ref=None, ref_text=None):
        with self.lock:
            self.calls += 1
        time.sleep(self.delay)
        return np.concatenate([np.zeros(SR // 4, np.float32), 0.4 * np.sin(np.arange(int(SR * 0.05 * len(text))) / 5).astype(np.float32)])


@pytest.fixture()
def eng(monkeypatch):
    e = Slow()
    monkeypatch.setattr(tts, "engine", e)
    monkeypatch.setattr(settings, "default_voice", "fake")
    monkeypatch.setattr(settings, "api_keys", "")
    monkeypatch.setattr(settings, "cache_size", 0)
    monkeypatch.setattr(security, "_hits", type(security._hits)(security._hits.default_factory))
    return e


@pytest.fixture()
def client(eng):
    from app.main import app
    with TestClient(app) as c:
        yield c


LONG = "यह एक लंबा वाक्य है। " * 20


# ---------- auth ----------
def test_auth_required_when_keys_set(client, monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "k1,k2")
    body = {"input": "नमस्ते।"}
    for path in ("/v1/audio/speech", "/v1/audio/speech/stream"):
        assert client.post(path, json=body).status_code == 401
        assert client.post(path, json=body, headers={"Authorization": "Bearer nope"}).status_code == 401
        assert client.post(path, json=body, headers={"Authorization": "Bearer k2"}).status_code == 200
    for path in ("/v1/voices", "/v1/capabilities", "/v1/speakers"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"Authorization": "Bearer k1"}).status_code == 200
    assert client.get("/health").status_code == 200  # probes stay open
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect) as e, client.websocket_connect("/v1/audio/ws"):
        pass
    assert e.value.code == 1008
    with client.websocket_connect("/v1/audio/ws?api_key=k1") as ws:
        ws.send_json({"type": "speak", "id": "a", "text": "नमस्ते।"})
        assert json.loads(ws.receive()["text"])["type"] == "start"


def test_rate_limit_429(client, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
    codes = [client.post("/v1/audio/speech", json={"input": "नमस्ते।"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


# ---------- overload ----------
def test_overload_503_on_both_http_endpoints_and_counts(client, monkeypatch):
    monkeypatch.setattr(settings, "max_streams", 0)
    before = tts.stats["streams_rejected"]
    for path in ("/v1/audio/speech", "/v1/audio/speech/stream"):
        r = client.post(path, json={"input": "नमस्ते।"})
        assert r.status_code == 503 and r.headers["retry-after"] == "1"
    assert tts.stats["streams_rejected"] == before + 2 and tts.stats["streams_active"] == 0


def test_ws_overload_and_queue_full(client, monkeypatch):
    monkeypatch.setattr(settings, "max_streams", 0)
    with client.websocket_connect("/v1/audio/ws") as ws:
        ws.send_json({"type": "speak", "id": "o", "text": "नमस्ते।"})
        seen = [json.loads(ws.receive()["text"]) for _ in range(2)]
        assert any(m.get("code") == "overloaded" for m in seen)
    monkeypatch.setattr(settings, "max_streams", 64)
    monkeypatch.setattr(ws_mod, "MAX_PENDING", 2)
    with client.websocket_connect("/v1/audio/ws") as ws:
        for i in range(5):
            ws.send_json({"type": "speak", "id": f"r{i}", "text": LONG})
        codes = []
        for _ in range(60):
            m = ws.receive()
            if m.get("text"):
                ev = json.loads(m["text"])
                if ev.get("code"):
                    codes.append(ev["code"])
                    if "queue_full" in codes:
                        break
        assert "queue_full" in codes
        ws.send_json({"type": "cancel"})


# ---------- cancellation / barge-in ----------
def test_barge_in_stops_audio_and_frees_slot(client, eng):
    cancelled0 = tts.stats["streams_cancelled"]
    with client.websocket_connect("/v1/audio/ws") as ws:
        ws.send_json({"type": "speak", "id": "long", "text": LONG})
        ws.send_json({"type": "speak", "id": "next", "text": LONG})
        while ws.receive().get("bytes") is None:
            pass
        t0 = time.perf_counter()
        ws.send_json({"type": "cancel"})  # barge-in
        got = []
        while len(got) < 2:
            m = ws.receive()
            if m.get("text"):
                ev = json.loads(m["text"])
                if ev["type"] == "cancelled":
                    got.append(ev["id"])
        assert set(got) == {"long", "next"} and time.perf_counter() - t0 < 2.0
        n = eng.calls
        time.sleep(0.4)
        assert eng.calls - n <= 2  # in-flight chunks may finish; nothing new is synthesized after the cancel
        assert tts.stats["streams_active"] == 0
        ws.send_json({"type": "speak", "id": "after", "text": "नमस्ते।"})  # connection still usable
        while True:
            m = ws.receive()
            if m.get("text") and json.loads(m["text"])["type"] == "end":
                break
    assert tts.stats["streams_cancelled"] > cancelled0


def test_cancel_task_skips_queued_chunks(eng):
    async def main():
        tts.scheduler = Scheduler(1)
        task = asyncio.create_task(tts.synthesize(LONG, "fake", sample_rate=8000))
        await asyncio.sleep(0.12)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        n = eng.calls
        await asyncio.sleep(0.4)
        assert eng.calls - n <= 1 and tts.stats["streams_active"] == 0
    asyncio.run(main())


def test_ws_disconnect_frees_slot(client):
    with client.websocket_connect("/v1/audio/ws") as ws:
        ws.send_json({"type": "speak", "id": "a", "text": LONG})
        while ws.receive().get("bytes") is None:
            pass
    time.sleep(0.5)
    assert tts.stats["streams_active"] == 0


# ---------- concurrency ----------
def test_concurrent_streams_all_complete(client):
    results, errs = [], []

    def one(i):
        try:
            r = client.post("/v1/audio/speech", json={"input": f"नमस्ते {i}। यह दूसरा वाक्य है।", "response_format": "pcm", "sample_rate": 8000})
            results.append((r.status_code, len(r.content)))
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=one, args=(i,)) for i in range(12)]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert not errs and len(results) == 12 and all(c == 200 and n > 0 and n % 2 == 0 for c, n in results)
    assert tts.stats["streams_active"] == 0


def test_concurrency_respects_max_streams(client, monkeypatch):
    monkeypatch.setattr(settings, "max_streams", 3)
    codes = []
    ts = [threading.Thread(target=lambda: codes.append(client.post("/v1/audio/speech", json={"input": LONG[:60]}).status_code)) for _ in range(10)]
    [t.start() for t in ts]
    [t.join(60) for t in ts]
    assert set(codes) <= {200, 503} and 200 in codes and tts.stats["streams_active"] == 0


# ---------- backward compatibility ----------
def test_backward_compat_shapes(client):
    r = client.post("/v1/audio/speech", json={"input": "नमस्ते।"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.headers["x-sample-rate"] == "24000"
    assert not any(h.startswith("x-tts-") for h in r.headers)
    with client.websocket_connect("/v1/audio/ws") as ws:
        ws.send_json({"type": "speak", "id": "a", "text": "नमस्ते।", "sample_rate": 8000})
        start = json.loads(ws.receive()["text"])
        assert start == {"type": "start", "id": "a", "sample_rate": 8000, "encoding": "pcm_s16le"}
    assert set(client.get("/v1/voices").json()["data"][0]) >= {"voice_id", "sample_rate"}
    assert client.get("/v1/models").json()["data"][0]["object"] == "model"


def test_cancel_right_after_a_frame_is_never_swallowed(client):
    """Guards the ws.py fix (asyncio.timeout instead of wait_for around send_bytes; on Python 3.11 wait_for swallowed a cancel
    landing as the send completed, so the request ran to 'end'). Found with the real Piper engine over a real socket; this fake
    engine + TestClient does not reproduce the race on the old code, so it is a smoke test, not proof."""
    with client.websocket_connect("/v1/audio/ws") as ws:
        for k in range(8):
            ws.send_json({"type": "speak", "id": f"c{k}", "text": LONG, "sample_rate": 16000})
            while ws.receive().get("bytes") is None:
                pass
            ws.send_json({"type": "cancel"})
            while True:
                m = ws.receive()
                if m.get("text"):
                    ev = json.loads(m["text"])
                    if ev["type"] in ("cancelled", "end"):
                        break
            assert ev["type"] == "cancelled", f"trial {k}: cancel ignored, request completed"
