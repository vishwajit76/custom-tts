"""Pipeline + API tests against a fake engine (no model files needed)."""
import asyncio
import json
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services import tts
from app.services.scheduler import Scheduler

SR = 22050


class FakeEngine:
    supports_cloning = False
    max_workers = 2
    ready = True

    def load(self):
        pass

    def voices(self):
        return [{"voice_id": "fake", "sample_rate": SR}]

    def has_voice(self, v):
        return v == "fake"

    def sample_rate(self, v):
        return SR

    def synth(self, text, voice, speed, ref=None, ref_text=None):
        time.sleep(0.02)
        n = int(SR * 0.05 * len(text) / speed)  # 50 ms of audio per char
        return np.concatenate([np.zeros(SR // 4, np.float32), 0.5 * np.sin(np.arange(n, dtype=np.float32) / 5)])


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(tts, "engine", FakeEngine())
    monkeypatch.setattr(settings, "default_voice", "fake")
    monkeypatch.setattr(settings, "api_keys", "")
    monkeypatch.setattr(settings, "cache_size", 0)
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_split_for_stream():
    text = "नमस्ते, मैं आपकी बैंक से बात कर रही हूँ और आपकी किस्त के बारे में पूछना चाहती हूँ। दूसरा वाक्य है?"
    parts = tts.split_for_stream(text)
    assert len(parts[0]) <= settings.first_chunk_chars
    assert parts[-1] == "दूसरा वाक्य है?"
    assert " ".join(parts).split() == text.split()
    # nothing speakable ("।", "...", a quote, a lone virama) never becomes its own chunk: merged back, or dropped
    assert tts.split_for_stream('... नमस्ते। । " ्') == ['नमस्ते। । " ्'] and tts.split_for_stream("।") == []


def test_trim_lead():
    wav = np.concatenate([np.zeros(SR), np.ones(10)]).astype(np.float32)
    assert len(tts.trim_lead(wav, SR, 30)) == SR * 30 // 1000 + 10


def test_trim_tail_caps_pause_mid_sentence_more_than_at_sentence_end():
    # a 400 ms end-of-utterance tail left on a chunk cut at a comma was heard as a pause after "ठीक है,"
    wav = np.concatenate([np.ones(10), np.zeros(SR)]).astype(np.float32)
    assert len(tts.trim_tail(wav, SR, 150)) == 10 + SR * 150 // 1000
    assert len(tts.trim_tail(np.zeros(5, np.float32), SR, 150)) == 5
    ends = [c for c in ["ठीक है,", "ठीक है।", "आप कैसे हैं?", 'उसने कहा "ठीक है।"', "और बाकी"] if tts._SENTENCE_END.search(c)]
    assert ends == ["ठीक है।", "आप कैसे हैं?", 'उसने कहा "ठीक है।"']


def test_scheduler_runs_earliest_deadline_first():
    s = Scheduler(1)
    gate, order = threading.Event(), []

    async def main():
        blocker = asyncio.ensure_future(s.run(0, gate.wait))
        await asyncio.sleep(0.05)  # worker is now busy
        jobs = [asyncio.ensure_future(s.run(d, order.append, d)) for d in (3, 1, 2)]
        skipped = asyncio.ensure_future(s.run(0.5, order.append, "cancelled"))
        await asyncio.sleep(0.01)
        skipped.cancel()
        gate.set()
        await asyncio.gather(blocker, *jobs)

    asyncio.run(main())
    assert order == [1, 2, 3]


def test_scheduler_worker_survives_a_closed_loop():
    s, gate = Scheduler(1), threading.Event()

    async def abandon():  # queue a job, then let asyncio.run close the loop under it
        asyncio.ensure_future(s.run(0, gate.wait))
        await asyncio.sleep(0.05)

    asyncio.run(abandon())
    gate.set()
    time.sleep(0.05)  # the worker resolves into the closed loop: must not die

    async def again():
        return await asyncio.wait_for(s.run(0, lambda: "alive"), 2)

    assert asyncio.run(again()) == "alive"


def test_stream_synthesizes_the_next_chunk_while_the_current_one_plays(monkeypatch):
    # a long chunk after a short first one started only once the first was consumed: the player ran dry mid-sentence
    started = {}

    class Slow(FakeEngine):
        def synth(self, text, voice, speed, ref=None, ref_text=None):
            started[text] = time.monotonic()
            return super().synth(text, voice, speed)

    monkeypatch.setattr(tts, "engine", Slow())
    monkeypatch.setattr(tts, "scheduler", Scheduler(2))
    monkeypatch.setattr(settings, "cache_size", 0)

    async def main():
        chunks = tts.split_for_stream("नमस्ते, मैं आपकी बैंक से बात कर रही हूँ और आपकी किस्त के बारे में पूछना चाहती हूँ।")
        gen = tts.stream("नमस्ते, मैं आपकी बैंक से बात कर रही हूँ और आपकी किस्त के बारे में पूछना चाहती हूँ।", "fake")
        await gen.__anext__()  # first chunk handed over; the client is now playing it
        playing = time.monotonic()
        await asyncio.sleep(0.2)
        assert len(chunks) > 1 and started.get(chunks[1], float("inf")) < playing + 0.1  # running, not awaiting the next pull
        await gen.aclose()

    asyncio.run(main())


def test_http_speech_and_stream(client):
    r = client.post("/v1/audio/speech", json={"input": "नमस्ते।", "voice": "default"})
    assert r.status_code == 200 and r.content[:4] == b"RIFF"
    r = client.post("/v1/audio/speech/stream", json={"input": "नमस्ते। दूसरा वाक्य।", "sample_rate": 8000})
    assert r.status_code == 200 and r.headers["x-sample-rate"] == "8000" and len(r.content) > 8000
    # no sample_rate: 24 kHz, the OpenAI pcm contract the calling platform's adapter assumes
    r = client.post("/v1/audio/speech/stream", json={"input": "नमस्ते।"})
    assert r.headers["x-sample-rate"] == "24000"
    assert client.post("/v1/audio/speech", json={"input": "x", "voice": "nope"}).status_code == 404


def test_http_overload(client, monkeypatch):
    monkeypatch.setattr(settings, "max_streams", 0)
    r = client.post("/v1/audio/speech/stream", json={"input": "नमस्ते।"})
    assert r.status_code == 503 and r.headers["retry-after"] == "1"


def _events(ws, until):
    frames, events = [], []
    while True:
        m = ws.receive()
        if m.get("bytes") is not None:
            frames.append(m["bytes"])
            continue
        ev = json.loads(m["text"])
        events.append(ev)
        if ev["type"] in until:
            return frames, events


def test_ws_speak_frames_and_order(client):
    with client.websocket_connect("/v1/audio/ws") as ws:
        ws.send_json({"type": "speak", "id": "a", "text": "नमस्ते।", "sample_rate": 8000, "frame_ms": 20})
        ws.send_json({"type": "speak", "id": "b", "text": "फिर मिलेंगे।", "sample_rate": 16000})
        frames, ev = _events(ws, {"end"})
        assert ev[0] == {"type": "start", "id": "a", "sample_rate": 8000, "encoding": "pcm_s16le"}
        assert all(len(f) == 320 for f in frames[:-1]) and ev[-1]["id"] == "a"
        frames, ev = _events(ws, {"end"})
        assert ev[0]["id"] == "b" and ev[-1]["audio_ms"] > 0


def test_ws_cancel_and_errors(client):
    with client.websocket_connect("/v1/audio/ws") as ws:
        ws.send_json({"type": "speak", "id": "long", "text": "यह एक लंबा वाक्य है। " * 20})
        ws.send_json({"type": "speak", "id": "queued", "text": "नमस्ते।"})
        m = ws.receive()
        while m.get("bytes") is None:
            m = ws.receive()
        ws.send_json({"type": "cancel"})
        _, ev = _events(ws, {"cancelled"})
        _, ev2 = _events(ws, {"cancelled"})
        assert {ev[-1]["id"], ev2[-1]["id"]} == {"long", "queued"}
        ws.send_json({"type": "speak", "text": ""})
        assert ws.receive_json()["code"] == "bad_request"
        ws.send_json({"type": "speak", "text": "x", "voice": "nope"})
        _, ev = _events(ws, {"error"})
        assert ev[-1]["code"] == "not_found"


class OtherEngine(FakeEngine):
    def voices(self):
        return [{"voice_id": "other:v", "sample_rate": 16000}]

    def has_voice(self, v):
        return v == "other:v"

    def sample_rate(self, v):
        return 16000

    def synth(self, text, voice, speed, ref=None, ref_text=None):
        return np.full(1600, 0.5, np.float32)


def test_multi_engine_routes_by_voice_id(client, monkeypatch):
    multi = tts.MultiEngine([FakeEngine(), OtherEngine()])
    monkeypatch.setattr(tts, "engine", multi)
    assert [v["voice_id"] for v in client.get("/v1/voices").json()["data"]] == ["fake", "other:v"]
    assert multi.sample_rate("fake") == SR and multi.sample_rate("other:v") == 16000
    r = client.post("/v1/audio/speech", json={"input": "नमस्ते।", "voice": "other:v", "response_format": "pcm", "sample_rate": 16000})
    assert r.status_code == 200 and abs(len(r.content) - (3200 + 2 * 16 * settings.lead_silence_ms)) <= 20  # + the padded lead (less the fade-in's quiet start); OtherEngine's 1600 samples, not FakeEngine's sine
    assert client.post("/v1/audio/speech", json={"input": "x", "voice": "nope"}).status_code == 404
    with pytest.raises(KeyError):
        multi.synth("x", "nope", 1.0)
    with pytest.raises(RuntimeError):  # same id in two engines
        tts.MultiEngine([FakeEngine(), FakeEngine()]).load()
    with pytest.raises(ValueError):
        tts._make_engine("piper,qwen3")


def test_second_chunk_is_small_enough_to_be_ready_before_the_first_ends():
    parts = tts.split_for_stream("नमस्ते, मैं श्रेया बोल रही हूँ और आपके loan के बारे में बात करना चाहती हूँ, क्या आप अभी बात कर सकते हैं?")
    assert parts[:2] == ["नमस्ते,", "मैं श्रेया बोल रही हूँ"] and len(parts[1]) <= 3 * len(parts[0]) + 20
    assert tts.split_for_stream("नमस्ते। आप कैसे हैं और घर पर सब कैसे हैं?")[1] == "आप कैसे हैं और घर पर सब कैसे हैं?"  # after a full sentence: untouched
