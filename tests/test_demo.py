"""/demo voice-agent tests: fake TTS engine, fake STT/LLM providers, mocked vendor HTTP (no network)."""
import asyncio
import io
import json
import time
import wave

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api import demo
from app.core.config import settings
from app.services import providers, tts
from app.services.persona_grammar import Persona, llm_hint

SR = 22050
FRAME_24K = 24000 * 40 // 1000 * 2  # bytes in a 40 ms frame at the default 24 kHz
SHORT = ("हाँ ", "जी, ", "मैं ", "आपकी ", "मदद ", "कर ", "सकती ", "हूँ।")  # one segment: no cut before the final ।


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


async def fake_chat(provider, messages):
    calls["llm"].append((provider, [dict(m) for m in messages]))
    last = messages[-1]["content"]
    if last == "boom":
        raise providers.ProviderError("openai HTTP 401: bad key")
    if last.startswith("long"):
        for i in range(30):  # ~1 s of speech per sentence, produced slowly: a turn that is still running when we interrupt
            yield f"यह वाक्य नंबर {i} है। "
            await asyncio.sleep(0.02)
    else:
        for d in SHORT:
            yield d


async def fake_transcribe(provider, wav):
    calls["stt"].append((provider, len(wav)))
    return "मुझे लोन चाहिए"


calls: dict = {}


@pytest.fixture()
def client(monkeypatch):
    calls.clear()
    calls.update(stt=[], llm=[])
    monkeypatch.setattr(tts, "engine", FakeEngine())
    for name, value in dict(
        default_voice="fake", api_keys="", cache_size=0, demo_enabled=True, demo_stt="", demo_llm="",
        platform_openai_api_key="sk-openai-secret", platform_sarvam_api_key="sarvam-secret",
        platform_gemini_api_key="", platform_elevenlabs_api_key="",
    ).items():
        monkeypatch.setattr(settings, name, value)
    monkeypatch.setattr(providers, "chat_stream", fake_chat)
    monkeypatch.setattr(providers, "transcribe", fake_transcribe)
    from app.main import app

    with TestClient(app, client=("127.0.0.1", 50000)) as c:
        yield c


def wav_bytes(seconds=0.5, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(rate * seconds))
    return buf.getvalue()


def until(ws, kinds):
    """Everything received (JSON events as dicts, PCM frames as bytes) up to and including an event of `kinds`."""
    items = []
    while True:
        m = ws.receive()
        if m.get("bytes") is not None:
            items.append(m["bytes"])
            continue
        ev = json.loads(m["text"])
        items.append(ev)
        if ev["type"] in kinds:
            return items


def first_frame(ws):
    while ws.receive().get("bytes") is None:
        pass


def types(items):
    return [i["type"] if isinstance(i, dict) else "pcm" for i in items]


# --- HTTP ---
def test_config_lists_only_providers_with_keys(client):
    r = client.get("/demo/config")
    assert r.status_code == 200
    body = r.json()
    assert body["stt"] == ["sarvam", "openai"] and body["llm"] == ["openai", "sarvam"]
    assert body["defaults"] == {"stt": "sarvam", "llm": "openai", "voice": "fake", "system_prompt": settings.demo_system_prompt}
    assert body["voices"] == [{"voice_id": "fake", "sample_rate": SR}]
    assert "secret" not in r.text


def test_config_defaults_follow_env(client, monkeypatch):
    monkeypatch.setattr(settings, "demo_stt", "openai")
    monkeypatch.setattr(settings, "demo_llm", "gemini")  # no key: falls back to the first available
    d = client.get("/demo/config").json()["defaults"]
    assert d["stt"] == "openai" and d["llm"] == "openai"


def test_config_without_keys(client, monkeypatch):
    monkeypatch.setattr(settings, "platform_openai_api_key", "")
    monkeypatch.setattr(settings, "platform_sarvam_api_key", "")
    body = client.get("/demo/config").json()
    assert body["stt"] == [] and body["llm"] == [] and body["defaults"]["stt"] == ""
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_json({"type": "text", "text": "नमस्ते"})
        assert "no STT/LLM provider" in ws.receive_json()["message"]


def test_page_served_without_auth(client, monkeypatch, tmp_path):
    page = tmp_path / "demo.html"
    page.write_text("<!doctype html><title>demo</title>")
    monkeypatch.setattr(demo, "PAGE", page)
    monkeypatch.setattr(settings, "api_keys", "secret")
    r = client.get("/demo")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html") and "<title>demo</title>" in r.text


def test_auth_enforced(client, monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "secret")
    assert client.get("/demo/config").status_code == 401
    assert client.get("/demo/config", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/demo/config", headers={"Authorization": "Bearer secret"}).status_code == 200
    with pytest.raises(WebSocketDisconnect) as e, client.websocket_connect("/demo/ws"):
        pass
    assert e.value.code == 1008
    with pytest.raises(WebSocketDisconnect) as e, client.websocket_connect("/demo/ws?api_key=wrong"):
        pass
    assert e.value.code == 1008
    for kwargs in ({"url": "/demo/ws?api_key=secret"}, {"url": "/demo/ws", "headers": {"Authorization": "Bearer secret"}}):
        with client.websocket_connect(**kwargs) as ws:
            ws.send_json({"type": "reset"})
            assert ws.receive_json()["type"] == "ready"


def test_disabled_is_404(client, monkeypatch):
    monkeypatch.setattr(settings, "demo_enabled", False)
    assert client.get("/demo").status_code == 404
    assert client.get("/demo/config").status_code == 404
    with pytest.raises(WebSocketDisconnect) as e, client.websocket_connect("/demo/ws"):
        pass
    assert e.value.code == 1008


# --- websocket turns ---
def test_text_turn(client):
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_json({"type": "text", "text": "नमस्ते"})
        items = until(ws, {"turn_end"})
    t = types(items)
    assert t[0] == "llm_delta" and t.count("audio_start") == 1 and t[-1] == "turn_end" and "transcript" not in t
    assert items[t.index("audio_start")] == {"type": "audio_start", "sample_rate": 24000}
    assert t.index("audio_start") < t.index("pcm")  # announced before the first frame
    frames = [i for i in items if isinstance(i, bytes)]
    assert len(frames) > 2 and all(len(f) == FRAME_24K for f in frames[:-1]) and 0 < len(frames[-1]) <= FRAME_24K
    end = items[-1]
    assert end["reply"] == "".join(SHORT) == "".join(i["text"] for i in items if isinstance(i, dict) and i["type"] == "llm_delta")
    m = end["metrics"]
    assert set(m) == {"stt_ms", "llm_ttft_ms", "tts_ttfa_ms", "e2e_ms", "audio_ms"}
    assert m["stt_ms"] is None and m["llm_ttft_ms"] >= 0 and 0 < m["tts_ttfa_ms"] <= m["e2e_ms"]
    assert m["audio_ms"] == round(sum(map(len, frames)) / 2 / 24000 * 1000)
    assert calls["llm"] == [("openai", [{"role": "system", "content": settings.demo_system_prompt}, {"role": "user", "content": "नमस्ते"}])]


def test_system_prompt_carries_voice_gender(client, monkeypatch):
    monkeypatch.setattr(tts.engine, "voices", lambda: [{"voice_id": "fake", "sample_rate": SR, "gender": "F"}])
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_json({"type": "text", "text": "नमस्ते"})
        until(ws, {"turn_end"})
    assert calls["llm"][0][1][0]["content"] == f"{settings.demo_system_prompt} {llm_hint(Persona('female'))}"


def test_wav_turn_yields_transcript_first(client):
    wav = wav_bytes()
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_bytes(wav)
        items = until(ws, {"turn_end"})
    assert items[0]["type"] == "transcript" and items[0]["text"] == "मुझे लोन चाहिए" and items[0]["stt_ms"] >= 0
    assert types(items)[1] == "llm_delta" and "audio_start" in types(items)
    assert items[-1]["metrics"]["stt_ms"] == items[0]["stt_ms"]
    assert calls["stt"] == [("sarvam", len(wav))]
    assert calls["llm"][0][1][-1] == {"role": "user", "content": "मुझे लोन चाहिए"}


def test_empty_transcript_ends_turn_quietly(client, monkeypatch):
    async def silence(provider, wav):
        return ""

    monkeypatch.setattr(providers, "transcribe", silence)
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_bytes(wav_bytes())
        assert types(until(ws, {"turn_end"})) == ["turn_end"]
    assert calls["llm"] == []


def test_bad_utterances_rejected(client):
    long_wav = wav_bytes(seconds=31)
    assert len(long_wav) < demo.MAX_WAV_BYTES
    with client.websocket_connect("/demo/ws") as ws:
        for data, msg in ((long_wav, "30 s"), (b"\0" * (demo.MAX_WAV_BYTES + 1), "2 MB"), (b"not a wav file", "WAV"), (b"", "WAV")):
            ws.send_bytes(data)
            ev = ws.receive_json()
            assert ev["type"] == "error" and msg in ev["message"]
        ws.send_bytes(wav_bytes(seconds=29))  # still fine after the rejections
        assert until(ws, {"turn_end"})[0]["type"] == "transcript"


def test_config_message(client):
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_json({"type": "config", "stt": "openai", "llm": "sarvam", "voice": "default", "speed": 1.2,
                      "sample_rate": 16000, "system_prompt": "  be brief  "})
        assert ws.receive_json() == {"type": "ready", "config": {
            "stt": "openai", "llm": "sarvam", "voice": "fake", "speed": 1.2, "system_prompt": "be brief", "sample_rate": 16000}}
        ws.send_json({"type": "text", "text": "hi"})
        items = until(ws, {"turn_end"})
        assert items[types(items).index("audio_start")]["sample_rate"] == 16000
        assert all(len(f) == 1280 for f in [i for i in items if isinstance(i, bytes)][:-1])
        ws.send_json({"type": "config", "system_prompt": ""})  # empty = back to the default prompt
        assert ws.receive_json()["config"]["system_prompt"] == settings.demo_system_prompt
    assert calls["llm"][0] == ("sarvam", [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}])


@pytest.mark.parametrize("msg, needle", [
    ({"stt": "elevenlabs"}, "not available"),
    ({"llm": "nope"}, "not available"),
    ({"voice": "nope"}, "not found"),
    ({"speed": 5}, "speed"),
    ({"sample_rate": 12345}, "sample_rate"),
])
def test_config_message_rejects_bad_values(client, msg, needle):
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_json({"type": "config", **msg})
        ev = ws.receive_json()
        assert ev["type"] == "error" and needle in ev["message"]
        ws.send_json({"type": "config"})  # nothing was applied
        assert ws.receive_json()["config"]["speed"] == 1.0


def test_provider_error_keeps_connection_open(client):
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_json({"type": "text", "text": "boom"})
        assert ws.receive_json() == {"type": "error", "message": "openai HTTP 401: bad key", "scope": "turn"}
        ws.send_json({"type": "text", "text": "ठीक है"})
        assert until(ws, {"turn_end"})[-1]["reply"] == "".join(SHORT)


def test_unknown_and_malformed_messages(client):
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_text("[1]")
        assert ws.receive_json()["type"] == "error"
        ws.send_text("not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "dance"})
        assert "dance" in ws.receive_json()["message"]
        ws.send_json({"type": "text", "text": "   "})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "interrupt"})  # nothing running: no reply (the page did not wait for one)
        ws.send_json({"type": "reset"})
        assert ws.receive_json()["type"] == "ready"


def test_history_and_reset(client):
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_json({"type": "text", "text": "one"})
        until(ws, {"turn_end"})
        ws.send_json({"type": "text", "text": "two"})
        until(ws, {"turn_end"})
        ws.send_json({"type": "reset"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "text", "text": "three"})
        until(ws, {"turn_end"})
    roles = lambda i: [(m["role"], m["content"]) for m in calls["llm"][i][1]]  # noqa: E731
    assert roles(1) == [("system", settings.demo_system_prompt), ("user", "one"), ("assistant", "".join(SHORT)), ("user", "two")]
    assert roles(2) == [("system", settings.demo_system_prompt), ("user", "three")]


def test_history_is_capped(client):
    with client.websocket_connect("/demo/ws") as ws:
        for i in range(demo.MAX_HISTORY):  # 2 messages per turn
            ws.send_json({"type": "text", "text": f"q{i}"})
            until(ws, {"turn_end"})
    assert len(calls["llm"][-1][1]) == 1 + demo.MAX_HISTORY  # the system prompt + the newest 20 (incl. this question)


def test_interrupt_mid_turn(client):
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_json({"type": "text", "text": "long answer please"})
        first_frame(ws)
        ws.send_json({"type": "interrupt"})
        until(ws, {"interrupted"})
        ws.send_json({"type": "reset"})  # anything the old turn still sent would arrive before this reply
        assert types(until(ws, {"ready"})) == ["ready"]
        assert tts.stats["streams_active"] == 0  # the TTS stream was closed, not left running


def test_interrupted_reply_is_kept_in_history(client):
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_json({"type": "text", "text": "long answer please"})
        first_frame(ws)
        ws.send_json({"type": "interrupt"})
        until(ws, {"interrupted"})
        ws.send_json({"type": "text", "text": "next"})
        until(ws, {"turn_end"})
    hist = calls["llm"][1][1]
    assert [m["role"] for m in hist[1:]] == ["user", "assistant", "user"]
    assert hist[2]["content"].startswith("यह वाक्य नंबर 0 है।") and hist[2]["content"].endswith("…")


def test_barge_in_cancels_old_turn(client):
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_json({"type": "text", "text": "long answer please"})
        first_frame(ws)
        ws.send_json({"type": "text", "text": "short"})
        items = until(ws, {"turn_end"})
        t = types(items)
        cut = t.index("interrupted")
        assert t.count("interrupted") == 1 and t.count("turn_end") == 1
        after = items[cut + 1:]
        assert after[0] == {"type": "llm_delta", "text": SHORT[0]}  # nothing of the old turn after the acknowledgement
        assert all(i["type"] != "llm_delta" or i["text"] in SHORT for i in after if isinstance(i, dict))
        assert items[-1]["reply"] == "".join(SHORT)
        ws.send_json({"type": "reset"})
        assert until(ws, {"ready"})[0]["type"] == "ready"  # and the old turn never ends on its own later
    assert [m["role"] for m in calls["llm"][1][1][1:]] == ["user", "assistant", "user"]
    assert calls["llm"][1][1][2]["content"].endswith("…")


def test_barge_in_with_new_utterance(client):
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_json({"type": "text", "text": "long answer please"})
        first_frame(ws)
        ws.send_bytes(wav_bytes())
        items = until(ws, {"turn_end"})
    t = types(items)
    assert t.index("interrupted") < t.index("transcript") < t.index("audio_start", t.index("transcript"))


# --- segmenter ---
def segments(deltas):
    seg, out = demo.Segmenter(), []
    for d in deltas:
        out += seg.feed(d)
    return out + seg.flush()


def test_first_segment_is_cut_early():
    text = "जी हाँ, मैं आपकी मदद कर सकती हूँ, बताइए क्या चाहिए। धन्यवाद।"
    expect = ["जी हाँ, मैं आपकी मदद कर सकती हूँ,", "बताइए क्या चाहिए।", "धन्यवाद।"]
    assert segments([text]) == segments(list(text)) == expect  # same however the LLM chunks its tokens
    assert demo.cut("जी हाँ, ", True) == 0  # a comma before 12 chars is too early


def test_sentence_comma_and_length_rules():
    assert demo.cut("नमस्ते जी। ", True) == 0  # first segment needs >= 12 chars
    assert demo.cut("नमस्ते जी। ", False) == len("नमस्ते जी।")
    assert demo.cut("Hello there! Next", False) == len("Hello there!")
    assert demo.cut("What? ", False) == 5
    assert demo.cut("नमस्ते।", False) == 7  # । and ? ! also cut at the end of the buffer
    assert demo.cut("done.", False) == 0  # a bare . may still turn into "3.5"
    short_comma = "कुछ शब्द, फिर बाकी बात"
    assert demo.cut(short_comma, False) == 0  # non-first comma cut needs >= 40 chars
    long_comma = "यह एक काफ़ी लंबा वाक्य है जो चालीस अक्षर से बड़ा है, और आगे भी चलता है"
    assert demo.cut(long_comma, False) == long_comma.index(",") + 1
    words = " ".join(["शब्द"] * 60)  # no punctuation at all
    assert demo.cut(words, False) == words.index(" ", 119) + 1
    assert demo.cut(words[:119], False) == 0
    assert segments([words]) == segments(list(words)) and all(len(s) <= 125 for s in segments([words]))
    assert " ".join(segments([words])) == words


def test_numbers_do_not_split():
    text = "ब्याज 8.5 प्रतिशत है। Rs 1,000 का भुगतान बाकी है।"
    assert segments(list(text)) == ["ब्याज 8.5 प्रतिशत है।", "Rs 1,000 का भुगतान बाकी है।"]


def test_speakable_strips_markdown_and_emoji():
    assert demo.speakable("**नमस्ते** `code` # शीर्षक\n- पहला\n* दूसरा\n1. तीसरा 😀👍🏽") == "नमस्ते code शीर्षक पहला दूसरा तीसरा"
    assert demo.speakable("भुगतान ₹500 है ✅") == "भुगतान ₹500 है"
    assert demo.speakable("1. ") == demo.speakable("।") == demo.speakable("🙂") == demo.speakable("**") == ""
    assert segments(["**नमस्ते", "** जी। 😀 ठीक है।"]) == ["नमस्ते जी।", "ठीक है।"]  # markdown split across tokens


# --- vendor HTTP (mocked transport) ---
def mock_vendors(monkeypatch, handler):
    for p in ("openai", "gemini", "sarvam", "elevenlabs"):
        monkeypatch.setattr(settings, f"platform_{p}_api_key", f"key-{p}")
    monkeypatch.setattr(providers, "_client", httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def test_transcribe_requests(monkeypatch):
    seen = {}

    def handler(req):
        seen[req.url.host] = req
        return httpx.Response(200, json={"transcript": " नमस्ते "} if "sarvam" in req.url.host else {"text": " नमस्ते "})

    mock_vendors(monkeypatch, handler)

    async def main():
        return [await providers.transcribe(p, b"RIFF....WAVE") for p in ("sarvam", "openai", "elevenlabs")]

    assert asyncio.run(main()) == ["नमस्ते"] * 3
    sarvam, openai, eleven = seen["api.sarvam.ai"], seen["api.openai.com"], seen["api.elevenlabs.io"]
    assert (sarvam.url.path, sarvam.headers["api-subscription-key"]) == ("/speech-to-text", "key-sarvam")
    assert (openai.url.path, openai.headers["authorization"]) == ("/v1/audio/transcriptions", "Bearer key-openai")
    assert (eleven.url.path, eleven.headers["xi-api-key"]) == ("/v1/speech-to-text", "key-elevenlabs")
    for req, fields in ((sarvam, (b"saaras:v3", b"hi-IN")), (openai, (b"gpt-transcribe", b'name="languages[]"')),
                        (eleven, (b"scribe_v2", b'name="model_id"', b"tag_audio_events"))):
        assert b'name="file"; filename="audio.wav"' in req.content and all(f in req.content for f in fields)


def test_openai_legacy_stt_model_takes_single_language(monkeypatch):
    seen = []
    mock_vendors(monkeypatch, lambda req: seen.append(req) or httpx.Response(200, json={"text": "ok"}))
    monkeypatch.setattr(settings, "demo_openai_stt_model", "gpt-4o-mini-transcribe")
    assert asyncio.run(providers.transcribe("openai", b"x")) == "ok"
    assert b'name="language"' in seen[0].content and b"languages" not in seen[0].content


def test_chat_stream_requests_and_parsing(monkeypatch):
    seen = {}
    sse = "".join(f"data: {json.dumps({'choices': [{'delta': d}]})}\n\n" for d in ({"role": "assistant"}, {"content": "नम"}, {"content": None}, {"content": "स्ते"}))
    sse += 'data: {"choices":[]}\n\ndata: [DONE]\n\ndata: {"choices":[{"delta":{"content":"never"}}]}\n\n'

    def handler(req):
        seen[req.url.host] = req
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})

    mock_vendors(monkeypatch, handler)
    messages = [{"role": "user", "content": "hi"}]

    async def main():
        return {p: [d async for d in providers.chat_stream(p, messages)] for p in ("openai", "gemini", "sarvam")}

    assert asyncio.run(main()) == {p: ["नम", "स्ते"] for p in ("openai", "gemini", "sarvam")}
    for host, path, model in (("api.openai.com", "/v1/chat/completions", settings.demo_openai_llm_model),
                              ("generativelanguage.googleapis.com", "/v1beta/openai/chat/completions", settings.demo_gemini_llm_model),
                              ("api.sarvam.ai", "/v1/chat/completions", settings.demo_sarvam_llm_model)):
        req = seen[host]
        body = json.loads(req.content)
        assert req.url.path == path and body["model"] == model and body["stream"] is True and body["messages"] == messages
        assert "reasoning_effort" in body
        assert body["max_completion_tokens" if host == "api.openai.com" else "max_tokens"] == settings.demo_max_tokens == 300
        assert req.headers["authorization"].startswith("Bearer key-")
    assert seen["api.sarvam.ai"].headers["api-subscription-key"] == "key-sarvam"


def test_vendor_errors_are_clear_and_truncated(monkeypatch):
    body = "key-openai " + "x" * 500
    mock_vendors(monkeypatch, lambda req: httpx.Response(401, text=body))

    async def stt():
        await providers.transcribe("openai", b"x")

    async def llm():
        return [d async for d in providers.chat_stream("openai", [])]

    for coro in (stt, llm):
        with pytest.raises(providers.ProviderError) as e:
            asyncio.run(coro())
        msg = str(e.value)
        assert msg.startswith("openai HTTP 401: ***") and "key-openai" not in msg and len(msg) == len("openai HTTP 401: ") + 160


def test_vendor_transport_error_is_wrapped(monkeypatch):
    def handler(req):
        raise httpx.ConnectTimeout("connect timed out", request=req)

    mock_vendors(monkeypatch, handler)
    with pytest.raises(providers.ProviderError, match="sarvam STT request failed: ConnectTimeout"):
        asyncio.run(providers.transcribe("sarvam", b"x"))


# --- loopback gate, limits (fix 1) ---
def test_non_loopback_needs_keys_or_opt_in(client, monkeypatch):
    from app.main import app

    with TestClient(app, client=("10.1.2.3", 5)) as far:
        assert far.get("/demo").status_code == 404 and far.get("/demo/config").status_code == 404
        with pytest.raises(WebSocketDisconnect) as e, far.websocket_connect("/demo/ws"):
            pass
        assert e.value.code == 1008
        monkeypatch.setattr(settings, "demo_allow_open", True)
        assert far.get("/demo/config").status_code == 200
        monkeypatch.setattr(settings, "demo_allow_open", False)
        monkeypatch.setattr(settings, "api_keys", "secret")
        assert far.get("/demo/config", headers={"Authorization": "Bearer secret"}).status_code == 200
    monkeypatch.setattr(settings, "api_keys", "")
    assert client.get("/demo/config", headers={"X-Forwarded-For": "8.8.8.8"}).status_code == 404  # behind a proxy


def test_turn_rate_cap_is_per_connection(client, monkeypatch):
    monkeypatch.setattr(settings, "demo_turns_per_minute", 2)
    with client.websocket_connect("/demo/ws") as ws:
        for _ in range(2):
            ws.send_json({"type": "text", "text": "hi"})
            until(ws, {"turn_end"})
        ws.send_json({"type": "text", "text": "hi"})
        e = ws.receive_json()
        assert e["type"] == "error" and e["scope"] == "request" and "too many" in e["message"]


def test_turn_deadline(client, monkeypatch):
    monkeypatch.setattr(settings, "demo_turn_timeout_s", 0.3)
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_json({"type": "text", "text": "long answer please"})
        e = until(ws, {"error"})[-1]
        assert e["scope"] == "turn" and "timed out" in e["message"]
        ws.send_json({"type": "reset"})
        until(ws, {"ready"})


# --- access-log redaction (fix 2) ---
def test_access_log_redacts_api_key():
    import logging

    from app.core.logging import RedactApiKey

    rec = logging.LogRecord("uvicorn.access", 20, "", 0, '%s - "%s %s HTTP/%s" %d', ("1.2.3.4", "GET", "/demo/ws?api_key=s3cret&x=1", "1.1", 101), None)
    assert RedactApiKey().filter(rec)
    assert rec.args[2] == "/demo/ws?api_key=***&x=1" and "s3cret" not in rec.getMessage() and rec.args[4] == 101


# --- error scope (3), idle interrupt (4) ---
def test_request_error_does_not_end_the_turn(client):
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_json({"type": "text", "text": "long answer please"})
        first_frame(ws)
        ws.send_json({"type": "config", "voice": "nope"})
        ws.send_json({"type": "dance"})
        ws.send_json({"type": "interrupt"})
        errs = [i for i in until(ws, {"interrupted"}) if isinstance(i, dict) and i["type"] == "error"]
        assert [e["scope"] for e in errs] == ["request", "request"]


# --- history = what the user heard (5), hygiene (6) ---
def next_history(ws, text="next"):
    ws.send_json({"type": "text", "text": text})
    until(ws, {"turn_end"})
    return [(m["role"], m["content"]) for m in calls["llm"][-1][1][1:]]


def test_heard_nothing_drops_the_reply_while_running(client):
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_json({"type": "text", "text": "long answer please"})
        first_frame(ws)
        ws.send_json({"type": "interrupt", "heard_ms": 0})
        assert until(ws, {"interrupted"})[-1] == {"type": "interrupted", "cancelled": True}
        assert next_history(ws) == [("user", "next")]


def test_heard_ms_cuts_a_finished_reply(client, monkeypatch):
    async def two(provider, messages):
        calls["llm"].append((provider, [dict(m) for m in messages]))
        if messages[-1]["content"] == "two":
            yield "पहला वाक्य यहाँ है। "
            yield "दूसरा वाक्य यहाँ है।"
        else:
            yield "ठीक है"

    monkeypatch.setattr(providers, "chat_stream", two)
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_json({"type": "text", "text": "two"})
        until(ws, {"turn_end"})
        ws.send_json({"type": "interrupt", "heard_ms": 1200})  # first sentence ~0.9 s, both ~1.9 s
        assert ws.receive_json() == {"type": "interrupted", "cancelled": False}
        assert next_history(ws) == [("user", "two"), ("assistant", "पहला वाक्य यहाँ है।…"), ("user", "next")]
        ws.send_json({"type": "interrupt", "heard_ms": 0})  # nothing of "ठीक है" heard: the exchange is forgotten
        assert ws.receive_json()["cancelled"] is False
        assert next_history(ws, "again") == [("user", "two"), ("assistant", "पहला वाक्य यहाँ है।…"), ("user", "again")]


def test_heard_everything_or_idle_changes_nothing(client):
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_json({"type": "text", "text": "short"})
        until(ws, {"turn_end"})
        ws.send_json({"type": "interrupt", "heard_ms": 10**6})  # heard it all: no rewrite, no ack
        ws.send_json({"type": "interrupt"})
        assert next_history(ws)[:2] == [("user", "short"), ("assistant", "".join(SHORT))]


def test_empty_or_cancelled_before_any_text_leaves_no_dangling_user(client, monkeypatch):
    async def quiet(provider, messages):
        calls["llm"].append((provider, [dict(m) for m in messages]))
        if messages[-1]["content"] == "slow":
            await asyncio.sleep(5)
        if False:
            yield ""

    monkeypatch.setattr(providers, "chat_stream", quiet)
    with client.websocket_connect("/demo/ws") as ws:
        ws.send_json({"type": "text", "text": "empty"})
        assert until(ws, {"turn_end"})[-1]["reply"] == ""
        ws.send_json({"type": "text", "text": "slow"})
        ws.send_json({"type": "interrupt"})
        until(ws, {"interrupted"})
        ws.send_json({"type": "text", "text": "last"})
        until(ws, {"turn_end"})
    assert calls["llm"][-1][1][1:] == [{"role": "user", "content": "last"}]


# --- segmenter holds bare numbers / abbreviations (7) ---
def test_list_numbers_and_abbreviations_are_not_segments():
    assert segments(["1. पहला बिंदु यहाँ है। 2. दूसरा बिंदु यहाँ है।"]) == ["पहला बिंदु यहाँ है।", "दूसरा बिंदु यहाँ है।"]  # "1." / "2." never alone
    assert segments(["कीमत Rs. 500 है। Dr. शर्मा आएँगे। ठीक है।"]) == ["कीमत Rs. 500 है।", "Dr. शर्मा आएँगे।", "ठीक है।"]
    assert segments(["2. ", "पहला"]) == ["पहला"]  # streamed: "2." waits for the rest
    assert segments(["Dr."]) == ["Dr."]  # ...but is still spoken when nothing follows


# --- WAV format (8), vendor error text (9) ---
def test_wav_format_is_checked():
    assert demo.wav_error(wav_bytes(1)) is None
    assert "96000 Hz" in demo.wav_error(wav_bytes(1, rate=96000))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(b"\0" * 400)
    assert "2 channel" in demo.wav_error(buf.getvalue())


def test_vendor_error_shows_only_the_parsed_message(monkeypatch, caplog):
    body = json.dumps({"error": {"message": "Incorrect API key key-openai " + "y" * 400, "type": "invalid"}})
    mock_vendors(monkeypatch, lambda req: httpx.Response(401, text=body))
    with pytest.raises(providers.ProviderError) as e, caplog.at_level("WARNING"):
        asyncio.run(providers.transcribe("openai", b"x"))
    msg = str(e.value)
    assert msg.startswith("openai HTTP 401: Incorrect API key ***") and "invalid" not in msg and len(msg) <= len("openai HTTP 401: ") + 160
    logged = caplog.records[-1].extra_fields["body"]
    assert "invalid" in logged and "key-openai" not in logged  # full body logged, key scrubbed
