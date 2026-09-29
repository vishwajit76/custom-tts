"""Voice-agent test page: browser mic -> vendor STT -> vendor LLM (streamed) -> our TTS -> browser.

GET /demo         the page (holds no secrets)
GET /demo/config  providers that have a key, defaults, voices (API key required)
WS  /demo/ws      auth like /v1/audio/ws (Authorization Bearer or ?api_key=)
With API_KEYS empty all three answer loopback clients only (404 / WS 1008) unless DEMO_ALLOW_OPEN=true.

Client -> server:
  JSON {"type":"config","stt","llm","voice","speed","system_prompt","sample_rate"}  any subset -> {"type":"ready","config"}
  JSON {"type":"text","text"}   typed user turn
  JSON {"type":"interrupt","heard_ms"?}  cancels the running turn -> {"type":"interrupted","cancelled":true}. heard_ms = ms of
        this reply's audio the user actually heard: the stored assistant message is cut to it ("cancelled":false if only that
        happened). An idle interrupt without a cut gets no reply.
  JSON {"type":"reset"}         clear history -> ready
  binary: one utterance as a WAV file (16 kHz mono PCM16, <= 30 s, <= 2 MB)
  A text/utterance that arrives while a turn runs is a barge-in: the running turn is cancelled first.
Server -> client:
  {"type":"transcript","text","stt_ms"}  {"type":"llm_delta","text"}  {"type":"audio_start","sample_rate"}
  binary PCM s16le mono, 40 ms frames  {"type":"turn_end","reply","metrics"}  {"type":"interrupted"}  {"type":"error","message","scope"}  scope "turn" = the turn ended, "request" = only that message was rejected
"""
import asyncio
import io
import json
import logging
import re
import time
import uuid
import wave
from collections import deque
from contextlib import aclosing
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from starlette.requests import HTTPConnection
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError

from app.core.config import settings
from app.core.security import authorize, require_api_key
from app.models.schemas import VOICE_ID, SampleRate
from app.services import providers, tts

log = logging.getLogger(__name__)
PAGE = Path(__file__).resolve().parent.parent / "static" / "demo.html"
MAX_HISTORY = 20  # messages kept besides the system prompt
MAX_WAV_BYTES = 2 * 1024 * 1024
MAX_UTTERANCE_S = 30  # also Sarvam's limit for its synchronous API
FRAME_MS = 40


def _enabled(conn: HTTPConnection) -> None:
    """Demo on, and (with no API keys configured) only loopback clients: the page spends the vendor keys."""
    if not settings.demo_enabled:
        raise HTTPException(404, "Not found")
    if not settings.keys and not settings.demo_allow_open:
        host = conn.client.host if conn.client else ""
        # a proxy in front of us shows up as loopback: X-Forwarded-For means the real client is elsewhere
        if host not in ("127.0.0.1", "::1") or "x-forwarded-for" in conn.headers:
            raise HTTPException(404, "Not found")


router = APIRouter(prefix="/demo")


@router.get("", dependencies=[Depends(_enabled)])
def page():
    return FileResponse(PAGE, media_type="text/html")


@router.get("/config", dependencies=[Depends(_enabled), Depends(require_api_key)])
def config():
    return {
        **providers.available_providers(),
        "defaults": {
            "stt": providers.default_provider("stt"), "llm": providers.default_provider("llm"),
            "voice": settings.default_voice, "system_prompt": settings.demo_system_prompt,
        },
        "voices": tts.engine.voices(),
    }


# --- turning streamed LLM text into speakable segments ---
_END = re.compile(r"(?:[।?!]+|\.+)(?=\s)|[।?!]+$")
_COMMA = re.compile(r",(?=\s)")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿⬀-⯿️]")
_HOLD = re.compile(r"\d+\.|\S{1,3}\.")  # "2." "Dr." "No.": a list number or abbreviation, not a sentence
_BULLET = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+", re.M)


def cut(buf: str, first: bool) -> int:
    """Index just past the next speakable segment in the text buffered so far, or 0 if none is ready.

    Sentence ends cut at once, commas from 40 chars on, any space from 120 chars on; the first segment cuts at a
    sentence end or comma from 12 chars on so speech starts sooner. An end mark needs a following space (or, for
    । ? !, the end of the buffer) so that "3.5" and "1,000" never split; a bare "2." or "Dr." waits for more text.
    """
    end = next((m.end() for m in _END.finditer(buf) if m.end() >= (12 if first else 1) and not _HOLD.fullmatch(buf[:m.end()].strip())), 0)
    comma = _COMMA.search(buf, (12 if first else 40) - 1)
    if comma and (not end or comma.end() < end):
        end = comma.end()
    if not end and (sp := buf.find(" ", 119)) > 0:
        end = sp + 1
    return end


def speakable(text: str) -> str:
    """Drop markdown symbols and emojis; '' when nothing pronounceable is left."""
    text = _BULLET.sub("", text)
    text = re.sub(r"\s+", " ", re.sub(r"[*#`]", "", _EMOJI.sub("", text))).strip()
    return text if re.search(r"\w", text) else ""


class Segmenter:
    def __init__(self):
        self.buf, self.sent = "", False

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out = []
        while end := cut(self.buf, not self.sent):
            seg, self.buf = speakable(self.buf[:end]), self.buf[end:]
            if seg:
                out.append(seg)
                self.sent = True
        return out

    def flush(self) -> list[str]:
        seg, self.buf = speakable(self.buf), ""
        return [seg] if seg else []


_PERSONA = {
    "F": "You are a woman: use feminine first-person verb forms (करती हूँ, रही हूँ, सकती हूँ).",
    "M": "You are a man: use masculine first-person verb forms (करता हूँ, रहा हूँ, सकता हूँ).",
}


def persona(voice: str) -> str:
    """Hindi verbs agree with the speaker's gender, so tell the LLM which voice speaks ('' when unknown)."""
    return _PERSONA.get(next((v.get("gender") for v in tts.engine.voices() if v["voice_id"] == voice), None), "")


# --- websocket ---
class Cfg(BaseModel):
    stt: str | None = None
    llm: str | None = None
    voice: str | None = Field(None, pattern=VOICE_ID)
    speed: float | None = Field(None, ge=0.5, le=2.0)
    system_prompt: str | None = Field(None, max_length=4000)
    sample_rate: SampleRate | None = None


def wav_error(data: bytes) -> str | None:
    if len(data) > MAX_WAV_BYTES:
        return "utterance exceeds 2 MB"
    try:
        with wave.open(io.BytesIO(data)) as w:
            seconds = w.getnframes() / w.getframerate()
            w_ch, w_width, w_rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
    except (wave.Error, EOFError, ZeroDivisionError):
        return "expected one WAV file (16 kHz mono PCM16)"
    if (w_ch, w_width, w_rate) != (1, 2, 16000):
        return f"expected 16 kHz mono 16-bit PCM WAV, got {w_rate} Hz, {w_ch} channel(s), {8 * w_width}-bit"
    return f"utterance exceeds {MAX_UTTERANCE_S} s" if seconds > MAX_UTTERANCE_S else None


def _ms(since: float) -> int:
    return round((time.monotonic() - since) * 1000)


def _validate(msg: dict) -> dict:
    """Config message -> the fields to change. Raises ValueError with a user-facing message."""
    try:
        upd = Cfg.model_validate(msg).model_dump(exclude_none=True)
    except ValidationError as e:
        err = e.errors(include_url=False)[0]
        raise ValueError(f"{'.'.join(map(str, err['loc']))}: {err['msg']}") from None
    avail = providers.available_providers()
    for kind in ("stt", "llm"):
        if kind in upd and upd[kind] not in avail[kind]:
            raise ValueError(f"{kind} provider {upd[kind]!r} not available (have: {', '.join(avail[kind]) or 'none'})")
    if "voice" in upd:
        try:
            upd["voice"] = tts.resolve_voice(upd["voice"])
        except KeyError:
            raise ValueError(f"voice {upd['voice']!r} not found") from None
    if "system_prompt" in upd:
        upd["system_prompt"] = upd["system_prompt"].strip() or settings.demo_system_prompt
    return upd


@router.websocket("/ws")
async def demo_ws(ws: WebSocket):
    auth = ws.headers.get("authorization", "")
    key = auth[7:] if auth.lower().startswith("bearer ") else ws.query_params.get("api_key", "")
    try:
        _enabled(ws)
        authorize(key)
    except HTTPException as e:
        await ws.close(code=1008, reason=e.detail)
        return
    await ws.accept()
    cfg = {
        "stt": providers.default_provider("stt"), "llm": providers.default_provider("llm"),
        "voice": settings.default_voice, "speed": 1.0, "system_prompt": settings.demo_system_prompt,
        "sample_rate": settings.default_sample_rate,
    }
    history: list[dict] = []
    lock = asyncio.Lock()  # a JSON event and a binary frame from different tasks must not reorder
    turn: asyncio.Task | None = None

    async def send(data: dict | bytes) -> None:
        async with lock:
            try:
                await (ws.send_bytes(data) if isinstance(data, bytes) else ws.send_text(json.dumps(data, ensure_ascii=False)))
            except (RuntimeError, WebSocketDisconnect):
                pass  # connection already closing

    async def error(message: str, scope: str = "request") -> None:
        """scope "turn": the running turn is over. "request": only this message was rejected; the turn (if any) goes on."""
        await send({"type": "error", "message": message, "scope": scope})

    def remember(role: str, content: str) -> dict:
        history.append(msg := {"role": role, "content": content})
        del history[:-MAX_HISTORY]
        return msg

    # The current (or latest) turn's reply as spoken: (text, cumulative audio ms at its end), and its history entry.
    last: dict = {}
    starts: deque[float] = deque()  # turn start times in the last minute

    def rate_limited() -> bool:
        now = time.monotonic()
        while starts and now - starts[0] > 60:
            starts.popleft()
        if len(starts) >= settings.demo_turns_per_minute:
            return True
        starts.append(now)
        return False

    def apply_heard(heard_ms: int) -> bool:
        """The user heard only the first heard_ms of the reply: cut the stored assistant message to that. True if changed."""
        msg = last.get("msg")
        if msg is None or not any(h is msg for h in history):
            return False
        kept = [t for t, end in last["segs"] if end <= heard_ms]
        if last["finished"] and len(kept) == len(last["segs"]):
            return False  # heard it all
        i = next(i for i, h in enumerate(history) if h is msg)
        if kept:
            msg["content"] = " ".join(kept) + "…"
        else:  # nothing heard: forget the reply and the question that got no answer
            del history[i - 1 if i and history[i - 1]["role"] == "user" else i : i + 1]
        last.clear()
        return True

    async def run_turn(wav: bytes | None, text: str | None) -> None:
        try:
            async with asyncio.timeout(settings.demo_turn_timeout_s):
                await turn_body(wav, text)
        except TimeoutError:
            await error(f"turn timed out after {settings.demo_turn_timeout_s:g} s", "turn")

    async def turn_body(wav: bytes | None, text: str | None) -> None:
        rid, t0, c = uuid.uuid4().hex[:12], time.monotonic(), dict(cfg)
        m = {"stt_ms": None, "llm_ttft_ms": 0, "tts_ttfa_ms": None, "e2e_ms": None, "audio_ms": 0}
        reply: list[str] = []
        asked = finished = False
        umsg = None
        segs: list[list] = []  # [text, audio ms at its end]
        last.clear()
        last.update(segs=segs, finished=False, msg=None)
        try:
            if not c["llm"] or (wav is not None and not c["stt"]):
                raise providers.ProviderError("no STT/LLM provider configured: set PLATFORM_*_API_KEY in .env")
            if wav is not None:
                t = time.monotonic()
                text = await providers.transcribe(c["stt"], wav)
                m["stt_ms"] = _ms(t)
                if not text:  # silence or noise: nothing to answer
                    await send({"type": "turn_end", "reply": "", "metrics": m})
                    return
                await send({"type": "transcript", "text": text, "stt_ms": m["stt_ms"]})
            umsg = remember("user", text)
            asked = True
            messages = [{"role": "system", "content": f"{c['system_prompt']} {persona(c['voice'])}".strip()}, *history]
            segments: asyncio.Queue[str | None] = asyncio.Queue()

            async def produce() -> None:
                seg, t = Segmenter(), time.monotonic()
                async with aclosing(providers.chat_stream(c["llm"], messages)) as deltas:
                    async for d in deltas:
                        if not reply:
                            m["llm_ttft_ms"] = _ms(t)
                        reply.append(d)
                        await send({"type": "llm_delta", "text": d})
                        for s in seg.feed(d):
                            segments.put_nowait(s)
                for s in seg.flush():
                    segments.put_nowait(s)
                segments.put_nowait(None)

            async def consume() -> None:
                sr, t_seg, nbytes = c["sample_rate"], None, 0
                while (s := await segments.get()) is not None:
                    t_seg = t_seg or time.monotonic()
                    segs.append(entry := [s, round(nbytes / 2 / sr * 1000)])
                    stream = tts.stream(s, c["voice"], c["speed"], sample_rate=sr, frame_ms=FRAME_MS, request_id=rid)
                    async with aclosing(stream) as frames:
                        async for frame in frames:
                            if not nbytes:
                                await send({"type": "audio_start", "sample_rate": sr})
                            await send(frame)
                            if not nbytes:
                                m["tts_ttfa_ms"], m["e2e_ms"] = _ms(t_seg), _ms(t0)
                            nbytes += len(frame)
                            entry[1] = round(nbytes / 2 / sr * 1000)
                m["audio_ms"] = round(nbytes / 2 / sr * 1000)

            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(produce())
                    tg.create_task(consume())
            except ExceptionGroup as eg:
                raise eg.exceptions[0] from None
            finished = last["finished"] = True
            if "".join(reply).strip():
                last["msg"] = remember("assistant", "".join(reply))
            await send({"type": "turn_end", "reply": "".join(reply), "metrics": m})
            log.info("demo turn", extra={"extra_fields": {"request_id": rid, "stt": c["stt"], "llm": c["llm"], **m}})
        except providers.ProviderError as e:
            await error(str(e), "turn")
        except tts.Overloaded as e:
            await error(f"TTS {e}", "turn")
        except Exception:
            log.exception("demo turn failed", extra={"extra_fields": {"request_id": rid}})
            await error("internal error", "turn")
        finally:
            if asked and not finished and "".join(reply).strip():  # interrupted or failed mid-reply: keep what was said
                last["msg"] = remember("assistant", "".join(reply) + "…")
            if asked and not last.get("msg") and history and history[-1] is umsg:
                history.pop()  # no reply at all: never leave two user messages in a row

    async def stop_turn() -> bool:
        """Cancel the running turn (LLM stream and TTS) and wait until it has fully stopped, then acknowledge."""
        if turn is None or turn.done():
            return False
        turn.cancel()
        await asyncio.wait({turn})
        await send({"type": "interrupted", "cancelled": True})
        return True

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if err := wav_error(msg["bytes"]):
                    await error(err)
                    continue
                if rate_limited():
                    await error("too many turns per minute")
                    continue
                await stop_turn()
                turn = asyncio.create_task(run_turn(msg["bytes"], None))
                continue
            try:
                body = json.loads(msg["text"])
                kind = body.get("type")
            except (ValueError, AttributeError, TypeError):
                await error("expected a JSON object")
                continue
            if kind == "config":
                try:
                    cfg.update(_validate(body))
                except ValueError as e:
                    await error(str(e))
                else:
                    await send({"type": "ready", "config": cfg})
            elif kind == "text":
                text = body.get("text")
                if not isinstance(text, str) or not text.strip() or len(text) > settings.max_input_chars:
                    await error(f"text must be 1-{settings.max_input_chars} characters")
                    continue
                if rate_limited():
                    await error("too many turns per minute")
                    continue
                await stop_turn()
                turn = asyncio.create_task(run_turn(None, text.strip()))
            elif kind == "interrupt":
                cancelled = await stop_turn()
                heard = body.get("heard_ms")
                cut = isinstance(heard, (int, float)) and not isinstance(heard, bool) and apply_heard(heard)
                if cut and not cancelled:  # an idle interrupt gets no reply: the page must not count an ack it did not wait for
                    await send({"type": "interrupted", "cancelled": False})
            elif kind == "reset":
                await stop_turn()
                history.clear()
                last.clear()
                await send({"type": "ready", "config": cfg})
            else:
                await error(f"unknown type {kind!r}")
    except (WebSocketDisconnect, RuntimeError):  # RuntimeError: receive after the socket closed
        pass
    finally:
        if turn:
            turn.cancel()
