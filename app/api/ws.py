"""WebSocket streaming TTS for calling.

Client -> server (JSON text):
  {"type":"speak","id":"r1","text":"...","voice":"default","speed":1.0,"sample_rate":8000,"frame_ms":40,
   "condition":{"emotion":"calm","fallback":"ignore"}}     (condition optional; see docs/voice-system.md)
  {"type":"cancel","id":"r1"}   cancel one request (playing or queued)
  {"type":"cancel"}             cancel everything (barge-in)
Server -> client:
  {"type":"start","id","sample_rate","encoding":"pcm_s16le"[,"applied_controls":[..],"ignored_controls":[..]]}, binary PCM frames..., {"type":"end","id","audio_ms","ttfa_ms"}
  {"type":"cancelled","id"} | {"type":"error","id","code","message"}
Speak requests on one connection play in order, so a client can send LLM output sentence by sentence.
"""
import asyncio
import json
import logging
import time

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.api.speech import prepare_ex
from app.core.security import authorize
from app.services.speaker_registry import owner_id
from app.models.schemas import WsSpeak
from app.services import tts

log = logging.getLogger(__name__)
router = APIRouter()
MAX_PENDING = 32
SEND_TIMEOUT_S = 10  # ponytail: a client that stops reading for this long is dropped, freeing its slot
_CODES = {400: "bad_request", 404: "not_found", 503: "unavailable", 422: "unsupported_control", 403: "forbidden"}


@router.websocket("/v1/audio/ws")
async def ws_tts(ws: WebSocket):
    auth = ws.headers.get("authorization", "")
    key = auth[7:] if auth.lower().startswith("bearer ") else ws.query_params.get("api_key", "")
    try:
        authorize(key)
    except HTTPException as e:
        await ws.close(code=1008, reason=e.detail)
        return
    await ws.accept()
    pending: asyncio.Queue[WsSpeak] = asyncio.Queue(MAX_PENDING)
    dropped: set[str] = set()
    cur: dict = {"id": None, "task": None}

    async def send(obj: dict) -> None:
        try:
            await ws.send_text(json.dumps(obj, ensure_ascii=False))
        except (RuntimeError, WebSocketDisconnect):
            pass  # connection already closing

    async def run(m: WsSpeak) -> None:
        t0 = time.monotonic()
        try:
            info: dict = {}
            kwargs, applied, ignored = prepare_ex(m, owner_id(key), info)
            sr = kwargs["sample_rate"]
            ttfa, nbytes = None, 0
            start = {"type": "start", "id": m.id, "sample_rate": sr, "encoding": "pcm_s16le"}
            if m.condition:  # only when conditioning was requested, so old clients see the old message
                start |= {"applied_controls": applied, "ignored_controls": ignored}
            if info.get("policy"):  # only when a routing_policy was sent
                start |= {"routing": info["routing"], **({"routed_engine": info["engine"], "routed_voice": info["voice"]} if info["routing"] == "policy" else {})}
            await send(start)
            async for frame in tts.stream(**kwargs, frame_ms=m.frame_ms, request_id=m.id):
                ttfa = ttfa or time.monotonic() - t0
                nbytes += len(frame)
                # asyncio.timeout, not wait_for: on Python 3.11 wait_for swallows a cancel that lands right as the send
                # completes, so barge-in right after a frame was ignored (found by bench cancel test, 2026-09-29)
                async with asyncio.timeout(SEND_TIMEOUT_S):
                    await ws.send_bytes(frame)
            await send({"type": "end", "id": m.id, "audio_ms": round(nbytes / 2 / sr * 1000), "ttfa_ms": round((ttfa or 0) * 1000)})
        except asyncio.CancelledError:
            await send({"type": "cancelled", "id": m.id})
        except HTTPException as e:
            detail = e.detail["message"] if isinstance(e.detail, dict) else e.detail
            await send({"type": "error", "id": m.id, "code": _CODES.get(e.status_code, "error"), "message": detail,
                        **({"unsupported": e.detail["unsupported"]} if isinstance(e.detail, dict) else {})})
        except tts.Overloaded as e:
            await send({"type": "error", "id": m.id, "code": "overloaded", "message": str(e)})
        except TimeoutError:
            log.warning("ws client too slow, closing", extra={"extra_fields": {"request_id": m.id}})
            await ws.close(code=1008, reason="client not reading")
        except Exception:
            log.exception("ws synthesis failed", extra={"extra_fields": {"request_id": m.id}})
            await send({"type": "error", "id": m.id, "code": "internal", "message": "synthesis failed"})

    async def pump() -> None:
        while True:
            m = await pending.get()
            if m.id in dropped:
                dropped.discard(m.id)
                await send({"type": "cancelled", "id": m.id})
                continue
            cur["id"], cur["task"] = m.id, asyncio.create_task(run(m))
            await asyncio.wait({cur["task"]})  # a cancelled request must not stop the pump
            cur["id"], cur["task"] = None, None

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
                kind = msg.get("type")
            except (ValueError, AttributeError):
                await send({"type": "error", "code": "bad_request", "message": "expected a JSON object"})
                continue
            if kind == "speak":
                try:
                    m = WsSpeak.model_validate(msg)
                except ValidationError as e:
                    await send({"type": "error", "id": msg.get("id"), "code": "bad_request", "message": e.errors(include_url=False)[0]["msg"]})
                    continue
                try:
                    pending.put_nowait(m)
                except asyncio.QueueFull:
                    await send({"type": "error", "id": m.id, "code": "queue_full", "message": f"max {MAX_PENDING} queued requests"})
            elif kind == "cancel":
                rid = msg.get("id")
                if rid is None:  # barge-in: drop everything
                    while not pending.empty():
                        await send({"type": "cancelled", "id": pending.get_nowait().id})
                    if cur["task"]:
                        cur["task"].cancel()
                elif rid == cur["id"]:
                    cur["task"].cancel()
                else:
                    dropped.add(rid)
            else:
                await send({"type": "error", "code": "bad_request", "message": f"unknown type {kind!r}"})
    except (WebSocketDisconnect, RuntimeError):  # RuntimeError: receive after we closed a slow client
        pass
    finally:
        pump_task.cancel()
        if cur["task"]:
            cur["task"].cancel()
