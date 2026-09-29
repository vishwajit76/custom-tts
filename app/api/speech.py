import binascii
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.core.config import settings
from app.core.security import require_api_key
from app.models.schemas import SpeechRequest, WsSpeak
from app.services import audio_utils, tts
from app.services.conditioning import UnsupportedControl, validate_condition

router = APIRouter(prefix="/v1/audio", dependencies=[Depends(require_api_key)])
MEDIA = {"wav": "audio/wav", "mp3": "audio/mpeg", "pcm": "audio/L16"}


def prepare_ex(req: SpeechRequest | WsSpeak) -> tuple[dict, list[str], list[str]]:
    """Validate a request against the loaded engine: (kwargs for tts.stream(), applied controls, ignored controls).

    Applied/ignored are empty unless the request carries a `condition`. Raises HTTPException (422 for controls
    the engine cannot honour with fallback="reject").
    """
    if not tts.engine.ready:
        raise HTTPException(503, "Model not loaded")
    text = req.input if isinstance(req, SpeechRequest) else req.text
    if len(text) > settings.max_input_chars:
        raise HTTPException(400, f"input exceeds {settings.max_input_chars} chars")
    cond = req.condition
    ref_b64 = getattr(req, "reference_audio", None) or (cond.reference_audio if cond else None)
    ref_text = getattr(req, "reference_text", None) or (cond.reference_text if cond else None)
    ref = None
    if ref_b64:
        if not tts.engine.supports_cloning:
            raise HTTPException(400, f"reference_audio needs ENGINES=qwen3 (running {settings.engines})")
        try:
            ref = audio_utils.decode_audio_b64(ref_b64)
        except (ValueError, binascii.Error, RuntimeError) as e:
            raise HTTPException(400, f"Bad reference_audio: {e}") from e
    voice = req.voice
    if ref is None:
        try:
            voice = tts.resolve_voice(req.voice)
        except KeyError:
            raise HTTPException(404, f"Voice '{req.voice}' not found") from None
    speed = cond.speed if cond and "speed" in cond.model_fields_set else req.speed
    applied, ignored = [], []
    if cond:
        try:
            eff, applied, ignored = validate_condition(
                cond.model_copy(update={"speed": speed, "reference_audio": ref_b64, "reference_text": ref_text}),
                tts.capabilities_for(None if ref is not None else voice), tts.engine_name(None if ref is not None else voice))
        except UnsupportedControl as e:
            raise HTTPException(422, {"message": str(e), "unsupported": e.controls}) from e
        speed = eff.speed
    sr = req.sample_rate or settings.default_sample_rate
    return dict(text=text, voice=voice, speed=speed, sample_rate=sr, ref=ref, ref_text=ref_text), applied, ignored


def prepare(req: SpeechRequest | WsSpeak) -> dict:
    return prepare_ex(req)[0]


def control_headers(req: SpeechRequest, applied: list[str], ignored: list[str]) -> dict:
    if not req.condition:
        return {}
    return {"X-TTS-Applied-Controls": ",".join(applied), "X-TTS-Ignored-Controls": ",".join(ignored)}


def _rid(request: Request) -> str:
    return request.headers.get("x-request-id") or uuid.uuid4().hex[:12]


@router.post("/speech")
async def speech(req: SpeechRequest, request: Request):
    rid = _rid(request)
    kwargs, applied, ignored = prepare_ex(req)
    try:
        wav = await tts.synthesize(**kwargs, request_id=rid)
    except tts.Overloaded as e:
        raise HTTPException(503, str(e), headers={"Retry-After": "1"}) from e
    sr = kwargs["sample_rate"]
    return Response(audio_utils.encode(wav, sr, req.response_format), media_type=MEDIA[req.response_format],
                    headers={"X-Request-ID": rid, "X-Sample-Rate": str(sr), **control_headers(req, applied, ignored)})


@router.post("/speech/stream")
async def speech_stream(req: SpeechRequest, request: Request):
    """Raw PCM s16le mono, streamed as each chunk is synthesized; rate in X-Sample-Rate. Format is always pcm."""
    rid = _rid(request)
    kwargs, applied, ignored = prepare_ex(req)
    gen = tts.stream(**kwargs, request_id=rid)
    try:
        first = await anext(gen)  # admission + first chunk before headers, so overload is a clean 503
    except tts.Overloaded as e:
        raise HTTPException(503, str(e), headers={"Retry-After": "1"}) from e
    except StopAsyncIteration:
        first = b""
    sr = kwargs["sample_rate"]

    async def body():
        yield first
        async for b in gen:
            yield b

    return StreamingResponse(body(), media_type=MEDIA["pcm"], headers={"X-Sample-Rate": str(sr), "X-Request-ID": rid, **control_headers(req, applied, ignored)})
