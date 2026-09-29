import binascii
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.core.config import settings
from app.core.security import require_api_key
from app.models.schemas import SpeechRequest
from app.services import audio_utils, tts

router = APIRouter(prefix="/v1/audio", dependencies=[Depends(require_api_key)])
MEDIA = {"wav": "audio/wav", "mp3": "audio/mpeg", "pcm": "audio/L16"}


def prepare(req: SpeechRequest) -> dict:
    """Validate a request against the loaded engine; returns kwargs for tts.stream(). Raises HTTPException."""
    if not tts.engine.ready:
        raise HTTPException(503, "Model not loaded")
    if len(req.input) > settings.max_input_chars:
        raise HTTPException(400, f"input exceeds {settings.max_input_chars} chars")
    ref = None
    if req.reference_audio:
        if not tts.engine.supports_cloning:
            raise HTTPException(400, f"reference_audio needs ENGINES=qwen3 (running {settings.engines})")
        try:
            ref = audio_utils.decode_audio_b64(req.reference_audio)
        except (ValueError, binascii.Error, RuntimeError) as e:
            raise HTTPException(400, f"Bad reference_audio: {e}") from e
    voice = req.voice
    if ref is None:
        try:
            voice = tts.resolve_voice(req.voice)
        except KeyError:
            raise HTTPException(404, f"Voice '{req.voice}' not found") from None
    sr = req.sample_rate or settings.default_sample_rate
    return dict(text=req.input, voice=voice, speed=req.speed, sample_rate=sr, ref=ref, ref_text=req.reference_text)


def _rid(request: Request) -> str:
    return request.headers.get("x-request-id") or uuid.uuid4().hex[:12]


@router.post("/speech")
async def speech(req: SpeechRequest, request: Request):
    rid = _rid(request)
    kwargs = prepare(req)
    try:
        wav = await tts.synthesize(**kwargs, request_id=rid)
    except tts.Overloaded as e:
        raise HTTPException(503, str(e), headers={"Retry-After": "1"}) from e
    sr = kwargs["sample_rate"]
    return Response(audio_utils.encode(wav, sr, req.response_format), media_type=MEDIA[req.response_format],
                    headers={"X-Request-ID": rid, "X-Sample-Rate": str(sr)})


@router.post("/speech/stream")
async def speech_stream(req: SpeechRequest, request: Request):
    """Raw PCM s16le mono, streamed as each chunk is synthesized; rate in X-Sample-Rate. Format is always pcm."""
    rid = _rid(request)
    kwargs = prepare(req)
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

    return StreamingResponse(body(), media_type=MEDIA["pcm"], headers={"X-Sample-Rate": str(sr), "X-Request-ID": rid})
