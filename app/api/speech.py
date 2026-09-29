import binascii
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.core.config import settings
from app.core.security import require_api_key
from app.models.schemas import SpeechRequest, WsSpeak
from app.services import audio_utils, routing, tts
from app.services import speaker_registry as sreg
from app.services.conditioning import UnsupportedControl, split_controls, validate_condition

router = APIRouter(prefix="/v1/audio", dependencies=[Depends(require_api_key)])
MEDIA = {"wav": "audio/wav", "mp3": "audio/mpeg", "pcm": "audio/L16"}


def _engine_items() -> list[tuple[str, object]]:
    return [(type(e).__name__.removesuffix("Engine").lower(), e) for e in getattr(tts.engine, "engines", [tts.engine])]


def resolve_speaker(cond, owner: str | None) -> dict:
    """condition.speaker_id -> {voice} (engine binding) or {ref, ref_text} (registry reference, cloning engines), or {}
    when unresolvable and fallback=ignore. 404 unknown/foreign speaker, 403 consent missing/not permitted, 422 no binding."""
    reg = sreg.get_registry()
    try:
        sp = reg.get(cond.speaker_id, owner if owner is not None else sreg.owner_id(""))
    except (sreg.SpeakerNotFound, sreg.SpeakerForbidden, sreg.SpeakerError) as e:
        raise HTTPException(404, "speaker not found") from e
    try:
        reg.check_use(sp, "tts")
    except sreg.SpeakerForbidden as e:
        raise HTTPException(403, str(e)) from e
    for name, e in _engine_items():
        bound = sp.engine_bindings.get(name)
        if bound and e.has_voice(bound):
            return {"voice": bound}
    best = reg.best_reference(sp) if getattr(tts.engine, "supports_cloning", False) else None
    if best is not None:
        try:
            reg.check_use(sp, "cloning")
        except sreg.SpeakerForbidden as e:
            raise HTTPException(403, str(e)) from e
        return {"ref": reg.read_reference(sp, best), "ref_text": best.transcript}
    if cond.fallback == "ignore":
        return {}
    engines = [n for n, _ in _engine_items()]
    raise HTTPException(422, {"message": f"speaker {sp.id!r} has no binding or usable reference for engine(s) {engines}; "
                                         "use fallback=\"ignore\" to synthesize with the default voice", "unsupported": ["speaker_id"]})


def prepare_ex(req: SpeechRequest | WsSpeak, owner: str | None = None, info: dict | None = None) -> tuple[dict, list[str], list[str]]:
    """Validate a request against the loaded engine: (kwargs for tts.stream(), applied controls, ignored controls).

    Applied/ignored are empty unless the request carries a `condition`. Raises HTTPException (422 for controls
    the engine cannot honour with fallback="reject").

    `info` (optional out-dict) receives routing details when a routing_policy was sent: {"policy", "routing":
    "policy"|"explicit", "engine", "voice"}. Policy routing only happens for voice="default" without reference/speaker.
    """
    if not tts.engine.ready:
        raise HTTPException(503, "Model not loaded")
    text = req.input if isinstance(req, SpeechRequest) else req.text
    if len(text) > settings.max_input_chars:
        raise HTTPException(400, f"input exceeds {settings.max_input_chars} chars")
    cond = req.condition
    ref_b64 = getattr(req, "reference_audio", None) or (cond.reference_audio if cond else None)
    ref_text = getattr(req, "reference_text", None) or (cond.reference_text if cond else None)
    req_ref_text = ref_text
    ref = None
    voice = req.voice
    speaker_ok = False
    if cond and cond.speaker_id:
        if ref_b64:
            raise HTTPException(400, "speaker_id and reference_audio are mutually exclusive")
        res = resolve_speaker(cond, owner)
        speaker_ok = bool(res)
        if "voice" in res:
            voice = res["voice"]
        elif "ref" in res:
            ref, ref_text = res["ref"], res["ref_text"]
    if ref_b64:
        if not tts.engine.supports_cloning:
            raise HTTPException(400, f"reference_audio needs ENGINES=qwen3 (running {settings.engines})")
        try:
            ref = audio_utils.decode_audio_b64(ref_b64)
        except (ValueError, binascii.Error, RuntimeError) as e:
            raise HTTPException(400, f"Bad reference_audio: {e}") from e
    policy = req.routing_policy or (cond.routing_policy if cond else None)
    extra_ignored: list[str] = []
    if policy:
        if ref is None and not speaker_ok and voice == "default":
            try:
                r = routing.route(policy, cond, _engine_items(), routing.parse_tiers(settings.engine_latency_tiers),
                                  settings.default_voice, settings.dsp_prosody)
            except routing.NoCapableEngine as e:
                raise HTTPException(422, {"message": str(e), "unsupported": e.missing}) from e
            voice = r.voice
            if r.degraded:
                extra_ignored.append("routing_policy")
            if info is not None:
                info.update(policy=policy, routing="policy", engine=r.engine, voice=r.voice)
        elif info is not None:
            info.update(policy=policy, routing="explicit")  # explicit voice / speaker / reference wins over the policy
    if ref is None:
        try:
            voice = tts.resolve_voice(voice)
        except KeyError:
            raise HTTPException(404, f"Voice '{req.voice}' not found") from None
    speed = cond.speed if cond and "speed" in cond.model_fields_set else req.speed
    applied, ignored = [], []
    extra: dict = {}
    if cond:
        caps = tts.capabilities_for(None if ref is not None else voice)
        try:
            eff, applied, ignored = validate_condition(
                cond.model_copy(update={"speed": speed, "reference_audio": ref_b64, "reference_text": req_ref_text}),
                caps, tts.engine_name(None if ref is not None else voice), speaker_ok, settings.dsp_prosody)
        except UnsupportedControl as e:
            raise HTTPException(422, {"message": str(e), "unsupported": e.controls}) from e
        speed = eff.speed
        native, post = split_controls(eff, caps, settings.dsp_prosody)
        if native:
            extra["controls"] = native
        if post:
            extra["dsp_controls"] = post
    sr = req.sample_rate or settings.default_sample_rate
    if info is not None and "voice" in info:
        info["voice"] = voice
    return dict(text=text, voice=voice, speed=speed, sample_rate=sr, ref=ref, ref_text=ref_text, **extra), applied, ignored + extra_ignored


def prepare(req: SpeechRequest | WsSpeak) -> dict:
    return prepare_ex(req)[0]


def control_headers(req: SpeechRequest, applied: list[str], ignored: list[str], info: dict | None = None) -> dict:
    h = {}
    if req.condition:
        h = {"X-TTS-Applied-Controls": ",".join(applied), "X-TTS-Ignored-Controls": ",".join(ignored)}
    if info and info.get("policy"):
        h["X-TTS-Routing"] = info["routing"] + (f";engine={info['engine']};voice={info['voice']}" if info["routing"] == "policy" else "")
    return h


def _rid(request: Request) -> str:
    return request.headers.get("x-request-id") or uuid.uuid4().hex[:12]


@router.post("/speech")
async def speech(req: SpeechRequest, request: Request, key: str = Depends(require_api_key)):
    rid = _rid(request)
    info: dict = {}
    kwargs, applied, ignored = prepare_ex(req, sreg.owner_id(key), info)
    try:
        wav = await tts.synthesize(**kwargs, request_id=rid)
    except tts.Overloaded as e:
        raise HTTPException(503, str(e), headers={"Retry-After": "1"}) from e
    sr = kwargs["sample_rate"]
    return Response(audio_utils.encode(wav, sr, req.response_format), media_type=MEDIA[req.response_format],
                    headers={"X-Request-ID": rid, "X-Sample-Rate": str(sr), **control_headers(req, applied, ignored, info)})


@router.post("/speech/stream")
async def speech_stream(req: SpeechRequest, request: Request, key: str = Depends(require_api_key)):
    """Raw PCM s16le mono, streamed as each chunk is synthesized; rate in X-Sample-Rate. Format is always pcm."""
    rid = _rid(request)
    info: dict = {}
    kwargs, applied, ignored = prepare_ex(req, sreg.owner_id(key), info)
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

    return StreamingResponse(body(), media_type=MEDIA["pcm"], headers={"X-Sample-Rate": str(sr), "X-Request-ID": rid, **control_headers(req, applied, ignored, info)})
