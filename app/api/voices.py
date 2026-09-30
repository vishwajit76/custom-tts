import re

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from pydantic import TypeAdapter, constr

from app.core.security import require_api_key
from app.models.schemas import VOICE_ID, SampleRate
from app.services import audio_utils, tts
from app.services import speaker_registry as sreg
from app.core.config import settings
from app.services.conditioning import DEFAULT_CAPABILITIES, Emotion, Role, Style, control_kinds
from app.services.routing import POLICIES, parse_tiers

router = APIRouter(prefix="/v1/voices", dependencies=[Depends(require_api_key)])
capabilities_router = APIRouter(prefix="/v1/capabilities", dependencies=[Depends(require_api_key)])
VoiceId = constr(pattern=f"^{VOICE_ID}$")  # pydantic patterns are searches: unanchored, "../evil" matched via "evil" and later raised a 500


def _cloning_engine():
    if not tts.engine.supports_cloning:
        raise HTTPException(400, "voice upload needs ENGINES=qwen3; for piper, fine-tune a voice (docs/training.md) and drop the .onnx into MODELS_DIR")
    return tts.engine


def _mirror_upload(owner: str, voice_id: str, consent_record_id: str | None, granted_by: str | None) -> None:
    """Legacy qwen3 upload also registers a registry speaker bound to the clip (consent pending unless the optional
    consent_record_id + granted_by form fields are sent). The `voice=` request path is unchanged; `speaker_id` needs consent."""
    reg = sreg.get_registry()
    engine_name = type(tts.engine).__name__.removesuffix("Engine").lower()
    try:
        try:
            reg.create(owner, voice_id, voice_id, engine_bindings={engine_name: voice_id})
        except sreg.SpeakerExists:
            reg.bind_engine(voice_id, owner, engine_name, voice_id)  # atomic: no lost update against a concurrent PATCH/upload
        if consent_record_id and granted_by:
            reg.set_consent(voice_id, owner, sreg.Consent(status="granted", consent_record_id=consent_record_id, granted_by=granted_by,
                                                            permitted_uses=["tts", "cloning"]))
    except sreg.SpeakerError:
        pass  # id not registry-safe (e.g. contains '.'/':') or owned by another key: the legacy voice still works


def _guard_foreign_speaker(owner: str, voice_id: str) -> None:
    """A voice id that is a registry speaker of another owner must not be overwritten or deleted through the legacy API."""
    try:
        sreg.get_registry().get(voice_id, owner)
    except sreg.SpeakerForbidden as e:
        raise HTTPException(409, "voice id belongs to another owner's speaker") from e
    except sreg.SpeakerError:
        pass  # not registered (or not registry-safe): legacy voice, unchanged behaviour


def _mirror_delete(owner: str, voice_id: str) -> None:
    reg = sreg.get_registry()
    try:
        sp = reg.get(voice_id, owner)
        if voice_id in sp.engine_bindings.values() and not sp.references:
            reg.delete(voice_id, owner)
    except sreg.SpeakerError:
        pass


@router.get("")
def list_voices():
    # `capabilities` is additive: what conditioning controls the engine behind this voice really honours
    return {"data": [{**v, "capabilities": tts.capabilities_for(v["voice_id"]).as_dict()} for v in tts.engine.voices()]}


@capabilities_router.get("")
def capabilities():
    """Per-engine capabilities plus the accepted values of every conditioning control."""
    engines = getattr(tts.engine, "engines", [tts.engine])
    out = {}
    for e in engines:
        name = type(e).__name__.removesuffix("Engine").lower()
        caps = getattr(e, "capabilities", DEFAULT_CAPABILITIES)
        out[name] = {**caps.as_dict(), "control_kinds": control_kinds(caps, settings.dsp_prosody),
                     "latency_tier": parse_tiers(settings.engine_latency_tiers).get(name, "slow"), "voices": [v["voice_id"] for v in e.voices()]}
    return {
        "engines": out,
        "controls": {"emotion": [x.value for x in Emotion], "style": [x.value for x in Style], "role": [x.value for x in Role],
                     "fallback": ["reject", "ignore"], "routing_policy": list(POLICIES)},
        "dsp": {"enabled": settings.dsp_prosody, "controls": ["pitch", "energy", "prosody_strength"] if settings.dsp_prosody else [],
                "note": "signal processing on the output, labelled pitch:dsp / energy:dsp; never emotion"},
        "output_sample_rates": list(TypeAdapter(SampleRate).json_schema()["enum"]),
    }


@router.post("")
async def upload_voice(voice_id: VoiceId = Form(), audio: UploadFile = ..., transcript: str | None = Form(None),
                       consent_record_id: str | None = Form(None), granted_by: str | None = Form(None), key: str = Depends(require_api_key)):
    """Register a reference voice (3-10s clean WAV) for zero-shot cloning. A transcript improves quality."""
    # FastAPI does not enforce the pattern on this Form field (a 70-char id, "a b", "../x" all arrived here): check it explicitly
    if not re.fullmatch(VOICE_ID, voice_id):
        raise HTTPException(422, "voice_id must match " + VOICE_ID)
    engine = _cloning_engine()
    _guard_foreign_speaker(sreg.owner_id(key), voice_id)
    try:
        wav, sr = audio_utils.decode_audio(await audio.read())
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, f"Bad audio: {e}") from e
    engine.save_voice(voice_id, wav, sr, transcript)
    tts.clear_cache(voice_id)
    _mirror_upload(sreg.owner_id(key), voice_id, consent_record_id, granted_by)
    return {"voice_id": voice_id, "has_transcript": bool(transcript)}


@router.delete("/{voice_id}")
def delete_voice(voice_id: VoiceId, key: str = Depends(require_api_key)):
    _guard_foreign_speaker(sreg.owner_id(key), voice_id)
    if not _cloning_engine().delete_voice(voice_id):
        raise HTTPException(404, "Voice not found")
    tts.clear_cache(voice_id)
    _mirror_delete(sreg.owner_id(key), voice_id)
    return {"deleted": voice_id}
