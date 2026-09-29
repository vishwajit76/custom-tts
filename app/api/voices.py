from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from pydantic import TypeAdapter, constr

from app.core.security import require_api_key
from app.models.schemas import VOICE_ID, SampleRate
from app.services import audio_utils, tts
from app.services.conditioning import DEFAULT_CAPABILITIES, Emotion, Role, Style

router = APIRouter(prefix="/v1/voices", dependencies=[Depends(require_api_key)])
capabilities_router = APIRouter(prefix="/v1/capabilities", dependencies=[Depends(require_api_key)])
VoiceId = constr(pattern=VOICE_ID)


def _cloning_engine():
    if not tts.engine.supports_cloning:
        raise HTTPException(400, "voice upload needs ENGINES=qwen3; for piper, fine-tune a voice (docs/training.md) and drop the .onnx into MODELS_DIR")
    return tts.engine


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
        out[name] = {**getattr(e, "capabilities", DEFAULT_CAPABILITIES).as_dict(), "voices": [v["voice_id"] for v in e.voices()]}
    return {
        "engines": out,
        "controls": {"emotion": [x.value for x in Emotion], "style": [x.value for x in Style], "role": [x.value for x in Role],
                     "fallback": ["reject", "ignore"]},
        "output_sample_rates": list(TypeAdapter(SampleRate).json_schema()["enum"]),
    }


@router.post("")
async def upload_voice(voice_id: VoiceId = Form(), audio: UploadFile = ..., transcript: str | None = Form(None)):
    """Register a reference voice (3-10s clean WAV) for zero-shot cloning. A transcript improves quality."""
    engine = _cloning_engine()
    try:
        wav, sr = audio_utils.decode_audio(await audio.read())
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, f"Bad audio: {e}") from e
    engine.save_voice(voice_id, wav, sr, transcript)
    tts.clear_cache(voice_id)
    return {"voice_id": voice_id, "has_transcript": bool(transcript)}


@router.delete("/{voice_id}")
def delete_voice(voice_id: VoiceId):
    if not _cloning_engine().delete_voice(voice_id):
        raise HTTPException(404, "Voice not found")
    tts.clear_cache(voice_id)
    return {"deleted": voice_id}
