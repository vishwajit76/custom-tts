from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from pydantic import constr

from app.core.security import require_api_key
from app.models.schemas import VOICE_ID
from app.services import audio_utils, tts

router = APIRouter(prefix="/v1/voices", dependencies=[Depends(require_api_key)])
VoiceId = constr(pattern=VOICE_ID)


def _cloning_engine():
    if not tts.engine.supports_cloning:
        raise HTTPException(400, "voice upload needs ENGINES=qwen3; for piper, fine-tune a voice (docs/training.md) and drop the .onnx into MODELS_DIR")
    return tts.engine


@router.get("")
def list_voices():
    return {"data": tts.engine.voices()}


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
