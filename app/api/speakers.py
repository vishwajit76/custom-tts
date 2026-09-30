"""Speaker registry REST API (docs/voice-system.md). All routes are owner-scoped: a speaker is visible only to the
API key that created it (other owners get 404, so existence does not leak)."""
import asyncio
import logging

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.security import require_api_key
from app.services import speaker_registry as sr
from app.services import tts
from app.services.speaker_encoder import BackendUnavailable, EncoderAudioError, get_encoder

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/speakers", dependencies=[Depends(require_api_key)])
# Warning threshold for neural backends only (the mfcc baseline gets no threshold). An uncalibrated default: it was NOT fitted on
# same/different-speaker trials for resemblyzer or speechbrain (their cosine scales differ), so it only triggers a warning,
# never a rejection, and is not speaker verification.
NEURAL_MIN_SIMILARITY = 0.75


class SpeakerCreate(BaseModel):
    id: str
    display_name: str = Field(min_length=1, max_length=200)
    languages: list[str] = Field(default_factory=list, max_length=16)
    gender: str | None = Field(None, max_length=32)  # descriptive only
    presentation: str | None = Field(None, max_length=64)
    retention: sr.Retention | None = None
    consent: sr.Consent | None = None
    training: dict = Field(default_factory=dict)


class SpeakerPatch(BaseModel):
    display_name: str | None = Field(None, min_length=1, max_length=200)
    languages: list[str] | None = Field(None, max_length=16)
    gender: str | None = Field(None, max_length=32)
    presentation: str | None = Field(None, max_length=64)
    engine_bindings: dict[str, str] | None = None
    training: dict | None = None
    retention: sr.Retention | None = None


def _engines() -> dict:
    return {type(e).__name__.removesuffix("Engine").lower(): e for e in getattr(tts.engine, "engines", [tts.engine])}


def _err(e: sr.SpeakerError) -> HTTPException:
    # another owner's speaker is reported as 404 so ids cannot be enumerated
    if isinstance(e, sr.SpeakerForbidden) and "another owner" in str(e):
        return HTTPException(404, "speaker not found")
    return HTTPException(e.status, str(e))


def _run(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except sr.SpeakerError as e:
        raise _err(e) from e


def _on_expiry(sp: sr.Speaker) -> None:
    purge_bound_voices(sp)


def purge_bound_voices(sp: sr.Speaker) -> None:
    """Cloning engines keep their own copy of a clip (qwen3 voices_dir): remove it with the speaker/consent."""
    for name, voice in sp.engine_bindings.items():
        e = _engines().get(name)
        if e is not None and getattr(e, "supports_cloning", False) and hasattr(e, "delete_voice"):
            try:
                if e.delete_voice(voice):
                    tts.clear_cache(voice)
            except ValueError:
                pass


@router.post("", status_code=201)
def create_speaker(body: SpeakerCreate, key: str = Depends(require_api_key)):
    reg = sr.get_registry()
    sp = _run(reg.create, sr.owner_id(key), body.id, body.display_name, body.languages, body.gender, body.presentation, body.retention,
              body.training)
    if body.consent:
        sp = _run(reg.set_consent, body.id, sp.owner, body.consent)
    return sp.public()


@router.get("")
def list_speakers(key: str = Depends(require_api_key)):
    return {"data": [s.public() for s in sr.get_registry().list_for(sr.owner_id(key))]}


@router.get("/{speaker_id}")
def get_speaker(speaker_id: str, key: str = Depends(require_api_key)):
    return _run(sr.get_registry().get, speaker_id, sr.owner_id(key)).public()


@router.patch("/{speaker_id}")
def patch_speaker(speaker_id: str, body: SpeakerPatch, key: str = Depends(require_api_key)):
    fields = body.model_dump(exclude_none=True)
    fields.pop("retention", None)
    if body.retention:
        fields["retention"] = body.retention
    for eng, voice in (body.engine_bindings or {}).items():
        e = _engines().get(eng)
        if e is None or not e.has_voice(voice):
            raise HTTPException(422, f"engine {eng!r} is not loaded or has no voice {voice!r}")
    return _run(sr.get_registry().update, speaker_id, sr.owner_id(key), **fields).public()


@router.put("/{speaker_id}/consent")
def put_consent(speaker_id: str, body: sr.Consent, key: str = Depends(require_api_key)):
    reg, owner = sr.get_registry(), sr.owner_id(key)
    sp = _run(reg.set_consent, speaker_id, owner, body)
    if body.status == "revoked":
        purge_bound_voices(sp)
    return sp.public()


@router.delete("/{speaker_id}")
def delete_speaker(speaker_id: str, key: str = Depends(require_api_key)):
    reg, owner = sr.get_registry(), sr.owner_id(key)
    sp = _run(reg.get, speaker_id, owner)
    purge_bound_voices(sp)
    _run(reg.delete, speaker_id, owner)
    return {"deleted": speaker_id}


@router.post("/{speaker_id}/references", status_code=201)
async def add_reference(speaker_id: str, audio: UploadFile = File(...), transcript: str | None = Form(None, max_length=2000),
                        key: str = Depends(require_api_key)):
    """Upload one reference clip (WAV, 3-30 s, >=16 kHz, not clipped/silent). Returns quality metrics and, with 2+
    references, the embedding consistency check (informational; see docs/voice-system.md)."""
    raw = await audio.read(reg_limit() + 1)
    # decode + encoder + fsync'd file writes + flock are blocking: on the event loop they would stall every live WebSocket stream
    return await asyncio.to_thread(_ingest_reference, speaker_id, sr.owner_id(key), raw, audio.content_type, transcript)


def _ingest_reference(speaker_id: str, owner: str, raw: bytes, content_type: str | None, transcript: str | None) -> dict:
    reg = sr.get_registry()
    sp = _run(reg.get, speaker_id, owner)
    if sp.consent.status == "revoked":
        raise HTTPException(403, "consent revoked; cannot add references")  # registry re-checks under its lock
    wav, rate, metrics = _run(sr.analyze_reference, raw, content_type)
    enc = get_encoder()
    emb, warnings, sim = None, [], None
    try:
        emb = enc.embed(wav, rate)
        prev = reg.reference_embeddings(sp) if sp.embedding_backend in (None, enc.name) else []
        if prev:
            sim = round(float(np.mean([enc.similarity(emb, p) for p in prev])), 4)
            if enc.neural and sim < NEURAL_MIN_SIMILARITY:
                warnings.append(f"low similarity to existing references ({sim}); check this is the same speaker")
            elif not enc.neural:
                warnings.append("similarity from the mfcc baseline is uncalibrated and not a speaker verification")
        elif sp.embedding_backend not in (None, enc.name):
            warnings.append(f"embedding not stored: this speaker's embeddings come from the {sp.embedding_backend!r} encoder, not {enc.name!r}")
    except (BackendUnavailable, EncoderAudioError) as e:
        warnings.append(f"embedding skipped: {e}")
    ref = _run(reg.add_reference, speaker_id, owner, wav, rate, metrics, transcript, emb, enc.name if emb is not None else None)
    return {**ref.model_dump(exclude={"path", "embedding_path"}), "raw_retained": ref.path is not None, "similarity_to_existing": sim,
            "encoder": {"name": enc.name, "neural": enc.neural, "similarity_kind": enc.similarity_kind}, "warnings": warnings}


def reg_limit() -> int:
    from app.core.config import settings

    return settings.ref_max_bytes


@router.delete("/{speaker_id}/references/{ref_id}")
def delete_reference(speaker_id: str, ref_id: str, key: str = Depends(require_api_key)):
    _run(sr.get_registry().delete_reference, speaker_id, sr.owner_id(key), ref_id)
    return {"deleted": ref_id}


sr.expiry_hooks.append(_on_expiry)
