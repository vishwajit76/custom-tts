"""Persistent speaker registry: one directory per speaker under settings.speakers_dir.

    <root>/<speaker_id>/speaker.json      the record (atomic write)
    <root>/<speaker_id>/refs/<ref>.wav    reference audio (only when retention keeps raw references)
    <root>/<speaker_id>/refs/<ref>.npy    per-reference embedding (informational)
    <root>/<speaker_id>/embedding.npy     mean embedding of the references (informational)

Safeguards: strict speaker id (no traversal), every stored path is relative and re-resolved inside the speaker dir,
per-owner authorization (owner = stable hash of the API key), consent gate for synthesis, secure delete
(overwrite then unlink). gender/presentation are descriptive metadata only and never used as identity.
Single-node file store: writes are atomic and serialized by a process lock plus an flock on the registry dir.
"""
import contextlib
import hashlib
import io
import json
import logging
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from pydantic import BaseModel, Field

from app.core.config import settings

try:
    import fcntl
except ImportError:  # pragma: no cover (non-POSIX)
    fcntl = None

log = logging.getLogger(__name__)
SPEAKER_ID = r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}"
PERMITTED_USES = ("tts", "cloning", "training", "evaluation")
ACCEPTED_CONTENT_TYPES = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave", "audio/flac", "audio/x-flac", "application/octet-stream"}


class SpeakerError(Exception):
    status = 400


class SpeakerNotFound(SpeakerError):
    status = 404


class SpeakerForbidden(SpeakerError):
    status = 403


class SpeakerExists(SpeakerError):
    status = 409


class ReferenceRejected(SpeakerError):
    status = 422


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def owner_id(api_key: str) -> str:
    """Stable, non-reversible owner identity for an API key (empty key = auth disabled)."""
    return "key_" + hashlib.sha256(api_key.encode()).hexdigest()[:16] if api_key else "anonymous"


class Consent(BaseModel):
    status: Literal["granted", "revoked", "pending"] = "pending"
    consent_record_id: str | None = Field(None, max_length=128)
    granted_by: str | None = Field(None, max_length=200)
    date: str | None = None
    permitted_uses: list[Literal["tts", "cloning", "training", "evaluation"]] = []


class Retention(BaseModel):
    keep_raw_reference: bool = True
    delete_after_days: int | None = Field(None, ge=1, le=3650)  # purge raw references this long after upload


class Reference(BaseModel):
    ref_id: str
    path: str | None = None  # relative to the speaker dir; None when raw audio is not retained
    sha256: str
    duration_s: float
    sample_rate: int
    rms_dbfs: float
    clip_fraction: float
    speech_fraction: float
    transcript: str | None = None
    embedding_path: str | None = None
    quality: float = 0.0  # 0..1 heuristic used to pick the best reference; NOT a voice-similarity score
    created_at: str = Field(default_factory=now_iso)


class Speaker(BaseModel):
    id: str
    owner: str
    display_name: str
    languages: list[str] = []
    gender: str | None = None  # descriptive metadata only, never identity
    presentation: str | None = None
    engine_bindings: dict[str, str] = {}  # engine name -> voice id inside that engine
    references: list[Reference] = []
    embedding_path: str | None = None
    embedding_backend: str | None = None
    training: dict = {}  # provenance: dataset manifest id, checkpoint, date, ...
    consent: Consent = Consent()
    retention: Retention = Retention()
    previews: list[str] = []
    clone_capable: bool = True  # False once retention removed every raw reference (and the engine-side clip copies)
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)

    def public(self) -> dict:
        d = self.model_dump()
        d.pop("owner")
        return d


# ---------- files ----------
def secure_unlink(path: Path) -> None:
    """Overwrite with random bytes, fsync, unlink. Best effort: journaling/SSD/backups may retain copies."""
    try:
        if path.is_file() and not path.is_symlink():
            size = path.stat().st_size
            with open(path, "r+b") as f:
                f.write(os.urandom(size))
                f.flush()
                os.fsync(f.fileno())
    except OSError:
        pass
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------- reference validation ----------
def analyze_reference(raw: bytes, content_type: str | None = None) -> tuple[np.ndarray, int, dict]:
    """Decode and validate an uploaded reference. Returns (mono float32, sr, metrics); raises ReferenceRejected."""
    if len(raw) > settings.ref_max_bytes:
        raise ReferenceRejected(f"file too large: {len(raw)} > {settings.ref_max_bytes} bytes")
    if not raw:
        raise ReferenceRejected("empty file")
    if content_type and content_type.split(";")[0].strip().lower() not in ACCEPTED_CONTENT_TYPES:
        raise ReferenceRejected(f"unsupported content-type {content_type!r}; send WAV")
    try:  # header first: a small FLAC/OGG can decode to gigabytes, so bound the duration before decoding
        info = sf.info(io.BytesIO(raw))
    except Exception as e:
        raise ReferenceRejected(f"cannot decode audio: {e}") from e
    if info.samplerate <= 0 or info.channels > 8 or info.frames > settings.ref_max_seconds * info.samplerate:
        raise ReferenceRejected(f"too long or malformed header ({info.frames} frames at {info.samplerate} Hz, {info.channels} ch)")
    try:
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as e:
        raise ReferenceRejected(f"cannot decode audio: {e}") from e
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if not np.all(np.isfinite(wav)):
        raise ReferenceRejected("audio contains NaN/inf")
    dur = len(wav) / sr
    if dur < settings.ref_min_seconds:
        raise ReferenceRejected(f"too short: {dur:.1f}s < {settings.ref_min_seconds:g}s")
    if dur > settings.ref_max_seconds:
        raise ReferenceRejected(f"too long: {dur:.1f}s > {settings.ref_max_seconds:g}s")
    if sr < settings.ref_min_sample_rate:
        raise ReferenceRejected(f"sample rate {sr} Hz < {settings.ref_min_sample_rate} Hz")
    clip = float(np.mean(np.abs(wav) >= 0.999))
    if clip > 0.01:
        raise ReferenceRejected(f"clipped: {clip:.1%} of samples at full scale")
    rms = float(np.sqrt(np.mean(wav**2)))
    rms_db = 20 * np.log10(max(rms, 1e-9))
    if rms_db < -45:
        raise ReferenceRejected(f"too quiet/silent: {rms_db:.1f} dBFS")
    frame = max(1, int(sr * 0.025))
    n = len(wav) // frame
    fr = np.sqrt(np.mean(wav[: n * frame].reshape(n, frame) ** 2, axis=1))
    peak = float(fr.max()) if n else 0.0
    speech = float(np.mean(fr > peak * 0.1)) if n else 0.0  # frames within 20 dB of the loudest frame
    if speech < 0.3:
        raise ReferenceRejected(f"mostly silence: only {speech:.0%} of frames contain speech")
    # heuristic quality: prefer ~5-12 s, low clipping, healthy level, little silence
    dur_score = 1.0 if 5 <= dur <= 12 else max(0.3, min(dur / 5, 12 / dur))
    level_score = float(np.clip((rms_db + 40) / 20, 0.2, 1.0))
    quality = round(dur_score * level_score * (0.5 + 0.5 * speech) * (1 - clip * 20), 4)
    return wav, sr, {"duration_s": round(dur, 3), "sample_rate": sr, "rms_dbfs": round(rms_db, 1), "clip_fraction": round(clip, 5),
                     "speech_fraction": round(speech, 3), "quality": quality, "sha256": hashlib.sha256(raw).hexdigest()}


# ---------- registry ----------
class SpeakerRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @contextlib.contextmanager
    def _locked(self):
        with self._lock:
            fd = None
            if fcntl:
                fd = os.open(self.root / ".lock", os.O_CREAT | os.O_RDWR, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fd is not None:
                    os.close(fd)

    # -- paths --
    @staticmethod
    def check_id(speaker_id: str) -> str:
        if not isinstance(speaker_id, str) or not re.fullmatch(SPEAKER_ID, speaker_id):
            raise SpeakerError(f"invalid speaker id {speaker_id!r}: use letters, digits, _ and - (max 64)")
        return speaker_id

    def _dir(self, speaker_id: str) -> Path:
        d = (self.root / self.check_id(speaker_id)).resolve()
        if d.parent != self.root.resolve():
            raise SpeakerError("invalid speaker id")
        return d

    def resolve_path(self, speaker_id: str, rel: str) -> Path:
        """Stored relative path -> absolute path, refusing anything outside the speaker dir (or symlinks)."""
        base = self._dir(speaker_id)
        p = (base / rel).resolve()
        if base not in p.parents or (base / rel).is_symlink():
            raise SpeakerError(f"path escapes registry: {rel!r}")
        return p

    # -- record io --
    def _load(self, speaker_id: str) -> Speaker:
        f = self._dir(speaker_id) / "speaker.json"
        if not f.is_file():
            raise SpeakerNotFound(f"speaker {speaker_id!r} not found")
        return Speaker.model_validate_json(f.read_text("utf-8"))

    def _save(self, sp: Speaker) -> None:
        sp.updated_at = now_iso()
        d = self._dir(sp.id)
        d.mkdir(exist_ok=True)
        _atomic_write(d / "speaker.json", sp.model_dump_json(indent=2).encode())

    def _authorized(self, speaker_id: str, owner: str) -> Speaker:
        sp = self._load(speaker_id)
        if sp.owner != owner:
            raise SpeakerForbidden("speaker belongs to another owner")
        return sp

    # -- CRUD --
    def create(self, owner: str, speaker_id: str, display_name: str, languages: list[str] | None = None, gender: str | None = None,
               presentation: str | None = None, retention: Retention | None = None, training: dict | None = None,
               engine_bindings: dict[str, str] | None = None) -> Speaker:
        with self._locked():
            if (self._dir(speaker_id) / "speaker.json").exists():
                raise SpeakerExists(f"speaker {speaker_id!r} already exists")
            sp = Speaker(id=speaker_id, owner=owner, display_name=display_name.strip()[:200] or speaker_id, languages=languages or [],
                         gender=gender, presentation=presentation, training=training or {}, engine_bindings=engine_bindings or {},
                         retention=retention or Retention(keep_raw_reference=settings.keep_raw_reference,
                                                          delete_after_days=settings.reference_delete_after_days))
            self._save(sp)
            return sp

    def get(self, speaker_id: str, owner: str) -> Speaker:
        with self._locked():
            sp = self._authorized(speaker_id, owner)
            return self._purge_if_expired(sp)

    def list_for(self, owner: str) -> list[Speaker]:
        out = []
        with self._locked():
            for d in sorted(self.root.iterdir()):
                if d.is_dir() and re.fullmatch(SPEAKER_ID, d.name) and (d / "speaker.json").is_file():
                    with contextlib.suppress(Exception):
                        sp = self._load(d.name)
                        if sp.owner == owner:
                            out.append(self._purge_if_expired(sp))
        return out

    def update(self, speaker_id: str, owner: str, **fields) -> Speaker:
        allowed = {"display_name", "languages", "gender", "presentation", "engine_bindings", "training", "retention"}
        if bad := set(fields) - allowed:
            raise SpeakerError(f"cannot update {sorted(bad)}")
        with self._locked():
            sp = self._authorized(speaker_id, owner)
            for k, v in fields.items():
                if v is not None:
                    setattr(sp, k, Retention.model_validate(v) if k == "retention" else v)
            self._save(sp)
            return sp

    def set_consent(self, speaker_id: str, owner: str, consent: Consent) -> Speaker:
        """granted needs consent_record_id, granted_by and permitted_uses. revoked purges references and embeddings."""
        with self._locked():
            sp = self._authorized(speaker_id, owner)
            if consent.status == "granted":
                if not (consent.consent_record_id and consent.granted_by and consent.permitted_uses):
                    raise SpeakerError("granted consent needs consent_record_id, granted_by and permitted_uses")
                consent = consent.model_copy(update={"date": consent.date or now_iso()})
            elif consent.status == "revoked":
                consent = consent.model_copy(update={"date": now_iso(), "permitted_uses": []})
                self._purge_assets(sp)
            sp.consent = consent
            self._save(sp)
            return sp

    def delete(self, speaker_id: str, owner: str) -> None:
        """Secure delete: overwrite + unlink every file, then the directory."""
        with self._locked():
            self._authorized(speaker_id, owner)
            d = self._dir(speaker_id)
            for f in sorted(d.rglob("*"), reverse=True):
                if f.is_file() or f.is_symlink():
                    secure_unlink(f)
            shutil.rmtree(d, ignore_errors=True)

    # -- consent gate --
    def check_use(self, sp: Speaker, use: str = "tts") -> None:
        if sp.consent.status != "granted":
            raise SpeakerForbidden(f"speaker {sp.id!r} consent is {sp.consent.status}; synthesis needs granted consent")
        if use not in sp.consent.permitted_uses:
            raise SpeakerForbidden(f"speaker {sp.id!r} consent does not permit use {use!r} (permitted: {sp.consent.permitted_uses})")

    # -- references --
    def add_reference(self, speaker_id: str, owner: str, wav: np.ndarray, sr: int, metrics: dict, transcript: str | None = None,
                      embedding: np.ndarray | None = None, embedding_backend: str | None = None) -> Reference:
        with self._locked():
            sp = self._authorized(speaker_id, owner)
            if sp.consent.status == "revoked":  # checked under the lock: a concurrent revoke must not be followed by an upload
                raise SpeakerForbidden("consent revoked; cannot add references")
            if len(sp.references) >= settings.max_references_per_speaker:
                raise SpeakerError(f"speaker already has {settings.max_references_per_speaker} references")
            if any(r.sha256 == metrics["sha256"] for r in sp.references):
                raise SpeakerExists("this reference audio is already registered")
            rid = uuid.uuid4().hex[:12]
            d = self._dir(speaker_id)
            (d / "refs").mkdir(parents=True, exist_ok=True)
            path = emb = None
            if sp.retention.keep_raw_reference:
                path = f"refs/{rid}.wav"
                buf = io.BytesIO()
                sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
                _atomic_write(d / path, buf.getvalue())
            if embedding is not None:
                emb = f"refs/{rid}.npy"
                b = io.BytesIO()
                np.save(b, embedding)
                _atomic_write(d / emb, b.getvalue())
            ref = Reference(ref_id=rid, path=path, transcript=transcript or None, embedding_path=emb,
                            **{k: metrics[k] for k in ("sha256", "duration_s", "sample_rate", "rms_dbfs", "clip_fraction", "speech_fraction", "quality")})
            sp.references.append(ref)
            if embedding is not None:
                self._update_centroid(sp, embedding_backend)
            self._save(sp)
            return ref

    def _update_centroid(self, sp: Speaker, backend: str | None) -> None:
        d = self._dir(sp.id)
        embs = [np.load(self.resolve_path(sp.id, r.embedding_path)) for r in sp.references if r.embedding_path]
        if not embs:
            return
        m = np.mean(embs, axis=0)
        m = (m / np.linalg.norm(m)).astype(np.float32)
        b = io.BytesIO()
        np.save(b, m)
        _atomic_write(d / "embedding.npy", b.getvalue())
        sp.embedding_path, sp.embedding_backend = "embedding.npy", backend

    def reference_embeddings(self, sp: Speaker) -> list[np.ndarray]:
        return [np.load(self.resolve_path(sp.id, r.embedding_path)) for r in sp.references if r.embedding_path]

    def delete_reference(self, speaker_id: str, owner: str, ref_id: str) -> Speaker:
        with self._locked():
            sp = self._authorized(speaker_id, owner)
            ref = next((r for r in sp.references if r.ref_id == ref_id), None)
            if ref is None:
                raise SpeakerNotFound(f"reference {ref_id!r} not found")
            for rel in (ref.path, ref.embedding_path):
                if rel:
                    secure_unlink(self.resolve_path(sp.id, rel))
            sp.references.remove(ref)
            if sp.references and any(r.embedding_path for r in sp.references):
                self._update_centroid(sp, sp.embedding_backend)
            elif sp.embedding_path:
                secure_unlink(self.resolve_path(sp.id, sp.embedding_path))
                sp.embedding_path = sp.embedding_backend = None
            self._save(sp)
            return sp

    def best_reference(self, sp: Speaker) -> Reference | None:
        """Highest heuristic quality among references whose raw audio is retained. Qwen3-TTS takes ONE reference clip
        per voice prompt (the API batches references across items, it does not merge several), so we pick the best."""
        usable = [r for r in sp.references if r.path]
        return max(usable, key=lambda r: (r.quality, r.duration_s)) if usable else None

    def read_reference(self, sp: Speaker, ref: Reference) -> tuple[np.ndarray, int]:
        wav, sr = sf.read(self.resolve_path(sp.id, ref.path), dtype="float32")
        return wav, sr

    # -- retention --
    def _purge_assets(self, sp: Speaker) -> None:
        for r in sp.references:
            for rel in (r.path, r.embedding_path):
                if rel:
                    secure_unlink(self.resolve_path(sp.id, rel))
        if sp.embedding_path:
            secure_unlink(self.resolve_path(sp.id, sp.embedding_path))
        sp.references, sp.embedding_path, sp.embedding_backend = [], None, None

    def _purge_if_expired(self, sp: Speaker) -> Speaker:
        days = sp.retention.delete_after_days
        if not days:
            return sp
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        old = [r for r in sp.references if r.path and datetime.fromisoformat(r.created_at) < cutoff]
        for r in old:
            secure_unlink(self.resolve_path(sp.id, r.path))
            r.path = None  # raw audio gone; hash, metrics and embedding stay as provenance
        if old:
            if not any(r.path for r in sp.references):  # nothing left to clone from: drop engine-side copies too
                sp.clone_capable = False
                for hook in expiry_hooks:
                    try:
                        hook(sp)
                    except Exception:
                        log.exception("expiry hook failed")
            self._save(sp)
        return sp

    def purge_expired(self) -> int:
        n = 0
        with self._locked():
            for d in self.root.iterdir():
                if d.is_dir() and re.fullmatch(SPEAKER_ID, d.name) and (d / "speaker.json").is_file():
                    sp = self._load(d.name)
                    before = sum(1 for r in sp.references if r.path)
                    self._purge_if_expired(sp)
                    n += before - sum(1 for r in sp.references if r.path)
        return n


expiry_hooks: list = []  # callables(speaker) run when retention removed a speaker's last raw reference (see api/speakers.py)
_registries: dict[str, SpeakerRegistry] = {}


def get_registry() -> SpeakerRegistry:
    key = str(settings.speakers_dir)
    if key not in _registries:
        _registries[key] = SpeakerRegistry(settings.speakers_dir)
    return _registries[key]
