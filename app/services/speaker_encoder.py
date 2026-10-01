"""Speaker embeddings for reference quality checks and evaluation.

IMPORTANT: embeddings are informational only. No TTS engine declares the `speaker_embedding` capability, so nothing
here conditions synthesis. The default `mfcc` backend is a deterministic MFCC-statistics baseline and is NOT a neural
speaker verifier: its cosine scores are uncalibrated (unrelated voices still score high) and must not be used for
identity decisions. Use the optional neural backends (`resemblyzer`, `speechbrain`) for meaningful similarity.
"""
import hashlib
import logging
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
TARGET_SR = 16000
MIN_SECONDS = 1.0
MAX_CLIP_FRACTION = 0.01
MIN_RMS_DBFS = -50.0
CACHE_MAX = 512


class EncoderAudioError(ValueError):
    """Audio is unusable for embedding (too short, silent, clipped)."""


class BackendUnavailable(RuntimeError):
    """Optional neural backend is not installed."""


def to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != TARGET_SR:
        import soxr

        wav = soxr.resample(wav, sr, TARGET_SR).astype(np.float32)
    return wav


def check_audio(wav16: np.ndarray, min_seconds: float = MIN_SECONDS) -> dict:
    """Quality metrics on 16 kHz mono audio; raises EncoderAudioError for unusable input."""
    if wav16.size == 0 or not np.all(np.isfinite(wav16)):
        raise EncoderAudioError("empty or non-finite audio")
    dur = wav16.size / TARGET_SR
    if dur < min_seconds:
        raise EncoderAudioError(f"audio too short: {dur:.2f}s < {min_seconds}s")
    rms = float(np.sqrt(np.mean(wav16**2)))
    rms_db = 20 * np.log10(max(rms, 1e-9))
    if rms_db < MIN_RMS_DBFS:
        raise EncoderAudioError(f"audio is (nearly) silent: {rms_db:.1f} dBFS")
    clip = float(np.mean(np.abs(wav16) >= 0.999))
    if clip > MAX_CLIP_FRACTION:
        raise EncoderAudioError(f"audio is clipped: {clip:.1%} of samples at full scale")
    return {"duration_s": round(dur, 3), "rms_dbfs": round(rms_db, 1), "clip_fraction": round(clip, 5)}


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n == 0 or not np.isfinite(n):
        raise EncoderAudioError("degenerate embedding")
    return v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class _MfccBackend:
    name = "mfcc"
    neural = False
    dim = 2 * 19 + 2 * 19

    kind = "mfcc_statistics_cosine"

    def load(self) -> None:
        try:
            import librosa  # noqa: F401  (not in requirements.txt: only requirements-dsp.txt / requirements-qwen.txt have it)
        except ImportError as e:
            raise BackendUnavailable("the mfcc baseline needs librosa: pip install -r requirements-dsp.txt") from e

    def embed(self, wav16: np.ndarray) -> np.ndarray:
        import librosa

        m = librosa.feature.mfcc(y=wav16, sr=TARGET_SR, n_mfcc=20, n_fft=512, hop_length=160, n_mels=40)[1:]  # drop c0 (loudness)
        d = librosa.feature.delta(m, width=5)
        return np.concatenate([m.mean(1), m.std(1), d.mean(1), d.std(1)]).astype(np.float32)


class _ResemblyzerBackend:
    kind = "neural_embedding_cosine"
    name = "resemblyzer"  # Apache-2.0 code + bundled GE2E weights (pretrained.pt, VoxCeleb/LibriSpeech-trained); see docs/licenses.md
    neural = True

    def load(self) -> None:
        try:
            from resemblyzer import VoiceEncoder
        except ImportError as e:
            raise BackendUnavailable("pip install resemblyzer") from e
        self._enc = VoiceEncoder("cpu")

    def embed(self, wav16: np.ndarray) -> np.ndarray:
        # raw 16 kHz audio, no resemblyzer.preprocess_wav: measured the same separation, 3x faster (docs/benchmarks.md)
        return np.asarray(self._enc.embed_utterance(wav16), np.float32)


class _SpeechbrainBackend:
    kind = "neural_embedding_cosine"
    name = "speechbrain"  # ECAPA-TDNN spkrec-ecapa-voxceleb (Apache-2.0); weights download on first use
    neural = True

    def load(self) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as e:
            raise BackendUnavailable("pip install speechbrain") from e
        self._enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    def embed(self, wav16: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self._enc.encode_batch(torch.from_numpy(wav16)[None]).squeeze().cpu().numpy().astype(np.float32)


BACKENDS = {"mfcc": _MfccBackend, "resemblyzer": _ResemblyzerBackend, "speechbrain": _SpeechbrainBackend}


def available_backends() -> dict[str, bool]:
    import importlib.util as u

    return {"mfcc": True, "resemblyzer": u.find_spec("resemblyzer") is not None, "speechbrain": u.find_spec("speechbrain") is not None}


class SpeakerEncoder:
    """Lazy-loading, caching encoder. Embeddings are L2-normalized, so cosine similarity is a dot product."""

    def __init__(self, backend: str = "mfcc", cache_dir: Path | None = None, min_seconds: float = MIN_SECONDS) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"unknown speaker encoder {backend!r}; have {list(BACKENDS)}")
        self._backend = BACKENDS[backend]()
        self.min_seconds = min_seconds
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()  # LRU: embeddings of uploaded voices must not pile up
        self._loaded = False
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._backend.name

    @property
    def neural(self) -> bool:
        return self._backend.neural

    @property
    def similarity_kind(self) -> str:
        """What a similarity score from this encoder IS. Never "speaker_verification": nothing here is calibrated on
        same/different-speaker trials for this service's audio, so a score is a QA signal, not an identity decision."""
        return self._backend.kind

    def _ensure(self) -> None:
        with self._lock:
            if not self._loaded:
                self._backend.load()
                self._loaded = True

    @staticmethod
    def audio_sha256(wav: np.ndarray, sr: int) -> str:
        w = np.ascontiguousarray(np.asarray(wav, dtype="<f4"))
        return hashlib.sha256(w.tobytes() + str(sr).encode()).hexdigest()

    def embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        key = f"{self.name}:{self.audio_sha256(wav, sr)}"
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        disk = self.cache_dir / self.name / f"{key.split(':')[1]}.npy" if self.cache_dir else None
        if disk is not None and disk.exists():
            emb = np.load(disk)
        else:
            w16 = to_16k(wav, sr)
            check_audio(w16, self.min_seconds)
            self._ensure()
            emb = l2_normalize(self._backend.embed(w16))
            if disk is not None:
                disk.parent.mkdir(parents=True, exist_ok=True)
                tmp = disk.with_suffix(".tmp.npy")
                np.save(tmp, emb)
                tmp.replace(disk)
        self._cache[key] = emb
        while len(self._cache) > CACHE_MAX:
            self._cache.popitem(last=False)
        return emb

    def check(self, wav: np.ndarray, sr: int) -> dict:
        return check_audio(to_16k(wav, sr), self.min_seconds)

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        return cosine(a, b)

    def consistency(self, embeddings: list[np.ndarray]) -> dict:
        """Similarity of each embedding to the mean of the others (reference QA). Needs >= 2."""
        if len(embeddings) < 2:
            return {"scores": [], "min": None, "calibrated": False, "kind": self.similarity_kind}
        E = np.stack(embeddings)
        scores = []
        for i in range(len(E)):
            others = np.delete(E, i, axis=0).mean(0)
            scores.append(round(cosine(E[i], others), 4))
        # "calibrated": False for every backend. A neural backend is meaningful, not calibrated for this service's audio
        return {"scores": scores, "min": min(scores), "calibrated": False, "kind": self.similarity_kind}


_default: dict[str, SpeakerEncoder] = {}


def get_encoder(backend: str | None = None) -> SpeakerEncoder:
    from app.core.config import settings

    name = backend or settings.speaker_encoder
    if name not in _default:
        _default[name] = SpeakerEncoder(name)
    return _default[name]
