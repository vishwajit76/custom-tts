"""Streaming synthesis pipeline shared by the HTTP and WebSocket APIs.

text -> normalize -> split (short first chunk) -> EDF worker pool -> trim lead silence -> resample -> PCM s16le frames
"""
import asyncio
import importlib
import logging
import re
import time
from collections import OrderedDict, deque
from collections.abc import AsyncIterator
from contextlib import contextmanager

import numpy as np
import soxr

from app.core.config import settings
from app.services import audio_utils, dsp, expressive_engine, text_normalizer
from app.services.conditioning import DEFAULT_CAPABILITIES, EngineCapabilities
from app.services.scheduler import Scheduler

log = logging.getLogger(__name__)
_SPEAKABLE = re.compile(r"[^\W_]")  # a letter or digit; matras and virama are marks, not letters


class Overloaded(Exception):
    pass


class MultiEngine:
    """Several engines behind the one-engine interface. Voice ids are unique across engines; each routes to its owner."""

    supports_cloning = False  # ponytail: cloning (qwen3) runs alone; routing `ref` requests would need a default cloner

    def __init__(self, engines: list) -> None:
        self.engines = engines
        self.max_workers = max(e.max_workers for e in engines)  # workers share the CPU; any worker runs any engine

    @property
    def ready(self) -> bool:
        return all(e.ready for e in self.engines)

    def load(self) -> None:
        for e in self.engines:
            e.load()
        ids = [v["voice_id"] for v in self.voices()]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"voice ids collide across engines: {sorted({i for i in ids if ids.count(i) > 1})}")

    def voices(self) -> list[dict]:
        return [v for e in self.engines for v in e.voices()]

    def _owner(self, voice: str):
        for e in self.engines:
            if e.has_voice(voice):
                return e
        raise KeyError(voice)

    def has_voice(self, voice: str) -> bool:
        return any(e.has_voice(voice) for e in self.engines)

    def sample_rate(self, voice: str) -> int:
        return self._owner(voice).sample_rate(voice)

    def capabilities_for(self, voice: str) -> EngineCapabilities:
        return getattr(self._owner(voice), "capabilities", DEFAULT_CAPABILITIES)

    def synth(self, text: str, voice: str, speed: float, ref=None, ref_text=None) -> np.ndarray:
        return self._owner(voice).synth(text, voice, speed, ref, ref_text)

    def synth_native(self, text: str, voice: str, speed: float, controls: dict, ref=None, ref_text=None) -> np.ndarray:
        return self._owner(voice).synth_native(text, voice, speed, controls, ref, ref_text)


_ENGINES = {"piper": "piper_engine.PiperEngine", "supertonic": "supertonic_engine.SupertonicEngine",
            "kokoro": "kokoro_engine.KokoroEngine", "qwen3": "qwen_engine.QwenEngine",
            "expressive": "expressive_engine.ExpressiveEngine"}  # "expressive": class taken from EXPRESSIVE_ENGINE, off by default


def _make_engine(names: str):
    """'piper' -> that engine; 'piper,kokoro' -> MultiEngine. Imports only the engines asked for."""
    names = [n.strip() for n in names.split(",") if n.strip()]
    if unknown := [n for n in names if n not in _ENGINES]:
        raise ValueError(f"unknown engine(s) {unknown}; have {list(_ENGINES)}")
    if len(names) > 1 and "qwen3" in names:
        raise ValueError("qwen3 (voice cloning) runs alone: ENGINES=qwen3")
    engines = []
    for n in names:
        if n == "expressive":
            engines.append(expressive_engine.build_configured(settings.expressive_engine))
            continue
        mod, cls = _ENGINES[n].split(".")
        engines.append(getattr(importlib.import_module(f"app.services.{mod}"), cls)())
    return engines[0] if len(engines) == 1 else MultiEngine(engines)


engine = _make_engine(settings.engines)
scheduler: Scheduler | None = None
_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
stats = {
    "streams_active": 0, "streams_total": 0, "streams_rejected": 0, "streams_cancelled": 0, "streams_failed": 0,
    "chunks_total": 0, "cache_hits": 0, "audio_seconds_total": 0.0, "synth_seconds_total": 0.0,
}
ttfa_recent: deque[float] = deque(maxlen=2000)  # seconds, for /metrics quantiles


def load() -> None:
    global scheduler
    engine.load()
    if settings.dsp_prosody:  # first librosa call JIT-compiles for seconds; keep that off the first request
        try:
            dsp.apply_prosody(np.zeros(8192, np.float32), 24000, pitch=1.0)
        except ImportError as e:
            raise RuntimeError("DSP_PROSODY=true needs librosa: pip install -r requirements-dsp.txt") from e
    scheduler = Scheduler(engine.max_workers)


def resolve_voice(voice: str) -> str:
    if engine.has_voice(voice):
        return voice
    if voice == "default" and engine.has_voice(settings.default_voice):
        return settings.default_voice
    raise KeyError(voice)


def capabilities_for(voice: str | None = None) -> EngineCapabilities:
    """Capabilities of the engine that would serve `voice` (None or a cloning reference: the single engine's own)."""
    if voice is not None and hasattr(engine, "capabilities_for") and engine.has_voice(voice):
        return engine.capabilities_for(voice)
    return getattr(engine, "capabilities", DEFAULT_CAPABILITIES)


def engine_name(voice: str | None = None) -> str:
    owner = engine._owner(voice) if voice is not None and hasattr(engine, "_owner") and engine.has_voice(voice) else engine
    return type(owner).__name__.removesuffix("Engine").lower()


def split_for_stream(text: str) -> list[str]:
    """One chunk per sentence (long ones split at clauses/words); first chunk kept short for time-to-first-audio."""
    out = []
    for s in re.split(r"(?<=[।.?!])\s+", text):
        if s.strip():
            out += text_normalizer.chunk(s, settings.max_chunk_chars)
    if out and len(out[0]) > settings.first_chunk_chars:
        head = text_normalizer.chunk(out[0], settings.first_chunk_chars)
        out[:1] = [head[0], " ".join(head[1:])] if len(head) > 1 else head
    kept: list[str] = []
    for c in out:  # a chunk with no letter or digit ("।", "...", a quote, a lone matra) must never reach an engine
        if _SPEAKABLE.search(c):
            kept.append(c)
        elif kept:
            kept[-1] += " " + c
    return kept


def trim_lead(wav: np.ndarray, sr: int, keep_ms: int, threshold: float = 0.01) -> np.ndarray:
    loud = np.flatnonzero(np.abs(wav) > threshold)
    if not len(loud):
        return wav
    return wav[max(0, loud[0] - sr * keep_ms // 1000):]


@contextmanager
def _slot():
    if stats["streams_active"] >= settings.max_streams:
        stats["streams_rejected"] += 1
        raise Overloaded(f"at capacity ({settings.max_streams} streams)")
    stats["streams_active"] += 1
    stats["streams_total"] += 1
    try:
        yield
    except (asyncio.CancelledError, GeneratorExit):
        stats["streams_cancelled"] += 1
        raise
    except Exception:
        stats["streams_failed"] += 1
        raise
    finally:
        stats["streams_active"] -= 1


async def _synth_chunk(text: str, voice: str, speed: float, deadline: float, ref, ref_text, controls: dict | None = None) -> np.ndarray:
    key = (voice, text, speed, tuple(sorted(controls.items())) if controls else ())
    if ref is None and key in _cache:
        _cache.move_to_end(key)
        stats["cache_hits"] += 1
        return _cache[key]
    t = time.perf_counter()
    if controls:  # native controls only reach an engine that declared them (prepare_ex/validate_condition guarantee it)
        wav = await scheduler.run(deadline, engine.synth_native, text, voice, speed, controls, ref, ref_text)
    else:
        wav = await scheduler.run(deadline, engine.synth, text, voice, speed, ref, ref_text)
    wav = trim_lead(wav, engine.sample_rate(voice), settings.lead_silence_ms)
    stats["synth_seconds_total"] += time.perf_counter() - t
    if ref is None and settings.cache_size:
        _cache[key] = wav
        if len(_cache) > settings.cache_size:
            _cache.popitem(last=False)
    return wav


async def stream(
    text: str, voice: str, speed: float = 1.0, sample_rate: int | None = None,
    ref=None, ref_text: str | None = None, frame_ms: int = 0, request_id: str = "",
    controls: dict | None = None, dsp_controls: dict | None = None,
) -> AsyncIterator[bytes]:
    """Yield PCM s16le mono bytes at `sample_rate` (None = model rate; the APIs always pass one).

    `controls`: validated native controls for engine.synth_native. `dsp_controls`: {pitch, energy, strength} applied as
    post-processing (app/services/dsp.py) to each chunk after the cache, before resampling.

    Raises Overloaded on the first iteration when at capacity. Cancelling the consuming task stops
    synthesis at once: queued chunks are skipped by the workers, nothing more is yielded.
    """
    with _slot():
        voice = resolve_voice(voice) if ref is None else voice
        sr_in = engine.sample_rate(voice)
        sr_out = sample_rate or sr_in
        rs = soxr.ResampleStream(sr_in, sr_out, 1, dtype="float32", quality="HQ") if sr_out != sr_in else None
        frame = sr_out * frame_ms // 1000 * 2  # bytes; 0 = whole chunks
        chunks = split_for_stream(text_normalizer.normalize(text))
        t0 = time.monotonic()
        sent_s, buf, ttfa = 0.0, b"", None
        for i, c in enumerate(chunks):
            # deadline = when the audio already produced finishes playing (client plays in real time)
            wav = await _synth_chunk(c, voice, speed, t0 + sent_s, ref, ref_text, controls)
            if dsp_controls:
                wav = dsp.apply_prosody(wav, sr_in, dsp_controls.get("pitch"), dsp_controls.get("energy"), dsp_controls.get("strength"))
            sent_s += len(wav) / sr_in
            stats["chunks_total"] += 1
            if rs is not None:
                wav = rs.resample_chunk(wav, last=i == len(chunks) - 1)
            buf += audio_utils.to_pcm16(wav)
            if ttfa is None and buf:
                ttfa = time.monotonic() - t0
                ttfa_recent.append(ttfa)
            if frame:
                whole = len(buf) - len(buf) % frame
                for off in range(0, whole, frame):
                    yield buf[off:off + frame]
                buf = buf[whole:]
            elif buf:
                yield buf
                buf = b""
        if buf:
            yield buf
        stats["audio_seconds_total"] += sent_s
        log.info("stream done", extra={"extra_fields": {
            "request_id": request_id, "chunks": len(chunks), "audio_s": round(sent_s, 2),
            "ttfa_ms": round(ttfa * 1000) if ttfa else None, "total_ms": round((time.monotonic() - t0) * 1000),
        }})


async def synthesize(text: str, voice: str, speed: float = 1.0, sample_rate: int | None = None, ref=None, ref_text=None,
                     request_id: str = "", controls: dict | None = None, dsp_controls: dict | None = None) -> np.ndarray:
    pcm = b"".join([b async for b in stream(text, voice, speed, sample_rate, ref, ref_text, request_id=request_id,
                                            controls=controls, dsp_controls=dsp_controls)])
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32767


def clear_cache(voice: str) -> None:
    for k in [k for k in _cache if k[0] == voice]:
        del _cache[k]
