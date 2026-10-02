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
from app.services import audio_utils, dsp, expressive_engine, text_normalizer, voice_catalog
from app.services.conditioning import DEFAULT_CAPABILITIES, EngineCapabilities
from app.services.scheduler import Scheduler

log = logging.getLogger(__name__)
_SPEAKABLE = re.compile(r"[^\W_]")  # a letter or digit; matras and virama are marks, not letters
_SENTENCE_END = re.compile(r"[।.?!]\W*$")  # closing quotes/brackets may follow
_MIN_S_PER_CHAR = 0.05  # speech per character at speed 1 is ~0.065-0.085 s (Kokoro/Supertonic/Piper Hindi); a floor


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
        owner = self._owner(voice)
        return owner.capabilities_for(voice) if hasattr(owner, "capabilities_for") else getattr(owner, "capabilities", DEFAULT_CAPABILITIES)

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
_cache_epoch = 0  # bumped by clear_cache(): a chunk synthesized before a voice was replaced/deleted must not be cached after it
stats = {
    "streams_active": 0, "streams_total": 0, "streams_rejected": 0, "streams_cancelled": 0, "streams_failed": 0,
    "chunks_total": 0, "cache_hits": 0, "audio_seconds_total": 0.0, "synth_seconds_total": 0.0,
}
ttfa_recent: deque[float] = deque(maxlen=2000)  # seconds, for /metrics quantiles


def load() -> None:
    global scheduler
    check_audio_settings()
    voice_catalog.get()  # validate voices/catalog.json up front: a bad catalog stops startup, not the first request
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
    if (target := voice_catalog.get().alias_target(voice)) and engine.has_voice(target):  # catalog alias -> its voice id
        return target
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
    # The second chunk synthesizes while the first one plays (stream's look-ahead), so it must not dwarf it: after a
    # short "नमस्ते," a 120-char rest was still synthesizing when the first had finished playing, a 0.2-0.5 s stall mid-
    # sentence. ~3x the first chunk + 20 chars keeps it inside that window at the measured RTF ~0.25 on CPU.
    if len(out) > 1 and not _SENTENCE_END.search(out[0]):
        out[1:2] = text_normalizer.cut_at_phrase(out[1], 3 * len(out[0]) + 20)
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


def trim_tail(wav: np.ndarray, sr: int, keep_ms: int, threshold: float = 0.01) -> np.ndarray:
    loud = np.flatnonzero(np.abs(wav) > threshold)
    if not len(loud):
        return wav
    return wav[:loud[-1] + 1 + sr * keep_ms // 1000]


def _trailing(wav: np.ndarray, threshold: float = 0.01) -> int:
    loud = np.flatnonzero(np.abs(wav) > threshold)
    return len(wav) - 1 - loud[-1] if len(loud) else 0


def pad_lead(wav: np.ndarray, sr: int, ms: int, threshold: float = 0.01) -> np.ndarray:
    """Zeros in front so the silence before the first sample above `threshold` is exactly `ms` (trim_lead only shortens)."""
    loud = np.flatnonzero(np.abs(wav) > threshold)
    short = sr * ms // 1000 - (loud[0] if len(loud) else 0)
    return np.concatenate([np.zeros(short, wav.dtype), wav]) if len(loud) and short > 0 else wav


def pad_tail(wav: np.ndarray, sr: int, ms: int, threshold: float = 0.01) -> np.ndarray:
    """Zeros appended so the silence after the last sample above `threshold` is `ms`; never shortens."""
    short = sr * ms // 1000 - _trailing(wav, threshold)
    return np.concatenate([wav, np.zeros(short, wav.dtype)]) if len(wav) and short > 0 else wav


_GAP_CLASSES = {"comma", "colon", "dash", "ellipsis", "sentence", "question", "exclaim", "phrase"}
_CLOSERS = "\"'”’»)]}"


def gap_class(text: str) -> str:
    """How a piece ends, for settings.pause_plan: its last punctuation mark (closing quotes/brackets ignored)."""
    t = text.rstrip().rstrip(_CLOSERS).rstrip()
    if t.endswith(("...", "…")):
        return "ellipsis"
    return {"।": "sentence", ".": "sentence", "?": "question", "!": "exclaim", ",": "comma", ";": "colon", ":": "colon",
            "-": "dash", "–": "dash", "—": "dash"}.get(t[-1:], "phrase")


def gap_ms(text: str) -> int | None:
    """Planned silence after this piece (settings.pause_plan), None when the plan is off."""
    plan = audio_utils.parse_kv(settings.pause_plan)
    return round(plan[gap_class(text)]) if plan else None


def check_audio_settings() -> None:
    """Fail at startup, not on the first request, if the pause plan or the voice gains do not parse."""
    if unknown := set(audio_utils.parse_kv(settings.pause_plan)) - _GAP_CLASSES:
        raise ValueError(f"PAUSE_PLAN: unknown class(es) {sorted(unknown)}; have {sorted(_GAP_CLASSES)}")
    audio_utils.parse_kv(settings.voice_gain_db)


def voice_gain(voice: str) -> float:
    gains = audio_utils.parse_kv(settings.voice_gain_db)
    db = gains.get(voice, gains.get(voice.split(":")[0], 0.0))
    return 10 ** (db / 20)


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
    key = (voice, text, speed, tuple(sorted(controls.items())) if controls else (), settings.pause_plan, settings.fade_ms)
    if ref is None and key in _cache:
        _cache.move_to_end(key)
        stats["cache_hits"] += 1
        return _cache[key]
    t = time.perf_counter()
    epoch = _cache_epoch
    if controls:  # native controls only reach an engine that declared them (prepare_ex/validate_condition guarantee it)
        wav = await scheduler.run(deadline, engine.synth_native, text, voice, speed, controls, ref, ref_text)
    else:
        wav = await scheduler.run(deadline, engine.synth, text, voice, speed, ref, ref_text)
    sr = engine.sample_rate(voice)
    wav = trim_lead(wav, sr, settings.lead_silence_ms)
    gap = gap_ms(text)
    if gap is None:
        wav = trim_tail(wav, sr, settings.sentence_pause_ms if _SENTENCE_END.search(text) else settings.pause_ms)
    else:  # the trailing silence is capped to the planned gap less the next piece's lead; stream() pads it up
        wav = trim_tail(wav, sr, max(0, gap - settings.lead_silence_ms))
    if settings.fade_ms:
        wav = audio_utils.edge_fades(wav, sr, settings.fade_ms)  # before the lead pad: fades the cut, not the padding
    if gap is not None:
        wav = pad_lead(wav, sr, settings.lead_silence_ms)
    stats["synth_seconds_total"] += time.perf_counter() - t
    if ref is None and settings.cache_size and epoch == _cache_epoch:
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
        chunks = split_for_stream(text_normalizer.normalize(text, voice_catalog.rules_of(voice)))
        t0 = time.monotonic()
        sent_s, buf, ttfa = 0.0, b"", None

        def synth(c: str, deadline: float) -> asyncio.Future:
            return asyncio.ensure_future(_synth_chunk(c, voice, speed, deadline, ref, ref_text, controls))

        # deadline = when the audio already produced finishes playing (client plays in real time)
        ahead = synth(chunks[0], t0) if chunks else None
        try:
            for i, c in enumerate(chunks):
                cur, ahead = ahead, None
                # One chunk of look-ahead: chunk i+1 synthesizes while chunk i is sent and played. Without it a long
                # chunk after the short first one started only once that one was out, and the player ran dry mid-
                # sentence (measured 0.2-0.5 s on Kokoro, heard as a pause after "ठीक है,"). Its deadline estimates
                # chunk i's length from its text, on the short side so the prefetch is never later than reality.
                if i + 1 < len(chunks):
                    ahead = synth(chunks[i + 1], t0 + sent_s + _MIN_S_PER_CHAR * len(c) / speed)
                wav = await cur
                if i + 1 < len(chunks) and (gap := gap_ms(c)) is not None:
                    wav = pad_tail(wav, sr_in, gap - settings.lead_silence_ms)
                if (g := voice_gain(voice) if ref is None else 1.0) != 1.0:
                    wav = wav * np.float32(g)  # a new array: `wav` may be the cached chunk
                if dsp_controls:
                    wav = await asyncio.to_thread(dsp.apply_prosody, wav, sr_in, dsp_controls.get("pitch"), dsp_controls.get("energy"), dsp_controls.get("strength"))
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
        finally:  # cancelled, failed or abandoned: the look-ahead chunk must not keep a worker busy
            if ahead is not None:
                ahead.cancel()
                if ahead.done() and not ahead.cancelled():
                    ahead.exception()  # retrieved: a failure nobody will await is not "never retrieved"
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
    global _cache_epoch
    _cache_epoch += 1  # also drops the result of any synthesis still in flight for this voice (store is skipped)
    for k in [k for k in _cache if k[0] == voice]:
        del _cache[k]
