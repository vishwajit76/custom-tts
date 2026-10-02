"""Objective audio-quality bench through the real server pipeline (tts.stream / tts.synthesize, no HTTP).

Usage: python -m bench.audio_quality --out bench/results/audio_quality_before.json [--voices hi_IN-custom-medium,hi_IN-rohan-medium]
           [--set pause_ms=150 ...]   override a setting, e.g. to reproduce the pre-T6 behaviour
           [--ttfa 20]                TTFA repeats (warm) per voice; 0 = skip
           [--calibrate]              print the per-voice static gain (dB) that puts each voice at --target-lufs

Per utterance (native rate, streamed chunk by chunk): lead/trailing silence, silent gap at each chunk seam, player stall at
each seam, loudness, peak / true peak, clipped fraction, DC offset, discontinuity at each seam; per telephony rate (8/16 kHz):
level change, guard-band and alias energy, length ratio.

Methods
  loudness   ITU-R BS.1770-4 K-weighted, 400 ms blocks / 75 % overlap, absolute gate -70 LUFS, relative gate -10 LU,
             mono (G = 1). Audio shorter than one block (a one-word reply) is measured over its own length, not zero-padded.
  true peak  4x oversampled (scipy resample_poly) max |x|, dBTP.
  edge       largest |sample| at the first/last sample of any chunk, dBFS (what a hard trim leaves: a step this size is a click).
  clipped    fraction of s16 samples at +-32767 after the server's limiter.
  silence    first/last sample above SILENCE_THR (-50 dBFS); gap at a seam = trailing silence of chunk i + lead of chunk i+1.
  stall      arrival time of chunk i+1 minus the moment the audio of chunks 0..i has finished playing from the first byte
             (> 0 = a real-time player ran dry). On an idle machine the look-ahead should make it <= 0.
  seam jump  |x[s] - x[s-1]| at the first sample of the next chunk, and that over the 99th percentile |diff| of the utterance.
  flux spike largest half-wave-rectified spectral flux (5 ms window, 2.5 ms hop) within 5 ms of the seam over the 95th
             percentile flux of the utterance: > ~2 = a spectral event that is not an ordinary onset.
  alias      8/16 kHz stream vs an ideal reference (the native audio FFT-resampled: brick-wall at the new Nyquist, so nothing
             folds): guard_db = energy of the stream in the top 10 % below the new Nyquist over the reference's, positive =
             aliasing/imaging (a good filter rolls off: negative); pass_db = same for 100 Hz..0.9 Nyquist (should be ~0);
             fold_db = energy the native holds above the new Nyquist, dB re native total (what a bad decimation would fold in);
             level_change_db = LUFS(stream) - LUFS(native).
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy import signal

SILENCE_THR = 10 ** (-50 / 20)
SENTENCES = [  # id, text: short replies, questions, long comma sentences, Hinglish, every punctuation class
    ("yes", "जी हाँ।"),
    ("no", "जी नहीं।"),
    ("ok", "ठीक है।"),
    ("wait", "एक मिनट रुकिए।"),
    ("yn-question", "क्या आप अभी बात कर सकते हैं?"),
    ("wh-question", "आपका नाम क्या है?"),
    ("exclaim", "बहुत बढ़िया!"),
    ("apology", "माफ़ कीजिए, मुझे आपकी बात समझ नहीं आई।"),
    ("greeting", "नमस्ते, मैं श्रेया बोल रही हूँ और आपके loan के बारे में बात करना चाहती हूँ, क्या आप अभी बात कर सकते हैं?"),
    ("long-commas", "आपकी किस्त की तारीख पंद्रह अगस्त है, इसलिए कृपया उससे पहले भुगतान कर दीजिए, नहीं तो आपके खाते पर अतिरिक्त शुल्क लगेगा, और यह आपकी क्रेडिट रेटिंग पर भी असर डाल सकता है।"),
    ("hinglish", "आपका EMI ₹5,000 है, due date 5 तारीख को है, please payment time पर कर दीजिए।"),
    ("three-sentences", "नमस्ते। मैं आपकी मदद के लिए यहाँ हूँ। बताइए, मैं आपकी क्या सहायता कर सकती हूँ?"),
    ("ellipsis-dash", "हम्म... मुझे सोचने दीजिए। शायद कल - या परसों - हो जाएगा।"),
    ("colon-semicolon", "ध्यान दीजिए: आपका OTP चार अंकों का है; इसे किसी से साझा न करें।"),
]
TTFA_TEXT = "नमस्ते, मैं श्रेया बोल रही हूँ और आपके loan के बारे में बात करना चाहती हूँ।"


# ---------------------------------------------------------------- metrics (no app import: tests use these)

def _shelf_hp(sr: int) -> np.ndarray:
    """BS.1770-4 pre-filter (high shelf) and RLB (high-pass), as second-order sections for any sample rate."""
    g, q, fc = 3.999843853973347, 0.7071752369554196, 1681.9744509555319
    k = np.tan(np.pi * fc / sr)
    vh = 10 ** (g / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = [(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0, 1, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    q, fc = 0.5003270373253953, 38.13547087613982
    k = np.tan(np.pi * fc / sr)
    a0 = 1 + k / q + k * k
    hp = [1, -2, 1, 1, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    return np.array([shelf, hp])


def lufs(x: np.ndarray, sr: int) -> float:
    """Integrated loudness, BS.1770-4 (see module docstring). -inf for digital silence."""
    loud = np.flatnonzero(np.abs(x) > 1e-5)
    if not len(loud):
        return float("-inf")
    y = signal.sosfilt(_shelf_hp(sr), x[loud[0]:loud[-1] + 1].astype(np.float64))
    n, hop = int(0.4 * sr), int(0.1 * sr)
    if len(y) < n:
        z = np.array([np.mean(y ** 2)])
    else:
        z = np.array([np.mean(y[i:i + n] ** 2) for i in range(0, len(y) - n + 1, hop)])
    z = z[z > 0]
    if not len(z):
        return float("-inf")
    l = -0.691 + 10 * np.log10(z)
    z = z[l > -70]
    if not len(z):
        return float("-inf")
    z = z[-0.691 + 10 * np.log10(z) > -0.691 + 10 * np.log10(z.mean()) - 10]
    return float(-0.691 + 10 * np.log10(z.mean()))


def db(v: float) -> float:
    return float(20 * np.log10(max(v, 1e-9)))


def true_peak_db(x: np.ndarray) -> float:
    return db(float(np.abs(signal.resample_poly(x.astype(np.float64), 4, 1)).max(initial=0)))


def lead_trail_ms(x: np.ndarray, sr: int, thr: float = SILENCE_THR) -> tuple[float, float]:
    loud = np.flatnonzero(np.abs(x) > thr)
    if not len(loud):
        return len(x) / sr * 1000, 0.0
    return loud[0] / sr * 1000, (len(x) - 1 - loud[-1]) / sr * 1000


def clipped_fraction(x: np.ndarray) -> float:
    return float(np.mean(np.abs(np.rint(x * 32767)) >= 32767)) if len(x) else 0.0


def flux(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Half-wave-rectified spectral flux per frame and the hop in samples (5 ms Hann window, 2.5 ms hop)."""
    nper = max(16, int(sr * 0.005))
    hop = nper // 2
    _, _, s = signal.stft(x, sr, window="hann", nperseg=nper, noverlap=nper - hop, boundary=None, padded=False)
    m = np.abs(s)
    return np.maximum(m[:, 1:] - m[:, :-1], 0).sum(axis=0), hop


def seam_stats(x: np.ndarray, sr: int, idx: int) -> dict:
    d = np.abs(np.diff(x))
    jump = float(abs(x[idx] - x[idx - 1])) if 0 < idx < len(x) else 0.0
    p99 = float(np.percentile(d, 99)) if len(d) else 0.0
    f, hop = flux(x, sr)
    w = max(2, int(0.005 * sr / hop))
    c = idx // hop
    p95 = float(np.percentile(f, 95)) if len(f) else 0.0
    near = f[max(0, c - w):c + w + 1]
    return {"jump": round(jump, 5), "jump_ratio": round(jump / p99, 3) if p99 > 1e-6 else 0.0,
            "flux_spike": round(float(near.max(initial=0)) / p95, 3) if p95 > 1e-9 else 0.0}


def band_energy(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    fr = np.fft.rfftfreq(len(x), 1 / sr)
    return float(sp[(fr >= lo) & (fr < hi)].sum())


def telephony_stats(native: np.ndarray, sr_in: int, out: np.ndarray, sr_out: int) -> dict:
    nyq = sr_out / 2
    ref = signal.resample(native.astype(np.float64), int(round(len(native) * sr_out / sr_in)))
    ref = np.rint(ref * 32767) / 32767  # s16 like the stream: both carry the same quantization noise floor
    tot_in = band_energy(native, sr_in, 0, sr_in / 2 + 1)
    lead, trail = lead_trail_ms(out, sr_out)
    ln, lo = lufs(native, sr_in), lufs(out, sr_out)

    def ratio(f0, f1):
        return round(10 * np.log10((band_energy(out, sr_out, f0, f1) + 1e-12) / (band_energy(ref, sr_out, f0, f1) + 1e-12)), 2)

    return {
        "level_change_db": round(lo - ln, 3),
        "guard_db": ratio(0.9 * nyq, nyq + 1), "pass_db": ratio(100, 0.9 * nyq),
        "fold_db": round(10 * np.log10(band_energy(native, sr_in, nyq, sr_in / 2 + 1) / tot_in + 1e-12), 2),
        "dur_ratio": round((len(out) / sr_out) / (len(native) / sr_in), 4),
        "lead_ms": round(lead, 1), "trail_ms": round(trail, 1),
        "peak_db": round(db(float(np.abs(out).max(initial=0))), 2), "clip_frac": round(clipped_fraction(out), 6),
    }


def utterance_stats(chunks: list[np.ndarray], sr: int, arrivals: list[float] | None = None) -> dict:
    """Metrics of one streamed utterance given its chunks (float, native rate) and optional arrival times (s)."""
    wav = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    lead, trail = lead_trail_ms(wav, sr)
    gaps, stalls, seams, starts = [], [], [], np.cumsum([0] + [len(c) for c in chunks])
    for i in range(1, len(chunks)):
        gaps.append(round(lead_trail_ms(chunks[i - 1], sr)[1] + lead_trail_ms(chunks[i], sr)[0], 1))
        seams.append(seam_stats(wav, sr, int(starts[i])))
        if arrivals:
            stalls.append(round((arrivals[i] - (arrivals[0] + starts[i] / sr)) * 1000, 1))
    cl = [lufs(c, sr) for c in chunks if len(c) > sr // 10]
    cl = [v for v in cl if np.isfinite(v)]
    return {
        "dur_s": round(len(wav) / sr, 3), "chunks": len(chunks), "lead_ms": round(lead, 1), "trail_ms": round(trail, 1),
        "gaps_ms": gaps, "stall_ms": stalls, "lufs": round(lufs(wav, sr), 2),
        "peak_db": round(db(float(np.abs(wav).max(initial=0))), 2), "true_peak_db": round(true_peak_db(wav), 2),
        "clip_frac": round(clipped_fraction(wav), 6), "dc": round(float(wav.mean()) if len(wav) else 0.0, 6),
        "edge_db": round(db(max((max(abs(float(c[0])), abs(float(c[-1]))) for c in chunks if len(c)), default=0.0)), 1),
        "seams": seams, "chunk_lufs": [round(v, 2) for v in cl],
        "chunk_lufs_spread_db": round(max(cl) - min(cl), 2) if len(cl) > 1 else 0.0,
    }


# ---------------------------------------------------------------- rendering through the server pipeline

def pcm(b: bytes) -> np.ndarray:
    return np.frombuffer(b, "<i2").astype(np.float32) / 32767


async def render(tts, text: str, voice: str) -> tuple[list[np.ndarray], list[float], int]:
    """Chunks as the server yields them at the native rate (frame_ms=0: one yield per synthesized chunk)."""
    sr = tts.engine.sample_rate(voice)
    chunks, arrivals, t0 = [], [], time.monotonic()
    async for b in tts.stream(text, voice, 1.0, sr, frame_ms=0):
        chunks.append(pcm(b))
        arrivals.append(time.monotonic() - t0)
    return chunks, arrivals, sr


async def bench_voice(tts, voice: str) -> list[dict]:
    rows = []
    for sid, text in SENTENCES:
        chunks, arrivals, sr = await render(tts, text, voice)
        row = {"voice": voice, "id": sid, "text": text, "pieces": tts.split_for_stream(tts.text_normalizer.normalize(text)),
               **utterance_stats(chunks, sr, arrivals)}
        native = await tts.synthesize(text, voice, 1.0, sr)
        row["synthesize_matches_stream"] = bool(len(native) == sum(len(c) for c in chunks))
        row["tel"] = {}
        for rate in (8000, 16000):
            out = await tts.synthesize(text, voice, 1.0, rate)
            row["tel"][str(rate)] = telephony_stats(native, sr, out, rate)
        rows.append(row)
    return rows


async def ttfa(tts, voice: str, n: int, rate: int | None) -> dict:
    sr = rate or tts.engine.sample_rate(voice)
    xs = []
    for i in range(n + 3):
        t = time.perf_counter()
        async for _ in tts.stream(TTFA_TEXT, voice, 1.0, sr, frame_ms=0):
            xs.append(time.perf_counter() - t)
            break
        await asyncio.sleep(0.05)  # let the look-ahead chunk finish so runs do not queue on each other
    xs = sorted(x * 1000 for x in xs[3:])  # first 3 are warm-up
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"n": len(xs), "p50_ms": round(statistics.median(xs), 1), "p95_ms": round(q(0.95), 1), "min_ms": round(xs[0], 1), "max_ms": round(xs[-1], 1)}


def summarize(rows: list[dict]) -> dict:
    def agg(vals):
        v = sorted(vals)
        return {"min": round(v[0], 2), "median": round(statistics.median(v), 2), "max": round(v[-1], 2)} if v else None

    end = [r for r in rows if r["text"].rstrip()[-1] in "।.?!"]
    return {
        "lead_ms": agg([r["lead_ms"] for r in rows]),
        "trail_ms_sentence_end": agg([r["trail_ms"] for r in end]),
        "gap_ms": agg([g for r in rows for g in r["gaps_ms"]]),
        "stall_ms": agg([s for r in rows for s in r["stall_ms"]]),
        "lufs": agg([r["lufs"] for r in rows if np.isfinite(r["lufs"])]),
        "lufs_std": round(float(np.std([r["lufs"] for r in rows if np.isfinite(r["lufs"])])), 2),
        "chunk_lufs_spread_db": agg([r["chunk_lufs_spread_db"] for r in rows if r["chunks"] > 1]),
        "true_peak_db": agg([r["true_peak_db"] for r in rows]),
        "edge_db_max": max(r["edge_db"] for r in rows),
        "clip_frac_max": max(r["clip_frac"] for r in rows),
        "dc_abs_max": max(abs(r["dc"]) for r in rows),
        "seam_jump_ratio": agg([s["jump_ratio"] for r in rows for s in r["seams"]]),
        "seam_flux_spike": agg([s["flux_spike"] for r in rows for s in r["seams"]]),
        **{f"tel{k}_{m}": agg([r["tel"][k][m] for r in rows]) for k in ("8000", "16000") for m in ("level_change_db", "guard_db", "pass_db", "fold_db")},
    }


def _coerce(old, v: str):
    return v.lower() in ("1", "true", "yes") if isinstance(old, bool) else type(old)(v)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--voices", default="")
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "models/piper"))
    ap.add_argument("--models-extra", default=os.environ.get("MODELS_EXTRA", "voices"))
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--ttfa", type=int, default=20)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--target-lufs", type=float, default=-23.0)
    a = ap.parse_args()
    os.environ.update(ENGINES="piper", MODELS_DIR=a.models_dir, MODELS_EXTRA=a.models_extra)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.core.config import settings
    from app.services import tts

    for kv in a.set:
        k, v = kv.split("=", 1)
        setattr(settings, k, _coerce(getattr(settings, k), v))
    tts.load()
    have = [v["voice_id"] for v in tts.engine.voices()]
    voices = [v for v in a.voices.split(",") if v] or have
    out = {"settings": {k: getattr(settings, k) for k in ("lead_silence_ms", "pause_ms", "sentence_pause_ms") + tuple(
        k for k in type(settings).model_fields if k.startswith(("pause_", "fade_", "voice_gain", "telephony_")))},
        "overrides": a.set, "sentences": [{"id": i, "text": t} for i, t in SENTENCES], "voices": {}}
    for v in voices:
        # VITS is stochastic (noise_w changes durations by +-10 % per run): the chunk cache makes the native, 8 kHz and 16 kHz
        # renders of one utterance the same samples, so telephony is compared against the audio it was made from.
        settings.cache_size = 4096
        rows = await bench_voice(tts, v)
        out["voices"][v] = {"summary": summarize(rows), "utterances": rows}
        settings.cache_size = 0  # TTFA: every run is a real synthesis
        tts._cache.clear()
        if a.ttfa:
            out["voices"][v]["ttfa_native"] = await ttfa(tts, v, a.ttfa, None)
            out["voices"][v]["ttfa_8k"] = await ttfa(tts, v, a.ttfa, 8000)
        if a.calibrate:
            ls = [r["lufs"] for r in rows if np.isfinite(r["lufs"]) and r["dur_s"] > 1.0]
            print(f"{v}: median LUFS {statistics.median(ls):.2f} -> gain {a.target_lufs - statistics.median(ls):+.2f} dB")
        print(f"{v}: {json.dumps(out['voices'][v]['summary'])}")
        if a.ttfa:
            print(f"{v}: ttfa native {out['voices'][v]['ttfa_native']} 8k {out['voices'][v]['ttfa_8k']}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    sys.stdout.flush()
    os._exit(0)  # ONNX Runtime threads abort the interpreter during normal teardown


if __name__ == "__main__":
    asyncio.run(main())
