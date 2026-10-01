"""Objective voice/control evaluation: identical sentences x voices x conditions -> one JSON report.

Usage: [ENGINES=piper,kokoro] [DSP_PROSODY=1] python -m bench.eval [--voices v1,v2] [--conditions conds.json]
           [--sentences file.txt] [--reference speaker.wav] [--cer] [--label eval]
Writes bench/results/<label>.json. Conditions default to: neutral, speed 0.85/1.15, pitch +2 / energy 1.3 (each only
counts if the engine or DSP really applies it; otherwise the row says status="unsupported" with the 422 reason).

What is measured (all objective signal statistics, none of it is a naturalness or emotion score):
  duration, chars/s, F0 median / spread in semitones / voiced fraction (librosa pyin), RMS level mean/std, pause count and
  length, clipping ratio, silence ratio and lead/trail silence, speaker similarity, 8 kHz telephony round trip
  (log-spectral distance in the 0.1-3.4 kHz band, F0 shift, similarity loss) and optional CER.
Caveats, on purpose:
  * Speaker similarity uses SPEAKER_ENCODER (default mfcc = statistics of MFCCs, NOT a neural verifier). Only compare
    scores within one backend; with mfcc, different voices can still score high. It is similarity to a reference clip
    (--reference) or, without one, to the same voice's neutral output.
  * CER needs a local Whisper (transformers + weights already cached; nothing is downloaded here). If unavailable the
    field is {"skipped": reason}. ASR error is a proxy for intelligibility only.
  * Predicted MOS (bench/quality.py's UTMOS) is NOT a human MOS: an English-trained model's guess. Only a blind
    native-listener test measures "sounds natural"; none has been run.
  * F0 from pyin on synthetic speech is noisy (octave errors, voiced fraction 0.5-0.65): compare medians over many
    sentences, not single clips.
  * Pitch/energy rows produced by DSP are signal processing, not expression, and are labelled applied="pitch:dsp".
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import numpy as np
import soxr

HERE = Path(__file__).parent
DEFAULT_CONDITIONS = [
    {"name": "neutral", "condition": None},
    {"name": "speed-0.85", "condition": {"speed": 0.85}},
    {"name": "speed-1.15", "condition": {"speed": 1.15}},
    {"name": "pitch+2", "condition": {"pitch": 2.0}},
    {"name": "energy-1.3", "condition": {"energy": 1.3}},
]
FRAME_MS, HOP_MS = 25, 10


def _frames_db(wav: np.ndarray, sr: int) -> np.ndarray:
    n, h = int(sr * FRAME_MS / 1000), int(sr * HOP_MS / 1000)
    if len(wav) < n:
        return np.array([-120.0])
    idx = np.arange(0, len(wav) - n + 1, h)
    rms = np.sqrt(np.array([np.mean(wav[i:i + n] ** 2) for i in idx]) + 1e-12)
    return 20 * np.log10(rms + 1e-9)


def signal_metrics(wav: np.ndarray, sr: int, text: str = "", silence_db: float = -45.0, min_pause_ms: int = 120) -> dict:
    wav = np.asarray(wav, dtype=np.float32)
    db = _frames_db(wav, sr)
    speech = db > silence_db
    dur = len(wav) / sr
    out = {"duration_s": round(dur, 3), "chars_per_s": round(len(text) / dur, 2) if dur and text else None,
           "clipping_ratio": float(np.mean(np.abs(wav) >= 0.999)) if len(wav) else 0.0,
           "peak_dbfs": round(20 * np.log10(float(np.max(np.abs(wav))) + 1e-9), 1) if len(wav) else None,
           "silence_ratio": round(float(1 - speech.mean()), 3), "silent": not speech.any()}
    if speech.any():
        first, last = int(np.argmax(speech)), len(speech) - 1 - int(np.argmax(speech[::-1]))
        out["lead_silence_ms"], out["trail_silence_ms"] = first * HOP_MS, (len(speech) - 1 - last) * HOP_MS
        out["rms_dbfs_mean"], out["rms_dbfs_std"] = round(float(db[speech].mean()), 1), round(float(db[speech].std()), 1)
        inner = speech[first:last + 1]
        runs, cur = [], 0
        for s in inner:
            if not s:
                cur += 1
            elif cur:
                runs.append(cur * HOP_MS)
                cur = 0
        runs = [r for r in runs if r >= min_pause_ms]
        out["pauses"] = {"count": len(runs), "total_ms": sum(runs), "max_ms": max(runs, default=0),
                         "mean_ms": round(statistics.mean(runs)) if runs else 0}
    out["f0"] = f0_stats(wav, sr)
    return out


def f0_stats(wav: np.ndarray, sr: int) -> dict:
    import librosa

    w16 = soxr.resample(wav, sr, 16000, quality="HQ") if sr != 16000 else wav
    if len(w16) < 4096:
        return {"skipped": "too short"}
    f0, voiced, _ = librosa.pyin(w16, fmin=65, fmax=500, sr=16000, frame_length=1024)
    v = f0[voiced & np.isfinite(f0)]
    if len(v) < 5:
        return {"voiced_fraction": 0.0}
    st = 12 * np.log2(v / np.median(v))
    return {"median_hz": round(float(np.median(v)), 1), "spread_semitones_p5_p95": round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2),
            "std_semitones": round(float(st.std()), 2), "voiced_fraction": round(float(len(v) / len(f0)), 3)}


def roundtrip_8k(wav: np.ndarray, sr: int, encoder=None) -> dict:
    """wav -> 8 kHz -> back to `sr` (what a PSTN listener's audio loses). LSD is over 100-3400 Hz."""
    wav = np.asarray(wav, dtype=np.float32)
    rt = soxr.resample(soxr.resample(wav, sr, 8000, quality="HQ"), 8000, sr, quality="HQ")
    n = min(len(wav), len(rt))
    a, b = wav[:n], rt[:n]
    nfft = 1024
    def spec(x):
        f = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(x, nfft)[::nfft // 2] * np.hanning(nfft), axis=1)) + 1e-8
        return 20 * np.log10(f)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    band = (freqs >= 100) & (freqs <= 3400)
    if n >= nfft:
        sa, sb = spec(a)[:, band], spec(b)[:, band]
        lsd = float(np.sqrt(np.mean((sa - sb) ** 2, axis=1)).mean())
    else:
        lsd = None
    total = float(np.sum(np.abs(np.fft.rfft(a)) ** 2)) + 1e-12
    hi = float(np.sum(np.abs(np.fft.rfft(a))[np.fft.rfftfreq(n, 1 / sr) > 4000] ** 2))
    out = {"lsd_db_100_3400hz": round(lsd, 2) if lsd is not None else None, "energy_above_4khz_lost_pct": round(100 * hi / total, 2),
           "f0_median_hz_after": f0_stats(rt, sr).get("median_hz")}
    if encoder is not None:
        out["speaker_similarity_after"] = _similarity(encoder, encoder_embed(encoder, wav, sr), encoder_embed(encoder, rt, sr))
    return out


def encoder_embed(encoder, wav, sr):
    try:
        return encoder.embed(wav, sr)
    except Exception:  # noqa: BLE001 - too short / silent / backend missing: reported as null
        return None


def _similarity(encoder, a, b):
    return None if a is None or b is None else round(float(encoder.similarity(a, b)), 4)


def try_asr():
    """(transcribe, cer) from training.asr if a local Whisper can be loaded, else (None, reason). Never downloads on purpose."""
    try:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from training.asr import cer, transcribe

        transcribe(np.zeros(16000, np.float32), 16000)
        return (transcribe, cer), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def evaluate(synth, voices: list[str], sentences: list[str], conditions: list[dict], encoder=None,
             reference: tuple[np.ndarray, int] | None = None, asr=None, asr_skip: str = "not requested") -> dict:
    """synth(text, voice, condition|None) -> (wav float32, sr, applied list, ignored list); raises ValueError(reason) if the
    combination is unsupported. Identical sentences and conditions for every voice."""
    ref_emb = encoder_embed(encoder, *reference) if encoder is not None and reference else None
    report = {"encoder": getattr(encoder, "name", None), "encoder_neural": getattr(encoder, "neural", None),
              "reference": "given clip" if reference else "same voice, neutral condition", "voices": {}}
    for voice in voices:
        vrep, neutral_embs = {}, {}
        for cond in conditions:
            rows = []
            for i, text in enumerate(sentences):
                try:
                    wav, sr, applied, ignored = synth(text, voice, cond["condition"])
                except ValueError as e:
                    rows.append({"i": i, "status": "unsupported", "reason": str(e)})
                    continue
                emb = encoder_embed(encoder, wav, sr) if encoder is not None else None
                if cond["name"] == "neutral":
                    neutral_embs[i] = emb
                base = ref_emb if ref_emb is not None else neutral_embs.get(i)
                row = {"i": i, "status": "ok", "applied": applied, "ignored": ignored, **signal_metrics(wav, sr, text),
                       "speaker_similarity": _similarity(encoder, emb, base) if encoder is not None and emb is not None and base is not None else None,
                       "roundtrip_8k": roundtrip_8k(wav, sr, encoder)}
                if encoder is not None and emb is None:
                    row["speaker_similarity_note"] = "encoder rejected the audio (under 1 s, silent, or backend unavailable)"
                if asr is not None:
                    row["cer"] = round(asr[1](text, asr[0](wav, sr)), 3)
                rows.append(row)
            ok = [r for r in rows if r["status"] == "ok"]
            vrep[cond["name"]] = {
                "condition": cond["condition"], "n_ok": len(ok), "n_unsupported": len(rows) - len(ok),
                "cer": {"skipped": asr_skip} if asr is None else {"mean": round(statistics.mean(r["cer"] for r in ok), 3) if ok else None},
                "rows": rows}
        report["voices"][voice] = vrep
    return report


def default_synth(dsp_only_label: bool = True):
    """Synthesis through the real request path (prepare_ex -> tts.synthesize), so support/labels match the API."""
    from fastapi import HTTPException

    from app.api.speech import prepare_ex
    from app.models.schemas import SpeechRequest
    from app.services import tts

    def synth(text, voice, condition):
        req = SpeechRequest(input=text, voice=voice, sample_rate=None, condition=condition)
        try:
            kwargs, applied, ignored = prepare_ex(req)
        except HTTPException as e:
            raise ValueError(str(e.detail)) from e
        wav = asyncio.run(tts.synthesize(**kwargs))
        return wav, kwargs["sample_rate"], applied, ignored

    return synth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", default="")
    ap.add_argument("--conditions", type=Path, help="JSON list of {name, condition}")
    ap.add_argument("--sentences", type=Path, default=HERE / "sentences.txt")
    ap.add_argument("--limit", type=int, default=5, help="sentences per cell")
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--cer", action="store_true")
    ap.add_argument("--label", default="eval")
    a = ap.parse_args()
    import soundfile as sf

    from app.core.config import settings
    from app.services import tts
    from app.services.speaker_encoder import get_encoder

    tts.load()
    voices = a.voices.split(",") if a.voices else [v["voice_id"] for v in tts.engine.voices()]
    sentences = [s for s in a.sentences.read_text("utf-8").splitlines() if s.strip()][:a.limit]
    conditions = json.loads(a.conditions.read_text("utf-8")) if a.conditions else DEFAULT_CONDITIONS
    asr, skip = (None, "not requested")
    if a.cer:
        asr, skip = try_asr()
        skip = skip or "n/a"
    ref = sf.read(a.reference, dtype="float32") if a.reference else None
    ref = (ref[0].mean(axis=1) if ref is not None and ref[0].ndim > 1 else ref[0], ref[1]) if ref is not None else None
    t0 = time.time()
    rep = evaluate(default_synth(), voices, sentences, conditions, get_encoder(settings.speaker_encoder), ref, asr, skip)
    rep |= {"label": a.label, "date": time.strftime("%Y-%m-%d"), "engines": settings.engines, "dsp_prosody": settings.dsp_prosody,
            "elapsed_s": round(time.time() - t0, 1),
            "notes": "Objective signal statistics only. Predicted MOS is not human MOS; speaker similarity backend is "
                     "reported in `encoder` (mfcc is not a neural verifier); DSP pitch/energy is not emotion."}
    out = HERE / "results" / f"{a.label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
