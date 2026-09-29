"""Pronunciation / intelligibility / signal-quality eval: synthesize a fixed set, transcribe with local Whisper, score CER.

Usage: [ENGINES=piper,supertonic,kokoro] python -m bench.quality [--voices v1,v2] [--no-normalize] [--test-csv data/x/test.csv] [--label name]
Intelligibility = PER (phoneme error rate) between the normalized input and the normalized Whisper transcript,
both phonemized by espeak-ng: script-neutral, so "12,500" vs "बारह हज़ार पाँच सौ" or "loan" vs "लोन" mostly cancel.
Naturalness = UTMOS22 predicted MOS (English-trained; use it to compare voices/settings, not as an absolute).
--no-normalize feeds raw text to the model to show what the normalizer buys.
"""
import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from app.core.config import settings
from app.services import tts
from app.services.text_normalizer import normalize
from training.asr import cer, per, transcribe

_UTMOS = None


def utmos(wav: np.ndarray, sr: int) -> float:
    """UTMOS22 predicted MOS (1-5). English-trained: compare engines/voices/settings, don't read as absolute."""
    global _UTMOS
    import torch

    if _UTMOS is None:
        _UTMOS = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
    with torch.inference_mode():
        return float(_UTMOS(torch.from_numpy(wav.astype(np.float32))[None], sr))

HERE = Path(__file__).parent


def signal(wav: np.ndarray, sr: int, text: str) -> dict:
    loud = np.flatnonzero(np.abs(wav) > 0.01)
    return {
        "clip_ratio": float(np.mean(np.abs(wav) >= 0.999)),
        "rms_dbfs": float(20 * np.log10(np.sqrt(np.mean(wav**2)) + 1e-9)),
        "chars_per_s": len(text.replace(" ", "")) / (len(wav) / sr),
        "silent": not len(loud),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--voices", help="comma separated; default: all installed")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--test-csv", type=Path, help="file|text or file|speaker|text rows (a prepare_dataset test.csv)")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--label", default=time.strftime("quality-%Y%m%d-%H%M%S"))
    a = p.parse_args()

    if a.test_csv:
        items = [("test", r.split("|")[-1]) for r in a.test_csv.read_text("utf-8").splitlines() if r.strip()]
    else:
        items = [tuple(line.split("\t")) for line in (HERE / "quality_set.tsv").read_text("utf-8").splitlines() if line and line[0] != "#"]
    eng = tts.engine  # every engine in ENGINES (default piper); pick voices with --voices
    eng.load()
    voices = a.voices.split(",") if a.voices else [v["voice_id"] for v in eng.voices()]
    out_dir = HERE / "results" / a.label
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"label": a.label, "normalize": not a.no_normalize, "speed": a.speed, "voices": {},
              "settings": {k: getattr(settings, k) for k in ("engines", "indian_english", "noise_scale", "noise_w", "supertonic_steps")}}
    for voice in voices:
        sr = eng.sample_rate(voice)
        per_cat, rows = defaultdict(list), []
        for i, (cat, text) in enumerate(items):
            ref = normalize(text)
            wav = eng.synth(text if a.no_normalize else ref, voice, a.speed)
            sf.write(out_dir / f"{voice}_{i:02d}.wav", wav, sr)
            hyp = transcribe(wav, sr)
            hyp_norm = normalize(hyp)  # ASR writes 12,500 / 5% / English in either script: normalize like ref
            e = per(ref, hyp_norm)
            per_cat[cat].append(e)
            rows.append({"cat": cat, "ref": ref, "hyp": hyp, "per": round(e, 3), "cer": round(cer(ref, hyp_norm), 3),
                         "utmos": round(utmos(wav, sr), 3), **signal(wav, sr, ref)})
        report["voices"][voice] = {
            "per_by_category": {c: round(statistics.mean(v), 3) for c, v in per_cat.items()},
            "per_mean": round(statistics.mean(r["per"] for r in rows), 3),
            "cer_mean": round(statistics.mean(r["cer"] for r in rows), 3),
            "utmos_mean": round(statistics.mean(r["utmos"] for r in rows), 3),
            "clipping_max": max(r["clip_ratio"] for r in rows), "silent_outputs": sum(r["silent"] for r in rows),
            "rms_dbfs_mean": round(statistics.mean(r["rms_dbfs"] for r in rows), 1),
            "chars_per_s_mean": round(statistics.mean(r["chars_per_s"] for r in rows), 1),
            "rows": rows,
        }
        print(voice, json.dumps({k: v for k, v in report["voices"][voice].items() if k != "rows"}, ensure_ascii=False), flush=True)
    (HERE / "results" / f"{a.label}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    cats = sorted({c for c, _ in items})
    print(f"\n| voice | PER {' | PER '.join(cats)} | mean PER | mean CER | UTMOS | clipping | silent |\n|---|{'---|' * len(cats)}---|---|---|---|---|")
    for v, r in report["voices"].items():
        print(f"| {v} | " + " | ".join(f"{r['per_by_category'].get(c, float('nan')):.3f}" for c in cats)
              + f" | {r['per_mean']:.3f} | {r['cer_mean']:.3f} | {r['utmos_mean']:.2f} | {r['clipping_max']:.4f} | {r['silent_outputs']} |")


if __name__ == "__main__":
    main()
