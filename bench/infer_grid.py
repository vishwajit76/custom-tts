"""Inference-parameter grid for a milestone: noise_scale x noise_w x length_scale -> CER (+UTMOS) on a fixed 15-sentence subset.

Usage: python -m bench.infer_grid --hf-step 350000 [--asr small] [--repeats 2]
       python -m bench.infer_grid --onnx voices/hi_IN-custom-medium.onnx --step 350000
Grid: noise_scale {0.5, 0.667, 0.8} x noise_w {0.6, 0.8, 1.0} x length_scale {1.0, 1.1} (18 configs; 0.667/0.8/1.0 is the default).
Scores the SUBSET of bench/hi_eval_50.txt (1-based line numbers in SUBSET below, spread over all five categories).
Writes bench/results/infer_grid_<step>.json (all configs, ranked) and keeps audio of 3 sentences only for the 5 best configs plus the
default in docs/samples/infer_grid/<config>/ (config = ns<noise_scale>_nw<noise_w>_ls<length_scale>).
Ranking = rank-sum of (CER mean ascending, UTMOS mean descending). With 15 sentences the CER noise is about +-0.02, so differences below
that are not real: treat the result as "which region of the grid to prefer", not a precise optimum. Server defaults are NOT changed.
"""
import argparse
import itertools
import json
import shutil
import statistics
import time
from pathlib import Path

import soundfile as sf

from bench import milestone_eval as me

NS, NW, LS = (0.5, 0.667, 0.8), (0.6, 0.8, 1.0), (1.0, 1.1)
SUBSET = [1, 4, 7, 10, 15, 18, 22, 27, 29, 33, 37, 41, 45, 47, 49]  # 1-based positions among the non-comment lines of hi_eval_50.txt
AUDIO_SUBSET = [4, 29, 47]  # short-ish conversational / numbers / question, saved for listening
DEFAULT = (0.667, 0.8, 1.0)
AUDIO_DIR = me.ROOT / "docs" / "samples" / "infer_grid"


def cfg_name(ns, nw, ls) -> str:
    return f"ns{ns}_nw{nw}_ls{ls}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--onnx")
    src.add_argument("--hf-step", type=int)
    ap.add_argument("--hf-folder")
    ap.add_argument("--step", type=int)
    ap.add_argument("--asr", default="small")
    ap.add_argument("--repeats", type=int, default=2, help="synthesize each sentence N times per config and average CER")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--limit-configs", type=int, help="smoke test: only the first N configs")
    ap.add_argument("--out")
    a = ap.parse_args()

    if a.hf_step:
        onnx, folder = me.resolve_hf(a.hf_step, a.hf_folder)
        step = a.hf_step
    else:
        onnx, step, folder = Path(a.onnx), a.step or 0, a.onnx
    allsents = me.load_sentences(me.DEFAULT_SENTENCES)
    sents = {i: allsents[i - 1] for i in SUBSET}
    from app.services.text_normalizer import normalize
    from training.asr import cer

    utmos = me.Utmos()
    results = []
    configs = list(itertools.product(NS, NW, LS))[:a.limit_configs]
    t0 = time.time()
    for ns, nw, ls in configs:
        synth = me.make_synth(onnx, {"noise_scale": ns, "noise_w": nw, "length_scale": ls})
        cs, us, per_sent, audio = [], [], {}, {}
        for i, text in sents.items():
            ref = normalize(text)
            cc = []
            for r in range(a.repeats):
                wav, sr = synth(ref)
                if r == 0:
                    us.append(utmos(wav, sr))
                    if i in AUDIO_SUBSET:
                        audio[i] = (wav, sr)
                cc.append(cer(ref, normalize(me.transcribe(a.asr, wav, sr))))
            cs.append(statistics.mean(cc)); per_sent[i] = round(cs[-1], 4)
        row = {"name": cfg_name(ns, nw, ls), "noise_scale": ns, "noise_w": nw, "length_scale": ls, "cer_mean": round(statistics.mean(cs), 4),
               "cer_median": round(statistics.median(cs), 4), "utmos_mean": round(statistics.mean(us), 3) if utmos.fn else None,
               "per_sentence_cer": per_sent, "is_default": (ns, nw, ls) == DEFAULT}
        results.append((row, audio))
        print(f"{row['name']:22s} cer {row['cer_mean']:.4f} utmos {row['utmos_mean']} ({time.time()-t0:.0f}s)", flush=True)

    rows = [r for r, _ in results]
    by_cer = {r["name"]: k for k, r in enumerate(sorted(rows, key=lambda r: r["cer_mean"]))}
    by_utm = {r["name"]: k for k, r in enumerate(sorted(rows, key=lambda r: -(r["utmos_mean"] or 0)))}
    for r in rows:
        r["rank_sum"] = by_cer[r["name"]] + by_utm[r["name"]]
    ranked = sorted(rows, key=lambda r: (r["rank_sum"], r["cer_mean"]))
    keep = {r["name"] for r in ranked[:a.top]} | {r["name"] for r in rows if r["is_default"]}
    if not a.no_audio:
        for row, audio in results:
            if row["name"] in keep:
                d = AUDIO_DIR / row["name"]
                shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
                for i, (wav, sr) in audio.items():
                    sf.write(d / f"s{i:02d}.wav", wav, sr, subtype="PCM_16")
    out = Path(a.out or me.HERE / "results" / f"infer_grid_{step}.json")
    out.write_text(json.dumps({"step": step, "source": str(folder), "asr": a.asr, "repeats": a.repeats, "subset_lines": SUBSET, "audio_lines": AUDIO_SUBSET,
                               "audio_kept_for": sorted(keep), "utmos": utmos.note, **me.now_stamps(), "ranked": ranked}, ensure_ascii=False, indent=1), encoding="utf8")
    print("\nrank cfg cer utmos")
    for k, r in enumerate(ranked, 1):
        print(f"{k:2d} {r['name']:22s} cer {r['cer_mean']:.4f} utmos {r['utmos_mean']}{'  (default)' if r['is_default'] else ''}")
    print("wrote", out)


if __name__ == "__main__":
    main()
