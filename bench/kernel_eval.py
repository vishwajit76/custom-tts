"""Milestone evaluation run INSIDE the Kaggle kernel (spawned as a subprocess by training/kaggle/train_kernel.py right after a milestone export).

  python -m bench.kernel_eval --onnx m.onnx --step 5000 --data-dir <data/hi_f> --out evaluation.json [--prev prev_evaluation.json] [--budget-s 900]
        [--repeats 3] [--asr small] [--device cuda --compute float16] [--limit N] [--meta-json '{...}'] [--fake]

It reuses bench.compare_checkpoints.score_checkpoint (same entry as a local `compare_checkpoints` run: CER, PER, UTMOS, speaker similarity, 8k/16k telephony
CER, two-level bootstrap CIs, raw matrices) and adds paired deltas against the previous stored evaluation (its `results[0].matrices`). Writes one JSON in the
compare_checkpoints/v1 layout (`results` = [entry]) plus `paired_vs_previous`. Exit 0 = written, 3 = budget exhausted (nothing written), 1 = error.
--fake replaces ASR and UTMOS with deterministic stubs (CPU smoke test of the kernel path: no model download, no GPU).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from bench import compare_checkpoints as cc
from bench import milestone_eval as me


def local_refs(data_dir: Path, n: int) -> list[Path]:
    """Real held-out clips (3-10 s) from <data_dir>/test.csv and <data_dir>/wavs (the kernel downloads data/hi_f from HF): same rule as milestone_eval.pick_refs."""
    csv = Path(data_dir) / "test.csv"
    if not csv.exists():
        return []
    out = []
    for line in csv.read_text("utf-8").splitlines():
        if not line.strip():
            continue
        p = Path(data_dir) / "wavs" / line.split("|", 1)[0]
        if p.exists() and 3.0 <= sf.info(p).duration <= 10.0:
            out.append(p)
        if len(out) == n:
            break
    return out


class FakeUtmos:
    note = "FAKE (smoke test)"
    fn = True

    def __call__(self, wav, sr):
        return 3.0 + float(np.abs(wav).mean())


def fake_transcribe(model, wav, sr):
    return "नमस्ते आप कैसे हैं" if len(wav) % 2 else "नमस्ते आप कैसे है"


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prev", help="previous evaluation json (for paired deltas)")
    ap.add_argument("--budget-s", type=float, default=900)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--asr", default="small")
    ap.add_argument("--device", default=None, help="faster-whisper device (default: ASR_DEVICE env or cpu)")
    ap.add_argument("--compute", default=None, help="faster-whisper compute type (default: ASR_COMPUTE_TYPE env or int8)")
    ap.add_argument("--sentences", default=str(me.DEFAULT_SENTENCES))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--n-refs", type=int, default=12)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--meta-json", default="{}")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    if a.device:
        me.ASR_DEVICE = a.device
    if a.compute:
        me.ASR_COMPUTE = a.compute
    if a.fake:
        me.transcribe = fake_transcribe
    onnx = Path(a.onnx)
    meta = {"path": None, "global_step": a.step, "onnx_sha256": sha256(onnx), "hf_commit": None, **json.loads(a.meta_json), "onnx": onnx}
    sents = me.load_sentences(a.sentences)[:a.limit]
    params = dict(me.DEFAULT_PARAMS)
    utmos = FakeUtmos() if a.fake else me.Utmos()
    refs = local_refs(Path(a.data_dir), a.n_refs)
    print(f"kernel_eval step {a.step}: {len(sents)} sentences x {a.repeats}, asr {a.asr} {me.ASR_DEVICE}/{me.ASR_COMPUTE}, {len(refs)} refs, budget {a.budget_s}s", flush=True)
    try:
        entry, raw = cc.score_checkpoint(meta, sents, params, a.repeats, a.asr, utmos, refs, b=a.bootstrap, seed=0, deadline=t0 + a.budget_s)
    except cc.EvalTimeout as e:
        print("budget exhausted:", e, flush=True)
        sys.exit(3)
    out = {"schema": cc.SCHEMA, "kind": "kernel_milestone_eval", "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "eval_set": cc.eval_set_info(a.sentences), "asr": a.asr, "asr_backend": f"faster-whisper {me.ASR_DEVICE}/{me.ASR_COMPUTE}, beam 5, hi", "inference_params": params,
           "repeats": a.repeats, "n_reference_clips": len(refs), "fake": bool(a.fake), "results": [entry], "paired_vs_previous": None}
    if a.prev:
        try:
            pe = json.loads(Path(a.prev).read_text("utf-8"))
            pr = pe["results"][0]
            pm = pr["matrices"]
            if len(pm["cer"]) != len(sents):
                raise ValueError(f"previous has {len(pm['cer'])} sentences, this run {len(sents)}")
            out["paired_vs_previous"] = {"baseline": {"step": pr["checkpoint"].get("global_step"), "hf_path": pr["checkpoint"].get("hf_path"), "onnx_sha256": pr["checkpoint"].get("onnx_sha256"),
                                                      "experiment_id": pr.get("experiment_id")},
                                         "candidate": {"step": a.step, "hf_path": meta.get("path")}, "delta_is": "candidate minus baseline (negative CER/PER = better)",
                                         **cc.paired_block(pm, entry["matrices"], a.bootstrap, 0)}
        except Exception as e:  # noqa: BLE001
            out["paired_vs_previous"] = {"error": repr(e)[:200]}
    out["eval_seconds"] = round(time.time() - t0)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    c = entry["cer"]
    print(f"wrote {a.out} in {out['eval_seconds']}s: CER {c['mean']} {c['ci95']} PER {entry['per']['mean']} UTMOS {entry['utmos'].get('mean')}", flush=True)


if __name__ == "__main__":
    main()
