"""Blinded A/B listening-test pack generator and result merger. No results are produced here: only the stimuli, a blank rating sheet and the analysis
of ratings that real listeners filled in. Protocol: docs/listening-test/README.md.

  # 1. make a pack from two models (HF milestone folders or local .onnx); needs the ONNX voices, no ASR
  python -m bench.listening_pack make --a milestones/step_340000 --b milestones/step_355000 --n 20 --seed 7 --out /tmp/pack_340k_vs_355k
  python -m bench.listening_pack make --a-onnx a.onnx --b-onnx b.onnx --a-label old --b-label new --n 20 --out /tmp/pack
        -> <out>/audio/<sample_id>_A.wav, <sample_id>_B.wav   (give ONLY audio/ + ratings_sheet.csv to listeners)
           <out>/ratings_sheet.csv                            (blank; columns below, checkpoint_A/B left empty)
           <out>/KEY_DO_NOT_SHARE.csv                         (sample_id -> which checkpoint is A and B, seed)
  # 2. after listeners return filled sheets (one csv per listener, same sample ids):
  python -m bench.listening_pack merge --key <out>/KEY_DO_NOT_SHARE.csv --ratings l1.csv l2.csv l3.csv --out results.csv
        -> unblinded long table + per-dimension counts, exact two-sided sign tests (ties excluded), listener consistency on the hidden repeats.

Rating cell values: A, B or tie (lower case ok) for naturalness, pronunciation, speaker_similarity, prosody, overall_preference.
Design: same sentence for A and B, random A/B side per item, random item order, RMS-matched loudness, one synthesis per model per item (Piper's noise is
unseeded, so a single draw is one sample: use >= 20 items and several listeners; --consistency N adds N hidden repeats with swapped sides).
"""
import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
DIMS = ["naturalness", "pronunciation", "speaker_similarity", "prosody", "overall_preference"]
SHEET_COLS = ["sample_id", "checkpoint_A", "checkpoint_B", "sentence", *DIMS, "comment"]
TARGET_RMS = 0.05


def sign_test_p(k: int, n: int) -> float:
    """Exact two-sided binomial (sign) test p-value for k wins out of n non-tie votes under p=0.5."""
    if n == 0:
        return 1.0
    k = max(k, n - k)
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * p)


def min_wins_for_significance(n: int, alpha: float = 0.05) -> int | None:
    """Smallest number of wins (out of n non-tie votes) that gives two-sided p < alpha."""
    for k in range(n // 2 + 1, n + 1):
        if sign_test_p(k, n) < alpha:
            return k
    return None


def rms_match(w: np.ndarray, target: float = TARGET_RMS) -> np.ndarray:
    r = float(np.sqrt(np.mean(w.astype(np.float64) ** 2)))
    out = w * (target / r) if r > 1e-6 else w
    peak = float(np.max(np.abs(out)))
    return (out / peak * 0.98 if peak > 0.98 else out).astype(np.float32)


def plan_items(sentences: list[str], n: int, seed: int, consistency: int = 0) -> list[dict]:
    """Choose n sentences, random A/B side per item, then `consistency` hidden repeats (same sentence, sides swapped), shuffled. Pure and seeded."""
    rng = random.Random(seed)
    idx = rng.sample(range(len(sentences)), min(n, len(sentences)))
    items = [{"sentence_index": i, "a_is": rng.choice(["first", "second"]), "repeat_of": None} for i in idx]
    for src in rng.sample(items, min(consistency, len(items))):
        items.append({"sentence_index": src["sentence_index"], "a_is": "second" if src["a_is"] == "first" else "first", "repeat_of": None, "_src": id(src)})
    rng.shuffle(items)
    src_pos = {id(it): k for k, it in enumerate(items)}
    for k, it in enumerate(items):
        it["sample_id"] = f"item{k + 1:02d}"
    for it in items:
        if "_src" in it:
            it["repeat_of"] = items[src_pos[it.pop("_src")]]["sample_id"]
    return items


def cmd_make(a) -> None:
    from bench import compare_checkpoints as cc
    from bench import milestone_eval as me
    from app.services.text_normalizer import normalize

    def model(path, onnx, label):
        if path:
            m = cc.resolve_hf(path)
            return m["onnx"], label or path, m
        return Path(onnx), label or Path(onnx).stem, {"onnx_sha256": cc.sha256(onnx)}

    oa, la, ma = model(a.a, a.a_onnx, a.a_label)
    ob, lb, mb = model(a.b, a.b_onnx, a.b_label)
    if la == lb:
        sys.exit("labels of A and B are identical")
    sents = me.load_sentences(a.sentences)
    items = plan_items(sents, a.n, a.seed, a.consistency)
    synth = {"a": me.make_synth(oa, me.DEFAULT_PARAMS), "b": me.make_synth(ob, me.DEFAULT_PARAMS)}
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    sheet, key = [], []
    for it in items:
        text = sents[it["sentence_index"]]
        ref = normalize(text)
        wa, sra = synth["a"](ref)
        wb, srb = synth["b"](ref)
        first, second = ((wa, sra, la), (wb, srb, lb)) if it["a_is"] == "first" else ((wb, srb, lb), (wa, sra, la))
        for side, (w, sr, _) in zip("AB", (first, second)):
            sf.write(out / "audio" / f"{it['sample_id']}_{side}.wav", rms_match(w), sr)
        sheet.append({"sample_id": it["sample_id"], "checkpoint_A": "", "checkpoint_B": "", "sentence": text, **{d: "" for d in DIMS}, "comment": ""})
        key.append({"sample_id": it["sample_id"], "checkpoint_A": first[2], "checkpoint_B": second[2], "sentence": text, "repeat_of": it["repeat_of"] or "", "seed": a.seed})
        print(it["sample_id"], "ok", flush=True)
    with open(out / "ratings_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, SHEET_COLS); w.writeheader(); w.writerows(sheet)
    with open(out / "KEY_DO_NOT_SHARE.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(key[0])); w.writeheader(); w.writerows(key)
    (out / "pack.json").write_text(json.dumps({"a": {"label": la, **{k: str(v) for k, v in ma.items() if k != "onnx"}}, "b": {"label": lb, **{k: str(v) for k, v in mb.items() if k != "onnx"}},
                                               "n_items": len(items), "consistency_repeats": a.consistency, "seed": a.seed, "inference_params": me.DEFAULT_PARAMS,
                                               "loudness": f"RMS matched to {TARGET_RMS}", "needed_wins_p<0.05_if_no_ties": min_wins_for_significance(len(items))}, indent=1, default=str))
    print(f"pack in {out}: give listeners audio/ and ratings_sheet.csv only; keep KEY_DO_NOT_SHARE.csv")


def read_csv(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merge(key_rows: list[dict], sheets: list[list[dict]]) -> dict:
    """Unblind and tally. Returns {rows, tallies, consistency}. A cell is A/B/tie/'' (blank = unrated, skipped)."""
    key = {r["sample_id"]: r for r in key_rows}
    rows, tallies = [], defaultdict(Counter)
    for li, sheet in enumerate(sheets, 1):
        for r in sheet:
            k = key[r["sample_id"]]
            row = {"listener": li, "sample_id": r["sample_id"], "checkpoint_A": k["checkpoint_A"], "checkpoint_B": k["checkpoint_B"], "sentence": k["sentence"]}
            for d in DIMS:
                v = (r.get(d) or "").strip()
                v = {"a": "A", "b": "B", "tie": "tie", "": ""}.get(v.lower(), None)
                if v is None:
                    raise ValueError(f"{r['sample_id']} {d}: value must be A, B or tie")
                row[d] = v
                if v == "A" or v == "B":
                    tallies[d][k["checkpoint_" + v]] += 1
                elif v == "tie":
                    tallies[d]["tie"] += 1
            rows.append(row)
    cons = {"pairs": 0, "agree": 0}
    by = {(x["listener"], x["sample_id"]): x for x in rows}
    for sid, k in key.items():
        if k.get("repeat_of"):
            for li in range(1, len(sheets) + 1):
                x, y = by.get((li, sid)), by.get((li, k["repeat_of"]))
                if not x or not y or not x["overall_preference"] or not y["overall_preference"]:
                    continue
                pick = lambda row: row["checkpoint_" + row["overall_preference"]] if row["overall_preference"] in ("A", "B") else "tie"
                cons["pairs"] += 1
                cons["agree"] += pick(x) == pick(y)
    summary = {}
    names = sorted({r["checkpoint_A"] for r in rows} | {r["checkpoint_B"] for r in rows})
    for d, c in tallies.items():
        if len(names) == 2:
            w0, w1 = c[names[0]], c[names[1]]
            summary[d] = {names[0]: w0, names[1]: w1, "tie": c["tie"], "n_non_tie": w0 + w1, "sign_test_p_two_sided": round(sign_test_p(w0, w0 + w1), 4)}
    return {"rows": rows, "summary": summary, "consistency": cons}


def cmd_merge(a) -> None:
    res = merge(read_csv(a.key), [read_csv(p) for p in a.ratings])
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(res["rows"][0])); w.writeheader(); w.writerows(res["rows"])
    print(json.dumps({"summary": res["summary"], "consistency": res["consistency"], "listeners": len(a.ratings)}, indent=1))
    print("Votes from several listeners on the same items are not independent: treat the p-values as optimistic; report per-listener counts too.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--a"); m.add_argument("--b"); m.add_argument("--a-onnx"); m.add_argument("--b-onnx"); m.add_argument("--a-label"); m.add_argument("--b-label")
    m.add_argument("--n", type=int, default=20); m.add_argument("--seed", type=int, default=7); m.add_argument("--consistency", type=int, default=3)
    m.add_argument("--sentences", default=str(HERE / "hi_eval_50.txt")); m.add_argument("--out", required=True)
    g = sub.add_parser("merge")
    g.add_argument("--key", required=True); g.add_argument("--ratings", nargs="+", required=True); g.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "make":
        if not ((a.a or a.a_onnx) and (a.b or a.b_onnx)):
            ap.error("give --a/--a-onnx and --b/--b-onnx")
        cmd_make(a)
    else:
        cmd_merge(a)


if __name__ == "__main__":
    main()
