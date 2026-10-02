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

V7 test (any set of systems, 1-5 ratings on six dimensions + blinded A/B, sealed key, anchor, analysis with CIs): docs/listening-test/v7/README.md
  python -m bench.listening_pack make-v7 --system v6=voices/hi_IN-custom-medium.onnx --system v7=exp/v7.onnx@young_female --n-items 20 --listeners 5 --out /tmp/v7pack
  python -m bench.listening_pack analyze-v7 --key /tmp/v7pack/organizer/KEY_DO_NOT_SHARE.json --rating l1_rating.csv ... --ab l1_ab.csv ... --out docs/listening-test/v7/results/run1.json

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


# ---------------------------------------------------------------- V7 test (docs/listening-test/v7/README.md): any set of systems, blinded ids, sealed key
DIMS_V7 = ["naturalness", "pronunciation", "prosody", "clarity", "conversational_realism", "speaker_consistency"]
AB_COLS = ["item_id", "text", *DIMS_V7, "overall_preference", "comment"]
RATE_COLS = ["stimulus_id", "text", *DIMS_V7, "comment"]
ANCHOR = "ANCHOR"
V7_RATE = 24000  # every stimulus is resampled to one rate: a differing file sample rate would reveal the model


def _tokens(rng: random.Random, n: int) -> list[str]:
    out: set[str] = set()
    while len(out) < n:
        out.add(f"{rng.getrandbits(32):08x}")
    lst = sorted(out)  # sorted first: set order is hash-randomized per process
    rng.shuffle(lst)
    return lst


def plan_v7(systems: list[str], item_ids: list[str], seed: int, pairs: list[tuple[str, str]] | None = None, anchor: bool = True) -> dict:
    """Pure and seeded. rating stimuli = item x (system + anchor), blind id each; A/B items = item x pair, random side, blind id each.
    The key (blind id -> system) is what must stay sealed; nothing else in the plan names a system."""
    rng = random.Random(seed)
    pairs = pairs if pairs is not None else [(systems[i], systems[j]) for i in range(len(systems)) for j in range(i + 1, len(systems))]
    names = [*systems, *([ANCHOR] if anchor else [])]
    cells = [(it, s) for it in item_ids for s in names]
    ab_cells = [(it, p) for it in item_ids for p in pairs]
    tok = _tokens(rng, len(cells) + len(ab_cells))
    stimuli = {tok[k]: {"item": it, "system": s} for k, (it, s) in enumerate(cells)}
    ab = {}
    for k, (it, (x, y)) in enumerate(ab_cells):
        a, b = (x, y) if rng.random() < 0.5 else (y, x)
        ab[tok[len(cells) + k]] = {"item": it, "A": a, "B": b}
    return {"seed": seed, "systems": systems, "pairs": [list(p) for p in pairs], "anchor": anchor, "stimuli": stimuli, "ab": ab}


def degrade_anchor(wav: np.ndarray, sr: int, seed: int = 0) -> np.ndarray:
    """Low anchor: band-limited to 4 kHz and noisy. Listeners who rate it as good as the systems are not discriminating."""
    import soxr

    lo = soxr.resample(soxr.resample(np.asarray(wav, np.float32), sr, 8000, quality="HQ"), 8000, sr, quality="HQ") if sr > 8000 else np.asarray(wav, np.float32)
    lo = soxr.resample(soxr.resample(lo, sr, 4000, quality="HQ"), 4000, sr, quality="HQ")
    rms = float(np.sqrt(np.mean(lo ** 2))) + 1e-9
    return (lo + np.random.default_rng(seed).normal(0, rms * 0.1, len(lo))).astype(np.float32)


def _prep(wav: np.ndarray, sr: int) -> np.ndarray:
    import soxr

    return rms_match(soxr.resample(np.asarray(wav, np.float32), sr, V7_RATE, quality="HQ") if sr != V7_RATE else np.asarray(wav, np.float32))


def render_v7(plan: dict, audio: dict, texts: dict, out: Path, listeners: int = 3, provenance: dict | None = None) -> dict:
    """audio[(system, item)] = (float wav, sr) for every system AND item (the anchor is derived). Writes out/listener_pack/{audio,sheets} and out/organizer/
    {KEY_DO_NOT_SHARE.json, pack_manifest.json}. Returns the key. The listener pack contains only blind ids and corpus text."""
    import hashlib

    lp, org = out / "listener_pack", out / "organizer"
    (lp / "audio").mkdir(parents=True, exist_ok=True)
    (lp / "sheets").mkdir(exist_ok=True)
    org.mkdir(parents=True, exist_ok=True)
    for sid, s in plan["stimuli"].items():
        w, sr = audio[(s["system"], s["item"])] if s["system"] != ANCHOR else (lambda a: (degrade_anchor(a[0], a[1], plan["seed"]), a[1]))(audio[(plan["systems"][0], s["item"])])
        sf.write(lp / "audio" / f"r_{sid}.wav", _prep(w, sr), V7_RATE, subtype="PCM_16")
    for iid, s in plan["ab"].items():
        for side in "AB":
            w, sr = audio[(s[side], s["item"])]
            sf.write(lp / "audio" / f"ab_{iid}_{side}.wav", _prep(w, sr), V7_RATE, subtype="PCM_16")
    for k in range(1, listeners + 1):  # same blind audio, a different order per listener
        rng = random.Random(plan["seed"] * 1000 + k)
        for name, cols, ids, idcol, extra in (("rating", RATE_COLS, list(plan["stimuli"]), "stimulus_id", plan["stimuli"]), ("ab", AB_COLS, list(plan["ab"]), "item_id", plan["ab"])):
            rng.shuffle(ids)
            with open(lp / "sheets" / f"listener_{k:02d}_{name}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, cols)
                w.writeheader()
                w.writerows({idcol: i, "text": texts[extra[i]["item"]]} for i in ids)
    key = {"seed": plan["seed"], "rate_hz": V7_RATE, "stimuli": plan["stimuli"], "ab": plan["ab"], "systems": plan["systems"], "pairs": plan["pairs"], "anchor": plan["anchor"],
           "provenance": provenance or {}}
    kb = json.dumps(key, ensure_ascii=False, indent=1, sort_keys=True).encode("utf-8")
    (org / "KEY_DO_NOT_SHARE.json").write_bytes(kb)
    (org / "pack_manifest.json").write_text(json.dumps({
        "key_sha256": hashlib.sha256(kb).hexdigest(), "n_rating_stimuli": len(plan["stimuli"]), "n_ab_items": len(plan["ab"]), "listeners": listeners, "seed": plan["seed"],
        "note": "key_sha256 commits to the sealed key at pack time: record it (e.g. in the repo) BEFORE listeners start, so the assignment cannot be changed afterwards; analyze verifies it"}, indent=1))
    return key


def _read_scores(path, cols, allowed):
    rows = read_csv(path)
    for r in rows:
        for c in cols:
            v = (r.get(c) or "").strip()
            if v and v.lower() not in allowed:
                raise ValueError(f"{path}: {r.get('stimulus_id') or r.get('item_id')} {c}: {v!r} not in {sorted(allowed)}")
    return rows


def _boot_mean(by_item: dict, b: int, seed: int) -> list[float]:
    """95% CI of the mean over all ratings, resampling ITEMS (a listener's ratings of one sentence are not independent of each other's)."""
    items = list(by_item)
    s = np.array([sum(by_item[i]) for i in items], float)
    c = np.array([len(by_item[i]) for i in items], float)
    idx = np.random.default_rng(seed).integers(0, len(items), size=(b, len(items)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def analyze_v7(key: dict, rating_sheets: dict[str, list[dict]], ab_sheets: dict[str, list[dict]], b: int = 2000, seed: int = 0) -> dict:
    """Unblind and summarize. Ratings are integers 1-5 (blank = not rated); A/B cells are A, B or tie. Never mixed with automated metrics: this output is the human evidence."""
    out: dict = {"n_listeners": {"rating": len(rating_sheets), "ab": len(ab_sheets)}, "scale": "1 (bad) to 5 (excellent); means are of ratings, not MOS predictions",
                 "ratings": {}, "rating_pair_diffs": [], "ab": [], "listener_checks": {}}
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # dim -> system -> item -> [values]
    per_listener_anchor: dict = defaultdict(lambda: defaultdict(list))
    for lid, sheet in rating_sheets.items():
        for r in sheet:
            k = key["stimuli"][r["stimulus_id"]]
            for d in DIMS_V7:
                v = (r.get(d) or "").strip()
                if v:
                    data[d][k["system"]][k["item"]].append(int(v))
                    if d == "naturalness":
                        per_listener_anchor[lid][k["system"]].append(int(v))
    for d in DIMS_V7:
        out["ratings"][d] = {}
        for s, by in data[d].items():
            vals = [x for v in by.values() for x in v]
            out["ratings"][d][s] = {"mean": round(float(np.mean(vals)), 3), "ci95": _boot_mean(by, b, seed), "n_ratings": len(vals), "n_items": len(by)}
        systems = [s for s in key["systems"] if s in data[d]]
        for i, x in enumerate(systems):
            for y in systems[i + 1:]:
                common = sorted(set(data[d][x]) & set(data[d][y]))
                if len(common) < 3:
                    continue
                dx = np.array([np.mean(data[d][y][c]) - np.mean(data[d][x][c]) for c in common])
                idx = np.random.default_rng(seed).integers(0, len(dx), size=(b, len(dx)))
                bm = dx[idx].mean(1)
                lo, hi = float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))
                out["rating_pair_diffs"].append({"dimension": d, "baseline": x, "candidate": y, "delta_mean": round(float(dx.mean()), 3), "ci95": [round(lo, 3), round(hi, 3)],
                                                 "excludes_zero": bool(lo > 0 or hi < 0), "n_items": len(common)})
    for lid, by in per_listener_anchor.items():
        sysm = {s: float(np.mean(v)) for s, v in by.items() if s != ANCHOR}
        anc = float(np.mean(by[ANCHOR])) if ANCHOR in by else None
        out["listener_checks"][lid] = {"anchor_naturalness_mean": None if anc is None else round(anc, 2), "min_system_naturalness_mean": round(min(sysm.values()), 2) if sysm else None,
                                       "flag_anchor_not_below_all_systems": bool(anc is not None and sysm and anc >= min(sysm.values()))}
    wins: dict = defaultdict(lambda: defaultdict(Counter))  # (a,b) sorted -> dim -> Counter(system|tie)
    per_l: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    for lid, sheet in ab_sheets.items():
        for r in sheet:
            k = key["ab"][r["item_id"]]
            pair = tuple(sorted((k["A"], k["B"])))
            for d in [*DIMS_V7, "overall_preference"]:
                v = (r.get(d) or "").strip().lower()
                if v in ("a", "b"):
                    w = k[v.upper()]
                elif v == "tie":
                    w = "tie"
                elif v == "":
                    continue
                else:
                    raise ValueError(f"{r['item_id']} {d}: {v!r} not A, B or tie")
                wins[pair][d][w] += 1
                per_l[pair][d][lid][w] += 1
    n_tests = max(1, sum(len(v) for v in wins.values()))
    for pair, dims in sorted(wins.items()):
        for d, c in dims.items():
            x, y = pair
            n = c[x] + c[y]
            out["ab"].append({"pair": [x, y], "dimension": d, x: c[x], y: c[y], "tie": c["tie"], "n_non_tie": n, "sign_test_p_two_sided": round(sign_test_p(c[x], n), 4),
                              "bonferroni_alpha": round(0.05 / n_tests, 5), "per_listener": {l: {x: v[x], y: v[y], "tie": v["tie"]} for l, v in per_l[pair][d].items()}})
    out["ab_note"] = ("sign test on non-tie votes pooled over listeners; votes on the same items are not independent, so p-values are optimistic: read per_listener and the item count. "
                      "bonferroni_alpha is 0.05 divided by the number of (pair, dimension) tests")
    return out


def cmd_make_v7(a) -> None:
    from bench import corpus as corpus_mod
    from bench import v7_eval as ve

    specs = list(a.system)
    if a.systems_file:
        for e in json.loads(Path(a.systems_file).read_text("utf-8")):
            if e.get("spec"):
                specs.append(f"{e['name']}={e['spec']}")
            else:
                print(f"skip {e['name']}: placeholder, no voice yet")
    if len(specs) < 2:
        sys.exit("need at least two systems")
    rows = corpus_mod.stratified(corpus_mod.load(a.corpus), a.n_items, a.seed)
    systems = ve.build_piper_systems(specs, True, a.seed)
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",")] if a.pairs else None
    audio = {}
    for s in systems:
        syn = ve.synthesize_all(s, rows, 1)
        for r, w in zip(rows, syn["audio"]):
            audio[(s.name, r["id"])] = (w.astype(np.float32) / 32767, syn["sr"])
    plan = plan_v7([s.name for s in systems], [r["id"] for r in rows], a.seed, pairs, not a.no_anchor)
    prov = {s.name: s.info() for s in systems} | {"corpus": a.corpus, "corpus_sha256": corpus_mod.VERSIONS[a.corpus][1]}
    render_v7(plan, audio, {r["id"]: r["text"] for r in rows}, Path(a.out), a.listeners, prov)
    print(f"pack in {a.out}: give listeners ONLY {a.out}/listener_pack; keep {a.out}/organizer sealed (record pack_manifest.json key_sha256 before the test)")


def cmd_analyze_v7(a) -> None:
    import hashlib

    kb = Path(a.key).read_bytes()
    man = Path(a.key).with_name("pack_manifest.json")
    if man.exists() and json.loads(man.read_text())["key_sha256"] != hashlib.sha256(kb).hexdigest():
        sys.exit("KEY sha256 does not match pack_manifest.json: the key was changed after the pack was made")
    key = json.loads(kb)
    rating = {Path(p).stem: _read_scores(p, DIMS_V7, {"1", "2", "3", "4", "5"}) for p in a.rating}
    ab = {Path(p).stem: _read_scores(p, [*DIMS_V7, "overall_preference"], {"a", "b", "tie"}) for p in a.ab}
    res = analyze_v7(key, rating, ab, a.bootstrap, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", "utf-8")
    for d, m in res["ratings"].items():
        print(d, {s: f"{v['mean']} {v['ci95']}" for s, v in m.items()})
    print("wrote", a.out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--a"); m.add_argument("--b"); m.add_argument("--a-onnx"); m.add_argument("--b-onnx"); m.add_argument("--a-label"); m.add_argument("--b-label")
    m.add_argument("--n", type=int, default=20); m.add_argument("--seed", type=int, default=7); m.add_argument("--consistency", type=int, default=3)
    m.add_argument("--sentences", default=str(HERE / "hi_eval_50.txt")); m.add_argument("--out", required=True)
    v = sub.add_parser("make-v7")
    v.add_argument("--system", action="append", default=[], help="NAME=PATH.onnx[@SPEAKER] | NAME=voice:ID (see bench/v7_eval.py)")
    v.add_argument("--systems-file", help="docs/listening-test/v7/systems.json; entries with spec null are placeholders and skipped")
    v.add_argument("--corpus", default="v2"); v.add_argument("--n-items", type=int, default=20); v.add_argument("--seed", type=int, default=7)
    v.add_argument("--listeners", type=int, default=5); v.add_argument("--pairs", help="a:b,c:d (default: every pair)"); v.add_argument("--no-anchor", action="store_true")
    v.add_argument("--out", required=True)
    z = sub.add_parser("analyze-v7")
    z.add_argument("--key", required=True); z.add_argument("--rating", nargs="*", default=[]); z.add_argument("--ab", nargs="*", default=[])
    z.add_argument("--bootstrap", type=int, default=2000); z.add_argument("--seed", type=int, default=0); z.add_argument("--out", required=True)
    g = sub.add_parser("merge")
    g.add_argument("--key", required=True); g.add_argument("--ratings", nargs="+", required=True); g.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "make-v7":
        return cmd_make_v7(a)
    if a.cmd == "analyze-v7":
        return cmd_analyze_v7(a)
    if a.cmd == "make":
        if not ((a.a or a.a_onnx) and (a.b or a.b_onnx)):
            ap.error("give --a/--a-onnx and --b/--b-onnx")
        cmd_make(a)
    else:
        cmd_merge(a)


if __name__ == "__main__":
    main()
