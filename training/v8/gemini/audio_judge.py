"""Gemini multimodal audio judge (offline automated proxy, NOT human MOS). Judges only existing wavs; generates no audio.
usage: audio_judge.py [--limit N] [--ab engA,engB]"""
import argparse, csv, hashlib, json, random, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import gem

DIMS = ["naturalness", "pronunciation", "prosody", "tone_match", "conversational"]
SCORE_PROMPT = """You are a strict Hindi TTS evaluator. Listen to the audio. Intended text: "{text}". Intended style: {style}.
Score 1-5 (5 best) and return JSON: {{"naturalness":int,"pronunciation":int,"prosody":int,"tone_match":int,"conversational":int,"issues":[short strings, max 5]}}.
pronunciation = does the speech match the intended text; tone_match = matches intended style."""
AB_PROMPT = """Two audio clips of the same Hindi text: "{text}" (intended style: {style}). Clip 1 first, then Clip 2.
Which sounds better overall (naturalness, pronunciation, prosody, tone)? Return JSON {{"winner":"1"|"2"|"tie","reason":str}}."""


def parse_score(o):
    """Validate a judge result; returns dict or None."""
    if not isinstance(o, dict):
        return None
    out = {}
    for d in DIMS:
        v = o.get(d)
        if not isinstance(v, (int, float)) or not 1 <= v <= 5:
            return None
        out[d] = float(v)
    out["issues"] = [str(i) for i in o.get("issues", [])][:5]
    return out


def ab_order(key, seed=0):
    """Deterministic randomized order: True means (A,B) presented as (1,2)."""
    return random.Random(f"{seed}:{key}").random() < 0.5


def tally(results):
    """Mean per-dim per-engine over parsed results {engine: [score dicts]}."""
    return {e: {**{d: round(statistics.mean(r[d] for r in rs), 2) for d in DIMS}, "n": len(rs)} for e, rs in results.items() if rs}


def part(path):
    from google.genai import types
    return types.Part.from_bytes(data=Path(path).read_bytes(), mime_type="audio/wav")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style_of(row):
    return next((n.split("=")[1] for n in row["notes"].split(";") if n.strip().startswith("emotion=")), "neutral")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ab", default="piper_v7a_cpu,kokoro_goonj")
    a = ap.parse_args()
    bk = gem.ROOT / "benchmarks/v8_model_bakeoff"
    corpus = {r["id"]: r for r in csv.DictReader(open(gem.ROOT / "bench/corpus/hi_eval_v2.tsv", encoding="utf-8"), delimiter="\t")}
    res, per = {}, {}
    for eng in sorted(p.name for p in (bk / "samples").iterdir() if p.is_dir()):
        for w in sorted((bk / "samples" / eng).glob("*.wav"))[: a.limit or None]:
            row = corpus.get(w.stem)
            if not row:
                continue
            p = SCORE_PROMPT.format(text=row["text"], style=style_of(row))
            try:
                s = parse_score(gem.ask_json(gem.PRO, p, parts=[part(w)], extra=sha(w)))
            except Exception as e:
                print("fail", eng, w.name, type(e).__name__)
                continue
            if s:
                res.setdefault(eng, []).append(s)
                per.setdefault(eng, {})[w.stem] = s
    table = tally(res)
    ab = {}
    ea, eb = a.ab.split(",")
    for w in sorted((bk / "samples" / ea).glob("*.wav")):
        wb = bk / "samples" / eb / w.name
        row = corpus.get(w.stem)
        if not wb.exists() or not row:
            continue
        first = ab_order(w.stem)
        x, y = (w, wb) if first else (wb, w)
        try:
            r = gem.ask_json(gem.PRO, AB_PROMPT.format(text=row["text"], style=style_of(row)), parts=[part(x), part(y)],
                             extra="ab" + sha(x) + sha(y))
        except Exception as e:
            print("ab fail", w.name, type(e).__name__)
            continue
        win = {"1": ea if first else eb, "2": eb if first else ea}.get(str(r.get("winner")), "tie")
        ab[w.stem] = {"winner": win, "reason": r.get("reason", ""), "order": [x.parent.name, y.parent.name]}
    out = {"label": "automated proxy, not human MOS", "judge_model": gem.PRO, "per_engine": table, "per_clip": per,
           "pairwise": {"A": ea, "B": eb, "results": ab}, "usage": gem.USAGE}
    (bk / "results").mkdir(exist_ok=True)
    (bk / "results/gemini_judge.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["", "## Gemini judge (automated proxy, not human MOS)", "",
             f"Judge: `{gem.PRO}`, scores 1-5 on the bake-off sample wavs. LLM-judge proxy; a listening test (Phase 19) is still required.", "",
             "| engine | n | " + " | ".join(DIMS) + " |", "|---|---|" + "---|" * len(DIMS)]
    lines += [f"| {e} | {t['n']} | " + " | ".join(str(t[d]) for d in DIMS) + " |"
              for e, t in sorted(table.items(), key=lambda kv: -sum(kv[1][d] for d in DIMS))]
    wins = [v["winner"] for v in ab.values()]
    lines += ["", f"Blind pairwise (randomized order) {ea} vs {eb}: " + ", ".join(f"{k} {wins.count(k)}" for k in (ea, eb, "tie")), ""]
    with open(bk / "REPORT.md", "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(gem.USAGE)


if __name__ == "__main__":
    main()
