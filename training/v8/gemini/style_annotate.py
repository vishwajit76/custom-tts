"""Label transcripts with style/emotion via Gemini Flash (text only; offline annotation).
usage: style_annotate.py [--n 200] [--full]  -> datasets/manifest/style_labels.jsonl"""
import argparse, collections, json, random, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import gem

STYLES = ["neutral", "conversational", "professional", "happy", "sad", "angry", "excited", "empathetic", "serious"]
BATCH = 20
PROMPT = """Label each Hindi TTS training transcript with the speaking style a natural speaker would use.
styles: {styles}. Return a JSON array, one object per input id:
{{"id":int,"style":<one of styles>,"intensity":0..1,"pauses":[char indexes (0-based, in the text) where a short pause fits],"emphasis":[words to stress]}}
Judge from text only; plain commands/facts are neutral. Inputs:
{items}"""


def valid(o, n):
    return (isinstance(o, dict) and o.get("style") in STYLES and isinstance(o.get("intensity"), (int, float))
            and 0 <= o["intensity"] <= 1 and isinstance(o.get("id"), int) and 0 <= o["id"] < n)


def annotate_batch(rows):
    items = "\n".join(json.dumps({"id": i, "text": r["normalized_text"]}, ensure_ascii=False) for i, r in enumerate(rows))
    res = gem.ask_json(gem.FLASH, PROMPT.format(styles=", ".join(STYLES), items=items))
    out = {}
    for o in res if isinstance(res, list) else []:
        if valid(o, len(rows)):
            r = rows[o["id"]]
            out[r["audio_path"]] = {"audio_path": r["audio_path"], "style": o["style"], "intensity": float(o["intensity"]),
                                    "pauses": [p for p in o.get("pauses", []) if isinstance(p, int) and 0 <= p <= len(r["normalized_text"])],
                                    "emphasis": [w for w in o.get("emphasis", []) if isinstance(w, str)]}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(gem.ROOT / "datasets/manifest/all.jsonl", encoding="utf-8")]
    if not a.full:
        random.Random(0).shuffle(rows); rows = rows[:a.n]
    batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    labels = {}
    with ThreadPoolExecutor(4) as ex:
        for d in ex.map(annotate_batch, batches):
            labels.update(d)
    out = gem.ROOT / "datasets/manifest" / ("style_labels.jsonl" if a.full else "style_labels_sample.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for l in labels.values():
            f.write(json.dumps(l, ensure_ascii=False) + "\n")
    print(len(labels), "/", len(rows), dict(collections.Counter(l["style"] for l in labels.values())))
    print("usage", gem.USAGE, "(cached calls cost 0)")


if __name__ == "__main__":
    main()
