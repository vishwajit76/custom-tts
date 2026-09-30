"""Check that the fixed eval sentences do not overlap the training text.

Usage: python -m bench.check_eval_overlap [--eval bench/hi_eval_50.txt] [--csv data/hi_f/metadata.csv --csv data/hi_f/test.csv] [-n 5]
Reports (a) exact sentence matches and (b) any run of N consecutive words shared with a training/test line. Exit 1 on overlap.
"""
import argparse
import sys
import unicodedata
from pathlib import Path


def words(s: str) -> list[str]:
    # drop punctuation by Unicode category (regex \w would split Devanagari words at vowel signs)
    return "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s.replace("़", "")).split()


def load_eval(path: str) -> list[str]:
    return [l.strip() for l in Path(path).read_text(encoding="utf8").splitlines() if l.strip() and not l.startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="bench/hi_eval_50.txt")
    ap.add_argument("--csv", action="append", default=None, help="file|text lines (default data/hi_f/metadata.csv and test.csv)")
    ap.add_argument("-n", type=int, default=5)
    a = ap.parse_args()
    csvs = a.csv or ["data/hi_f/metadata.csv", "data/hi_f/test.csv"]
    grams: dict[tuple, str] = {}
    exact = set()
    for c in csvs:
        for line in Path(c).read_text(encoding="utf8").splitlines():
            t = line.split("|", 1)[-1]
            w = words(t)
            exact.add(" ".join(w))
            for i in range(len(w) - a.n + 1):
                grams.setdefault(tuple(w[i:i + a.n]), t)
    bad = 0
    ev = load_eval(a.eval)
    for s in ev:
        w = words(s)
        if " ".join(w) in exact:
            print("EXACT:", s); bad += 1; continue
        hit = next((grams[tuple(w[i:i + a.n])] for i in range(len(w) - a.n + 1) if tuple(w[i:i + a.n]) in grams), None)
        if hit:
            print(f"{a.n}-gram overlap:\n  eval : {s}\n  train: {hit}"); bad += 1
    print(f"{len(ev)} eval sentences, {bad} overlapping (checked {', '.join(csvs)})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
