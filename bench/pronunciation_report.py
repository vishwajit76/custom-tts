"""Machine-readable per-category report of the pronunciation regression corpus.

    python bench/pronunciation_report.py                      # JSON to stdout
    python bench/pronunciation_report.py --out bench/results/pronunciation.json

Scores the TEXT normalizer only (no audio). Expected strings are machine-drafted and unreviewed by a native speaker
unless a row says "native-reviewed"; `review_status_counts` makes that visible in every report. Rows whose notes start
with "XFAIL" are counted as known_gaps and reported separately, not as passes. Exit status is 1 if any non-xfail row
fails (so it can gate CI) or if a known gap now passes (update the corpus).
"""
import argparse
import csv
import json
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.services.text_normalizer import normalize  # noqa: E402

CORPUS = ROOT / "tests" / "data" / "pronunciation_corpus.tsv"


def run(corpus: Path = CORPUS) -> dict:
    with corpus.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    cats: dict[str, dict] = defaultdict(lambda: {"total": 0, "pass": 0, "fail": 0, "known_gap": 0, "unexpected_pass": 0})
    failures = []
    for r in rows:
        got = normalize(r["input"])
        ok = got == r["expected_normalized"]
        gap = r["notes"].startswith("XFAIL")
        c = cats[r["category"]]
        c["total"] += 1
        if gap:
            c["unexpected_pass" if ok else "known_gap"] += 1
        else:
            c["pass" if ok else "fail"] += 1
        if (not ok and not gap) or (ok and gap):
            failures.append({"id": r["id"], "category": r["category"], "input": r["input"],
                             "expected": r["expected_normalized"], "got": got, "kind": "unexpected_pass" if ok else "fail"})
    for c in cats.values():
        scored = c["total"] - c["known_gap"] - c["unexpected_pass"]
        c["pass_rate"] = round(c["pass"] / scored, 4) if scored else None
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = ""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": sha, "python": platform.python_version(), "corpus": str(corpus.relative_to(ROOT)),
        "scope": "text normalization only; expected strings machine-drafted unless native-reviewed",
        "review_status_counts": dict(Counter(r["review_status"] for r in rows)),
        "totals": {k: sum(c[k] for c in cats.values()) for k in ("total", "pass", "fail", "known_gap", "unexpected_pass")},
        "categories": dict(sorted(cats.items())),
        "failures": failures,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rep = run(a.corpus)
    text = json.dumps(rep, ensure_ascii=False, indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    sys.exit(1 if rep["failures"] else 0)
