"""Versioned evaluation corpora. `load("v2")` verifies the sha256 of the TSV and fails loudly on any mismatch.

A published version is never edited: add v3 (new file, new entry in VERSIONS, new README row) instead. See README.md.
"""
import csv
import hashlib
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
# version -> (file, sha256 of the file bytes). Keep in sync with README.md (tests/test_corpus.py checks both).
VERSIONS = {"v2": ("hi_eval_v2.tsv", "814850bf4e71617f28355c4898ea54c6832246b68de9164944e77363bfb2f601")}
COLUMNS = ["id", "category", "subcategory", "text", "notes"]
PROSODY = ["statement", "yes_no_question", "wh_question", "exclamation", "confirmation", "uncertainty", "apology", "request", "instruction"]


class CorpusHashMismatch(RuntimeError):
    pass


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def path_of(version: str) -> Path:
    if version not in VERSIONS:
        raise KeyError(f"unknown corpus version {version!r}; have {sorted(VERSIONS)}")
    return HERE / VERSIONS[version][0]


def load(version: str = "v2", verify: bool = True) -> list[dict]:
    p = path_of(version)
    got = sha256_file(p)
    if verify and got != VERSIONS[version][1]:
        raise CorpusHashMismatch(f"corpus {version} ({p.name}) sha256 {got} != pinned {VERSIONS[version][1]}: a published corpus must never be edited; add a new version")
    with p.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    if rows and list(rows[0]) != COLUMNS:
        raise ValueError(f"{p.name}: columns {list(rows[0])} != {COLUMNS}")
    return rows


def notes_of(row: dict) -> dict:
    """notes column 'k=v; k=v' -> dict (free text without '=' is ignored)."""
    return dict(kv.strip().split("=", 1) for kv in row["notes"].split(";") if "=" in kv)


def counts(rows: list[dict]) -> dict:
    cat = Counter(r["category"] for r in rows)
    pro = Counter(r["subcategory"] for r in rows if r["subcategory"] in PROSODY)
    return {"total": len(rows), "category": dict(sorted(cat.items())), "prosody": {k: pro.get(k, 0) for k in PROSODY},
            "subcategory": dict(sorted(Counter(f"{r['category']}/{r['subcategory']}" for r in rows).items()))}


def stratified(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """~n rows: per-category quota proportional to its size (min 1), filled round-robin across that category's subcategories
    (seeded shuffle inside each), returned in corpus order. Deterministic for (corpus, n, seed)."""
    import random

    rng = random.Random(seed)
    cats: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        cats.setdefault(r["category"], {}).setdefault(r["subcategory"], []).append(r)
    total = len(rows)
    quota = {c: max(1, round(n * sum(map(len, g.values())) / total)) for c, g in cats.items()}
    ids = set()
    for c, g in sorted(cats.items()):
        pools = [rng.sample(v, len(v)) for _, v in sorted(g.items())]
        order = rng.sample(range(len(pools)), len(pools))
        while len(ids & {r["id"] for v in g.values() for r in v}) < min(quota[c], sum(map(len, pools))):
            for k in order:
                if pools[k] and len(ids & {r["id"] for v in g.values() for r in v}) < quota[c]:
                    ids.add(pools[k].pop()["id"])
    return [r for r in rows if r["id"] in ids]
