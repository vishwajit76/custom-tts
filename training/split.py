"""Deterministic, leakage-free train/val/test split.

Clips sharing an audio hash or a normalized transcript are grouped and always land in the same split.
--speaker-disjoint-test holds out whole speakers for test and drops any train/val clip whose transcript or
audio hash also occurs in test. The test set is written separately (test.jsonl + test.jsonl.heldout flag, each
row heldout=true); training scripts call assert_not_heldout() and refuse to load it.

  python -m training.split --manifest clean.jsonl --out splits/ [--seed 1234] [--speaker-disjoint-test]
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from training.manifest import read_rows, write_jsonl


class HeldOutError(Exception):
    pass


def _key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _h(seed: int, s: str) -> int:
    return int(hashlib.sha256(f"{seed}|{s}".encode()).hexdigest()[:12], 16)


def _groups(rows: list[dict]) -> list[list[dict]]:
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[str, int] = {}
    for i, r in enumerate(rows):
        for k in (f"t:{_key(r['text'])}", f"a:{r['audio_sha256']}" if r.get("audio_sha256") else None):
            if k is None:
                continue
            if k in seen:
                parent[find(i)] = find(seen[k])
            else:
                seen[k] = i
    out: dict[int, list[dict]] = {}
    for i, r in enumerate(rows):
        out.setdefault(find(i), []).append(r)
    return list(out.values())


def make_splits(rows: list[dict], seed: int = 1234, val_fraction: float = 0.05, test_fraction: float = 0.05,
                speaker_disjoint_test: bool = False) -> tuple[dict[str, list[dict]], list[dict]]:
    """-> ({"train","val","test"}, dropped_rows). Same input + seed gives the same output."""
    rows = sorted(rows, key=lambda r: r["id"])
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    dropped: list[dict] = []
    pool = rows
    if speaker_disjoint_test and test_fraction > 0:
        speakers = sorted({r["speaker_id"] for r in rows}, key=lambda s: _h(seed, s))
        if len(speakers) < 2:
            raise ValueError("speaker-disjoint test split needs at least 2 speakers")
        chosen, n = set(), 0
        for s in speakers[:-1]:  # always leave at least one speaker for training
            if n >= test_fraction * len(rows):
                break
            chosen.add(s)
            n += sum(r["speaker_id"] == s for r in rows)
        splits["test"] = [r for r in rows if r["speaker_id"] in chosen]
        tkeys = {_key(r["text"]) for r in splits["test"]}
        thash = {r["audio_sha256"] for r in splits["test"] if r.get("audio_sha256")}
        pool = []
        for r in rows:
            if r["speaker_id"] in chosen:
                continue
            (dropped if _key(r["text"]) in tkeys or r.get("audio_sha256") in thash else pool).append(r)
    groups = sorted(_groups(pool), key=lambda g: _h(seed, min(r["id"] for r in g)))
    n_test = 0 if speaker_disjoint_test else test_fraction * len(pool)
    n_val = val_fraction * len(pool)
    for g in groups:
        if len(splits["test"]) < n_test and not speaker_disjoint_test:
            splits["test"] += g
        elif len(splits["val"]) < n_val:
            splits["val"] += g
        else:
            splits["train"] += g
    for name in splits:
        splits[name].sort(key=lambda r: r["id"])
    return splits, dropped


def write_splits(splits: dict[str, list[dict]], out: Path, dropped: list[dict] = (), meta: dict | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        if name == "test":
            rows = [{**r, "heldout": True} for r in rows]
            (out / "test.jsonl.heldout").write_text(
                "Held-out test set. Training scripts must refuse to load test.jsonl.\n", encoding="utf-8")
        write_jsonl(out / f"{name}.jsonl", [{**r, "split": name} for r in rows])
    summary = {"counts": {k: len(v) for k, v in splits.items()}, "dropped_for_leakage": len(dropped),
               "speakers": {k: sorted({r["speaker_id"] for r in v}) for k, v in splits.items()}, **(meta or {})}
    (out / "split_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def assert_not_heldout(path: Path) -> None:
    """Call before loading any training data file/dir. Raises HeldOutError for test/held-out data."""
    path = Path(path)
    if path.is_dir():
        if (path / "HELDOUT_ONLY").exists():
            raise HeldOutError(f"{path} is marked HELDOUT_ONLY")
        return
    if re.match(r"test", path.stem, re.I) or Path(str(path) + ".heldout").exists():
        raise HeldOutError(f"{path} is a held-out test set; refusing to load it for training")
    if path.suffix == ".jsonl" and any(r.get("heldout") for r in read_rows(path)):
        raise HeldOutError(f"{path} contains held-out rows; refusing to load it for training")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--test-fraction", type=float, default=0.05)
    p.add_argument("--speaker-disjoint-test", action="store_true")
    a = p.parse_args()
    rows = [r for r in read_rows(a.manifest) if not r.get("rejected")]
    splits, dropped = make_splits(rows, a.seed, a.val_fraction, a.test_fraction, a.speaker_disjoint_test)
    write_splits(splits, a.out, dropped, {"seed": a.seed, "speaker_disjoint_test": a.speaker_disjoint_test})
    print(json.dumps({k: len(v) for k, v in splits.items()} | {"dropped_for_leakage": len(dropped)}))


if __name__ == "__main__":
    main()
