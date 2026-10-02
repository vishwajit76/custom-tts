import re
from pathlib import Path

import pytest

from bench import check_eval_overlap as ov
from bench import corpus

ROOT = Path(__file__).resolve().parent.parent
MIN_CAT = {"hindi": 100, "hinglish": 50, "numbers": 30, "pronunciation": 30, "questions": 20, "expressive": 20}


def test_counts_meet_minimums_and_are_documented():
    rows = corpus.load("v2")
    c = corpus.counts(rows)
    assert c["total"] >= 250 and all(c["category"].get(k, 0) >= v for k, v in MIN_CAT.items())
    assert all(v >= 8 for v in c["prosody"].values()), c["prosody"]
    readme = (ROOT / "bench/corpus/README.md").read_text("utf-8")
    assert corpus.VERSIONS["v2"][1] in readme and f"| {c['total']} |" in readme
    for k, v in c["category"].items():
        assert re.search(rf"\| {k} \| {v} \|", readme), k


def test_rows_are_unique_devanagari_and_not_in_v1():
    rows = corpus.load("v2")
    assert len({r["id"] for r in rows}) == len(rows) == len({r["text"] for r in rows})
    assert all(re.search(r"[ऀ-ॿ]|[A-Za-z]", r["text"]) and r["text"] == r["text"].strip() for r in rows)
    v1 = {" ".join(ov.words(s)) for s in ov.load_eval(ROOT / "bench/hi_eval_50.txt")}
    assert not [r["id"] for r in rows if " ".join(ov.words(r["text"])) in v1]
    assert sum(1 for r in rows if "speaker=f" in r["notes"]) >= 10 and sum(1 for r in rows if "speaker=m" in r["notes"]) >= 10


def test_hash_pinned_and_mismatch_fails_loudly(tmp_path, monkeypatch):
    assert corpus.sha256_file(corpus.path_of("v2")) == corpus.VERSIONS["v2"][1]
    bad = tmp_path / "x.tsv"
    bad.write_bytes(corpus.path_of("v2").read_bytes() + b"\n")
    monkeypatch.setitem(corpus.VERSIONS, "v9", (bad.name, corpus.VERSIONS["v2"][1]))
    monkeypatch.setattr(corpus, "HERE", tmp_path)
    with pytest.raises(corpus.CorpusHashMismatch):
        corpus.load("v9")
    with pytest.raises(KeyError):
        corpus.load("v0")


def test_stratified_is_seeded_and_covers_every_category():
    rows = corpus.load("v2")
    q = corpus.stratified(rows, 30, seed=1)
    assert q == corpus.stratified(rows, 30, seed=1) and q != corpus.stratified(rows, 30, seed=2)
    assert 28 <= len(q) <= 34 and {r["category"] for r in q} == set(MIN_CAT)


def test_overlap_check_accepts_tsv_and_flags_v1_and_training(tmp_path, capsys, monkeypatch):
    import sys

    train = tmp_path / "m.csv"
    train.write_text("a.wav|" + corpus.load("v2")[0]["text"] + "\n", "utf-8")
    ok = tmp_path / "ok.csv"
    ok.write_text("a.wav|कुछ बिल्कुल अलग पंक्ति जो कहीं नहीं है\n", "utf-8")
    run = lambda *args: (monkeypatch.setattr(sys, "argv", ["x", *args]), ov.main())[1]
    assert run("--eval", str(corpus.path_of("v2")), "--also-eval", str(ROOT / "bench/hi_eval_50.txt"), "--csv", str(ok)) == 0
    assert run("--eval", str(corpus.path_of("v2")), "--csv", str(train)) == 1
    assert run("--eval", str(ROOT / "bench/hi_eval_50.txt"), "--also-eval", str(ROOT / "bench/hi_eval_50.txt"), "--csv", str(ok)) == 1
