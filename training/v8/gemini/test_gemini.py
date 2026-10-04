"""Offline tests (no network): python -m pytest training/v8/gemini/test_gemini.py"""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import gem, audio_judge as aj, eval_corpus as ec, style_annotate as sa


def test_parse_json_fenced():
    assert gem.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert gem.parse_json('noise [1,2] tail') == [1, 2]


def test_cache_hits_once():
    calls = []

    def fn():
        calls.append(1)
        return '{"x": 2}'
    with tempfile.TemporaryDirectory() as d:
        assert gem.cached("m", "p", "", fn, d) == {"x": 2}
        assert gem.cached("m", "p", "", fn, d) == {"x": 2}
        gem.cached("m", "p2", "", fn, d)
    assert len(calls) == 2


def test_score_and_order():
    ok = {**{d: 4 for d in aj.DIMS}, "issues": ["a"]}
    assert aj.parse_score(ok)["naturalness"] == 4.0
    assert aj.parse_score({**ok, "prosody": 9}) is None
    assert aj.ab_order("hi-001") == aj.ab_order("hi-001")
    assert {aj.ab_order(str(i)) for i in range(40)} == {True, False}


def test_corpus_validation_and_dedupe():
    assert ec.valid_row("hindi", "नमस्ते जी।") and not ec.valid_row("hindi", "hello")
    assert ec.valid_row("hinglish", "App खोलिए")
    base = [("hi-001", "hindi", "s", "नमस्ते जी।", "")]
    new = [("hi", "hindi", "s", "नमस्ते जी", ""), ("hi", "hindi", "s", "ठीक है।", ""), ("hi", "hindi", "s", "ok", "")]
    assert [r[3] for r in ec.merge(base, new)] == ["ठीक है।"]


def test_style_valid():
    assert sa.valid({"id": 0, "style": "sad", "intensity": 0.5}, 1)
    assert not sa.valid({"id": 0, "style": "bored", "intensity": 0.5}, 1)
