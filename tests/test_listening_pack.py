import numpy as np
import pytest

from bench import listening_pack as lp

SENTS = [f"s{i}" for i in range(50)]


def test_plan_is_seeded_unique_and_swaps_repeat_sides():
    a = lp.plan_items(SENTS, 20, 7, 3)
    assert a == lp.plan_items(SENTS, 20, 7, 3) and a != lp.plan_items(SENTS, 20, 8, 3)
    assert len(a) == 23 and len({x["sample_id"] for x in a}) == 23
    by = {x["sample_id"]: x for x in a}
    reps = [x for x in a if x["repeat_of"]]
    assert len(reps) == 3
    for r in reps:
        o = by[r["repeat_of"]]
        assert o["sentence_index"] == r["sentence_index"] and o["a_is"] != r["a_is"]
    firsts = [x for x in a if not x["repeat_of"]]
    assert len({x["sentence_index"] for x in firsts}) == 20
    assert 4 <= sum(x["a_is"] == "first" for x in firsts) <= 16  # both sides used


def test_sign_test_exact_values():
    assert lp.sign_test_p(5, 10) == 1.0
    assert lp.sign_test_p(10, 10) == pytest.approx(2 / 1024)
    assert lp.sign_test_p(9, 10) == pytest.approx(22 / 1024)
    assert lp.sign_test_p(0, 0) == 1.0
    assert lp.min_wins_for_significance(20) == 15 and lp.min_wins_for_significance(4) is None


def test_rms_match_equalizes_level_and_never_clips():
    rng = np.random.default_rng(0)
    quiet, loud = rng.normal(0, 0.01, 8000).astype("f4"), rng.normal(0, 0.3, 8000).astype("f4")
    for w in (quiet, loud):
        o = lp.rms_match(w)
        assert abs(float(np.sqrt(np.mean(o ** 2))) - lp.TARGET_RMS) < 5e-3 and float(np.abs(o).max()) <= 0.98 + 1e-6


def _key():
    return [{"sample_id": "item01", "checkpoint_A": "old", "checkpoint_B": "new", "sentence": "x", "repeat_of": ""},
            {"sample_id": "item02", "checkpoint_A": "new", "checkpoint_B": "old", "sentence": "y", "repeat_of": ""},
            {"sample_id": "item03", "checkpoint_A": "new", "checkpoint_B": "old", "sentence": "x", "repeat_of": "item01"}]


def _sheet(picks):
    return [{"sample_id": s, **{d: v for d in lp.DIMS}} for s, v in picks.items()]


def test_merge_unblinds_sides_and_tallies():
    # synthetic ratings for the test only: the listener always prefers "new" (side B in item01, side A in item02 and item03)
    res = lp.merge(_key(), [_sheet({"item01": "B", "item02": "A", "item03": "A"})])
    s = res["summary"]["overall_preference"]
    assert s["new"] == 3 and s["old"] == 0 and s["n_non_tie"] == 3 and s["sign_test_p_two_sided"] == 0.25
    assert res["consistency"] == {"pairs": 1, "agree": 1}  # item03 repeats item01 with swapped sides and got the same model


def test_merge_detects_inconsistent_listener_and_ties_and_bad_values():
    res = lp.merge(_key(), [_sheet({"item01": "B", "item02": "tie", "item03": "B"})])  # item03 B = old, item01 B = new -> disagree
    assert res["consistency"] == {"pairs": 1, "agree": 0} and res["summary"]["overall_preference"]["tie"] == 1
    with pytest.raises(ValueError):
        lp.merge(_key(), [_sheet({"item01": "C", "item02": "A", "item03": "A"})])
    blank = lp.merge(_key(), [[{"sample_id": "item01", **{d: "" for d in lp.DIMS}}]])
    assert blank["summary"] == {}  # blank = unrated, never counted
