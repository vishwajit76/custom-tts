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


# ---------------------------------------------------------------- V7 pack: blinding, key round trip, analysis on a synthetic sheet
import csv
import hashlib
import json
import re

import soundfile as sf

SYSTEMS = ["v6", "v7_baseline", "young_female"]  # names that must never reach the listener pack
ITEMS = [f"hi-{i:03d}" for i in range(1, 7)]


def _audio():
    rng = np.random.default_rng(0)
    return {(s, it): (rng.normal(0, 0.1, 22050).astype("f4"), 22050 if s == "v6" else 24000) for s in SYSTEMS for it in ITEMS}


def _pack(tmp_path, listeners=2):
    plan = lp.plan_v7(SYSTEMS, ITEMS, seed=5)
    key = lp.render_v7(plan, _audio(), {i: "पाठ " + i for i in ITEMS}, tmp_path, listeners, {"v6": {"model_sha256": "x"}})
    return plan, key


def test_v7_plan_is_seeded_and_covers_every_cell():
    a = lp.plan_v7(SYSTEMS, ITEMS, 5)
    assert a == lp.plan_v7(SYSTEMS, ITEMS, 5) and a != lp.plan_v7(SYSTEMS, ITEMS, 6)
    assert len(a["stimuli"]) == len(ITEMS) * (len(SYSTEMS) + 1)  # + anchor
    assert len(a["ab"]) == len(ITEMS) * 3 and all(re.fullmatch(r"[0-9a-f]{8}", k) for k in a["stimuli"])
    first_is_a = [v["A"] == min(v["A"], v["B"]) for v in a["ab"].values()]
    assert 3 <= sum(first_is_a) <= len(first_is_a) - 3  # both sides used


def test_v7_pack_leaks_no_system_name_in_names_audio_sheets_or_manifest(tmp_path):
    _pack(tmp_path)
    listener, org = tmp_path / "listener_pack", tmp_path / "organizer"
    files = [p for p in listener.rglob("*") if p.is_file()]
    assert files
    for p in files:
        blob = p.name.encode() + p.read_bytes()
        assert not any(s.lower().encode() in blob.lower() for s in [*SYSTEMS, lp.ANCHOR]), p
    assert all(re.fullmatch(r"(r_[0-9a-f]{8}|ab_[0-9a-f]{8}_[AB])\.wav|listener_\d\d_(rating|ab)\.csv", p.name) for p in files)
    assert {sf.info(p).samplerate for p in files if p.suffix == ".wav"} == {lp.V7_RATE}  # one rate: it must not reveal the model
    assert not any(s.lower().encode() in (org / "pack_manifest.json").read_bytes().lower() for s in SYSTEMS)
    assert any(s.encode() in (org / "KEY_DO_NOT_SHARE.json").read_bytes() for s in SYSTEMS)  # the sealed key does name them


def test_v7_key_round_trip_and_commitment(tmp_path):
    plan, _ = _pack(tmp_path)
    org = tmp_path / "organizer"
    again = json.loads((org / "KEY_DO_NOT_SHARE.json").read_text())
    assert again["stimuli"] == plan["stimuli"] and again["ab"] == plan["ab"]
    assert json.loads((org / "pack_manifest.json").read_text())["key_sha256"] == hashlib.sha256((org / "KEY_DO_NOT_SHARE.json").read_bytes()).hexdigest()
    orders = []
    for k in (1, 2):  # same stimuli per listener, different order
        rows = list(csv.DictReader(open(tmp_path / "listener_pack/sheets" / f"listener_{k:02d}_rating.csv", encoding="utf-8")))
        assert {r["stimulus_id"] for r in rows} == set(plan["stimuli"]) and all(r["naturalness"] == "" for r in rows)
        orders.append([r["stimulus_id"] for r in rows])
    assert orders[0] != orders[1]


def _fill(plan, better="v7_baseline", seed=0):
    """Synthetic listener: `better` gets ~4.5, other systems ~3, the anchor ~1; A/B prefers `better` when it is in the pair."""
    rng = np.random.default_rng(seed)
    base = lambda s: 4.5 if s == better else 1 if s == lp.ANCHOR else 3
    rate = [{"stimulus_id": sid, **{d: str(int(np.clip(round(base(s["system"]) + rng.normal(0, 0.4)), 1, 5))) for d in lp.DIMS_V7}} for sid, s in plan["stimuli"].items()]
    ab = []
    for iid, s in plan["ab"].items():
        pick = "A" if s["A"] == better else "B" if s["B"] == better else ("A" if rng.random() < 0.5 else "B")
        ab.append({"item_id": iid, **{d: pick for d in [*lp.DIMS_V7, "overall_preference"]}})
    return rate, ab


def test_v7_analysis_on_a_synthetic_sheet(tmp_path):
    plan, key = _pack(tmp_path)
    r1, a1 = _fill(plan, seed=1)
    r2, a2 = _fill(plan, seed=2)
    res = lp.analyze_v7(key, {"l1": r1, "l2": r2}, {"l1": a1, "l2": a2}, b=300, seed=1)
    m = res["ratings"]["naturalness"]
    assert m["v7_baseline"]["mean"] > m["v6"]["mean"] > m[lp.ANCHOR]["mean"] and m["v7_baseline"]["ci95"][0] <= m["v7_baseline"]["mean"] <= m["v7_baseline"]["ci95"][1]
    d = next(x for x in res["rating_pair_diffs"] if x["dimension"] == "naturalness" and x["baseline"] == "v6" and x["candidate"] == "v7_baseline")
    assert d["delta_mean"] > 1 and d["excludes_zero"]
    ab = next(x for x in res["ab"] if x["pair"] == ["v6", "v7_baseline"] and x["dimension"] == "overall_preference")
    assert ab["v7_baseline"] == 2 * len(ITEMS) and ab["v6"] == 0 and ab["sign_test_p_two_sided"] == pytest.approx(lp.sign_test_p(0, 12), abs=1e-4) and ab["sign_test_p_two_sided"] < 0.01
    assert set(ab["per_listener"]) == {"l1", "l2"} and not any(c["flag_anchor_not_below_all_systems"] for c in res["listener_checks"].values())
    json.dumps(res)


def test_v7_analysis_flags_inattentive_listener_and_rejects_bad_cells(tmp_path):
    plan, key = _pack(tmp_path)
    rate, _ = _fill(plan)
    for r in rate:  # rates everything 5, anchor included
        r.update({d: "5" for d in lp.DIMS_V7})
    assert lp.analyze_v7(key, {"l": rate}, {})["listener_checks"]["l"]["flag_anchor_not_below_all_systems"]
    with pytest.raises(ValueError):
        lp.analyze_v7(key, {}, {"l": [{"item_id": next(iter(plan["ab"])), "naturalness": "C"}]})
    p = tmp_path / "bad.csv"
    p.write_text("stimulus_id,naturalness\nx,7\n")
    with pytest.raises(ValueError):
        lp._read_scores(p, ["naturalness"], {"1", "2", "3", "4", "5"})
    blank = [{"stimulus_id": next(iter(plan["stimuli"])), **{d: "" for d in lp.DIMS_V7}}]
    assert lp.analyze_v7(key, {"l": blank}, {})["ratings"]["naturalness"] == {}
