import copy
import json

import pytest

from training import experiments as ex


def test_committed_records_are_valid_and_named_by_id():
    res = ex.validate_all()
    assert len(res) >= 3 and not {k: v for k, v in res.items() if v}


def _rec():
    p = next(ex.DIR.glob("hi_f-v4-*.json"))
    return json.loads(p.read_text("utf-8"))


def test_write_record_is_create_once(tmp_path):
    r = _rec(); r["experiment_id"] = "hi_f-unit-1"
    ex.write_record(r, tmp_path)
    with pytest.raises(FileExistsError):
        ex.write_record(r, tmp_path)


def test_validation_catches_bad_records():
    r = _rec()
    bad = copy.deepcopy(r); bad["seed"] = 99
    assert any("seed" in e for e in ex.validate(bad))
    bad = copy.deepcopy(r); del bad["dataset"]
    assert any("dataset" in e for e in ex.validate(bad))
    bad = copy.deepcopy(r); bad["dataset"] = {"n_wavs": 1}
    assert any("metadata_sha256" in e for e in ex.validate(bad))
    bad = copy.deepcopy(r); bad["experiment_id"] = "x y"
    assert ex.validate(bad)
    assert ex.validate(r, name="other")


def test_v4_record_states_what_was_verified():
    r = _rec()
    assert r["outputs"]["final_global_step_verified"] == 357212 and "git_sha" in r["unknown"]
    assert r["config"]["lr"]["lr_g"] == pytest.approx(1.5258e-4, rel=1e-4)
