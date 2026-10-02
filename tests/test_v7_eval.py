"""bench/v7_eval.py with a fake engine and a fake ASR/UTMOS/encoder: no models, no network."""
import json

import numpy as np
import pytest

from bench import corpus
from bench import v7_eval as ve

SR = 22050


class FakeEngine:
    """Deterministic: a tone whose pitch and length depend on the chunk text. Same call signature as PiperEngine."""
    max_workers = 1

    def sample_rate(self, voice):
        return SR

    def has_voice(self, voice):
        return True

    def synth(self, text, voice, speed, ref=None, ref_text=None):
        n = int(SR * (0.4 + 0.02 * len(text)))
        f = 150 + (sum(map(ord, text)) % 100) + (0 if voice == "a" else 40)
        t = np.arange(n) / SR
        return (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)


class FakeEncoder:
    name = "fake"

    def embed(self, wav, sr):
        v = np.array([np.mean(np.abs(wav)), np.std(wav), len(wav) / sr, 1.0])
        return v / np.linalg.norm(v)


def fake_asr(wav, sr):  # a deterministic function of the audio only; its CER is meaningless, its reproducibility is not
    return "आपका" + " ऑर्डर" * (len(wav) % 3)


def systems():
    return [ve.System("a", FakeEngine(), "a", None, {"noise_scale": 0.667}), ve.System("b", FakeEngine(), "b", None, {"noise_scale": 0.667})]


def run(rows, **kw):
    return ve.run(systems(), rows, corpus_version="v2", asr=fake_asr, asr_name="fake-asr", utmos=lambda w, sr: 3.0 + float(np.mean(np.abs(w))),
                  utmos_note="fake utmos", encoder=FakeEncoder(), encoder_name="fake", latency_repeats=2, seed=3, bootstrap=100, log=lambda *_: None, **kw)


@pytest.fixture(scope="module")
def rows():
    return corpus.stratified(corpus.load("v2"), 30, seed=3)


def test_same_seed_same_fake_engine_gives_identical_metrics_except_timing(rows):
    a, b = ve.strip_timing(run(rows)), ve.strip_timing(run(rows))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["corpus"]["sha256"] == corpus.VERSIONS["v2"][1] and a["asr"]["model"] == "fake-asr" and a["seed"] == 3


def test_report_has_required_fields_and_honest_labels(rows):
    rep = run(rows)
    s = rep["systems"]["a"]
    o = s["aggregate"]["overall"]
    for k in ("cer", "per", "cer_8k", "cer_16k", "utmos", "spk_self"):
        assert o[k]["n"] == len(rows) and o[k]["ci95"][0] <= o[k]["mean"] <= o[k]["ci95"][1], k
    assert set(s["aggregate"]["per_category"]) == {r["category"] for r in rows}
    assert s["timing"]["ttfa_ms"]["n_streams"] == 2 * len(rows) and s["timing"]["ttfa_ms"]["p95"] >= s["timing"]["ttfa_ms"]["p50"] > 0
    assert s["rss"]["process_peak_mb"] > 0 and s["aggregate"]["audio"]["total_s"] > 0 and s["aggregate"]["audio"]["seconds_per_char"] > 0
    assert "PREDICTED MOS" in rep["utmos"]["label"] and "NOT a human MOS" in rep["utmos"]["label"]
    assert rep["environment"]["cpu_count"] and "git_commit" in rep["environment"] and rep["telephony"]["variants_hz"] == [8000, 16000]
    json.dumps(rep, ensure_ascii=False)


def test_paired_delta_between_two_systems_and_same_system_is_zero(rows):
    rep = run(rows)
    d = rep["paired_vs_first"][0]
    assert d["baseline"] == "a" and d["candidate"] == "b" and "cer" in d and "excludes_zero" in d["cer"]
    same = ve.paired({"x": rep["systems"]["a"]["rows"], "y": rep["systems"]["a"]["rows"]}, 100, 0)[0]
    assert same["cer"]["delta_mean"] == 0 and not same["cer"]["excludes_zero"]


def test_skips_are_reported_not_invented(rows):
    rep = ve.run(systems()[:1], rows[:6], corpus_version="v2", asr=None, utmos=None, utmos_note="skipped: offline", encoder=None, latency_repeats=1, bootstrap=50, log=lambda *_: None)
    o = rep["systems"]["a"]["aggregate"]["overall"]
    assert "cer" not in o and "utmos" not in o and rep["utmos"]["skipped"] == "skipped: offline" and rep["asr"]["skipped"] and rep["speaker_similarity"]["skipped"]


def test_aggregate_per_category_math():
    rows_ = [{"id": str(i), "category": "x" if i < 2 else "y", "ref": "ab", "dur_s": 1.0, "clip_fraction": 0.0, "silent": False, "cer": c}
             for i, c in enumerate([0.0, 0.2, 0.4, 0.6])]
    agg = ve.aggregate(rows_, {"sr": SR, "ttfa_ms": [10.0, 20.0], "rtf": [0.1, 0.2], "synth_wall_s": 1, "threads": {}}, 50, 0)
    assert agg["overall"]["cer"]["mean"] == pytest.approx(0.3) and agg["per_category"]["x"]["cer"]["mean"] == pytest.approx(0.1)
    assert agg["per_category"]["y"]["cer"]["mean"] == pytest.approx(0.5) and agg["audio"]["seconds_per_char"] == 0.5


def test_server_resample_matches_stream_path_rates():
    t = np.arange(SR) / SR
    x = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    assert len(ve.server_resample(x, SR, 8000)) in range(7990, 8011) and len(ve.server_resample(x, SR, 16000)) in range(15990, 16011)
    assert len(ve.server_resample(x, SR, SR)) == SR


def test_seeded_noise_graph_patch_sets_seed_only_on_random_nodes():
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    g = helper.make_graph([helper.make_node("RandomNormalLike", ["x"], ["y"], dtype=1), helper.make_node("Relu", ["y"], ["z"])], "g",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])], [helper.make_tensor_value_info("z", TensorProto.FLOAT, [2])])
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.onnx"
        onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)], ir_version=8), str(p))
        sess = ve.seeded_make_session(5)(p, 1, False)
        a = sess.run(None, {"x": np.zeros(2, np.float32)})[0]
        b = ve.seeded_make_session(5)(p, 1, False).run(None, {"x": np.zeros(2, np.float32)})[0]
        c = ve.seeded_make_session(6)(p, 1, False).run(None, {"x": np.zeros(2, np.float32)})[0]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_reference_uses_the_voices_pronunciation_rules(monkeypatch, rows):
    from app.services import voice_catalog

    monkeypatch.setattr(voice_catalog, "rules_of", lambda v: "all" if v == "b" else None)
    s = systems()
    assert s[0].info()["pronunciation_rules"] is None and s[1].info()["pronunciation_rules"] == "all"
    assert run(rows[:6])["systems"]["b"]["system"]["pronunciation_rules"] == "all"
