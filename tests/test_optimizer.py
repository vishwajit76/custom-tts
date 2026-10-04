"""Optimizer unit tests: deterministic, no GPU, no network, no git."""
import numpy as np
import pytest

from optimizer import bench, core, evaluators as ev
from optimizer import loop as loopmod


def test_hindi_number_words_independent_of_app():
    assert bench.words(125000) == "एक लाख पच्चीस हज़ार"
    assert bench.words(999999) == "नौ लाख निन्यानवे हज़ार नौ सौ निन्यानवे"
    assert bench.indian(125000) == "1,25,000" and bench.indian(1000000) == "10,00,000"
    assert bench.clock(1, 30) == "डेढ़ बजे" and bench.clock(10, 45) == "पौने ग्यारह बजे" and bench.year(1998) == "उन्नीस सौ अट्ठानवे"


def test_text_evaluator_canonical_and_alternatives():
    case = {"text": "x", "expected_normalized": "पाँच हज़ार", "alternatives": ["पांच हजार रुपये"]}
    assert ev.TextEvaluator().evaluate({"normalized": "पांच हजार!"}, case)["text_exact"] == 1.0
    assert ev.TextEvaluator().evaluate({"normalized": "पाँच सौ"}, case)["text_score"] < 1


def test_asr_detects_word_split():
    m = ev.asr_metrics("मेरा नाम विश्व जीत है", {"text": "मेरा नाम विश्वजीत है।", "expected_normalized": "मेरा नाम विश्वजीत है।", "important": ["विश्वजीत"]})
    assert m["splits"] == ["विश्वजीत"] and m["important_acc"] == 0 and m["cer"] == 0


def test_audio_metrics_flags_clipping_and_silence():
    sr = 16000
    tone = 0.3 * np.sin(np.arange(sr) / sr * 2 * np.pi * 220).astype(np.float32)
    clean = ev.audio_metrics(np.concatenate([tone, np.zeros(sr // 4, np.float32), tone]), sr, 20)
    assert clean["issues"] == [] and clean["pauses"] == 1
    bad = ev.audio_metrics(np.concatenate([np.clip(tone * 10, -1, 1), np.zeros(2 * sr, np.float32), tone]), sr)
    assert {"clipping", "abnormal_silence"} <= set(bad["issues"]) and bad["audio_quality"] < clean["audio_quality"]


def test_total_and_gates():
    w = {"pronunciation": 0.4, "naturalness": 0.25, "prosody": 0.15, "audio_quality": 0.1, "consistency": 0.1}
    assert core.total(dict.fromkeys(w, 1.0), w) == 1.0
    g = {"min_target_improvement": 0.05, "critical_regression": 0.05, "max_category_regression": 0.01, "max_rtf_degradation": 0.1}
    cases = {"a": {"category": "x", "priority": 1}, "b": {"category": "x", "priority": 2}, "c": {"category": "y", "priority": 2}}
    r = lambda t, p=0.9: {"total": t, "pronunciation": p}
    old = {"a": r(0.5), "b": r(0.9), "c": r(0.9)}
    assert core.gates(0.5, 0.7, old, {"a": r(0.7)}, cases, g)[0]
    assert not core.gates(0.5, 0.52, old, {"a": r(0.52)}, cases, g)[0]                          # too small
    ok, why, regr = core.gates(0.5, 0.7, old, {"a": r(0.7), "c": r(0.7)}, cases, g)
    assert not ok and any("category y" in x for x in why) and regr == pytest.approx(0.2)        # category regression
    assert not core.gates(0.5, 0.7, old, {"a": r(0.7, 0.8)}, cases, g)[0]                       # critical pron drop
    assert not core.gates(0.5, 0.7, old, {"a": r(0.7)}, cases, g, 0.1, 0.2)[0]                  # rtf


def test_db_cache_state_results(tmp_path):
    db = core.DB(tmp_path / "x.db")
    assert db.get("k") is None
    db.put("k", {"hyp": "नमस्ते"})
    assert core.DB(tmp_path / "x.db").get("k") == {"hyp": "नमस्ते"}
    db.set_state("next_iteration", 7)
    assert core.DB(tmp_path / "x.db").state("next_iteration") == 7
    db.save_results("current:e", "e", [{"id": "a", "category": "c", "priority": 1, "total": 0.5}])
    assert db.results("current:e", "e")["a"]["total"] == 0.5
    db.log(iteration=1, test_id="a", change={"type": "rule", "name": "iso_date"}, decision="rejected")
    assert db.q("SELECT decision FROM experiments") == [("rejected",)]


def test_cache_key_and_frontend_dictionary():
    a = core.DEFAULT_ACTIVE
    n1, ch1, p1 = core.frontend("Zorblax पर भेजिए।", "x", a, {})
    n2, ch2, p2 = core.frontend("Zorblax पर भेजिए।", "x", a, {"Zorblax": {"spoken": "व्हाट्सऐप"}})
    assert "व्हाट्सऐप" in n2 and core.key_of("m", ch1, p1) != core.key_of("m", ch2, p2)
    assert core.key_of("m", ch1, p1) == core.key_of("m", list(ch1), dict(p1)) != core.key_of("m2", ch1, p1)
    assert core.apply_rules("तारीख 2026-10-04 है", ["iso_date"]) == "तारीख 4/10/2026 है"


def test_candidates_bounded_and_ordered():
    case = {"id": "t", "category": "acronyms", "text": "कृपया OTP बताइए, और नाम भी।", "terms": {"OTP": "ओ टी पी"}, "important": []}
    res = {"failure_type": "normalization", "metrics": {"splits": []}}
    cs = core.candidates(case, res, core.DEFAULT_ACTIVE, {}, {"pause_ms": [60, 120, 250], "speed": [0.9, 1.0, 1.1]}, 5)
    assert len(cs) == 5 and cs[0] == {"type": "dict", "source": "OTP", "spoken": "ओ टी पी"}
    assert len({str(c) for c in cs}) == len(cs)
    a, p = core.apply_change(core.DEFAULT_ACTIVE, {}, {"type": "setting", "category": "acronyms", "key": "pause_ms", "value": 60})
    assert core.settings_for(a, "acronyms")["pause_ms"] == 60 and core.DEFAULT_ACTIVE["category"] == {}


# ---------- loop with fake engine/ASR: resume + accept path ----------
class FakeEngine:
    version = "fake"

    def synth(self, chunks, speed=1.0, pause_ms=120):
        n = sum(len(c) for c in chunks)
        t = np.arange(int(16000 * max(0.2, n / 14)), dtype=np.float32)
        return (0.3 * np.sin(t / 16000 * 2 * np.pi * 200)).astype(np.float32), 16000


class FakeASR:  # "hears" exactly the normalized text that was synthesized, via a side table
    def __init__(self):
        self.next = []

    def transcribe(self, wavs):
        return [self.next.pop(0) if self.next else "" for _ in wavs]


@pytest.fixture
def session(tmp_path, monkeypatch):
    for k, v in {"STATE": tmp_path / "state", "REPORTS": tmp_path / "rep", "PRON": tmp_path / "pron.json", "ACTIVE": tmp_path / "act.json"}.items():
        monkeypatch.setattr(core, k, v)
    monkeypatch.setattr(core, "git_commit", lambda *a, **k: "deadbeef")
    cases = [{"id": "c1", "category": "acronyms", "priority": 1, "text": "कृपया OTP बताइए।", "expected_normalized": "कृपया ओ टी पी बताइए।",
              "terms": {"OTP": "ओटीपी"}, "important": ["ओटीपी"]},
             {"id": "c2", "category": "basic_hindi", "priority": 2, "text": "नमस्ते।", "expected_normalized": "नमस्ते।"}]
    monkeypatch.setattr(bench, "load", lambda *a, **k: cases)
    asr = FakeASR()
    orig = core.Evaluator.run

    def run(self, cs, engine, active, pron):  # fake ASR echoes the frontend text = perfect recognition of what was said
        asr.next = [core.frontend(c["text"], c["category"], active, pron)[0] for c in cs]
        return orig(self, cs, engine, active, pron)
    monkeypatch.setattr(core.Evaluator, "run", run)
    cfg = core.config()
    cfg["engines"], cfg["primary_engine"] = ["fake"], "fake"
    db = core.DB(tmp_path / "state" / "o.db")
    return lambda: loopmod.Session(cfg, db, {"fake": FakeEngine()}, asr)


def test_loop_accepts_dictionary_fix_and_resumes(session, monkeypatch):
    s = session()
    s.benchmark()
    calls = []
    orig = loopmod.Session.iterate

    def crash(self, it, *a):
        calls.append(it)
        raise KeyboardInterrupt
    monkeypatch.setattr(loopmod.Session, "iterate", crash)
    with pytest.raises(KeyboardInterrupt):
        s.run(3)
    assert s.db.state("next_iteration", 1) == 1 and s.db.state("in_progress") == 1
    monkeypatch.setattr(loopmod.Session, "iterate", orig)
    s2 = session()
    rs = s2.run(1, resume=True)
    assert rs[0]["iteration"] == 1 and s2.db.state("next_iteration") == 2
    assert rs[0]["decision"] == "accepted" and "OTP" in s2.pron and s2.pron["OTP"]["iteration"] == 1
    assert rs[0]["after"] > rs[0]["before"]
