import re
from pathlib import Path

import pytest

from bench import check_eval_overlap as ov
from bench import milestone_eval as me

ROOT = Path(__file__).resolve().parent.parent


def test_eval_set_is_50_unique_devanagari_sentences():
    s = me.load_sentences(ROOT / "bench" / "hi_eval_50.txt")
    assert len(s) == 50 and len(set(s)) == 50
    assert all(re.search(r"[ऀ-ॿ]", x) for x in s)


def test_overlap_words_keep_devanagari_words_whole():
    assert ov.words("बहुत, बहुत शुक्रिया।") == ["बहुत", "बहुत", "शुक्रिया"]
    assert ov.words("ज़रूर") == ov.words("जरूर")


@pytest.mark.skipif(not (ROOT / "data/hi_f/metadata.csv").exists(), reason="training text not present")
def test_eval_set_disjoint_from_training_text():
    import sys
    argv, sys.argv = sys.argv, ["x"]
    try:
        assert ov.main() == 0
    finally:
        sys.argv = argv


def test_kernel_wrapper_source_compiles_and_has_lr_control():
    src = (ROOT / "training/kaggle/train_kernel.py").read_text(encoding="utf8")
    wrap = re.search(r"WRAP_SRC = r'''(.*?)'''\n", src, re.S).group(1)
    compile(wrap, "wrap.py", "exec")
    compile(src, "train_kernel.py", "exec")
    assert 'state_key = "lr_ctl"' in wrap and "LR_MODE" in wrap
    assert "milestones/step_{n}" in src and "experiments/{self.id}/" in src


def test_percentiles_and_stamps():
    assert me.pct([0.0, 1.0], 50) == 0.5
    st = me.now_stamps()
    assert st["timestamp_utc"].endswith("Z") and st["timestamp_ist"].endswith("+05:30")
