"""CER must see Devanagari vowel signs. Regression for the `\\w` bug: Python's \\w excludes combining marks, so the old cleaner deleted every matra
and virama and CER only compared the consonant skeleton (cer('की', 'कु') == 0.0)."""
from training.asr import _clean, cer


def test_vowel_signs_and_virama_are_kept():
    assert _clean("नमस्ते, आप कैसे हैं?") == "नमस्तेआपकैसेहैं"
    assert cer("की", "कु") == 0.5 and cer("कैसे", "कसे") > 0


def test_punctuation_spaces_nukta_and_chandrabindu_do_not_count():
    assert cer("नमस्ते, जी।", "नमस्ते जी") == 0.0
    assert cer("ज़रूर", "जरूर") == 0.0
    assert cer("हाँ", "हां") == 0.0
    assert abs(cer("abcd", "abxd") - 0.25) < 1e-9


def test_legacy_mode_reproduces_the_consonant_skeleton():
    assert _clean("नमस्ते", keep_marks=False) == "नमसत" and cer("की", "कु", keep_marks=False) == 0.0
