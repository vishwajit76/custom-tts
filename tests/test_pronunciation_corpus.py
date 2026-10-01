"""Pronunciation regression corpus (Phase 6).

tests/data/pronunciation_corpus.tsv is MACHINE-DRAFTED: expected strings were written from Hindi conventions by the
model, not reviewed by a native speaker. review_status says so on every row. A row whose notes start with "XFAIL" is a
known gap or a genuinely ambiguous case: it is strict-xfail (an unexpected pass fails the run, so the marker gets
removed) and its expectation is never weakened to match the normalizer.
"""
import csv
import re
from pathlib import Path

import pytest

from app.services import hinglish
from app.services.text_normalizer import normalize

CORPUS = Path(__file__).parent / "data" / "pronunciation_corpus.tsv"
REQUIRED = {
    "devanagari", "roman_hindi", "code_switch", "names_places", "loanwords", "numbers", "lakh_crore", "rupees",
    "dates", "times", "decimals", "percent", "otp", "phone", "abbreviations", "technical",
}
LATIN, DEV = re.compile(r"[A-Za-z]"), re.compile(r"[ऀ-ॿ]")


def load() -> list[dict]:
    with CORPUS.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


ROWS = load()


def _params():
    out = []
    for r in ROWS:
        marks = [pytest.mark.xfail(strict=True, reason=r["notes"])] if r["notes"].startswith("XFAIL") else []
        out.append(pytest.param(r, id=r["id"], marks=marks))
    return out


def test_corpus_shape():
    assert len(ROWS) >= 200
    assert len({r["id"] for r in ROWS}) == len(ROWS)
    assert REQUIRED <= {r["category"] for r in ROWS}
    assert list(ROWS[0]) == ["id", "category", "input", "expected_normalized", "review_status", "notes"]
    # honest provenance: nobody has reviewed these yet; flip a row only when a native reviewer signs it off
    assert {r["review_status"] for r in ROWS} <= {"machine-drafted, needs native review", "native-reviewed"}
    assert all(r["input"] and r["expected_normalized"] for r in ROWS)


@pytest.mark.parametrize("row", _params())
def test_corpus_row(row):
    assert normalize(row["input"]) == row["expected_normalized"]


def test_normalize_is_idempotent_on_corpus_expectations():
    # feeding already-normalized text back in must not change it (no double reading), except rows that still hold digits
    for r in ROWS:
        exp = r["expected_normalized"]
        if r["notes"].startswith("XFAIL") or re.search(r"\d", exp):
            continue
        assert normalize(exp) == exp, (r["id"], exp, normalize(exp))


def test_no_token_mixes_scripts():
    """A whitespace-delimited token must not glue Latin and Devanagari letters ("बेंगलुरुपये", "SBINशून्य")."""
    for r in ROWS:
        if r["notes"].startswith("XFAIL"):
            continue
        for tok in normalize(r["input"]).split():
            assert not (LATIN.search(tok) and DEV.search(tok)), (r["id"], tok)


# --- English must stay English -------------------------------------------------------------------------------------
# Words whose spelling looks Hindi-shaped to the romanization heuristic (ee, aa, kh, bh, dh...) but are English.
ENGLISH_LOOKALIKES = [
    "meeting", "feedback", "speed", "free", "green", "screen", "agreement", "payment", "statement", "processing",
    "banking", "reach", "team", "deal", "video", "keep", "week",
]


@pytest.mark.parametrize("word", ENGLISH_LOOKALIKES)
def test_english_words_not_transliterated(word):
    for sentence in (f"kal aapka {word} hai, theek hai?", f"aap {word} dekh lijiye", f"आपका {word} ready है"):
        out = normalize(sentence)
        assert word in out.split() or word in re.findall(r"[A-Za-z]+", out), (sentence, out)


def test_hindi_words_still_transliterated_when_english_present():
    assert normalize("aapka meeting kal hai") == "आपका meeting कल है"


def test_ambiguous_hindi_english_words_stay_english_in_english_sentence():
    assert normalize("Please call me tomorrow.") == "Please call me tomorrow."
    assert normalize("Do you like it?") == "Do you like it?"


# --- script / phoneme boundaries -----------------------------------------------------------------------------------
def test_devanagari_only_input_passes_through_untouched():
    text = "आपका खाता सुरक्षित है, चिंता न करें।"
    assert hinglish.convert(text) == text and normalize(text) == text


def test_latin_acronym_boundary_next_to_devanagari():
    assert normalize("आपकी EMI due है") == "आपकी ई एम आई due है"


def test_phoneme_blocks_only_wrap_english_runs():
    ie = pytest.importorskip("app.services.indian_english")
    try:
        out = ie.mark("आपका हुआ loan approve।")
    except Exception as e:  # espeak-ng data unavailable in this environment
        pytest.skip(f"espeak-ng unavailable: {e}")
    blocks = re.findall(r"\[\[(.*?)\]\]", out)
    assert len(blocks) == 1  # "loan approve" is one contiguous English run
    assert not DEV.search(blocks[0]) and blocks[0].strip()
    outside = re.sub(r"\[\[.*?\]\]", "", out)
    assert not LATIN.search(outside)
    assert outside.startswith("आपका हुआ ")
    assert out.endswith("]].")  # danda right after a block becomes "."


def test_phoneme_marking_leaves_pure_devanagari_alone():
    ie = pytest.importorskip("app.services.indian_english")
    assert ie.mark("नमस्ते, आप कैसे हैं?") == "नमस्ते, आप कैसे हैं?"


# --- opt-in persona gender agreement --------------------------------------------------------------------------------
def test_persona_gender_female():
    f = hinglish.apply_persona_gender
    assert f("मैं आपकी मदद कर सकता हूँ", "female") == "मैं आपकी मदद कर सकती हूँ"
    assert f("मैं कल फ़ोन करूँगा", "f") == "मैं कल फ़ोन करूँगी"
    assert f("मैं गया था", "female") == "मैं गई थी"
    assert f("मैं समझता हूँ, मैं बताता हूँ", "female") == "मैं समझती हूँ, मैं बताती हूँ"


def test_persona_gender_male():
    f = hinglish.apply_persona_gender
    assert f("मैं आपकी मदद कर सकती हूँ", "male") == "मैं आपकी मदद कर सकता हूँ"
    assert f("मैं कल फ़ोन करूँगी", "m") == "मैं कल फ़ोन करूँगा"
    assert f("मैं गई थी", "male") == "मैं गया था"
    assert f("आपकी सहायक हूँ", "male") == "आपका सहायक हूँ"


def test_persona_gender_does_not_touch_third_person_or_unknown_gender():
    f = hinglish.apply_persona_gender
    third = "वह बोल रहा है और वो कर रहा है"
    assert f(third, "female") == third
    text = "मैं कर सकता हूँ"
    for g in (None, "", "neutral", "other"):
        assert f(text, g) == text  # no silent rewrite for unspecified gender


def test_persona_gender_is_never_applied_automatically():
    text = "मैं आपकी मदद कर सकता हूँ, मैं कल फ़ोन करूँगा"
    assert normalize(text) == text
    assert hinglish.convert(text) == text
    assert normalize("main aapko call karunga") == "मैं आपको call करूँगा"  # masculine stays masculine
    assert normalize("main aapko call karungi") == "मैं आपको call करूँगी"
