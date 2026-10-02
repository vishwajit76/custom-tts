"""Prosody by category: punctuation survives normalization and chunking, the pause plan follows it, short sentences stay whole.

Text only (no model): the sound of a pause is checked in tests/test_audio_quality.py, the '?' reaching the VITS model below.
"""
import pytest

from app.core.config import settings
from app.services import audio_utils, indian_english, text_normalizer, tts

# category -> (sentence, terminal class gap_class() must give it)
CASES = {
    "statement": ("मैं आपकी किस्त के बारे में बात कर रही हूँ।", "sentence"),
    "yes/no question": ("क्या आप अभी बात कर सकते हैं?", "question"),
    "wh-question": ("आपका नाम क्या है?", "question"),
    "exclamation": ("बहुत बढ़िया!", "exclaim"),
    "confirmation": ("जी हाँ, बिल्कुल सही है।", "sentence"),
    "uncertainty": ("शायद कल, पक्का पता नहीं।", "sentence"),
    "apology": ("माफ़ कीजिए, मुझे समझ नहीं आया।", "sentence"),
    "request": ("कृपया थोड़ा इंतज़ार कीजिए।", "sentence"),
    "instruction": ("अपना OTP किसी को न बताएँ।", "sentence"),
}
NEXT = "ठीक है।"


@pytest.mark.parametrize("name", CASES)
def test_punctuation_survives_and_short_sentence_is_one_piece(name):
    s, _ = CASES[name]
    marks = [c for c in s if c in ",।?!"]
    norm = text_normalizer.normalize(s)
    assert [c for c in norm if c in ",।?!"] == marks
    pieces = tts.split_for_stream(norm)
    assert len(pieces) == 1 and [c for c in pieces[0] if c in ",।?!"] == marks and pieces[0][-1] == s[-1]


@pytest.mark.parametrize("name", CASES)
def test_pause_plan_follows_the_ending(name):
    s, cls = CASES[name]
    first, second = tts.split_for_stream(text_normalizer.normalize(f"{s} {NEXT}"))
    assert second == NEXT and tts.gap_class(first) == cls
    assert tts.gap_ms(first) == audio_utils.parse_kv(settings.pause_plan)[cls]


def test_plan_orders_pauses_by_how_strong_the_break_is():
    plan = audio_utils.parse_kv(settings.pause_plan)
    assert plan["phrase"] < plan["comma"] < plan["colon"] < plan["sentence"] <= plan["question"] < plan["ellipsis"]
    assert plan["exclaim"] <= plan["question"] and plan["dash"] > plan["comma"]


def test_gap_classes_cover_every_mark_and_ignore_closing_quotes():
    for text, cls in [("ठीक है,", "comma"), ("ठीक है;", "colon"), ("ध्यान दें:", "colon"), ("ठीक है -", "dash"), ("ठीक है —", "dash"),
                      ("हम्म...", "ellipsis"), ("हम्म…", "ellipsis"), ("ठीक है।", "sentence"), ("Okay.", "sentence"),
                      ("ठीक है?", "question"), ("ठीक है!", "exclaim"), ('उसने कहा "ठीक है।"', "sentence"), ("मैं आपकी", "phrase")]:
        assert tts.gap_class(text) == cls, text


def test_a_long_sentence_is_cut_at_a_comma_with_the_comma_kept():
    s = ("आपकी किस्त की तारीख पंद्रह अगस्त है, इसलिए कृपया उससे पहले भुगतान कर दीजिए, नहीं तो आपके खाते पर अतिरिक्त शुल्क लगेगा, "
         "और यह आपकी क्रेडिट रेटिंग पर भी असर डाल सकता है।")
    pieces = tts.split_for_stream(text_normalizer.normalize(s))
    assert len(pieces) > 1 and " ".join(pieces).count(",") == s.count(",") and pieces[-1].endswith("।")
    assert all(tts.gap_class(p) in ("comma", "phrase") for p in pieces[:-1])  # never a sentence pause inside a sentence


def test_ellipsis_between_words_is_its_own_longer_pause():
    first, second = tts.split_for_stream("हम्म... ठीक है।")
    assert tts.gap_class(first) == "ellipsis" and second == "ठीक है।"


@pytest.mark.parametrize("name,mark", [("yes/no question", "?"), ("wh-question", "?"), ("exclamation", "!")])
def test_question_and_exclamation_marks_reach_the_vits_model(name, mark):
    """The VITS voice learned question intonation from '?': the phoneme string it gets must end with the mark."""
    phonemize = pytest.importorskip("piper.phonemize_espeak").EspeakPhonemizer().phonemize
    piece = tts.split_for_stream(text_normalizer.normalize(CASES[name][0]))[0]
    sent = indian_english.mark(piece)  # what PiperEngine.synth hands to espeak
    assert phonemize("hi", sent)[-1][-1] == mark
