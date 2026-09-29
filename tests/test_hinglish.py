import re

from app.services.hinglish import _LEX, convert, roman_to_devanagari

LATIN = re.compile(r"[A-Za-z]")


def test_romanized_sentence_becomes_devanagari():
    out = convert("kya aap abhi baat kar sakte hain?")
    assert out == "क्या आप अभी बात कर सकते हैं?"


def test_english_words_kept_in_hindi_sentence():
    assert convert("आपका loan approve हो गया है") == "आपका loan approve हो गया है"
    assert convert("Please call me tomorrow.") == "Please call me tomorrow."  # "me" is amb: English here


def test_names_always_converted():
    assert convert("मेरा नाम Rahul है।") == "मेरा नाम राहुल है।"
    assert convert("Hello Rahul ji, aapki EMI due hai.") == "Hello राहुल जी, आपकी EMI due है."


def test_romanized_mode_keeps_acronyms_and_english():
    assert convert("aapka OTP kya hai") == "आपका OTP क्या है"
    assert convert("Main kal aapko call karungi, theek hai?") == "मैं कल आपको call करूँगी, ठीक है?"


def test_rules():
    assert roman_to_devanagari("kal") == "कल"
    assert roman_to_devanagari("abhi") == "अभी"
    assert roman_to_devanagari("hain") == "हैं"
    assert not LATIN.search(roman_to_devanagari("chhutti"))


def test_lexicon_is_clean():
    dev = re.compile(r"^[ऀ-ॿ ]+$")
    for kind, words in _LEX.items():
        for latin, d in words.items():
            assert latin == latin.lower() and latin.isalpha(), latin
            assert dev.match(d), (latin, d)
    assert not set(_LEX["hi"]) & set(_LEX["amb"])
