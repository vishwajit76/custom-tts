"""Pronunciation layer (app/services/pronunciation). Evidence for every rule: bench/results/pronunciation_probe_v7.tsv.
IPA assertions use the espeak-ng `hi` voice the Piper voices use and are skipped when it is unavailable. They check that
the respelled text reads the way the rule intends, not that the audio is right (nobody native has listened yet)."""
import re
import unicodedata

import pytest

from app.services import pronunciation as pron
from app.services.pronunciation import lexical, phonological
from app.services.text_normalizer import normalize

try:
    from app.services.indian_english import espeak

    espeak("hi", "है")
except Exception:  # pragma: no cover - espeak-ng / piper not installed
    espeak = None
needs_espeak = pytest.mark.skipif(espeak is None, reason="espeak-ng not available")

ALL = lexical.parse("all")


def ipa(text: str) -> str:
    return " ".join("".join(s) for s in espeak("hi", text)).replace("ˈ", "").replace("ˌ", "")


def fix(text: str, rules=ALL) -> str:
    return pron.apply(text, rules)


# --- setting / flag ---------------------------------------------------------------------------------------------
def test_default_is_off_and_normalize_is_unchanged():
    assert lexical.parse("off") == frozenset() and lexical.parse("") == frozenset()
    assert normalize("यह WhatsApp पर चाहता है") == "यह WhatsApp पर चाहता है"  # pinned by tests/data/pronunciation_corpus.tsv
    assert normalize("यह WhatsApp पर चाहता है", rules="all") == "ये वाट्सैप पर चाह्ता है"


def test_parse_rejects_unknown_group():
    with pytest.raises(ValueError):
        lexical.parse("names,typo")
    assert lexical.parse("names, english") == {"names", "english"}


def test_every_table_row_has_category_and_reason():
    for name in ("lexical.tsv", "names.tsv", "english.tsv"):
        rows = lexical._read(name)
        assert rows and all(r["category"] and len(r["reason"]) > 15 for r in rows)


# --- lexical (Hindi grammar words and word-level respellings) -------------------------------------------------------
GRAMMAR_UNCHANGED = ("है हैं हूँ हो था थी थे थीं करता करती करते रहा रही रहे गया गई गए सकता सकती जाएगा जाएगी करेगा करेगी "
                     "मिलेगा मिलेगी देगा देगी ने को से में पर ये वे मेरा मेरी मेरे आपका आपकी आपके इसका इसकी इसके").split()


@pytest.mark.parametrize("w", GRAMMAR_UNCHANGED)
def test_grammar_words_pass_through_unchanged(w):
    """The probe shows espeak reads each of these correctly (up to notation), so no rule may touch them."""
    assert fix(w) == w


def test_yah_vah_are_respelled_colloquially():
    assert fix("यह है और वह थी") == "ये है और वो थी"
    assert fix("यहाँ वहाँ यही वही") == "यहाँ वहाँ यही वही"  # only the exact words


def test_chahta_family_is_fixed_by_the_h_rule():
    assert [fix(w) for w in ("चाहता", "चाहती", "चाहते")] == ["चाह्ता", "चाह्ती", "चाह्ते"]


@needs_espeak
def test_lexical_ipa():
    assert ipa("यह") != ipa(fix("यह")) == ipa("ये") and ipa(fix("वह")) == ipa("वो")
    assert ipa("दशमलव").count("ə") > ipa(fix("दशमलव")).count("ə")  # 4 syllables -> dash-ma-lav
    assert "nʋ" in ipa(fix("जनवरी")) and "nʋ" not in ipa("जनवरी")  # jan-va-ri, not janavri
    assert "ʌhət" in ipa("रहता") and "ʌht" in ipa(fix("रहता"))


def test_number_and_month_words_the_normalizer_writes_are_covered():
    out = normalize("15 जनवरी, 3.5, 53, 61 और 2 बजकर 5 मिनट", rules="all")
    assert "जन्वरी" in out and "बज्कर" in out
    assert normalize("62", rules="all") == "बासठ"  # espeak reads बासठ correctly: not in the table
    assert normalize("61", rules="all") == "इक्सठ" and normalize("53", rules="all") == "तिर्पन"


# --- phonological: schwa -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("src,out", [
    ("रहता", "रह्ता"), ("पहचान", "पह्चान"), ("मेहनत", "मेह्नत"), ("कहना", "कह्ना"), ("गहरा", "गह्रा"), ("महसूस", "मह्सूस"),
    ("अहमद", "अह्मद"), ("दोहराना", "दोह्राना"), ("पढ़ता", "पढ़्ता"),
    ("देखकर", "देख्कर"), ("उठकर", "उठ्कर"), ("बजकर", "बज्कर"), ("सुनकर", "सुन्कर"), ("तेंदुलकर", "तेंदुल्कर"),
])
def test_schwa_rules_fire(src, out):
    assert phonological.word(unicodedata.normalize("NFC", src)) == unicodedata.normalize("NFC", out)


@pytest.mark.parametrize("w", [
    "कमला", "समझना", "अपना", "सरकार", "नमकीन", "बदलना", "धनवान", "राम", "कमल", "कर",  # espeak already right
    "बाहर", "शहर", "लहर", "दोपहर", "बहन", "रोहन", "सहन",  # ह before a final consonant keeps its schwa
    "बहुत", "सहारा", "कहानी", "नहीं", "जगह",  # ह with a vowel sign
    "नौकर", "शंकर", "भास्कर", "मुकर",  # not stem + -कर
    "कर्म", "जन्म", "धर्म", "सिस्टम", "प्रोग्राम", "ट्रेन",  # nasal-final words espeak reads right
    "नमस्ते", "123", "।",
])
def test_schwa_rules_leave_alone(w):
    assert phonological.word(w) == w


@pytest.mark.parametrize("src", ["प्रश्न", "कृष्ण", "फ़िल्म", "ब्रह्म", "रत्न", "स्वप्न", "भस्म"])
def test_final_cluster_nasal_loses_its_schwa(src):
    out = phonological.word(unicodedata.normalize("NFC", src))
    assert out == unicodedata.normalize("NFC", src) + "्"


@needs_espeak
@pytest.mark.parametrize("src", ["चाहता", "रहता", "पहचान", "मेहनत", "देखकर", "बजकर", "प्रश्न", "कृष्ण", "फ़िल्म", "ब्रह्म", "पढ़ता"])
def test_schwa_rules_remove_the_wrong_schwa(src):
    before, after = ipa(src), ipa(phonological.word(unicodedata.normalize("NFC", src)))
    assert len(re.findall("[əʌ]", after)) < len(re.findall("[əʌ]", before))


def test_lexical_schwa_words():
    assert fix("नमस्कार") == "नमसकार" and fix("राजस्थान") == "राजसथान" and fix("रामचरित") == "राम्चरित"


# --- phonological: halant, nukta, anusvara, chandrabindu, ऋ ---------------------------------------------------------
@needs_espeak
def test_explicit_halant_spellings_equal_the_conjunct_reading():
    assert ipa("सम्बन्ध") == ipa("संबंध") and ipa("अन्त") == ipa("अंत")


@needs_espeak
def test_nukta_forms_read_identically_and_are_matched_either_way():
    pre, dec = "ज़रा", "ज़रा"  # ज़रा precomposed / base + nukta
    assert ipa(pre) == ipa(dec) and ipa(pre).startswith("z")
    assert pron.apply("फ़रवरी", frozenset({"schwa"})) == "फ़र्वरी"  # फ़रवरी
    assert pron.apply("फ़रवरी", frozenset({"schwa"})) == "फ़र्वरी"


def test_nukta_letters_are_not_rewritten_by_policy():
    for w in ("ज़रूर", "फ़ोन", "क़ीमत", "ख़बर", "ग़लत", "लड़का", "बड़ा"):
        assert fix(w) == w  # espeak reads each correctly; no NFC/NFD rewriting of unmatched words


@needs_espeak
@pytest.mark.parametrize("w", ["संबंध", "अंत", "गंगा", "संपर्क", "हाँ", "माँ", "आँख", "पाँच"])
def test_anusvara_and_chandrabindu_are_read_by_espeak_and_left_alone(w):
    assert fix(w) == w and re.search("[nmŋɲ\u0303]", unicodedata.normalize("NFD", ipa(w)))  # a nasal is spoken


@needs_espeak
@pytest.mark.parametrize("w", ["कृपया", "ऋषि", "गृह", "मृत्यु"])
def test_ri_vowel_reads_as_ri(w):
    assert fix(w) == w and ipa(w).startswith(("kɾɪ", "ɾɪ", "ɡɾɪ", "mɾɪ"))


# --- brand / name lexicon -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("src,out", [
    ("Zerodha", "ज़ेरोधा"), ("Groww", "ग्रो"), ("WhatsApp", "वाट्सैप"), ("whatsapp", "वाट्सैप"), ("Aadhaar", "आधार"),
    ("Priyanka", "प्रियंका"), ("Mysuru", "मैसूरु"), ("Thiruvananthapuram", "तिरुवनंतपुरम"),
])
def test_names_and_brands(src, out):
    assert fix(src) == out
    assert normalize(f"{src} पर", rules="names") == f"{out} पर"


@needs_espeak
def test_whatsapp_and_zerodha_ipa():
    assert "ʋˈaːʈsɛːp".replace("ˈ", "") in ipa(fix("WhatsApp")) and ipa(fix("Zerodha")).startswith("zeːɾ")


def test_existing_lexicon_names_win_and_unknown_latin_is_untouched():
    assert normalize("Rahul Flipkart Paytm", rules="all") == "राहुल फ़्लिपकार्ट पेटीएम"
    assert fix("Zomato Swiggy Jio") == "Zomato Swiggy Jio"  # hinglish.convert (earlier in normalize) owns these
    assert normalize("Zomato Swiggy Jio", rules="all") == "ज़ोमैटो स्विगी जियो"
    assert fix("Quasar Xylophone") == "Quasar Xylophone"


def test_names_group_is_independent_of_english_and_schwa():
    assert fix("WhatsApp sorry", frozenset({"names"})) == "वाट्सैप sorry"
    assert fix("WhatsApp sorry", frozenset({"english"})) == "WhatsApp सॉरी"
    assert fix("WhatsApp sorry", frozenset({"schwa"})) == "WhatsApp sorry"


# --- English exceptions ---------------------------------------------------------------------------------------------
def test_english_exceptions():
    assert fix("sorry") == "सॉरी" and fix("Password reset") == "पास्वर्ड reset" and fix("insurance") == "इंश्योरेंस"
    assert fix("sorrybye") == "sorrybye" and fix("loan approve") == "loan approve"  # not in the table: Indian-English path


def test_letter_f_is_an_f_not_an_aspirated_p():
    out = normalize("PDF और HDFC", rules="all")
    assert out == "पी डी एफ़ और एच डी एफ़ सी"


@needs_espeak
def test_english_exceptions_ipa():
    assert ipa("एफ").endswith("pʰ") and ipa("एफ़").endswith("f")


# --- code-switch detection ------------------------------------------------------------------------------------------
def test_tags_for_mixed_text():
    tags = dict(pron.tag("मैं Flipkart पर order 25 करता हूँ।"))
    assert tags["मैं"] == "hi" and tags["25"] == "num" and tags["।"] == "punct" and tags["order"] == "en"


def test_romanized_hindi_is_hi_latn_and_english_stays_en():
    t = pron.tag("kya aap meeting me aayenge, please confirm")
    d = dict(t)
    assert d["kya"] == d["aap"] == "hi-Latn" and d["meeting"] == "en" and d["confirm"] == "en"


def test_pure_english_sentence_is_all_en():
    assert {t for _, t in pron.tag("The AI is ready")} == {"en"}


def test_hindi_rules_never_touch_latin_or_digits():
    # a Latin word that looks like a Hindi key, digits and punctuation: unchanged
    assert fix("rahta 123, ok.") == "rahta 123, ok."
    assert fix("पहला 1st") == "पह्ला 1st"


def test_apply_with_no_groups_is_identity():
    s = "यह चाहता WhatsApp sorry"
    assert pron.apply(s, frozenset()) is s


def test_normalize_still_idempotent_with_all_rules():
    for s in ("यह मेरा पहला WhatsApp है", "मैं पढ़ना चाहता हूँ", "15 जनवरी को देखकर बताऊँगा"):
        once = normalize(s, rules="all")
        assert normalize(once, rules="all") == once
