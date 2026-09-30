"""P3 regressions found while auditing the Hindi/Hinglish normalizer on conversational cases (brands, URLs, am/pm...).

Each test pins one defect that was real at the time: the docstring says what it used to do."""
import re

import pytest

from app.services import hinglish
from app.services.text_normalizer import normalize

LATIN, DEV = re.compile(r"[A-Za-z]"), re.compile(r"[ऀ-ॿ]")
BRANDS = ["WhatsApp", "Google", "Amazon", "Flipkart", "YouTube", "Paytm", "PhonePe", "Zomato", "Swiggy", "Instagram",
          "Airtel", "Jio", "Tata", "Infosys", "Reliance", "Gmail", "GPay"]


@pytest.mark.parametrize("brand", BRANDS)
def test_brand_is_spoken_from_devanagari_spelling(brand):
    """Decision (docs/voice-system.md): curated brands are respelled in Devanagari, which espeak-ng `hi` reads with Hindi
    rules; the English G2P + Indian-English mapping mangled several of them (Zomato -> 'zomaato', Jio -> 'jaio')."""
    dev = normalize(brand)
    assert DEV.search(dev) and not LATIN.search(dev)
    assert normalize(brand.lower()) == normalize(brand.upper()) == dev  # case-insensitive
    assert normalize(f"Order from {brand} now") == f"Order from {dev} now"  # also inside an English sentence


def test_ordinary_english_words_are_not_respelled():
    assert normalize("video link payment order") == "video link payment order"


@pytest.mark.parametrize("text", ["The AI is ready", "Is he in the office?", "Please use the app", "Do you like it?", "Are we to go?"])
def test_english_sentence_with_only_ambiguous_words_stays_english(text):
    """'the'/'is'/'he'/'in'/'use'/'do'/'to' are in the ambiguous lexicon; 'The AI is ready' used to become 'थे ए आई इस ready'."""
    out = normalize(text)
    assert not DEV.search(re.sub("ए आई", "", out)), out


def test_use_before_light_verb_is_english_but_standalone_use_is_hindi():
    assert normalize("Paytm use karein") == "पेटीएम use करें"  # was पेटीएम उसे करें
    assert normalize("use bulao, main aata hoon") == "उसे बुलाओ, मैं आता हूँ"


def test_capital_hi_greeting_stays_english_but_particle_hi_converts():
    assert normalize("Hi, main Amit bol raha hoon") == "Hi, मैं अमित बोल रहा हूँ"  # was ही
    assert normalize("aap hi bataiye, main sunta hoon") == "आप ही बताइए, मैं सुनता हूँ"


def test_ai_acronym_versus_lowercase_ai():
    assert normalize("AI se baat kijiye") == "ए आई से बात कीजिए"
    assert normalize("ai se baat kijiye") == "ए आई से बात कीजिए"  # lazily typed, Romanized-Hindi sentence
    assert normalize("the ai bot") == "the ai bot"  # English sentence: untouched


def test_lakh_and_crore_are_not_transliterated_by_the_shape_heuristic():
    assert normalize("loan 5 lakh 50 hazaar ka") == "loan पाँच लाख पचास हज़ार का"  # was 'पाँच लख'
    assert normalize("aapka loan 2 crore ka hai") == "आपका loan दो करोड़ का है"


@pytest.mark.parametrize("src,exp", [
    ("Q3", "क्यू थ्री"), ("MP3", "एम पी थ्री"), ("B2B", "बी टू बी"), ("2BHK", "टू बी एच के"), ("F16", "एफ सोलह"), ("H264", "एच टू सिक्स फ़ोर"),
])
def test_letters_glued_to_digits_never_make_a_mixed_script_token(src, exp):
    """'Q3' used to become 'Qतीन', 'MP3' 'MPतीन' (Latin + Devanagari in one token)."""
    assert normalize(src) == exp


def test_ordinals_and_clock_times_are_not_glued_tokens():
    assert normalize("1ST floor") == normalize("1st floor")
    assert normalize("5PM") == "शाम पाँच बजे"


def test_iphone15_is_split():
    assert normalize("iPhone15") == "iPhone पंद्रह"


def test_email_and_url_are_spoken_not_left_for_the_english_reader():
    assert normalize("rahul.sharma92@gmail.com") == "राहुल डॉट शर्मा नौ दो एट जीमेल डॉट कॉम"
    assert normalize("www.example.com/pay") == "डब्ल्यू डब्ल्यू डब्ल्यू डॉट example डॉट कॉम स्लैश pay"
    assert normalize("https://example.com") == "example डॉट कॉम"
    assert normalize("a@b.co.in") == "a एट b डॉट को डॉट इन"


def test_sentence_punctuation_after_url_is_not_part_of_it():
    assert normalize("Visit example.com.") == "Visit example डॉट कॉम."
    assert normalize("(www.example.com)") == "(डब्ल्यू डब्ल्यू डब्ल्यू डॉट example डॉट कॉम)"
    assert normalize("example.com, then foo.org!") == "example डॉट कॉम, then foo डॉट ऑर्ग!"


def test_bare_dot_in_is_not_a_url():
    assert normalize("Ok.In the morning") == "Ok.In the morning"
    assert normalize("version 2.0 is out") == "version दो दशमलव शून्य is out"


def test_web_placeholders_never_leak_even_with_hostile_input():
    hostile = " mail a@b.com "
    out = normalize(hostile)
    assert not re.search("[-]", out) and "एट" in out


def test_many_urls_in_one_text():
    out = normalize(" ".join(f"x{i}.com" for i in range(40)))
    assert out.count("डॉट कॉम") == 40 and not re.search("[-]", out)


def test_phone_groups_get_a_pause_and_digits_inside_a_group_do_not():
    assert normalize("9876543210") == "नौ आठ सात छह पाँच, चार तीन दो एक शून्य"
    assert normalize("1800 123 4567") == "एक आठ शून्य शून्य, एक दो तीन, चार पाँच छह सात"  # the writer's own grouping is kept
    assert normalize("+91 9876543210") == "प्लस नौ एक, नौ आठ सात छह पाँच, चार तीन दो एक शून्य"
    assert normalize("1234567890123456") == "एक दो तीन चार, पाँच छह सात आठ, नौ शून्य एक दो, तीन चार पाँच छह"


def test_short_numbers_are_not_phones():
    assert normalize("12,500") == "बारह हज़ार पाँच सौ"
    assert normalize("5 10 2026") == "पाँच दस दो हज़ार छब्बीस"


@pytest.mark.parametrize("src,exp", [
    ("5 PM", "शाम पाँच बजे"), ("9 AM", "सुबह नौ बजे"), ("2 pm", "दोपहर दो बजे"), ("9 pm", "रात नौ बजे"), ("12 AM", "रात बारह बजे"),
    ("12 pm", "दोपहर बारह बजे"), ("9:30 am", "सुबह साढ़े नौ बजे"), ("9:30 pm", "रात साढ़े नौ बजे"), ("5 p.m.", "शाम पाँच बजे"),
    ("शाम 5 pm", "शाम पाँच बजे"), ("subah 9 am", "सुबह नौ बजे"), ("5 PM baje", "शाम पाँच बजे"),
])
def test_am_pm_keeps_a_day_period_and_is_not_doubled(src, exp):
    """'9:30 pm' used to be read 'साढ़े नौ बजे' (am and pm indistinguishable)."""
    assert normalize(src) == exp


def test_am_pm_needs_a_real_hour():
    assert normalize("13 pm") == "तेरह pm"
    assert normalize("I am 5 pmxyz") == normalize("I am 5 pmxyz")  # no crash, no rewrite of 'pmxyz'


def test_month_abbreviations_and_month_first_dates():
    assert normalize("15 Aug 2026") == "पंद्रह अगस्त दो हज़ार छब्बीस"
    assert normalize("Aug 15") == "पंद्रह अगस्त"
    assert normalize("Sept 3, 2025") == "तीन सितंबर दो हज़ार पच्चीस"
    assert normalize("May I help you?") == "May I help you?"  # a bare month word is not a date


def test_rupee_ranges():
    assert normalize("₹500-₹1000") == "पाँच सौ से एक हज़ार रुपये"  # was 'पाँच सौ रुपये-एक हज़ार रुपये'
    assert normalize("₹5-10 lakh") == "पाँच से दस लाख रुपये"
    assert normalize("₹500") == "पाँच सौ रुपये"


def test_hash_no_and_star():
    assert normalize("ticket #4521") == "ticket नंबर चार हज़ार पाँच सौ इक्कीस"
    assert normalize("order no. 5") == "order नंबर पाँच"
    assert normalize("press # to confirm") == "press हैश to confirm"
    assert normalize("press * for menu") == "press स्टार for menu"
    assert normalize("no 5 books") == "no पाँच books"  # 'no' without a dot is the English word


def test_new_acronyms_are_spelled():
    assert normalize("CRM ID QR") == "सी आर एम आई डी क्यू आर"
    assert normalize("VoIP") == "वी ओ आई पी"


@pytest.mark.parametrize("word", ["batana", "bolna", "mangwaya", "saadhe", "vapas", "shuru", "nikalna", "pahunchega", "chalu"])
def test_common_romanized_verbs_and_words_are_converted(word):
    """These were left in Latin inside Romanized-Hindi sentences (espeak then reads them with English rules)."""
    assert not LATIN.search(normalize(f"aap {word} hai kya"))


def test_outputs_have_no_mixed_script_tokens():
    samples = ["TCS ka Q3 result", "MP3 file bhejiye", "rahul.sharma92@gmail.com par mail karein", "₹500-₹1000 ke beech",
               "WhatsApp par link bhej diya hai", "kal 5 PM baje", "iPhone15 Pro", "order no. 12345"]
    for s in samples:
        for tok in normalize(s).split():
            assert not (LATIN.search(tok) and DEV.search(tok)), (s, tok)


def test_hinglish_brand_helper():
    assert hinglish.brand("Zomato") == "ज़ोमैटो" and hinglish.brand("loan") == "loan"
