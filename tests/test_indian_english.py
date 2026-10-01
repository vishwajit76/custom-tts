from concurrent.futures import ThreadPoolExecutor

import pytest

from app.services.indian_english import espeak, indianize, mark


def test_indianize_rules():
    assert indianize("pˈeɪmənt") == "pˈeːmənʈ"  # monophthong e, retroflex t
    assert indianize("θɹˈiː") == "tʰɾˈiː"  # dental aspirated th, tapped r
    assert indianize("pˈɑːləsi") == "pˈɔləsi" and indianize("kˈɑːɹd") == "kˈaːɾɖ"


def test_indianize_keeps_flapped_t_affricates_and_single_r():
    # heard on a call as unclear: "property" lost its t, "enquiry" got a double r, "budget" became "budzit"
    assert indianize("pɹˈɑːpɚɾi") == "pɾˈɔpəɾʈi"  # property: प्रॉपर्टी
    assert indianize("ɛŋkwˈaɪɚɹi") == "ɛŋkwˈaːɪəɾi"  # enquiry: इंक्वायरी (kʋ was heard as एंखारी)
    assert indianize("bˈʌdʒɪt") == "bˈʌɟɪʈ" and indianize("tʃˈɛk") == "cˈɛk"  # budget, check: ज/च


def test_mark_only_touches_english():
    out = mark("आपका loan approve हो गया")
    assert out.startswith("आपका [[") and out.endswith("]] हो गया") and "ɾ" in out
    assert mark("अभी payment।").endswith("]].")  # a lone । is read aloud as पूर्णविराम by espeak-ng


def test_espeak_is_thread_safe_across_voices():
    """Hindi and English phonemization interleaved from many threads (what the worker pool does) must be stable."""
    jobs = [("hi", "नमस्ते, आपका दिन शुभ हो।"), ("en-us", "please confirm your payment")] * 150
    expected = {v: espeak(v, t) for v, t in jobs[:2]}
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda j: (j[0], espeak(*j)), jobs))
    assert all(r == expected[v] for v, r in results)


def test_kokoro_g2p_indian_english_and_no_spoken_danda():
    kokoro = pytest.importorskip("app.services.kokoro_engine")  # optional engine (requirements-engines.txt)
    ps = kokoro.phonemes("आपका loan approve हुआ, online payment।").replace("ˈ", "")
    assert "loːn" in ps and "puːɾn" not in ps and "puːrn" not in ps
