"""English words inside Hindi speech, pronounced the Indian way, as espeak-ng IPA a Hindi voice knows.

espeak-ng's Hindi voice reads Latin words with US/UK English phonemes (ɹ, w, æ, əʊ...) that a Hindi-trained
voice has barely seen, so "approve", "WhatsApp", "policy" come out garbled. Indian English differs from those
phonemes in systematic ways: retroflex t/d, tapped r, v/w merge to ʋ, monophthong o/e, rhotic vowels.
We phonemize each English run with en-us and map it onto the Hindi voice's inventory.
"""
import re

# ponytail: piper-tts==1.8.0 internals (pinned in requirements.txt). espeak-ng is not thread-safe; piper guards
# every call with this lock, and so must we.
from piper import voice as _pv
from piper.phonemize_espeak import ESPEAK_DATA_DIR, EspeakPhonemizer

# multi-symbol rules first. en-us merges "policy/document" (Indian ɔ: पॉलिसी) with "car" (Indian aː before r)
_SEQ = [("əʊ", "oː"), ("oʊ", "oː"), ("eɪ", "eː"), ("ɜː", "əɾ"), ("ɑːɹ", "aːɾ"), ("ɑː", "ɔ"), ("ɚ", "əɾ")]
_CHAR = str.maketrans({
    "t": "ʈ", "d": "ɖ", "ɹ": "ɾ", "v": "ʋ", "w": "ʋ", "æ": "ɛ", "ð": "d", "ɐ": "ʌ", "ɑ": "a", "ɜ": "ə",
    "ʒ": "z", "ᵻ": "ɪ", "ɒ": "ɔ",
})
_ENGLISH_RUN = re.compile(r"[A-Za-z][A-Za-z']*(?:[ \t]+[A-Za-z][A-Za-z']*)*")
_DANDA_AFTER_BLOCK = re.compile(r"\]\](\s*)।")


def indianize(ipa: str) -> str:
    for a, b in _SEQ:
        ipa = ipa.replace(a, b)
    return ipa.translate(_CHAR).replace("θ", "tʰ")  # after translate, so the dental t of θ stays dental


def espeak(voice: str, text: str) -> list[list[str]]:
    """The process's single espeak-ng instance. A second EspeakPhonemizer re-initializes the global C library
    under the first one and corrupts its phoneme tables (seen: 'Invalid instruction for phoneme', then SIGSEGV)."""
    with _pv._ESPEAK_PHONEMIZER_LOCK:
        if _pv._ESPEAK_PHONEMIZER is None:
            _pv._ESPEAK_PHONEMIZER = EspeakPhonemizer(ESPEAK_DATA_DIR)
        return _pv._ESPEAK_PHONEMIZER.phonemize(voice, text)


def english_ipa(words: str) -> str:
    return indianize("".join("".join(s) for s in espeak("en-us", words)).strip())


def mark(text: str) -> str:
    """Replace every English run with a Piper raw-phoneme block: 'आपका loan approve हुआ' -> 'आपका [[loːn ʌpɾuːʋ]] हुआ'.

    A danda right after a block becomes '.': espeak-ng reads a '।' with no Hindi before it as the word पूर्णविराम."""
    return _DANDA_AFTER_BLOCK.sub(r"]]\1.", _ENGLISH_RUN.sub(lambda m: f"[[{english_ipa(m[0])}]]", text))


if __name__ == "__main__":
    assert indianize("pˈeɪmənt") == "pˈeːmənʈ"
    assert indianize("θɹˈiː") == "tʰɾˈiː"
    assert indianize("lˈəʊn") == "lˈoːn"
    assert indianize("pˈɑːləsi") == "pˈɔləsi" and indianize("kˈɑːɹd") == "kˈaːɾɖ"
    print(mark("आपका loan approve हो गया, WhatsApp पर भेजा"))
