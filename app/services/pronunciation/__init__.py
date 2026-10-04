"""Hindi pronunciation layer (docs/pronunciation.md).

  raw text -> hinglish.convert + text_normalizer rules -> apply() -> TTS

apply() runs last in normalize(): numbers are words, romanized Hindi is Devanagari, only then do the evidence-backed
respellings fix what espeak-ng `hi` reads wrongly. Devanagari tokens get the Hindi lexicon and phonological rules; Latin
tokens get only the name/English tables (anything else stays Latin for the Indian-English path); numbers and punctuation
are never touched. Which rule groups run is settings.pronunciation_rules.
"""
import re
import unicodedata

from app.services.pronunciation import lexical, phonological
from app.services.pronunciation.detect import tag

__all__ = ["apply", "tag", "lexical"]

# Devanagari letters and signs, without the danda and Devanagari digits; or a Latin word
_WORD = re.compile(r"[A-Za-z][A-Za-z']*|[ऀ-ॣॱ-ॿ]+")


def apply(text: str, rules: frozenset[str]) -> str:
    """`text` with the enabled rule groups applied. No groups enabled = `text` unchanged, at the cost of one set test."""
    if not rules:
        return text
    hi, latin = lexical.tables(rules)
    phon = "schwa" in rules

    def sub(m: re.Match) -> str:
        w = m[0]
        if w[0].isascii():
            return latin.get(w.lower(), w)
        k = w if w.isascii() else unicodedata.normalize("NFC", w)
        r = hi.get(k)
        if r is not None:
            return r
        return phonological.word(k) if phon else w

    return _WORD.sub(sub, text)

