"""Romanized Hindi (Hinglish) -> Devanagari, before phonemization.

espeak-ng's Hindi voice reads Devanagari with Hindi rules and Latin words with English rules. That is right
for real English ("loan", "confirm") but wrong for Hindi typed in Latin ("kya aap baat kar sakte hain")
and for Indian names ("Rahul"). So: lexicon words and names -> Devanagari; English stays Latin.
"""
import re
from pathlib import Path

_LEX: dict[str, dict[str, str]] = {"hi": {}, "amb": {}, "name": {}}
for _line in (Path(__file__).parent / "lexicon_hi.tsv").read_text("utf-8").splitlines():
    if _line and not _line.startswith("#"):
        _latin, _dev, _kind = _line.split("\t")
        _LEX[_kind][_latin] = _dev
_HINDI = _LEX["hi"] | _LEX["amb"]
_ALWAYS = _LEX["hi"] | _LEX["name"]

# Greedy longest match. Casual romanization is lossy (t = त or ट); the lexicon covers the common words.
_CONS = {
    "chh": "छ", "cch": "च्छ", "ksh": "क्ष", "kh": "ख", "gh": "घ", "ch": "च", "jh": "झ", "th": "थ", "dh": "ध",
    "ph": "फ", "bh": "भ", "sh": "श", "cc": "च्च", "k": "क", "g": "ग", "c": "क", "j": "ज", "t": "त", "d": "द",
    "n": "न", "p": "प", "f": "फ़", "b": "ब", "m": "म", "y": "य", "r": "र", "l": "ल", "v": "व", "w": "व",
    "s": "स", "h": "ह", "z": "ज़", "x": "क्स", "q": "क़",
}
_VOWELS = {  # latin: (independent, matra)
    "aa": ("आ", "ा"), "ee": ("ई", "ी"), "ii": ("ई", "ी"), "oo": ("ऊ", "ू"), "uu": ("ऊ", "ू"), "ai": ("ऐ", "ै"),
    "au": ("औ", "ौ"), "a": ("अ", ""), "i": ("इ", "ि"), "u": ("उ", "ु"), "e": ("ए", "े"), "o": ("ओ", "ो"),
}
_TOKENS = sorted([*_CONS, *_VOWELS], key=len, reverse=True)
_LATIN = re.compile(r"[A-Za-z]+")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_SENTENCE = re.compile(r"[^.?!।]+[.?!।]*|[.?!।]+")
# ponytail: shape heuristic for unknown words inside Romanized-Hindi sentences; anything else is assumed
# English ("call", "office"). Add frequent misses to lexicon_hi.tsv instead of growing this regex.
_HINDI_SHAPE = re.compile(r"aa|ee|kh|gh|bh|dh|jh|chh|iye$|enge$|unga$|ungi$|oge$|ogi$|iyo$")


def roman_to_devanagari(word: str) -> str:
    w, out, i, after_cons = word.lower(), [], 0, False
    while i < len(w):
        tok = next((t for t in _TOKENS if w.startswith(t, i)), None)
        if tok is None:  # unreachable for [a-z], kept for safety
            out.append(w[i])
            i += 1
            continue
        if tok in _CONS:
            if after_cons:
                out.append("्")  # consonant cluster
            out.append(_CONS[tok])
            after_cons = True
        else:
            indep, matra = _VOWELS[tok]
            if i + 1 == len(w) and tok in ("a", "i"):  # final written a/i are spoken long: kya, tha, abhi, sabhi
                indep, matra = {"a": ("आ", "ा"), "i": ("ई", "ी")}[tok]
            if after_cons:
                out.append(matra)
            else:
                out.append(indep)
            after_cons = False
        i += len(tok)
    s = "".join(out)
    # final nasal after a long vowel: hain -> हैं, nahin -> नहीं, haan -> हाँ
    return re.sub(r"([ाीूैोे])न$", lambda m: m[1] + ("ँ" if m[1] in "ाू" else "ं"), s)


def _sentence(m: re.Match) -> str:
    s = m[0]
    words = [w.lower() for w in _LATIN.findall(s)]
    hits = sum(w in _HINDI for w in words)
    romanized = not _DEVANAGARI.search(s) and hits >= 2 and 2 * hits >= len(words)

    def word(wm: re.Match) -> str:
        w = wm[0]
        k = w.lower()
        if k in _ALWAYS:
            return _ALWAYS[k]
        if not romanized:
            return w
        if k in _LEX["amb"]:
            return _LEX["amb"][k]
        if (w.isupper() and len(w) > 1) or not _HINDI_SHAPE.search(k):
            return w  # acronym (OTP) or probably English (call, office)
        return roman_to_devanagari(k)

    return _LATIN.sub(word, s)


def convert(text: str) -> str:
    return _SENTENCE.sub(_sentence, text)
