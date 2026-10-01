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
# English words that happen to match _HINDI_SHAPE ("meeting" has "ee"). Never transliterated: the shape heuristic is
# vetoed by these suffixes and by the explicit set.
_ENGLISH_SUFFIX = re.compile(r"(?:ing|tion|sion|ment|ness|ity|ous|able|ible)s?$")
_ENGLISH_SHAPE_WORDS = frozenset(
    "meeting meetings feedback speed free need deep green screen agree agreed degree between week weekend keep "
    "sleep street sheet three teen queen been seen team feel steel wheel proceed succeed indeed speech reach "
    "teach beach each real deal appeal".split()
)


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


# "use" is उसे in Romanized Hindi ("use bulao") but the English verb before a light verb ("Paytm use karein").
_LIGHT_VERB = re.compile(r"\s+(?:kar|kij|kare|karo|karn|karke|karein|karen)", re.IGNORECASE)


def _sentence(m: re.Match) -> str:
    s = m[0]
    words = [w.lower() for w in _LATIN.findall(s)]
    # A sentence counts as Romanized Hindi only with at least one word that is never English; "The AI is ready" has only
    # the ambiguous "the"/"is" and must stay English. Names/brands are converted anyway, so they don't dilute the ratio.
    sure = sum(w in _ALWAYS and w not in _LEX["name"] for w in words)
    hits = sum(w in _HINDI for w in words)
    total = sum(w not in _LEX["name"] for w in words)
    romanized = not _DEVANAGARI.search(s) and sure >= 1 and hits >= 2 and 2 * hits >= total

    def word(wm: re.Match) -> str:
        w = wm[0]
        k = w.lower()
        if k in _ALWAYS:
            return _ALWAYS[k]
        if not romanized:
            return w
        if k == "hi" and w == "Hi":
            return w  # the greeting ("Hi, main Priya bol rahi hoon"); the Hindi particle is written lowercase
        if k == "use" and _LIGHT_VERB.match(s, wm.end()):
            return w  # English verb: "use karein"
        if k == "ai" and w == "ai":
            return "ए आई"  # lazily typed "AI" (Hindi has no common Romanized "ai"); uppercase AI is an acronym rule
        if k in _LEX["amb"]:
            return _LEX["amb"][k]
        if (w.isupper() and len(w) > 1) or not _HINDI_SHAPE.search(k):
            return w  # acronym (OTP) or probably English (call, office)
        if k in _ENGLISH_SHAPE_WORDS or _ENGLISH_SUFFIX.search(k):
            return w  # English word that looks Hindi-shaped (meeting, feedback)
        return roman_to_devanagari(k)

    return _LATIN.sub(word, s)


def brand(word: str) -> str:
    """Devanagari spelling of a known Indian name or brand (lexicon kind "name"); any other word unchanged."""
    return _LEX["name"].get(word.lower(), word)


def convert(text: str) -> str:
    return _SENTENCE.sub(_sentence, text)


# --- opt-in persona gender agreement -------------------------------------------------------------------------
# NEVER called by normalize()/convert() or by the API. The application calls it explicitly, only on text it wrote
# for its own persona (a system-authored script), never on arbitrary user content. Scope is first-person verb
# agreement only; it does not touch third-person or quoted speech, and it cannot know which "गया" refers to the
# speaker, so it is deliberately limited to fixed, unambiguous first-person phrases.
_FEM = {  # masculine -> feminine. Only phrases that are unambiguously first person (explicit हूँ/ऊँगा or मैं)
    "करता हूँ": "करती हूँ", "करता हूं": "करती हूं", "रहा हूँ": "रही हूँ", "रहा हूं": "रही हूं",
    "सकता हूँ": "सकती हूँ", "सकता हूं": "सकती हूं", "चाहता हूँ": "चाहती हूँ", "चाहता हूं": "चाहती हूं",
    "जानता हूँ": "जानती हूँ", "समझता हूँ": "समझती हूँ", "देता हूँ": "देती हूँ", "लेता हूँ": "लेती हूँ",
    "बताता हूँ": "बताती हूँ", "भेजता हूँ": "भेजती हूँ", "बोलता हूँ": "बोलती हूँ",
    "करूँगा": "करूँगी", "करूंगा": "करूंगी", "आऊँगा": "आऊँगी", "दूँगा": "दूँगी", "बताऊँगा": "बताऊँगी",
    "भेजूँगा": "भेजूँगी", "देखूँगा": "देखूँगी", "मैं गया था": "मैं गई थी", "मैं आया था": "मैं आई थी",
    "मैं गया": "मैं गई", "मैं आया": "मैं आई",
    "आपका सहायक हूँ": "आपकी सहायक हूँ", "आपका असिस्टेंट हूँ": "आपकी असिस्टेंट हूँ",
}
_MASC = {f: m for m, f in _FEM.items()}


def apply_persona_gender(text: str, gender: str) -> str:
    """Rewrite first-person Hindi verb endings to the persona's grammatical gender ("male"/"female"/"m"/"f").

    Explicit, opt-in, phrase-level (fixed phrases only, longest first). Returns text unchanged for any other gender
    value (including "neutral"/None), so callers cannot get a silent rewrite. Never called automatically."""
    g = (gender or "").strip().lower()
    if g in ("female", "f", "woman", "feminine"):
        table = _FEM
    elif g in ("male", "m", "man", "masculine"):
        table = _MASC
    else:
        return text
    for src in sorted(table, key=len, reverse=True):
        if src != table[src]:
            text = text.replace(src, table[src])
    return text
