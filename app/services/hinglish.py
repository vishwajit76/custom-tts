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
        if k == "hey" and not s[:wm.start()].strip():
            return w  # greeting opens the sentence ("Hey, kya haal hai?"); later it is "hai" misspelt ("thik hey")
        if k == "use" and _LIGHT_VERB.match(s, wm.end()):
            return w  # English verb: "use karein"
        if k == "ai" and w == "ai":
            return "ए आई"  # lazily typed "AI" (Hindi has no common Romanized "ai"); uppercase AI is an acronym rule
        if k in _LEX["amb"]:
            if "-" in (s[wm.start() - 1:wm.start()], s[wm.end():wm.end() + 1]):
                return w  # inside an English compound: "ready-to-move", "up-to-date"
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
# NEVER called by normalize()/convert() or by the API. Thin wrapper over app/services/persona_grammar.py (the rules
# live there); speaker-authored text only.
def apply_persona_gender(text: str, gender: str) -> str:
    """Rewrite first-person Hindi agreement to the persona's gender ("male"/"female"/"m"/"f"); any other value
    (including "neutral"/None) returns text unchanged."""
    from app.services import persona_grammar  # lazy: hinglish.py is vendored into the Kaggle bundle without it

    return persona_grammar.apply(text, persona_grammar.Persona(persona_grammar.normalize_gender(gender)))
