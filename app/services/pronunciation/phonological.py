"""Phonological rules on one Devanagari word (NFC). Each rule fixes a measured espeak-ng `hi` error and is gated by the
`schwa` category (docs/pronunciation.md has the evidence and the words it was checked on).

The fix is a *respelling*: an explicit virama (halant) where espeak keeps a schwa that Hindi drops. espeak then reads the
cluster, so no engine-specific phoneme string is needed. A word the parser does not fully understand is returned as is.

  H     ह loses its schwa between a vowelled syllable and a following syllable: रहता, पहचान, मेहनत, चाहता
        (espeak: rahataa, pahachaan, mehanat)
  KAR   the absolutive -कर after a consonant stem: देखकर, उठकर, बजकर (espeak: dekhakar)
  FINAL a final stop/sibilant/h/l cluster ending in a nasal drops its schwa: प्रश्न, कृष्ण, फ़िल्म, ब्रह्म (espeak: prashna)

Not done on purpose: a general schwa-deletion algorithm. Directional Ohala-style deletion contradicts espeak AND native
usage on जनवरी/दशमलव/नमस्कार, so those words are listed one by one in data/lexical.tsv instead.
"""
from functools import lru_cache

VIRAMA, NUKTA = "्", "़"
_MATRAS = frozenset("ािीुूृॄॅॆेैॉॊोौ")
_MARKS = frozenset("ँंः")
_CONS = frozenset(chr(c) for c in range(0x0915, 0x093A)) | frozenset("क़ख़ग़ज़ड़ढ़फ़य़")
_INDEP = frozenset(chr(c) for c in range(0x0904, 0x0915))
_H = ("ह", "ढ\u093c")  # ढ़ is read r.h: its h part behaves like ह (पढ़ता: espeak pa-r.h-a-taa)
_NASALS = frozenset("नमण")
_NOT_X = frozenset("रनम")  # espeak already drops the schwa after र्/न्/म् + nasal (कर्म, जन्म): leave those alone


class _U:
    """One orthographic unit: a consonant (with its vowel sign / virama) or an independent vowel."""
    __slots__ = ("cons", "v", "text", "mark")

    def __init__(self, cons: bool, v: str, text: str, mark: str) -> None:
        self.cons, self.v, self.text, self.mark = cons, v, text, mark  # v: 'a' inherent, 'm' matra, 'v' virama, 'i' independent

    @property
    def bare(self) -> bool:
        """Inherent vowel, no anusvara/chandrabindu."""
        return self.cons and self.v == "a" and not self.mark

    def halant(self) -> None:
        self.v, self.text = "v", self.text + VIRAMA


def _units(w: str) -> list[_U] | None:
    """Units of `w`; None if it holds anything but consonants, vowels, signs (digits, danda, stray marks)."""
    out, i, n = [], 0, len(w)
    while i < n:
        ch = w[i]
        if ch in _CONS:
            j = i + 1
            if j < n and w[j] == NUKTA:
                j += 1
            v = "a"
            if j < n and w[j] in _MATRAS:
                v, j = "m", j + 1
            elif j < n and w[j] == VIRAMA:
                v, j = "v", j + 1
            k = j
            while j < n and w[j] in _MARKS:
                j += 1
            out.append(_U(True, v, w[i:k], w[k:j]))
            i = j
        elif ch in _INDEP:
            j = i + 1
            while j < n and w[j] in _MARKS:
                j += 1
            out.append(_U(False, "i", w[i:j], ""))
            i = j
        else:
            return None
    return out


def _vowelled(u: list[_U], i: int) -> bool:
    """Unit i surely has a spoken vowel: independent vowel, matra, or the inherent schwa of the first unit."""
    x = u[i]
    return not x.cons or x.v == "m" or (x.v == "a" and i == 0)


@lru_cache(maxsize=8192)
def word(w: str) -> str:
    """`w` (NFC) with the phonological rules applied; unchanged when no rule fires."""
    u = _units(w) if len(w) >= 3 else None
    if not u or len(u) < 3:
        return w
    n, changed = len(u), False
    last, x = u[-1], u[-2]
    if last.bare and last.text[0] in _NASALS and last.text[-1] != NUKTA and x.cons and x.v == "v" and x.text[0] not in _NOT_X:
        last.halant()  # FINAL
        changed = True
    elif (n >= 4 and last.bare and last.text == "र" and x.bare and x.text == "क" and u[-3].bare
          and (_vowelled(u, n - 4))):
        u[-3].halant()  # KAR
        changed = True
    for i in range(n - 2, 0, -1):  # H
        if u[i].bare and u[i].text in _H and _vowelled(u, i - 1) and not u[i - 1].mark:
            nxt = u[i + 1]
            if nxt.cons and (nxt.v == "m" or (nxt.v == "a" and i + 1 < n - 1)):
                u[i].halant()
                changed = True
    return "".join(x.text + x.mark for x in u) if changed else w
