"""Hindi text normalization: numbers, currency, dates, percentages, abbreviations, chunking.

Latin-script (Hinglish) words are left untouched; the model reads them as English.
"""
import re

from app.services import hinglish

_UNITS = (
    "शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस "
    "बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस "
    "पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस "
    "सैंतालीस अड़तालीस उनचास पचास इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ "
    "इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर "
    "पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी "
    "अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पचानवे छियानवे सत्तानवे अट्ठानवे निन्यानवे"
).split()
assert len(_UNITS) == 100

_SCALES = [(10**7, "करोड़"), (10**5, "लाख"), (1000, "हज़ार"), (100, "सौ")]
_MONTHS = "जनवरी फ़रवरी मार्च अप्रैल मई जून जुलाई अगस्त सितंबर अक्टूबर नवंबर दिसंबर".split()
_ABBR = {
    "डॉ.": "डॉक्टर", "श्री.": "श्री", "कि.मी.": "किलोमीटर", "किमी": "किलोमीटर",
    "Dr.": "डॉक्टर", "Mr.": "मिस्टर", "Mrs.": "मिसेज़", "Rs.": "रुपये", "Rs": "रुपये",
    "km": "किलोमीटर", "kg": "किलो", "EMI": "ई एम आई", "24/7": "चौबीसों घंटे", "24x7": "चौबीसों घंटे", "OTP": "ओ टी पी", "KYC": "के वाई सी",
    "Ltd.": "लिमिटेड", "Pvt.": "प्राइवेट", "No.": "नंबर",
    "Sr.": "सीनियर", "Jr.": "जूनियर", "Smt.": "श्रीमती", "Prof.": "प्रोफ़ेसर",
}
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_NUM = r"\d[\d,]*(?:\.\d+)?"

_SCALE_WORDS = {"lakh": "लाख", "lac": "लाख", "crore": "करोड़", "लाख": "लाख", "करोड़": "करोड़", "हज़ार": "हज़ार", "हजार": "हज़ार"}
_ORDINAL_WORDS = {1: "पहला", 2: "दूसरा", 3: "तीसरा", 4: "चौथा", 5: "पाँचवाँ", 6: "छठा"}

# Currency / amount expressions run before ABBR so "Rs 500" reads the amount first.
_RUPEE_RE = re.compile(rf"₹\s*({_NUM})(?:\s*(lakh|lac|crore|लाख|करोड़))?", re.IGNORECASE)
_RS_RE = re.compile(rf"(?<!\w)Rs\.?\s*({_NUM})")
_INR_HEAD_RE = re.compile(rf"(?<!\w)INR\s*({_NUM})", re.IGNORECASE)
_INR_TAIL_RE = re.compile(rf"({_NUM})\s*INR(?!\w)", re.IGNORECASE)
_DOLLAR_HEAD_RE = re.compile(rf"\$\s*({_NUM})")
_DOLLAR_TAIL_RE = re.compile(rf"({_NUM})\s*USD(?!\w)", re.IGNORECASE)
_SCALE_RE = re.compile(rf"({_NUM})\s*(lakh|lac|crore|लाख|करोड़|हज़ार|हजार)(?![a-z])", re.IGNORECASE)
_RUPEE_TAIL_RE = re.compile(rf"({_NUM})\s*(?:रुपये|रुपए)")

# Units directly after a number (before ABBR so "5 kg" -> "किलो", not "किलोमीटर").
_UNITS_MAP = {
    "km": "किलोमीटर", "kg": "किलो", "g": "ग्राम", "gm": "ग्राम",
    "ml": "मिलीलीटर", "l": "लीटर", "ltr": "लीटर",
    "gb": "जीबी", "mb": "एमबी", "min": "मिनट", "mins": "मिनट",
    "hr": "घंटे", "hrs": "घंटे", "sec": "सेकंड",
}
_UNIT_RE = re.compile(
    r"(?<!\w)(" + _NUM + r")\s*("
    + "|".join(sorted(_UNITS_MAP, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)

_DATE_RE = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")
# Optional trailing am/pm/PM/बजे is consumed so "10:30 बजे" does not double "बजे".
# 10:30 [pm|बजे], or Indian-style 5.30 only when followed by pm/बजे (else it is a decimal)
_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2})(?:\s*(?:[ap]\.?m\.?|बजे))?|\.(\d{2})\s*(?:[ap]\.?m\.?|बजे))(?!\d)", re.IGNORECASE)
# +91 style prefixes, or 10+ digit sequences separated by spaces/dashes.
_PHONE_RE = re.compile(r"\+(\d+)(?:[\s\-]+(\d[\d\s\-]*\d))?|\d[\d\s\-]*\d")
_RANGE_RE = re.compile(rf"({_NUM})\s*[-–]\s*({_NUM})")
_PCT_RE = re.compile(rf"({_NUM})\s*%")
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_AMP_RE = re.compile(r"&")
_AT_RE = re.compile(r"@")
_PLUS_RE = re.compile(r"(?<=\w)\s*\+\s*(?=\w)")
_EQ_RE = re.compile(r"=")
_SLASH_RE = re.compile(r"([^\s/]+)\s*/\s*([^\s/]+)")
_NUM_RE = re.compile(_NUM)


def num_to_words(n: int) -> str:
    if n < 100:
        return _UNITS[n]
    for value, name in _SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            words = f"{num_to_words(head)} {name}"
            return f"{words} {num_to_words(rest)}" if rest else words
    raise AssertionError  # unreachable


def _digits(s: str) -> str:
    return " ".join(_UNITS[int(d)] for d in s)


def _number(s: str) -> str:
    s = s.replace(",", "")
    if "." in s:
        whole, frac = s.split(".", 1)
        return f"{num_to_words(int(whole or 0))} दशमलव {_digits(frac)}"
    # phone numbers / long ids / leading zeros: read digit by digit
    if len(s) > 9 or (len(s) > 1 and s[0] == "0"):
        return _digits(s)
    return num_to_words(int(s))


def _date(m: re.Match) -> str:
    d, mo, y = int(m[1]), int(m[2]), m[3]
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return m[0]
    y = int(y) + 2000 if len(y) == 2 else int(y)
    return f"{num_to_words(d)} {_MONTHS[mo - 1]} {num_to_words(y)}"


def _currency(m: re.Match) -> str:
    amount = m[1].replace(",", "")
    whole, _, paise = amount.partition(".")
    out = f"{num_to_words(int(whole))} रुपये"
    if paise and int(paise):
        out += f" {num_to_words(int(paise[:2].ljust(2, '0')))} पैसे"
    return out


def _amount(s: str) -> str:
    """Like _number, but x.5 reads the spoken way: 1.5 -> डेढ़, 2.5 -> ढाई, 3.5 -> साढ़े तीन."""
    whole, _, frac = s.replace(",", "").partition(".")
    if frac == "5" and whole:
        return {"1": "डेढ़", "2": "ढाई"}.get(whole) or f"साढ़े {num_to_words(int(whole))}"
    return _number(s)


def _rupee(m: re.Match) -> str:
    amount, scale = m[1], m[2]
    if scale:
        return f"{_amount(amount)} {_SCALE_WORDS[scale.lower()]} रुपये"
    return _currency(m)


def _dollar(m: re.Match) -> str:
    return f"{_number(m[1])} डॉलर"


def _scale(m: re.Match) -> str:
    return f"{_amount(m[1])} {_SCALE_WORDS[m[2].lower()]}"


def _time(m: re.Match) -> str:
    h = int(m[1])
    mi = int(m[2] or m[3])
    hour = num_to_words(h % 12 or 12)
    if mi == 0:
        return f"{hour} बजे"
    if mi == 15:
        return f"सवा {hour} बजे"
    if mi == 30:
        if h == 1:
            return "डेढ़ बजे"
        if h == 2:
            return "ढाई बजे"
        return f"साढ़े {hour} बजे"
    if mi == 45:
        return f"पौने {num_to_words((h % 12) + 1)} बजे"
    return f"{hour} बजकर {num_to_words(mi)} मिनट"


def _phone(m: re.Match) -> str:
    s = m[0]
    if s[0] == "+":
        cc = ""
        i = 1
        while i < len(s) and s[i].isdigit():
            cc += s[i]
            i += 1
        rest = re.sub(r"\D", "", s[i:])
        out = "प्लस " + _digits(cc)
        if rest:
            out += " " + _digits(rest)
        return out
    digits = re.sub(r"\D", "", s)
    if len(digits) < 10:
        return s
    return _digits(digits)


def _range(m: re.Match) -> str:
    return f"{_number(m[1])} से {_number(m[2])}"


def _unit(m: re.Match) -> str:
    return f"{m[1]} {_UNITS_MAP[m[2].lower()]}"


def _ordinal(m: re.Match) -> str:
    n = int(m[1])
    return _ORDINAL_WORDS.get(n, num_to_words(n) + "वाँ")


def normalize(text: str) -> str:
    text = text.translate(_DEVANAGARI_DIGITS)
    # first, while sentences are still all-Latin: Romanized Hindi + Indian names -> Devanagari
    text = hinglish.convert(text)

    # Currency / amount expressions (before ABBR so "Rs 500" reads the amount first).
    text = _RUPEE_RE.sub(_rupee, text)
    text = _RS_RE.sub(_currency, text)
    text = _INR_HEAD_RE.sub(_currency, text)
    text = _INR_TAIL_RE.sub(_currency, text)
    text = _DOLLAR_HEAD_RE.sub(_dollar, text)
    text = _DOLLAR_TAIL_RE.sub(_dollar, text)
    text = _SCALE_RE.sub(_scale, text)
    text = _RUPEE_TAIL_RE.sub(_currency, text)

    # Units directly after a number (before ABBR so "5 kg" -> "किलो", not "किलोमीटर").
    text = _UNIT_RE.sub(_unit, text)

    # Abbreviations.
    for k, v in _ABBR.items():
        text = re.sub(rf"(?<!\w){re.escape(k)}(?!\w)", v, text)

    # Dates, times, phone numbers, ranges (specific patterns before the generic number pass).
    text = _DATE_RE.sub(_date, text)
    text = _TIME_RE.sub(_time, text)
    text = _PHONE_RE.sub(_phone, text)
    text = _RANGE_RE.sub(_range, text)
    text = _PCT_RE.sub(lambda m: f"{_number(m[1])} प्रतिशत", text)

    # Ordinals and symbol operators.
    text = _ORDINAL_RE.sub(_ordinal, text)
    text = _AMP_RE.sub(" और ", text)
    text = _AT_RE.sub(" एट ", text)
    text = _PLUS_RE.sub(" प्लस ", text)
    text = _EQ_RE.sub(" बराबर ", text)
    text = _SLASH_RE.sub(r"\1 या \2", text)

    # Generic numbers last.
    text = _NUM_RE.sub(lambda m: _number(m[0]), text)
    return re.sub(r"\s+", " ", text).strip()


def chunk(text: str, max_chars: int) -> list[str]:
    """Split on sentence ends (। . ? !), then on commas if a sentence is too long."""
    sentences = [s.strip() for s in re.split(r"(?<=[।.?!])\s+", text) if s.strip()]
    chunks, cur = [], ""
    for s in sentences:
        parts = [s] if len(s) <= max_chars else re.split(r"(?<=[,;])\s+", s)
        for p in parts:
            if len(p) > max_chars and cur:
                chunks.append(cur)
                cur = ""
            while len(p) > max_chars:  # no punctuation at all: hard split on space
                cut = p.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                chunks.append(p[:cut].strip())
                p = p[cut:].strip()
            if cur and len(cur) + len(p) + 1 > max_chars:
                chunks.append(cur)
                cur = p
            else:
                cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    return chunks