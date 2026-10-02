"""Persona grammar: make a Hindi script's first-person verbs agree with the speaker's grammatical gender.

Hindi verbs, participles and a few predicate adjectives agree with the SPEAKER's gender (करता/करती हूँ, गया/गई, करूँगा/
करूँगी). The voice is a woman or a man, so a script written for the other gender sounds wrong. `apply()` rewrites
only what it can prove is the speaker's own agreement; everything else is returned byte-for-byte.

Opt-in and speaker-authored text only. Never call it on text a user wrote (a user's "मैं जाता हूँ" is the user's).

Model (rule-based, one pass, clause-bounded):
  1. Tokenise. Quoted spans (straight/curly/guillemets/backticks) are opaque and end the clause around them.
  2. Split into clauses at sentence/clause punctuation and at conjunction/subordinator words (और, लेकिन, कि, जब, ...).
  3. A clause is skipped when it is reported speech: a non-first-person reporting verb (उसने कहा, वह कहता है, he said)
     blocks its comma-neighbours; the clause after कि is eligible only if the clause before it is first person.
  4. Anchors inside an eligible clause: हूँ/हूं, the 1sg future (-ूँगा/-ऊँगा/...), and था/थी when the clause has an
     explicit nominative मैं (no ने, no competing subject pronoun). Each anchor flips itself (future, था/थी) and
     the chain of agreeing words directly to its left (करता/सकता/चाहता, रहा, चुका, गया, passive participle + गया,
     and predicate adjectives such as अच्छा/नया/थका when they touch the auxiliary).
  5. With an explicit मैं, clause-final perfectives (मैं घर गया) and आपका/आपकी + role noun (सहायक, असिस्टेंट) flip too.

Not handled on purpose (precision over recall): हम (plural or royal, may include other genders, so no safe flip),
आप/तुम agreement, ते/ती plural/honorific forms, subject ellipsis across conjunctions (मैं आया और सो गया leaves
the second clause), मैंने (ergative: the verb agrees with the object), मुझे-constructions (मुझे पता है, मुझे जाना
है: the verb agrees with something else), nouns/names (मैं लड़का हूँ, मैं गीता हूँ). Hindi grammar does not inflect
for age: `age_group` changes no rewrite; it is voice-selection / LLM-prompt metadata only.
"""
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

Gender = Literal["male", "female", "neutral"]
AgeGroup = Literal["young_adult", "adult", "mature"]


@pydantic_dataclass(frozen=True)
class Persona:
    """The speaking persona. Only `gender` drives rewrites; `persona` (role label) and `age_group` are metadata."""
    gender: Gender
    persona: str = Field("assistant", max_length=64, pattern=r"^[^\r\n]*$")
    age_group: AgeGroup = "adult"


@dataclass(frozen=True)
class Rewrite:
    start: int
    end: int
    old: str
    new: str
    rule: str  # ta | aspect | result | perfective | passive | future | aux | possessive


@dataclass(frozen=True)
class Result:
    text: str
    rewrites: tuple[Rewrite, ...]


_GENDERS = {"female": "female", "f": "female", "woman": "female", "feminine": "female",
            "male": "male", "m": "male", "man": "male", "masculine": "male"}


def normalize_gender(raw: str | None) -> Gender:
    """'F'/'female'/'woman' -> female, 'M'/'male'/'man' -> male, anything else -> neutral (no rewrite)."""
    return _GENDERS.get((raw or "").strip().lower(), "neutral")  # type: ignore[return-value]


def llm_hint(p: Persona) -> str:
    """Hindi system-prompt sentence telling the LLM which gendered first-person forms to write ('' for neutral)."""
    if p.gender == "neutral":
        return ""
    role = f" (भूमिका: {p.persona})" if p.persona != "assistant" else ""
    if p.gender == "female":
        return (f"आप एक महिला आवाज़ हैं{role}। अपने बारे में बोलते समय हमेशा स्त्रीलिंग क्रिया-रूप लिखें: "
                "करती हूँ, रही हूँ, सकती हूँ, गई थी, करूँगी; पुल्लिंग रूप (करता हूँ, करूँगा) कभी नहीं।")
    return (f"आप एक पुरुष आवाज़ हैं{role}। अपने बारे में बोलते समय हमेशा पुल्लिंग क्रिया-रूप लिखें: "
            "करता हूँ, रहा हूँ, सकता हूँ, गया था, करूँगा; स्त्रीलिंग रूप (करती हूँ, करूँगी) कभी नहीं।")


# --- vocabulary ---------------------------------------------------------------------------------------------------
_ME = frozenset({"मैं", "मै"})
_ERG = frozenset({"मैंने", "मैने"})
_OBL1 = frozenset({"मुझे", "मुझको", "मुझसे", "मेरा", "मेरी", "मेरे"})  # first-person evidence for the कि rule only
_HOON = frozenset({"हूँ", "हूं"})
_THA = {"था": "थी", "थी": "था"}
_FUT = re.compile(r"^.+[ूऊुउ][ँं]ग[ाी]$")  # करूँगा, जाऊँगा, दूँगा, करूंगी, जाउंगा: 1sg future only
# subject pronouns that make a clause "someone else's"; आप/तुम/हम are subjects only when no postposition follows
_OTHER = frozenset({"वह", "वो", "वे", "उसने", "उन्होंने", "उन्होने", "इसने", "इन्होंने", "आप", "आपने", "तुम", "तुमने",
                    "तू", "तूने", "हम", "हमने", "कोई", "कौन"})
_POLITE = frozenset({"आप", "तुम", "तू", "हम"})
_POSTP = frozenset({"को", "से", "के", "की", "का", "में", "पर", "तक"})
_PARTICLES = frozenset({"नहीं", "न", "भी", "ही", "तो", "अब", "अभी", "कभी", "हमेशा", "बिल्कुल", "ज़रूर", "सिर्फ़", "केवल"})
_BOUND = {w: "c" for w in ("और", "तथा", "व", "या", "लेकिन", "मगर", "परंतु", "किंतु", "बल्कि", "क्योंकि", "इसलिए", "ताकि",
                           "जबकि", "जब", "तब", "अगर", "यदि", "चूँकि", "वरना", "वर्ना", "फिर", "जो")} | {"कि": "k"}
_REP_PAST = frozenset({"कहा", "बोला", "बोली", "बोले", "बोलीं", "बताया", "बताई", "पूछा", "पूछी", "पूछे", "चिल्लाया",
                       "चिल्लाई"})
_REP_TA = frozenset({"कहता", "कहती", "कहते", "बोलता", "बोलती", "बोलते", "बताता", "बताती", "बताते", "पूछता", "पूछती",
                     "पूछते"})
_REP_STEM = frozenset({"कह", "बोल", "बता", "पूछ"})
_REP_EN = frozenset({"said", "says", "asked", "replied", "told", "added", "wrote"})
_REL = frozenset({"जैसा", "जैसे", "जैसी", "जितना"})  # "जैसा आपने कहा, मैं ..." is not a quote introduction
_AUX3 = frozenset({"है", "हैं", "था", "थी", "थे", "हो", "होगा", "होगी", "होंगे"})
_ROOT_OK = frozenset({"आ", "जा", "खा", "पा", "छा", "ला", "सो", "हो", "रो", "खो", "धो", "ले", "दे", "पी", "जी", "सी"})
_POSS = {"आपका": "आपकी", "तुम्हारा": "तुम्हारी", "तेरा": "तेरी"}
_POSS_F = {f: m for m, f in _POSS.items()}
_ROLE = frozenset({"सहायक", "असिस्टेंट", "असिस्टैंट", "एजेंट", "साथी", "दोस्त", "मित्र", "सलाहकार", "मददगार", "गाइड", "हेल्पर",
                   "प्रतिनिधि", "assistant", "agent", "helper", "guide", "companion", "friend", "advisor"})
_ROLE_MOD = frozenset({"वर्चुअल", "निजी", "डिजिटल", "खास", "ख़ास", "भरोसेमंद", "ai", "virtual", "digital", "personal", "smart"})

# word shapes that look like a -ता/-ती verb form but are nouns or names; the shape rules cover most, this lists the rest
_NOT_TA = frozenset({"पता", "लापता", "नाता", "पिता", "माता", "नेता", "दाता", "देवता", "जनता", "ममता", "समता", "क्षमता",
                     "ज्ञाता", "विधाता", "भ्राता", "त्राता", "निर्माता", "श्रोता", "कर्ता", "वक्ता", "चिंता", "गीता", "सीता",
                     "रीता", "लता", "भारती", "आरती", "सती", "मालती", "खेती", "गिनती", "सरस्वती", "पार्वती"})
_TA_EXC = frozenset({"देता", "लेता", "देती", "लेती", "पीता", "जीता", "पीती", "जीती", "सीती"})  # real verbs with े/ी stems
_VOWEL = re.compile("[अ-औा-ौ]")
_CONS = re.compile("[क-हक़-य़]")


def _ta_ok(w: str) -> bool:
    """w ends in ता/ती: is it a verb form (करता, जाती) rather than a noun or name (रास्ता, भारती, सविता)?"""
    core = w[:-2]
    if not core or w in _NOT_TA:
        return False
    if w in _TA_EXC:
        return True
    if core[-1] in "्ंँिुेी" or w.endswith(("वती", "मती")):
        return False
    return bool(_VOWEL.search(core)) or len(_CONS.findall(core)) > 1


def _fem(m: str) -> str:
    """Masculine perfective/adjective form -> feminine: गया->गई, हुआ->हुई, किया->की, बड़ा->बड़ी, नया->नई."""
    if m.endswith("िया"):
        return m[:-3] + "ी"
    if m.endswith("या"):
        return m[:-2] + "ई"
    if m.endswith("आ"):
        return m[:-1] + "ई"
    return m[:-1] + "ी"


def _lex(words: str, table: dict) -> None:
    for m in words.split():
        f = _fem(m)
        table[m] = (m, f)
        table[f] = (m, f)
        if f.endswith("ई") and m.endswith("या"):
            table[f[:-1] + "यी"] = (m, f)  # गयी, आयी


# chain-licensed anywhere in the chain; fixed forms only (गयी/हुयी are the common alternate spellings)
_ASPECT: dict[str, tuple[str, str]] = {}
_lex("रहा चुका वाला गया हुआ आया", _ASPECT)
_ASPECT["हुयी"] = ("हुआ", "हुई")
# predicate adjectives and resting participles: only when they touch the auxiliary (so "अच्छा काम करता हूँ" is untouched)
_RESULT: dict[str, tuple[str, str]] = {}
_lex("अच्छा बुरा बड़ा छोटा नया पुराना अकेला भूखा प्यासा सच्चा झूठा सीधा मोटा पतला लंबा बूढ़ा भोला शर्मीला थका डरा बैठा खड़ा "
     "लेटा सोया फँसा उलझा लगा जुड़ा भरा खोया जागा उठा घबराया रुका पड़ा बँधा ठहरा लाया भूला", _RESULT)
# clause-final perfectives with an explicit मैं (मैं घर गया); clause-final is required because a bare root looks the same
_PERF: dict[str, tuple[str, str]] = {}
_lex("गया आया सोया रोया निकला रुका ठहरा उतरा घबराया मुस्कुराया शर्माया पछताया गुस्साया पहुँचा लौटा चला बैठा उठा पड़ा हँसा गिरा "
     "जागा बना", _PERF)
_AUX_LIKE = _HOON | frozenset({"था", "थी", "हुआ", "हुई", "हुयी", "रहता", "रहती", "रहा", "रही", "रह", "महसूस"})


def _part(p: str) -> tuple[str, str] | None:
    """Open-class perfective participle (masc, fem) for passives (बनाया गया हूँ), or None if p is not one."""
    if len(p) < 3 or p in _ROOT_OK:
        return None
    if p.endswith("ी") or p.endswith("ई"):
        if p in ("की", "दी", "ली", "पी"):
            return None  # also the genitive की / ambiguous: only the masculine side is rewritten
        return (p[:-1] + "या", p) if p.endswith("ई") else (p[:-1] + "ा", p)
    if p.endswith("ा"):
        return p, _fem(p)
    return None


def _pair(w: str, nxt: str) -> tuple[str, str] | None:
    """(masculine, feminine) of an agreeing chain word, or None. `nxt` is the word to its right in the chain."""
    if w.endswith("ता") and _ta_ok(w):
        return w, w[:-1] + "ी"
    if w.endswith("ती") and _ta_ok(w):
        return w[:-1] + "ा", w
    p = _ASPECT.get(w)
    if p is None and (nxt in _AUX_LIKE or _FUT.match(nxt)):
        p = _RESULT.get(w)
    return p


# --- tokeniser ----------------------------------------------------------------------------------------------------
_DEV = "ऀ-ॣ०-ॿ‌‍"
_TOK = re.compile(
    r'(?P<q>"[^"]*(?:"|\Z)|“[^”]*(?:”|\Z)|‘[^’]*(?:’|\Z)|«[^»]*(?:»|\Z)|„[^“”]*(?:[“”]|\Z)|「[^」]*(?:」|\Z)'
    r"|`[^`]*(?:`|\Z)|(?<![\w" + _DEV + r"])'[^'\n]*'(?![\w" + _DEV + r"]))"
    r"|(?P<s>[।॥!?\n…]|\.(?!\d))"
    r"|(?P<p>[,;:()\[\]|—–-])"
    r"|(?P<d>[" + _DEV + r"]+)"
    r"|(?P<l>[A-Za-z]+(?:['’][A-Za-z]+)?)")
_TRIGGER = re.compile("मैं|मै|हू[ँं]|[ूऊुउ][ँं]ग[ाी]|था|थी")  # cheap prefilter: nothing to anchor on -> return the text


def _split(text: str) -> list[tuple[frozenset, list]]:
    """Clauses as (boundary kinds before it, [(word, start, end, is_latin)]). Kinds: s sentence, p punct, c conj,
    k कि, q quote. Empty clauses merge their boundaries, so `, "..." ,` is one boundary set."""
    out: list[tuple[frozenset, list]] = []
    cur: list = []
    cb = {"s"}
    for m in _TOK.finditer(text):
        k = m.lastgroup
        if k == "d" or k == "l":
            w = m.group()
            kind = _BOUND.get(w) if k == "d" else None
            if kind is None:
                cur.append((w, m.start(), m.end(), k == "l"))
                continue
        else:
            kind = k
        if cur:
            out.append((frozenset(cb), cur))
            cur, cb = [], {kind}
        else:
            cb.add(kind)
    if cur:
        out.append((frozenset(cb), cur))
    return out


@dataclass
class _Flags:
    me: bool = False        # nominative मैं
    erg: bool = False       # मैंने / X ने: verbs agree with the object
    fp: bool = False        # any first-person evidence (also मुझे/मेरा): lets a following कि-clause through
    anchor: bool = False    # हूँ or 1sg future
    other: bool = False     # competing subject pronoun
    rep: bool = False       # contains a non-first-person reporting verb


def _next(ws: list[str], k: int) -> str:
    while k < len(ws) and ws[k] in _PARTICLES:
        k += 1
    return ws[k] if k < len(ws) else ""


def _scan(toks: list) -> _Flags:
    ws = [t[0] for t in toks]
    f = _Flags()
    first_i = len(ws)
    rel = any(w in _REL for w in ws)
    for k, w in enumerate(ws):
        if w in _ME:
            f.me = f.fp = True
        elif w in _ERG:
            f.erg = f.fp = True
        elif w in _OBL1:
            f.fp = True
        elif w in _HOON or _FUT.match(w):
            f.anchor = f.fp = True
        elif w == "ने":
            f.erg = True
        elif w in _OTHER and not (w in _POLITE and k + 1 < len(ws) and ws[k + 1] in _POSTP):
            f.other = True
        if w in _ME or w in _ERG:
            first_i = min(first_i, k)
        if f.rep or rel or k > first_i:
            continue
        if toks[k][3]:
            f.rep = w.lower() in _REP_EN and (k == 0 or ws[k - 1].lower() != "i")
        else:
            f.rep = (w in _REP_PAST or (w in _REP_TA and _next(ws, k + 1) in _AUX3)
                     or (w in _REP_STEM and _next(ws, k + 1) in ("रहा", "रही", "रहे") and _next(ws, k + 2) in _AUX3))
    return f


def _eligible(cl: list, fl: list[_Flags]) -> list[bool]:
    ok = [not f.rep for f in fl]
    for i, f in enumerate(fl):  # a reporting clause also owns its comma-neighbours (direct speech without quotes)
        if f.rep:
            if i and cl[i][0] <= {"p"}:
                ok[i - 1] = False
            if i + 1 < len(fl) and cl[i + 1][0] <= {"p"}:
                ok[i + 1] = False
    emb = False  # previous clause was reported content that we refused: "... कि X है और मैं ..." stays untouched
    for i, (b, _) in enumerate(cl):
        if "k" in b:  # कि-clause: eligible only under a first-person, eligible matrix clause
            ok[i] = ok[i] and i > 0 and "s" not in b and ok[i - 1] and fl[i - 1].fp and not fl[i - 1].other
            emb = not ok[i]
        elif b == {"c"} and emb:
            ok[i] = False
        else:
            emb = False
    return ok


# --- rewriting ----------------------------------------------------------------------------------------------------
def _edit(toks: list, female: bool, f: _Flags, out: list[Rewrite]) -> None:
    ws = [t[0] for t in toks]
    n = len(ws)
    explicit = f.me and not f.erg and not f.other
    anchors = [k for k, w in enumerate(ws) if w in _HOON or _FUT.match(w) or (explicit and w in _THA)]
    if not anchors and not explicit:
        return
    done: set[int] = set()

    def put(k: int, new: str, rule: str) -> None:
        if new != ws[k] and k not in done:
            done.add(k)
            out.append(Rewrite(toks[k][1], toks[k][2], ws[k], new, rule))

    def pick(p: tuple[str, str]) -> str:
        return p[1] if female else p[0]

    def prev(j: int) -> int:
        while j >= 0 and ws[j] in _PARTICLES:
            j -= 1
        return j

    for a in anchors:
        w = ws[a]
        if w in _THA:
            put(a, "थी" if female else "था", "aux")
        elif w not in _HOON:
            put(a, (w[:-1] + "ी") if female and w.endswith("ा") else (w[:-1] + "ा") if not female and w.endswith("ी") else w,
                "future")
        j, nxt = prev(a - 1), w
        while j >= 0:
            c = ws[j]
            if c == "महसूस":  # "अकेला महसूस करता हूँ": the adjective agrees with the speaker
                nxt, j = c, prev(j - 1)
                continue
            p = _pair(c, nxt)
            if p is None:
                break
            put(j, pick(p), "ta" if c.endswith(("ता", "ती")) else "result" if c in _RESULT and c not in _ASPECT else "aspect")
            nxt = c
            j = prev(j - 1)
            if c in ("गया", "गई", "गयी"):  # passive: बनाया गया हूँ / बनाई गई हूँ
                q = _part(ws[j]) if j >= 0 else None
                if q is not None:
                    put(j, pick(q), "passive")
                    nxt, j = ws[j], prev(j - 1)
    if explicit:
        first = anchors[0] if anchors else n
        for k in range(first):
            w = ws[k]
            p = _PERF.get(w)
            if p is None or k in done or _next(ws, k + 1):
                continue
            if w in ("गया", "गई", "गयी"):
                j = prev(k - 1)
                q = _part(ws[j]) if j >= 0 else None
                if q is not None:  # "X-ा गया": only the motion participle चला/चली is safely the speaker's
                    if ws[j] not in ("चला", "चली"):
                        continue
                    put(j, pick(q), "perfective")
            put(k, pick(p), "perfective")
    if anchors or explicit:
        for k, w in enumerate(ws):
            poss = (_POSS if female else _POSS_F).get(w)
            if poss is None:
                continue
            for r in range(k + 1, min(k + 4, n)):
                lw = ws[r].lower()
                if lw in _ROLE:
                    put(k, poss, "possessive")
                    break
                if lw not in _ROLE_MOD:
                    break


def rewrite(text: str, p: Persona) -> Result:
    """Like apply() but also lists every rewrite (offsets into the ORIGINAL text) for tests and logs."""
    if p.gender == "neutral" or not _TRIGGER.search(text):
        return Result(text, ())
    cl = _split(text)
    fl = [_scan(t) for _, t in cl]
    ok = _eligible(cl, fl)
    hits: list[Rewrite] = []
    for (_, toks), f, good in zip(cl, fl, ok):
        if good:
            _edit(toks, p.gender == "female", f, hits)
    if not hits:
        return Result(text, ())
    hits.sort(key=lambda h: h.start)
    parts, pos = [], 0
    for h in hits:
        parts += [text[pos:h.start], h.new]
        pos = h.end
    parts.append(text[pos:])
    return Result("".join(parts), tuple(hits))


def apply(text: str, persona: Persona) -> str:
    """Return `text` with first-person Hindi agreement forced to persona.gender; neutral returns it unchanged."""
    return rewrite(text, persona).text
