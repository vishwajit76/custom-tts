"""Evidence table for the pronunciation layer: what espeak-ng `hi` (the phonemizer the Piper voices use) produces for each
probe item, before and after app/services/pronunciation, against the expected standard (Delhi/Hindustani colloquial).

    python bench/pronunciation_probe.py                       # TSV to bench/results/pronunciation_probe_v7.tsv + summary
    python bench/pronunciation_probe.py --check               # exit 1 if the committed TSV is stale

Items: bench/pronunciation_probe_items.tsv (text, category, expected IPA, note). `expected` is the author's reading of
standard Hindi, NOT native-reviewed. "Wrong" means the BROAD form differs (see `broad`): stress, vowel length, dental vs
retroflex, a nasal vowel vs vowel + nasal, gemination and the h/ɦ, ʌ/ə spelling variants are ignored, so a flagged item
differs in syllables or segments. IPA being right does not make the audio right: a voice trained on espeak labels may have
learned to compensate (docs/pronunciation.md, Limits).
"""
import argparse
import csv
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from piper.voice import _PHONEME_BLOCK_PATTERN  # noqa: E402

from app.services import indian_english  # noqa: E402
from app.services.pronunciation import lexical, phonological  # noqa: E402
from app.services.text_normalizer import normalize  # noqa: E402

ITEMS = ROOT / "bench" / "pronunciation_probe_items.tsv"
OUT = ROOT / "bench" / "results" / "pronunciation_probe_v7.tsv"
COLUMNS = ["text", "category", "espeak_before", "espeak_after", "expected", "fixed_by", "wrong_before", "wrong_after"]

_MAP = str.maketrans({"ʌ": "ə", "ɐ": "ə", "ɦ": "h", "ɡ": "g", "ɾ": "r", "ɽ": "r", "ɻ": "r", "ɪ": "i", "ʊ": "u", "ɛ": "e", "ɔ": "o",
                      "æ": "e", "ʂ": "ʃ", "ɳ": "n", "ɲ": "n", "ŋ": "n", "ʋ": "v", "w": "v", "ʒ": "z", "c": "C", "ɟ": "J",
                      "ʈ": "t", "ɖ": "d", "ʰ": "h", "ʲ": "", "ː": "", "ˈ": "", "ˌ": "", " ": "", ".": "", "̃": "", "̪": "",
                      "͡": "", "̩": ""})


def broad(ipa: str) -> str:
    s = unicodedata.normalize("NFD", ipa).replace("\u0361", "").replace("tʃ", "c").replace("dʒ", "ɟ").translate(_MAP)
    s = s.replace("C", "tʃ").replace("J", "dʒ")
    s = re.sub(r"[nm](?=[ptkbdgfsʃzxqhjvlrmn])", "", s)  # nasal before a consonant: nasalised vowel and V+N are one thing
    s = re.sub(r"(.)\1+", r"\1", s)  # gemination
    return s


def espeak_ipa(text: str, rules: str) -> tuple[str, str]:
    """(normalized text, IPA) the way PiperEngine.synth gets them: normalize -> Indian-English marking -> Piper phonemize."""
    norm = normalize(text, rules)
    marked = indian_english.mark(norm)
    out = []
    for part in _PHONEME_BLOCK_PATTERN.split(marked):  # [[...]] blocks are raw phonemes (piper/voice.py)
        if part.startswith("[["):
            out.append(part[2:-2].strip())
        elif part.strip():
            out.append(" ".join("".join(s) for s in indian_english.espeak("hi", part)).strip())
    return norm, " ".join(out)


_PRECOMPOSED = {"ज़": "ज़", "ख़": "ख़", "ग़": "ग़", "ड़": "ड़",
                "ढ़": "ढ़", "फ़": "फ़", "क़": "क़"}


def item_text(text: str, category: str) -> str:
    text = unicodedata.normalize("NFC", text)  # base + nukta
    if category == "nukta-nfd":
        for k, v in _PRECOMPOSED.items():
            text = text.replace(k, v)
    return text


def fixed_by(text: str, norm_off: str, norm_on: str) -> str:
    if norm_off == norm_on:
        return ""
    for g in lexical.GROUPS:
        if g in ("schwa",):
            continue
        if normalize(text, g) == norm_on:
            return f"lexical:{g}"
    if normalize(text, "schwa") == norm_on:
        toks = re.findall(r"[ऀ-ॣॱ-ॿ]+", unicodedata.normalize("NFC", norm_off))
        return "lexical:schwa" if any(t in lexical.HI for t in toks) else "phonological:schwa"
    return "combined"


def run() -> list[dict]:
    rows = []
    with ITEMS.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            text = item_text(r["text"], r["category"])
            n0, before = espeak_ipa(text, "off")
            n1, after = espeak_ipa(text, "all")
            exp = broad(r["expected"])
            rows.append({"text": text, "category": r["category"], "espeak_before": before, "espeak_after": after,
                         "expected": r["expected"], "fixed_by": fixed_by(text, n0, n1),
                         "wrong_before": int(broad(before) != exp), "wrong_after": int(broad(after) != exp)})
    return rows


def render(rows: list[dict]) -> str:
    lines = ["\t".join(COLUMNS)] + ["\t".join(str(r[c]) for c in COLUMNS) for r in rows]
    return "\n".join(lines) + "\n"


def summary(rows: list[dict]) -> str:
    cat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        c = cat[r["category"]]
        c[0] += 1
        c[1] += r["wrong_before"]
        c[2] += r["wrong_after"]
    out = [f"{'category':16} {'items':>5} {'wrong_before':>12} {'wrong_after':>11}"]
    out += [f"{k:16} {v[0]:5d} {v[1]:12d} {v[2]:11d}" for k, v in cat.items()]
    t = [sum(v[i] for v in cat.values()) for i in range(3)]
    out.append(f"{'TOTAL':16} {t[0]:5d} {t[1]:12d} {t[2]:11d}")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    data = run()
    if a.check:
        sys.exit(0 if OUT.read_text("utf-8") == render(data) else 1)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(render(data), "utf-8")
    print(summary(data))
