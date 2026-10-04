"""Phoneme-level fixes for goonj: per-word IPA overrides + regex patterns on the espeak phoneme string (g2p_fixes.json)."""
import json
import re
from pathlib import Path

FIXES = Path(__file__).with_name("g2p_fixes.json")
_WORD = re.compile(r"[\wऀ-ॿ]")
_TAIL = re.compile(r"^(.*?)([,.?!;:…]*)$")


def load():
    return json.loads(FIXES.read_text("utf-8")) if FIXES.exists() else {"words": {}, "patterns": []}


CFG = load()


def fix(text: str, ps: str, cfg=None) -> str:
    """text = what was phonemized, ps = its espeak IPA. Words are swapped by position when counts agree
    (ponytail: any mismatch, e.g. hyphenated English, skips word overrides; patterns still apply)."""
    cfg = cfg or CFG
    words = [w.strip(" ,.?!;:।…()\"'") for w in text.split() if _WORD.search(w)]
    toks = ps.split(" ")
    if cfg["words"] and len(words) == len(toks):
        for i, w in enumerate(words):
            if w in cfg["words"]:
                body, tail = _TAIL.match(toks[i]).groups()
                toks[i] = cfg["words"][w] + tail
    ps = " ".join(toks)
    for pat, rep in cfg["patterns"]:
        ps = re.sub(pat, rep, ps)
    return ps


if __name__ == "__main__":
    c = {"words": {"कृपया": "X"}, "patterns": [[r"r\.h", "ɽ"]]}
    assert fix("कृपया साढ़े", "kɾˈɪpjˌaː sˈaːr.heː", c) == "X sˈaːɽeː"
    assert fix("जी, कृपया।", "ɟˈi, kɾˈɪpjˌaː", c) == "ɟˈi, X"
    print("ok")
