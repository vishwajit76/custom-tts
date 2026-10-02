"""Per-token language tags for code-switched text: hi (Devanagari), hi-Latn (romanized Hindi or an Indian name), en, num, punct.

Reuses hinglish.py's own decision instead of a second heuristic: a Latin word is hi-Latn exactly when hinglish.convert()
would rewrite it. Diagnostic and test API; the hot path (apply) needs no tags because hinglish.convert has already run, so
every Latin token it sees is English or a name/brand.
"""
import re

from app.services import hinglish

_TOKEN = re.compile(r"[A-Za-z][A-Za-z']*|[ऀ-ॣॱ-ॿ]+|[0-9०-९][0-9०-९.,]*|\S")


def tag(text: str) -> list[tuple[str, str]]:
    """[(token, tag)] for every non-space token."""
    out: list[tuple[str, str]] = []
    for m in hinglish._SENTENCE.finditer(text):
        sent = m[0]
        toks = _TOKEN.findall(sent)
        # hinglish only ever rewrites Latin words in place, so the Latin words it left alone appear in order in its output
        left = iter(hinglish._LATIN.findall(hinglish._sentence(m)))
        nxt = next(left, None)
        for t in toks:
            if t[0].isascii() and t[0].isalpha():
                if nxt == t:
                    nxt = next(left, None)
                    out.append((t, "en"))
                else:
                    out.append((t, "hi-Latn"))
            elif "ऀ" <= t[0] <= "ॣ" or "ॱ" <= t[0] <= "ॿ":  # not the danda or Devanagari digits
                out.append((t, "hi"))
            elif t[0].isdigit() or "०" <= t[0] <= "९":
                out.append((t, "num"))
            else:
                out.append((t, "punct"))
    return out
