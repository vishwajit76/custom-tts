"""Word-level respelling tables, loaded once from data/*.tsv (header row + `category` and `reason` columns on every row)."""
import unicodedata
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data"
GROUPS = ("grammar", "schwa", "names", "english")  # what settings.pronunciation_rules can switch on


def _read(name: str) -> list[dict[str, str]]:
    lines = [ln for ln in (DATA / name).read_text("utf-8").splitlines() if ln and not ln.startswith("#")]
    head = lines[0].split("\t")
    rows = [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]
    for r in rows:  # a row without a category and a reason is a bug in the data, not a default
        if len(r) != 4 or not r["category"] or not r["reason"]:
            raise ValueError(f"{name}: row needs word, respelling, category, reason: {r}")
    return rows


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# (key, respelling, group); Devanagari keys are NFC (a nukta letter may arrive precomposed or as base+nukta)
HI = {_nfc(r["word"]): (r["respelling"], r["category"]) for r in _read("lexical.tsv")}
LATIN = {r["word"].lower(): (r["respelling"], g) for g, f in (("names", "names.tsv"), ("english", "english.tsv")) for r in _read(f)}
assert {g for _, g in HI.values()} <= set(GROUPS) and len(HI) and len(LATIN)


@lru_cache(maxsize=16)
def parse(setting: str) -> frozenset[str]:
    """'names,english' -> groups; 'all' -> every group; ''/'off'/'none' -> none. Unknown names are an error, not ignored."""
    s = {x.strip().lower() for x in (setting or "").split(",") if x.strip()}
    if s <= {"off", "none"}:
        return frozenset()
    if "all" in s:
        return frozenset(GROUPS)
    if not s <= set(GROUPS):
        raise ValueError(f"unknown pronunciation_rules {sorted(s - set(GROUPS))}; valid: {', '.join(GROUPS)}, all, off")
    return frozenset(s)


@lru_cache(maxsize=16)
def tables(groups: frozenset[str]) -> tuple[dict[str, str], dict[str, str]]:
    """(Devanagari word -> respelling, lowercase Latin word -> respelling), limited to the enabled groups."""
    return ({k: v for k, (v, g) in HI.items() if g in groups}, {k: v for k, (v, g) in LATIN.items() if g in groups})
