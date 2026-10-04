"""Expand bench/corpus/hi_eval_v2.tsv -> hi_eval_v3.tsv (>=1000 rows) with Gemini Flash text generation.
Original 258 rows kept verbatim at top. Text-only; no audio."""
import re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import gem

SRC = gem.ROOT / "bench/corpus/hi_eval_v2.tsv"
DST = gem.ROOT / "bench/corpus/hi_eval_v3.tsv"
EMO = "neutral conversational professional happy sad angry excited empathetic serious".split()
# (id prefix, category, subcategories, instruction, rows per call, calls)
SPECS = [
    ("hi", "hindi", "statement request instruction confirmation apology", "pure Devanagari Hindi call-centre/assistant sentences (6-25 words), varied domains: banking, booking, delivery, telecom, health, utilities, education", 25, 4),
    ("hg", "hinglish", "statement instruction confirmation apology", "Hinglish; mix Devanagari with English words in Latin script (notes script=mixed) or fully romanised Hindi (notes script=roman)", 25, 4),
    ("num", "numbers", "quantity phone_otp ordinal_id percent_quantity", "sentences with digits/numerals to be read aloud (quantities, OTPs, phone numbers, IDs, percentages, decimals)", 25, 3),
    ("cur", "currency", "currency", "sentences with rupee amounts written as numerals/symbols (Rs, rupee sign, lakh, crore, paise)", 25, 2),
    ("dt", "dates", "date time", "sentences with dates and clock times in numeral form (15/10/2026, 3:45 PM, 2 Jan)", 25, 2),
    ("nm", "names", "indian_name city brand", "sentences with Indian person names, cities, brands and titles", 25, 2),
    ("ad", "addresses", "address", "sentences reading out Indian postal addresses with house no, sector, landmarks, PIN code", 25, 2),
    ("q", "questions", "yes_no_question wh_question", "Hindi questions of all kinds", 25, 3),
    ("long", "long", "paragraph", "long responses of 40-80 words with commas and multiple clauses", 15, 3),
    ("short", "short", "backchannel confirmation", "very short responses of 1-4 words (haan ji, theek hai, ek minute, ...)", 25, 2),
    ("abbr", "abbreviations", "abbreviation", "sentences containing abbreviations/acronyms (OTP, KYC, EMI, PNR, ATM, GST, Dr., Mr.)", 25, 2),
    ("call", "calling_agent", "turn", "calling-agent conversational turns (sales, support, collections, reminders); notes must contain emotion=<tag>, tag one of {emo}", 25, 8),
]
PROMPT = """Generate {n} NEW distinct evaluation sentences for a Hindi text-to-speech test set: {what}.
Subcategories to spread across: {subs}. Batch seed: {seed}. Return JSON array of
{{"subcategory":str,"text":str,"notes":str}} where notes is like "domain=banking" (domain one of banking booking delivery support appointment telecom utilities education).
Natural, realistic, no profanity, no real personal data."""
DEV = re.compile(r"[ऀ-ॿ]")
LAT = re.compile(r"[A-Za-z]")


def valid_row(cat, text):
    text = text.strip()
    if not text or "\t" in text or "\n" in text or len(text) > 600:
        return False
    if cat == "hinglish":
        return bool(DEV.search(text) or LAT.search(text))
    if cat in ("abbreviations", "calling_agent", "names", "addresses", "numbers", "currency", "dates"):
        return bool(DEV.search(text))  # Latin acronyms / digits allowed
    return bool(DEV.search(text)) and not LAT.search(text)


def norm(t):
    return re.sub(r"[\s।.,?!]+", "", t)


def merge(base_rows, new_rows):
    """Append new_rows (id,cat,sub,text,notes) skipping dupes vs base/each other and invalid text."""
    seen = {norm(r[3]) for r in base_rows}
    out = []
    for r in new_rows:
        k = norm(r[3])
        if k in seen or not valid_row(r[1], r[3]):
            continue
        seen.add(k)
        out.append(r)
    return out


def gen(job):
    pre, cat, subs, what, n, seed = job
    p = PROMPT.format(n=n, what=what.replace("{emo}", ",".join(EMO)), subs=subs, seed=seed)
    try:
        res = gem.ask_json(gem.FLASH, p, extra=str(seed))
    except Exception as e:
        print("skip", pre, seed, type(e).__name__)
        return []
    return [(pre, cat, str(o.get("subcategory", "")).strip(), str(o.get("text", "")).strip(), str(o.get("notes", "")).strip())
            for o in res if isinstance(o, dict)]


def main():
    base = [l.rstrip("\n").split("\t") for l in open(SRC, encoding="utf-8")]
    header, base = base[0], base[1:]
    jobs = [(p, c, s, w, n, i) for p, c, s, w, n, k in SPECS for i in range(k)]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(gen, jobs))
    cand = [r for rs in res for r in rs]
    cand = [r for r in cand if r[1] != "calling_agent" or any(f"emotion={e}" in r[4] for e in EMO)]
    new = merge(base, cand)
    cnt = {}
    with open(DST, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for r in base:
            f.write("\t".join(r) + "\n")
        for pre, cat, sub, text, notes in new:
            cnt[pre] = cnt.get(pre, 0) + 1
            f.write(f"{pre}3-{cnt[pre]:03d}\t{cat}\t{sub}\t{text}\t{notes}\n")
    print(len(base), "+", len(new), "=", len(base) + len(new), cnt, gem.USAGE)


if __name__ == "__main__":
    main()
