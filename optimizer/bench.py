"""Pronunciation benchmark: tests/pronunciation_bench/<category>/cases.json (spec §6).

Expected spoken forms come from the independent rules below (own number table), never from app's normalizer, so a
normalizer regression shows up as a failure. Case: id, text, expected_normalized, category, priority (1 = critical),
optional alternatives (other acceptable spoken forms), important (words that must be heard), terms (token -> spoken), source.
"""
import csv
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "tests/pronunciation_bench"

ONES = ("शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस "
        "इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस "
        "सैंतीस अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चौवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास "
        "इक्यावन बावन तिरेपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ इकसठ बासठ तिरेसठ चौंसठ पैंसठ छियासठ सड़सठ "
        "अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी इक्यासी "
        "बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पंचानवे छियानवे "
        "सत्तानवे अट्ठानवे निन्यानवे").split()
assert len(ONES) == 100
MONTHS = "जनवरी फ़रवरी मार्च अप्रैल मई जून जुलाई अगस्त सितंबर अक्टूबर नवंबर दिसंबर".split()
EN_MONTHS = "January February March April May June July August September October November December".split()


def words(n: int) -> str:
    """Indian-system cardinal, 0 <= n < 10**9."""
    if n < 100:
        return ONES[n]
    out = []
    for unit, name in ((10**7, "करोड़"), (10**5, "लाख"), (1000, "हज़ार"), (100, "सौ")):
        if n >= unit:
            out += [words(n // unit), name]
            n %= unit
    if n:
        out.append(ONES[n])
    return " ".join(out)


def alts(n: int) -> list[str]:
    """"एक सौ"/"एक हज़ार" may be said without "एक"."""
    w = words(n)
    return [w[3:]] if w.startswith("एक सौ") or w.startswith("एक हज़ार") else []


def indian(n: int) -> str:
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    return ",".join([head, *parts, tail]) if head else ",".join([*parts, tail])


def year(y: int) -> str:
    return f"{words(y // 100)} सौ {ONES[y % 100]}" if 1100 <= y < 2000 and y % 100 else words(y)


def clock(h: int, m: int) -> str:
    """12-hour h. साढ़े/सवा/पौने conventions; 1:30 डेढ़, 2:30 ढाई."""
    nxt = h % 12 + 1
    if m == 0:
        return f"{ONES[h]} बजे"
    if m == 30:
        return {1: "डेढ़ बजे", 2: "ढाई बजे"}.get(h, f"साढ़े {ONES[h]} बजे")
    if m == 15:
        return f"सवा {ONES[h]} बजे"
    if m == 45:
        return "पौने एक बजे" if nxt == 1 else f"पौने {ONES[nxt]} बजे"
    return f"{ONES[h]} बजकर {ONES[m]} मिनट"


def case(cid, cat, text, exp, prio=2, **kw):
    return {"id": cid, "category": cat, "text": text, "expected_normalized": exp, "priority": prio, **kw}


def gen_numbers(r):
    out = []
    picks = [7, 12, 19, 25, 38, 47, 59, 66, 79, 83, 99, 100, 101, 150, 345, 999, 1000, 1500, 2026, 4567, 10000, 12345,
             99999, 100000, 125000, 250000, 512000, 999999, 1000000] + r.sample(range(2, 1000000), 21)
    tmpl = [("मेरे पास {} किताबें हैं।", "मेरे पास {} किताबें हैं।"), ("कुल {} लोग आए थे।", "कुल {} लोग आए थे।"),
            ("इस गाँव की आबादी {} है।", "इस गाँव की आबादी {} है।")]
    for i, n in enumerate(picks):
        t, e = tmpl[i % 3]
        num = indian(n) if n >= 1000 and i % 2 else str(n)
        out.append(case(f"num-{i:03d}", "numbers", t.format(num), e.format(words(n)), 1 if n in (47, 1500, 125000) else 2,
                        alternatives=[e.format(a) for a in alts(n)], important=[words(n)]))
    return out


def gen_currency(r):
    out, i = [], 0
    for n in [5, 10, 49, 99, 250, 500, 999, 1200, 2499, 25000, 125000, 300000, 750000] + r.sample(range(2, 1000000), 12):
        w = words(n)
        for fmt in (f"₹{indian(n)}", f"Rs. {indian(n)}", f"INR {n}"):
            if i % 3 == 0 or fmt.startswith("₹"):
                out.append(case(f"cur-{i:03d}", "currency", f"आपको {fmt} देने हैं।", f"आपको {w} रुपये देने हैं।",
                                1 if n in (500, 25000, 125000) else 2, alternatives=[f"आपको {a} रुपये देने हैं।" for a in alts(n)],
                                important=[w]))
            i += 1
    for rs, p in [(12, 50), (2499, 50), (99, 75), (349, 25)]:
        out.append(case(f"cur-{i:03d}", "currency", f"बिल ₹{indian(rs)}.{p} है।", f"बिल {words(rs)} रुपये {ONES[p]} पैसे है।",
                        important=[words(rs), ONES[p]]))
        i += 1
    return out


def gen_dates(r):
    out = []
    for i in range(36):
        d, m, y = r.randint(1, 28), r.randint(1, 12), r.choice([1995, 1998, 2001, 2010, 2024, 2025, 2026, 2027])
        exp = f"{ONES[d]} {MONTHS[m - 1]} {year(y)}"
        text = [f"{d:02d}/{m:02d}/{y}", f"{d} {EN_MONTHS[m - 1]} {y}", f"{y}-{m:02d}-{d:02d}", f"{d} {MONTHS[m - 1]} {y}"][i % 4]
        out.append(case(f"date-{i:03d}", "dates", f"आपकी अपॉइंटमेंट {text} को है।", f"आपकी अपॉइंटमेंट {exp} को है।",
                        1 if i < 4 else 2, important=[ONES[d], MONTHS[m - 1], year(y)]))
    return out


def gen_time(r):
    out, i = [], 0
    for h in range(1, 13):
        for m in (0, 15, 30, 45, 20):
            if (h + m) % 2 and m not in (30,):
                continue
            exp = clock(h, m)
            out.append(case(f"time-{i:03d}", "time", f"मीटिंग का समय {h}:{m:02d} है।", f"मीटिंग का समय {exp} है।",
                            1 if m == 30 else 2, important=[exp.split()[0]]))
            i += 1
    for h, ap, period in [(7, "AM", "सुबह"), (9, "AM", "सुबह"), (10, "AM", "सुबह"), (11, "AM", "सुबह"), (5, "PM", "शाम"),
                          (6, "PM", "शाम"), (8, "PM", "रात"), (9, "PM", "रात"), (10, "PM", "रात")]:
        for m in (0, 30):
            out.append(case(f"time-{i:03d}", "time", f"कॉल {h}:{m:02d} {ap} पर करें।", f"कॉल {period} {clock(h, m)} पर करें।",
                            important=[period]))
            i += 1
    return out


def gen_phone(r):
    out = []
    digit = lambda s: " ".join(ONES[int(c)] for c in s)
    for i in range(24):
        num = str(r.choice([6, 7, 8, 9])) + "".join(str(r.randint(0, 9)) for _ in range(9))
        text = [num, f"{num[:5]} {num[5:]}", f"{num[:5]}-{num[5:]}"][i % 3]
        exp = f"मेरा नंबर {digit(num)} है।"
        out.append(case(f"phone-{i:03d}", "phone_numbers", f"मेरा नंबर {text} है।", exp, 1 if i < 3 else 2,
                        alternatives=[exp.replace("शून्य", "ज़ीरो")], important=[ONES[int(c)] for c in num[:3]]))
    return out


def gen_percent(r):
    out = []
    for i, n in enumerate([1, 5, 10, 12, 18, 25, 33, 45, 50, 67, 75, 90, 99, 100]):
        out.append(case(f"pct-{i:03d}", "percent", f"ब्याज दर {n}% है।", f"ब्याज दर {words(n)} प्रतिशत है।",
                        alternatives=[f"ब्याज दर {a} प्रतिशत है।" for a in alts(n)] + [f"ब्याज दर {words(n)} परसेंट है।"],
                        important=[words(n)]))
    for j, (a, b) in enumerate([(12, 5), (7, 5), (2, 5)], len(out)):
        out.append(case(f"pct-{j:03d}", "percent", f"बढ़त {a}.{b}% रही।", f"बढ़त साढ़े {ONES[a]} प्रतिशत रही।",
                        alternatives=[f"बढ़त {ONES[a]} दशमलव {ONES[b]} प्रतिशत रही।"], important=[ONES[a]]))
    return out


# token -> spoken. Acronyms are letter-spelled; brands/terms as commonly said in Hindi.
TERMS = {"API": "ए पी आई", "AI": "ए आई", "CRM": "सी आर एम", "SIP": "एस आई पी", "TTS": "टी टी एस", "OTP": "ओ टी पी",
         "URL": "यू आर एल", "HTTP": "एच टी टी पी", "PNR": "पी एन आर", "ATM": "ए टी एम", "PAN": "पैन", "KYC": "के वाई सी",
         "EMI": "ई एम आई", "GST": "जी एस टी", "UPI": "यू पी आई", "SMS": "एस एम एस", "PDF": "पी डी एफ", "IT": "आई टी",
         "HDFC": "एच डी एफ सी", "SBI": "एस बी आई", "ICICI": "आई सी आई सी आई", "IRCTC": "आई आर सी टी सी"}
WORDS = {"WhatsApp": "व्हाट्सऐप", "Google": "गूगल", "Paytm": "पेटीएम", "Amazon": "अमेज़न", "Flipkart": "फ्लिपकार्ट",
         "YouTube": "यूट्यूब", "Facebook": "फेसबुक", "Instagram": "इंस्टाग्राम", "laptop": "लैपटॉप", "software": "सॉफ्टवेयर",
         "password": "पासवर्ड", "email": "ईमेल", "server": "सर्वर", "update": "अपडेट", "download": "डाउनलोड",
         "account": "अकाउंट", "meeting": "मीटिंग", "online": "ऑनलाइन", "website": "वेबसाइट", "mobile": "मोबाइल"}


def gen_terms():
    out = []
    frames = ["कृपया {} की जानकारी दीजिए।", "आपका {} अपडेट हो गया है।", "{} से जुड़ी समस्या बताइए।"]
    for i, (k, v) in enumerate(TERMS.items()):
        f = frames[i % 3]
        out.append(case(f"acr-{i:03d}", "acronyms", f.format(k), f.format(v),
                        1 if k in ("OTP", "API", "UPI") else 2, important=[v], terms={k: v},
                        alternatives=[f.format(v.replace(" ", ""))]))
    for i, (k, v) in enumerate(WORDS.items()):
        f = frames[i % 3]
        out.append(case(f"en-{i:03d}", "english_words", f.format(k), f.format(v), 1 if k == "WhatsApp" else 2,
                        important=[v], terms={k: v}))
    return out


DIFFICULT = ["विश्वजीत", "प्रतिष्ठा", "उज्ज्वल", "क्षत्रिय", "स्वास्थ्य", "संक्षिप्त", "द्वारा", "श्रृंखला", "ऋषिकेश", "कृतज्ञता",
             "अर्थव्यवस्था", "सम्प्रेषण", "विद्यार्थी", "आत्मनिर्भर", "प्रज्ज्वलित", "ज्योत्स्ना", "उत्कृष्ट", "वृद्धि",
             "स्पष्टीकरण", "मुख्यमंत्री", "आश्चर्य", "दृष्टिकोण", "त्र्यंबकेश्वर", "स्वतंत्रता", "धृतराष्ट्र"]
NAMES = {"Vishvajeet": "विश्वजीत", "Rahul": "राहुल", "Priya": "प्रिया", "Aishwarya": "ऐश्वर्या", "Siddharth": "सिद्धार्थ",
         "Shreya": "श्रेया", "Abhishek": "अभिषेक", "Kshitij": "क्षितिज", "Gyanendra": "ज्ञानेंद्र", "Pradyumn": "प्रद्युम्न",
         "Chauhan": "चौहान", "Chaturvedi": "चतुर्वेदी", "Tripathi": "त्रिपाठी", "Bhattacharya": "भट्टाचार्य", "Ananya": "अनन्या"}
PLACES = {"Thiruvananthapuram": "तिरुवनंतपुरम", "Visakhapatnam": "विशाखापत्तनम", "Bhubaneswar": "भुवनेश्वर",
          "Gandhinagar": "गांधीनगर", "Vadodara": "वडोदरा", "Prayagraj": "प्रयागराज", "Guwahati": "गुवाहाटी",
          "Puducherry": "पुडुचेरी", "Coimbatore": "कोयंबटूर", "Aurangabad": "औरंगाबाद"}


def gen_curated():
    out = []
    for i, w in enumerate(DIFFICULT):
        t = f"{w} शब्द को ध्यान से पढ़िए।"
        out.append(case(f"dif-{i:03d}", "difficult_hindi", t, t, 1 if w == "विश्वजीत" else 2, important=[w]))
    for i, (k, v) in enumerate(NAMES.items()):
        out.append(case(f"name-{i:03d}", "names", f"मेरा नाम {k} है।", f"मेरा नाम {v} है।", 1 if i < 2 else 2, important=[v], terms={k: v}))
        out.append(case(f"name-d{i:03d}", "names", f"{v} जी, आपका स्वागत है।", f"{v} जी, आपका स्वागत है।", important=[v]))
    for i, (k, v) in enumerate(PLACES.items()):
        out.append(case(f"pn-{i:03d}", "proper_nouns", f"मैं {k} से बोल रही हूँ।", f"मैं {v} से बोल रही हूँ।", important=[v], terms={k: v}))
        out.append(case(f"pn-d{i:03d}", "proper_nouns", f"अगली ट्रेन {v} जाएगी।", f"अगली ट्रेन {v} जाएगी।", important=[v]))
    abbr = [("डॉ. मेहता आज नहीं आएँगे।", "डॉक्टर मेहता आज नहीं आएँगे।"), ("Dr. Sharma से मिलिए।", "डॉक्टर शर्मा से मिलिए।"),
            ("Mr. Gupta लाइन पर हैं।", "मिस्टर गुप्ता लाइन पर हैं।"), ("5 kg चावल भेजिए।", "पाँच किलो चावल भेजिए।"),
            ("दूरी 12 km है।", "दूरी बारह किलोमीटर है।"), ("कृपया आदि बातें नोट करें, जैसे नाम, पता इत्यादि।", "कृपया आदि बातें नोट करें, जैसे नाम, पता इत्यादि।"),
            ("नं. 45 वाली फ़ाइल दीजिए।", "नंबर पैंतालीस वाली फ़ाइल दीजिए।"), ("लगभग 2 km दूर है।", "लगभग दो किलोमीटर दूर है।")]
    out += [case(f"abbr-{i:03d}", "abbreviations", t, e) for i, (t, e) in enumerate(abbr)]
    punct = ["रुकिए... मैं देखती हूँ।", "हाँ, बिल्कुल; आप सही हैं।", "अरे वाह! यह तो बहुत अच्छा है!", "पहला, दूसरा, और तीसरा विकल्प देखिए।",
             "उसने कहा, \"मैं कल आऊँगा।\"", "नाम (पूरा) लिखिए।", "हाँ — बिल्कुल सही।", "ठीक है, ठीक है, मैं समझ गई।"]
    out += [case(f"pun-{i:03d}", "punctuation", t, t) for i, t in enumerate(punct)]
    cmds = ["कृपया अपना पासवर्ड बदलिए।", "दरवाज़ा बंद कर दीजिए।", "अभी लाइन पर बने रहिए।", "एक दबाइए।", "अपना नाम बोलिए।",
            "बीप के बाद संदेश छोड़िए।", "बत्ती बुझा दो।", "यहाँ हस्ताक्षर कीजिए।"]
    out += [case(f"cmd-{i:03d}", "commands", t, t) for i, t in enumerate(cmds)]
    conv = ["अच्छा, तो फिर कल मिलते हैं।", "हम्म, यह थोड़ा मुश्किल लग रहा है।", "अरे, आप तो बहुत जल्दी आ गए!", "जी हाँ, मैं सुन रही हूँ।",
            "कोई बात नहीं, हो जाता है।", "सच में? मुझे तो पता ही नहीं था।", "चलिए, फिर मिलते हैं।", "बस दो मिनट रुकिए, अभी बताती हूँ।"]
    out += [case(f"conv-{i:03d}", "conversational", t, t) for i, t in enumerate(conv)]
    edge = [("", None), ("।", None), ("A", "ए"), ("ok", "ओके"), ("हाँ", "हाँ"), ("100%", "सौ प्रतिशत"), ("1", "एक"),
            ("नमस्ते!!!", "नमस्ते!!!"), ("call करें ASAP", "call करें ए एस ए पी")]
    out += [case(f"edge-{i:03d}", "edge_cases", t, e, alternatives=["एक सौ प्रतिशत"] if t == "100%" else [])
            for i, (t, e) in enumerate(edge) if e]
    return out


def from_tsv():
    out = []
    rows = list(csv.DictReader(open(ROOT / "bench/corpus/hi_eval_v2.tsv", encoding="utf-8"), delimiter="\t"))
    pick = {"hindi": ("basic_hindi", 40), "questions": ("questions", 22), "expressive": ("conversational", 10),
            "pronunciation": ("difficult_hindi", 32)}
    seen = {}
    for r in rows:
        if r["category"] in pick and not re.search("[A-Za-z0-9]", r["text"]):  # target = text only when nothing to normalize
            cat, n = pick[r["category"]]
            if seen.get(r["category"], 0) < n:
                seen[r["category"]] = seen.get(r["category"], 0) + 1
                out.append(case(r["id"], cat, r["text"], r["text"], 2, source="hi_eval_v2"))
        elif r["category"] == "hinglish" and re.search("[ऀ-ॿ]", r["text"]):
            # mixed-script rows: English loans stay Latin by policy, so the text-level target is the text itself
            out.append(case(r["id"], "hinglish", r["text"], r["text"], 2, source="hi_eval_v2"))
    for r in csv.DictReader(open(ROOT / "bench/corpus/hi_eval_v3.tsv", encoding="utf-8"), delimiter="\t"):
        if r["category"] == "calling_agent" and not re.search("[A-Za-z0-9]", r["text"]) and sum(c.get("source") == "hi_eval_v3:calling_agent" for c in out) < 40:
            out.append(case(r["id"], "conversational", r["text"], r["text"], 2, source="hi_eval_v3:calling_agent"))
        if r["category"] == "long" and sum(c["category"] == "long_sentences" for c in out) < 20:
            out.append(case(r["id"], "long_sentences", r["text"], r["text"], 3, source="hi_eval_v3"))
    # roman Hindi with reviewed-looking Devanagari targets (machine-drafted; priority 3 until native review)
    for r in csv.DictReader(open(ROOT / "tests/data/pronunciation_corpus.tsv", encoding="utf-8"), delimiter="\t"):
        if r["category"] in ("roman_hindi", "names_places"):
            cat = "hinglish" if r["category"] == "roman_hindi" else "names"
            out.append(case(r["id"], cat, r["input"], r["expected_normalized"], 3, source="pronunciation_corpus"))
    return out


def generate() -> dict[str, list]:
    r = random.Random(20261004)
    cases = (gen_numbers(r) + gen_currency(r) + gen_dates(r) + gen_time(r) + gen_phone(r) + gen_percent(r) + gen_terms()
             + gen_curated() + from_tsv())
    by = {}
    for c in cases:
        by.setdefault(c["category"], []).append(c)
    return by


def write():
    for cat, cs in generate().items():
        d = BENCH / cat
        d.mkdir(parents=True, exist_ok=True)
        (d / "cases.json").write_text(json.dumps(cs, ensure_ascii=False, indent=1), encoding="utf-8")
    return load()


def load(category=None, test=None) -> list[dict]:
    out = []
    for f in sorted(BENCH.glob("*/cases.json")):
        out += json.loads(f.read_text(encoding="utf-8"))
    return [c for c in out if (not category or c["category"] == category) and (not test or c["id"] == test)]


if __name__ == "__main__":
    cs = write()
    from collections import Counter
    print(len(cs), dict(Counter(c["category"] for c in cs)))
