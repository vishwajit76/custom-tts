import sys, copy, json
sys.stdout.reconfigure(encoding="utf-8")
from optimizer import core
from optimizer.loop import Session
CATS = "punctuation questions commands conversational long_sentences edge_cases basic_hindi difficult_hindi".split()
eng = sys.argv[1]; variants = json.loads(sys.argv[2])   # {name: default-override dict}
s = Session()
for name, ov in variants.items():
    a = copy.deepcopy(s.active); a["default"].update(ov)
    for c in CATS: a["category"].pop(c, None)
    row = []
    for c in CATS:
        cs = [x for x in s.cases.values() if x["category"] == c]
        rs = s.ev.run(cs, eng, a, s.pron)
        m = lambda k: sum(r[k] for r in rs) / len(rs)
        cer = sum(r["metrics"]["cer"] for r in rs) / len(rs)
        row.append(f"{c[:6]} T{m('total'):.4f} P{m('prosody'):.2f} C{cer:.3f}")
    print(name, "|", " | ".join(row), flush=True)
