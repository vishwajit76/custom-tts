import sys; sys.path.insert(0, "D:/Github/custom-tts")
import sys, re, json
sys.stdout.reconfigure(encoding="utf-8")
from app.services import text_normalizer as tn
from optimizer import core
from optimizer.loop import Session
D = "शून्य एक दो तीन चार पाँच छह सात आठ नौ".split()
RUN = re.compile(r"(?:(?:%s)(?:,? |$|(?=[^\w])))+" % "|".join(D))
def regroup(n, sizes, sep):
    def f(m):
        ws = re.findall("|".join(D), m[0])
        if len(ws) < 8: return m[0]
        out, i, k = [], 0, 0
        while i < len(ws):
            sz = sizes[k % len(sizes)] if isinstance(sizes, list) else sizes; out.append(" ".join(ws[i:i+sz])); i += sz; k += 1
        return sep.join(out) + (" " if m[0].endswith(" ") else "")
    return RUN.sub(f, n)
V = {
 "base": lambda n: n,
 "hajar": lambda n: n.replace("हज़ार", "हजार"),
 "rupe": lambda n: n.replace("रुपये", "रुपए"),
 "ph_nocomma": lambda n: regroup(n, 5, " "),
 "ph_3": lambda n: regroup(n, 3, ", "),
 "ph_2": lambda n: regroup(n, 2, ", "),
 "ph_4": lambda n: regroup(n, [3,3,4], ", "),
 "date_comma": lambda n: re.sub(r"((?:%s)) (?=दो हज़ार|उन्नीस सौ)" % "|".join(tn._MONTHS), r"\1, ", n),
}
cats = sys.argv[1].split(","); names = sys.argv[2].split(","); eng = sys.argv[3] if len(sys.argv) > 3 else "goonj"
s = Session(); orig = tn.normalize
for name in names:
    tn.normalize = lambda t, rules=None, f=V[name]: f(orig(t, rules))
    row = []
    for c in cats:
        cs = [x for x in s.cases.values() if x["category"] == c]
        rs = s.ev.run(cs, eng, s.active, s.pron)
        row.append(f"{c[:8]} T{sum(r['total'] for r in rs)/len(rs):.4f} P{sum(r['pronunciation'] for r in rs)/len(rs):.3f}")
    print(name, "|", " | ".join(row), flush=True)
