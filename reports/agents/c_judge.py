import sys, copy, json
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "training/v8/gemini")
import audio_judge as aj, gem
from optimizer.loop import Session
NEW = {"questions": .95, "conversational": 1.05, "long_sentences": 1.05, "basic_hindi": 1.05}
s = Session(); out = {}
for c, sp in NEW.items():
    cs = [x for x in s.cases.values() if x["category"] == c][:8]
    wavs = {}
    for tag, v in (("old", 1.0), ("new", sp)):
        a = copy.deepcopy(s.active); a["category"][c] = {"speed": v}
        wavs[tag] = {r["id"]: r["wav"] for r in s.ev.run(cs, "goonj", a, s.pron)}
    w = {"old": 0, "new": 0, "tie": 0}
    for x in cs:
        first = aj.ab_order(x["id"] + c)
        p1, p2 = (wavs["new"][x["id"]], wavs["old"][x["id"]]) if first else (wavs["old"][x["id"]], wavs["new"][x["id"]])
        try:
            r = gem.ask_json(gem.PRO, aj.AB_PROMPT.format(text=x["text"], style="neutral"), parts=[aj.part(p1), aj.part(p2)], extra="ab" + aj.sha(p1) + aj.sha(p2))
        except Exception as e:
            print("fail", type(e).__name__); continue
        win = {"1": "new" if first else "old", "2": "old" if first else "new"}.get(str(r.get("winner")), "tie")
        w[win] += 1
    out[c] = w; print(c, w, flush=True)
