"""Agent D harness: python d_exp.py variants.json  -> per-variant CER delta on bench cases whose phonemes change (goonj, seed 0)."""
import json, sys, statistics as st
sys.path.insert(0, ".")
from optimizer import bench, core, evaluators as ev, g2p
from optimizer.engines import make
from app.services import kokoro_engine

active = core.load_json(core.ACTIVE, core.DEFAULT_ACTIVE); pron = core.load_json(core.PRON, {})
cases = bench.load()
fr = {c["id"]: core.frontend(c["text"], c["category"], active, pron) for c in cases}
import os
DRY = os.environ.get("DRY")
eng = None; asr = None
EMPTY = {"words": {}, "patterns": []}
memo = {}

def run(cfg, cs):
    global eng, asr
    if eng is None: eng = make("goonj"); eng.ensure(); asr = ev.ASREvaluator(batch_size=4)
    g2p.CFG = cfg; out = {}
    todo = {}
    for c in cs:
        norm, ch, p = fr[c["id"]]
        k = (json.dumps(cfg, sort_keys=True, ensure_ascii=False), c["id"])
        if k not in memo: todo[k] = (c, ch, p)
    wavs = [(eng.synth(ch, p["speed"], p["pause_ms"]), k) for k, (c, ch, p) in todo.items()]
    hyps = asr.transcribe([w for w, _ in [(a[0], 0) for a, _ in wavs]]) if False else asr.transcribe([a for a, _ in wavs])
    for ((w, sr), k), h in zip(wavs, hyps):
        memo[k] = ev.asr_metrics(h, todo[k][0])
    return {c["id"]: memo[(json.dumps(cfg, sort_keys=True, ensure_ascii=False), c["id"])] for c in cs}

def phon(cfg, c):
    g2p.CFG = cfg
    return [g2p.fix(t, kokoro_engine.phonemes(t), cfg) for t in fr[c["id"]][1]]

variants = json.load(open(sys.argv[1], encoding="utf-8"))
base_ph = {c["id"]: phon(EMPTY, c) for c in cases if c["id"] in fr and fr[c["id"]][1]}
for name, cfg in variants.items():
    cfg = {"words": cfg.get("words", {}), "patterns": cfg.get("patterns", [])}
    aff = [c for c in cases if c["id"] in base_ph and phon(cfg, c) != base_ph[c["id"]]]
    if not aff: print(name, "no affected cases", flush=True); continue
    print(f"-- {name}: affected {len(aff)}", flush=True)
    if DRY: continue
    b, n = run(EMPTY, aff), run(cfg, aff)
    d = [n[c["id"]]["cer"] - b[c["id"]]["cer"] for c in aff]
    print(f"== {name}: affected {len(aff)} meanCER {st.mean(b[c['id']]['cer'] for c in aff):.3f}->{st.mean(n[c['id']]['cer'] for c in aff):.3f} "
          f"better {sum(x < -1e-9 for x in d)} worse {sum(x > 1e-9 for x in d)}", flush=True)
    for c, x in sorted(zip(aff, d), key=lambda t: t[1])[:: max(1, len(aff) // 12)]:
        print(f"  {c['id']:10} {b[c['id']]['cer']:.2f}->{n[c['id']]['cer']:.2f} {c['text'][:30]} | {b[c['id']]['hyp'][:30]} => {n[c['id']]['hyp'][:30]}")
