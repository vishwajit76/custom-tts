"""python single.py <engine> [n_rows]: load/cold/warm latency, TTFA, RTF, resources. Chunks are synthesized sequentially (as the app streams)."""
import sys
from common import *
import soundfile as sf

name = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
rows = corpus(n)
out = {"engine": name, "rows": n, "errors": []}
sdir = HERE / "samples" / name
sdir.mkdir(parents=True, exist_ok=True)
out["vram_baseline_mib"] = round(vram_now(), 1)
recs = []
with Sampler() as S:
    t = time.perf_counter(); e = ENGINES[name](); e.load(); out["load_s"] = round(time.perf_counter() - t, 2)
    out["vram_after_load_mib"] = round(vram_now(), 1)
    for i, r in enumerate(rows):
        try:
            ch = chunks_of(r["text"]); t0 = time.perf_counter(); wavs = []; ttfa = None
            for c in ch:
                w, sr = e.synth(c); wavs.append(w)
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
            tot = time.perf_counter() - t0; w = np.concatenate(wavs); dur = len(w) / sr
            recs.append({"id": r["id"], "i": i, "chunks": len(ch), "ttfa": ttfa, "total": tot, "dur": dur, "rtf": tot / dur if dur else None})
            if i % 5 == 0 and i < 50 and len(w):
                sf.write(sdir / f"{r['id']}.wav", w, sr)
        except Exception as ex:
            out["errors"].append(f"{r['id']}: {type(ex).__name__}: {ex}"[:300]); recs.append({"id": r["id"], "i": i, "error": True})
    if torch.cuda.is_available():
        torch.cuda.synchronize()
out.update(S.summary())
ok = [r for r in recs if not r.get("error")]
cold, warm = ok[:1], ok[1:]
f = lambda k, L: [r[k] for r in L if r.get(k) is not None]
out["active_provider"] = getattr(e, "active", "torch")
out["cold"] = cold[0] if cold else None
out["warm"] = {"n": len(warm), "ttfa_p50": pct(f("ttfa", warm), 50), "ttfa_p95": pct(f("ttfa", warm), 95), "ttfa_p99": pct(f("ttfa", warm), 99),
               "total_mean": round(st.mean(f("total", warm)), 4), "dur_mean": round(st.mean(f("dur", warm)), 3),
               "rtf_mean": round(sum(f("total", warm)) / sum(f("dur", warm)), 4), "rtf_p95": pct(f("rtf", warm), 95)} if warm else None
out["per_utt"] = recs
(HERE / "results" / f"single_{name}.json").write_text(json.dumps(out, indent=1))
print(name, json.dumps({k: v for k, v in out.items() if k != "per_utt"}, ensure_ascii=False))
