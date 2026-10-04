"""python concurrency.py <engine> [Ns comma list]. N simulated calling agents share ONE loaded model behind a FIFO queue
(engine.slots worker threads). Each agent says UTT sentences; text arrives chunk by chunk every ARRIVAL s (LLM streaming);
a chunk is submitted when it has arrived AND the previous chunk of that sentence is done (look-ahead of one, like the app).
Playback clock: audio starts when chunk 0 is ready; a gap = next chunk ready after the previous chunk's audio ended."""
import sys, queue
from common import *

ARRIVAL, UTT, DEADLINE = 0.25, 4, 420.0
name = sys.argv[1]
Ns = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4,6,8,12,16").split(",")]
rows = corpus(50)
jobs = {i: chunks_of(r["text"]) for i, r in enumerate(rows)}

e = ENGINES[name](); t = time.perf_counter(); e.load(); load_s = time.perf_counter() - t
e.synth(chunks_of(rows[0]["text"])[0])  # warm-up (cold latency is measured in single.py)
q = queue.Queue()


def gpu_worker():
    while True:
        job = q.get()
        if job is None:
            return
        text, res, ev = job
        res["start"] = time.perf_counter()
        try:
            res["wav"], res["sr"] = e.synth(text)
        except Exception as ex:
            res["err"] = f"{type(ex).__name__}: {ex}"[:200]
        res["end"] = time.perf_counter(); ev.set()


workers = [threading.Thread(target=gpu_worker, daemon=True) for _ in range(e.slots)]
[w.start() for w in workers]


def agent(w, N, t_end, out):
    for u in range(UTT):
        ch = jobs[(w * UTT + u) % len(jobs)]
        t_sub0 = time.perf_counter(); prev_done = t_sub0
        ready, durs, waits, comp, failed = [], [], [], 0.0, False
        for k, c in enumerate(ch):
            arrive = t_sub0 + k * ARRIVAL
            time.sleep(max(0, max(arrive, prev_done) - time.perf_counter()))
            res, ev = {}, threading.Event(); sub = time.perf_counter(); q.put((c, res, ev))
            if not ev.wait(max(1, t_end - time.perf_counter())) or "err" in res:
                failed = True; out["fail"].append(res.get("err", "timeout")); break
            prev_done = res["end"]; waits.append(res["start"] - sub); comp += res["end"] - res["start"]
            ready.append(res["end"]); durs.append(len(res["wav"]) / res["sr"])
        if failed or not durs:
            out["fail_utt"] += 1; continue
        gap, end_clock = 0.0, ready[0] + durs[0]
        for r, d in zip(ready[1:], durs[1:]):
            if r > end_clock:
                gap += r - end_clock; end_clock = r + d
            else:
                end_clock += d
        out["utt"].append({"ttfa": ready[0] - t_sub0, "wait": waits, "comp": comp, "dur": sum(durs), "e2e": ready[-1] - t_sub0, "gap_s": gap,
                           "underrun": gap > 0.05})


results = []
for N in Ns:
    out = {"utt": [], "fail": [], "fail_utt": 0}
    t_end = time.perf_counter() + DEADLINE
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    with Sampler() as S:
        t0 = time.perf_counter()
        ths = [threading.Thread(target=agent, args=(w, N, t_end, out)) for w in range(N)]
        [x.start() for x in ths]; [x.join() for x in ths]
        wall = time.perf_counter() - t0
    U = out["utt"]
    row = {"engine": name, "N": N, "wall_s": round(wall, 1), "utterances_ok": len(U), "utterances_failed": out["fail_utt"], "fail_msgs": list(set(out["fail"]))[:3],
           "ttfa_p50": pct([u["ttfa"] for u in U], 50), "ttfa_p95": pct([u["ttfa"] for u in U], 95), "ttfa_p99": pct([u["ttfa"] for u in U], 99),
           "rtf_compute": round(sum(u["comp"] for u in U) / max(1e-9, sum(u["dur"] for u in U)), 4),
           "rtf_e2e_mean": round(float(np.mean([u["e2e"] / u["dur"] for u in U])), 4) if U else None,
           "rtf_e2e_p95": pct([u["e2e"] / u["dur"] for u in U], 95),
           "queue_wait_mean": round(float(np.mean([x for u in U for x in u["wait"]])), 4) if U else None,
           "queue_wait_p95": pct([x for u in U for x in u["wait"]], 95),
           "gap_total_s": round(sum(u["gap_s"] for u in U), 2), "utt_with_underrun": sum(u["underrun"] for u in U),
           "audio_throughput_x": round(sum(u["dur"] for u in U) / wall, 2), **S.summary()}
    row["REALTIME"] = bool(U and row["rtf_e2e_mean"] < 0.80 and not out["fail_utt"])
    results.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    (HERE / "results" / f"conc_{name}.json").write_text(json.dumps({"engine": name, "load_s": round(load_s, 2), "arrival_s": ARRIVAL, "utt_per_agent": UTT, "runs": results}, indent=1))
    if out["fail_utt"] or (row["rtf_e2e_mean"] or 9) > 3:  # saturated: higher N adds nothing
        print("saturated/failing; stopping ladder", flush=True); break
for _ in workers:
    q.put(None)
