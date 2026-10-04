"""Builds REPORT.md tables from results/*.json (measured data only). Prose/recommendation is appended by hand after reading tables."""
import json
from pathlib import Path

R = Path(__file__).parent / "results"
singles = {p.stem[7:]: json.load(open(p, encoding="utf-8")) for p in sorted(R.glob("single_*.json"))}
conc = {p.stem[5:]: json.load(open(p, encoding="utf-8")) for p in sorted(R.glob("conc_*.json"))}
cer = json.load(open(R / "asr_cer.json", encoding="utf-8")) if (R / "asr_cer.json").exists() else {}
L = ["## Single-stream (first 50 rows of hi_eval_v2, all category=hindi; sentence chunks synthesized sequentially)", "",
     "| engine | provider | load s | cold TTFA s | warm TTFA p50/p95 s | warm RTF | audio s/utt | peak VRAM nvml MiB (torch MiB) | GPU util mean% | CPU% | RSS MiB | errors | spilled | CER (10 samples) |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for n, s in singles.items():
    w = s["warm"] or {}
    c = s["cold"] or {}
    L.append(f"| {n} | {s['active_provider'].replace('ExecutionProvider','')} | {s['load_s']} | {c.get('ttfa',0):.2f} | {w.get('ttfa_p50')}/{w.get('ttfa_p95')} | {w.get('rtf_mean')} | {w.get('dur_mean')} | "
             f"{s['nvml_vram_peak_mib']} ({s['torch_max_alloc_mib']}) | {s['gpu_util_mean']} | {s['cpu_pct_machine_mean']} | {s['rss_peak_mib']} | {len(s['errors'])} | {'SPILLED' if s['spilled'] else 'no'} | {cer.get(n, {}).get('cer_mean', '-')} |")
L += ["", "VRAM baseline with desktop only: %s MiB (nvml total used includes it)." % next(iter(singles.values()))["vram_baseline_mib"], "",
      "## Concurrency (N agents, 4 sentences each, simultaneous start, one shared model, FIFO queue; text arrives every 0.25 s per chunk)", "",
      "REALTIME = mean end-to-end RTF < 0.80 and no failed utterance. Gaps = playback-clock underruns.", "",
      "| engine | N | TTFA p50 | p95 | p99 | RTF compute | RTF e2e mean | queue wait mean/p95 s | underrun utt / total gap s | failed | peak VRAM nvml MiB | GPU% | spilled | REALTIME |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for n, c in conc.items():
    for r in c["runs"]:
        L.append(f"| {n} | {r['N']} | {r['ttfa_p50']} | {r['ttfa_p95']} | {r['ttfa_p99']} | {r['rtf_compute']} | {r['rtf_e2e_mean']} | {r['queue_wait_mean']}/{r['queue_wait_p95']} | "
                 f"{r['utt_with_underrun']}/{r['utterances_ok']} / {r['gap_total_s']} | {r['utterances_failed']} | {r['nvml_vram_peak_mib']} | {r['gpu_util_mean']} | {'SPILLED' if r['spilled'] else 'no'} | {'YES' if r['REALTIME'] else 'NO'} |")
(Path(__file__).parent / "tables.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
