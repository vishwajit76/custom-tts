"""Offline ONNX Runtime tuning: latency + peak RSS per session-option variant (one process per variant).

Usage: python -m bench.ort_variants [model.onnx]
"""
import json
import resource
import statistics
import subprocess
import sys
import time

VARIANTS = {
    "default(t1)": dict(threads=1),
    "t2": dict(threads=2),
    "t4": dict(threads=4),
    "t1-noarena": dict(threads=1, arena=False),
    "t4-noarena": dict(threads=4, arena=False),
    "t1-nomempattern-noarena": dict(threads=1, arena=False, mem_pattern=False),
    "t4-nospin": dict(threads=4, spin=False),
    "t2-nospin": dict(threads=2, spin=False),
}


def child(model: str, threads: int, arena: bool = True, mem_pattern: bool = True, spin: bool = True, concurrency: int = 1) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    import onnxruntime as ort
    from piper import PiperVoice

    from pathlib import Path

    SENTENCES = [s for s in (Path(__file__).parent / "sentences.txt").read_text("utf-8").splitlines() if s.strip()]
    v = PiperVoice.load(model)
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    so.enable_cpu_mem_arena, so.enable_mem_pattern = arena, mem_pattern
    so.add_session_config_entry("session.intra_op.allow_spinning", "1" if spin else "0")
    v.session = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    ids = [v.phonemes_to_ids(v.phonemize(s)[0]) for s in SENTENCES]
    v.phoneme_ids_to_audio(ids[0])

    def one(i):
        t = time.perf_counter()
        a = v.phoneme_ids_to_audio(ids[i % len(ids)])
        return (time.perf_counter() - t) * 1000, len(a) / v.config.sample_rate

    with ThreadPoolExecutor(concurrency) as ex:
        t = time.perf_counter()
        res = list(ex.map(one, range(len(ids) * 3 * concurrency)))
        wall = time.perf_counter() - t
    ms = sorted(r[0] for r in res)
    return {
        "p50_ms": round(statistics.median(ms)), "p95_ms": round(ms[int(0.95 * len(ms))]),
        "throughput_x_realtime": round(sum(r[1] for r in res) / wall, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10)),
    }


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        print(json.dumps(child(sys.argv[2], **json.loads(sys.argv[3]))))
        sys.exit()
    model = sys.argv[1] if len(sys.argv) > 1 else "models/piper/hi_IN-rohan-medium.onnx"
    print("| variant | conc | p50 ms | p95 ms | x realtime | peak RSS MB |\n|---|---|---|---|---|---|")
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else VARIANTS
    for name, kw in ((n, VARIANTS[n]) for n in only):
        for conc in (1, 4):
            out = subprocess.run([sys.executable, "-m", "bench.ort_variants", "--child", model, json.dumps({**kw, "concurrency": conc})],
                                 capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
            r = json.loads(out)
            print(f"| {name} | {conc} | {r['p50_ms']} | {r['p95_ms']} | {r['throughput_x_realtime']} | {r['peak_rss_mb']} |", flush=True)
