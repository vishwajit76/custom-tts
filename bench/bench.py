"""Reproducible TTS benchmark over the WebSocket API (what a calling client sees).

Usage: python -m bench.bench --url ws://localhost:8000/v1/audio/ws [--key K] [--concurrency 1,5,10] [--label cpu-w5t1]
       python -m bench.bench --soak 600 --concurrency 10          # long-running reliability
Writes bench/results/<label>.json and prints a markdown summary.

Metrics per request: ttfa (speak sent -> first audio frame), total, audio seconds, rtf (total/audio),
underrun (a real-time player that started at the first frame would have run dry, even with a 0 ms jitter buffer).
"""
import argparse
import asyncio
import json
import statistics
import time
import urllib.request
import uuid
from pathlib import Path

import websockets

SENTENCES = [s for s in (Path(__file__).parent / "sentences.txt").read_text("utf-8").splitlines() if s.strip()]
RESULTS = Path(__file__).parent / "results"


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


async def speak(ws, text: str, sr: int, voice: str) -> dict:
    rid = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    await ws.send(json.dumps({"type": "speak", "id": rid, "text": text, "sample_rate": sr, "voice": voice}))
    first, nbytes, underrun_ms = None, 0, 0.0
    while True:
        msg = await ws.recv()
        now = time.perf_counter()
        if isinstance(msg, bytes):
            if first is None:
                first = now
            # when this frame is needed by a player that started at `first`
            due = first + nbytes / 2 / sr
            underrun_ms += max(0.0, now - due) * 1000
            nbytes += len(msg)
            continue
        ev = json.loads(msg)
        if ev["type"] == "end":
            break
        if ev["type"] in ("error", "cancelled"):
            return {"error": ev.get("code", ev["type"]), "text_len": len(text)}
    audio_s = nbytes / 2 / sr
    total = now - t0
    return {"ttfa": first - t0, "total": total, "audio_s": audio_s, "rtf": total / audio_s, "underrun_ms": underrun_ms, "text_len": len(text)}


async def client(url: str, n: int, sr: int, voice: str, offset: int, deadline: float | None, call_sim: float) -> list[dict]:
    out = []
    async with websockets.connect(url, max_size=None) as ws:
        await asyncio.sleep(offset * 0.05)  # stagger connects
        i = 0
        while (deadline is None and i < n) or (deadline is not None and time.perf_counter() < deadline):
            r = await speak(ws, SENTENCES[(offset + i) % len(SENTENCES)], sr, voice)
            r["t"] = time.perf_counter()
            out.append(r)
            i += 1
            if call_sim and "audio_s" in r:  # real call: bot audio plays out, then the caller talks
                await asyncio.sleep(max(0.0, r["audio_s"] - (r["total"] - r["ttfa"])) + call_sim)
    return out


async def cancel_test(url: str, sr: int, voice: str, trials: int = 5) -> dict:
    """Barge-in: cancel right after first audio; time until the server acks 'cancelled'.

    Audio received between our cancel and the ack was already in flight (the server sends faster than real time),
    so clients must also flush their own playback buffer on barge-in; reported as buffered_ahead_ms."""
    lat, leaked = [], []
    long_text = " ".join(SENTENCES[-4:])
    async with websockets.connect(url, max_size=None) as ws:
        for _ in range(trials):
            await ws.send(json.dumps({"type": "speak", "id": "c", "text": long_text, "sample_rate": sr, "voice": voice}))
            while not isinstance(await ws.recv(), bytes):
                pass
            t = time.perf_counter()
            await ws.send(json.dumps({"type": "cancel"}))
            extra = 0
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    extra += len(msg)
                elif json.loads(msg)["type"] in ("cancelled", "end"):
                    break
            lat.append((time.perf_counter() - t) * 1000)
            leaked.append(extra / 2 / sr * 1000)
    return {"cancel_ack_ms_p50": statistics.median(lat), "cancel_ack_ms_max": max(lat), "buffered_ahead_ms_max": max(leaked)}


def rss_mb(http: str) -> float | None:
    try:
        for line in urllib.request.urlopen(f"{http}/metrics", timeout=5).read().decode().splitlines():
            if line.startswith("process_resident_memory_bytes"):
                return round(float(line.split()[1]) / 2**20, 1)
    except OSError:
        return None


def summarize(rs: list[dict]) -> dict:
    ok = [r for r in rs if "error" not in r]
    ttfa = [r["ttfa"] * 1000 for r in ok]
    return {
        "requests": len(rs), "errors": len(rs) - len(ok),
        "ttfa_ms_p50": pct(ttfa, 0.5), "ttfa_ms_p95": pct(ttfa, 0.95), "ttfa_ms_p99": pct(ttfa, 0.99),
        "rtf_p50": pct([r["rtf"] for r in ok], 0.5), "rtf_p95": pct([r["rtf"] for r in ok], 0.95),
        "underrun_requests": sum(r["underrun_ms"] > 0 for r in ok), "underrun_ms_max": max((r["underrun_ms"] for r in ok), default=0),
        "audio_s_total": sum(r["audio_s"] for r in ok),
    }


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://localhost:8000/v1/audio/ws")
    p.add_argument("--key", default="")
    p.add_argument("--voice", default="default")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--concurrency", default="1,5,10")
    p.add_argument("--requests", type=int, default=13, help="per client (ignored with --soak)")
    p.add_argument("--soak", type=float, default=0, help="seconds to run each concurrency level instead of --requests")
    p.add_argument("--call-sim", type=float, default=0, metavar="SECONDS",
                   help="simulate real calls: after each reply wait for its playback plus SECONDS of caller speech")
    p.add_argument("--label", default=time.strftime("%Y%m%d-%H%M%S"))
    a = p.parse_args()
    url = f"{a.url}?api_key={a.key}" if a.key else a.url
    http = a.url.replace("ws", "http", 1).split("/v1/")[0]

    await client(url, 2, a.sample_rate, a.voice, 0, None, 0)  # warm-up; run the server with CACHE_SIZE=0 for honest numbers
    report = {"label": a.label, "sample_rate": a.sample_rate, "call_sim": a.call_sim, "rss_mb_start": rss_mb(http), "levels": {}}
    for c in [int(x) for x in a.concurrency.split(",")]:
        deadline = time.perf_counter() + a.soak if a.soak else None
        t = time.perf_counter()
        res = await asyncio.gather(*[client(url, a.requests, a.sample_rate, a.voice, i, deadline, a.call_sim) for i in range(c)])
        flat = [r for rs in res for r in rs]
        s = summarize(flat)
        s["wall_s"] = time.perf_counter() - t
        s["rss_mb"] = rss_mb(http)
        if a.soak:  # drift: first vs last minute p95
            t_end = max(r["t"] for r in flat)
            s["ttfa_ms_p95_first_min"] = summarize([r for r in flat if r["t"] < t + 60])["ttfa_ms_p95"]
            s["ttfa_ms_p95_last_min"] = summarize([r for r in flat if r["t"] > t_end - 60])["ttfa_ms_p95"]
        report["levels"][c] = s
        print(f"c={c}: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()), flush=True)
    report["cancel"] = await cancel_test(url, a.sample_rate, a.voice)
    print("cancel:", report["cancel"])
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{a.label}.json").write_text(json.dumps(report, indent=2))

    print(f"\n| concurrency | requests | errors | TTFA p50 / p95 / p99 (ms) | RTF p50 / p95 | underrun reqs | RSS MB |")
    print("|---|---|---|---|---|---|---|")
    for c, s in report["levels"].items():
        print(f"| {c} | {s['requests']} | {s['errors']} | {s['ttfa_ms_p50']:.0f} / {s['ttfa_ms_p95']:.0f} / {s['ttfa_ms_p99']:.0f} "
              f"| {s['rtf_p50']:.3f} / {s['rtf_p95']:.3f} | {s['underrun_requests']} | {s['rss_mb']} |")


if __name__ == "__main__":
    asyncio.run(main())
