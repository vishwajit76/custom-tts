# Benchmarks

Hardware: Apple M4 (4 performance + 6 efficiency cores, 16 GB), macOS 26, Python 3.12, ONNX Runtime 1.30,
piper-tts 1.8, voice `hi_IN-rohan-medium` (22.05 kHz), unless stated. Linux numbers come from Docker Desktop
(linux/arm64 VM on the same Mac). **No NVIDIA GPU and no x86 server were available for sections 1-8** (section 9 is a 4 vCPU x86 container), so GPU figures
are not measured. Raw JSON is in `bench/results/`. Every table can be reproduced with the command above it.
Servers ran with `CACHE_SIZE=0` so repeated sentences are not served from cache.

Definitions:
- **TTFA** (time to first audio): WebSocket `speak` sent → first binary frame received, measured client-side
  on localhost.
- **RTF**: request wall time ÷ audio duration.
- **Underrun**: a player that started at the first frame would have run dry, with zero jitter buffer.
- **Call simulation** (`--call-sim 2`): every client waits for its audio to finish playing plus 2 s of
  "caller speech" before the next sentence. The bot talks ~70% of the time, which is heavier than a real call.

## 1. Model selection (single sentence, CPU)

| Engine | Threads | Median latency / sentence | RTF | Notes |
|---|---|---|---|---|
| **Piper hi_IN medium (rohan / priyamvada / pratham)** | 4 | 73–86 ms | **0.022** | chosen |
| Piper hi_IN medium | 1 | 161–197 ms | 0.051 | |
| Kokoro-82M fp32, `hf_alpha` | default | 901 ms | 0.237 | |
| Kokoro-82M int8 | default | 1794 ms | 0.498 | int8 slower |
| Qwen3-TTS 1.7B / 0.6B (previous engine) | CPU/MPS | seconds | ~2.1 / ~1.8 | slower than real time |

## 2. ONNX Runtime tuning (`python -m bench.ort_variants`)

All 13 benchmark sentences × 3; "conc" = parallel Run() calls on one shared session.

| Variant | conc | p50 ms | p95 ms | × real time | peak RSS MB |
|---|---|---|---|---|---|
| 1 intra-op thread | 1 | 304 | 696 | 17.4 | 470 |
| 1 thread | 4 | 386 | 813 | 55.9 | 1448 |
| 2 threads | 1 | 186 | 389 | 29.6 | 489 |
| **4 threads** | 1 | **141** | **299** | 39.1 | 485 |
| **4 threads** | 4 | 333 | 616 | **66.1** | 1576 |
| arena off, 1 thread | 1 | 314 | 701 | 17.3 | 1162 |
| arena + mem-pattern off | 1 | 321 | 670 | 17.4 | 950 |
| 4 threads, no spin-wait | 4 | 401 | 720 | 54.1 | 1738 |

Findings: 4 intra-op threads cut single-chunk latency 2.2× at equal or better throughput, so the default is
`THREADS_PER_WORKER=min(4, cpus)`. Turning the memory arena off *raises* peak RSS on macOS, so defaults stay.
Disabling spin-wait gave nothing. (macOS reports peak RSS; the Linux container's current RSS is 233 MB idle,
~430–640 MB under load.)

| Optimization | Result | Decision |
|---|---|---|
| INT8 dynamic quantization, MatMul/Gemm only | 131 ms vs 130 ms p50, 63 MB (model is conv-dominated) | rejected: no gain |
| INT8 dynamic, all ops (ConvInteger) | 556 ms vs 130 ms p50 (4.3× slower), 18 MB file, −60 MB RSS | rejected |
| CoreML EP (Apple GPU/ANE) | 197 ms vs 130 ms p50 | rejected: dynamic shapes fall back / partition |
| FP16 + CUDA EP | not measurable here | `Dockerfile.gpu`, `USE_CUDA=1`: untested |
| Dynamic batching | not implemented | VITS output needs per-item length trimming, and batching adds queueing delay; parallel sessions already saturate the cores |
| Warm-up at load | first Run allocates; the warm-up hides it | on |
| Lead-silence trim | the model emits 230–280 ms of silence before speech | trimmed to 30 ms: ~200 ms earlier audible audio |

## 3. Server configuration sweep (native, burst: every client speaks back-to-back)

`bench/matrix.sh "<env>" -- --concurrency 1,5,10`. TTFA p50 / p95 / p99 in ms.

| Config | 1 stream | 5 streams | 10 streams | RTF p50 @10 | RSS MB @10 |
|---|---|---|---|---|---|
| 5 workers × 1 thread | 155 / 312 / 312 | 205 / 424 / 455 | 301 / 536 / 645 | 0.118 | 1926 |
| **2 workers × 4 threads** | **74 / 140 / 140** | 192 / 403 / 434 | 210 / 412 / 509 | 0.111 | **1076** |
| 3 workers × 2 threads | 96 / 200 / 200 | 249 / 417 / 518 | 269 / 485 / 537 | 0.120 | 1488 |
| 4 × 2, first chunk 40 chars | 96 / 142 / 142 | 229 / 367 / 393 | 282 / 444 / 565 | 0.144 | 1474 |

Zero errors and zero underruns in every run. "Burst" means 10 streams talking non-stop, i.e. ~10× the load of
10 real calls.

## 4. Chunk size under realistic call load (native, 2 × 4, `--call-sim 2`)

| MAX_CHUNK_CHARS | 10 calls | 20 calls | 40 calls |
|---|---|---|---|
| 200 | 99 / 165 / 289 | 108 / 293 / 368 | 124 / 323 / 380 (1 underrun) |
| **120** (default) | 100 / 184 / 224 | **99 / 215 / 259** | 134 / 277 / 310 (1 underrun) |

Shorter chunks mean shorter non-preemptible inference runs, so new calls' first chunks wait less (EDF can only
reorder the queue, not interrupt a run).

## 5. Linux container, 4 vCPU (`docker run --cpus 4`, Docker Desktop VM)

| Layout | 1 stream (burst) | 5 streams (burst) | 10 calls (sim) | 20 calls (sim) |
|---|---|---|---|---|
| 1 × 4 (auto from cgroup quota) | 111 / 202 / 202 | 267 / 520 / 623 | 174 / 596 / 866 | 183 / 950 / 1248 |
| 2 × 2 | 118 / 208 / 208 | 304 / 518 / 655 | 180 / 579 / 852 | 202 / 1025 / 1319 |
| 4 × 1 | 127 / 230 / 230 | 293 / 528 / 649 | 174 / 643 / 937 | 199 / 997 / 1315 |

Per core, the container is ~2–3× slower than native macOS on the same machine: VM overhead, plus a different
ONNX Runtime build (linux-aarch64 vs macOS arm64). Before these numbers existed, `os.cpu_count()` inside the
container reported all 10 VM cores and oversubscribed the 4-CPU quota. The server now reads the cgroup quota
(`app/core/config.py::cpu_count`).

## 6. Cancellation

The server acknowledges `cancel` in 2.4–4.5 ms p50 (native and container), and no frame of a cancelled request
follows the ack. It sends audio faster than real time, so up to ~4.6 s of audio was already in flight or in the
client's buffer at barge-in. Clients must flush their own playback buffer on barge-in (documented in README).

## 7. Engine comparison (Sep 29)

### Pronunciation and naturalness

Same 27-sentence set (`bench/quality_set.tsv`: Hindi, numbers, Hinglish, names, romanized), same normalizer, speed
1.0. Intelligibility = PER/CER of local Whisper transcripts (lower is better). Naturalness = UTMOS22 and
Audiobox-Aesthetics CE (higher is better). Both are proxies: UTMOS is English-trained, and CE is not validated for
Hindi. They agree on the ranking, but a blind native-listener test is still owed (docs/research.md). Piper rows come from
`q2-ie-on` (Indian-English markup on). Other rows come from `bench/results/q3-{supertonic3,kokoro,syspin}.json` and
`q3-aesthetics.json`. No voice produced silence or clipping.

| Engine | Voice | Sex | PER | PER Hinglish | PER numbers | CER | UTMOS | CE |
|---|---|---|---|---|---|---|---|---|
| Kokoro-82M | hf_alpha | F | .059 | .063 | .100 | .124 | **4.33** | 5.82 |
| Kokoro-82M | hf_beta | F | .057 | .066 | .093 | .147 | 4.25 | 5.78 |
| Kokoro-82M | hm_psi | M | .055 | .039 | .104 | .120 | 4.18 | **5.90** |
| Kokoro-82M | hm_omega | M | .054 | .042 | .091 | .127 | 4.13 | 5.74 |
| Supertonic 3 | F3 | F | .038 | .019 | .060 | .097 | 4.22 | 5.35 |
| Supertonic 3 | F1 | F | .050 | .044 | .079 | .128 | 4.18 | 5.55 |
| Supertonic 3 | M4 | M | .043 | .007 | .086 | .072 | 4.15 | 5.38 |
| Supertonic 3 | F2 | F | .077 | .044 | .190 | .155 | 4.06 | 5.53 |
| Supertonic 3 | F5 | F | .040 | .019 | .073 | **.069** | 4.04 | 5.60 |
| Supertonic 3 | M1 | M | .051 | .053 | .075 | .132 | 3.98 | 5.63 |
| Supertonic 3 | M2 | M | .049 | .035 | .083 | .094 | 3.89 | 5.31 |
| Supertonic 3 | F4 | F | .127 | .032 | .367 | .170 | 3.89 | 5.47 |
| Supertonic 3 | M5 | M | .075 | .032 | .187 | .119 | 3.88 | 5.63 |
| Supertonic 3 | M3 | M | .043 | **.004** | .078 | .090 | 3.67 | 5.21 |
| Piper | rohan | M | .045 | .042 | .056 | .111 | 3.89 | 5.51 |
| Piper | pratham | M | .065 | .061 | .100 | .137 | 3.91 | 5.49 |
| Piper | priyamvada | F | .073 | .088 | .108 | .180 | 3.83 | 5.30 |
| SYSPIN VITS | male | M | .213 | .492 | .197 | .266 | 3.80 | 5.37 |
| SYSPIN VITS | female | F | .251 | .513 | .219 | .301 | 3.65 | 5.70 |

- **Kokoro** is the most natural on both metrics, and its intelligibility is on par with Piper.
- **Supertonic 3** has the best Hinglish intelligibility. Only M3 stands out, with low naturalness.
- **Piper** has the lowest naturalness of the usable engines, and none of its Hindi voices is commercially clean (docs/research.md).
- **SYSPIN** drops English words entirely (Hinglish PER about 0.5), so it is not usable for call text.
- **Pocket TTS Hindi** was skipped: it ships no voice we are licensed to use.

**Scoring artifact.** The high "numbers" PER for Supertonic F4, F2 and M5 (and one Kokoro row) comes from a single
sentence, "कीमत डेढ़ लाख रुपये है". For a spoken "डेढ़ लाख", Whisper wrote "1.500,000" or "1.5 लाग", and normalizing
that transcript does not give back "डेढ़ लाख". With that sentence left out, number PER is:

| Voices | Number PER |
|---|---|
| All Supertonic voices | 0.060-0.086 |
| Kokoro | 0.073-0.083 |
| Piper rohan | 0.041 |
| Piper pratham and priyamvada | 0.101-0.108 |

Read the PER columns with that in mind.

### Latency

Live server with all three engines loaded (`ENGINES=piper,supertonic,kokoro`) on an idle M4. Default layout is
WORKERS=2 and THREADS_PER_WORKER=4. Load average was about 4-6 during the runs; the idle baseline was 4.0. Burst
mode means every client starts at once, which is the worst case for time to first audio (TTFA). Results are in
`bench/results/eng-*.json`. Peak RSS with all three engines loaded was 1.4-2.2 GB.

| Voice | TTFA p50/p95, 1 stream | RTF, 1 stream | TTFA p50/p95, 4 streams | RTF, 4 streams | Underruns, 4 streams |
|---|---|---|---|---|---|
| Piper rohan | 68 / 127 ms | 0.025 | 174 / 255 ms | 0.062 | 0/52 |
| Supertonic F1, 8 steps (default) | 640 / 976 ms | 0.25 | 1499 / 2297 ms | 0.66 | 11/52 |
| Supertonic F1, `SUPERTONIC_STEPS=4` | 298 / 472 ms | 0.12 | 752 / 1255 ms | 0.32 | 0/52 |
| Supertonic F1, 4 workers × 2 threads | 566 / 891 ms | 0.23 | 1443 / 2235 ms | 0.56 | 0/52 |
| Kokoro hf_alpha | 789 / 1314 ms | 0.25 | 2588 / 4128 ms | 0.84 | 18/52 |
| Kokoro hf_alpha, 4 workers × 2 threads | 1166 / 1858 ms | 0.36 | 3202 / 4883 ms | 1.02 | 23/52 |

**Rerun with `CACHE_SIZE=0` (Sep 29, load average 3-6).** "Realistic calls" uses `--call-sim 2`: after each reply, the
client waits for playback plus 2 s of caller speech. Results are in `eng-supertonic-steps{8,6}-{nocache,callsim}.json` and
`eng-kokoro-callsim.json`.

| Voice | TTFA p50/p95, 1 stream | 4 streams, burst | 4 realistic calls | 8 realistic calls | UTMOS (F1/F3/M4) |
|---|---|---|---|---|---|
| Supertonic F1, 8 steps | 573 / 1004 ms | 1510 / 2533 ms, 7 underruns | 743 / 1208 ms, 0 underruns | 1144 / 2191 ms, 8 underruns | 4.18 / 4.20 / 4.10 |
| Supertonic F1, 6 steps | 557 / 899 ms | 1389 / 2156 ms, 5 underruns | 566 / 910 ms, 0 underruns | 779 / 1321 ms, 0 underruns | 4.13 / 4.10 / 3.97 |
| Supertonic F1, 4 steps | rejected | | | | 3.33 / 3.43 / 3.27, one PER blow-up |
| Kokoro hf_alpha | see above | see above | 930 / 1643 ms, 0 underruns (2 calls: 708 / 1133 ms) | not run | 4.33 (hf_alpha) |

- `SUPERTONIC_STEPS=6` costs about 0.05-0.13 UTMOS, with PER unchanged (`q4-supertonic-steps{8,6,4}.json`). In exchange, the M4 holds 8 realistic calls without underruns.
- At 4 steps, quality falls below Piper.
- Recommendation: 8 steps where quality comes first (demo, few calls), and 6 steps for call capacity.

On this CPU, only Piper meets the 200 ms TTFA target at 4 concurrent streams. Kokoro is the most natural but costs
about 0.55 RTF per stream and barely speeds up with more threads, so the M4 can hold about 2 real-time Kokoro
streams. Supertonic with 4 steps is the middle ground. Serving the natural voices to many concurrent calls needs a
GPU host (docs/research.md, Tier B) or many more CPU cores.

## 8. Final configuration and soak: pending

## 9. Baseline on the 4 vCPU x86 container (2026-09-29, Phase 0 measurement)

Measured this session, Piper `hi_IN-rohan-medium` only (`scripts/download_voices.py` succeeded here; Kokoro/Supertonic/Qwen were
not downloaded or run, so there are no numbers for them on this box).

**Hardware / software:** Intel Xeon @ 2.10 GHz, 4 vCPU (1 thread per core), 15 GiB RAM, Linux 6.18 VM, Python 3.11.15,
onnxruntime 1.30.0, piper-tts 1.8.0. Server defaults on 4 CPUs: `WORKERS=1`, `THREADS_PER_WORKER=4`. The load generator ran on the
same machine (its own CPU use is small: about 1 s of user time for a 105 s run), so a separate client host would look slightly better.

**Commands (exact):**
```
python scripts/download_voices.py                       # hi_IN-rohan-medium -> models/piper
CACHE_SIZE=0 python -m uvicorn app.main:app --port 8000
python -m bench.bench --url ws://localhost:8000/v1/audio/ws --concurrency 1,5,10,20 --sample-rate 16000 --label baseline-20260929-4vcpu
```
13 requests per client (burst: each client speaks back-to-back, harsher than real calls), 16 kHz, phrase cache off, warm-up first.
Raw data: `bench/results/baseline-20260929-4vcpu.json`. CPU: `vmstat 1` alongside; RAM: `/metrics` and `ps` RSS.

| concurrency | requests | errors | TTFA p50 / p95 / p99 (ms) | RTF p50 / p95 | underrun requests | server RSS MB (at end of level) |
|---|---|---|---|---|---|---|
| 1 | 13 | 0 | 119 / 195 / 195 | 0.042 / 0.055 | 0 | 403 |
| 5 | 65 | 0 | 274 / 608 / 750 | 0.129 / 0.382 | 0 | 423 |
| 10 | 130 | 0 | 266 / 817 / 1081 | 0.271 / 0.728 | 3 (max 242 ms) | 430 |
| 20 | 260 | 0 | 1699 / 2282 / 2620 | 0.714 / 1.808 | 95 (max 1417 ms) | 440 |

- CPU: whole-machine busy (user+system) averaged 85 % over the 121 s run (idle gaps included), with peaks at 100 % (vmstat, 1 s samples). Peak server RSS 440 MB.
- Reading: one Piper worker with 4 threads is real-time up to about 5 burst streams here (0 underruns) and about 10 with occasional
  underruns; at 20 burst streams it is past capacity (median RTF 0.71, p95 above 1.0, 95 of 260 requests underran). This
  burst load is heavier than real calls (`--call-sim`, section 4), so it under-states call capacity; it was not re-run with `--call-sim` here.
- TTFA at 1 stream (119 ms p50) is slower than the M4 records in section 3 (68 ms), as expected for a 2.1 GHz virtualised Xeon.
  These are different machines: do not compare the rows as a regression.
- **Cancellation (5 trials, barge-in right after first audio):** ack p50 5.7 ms, max 7.6 ms, 0 ms of audio sent after the cancel.
  An earlier run with the same commands **before a fix** measured ack about 850 ms and 23 s of audio sent after the cancel: on Python 3.11
  `asyncio.wait_for(ws.send_bytes(...))` swallowed a cancel that arrived as the frame send completed, so the request ran to its end.
  `app/api/ws.py` now uses `asyncio.timeout`. Clients must still flush their own buffered audio (voice-system.md, telephony section).
- Not measured: 50/100/200 concurrent calls, Kokoro/Supertonic/Qwen on this box, GPU, a soak run, `--call-sim` on this box.

### Objective eval sample (`bench/eval.py`, Piper rohan, `DSP_PROSODY=true`, 6 sentences)

```
DSP_PROSODY=true python -m bench.eval --voices hi_IN-rohan-medium --limit 6 --label eval-piper-dsp-20260929 [--cer]
```
Output `bench/results/eval-piper-dsp-20260929.json`. Median over the 6 sentences: F0 148 Hz neutral, 143-145 Hz at speed 0.85/1.15, about 170 Hz with
`pitch:dsp` +2 (expected about 166 Hz; pyin F0 on synthetic speech is noisy, single clips vary by several semitones); RMS -19.0 dBFS neutral, -17.1 with
`energy:dsp` 1.3; no clipping in neutral rows. 8 kHz round trip: about 0.4 % of energy lies above 4 kHz and is lost. Speaker similarity to the
same voice's neutral output is 0.97-1.0 for speed/pitch/energy variants (neutral is 1.0 by construction) and 0.92-0.98 after the 8 kHz round trip (5 of 6 sentences; the shortest is under the encoder's 1 s minimum), with the **mfcc backend, which is not
a neural speaker verifier**. CER was **skipped** (`--cer`: Whisper weights not reachable offline here; recorded as skipped in the JSON).
These are signal statistics. **Predicted MOS is not human MOS**, and none of it says whether the voice sounds natural or expressive.

## Scaling to 200 calls

Pending final numbers.
