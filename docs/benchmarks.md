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

### 9b. Kokoro and Supertonic on the same 4 vCPU box (2026-09-29)

Same box, commands and settings as section 9 (burst, 13 requests per client, 16 kHz, `CACHE_SIZE=0`, warm-up first), concurrency 1 and 5 only. One voice per engine
(`kokoro:hf_alpha`, `supertonic:F3`, Supertonic at the default 8 steps), served with `ENGINES=piper,<engine> API_KEYS=` (auth off; the shell had an `API_KEYS` set and the bench sent no key).
Weights: `scripts/download_voices.py kokoro supertonic` succeeded (Kokoro from the GitHub release, Supertonic from the HF hub, which was reachable for that repo). Raw data:
`bench/results/bl-kokoro-hf_alpha-20260929-4vcpu.json`, `bl-supertonic-F3-20260929-4vcpu.json`. **Caveat:** another worker was editing/testing in the same container, so background CPU load was not controlled; treat these as indicative (single run each, no repeats).

| engine / voice | concurrency | errors | TTFA p50 / p95 / p99 (ms) | RTF p50 / p95 | underrun reqs (max ms) | RSS MB |
|---|---|---|---|---|---|---|
| Piper rohan (section 9) | 1 | 0 | 119 / 195 / 195 | 0.042 / 0.055 | 0 | 403 |
| Piper rohan (section 9) | 5 | 0 | 274 / 608 / 750 | 0.129 / 0.382 | 0 | 423 |
| Kokoro hf_alpha | 1 | 0 | 1074 / 1360 / 1360 | 0.299 / 0.356 | 0 | 797 |
| Kokoro hf_alpha | 5 | 0 | 4986 / 8291 / 10167 | 1.308 / 2.449 | 27 of 65 (6279) | 833 |
| Supertonic F3 | 1 | 0 | 738 / 1068 / 1068 | 0.278 / 0.460 | 0 | 716 |
| Supertonic F3 | 5 | 0 | 4584 / 6271 / 6734 | 1.304 / 2.560 | 27 of 65 (1995) | 730 |

- Both are real-time for one stream on this box (RTF 0.28-0.30) but not for 5 burst streams (median RTF 1.3, 27 of 65 requests underran). Piper is about 7x lower RTF and 9x lower TTFA at one stream.
  Consistent with the `balanced` latency tier in `engine_latency_tiers` (about 2 streams per 4 cores; 5 is past that).
- Cancel ack: Kokoro p50 5.5 ms / max 8.0 ms, Supertonic p50 10.9 ms / max 17.0 ms, 0 ms buffered ahead.

### 9c. bench/eval.py across three engines (6 sentences, neutral, 2026-09-29)

`ENGINES=piper,kokoro,supertonic python -m bench.eval --voices hi_IN-rohan-medium,kokoro:hf_alpha,supertonic:F3 --limit 6 --label eval-engines-20260929 --cer`
(`bench/results/eval-engines-20260929.json`). Median over 6 sentences: 

| voice | duration s | chars/s | F0 median Hz | RMS dBFS | worst peak dBFS | max clipping | lead / trail silence ms |
|---|---|---|---|---|---|---|---|
| Piper rohan | 4.40 | 11.4 | 151 | -18.9 | -1.4 | 0 | 10 / 230 |
| Kokoro hf_alpha | 5.03 | 9.7 | 223 | -18.8 | 0.0 | 0.0002 | 10 / 475 |
| Supertonic F3 | 4.19 | 11.1 | 171 | -27.0 | -7.1 | 0 | 10 / 670 |

Supertonic output is about 8 dB quieter than the others (level-normalise before comparing in a listening test); Kokoro touches full scale on some samples (0.02 % of samples); none was silent. `eval.py --cer` still records **skipped**
(its transformers Whisper needs `openai/whisper-large-v3-turbo` from the HF hub: `OSError: We couldn't connect to 'https://huggingface.co'`). The speaker-similarity column in that JSON is the mfcc backend (not neural).

### 9d. CER/PER with faster-whisper (new `bench/quality_fw.py`)

`faster-whisper` 1.2.1 installed from PyPI and `WhisperModel("small", compute_type="int8")` **did download and run** (Systran weights via the HF hub, which was reachable for those repos; `openai/whisper-large-v3-turbo` for
`training/asr.py` was not). Command: `ENGINES=piper,kokoro,supertonic WHISPER_MODEL=small python -m bench.quality_fw --voices hi_IN-rohan-medium,kokoro:hf_alpha,supertonic:F3 --label quality-fw-small-20260929`
on the 27-sentence `bench/quality_set.tsv` (raw output: `bench/results/quality-fw-small-20260929/`, `.json`). PER/CER vs the normalised input text; UTMOS **blocked**
(`torch.hub` load of `tarepan/SpeechMOS:v1.2.0` from GitHub: `HTTP Error 403: Forbidden`; reported as NaN).

| voice | PER hindi | PER hinglish | PER names | PER numbers | PER romanized | mean PER | mean CER |
|---|---|---|---|---|---|---|---|
| Piper rohan | 0.129 | 0.147 | 0.178 | 0.223 | 0.121 | 0.161 | 0.309 |
| Kokoro hf_alpha | 0.131 | 0.226 | 0.175 | 0.161 | 0.182 | 0.169 | 0.270 |
| Supertonic F3 | 0.113 | 0.184 | 0.125 | 0.224 | 0.122 | 0.158 | 0.295 |

**Not comparable with the section 7 table**: that used a larger local Whisper and reports PER 0.04-0.06 for the same voices; Whisper `small` mis-transcribes Hindi more, so these absolute values are inflated by the ASR.
Differences of about 0.01 between voices are within ASR noise; use these only for relative sanity checks with the same ASR.

### 9e. Neural speaker encoder (resemblyzer 0.1.4) vs the mfcc baseline (2026-09-29)

`pip install resemblyzer` (pulls webrtcvad; `pretrained.pt` ships in the package; licence Apache-2.0, see `docs/licenses.md`). `tests/test_speaker_encoder.py`: 5 passed (its only neural-backend test is a skip-if-missing check, so it does not exercise
the neural path; the measurement below does). The backend worked without a fix; only its licence comment was wrong (said MIT).
Command: `ENGINES=piper,kokoro,supertonic python -m bench.spk_sim --voices hi_IN-rohan-medium,hi_IN-pratham-medium,hi_IN-priyamvada-medium,kokoro:hf_alpha,kokoro:hm_psi,supertonic:F3,supertonic:M4 --label spk-sim-20260929`.
7 voices x 4 different sentences (about 6 s each, median). "same" = one voice, two different sentences (42 pairs); "diff" = two voices, same sentence (84 pairs). All synthetic voices; **no real speakers, no human-recorded audio.**

| backend | embed latency median / max (ms, about 6 s audio, 4 vCPU) | same-voice cosine mean (min) | different-voice cosine mean (max) |
|---|---|---|---|
| mfcc | 5.2 / 6.1 | 0.992 (0.968) | 0.962 (0.989) |
| resemblyzer (raw 16 kHz audio, as shipped) | 30.9 / 159.6 (first calls) | 0.935 (0.879) | 0.643 (0.830) |
| resemblyzer with `preprocess_wav` (rejected variant) | 99.7 / 176.0 | 0.927 (0.835) | 0.633 (0.834) |

- On these voices resemblyzer separates same from different clearly (the lowest same-voice score, 0.879, is above the highest different-voice score, 0.830, so a threshold exists on this set), while mfcc does not (0.962 for different voices). Its closest different pairs were
  `priyamvada|kokoro:hf_alpha` 0.783 and `rohan|supertonic:M4` 0.738. A margin of 0.05 on 7 TTS voices is thin: do not pick an identity threshold from this; it is not calibrated on real speech.
- `preprocess_wav` (volume normalisation + webrtcvad trim) did not improve separation and was 3x slower, so the backend keeps raw audio. Raw JSON: `bench/results/spk-sim-20260929.json`, `spk-sim-preprocess-20260929.json`.

### 9f. Docker

`docker` CLI 29.3.1 is present but `docker info` fails: `failed to connect to the docker API at unix:///var/run/docker.sock ... no such file or directory` (no daemon in this sandbox). `docker build` and the `/health` smoke were **not run**.

## 10. Custom Hindi voice: checkpoint comparison protocol and first run (2026-09-30)

Tool: `python -m bench.compare_checkpoints` (protocol and metric labels in its docstring and in [custom-voice-runbook.md](custom-voice-runbook.md) 6.1). Fixed 50-sentence set `bench/hi_eval_50.txt`
(sha256 661fd8e5...; 0 exact or 5-word overlaps with `data/hi_f/metadata.csv` + `test.csv`, `python -m bench.check_eval_overlap`; 4-word overlap: 1 sentence, 3-word: 32, i.e. only common short phrases;
overlap with the unknown pretraining text of the rohan base voice cannot be checked). 3 synthesis repeats per sentence (Piper's ONNX noise is unseeded), faster-whisper `small` int8, noise_scale 0.667 / noise_w 0.8 /
length_scale 1.0, 12 real held-out clips as speaker reference, 95% two-level bootstrap CIs (2000 resamples). Raw: `bench/results/compare_340k_350k_355k_small.json`.

The three models are the current HF folders `milestones/step_340000` (uploaded 03:31 UTC), `step_350000` (05:50) and `step_355000` (07:00), sha256 dcccfb13... / 0fe65847... / 673ba46c...: **all three are v5 uploads**
(v5 overwrote v4's 350000 and 355000 after v4 ended / while this run was being prepared; v4's 355000 ONNX no longer exists on HF), so this is one training timeline at constant LR 1.5258e-4.

| Metric (mean, 95% CI) | 340000 | 350000 | 355000 |
|---|---|---|---|
| CER (vowel signs counted), small ASR | 0.166 [0.148, 0.187] | 0.170 [0.148, 0.194] | 0.159 [0.140, 0.180] |
| CER legacy consonant-only (comparable to old rows) | 0.145 | 0.148 | 0.136 |
| PER (espeak phonemes) | 0.178 [0.157, 0.201] | 0.177 [0.155, 0.202] | 0.173 [0.151, 0.198] |
| UTMOS22 **predicted MOS** | 3.75 [3.65, 3.85] | 3.80 [3.71, 3.88] | 3.87 [3.78, 3.96] |
| Speaker similarity, Resemblyzer cosine (real-vs-real ceiling 0.918) | 0.915 [0.909, 0.921] | 0.914 [0.907, 0.921] | 0.922 [0.916, 0.928] |
| Speaker similarity, MFCC cosine (saturated, weak) | 0.992 | 0.990 | 0.990 |
| Telephony CER, 8 kHz mu-law channel (synthetic) | 0.167 | 0.170 | 0.173 |
| Telephony CER, 16 kHz 8-bit mu-law (synthetic) | 0.167 | 0.165 | 0.156 |
| Repeat std of mean CER (3 repeats) | 0.0067 | 0.0085 | 0.0023 |

Paired deltas vs 340000 (same sentences): 350000: CER +0.004 [-0.007, 0.016], PER -0.001, UTMOS +0.044 [-0.021, 0.115], speaker sim -0.001: nothing distinguishable. 355000: CER -0.007 [-0.019, 0.006], PER -0.005 [-0.016, 0.009]
(not distinguishable), UTMOS +0.121 [0.042, 0.201] and speaker sim +0.007 [0.004, 0.010] (intervals exclude 0, but both are small and UTMOS is an English-trained predictor; the speaker gain is 0.007 on a
scale whose ceiling is 0.918). Reading: 15k steps at constant LR bought at most a small predicted-MOS gain; intelligibility did not move measurably. **This says nothing about how natural the voice sounds: no listening test has been run.**
The simulated telephony channels changed CER by under 0.01 (the ASR resamples to 16 kHz, and mu-law at 8 bit is mild): they are a sanity check, not a carrier-quality result. Small ASR floor is high (about 0.16 CER on this set), so use `--asr large-v3` when a decision hangs on CER.

Validity notes: (1) CER before 2026-09-30 ignored vowel signs (see training-progress.md), PER never did. (2) A single-pass `milestone_eval` row is one random draw; differences under about 0.01 to 0.02 CER are noise (repeat std above is 0.002 to 0.009 and sentence sampling adds more: CIs are about +-0.02).
(3) Speaker similarity is an embedding cosine, not speaker verification; UTMOS is predicted MOS. (4) Ranking by these metrics is not a substitute for `docs/listening-test/README.md`.

## Scaling to 200 calls

Pending final numbers.

## 10. P3-P7 audit measurements, custom voice (2026-09-30)

Separate section on purpose (another worker edits this file too). Everything here was measured on this session's box with **`bench/bench.py` used read-only** (same code, result JSON redirected
to the scratchpad so nothing was written under `bench/`; `python -c` wrapper that sets `bench.bench.RESULTS`) and scratch scripts for the ASR runs. Voice: `hi_IN-custom-medium`
(personal fine-tune, intermediate checkpoint, selected with `MODELS_EXTRA=voices`), Piper, 22.05 kHz native.

**Read this first: these latency numbers are contended.** The VM is the same 4 vCPU Xeon @ 2.1 GHz as section 9, but another worker's jobs (`bench.compare_checkpoints`, training/export
tests, ~200 % CPU for the whole session) were running on it: load average 4-8 on 4 cores during every run here. The same `hi_IN-rohan-medium` that measured 119 ms TTFA p50 / RTF 0.042 at 1 stream in
section 9 measured **330 ms / 0.113 under this contention** (2.5-2.8x slower), so treat section 10 latencies as an upper bound on an idle box, not as capacity. The custom voice costs the same as rohan
(same VITS architecture and size): 291 ms / 0.118 vs 330 ms / 0.113 at 1 stream, 725 vs 723 ms p50 at 5 streams, in the same conditions. No quiet window appeared; nothing below was re-run uncontended.

```
ENGINES=piper MODELS_EXTRA=voices CACHE_SIZE=0 API_KEYS= LOG_LEVEL=WARNING python -m uvicorn app.main:app --port 8765 --ws-ping-interval 20 --ws-ping-timeout 20 --timeout-graceful-shutdown 10
python -m bench.bench --url ws://localhost:8765/v1/audio/ws --voice hi_IN-custom-medium --concurrency 1,5 --sample-rate 8000 --label custom-8k-c1-5   # defaults: WORKERS=1 x THREADS_PER_WORKER=4
python -m bench.bench ... --voice hi_IN-rohan-medium --concurrency 1,5 --sample-rate 8000                                                          # same conditions, for the contention factor
python -m bench.bench ... --voice hi_IN-custom-medium --concurrency 3 --sample-rate 8000 --soak 600 --call-sim 1 --label custom-soak600-c3-callsim1
```

### TTFA / RTF (WebSocket, 8 kHz PCM, 13 requests per client, burst = back-to-back, phrase cache off)

| voice | concurrency | requests | errors | TTFA p50 / p95 / p99 (ms) | RTF p50 / p95 | underrun requests | server RSS MB |
|---|---|---|---|---|---|---|---|
| custom | 1 | 13 | 0 | 291 / 503 / 503 | 0.118 / 0.135 | 0 | 591 |
| custom | 5 | 65 | 0 | 725 / 1544 / 1671 | 0.521 / 0.864 | 15 (max 787 ms) | 669 |
| rohan (same conditions) | 1 | 13 | 0 | 330 / 582 / 582 | 0.113 / 0.177 | 0 | 784 |
| rohan (same conditions) | 5 | 65 | 0 | 723 / 1612 / 1869 | 0.500 / 0.788 | 12 (max 1225 ms) | 863 |

Section 9's uncontended rohan rows on this box (119 / 274 ms p50 at 1 / 5 streams, 0 underruns at 5) are the better guide to an idle machine. With 13 requests per client the p95/p99 at 1 stream are the
2nd-largest/largest of 13 samples: indicative only. RSS is higher than section 9's because 4 voices are loaded (3 bundled + custom).

### Barge-in and the 10-minute soak

- Cancel ack (bench `cancel_test`, 5 trials, cancel right after first audio): p50 0.5 ms, max 3.4 ms, **0 ms of audio** after the cancel (client-side buffers still need flushing: voice-system.md).
- Soak, **600 s, 3 concurrent calls with `--call-sim 1`** (each waits for its reply to "play" plus 1 s): 313 requests, **0 errors, 0 underruns**, TTFA p50/p95/p99 213 / 588 / 814 ms, RTF p50/p95 0.089 / 0.215,
  1409 s of audio. **RSS 864 MB at start -> 890 MB at the end, 890.6 MB in a fresh run afterwards**: about +26 MB in the first minutes (allocator warm-up), then flat: no growth trend over 10 minutes.
  p95 TTFA in the first vs last minute was 347 vs 750 ms; that is the contending job's load, not drift: a fresh 3-call run right after the soak gave the same 239 / 603 / 719 ms.
- **Churn** (`python` script against the live server, about 4.5 minutes in total over three runs, one of which hung on a client bug and was killed): per run concurrent (a) WS `speak` then `cancel` after 1-3 frames, (b) WS dropped
  with no close frame after 0-2 frames (`transport.abort()`), (c) HTTP `/v1/audio/speech/stream` read for 1-2 chunks then the connection closed mid-body. Roughly 120 cancels, 320 abrupt WS drops and 290 HTTP aborts completed.
  Server counters afterwards: `streams_total` 1478, `streams_cancelled` 898, `streams_failed` **0**, `streams_rejected` 0, **`streams_active` 0** (no slot leaked), RSS 906 -> 908 MB, no exception in the server log. The per-key rate
  limit (600 requests/connections per minute) throttled the client during the run (observed: WS handshakes refused with HTTP 403, which is what Starlette turns a close-before-accept into; some HTTP streams got an empty body, status not captured), so the load was lower than the generator intended.
- SIGTERM with a long WebSocket stream in flight: the socket was closed with code 1012 after 0.1 s and the process exited after 1.7 s (`Application shutdown complete`), well inside `--timeout-graceful-shutdown 10`.

### Output-rate intelligibility, brand A/B, onset trim (details in voice-system.md)

- CER by output rate, same cached waveform through `tts.stream`, 20 utterances, Whisper-small int8: 22.05 kHz 0.300, 16 kHz 0.311, 8 kHz 0.329, 8 kHz + mu-law round trip (`audioop`, quantisation only) 0.340. Deltas vs 22.05 kHz:
  +0.011 +/- 0.005, +0.029 +/- 0.020, +0.040 +/- 0.045 (mean +/- s.e.). Peak -2.4 dBFS, RMS -20.9 dBFS, 0 full-scale samples at every rate; 120 ms+ pauses identical in length at 8 kHz and 22.05 kHz.
  (An earlier run that re-synthesized per rate gave 0.330 / 0.318 / 0.362 / 0.377 but is confounded by VITS sampling noise and was discarded.)
- Brand respelling A/B, 15 brands x 2 carrier sentences, Whisper-small CER: custom voice 0.331 (English route) vs 0.303 (Devanagari route); rohan 0.297 vs 0.253. WhatsApp is worse with every Devanagari spelling tried
  (0.156 vs 0.21-0.33) and stays on the English route; Flipkart measured 0.325 vs 0.239 (custom) and 0.251 vs 0.185 (rohan) with the nukta spelling. Noisy proxy; the espeak phoneme outputs in voice-system.md are the stronger evidence.
- Lead-silence trim (`trim_lead`, threshold 0.01, keep 30 ms): custom voice pads about 110-140 ms of leading silence (rohan about 230-320 ms); over 10 fricative-heavy starts x 2 voices it cut at most 12 ms of audio above 0.002 (one rohan start), 0 ms otherwise.
- Normalizer cost on the event loop: 0.4 ms for a 49-character sentence, 0.8 ms for 136 characters, 15 ms for 4000 characters.

### Not measured here

Any uncontended run of the above; concurrency above 5 for the custom voice; Kokoro/Supertonic/Qwen; GPU; real G.711/network codec; listening tests (brand spellings, infer-grid settings, 8 kHz naturalness); `--call-sim` at 10+ calls.

## 11. Audio quality and prosody, T6 (2026-10-02)

`python -m bench.audio_quality --out bench/results/audio_quality_{before,after}.json` renders 14 fixed sentences (one-word replies,
questions, long comma sentences, Hinglish, every punctuation class) through `tts.stream`/`tts.synthesize` for each Piper voice
(M4, idle apart from other agents' jobs; VITS is stochastic, so rows move a little run to run). Methods are in the script's
docstring: loudness = BS.1770-4 K-weighted gated LUFS (validated on a 997 Hz sine, -3.01 +-0.04), true peak = 4x oversampled,
silence threshold -50 dBFS, telephony compared against an FFT-resampled ideal of the same cached audio.
"Before" = the code at de40845; "after" = defaults below (`PAUSE_PLAN=""`, `FADE_MS=0`, `VOICE_GAIN_DB=""` reproduce "before").

What the measurement showed, and what was changed:

| | before | after |
|---|---|---|
| Median speech level per voice (custom / pratham / rohan / priyamvada), LUFS | -20.5 / -18.4 / -16.3 / -15.7 (4.8 LU apart) | -20.0 / -19.7 / -19.8 / -19.9 (0.3 LU apart) |
| True peak, worst utterance per voice, dBTP | -3.6 / -1.9 / -0.8 / -1.3 | -2.9 / -3.2 / -4.9 / -5.6 |
| Lead silence min across utterances, ms (target 30) | custom 5, pratham 0 (onset at sample 0) | custom 9, pratham 16: onset-in-first-ms remains where the model starts voiced, but never 0 for a padded chunk |
| Chunk edge, loudest first/last sample, dBFS | custom -73, pratham -43, rohan -54, priyamvada -51 | -84, -67, -65, -64 |
| Gap at a sentence seam, min / median / max ms (12 seams, 4 voices) | 94 / 190 / 245 | 287 / 310 / 319 |
| Gap at a colon/semicolon seam | 44 / 150 / 155 | 231 / 232 / 234 |
| Gap at an ellipsis seam | 103 / 110 / 169 | 388 / 409 / 415 |
| Gap at a mid-sentence phrase cut | 152 / 160 / 169 | 117 / 120 / 129 |
| Gap at a comma seam | 92 / 152 / 175 | 62 / 163 / 178 (see note) |
| Clipped samples, seam jump, seam spectral-flux spike, DC | 0, 0, 0, < 0.002 | unchanged |
| Chunk loudness spread inside one utterance, median | 0.7-1.5 dB | unchanged (a per-chunk gain would pump; not done) |
| 8 / 16 kHz: length ratio, passband error, level change | 1.000, 0.0 dB, +0.1 / 0.0 dB median | unchanged |
| 8 / 16 kHz: top-10 % band vs ideal (negative = rolled off, no aliasing) | -1 to -4 dB | unchanged |

Notes. Nothing needed fixing in telephony resampling (soxr HQ is transparent: passband 0.0 dB, no energy added near Nyquist),
in clipping (the existing soft limiter never engaged on Piper) or in seams (every seam sits in silence, so no clicks existed to
remove; the edge fade only matters for chunks cut while audible, seen on pratham). Gaps are measured at -50 dBFS while the plan is
applied at -40 dBFS (the trim threshold), so measured values sit slightly under the plan (sentence 320 -> 287-319); the spread
inside a class dropping from ~150 ms to ~30 ms is the gain. The comma row did not improve: the spread comes from soft onsets
between -50 and -40 dBFS in the next piece. The plan values are defaults chosen to keep the old median where it was reasonable
(comma, phrase) and to lengthen between-sentence gaps from the engines' 100-250 ms to ~320 ms; the length is a listening
decision, not something these metrics can confirm. No listening test was run.

Time to first audio (`tts.stream`, first byte, 20 warm repeats after 3 warm-ups, cache off, custom voice, native rate):

| | p50 | p95 |
|---|---|---|
| before | 33.3 ms | 43.6 ms |
| after (run 1, concurrent load from other jobs: load average 4.7) | 47.1 ms | 146.0 ms |
| after (runs 2 and 3, repeated) | 37.8 / 34.1 ms | 48.3 / 37.8 ms |

Target p50 < 100 ms holds. The added work per chunk is a few numpy passes over milliseconds of audio.

## 11. V7 evaluation (2026-10-02)

Harness `python -m bench.v7_eval` (docstring has the full contract), corpus `bench/corpus/hi_eval_v2.tsv` (258 rows, sha256 `814850bf...2f601`, [bench/corpus/README.md](../bench/corpus/README.md), not native-reviewed),
human test protocol [listening-test/v7/README.md](listening-test/v7/README.md). One JSON per run in `bench/results/v7_eval/`: CER/PER overall and per category with bootstrap CIs and the ASR model name, UTMOS as *predicted* MOS,
speaker similarity, TTFA/RTF p50/p95, RSS, audio duration, clipping, 8 and 16 kHz variants, environment, corpus and model sha256, git commit, seed; paired deltas when two systems are given (medium vs high: same corpus, same params, `--system med=...onnx --system high=...onnx`).
Synthesis goes through `app.services.tts.stream` on a real `PiperEngine` (voice-catalog pronunciation rules applied to text and reference); Piper's graph noise is seeded so reruns reproduce the audio. `--quick` = 32 stratified rows.

**V6 quick baseline** (`bench/results/v7_eval/v6_baseline_quick_small.json`, `hi_IN-custom-medium.onnx` sha256 e18a819a...6596e, Apple M4, seed 0):
ASR **faster-whisper small** (int8 CPU, beam 5, hi), **n = 32 rows**, 95% bootstrap CIs over rows.

| metric | mean [95% CI] |
|---|---|
| CER (vowel-aware) | 0.206 [0.160, 0.254] |
| PER | 0.168 [0.144, 0.191] |
| CER after 8 kHz / 16 kHz server resampler | 0.220 [0.174, 0.271] / 0.209 [0.162, 0.260] |
| UTMOS22 **predicted** MOS (English-trained, not a MOS) | 4.04 [3.93, 4.13] |
| speaker self-consistency (Resemblyzer, leave-one-out; no real reference clips were available, so NOT similarity to the target speaker) | 0.944 [0.937, 0.950] |
| TTFA p50 / p95 (64 warm streams, 1 worker x 4 threads, cache off) | 113 / 166 ms |
| RTF p50 / p95 | 0.036 / 0.041 |
| RSS after synthesis phase / process peak | 684 / 926 MB |
| clipping | 0 of 32 clips; 119 s audio, 0.087 s per character |

Per category CER (n tiny, indicative only): hindi 0.151 (12), questions 0.102 (3), expressive 0.133 (3), numbers 0.183 (4), pronunciation 0.198 (4), hinglish 0.429 (6; romanized rows and Latin words inflate CER, PER 0.211).
Wall time: quick run 259 s (whisper small, 3 ASR passes per row). The full corpus is ~8x the rows, so roughly 35 min with small and longer with large-v3-turbo (not measured).
The corpus is new, so these numbers are not comparable with section 10 (50-sentence v1).
Not claimed: naturalness (no listening test); overlap of v2 with the real IndicTTS training text (not present on this machine).
