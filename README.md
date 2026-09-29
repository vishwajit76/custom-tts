# custom-tts: self-hosted Hindi / Hinglish TTS for AI calling

Streaming Hindi and Hinglish text-to-speech server built for phone agents. It runs fully offline on CPU.

- **Engine:** Piper VITS on ONNX Runtime, ~63 MB per voice. The earlier Qwen3-TTS engine is still available
  (`ENGINE=qwen3`) for zero-shot cloning on a GPU.
- **Latency:** first audio 74 ms p50 / 140 ms p95; RTF 0.027 for a single stream on an Apple M4 CPU.
  With 20 simulated concurrent calls: 99 / 215 ms (full tables in [docs/benchmarks.md](docs/benchmarks.md)).
- **APIs:** WebSocket streaming with ordered queueing and instant cancel (barge-in); HTTP streaming; an
  OpenAI-compatible `/v1/audio/speech`. PCM s16le at 8/16/22.05/24/44.1/48 kHz, resampled with soxr HQ.
- **Text pipeline:** Hindi normalization for numbers (lakh/crore), ₹, dates, times (साढ़े दस बजे), phone
  numbers, ordinals, units and symbols. Romanized Hindi ("kya aap free hain?") and Indian names
  (Rahul, Lucknow) are converted to Devanagari; English words stay English.
- **Voice training:** dataset prep → Piper fine-tune → ONNX export ([docs/training.md](docs/training.md)).
- **Licensing:** read [docs/research.md](docs/research.md) before commercial use. The bundled Hindi voices are
  for evaluation; train your own voice for production.

## Quick start

```bash
cp .env.example .env                       # set API_KEYS
docker compose up --build                  # CPU image, voices baked in, offline at runtime
curl localhost:8000/health
```

Local (Python 3.12):

```bash
uv venv && uv pip install -r requirements-dev.txt
python scripts/download_voices.py          # hi_IN-rohan-medium -> models/piper/
uvicorn app.main:app --port 8000
python scripts/ws_client.py "नमस्ते, मैं आपकी कैसे मदद कर सकती हूँ?" --sample-rate 8000 --key $KEY
```

## API

Auth: `Authorization: Bearer <key>` (WebSocket: the same header, or `?api_key=`). An empty `API_KEYS` disables auth.

### WebSocket `/v1/audio/ws` (recommended for calls)

```jsonc
// client -> server (text frames)
{"type":"speak","id":"r1","text":"आपकी EMI ₹12,500 है।","voice":"default","speed":1.0,"sample_rate":8000,"frame_ms":40}
{"type":"cancel","id":"r1"}      // cancel one request (playing or queued)
{"type":"cancel"}                // barge-in: cancel everything
// server -> client
{"type":"start","id":"r1","sample_rate":8000,"encoding":"pcm_s16le"}
<binary PCM frames, frame_ms each>
{"type":"end","id":"r1","audio_ms":2140,"ttfa_ms":71}
{"type":"cancelled","id":"r1"} | {"type":"error","id":"r1","code":"overloaded|not_found|bad_request|queue_full|internal","message":"..."}
```

- Speak requests on one connection play **in order**, so send LLM output sentence by sentence as it streams.
  Up to 32 can be queued.
- Cancel is acknowledged in ~3 ms, and no frame of a cancelled request is sent after its `cancelled` event. The
  server sends faster than real time, though, so on barge-in the client must also **flush its own playback
  buffer**: up to a few seconds of audio can already be in it.
- `speed` 0.5–2.0 (native VITS length scale, no time-stretch artifacts). `frame_ms: 0` sends one frame per
  synthesized chunk.
- Backpressure: a slow reader pauses synthesis. A client that reads nothing for 10 s is disconnected so it
  cannot hold a slot forever.

### HTTP

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/audio/speech` | OpenAI-style body `{input, voice, speed, response_format: wav/pcm/mp3, sample_rate}`; full file |
| POST | `/v1/audio/speech/stream` | same body, chunked raw PCM s16le as it is synthesized; `X-Sample-Rate` header |
| GET | `/v1/voices` | installed voices |
| POST/DELETE | `/v1/voices` | reference-clip upload for zero-shot cloning (`ENGINE=qwen3` only) |
| GET | `/v1/models` | |
| GET | `/health` | no auth; 503 while loading; load snapshot (streams, queue depth, busy workers) |
| GET | `/metrics` | no auth; Prometheus text: streams, rejects, cancels, TTFA p50/p95/p99, RSS |

Every response carries `X-Request-ID` (sent or generated), and it appears in the JSON logs. At capacity
(`MAX_STREAMS`) HTTP returns `503` + `Retry-After: 1` and WebSocket returns an `overloaded` error.

## How it works

```text
text ─ normalize (Hinglish → numbers/₹/dates/times → abbreviations) ─ split: short first chunk (≤60 chars), then ≤120-char clauses
     ─ EDF worker pool (ONNX Runtime, shared session per voice) ─ trim model's ~250 ms lead silence ─ soxr resample ─ PCM frames
```

- **Time to first audio:** the first chunk is cut at a clause or word boundary (`FIRST_CHUNK_CHARS`). The model
  pads every utterance with ~250 ms of leading silence; it is trimmed to 30 ms (`LEAD_SILENCE_MS`).
- **Scheduling:** each chunk is queued with a deadline, the moment the audio already sent to that call runs out.
  Workers take the earliest deadline first, so a new call's first chunk jumps ahead of chunks that still have
  seconds of slack. `MAX_CHUNK_CHARS=120` bounds how long any single (non-preemptible) run blocks a worker.
- **Concurrency:** `WORKERS` threads × `THREADS_PER_WORKER` ONNX intra-op threads. Defaults come from the
  container's cgroup CPU quota: min(4, cpus) threads per worker, cpus/threads workers. `MAX_STREAMS` caps
  admission; a phrase cache (`CACHE_SIZE`) serves repeated prompts from memory.
- **One process per container.** The model lives in-process; scale out with replicas behind a load balancer.
  The rate limiter is in-memory per process.

## Engines and voices

`ENGINES` (comma list) loads several engines side by side; every API (`/v1/voices`, HTTP, WebSocket, `/demo`) sees
all their voices. Each voice in `/v1/voices` carries `engine`, `gender` (M/F) and a display `name`.

| Engine | Voice ids | Weights | License (code / weights) | Notes (27-sentence eval: Whisper PER, UTMOS) |
|---|---|---|---|---|
| `piper` (default) | `hi_IN-rohan-medium` (M), `hi_IN-pratham-medium` (M), `hi_IN-priyamvada-medium` (F) | ~63 MB each | GPL-3.0 / per MODEL_CARD | fastest; least natural (UTMOS 3.83–3.91). No voice is commercially clean: pratham, priyamvada CC BY-NC-SA data; rohan fine-tuned from a research-only base |
| `supertonic` | `supertonic:F1`–`F5`, `supertonic:M1`–`M5` | ~400 MB, 44.1 kHz | MIT / OpenRAIL-M, see below | UTMOS 3.98–4.22, best Hinglish; F3, F5, M4 most intelligible; M3 least natural (UTMOS 3.67) |
| `kokoro` | `kokoro:hf_alpha`, `kokoro:hf_beta` (F), `kokoro:hm_omega`, `kokoro:hm_psi` (M) | ~354 MB, 24 kHz | MIT / Apache-2.0 | most natural (UTMOS 4.13–4.33, hf_alpha best), PER ~0.055 |

Latency on an idle Apple M4 (10 cores), all three engines loaded, default `WORKERS=2 THREADS_PER_WORKER=4`,
`python -m bench.bench --concurrency 1,4` (13 sentences per client, results in `bench/results/eng-*.json`):

| Voice | TTFA p50 / p95, 1 stream | RTF p50, 1 stream | TTFA p50 / p95, 4 streams | RTF p50, 4 streams | underruns at 4 |
|---|---|---|---|---|---|
| piper rohan | 68 / 127 ms | 0.025 | 174 / 255 ms | 0.062 | 0 / 52 |
| supertonic F1 | 640 / 976 ms | 0.25 | 1499 / 2297 ms | 0.66 | 11 / 52 |
| supertonic F1, `SUPERTONIC_STEPS=4` (rejected: quality below Piper) | 298 / 472 ms | 0.12 | 752 / 1255 ms | 0.32 | 0 / 52 |
| kokoro hf_alpha | 789 / 1314 ms | 0.25 | 2588 / 4128 ms | 0.84 | 18 / 52 |

Supertonic and Kokoro cost about 10× Piper per second of audio. With realistic call pacing (`--call-sim 2`, cache off),
an M4 held about 4 Kokoro calls (TTFA p50 930 ms) and about 8 Supertonic calls at `SUPERTONIC_STEPS=6` (TTFA p50 779 ms) with
no underruns. 6 steps costs about 0.1 UTMOS. 4 steps is rejected: quality falls below Piper (docs/benchmarks.md §7). `WORKERS=4 THREADS_PER_WORKER=2`
helped Supertonic slightly (0 underruns at 4 streams) and made Kokoro worse.

```bash
uv pip install -r requirements-engines.txt
python scripts/download_voices.py hi_IN-rohan-medium supertonic kokoro   # -> models/{piper,supertonic3,kokoro}
ENGINES=piper,supertonic,kokoro uvicorn app.main:app --port 8000
```

- **Supertonic 3 license:** the weights are BigScience OpenRAIL-M. Commercial use and hosting as a service are
  allowed, but its Attachment A use restrictions apply to you and must be passed on to your users. For calling, the
  one that matters: output must not be placed "in any context ... without expressly and intelligibly disclaiming
  that the information and/or content is machine generated", so callers must be told they hear an AI; no
  impersonation without consent. It is not a default voice.
- English words inside Hindi: Kokoro's G2P runs on the server's single espeak-ng instance (Kokoro's own Hindi recipe)
  and, like Piper, speaks English runs with the Indian-English phoneme mapping (`INDIAN_ENGLISH`). Supertonic reads
  the raw text (Devanagari and Latin) itself; it says "approve" fine but "loan" comes out closer to "loon".
- `ENGINES=qwen3` (cloning) runs alone. The older `ENGINE=` variable still works.

## Configuration

See [.env.example](.env.example). The ones that matter: `API_KEYS`, `DEFAULT_VOICE`, `MODELS_DIR`, `MAX_STREAMS`,
`WORKERS`, `THREADS_PER_WORKER`, `FIRST_CHUNK_CHARS`, `MAX_CHUNK_CHARS`, `USE_CUDA`.

## Deployment

- `Dockerfile`: CPU, python:3.12-slim, non-root, voices baked in at build (`--build-arg VOICES="hi_IN-rohan-medium ..."`),
  `HF_HUB_OFFLINE=1`, health check, graceful shutdown. 820 MB image, ~230–470 MB RSS under load.
- `Dockerfile.gpu`: CUDA 12 + `onnxruntime-gpu` (`USE_CUDA=1`). **Untested**: no NVIDIA GPU was available here.
- `docker-compose.yml`: CPU service with CPU/memory limits, log rotation, restart policy; `--profile gpu` for GPU.
- Custom voices: mount a directory of `*.onnx` + `*.onnx.json` at `/srv/models/piper`.
- Sizing: one 4-vCPU container handles ~10 simulated concurrent calls with p95 TTFA under ~500 ms. For 200 calls,
  see [docs/benchmarks.md](docs/benchmarks.md#scaling-to-200-calls).
- License: `piper-tts` and espeak-ng are GPL-3.0. Fine for a hosted service; distributing the image to third
  parties requires offering source.

## Testing and benchmarks

```bash
pytest tests                                                   # normalizer, Hinglish, scheduler, HTTP + WS protocol
python -m bench.bench --concurrency 1,5,10                     # latency / RTF / underruns / cancel (server: CACHE_SIZE=0)
python -m bench.bench --concurrency 10,20,40 --call-sim 2      # realistic calls: playback + 2 s caller turns
python -m bench.bench --concurrency 10 --soak 600              # long-running reliability (drift, errors, RSS)
bench/matrix.sh "WORKERS=2 THREADS_PER_WORKER=4" -- --concurrency 1,5,10   # config sweeps
python -m bench.ort_variants                                   # ONNX Runtime session-option sweep
python -m bench.quality                                        # pronunciation: Whisper CER per category
```

Results are saved to `bench/results/`. Methodology and numbers: [docs/benchmarks.md](docs/benchmarks.md).
Progress checklist: [docs/PROGRESS.md](docs/PROGRESS.md).
