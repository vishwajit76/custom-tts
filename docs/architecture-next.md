# Architecture audit (baseline for the next upgrade)

Audited at the checked-out revision on 2026-09-29 by reading code, tests and docs. Nothing here was re-benchmarked:
numbers are **prior recorded** values from `docs/benchmarks.md` (Apple M4, CPU only, no NVIDIA GPU was available).

## 1. Current architecture

- **Server:** FastAPI (`app/main.py`), one process per container. Routers: `health` (/health, /metrics, /v1/models),
  `speech` (`/v1/audio/speech`, `/v1/audio/speech/stream`), `ws` (`/v1/audio/ws`), `voices` (`/v1/voices`), `demo` (/demo voice agent
  using vendor STT/LLM keys from settings). Auth: `app/core/security.py` (API keys, in-memory rate limit).
- **Engines** (`app/services/*_engine.py`), selected by `ENGINES=piper,supertonic,kokoro` or `qwen3` (alone). Common
  duck-typed interface: `load, ready, voices, has_voice, sample_rate, synth(text, voice, speed, ref, ref_text), max_workers, supports_cloning`.
  `MultiEngine` (in `tts.py`) routes by voice id; ids must be unique across engines.
- **Pipeline** (`app/services/tts.py::stream`): normalize -> `split_for_stream` (short first chunk, then <=120-char clauses; unspeakable chunks merged)
  -> EDF scheduler (`scheduler.py`, deadline = when already-sent audio finishes playing) -> engine `synth` in worker threads
  -> `trim_lead` (~250 ms model silence to 30 ms) -> soxr `ResampleStream` (HQ) -> PCM s16le, optional fixed `frame_ms` frames.
  Phrase cache (`CACHE_SIZE`, key voice+text+speed; bypassed for cloning), admission limit `MAX_STREAMS` (503 / WS `overloaded`).
- **WS** (`app/api/ws.py`): ordered per-connection queue (32), per-id or global cancel (barge-in), slow-client drop after 10 s, typed error codes.
- **Text:** `text_normalizer.py` (numbers/lakh/crore, rupees, dates, times, phones, abbreviations), `hinglish.py` (romanized Hindi lexicon `lexicon_hi.tsv`),
  `indian_english.py` (Latin runs -> Indian-English phonemes for Piper/Kokoro).
- **Config:** `app/core/config.py` (pydantic-settings, cgroup-aware CPU count).
- **Schemas:** `app/models/schemas.py`. Note: this file was missing from the repo (`.gitignore` had `models/`, which matched `app/models/`); it was
  reconstructed from its call sites and `.gitignore` now anchors `/models/`.

## 2. Engine capability matrix (from code)

| | Piper | Supertonic 3 | Kokoro-82M | Qwen3-TTS |
|---|---|---|---|---|
| Runtime | ONNX Runtime CPU/CUDA | ONNX Runtime | ONNX Runtime (kokoro-onnx) | PyTorch (GPU needed) |
| Cloning | no | no | no | yes (reference audio + optional transcript; x-vector only without) |
| Emotion / style / role | none | none | none | none passed to the model |
| Pitch / energy | none | none | none | none |
| Speed | native length_scale | native, clamped 0.7-2.0 | native | librosa time-stretch |
| Max workers | `WORKERS` | `WORKERS` | `WORKERS` | 1 (not thread-safe) |
| Streaming | sentence/clause chunks | same | same | same |
| Native rate | per voice (22.05 kHz) | model rate | 24 kHz | 24 kHz |
| Hindi G2P | espeak-ng `hi` + Indian-English markup | reads Devanagari directly | misaki-style espeak IPA + Indian-English | model-internal |

Exposed as `EngineCapabilities` (`app/services/conditioning.py`), `/v1/voices`, `/v1/capabilities` (see docs/voice-system.md).

## 3. Voice inventory (17 preset + user clones)

- Piper: `hi_IN-rohan-medium` (M), `hi_IN-pratham-medium` (M), `hi_IN-priyamvada-medium` (F). Gender labels in `piper_engine.GENDER` (rohan from the model card dataset name; the other two by name/pitch). Any extra `*.onnx` in `MODELS_DIR` becomes a voice (multi-speaker models expand to `model:speaker`).
- Supertonic: `supertonic:F1-F5`, `M1-M5`. Kokoro: `kokoro:hf_alpha`, `hf_beta` (F), `hm_omega`, `hm_psi` (M).
- Qwen3: any WAV uploaded to `voices/` via `POST /v1/voices` (plus optional `.txt` transcript). No persistent registry, consent record or ownership yet.

## 4. Streaming pipeline

See section 1. Cancellation: cancelling the consuming task skips queued chunks; server acks in ms, but up to ~4.6 s of audio may already be client-side, so clients must flush their buffer (README).

## 5. Training pipeline

`training/` (Piper VITS only): `prepare_dataset.py` (layouts, denoise, trim, ASR segmentation, validation, splits, report) -> `train.py` (fine-tune, auto-resume, `--seed`, AMP `--precision`, `--warmstart-vocoder base`; gradient accumulation NOT wired) -> `export.py` (ONNX) -> drop into `MODELS_DIR`.
`training/smoke_test.sh` validates plumbing with synthetic data only. **No commercially clean production voice has been trained** (no authorized data); a personal-use custom Hindi voice has been training on a Kaggle T4 since 2026-09-29, see [custom-voice-runbook.md](custom-voice-runbook.md) (audited state at that date: none). Multi-speaker rows are supported by Piper; there is no emotion/style-conditioned training. (Other workers are editing `training/`; re-check before relying on this section.)

## 6. License notes (as found in the repo docs; see docs/licenses.md when it lands)

- Piper code GPL-3.0. Hindi voices: pratham/priyamvada data CC BY-NC-SA (not commercial); rohan fine-tuned from research-only lessac lineage (not clean). `docs/research.md` concludes none of the three is commercially clean.
- Kokoro: Apache-2.0 weights (README says MIT code); some training data is synthetic audio from unnamed closed TTS (provenance caveat).
- Supertonic 3: OpenRAIL-M; commercial use allowed but use restrictions must be passed on (disclose machine-generated speech; no deceptive use).
- Qwen3-TTS: license not audited in-repo; cloning must only use consented speakers.
- Never train on vendor-API audio (ToS, docs/research.md). SYSPIN / LIMMITS / IITM data reported CC BY 4.0 (LIMMITS unverified).

## 7. Prior recorded benchmarks (docs/benchmarks.md; NOT re-measured here)

Apple M4, CPU, `bench/bench.py` over WS, `CACHE_SIZE=0`.

- Piper rohan: single-sentence median 73-86 ms, RTF 0.022 (4 threads). TTFA p50/p95 68/127 ms at 1 stream, 174/255 ms at 4 streams (RTF 0.062, 0/52 underruns).
- Burst config sweep, 2 workers x 4 threads: TTFA p50/p95/p99 74/140/140 (1), 192/403/434 (5), 210/412/509 (10 streams); RTF p50 0.111 @10; RSS 1076 MB @10.
- Call-sim at 120-char chunks: 100/184/224 (10 calls), 99/215/259 (20), 134/277/310 (40, one underrun).
- Container 4 vCPU (Docker VM): 20 sim calls 183/950/1248 ms; per-core ~2-3x slower than native.
- Cancel ack 2.4-4.5 ms p50.
- Supertonic F1 8 steps: TTFA 640/976 ms (1 stream), RTF 0.25; Kokoro hf_alpha 789/1314 ms, RTF 0.25, ~2 real-time streams on the M4.
- Quality proxies (27 sentences, Whisper PER, UTMOS): Kokoro 4.13-4.33, Supertonic 3.67-4.22, Piper 3.83-3.91. Proxies only; no native-listener test exists.
- Not measured: any GPU, x86, Qwen at current revision (prior ~RTF 2), soak/final config (section 8 "pending"), 50/100/200 concurrent calls (extrapolated only).

## 8. Regressions to avoid

Piper fast path (TTFA ~70 ms, RTF 0.022); 4 intra-op threads default and cgroup-aware CPU count; 30 ms lead-silence trim; chunk cap 120 / first chunk 60; EDF scheduling and admission 503; WS cancel semantics and error codes; unchanged `/v1/audio/*` bodies and `X-Sample-Rate`/`X-Request-ID` headers; 24 kHz default rate; skipping unspeakable chunks; cache bypass for cloning; tests use fake engines (no models needed).

## 9. Missing capabilities, ranked

1. Any engine with real emotion/style/prosody control (none exists; all conditioning is currently rejected).
2. Persistent speaker registry with consent, ownership, retention, deletion (current cloning voices are loose files with no auth model beyond the API key).
3. Speaker embedding / encoder and few-shot cloning workflow; cloning is qwen3-only and runs alone (cannot mix with Piper).
4. Commercially clean expressive Hindi voice (all Piper Hindi voices are encumbered; no trained custom voice).
5. Routing across engines by capability/latency (MultiEngine routes only by voice id; `supports_cloning=False` on MultiEngine).
6. GPU path: `Dockerfile.gpu`/`USE_CUDA` untested; no GPU or 50-200-call measurements.
7. Human/objective eval: native-listener A/B, speaker similarity, emotion recognition; pronunciation corpus of 200+ reviewed cases.
8. Training: gradient accumulation, emotion/style labels, expressive multi-speaker architecture.

## 10. Implemented vs planned (final state, 2026-09-29)

| Item | Status |
|---|---|
| Piper/Supertonic/Kokoro serving, HTTP+WS streaming, EDF, cancel, metrics, Docker CPU | implemented, tested (fake-engine tests; benchmarks on M4 and on a 4 vCPU Xeon container, `benchmarks.md` sections 1-9) |
| Hindi/Hinglish normalizer, Indian-English phonemes | implemented; 326-row regression corpus |
| Qwen3 reference cloning | implemented, needs GPU; not exercised beyond capability/contract tests |
| Piper training/export pipeline | implemented, smoke-tested with synthetic data only; production fine-tune blocked |
| `VoiceCondition`, `EngineCapabilities`, validation, `/v1/capabilities`, headers | implemented, tested |
| Speaker registry + consent + secure deletion | implemented, tested (`tests/test_speakers.py`) |
| Speaker encoder | implemented (mfcc baseline; resemblyzer/speechbrain optional); informational only, never conditions synthesis |
| Model-native emotion/style/role synthesis | **blocked**: no engine supports it. Only the `ExpressiveEngine` boundary exists (`app/services/expressive_engine.py`), tested with a fake engine, disabled unless `EXPRESSIVE_ENGINE` is set |
| Expressive engine selection / bake-off | research done (`model-selection.md`); bake-off **not run** (needs GPU + weights) |
| DSP pitch/energy | implemented, opt-in (`DSP_PROSODY`), labelled `:dsp`, needs librosa (`requirements-dsp.txt`) |
| Policy routing (`routing_policy`) | implemented (`app/services/routing.py`), tested; tiers from config, not load-aware |
| 8/16 kHz telephony PCM + barge-in docs | implemented, tested |
| Eval + previews tooling | implemented (`bench/eval.py`, `bench/previews.py`); no human listening test |
| Baseline benchmark on this container | done for Piper rohan, concurrency 1/5/10/20 (`benchmarks.md` section 9) |
| GPU deployment validation, 50-200 call benchmarks | planned, blocked on hardware |

## 11. Docs vs code audit for P3-P7 (2026-09-30)

Every claim in `README.md`, this file, `voice-system.md` and `model-selection.md` about the serving path, text pipeline, speakers, telephony and real time was read against the code
and tests. Classes: **IMPLEMENTED** (code + a test that would fail without it), **PARTIAL** (works, but narrower than the doc wording), **UNVERIFIED** (code exists; nothing here
proves the quality/latency claim), **BLOCKED** (needs hardware/a model/people we do not have), **REGRESSION RISK** (a hot-path property that a careless change breaks).

| Claim | Class | Basis / what the doc should say |
|---|---|---|
| HTTP + WS streaming, EDF scheduling, admission 503 / WS `overloaded`, ordered WS queue, cancel + barge-in | IMPLEMENTED | `tests/test_robustness.py`, `test_streaming.py`. A stale `cancel{id}` used to poison a later request with the same id (fixed, `test_p6_production.py`) |
| "Streaming" | PARTIAL | sentence/clause chunking in the pipeline only; no engine streams inside a model call (precise wording: voice-system.md "Real-time behaviour") |
| Latency numbers in README (74/140 ms, RTF 0.027) | UNVERIFIED here | recorded on an Apple M4 on 2026-09-29; the custom voice was re-measured on this 4 vCPU box: benchmarks.md section 10 |
| Hindi/Hinglish normalizer, 326-row corpus | PARTIAL | IMPLEMENTED and tested, but every expected string is **machine-drafted, not native-reviewed**. Audit added 58 rows + `tests/test_normalizer_p3.py`; README's "English words stay English" is now "except a curated brand table respelled in Devanagari" |
| Brand/company pronunciation | IMPLEMENTED, listening UNVERIFIED | lexicon kind `name`; phoneme evidence + ASR proxy only (voice-system.md "Text normalization") |
| Speaker registry, consent, retention, secure delete | IMPLEMENTED | `test_speakers.py`, `test_p4_hardening.py`. Audit fixed: orphan files on failed upload/delete, lost concurrent bindings, revocation not durable, expiry only lazy |
| Speaker encoder | IMPLEMENTED (informational) | the default `mfcc` backend needed librosa, which `requirements.txt` lacked (was HTTP 500). Verification (accept/reject decision) is **not implemented**; neural similarity is uncalibrated |
| "Embedding conditions synthesis" (nowhere claimed, easy to assume) | not implemented | no engine declares `speaker_embedding`; Piper cannot consume one |
| Qwen3 cloning | PARTIAL / BLOCKED | code + contract tests only; never run (no GPU). Hindi is not in its documented language list (model-selection.md) while the engine passes `language="Auto"`: UNVERIFIED |
| Emotion / style / role synthesis | BLOCKED | no model; only the `ExpressiveEngine` boundary exists |
| DSP pitch/energy | IMPLEMENTED (opt-in), quality UNVERIFIED | labelled `:dsp`; no listening test |
| `routing_policy` | IMPLEMENTED | static tiers, not load-aware, no failover |
| 8/16 kHz PCM output | IMPLEMENTED | aliasing/seam tests, pause and mu-law tests added; intelligibility measured (benchmarks.md section 10) |
| G.711 / codec quality | not implemented | the gateway must encode; only a mu-law *quantisation* round trip is tested, labelled as such |
| API keys, per-key rate limit | PARTIAL | constant-time key compare; limit is per key, per process, in memory; counts HTTP requests and WS *connections*, not WS `speak` messages (concurrency is bounded by `MAX_STREAMS` and the 32-deep queue instead); failed auth is not throttled (use a gateway) |
| Body/upload limits | PARTIAL -> IMPLEMENTED | uploads were capped, the JSON speech routes were not (fixed: 413) |
| Path traversal | IMPLEMENTED | speaker ids strict; qwen voice ids `fullmatch`; `POST /v1/voices` did not validate its Form id (fixed, 422 not 500) |
| Custom voice selectable | was PARTIAL | only by pointing `MODELS_DIR` at it (or copying into `models/piper`); now `MODELS_EXTRA=voices` |
| Graceful shutdown | PARTIAL | `--timeout-graceful-shutdown 10` + compose `stop_grace_period 15s`; no drain mode (new streams are accepted until uvicorn stops), in-flight WS tasks are cancelled at exit |
| Memory over a 10-minute soak | see benchmarks.md section 10 | bounded structures: phrase cache (LRU), TTFA ring (2000), encoder cache (LRU 512), rate-limit deques (<= limit per key) |
| Piper fast path, 4 intra-op threads, 30 ms lead trim, 60/120 char chunking, EDF, cache bypass for cloning | REGRESSION RISK | unchanged by this audit; `trim_lead` measured not to cut speech onsets (benchmarks.md section 10) |
