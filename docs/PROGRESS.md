# Progress

## Upgrade status by phase (next-upgrade plan, final state 2026-09-29)

Status: **done** = implemented and covered by tests, **partial** = implemented with a stated limit, **blocked** = cannot be
done without something we do not have. "Evidence" names the test file / doc; no GPU and no listeners were available.

| Phase | Status | Evidence | What is missing |
|---|---|---|---|
| 0 Audit + baseline | done | `architecture-next.md`; baseline run in `benchmarks.md` section 9 (this 4 vCPU box, Piper only) | no GPU, no 50-200 call runs, no Qwen/Kokoro/Supertonic numbers on this box |
| 1 Conditioning API + capabilities | done | `tests/test_conditioning.py`, `/v1/capabilities` | none |
| 2 Speaker registry + consent | done | `tests/test_speakers.py` | consent evidence is not verified by the service (operator's job) |
| 3 Speaker encoder + cloning | partial | `tests/test_speaker_encoder.py`; qwen3 cloning path | default `mfcc` encoder is not neural; no engine consumes an embedding; cloning needs a GPU (never run here) |
| 4 Expressive model research | done (research only) | `model-selection.md`, `licenses.md` | bake-off not run: needs GPU and HF weights; licences from HF cards not re-read |
| 5 Emotion / style / role / prosody | **blocked** for native emotion/style/role; partial for prosody | `tests/test_expressive_dsp.py` (fake native engine proves the plumbing and the capability gate); DSP pitch/energy | no engine supports emotion/style/role, so they are rejected/ignored on all real engines. DSP pitch/energy is opt-in (`DSP_PROSODY`), labelled `:dsp`, never emotion; its perceptual quality was not listened to |
| 6 Hindi/Hinglish quality | done | `tests/test_pronunciation_corpus.py` (326 rows), normalizer tests | no native-listener review |
| 7 Training/data pipeline | done (plumbing) | `tests/test_training_pipeline.py`, `test_prepare_dataset.py` | no commercially clean fine-tune (no authorised data); a personal-use voice is training on Kaggle, see the 2026-09-30 update |
| 8 Serving / telephony | done | `tests/test_telephony.py` (8k/16k PCM s16le, alias rejection, seam-free streaming resampler); barge-in docs | streaming is sentence-level, not model-internal; no G.711 encoding; GPU/ONNX-FP16 export unverified |
| 9 Previews + evaluation | done (tooling) | `bench/eval.py`, `bench/previews.py`, `tests/test_bench_eval.py`; sample run in `benchmarks.md` | predicted MOS != human MOS; CER only if a local Whisper is cached; no listening test |
| 10 Routing, API tests, deployment | done / partial | `tests/test_routing.py`, `tests/test_robustness.py`; CPU Dockerfile unchanged | routing tiers are config, not load-aware; GPU Dockerfile unverified; Docker not built in this environment |

Bug found and fixed while measuring (Phase 0 baseline): on Python 3.11, `asyncio.wait_for(ws.send_bytes(...))` in `app/api/ws.py`
swallowed a barge-in cancel that arrived right after a frame was sent, so the request ran to completion (measured: cancel ack
about 850 ms and 23 s of audio sent after the cancel). Replaced by `asyncio.timeout`; re-measured in `benchmarks.md` section 9.
The Docker image uses Python 3.12, where `wait_for` no longer has this race.

---

# Earlier progress checklist (original ten-step plan)

Mirrors the ten execution steps in [plan.md](plan.md). `[x]` = done and verified this session; `[~]` = implemented,
verification limited (reason given); `[ ]` = not done (blocked, reason given).

## 1. Research and licenses: [research.md](research.md)
- [x] Candidates surveyed: Piper, AI4Bharat Indic-TTS, IndicF5, Indic-Parler, Kokoro, Veena, MMS, Qwen3 (existing)
- [x] Licenses verified from primary sources; Piper Hindi voices flagged (2 non-commercial, 1 unresolved)
- [x] Commercially clean path identified: Piper `_base_model` (CC BY 4.0) + own / SYSPIN / LIMMITS data
- [x] Vendor-API distillation evaluated and rejected on ToS grounds (10 vendors checked)
- [~] IndicTTS license.pdf could not be retrieved (site timeouts): rohan voice status stays "unresolved"

## 2. Baseline
- [x] Piper hi_IN voices running; Hindi + Hinglish phonemization checked (espeak switches to English phonemes for Latin words)

## 3. Benchmark latency / quality: [benchmarks.md](benchmarks.md)
- [x] Piper vs Kokoro (fp32/int8) vs Qwen3 on the same CPU
- [x] Whisper-CER pronunciation eval across voices and text categories (`bench/quality.py`)

## 4. Optimize inference
- [x] ONNX Runtime thread layouts (1/2/4 intra-op × workers): 4 threads = 2.3× lower latency, same throughput
- [x] Memory arena on/off (off *raises* RSS), mem-pattern: keep defaults
- [x] INT8 dynamic quantization: rejected (MatMul-only: no gain; full: 1.8–4× slower)
- [x] CoreML EP (Apple GPU/ANE): rejected (slower than CPU)
- [~] FP16 / CUDA EP: `Dockerfile.gpu` + `USE_CUDA` written, **untested** (no NVIDIA GPU)
- [x] Warm-up, persistent sessions, sentence/clause chunking, lead-silence trim, EDF scheduling, chunk-size cap
- [x] Dynamic batching: evaluated, not implemented. VITS output needs per-item trimming, and batching adds queueing delay on CPU; parallel sessions already saturate cores

## 5. Hindi / Hinglish text
- [x] Numbers (lakh/crore), ₹/Rs/INR/$, dates, times incl. `5.30 बजे`, phone numbers, ordinals, units, ranges, symbols, `24/7`, डेढ़/ढाई/साढ़े
- [x] Romanized Hindi → Devanagari (lexicon + rules, English kept), Indian names/places
- [x] Speed control (native length scale), voice selection
- [x] Expressive/emotion controls: not added. Piper VITS has no emotion conditioning, and the plan asks for them only where the model supports it

## 6. Streaming API
- [x] WebSocket: ordered queue, per-request/all cancel, request ids, framing, backpressure, slow-client drop
- [x] HTTP streaming + OpenAI-compatible endpoint; 8/16/22.05/24/44.1/48 kHz via soxr stream resampler
- [x] Graceful errors (typed error codes, overload 503/Retry-After), structured logs with request ids

## 7. Concurrency
- [x] Bounded worker pool, EDF scheduling, admission limit, per-connection isolation, `/metrics` + `/health` load snapshot
- [x] 1 / 5 / 10 concurrent streams benchmarked; call simulation at 10 / 20 / 40; container at 4 vCPU
- [~] 200-call sizing extrapolated from measured per-core capacity (no multi-node hardware here)

## 8. Tests
- [x] `pytest tests`: normalizer, Hinglish, lexicon hygiene, EDF scheduler, HTTP + WS protocol (fake engine)
- [x] Benchmarks: TTFA p50/p95/p99, RTF, underruns, cancel ack, RSS, soak mode; results in `bench/results/`
- [x] Soak run completed (see benchmarks.md)

## 9. Deployment
- [x] CPU Docker image built and run: health, auth, synthesis, cgroup-aware CPU defaults verified
- [x] Compose with limits / logging / restart; `.env.example`; offline voices baked into the image
- [~] GPU image: written, not built (no NVIDIA hardware)

## 10. Training pipeline: [training.md](training.md)
- [x] Dataset prep: layouts, speakers, denoise, trim, segmentation (+ASR), normalization, validation, splits, report
- [x] Train with checkpoints + auto-resume; export to ONNX voice; smoke test passes end to end
- [~] **Production voice fine-tune**: personal-use custom Hindi voice now training on a Kaggle T4 (see the 2026-09-30 update and [custom-voice-runbook.md](custom-voice-runbook.md)); a commercial voice stays blocked on (a) authorized Hindi recordings or a downloaded CC BY 4.0 set and (b) an NVIDIA GPU. MPS measured at ~7.7 s/step, which is impractical

## 11. Human-like, multi-voice push (Sep 29): [research.md](research.md), [benchmarks.md §7](benchmarks.md)
- [x] Research: 6 dimension reports + 17 primary-source license/speed verifications, written up in research.md (CPU tier vs GPU tier, voice inventory, fine-tune path, eval method, open questions)
- [x] License finding: none of the 3 Piper Hindi voices is commercially clean (pratham/priyamvada CC BY-NC-SA data; rohan fine-tuned from research-only lessac). training.md no longer recommends `--init rohan` for commercial voices
- [x] Hands-on quality eval on the same 27 sentences: Kokoro most natural (UTMOS 4.13-4.33 vs Piper 3.83-3.91); Supertonic 3 best on Hinglish; SYSPIN drops English words (rejected); Pocket TTS Hindi ships no licensed voice (skipped)
- [x] Multi-engine server: `ENGINES=piper,supertonic,kokoro` serves 17 voices (8F, 9M) side by side on every API and the demo dropdown; routing tests; Kokoro gets Indian-English phonemes for Latin words
- [x] Live latency per engine at 1 and 4 streams: only Piper meets 200 ms TTFA at 4 streams on the M4; Kokoro holds ~2 real-time streams; Supertonic at 4 steps is the middle ground
- [x] Demo page reviewed (10 findings) and fixed: loopback-only without API keys, max_tokens / turn deadline / turn-rate cap, API key redacted from uvicorn logs, error scope, history = what the caller heard, 300 ms barge-in threshold
- [x] VoiceStudio (AGPL-3.0) studied for ideas only; adopted: skip unspeakable chunks. Noted: Devanagari-safe word boundaries, revision pinning
- [x] Demo end to end with real vendor keys (scripted, no mic):
  - typed turn with OpenAI and Kokoro: 2.6 s to first audio, most of it LLM first token (1.7 s);
  - WAV turn with Sarvam STT, Gemini and Supertonic F3: 2.3 s (exact transcript in 769 ms).
- [x] The LLM system prompt now carries the voice's gender, because Hindi verbs agree with the speaker ("करती/करता हूँ"). Before, the female voice said "कर रहा हूँ".
- [~] Demo with a real microphone and Firefox/Safari: not exercised
- [~] Docker build with `REQUIREMENTS=requirements-engines.txt`: not built
- [ ] Blind native-listener A/B (the only real test of "sounds human"): needs listeners
- [ ] GPU tier (Chatterbox Hindi pack / Magpie) bake-off and custom-voice fine-tune: need an NVIDIA GPU and consented recordings

## Update 2026-09-29 — verification pass

- Phase 0: Kokoro hf_alpha / Supertonic F3 measured on 4 vCPU (benchmarks §9b): 1 stream RTF 0.30 / 0.28, TTFA p50 1.07 s / 0.74 s; at 5 burst streams both past capacity (RTF ≈1.3, 27/65 requests underran). Single runs on a shared CPU — indicative only. Piper remains ~7× faster.
- Phase 3: neural encoder measured (resemblyzer 0.1.4, Apache-2.0 code; weight/data provenance unknown): 31 ms/embed; same-voice cosine mean 0.935 vs different-voice 0.643 (mfcc: 0.99 vs 0.96). Thin min/max margin — not a calibrated identity threshold. Still informational only.
- Phase 7: `piper.train` flags verified against live `--help` (piper-tts 1.8.0). `accumulate_grad_batches` fails at runtime (manual optimization) and is now rejected by `train.py`. CPU 2-step train + `--init` resume smoke passed on synthetic data (plumbing only). HF checkpoints and 16-mixed AMP unexercised.
- Phase 9: CER runs locally via `bench/quality_fw.py` (faster-whisper small): mean CER Piper 0.309 / Kokoro 0.270 / Supertonic 0.295 — small-model ASR noise dominates; not comparable to §7. UTMOS blocked (torch.hub 403).
- Security review: 13 fixes with regression tests (`tests/test_review_fixes.py`) — decode bombs, consent race, cross-owner legacy voice overwrite, DSP blocking the event loop, annotate CSRF/DNS-rebinding, Devanagari split key, rights validation, cloning consent for bound voices, retention purge of engine copies, upload body cap.
- Phase 10: Docker not built — CLI present, no daemon in this environment.

## Update 2026-09-30 — custom Hindi voice training phase (in progress)

Status: **in progress, not finished, not validated by listeners.**

- What runs: personal, non-commercial Piper fine-tune on IndicTTS Hindi female (7.9 h), init from Piper `hi_IN-rohan-medium`
  (step 309852), on a Kaggle Tesla T4 (batch 24, fp16-mixed, about 1.2 global steps/s (0.83 s per step), 11 h runs, checkpoint to a private HF repo every 20 min,
  milestone ONNX every 5000 steps). Operating guide: [custom-voice-runbook.md](custom-voice-runbook.md); results:
  [training-progress.md](training-progress.md).
- Progress: milestones exported from step 310300 (CPU) through 345000 (Kaggle); the earlier "blocked on GPU/data" status in the training
  section above is superseded for this personal voice (authorized commercial data is still missing, so no commercial voice).
- Quality evidence is weak: CER (faster-whisper small, 3 sentences) fell from 0.163 (310.3k) to about 0.04 to 0.10 and then shows no trend; it is
  noisy (the same step scored 0.061 and 0.039 on re-upload). **No human listening test has been done.** Predicted MOS (UTMOS) was not run.
- Known process problem: two concurrent Kaggle sessions (v4, v5) overwrote each other's HF milestones, so step labels from the two
  are not on one timeline. Fix is procedural: one session at a time.
- Tokens used during setup were pasted in chat and should be rotated (runbook section 3).
- The CPU long-train scripts remain as a fallback only (the container pauses when idle).

## Review & hardening pass — 2026-09-30 (08:30 UTC / 14:00 IST)

**Training / evaluation (P0–P2)**
- Critical: piper-tts 1.8.0 never steps its LR scheduler → v4/v5 ran at constant LR 1.5258e-4 for ~47k steps (verified in source + checkpoint). Kernel now anneals 1e-4 → 5e-6 over 150 epochs (~45k steps); `tests/test_lr_anneal.py`.
- Critical: stale-seed checkpoint uploads (version_0/version_1 path bug) and v4/v5 milestone name collisions (v5 overwrote v4's 350000/355000 ONNX). Fixed: experiment-scoped artifacts `experiments/<id>/`, create-once upload guard, explicit verified resume (`RESUME_FROM`), sha256 read-back.
- Critical: CER ignored Devanagari vowel signs (`\w`), so all earlier CERs were consonant-only; fixed, PER unaffected.
- Reproducibility: manifest/environment/metrics/val_split per experiment; records in `training/experiments/`; seed 1234 fixed (also drives the val split).
- `bench/compare_checkpoints.py` (repeats, bootstrap CIs, paired deltas). 340k/350k/355k (all v5): CER/PER indistinguishable; 355k UTMOS +0.12 and speaker cosine +0.007 (CIs exclude 0). Blind A/B kit in `docs/listening-test/` — no listening results yet.
- v6 launched 07:31 UTC / 13:01 IST from v4 final (357212) with anneal; verified resumed, LR decaying, val_mel 0.419 → 0.415, checkpoint uploads OK. Kernel CPU smoke 19/19; `lr_dryrun` scenarios B–D not completed.

**Server (P3–P7)** — tests 575 → 777 passed
- Hinglish normalizer: AI/ai ambiguity, lakh/crore, URLs/emails, am/pm, phone grouping, glued tokens, brand respelling (Devanagari route better CER on both voices; WhatsApp exception). Corpus 326 → 384 rows (not native-reviewed).
- Speaker registry: orphan-file rollback, retryable delete, durable revocation, atomic engine bind, no mixed-backend embeddings, librosa-missing 500 fixed, ingest off the event loop; similarity labelled `mfcc_statistics_cosine` / `neural_embedding_cosine` (no verification).
- Production: JSON speech body cap (413), voice_id validation, WS stale-cancel fix, periodic retention sweep, `MODELS_EXTRA=voices` exposes `hi_IN-custom-medium`.
- Telephony CER (custom voice, whisper-small, n=20): 22.05k 0.300 / 16k 0.311 / 8k 0.329 / 8k+μ-law 0.340 (quantisation only, not a network codec). 10-min soak: 0 errors, flat RSS; cancel ack ≤3.4 ms. Latencies measured under contention (upper bounds).

**Unverified / blocked:** anneal's audible effect; listening tests; native corpus review; Qwen3 cloning (no GPU); real codec/network; uncontended latency for the custom voice.
