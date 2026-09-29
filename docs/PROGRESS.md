# Progress checklist

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
- [ ] **Production voice fine-tune**: blocked on (a) authorized Hindi recordings or a downloaded CC BY 4.0 set and (b) an NVIDIA GPU. MPS measured at ~7.7 s/step, which is impractical

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
