# V8 implementation status (Phase 0 audit)

Audit date 2026-10-04, branch `main` (V7 merged, commit 2374a26). Read-only audit; no V7 file changed.
Host: Windows 11, i5-11300H, RTX 3060 Laptop 6 GB (driver 555.97), Python 3.14.0 (system), docker CLI present, no espeak-ng/ffmpeg on PATH, no venv.

## 1. Existing architecture

| Area | What exists | Path |
|---|---|---|
| Engine interface (duck-typed, no ABC) | `ready`, `max_workers`, `load()`, `voices()`, `has_voice()`, `sample_rate(voice)`, `synth(text, voice, speed, ref, ref_text) -> np.ndarray`, `synth_native(..., controls, ...)`, optional `capabilities_for(voice)` | `app/services/tts.py` (`MultiEngine`, `_ENGINES`, `_make_engine`), `app/services/conditioning.py` (`EngineCapabilities`) |
| Engines | piper (VITS/ONNX), supertonic, kokoro (kokoro-onnx, Hindi hf_alpha/beta, hm_omega/psi), qwen3 (cloning, runs alone), expressive (off), goonj (experiment, needs `exp/goonj` + `bench.kokoro_pt`) | `app/services/{piper,supertonic,kokoro,qwen,expressive}_engine.py`; selected by env `ENGINES` |
| Normalization | Hindi number/date/currency/time/phone/unit expansion | `app/services/text_normalizer.py` (556 lines; `num_to_words`) |
| Pronunciation | code-switch detect, lexical respellings, phonological rules (schwa, clusters); per-voice opt-in via catalog, global default off | `app/services/pronunciation/{detect,lexical,phonological}.py`, `hinglish.py`, `indian_english.py`; `docs/pronunciation.md` |
| G2P | one shared espeak-ng instance under Piper's lock | `app/services/indian_english.py` (`espeak`) |
| Persona grammar | first-person gender agreement | `app/services/persona_grammar.py` |
| Voice registry | catalog (gender, age, style, family, status, rules) + speaker registry with consent (cloned voices) | `voices/catalog.json`, `app/services/voice_catalog.py`, `speaker_registry.py`, `speaker_encoder.py` |
| Streaming | sentence split + per-chunk synth via scheduler; HTTP stream and WS | `tts.py` (`split_for_stream`, `synthesize`), `app/api/{speech,ws}.py`, `scheduler.py`, `scripts/ws_client.py` |
| Audio conversion | soxr resample, trim/pad, gain, pause plan, fades, 8/16 kHz telephony | `app/services/audio_utils.py`, `tts.py`, `dsp.py` (optional prosody) |
| API | FastAPI: `/v1/audio/speech`, `/stream`, `/v1/voices`, speakers, health, `/demo` voice agent (vendor STT/LLM) | `app/main.py`, `app/api/*`, `providers.py`, `app/core/{config,limits,security}.py` |
| ONNX models present | one: `voices/hi_IN-custom-medium.onnx` (60.6 MB + `.json`, gitignored); catalog lists community Piper voices (rohan, pratham, priyamvada ...) fetched by `scripts/download_voices.py` | `voices/` |
| Training | Piper VITS: train, export ONNX, dataset prep + quality gates, HF ingest, Kaggle kernel, experiment ledger | `training/{train,export,prepare_dataset,quality_gates,ingest_hf,asr,split,manifest}.py`, `training/kaggle/`, `training/experiments/*.json` |
| Benchmarks | latency/concurrency (`bench.py`, `matrix.sh`), quality CER/PER (`quality*.py`), v7 multi-metric eval with bootstrap CIs, listening kit, pronunciation probe | `bench/*.py`, `bench/results/*.json`, `docs/benchmarks.md` |
| Tests | 38 files; V7 report claims 1490 pass / 2 skip / 4 xfail | `tests/` |
| Config | pydantic-settings, env-driven | `app/core/config.py`, `.env` (gitignored) |
| Docker | CPU `Dockerfile`; `Dockerfile.gpu` (CUDA 12.6 ubuntu24.04, self-described "UNVERIFIED ON GPU") | repo root, `docker-compose.yml` |

## 2. Local data and models (this machine)

| Item | Finding |
|---|---|
| `exp/`, `work/` | **Do not exist** in this checkout (gitignored; weights live elsewhere or were never copied). `dev/` is untracked and empty apart from a stray dir named `null`. |
| Local checkpoints | none (`*.ckpt`/`*.pt`/`*.safetensors` absent; `training/runs/` absent). Only `voices/hi_IN-custom-medium.onnx` (60.6 MB). |
| Local datasets | none (`/data/` gitignored, absent). `D:\AI Models` holds only LM Studio LLM files (irrelevant). HF cache holds only `gpahal/bge-m3-onnx-int8`. |
| Datasets used historically (per `V7_IMPLEMENTATION_REPORT.md`; not on disk) | AI4Bharat Rasa Hindi train, CC-BY-4.0: 44.8 h ingested, 38.2 h accepted (F 20.6, M 17.7); IndicTTS Hindi female 7.9 h (personal use only); IndicVoices-R Hindi 71.9 h rejected (<=0.37 h/speaker) |
| Eval corpus | `bench/corpus/hi_eval_v2.tsv` (258 rows, hash-pinned; id/category/subcategory/text/notes); also `bench/hi_eval_50.txt`, `bench/sentences.txt`, `bench/quality_set.tsv`, `bench/pronunciation_probe_items.tsv`, `tests/data/pronunciation_corpus.tsv` |
| V7 young female/male voices | **never trained** (report pending) |

## 3. Production-ready vs experimental

| Status | Components |
|---|---|
| Production-grade (tested, documented) | Piper engine + catalog, normalizer, pronunciation rules (opt-in), persona grammar, streaming/WS API, rate limits/auth, telephony resample, speaker registry w/ consent, dataset quality gates, eval harness |
| Working, optional | Kokoro (Hindi voices, ONNX), Supertonic, DSP prosody |
| Experimental | goonj engine (needs absent `exp/goonj`), Qwen3 cloning (NVIDIA GPU, own reqs), expressive engine (off), `Dockerfile.gpu` (never built), `/demo` agent |
| Unverified on this hardware | all GPU paths; all benchmark numbers (taken on a 4 vCPU Linux box) |

## 4. Reusable for V8

| Need | Reuse | Path |
|---|---|---|
| Engine contract + routing | `MultiEngine`, `_ENGINES` registry, `EngineCapabilities` | `app/services/tts.py`, `conditioning.py` |
| Text front end | normalizer + pronunciation + hinglish | `app/services/text_normalizer.py`, `pronunciation/`, `hinglish.py` |
| Voice metadata | catalog schema/loader (add engine/ref fields) | `voices/catalog.json`, `voice_catalog.py` |
| Cloning / refs | speaker registry + encoder | `speaker_registry.py`, `speaker_encoder.py` |
| Streaming + telephony | `split_for_stream`, scheduler, soxr 8/16 kHz | `tts.py`, `scheduler.py`, `audio_utils.py` |
| Kokoro | existing ONNX engine; `make_session` has CUDA EP option | `kokoro_engine.py`, `piper_engine.py` |
| Fallback | PiperEngine as-is | `piper_engine.py` |
| Evaluation | corpus v2, v7_eval (CER/PER/spk-sim/bootstrap), latency bench | `bench/corpus/hi_eval_v2.tsv`, `bench/v7_eval.py`, `bench/bench.py`, `bench/spk_sim.py` |
| Data pipeline | quality gates, HF ingest, data-rights | `training/quality_gates.py`, `ingest_hf.py`, `data_rights.py` |
| Template for a torch/GPU engine | lazy load, runs alone | `qwen_engine.py`, `requirements-qwen.txt` |

## 5. Broken / missing

| Item | Detail |
|---|---|
| Test env not runnable | no deps installed; `pytest` fails at conftest (`No module named 'fastapi'`). Python 3.14 is likely too new for `piper-tts==1.8.0`, onnxruntime, torch wheels; use a 3.11/3.12 venv. |
| Missing | IndicF5 engine; GPU-resident torch engine with VRAM budgeting; VoiceManager (registry split between catalog and speaker_registry); reference-audio library for IndicF5 (ref wav + exact ref text); `scripts/windows/*`; GPU benchmark harness; single-GPU concurrency test; V7 young voices; CER/MOS numbers for any non-Piper model |
| Stale | `Dockerfile.gpu` unverified; `goonj` hardcodes `exp/goonj`; docs state "no GPU" throughout |
| Licence risk | Piper community Hindi voices non-commercial (`docs/licenses.md`); IndicTTS data personal-use; check IndicF5 weights terms and Kokoro (Apache-2.0) before shipping |

## 6. Windows compatibility issues

| Issue | Where | Note |
|---|---|---|
| espeak-ng not on PATH | `indian_english.py`, `kokoro_engine.py`, piper | piper-tts wheels bundle espeak-ng data; kokoro-onnx dlopens its own libespeak-ng. Verify on Windows; V7 saw SIGSEGV with two espeak instances |
| Python 3.14 | system python | use 3.11/3.12 venv; multi-process `WORKERS` uses spawn on Windows, not fork |
| Shell-only tooling | `training/*.sh`, `bench/matrix.sh`, `training/kaggle/push.sh`, git hooks | need PowerShell equivalents; V8 requires `scripts/windows/*.ps1` |
| `onnxruntime-gpu` | `requirements.txt` pulls CPU onnxruntime via piper-tts | swap to GPU build; needs CUDA 12 + cuDNN 9 DLLs on PATH |
| Docker assumptions | `Dockerfile.gpu`: Linux ubuntu24.04, `/srv`, `/venv` | Windows would need Docker Desktop + WSL2 GPU; native Windows venv preferred |
| cwd-relative paths | `voices_dir=Path("voices")`, `exp/goonj` | run from repo root or make absolute |
| ffmpeg missing | PATH | needed only for IndicF5 ref-audio loading/conversion; install via winget |
| Line endings | `.sh` files | keep LF (`.gitattributes`) |

## 7. Proposed V8 architecture

V7 stays untouched. V8 engines go behind the existing duck-typed interface and register in `_ENGINES`.

| Component | Role | Notes |
|---|---|---|
| `IndicF5Engine` (`app/services/indicf5_engine.py`) | high-quality Hindi, reference-conditioned (flow-matching, torch, fp16 CUDA) | benchmark on 3060 first (VRAM, RTF, TTFA); one model instance + GPU semaphore; cached ref per voice; `cloning=True`; `max_workers` 1-2; sentence-chunked streaming via `split_for_stream` |
| `KokoroEngine` | low-latency Hindi (4 voices) | existing; add CUDA EP option; candidate default for live calls |
| `PiperFallbackEngine` | wrap existing `PiperEngine` unchanged | used when GPU engine unavailable/overloaded/VRAM pressure |
| `VoiceManager` (`app/services/voice_manager.py`) | `voice_id -> engine + ref audio + ref text + gender/age/style + fallback chain` | built on `voice_catalog.py` (+ `engine`, `ref_audio`, `ref_text`, `fallback` fields); `MultiEngine._owner` delegates to it; falls back on `Overloaded`/engine error |
| Front end | unchanged normalizer + pronunciation | per-engine text mode (IndicF5 Devanagari, Kokoro IPA) |
| Output | existing soxr 8/16 kHz path | |
| Eval | `bench/v7_eval.py` + corpus v2 for all candidates; add GPU latency/concurrency script | |

Next (Phase 1+): Python 3.11/3.12 venv + `scripts/windows/{setup,check-gpu,benchmark-gpu}.ps1`; get existing suite green on Windows; benchmark IndicF5 / Kokoro-GPU / Piper on corpus v2; then pick the training target.

## 8. Run existing tests

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest -q
```
Not run here: no deps installed, only Python 3.14. V7 baseline: 1490 passed, 2 skipped, 4 xfailed.

## Optimizer integration (2026-10-04, first milestone)

Spec: `docs/V8-OPTIMIZER-SPEC.md`. The optimizer wraps the existing pipeline; it never edits `app/` or retrains.

| Stage | Existing code | How the optimizer plugs in |
|---|---|---|
| Raw text -> spoken text | `text_normalizer.normalize(text, rules)` (numbers, currency, dates, time, phone, %, abbreviations, `hinglish.convert` for romanized Hindi/names, then `pronunciation.apply`) | `optimizer/core.py::frontend` runs, in order: active library rules (regex, `RULES`), then the optimizer dictionary `optimizer/pron_dict.json` (source token -> spoken form, word-bounded), then the untouched `normalize()` |
| Pronunciation respelling | `app/services/pronunciation/` (lexicon + schwa/cluster rules; groups via `rules=`; global default off) | `pron_rules` key in `optimizer/active_config.json`; `phonological.word` is a word-boundary candidate |
| Hinglish / Indian English | `hinglish.py` (roman -> Devanagari, `brand`), `indian_english.mark` (Piper) and `kokoro_engine.phonemes` (Kokoro, Indian-English IPA for Latin runs) | candidate spoken forms for Latin tokens (`brand`, `roman_to_devanagari`, letter spelling) |
| Chunking | `tts.split_for_stream` -> `text_normalizer.chunk`/`cut_at_phrase` | used as-is; optimizer adds per-category inter-chunk pause and punctuation variants (`comma_to_danda`, `conj_comma`) |
| Engines | `tts._ENGINES`; goonj = `bench.kokoro_pt.KokoroPTEngine("exp/goonj")`; Piper = `PiperEngine._run` | `optimizer/engines.py`: goonj on CUDA (subclass moves weights to GPU), Piper v7a/base on CPU; selected by `engines`/`primary_engine` in `optimizer/config.yaml`. Honored controls: text, `speed`, inter-chunk pause. Nothing else (no pitch/emphasis API in either engine) |
| Eval | `bench/v7_eval.py`, bake-off `asr_cer.py` (Whisper large-v3-turbo) | `optimizer/evaluators.py`: Text, ASR (Whisper turbo fp16 GPU, CER/WER/token/important-word acc, split detection), Audio (numpy), Laya |

Missing / known gaps found while wiring:
- **Laya** = convaiinnovations/laya (pip `laya` 0.3.26 installed; text-only System-1 classifier). Backend `laya.backend: laya` implemented against the package source API (`laya.load("ml")`, `predict_batch(states, questions)`; `score` quality ordinal blended at `laya.weight`, `noul` acceptability, `choice` best-pick + failure type; answers/confidence logged to `experiments.laya` and iteration reports). **Not yet run**: HF weight download (644 MB multilingual) fails here with SSL EOF / ~70 KB/s, so the default stays `deterministic`. Shipped multilingual checkpoint has no fitted temperatures; fine-tuning notebooks exist upstream (later step, not now).
- Naturalness/prosody/consistency are cheap audio proxies (speaking rate, pauses at punctuation, loudness), not MOS.
- Normalizer bugs surfaced by the independent benchmark (e.g. ISO date `2026-10-04` -> "दो हज़ार छब्बीस से दस-शून्य चार").
- No per-word stress/pitch control in either engine, so stress/emphasis candidates are not generated.
- Piper VITS is stochastic (ORT noise not seedable); audio is cached per text hash so unchanged cases score identically across runs. Kokoro is seeded per synth.
