# V7 roadmap (Hindi voice quality upgrade)

Started 2026-10-01. Living plan: each track updates its row when it lands. Results go to `docs/V7.md` and
`V7_IMPLEMENTATION_REPORT.md`; this file is the plan, not the evidence.

## Goal

Hindi that sounds like a person, not a robot: correct pronunciation, natural prosody, real (learned, not DSP) tone
variation, and two new real-speaker voices (`hi-IN-young-female`, `hi-IN-young-male`), while keeping the Piper CPU
fast path (TTFA p50 < 100 ms on the M4) and every existing API and voice id working.

## Starting point (V6, audited 2026-10-01)

- Serving: FastAPI, multi-engine (`piper`, `kokoro`, `supertonic`, `qwen3`), sentence-level streaming, EDF scheduler,
  telephony 8/16 kHz, 788 tests passing.
- Text: `text_normalizer.py` (numbers, currency, dates, phones, units, URLs), `hinglish.py` (romanized Hindi to
  Devanagari, brand respelling, small fixed persona-gender table), `lexicon_hi.tsv`, `indian_english.py`.
- Custom voice: Piper medium VITS, IndicTTS Hindi female 7.9 h (personal use only), init rohan, v6 final step 402504,
  `voices/hi_IN-custom-medium.onnx`. No listening test yet.
- Training: Kaggle T4 kernel with experiment isolation (`experiments/<id>/`), verified resume, sha256 read-back,
  LR anneal; local `training/train.py`; piper-tts training package has a non-strict `warmstart_ckpt`.

## Decisions

| Decision | Choice | Why | Rejected |
|---|---|---|---|
| Training data | AI4Bharat **Rasa** Hindi (CC-BY-4.0): female 27.05 h, male 23.78 h, 48 kHz, studio, labelled styles | Commercial licence, both genders, real expressive and conversational speech in one corpus | IndicTTS (personal-use only in our records); SYSPIN (read speech only, kept as fallback); IndicVoices-R (has age labels but crowd-sourced phone audio, minutes per speaker) |
| Tone/expression | Learned from Rasa style labels, exposed as Piper speakers (one speaker id per voice x style) | Real recorded emotion, zero inference cost, works in the existing ONNX path | DSP pitch/energy as "emotion" (forbidden); a GPU-only expressive model (breaks CPU objective) |
| Model | One Piper medium multi-speaker model, warm-started from v6 final (non-strict: speaker conditioning layers start fresh) | One training run serves both voices; reuses v6's Hindi text encoder | Two single-speaker runs (twice the GPU time we do not have) |
| Medium vs high | Run as a bounded experiment with equal data and step budget | Spec requires it; compute decides how far it goes | Assuming high is better |
| Age labels | `age_group` comes from documented evidence: dataset metadata if present, otherwise perceptual screening by listeners, stated as such | Rasa ships no speaker age | Inferring age from a voice id or pitch alone |
| Compute | Local M4 (MPS) for data prep, smoke runs, and as much training as speed allows; Kaggle T4 when the weekly quota resets; one Kaggle session at a time | Kaggle has ~30 min left this week | Paid GPUs (not authorised) |

## Tracks and file ownership

Each worker owns its files; shared files (`README.md`, `docs/PROGRESS.md`) are edited only at integration.

| Track | Scope | Owns |
|---|---|---|
| T1 Pronunciation engine | code-switch detection, lexical rules, phonological rules (schwa, nukta, nasals, conjuncts, ऋ), brand/name lexicon, English exceptions, phoneme-safe output; unit tests per category; `docs/pronunciation.md` | `app/services/pronunciation/`, `app/services/text_normalizer.py` (integration point only), `tests/test_pronunciation_*.py` |
| T2 Persona grammar | structured first-person agreement (`gender`, `persona`, `age_group`), quote/third-person safety, compat wrapper for `hinglish.apply_persona_gender` | `app/services/persona_grammar.py`, `app/services/hinglish.py` (wrapper only), `tests/test_persona_grammar.py` |
| T3 Dataset pipeline | quality gates (RMS, clipping, SNR, silence/speech ratio, duration, rate, ASR consistency, duplicates, speaker leakage, condition consistency), review metadata, dataset report, HF parquet ingestion with speaker/style/gender/age columns | `training/prepare_dataset.py`, `training/quality_gates.py`, `training/ingest_hf.py`, `training/audio_report.py`, `tests/test_prepare_dataset.py`, `tests/test_quality_gates.py` |
| T4 Voice metadata + multi-speaker serving | explicit voice catalog (gender, age_group, style, family, speaker_id), Piper `sid` selection, `/v1/voices` metadata, persona gender taken from metadata, legacy ids unchanged | `app/services/piper_engine.py`, `app/services/voice_catalog.py`, `voices/catalog.json`, `app/api/voices.py`, `tests/test_voice_catalog.py`, `docs/voices.md` |
| T5 Evaluation + listening test | versioned corpus v2 (>= 250 cases in the required categories), medium-vs-high harness (CER, PER, UTMOS, speaker sim, TTFA, RTF, RSS, duration, clipping, 8k/16k), reproducibility (corpus hash, seeds), blind A/B package | `bench/`, `docs/listening-test/v7/`, `tests/test_bench_*.py` |
| T6 Audio quality + prosody | lead/trail silence, loudness consistency, clipping, boundary fades, seams, telephony resampling, punctuation-driven pauses, prosody test categories; objective regression tests | `app/services/tts.py`, `app/services/audio_utils.py`, `tests/test_audio_quality.py`, `tests/test_prosody.py` |
| T7 Data + training (advisor) | Rasa download and preparation, multi-speaker training, Kaggle kernel changes, checkpoints, export, evaluation | `training/kaggle/`, `training/experiments/`, `voices/` |

## Gates (V7 definition of done, from the brief)

1. Pronunciation better on the expanded benchmark (PER/CER with CIs, same ASR, same corpus version).
2. V7 better than V6 in a blind native-listener test. **Needs people; cannot be done by the agent.**
3. `hi-IN-young-female` and `hi-IN-young-male` exist as trained voices from real recordings.
4. Existing voices and APIs unchanged. 5. Persona agreement robust (tests). 6. Benchmarks reproducible.
7. All tests pass. 8. Telephony stable. 9. Docs match code. 10. Data rights documented.
11. No quality claim from proxy metrics alone. "Production ready" only after gate 2.

## Status

| Track | Status | Notes |
|---|---|---|
| T1 | in progress | |
| T2 | in progress | |
| T3 | landed, untuned on real data | `training/ingest_hf.py`, `quality_gates.py`, gated `prepare_dataset`; docs/training.md section 2c. Thresholds need a first look at real Rasa audio |
| T4 | in progress | |
| T5 | in progress | |
| T6 | in progress | |
| T7 | blocked on dataset access | Rasa and IndicVoices-R are gated: the owner must accept the terms on Hugging Face |
