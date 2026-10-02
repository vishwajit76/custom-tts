# V7 implementation report

Report date: 2026-10-02 (IST). Branch: `v7` (not merged to `main`). This is an interim report: everything that does not depend
on trained V7 weights is implemented and tested; the young female and young male voices are **not trained yet** because GPU time
is exhausted until the Kaggle quota resets on 2026-10-03 00:00 UTC. Sections marked **pending** are filled in after training and
evaluation. Architecture and data details: [docs/V7.md](docs/V7.md). Plan and status: [docs/V7-ROADMAP.md](docs/V7-ROADMAP.md).

## V7 definition of done: where it stands

| # | Gate | State |
|---|---|---|
| 1 | Pronunciation materially better on the expanded benchmark | **partial**: espeak IPA errors 56 -> 4 of 197 probe words; audio-level CER/PER on corpus v2 pending a trained V7 model |
| 2 | V7 better than V6 in blind listening | **pending**: listening kit ready, no listeners yet |
| 3 | `hi-IN-young-female` is a real trained voice | **pending**: data ready (20.6 h, real recordings), training not started |
| 4 | `hi-IN-young-male` is a real trained voice | **pending**: data ready (17.7 h, real recordings), training not started |
| 5 | Existing voices keep working | **done**: legacy ids pinned by tests; pronunciation rules default off |
| 6 | Persona gender agreement robust | **done** within stated limits (tests/test_persona_grammar.py, 429 cases) |
| 7 | Benchmarks reproducible | **done** for the harness (hash-pinned corpus, seeds, recorded env); no V7 numbers yet |
| 8 | All tests pass | **done**: 1490 passed, 2 skipped, 4 xfailed; repeated clean full runs (exit 0) |
| 9 | Telephony stable | **done** for the pipeline (8/16 kHz length, level, aliasing within bounds); V7 voices not measured yet |
| 10 | Docs match implementation | **done** for landed code |
| 11 | Voice/data rights documented | **done**; the V7 weights inherit a non-commercial lineage (see Licensing) |
| 12 | No quality claim from proxy metrics alone | upheld: no "better" or "production ready" claim is made |

## What changed

- **Pronunciation engine** (`app/services/pronunciation/`): code-switch tagging, lexical respellings, names/brands, English
  exceptions, phonological rules (schwa deletion/retention, final clusters). Every lexicon row carries a category and a reason.
  Per-voice opt-in through the catalog; global default off.
- **Persona grammar** (`app/services/persona_grammar.py`): replaces the fixed 30-phrase table with morphology-driven, clause-bounded
  first-person agreement; optional `persona` object on `/v1/audio/speech`, `/v1/audio/speech/stream` and WS `speak`.
- **Dataset pipeline** (`training/quality_gates.py`, `prepare_dataset.py`, `ingest_hf.py`): loudness, clipping, SNR, silence and
  speech ratio, duration, speech rate, optional ASR consistency, exact and near duplicates, speaker leakage, recording-condition
  outliers; human review sidecar; JSON + Markdown dataset report; HF parquet ingestion with a data-rights entry.
- **Voice catalog + multi-speaker serving** (`voices/catalog.json`, `app/services/voice_catalog.py`): explicit gender, age group
  with required evidence, style, family, status, recorded styles, pronunciation rules; Piper `sid` routing; additive `/v1/voices`
  fields. Recorded emotions `angry`, `fearful`, `surprised`, `disgusted` added to the emotion enum (served only by voices that
  recorded them).
- **Audio + prosody** (`tts.py`, `audio_utils.py`): per-voice static gain, punctuation pause plan, edge-only fades; objective
  regression tests.
- **Evaluation** (`bench/corpus/`, `bench/v7_eval.py`, `docs/listening-test/v7/`): corpus v2, multi-metric harness with paired
  bootstrap CIs, blinded listening kit with sealed key and analysis.
- **Training infrastructure**: `train.py --warmstart` (single- to multi-speaker), Kaggle kernel `VOICE_NAME` / `INIT_MODE` /
  `VOCODER_WARMSTART` / `EXTRA_ARGS`, `push.sh --env/--slug`, a CPU Kaggle data kernel (`training/kaggle/datakernel/`).
- **Fixes found on the way**: a test built a second espeak instance and crashed the suite about 2 runs in 3 (SIGSEGV/SIGBUS);
  the shared rate limiter made combined suites return 429; a `data/` ignore rule hid the pronunciation TSVs from git.

## Files changed

63 files, about 12.9k lines added, versus `main` (`git diff --stat main..v7`). New modules: `app/services/pronunciation/*`,
`app/services/persona_grammar.py`, `app/services/voice_catalog.py`, `training/quality_gates.py`, `training/ingest_hf.py`,
`training/kaggle/datakernel/*`, `bench/audio_quality.py`, `bench/pronunciation_probe.py`, `bench/v7_eval.py`, `bench/corpus/*`.
New tests: `test_pronunciation_rules.py`, `test_persona_grammar.py`, `test_quality_gates.py`, `test_voice_catalog.py`,
`test_audio_quality.py`, `test_prosody.py`, `conftest.py`, and the T5 bench tests. New docs: `docs/V7.md`, `docs/V7-ROADMAP.md`,
`docs/pronunciation.md`, `docs/voices.md`, `docs/listening-test/v7/README.md`.

## Datasets used

| Dataset | Licence | Hours | Use |
|---|---|---|---|
| AI4Bharat Rasa Hindi, train | CC-BY-4.0 | 44.8 ingested, 38.2 accepted (F 20.6, M 17.7) | V7 training |
| AI4Bharat Rasa Hindi, test | CC-BY-4.0 | held out | evaluation of the trained voices only |
| IndicVoices-R Hindi | CC-BY-4.0 | 71.9 over 368 speakers | **rejected**: at most 0.37 h per speaker |
| IndicTTS Hindi female | personal use only (our records) | 7.9 | V6 only; reaches V7 through the warm-start |

Rejected by the gates: 3146 clips / 6.6 h (too long 5.1 h, too short, too little speech, condition outliers, clipping, duplicates).

## Training experiments

**Pending.** Planned: experiment A, Piper medium, 16 speakers (2 voices x 8 recorded styles), warm-started from v6 final; experiment
B, Piper high, same data, decoder from `cori-high` (public-domain data). Same seed, same corpus, same evaluation; one 11-hour Kaggle T4
session each. Measured feasibility: on the M4 GPU, medium trains at about 3 s per step, roughly 4x slower than the T4, and high about
25x slower again, so both run on Kaggle.

## Best checkpoints

**Pending** (no V7 training yet). V6 reference: `experiments/hi_f-v6-0930T0731Z/checkpoints/final_step402504.ckpt`, exported as
`voices/hi_IN-custom-medium.onnx` (sha256 `e18a819a...`).

## Benchmark tables

**Pending** for V7. V6 baseline (`bench/results/v7_eval/v6_baseline_quick_small.json`; `bench/v7_eval.py --quick`: corpus v2
sha256 `814850bf...`, 32 stratified rows, faster-whisper small, seeded VITS noise, 95% bootstrap CIs over rows, Apple M4):

| Metric | V6 `hi_IN-custom-medium` | V7 young female | V7 young male |
|---|---|---|---|
| CER | 0.206 [0.160, 0.254] | pending | pending |
| PER | 0.168 [0.144, 0.191] | pending | pending |
| CER 16 kHz / 8 kHz | 0.209 / 0.220 | pending | pending |
| Predicted MOS (UTMOS22, not human MOS) | 4.04 [3.93, 4.13] | pending | pending |
| TTFA p50 / p95 (harness: one scheduler worker, seeded graph copy) | 113 / 166 ms | pending | pending |
| RTF p50 | 0.036 | pending | pending |
| Clipped samples | 0 | pending | pending |

Per-category CER (n of 3-8 rows each, indicative only): Hindi 0.151, questions 0.102, expressive 0.133, numbers 0.183, pronunciation
0.198, Hinglish 0.429. Whisper small explains much of the CER level; compare systems only within one run of the harness. The harness
TTFA (113 ms) is higher than the T6 measurement (34-38 ms, `tts.stream` with the normal worker pool and an unmodified session); the
cause is not yet isolated, so the latency target is only claimed for the T6 setup.

Corpus v2 vs training text: no exact sentence overlap with Rasa train (`data/hi_v7`) or IndicTTS (`data/hi_f`); 12 and 2 rows
share a common 5-word phrase (e.g. "मैं आपकी क्या मदद कर सकती हूँ"), accepted and recorded here.

## Audio-quality findings

| Metric (14 sentences, 4 Piper voices) | Before | After |
|---|---|---|
| Speech level spread across voices | 4.8 LU | 0.3 LU |
| Worst true peak (rohan) | -0.8 dBTP | -4.9 dBTP |
| Loudest chunk-edge sample (pratham) | -43 dBFS | -67 dBFS |
| Sentence-seam gap min / median / max | 94 / 190 / 245 ms | 287 / 310 / 319 ms |
| TTFA p50 / p95, custom voice, warm | 33 / 44 ms | 34-38 / 38-48 ms |

Clipping, clicks at seams, DC offset and 8/16 kHz resampling were already clean. Pause lengths are design values, not yet
validated by listeners.

## Pronunciation findings

espeak-ng `hi`, 197 probe items: wrong 56 before, 4 after. Grammar words (है/हैं, था/थी, गया/गई/गए, ...) were already right except
यह, वह, चाहता, चाहती. Nukta, anusvara, chandrabindu and ऋ were already right. The main errors were schwa handling, names and brands, and
number/month words that the normalizer itself writes. Still wrong: कृपया, बेंगलुरु/Bengaluru, डिलीवरी. The probe measures IPA of
isolated words, not audio.

## Young female results

**Pending.** Data: Rasa Hindi female, 20.6 h accepted, 8 recorded styles (conversational 3.2 h, neutral 11.8 h, six emotions
0.87-0.96 h each). Age group: unspecified (no age metadata; listener screening required).

## Young male results

**Pending.** Data: Rasa Hindi male, 17.7 h accepted, 8 recorded styles (conversational 3.2 h, neutral 10.3 h, six emotions
0.61-0.74 h each). Age group: unspecified.

## Performance impact

Fast Piper path unchanged in kind: catalog lookup is one dict hit, pronunciation rules add about 0.04 ms per request when on,
persona grammar about 0.04 ms. TTFA p50 34-38 ms warm on the M4 (target < 100 ms). The multi-speaker medium model adds speaker
conditioning layers (`gin_channels` 512); its latency is measured after training. A high model, if it wins on quality, ships as a
separate tier.

## Licensing

Rasa is CC-BY-4.0 (attribution in docs/V7.md). The V7 weights are warm-started from v6 (IndicTTS personal use; rohan <- lessac,
research only), so **V7 voices are not cleared for commercial use** until retrained from a clean initialisation.

## Remaining limitations

- No native-listener evaluation of anything in V7.
- Voices untrained; emotion speakers have under 1 h each.
- Commercial lineage of the warm-start (above).
- Persona grammar: no subject ellipsis across conjunctions; closed adjective list.
- Four pronunciation probe items unfixable by respelling (need the Piper inline-phoneme path).

## Recommended V8 work

1. Clean-lineage V7: train from scratch on Rasa (or a cleared base) once GPU budget allows, so the voices can ship commercially.
2. Inline phonemes (`[[...]]`) on the Piper path for the words respelling cannot fix.
3. Native-speaker review of the pronunciation lexicon and the pause plan.
4. Age screening by listeners to set `age_group` with evidence.
5. More emotion data per voice (Rasa has about 1 h per emotion), or merge rarely used emotions.
