# Custom Hindi voice: training progress log

Goal: natural, human-sounding (not robotic) Hindi voice fine-tuned from a public dataset (IndicTTS Hindi female, 7.9 h) on top of
Piper `hi_IN-rohan-medium`. **Personal, non-commercial use only.** How it is run: [custom-voice-runbook.md](custom-voice-runbook.md).

Voice files for every milestone (private): https://huggingface.co/vishwajit76/custom-tts-hindi-train/tree/main/milestones
(`milestones/step_N/`). Samples of each are mirrored in `docs/samples/step_N/` (the copy there is the last one evaluated).

## How to read this table

- **Session**: `CPU` = the 4 vCPU container (long-train path, fallback). `Kaggle v1` = first Kaggle T4 run. `v4` and `v5` are two
  Kaggle sessions of the same kernel that ran concurrently and both wrote milestones to the same HF path. Later uploads
  overwrote earlier ones at the same step, so several steps have two rows (one per upload). **Step labels of v4 and v5 are
  not on one timeline** (v5 was re-trained, v4 ran ahead), so do not read the table as one monotonic curve.
- **Time**: UTC / IST (IST = UTC+5:30). "-" = not recorded. Early rows only have a date.
- **CER**: faster-whisper `small`, beam 5, `hi`, int8, `training.asr.cer`, 3 fixed sentences (s1/s2/s3) and their mean. **CER is
  noisy**: 3 short sentences, one small ASR model; differences under about 0.05 are noise, the same voice was scored 0.061 and 0.039
  at step 325000, and identical triples recur. It measures intelligibility, not naturalness. **No human listening test yet.**
- **val_mel / loss**: only when recorded (`train_mel` where noted); many rows have none because no log was retrievable.
- Steps are the checkpoint `global_step` (2 per batch, both GAN optimizers). TensorBoard's step column is offset (about 832.9k).

| Date UTC / IST | Step | Session | CER s1 / s2 / s3 | CER mean | val_mel / loss | Notes |
|---|---|---|---|---|---|---|
| 2026-09-29 (time -) | 0 | CPU | - | - | - | Dataset download and fine-tune setup started (dataset choice Rasa -> IndicTTS -> FLEURS; IndicTTS chosen). |
| 2026-09-29 (time -) | 310300 | CPU (last.ckpt, epoch 3191) | - | 0.163 | mel 0.547; loss_g 40.28, loss_d 1.83, kl 2.72, dur 1.35 | Milestone 1. About 6.85 s per log step. Samples: docs/samples/step_310300. |
| 2026-09-29 (time -) | 315000 | Kaggle v1 | - | 0.047 | - | Milestone 2, HF milestones/step_315000. last.ckpt was never uploaded by this run, so resume still came from 310300. s/step, losses, GPU name not recorded. |
| 2026-09-29 (time -) | 320000 | Kaggle (first run) | - | 0.102 | - | Milestone 3, first export. Superseded by the re-export below. |
| 2026-09-29 22:55 / 2026-09-30 04:25 | 320000 | Kaggle v5 (re-trained) | 0.000 / 0.105 / 0.115 | 0.074 | val_mel 0.448; loss_g 34.29 (heartbeat at trainer step 321860) | Re-uploaded. About 0.835 s per global step (1.2 steps/s, bs 24), Tesla T4; 'sps' in the heartbeat is seconds per step. |
| 2026-09-29 21:46 / 2026-09-30 03:16 | 325000 | Kaggle (first upload) | 0.000 / 0.105 / 0.077 | 0.061 | - | Overwritten by the 00:04Z upload below. |
| 2026-09-30 00:04 / 05:34 | 325000 | Kaggle v5 (re-trained) | 0.000 / 0.079 / 0.038 | 0.039 | - | Best CER so far, but see the CER caveat; this is within noise of 0.061. |
| 2026-09-29 22:55 / 2026-09-30 04:25 | 330000 | Kaggle (first upload) | 0.000 / 0.132 / 0.154 | 0.095 | - | Overwritten by the 01:13Z upload below. |
| 2026-09-30 01:13 / 06:43 | 330000 | Kaggle v5 (re-trained) | 0.000 / 0.105 / 0.115 | 0.074 | train_mel 0.430; loss_g 33.27 (heartbeat at trainer step 330600) | |
| 2026-09-30 00:04 / 05:34 | 335000 | Kaggle v4 session | 0.000 / 0.105 / 0.077 | 0.061 | - | Earlier copy was placed in `voices/hi_IN-custom-medium.onnx` (local server voice); later overwritten by v5 upload. |
| 2026-09-30 02:22 / 07:52 | 335000 | Kaggle v5 (re-trained) | 0.045 / 0.105 / 0.115 | 0.089 | - | Overwrote the 00:04Z upload. Samples in docs/samples/step_335000 are this one. |
| 2026-09-30 01:13 / 06:43 | 340000 | Kaggle v4 session (older concurrent session) | 0.000 / 0.132 / 0.077 | 0.070 | - | Overwritten by the 03:31Z v5 upload below; docs/samples/step_340000 now holds the v5 copy. |
| 2026-09-30 02:21 / 07:51 | 345000 | Kaggle v4 session (older concurrent session) | 0.000 / 0.132 / 0.077 | 0.070 | - | Samples: docs/samples/step_345000. |
| 2026-09-30 03:31 / 09:01 | 340000 | Kaggle v5 (re-trained) | 0.000 / 0.105 / 0.077 | 0.061 | train_mel 0.430; loss_g 33.69 (v5 heartbeat 04:19Z at trainer step 343440) | Re-uploaded, overwrote the 01:13Z v4 copy. Samples in docs/samples/step_340000 are this one. |
| 2026-09-30 03:30 / 09:00 | 350000 | Kaggle v4 session (older concurrent session) | 0.000 / 0.079 / 0.192 | 0.090 | - | Overwritten at 05:50 UTC by v5's own 350000 (no longer on HF). s3 worse than earlier rows (0.192), within the noise caveat. Samples: docs/samples/step_350000. |
| 2026-09-30 04:39 / 10:09 | 355000 | Kaggle v4 session (older concurrent session) | 0.000 / 0.158 / 0.077 | 0.078 | - | v4's last milestone. Samples: docs/samples/step_355000. |
| 2026-09-30 04:41 / 10:11 | 345000 | Kaggle v5 (re-trained) | 0.091 / 0.079 / 0.115 | 0.095 | - | Re-uploaded, overwrote the 02:21Z v4 copy. Samples in docs/samples/step_345000 are this one. |
| 2026-09-30 08:35 / 14:05 | 360000 | Kaggle v6 `hi_f-v6-0930T0731Z` (LR anneal, LR ~8.2e-5 at this step; ONNX commit 08:26Z) | 0.059 / 0.117 / 0.070 | 0.082 | val_mel 0.418 (heartbeat) | First annealed milestone. Samples: docs/samples/hi_f-v6-0930T0731Z/step_360000 (trainer wavs). 50-sentence small-ASR, 3 repeats vs v5 355000 (07:00Z, last constant LR): CER 0.154 vs 0.166 (paired delta -0.012, CI [-0.024, -0.000]), PER -0.011 (CI [-0.022, -0.001]), UTMOS +0.056 (CI [-0.014, 0.125], includes 0), speaker cos +0.005 (CI [0.002, 0.008]). CERs are borderline (CI edge at 0), single run, small ASR: suggestive, not proof. Details: bench/results/compare_355k_v6-360k_small.json. Note: folder milestones/step_355000 is v5's 07:00Z upload. |


**Provenance of the `milestones/step_N` folders on HF as of 2026-09-30 06:25 UTC** (ONNX upload time from the HF commit; two sessions wrote these names, so this, not the step
number, says which run a file came from): 315000 09-29 21:46, 320000 09-29 22:55, 325000 09-30 00:04, 330000 01:13, 335000 02:22, 340000 03:31 (v5), 345000 04:41 (v5),
350000 05:50 (**v5**: the v4 copy from 03:30 was overwritten after v4 ended at 05:10), 355000 04:39 (v4). Rows above that evaluated a folder before its overwrite describe the earlier copy.
`python -m bench.compare_checkpoints --list` prints this table live, and every new row records the ONNX sha256 and commit time. From the next kernel push on, milestones live in
`experiments/<id>/milestones/step_N/` (unique per run, never overwritten).

Note: v4 finished at 05:10 UTC / 10:40 IST; its final checkpoint (global_step 357212) is preserved as `runs/hi_f/v4_final_step357212.ckpt` on HF.
Open items: no listening test; per-sentence CER for the first three rows and losses for most rows were not recorded; run one
Kaggle session at a time from now on (runbook section 9).

## 50-sentence eval (from 2026-09-30)

Protocol, metrics and how to run it: [custom-voice-runbook.md](custom-voice-runbook.md) section 6.1 (`python -m bench.milestone_eval`). Fixed text
`bench/hi_eval_50.txt` (50 sentences, not in the training text), fixed params noise_scale 0.667 / noise_w 0.8 / length_scale 1.0, ASR = faster-whisper
int8 (`small` comparable across rows; `large-v3` is the stronger one). Raw rows: `bench/results/milestones.jsonl`; per-sentence details:
`bench/results/milestone_details/`. **Only compare rows with the same ASR.** Noise band (measured: three re-scores of 345000, small ASR: CER 0.1355 /
0.1411 / 0.1453): about 0.01 CER, so 340000 vs 345000 is a tie on every metric.

| Time UTC / IST | Step | Session | ASR (repeats) | CER mean / median / p90 | PER mean | UTMOS mean (min) | Spk-sim (real-vs-real 0.92) |
|---|---|---|---|---|---|---|---|
| 2026-09-30 03:53 / 09:23 | 340000 | v5 (corrected: the folder was overwritten by v5 at 03:31:33 UTC, before this eval; first published as v4) | small (x2) | 0.130 / 0.125 / 0.209 | 0.159 | 3.79 (2.39) | 0.908 |
| 2026-09-30 04:09 / 09:39 | 345000 | v4 (older concurrent session) | small (x2) | 0.141 / 0.134 / 0.217 | 0.179 | 3.84 (2.88) | 0.913 |
| 2026-09-30 04:25 / 09:55 | 340000 | v5 (corrected, same folder state as above) | large-v3 | 0.050 / 0.020 / 0.076 | 0.056 | 3.84 (2.49) | 0.906 |
| 2026-09-30 04:35 / 10:05 | 345000 | v4 (older concurrent session) | large-v3 | 0.047 / 0.024 / 0.080 | 0.062 | 3.89 (2.97) | 0.911 |

**CER metric fix (2026-09-30).** All CER values in this file, the 3-sentence table and every row of `bench/results/milestones.jsonl` up to now are **consonant-only** CERs: `training.asr._clean`
removed punctuation with the regex `[^\w]`, and Python's `\w` does not match Devanagari vowel signs or virama, so they were deleted from reference and hypothesis alike (`cer('की', 'कु')` was 0.0).
`cer` now keeps them (`tests/test_cer_marks.py`); the old definition survives as `cer(..., keep_marks=False)` / the `cer_legacy_skeleton` field. Re-scoring the stored first-repeat transcripts of the
four 50-sentence rows: large-v3 0.0497 -> 0.0507 (340000) and 0.0470 -> 0.0492 (345000), small 0.1237 -> 0.1571 and 0.1429 -> 0.1703. Comparisons inside the old rows remain valid (same definition), new rows
(`bench.compare_checkpoints`, new `milestone_eval` rows) are not comparable to them without the legacy column. PER (espeak phonemes) always included vowels and is unaffected. The dataset-prep
`--asr-validate` threshold (0.35) now sees slightly higher CERs and is therefore slightly stricter; the 0.35 was not re-tuned.

Reading the numbers:
- Small ASR sits at CER about 0.13 on this set, much of it Whisper's own errors; large-v3 gives about 0.05. The old 3-sentence CER (0.07) is not comparable.
- CER max about 0.70 is one sentence (#38): large-v3 writes "online order / delivery / upgrade" in Latin script. PER for it is 0.11. Sentence #30 (year "उन्नीस सौ अठासी") is
  penalised because the normalizer reads the ASR's "1988" as "एक हज़ार नौ सौ अट्ठासी". Both are evaluation artefacts, not voice errors. PER is the fairer number for those.
- UTMOS min 2.4-3.0 identifies the weakest sentences per checkpoint (details JSON), useful for listening. UTMOS is English-trained: relative use only.
- Speaker similarity 0.91 vs 0.92 real-vs-real ceiling: the voice is close to the dataset speaker; it will not show naturalness.
- No listening test yet: none of these numbers say the voice sounds natural.

**Repeat/CI comparison (2026-09-30, `bench.compare_checkpoints`, 3 repeats, small ASR, vowel-aware CER):** 340000 / 350000 / 355000 (all v5 uploads): CER 0.166 / 0.170 / 0.159, PER 0.178 / 0.177 / 0.173 (all CIs about +-0.02, paired deltas include 0),
UTMOS predicted MOS 3.75 / 3.80 / 3.87 (355000 vs 340000: +0.12, CI [0.04, 0.20]), speaker embedding cosine 0.915 / 0.914 / 0.922. Details and caveats: benchmarks.md section 10.

### Inference-parameter grid (milestone 350000, small ASR, 15 sentences x 2 repeats)

`python -m bench.infer_grid --hf-step 350000`, full table in `bench/results/infer_grid_350000.json`, audio for the top-5 and default in
`docs/samples/infer_grid/<config>/` (3 sentences each). Differences between single configs (CER 0.136-0.166) are inside the noise band (about 0.02 for 15 sentences);
only the marginal trends are informative:

| Factor | Value | mean CER | mean UTMOS |
|---|---|---|---|
| noise_scale | 0.5 / 0.667 / 0.8 | 0.142 / 0.147 / 0.156 | 3.91 / 3.81 / 3.74 |
| noise_w | 0.6 / 0.8 / 1.0 | 0.152 / 0.146 / 0.147 | 3.82 / 3.79 / 3.85 |
| length_scale | 1.0 / 1.1 | 0.147 / 0.150 | 3.82 / 3.82 |

Top by rank-sum of CER and UTMOS: ns0.5/nw0.6/ls1.0 (CER 0.140, UTMOS 3.94), ns0.5/nw1.0/ls1.1 (0.142, 4.01), ns0.5/nw0.8/ls1.0 (0.139, 3.88),
ns0.5/nw0.8/ls1.1 (0.138, 3.85), ns0.667/nw1.0/ls1.0 (0.136, 3.82). The default ns0.667/nw0.8/ls1.0 ranked 12th of 18 (CER 0.151, UTMOS 3.83).
**Recommendation (not applied; server defaults unchanged):** noise_scale 0.5, noise_w 0.8 to 1.0 (no reliable difference), length_scale 1.0 (1.1 gives nothing).
Lower noise_scale is the only consistent effect (cleaner, per the automatic metrics); the research note says it can also sound flatter, which these metrics cannot show:
listen to `docs/samples/infer_grid/ns0.5_nw1.0_ls1.1` vs `ns0.667_nw0.8_ls1.0` before adopting it. Per-request: `noise_scale`/`noise_w` in the voice `config.json` `inference` block.

### Learning rate (read from HF `runs/hi_f/last.ckpt`, 2026-09-30)

Seed checkpoint (global_step 310300, epoch 3191): generator and discriminator `lr = 1.5258e-4`, ExponentialLR gamma 0.999875, `last_epoch` 2165. piper-tts 1.8.0 never steps the
scheduler (manual optimization), so v1-v5 all trained at a **constant 1.5258e-4**. Also: HF `last.ckpt` is still this seed (the Kaggle 20-minute upload re-sent the
unchanged seed, see training/kaggle/README.md); the trained weights appear to exist only in the running Kaggle session until its deadline save (corrected 2026-09-30: HF `last.ckpt` did change, at 05:10 UTC v4 uploaded its final there (step 357212, now also `v4_final_step357212.ckpt`), and at 05:18 v5's periodic upload replaced it
with the stale 310300 seed; its sha256 25b56359... equals the seed's). Verified directly: `runs/hi_f/last.ckpt` global_step 310300, `v4_final_step357212.ckpt` global_step 357212, epoch 3347,
both optimizers still lr 1.5258e-4 and the schedulers still `last_epoch` 2165, i.e. 46.9k steps at exactly constant LR. Measured steps per epoch 300.7 (302 expected: 3607 train clips after piper's
10% validation split / 24 x 2 optimizers); the earlier 336 (54k steps for 160 epochs) was wrong, 160 epochs = 48.3k steps. Kernel default is now `ANNEAL_EPOCHS=150` (45k steps, one session).
Anneal implementation and its tests: runbook 6.4, `tests/test_lr_anneal.py`, `training/lr_dryrun.py`.
| 2026-09-30 09:37 / 15:07 | 365000 | Kaggle v6 `hi_f-v6-0930T0731Z` (LR anneal, LR 5.95e-5 at this step; ONNX commit 09:37Z) | 0.088 / 0.117 / 0.070 | 0.092 | val_mel 0.4195 (metrics.jsonl, step 364992; 360000: 0.4194) | Samples: docs/samples/hi_f-v6-0930T0731Z/step_365000 (trainer wavs). 50-sentence small-ASR, 3 repeats (bench/results/compare_355k_v6-360k-365k_small.json; note the 355000/360000 means differ slightly from the earlier run, synthesis noise). Means: CER 0.1605 / 0.1550 / 0.1528 (355k / 360k / 365k), PER 0.171 / 0.168 / 0.163, UTMOS 3.858 / 3.934 / 3.949. Paired vs 355000: CER -0.0077 (CI [-0.021, 0.004], includes 0), PER -0.0077 ([-0.023, 0.004], includes 0), UTMOS +0.091 ([0.026, 0.157], excludes 0), speaker cos +0.0056 ([0.002, 0.009], excludes 0). 360000 vs 355000: CER -0.0055 ([-0.019, 0.006]), UTMOS +0.076 ([0.032, 0.123]). 365000 vs 360000: point deltas only (the JSON keeps no per-sentence matrices, so no paired CI): CER -0.0022, PER -0.0052, UTMOS +0.015, speaker cos +0.0025, well inside the noise. Stopping rule 6.2 does not fire: only one of three milestones is available for the window, CER/PER still drift down within noise, no large-v3 or blind A/B done. |
| 2026-09-30 10:48 / 16:18 | 370000 | Kaggle v6 `hi_f-v6-0930T0731Z` (LR anneal, LR 4.32e-5 at this step; upload 10:48Z / 16:18 IST) | 0.029 / 0.150 / 0.047 | 0.075 | val_mel 0.4151 | Samples: docs/samples/hi_f-v6-0930T0731Z/step_370000 (trainer wavs). Legacy 3-sentence CER only (bench/results/legacy_3sentence.jsonl). The 50-sentence comparison was not run locally (about 10 h on this CPU); scoring moves in-kernel from the next Kaggle run (runbook 6.0). These milestones can be scored later via stored matrices if needed. |
| 2026-09-30 11:59 / 17:29 | 375000 | Kaggle v6 `hi_f-v6-0930T0731Z` (LR anneal, LR 3.08e-5 at this step; upload 11:59Z / 17:29 IST) | 0.059 / 0.150 / 0.070 | 0.093 | val_mel 0.4126 | Samples: docs/samples/hi_f-v6-0930T0731Z/step_375000 (trainer wavs). Legacy 3-sentence CER only (bench/results/legacy_3sentence.jsonl). The 50-sentence comparison was not run locally (about 10 h on this CPU); scoring moves in-kernel from the next Kaggle run (runbook 6.0). These milestones can be scored later via stored matrices if needed. |
| 2026-09-30 13:11 / 18:41 | 380000 | Kaggle v6 `hi_f-v6-0930T0731Z` (LR anneal, LR 2.19e-5 at this step; upload 13:11Z / 18:41 IST) | 0.088 / 0.150 / 0.116 | 0.118 | val_mel 0.4097 | Samples: docs/samples/hi_f-v6-0930T0731Z/step_380000 (trainer wavs). Legacy 3-sentence CER only (bench/results/legacy_3sentence.jsonl). The 50-sentence comparison was not run locally (about 10 h on this CPU); scoring moves in-kernel from the next Kaggle run (runbook 6.0). These milestones can be scored later via stored matrices if needed. |
| 2026-09-30 14:23 / 19:53 | 385000 | Kaggle v6 `hi_f-v6-0930T0731Z` (LR anneal, LR 1.59e-5 at this step; upload 14:23Z / 19:53 IST) | 0.029 / 0.167 / 0.093 | 0.096 | val_mel 0.4116 | Samples: docs/samples/hi_f-v6-0930T0731Z/step_385000 (trainer wavs). Legacy 3-sentence CER only (bench/results/legacy_3sentence.jsonl). The 50-sentence comparison was not run locally (about 10 h on this CPU); scoring moves in-kernel from the next Kaggle run (runbook 6.0). These milestones can be scored later via stored matrices if needed. |
| 2026-09-30 15:34 / 21:04 | 390000 | Kaggle v6 `hi_f-v6-0930T0731Z` (LR anneal, LR 1.13e-5 at this step; upload 15:34Z / 21:04 IST) | 0.059 / 0.150 / 0.070 | 0.093 | val_mel 0.4043 | Samples: docs/samples/hi_f-v6-0930T0731Z/step_390000 (trainer wavs). Legacy 3-sentence CER only (bench/results/legacy_3sentence.jsonl). The 50-sentence comparison was not run locally (about 10 h on this CPU); scoring moves in-kernel from the next Kaggle run (runbook 6.0). These milestones can be scored later via stored matrices if needed. |
| 2026-09-30 16:45 / 22:15 | 395000 | Kaggle v6 `hi_f-v6-0930T0731Z` (LR anneal, LR 8.07e-6 at this step; upload 16:45Z / 22:15 IST) | 0.059 / 0.150 / 0.140 | 0.116 | val_mel 0.4030 | Samples: docs/samples/hi_f-v6-0930T0731Z/step_395000 (trainer wavs). Legacy 3-sentence CER only (bench/results/legacy_3sentence.jsonl). The 50-sentence comparison was not run locally (about 10 h on this CPU); scoring moves in-kernel from the next Kaggle run (runbook 6.0). These milestones can be scored later via stored matrices if needed. |
| 2026-09-30 17:56 / 23:26 | 400000 | Kaggle v6 `hi_f-v6-0930T0731Z` (LR anneal, LR 5.87e-6 at this step; upload 17:56Z / 23:26 IST) | 0.059 / 0.133 / 0.023 | 0.072 | val_mel 0.4076 | Samples: docs/samples/hi_f-v6-0930T0731Z/step_400000 (trainer wavs). Legacy 3-sentence CER only (bench/results/legacy_3sentence.jsonl). The 50-sentence comparison was not run locally (about 10 h on this CPU); scoring moves in-kernel from the next Kaggle run (runbook 6.0). These milestones can be scored later via stored matrices if needed. |

**v6 finished** — 2026-09-30 18:32 UTC / 2026-10-01 00:02 IST: final checkpoint `experiments/hi_f-v6-0930T0731Z/checkpoints/final_step402504.ckpt` (step 402504, LR 5e-6 floor reached, sha256 read-back OK). Kaggle weekly GPU quota exhausted; training paused pending user decision. Candidate voices: v6 final (402504) and milestone 400000.
