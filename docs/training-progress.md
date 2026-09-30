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
| 2026-09-29 22:55 / 2026-09-30 04:25 | 320000 | Kaggle v5 (re-trained) | 0.000 / 0.105 / 0.115 | 0.074 | val_mel 0.448; loss_g 34.29 (heartbeat at trainer step 321860) | Re-uploaded. About 1.20 s/step (0.835 steps/s, bs 24), Tesla T4. |
| 2026-09-29 21:46 / 2026-09-30 03:16 | 325000 | Kaggle (first upload) | 0.000 / 0.105 / 0.077 | 0.061 | - | Overwritten by the 00:04Z upload below. |
| 2026-09-30 00:04 / 05:34 | 325000 | Kaggle v5 (re-trained) | 0.000 / 0.079 / 0.038 | 0.039 | - | Best CER so far, but see the CER caveat; this is within noise of 0.061. |
| 2026-09-29 22:55 / 2026-09-30 04:25 | 330000 | Kaggle (first upload) | 0.000 / 0.132 / 0.154 | 0.095 | - | Overwritten by the 01:13Z upload below. |
| 2026-09-30 01:13 / 06:43 | 330000 | Kaggle v5 (re-trained) | 0.000 / 0.105 / 0.115 | 0.074 | train_mel 0.430; loss_g 33.27 (heartbeat at trainer step 330600) | |
| 2026-09-30 00:04 / 05:34 | 335000 | Kaggle v4 session | 0.000 / 0.105 / 0.077 | 0.061 | - | Earlier copy was placed in `voices/hi_IN-custom-medium.onnx` (local server voice); later overwritten by v5 upload. |
| 2026-09-30 02:22 / 07:52 | 335000 | Kaggle v5 (re-trained) | 0.045 / 0.105 / 0.115 | 0.089 | - | Overwrote the 00:04Z upload. Samples in docs/samples/step_335000 are this one. |
| 2026-09-30 01:13 / 06:43 | 340000 | Kaggle v4 session (older concurrent session) | 0.000 / 0.132 / 0.077 | 0.070 | - | Overwritten by the 03:31Z v5 upload below; docs/samples/step_340000 now holds the v5 copy. |
| 2026-09-30 02:21 / 07:51 | 345000 | Kaggle v4 session (older concurrent session) | 0.000 / 0.132 / 0.077 | 0.070 | - | Samples: docs/samples/step_345000. |
| 2026-09-30 03:31 / 09:01 | 340000 | Kaggle v5 (re-trained) | 0.000 / 0.105 / 0.077 | 0.061 | train_mel 0.430; loss_g 33.69 (v5 heartbeat 04:19Z at trainer step 343440) | Re-uploaded, overwrote the 01:13Z v4 copy. Samples in docs/samples/step_340000 are this one. |
| 2026-09-30 03:30 / 09:00 | 350000 | Kaggle v4 session (older concurrent session) | 0.000 / 0.079 / 0.192 | 0.090 | - | Latest recorded milestone. s3 worse than earlier rows (0.192), within the noise caveat. Samples: docs/samples/step_350000. |
| 2026-09-30 04:39 / 10:09 | 355000 | Kaggle v4 session (older concurrent session) | 0.000 / 0.158 / 0.077 | 0.078 | - | v4's last milestone. Samples: docs/samples/step_355000. |
| 2026-09-30 04:41 / 10:11 | 345000 | Kaggle v5 (re-trained) | 0.091 / 0.079 / 0.115 | 0.095 | - | Re-uploaded, overwrote the 02:21Z v4 copy. Samples in docs/samples/step_345000 are this one. |


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
| 2026-09-30 03:53 / 09:23 | 340000 | v4 (older concurrent session) | small (x2) | 0.130 / 0.125 / 0.209 | 0.159 | 3.79 (2.39) | 0.908 |
| 2026-09-30 04:09 / 09:39 | 345000 | v4 (older concurrent session) | small (x2) | 0.141 / 0.134 / 0.217 | 0.179 | 3.84 (2.88) | 0.913 |
| 2026-09-30 04:25 / 09:55 | 340000 | v4 (older concurrent session) | large-v3 | 0.050 / 0.020 / 0.076 | 0.056 | 3.84 (2.49) | 0.906 |
| 2026-09-30 04:35 / 10:05 | 345000 | v4 (older concurrent session) | large-v3 | 0.047 / 0.024 / 0.080 | 0.062 | 3.89 (2.97) | 0.911 |

Reading the numbers:
- Small ASR sits at CER about 0.13 on this set, much of it Whisper's own errors; large-v3 gives about 0.05. The old 3-sentence CER (0.07) is not comparable.
- CER max about 0.70 is one sentence (#38): large-v3 writes "online order / delivery / upgrade" in Latin script. PER for it is 0.11. Sentence #30 (year "उन्नीस सौ अठासी") is
  penalised because the normalizer reads the ASR's "1988" as "एक हज़ार नौ सौ अट्ठासी". Both are evaluation artefacts, not voice errors. PER is the fairer number for those.
- UTMOS min 2.4-3.0 identifies the weakest sentences per checkpoint (details JSON), useful for listening. UTMOS is English-trained: relative use only.
- Speaker similarity 0.91 vs 0.92 real-vs-real ceiling: the voice is close to the dataset speaker; it will not show naturalness.
- No listening test yet: none of these numbers say the voice sounds natural.

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
unchanged seed, see training/kaggle/README.md); the trained weights appear to exist only in the running Kaggle session until its deadline save (last.ckpt has not changed on HF since 2026-09-29 16:23 UTC). Anneal implementation: runbook 6.4.
