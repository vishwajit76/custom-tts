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
| 2026-09-30 01:13 / 06:43 | 340000 | Kaggle v4 session (older concurrent session) | 0.000 / 0.132 / 0.077 | 0.070 | - | Samples: docs/samples/step_340000. |
| 2026-09-30 02:21 / 07:51 | 345000 | Kaggle v4 session (older concurrent session) | 0.000 / 0.132 / 0.077 | 0.070 | - | Latest recorded milestone. Samples: docs/samples/step_345000. |

Open items: no listening test; per-sentence CER for the first three rows and losses for most rows were not recorded; run one
Kaggle session at a time from now on (runbook section 9).
