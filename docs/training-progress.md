# Custom Hindi voice — training progress log

Goal: natural, human-sounding (not robotic) Hindi voice fine-tuned from a public dataset. **Personal, non-commercial use only.**

Hardware: 4 vCPU, 15 GB RAM, no GPU (cloud container — may be reclaimed; checkpoints are not in git, exported ONNX milestones are).

| Date (UTC) | Step | Event | Notes |
|---|---|---|---|
| 2026-09-29 | 0 | Dataset download + fine-tune setup started | Worker selecting dataset (Rasa → IndicTTS → FLEURS) |
| 2026-09-29 | 310300 | Milestone 1 export (last.ckpt, epoch 3191) | loss_g 40.28, loss_d 1.83, mel 0.547, kl 2.72, dur 1.35; CER 0.163 (faster-whisper small, 3 sentences); ~6.85 s/log-step; TB step column is offset (~832.9k), true global_step from ckpt; samples in docs/samples/step_310300 |
| 2026-09-29 | 315000 | Milestone 2 export (Kaggle GPU run v1, HF milestones/step_315000; last.ckpt never uploaded, resume still from 310300) | CER 0.047 (faster-whisper small, 3 sentences); s/step and losses not recorded (no log retrievable); GPU name not recorded; samples in docs/samples/step_315000 |
| 2026-09-29 | 320000 | Milestone 3 re-exported (re-trained run v5, Kaggle GPU, HF milestones/step_320000 re-uploaded 22:55Z) | CER 0.074 (faster-whisper small, 3 sentences: 0.000/0.105/0.115); heartbeat at trainer step 321860: val_mel 0.448, loss_g 34.29, ~1.20 s/step (0.835 steps/s, bs=24), Tesla T4; earlier run row had CER 0.102; samples in docs/samples/step_320000 |
| 2026-09-30 | 325000 | Milestone 4 re-exported (v5 re-trained, Kaggle session v5, HF milestones/step_325000 re-uploaded 2026-09-30 00:04Z, overwriting the 21:46Z upload) | CER 0.039 (faster-whisper small, 3 sentences: 0.000/0.079/0.038); previous upload had CER 0.061 (0.000/0.105/0.077); Tesla T4; samples in docs/samples/step_325000 (replaced) |
| 2026-09-30 | 330000 | Milestone 5 re-exported (v5 re-trained, Kaggle session v5, HF milestones/step_330000 re-uploaded ~2026-09-30 01:13Z, overwriting the 22:55Z upload) | CER 0.074 (faster-whisper small, 3 sentences: 0.000/0.105/0.115); previous upload had CER 0.095 (0.000/0.132/0.154); v5 heartbeat at trainer step 330600: train_mel 0.430, loss_g 33.27; samples in docs/samples/step_330000 (replaced) |
| 2026-09-30 02:22 UTC / 07:52 IST | 335000 | Milestone 6 re-exported (v5 re-trained, Kaggle session "v5 re-trained", HF milestones/step_335000 re-uploaded 2026-09-30 02:22Z / 07:52 IST, overwriting the 00:04Z upload; copied to voices/hi_IN-custom-medium.onnx earlier from the old upload) | CER 0.089 (faster-whisper small, 3 sentences: 0.045/0.105/0.115); previous upload (v4 session, 00:04Z) had CER 0.061 (0.000/0.105/0.077); step labels of v4 and v5 are not on one timeline; samples in docs/samples/step_335000 (replaced) |
| 2026-09-30 | 340000 | Milestone 7 export (v4 session, older concurrent Kaggle session v4, HF milestones/step_340000, uploaded ~2026-09-30 01:13Z) | CER 0.070 (faster-whisper small, 3 sentences: 0.000/0.132/0.077); step labels of v4 and v5 are not on one timeline; samples in docs/samples/step_340000 |
| 2026-09-30 02:21 UTC / 07:51 IST | 345000 | Milestone 8 export (older Kaggle session "v4 session", HF milestones/step_345000, uploaded 2026-09-30 02:21Z / 07:51 IST) | CER 0.070 (faster-whisper small, 3 sentences: 0.000/0.132/0.077); step labels of v4 and v5 are not on one timeline; samples in docs/samples/step_345000 |

Voice files for every milestone: https://huggingface.co/vishwajit76/custom-tts-hindi-train/tree/main/milestones (private)
