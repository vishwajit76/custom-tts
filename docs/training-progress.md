# Custom Hindi voice — training progress log

Goal: natural, human-sounding (not robotic) Hindi voice fine-tuned from a public dataset. **Personal, non-commercial use only.**

Hardware: 4 vCPU, 15 GB RAM, no GPU (cloud container — may be reclaimed; checkpoints are not in git, exported ONNX milestones are).

| Date (UTC) | Step | Event | Notes |
|---|---|---|---|
| 2026-09-29 | 0 | Dataset download + fine-tune setup started | Worker selecting dataset (Rasa → IndicTTS → FLEURS) |
| 2026-09-29 | 310300 | Milestone 1 export (last.ckpt, epoch 3191) | loss_g 40.28, loss_d 1.83, mel 0.547, kl 2.72, dur 1.35; CER 0.163 (faster-whisper small, 3 sentences); ~6.85 s/log-step; TB step column is offset (~832.9k), true global_step from ckpt; samples in docs/samples/step_310300 |
| 2026-09-29 | 315000 | Milestone 2 export (Kaggle GPU run v1, HF milestones/step_315000; last.ckpt never uploaded, resume still from 310300) | CER 0.047 (faster-whisper small, 3 sentences); s/step and losses not recorded (no log retrievable); GPU name not recorded; samples in docs/samples/step_315000 |
| 2026-09-29 | 320000 | Milestone 3 re-exported (re-trained run v5, Kaggle GPU, HF milestones/step_320000 re-uploaded 22:55Z) | CER 0.074 (faster-whisper small, 3 sentences: 0.000/0.105/0.115); heartbeat at trainer step 321860: val_mel 0.448, loss_g 34.29, ~1.20 s/step (0.835 steps/s, bs=24), Tesla T4; earlier run row had CER 0.102; samples in docs/samples/step_320000 |
| 2026-09-29 | 325000 | Milestone 4 export (re-trained run v5, HF milestones/step_325000, uploaded 21:46Z) | CER 0.061 (faster-whisper small, 3 sentences: 0.000/0.105/0.077); losses not recorded for this exact step; Tesla T4; samples in docs/samples/step_325000 |
| 2026-09-29 | 330000 | Milestone 5 export (re-trained run v5, HF milestones/step_330000, uploaded 22:55Z) | CER 0.095 (faster-whisper small, 3 sentences: 0.000/0.132/0.154); noisy; label is the ckpt filename step, which exceeds heartbeat trainer step (321860 at 23:20Z), so filename step is offset from true step; samples in docs/samples/step_330000 |
