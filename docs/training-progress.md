# Custom Hindi voice — training progress log

Goal: natural, human-sounding (not robotic) Hindi voice fine-tuned from a public dataset. **Personal, non-commercial use only.**

Hardware: 4 vCPU, 15 GB RAM, no GPU (cloud container — may be reclaimed; checkpoints are not in git, exported ONNX milestones are).

| Date (UTC) | Step | Event | Notes |
|---|---|---|---|
| 2026-09-29 | 0 | Dataset download + fine-tune setup started | Worker selecting dataset (Rasa → IndicTTS → FLEURS) |
| 2026-09-29 | 310300 | Milestone 1 export (last.ckpt, epoch 3191) | loss_g 40.28, loss_d 1.83, mel 0.547, kl 2.72, dur 1.35; CER 0.163 (faster-whisper small, 3 sentences); ~6.85 s/log-step; TB step column is offset (~832.9k), true global_step from ckpt; samples in docs/samples/step_310300 |
