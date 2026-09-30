# Kaggle GPU training (Hindi Piper fine-tune)

Kernel: https://www.kaggle.com/code/vishwajit76/custom-tts-hindi-train (private, GPU, internet on).
Operating guide, secrets, evaluation and troubleshooting: [docs/custom-voice-runbook.md](../../docs/custom-voice-runbook.md).

## Relaunch

```bash
pip install -U kaggle                       # >= 1.7 for KGAT tokens
echo -n '<KGAT token>' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
kaggle kernels push -p training/kaggle      # starts a new run; resumes from HF runs/hi_f/last.ckpt
```

Limits: <= 12 h per run (kernel stops itself at 11 h and uploads a final checkpoint), ~30 GPU h/week per account.
Only one writer to HF `runs/hi_f/last.ckpt` at a time: keep the local CPU trainer stopped
(`touch training/runs/hi_f/STOP`).

## What the kernel does (`train_kernel.py`)

- Reads the HF token from the private Kaggle dataset `vishwajit76/cttsh-secrets` (file `hf_token`). Tokens are never in git or kernel source.
- Installs `piper-tts[train]==1.8.0` (constrained to the preinstalled torch), builds `monotonic_align` with cythonize.
- Downloads `data/hi_f` + `runs/hi_f/{last.ckpt,config.json}` from private HF repo `vishwajit76/custom-tts-hindi-train`; Piper's cache is rebuilt on the VM.
- Resumes with `--ckpt_path last.ckpt`, fp16-mixed, batch 24 (falls back 16/12/8 on CUDA OOM; 32 OOMs on a 16 GB GPU).
- New checkpoints are written to `/tmp/w/ckpt/last.ckpt` (the downloaded seed is never rewritten; before 2026-09-30 the 20-min uploads re-sent the unchanged seed because Lightning wrote to `version_1`, so only the deadline save was fresh).
- Learning rate: `LR_MODE=anneal` (default in the script; `LR_MODE=keep` disables) sets LR per epoch from `LR_START` (1e-4) to `LR_START*LR_FINAL_RATIO` (0.05) over `ANNEAL_EPOCHS` (160), then holds; see runbook section 4 and `training/lr_dryrun.py`.
- Checkpoint every 500 global steps (Lightning counts both GAN optimizers, so 2 per batch); a thread uploads last.ckpt to HF every 20 min.
- Every 5000 global steps: legacy ONNX export, 3 test sentences synthesized, uploaded to HF `milestones/step_<N>_<session>/` (session = `$KERNEL_VERSION` or `k` + UTC start time, also in `session.json`).
- Heartbeat: `runs/hi_f/kaggle_progress_<session>.txt` (per session) and the legacy `runs/hi_f/kaggle_progress.txt` on HF (first line names the session; trainer lines, LR, every 3 min). Kaggle shows no live logs for script kernels. Crashes go to `runs/hi_f/kaggle_crash.txt`.

## Pull results

`training/hf_sync.sh restore` (or `hf_hub_download`) fetches last.ckpt; milestones are under `milestones/` in the HF repo.
