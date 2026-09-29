# Kaggle GPU training (Hindi Piper fine-tune)

Kernel: https://www.kaggle.com/code/vishwajit76/custom-tts-hindi-train (private, GPU, internet on).

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
- Checkpoint every 500 global steps (Lightning counts both GAN optimizers, so 2 per batch); a thread uploads last.ckpt to HF every 20 min.
- Every 5000 global steps: legacy ONNX export, 3 test sentences synthesized, uploaded to HF `milestones/step_<N>/`.
- Heartbeat: `runs/hi_f/kaggle_progress.txt` on HF (last trainer lines, every 3 min). Kaggle shows no live logs for script kernels. Crashes go to `runs/hi_f/kaggle_crash.txt`.

## Pull results

`training/hf_sync.sh restore` (or `hf_hub_download`) fetches last.ckpt; milestones are under `milestones/` in the HF repo.
