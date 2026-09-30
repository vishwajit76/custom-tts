# Kaggle GPU training (Hindi Piper fine-tune)

Kernel: https://www.kaggle.com/code/vishwajit76/custom-tts-hindi-train (private, GPU, internet on).
Operating guide, secrets, evaluation and troubleshooting: [docs/custom-voice-runbook.md](../../docs/custom-voice-runbook.md).

## Relaunch

```bash
pip install -U kaggle                       # >= 1.7 for KGAT tokens
echo -n '<KGAT token>' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
kaggle kernels status vishwajit76/custom-tts-hindi-train   # never push while a session is running
training/kaggle/push.sh --kernel-version v6   # stamps GIT_SHA / RESUME_FROM / EXPERIMENT_ID into a temp copy and pushes; --dry-run to inspect
kaggle kernels push -p training/kaggle        # plain push also works (git_sha recorded as unknown, RESUME_FROM=auto)
```

Limits: <= 12 h per run (kernel stops itself at 11 h and uploads a final checkpoint), ~30 GPU h/week per account.
Each run writes only to its own HF folder `experiments/<id>/` (guarded; see docs/training.md section 6), so a second session cannot overwrite it. Keep the local CPU trainer stopped
(`touch training/runs/hi_f/STOP`): it still uploads to the legacy `runs/hi_f/last.ckpt` via `hf_sync.sh`.

## What the kernel does (`train_kernel.py`)

- Reads the HF token from the private Kaggle dataset `vishwajit76/cttsh-secrets` (file `hf_token`). Tokens are never in git or kernel source.
- Installs `piper-tts[train]==1.8.0` (constrained to the preinstalled torch), builds `monotonic_align` with cythonize.
- Downloads `data/hi_f` + `runs/hi_f/config.json`; resolves the resume checkpoint (`RESUME_FROM` = HF path or `auto` = highest step among `experiments/*/checkpoints/*.ckpt` and `runs/hi_f/*.ckpt`), verifies sha256/loadable/finite/step, writes `manifest.json` + `environment.json`. Piper's cache is rebuilt on the VM.
- Resumes with `--ckpt_path <verified resume checkpoint>`, fp16-mixed, batch 24 (falls back 16/12/8 on CUDA OOM; 32 OOMs on a 16 GB GPU). Seed 1234 (piper's train/val split is drawn from it: never change it for one voice).
- New checkpoints are written to `/tmp/w/ckpt/last.ckpt` (the resume checkpoint is never rewritten) and uploaded to `experiments/<id>/checkpoints/` after a load/finite/step check with sha256 read-back; the end of the run adds the create-once `final_step<N>.ckpt` and `result.json`.
- Learning rate: `LR_MODE=anneal` (default in the script; `LR_MODE=keep` disables) sets LR per epoch from `LR_START` (1e-4) to `LR_START*LR_FINAL_RATIO` (0.05) over `ANNEAL_EPOCHS` (150, about 45k steps = one session), then holds; see runbook section 4 and `training/lr_dryrun.py`.
- Checkpoint every 500 global steps (Lightning counts both GAN optimizers, so 2 per batch); a thread uploads last.ckpt to HF every 20 min.
- Every 5000 global steps: legacy ONNX export, 3 test sentences synthesized, uploaded create-once to HF `experiments/<id>/milestones/step_<N>/` and `samples/step_<N>/`.
- Heartbeat: `experiments/<id>/heartbeat.txt` and the legacy `runs/hi_f/kaggle_progress.txt` (first line names session and experiment; `sps` = seconds per step), `metrics.jsonl` every 20 steps (uploaded every 15 min). Kaggle shows no live logs for script kernels. Crashes: `experiments/<id>/crash.txt` + legacy `runs/hi_f/kaggle_crash.txt`.

## Pull results

`hf_hub_download` of `experiments/<id>/checkpoints/final_step<N>.ckpt`; milestones are under `experiments/<id>/milestones/` (legacy runs: `milestones/`). `python -m bench.compare_checkpoints --list` lists them.
