# Custom Hindi voice: operational runbook

How the personal custom Hindi voice is trained, monitored, evaluated and served **as it actually works today**
(state of 2026-09-30). For the generic pipeline (any dataset, CPU/MPS, manifests, rights) see [training.md](training.md).
For per-milestone numbers see [training-progress.md](training-progress.md).

Scope and licence: personal, non-commercial use only. The init checkpoint (Piper `hi_IN-rohan-medium`) descends from
research-only data, see [research.md](research.md) and [licenses.md](licenses.md). Do not ship this voice commercially.

Time convention: UTC first, IST (UTC+5:30) second, e.g. 02:22 UTC / 07:52 IST.

## 1. Overview

```
IndicTTS Hindi female (7.9 h) --prepare_dataset--> data/hi_f --hf_sync backup-data--> HF private repo data/hi_f
Piper hi_IN rohan medium ckpt (step 309852) ---------------------------------------> HF runs/hi_f/last.ckpt (seed)
                                   Kaggle T4 kernel (resume, fp16-mixed, bs 24)
        last.ckpt every 20 min -> HF runs/hi_f/     heartbeat every 3 min -> HF runs/hi_f/kaggle_progress.txt
        every 5000 steps: ONNX + 3 test wavs -> HF milestones/step_N/
                                   evaluate (CER) -> copy to models/piper/ -> server
```

Status: training is in progress. No human listening test has been done. CER is a noisy proxy (section 6).

## 2. Assets and where they live

| Asset | Location | Notes |
|---|---|---|
| Dataset `hi_f` | private HF repo `vishwajit76/custom-tts-hindi-train`, path `data/hi_f/` (`wavs/`, `metadata.csv`) | also local `data/hi_f` (gitignored) |
| Init checkpoint | `rhasspy/piper-checkpoints` (HF dataset) `hi/hi_IN/rohan/medium/epoch=3190-step=309852.ckpt` | training started here, global_step 309852 |
| Live training checkpoint | HF `runs/hi_f/last.ckpt` + `runs/hi_f/config.json` | single source of truth for resuming |
| Heartbeat / crash | HF `runs/hi_f/kaggle_progress.txt`, `runs/hi_f/kaggle_crash.txt` | see section 4 |
| Milestone voices | HF `milestones/step_N/` (`hi_IN-custom-medium.onnx`, `.onnx.json`, `sample_1..3.wav`) | private; mirrored samples in `docs/samples/step_N/` |
| Kaggle notebook | https://www.kaggle.com/code/vishwajit76/custom-tts-hindi-train (private, GPU, internet on) | source: `training/kaggle/train_kernel.py` |
| Kaggle secrets dataset | `vishwajit76/cttsh-secrets` (private), file `hf_token` | read by the kernel |
| Ckpts are not in git | `*.ckpt`, `training/runs/`, `data/`, `voices/*.onnx*` are gitignored | only docs/samples and the code are committed |

### Dataset

IndicTTS Hindi, female speaker: 7.9 h after preparation. Prepared with `training/prepare_dataset.py`
(mono 22.05 kHz, silence trim, loudness -20 dBFS, transcripts through the server normalizer, rejection of clipped/too
short/too long/rate-implausible clips, deterministic 5% held-out `test.csv`), see [training.md](training.md) section 2.
Uploaded once with `training/hf_sync.sh backup-data` (needs `NAME=hi_f`, run from a checkout that has `data/hi_f`).
IndicTTS licence status is unresolved (see [research.md](research.md)): keep the repo private.

## 3. Secrets

Never commit tokens, paste them in chat, or put them in kernel source.

| Secret | Where it lives |
|---|---|
| HF token | private Kaggle dataset `vishwajit76/cttsh-secrets` (file `hf_token`); locally `~/.cache/huggingface/token` |
| Kaggle API token (KGAT) | `~/.kaggle/access_token` (chmod 600), exported as `KAGGLE_API_TOKEN` |

**Recommendation: rotate both tokens.** They were pasted into a chat session during setup, so treat them as exposed.
Rotate: create a new HF token (write scope, limited to the one repo if possible) and a new Kaggle token, then
(1) overwrite `hf_token` in the `cttsh-secrets` dataset (new dataset version) and (2) replace the two local files.
Revoke the old ones afterwards. The kernel reads whichever `hf_token` it finds under `/kaggle/input`.

## 4. Launching and relaunching training

Prerequisites: `pip install -U kaggle` (>= 1.7 for KGAT tokens), Kaggle token as above.

```bash
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
kaggle kernels push -p training/kaggle      # starts a new run; resumes from HF runs/hi_f/last.ckpt
kaggle kernels status vishwajit76/custom-tts-hindi-train
```

What a run does (`training/kaggle/train_kernel.py`; details in [../training/kaggle/README.md](../training/kaggle/README.md)):

- Installs `piper-tts[train]==1.8.0`, builds `monotonic_align`, downloads `data/hi_f` and `runs/hi_f/{last.ckpt,config.json}`.
- Trains on a T4 (16 GB) with `--trainer.precision 16-mixed`, batch 24 (falls back 16/12/8 on CUDA OOM; 32 OOMs).
  Speed about 0.83 steps/s (about 1.2 s per global step, Lightning counts both GAN optimizers so one batch = 2 steps),
  so 5000 steps take about 100 min.
- Local checkpoint every 500 steps; a thread uploads `last.ckpt` to HF every 20 min.
- Every 5000 steps: legacy ONNX export on CPU, 3 test sentences synthesized, folder uploaded to HF `milestones/step_N/`.
- Self-stops at 11 h (`MAX_HOURS`), saves `last.ckpt`, uploads it (`final`). Kaggle's hard cap is 12 h per session and
  about 30 GPU h per week per account, so relaunch after each run ends.
- Resume is lossy by up to 20 min: at most the steps since the last HF upload are redone.

### Heartbeat and crash files

`runs/hi_f/kaggle_progress.txt` on HF, rewritten every 3 min (Kaggle shows no live logs for script kernels): timestamp,
elapsed hours, start step, batch size, GPU, latest trainer step / s per step / losses (`trainer_progress`), age of the
local checkpoint, last HF upload status, last 20 log lines. A stale timestamp (older than about 10 min) means the
kernel is dead or restarting. `runs/hi_f/kaggle_crash.txt` holds a traceback or the last 60 trainer lines after a crash.

```bash
python - <<'P'
from huggingface_hub import hf_hub_download
print(open(hf_hub_download("vishwajit76/custom-tts-hindi-train","runs/hi_f/kaggle_progress.txt",force_download=True)).read())
P
```

## 5. Hourly check-in routine

A scheduled routine wakes the session hourly to: read the heartbeat (alive? step advancing? last upload ok?), list new
`milestones/step_N`, run the CER check (section 6) on new ones, copy the samples into `docs/samples/step_N/`, append a row
to [training-progress.md](training-progress.md), commit and push to the PR branch. If the heartbeat is stale for more
than about 30 min or `kaggle_crash.txt` is new, relaunch (section 4) after making sure no other session is running
(section 9). Pull with `git pull --rebase origin <branch>` before committing, because check-ins also push.

## 6. Evaluating a milestone (CER)

Method: synthesize the 3 fixed test sentences (the ones in `sample_1..3.wav`), transcribe with **faster-whisper `small`**
(beam 5, `language="hi"`, int8, CPU; see `bench/quality_fw.py`), score with `training.asr.cer` (Levenshtein over characters
after dropping punctuation, spaces and nukta), report per-sentence CER and the mean.

```bash
pip install faster-whisper       # weights Systran/faster-whisper-small from the HF hub
python - <<'P'
import glob, sys, soundfile as sf
sys.path.insert(0, ".")
from bench.quality_fw import transcribe          # patches training.asr with the faster-whisper backend
from training.asr import cer
REFS = ["नमस्ते, आप कैसे हैं? आज मौसम बहुत सुहावना है।",
        "भारत एक विशाल देश है, जहाँ अनेक भाषाएँ और संस्कृतियाँ एक साथ मिलकर रहती हैं।",
        "क्या आपने कल रात का खाना खा लिया? मुझे तो बहुत भूख लगी है!"]
d = sys.argv[1] if len(sys.argv) > 1 else "docs/samples/step_345000"
s = []
for i, r in enumerate(REFS, 1):
    w, sr = sf.read(f"{d}/sample_{i}.wav"); s.append(cer(r, transcribe(w, sr)))
print([round(x, 3) for x in s], "mean", round(sum(s) / 3, 3))
P
```

(`bench.quality_fw` also runs UTMOS via torch.hub, which is blocked in some sandboxes; the import only needs the
`transcribe` function.)

**Limitation, read before trusting the numbers:** 3 short sentences, one ASR model, and small Whisper's own errors dominate.
One character flips a sentence score by 0.03 to 0.1, the same voice re-scored differently between uploads (0.061 vs 0.039 at
step 325000), and identical per-sentence triples recur across steps. CER differences below about 0.05 are noise. CER
measures intelligibility only, not naturalness, prosody or speaker similarity. The only real test is human listening
(blind A/B by native speakers), which has **not** been done. Pick a final checkpoint by listening, not by CER.

## 7. Using a milestone voice in the server

```bash
python - <<'P'
from huggingface_hub import hf_hub_download as d
import shutil, os
N = 345000                                   # milestone to try
os.makedirs("models/piper", exist_ok=True)
for ext in ("onnx", "onnx.json"):
    shutil.copy(d("vishwajit76/custom-tts-hindi-train", f"milestones/step_{N}/hi_IN-custom-medium.{ext}"),
                f"models/piper/hi_IN-custom-medium.{ext}")
P
DEFAULT_VOICE=hi_IN-custom-medium MODELS_DIR=models/piper uvicorn app.main:app   # or set in .env
curl -s localhost:8000/v1/voices | grep custom
```

The server loads every `*.onnx` + `*.onnx.json` in `MODELS_DIR` (default `models/piper`); the voice id is the file stem
`hi_IN-custom-medium`. Select it per request with `voice`, or globally with `DEFAULT_VOICE`. In Docker, mount the
directory at `/srv/models/piper`. `training/export_latest.sh` (CPU path) writes to `voices/` instead (gitignored); copy from
there into `MODELS_DIR`. A new file needs a server restart. Note: a milestone is a snapshot, so keep the step number in
your notes; the file name does not carry it.

## 8. Stopping training

- Kaggle: `kaggle kernels status ...` shows the state; there is no reliable CLI cancel for a running script kernel, stop it
  from the notebook page in the browser (Stop session) or let it hit its 11 h deadline. Before stopping, wait until a
  checkpoint upload is fresh in the heartbeat (`last_upload=ok HH:MM:SS`) so at most a few minutes are lost.
  A stop between uploads loses up to 20 min.
- Do not push a new kernel version to "stop" it: a push starts another session (section 9).
- CPU fallback trainer: `touch training/runs/hi_f/STOP && kill $(cat training/runs/hi_f/train.pid)`.

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory`, log says "CUDA OOM -> smaller batch" | batch too large for the GPU (32 OOMs on 16 GB) | automatic fallback 24 -> 16 -> 12 -> 8; to start lower, change the `BS` default in `train_kernel.py` and relaunch |
| Two Kaggle sessions writing to HF at once (e.g. re-pushed while one was still running) | each session uploads `last.ckpt`, `kaggle_progress.txt` and `milestones/step_N` to the same paths, so the last writer wins, heartbeat flips between sessions, milestones get overwritten with different weights | run only one session. Check status before `kaggle kernels push`. This happened in the v4/v5 sessions, see [training-progress.md](training-progress.md); their step labels are not on one timeline |
| CPU container silent, training stopped, log frozen | the cloud container pauses when idle (no active foreground tool) | reason the CPU path is a fallback only; use Kaggle. If needed: `training/run_longtrain.sh` again (self-resuming) |
| Step numbers differ between TensorBoard, logs and file names | TensorBoard `step` column is an offset counter (about 832.9k for the CPU run); Lightning counts 2 global steps per batch (two optimizers); milestone names use the checkpoint's true `global_step` | trust `global_step` from the checkpoint (`training/status.sh` prints it) and the heartbeat `step` |
| ONNX export fails (dynamo exporter, torch >= 2.9) | `torch.onnx.export` defaults to the dynamo exporter, which fails on VITS | legacy exporter: `training/export_onnx_legacy.py` (or `dynamo=False`); the kernel already does this |
| `MisconfigurationException: Automatic gradient accumulation is not supported for manual optimization` | Piper's VITS trainer uses manual optimization | no accumulation; raise `--batch-size`. `training.train` rejects the flag |
| Heartbeat stale, no crash file | Kaggle killed the session (12 h cap, quota, preemption) | check `kaggle kernels status`, relaunch, resume is from the last HF upload |
| `hf_token not found under /kaggle/input` | secrets dataset not attached or renamed | keep `dataset_sources` in `training/kaggle/kernel-metadata.json` |
| HF upload FAILED in heartbeat | token expired/rotated, network, rate limit | fix the token in `cttsh-secrets`, relaunch; local ckpt is lost when the session ends |
| Milestone CER looks worse than the previous one | ASR noise (section 6) or another session overwrote it | re-listen; compare sample wavs, not one number |

## 10. Lessons learned

- Kaggle GPU is roughly 100x the throughput of the 4 vCPU container (about 1.2 s/step at bs 24 versus about 7 s per bs-8 step on CPU); the CPU path only makes sense as a fallback.
- Cloud CPU containers pause when idle and can be reclaimed: always keep the checkpoint and data on HF, never only on local disk.
- One writer per HF path. Concurrent sessions silently overwrite each other's `last.ckpt` and milestones and produce misleading timelines.
- Kaggle script kernels have no live logs: a heartbeat file on HF with the trainer step, s/step, losses and upload status is the only visibility. Record GPU name and losses at each milestone, several early rows lack them.
- Step labels are ambiguous (TensorBoard offset, 2 global steps per batch, resumed runs). Always read `global_step` from the checkpoint.
- Gradient accumulation does not work with Piper's manual optimization; scale the batch instead and use fp16-mixed.
- torch >= 2.9 breaks Piper's ONNX export; use the legacy exporter.
- Three-sentence CER with a small Whisper is too noisy to rank checkpoints 5000 steps apart. Improvement from 310k (0.163) to 315k (0.047) is real-looking, after that the numbers hover around 0.04 to 0.10 with no trend. Losses were still falling slowly when last recorded (mel 0.547 at 310.3k on CPU, 0.448 at about 321.9k on Kaggle; different logging points), so more steps may help, but listening is needed.
- Never paste tokens into chat; keep them in files with restricted permissions and rotate anything that leaked.
