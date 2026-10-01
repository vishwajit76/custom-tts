#!/usr/bin/env bash
# Idempotent, self-resuming CPU fine-tune launcher. Re-run after any restart: it resumes from the newest last.ckpt.
# Env: NAME (run name, default hi_f) DATA (default data/hi_f) INIT (init ckpt for a fresh run) BS (default 8; bench: 0.59 samples/s, peak 7.6 GB) WORKERS (1) THREADS (4) RSS_MAX_MB (11000 watchdog)
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${NAME:-hi_f}; DATA=${DATA:-data/hi_f}; BS=${BS:-8}
INIT=${INIT:-data/init/hi/hi_IN/rohan/medium/epoch=3190-step=309852.ckpt}
RUN=training/runs/$NAME; mkdir -p "$RUN"
PID=$RUN/train.pid
if [ -f "$PID" ] && kill -0 "$(cat "$PID")" 2>/dev/null; then echo "already running (pid $(cat "$PID"))"; exit 0; fi
if [ ! -f "$INIT" ] && ! ls "$RUN"/lightning_logs/version_*/checkpoints/last.ckpt >/dev/null 2>&1; then
  python - <<'P'
from huggingface_hub import hf_hub_download
hf_hub_download("rhasspy/piper-checkpoints","hi/hi_IN/rohan/medium/epoch=3190-step=309852.ckpt",repo_type="dataset",local_dir="data/init")
P
fi
# supervisor (training/supervise.sh): restarts training (resuming from last.ckpt) if it crashes; stops on clean exit or if STOP file exists
RUN=$RUN DATA=$DATA NAME=$NAME INIT=$INIT BS=$BS WORKERS=${WORKERS:-1} THREADS=${THREADS:-4} nohup setsid bash training/supervise.sh >/dev/null 2>&1 &
echo $! > "$PID"
echo "started pid $(cat "$PID"); log: $RUN/train.log (stop: touch $RUN/STOP && kill \$(cat $PID))"
