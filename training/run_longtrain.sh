#!/usr/bin/env bash
# Idempotent, self-resuming CPU fine-tune launcher. Re-run after any restart: it resumes from the newest last.ckpt.
# Env: NAME (run name, default hi_f) DATA (default data/hi_f) INIT (init ckpt for a fresh run) BS (default 12)
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${NAME:-hi_f}; DATA=${DATA:-data/hi_f}; BS=${BS:-4}
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
# supervisor loop: restarts training (resuming from last.ckpt) if it crashes; stops on clean exit or if STOP file exists
nohup setsid bash -c '
  while [ ! -f "'"$RUN"'/STOP" ]; do
    echo "=== $(date -u +%FT%TZ) launch" >> "'"$RUN"'/train.log"
    PIPER_TRAIN_MODULE=training.piper_cpu python -m training.train --data "'"$DATA"'" --run "'"$RUN"'" --init "'"$INIT"'" --name "'"$NAME"'" --epochs 100000 \
      --batch-size "'"$BS"'" --accelerator cpu --precision 32-true --trainer.log_every_n_steps 10 \
      >> "'"$RUN"'/train.log" 2>&1 && break
    echo "=== exited nonzero, retry in 30s" >> "'"$RUN"'/train.log"; sleep 30
  done' >/dev/null 2>&1 &
echo $! > "$PID"
echo "started pid $(cat "$PID"); log: $RUN/train.log (stop: touch $RUN/STOP && kill \$(cat $PID))"
