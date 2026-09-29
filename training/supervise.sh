#!/usr/bin/env bash
# Supervisor loop (launched by run_longtrain.sh): restarts training from last.ckpt on crash, logs RSS to mem.log every 20 s,
# and RSS watchdog: if trainer RSS > RSS_MAX_MB (or MemAvailable < MIN_AVAIL_MB) it restarts the trainer (SIGTERM, resumes from
# last.ckpt, losing <=100 steps) instead of letting the container OOM. Stops on clean exit or STOP file.
# Env: RUN DATA NAME INIT BS WORKERS THREADS RSS_MAX_MB MIN_AVAIL_MB ALLOC
cd "$(dirname "$0")/.."
: "${THREADS:=4}" "${WORKERS:=1}" "${RSS_MAX_MB:=11000}" "${MIN_AVAIL_MB:=1200}" "${ALLOC:=glibc}"
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS PIPER_THREADS=$THREADS PIPER_TRAIN_MODULE=training.piper_cpu
if [ "$ALLOC" = jemalloc ] && [ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 MALLOC_CONF=background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:0
fi
rss_mb() { local t=0 q r; for q in $(pgrep -f "training.piper_cpu fit"); do r=$(awk '/VmRSS/{print int($2/1024)}' /proc/$q/status 2>/dev/null); t=$((t + ${r:-0})); done; echo $t; }
avail_mb() { awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo; }
while [ ! -f "$RUN/STOP" ]; do
  echo "=== $(date -u +%FT%TZ) launch bs=$BS workers=$WORKERS threads=$THREADS alloc=$ALLOC" >> "$RUN/train.log"
  python -m training.train --data "$DATA" --run "$RUN" --init "$INIT" --name "$NAME" --epochs 100000 \
    --batch-size "$BS" --accelerator cpu --precision 32-true --trainer.log_every_n_steps 10 --data.num_workers "$WORKERS" \
    >> "$RUN/train.log" 2>&1 &
  TP=$!
  while kill -0 $TP 2>/dev/null; do
    sleep 20; r=$(rss_mb); a=$(avail_mb)
    echo "$(date +%T) ${r}MB avail=${a}MB $(ls -t "$RUN"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1 | xargs -r stat -c %y | cut -c12-19)" >> "$RUN/mem.log"
    if [ "$r" -gt "$RSS_MAX_MB" ] || [ "$a" -lt "$MIN_AVAIL_MB" ]; then
      echo "=== $(date -u +%FT%TZ) watchdog: rss=${r}MB avail=${a}MB -> graceful restart" | tee -a "$RUN/train.log" >> "$RUN/mem.log"
      pkill -TERM -f "training.piper_cpu fit"; sleep 15; pkill -KILL -f "training.piper_cpu fit"; sync; break
    fi
  done
  wait $TP; rc=$?
  [ "$rc" -eq 0 ] && break
  echo "=== exited rc=$rc, retry in 20s" >> "$RUN/train.log"; sleep 20
done
