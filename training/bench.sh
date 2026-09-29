#!/usr/bin/env bash
# usage: bench.sh BS WORKERS THREADS [ALLOC=glibc|jemalloc] [SECS=170]  -> prints s/step, samples/s, peak RSS (bench run dir, copy of last.ckpt)
cd "$(dirname "$0")/.."; BS=$1; W=$2; T=$3; AL=${4:-glibc}; SECS=${5:-170}; RUN=training/runs/bench
rm -f $RUN/log.txt; PRE=""
[ "$AL" = jemalloc ] && export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 MALLOC_CONF=background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:0
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T PIPER_THREADS=$T
PIPER_TRAIN_MODULE=training.piper_cpu python -m training.train --data data/hi_f --run $RUN --name hi_f --epochs 100000 --batch-size $BS --accelerator cpu --precision 32-true \
  --trainer.log_every_n_steps 4 --data.num_workers $W > $RUN/log.txt 2>&1 &
P=$!; peak=0; for i in $(seq $((SECS/5))); do sleep 5; for q in $(pgrep -f "training.piper_cpu fit"); do r=$(awk '/VmRSS/{print int($2/1024)}' /proc/$q/status 2>/dev/null); [ "${r:-0}" -gt "$peak" ] && peak=$r; done; done
pkill -f "training.piper_cpu fit"; kill $P 2>/dev/null; sleep 3
V=$(ls -d $RUN/lightning_logs/version_* | sort -V | tail -1)
python - $V $BS $peak "$W $T $AL" <<'P' 2>&1
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
a=EventAccumulator(sys.argv[1],size_guidance={"scalars":0});a.Reload()
ev=a.Scalars("loss_g"); ev=ev[3:] if len(ev)>6 else ev
s=(ev[-1].wall_time-ev[0].wall_time)/max(1,(ev[-1].step-ev[0].step)); bs=int(sys.argv[2])
print(f"bs={bs} w/t/alloc={sys.argv[4]} s/gstep={s:.2f} s/batch={2*s:.2f} samples/s={bs/(2*s):.2f} peakRSS={sys.argv[3]}MB")
P
