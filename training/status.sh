#!/usr/bin/env bash
# Training status: alive?, epoch/step, latest losses (from TensorBoard events), newest ckpt + age, disk.
NAME=${NAME:-hi_f}; cd "$(dirname "$0")/.."; RUN=training/runs/$NAME
if [ -f "$RUN/train.pid" ] && kill -0 "$(cat "$RUN/train.pid")" 2>/dev/null; then echo "alive: yes (supervisor pid $(cat "$RUN/train.pid"))"; else echo "alive: NO (run training/run_longtrain.sh)"; fi
pgrep -f "training.piper_cpu fit" >/dev/null && echo "trainer process: running" || echo "trainer process: not running"
V=$(ls -d "$RUN"/lightning_logs/version_* 2>/dev/null | sort -V | tail -1)
if [ -n "$V" ]; then
  python - "$V" <<'P' 2>/dev/null
import sys, time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
a = EventAccumulator(sys.argv[1], size_guidance={"scalars": 0}); a.Reload()
tags = a.Tags()["scalars"]
print("--- progress / losses (%s)" % sys.argv[1])
if not tags: print("  no scalars logged yet")
for t in sorted(tags):
    s = a.Scalars(t)[-1]
    print(f"  {t:12s} = {s.value:9.4f}  (step {s.step}, {int(time.time()-s.wall_time)}s ago)")
if "epoch" in tags:
    print("  epoch:", int(a.Scalars("epoch")[-1].value))
# s/step over the last logged window
for t in tags:
    ev = a.Scalars(t)
    if len(ev) > 5 and ev[-1].step > ev[-6].step:
        print(f"  ~{(ev[-1].wall_time-ev[-6].wall_time)/(ev[-1].step-ev[-6].step):.2f} s/global_step (from '{t}', last 5 log points)"); break
P
fi
echo "--- newest ckpt"; c=$(ls -t "$RUN"/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
[ -n "$c" ] && echo "TRUE global_step (from ckpt; the TB 'step' above is a stale/offset counter): $(python -c "import torch,sys;print(torch.load(sys.argv[1],map_location='cpu',weights_only=False)['global_step'])" "$c" 2>/dev/null)"
[ -n "$c" ] && echo "$c  age $(( $(date +%s) - $(stat -c %Y "$c") ))s  size $(du -h "$c" | cut -f1)" || echo none
echo "--- disk"; du -sh "$RUN" data 2>/dev/null; df -h . | tail -1
