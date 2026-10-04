"""Run a command while holding a machine-wide GPU lock, so parallel agents never share the 6 GB card.

usage: python scripts/gpu_lock.py <cmd> [args...]
"""
import msvcrt
import subprocess
import sys
import time
from pathlib import Path

lock = Path(__file__).resolve().parents[1] / ".gpu.lock"
with open(lock, "a+") as f:
    waited = time.time()
    while True:
        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            break
        except OSError:
            time.sleep(2)
    print(f"[gpu_lock] acquired after {time.time() - waited:.0f}s", file=sys.stderr, flush=True)
    try:
        cmd = sys.argv[1:]
        if Path(cmd[0]).exists():
            cmd[0] = str(Path(cmd[0]).resolve())
        sys.exit(subprocess.call(cmd))
    finally:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
