"""python benchmark.py [--engines goonj,piper_v7a] [--regen]: full pronunciation benchmark under the active config."""
import argparse
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
from optimizer import bench
from optimizer.loop import Session

ap = argparse.ArgumentParser()
ap.add_argument("--engines")
ap.add_argument("--regen", action="store_true", help="regenerate tests/pronunciation_bench from optimizer/bench.py")
a = ap.parse_args()
if a.regen:
    bench.write()
out = Session().benchmark(a.engines.split(",") if a.engines else None)
for e, s in out.items():
    print(e, json.dumps({k: v for k, v in s.items() if k != "categories"}))
    for c, v in s["categories"].items():
        print(f"  {c:16s} n={v['n']:3d} total={v['total']:.3f} pron={v['pronunciation']:.3f} exact={v['text_exact']:.2f} cer={v['cer']:.3f}")
