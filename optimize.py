"""python optimize.py [--category X] [--test ID] [--iterations N] [--resume] [--benchmark]"""
import argparse
import os
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
sys.stdout.reconfigure(encoding="utf-8")
from optimizer.loop import Session

ap = argparse.ArgumentParser()
ap.add_argument("--category")
ap.add_argument("--test")
ap.add_argument("--iterations", type=int, default=10)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--benchmark", action="store_true", help="refresh the full benchmark first")
a = ap.parse_args()
s = Session()
if a.benchmark:
    print(s.benchmark())
rs = s.run(a.iterations, a.resume, a.category, a.test)
acc = [r for r in rs if r["decision"] == "accepted"]
print(f"{len(rs)} iterations, {len(acc)} accepted; report: reports/optimizer/report.html")
