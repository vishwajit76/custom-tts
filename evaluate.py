"""python evaluate.py [--test ID | --category X | --text "..."] [--engine goonj]: score cases under the active config."""
import argparse
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
from optimizer import core
from optimizer.loop import Session

ap = argparse.ArgumentParser()
ap.add_argument("--test")
ap.add_argument("--category")
ap.add_argument("--text")
ap.add_argument("--engine")
a = ap.parse_args()
s = Session()
e = a.engine or s.cfg["primary_engine"]
if a.text:
    cases = [{"id": "adhoc", "category": "adhoc", "priority": 3, "text": a.text, "expected_normalized": a.text}]
else:
    cases = [c for c in s.cases.values() if (not a.test or c["id"] == a.test) and (not a.category or c["category"] == a.category)]
rs = s.ev.run(cases, e, s.active, s.pron)
for r in rs:
    print(json.dumps({k: r[k] for k in ("id", "total", "pronunciation", "naturalness", "prosody", "audio_quality", "consistency",
                                        "failure_type", "normalized", "wav")} | {"asr": r["metrics"]["hyp"], "cer": r["metrics"]["cer"]},
                     ensure_ascii=False))
if len(rs) > 1:
    print(json.dumps({k: v for k, v in core.summarize({r["id"]: r for r in rs}).items() if k != "categories"}))
