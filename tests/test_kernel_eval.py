"""In-kernel milestone evaluation: EvalManager (defer / timeout / never raises), previous-evaluation pick, and that the vendored bundle is import-complete."""
import base64
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "training/kaggle/train_kernel.py").read_text("utf-8")


@pytest.fixture(scope="module")
def K():
    block = re.search(r"# ---- BEGIN HELPERS.*?\n(.*?)# ---- END HELPERS ----", SRC, re.S).group(1)
    ns = {"__name__": "kernel_helpers"}
    import glob, hashlib, json as _j, os, pathlib, shutil, subprocess as sp, sys as _s, threading, time  # noqa: E401
    ns.update(glob=glob, hashlib=hashlib, json=_j, os=os, pathlib=pathlib, re=re, shutil=shutil, subprocess=sp, sys=_s, threading=threading, time=time)
    exec(compile(block, "kernel_helpers", "exec"), ns)
    return SimpleNamespace(**ns)


def test_pick_prev_eval(K):
    files = ["experiments/a/evaluations/step_5000.json", "experiments/a/evaluations/step_10000.json", "experiments/b/evaluations/step_10000.json",
             "experiments/a/evaluations/step_10000.skipped.json", "experiments/a/milestones/step_15000/x.onnx"]
    assert K.pick_prev_eval(files, 15000, "b") == "experiments/b/evaluations/step_10000.json"
    assert K.pick_prev_eval(files, 15000, "a") == "experiments/a/evaluations/step_10000.json"
    assert K.pick_prev_eval(files, 10000, "a") == "experiments/a/evaluations/step_5000.json"
    assert K.pick_prev_eval(files, 5000, "a") is None


def _mgr(K, run, free=None, put_log=None, prev=lambda s: None):
    put_log = [] if put_log is None else put_log
    def put(rel, body):
        put_log.append((rel, body)); return rel
    return K.EvalManager(run, put, prev, lambda: free, min_free_mb=3000, budget_s=10, log=lambda *a, **k: None, cleanup=lambda *a, **k: None), put_log


def test_manager_ok_timeout_error_never_raise(K):
    def run(item, prev, budget):
        return {1: ("ok", b"{}"), 2: ("timeout", "slow"), 3: ("error", "boom")}.get(item["step"]) or (_ for _ in ()).throw(RuntimeError("crash"))
    m, puts = _mgr(K, run)
    for n in (1, 2, 3, 4):
        m.enqueue(n, "/nonexistent")
    st = m.finish(30)
    assert [s["status"] for s in st] == ["ok", "timeout", "error", "error"]
    assert [p[0] for p in puts] == ["evaluations/step_1.json", "evaluations/step_2.skipped.json", "evaluations/step_3.skipped.json", "evaluations/step_4.skipped.json"]
    assert json.loads(puts[1][1])["status"] == "timeout"


def test_manager_upload_failure_is_contained(K):
    m = K.EvalManager(lambda i, p, b: ("ok", b"{}"), lambda rel, body: (_ for _ in ()).throw(OSError("hf down")), log=lambda *a, **k: None, cleanup=lambda *a, **k: None)
    m.enqueue(1, "/x")
    assert m.finish(30)[0]["status"] == "upload_failed"


def test_manager_defers_when_vram_low_then_runs_after_training(K):
    ran = []
    def run(item, prev, budget):
        ran.append((item["step"], budget)); return "ok", b"{}"
    m, puts = _mgr(K, run, free=1000)
    m.enqueue(5000, "/x")
    st = m.finish(600)
    assert ran == [(5000, 10)] and [s["status"] for s in st] == ["deferred", "ok"] and puts[0][0] == "evaluations/step_5000.json"


def test_manager_passes_previous_evaluation(K):
    seen = []
    m, _ = _mgr(K, lambda i, p, b: (seen.append(p), ("ok", b"{}"))[1], prev=lambda s: f"/prev_{s}.json")
    m.enqueue(7, "/x"); m.finish(30)
    assert seen == ["/prev_7.json"]


def test_vendored_bundle_is_import_complete(tmp_path):
    sys.path.insert(0, str(ROOT / "training/kaggle"))
    import make_bundle
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(make_bundle.build())))
    z.extractall(tmp_path)
    # imports the evaluation needs, resolved ONLY against the extracted files (repo root not on the path)
    code = "import bench.kernel_eval, bench.compare_checkpoints, bench.milestone_eval, training.asr, app.services.text_normalizer, app.services.speaker_encoder, app.services.indian_english; " \
           "from bench import milestone_eval as me; assert len(me.load_sentences(me.DEFAULT_SENTENCES)) == 50; print('ok')"
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env={"PYTHONPATH": str(tmp_path), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}, capture_output=True, text=True)
    assert r.stdout.strip().endswith("ok"), r.stderr[-800:]
