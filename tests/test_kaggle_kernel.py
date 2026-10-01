"""Unit tests for the pure helpers embedded in training/kaggle/train_kernel.py: resume selection, dataset fingerprint, checkpoint validation and the
experiment upload guard (with a fake HfApi). The helper block is exec'd from the kernel source, so these test the code that actually runs on Kaggle."""
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = (ROOT / "training/kaggle/train_kernel.py").read_text("utf-8")


@pytest.fixture(scope="module")
def K():
    block = re.search(r"# ---- BEGIN HELPERS.*?\n(.*?)# ---- END HELPERS ----", SRC, re.S).group(1)
    ns = {"__name__": "kernel_helpers"}
    import hashlib, json as _j, os, pathlib, shutil, subprocess, sys, time as _t  # noqa: E401  (names the kernel imports before the block)
    ns.update(hashlib=hashlib, json=_j, os=os, pathlib=pathlib, re=re, shutil=shutil, subprocess=subprocess, sys=sys, time=_t)
    exec(compile(block, "kernel_helpers", "exec"), ns)
    return SimpleNamespace(**ns)


class FakeHfApi:
    """Only what the guard touches; keeps bytes in memory and records every write."""

    def __init__(self, files=None):
        self.files = dict(files or {})
        self.writes = []

    def file_exists(self, repo_id, filename, **kw):
        return filename in self.files

    def list_repo_files(self, repo_id, **kw):
        return sorted(self.files)

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id=None, commit_message=None, **kw):
        self.files[path_in_repo] = path_or_fileobj if isinstance(path_or_fileobj, bytes) else Path(path_or_fileobj).read_bytes()
        self.writes.append((path_in_repo, commit_message))

    def upload_folder(self, *, folder_path, path_in_repo, repo_id=None, commit_message=None, **kw):
        for p in Path(folder_path).rglob("*"):
            if p.is_file():
                self.files[f"{path_in_repo}/{p.relative_to(folder_path)}"] = p.read_bytes()
                self.writes.append((f"{path_in_repo}/{p.relative_to(folder_path)}", commit_message))

    def read_text(self, path):
        return self.files[path].decode()


def guard(K, api, exp="hi_f-v6-0930T0730Z", sid="v6-0930T0730Z", **kw):
    return K.ExperimentGuard(api, "repo", exp, sid, read_text=api.read_text, **kw)


# ---------------------------------------------------------------- resume selection
def test_parse_step(K):
    assert K.parse_step("runs/hi_f/v4_final_step357212.ckpt") == 357212
    assert K.parse_step("experiments/x/checkpoints/final_step1.ckpt") == 1
    assert K.parse_step("runs/hi_f/last.ckpt") is None and K.parse_step("a/step_5.onnx") is None


def test_pick_resume_highest_step_ignores_milestones_and_non_ckpt(K):
    files = ["runs/hi_f/last.ckpt", "runs/hi_f/v4_final_step357212.ckpt", "milestones/step_355000/x.onnx", "runs/hi_f/config.json",
             "experiments/a/checkpoints/final_step350000.ckpt", "experiments/a/milestones/step_360000/m.ckpt", "data/hi_f/wavs/step_9.ckpt"]
    path, step, seen = K.pick_resume(files, lambda p: 310300)  # legacy last.ckpt is the stale seed
    assert (path, step) == ("runs/hi_f/v4_final_step357212.ckpt", 357212)
    assert {s["path"] for s in seen} == {"runs/hi_f/last.ckpt", "runs/hi_f/v4_final_step357212.ckpt", "experiments/a/checkpoints/final_step350000.ckpt"}
    path, step, _ = K.pick_resume(files, lambda p: 357900)  # a legacy last.ckpt that really advanced wins
    assert (path, step) == ("runs/hi_f/last.ckpt", 357900)


def test_pick_resume_tie_prefers_experiments_and_survives_probe_errors(K):
    files = ["runs/hi_f/final_step400.ckpt", "experiments/b/checkpoints/final_step400.ckpt", "runs/hi_f/last.ckpt"]
    def probe(p): raise OSError("torn")
    path, step, seen = K.pick_resume(files, probe)
    assert (path, step) == ("experiments/b/checkpoints/final_step400.ckpt", 400)
    assert any(s.get("error") for s in seen)
    assert K.pick_resume([], probe)[:2] == (None, None)


# ---------------------------------------------------------------- fingerprint and checkpoint validation
def test_dataset_fingerprint_changes_with_data(K, tmp_path):
    (tmp_path / "wavs").mkdir()
    (tmp_path / "metadata.csv").write_text("a.wav|x\nb.wav|y\n", "utf-8")
    (tmp_path / "wavs/a.wav").write_bytes(b"1234"); (tmp_path / "wavs/b.wav").write_bytes(b"12")
    f1 = K.dataset_fingerprint(tmp_path)
    assert f1["metadata_rows"] == 2 and f1["n_wavs"] == 2 and len(f1["metadata_sha256"]) == 64
    (tmp_path / "wavs/b.wav").write_bytes(b"123")
    assert K.dataset_fingerprint(tmp_path)["wavs_name_size_sha256"] != f1["wavs_name_size_sha256"]


def _ckpt(path, step=10, nan=False):
    import torch
    w = torch.zeros(3)
    if nan:
        w[0] = float("nan")
    torch.save({"global_step": step, "epoch": 2, "state_dict": {"w": w},
                "optimizer_states": [{"param_groups": [{"lr": 1.5e-4}]}, {"param_groups": [{"lr": 1.5e-4}]}],
                "lr_schedulers": [{"gamma": 0.9, "last_epoch": 5, "_last_lr": [1.5e-4]}]}, path)


def test_ckpt_meta_reads_step_lr_and_detects_nan_and_truncation(K, tmp_path):
    _ckpt(tmp_path / "ok.ckpt", 42)
    m = K.ckpt_meta(tmp_path / "ok.ckpt")
    assert m["global_step"] == 42 and m["epoch"] == 2 and m["lr_g"] == m["lr_d"] == 1.5e-4 and m["finite"] and m["lr_ctl"] is None
    _ckpt(tmp_path / "nan.ckpt", nan=True)
    m = K.ckpt_meta(tmp_path / "nan.ckpt")
    assert m["finite"] is False and m["nonfinite_tensors"] == ["w"]
    b = (tmp_path / "ok.ckpt").read_bytes()
    (tmp_path / "torn.ckpt").write_bytes(b[: len(b) // 2])  # a copy taken while Lightning was still writing
    with pytest.raises(Exception):
        K.ckpt_meta(tmp_path / "torn.ckpt")


# ---------------------------------------------------------------- upload guard (fake HfApi)
def test_guard_claims_and_writes_only_inside_the_experiment(K):
    api = FakeHfApi({"runs/hi_f/last.ckpt": b"legacy", "milestones/step_350000/a.onnx": b"old"})
    g = guard(K, api)
    g.claim({"git_sha": "abc"})
    assert json.loads(api.files["experiments/hi_f-v6-0930T0730Z/session.json"])["owner_session"] == "v6-0930T0730Z"
    g.put(b"x", "checkpoints/last.ckpt", "ckpt", overwrite=True)
    assert all(p.startswith("experiments/hi_f-v6-0930T0730Z/") for p, _ in api.writes)
    assert api.files["runs/hi_f/last.ckpt"] == b"legacy" and api.files["milestones/step_350000/a.onnx"] == b"old"
    assert all("hi_f-v6-0930T0730Z" in m for _, m in api.writes)  # every commit message names the experiment


def test_guard_refuses_foreign_owner_and_allows_same_session(K):
    api = FakeHfApi()
    guard(K, api, sid="v6-A").claim({})
    with pytest.raises(K.CollisionError):
        guard(K, api, sid="v6-B").claim({})  # same experiment id, different session
    guard(K, api, sid="v6-A").claim({})  # the owner may re-claim (e.g. after a crash-restart inside the same session)
    assert [p for p, _ in api.writes].count("experiments/hi_f-v6-0930T0730Z/session.json") == 1


def test_guard_create_once_files_are_never_overwritten(K):
    api = FakeHfApi(); g = guard(K, api); g.claim({})
    assert g.put(b"first", "checkpoints/final_step10.ckpt", "final") is not None
    assert g.put(b"second", "checkpoints/final_step10.ckpt", "final") is None
    assert api.files["experiments/hi_f-v6-0930T0730Z/checkpoints/final_step10.ckpt"] == b"first"
    g.put(b"a", "checkpoints/last.ckpt", "ckpt", overwrite=True); g.put(b"b", "checkpoints/last.ckpt", "ckpt", overwrite=True)
    assert api.files["experiments/hi_f-v6-0930T0730Z/checkpoints/last.ckpt"] == b"b"


def test_guard_folder_refuses_existing_and_rejects_escapes(K, tmp_path):
    api = FakeHfApi(); g = guard(K, api); g.claim({})
    (tmp_path / "m.onnx").write_bytes(b"1")
    assert g.put_folder(tmp_path, "milestones/step_5", "ms") is not None
    (tmp_path / "m.onnx").write_bytes(b"2")
    assert g.put_folder(tmp_path, "milestones/step_5", "ms") is None
    assert api.files["experiments/hi_f-v6-0930T0730Z/milestones/step_5/m.onnx"] == b"1"
    for bad in ("../other/x", "/abs", "a/../../b"):
        with pytest.raises(K.CollisionError):
            g.put(b"x", bad, "x")
    with pytest.raises(K.CollisionError):
        g.put_legacy(b"x", "runs/hi_f/last.ckpt", "x")  # never the shared resume path
    g.put_legacy(b"x", "runs/hi_f/kaggle_progress.txt", "hb")


def test_guard_detects_ownership_change_mid_run_and_needs_claim(K):
    api = FakeHfApi(); g = guard(K, api, recheck_s=0)
    with pytest.raises(K.CollisionError):
        g.put(b"x", "a", "m")  # not claimed
    g.claim({})
    api.files["experiments/hi_f-v6-0930T0730Z/session.json"] = json.dumps({"owner_session": "someone-else"}).encode()
    time.sleep(0.01)
    with pytest.raises(K.CollisionError):
        g.put(b"x", "metrics.jsonl", "m", overwrite=True)
    with pytest.raises(ValueError):
        K.ExperimentGuard(api, "r", "bad id!", "s")


def test_localapi_roundtrip_is_a_valid_fake(K, tmp_path):
    api = K.LocalApi(tmp_path / "hub"); g = guard(K, api, exp="hi_f-local-1", sid="s1", recheck_s=0)
    g.claim({"x": 1}); src = tmp_path / "f.bin"; src.write_bytes(b"abc")
    g.put(src, "checkpoints/last.ckpt", "ckpt", overwrite=True)
    assert api.get_paths_info("r", ["experiments/hi_f-local-1/checkpoints/last.ckpt"])[0].lfs.sha256 == K.sha256_file(src)
    assert K.remote_sha256(api, "r", "experiments/hi_f-local-1/checkpoints/last.ckpt") == K.sha256_file(src)
    assert all(p.startswith("experiments/hi_f-local-1/") for _, p in api.commits)


def test_kernel_uses_guard_and_new_layout_only():
    assert "milestones/step_{n}" in SRC and "GUARD.put_folder" in SRC
    assert not re.search(r'api\.upload_(file|folder)\(', SRC.split("# ---- END HELPERS ----")[1]), "kernel body must upload through the guard only"
    assert 'hf_hub_download(REPO, f"runs/{NAME}/last.ckpt")' not in SRC, "implicit last.ckpt resume must not come back"
