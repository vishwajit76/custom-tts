"""Kaggle GPU kernel: resume the Hindi Piper fine-tune from an explicit HF checkpoint, train <=11 h, sync back to HF under experiments/<id>/.
Secrets: HF token is read from the private Kaggle dataset cttsh-secrets (file hf_token). Never embed tokens here.

Kaggle script kernels cannot receive environment variables, so per-launch settings live in STAMP below (rewritten by training/kaggle/push.sh, or
edited by hand before `kaggle kernels push`); an environment variable of the same name always wins (used by the CPU smoke test).
  RESUME_FROM   HF path of the checkpoint to resume, or "auto" (default): highest step among experiments/*/checkpoints/*.ckpt and runs/hi_f/*.ckpt
  EXPERIMENT_ID default hi_f-<KERNEL_VERSION|k>-<UTC start>; every upload goes to experiments/<id>/ and is guarded (see ExperimentGuard)
  GIT_SHA       commit the kernel was built from (push.sh stamps it; otherwise "unknown")
Layout written on HF (repo vishwajit76/custom-tts-hindi-train), never anything outside it except the legacy heartbeat/crash files:
  experiments/<id>/{session.json,manifest.json,environment.json,metrics.jsonl,heartbeat.txt,result.json,crash.txt}
  experiments/<id>/checkpoints/{last.ckpt,last.meta.json,final_step<N>.ckpt,final_step<N>.meta.json}
  experiments/<id>/milestones/step_<N>/{*.onnx,*.onnx.json,session.json}   experiments/<id>/samples/step_<N>/sample_*.wav
  experiments/<id>/evaluations/step_<N>.json (compare_checkpoints/v1 entry + paired_vs_previous; create-once; step_<N>.skipped.json marks a timed-out/failed eval)
"""
import glob, hashlib, json, os, pathlib, re, shutil, subprocess, sys, threading, time

T0 = time.time()
STAMP = {}  # push.sh rewrites this line, e.g. {"GIT_SHA": "abc1234", "KERNEL_VERSION": "v6", "RESUME_FROM": "auto"}
for _k, _v in STAMP.items():
    os.environ.setdefault(_k, str(_v))


# ---- BEGIN HELPERS (pure functions and classes; unit-tested by tests/test_kaggle_kernel.py, which execs this block) ----
class CollisionError(RuntimeError):
    """An upload would overwrite a path that belongs to a different experiment/session."""


ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$")
LEGACY_ALLOWED = ("runs/hi_f/kaggle_progress.txt", "runs/hi_f/kaggle_crash.txt")  # the only paths outside experiments/<id>/ that may be written


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_step(path):
    """Step encoded in a checkpoint file name (..._step357212.ckpt), else None."""
    m = re.search(r"step[_-]?(\d+)\.ckpt$", str(path))
    return int(m.group(1)) if m else None


def dataset_fingerprint(data_dir):
    """sha256 of metadata.csv + row/wav counts + sha256 over the sorted 'name:size' list of the wavs (cheap, catches any added/removed/re-encoded clip)."""
    d = pathlib.Path(data_dir)
    meta = d / "metadata.csv"
    rows = [l for l in meta.read_text("utf-8").splitlines() if l.strip()]
    wavs = sorted((p.name, p.stat().st_size) for p in (d / "wavs").glob("*.wav")) if (d / "wavs").exists() else []
    return {"metadata_sha256": sha256_file(meta), "metadata_rows": len(rows), "n_wavs": len(wavs),
            "wavs_name_size_sha256": hashlib.sha256("\n".join(f"{n}:{s}" for n, s in wavs).encode()).hexdigest()}


def ckpt_meta(path, check_finite=True):
    """Load a Lightning checkpoint on CPU and return what the resume/upload logic needs. Raises if it cannot be loaded (truncated/torn file)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    out = {"global_step": int(ck["global_step"]), "epoch": int(ck.get("epoch", -1)), "size": os.path.getsize(path)}
    try:
        out["lr_g"] = ck["optimizer_states"][0]["param_groups"][0]["lr"]
        out["lr_d"] = ck["optimizer_states"][1]["param_groups"][0]["lr"]
    except Exception:  # noqa: BLE001
        out["lr_g"] = out["lr_d"] = None
    out["scheduler"] = [{k: v for k, v in sd.items() if k in ("gamma", "last_epoch", "base_lrs", "_last_lr")} for sd in ck.get("lr_schedulers", [])]
    out["lr_ctl"] = ck.get("callbacks", {}).get("lr_ctl")
    if check_finite:
        bad = [k for k, v in ck["state_dict"].items() if torch.is_tensor(v) and torch.is_floating_point(v) and not bool(torch.isfinite(v).all())]
        out["finite"], out["nonfinite_tensors"] = (not bad), bad[:5]
    return out


def pick_resume(files, probe_step):
    """Highest-step resumable checkpoint among the repo's files. probe_step(path)->int|None is used for names without a step (last.ckpt).
    Only experiments/*/checkpoints/*.ckpt and runs/hi_f/*.ckpt are candidates (milestone ONNX/other files never). Ties prefer experiments/.
    Returns (path, step, considered) or (None, None, considered)."""
    considered, best = [], None
    for f in sorted(files):
        if not f.endswith(".ckpt") or not (f.startswith("runs/hi_f/") or re.match(r"experiments/[^/]+/checkpoints/[^/]+\.ckpt$", f)):
            continue
        s = parse_step(f)
        if s is None:
            try:
                s = probe_step(f)
            except Exception as e:  # noqa: BLE001
                s = None
                considered.append({"path": f, "step": None, "error": repr(e)[:120]})
                continue
        considered.append({"path": f, "step": s})
        if s is None:
            continue
        key = (s, f.startswith("experiments/"), f)
        if best is None or key > best:
            best = key
    return (best[2], best[0], considered) if best else (None, None, considered)


def env_info(extra_env=("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_DOCKER_IMAGE", "KAGGLE_URL_BASE")):
    """Versions and hardware (whitelisted env only: never dumps the environment, tokens live there)."""
    import platform
    from importlib import metadata as md
    def v(name):
        try:
            return md.version(name)
        except Exception:  # noqa: BLE001
            return None
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": {p: v(p) for p in
            ("torch", "torchaudio", "lightning", "pytorch-lightning", "piper-tts", "numpy", "huggingface_hub", "onnx", "onnxruntime", "espeakng-loader")},
            "env": {k: os.environ[k] for k in extra_env if k in os.environ}}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_runtime"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            pr = torch.cuda.get_device_properties(0)
            info["gpu"] = {"name": pr.name, "memory_mb": pr.total_memory // 2**20, "capability": list(torch.cuda.get_device_capability(0))}
    except Exception as e:  # noqa: BLE001
        info["torch_error"] = repr(e)[:100]
    try:
        info["espeak_ng"] = subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True, timeout=10).stdout.strip()[:120]
    except Exception:  # noqa: BLE001
        info["espeak_ng"] = None
    return info


class ExperimentGuard:
    """Every kernel upload goes through here. Writes only under experiments/<id>/ (plus two allow-listed legacy files), and refuses to touch
    an experiment whose remote session.json names a different session, so two sessions (or a relaunch that reuses an id) can never overwrite each
    other's checkpoints, milestones or metrics. Create-once files (finals, milestones, manifest, result) are never overwritten, even by the owner.
    Not atomic (HF has no compare-and-swap): the check-then-write window is a few seconds; ids contain the start minute so a race needs an explicit id clash."""

    def __init__(self, api, repo, exp_id, sid, read_text=None, recheck_s=600):
        if not ID_RE.match(exp_id or ""):
            raise ValueError(f"bad experiment id {exp_id!r}")
        self.api, self.repo, self.id, self.sid = api, repo, exp_id, sid
        self.read_text, self.recheck_s, self._checked, self.claimed = read_text, recheck_s, 0.0, False
        self.log = []  # (commit_message, path)

    def rp(self, rel):
        rel = str(rel).replace("\\", "/")
        if rel.startswith("/") or ".." in rel.split("/"):
            raise CollisionError(f"path escapes the experiment folder: {rel}")
        return f"experiments/{self.id}/{rel}"

    def _owner(self):
        p = self.rp("session.json")
        if not self.api.file_exists(self.repo, p):
            return None
        return json.loads(self.read_text(p)).get("owner_session")

    def claim(self, info):
        owner = self._owner()
        if owner not in (None, self.sid):
            raise CollisionError(f"experiments/{self.id}/ is owned by session {owner!r}, this is {self.sid!r}: pick another EXPERIMENT_ID")
        if owner is None:
            body = json.dumps({**info, "experiment_id": self.id, "owner_session": self.sid}, indent=1, default=str).encode()
            self._up(body, "session.json", f"claim {self.id} ({self.sid})")
        self.claimed, self._checked = True, time.time()

    def _still_owner(self):
        if time.time() - self._checked > self.recheck_s:
            owner = self._owner()
            if owner != self.sid:
                raise CollisionError(f"ownership of experiments/{self.id}/ changed to {owner!r}; refusing to upload")
            self._checked = time.time()

    def _up(self, src, rel, msg):
        p = self.rp(rel)
        self.api.upload_file(path_or_fileobj=src, path_in_repo=p, repo_id=self.repo, commit_message=msg)
        self.log.append((msg, p))
        return p

    def put(self, src, rel, msg, overwrite=False):
        """src: local path or bytes. Returns the repo path, or None when a create-once file already exists (skipped, never overwritten)."""
        if not self.claimed:
            raise CollisionError("claim() first")
        self._still_owner()
        if not overwrite and self.api.file_exists(self.repo, self.rp(rel)):
            return None
        return self._up(src, rel, f"{msg} [{self.id}]")

    def put_folder(self, folder, rel, msg):
        if not self.claimed:
            raise CollisionError("claim() first")
        self._still_owner()
        prefix = self.rp(rel).rstrip("/") + "/"
        if any(f.startswith(prefix) for f in self.api.list_repo_files(self.repo)):
            return None
        self.api.upload_folder(folder_path=str(folder), path_in_repo=self.rp(rel), repo_id=self.repo, commit_message=f"{msg} [{self.id}]")
        self.log.append((msg, prefix))
        return prefix

    def put_legacy(self, src, path, msg):
        """Legacy heartbeat/crash files only (the hourly check-in still reads runs/hi_f/kaggle_progress.txt); allow-listed by exact path."""
        if path not in LEGACY_ALLOWED:
            raise CollisionError(f"{path} is not an allow-listed legacy path")
        self.api.upload_file(path_or_fileobj=src, path_in_repo=path, repo_id=self.repo, commit_message=f"{msg} [{self.id}]")


class LocalApi:
    """Directory-backed stand-in for the subset of huggingface_hub.HfApi the kernel uses (unit tests and the CPU smoke run: nothing leaves the machine)."""

    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.commits = []

    def _p(self, rel):
        return self.root / rel

    def _put(self, src, rel):
        dst = self._p(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if isinstance(src, (bytes, bytearray)):
            dst.write_bytes(bytes(src))
        else:
            try:
                os.link(src, dst)  # 850 MB checkpoints: hard link when the same filesystem, else copy
            except OSError:
                shutil.copy(src, dst)

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id=None, commit_message=None, **kw):
        self._put(path_or_fileobj, path_in_repo)
        self.commits.append((commit_message, path_in_repo))

    def upload_folder(self, *, folder_path, path_in_repo, repo_id=None, commit_message=None, **kw):
        for p in pathlib.Path(folder_path).rglob("*"):
            if p.is_file():
                self._put(p, f"{path_in_repo}/{p.relative_to(folder_path)}")
        self.commits.append((commit_message, path_in_repo + "/"))

    def file_exists(self, repo_id, filename, **kw):
        return self._p(filename).exists()

    def list_repo_files(self, repo_id, **kw):
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file() or p.is_symlink())

    def get_paths_info(self, repo_id, paths, **kw):
        from types import SimpleNamespace as NS
        return [NS(path=p, size=self._p(p).stat().st_size, lfs=NS(sha256=sha256_file(self._p(p)))) for p in paths if self._p(p).exists()]

    def download(self, repo_id, filename):
        return self._p(filename)

    def read_text(self, filename):
        return self._p(filename).read_text("utf-8")


def remote_sha256(api, repo, path):
    try:
        info = api.get_paths_info(repo, [path])
        return info[0].lfs.sha256 if info and info[0].lfs else None
    except Exception:  # noqa: BLE001
        return None

EVAL_RE = re.compile(r"experiments/([^/]+)/evaluations/step_(\d+)\.json$")


def pick_prev_eval(files, step, exp_id=None):
    """Stored evaluation of the closest EARLIER milestone (highest step < `step`, any experiment; ties prefer this experiment). Returns the repo path or None."""
    best = None
    for f in files:
        m = EVAL_RE.match(f)
        if m and int(m.group(2)) < step:
            key = (int(m.group(2)), m.group(1) == exp_id, f)
            best = key if best is None or key > best else best
    return best[2] if best else None


def cuda_lib_dirs():
    """nvidia/*/lib directories of the pip wheels that ship with torch: ctranslate2 (faster-whisper) needs cuDNN 9 / cuBLAS 12 and they are not on the loader path by default."""
    import site
    out = []
    for sp in set(site.getsitepackages() + [site.getusersitepackages()]):
        out += sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
    return out


def gpu_free_mb():
    """Free VRAM of GPU 0 in MB via nvidia-smi (no CUDA context is created in the calling process); None when unknown."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


class EvalManager:
    """Runs milestone evaluations one at a time on a background thread so training is never blocked, and NEVER raises into the caller.
      run_fn(item, prev_path, budget_s) -> ("ok", json_bytes) | ("timeout", msg) | ("error", msg)     item = {"step", "dir"}
      free_mb_fn() -> free VRAM in MB or None;  prev_fn(step) -> local path of the previous stored evaluation or None
      put_fn(rel, bytes) -> upload under experiments/<id>/ (create-once: returns None when the file already exists)
    If less than `min_free_mb` VRAM is free while training runs the item is DEFERRED and runs in finish(), after training has released the GPU.
    A timeout or error stores evaluations/step_<N>.skipped.json (a marker; the real step_<N>.json stays free) and is logged, nothing else happens."""

    def __init__(self, run_fn, put_fn, prev_fn=lambda s: None, free_mb_fn=lambda: None, min_free_mb=3000, budget_s=900, log=print, cleanup=shutil.rmtree):
        import queue
        self.run_fn, self.put_fn, self.prev_fn, self.free_mb_fn = run_fn, put_fn, prev_fn, free_mb_fn
        self.min_free_mb, self.budget_s, self.log, self.cleanup = min_free_mb, budget_s, log, cleanup
        self.q, self.deferred, self.status = queue.Queue(), [], []
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def enqueue(self, step, folder):
        self.q.put({"step": int(step), "dir": str(folder)})

    def _loop(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            try:
                self._process(item, defer_ok=True)
            except Exception as e:  # noqa: BLE001
                self.log(f"eval manager error step {item['step']}: {e!r}")

    def _process(self, item, defer_ok, budget_s=None):
        n, st = item["step"], {"step": item["step"]}
        free = None
        try:
            free = self.free_mb_fn()
        except Exception:  # noqa: BLE001
            pass
        if defer_ok and free is not None and free < self.min_free_mb:
            self.deferred.append(item)
            st.update(status="deferred", reason=f"only {free} MB VRAM free (< {self.min_free_mb}): will run after training")
            self.status.append(st); self.log(f"eval step {n}: {st['reason']}")
            return
        t = time.time()
        try:
            prev = self.prev_fn(n)
        except Exception as e:  # noqa: BLE001
            prev = None
            self.log(f"eval step {n}: previous evaluation unavailable ({e!r}); continuing without paired deltas")
        try:
            kind, payload = self.run_fn(item, prev, budget_s or self.budget_s)
        except Exception as e:  # noqa: BLE001
            kind, payload = "error", repr(e)[:300]
        try:
            if kind == "ok":
                p = self.put_fn(f"evaluations/step_{n}.json", payload)
                st.update(status="ok" if p else "exists", path=p)
            else:
                body = json.dumps({"step": n, "status": kind, "detail": str(payload)[:500], "budget_s": budget_s or self.budget_s, "free_vram_mb": free,
                                   "t_utc": time.strftime("%FT%TZ", time.gmtime())}).encode()
                self.put_fn(f"evaluations/step_{n}.skipped.json", body)
                st.update(status=kind, detail=str(payload)[:200])
        except Exception as e:  # noqa: BLE001
            st.update(status="upload_failed", detail=repr(e)[:200])
        st["seconds"] = round(time.time() - t)
        self.status.append(st)
        self.log(f"eval step {n}: {st}")
        try:
            self.cleanup(item["dir"], ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass

    def finish(self, total_budget_s=1500):
        """After training: drain the queue, then run the deferred items (the GPU is free now), all within `total_budget_s`. Returns the status list."""
        end = time.time() + total_budget_s
        self.q.put(None)
        self.th.join(max(1, end - time.time()))
        if self.th.is_alive():
            self.log("eval worker still busy at the end budget: abandoned (daemon)")
            return self.status
        for item in self.deferred:
            left = end - time.time()
            if left < 60:
                self.status.append({"step": item["step"], "status": "skipped", "detail": "no time left after training"}); continue
            try:
                self._process(item, defer_ok=False, budget_s=min(self.budget_s, left))
            except Exception as e:  # noqa: BLE001
                self.log(f"deferred eval error step {item['step']}: {e!r}")
        self.deferred = []
        return self.status
# ---- END HELPERS ----

MAX_H = float(os.environ.get("MAX_HOURS", "11"))
REPO = "vishwajit76/custom-tts-hindi-train"
NAME = "hi_f"
W = pathlib.Path(os.environ.get("W_ROOT", "/tmp/w")); W.mkdir(parents=True, exist_ok=True)  # big files stay out of /kaggle/working (kernel output)
BS = int(os.environ.get("BS", "24"))
MS_EVERY = int(os.environ.get("MS_EVERY", "5000"))
SMOKE = os.environ.get("SMOKE_LOCAL_DIR")  # CPU smoke test (training/kernel_smoke.py): local directory instead of HF/Kaggle/GPU
UPLOAD_EVERY_S = float(os.environ.get("UPLOAD_EVERY_S", str(20 * 60)))
HB_EVERY_S = float(os.environ.get("HB_EVERY_S", "180"))
METRICS_EVERY_S = float(os.environ.get("METRICS_EVERY_S", str(15 * 60)))
SEED_VALUE = int(os.environ.get("SEED", "1234"))  # NOTE: piper's train/val split is drawn from this seed (random_split): never change it between sessions of one voice
# Session id: unique per kernel run. KERNEL_VERSION (STAMP/env, e.g. "v6") + UTC start time, e.g. "v6-0930T0725Z".
SID = os.environ.get("SESSION_ID") or f"{os.environ.get('KERNEL_VERSION', 'k')}-{time.strftime('%m%dT%H%MZ', time.gmtime(T0))}"
EXPERIMENT_ID = os.environ.get("EXPERIMENT_ID") or f"{NAME}-{SID}"
GIT_SHA = os.environ.get("GIT_SHA", "unknown")
RESUME_FROM = os.environ.get("RESUME_FROM", "auto")
# Learning-rate control. piper-tts 1.8.0 never calls scheduler.step() (manual optimization), so without this LR is CONSTANT at the value
# stored in the checkpoint (1.5258e-4 in the seed and still in the 357212 final: 47k steps later), and it has no --lr_final_ratio.
# The wrapper below implements the anneal. Defaults (env/STAMP override; LR_MODE=keep = checkpoint LR untouched):
#   LR_MODE=anneal  LR_START=1e-4  LR_FINAL_RATIO=0.05  ANNEAL_EPOCHS=150   (LR_D_START defaults to LR_START)
# -> exponential decay LR_START -> LR_START*LR_FINAL_RATIO over ANNEAL_EPOCHS epochs, then held. One epoch = about 302 global steps at bs 24
# (3607 train clips = 4013 - 10% piper validation split - 5 test, 151 batches x 2 optimizers), so 150 epochs = about 45k steps = one 11 h session.
# Verified on CPU with training/lr_dryrun.py and tests/test_lr_anneal.py.
for _k, _v in dict(LR_MODE="anneal", LR_START="1e-4", LR_FINAL_RATIO="0.05", ANNEAL_EPOCHS="150").items():
    os.environ.setdefault(_k, _v)
LR_ENV = {k: os.environ[k] for k in ("LR_MODE", "LR_START", "LR_D_START", "LR_FINAL_RATIO", "ANNEAL_EPOCHS") if os.environ.get(k)}
if LR_ENV.get("LR_MODE", "").lower() != "anneal":
    LR_ENV = {"LR_MODE": "keep"}
SENTS = ["नमस्ते, आप कैसे हैं? आज मौसम बहुत सुहावना है।",
         "भारत एक विशाल देश है, जहाँ अनेक भाषाएँ और संस्कृतियाँ एक साथ मिलकर रहती हैं।",
         "क्या आपने कल रात का खाना खा लिया? मुझे तो बहुत भूख लगी है!"]
EVAL_ON = os.environ.get("EVAL_ENABLE", "1") != "0"
EVAL_BUDGET_S = float(os.environ.get("EVAL_BUDGET_S", str(15 * 60)))  # per milestone; over budget = skipped + logged
EVAL_REPEATS = int(os.environ.get("EVAL_REPEATS", "3"))
EVAL_LIMIT = int(os.environ["EVAL_LIMIT"]) if os.environ.get("EVAL_LIMIT") else None
EVAL_MIN_FREE_MB = int(os.environ.get("EVAL_MIN_FREE_MB", "3000"))  # whisper-small fp16 + CUDA context + beam-5 activations need about 1.5-2 GB; 3 GB margin
EVAL_END_BUDGET_S = float(os.environ.get("EVAL_END_BUDGET_S", str(25 * 60)))  # after the final checkpoint upload: drain + deferred evals
# Files of the repo the evaluation needs, zipped+base64 by training/kaggle/make_bundle.py and written here by push.sh (script kernels upload one file only).
EVAL_BUNDLE_B64 = ""
GUARD = None


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, **kw)


import traceback
def _crash(tp, v, tb):
    txt = "".join(traceback.format_exception(tp, v, tb)); print(txt, flush=True)
    body = (f"session {SID} experiment {EXPERIMENT_ID}\n" + txt).encode()
    try:
        if GUARD is not None and GUARD.claimed:
            GUARD.put(body, "crash.txt", "crash", overwrite=True)
        if GUARD is not None:
            GUARD.put_legacy(body, "runs/hi_f/kaggle_crash.txt", "crash")
    except Exception as e: print(e)
sys.excepthook = _crash

# ---- token (from private dataset; not printed) ----
if not SMOKE:
    cands = glob.glob("/kaggle/input/**/hf_token", recursive=True)
    assert cands, "hf_token not found under /kaggle/input"
    os.environ["HF_TOKEN"] = pathlib.Path(cands[0]).read_text().strip()

# ---- env ----
import torch  # preinstalled
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
if not SMOKE:
    cons = W / "constraints.txt"
    sh(f"pip freeze 2>/dev/null | grep -iE '^(torch|torchaudio|torchvision|numpy)==' > {cons}")
    sh(f"pip install -q -c {cons} 'piper-tts[train]==1.8.0' cython setuptools soundfile soxr huggingface_hub 2>&1 | tail -3")
import piper.train.vits as pv
MA = pathlib.Path(pv.__file__).parent / "monotonic_align"
if not SMOKE and not list((MA / "monotonic_align").glob("core*.so")):
    tmp = pathlib.Path("/tmp/ma"); tmp.mkdir(exist_ok=True)
    sh(f"curl -fsSL -o {tmp}/core.pyx https://raw.githubusercontent.com/OHF-Voice/piper1-gpl/main/src/piper/train/vits/monotonic_align/core.pyx && cd {tmp} && cythonize -i core.pyx >/dev/null 2>&1")
    (MA / "monotonic_align").mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("core*.so"):
        shutil.move(str(f), MA / "monotonic_align" / f.name)
sh("python -c 'from piper.train.vits.monotonic_align import maximum_path; print(\"monotonic_align ok\")'")
if not SMOKE:
    sh("apt-get install -y -q espeak-ng >/dev/null 2>&1 || true")

# ---- data + ckpt from HF ----
if SMOKE:
    api = LocalApi(SMOKE)
    _dl = lambda path: api.download(REPO, path)
    _read = api.read_text
    DATA = pathlib.Path(SMOKE) / "data" / NAME
else:
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
    api = HfApi()
    _dl = lambda path: pathlib.Path(hf_hub_download(REPO, path))
    _read = lambda path: pathlib.Path(hf_hub_download(REPO, path, force_download=True)).read_text("utf-8")
    snapshot_download(REPO, allow_patterns=[f"data/{NAME}/*"], local_dir=str(W / "hf"))
    DATA = W / "hf" / "data" / NAME
RUN = W / "run"; CK = RUN / "lightning_logs" / "version_0" / "checkpoints"; CK.mkdir(parents=True, exist_ok=True)
GUARD = ExperimentGuard(api, REPO, EXPERIMENT_ID, SID, read_text=_read)
shutil.copy(_dl(f"runs/{NAME}/config.json"), RUN / "config.json")

# Resume point: explicit (RESUME_FROM=<hf path>) or the highest-step known checkpoint. Never an implicit "last.ckpt": that name was overwritten by whichever
# session uploaded last (v5's 20-min upload replaced v4's final with the stale 310300 seed).
files = api.list_repo_files(REPO)
def _probe(path):
    m = ckpt_meta(_dl(path), check_finite=False)  # legacy runs/hi_f/last.ckpt has no step in its name: read global_step from the file
    return m["global_step"]
if RESUME_FROM.lower() == "auto":
    RESUME_PATH, RESUME_STEP_HINT, CONSIDERED = pick_resume(files, _probe)
    assert RESUME_PATH, f"no resumable checkpoint found: {CONSIDERED}"
else:
    RESUME_PATH, RESUME_STEP_HINT, CONSIDERED = RESUME_FROM, parse_step(RESUME_FROM), []
    assert RESUME_PATH in files, f"RESUME_FROM={RESUME_PATH} not found on {REPO}"
SEED = _dl(RESUME_PATH)  # resumed from in place and never written again
RESUME_SHA, REMOTE_SHA = sha256_file(SEED), remote_sha256(api, REPO, RESUME_PATH)
assert REMOTE_SHA is None or REMOTE_SHA == RESUME_SHA, f"resume checkpoint sha256 mismatch: local {RESUME_SHA} remote {REMOTE_SHA} (corrupt download)"
CK0 = ckpt_meta(SEED)
assert CK0["finite"], f"resume checkpoint has non-finite weights: {CK0['nonfinite_tensors']}"
assert RESUME_STEP_HINT is None or RESUME_STEP_HINT == CK0["global_step"], f"{RESUME_PATH}: name says step {RESUME_STEP_HINT}, file says {CK0['global_step']}"
START_STEP, START_EPOCH = CK0["global_step"], CK0["epoch"] + 1
CK_LR = dict(lr_g=CK0["lr_g"], lr_d=CK0["lr_d"], sched=CK0["scheduler"], lr_ctl=CK0["lr_ctl"])
# NEW checkpoints go to their own directory (the seed is never rewritten; Lightning would write version_1 next to it, see docs/training-progress.md).
OUT_CK = W / "ckpt"; OUT_CK.mkdir(exist_ok=True)
LAST = OUT_CK / "last.ckpt"
print(f"session {SID} experiment {EXPERIMENT_ID}: resuming {RESUME_PATH} global_step={START_STEP} epoch={START_EPOCH} sha256={RESUME_SHA[:12]} "
      f"LR in checkpoint={CK_LR} LR env={LR_ENV}", flush=True)
FP = dataset_fingerprint(DATA)
print("dataset", FP, flush=True)

# ---- trainer wrapper: ckpt every 500 steps, milestone copies, deadline stop, metrics.jsonl ----
(W / "ms").mkdir(exist_ok=True)
WRAP_SRC = r'''
import os, time, json, math, torch
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from piper.train import __main__ as m
W = os.environ.get("W_DIR", "/tmp/w"); LAST = os.environ["LAST_CKPT"]
DEADLINE = float(os.environ["DEADLINE"]); MS = int(os.environ.get("MS_EVERY", "5000")); MET = max(1, int(os.environ.get("METRICS_EVERY_STEPS", "20")))

def _lrs(tr):
    try: return [float(o.param_groups[0]["lr"]) for o in tr.optimizers]
    except Exception: return []

class LrCtl(Callback):
    """LR_MODE=anneal: set LR each epoch to start * ratio**(min(epoch-e0, N)/N). Overrides whatever the resumed checkpoint restored
    (Lightning restores optimizer param_groups lr and scheduler state BEFORE on_train_start, so CLI flags such as --model.learning_rate
    are ignored on resume). e0 is stored in the checkpoint, so a later session with the same settings continues the anneal instead of restarting it.
    Default (LR_MODE unset): does nothing and saves nothing."""
    state_key = "lr_ctl"
    def __init__(s):
        e = os.environ.get
        s.on = e("LR_MODE", "").lower() == "anneal"
        s.start = float(e("LR_START", "1e-4")); s.dstart = float(e("LR_D_START", "") or s.start)
        s.ratio = float(e("LR_FINAL_RATIO", "0.05")); s.n = max(1, int(e("ANNEAL_EPOCHS", "150")))
        s.e0 = None; s.saved = {}
    def cfg(s): return [s.start, s.dstart, s.ratio, s.n]
    def state_dict(s): return dict(e0=s.e0, cfg=s.cfg()) if s.on and s.e0 is not None else {}
    def load_state_dict(s, sd): s.saved = sd or {}
    def lr_at(s, epoch, base): return base * s.ratio ** (min(max(epoch - s.e0, 0), s.n) / s.n)
    def apply(s, tr):
        for opt, base in zip(tr.optimizers, (s.start, s.dstart)):
            for g in opt.param_groups: g["lr"] = s.lr_at(tr.current_epoch, base)
    def on_train_start(s, tr, pl):
        print(f"LR at train start (after checkpoint restore): {_lrs(tr)} epoch={tr.current_epoch} mode={'anneal' if s.on else 'keep'}", flush=True)
        if not s.on: return
        cont = s.saved.get("e0") is not None and s.saved.get("cfg") == s.cfg()
        s.e0 = s.saved["e0"] if cont else tr.current_epoch
        for c in tr.lr_scheduler_configs: c.scheduler.gamma = 1.0  # piper never steps them; make sure nothing else can decay
        s.apply(tr)
        print(f"LR anneal {'CONTINUES' if cont else 'STARTS'} e0={s.e0} cfg={s.cfg()} -> now {_lrs(tr)}", flush=True)
    def on_train_epoch_start(s, tr, pl):
        if s.on: s.apply(tr)
        print(f"epoch {tr.current_epoch} step {tr.global_step} lr {_lrs(tr)}", flush=True)

class Ctl(Callback):
    def __init__(s): s.last_ms = None; s.t = time.time()
    def on_train_start(s, tr, pl):
        s.last_ms = tr.global_step // MS; s.g0 = tr.global_step; s.t0 = time.time()
        open(W + "/prog.json", "w").write(json.dumps(dict(step=tr.global_step, sps=None, t=time.time(), started=True, lr=_lrs(tr))))
        try:  # piper draws train/val/test with random_split under --seed_everything: record the split so sessions can be checked for validation leakage
            import hashlib
            dm = tr.datamodule
            sp = {n: sorted(int(i) for i in getattr(dm, n + "_dataset").indices) for n in ("train", "val", "test")}
            sp = dict({n + "_n": len(v) for n, v in sp.items()}, **{n + "_sha256": hashlib.sha256(json.dumps(v).encode()).hexdigest() for n, v in sp.items()})
            open(W + "/val_split.json", "w").write(json.dumps(sp)); print("SPLIT", sp, flush=True)
        except Exception as e: print("split fingerprint unavailable:", repr(e), flush=True)
    def on_train_batch_end(s, tr, pl, *a, **k):
        g = tr.global_step
        if (g - s.g0) % MET == 0 and g > s.g0:
            d = dict(step=g, sps=(time.time() - s.t0) / (g - s.g0), t=time.time(), epoch=tr.current_epoch, lr=_lrs(tr))  # sps = SECONDS per global step
            try: d.update({k2: float(v) for k2, v in tr.callback_metrics.items()})
            except Exception: pass
            open(W + "/prog.json.tmp", "w").write(json.dumps(d)); os.replace(W + "/prog.json.tmp", W + "/prog.json")
            try:  # metrics.jsonl: one line per 20-step heartbeat, append-only
                row = {k2: v for k2, v in d.items() if k2 != "t"}
                row["t_utc"] = time.strftime("%FT%TZ", time.gmtime(d["t"]))
                row["nonfinite"] = [k2 for k2, v in row.items() if isinstance(v, float) and not math.isfinite(v)]
                open(W + "/metrics.jsonl", "a").write(json.dumps(row) + "\n")
            except Exception: pass
        if g // MS > s.last_ms:
            s.last_ms = g // MS
            tr.save_checkpoint(W + "/ms/tmp.ckpt"); os.replace(W + "/ms/tmp.ckpt", W + f"/ms/step_{g}.ckpt")
        if time.time() > DEADLINE and not tr.should_stop:
            print("DEADLINE reached: saving and stopping", flush=True)
            tr.save_checkpoint(LAST); tr.should_stop = True
m._DEFAULT_CALLBACKS[:] = [ModelCheckpoint(dirpath=os.environ["CKPT_DIR"], save_last=True, save_top_k=0, every_n_train_steps=int(os.environ.get("CKPT_EVERY", "500"))), LrCtl(), Ctl()]
if __name__ == "__main__": m.main()
'''
(W / "wrap.py").write_text(WRAP_SRC)

cmd = [sys.executable, str(W / "wrap.py"), "fit", "--seed_everything", str(SEED_VALUE),
       "--data.voice_name", NAME, "--data.csv_path", str(DATA / "metadata.csv"), "--data.audio_dir", str(DATA / "wavs"),
       "--data.espeak_voice", "hi", "--data.cache_dir", str(RUN / "cache"), "--data.config_path", str(RUN / "config.json"),
       "--data.batch_size", str(BS), "--data.num_workers", "0" if SMOKE else "3", "--model.sample_rate", "22050", "--model.num_speakers", "1",
       "--trainer.default_root_dir", str(RUN), "--trainer.accelerator", "cpu" if SMOKE else "gpu", "--trainer.devices", "1",
       "--trainer.precision", "32-true" if SMOKE else "16-mixed", "--trainer.max_epochs", str(START_EPOCH + (int(os.environ.get("SMOKE_EPOCHS", "1000")) if SMOKE else 100000)),
       "--trainer.log_every_n_steps", "50", "--ckpt_path", str(SEED)]
if SMOKE:
    cmd += ["--trainer.enable_progress_bar", "false"]
env = dict(os.environ, DEADLINE=str(T0 + MAX_H * 3600), PYTHONUNBUFFERED="1", W_DIR=str(W), LAST_CKPT=str(LAST), CKPT_DIR=str(OUT_CK), MS_EVERY=str(MS_EVERY),
           **({"CUDA_VISIBLE_DEVICES": ""} if SMOKE else {}))

# ---- experiment claim + launch manifest (create-once) ----
def _script_sha():
    try: return sha256_file(__file__)
    except Exception: return None
GUARD.claim({"session": SID, "git_sha": GIT_SHA, "t_utc": time.strftime("%FT%TZ", time.gmtime(T0))})
MANIFEST = {
    "experiment_id": EXPERIMENT_ID, "session": SID, "git_sha": GIT_SHA, "kernel_script_sha256": _script_sha(),
    "kernel": "vishwajit76/custom-tts-hindi-train", "started_utc": time.strftime("%FT%TZ", time.gmtime(T0)),
    "started_ist": time.strftime("%FT%T+05:30", time.gmtime(T0 + 19800)), "max_hours": MAX_H,
    "resume": {"requested": RESUME_FROM, "path": RESUME_PATH, "global_step": START_STEP, "epoch": CK0["epoch"], "sha256": RESUME_SHA,
               "remote_sha256": REMOTE_SHA, "considered": CONSIDERED},
    "dataset": {"repo_path": f"data/{NAME}", **FP}, "seed": SEED_VALUE, "batch_size": BS, "precision": "32-true" if SMOKE else "16-mixed",
    "lr": {"env": LR_ENV, "checkpoint": CK_LR}, "meta_note": "sps in metrics.jsonl/heartbeat is seconds per global step",
    "cli": [str(x) for x in cmd], "smoke": bool(SMOKE),
}
GUARD.put(json.dumps(MANIFEST, indent=1, default=str).encode(), "manifest.json", "manifest")
GUARD.put(json.dumps(env_info(), indent=1, default=str).encode(), "environment.json", "environment")

LAST_UP, LAST_UP_STEP, UPLOAD_LOCK = "never", -1, threading.Lock()
def _meta_json(m, extra=None):
    return json.dumps({**{k: m[k] for k in ("global_step", "epoch", "sha256", "size", "lr_g", "lr_d")}, "experiment_id": EXPERIMENT_ID, "session": SID,
                       "t_utc": time.strftime("%FT%TZ", time.gmtime()), **(extra or {})}, indent=1).encode()

def upload_ckpt(src, final=False):
    """Validate (loads, finite, step advanced), copy, upload with sha256 read-back, then the small meta file. Torn writes are detected by the load."""
    global LAST_UP, LAST_UP_STEP
    with UPLOAD_LOCK:
        tmp = W / "upload_last.ckpt"
        try:
            shutil.copy(src, tmp)
            m = ckpt_meta(tmp)
            if not m["finite"]:
                LAST_UP = f"REFUSED non-finite {m['nonfinite_tensors']}"; print("upload refused:", LAST_UP, flush=True); return
            if m["global_step"] <= LAST_UP_STEP and not final:
                return
            m["sha256"] = sha256_file(tmp)
            rel = f"checkpoints/final_step{m['global_step']}.ckpt" if final else "checkpoints/last.ckpt"
            p = GUARD.put(tmp, rel, "final" if final else "ckpt", overwrite=not final)
            if p is None:
                LAST_UP = f"exists, not overwritten: {rel}"; print(LAST_UP, flush=True); return
            rsha = remote_sha256(api, REPO, p)
            if rsha is not None and rsha != m["sha256"]:
                LAST_UP = f"FAILED sha256 read-back mismatch {rel}"; print(LAST_UP, flush=True); return
            GUARD.put(_meta_json(m, {"readback_sha256_ok": rsha is not None}), rel[:-5] + ".meta.json", "meta", overwrite=True)
            GUARD.put(RUN / "config.json", "config.json", "config", overwrite=True)
            LAST_UP_STEP = m["global_step"]; LAST_UP = f"ok {time.strftime('%T')} step {m['global_step']}"; print("HF upload ok", rel, LAST_UP, flush=True)
            if final: return m
        except Exception as e:
            LAST_UP = "FAILED " + repr(e)[:150]; print("HF upload failed", repr(e), flush=True)
        finally:
            tmp.unlink(missing_ok=True)


def milestone(ckpt):
    n = pathlib.Path(ckpt).stem.split("_")[1]
    out = W / f"export_{n}_{SID}"; out.mkdir(exist_ok=True)
    onnx = out / "hi_IN-custom-medium.onnx"
    exp = ("import functools,runpy,torch;torch.onnx.export=functools.partial(torch.onnx.export,dynamo=False);"
           "runpy.run_module('piper.train.export_onnx',run_name='__main__')")
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    try:
        r = subprocess.run([sys.executable, "-c", exp, "--checkpoint", ckpt, "--output-file", str(onnx)], env=e)
        if r.returncode:
            print("export failed", flush=True); return
        shutil.copy(RUN / "config.json", str(onnx) + ".json")
        (out / "session.json").write_text(json.dumps(dict(experiment_id=EXPERIMENT_ID, session=SID, git_sha=GIT_SHA, step=int(n), start_step=START_STEP, resume_from=RESUME_PATH,
                                                         lr_env=LR_ENV, lr_at_ckpt_start=CK_LR.get("lr_g"), t_utc=time.strftime("%FT%TZ", time.gmtime()))))
        smp = W / f"samples_{n}_{SID}"; smp.mkdir(exist_ok=True)
        for i, t in enumerate(SENTS, 1):
            subprocess.run([sys.executable, "-m", "piper", "-m", str(onnx), "-f", str(smp / f"sample_{i}.wav")], input=t.encode(), env=e)
        if GUARD.put_folder(out, f"milestones/step_{n}", f"milestone {n} {SID}") is None:
            print("milestone exists remotely, NOT overwritten:", n, flush=True)
        else:
            GUARD.put_folder(smp, f"samples/step_{n}", f"samples {n} {SID}")
            print("milestone uploaded", n, SID, flush=True)
            if EVAL_MGR is not None:  # never blocks or raises: copy the ONNX aside (the export dir is removed below) and queue the eval
                try:
                    q_ = W / f"evalq_{n}_{SID}"; q_.mkdir(exist_ok=True)
                    for f_ in ("hi_IN-custom-medium.onnx", "hi_IN-custom-medium.onnx.json"): shutil.copy(out / f_, q_ / f_)
                    EVAL_MGR.enqueue(int(n), q_)
                except Exception as ex2: print("eval enqueue failed", n, repr(ex2), flush=True)
    except Exception as ex:
        print("milestone failed", n, repr(ex), flush=True)
    finally:
        shutil.rmtree(out, ignore_errors=True); shutil.rmtree(W / f"samples_{n}_{SID}", ignore_errors=True); os.remove(ckpt)


# ---- in-kernel milestone evaluation (bench/kernel_eval.py in a subprocess; results -> experiments/<id>/evaluations/step_<N>.json, create-once) ----
# Why GPU + subprocess: on 4 CPUs the CER/PER/UTMOS/telephony protocol (50 sentences x 3 repeats, 3 ASR passes each) takes ~2.5 h per milestone; faster-whisper
# small fp16 on the T4 needs a few minutes. A subprocess isolates CUDA/ctranslate2 crashes and can be killed at the budget; it runs niced with 2 threads so the
# 3 dataloader workers keep their CPUs; Piper synthesis stays on CPU (onnxruntime). VRAM: whisper-small fp16 ~1.5-2 GB, only while an eval runs (~5-15 min per
# 5000 steps); if nvidia-smi shows < EVAL_MIN_FREE_MB free the eval is deferred until training ended instead of risking a trainer OOM (which would shrink the batch).
EVAL_SRC = W / "evalsrc"
EVAL_MGR = None
def _eval_setup():
    """Unpack the vendored repo files (or use the checkout in the smoke test). Returns the directory that goes on PYTHONPATH."""
    if SMOKE:
        return str(pathlib.Path(__file__).resolve().parents[2])
    import base64, io, zipfile
    assert EVAL_BUNDLE_B64, "EVAL_BUNDLE_B64 is empty: the kernel was not pushed with training/kaggle/push.sh"
    zipfile.ZipFile(io.BytesIO(base64.b64decode(EVAL_BUNDLE_B64))).extractall(EVAL_SRC)
    cons = W / "constraints.txt"
    # after training started (this runs on the first milestone): never let pip move torch/numpy; resemblyzer needs webrtcvad (wheels package) + librosa
    sh(f"pip install -q -c {cons} faster-whisper soxr onnxruntime librosa webrtcvad-wheels 2>&1 | tail -3")
    sh(f"pip install -q -c {cons} --no-deps resemblyzer 2>&1 | tail -2")
    return str(EVAL_SRC)

_EVAL_PP = []
def _eval_run(item, prev, budget_s):
    if not _EVAL_PP:
        _EVAL_PP.append(_eval_setup())
    out = pathlib.Path(item["dir"]) / "evaluation.json"
    meta = {"path": f"experiments/{EXPERIMENT_ID}/milestones/step_{item['step']}", "experiment_id": EXPERIMENT_ID, "git_sha": GIT_SHA, "session": SID}
    cmd_ = [sys.executable, "-m", "bench.kernel_eval", "--onnx", str(pathlib.Path(item["dir"]) / "hi_IN-custom-medium.onnx"), "--step", str(item["step"]),
            "--data-dir", str(DATA), "--out", str(out), "--budget-s", str(budget_s), "--repeats", str(EVAL_REPEATS), "--asr", "small", "--meta-json", json.dumps(meta)]
    if EVAL_LIMIT: cmd_ += ["--limit", str(EVAL_LIMIT)]
    if prev: cmd_ += ["--prev", str(prev)]
    if SMOKE: cmd_ += ["--fake", "--n-refs", "3"]
    else: cmd_ += ["--device", "cuda", "--compute", "float16"]
    libs = [] if SMOKE else cuda_lib_dirs()
    e_ = dict(os.environ, PYTHONPATH=_EVAL_PP[0] + os.pathsep + os.environ.get("PYTHONPATH", ""), OMP_NUM_THREADS="2", HF_HUB_DISABLE_PROGRESS_BARS="1", PYTHONUNBUFFERED="1",
              LD_LIBRARY_PATH=os.pathsep.join(libs + [os.environ.get("LD_LIBRARY_PATH", "")]))
    if SMOKE: e_["CUDA_VISIBLE_DEVICES"] = ""
    log_ = pathlib.Path(item["dir"]) / "eval.log"
    try:
        with open(log_, "w") as lf:
            r = subprocess.run(cmd_, env=e_, stdout=lf, stderr=subprocess.STDOUT, cwd=_EVAL_PP[0], timeout=budget_s + 120, preexec_fn=lambda: os.nice(10))
    except subprocess.TimeoutExpired:
        return "timeout", f"subprocess killed after {budget_s + 120:.0f}s"
    tail_ = log_.read_text(errors="replace")[-600:]
    print("eval log tail:", tail_.replace("\n", " | ")[-400:], flush=True)
    if r.returncode == 3: return "timeout", f"budget {budget_s:.0f}s exhausted"
    if r.returncode or not out.exists(): return "error", f"rc={r.returncode}: {tail_[-300:]}"
    return "ok", out.read_bytes()

def _eval_prev(step):
    p = pick_prev_eval(api.list_repo_files(REPO), step, EXPERIMENT_ID)
    return _dl(p) if p else None

def _eval_put(rel, body):
    return GUARD.put(body, rel, "evaluation")

if EVAL_ON:
    EVAL_MGR = EvalManager(_eval_run, _eval_put, _eval_prev, (lambda: None) if SMOKE else gpu_free_mb, EVAL_MIN_FREE_MB, EVAL_BUDGET_S)


SPLIT_UP = False
def upload_split():
    global SPLIT_UP
    p = W / "val_split.json"
    if not SPLIT_UP and p.exists():
        try: GUARD.put(p, "val_split.json", "split"); SPLIT_UP = True
        except Exception as e: print("split upload failed", repr(e), flush=True)


def upload_metrics():
    p = W / "metrics.jsonl"
    if p.exists() and p.stat().st_size:
        try: GUARD.put(p, "metrics.jsonl", "metrics", overwrite=True)
        except Exception as e: print("metrics upload failed", repr(e), flush=True)


stop = threading.Event()
def bg():
    last = last_m = time.time()
    while not stop.wait(15):
        for c in sorted(glob.glob(str(W / "ms" / "step_*.ckpt"))):
            milestone(c)
        if LAST.exists() and time.time() - last > UPLOAD_EVERY_S:
            upload_ckpt(LAST); last = time.time()
        upload_split()
        if time.time() - last_m > METRICS_EVERY_S:
            upload_metrics(); last_m = time.time()
th = threading.Thread(target=bg, daemon=True); th.start()

tail = []
def hb():  # heartbeat: small progress file on HF (Kaggle shows no live logs for script kernels)
    while not stop.wait(HB_EVERY_S):
        try:
            try: pr = json.loads((W / "prog.json").read_text()); pr["age_s"] = round(time.time() - pr.pop("t"))
            except Exception as e: pr = f"no trainer step yet ({e!r})"
            try: gpu = torch.cuda.get_device_name(0)
            except Exception: gpu = "?"
            lk = LAST.stat() if LAST.exists() else None
            txt = (f"{time.strftime('%FT%TZ', time.gmtime())} session={SID} experiment={EXPERIMENT_ID} elapsed_h={(time.time()-T0)/3600:.2f} start_step={START_STEP} bs={BS} gpu={gpu} lr_env={LR_ENV or 'keep'}\n"
                   f"trainer_progress={pr}  (sps = seconds per global step)\nlast_ckpt_local={'%d B, %ds ago' % (lk.st_size, time.time()-lk.st_mtime) if lk else None} last_upload={LAST_UP}\n"
                   + "\n".join(tail[-20:]))
            GUARD.put(txt.encode(), "heartbeat.txt", "heartbeat", overwrite=True)
            if os.environ.get("LEGACY_HEARTBEAT", "1") == "1":  # legacy path read by the hourly check-in; the first line names the session
                GUARD.put_legacy(txt.encode(), "runs/hi_f/kaggle_progress.txt", "heartbeat")
        except Exception as e:
            print("hb fail", repr(e), flush=True)
threading.Thread(target=hb, daemon=True).start()

BSS = [BS] + [b for b in (24, 16, 12, 8) if b < BS]
OOM_RETRIES = 0
for BS_TRY in BSS:
    cmd[cmd.index("--data.batch_size") + 1] = str(BS_TRY); BS = BS_TRY
    resume = SEED
    if LAST.exists():  # after an OOM retry keep any progress already saved, but only if the file is complete
        try:
            if ckpt_meta(LAST, check_finite=False)["global_step"] > START_STEP: resume = LAST
        except Exception as e: print("last.ckpt unreadable, resuming from the seed:", repr(e), flush=True)
    cmd[cmd.index("--ckpt_path") + 1] = str(resume)
    tail.clear(); tail.append(f"batch_size={BS_TRY}")
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    buf = b""; oom = False
    while True:
        ch = p.stdout.read1(4096)
        if not ch: break
        sys.stdout.buffer.write(ch); sys.stdout.buffer.flush()
        buf += ch; oom = oom or b"out of memory" in ch or b"OutOfMemory" in ch
        parts = buf.replace(b"\r", b"\n").split(b"\n"); buf = parts.pop()
        tail.extend(x.decode("utf8", "replace")[-300:] for x in parts if x.strip()); del tail[:-60]
    rc = p.wait()
    if rc and oom: print("CUDA OOM -> smaller batch", flush=True); OOM_RETRIES += 1; continue
    break

if rc:
    body = (f"session {SID} experiment {EXPERIMENT_ID}\n" + "\n".join(tail[-60:])).encode()
    try:
        GUARD.put(body, "crash.txt", "crash", overwrite=True); GUARD.put_legacy(body, "runs/hi_f/kaggle_crash.txt", "crash")
    except Exception: pass
print("trainer exit", rc, flush=True)
stop.set(); th.join()
for c in sorted(glob.glob(str(W / "ms" / "step_*.ckpt"))):
    milestone(c)
upload_metrics(); upload_split()
FINAL = None
if LAST.exists():
    upload_ckpt(LAST)  # checkpoints/last.ckpt == the final weights (skipped when that step is already uploaded)
    FINAL = upload_ckpt(LAST, final=True)  # create-once checkpoints/final_step<N>.ckpt
EVAL_STATUS = []
if EVAL_MGR is not None:  # AFTER the final checkpoint is safe on HF: drain queued evals and run the deferred ones on the now-free GPU (capped, never raises)
    try: EVAL_STATUS = EVAL_MGR.finish(EVAL_END_BUDGET_S)
    except Exception as e: print("eval finish failed", repr(e), flush=True)
try:
    GUARD.put(json.dumps({"experiment_id": EXPERIMENT_ID, "session": SID, "exit_code": rc, "batch_size_used": BS, "oom_retries": OOM_RETRIES,
                          "start_step": START_STEP, "final": FINAL, "last_upload": LAST_UP, "evaluations": EVAL_STATUS, "elapsed_h": round((time.time() - T0) / 3600, 3),
                          "ended_utc": time.strftime("%FT%TZ", time.gmtime())}, indent=1, default=str).encode(), "result.json", "result")
except Exception as e: print("result upload failed", repr(e), flush=True)
