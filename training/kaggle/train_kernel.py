"""Kaggle GPU kernel: resume the Hindi Piper fine-tune from HF (runs/hi_f/last.ckpt), train <=11 h, sync back to HF.
Secrets: HF token is read from the private Kaggle dataset cttsh-secrets (file hf_token). Never embed tokens here."""
import glob, json, os, pathlib, shutil, subprocess, sys, threading, time

T0 = time.time()
MAX_H = float(os.environ.get("MAX_HOURS", "11"))
REPO = "vishwajit76/custom-tts-hindi-train"
NAME = "hi_f"
W = pathlib.Path("/tmp/w"); W.mkdir(exist_ok=True)  # big files stay out of /kaggle/working (kernel output)
BS = int(os.environ.get("BS", "24"))
MS_EVERY = 5000
# Session id: unique per kernel run so concurrent sessions never share a milestone folder or heartbeat file.
# KERNEL_VERSION (optional env, e.g. "v6") + UTC start time, e.g. "v6-0930T0725Z".
SID = os.environ.get("SESSION_ID") or f"{os.environ.get('KERNEL_VERSION', 'k')}-{time.strftime('%m%dT%H%MZ', time.gmtime(T0))}"
# Learning-rate control. piper-tts 1.8.0 never calls scheduler.step() (manual optimization), so without this LR is CONSTANT at the value
# stored in last.ckpt (1.5258e-4 in the seed), and it has no --lr_final_ratio / max_epochs-derived decay. The wrapper below implements the anneal.
# Defaults (env overrides them; LR_MODE=keep restores the old behaviour = checkpoint LR untouched):
#   LR_MODE=anneal  LR_START=1e-4  LR_FINAL_RATIO=0.05  ANNEAL_EPOCHS=160   (LR_D_START defaults to LR_START)
# -> exponential decay LR_START -> LR_START*LR_FINAL_RATIO over ANNEAL_EPOCHS epochs (about 336 global steps each), then held.
# Verified on CPU with training/lr_dryrun.py (resume from the real HF checkpoint: default keeps LR, anneal overrides it, a later resume continues it).
for _k, _v in dict(LR_MODE="anneal", LR_START="1e-4", LR_FINAL_RATIO="0.05", ANNEAL_EPOCHS="160").items():
    os.environ.setdefault(_k, _v)
LR_ENV = {k: os.environ[k] for k in ("LR_MODE", "LR_START", "LR_D_START", "LR_FINAL_RATIO", "ANNEAL_EPOCHS") if os.environ.get(k)}
if LR_ENV.get("LR_MODE", "").lower() != "anneal":
    LR_ENV = {"LR_MODE": "keep"}
SENTS = ["नमस्ते, आप कैसे हैं? आज मौसम बहुत सुहावना है।",
         "भारत एक विशाल देश है, जहाँ अनेक भाषाएँ और संस्कृतियाँ एक साथ मिलकर रहती हैं।",
         "क्या आपने कल रात का खाना खा लिया? मुझे तो बहुत भूख लगी है!"]


def sh(cmd, **kw):
    print("+", cmd, flush=True)
    return subprocess.run(cmd, shell=True, **kw)


import traceback
def _crash(tp, v, tb):
    txt = "".join(traceback.format_exception(tp, v, tb)); print(txt, flush=True)
    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(path_or_fileobj=(f"session {SID}\n" + txt).encode(), path_in_repo=f"runs/{NAME}/kaggle_crash.txt", repo_id=REPO)
    except Exception as e: print(e)
sys.excepthook = _crash

# ---- token (from private dataset; not printed) ----
cands = glob.glob("/kaggle/input/**/hf_token", recursive=True)
assert cands, "hf_token not found under /kaggle/input"
os.environ["HF_TOKEN"] = pathlib.Path(cands[0]).read_text().strip()

# ---- env ----
import torch  # preinstalled
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
cons = W / "constraints.txt"
sh(f"pip freeze 2>/dev/null | grep -iE '^(torch|torchaudio|torchvision|numpy)==' > {cons}")
sh(f"pip install -q -c {cons} 'piper-tts[train]==1.8.0' cython setuptools soundfile soxr huggingface_hub 2>&1 | tail -3")
import piper.train.vits as pv
MA = pathlib.Path(pv.__file__).parent / "monotonic_align"
if not list((MA / "monotonic_align").glob("core*.so")):
    tmp = pathlib.Path("/tmp/ma"); tmp.mkdir(exist_ok=True)
    sh(f"curl -fsSL -o {tmp}/core.pyx https://raw.githubusercontent.com/OHF-Voice/piper1-gpl/main/src/piper/train/vits/monotonic_align/core.pyx && cd {tmp} && cythonize -i core.pyx >/dev/null 2>&1")
    (MA / "monotonic_align").mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("core*.so"):
        shutil.move(str(f), MA / "monotonic_align" / f.name)
sh("python -c 'from piper.train.vits.monotonic_align import maximum_path; print(\"monotonic_align ok\")'")
sh("apt-get install -y -q espeak-ng >/dev/null 2>&1 || true")

# ---- data + ckpt from HF ----
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
api = HfApi()
snapshot_download(REPO, allow_patterns=[f"data/{NAME}/*", f"runs/{NAME}/config.json"], local_dir=str(W / "hf"))
DATA = W / "hf" / "data" / NAME
RUN = W / "run"; CK = RUN / "lightning_logs" / "version_0" / "checkpoints"; CK.mkdir(parents=True, exist_ok=True)
shutil.copy(hf_hub_download(REPO, f"runs/{NAME}/last.ckpt"), CK / "last.ckpt")
shutil.copy(W / "hf" / "runs" / NAME / "config.json", RUN / "config.json")
SEED = CK / "last.ckpt"  # the downloaded resume point; resumed from in place and never written again
# NEW checkpoints go to their own directory. Lightning creates lightning_logs/version_1 (version_0 already exists because of SEED), so the
# old `LAST = version_0/last.ckpt` was never rewritten by ModelCheckpoint: the 20-min uploads re-sent the unchanged seed (HF skips the
# identical file, so no "ckpt kaggle" commit ever appeared) and only the DEADLINE save produced a fresh file.
OUT_CK = W / "ckpt"; OUT_CK.mkdir(exist_ok=True)
LAST = OUT_CK / "last.ckpt"
ck0 = torch.load(SEED, map_location="cpu", weights_only=False)
START_STEP, START_EPOCH = int(ck0["global_step"]), int(ck0.get("epoch", -1)) + 1
CK_LR = {}
try:
    CK_LR = dict(lr_g=ck0["optimizer_states"][0]["param_groups"][0]["lr"], lr_d=ck0["optimizer_states"][1]["param_groups"][0]["lr"],
                 sched=[{k: v for k, v in sd.items() if k in ("gamma", "last_epoch", "base_lrs", "_last_lr")} for sd in ck0.get("lr_schedulers", [])],
                 lr_ctl=ck0.get("callbacks", {}).get("lr_ctl"))
except Exception as e:
    CK_LR = dict(error=repr(e))
del ck0
print(f"session {SID}: resuming global_step={START_STEP} epoch={START_EPOCH} LR in checkpoint={CK_LR} LR env={LR_ENV or 'none (keep checkpoint LR)'}", flush=True)

# ---- trainer wrapper: ckpt every 500 steps, milestone copies, deadline stop ----
(W / "ms").mkdir(exist_ok=True)
WRAP_SRC = r'''
import os, time, json, math, torch
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from piper.train import __main__ as m
W = os.environ.get("W_DIR", "/tmp/w"); LAST = os.environ["LAST_CKPT"]
DEADLINE = float(os.environ["DEADLINE"]); MS = int(os.environ.get("MS_EVERY", "5000"))

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
        s.ratio = float(e("LR_FINAL_RATIO", "0.05")); s.n = max(1, int(e("ANNEAL_EPOCHS", "160")))
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
    def on_train_batch_end(s, tr, pl, *a, **k):
        g = tr.global_step
        if (g - s.g0) % 20 == 0 and g > s.g0:
            d = dict(step=g, sps=(time.time() - s.t0) / (g - s.g0), t=time.time(), epoch=tr.current_epoch, lr=_lrs(tr))
            try: d.update({k2: float(v) for k2, v in tr.callback_metrics.items()})
            except Exception: pass
            open(W + "/prog.json.tmp", "w").write(json.dumps(d)); os.replace(W + "/prog.json.tmp", W + "/prog.json")
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

cmd = [sys.executable, "/tmp/w/wrap.py", "fit", "--seed_everything", "1234",
       "--data.voice_name", NAME, "--data.csv_path", str(DATA / "metadata.csv"), "--data.audio_dir", str(DATA / "wavs"),
       "--data.espeak_voice", "hi", "--data.cache_dir", str(RUN / "cache"), "--data.config_path", str(RUN / "config.json"),
       "--data.batch_size", str(BS), "--data.num_workers", "3", "--model.sample_rate", "22050", "--model.num_speakers", "1",
       "--trainer.default_root_dir", str(RUN), "--trainer.accelerator", "gpu", "--trainer.devices", "1",
       "--trainer.precision", "16-mixed", "--trainer.max_epochs", str(START_EPOCH + 100000),
       "--trainer.log_every_n_steps", "50", "--ckpt_path", str(SEED)]
env = dict(os.environ, DEADLINE=str(T0 + MAX_H * 3600), PYTHONUNBUFFERED="1", W_DIR=str(W), LAST_CKPT=str(LAST), CKPT_DIR=str(OUT_CK), MS_EVERY=str(MS_EVERY))


LAST_UP = "never"
def upload_last(tag="ckpt"):
    global LAST_UP
    try:
        tmp = W / "upload_last.ckpt"; shutil.copy(LAST, tmp)
        api.upload_file(path_or_fileobj=str(tmp), path_in_repo=f"runs/{NAME}/last.ckpt", repo_id=REPO, commit_message=f"{tag} kaggle")
        api.upload_file(path_or_fileobj=str(RUN / "config.json"), path_in_repo=f"runs/{NAME}/config.json", repo_id=REPO)
        LAST_UP = "ok " + time.strftime("%T"); print("HF upload ok", tag, LAST_UP, flush=True)
    except Exception as e:
        LAST_UP = "FAILED " + repr(e)[:150]; print("HF upload failed", repr(e), flush=True)


def milestone(ckpt):
    n = pathlib.Path(ckpt).stem.split("_")[1]
    out = W / f"export_{n}_{SID}"; out.mkdir(exist_ok=True)
    onnx = out / "hi_IN-custom-medium.onnx"
    exp = ("import functools,runpy,torch;torch.onnx.export=functools.partial(torch.onnx.export,dynamo=False);"
           "runpy.run_module('piper.train.export_onnx',run_name='__main__')")
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-c", exp, "--checkpoint", ckpt, "--output-file", str(onnx)], env=e)
    if r.returncode:
        print("export failed", flush=True); return
    shutil.copy(RUN / "config.json", str(onnx) + ".json")
    (out / "session.json").write_text(json.dumps(dict(session=SID, step=int(n), start_step=START_STEP, lr_env=LR_ENV, lr_at_ckpt_start=CK_LR.get("lr_g"),
                                                     t_utc=time.strftime("%FT%TZ", time.gmtime()))))
    for i, t in enumerate(SENTS, 1):
        subprocess.run([sys.executable, "-m", "piper", "-m", str(onnx), "-f", str(out / f"sample_{i}.wav")], input=t.encode(), env=e)
    api.upload_folder(folder_path=str(out), path_in_repo=f"milestones/step_{n}_{SID}", repo_id=REPO, commit_message=f"milestone {n} {SID}")
    print("milestone uploaded", n, SID, flush=True)
    shutil.rmtree(out, ignore_errors=True); os.remove(ckpt)


stop = threading.Event()
def bg():
    last = time.time()
    while not stop.wait(15):
        for c in sorted(glob.glob(str(W / "ms" / "step_*.ckpt"))):
            milestone(c)
        if LAST.exists() and time.time() - last > 20 * 60:
            upload_last(); last = time.time()
th = threading.Thread(target=bg, daemon=True); th.start()

tail = []
def hb():  # heartbeat: small progress file on HF (Kaggle shows no live logs for script kernels)
    while not stop.wait(180):
        try:
            try: pr = json.loads((W / "prog.json").read_text()); pr["age_s"] = round(time.time() - pr.pop("t"))
            except Exception as e: pr = f"no trainer step yet ({e!r})"
            try: gpu = torch.cuda.get_device_name(0)
            except Exception: gpu = "?"
            lk = LAST.stat() if LAST.exists() else None
            txt = (f"{time.strftime('%FT%TZ', time.gmtime())} session={SID} elapsed_h={(time.time()-T0)/3600:.2f} start_step={START_STEP} bs={BS} gpu={gpu} lr_env={LR_ENV or 'keep'}\n"
                   f"trainer_progress={pr}\nlast_ckpt_local={'%d B, %ds ago' % (lk.st_size, time.time()-lk.st_mtime) if lk else None} last_upload={LAST_UP}\n"
                   + "\n".join(tail[-20:]))
            api.upload_file(path_or_fileobj=txt.encode(), path_in_repo=f"runs/{NAME}/kaggle_progress_{SID}.txt", repo_id=REPO)  # per session: never clobbered
            if os.environ.get("LEGACY_HEARTBEAT", "1") == "1":  # legacy path read by the hourly check-in; the first line names the session
                api.upload_file(path_or_fileobj=txt.encode(), path_in_repo=f"runs/{NAME}/kaggle_progress.txt", repo_id=REPO)
        except Exception as e:
            print("hb fail", repr(e), flush=True)
threading.Thread(target=hb, daemon=True).start()

BSS = [BS] + [b for b in (24, 16, 12, 8) if b < BS]
for BS_TRY in BSS:
    cmd[cmd.index("--data.batch_size") + 1] = str(BS_TRY); BS = BS_TRY
    cmd[cmd.index("--ckpt_path") + 1] = str(LAST if LAST.exists() else SEED)  # after an OOM retry keep any progress already saved
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
    if rc and oom: print("CUDA OOM -> smaller batch", flush=True); continue
    break

if rc:
    try: api.upload_file(path_or_fileobj=(f"session {SID}\n" + "\n".join(tail[-60:])).encode(), path_in_repo=f"runs/{NAME}/kaggle_crash.txt", repo_id=REPO)
    except Exception: pass
print("trainer exit", rc, flush=True)
stop.set(); th.join()
for c in sorted(glob.glob(str(W / "ms" / "step_*.ckpt"))):
    milestone(c)
upload_last("final")
