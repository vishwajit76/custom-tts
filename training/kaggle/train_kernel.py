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
        HfApi().upload_file(path_or_fileobj=txt.encode(), path_in_repo=f"runs/{NAME}/kaggle_crash.txt", repo_id=REPO)
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
LAST = CK / "last.ckpt"
ck0 = torch.load(LAST, map_location="cpu", weights_only=False)
START_STEP, START_EPOCH = int(ck0["global_step"]), int(ck0.get("epoch", -1)) + 1
del ck0
print(f"resuming global_step={START_STEP} epoch={START_EPOCH}", flush=True)

# ---- trainer wrapper: ckpt every 500 steps, milestone copies, deadline stop ----
(W / "ms").mkdir(exist_ok=True)
(W / "wrap.py").write_text(f'''
import os, time, torch
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from piper.train import __main__ as m
DEADLINE = float(os.environ["DEADLINE"]); MS = {MS_EVERY}
class Ctl(Callback):
    def __init__(s): s.last_ms = None; s.t = time.time()
    def on_train_start(s, tr, pl): s.last_ms = tr.global_step // MS
    def on_train_batch_end(s, tr, pl, *a, **k):
        g = tr.global_step
        if g // MS > s.last_ms:
            s.last_ms = g // MS
            tr.save_checkpoint("/tmp/w/ms/tmp.ckpt"); os.replace("/tmp/w/ms/tmp.ckpt", f"/tmp/w/ms/step_{{g}}.ckpt")
        if time.time() > DEADLINE and not tr.should_stop:
            print("DEADLINE reached: saving and stopping", flush=True)
            tr.save_checkpoint("{LAST}"); tr.should_stop = True
m._DEFAULT_CALLBACKS[:] = [ModelCheckpoint(save_last=True, save_top_k=0, every_n_train_steps=500), Ctl()]
if __name__ == "__main__": m.main()
''')

cmd = [sys.executable, "/tmp/w/wrap.py", "fit", "--seed_everything", "1234",
       "--data.voice_name", NAME, "--data.csv_path", str(DATA / "metadata.csv"), "--data.audio_dir", str(DATA / "wavs"),
       "--data.espeak_voice", "hi", "--data.cache_dir", str(RUN / "cache"), "--data.config_path", str(RUN / "config.json"),
       "--data.batch_size", str(BS), "--data.num_workers", "3", "--model.sample_rate", "22050", "--model.num_speakers", "1",
       "--trainer.default_root_dir", str(RUN), "--trainer.accelerator", "gpu", "--trainer.devices", "1",
       "--trainer.precision", "16-mixed", "--trainer.max_epochs", str(START_EPOCH + 100000),
       "--trainer.log_every_n_steps", "50", "--ckpt_path", str(LAST)]
env = dict(os.environ, DEADLINE=str(T0 + MAX_H * 3600), PYTHONUNBUFFERED="1")


def upload_last(tag="ckpt"):
    try:
        tmp = W / "upload_last.ckpt"; shutil.copy(LAST, tmp)
        api.upload_file(path_or_fileobj=str(tmp), path_in_repo=f"runs/{NAME}/last.ckpt", repo_id=REPO, commit_message=f"{tag} kaggle")
        api.upload_file(path_or_fileobj=str(RUN / "config.json"), path_in_repo=f"runs/{NAME}/config.json", repo_id=REPO)
        print("HF upload ok", tag, time.strftime("%T"), flush=True)
    except Exception as e:
        print("HF upload failed", repr(e), flush=True)


def milestone(ckpt):
    n = pathlib.Path(ckpt).stem.split("_")[1]
    out = W / f"export_{n}"; out.mkdir(exist_ok=True)
    onnx = out / "hi_IN-custom-medium.onnx"
    exp = ("import functools,runpy,torch;torch.onnx.export=functools.partial(torch.onnx.export,dynamo=False);"
           "runpy.run_module('piper.train.export_onnx',run_name='__main__')")
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-c", exp, "--checkpoint", ckpt, "--output-file", str(onnx)], env=e)
    if r.returncode:
        print("export failed", flush=True); return
    shutil.copy(RUN / "config.json", str(onnx) + ".json")
    for i, t in enumerate(SENTS, 1):
        subprocess.run([sys.executable, "-m", "piper", "-m", str(onnx), "-f", str(out / f"sample_{i}.wav")], input=t.encode(), env=e)
    api.upload_folder(folder_path=str(out), path_in_repo=f"milestones/step_{n}", repo_id=REPO, commit_message=f"milestone {n}")
    print("milestone uploaded", n, flush=True)
    shutil.rmtree(out, ignore_errors=True); os.remove(ckpt)


stop = threading.Event()
def bg():
    last = time.time()
    while not stop.wait(15):
        for c in sorted(glob.glob(str(W / "ms" / "step_*.ckpt"))):
            milestone(c)
        if time.time() - last > 20 * 60:
            upload_last(); last = time.time()
th = threading.Thread(target=bg, daemon=True); th.start()

tail = []
def hb():  # heartbeat: small progress file on HF (Kaggle shows no live logs for script kernels)
    while not stop.wait(180):
        try:
            txt = f"{time.strftime('%FT%TZ', time.gmtime())} elapsed_h={(time.time()-T0)/3600:.2f} start_step={START_STEP} bs={BS}\n" + "\n".join(tail[-12:])
            api.upload_file(path_or_fileobj=txt.encode(), path_in_repo=f"runs/{NAME}/kaggle_progress.txt", repo_id=REPO)
        except Exception as e:
            print("hb fail", repr(e), flush=True)
threading.Thread(target=hb, daemon=True).start()

BSS = [BS] + [b for b in (24, 16, 12, 8) if b < BS]
for BS_TRY in BSS:
    cmd[cmd.index("--data.batch_size") + 1] = str(BS_TRY); BS = BS_TRY
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
    try: api.upload_file(path_or_fileobj="\n".join(tail[-60:]).encode(), path_in_repo=f"runs/{NAME}/kaggle_crash.txt", repo_id=REPO)
    except Exception: pass
print("trainer exit", rc, flush=True)
stop.set(); th.join()
for c in sorted(glob.glob(str(W / "ms" / "step_*.ckpt"))):
    milestone(c)
upload_last("final")
