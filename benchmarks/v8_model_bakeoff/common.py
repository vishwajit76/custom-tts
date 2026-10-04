"""Shared harness: engine wrappers (synth(text)->(float32 wav, sr)), text pipeline, resource sampler. Run from repo root."""
import os, sys, time, threading, json, csv, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
HERE = Path(__file__).resolve().parent
import numpy as np
import torch

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)  # WDDM silently spills past 6 GB into system RAM
import onnxruntime as ort
import psutil
import pynvml

SPILL_MIB = 5.8 * 1024
pynvml.nvmlInit()
_h = pynvml.nvmlDeviceGetHandleByIndex(0)


def corpus(n=50):
    rows = list(csv.DictReader(open(ROOT / "bench/corpus/hi_eval_v2.tsv", encoding="utf-8"), delimiter="\t"))
    return rows[:n]


def chunks_of(text):
    from app.services import text_normalizer
    from app.services.tts import split_for_stream
    return split_for_stream(text_normalizer.normalize(text))


class Sampler:
    """Polls NVML (== nvidia-smi) used VRAM + GPU util, process CPU% and RSS every 100 ms."""
    def __init__(self):
        self.p = psutil.Process(); self.stop = threading.Event(); self.vram = []; self.util = []; self.cpu = []; self.rss = []

    def __enter__(self):
        self.p.cpu_percent(None)

        def run():
            while not self.stop.is_set():
                self.vram.append(pynvml.nvmlDeviceGetMemoryInfo(_h).used / 2**20)
                self.util.append(pynvml.nvmlDeviceGetUtilizationRates(_h).gpu)
                self.cpu.append(self.p.cpu_percent(None) / psutil.cpu_count())  # % of whole machine
                self.rss.append(self.p.memory_info().rss / 2**20)
                time.sleep(0.1)
        self.t = threading.Thread(target=run, daemon=True); self.t.start(); return self

    def __exit__(self, *a):
        self.stop.set(); self.t.join()

    def summary(self):
        f = lambda x, fn: round(fn(x), 1) if x else None
        torch_peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
        peak = max(self.vram or [0])
        return {"nvml_vram_peak_mib": f(self.vram, max), "nvml_vram_mean_mib": f(self.vram, st.mean),
                "torch_max_alloc_mib": round(torch_peak, 1), "gpu_util_mean": f(self.util, st.mean), "gpu_util_max": f(self.util, max),
                "cpu_pct_machine_mean": f(self.cpu, st.mean), "rss_peak_mib": f(self.rss, max),
                "spilled": bool(peak > SPILL_MIB or torch_peak > SPILL_MIB)}


def vram_now():
    return pynvml.nvmlDeviceGetMemoryInfo(_h).used / 2**20


# ---------------- engines ----------------
class PiperEng:
    def __init__(self, model, cuda):
        self.model, self.cuda = model, cuda
        self.slots = 1 if cuda else 2

    def load(self):
        from piper import PiperVoice
        from piper.config import PiperConfig
        from app.services.piper_engine import make_session
        if self.cuda:
            ort.preload_dlls()
        m = ROOT / self.model
        cfg = PiperConfig.from_dict(json.loads(Path(f"{m}.json").read_text("utf-8")))
        sess = make_session(m, 1 if self.cuda else 2, self.cuda)
        self.active = sess.get_providers()[0]
        self.v = PiperVoice(session=sess, config=cfg)
        self.sid = 0 if cfg.num_speakers > 1 else None  # multi-speaker V7: speaker id 0
        self.sr = cfg.sample_rate

    def synth(self, text):
        from app.services.piper_engine import PiperEngine
        return PiperEngine._run(self.v, text, 1.0, self.sid), self.sr


class GoonjEng:
    slots = 1

    def __init__(self, voice="hi_meera"):
        self.voice = voice

    def load(self):
        from bench.kokoro_pt import _kmodel
        d = ROOT / "exp/goonj"
        self.m = _kmodel()(repo_id="hexgrad/Kokoro-82M", config=str(d / "config.json"), model=str(next(d.glob("*.pth")))).eval().to("cuda")
        self.pack = torch.load(d / "voices" / f"{self.voice}.pt", map_location="cuda", weights_only=True)

    def synth(self, text):
        from app.services import kokoro_engine
        ps = kokoro_engine.phonemes(text)
        n = sum(c in self.m.vocab for c in ps)
        if not n:
            return np.zeros(0, np.float32), 24000
        with torch.inference_mode():
            w = self.m(ps, self.pack[min(n - 1, self.pack.shape[0] - 1)], speed=1.0)
        return w.float().cpu().numpy(), 24000


class F5Eng:
    slots = 1

    def __init__(self, nfe=32):
        self.nfe = nfe

    def load(self):
        from huggingface_hub import hf_hub_download
        from f5_tts.api import F5TTS
        repo = "SPRINGLab/F5-Hindi-24KHz"
        ck = hf_hub_download(repo, "model_2500000.safetensors"); vocab = hf_hub_download(repo, "vocab.txt")
        self.ref = str(HERE / "ref/ref.wav"); self.ref_text = (HERE / "ref/ref.txt").read_text("utf-8").strip()
        if not Path(self.ref).exists():
            import shutil
            shutil.copy(hf_hub_download(repo, "samples/output1.wav"), self.ref)
        self.tts = F5TTS(model="F5TTS_Small", ckpt_file=ck, vocab_file=vocab, device="cuda")

    def synth(self, text):
        w, sr, _ = self.tts.infer(self.ref, self.ref_text, text, nfe_step=self.nfe, show_info=lambda *a: None)
        return np.asarray(w, np.float32), sr


class IndicF5Eng(F5Eng):
    """ai4bharat/IndicF5 (F5TTS_Base arch, 336M). Weights from model.safetensors with the torch.compile '_orig_mod.' prefix
    stripped into exp/indicf5/ (vocoder keys dropped; f5_tts loads vocos itself). Same Hindi ref clip as the F5 runs."""
    def load(self):
        from f5_tts.api import F5TTS
        d = ROOT / "exp/indicf5"
        self.ref = str(HERE / "ref/ref.wav"); self.ref_text = (HERE / "ref/ref.txt").read_text("utf-8").strip()
        self.tts = F5TTS(model="F5TTS_Base", ckpt_file=str(d / "model.safetensors"), vocab_file=str(d / "vocab.txt"), device="cuda")
        self.tts.ema_model.float()  # f5_tts loads fp16 on sm>=7; IndicF5 in fp16 emits noise (Whisper: "झाल"), fp32 is intelligible


class MmsEng:
    slots = 1

    def load(self):
        from transformers import VitsModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained("facebook/mms-tts-hin")
        self.m = VitsModel.from_pretrained("facebook/mms-tts-hin").to("cuda").eval()

    def synth(self, text):
        x = self.tok(text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            w = self.m(**x).waveform[0]
        return w.float().cpu().numpy(), self.m.config.sampling_rate


ENGINES = {
    "piper_base_cpu": lambda: PiperEng("voices/hi_IN-custom-medium.onnx", False),
    "piper_base_cuda": lambda: PiperEng("voices/hi_IN-custom-medium.onnx", True),
    "piper_v7a_cpu": lambda: PiperEng("exp/v7/v7a.onnx", False),
    "piper_v7a_cuda": lambda: PiperEng("exp/v7/v7a.onnx", True),
    "piper_v7b_cpu": lambda: PiperEng("exp/v7/v7b.onnx", False),
    "piper_v7b_cuda": lambda: PiperEng("exp/v7/v7b.onnx", True),
    "kokoro_goonj": lambda: GoonjEng(),
    "f5_hindi_springlab": lambda: F5Eng(),
    "f5_hindi_nfe16": lambda: F5Eng(16),
    "indicf5": lambda: IndicF5Eng(),
    "indicf5_nfe16": lambda: IndicF5Eng(16),
    "mms_hin": lambda: MmsEng(),
}


def pct(x, p):
    return round(float(np.percentile(x, p)), 4) if len(x) else None
