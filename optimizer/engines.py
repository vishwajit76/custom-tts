"""Existing app engines behind one call: synth(chunks, speed, pause_ms) -> (float32 wav, sr). No new models."""
import json
import threading

import numpy as np

from optimizer.bench import ROOT


class Goonj:
    """bench.kokoro_pt.KokoroPTEngine (the app's `goonj` engine) with weights moved to CUDA."""
    version = "goonj:kokoro_hindi_final"
    sr = 24000

    def __init__(self, voice="hi_meera", device="cuda"):
        self.voice, self.device = voice, device

    def load(self):
        import torch
        from loguru import logger
        logger.disable("kokoro")  # per-call DEBUG dumps
        from bench.kokoro_pt import _kmodel  # same loader as the app's goonj engine (KokoroPTEngine)
        if self.device == "cuda":
            torch.cuda.set_per_process_memory_fraction(0.9)  # WDDM spills silently past VRAM
        d = ROOT / "exp/goonj"
        self.model = _kmodel()(repo_id="hexgrad/Kokoro-82M", config=str(d / "config.json"), model=str(next(d.glob("*.pth")))).eval().to(self.device)
        self.pack = torch.load(d / "voices" / f"{self.voice}.pt", map_location=self.device, weights_only=True)

    def chunk(self, text, speed):
        import torch
        from app.services import kokoro_engine
        ps = kokoro_engine.phonemes(text)
        n = sum(c in self.model.vocab for c in ps)
        if not n:
            return np.zeros(0, np.float32)
        torch.manual_seed(0)  # deterministic decoder noise
        with torch.inference_mode():
            return self.model(ps, self.pack[min(n - 1, self.pack.shape[0] - 1)], speed=speed).float().cpu().numpy()


class Piper:
    """app PiperEngine._run on a given ONNX (CPU keeps the GPU for Whisper/Kokoro)."""
    sr = 22050

    def __init__(self, model="exp/v7/v7a.onnx"):
        self.model = model
        self.version = f"piper:{model}"

    def load(self):
        from piper import PiperVoice
        from piper.config import PiperConfig
        from app.services.piper_engine import make_session
        m = ROOT / self.model
        cfg = PiperConfig.from_dict(json.loads(m.with_name(m.name + ".json").read_text("utf-8")))
        self.v = PiperVoice(session=make_session(m, 2, False), config=cfg)
        self.sid = 0 if cfg.num_speakers > 1 else None
        self.sr = cfg.sample_rate

    def chunk(self, text, speed):
        from app.services.piper_engine import PiperEngine
        return PiperEngine._run(self.v, text, speed, self.sid)


class Engine:
    """Loaded lazily; one inference at a time per engine (ponytail: per-engine lock; a pool if inference_workers > 1 matters)."""

    def __init__(self, impl):
        self.impl, self.lock, self.loaded = impl, threading.Lock(), False

    @property
    def version(self):
        return self.impl.version

    def synth(self, chunks, speed=1.0, pause_ms=120):
        with self.lock:
            if not self.loaded:
                self.impl.load()
                self.loaded = True
            gap = np.zeros(int(self.impl.sr * pause_ms / 1000), np.float32)
            parts = []
            for c in chunks:
                parts += [self.impl.chunk(c, speed), gap]
            return (np.concatenate(parts[:-1]) if parts else np.zeros(0, np.float32)), self.impl.sr


def make(name):
    return Engine({"goonj": lambda: Goonj(), "piper_v7a": lambda: Piper("exp/v7/v7a.onnx"),
                   "piper_base": lambda: Piper("voices/hi_IN-custom-medium.onnx")}[name]())
