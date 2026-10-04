"""Kokoro-format PyTorch checkpoints (e.g. BH-Builds/goonj-1-82M) as a server engine, for bench.v7_eval only.

Same G2P as app.services.kokoro_engine (misaki EspeakG2P('hi') recipe on our espeak), same EngineCapabilities, so the harness
path (tts.stream: normalize, chunking, trim, PCM16) is identical; only the acoustic model differs. A model dir holds one *.pth,
config.json and voices/*.pt ([510, 1, 256] packs). Needs the `kokoro` package (installed --no-deps: its pipeline imports misaki,
which we don't use, so the package __init__ is bypassed).
"""
from optimizer import g2p
import pathlib
import sys
import types
from importlib.util import find_spec

import numpy as np
import torch

from app.services import kokoro_engine


def _kmodel():
    if "kokoro" not in sys.modules:  # ponytail: skip kokoro/__init__ (pulls misaki); breaks if kokoro.model starts importing the pipeline
        m = types.ModuleType("kokoro")
        m.__path__ = list(find_spec("kokoro").submodule_search_locations)
        sys.modules["kokoro"] = m
    from kokoro.model import KModel

    return KModel


class KokoroPTEngine(kokoro_engine.KokoroEngine):
    def __init__(self, model_dir: str) -> None:
        super().__init__()
        self.dir = pathlib.Path(model_dir)
        self.prefix = f"{self.dir.name}:"

    def load(self) -> None:
        torch.set_num_threads(max(1, kokoro_engine.settings.threads_per_worker))
        pth = next(self.dir.glob("*.pth"))
        self.model = _kmodel()(repo_id="hexgrad/Kokoro-82M", config=str(self.dir / "config.json"), model=str(pth)).eval()
        self._styles = {self.prefix + p.stem: torch.load(p, map_location="cpu", weights_only=True) for p in sorted((self.dir / "voices").glob("*.pt"))}
        self.kokoro = self.model  # `ready`
        self.synth("नमस्ते।", next(iter(self._styles)), 1.0)

    def voices(self) -> list[dict]:
        return [{"voice_id": v, "sample_rate": kokoro_engine.SAMPLE_RATE, "language": "hi", "engine": "kokoro-pt",
                 "gender": "male" if v.split("_")[-1] in {"atul", "ravi", "aman", "arjun", "dev", "kabir", "sameer"} else "female", "name": f"goonj {v.split(':')[1]}"} for v in self._styles]

    def synth(self, text: str, voice: str, speed: float, ref=None, ref_text=None) -> np.ndarray:
        ps = g2p.fix(text, kokoro_engine.phonemes(text))  # measured goonj G2P fixes (optimizer/g2p_fixes.json)
        n = sum(c in self.model.vocab for c in ps)
        if not n:
            return np.zeros(0, np.float32)
        pack = self._styles[voice]
        with torch.no_grad():
            wav = self.model(ps, pack[min(n - 1, pack.shape[0] - 1)], speed=speed)
        return wav.numpy().astype(np.float32)
