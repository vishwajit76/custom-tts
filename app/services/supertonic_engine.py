"""Supertonic 3 (Supertone, flow matching on ONNX Runtime): 10 preset voices, reads Devanagari directly (no G2P).

One TTS instance shared by all worker threads: `synth` only runs ONNX sessions (thread-safe) and pure-Python
text prep, so no lock. Code MIT; weights BigScience OpenRAIL-M (commercial use allowed with use restrictions,
e.g. output must be disclosed as machine generated; see README).
"""
import logging
import time

import numpy as np
from supertonic import TTS

from app.core.config import settings
from app.services.conditioning import EngineCapabilities

log = logging.getLogger(__name__)
PREFIX = "supertonic:"


class SupertonicEngine:
    supports_cloning = False
    # preset voice styles only; speed is native but the model accepts 0.7-2.0 (synth clamps)
    capabilities = EngineCapabilities(speed=True, speed_range=(0.7, 2.0), streaming="sentence", languages=("hi",))

    def __init__(self) -> None:
        self.tts: TTS | None = None
        self._styles: dict = {}  # voice_id -> Style
        self.max_workers = settings.workers

    @property
    def ready(self) -> bool:
        return self.tts is not None

    def load(self) -> None:
        t = time.perf_counter()
        tts = TTS(model="supertonic-3", model_dir=settings.supertonic_dir, auto_download=False,
                  intra_op_num_threads=settings.threads_per_worker, inter_op_num_threads=1)
        self._styles = {PREFIX + n: tts.get_voice_style(n) for n in tts.voice_style_names}  # M1..M5, F1..F5
        self.tts = tts
        self.synth("नमस्ते।", next(iter(self._styles)), 1.0)  # warm-up: first run allocates
        log.info("supertonic voices loaded", extra={"extra_fields": {"voices": list(self._styles), "secs": round(time.perf_counter() - t, 2)}})

    def voices(self) -> list[dict]:
        sr = self.tts.sample_rate
        return [{"voice_id": v, "sample_rate": sr, "language": "hi", "engine": "supertonic", "gender": v[len(PREFIX)],
                 "name": f"Supertonic {v[len(PREFIX):]}"} for v in self._styles]

    def has_voice(self, voice: str) -> bool:
        return voice in self._styles

    def sample_rate(self, voice: str) -> int:
        return self.tts.sample_rate

    def synth(self, text: str, voice: str, speed: float, ref=None, ref_text=None) -> np.ndarray:
        # the model accepts speed 0.7-2.0; our API allows 0.5
        wav, _ = self.tts.synthesize(text, self._styles[voice], total_steps=settings.supertonic_steps,
                                     speed=min(max(speed, 0.7), 2.0), lang="hi")
        return wav[0].astype(np.float32, copy=False)
