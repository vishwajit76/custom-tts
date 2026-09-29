"""Piper (VITS) voices on ONNX Runtime. One shared InferenceSession per voice model; `synth` is thread-safe."""
import json
import logging
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from piper import PiperVoice
from piper.config import PiperConfig, SynthesisConfig

from app.core.config import settings
from app.services import indian_english
from app.services.conditioning import EngineCapabilities

log = logging.getLogger(__name__)
# speaker in hi_IN-<speaker>-medium -> gender. rohan: MODEL_CARD dataset "Hindi Mono Male"; the pratham and
# priyamvada cards don't say, so by name and median pitch (~98 Hz, ~214 Hz). Other (e.g. fine-tuned) voices: None.
GENDER = {"rohan": "M", "pratham": "M", "priyamvada": "F"}


def make_session(model: Path, threads: int, use_cuda: bool) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})] if use_cuda else []
    return ort.InferenceSession(str(model), so, providers=providers + ["CPUExecutionProvider"])


def _speaker(voice_id: str) -> str:
    """hi_IN-rohan-medium -> rohan; model:speaker -> speaker; anything else as is."""
    name = voice_id.split(":")[-1]
    parts = name.split("-")
    return parts[1] if ":" not in voice_id and len(parts) == 3 else name


class PiperEngine:
    supports_cloning = False
    # speed = VITS length_scale; no emotion/style/pitch/energy conditioning exists in the model or this code
    capabilities = EngineCapabilities(speed=True, streaming="sentence", languages=("hi", "hinglish"))

    def __init__(self) -> None:
        self._voices: dict[str, tuple[PiperVoice, int | None]] = {}  # voice_id -> (voice, speaker_id)
        self.max_workers = settings.workers

    @property
    def ready(self) -> bool:
        return bool(self._voices)

    def load(self) -> None:
        t = time.perf_counter()
        for model in sorted(settings.models_dir.glob("*.onnx")):
            cfg_path = Path(f"{model}.json")
            if not cfg_path.exists():
                log.warning("skipping voice without config", extra={"extra_fields": {"model": str(model)}})
                continue
            config = PiperConfig.from_dict(json.loads(cfg_path.read_text("utf-8")))
            voice = PiperVoice(session=make_session(model, settings.threads_per_worker, settings.use_cuda), config=config)
            if config.num_speakers > 1 and config.speaker_id_map:
                for name, sid in config.speaker_id_map.items():
                    self._voices[f"{model.stem}:{name}"] = (voice, sid)
            else:
                self._voices[model.stem] = (voice, None)
            self.synth("नमस्ते।", next(reversed(self._voices)), 1.0)  # warm-up: first run allocates
        if not self._voices:
            raise RuntimeError(f"no Piper voices (*.onnx + *.onnx.json) in {settings.models_dir}; run scripts/download_voices.py")
        log.info("piper voices loaded", extra={"extra_fields": {"voices": list(self._voices), "secs": round(time.perf_counter() - t, 2)}})

    def voices(self) -> list[dict]:
        return [
            {"voice_id": vid, "sample_rate": v.config.sample_rate, "language": v.config.espeak_voice, "speaker_id": sid,
             "engine": "piper", "gender": GENDER.get(_speaker(vid)), "name": f"Piper {_speaker(vid).title()}"}
            for vid, (v, sid) in self._voices.items()
        ]

    def has_voice(self, voice: str) -> bool:
        return voice in self._voices

    def sample_rate(self, voice: str) -> int:
        return self._voices[voice][0].config.sample_rate

    def synth(self, text: str, voice: str, speed: float, ref=None, ref_text=None) -> np.ndarray:
        v, sid = self._voices[voice]
        if settings.indian_english and v.config.espeak_voice == "hi":
            text = indian_english.mark(text)
        out = []
        for phonemes in v.phonemize(text):  # espeak splits sentences; normally one per chunk
            if phonemes:
                audio = v.phoneme_ids_to_audio(v.phonemes_to_ids(phonemes), SynthesisConfig(
                    length_scale=1.0 / speed, speaker_id=sid, noise_scale=settings.noise_scale, noise_w_scale=settings.noise_w))
                out.append(np.asarray(audio, dtype=np.float32))
        return np.concatenate(out) if out else np.zeros(0, np.float32)

