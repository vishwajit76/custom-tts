"""Optional Qwen3-TTS engine (ENGINE=qwen3): zero-shot voice cloning from a reference clip.

Not real-time on CPU (measured RTF ~2 on Apple M4); needs an NVIDIA GPU. Install requirements-qwen.txt.
"""
import logging
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from app.core.config import settings
from app.models.schemas import VOICE_ID
from app.services import audio_utils
from app.services.conditioning import EngineCapabilities

log = logging.getLogger(__name__)


class QwenEngine:
    supports_cloning = True
    # cloning via reference audio (x-vector only without a transcript); speed is librosa time-stretch (DSP), not native.
    # No emotion/style instruction is passed to generate_voice_clone, so none is claimed.
    capabilities = EngineCapabilities(cloning=True, speed=True, streaming="sentence", sample_rates=(24000,), languages=("auto",))
    max_workers = 1  # one model instance; generation is not thread-safe

    def __init__(self) -> None:
        self.model = None
        self.sr = 24000
        self._prompts: dict[str, object] = {}  # voice_id -> cached voice_clone_prompt

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        device = settings.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        kwargs = {"device_map": device, "dtype": torch.float32 if device == "cpu" else torch.bfloat16}
        try:
            import flash_attn  # noqa: F401

            kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass
        t = time.perf_counter()
        self.model = Qwen3TTSModel.from_pretrained(settings.resolve_model_dir(), **kwargs)
        log.info("qwen3 loaded", extra={"extra_fields": {"device": device, "secs": round(time.perf_counter() - t, 1)}})
        settings.voices_dir.mkdir(parents=True, exist_ok=True)

    # ---------- voices ----------
    def voices(self) -> list[dict]:
        return [
            {"voice_id": p.stem, "sample_rate": self.sr, "has_transcript": p.with_suffix(".txt").exists()}
            for p in sorted(settings.voices_dir.glob("*.wav"))
        ]

    def has_voice(self, voice: str) -> bool:
        return bool(re.fullmatch(VOICE_ID, voice)) and self._wav_path(voice).exists()

    def sample_rate(self, voice: str) -> int:
        return self.sr

    @staticmethod
    def _wav_path(voice_id: str) -> Path:
        if not re.fullmatch(VOICE_ID, voice_id):
            raise ValueError(f"invalid voice_id {voice_id!r}")
        return settings.voices_dir / f"{voice_id}.wav"

    def save_voice(self, voice_id: str, wav: np.ndarray, sr: int, transcript: str | None) -> None:
        path = self._wav_path(voice_id)
        sf.write(path, wav, sr)
        txt = path.with_suffix(".txt")
        if transcript:
            txt.write_text(transcript, encoding="utf-8")
        else:
            txt.unlink(missing_ok=True)
        self._prompts.pop(voice_id, None)

    def delete_voice(self, voice_id: str) -> bool:
        wav = self._wav_path(voice_id)
        if not wav.exists():
            return False
        wav.unlink()
        wav.with_suffix(".txt").unlink(missing_ok=True)
        self._prompts.pop(voice_id, None)
        return True

    def _make_prompt(self, audio, transcript: str | None):
        return self.model.create_voice_clone_prompt(ref_audio=audio, ref_text=transcript or None, x_vector_only_mode=not transcript)

    def _voice_prompt(self, voice_id: str):
        if voice_id not in self._prompts:
            wav = self._wav_path(voice_id)
            txt = wav.with_suffix(".txt")
            self._prompts[voice_id] = self._make_prompt(str(wav), txt.read_text("utf-8").strip() if txt.exists() else None)
        return self._prompts[voice_id]

    # ---------- synthesis ----------
    def synth(self, text: str, voice: str, speed: float, ref=None, ref_text=None) -> np.ndarray:
        prompt = self._make_prompt(ref, ref_text) if ref is not None else self._voice_prompt(voice)
        # ponytail: 12 codec frames/sec, ~10+ chars/sec of Hindi -> cap stops runaway generation (seen on MPS)
        cap = int(len(text) * 1.5) + 36
        wavs, self.sr = self.model.generate_voice_clone(text=text, language=settings.language, voice_clone_prompt=prompt, max_new_tokens=cap)
        return audio_utils.change_speed(np.asarray(wavs[0], dtype=np.float32), speed)
