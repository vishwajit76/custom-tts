"""Optional Qwen3-TTS engine (ENGINE=qwen3): zero-shot voice cloning from a reference clip.

Not real-time on CPU (measured RTF ~2 on Apple M4); needs an NVIDIA GPU. Install requirements-qwen.txt.
"""
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from app.core.config import settings
from app.models.schemas import VOICE_ID
from app.services import audio_utils
from app.services.conditioning import EngineCapabilities
from app.services.speaker_registry import secure_unlink

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
        self._lock = threading.Lock()  # guards _prompts and _epoch (save/delete run on request threads, synth on a worker)
        self._epoch = 0  # bumped by save/delete: a prompt built from the old clip must not be cached after it changed

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

    @staticmethod
    def _atomic_write(path: Path, write) -> None:
        """Write via a temp file + rename: synthesis reading the clip never sees a half-written WAV/transcript."""
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            write(tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def save_voice(self, voice_id: str, wav: np.ndarray, sr: int, transcript: str | None) -> None:
        path = self._wav_path(voice_id)
        txt = path.with_suffix(".txt")
        with self._lock:
            self._epoch += 1
            self._prompts.pop(voice_id, None)
        self._atomic_write(path, lambda t: sf.write(t, wav, sr, format="WAV"))
        if transcript:
            self._atomic_write(txt, lambda t: t.write_text(transcript, encoding="utf-8"))
        else:
            txt.unlink(missing_ok=True)
        with self._lock:  # again after the files changed: a synth that loaded the old clip meanwhile must not keep its prompt
            self._epoch += 1
            self._prompts.pop(voice_id, None)

    def delete_voice(self, voice_id: str) -> bool:
        wav = self._wav_path(voice_id)
        with self._lock:
            self._epoch += 1
            self._prompts.pop(voice_id, None)
        if not wav.exists():
            return False
        secure_unlink(wav)  # overwrite then unlink: the clip is someone's voice
        secure_unlink(wav.with_suffix(".txt"))
        with self._lock:
            self._epoch += 1
            self._prompts.pop(voice_id, None)
        return True

    def _make_prompt(self, audio, transcript: str | None):
        return self.model.create_voice_clone_prompt(ref_audio=audio, ref_text=transcript or None, x_vector_only_mode=not transcript)

    def _voice_prompt(self, voice_id: str):
        with self._lock:
            if voice_id in self._prompts:
                return self._prompts[voice_id]
            epoch = self._epoch
        wav = self._wav_path(voice_id)
        txt = wav.with_suffix(".txt")
        prompt = self._make_prompt(str(wav), txt.read_text("utf-8").strip() if txt.exists() else None)
        with self._lock:
            if epoch == self._epoch:  # the clip did not change while the prompt was being built
                self._prompts[voice_id] = prompt
        return prompt

    # ---------- synthesis ----------
    def synth(self, text: str, voice: str, speed: float, ref=None, ref_text=None) -> np.ndarray:
        # `ref` is (wav, sr) from a request or a registry reference. Qwen3-TTS builds one prompt from ONE clip
        # (create_voice_clone_prompt takes a single ref_audio per voice), so multi-reference speakers use their best clip.
        prompt = self._make_prompt(ref, ref_text) if ref is not None else self._voice_prompt(voice)
        # ponytail: 12 codec frames/sec, ~10+ chars/sec of Hindi -> cap stops runaway generation (seen on MPS)
        cap = int(len(text) * 1.5) + 36
        wavs, self.sr = self.model.generate_voice_clone(text=text, language=settings.language, voice_clone_prompt=prompt, max_new_tokens=cap)
        return audio_utils.change_speed(np.asarray(wavs[0], dtype=np.float32), speed)
