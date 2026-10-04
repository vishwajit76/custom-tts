"""Piper (VITS) voices on ONNX Runtime. One shared InferenceSession per voice model; `synth` is thread-safe.

Voice ids come from the catalog (voices/catalog.json, app/services/voice_catalog.py): a voice is (model, speaker id), and a
multi-speaker model can serve several voices and, for each, the recorded styles listed in its entry. Models with no
catalog entry behave as before: a single-speaker model is the voice `<file stem>`, a multi-speaker one `<stem>:<speaker>`.
"""
import json
import logging
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import onnxruntime as ort
from piper import PiperVoice
from piper.config import PiperConfig, SynthesisConfig

from app.core.config import settings
from app.services import indian_english, voice_catalog
from app.services.conditioning import Emotion, EngineCapabilities

log = logging.getLogger(__name__)
_EMOTIONS = frozenset(e.value for e in Emotion)


def make_session(model: Path, threads: int, use_cuda: bool) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})] if use_cuda else []
    return ort.InferenceSession(str(model), so, providers=providers + ["CPUExecutionProvider"])


def _speaker(voice_id: str) -> str:
    """hi_IN-rohan-medium -> rohan; model:speaker -> speaker; anything else as is. Display name only, never gender."""
    name = voice_id.split(":")[-1]
    parts = name.split("-")
    return parts[1] if ":" not in voice_id and len(parts) == 3 else name


def model_files(dirs: list[Path]) -> list[Path]:
    """*.onnx in each directory, in order; the first occurrence of a file stem wins (an extra directory cannot shadow a voice)."""
    seen: dict[str, Path] = {}
    for d in dirs:
        for model in sorted(Path(d).glob("*.onnx")):
            if model.stem in seen:
                log.warning("duplicate voice ignored", extra={"extra_fields": {"voice": model.stem, "kept": str(seen[model.stem]), "ignored": str(model)}})
            else:
                seen[model.stem] = model
    return list(seen.values())


@dataclass(frozen=True)
class _Voice:
    voice: PiperVoice
    sid: int | None  # speaker id when no style is asked for (None: single-speaker model, or the model's own default speaker)
    styles: dict[str, int] = field(default_factory=dict)  # catalog style/emotion name -> speaker id (same voice, other recorded style)
    caps: EngineCapabilities | None = None  # None: the engine-wide ones


class PiperEngine:
    supports_cloning = False
    # speed = VITS length_scale; no emotion/style/pitch/energy conditioning exists in the model or this code, except
    # voices whose catalog entry lists recorded styles (capabilities_for): those are separate speakers of a multi-speaker model
    capabilities = EngineCapabilities(speed=True, streaming="sentence", languages=("hi", "hinglish"))

    def __init__(self) -> None:
        self._voices: dict[str, _Voice] = {}
        self.max_workers = settings.workers

    @property
    def ready(self) -> bool:
        return bool(self._voices)

    def load(self) -> None:
        t = time.perf_counter()
        dirs = [settings.models_dir, *(Path(d.strip()) for d in settings.models_extra.split(",") if d.strip())]
        entries = voice_catalog.get().active("piper")
        stems = set()
        for model in model_files(dirs):
            cfg_path = Path(f"{model}.json")
            if not cfg_path.exists():
                log.warning("skipping voice without config", extra={"extra_fields": {"model": str(model)}})
                continue
            stems.add(model.stem)
            config = PiperConfig.from_dict(json.loads(cfg_path.read_text("utf-8")))
            voice = PiperVoice(session=make_session(model, settings.threads_per_worker, settings.use_cuda), config=config)
            multi = config.num_speakers > 1 and bool(config.speaker_id_map)
            mine = [e for e in entries if e.model == model.stem]
            added: dict[str, _Voice] = {}
            if not multi:  # the plain file-stem voice stays routable whatever the catalog says
                added[model.stem] = _Voice(voice, None)
            elif not mine:  # a multi-speaker model nobody catalogued: one voice per speaker, as before
                added = {f"{model.stem}:{name}": _Voice(voice, sid) for name, sid in config.speaker_id_map.items()}
            for e in mine:  # a catalogued multi-speaker model exposes only its catalog voices
                if (v := self._bind(e, voice, config, multi)) is not None:
                    added[e.voice_id] = v
            self._voices |= added
            if added:
                self.synth("नमस्ते।", next(reversed(added)), 1.0)  # warm-up: first run allocates
        if absent := [e.voice_id for e in entries if e.model not in stems]:
            log.warning("catalog voices skipped: model file absent", extra={"extra_fields": {"voices": absent}})
        if not self._voices:
            raise RuntimeError(f"no Piper voices (*.onnx + *.onnx.json) in {dirs}; run scripts/download_voices.py")
        log.info("piper voices loaded", extra={"extra_fields": {"voices": list(self._voices), "secs": round(time.perf_counter() - t, 2)}})

    def _bind(self, e: voice_catalog.Voice, voice: PiperVoice, config: PiperConfig, multi: bool) -> _Voice | None:
        """Catalog entry -> loaded model: speaker keys to speaker ids. None, with a warning, if they do not fit the model."""
        keys = {k for k in (e.default_speaker, *e.styles.values()) if k}
        if not multi:
            if keys:
                log.warning("catalog voice skipped: model is single-speaker", extra={"extra_fields": {"voice": e.voice_id, "model": e.model}})
                return None
            return _Voice(voice, None)
        ids = config.speaker_id_map
        if missing := sorted(k for k in keys if k not in ids):
            log.warning("catalog voice skipped: speaker not in the model's speaker_id_map",
                        extra={"extra_fields": {"voice": e.voice_id, "model": e.model, "missing": missing}})
            return None
        styles = {n: ids[k] for n, k in e.styles.items()}
        caps = None
        if styles:  # native only for the styles this voice was recorded in, labelled as plain names (not ":dsp"/":steered")
            emo, sty = tuple(n for n in styles if n in _EMOTIONS), tuple(n for n in styles if n not in _EMOTIONS)
            caps = replace(self.capabilities, native_emotion=bool(emo), emotion_values=emo, native_style=bool(sty), style_values=sty, discrete_styles=True)
        return _Voice(voice, ids[e.default_speaker] if e.default_speaker else None, styles, caps)

    def voices(self) -> list[dict]:
        cat = voice_catalog.get()
        out = []
        for vid, v in self._voices.items():
            e = cat.get(vid)
            out.append({"voice_id": vid, "sample_rate": v.voice.config.sample_rate, "language": v.voice.config.espeak_voice, "speaker_id": v.sid,
                        "engine": "piper", "gender": e.gender if e else None,
                        "name": (e.name or f"Piper {vid}") if e else f"Piper {_speaker(vid).title()}"})
        return out

    def has_voice(self, voice: str) -> bool:
        return voice in self._voices

    def sample_rate(self, voice: str) -> int:
        return self._voices[voice].voice.config.sample_rate

    def capabilities_for(self, voice: str) -> EngineCapabilities:
        return self._voices[voice].caps or self.capabilities

    def synth(self, text: str, voice: str, speed: float, ref=None, ref_text=None) -> np.ndarray:
        v = self._voices[voice]
        return self._run(v.voice, text, speed, v.sid)

    def synth_native(self, text: str, voice: str, speed: float, controls: dict, ref=None, ref_text=None) -> np.ndarray:
        """Style voices: `controls` carries the (already validated) emotion/style name, which picks that style's speaker."""
        v = self._voices[voice]
        sid = next((v.styles[c] for k in ("emotion", "style") if (c := controls.get(k)) in v.styles), v.sid)
        return self._run(v.voice, text, speed, sid)

    @staticmethod
    def _run(v: PiperVoice, text: str, speed: float, sid: int | None) -> np.ndarray:
        if settings.indian_english and v.config.espeak_voice == "hi":
            text = indian_english.mark(text)
        out = []
        for phonemes in v.phonemize(text):  # espeak splits sentences; normally one per chunk
            if phonemes:
                audio = v.phoneme_ids_to_audio(v.phonemes_to_ids(phonemes), SynthesisConfig(
                    length_scale=1.0 / speed, speaker_id=sid, noise_scale=settings.noise_scale, noise_w_scale=settings.noise_w))
                out.append(np.asarray(audio, dtype=np.float32))
        return np.concatenate(out) if out else np.zeros(0, np.float32)
