"""Kokoro-82M v1.0 Hindi voices (hf_alpha, hf_beta, hm_omega, hm_psi) via kokoro-onnx, fp32.

G2P is Kokoro's own Hindi recipe (misaki EspeakG2P('hi'): espeak-ng IPA with tied diphthongs/affricates mapped to
single symbols), but run on the process's one espeak-ng instance (indian_english.espeak, under piper's lock):
kokoro-onnx's phonemizer path would start a second espeak-ng, so it is never called (we pass phonemes).
kokoro-onnx still dlopens its bundled libespeak-ng at load; that alone initializes nothing.
The ONNX session is shared by all worker threads (run() is thread-safe). Code MIT, weights Apache-2.0.
KOKORO_TORCH=1 runs the same weights in PyTorch on CPU (hexgrad `kokoro` KModel) under kokoro-onnx's own text/style/batch
path: on the M4 it is ~1.7x faster than the ONNX export at equal CER/UTMOS (docs/benchmarks.md section 13).
"""
import logging
import re
import sys
import time
import types
from importlib.util import find_spec

import numpy as np
from kokoro_onnx import SAMPLE_RATE, Kokoro

from app.core.config import settings
from app.services import indian_english, voice_catalog
from app.services.conditioning import EngineCapabilities
from app.services.piper_engine import make_session

log = logging.getLogger(__name__)
PREFIX = "kokoro:"
NAMES = {"hf_alpha": "Alpha", "hf_beta": "Beta", "hm_omega": "Omega", "hm_psi": "Psi"}
# misaki E2M: espeak with tie='^' writes these as one tied phoneme; piper's espeak has no tie, so match the pairs.
# ponytail: untied pairs also match across a syllable edge (rare in Hindi); the tie flag would need a custom espeak call
_TIES = [("dʒ", "ʤ"), ("tʃ", "ʧ"), ("aɪ", "I"), ("aʊ", "W"), ("eɪ", "A"), ("oʊ", "O"), ("əʊ", "Q"), ("ɔɪ", "Y")]


def _kmodel():
    if "kokoro" not in sys.modules:  # ponytail: skip kokoro/__init__ (pulls misaki); breaks if kokoro.model starts importing the pipeline
        m = types.ModuleType("kokoro")
        m.__path__ = list(find_spec("kokoro").submodule_search_locations)
        sys.modules["kokoro"] = m
    from kokoro.model import KModel

    return KModel


class _TorchSession:
    """Stands in for the ORT session (no ORT load: its ~450 MB is never returned to the OS). kokoro-onnx reads the
    inputs/outputs/metadata below once, then only calls run(None, {tokens, style, speed}) -> [audio]."""

    def __init__(self, d) -> None:
        import torch

        self.torch, self._model_path = torch, str(d / "kokoro-v1_0.pth")
        self.get_inputs = lambda: [types.SimpleNamespace(name=n, type=t) for n, t in
                                   (("tokens", "tensor(int64)"), ("style", "tensor(float)"), ("speed", "tensor(float)"))]
        self.get_outputs = lambda: [types.SimpleNamespace(name="audio")]
        self.get_modelmeta = lambda: types.SimpleNamespace(custom_metadata_map={})
        torch.set_num_threads(settings.threads_per_worker)
        # CPU only: on MPS UTMOS fell 0.10 (hf_alpha) / 0.49 (hm_omega) vs ONNX; thread-safe, worker threads call it concurrently
        self.model = _kmodel()(repo_id="hexgrad/Kokoro-82M", config=str(d / "config.json"), model=str(d / "kokoro-v1_0.pth")).eval()

    def run(self, _outputs, inputs: dict) -> list[np.ndarray]:
        t = self.torch
        audio, _ = self.model.forward_with_tokens(t.from_numpy(inputs["tokens"]), t.from_numpy(inputs["style"]), float(inputs["speed"][0]))
        return [audio.numpy()]


def phonemes(text: str) -> str:
    """Hindi runs as espeak-ng 'hi' IPA; English runs in the Indian-English IPA Piper uses (INDIAN_ENGLISH): measured
    on 6 Hinglish sentences x 4 voices with Whisper, phoneme error 0.051 -> 0.029 ('loan' was heard as 'learn')."""
    out, pos = [], 0
    for m in [*indian_english._ENGLISH_RUN.finditer(text), None] if settings.indian_english else [None]:
        hindi = text[pos:m.start() if m else len(text)]
        if re.search(r"[^\W_]", hindi):  # espeak-ng reads a lone "।" aloud (पूर्णविराम); bare punctuation adds nothing
            out += ["".join(s) for s in indian_english.espeak("hi", hindi)]
        if m:
            out.append(indian_english.english_ipa(m[0]))
            pos = m.end()
    ps = " ".join(out).replace("ʲ", "")
    for a, b in _TIES:
        ps = ps.replace(a, b)
    return ps


class KokoroEngine:
    supports_cloning = False
    capabilities = EngineCapabilities(speed=True, streaming="sentence", sample_rates=(SAMPLE_RATE,), languages=("hi", "hinglish"))

    def __init__(self) -> None:
        self.kokoro: Kokoro | None = None
        self._styles: dict[str, np.ndarray] = {}  # voice_id -> (510, 1, 256) style table
        self.max_workers = settings.workers

    @property
    def ready(self) -> bool:
        return self.kokoro is not None

    def load(self) -> None:
        t = time.perf_counter()
        d = settings.kokoro_dir
        torch_ok = settings.kokoro_torch and (d / "pt/kokoro-v1_0.pth").exists()
        if settings.kokoro_torch and not torch_ok:
            log.warning("kokoro: no %s, staying on ONNX (scripts/download_voices.py kokoro-pt)", d / "pt/kokoro-v1_0.pth")
        sess = _TorchSession(d / "pt") if torch_ok else \
            make_session(d / "kokoro-v1.0.onnx", settings.threads_per_worker, settings.use_cuda)
        k = Kokoro.from_session(sess, str(d / "voices-v1.0.bin"))
        self._styles = {PREFIX + n: k.voices[n] for n in NAMES}  # read once: NpzFile unzips on every access
        self.kokoro = k
        self.synth("नमस्ते।", next(iter(self._styles)), 1.0)  # warm-up: first run allocates
        log.info("kokoro voices loaded", extra={"extra_fields": {"voices": list(self._styles), "secs": round(time.perf_counter() - t, 2)}})

    def voices(self) -> list[dict]:
        return [{"voice_id": v, "sample_rate": SAMPLE_RATE, "language": "hi", "engine": "kokoro",
                 "gender": voice_catalog.gender_of(v), "name": f"Kokoro {NAMES[v[len(PREFIX):]]}"} for v in self._styles]

    def has_voice(self, voice: str) -> bool:
        return voice in self._styles

    def sample_rate(self, voice: str) -> int:
        return SAMPLE_RATE

    def synth(self, text: str, voice: str, speed: float, ref=None, ref_text=None) -> np.ndarray:
        ps = phonemes(text)
        if not ps.strip():
            return np.zeros(0, np.float32)
        # trim=False keeps the model's own end-of-sentence pause; tts.trim_lead/trim_tail cap lead and tail silence
        wav, _ = self.kokoro.create(ps, self._styles[voice], speed=speed, is_phonemes=True, trim=False)
        return np.asarray(wav, dtype=np.float32)
