"""Run bench/quality.py with faster-whisper (CTranslate2, CPU int8) as the ASR instead of transformers Whisper.

Usage: WHISPER_MODEL=small python -m bench.quality_fw --voices hi_IN-rohan-medium --label quality-fw-small
Needs `pip install faster-whisper` and reachable weights (Systran/faster-whisper-<size> on the HF hub). Same args as bench.quality.
"""
import os

import numpy as np
import soxr
from faster_whisper import WhisperModel

import training.asr as asr

_model = WhisperModel(os.environ.get("WHISPER_MODEL", "small"), device="cpu", compute_type="int8")


def transcribe(wav: np.ndarray, sr: int) -> str:
    w = soxr.resample(np.asarray(wav, np.float32), sr, 16000) if sr != 16000 else np.asarray(wav, np.float32)
    segs, _ = _model.transcribe(w, language="hi", beam_size=5, condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segs)


asr.transcribe = transcribe
from bench import quality  # noqa: E402  (imports transcribe from training.asr after the patch)

_utmos = quality.utmos
_utmos_err: list = []


def _safe_utmos(wav, sr):  # UTMOS comes from torch.hub (GitHub); when unreachable report NaN instead of aborting the CER run
    if _utmos_err:
        return float("nan")
    try:
        return _utmos(wav, sr)
    except Exception as e:  # noqa: BLE001
        _utmos_err.append(repr(e)[:200])
        print("UTMOS unavailable:", _utmos_err[0])
        return float("nan")


quality.utmos = _safe_utmos

if __name__ == "__main__":
    quality.main()
