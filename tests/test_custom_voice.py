"""The fine-tuned voice (voices/hi_IN-custom-medium.onnx, gitignored) is selectable via MODELS_EXTRA without touching other voices."""
from pathlib import Path

import numpy as np
import pytest

from app.core.config import settings
from app.services import piper_engine

CUSTOM = Path(__file__).resolve().parent.parent / "voices" / "hi_IN-custom-medium.onnx"


def test_model_files_orders_dirs_and_first_stem_wins(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    for p in (a / "hi_IN-rohan-medium.onnx", b / "hi_IN-custom-medium.onnx", b / "hi_IN-rohan-medium.onnx", b / "notes.txt"):
        p.write_bytes(b"")
    got = piper_engine.model_files([a, b, tmp_path / "missing"])
    assert [(p.parent.name, p.name) for p in got] == [("a", "hi_IN-rohan-medium.onnx"), ("b", "hi_IN-custom-medium.onnx")]  # b's rohan cannot shadow a's


def test_models_extra_is_empty_by_default():
    assert settings.models_extra == ""  # no behaviour change for existing deployments


@pytest.mark.skipif(not CUSTOM.exists(), reason="custom voice not present (gitignored)")
def test_custom_voice_loads_and_speaks_through_the_engine(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "models_dir", tmp_path)  # an empty default dir: only the extra one provides voices
    monkeypatch.setattr(settings, "models_extra", str(CUSTOM.parent))
    e = piper_engine.PiperEngine()
    e.load()
    assert e.has_voice("hi_IN-custom-medium")
    v = next(v for v in e.voices() if v["voice_id"] == "hi_IN-custom-medium")
    assert v["sample_rate"] == 22050 and v["gender"] is None  # no gender claimed for a voice we have no card for
    wav = e.synth("नमस्ते, आप कैसे हैं?", "hi_IN-custom-medium", 1.0)
    assert wav.dtype == np.float32 and len(wav) / 22050 > 0.8 and np.abs(wav).max() > 0.05
