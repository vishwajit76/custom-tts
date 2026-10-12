import types

import numpy as np
import pytest

pytest.importorskip("kokoro_onnx")
from app.services import kokoro_engine  # noqa: E402


@pytest.mark.parametrize("on, has_pt, want", [(False, True, "onnx"), (True, False, "onnx"), (True, True, "torch")])
def test_kokoro_torch_picks_backend(monkeypatch, tmp_path, on, has_pt, want):
    if has_pt:
        (tmp_path / "pt").mkdir()
        (tmp_path / "pt/kokoro-v1_0.pth").touch()
    monkeypatch.setattr(kokoro_engine.settings, "kokoro_dir", tmp_path)
    monkeypatch.setattr(kokoro_engine.settings, "kokoro_torch", on)
    monkeypatch.setattr(kokoro_engine, "make_session", lambda *a: "onnx")
    monkeypatch.setattr(kokoro_engine, "_TorchSession", lambda *a: "torch")
    got = {}
    monkeypatch.setattr(kokoro_engine.Kokoro, "from_session", lambda sess, _v: got.setdefault("s", types.SimpleNamespace(
        sess=sess, voices={n: np.zeros((510, 1, 256), np.float32) for n in kokoro_engine.NAMES})))
    monkeypatch.setattr(kokoro_engine.KokoroEngine, "synth", lambda *a: None)
    kokoro_engine.KokoroEngine().load()
    assert got["s"].sess == want
