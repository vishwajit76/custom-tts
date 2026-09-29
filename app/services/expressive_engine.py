"""Integration boundary for an engine with model-NATIVE emotion / style / role control (Phase 5).

STATUS: no engine shipped here implements this. Piper, Supertonic, Kokoro and Qwen3-TTS (Base) take no emotion input, and the
candidate bake-off (docs/model-selection.md) has not run. This module only fixes HOW native controls would flow, so a
future engine is a drop-in and the pipeline stays honest:

  request.condition -> validate_condition(caps) -> split_controls() -> tts.stream(controls=...) -> engine.synth_native(...)

Rules for an implementation:
  * Declare `capabilities` truthfully (native_emotion / native_style / role / pitch / energy). Undeclared = rejected (422)
    or listed as ignored; the pipeline never passes an engine a control it did not declare.
  * `synth` (no controls) must keep working: neutral delivery, used by cache warm-up and unconditioned requests.
  * `synth_native` receives only the declared, validated controls as plain values, e.g.
    {"emotion": "calm", "emotion_strength": 0.6, "style": "warm", "role": "customer_support"}. How they reach the model
    (special tokens, a style embedding lookup, an instruction prompt) is the engine's business.
  * Prompt-steered engines (prompt_emotion=True) get the same dict but are labelled "emotion:steered" (unvalidated).

Disabled unless configured: set ENGINES=expressive and EXPRESSIVE_ENGINE=package.module:ClassName (a subclass of
ExpressiveEngine). Nothing is imported otherwise.
"""
import importlib

import numpy as np

from app.services.conditioning import EngineCapabilities


class ExpressiveEngine:
    """Base class / interface. Subclasses set `capabilities` and implement the methods below."""

    supports_cloning = False
    max_workers = 1
    ready = False
    capabilities = EngineCapabilities()

    def load(self) -> None:
        raise NotImplementedError

    def voices(self) -> list[dict]:
        raise NotImplementedError

    def has_voice(self, voice: str) -> bool:
        return any(v["voice_id"] == voice for v in self.voices())

    def sample_rate(self, voice: str) -> int:
        raise NotImplementedError

    def synth(self, text: str, voice: str, speed: float, ref=None, ref_text=None) -> np.ndarray:
        raise NotImplementedError

    def synth_native(self, text: str, voice: str, speed: float, controls: dict, ref=None, ref_text=None) -> np.ndarray:
        """Synthesize with native controls (see module docstring). Default: refuse, so an engine that declares native
        capabilities but forgot to implement them fails loudly instead of silently ignoring the request."""
        raise NotImplementedError(f"{type(self).__name__} declares native controls but does not implement synth_native")


class ExpressiveNotConfigured(RuntimeError):
    pass


def build_configured(path: str) -> ExpressiveEngine:
    """Instantiate `package.module:ClassName`. Raises ExpressiveNotConfigured when empty, ValueError for a wrong class."""
    if not path:
        raise ExpressiveNotConfigured("ENGINES includes 'expressive' but EXPRESSIVE_ENGINE is not set (package.module:ClassName)")
    mod, _, cls = path.partition(":")
    if not cls:
        raise ValueError(f"EXPRESSIVE_ENGINE must look like package.module:ClassName, got {path!r}")
    klass = getattr(importlib.import_module(mod), cls)
    if not (isinstance(klass, type) and issubclass(klass, ExpressiveEngine)):
        raise ValueError(f"{path} is not an ExpressiveEngine subclass")
    return klass()
