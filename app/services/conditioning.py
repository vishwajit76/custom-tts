"""Engine-independent voice conditioning contract and truthful capability checks.

A request may ask for controls (emotion, style, role, pitch, ...). Each engine advertises which of them it really
implements (`EngineCapabilities`); `validate_condition` refuses the rest (fallback="reject") or drops them and says so
(fallback="ignore"). A control is only reported as applied when the engine honours it.
"""
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Emotion(str, Enum):
    neutral = "neutral"
    happy = "happy"
    sad = "sad"
    excited = "excited"
    calm = "calm"
    concerned = "concerned"
    empathetic = "empathetic"
    apologetic = "apologetic"
    serious = "serious"


class Style(str, Enum):
    conversational = "conversational"
    professional = "professional"
    warm = "warm"
    narration = "narration"
    storytelling = "storytelling"
    energetic = "energetic"
    soft = "soft"


class Role(str, Enum):
    customer_support = "customer_support"
    assistant = "assistant"
    narrator = "narrator"
    teacher = "teacher"
    announcer = "announcer"


SPEED_MIN, SPEED_MAX = 0.5, 2.0  # same bounds as SpeechRequest / README


class VoiceCondition(BaseModel):
    speaker_id: str | None = Field(None, max_length=128)
    speaker_embedding: list[float] | None = Field(None, max_length=4096)
    reference_audio: str | None = None  # base64 audio (optionally a data: URI)
    reference_text: str | None = Field(None, max_length=2000)
    style_reference: str | None = None  # base64 audio whose delivery to imitate
    emotion: Emotion | None = None
    emotion_strength: float | None = Field(None, ge=0, le=1)
    style: Style | None = None
    style_strength: float | None = Field(None, ge=0, le=1)
    role: Role | None = None
    speed: float = Field(1.0, ge=SPEED_MIN, le=SPEED_MAX)
    pitch: float | None = Field(None, ge=-12, le=12)  # semitones relative to the voice
    energy: float | None = Field(None, ge=0, le=2)  # 1.0 = the voice's own
    prosody_strength: float | None = Field(None, ge=0, le=1)
    fallback: Literal["reject", "ignore"] = "reject"

    @model_validator(mode="after")
    def _dependents(self):
        for strength, base in (("emotion_strength", "emotion"), ("style_strength", "style")):
            if getattr(self, strength) is not None and getattr(self, base) is None:
                raise ValueError(f"{strength} needs {base}")
        if self.reference_text and not self.reference_audio:
            raise ValueError("reference_text needs reference_audio")
        return self


@dataclass(frozen=True)
class EngineCapabilities:
    """What an engine actually does today. Everything defaults to False: a capability must be claimed on purpose."""
    cloning: bool = False  # zero-shot cloning from reference_audio
    speaker_embedding: bool = False  # consumes an external speaker embedding
    native_emotion: bool = False  # model trained with emotion conditioning
    native_style: bool = False
    prompt_emotion: bool = False  # emotion/style steered by a text prompt: best effort, NOT validated
    role: bool = False
    style_reference: bool = False
    speed: bool = False
    speed_range: tuple[float, float] = (SPEED_MIN, SPEED_MAX)  # values outside are clamped by the engine
    pitch: bool = False
    energy: bool = False
    streaming: Literal["chunk", "sentence", "none"] = "none"  # "sentence": whole sentence/clause per synth call
    sample_rates: tuple[int, ...] = ()  # native output rates; () = per voice, see /v1/voices
    languages: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        d = asdict(self)
        d["speed_range"], d["sample_rates"], d["languages"] = list(self.speed_range), list(self.sample_rates), list(self.languages)
        return d


DEFAULT_CAPABILITIES = EngineCapabilities(speed=True, streaming="sentence")  # engines that declare nothing: speed only


class UnsupportedControl(ValueError):
    def __init__(self, controls: list[str], engine: str = "") -> None:
        self.controls = controls
        super().__init__(f"unsupported controls for {engine or 'this engine'}: {', '.join(controls)}; "
                         "use fallback=\"ignore\" to synthesize without them")


def validate_condition(cond: VoiceCondition, caps: EngineCapabilities, engine: str = "") -> tuple[VoiceCondition, list[str], list[str]]:
    """Return (effective condition, applied, ignored). Raises UnsupportedControl if anything is unsupported and fallback="reject".

    Ignored controls are reset to their defaults in the effective condition. Applied names for prompt-steered
    controls carry a ":steered" suffix (best effort, unvalidated).
    """
    applied: list[str] = []
    ignored: list[str] = []
    drop: dict[str, None] = {}

    def gate(names: list[str], ok: bool, tag: str = "") -> None:
        given = [n for n in names if getattr(cond, n) not in (None, "")]
        if not given:
            return
        for n in given:
            (applied if ok else ignored).append(n + tag if ok else n)
            if not ok:
                drop[n] = None

    gate(["speaker_id"], False)  # no engine binds speaker ids yet; voices are selected with `voice`
    gate(["speaker_embedding"], caps.speaker_embedding)
    gate(["reference_audio", "reference_text"], caps.cloning)
    gate(["style_reference"], caps.style_reference)
    gate(["emotion", "emotion_strength"], caps.native_emotion or caps.prompt_emotion, "" if caps.native_emotion else ":steered")
    gate(["style", "style_strength"], caps.native_style or caps.prompt_emotion, "" if caps.native_style else ":steered")
    gate(["role"], caps.role)
    gate(["pitch"], caps.pitch)
    gate(["energy"], caps.energy)
    gate(["prosody_strength"], caps.pitch or caps.energy)
    speed = cond.speed
    if speed != 1.0:
        if caps.speed:
            lo, hi = caps.speed_range
            clamped = min(max(speed, lo), hi)
            applied.append("speed" if clamped == speed else f"speed:clamped_to_{clamped:g}")
        else:
            ignored.append("speed")
            drop["speed"] = None
    if ignored and cond.fallback == "reject":
        raise UnsupportedControl(ignored, engine)
    eff = cond.model_copy(update={n: (1.0 if n == "speed" else None) for n in drop})
    return eff, applied, ignored
