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
    routing_policy: Literal["fast", "balanced", "expressive", "clone"] | None = None  # engine selection, see app/services/routing.py

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
    # Which values native_emotion / native_style cover; () = every value. A Piper multi-speaker voice has recorded only
    # the styles its catalog entry lists (voices/catalog.json), so only those are native for it.
    emotion_values: tuple[str, ...] = ()
    style_values: tuple[str, ...] = ()
    # emotion/style select a recorded speaker: no *_strength, and emotion and style cannot both be sent (one speaker per request)
    discrete_styles: bool = False
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
        for k in ("speed_range", "sample_rates", "languages", "emotion_values", "style_values"):
            d[k] = list(d[k])
        return d


DEFAULT_CAPABILITIES = EngineCapabilities(speed=True, streaming="sentence")  # engines that declare nothing: speed only


class UnsupportedControl(ValueError):
    def __init__(self, controls: list[str], engine: str = "") -> None:
        self.controls = controls
        super().__init__(f"unsupported controls for {engine or 'this engine'}: {', '.join(controls)}; "
                         "use fallback=\"ignore\" to synthesize without them")


def control_kinds(caps: EngineCapabilities, dsp: bool = False) -> dict[str, str]:
    """How each control would be realised on this engine: native | steered (prompt) | dsp (post-processing) | none.
    "embedding" (learned speaker/style embedding) is reported for speaker_embedding / style_reference / cloning."""
    def k(native: bool, steered: bool = False, dsp_ok: bool = False) -> str:
        return "native" if native else "steered" if steered else "dsp" if dsp_ok else "none"
    return {
        "emotion": k(caps.native_emotion, caps.prompt_emotion), "style": k(caps.native_style, caps.prompt_emotion),
        "role": k(caps.role), "pitch": k(caps.pitch, dsp_ok=dsp), "energy": k(caps.energy, dsp_ok=dsp),
        "speaker_embedding": "embedding" if caps.speaker_embedding else "none",
        "style_reference": "embedding" if caps.style_reference else "none",
        "cloning": "embedding" if caps.cloning else "none",
    }


def split_controls(eff: VoiceCondition, caps: EngineCapabilities, dsp: bool = False) -> tuple[dict, dict]:
    """(native controls to pass to the engine, DSP post-processing controls) from an effective (validated) condition."""
    native: dict = {}
    if caps.native_emotion or caps.prompt_emotion:
        native |= {k: getattr(eff, k) for k in ("emotion", "emotion_strength") if getattr(eff, k) is not None}
    if caps.native_style or caps.prompt_emotion:
        native |= {k: getattr(eff, k) for k in ("style", "style_strength") if getattr(eff, k) is not None}
    if caps.role and eff.role is not None:
        native["role"] = eff.role
    for k, ok in (("pitch", caps.pitch), ("energy", caps.energy)):
        if ok and getattr(eff, k) is not None:
            native[k] = getattr(eff, k)
    native = {k: (v.value if isinstance(v, Enum) else v) for k, v in native.items()}
    post: dict = {}
    if dsp:
        post = {k: getattr(eff, k) for k in ("pitch", "energy") if getattr(eff, k) is not None and not getattr(caps, k)}
        if post and eff.prosody_strength is not None:
            post["strength"] = eff.prosody_strength
    return native, post


def validate_condition(cond: VoiceCondition, caps: EngineCapabilities, engine: str = "", speaker_ok: bool = False, dsp: bool = False) -> tuple[VoiceCondition, list[str], list[str]]:
    """Return (effective condition, applied, ignored). Raises UnsupportedControl if anything is unsupported and fallback="reject".

    `speaker_ok` is set by the caller (speech.prepare_ex) when condition.speaker_id was resolved against the registry to a
    binding or reference this engine can serve, after the consent check; otherwise speaker_id is unsupported.

    `dsp` = DSP prosody post-processing is enabled (settings.dsp_prosody): pitch/energy/prosody_strength are then honoured
    by the pipeline for engines that lack them natively and are labelled "pitch:dsp" etc. DSP is never used for emotion/style/role.

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

    gate(["speaker_id"], speaker_ok)
    gate(["speaker_embedding"], caps.speaker_embedding)
    gate(["reference_audio", "reference_text"], caps.cloning)
    gate(["style_reference"], caps.style_reference)
    def native(on: bool, values: tuple[str, ...], v) -> bool:  # native for this value (a voice may cover only some)
        return on and (not values or v is None or getattr(v, "value", v) in values)

    emo_native, sty_native = native(caps.native_emotion, caps.emotion_values, cond.emotion), native(caps.native_style, caps.style_values, cond.style)
    emo_ok = emo_native or caps.prompt_emotion
    gate(["emotion"], emo_ok, "" if emo_native else ":steered")
    gate(["emotion_strength"], emo_ok and not caps.discrete_styles, "" if emo_native else ":steered")
    # discrete styles: one recorded speaker per request, so a style next to an applied emotion is not honoured
    sty_ok = (sty_native or caps.prompt_emotion) and not (caps.discrete_styles and cond.emotion is not None and emo_ok)
    gate(["style"], sty_ok, "" if sty_native else ":steered")
    gate(["style_strength"], sty_ok and not caps.discrete_styles, "" if sty_native else ":steered")
    gate(["role"], caps.role)
    gate(["pitch"], caps.pitch or dsp, "" if caps.pitch else ":dsp")
    gate(["energy"], caps.energy or dsp, "" if caps.energy else ":dsp")
    gate(["prosody_strength"], caps.pitch or caps.energy or dsp, "" if caps.pitch or caps.energy else ":dsp")
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
