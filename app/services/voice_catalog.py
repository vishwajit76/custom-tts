"""Explicit voice catalog (voices/catalog.json): which public voice is which model + speaker, with documented metadata.

Gender, age group, style and family are *declared here*, never derived from a voice id string. A voice that is not in
the catalog (a model dropped into MODELS_DIR, an uploaded qwen3 reference) simply has gender None / age_group
"unspecified". Format and rules: docs/voices.md.

`get()` loads and validates once per catalog path (startup calls it; a bad file raises CatalogError and the server does
not start). A missing file is an empty catalog with one warning. Engines decide availability: a Piper entry whose model
file is absent is skipped with a warning (PiperEngine.load); `planned` entries are never registered.
"""
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.core.config import settings
from app.models.schemas import VOICE_ID
from app.services.conditioning import Emotion, Style

log = logging.getLogger(__name__)
# style names a request can actually carry (condition.emotion / condition.style); a catalog style outside these is dead config
STYLE_NAMES = frozenset(x.value for x in (*Emotion, *Style))
_ID = f"^{VOICE_ID}$"


class CatalogError(ValueError):
    pass


class Voice(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo in a field name must not silently drop metadata

    voice_id: str = Field(pattern=_ID)
    engine: str = Field(min_length=1)
    model: str | None = None  # piper: model file stem (<model>.onnx + <model>.onnx.json)
    speaker: str | None = None  # key in the model's speaker_id_map; omit for single-speaker models
    styles: dict[str, str] = {}  # style/emotion name (a conditioning.Emotion or Style value) -> speaker key
    default_style: str | None = None  # key of `styles` used when the request names none (else `speaker`)
    gender: Literal["F", "M"] | None = None
    age_group: Literal["young_adult", "adult", "mature", "unspecified"] = "unspecified"
    age_evidence: str | None = None  # where the age label comes from; required unless age_group is unspecified
    style: str | None = None  # the voice's character, free text ("conversational, warm")
    family: str | None = None
    language: str | None = None
    accent: str | None = None
    role: str | None = None
    license: str | None = None
    data_source: str | None = None
    status: Literal["experimental", "production", "planned"]
    aliases: list[str] = []
    name: str | None = None  # display name for /v1/voices; default "Piper <voice_id>"
    # pronunciation rule groups for this voice's text (app/services/pronunciation); None = settings.pronunciation_rules.
    # A model trained on rule-corrected text must be served with the same rules, so V7 voices pin theirs here.
    pronunciation_rules: str | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.pronunciation_rules is not None:
            from app.services.pronunciation import lexical
            lexical.parse(self.pronunciation_rules)  # unknown group -> ValueError -> catalog rejected
        if self.age_group != "unspecified" and not (self.age_evidence or "").strip():
            raise ValueError(f"{self.voice_id}: age_group {self.age_group!r} needs age_evidence (dataset metadata or a stated listener screening)")
        if self.engine == "piper" and not self.model:
            raise ValueError(f"{self.voice_id}: a piper voice needs `model`")
        if (self.speaker or self.styles or self.default_style) and not self.model:
            raise ValueError(f"{self.voice_id}: speaker/styles/default_style need `model`")
        if bad := sorted(set(self.styles) - STYLE_NAMES):
            raise ValueError(f"{self.voice_id}: style names {bad} are not request values; use conditioning.Emotion/Style values {sorted(STYLE_NAMES)}")
        if self.default_style is not None:
            if self.default_style not in self.styles:
                raise ValueError(f"{self.voice_id}: default_style {self.default_style!r} is not in styles")
            if self.speaker is not None and self.styles[self.default_style] != self.speaker:
                raise ValueError(f"{self.voice_id}: default_style {self.default_style!r} maps to {self.styles[self.default_style]!r}, but speaker is {self.speaker!r}")
        if self.styles and self.speaker is None and self.default_style is None:
            raise ValueError(f"{self.voice_id}: styles need `speaker` or `default_style` (the voice used when no style is requested)")
        for a in self.aliases:
            if not re.fullmatch(VOICE_ID, a) or a == self.voice_id:
                raise ValueError(f"{self.voice_id}: bad alias {a!r}")
        return self

    @property
    def default_speaker(self) -> str | None:
        return self.styles[self.default_style] if self.default_style else self.speaker


class Catalog:
    def __init__(self, voices: list[Voice]) -> None:
        self.voices = tuple(voices)
        self._by_id: dict[str, Voice] = {}
        self._alias: dict[str, str] = {}
        for v in voices:
            for name in (v.voice_id, *v.aliases):
                if name in self._by_id or name in self._alias:
                    raise CatalogError(f"duplicate voice id or alias {name!r}")
            self._by_id[v.voice_id] = v
            self._alias |= {a: v.voice_id for a in v.aliases}

    def get(self, voice_id: str) -> Voice | None:
        return self._by_id.get(voice_id)

    def alias_target(self, name: str) -> str | None:
        """Voice id an alias stands for (None if `name` is not an alias or points at a planned voice)."""
        t = self._alias.get(name)
        return t if t and self._by_id[t].status != "planned" else None

    def active(self, engine: str | None = None) -> list[Voice]:
        """Entries that may be registered: everything except `planned`."""
        return [v for v in self.voices if v.status != "planned" and (engine is None or v.engine == engine)]


@lru_cache(maxsize=8)
def _load(path: str) -> Catalog:
    p = Path(path)
    if not p.exists():
        log.warning("voice catalog not found; voices carry no gender/age metadata", extra={"extra_fields": {"path": path}})
        return Catalog([])
    try:
        data = json.loads(p.read_text("utf-8"))
        if not isinstance(data, dict) or set(data) - {"version", "voices"} or not isinstance(data.get("voices"), list):
            raise CatalogError('expected {"version": 1, "voices": [...]}')
        return Catalog([Voice.model_validate(v) for v in data["voices"]])
    except (ValidationError, json.JSONDecodeError) as e:
        raise CatalogError(f"{path}: {e}") from e


def get() -> Catalog:
    return _load(str(settings.voice_catalog))


def gender_of(voice_id: str) -> str | None:
    """'F' | 'M' as declared in the catalog; None when the voice is not catalogued or no gender is declared."""
    v = get().get(voice_id)
    return v.gender if v else None


def rules_of(voice_id: str) -> str | None:
    """Pronunciation rule groups pinned by the catalog for this voice, or None (use the global setting)."""
    v = get().get(voice_id)
    return v.pronunciation_rules if v else None


def metadata(voice_id: str) -> dict:
    """The additive /v1/voices fields for a voice (None / "unspecified" / [] where the catalog says nothing)."""
    v = get().get(voice_id)
    if v is None:
        return {"gender": None, "age_group": "unspecified", "style": None, "family": None, "styles": [], "status": None, "language": None}
    return {"gender": v.gender, "age_group": v.age_group, "style": v.style, "family": v.family, "styles": list(v.styles),
            "status": v.status, "language": v.language}
