"""Request schemas shared by the HTTP and WebSocket APIs.

(Reconstructed: the original file was never committed because .gitignore's `models/` matched app/models.)
"""
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from app.services.conditioning import VoiceCondition

VOICE_ID = r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}"
SampleRate = Literal[8000, 16000, 22050, 24000, 44100, 48000]


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1)
    voice: str = Field("default", max_length=128)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    response_format: Literal["wav", "pcm", "mp3"] = "wav"
    sample_rate: SampleRate | None = None
    reference_audio: str | None = None  # base64; qwen3 cloning
    reference_text: str | None = None
    condition: VoiceCondition | None = None  # optional conditioning: emotion, style, role, ... (docs/voice-system.md)
    routing_policy: Literal["fast", "balanced", "expressive", "clone"] | None = None  # engine choice when voice is "default"


class WsSpeak(BaseModel):
    type: Literal["speak"] = "speak"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12], max_length=64)
    text: str = Field(min_length=1)
    voice: str = Field("default", max_length=128)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    sample_rate: SampleRate | None = None
    frame_ms: int = Field(0, ge=0, le=1000)
    condition: VoiceCondition | None = None
    routing_policy: Literal["fast", "balanced", "expressive", "clone"] | None = None
