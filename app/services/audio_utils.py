import base64
import io

import numpy as np
import soundfile as sf


def decode_audio_b64(data: str) -> tuple[np.ndarray, int]:
    if "," in data[:100]:  # strip data: URI prefix
        data = data.split(",", 1)[1]
    return decode_audio(base64.b64decode(data, validate=True))


def decode_audio(raw: bytes) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    dur = len(wav) / sr
    if not 1.0 <= dur <= 30.0:
        raise ValueError(f"reference audio must be 1-30s (3-10s recommended), got {dur:.1f}s")
    return wav, sr


def change_speed(wav: np.ndarray, speed: float) -> np.ndarray:
    if abs(speed - 1.0) < 1e-3:
        return wav
    import librosa

    return librosa.effects.time_stretch(wav, rate=speed)


def to_pcm16(wav: np.ndarray) -> bytes:
    return (np.clip(wav, -1, 1) * 32767).astype("<i2").tobytes()


def encode(wav: np.ndarray, sr: int, fmt: str) -> bytes:
    if fmt == "pcm":
        return to_pcm16(wav)
    buf = io.BytesIO()
    if fmt == "mp3":
        sf.write(buf, wav, sr, format="MP3")
    else:
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
