import base64
import io

import numpy as np
import soundfile as sf


def decode_audio_b64(data: str) -> tuple[np.ndarray, int]:
    if "," in data[:100]:  # strip data: URI prefix
        data = data.split(",", 1)[1]
    return decode_audio(base64.b64decode(data, validate=True))


def decode_audio(raw: bytes) -> tuple[np.ndarray, int]:
    info = sf.info(io.BytesIO(raw))  # header first: a small FLAC/OGG can decode to gigabytes
    if info.samplerate < 8000 or info.channels > 8 or info.frames > 30.0 * info.samplerate:
        raise ValueError(f"reference audio must be 1-30s, 8 kHz+ and <=8 channels (header: {info.frames} frames, "
                         f"{info.samplerate} Hz, {info.channels} ch)")
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


_KNEE = 0.9


def soft_limit(wav: np.ndarray) -> np.ndarray:
    """Bend peaks above 0.9 smoothly into (0.9, 1.0). Kokoro overshoots to 1.1-1.4 on a few transient samples; hard
    clipping those crackles, while a gain cut would quieten the whole chunk for a handful of samples."""
    a = np.abs(wav)
    if a.max(initial=0) <= _KNEE:
        return wav
    return np.where(a > _KNEE, np.sign(wav) * (_KNEE + (1 - _KNEE) * np.tanh((a - _KNEE) / (1 - _KNEE))), wav)


def to_pcm16(wav: np.ndarray) -> bytes:
    return np.rint(np.clip(soft_limit(wav), -1, 1) * 32767).astype("<i2").tobytes()  # round, not truncate: no DC bias


def encode(wav: np.ndarray, sr: int, fmt: str) -> bytes:
    if fmt == "pcm":
        return to_pcm16(wav)
    buf = io.BytesIO()
    if fmt == "mp3":
        sf.write(buf, wav, sr, format="MP3")
    else:
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
