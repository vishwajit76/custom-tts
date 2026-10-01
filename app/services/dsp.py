"""Optional DSP prosody post-processing: pitch shift (semitones) and energy (gain). Opt-in: settings.dsp_prosody.

This is signal processing on the finished waveform, NOT model-native control and never a substitute for emotion.
Pitch uses librosa's phase-vocoder time-stretch + soxr resampling: duration is preserved; formants move with pitch
(voice gets smaller/larger), so quality is fine for roughly +/-3 semitones and audibly artificial beyond ~+/-6.
Pitch shifting is followed by an RMS match to the input, so pitch does not change loudness.
Energy is a linear gain with a soft limiter above -1 dBFS-ish, so boosting never hard-clips.
"""
import numpy as np

MIN_SAMPLES = 2048  # librosa's phase vocoder needs a few frames; shorter chunks are returned unshifted


def soft_limit(wav: np.ndarray, knee: float = 0.8) -> np.ndarray:
    """Identity below `knee`; above it a tanh curve that approaches 1.0 asymptotically (no hard clipping)."""
    a = np.abs(wav)
    over = a > knee
    if not over.any():
        return wav
    out = wav.copy()
    room = 1.0 - knee
    out[over] = np.sign(wav[over]) * (knee + room * np.tanh((a[over] - knee) / room))
    return out


def apply_prosody(wav: np.ndarray, sr: int, pitch: float | None = None, energy: float | None = None, strength: float | None = None) -> np.ndarray:
    """pitch in semitones, energy = gain multiplier (1.0 = unchanged); `strength` (0..1, default 1) scales both toward neutral."""
    k = 1.0 if strength is None else strength
    if pitch and abs(pitch * k) >= 0.05 and len(wav) >= MIN_SAMPLES:
        import librosa

        src = wav.astype(np.float32)
        wav = librosa.effects.pitch_shift(src, sr=sr, n_steps=float(pitch * k), res_type="soxr_hq")
        r_in, r_out = float(np.sqrt(np.mean(src ** 2))), float(np.sqrt(np.mean(wav ** 2)))
        if r_out > 1e-6 and r_in > 1e-6:  # the phase vocoder loses ~3 dB: keep loudness independent of pitch
            wav = soft_limit(wav * np.float32(min(max(r_in / r_out, 0.5), 2.0)))
    if energy is not None and abs(energy - 1.0) * k > 1e-3:
        wav = soft_limit(wav * np.float32(1.0 + (energy - 1.0) * k))
    return wav.astype(np.float32, copy=False)
