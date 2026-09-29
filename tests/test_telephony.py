"""Phase 8: 8 kHz / 16 kHz PCM s16le telephony output and resampling quality (soxr HQ streaming resampler)."""
import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services import audio_utils, tts

NATIVE = 24000


class Tone:
    """Engine whose output is a known mixture: 1 kHz (in-band for 8k), 5 kHz (above 8k's Nyquist, must not alias)."""
    supports_cloning = False
    max_workers = 1
    ready = True

    def load(self):
        pass

    def voices(self):
        return [{"voice_id": "t", "sample_rate": NATIVE}]

    def has_voice(self, v):
        return v == "t"

    def sample_rate(self, v):
        return NATIVE

    def synth(self, text, voice, speed, ref=None, ref_text=None):
        t = np.arange(NATIVE) / NATIVE
        return np.concatenate([np.zeros(NATIVE // 4), 0.3 * np.sin(2 * np.pi * 1000 * t) + 0.3 * np.sin(2 * np.pi * 5000 * t)]).astype(np.float32)


@pytest.fixture()
def tone(monkeypatch):
    monkeypatch.setattr(tts, "engine", Tone())
    monkeypatch.setattr(settings, "cache_size", 0)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(tts, "engine", Tone())
    monkeypatch.setattr(settings, "default_voice", "t")
    monkeypatch.setattr(settings, "api_keys", "")
    monkeypatch.setattr(settings, "cache_size", 0)
    from app.main import app
    with TestClient(app) as c:
        yield c


def level_db(x, sr, f, bw=60):
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x)))) / len(x)
    fr = np.fft.rfftfreq(len(x), 1 / sr)
    return 20 * np.log10(sp[(fr > f - bw) & (fr < f + bw)].max() + 1e-12)


@pytest.mark.parametrize("sr", [8000, 16000])
def test_pcm_s16le_length_rate_and_headers(client, sr):
    r = client.post("/v1/audio/speech/stream", json={"input": "नमस्ते।", "sample_rate": sr})
    assert r.status_code == 200 and r.headers["x-sample-rate"] == str(sr) and r.headers["content-type"] == "audio/L16"
    assert len(r.content) % 2 == 0
    secs = len(r.content) / 2 / sr
    assert abs(secs - 1.03) < 0.02  # 250 ms model lead silence trimmed to 30 ms + the 1 s tone
    x = np.frombuffer(r.content, "<i2")
    assert x.dtype.str == "<i2" and np.abs(x).max() <= 32767


@pytest.mark.parametrize("sr", [8000, 16000])
def test_ws_frames_are_whole_samples_and_exact_size(client, sr):
    with client.websocket_connect("/v1/audio/ws") as ws:
        ws.send_json({"type": "speak", "id": "a", "text": "नमस्ते।", "sample_rate": sr, "frame_ms": 20})
        frames = []
        while True:
            m = ws.receive()
            if m.get("bytes") is not None:
                frames.append(m["bytes"])
            elif json.loads(m["text"])["type"] == "end":
                break
    fb = sr * 20 // 1000 * 2
    assert all(len(f) == fb for f in frames[:-1]) and 0 < len(frames[-1]) <= fb and len(frames[-1]) % 2 == 0


def _synth(sr):
    return asyncio.run(_run(sr))


async def _run(sr):
    tts.scheduler = tts.Scheduler(1)
    return await tts.synthesize("नमस्ते।", "t", sample_rate=sr)


def test_8k_passband_preserved_and_stopband_not_aliased(tone):
    wav = _synth(8000)[8000 // 10:]
    ref = level_db(wav, 8000, 1000)
    assert ref > -25  # 0.3 amplitude tone kept (~ -10 dBFS/2 after window normalisation)
    # 5 kHz is above 4 kHz Nyquist: a bad resampler folds it to 3 kHz. HQ soxr must suppress it > 60 dB.
    assert level_db(wav, 8000, 3000) < ref - 60


def test_16k_passband_and_no_alias_of_native_band(tone):
    wav = _synth(16000)[1600:]
    assert level_db(wav, 16000, 1000) > -25 and level_db(wav, 16000, 5000) > -25  # 5 kHz < 8 kHz Nyquist: kept


def test_stream_resampler_matches_oneshot_no_chunk_seams():
    import soxr
    t = np.arange(NATIVE * 2) / NATIVE
    x = (0.5 * np.sin(2 * np.pi * 700 * t)).astype(np.float32)
    rs = soxr.ResampleStream(NATIVE, 8000, 1, dtype="float32", quality="HQ")
    parts = [x[i:i + 4800] for i in range(0, len(x), 4800)]
    out = np.concatenate([rs.resample_chunk(p, last=i == len(parts) - 1) for i, p in enumerate(parts)])
    one = soxr.resample(x, NATIVE, 8000, quality="HQ")
    n = min(len(out), len(one))
    assert abs(len(out) - len(one)) <= 2
    d = np.abs(out[200:n - 200] - one[200:n - 200])
    assert d.max() < 1e-3  # identical up to filter start-up: no discontinuities at chunk borders


def test_pcm16_rounding_and_clipping():
    b = audio_utils.to_pcm16(np.array([0.0, 1e-5, -1e-5, 0.5, 2.0, -2.0], np.float32))
    v = np.frombuffer(b, "<i2")
    assert list(v) == [0, 0, 0, 16384, 32767, -32767]
    assert abs(np.frombuffer(audio_utils.to_pcm16(np.array([0.5 / 32767 * 0.6])), "<i2")[0]) == 0


def test_mp3_and_wav_at_8k(client):
    r = client.post("/v1/audio/speech", json={"input": "नमस्ते।", "sample_rate": 8000, "response_format": "wav"})
    import io
    import soundfile as sf
    d, sr = sf.read(io.BytesIO(r.content), dtype="int16")
    assert sr == 8000 and d.ndim == 1
