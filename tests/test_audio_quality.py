"""Objective audio checks on the streaming pipeline: a deterministic fake engine with known silence and level, plus (skipped
when the voice file is absent) the real custom Piper voice. Metrics come from bench/audio_quality.py."""
import asyncio
import time
from pathlib import Path

import numpy as np
import pytest

from app.core.config import settings
from app.services import tts
from app.services.scheduler import Scheduler
from bench import audio_quality as aq

SR = 22050
SENT = "ठीक है। क्या हुआ? बहुत बढ़िया! हम्म... ठीक है, अच्छा; और।"


def voiced(n: int, level: float, start: int = 0) -> np.ndarray:
    t = (np.arange(n) + start) / SR
    return (level * (0.6 * np.sin(2 * np.pi * 180 * t) + 0.3 * np.sin(2 * np.pi * 360 * t) + 0.1 * np.sin(2 * np.pi * 900 * t))).astype(np.float32)


class Fake:
    """Voice `a` quiet-ish, `b` louder; `lead`/`tail` seconds of exact silence around 50 ms of voiced signal per character."""
    supports_cloning = False
    max_workers = 2
    ready = True
    calls = 0

    def __init__(self, levels=None, lead=0.25, tail=0.4, hard_cut=False):
        self.levels, self.lead, self.tail, self.hard_cut = levels or {"a": 0.3, "b": 0.1}, lead, tail, hard_cut

    def load(self): ...
    def voices(self): return [{"voice_id": v, "sample_rate": SR} for v in self.levels]
    def has_voice(self, v): return v in self.levels
    def sample_rate(self, v): return SR

    def synth(self, text, voice, speed, ref=None, ref_text=None):
        type(self).calls += 1
        time.sleep(0.005)
        body = voiced(int(SR * 0.05 * len(text) / speed), self.levels[voice])
        if self.hard_cut:  # voiced signal that stops mid-wave, no silence either side
            return body + np.float32(0.2)
        return np.concatenate([np.zeros(int(SR * self.lead), np.float32), body, np.zeros(int(SR * self.tail), np.float32)])


@pytest.fixture()
def pipe(monkeypatch):
    def make(engine=None, **overrides):
        monkeypatch.setattr(tts, "engine", engine or Fake())
        monkeypatch.setattr(tts, "scheduler", Scheduler(2))
        monkeypatch.setattr(settings, "cache_size", 0)
        for k, v in overrides.items():
            monkeypatch.setattr(settings, k, v)
    return make


def render(text, voice="a", rate=None):
    async def go():
        chunks = []
        async for b in tts.stream(text, voice, 1.0, rate or SR, frame_ms=0):
            chunks.append(aq.pcm(b))
        return chunks
    return asyncio.run(go())


@pytest.mark.parametrize("lead", [0.25, 0.0])
def test_lead_silence_is_fixed_whatever_the_engine_pads(pipe, lead):
    pipe(Fake(lead=lead))
    for c in render("ठीक है। क्या हुआ?"):
        assert abs(aq.lead_trail_ms(c, SR)[0] - settings.lead_silence_ms) < 1.5


def test_gap_between_pieces_follows_the_ending_punctuation(pipe):
    pipe()
    chunks = render(SENT)
    pieces = tts.split_for_stream(SENT)
    assert len(chunks) == len(pieces) == 5
    for i, p in enumerate(pieces[:-1]):
        gap = aq.lead_trail_ms(chunks[i], SR)[1] + aq.lead_trail_ms(chunks[i + 1], SR)[0]
        assert abs(gap - tts.gap_ms(p)) < 3, (p, gap)
    assert len({round(aq.lead_trail_ms(chunks[i], SR)[1]) for i in (1, 2)}) == 2  # question != exclamation


def test_final_piece_trailing_silence_is_capped_not_padded(pipe):
    pipe(Fake(tail=0.6))
    last = render("ठीक है। अच्छा।")[-1]
    assert aq.lead_trail_ms(last, SR)[1] <= tts.gap_ms("अच्छा।") - settings.lead_silence_ms + 1
    pipe(Fake(tail=0.0))
    assert aq.lead_trail_ms(render("ठीक है। अच्छा।")[-1], SR)[1] < 1  # nothing is added after the last piece


def test_legacy_settings_reproduce_the_old_caps(pipe):
    pipe(pause_plan="", fade_ms=0, voice_gain_db="")
    c = render("ठीक है। क्या हुआ?")
    assert abs(aq.lead_trail_ms(c[0], SR)[1] - settings.sentence_pause_ms) < 1.5  # 400 ms engine tail capped at 350
    assert abs(aq.lead_trail_ms(c[1], SR)[0] - settings.lead_silence_ms) < 1.5
    pipe(Fake(lead=0.0), pause_plan="", fade_ms=0, voice_gain_db="")
    assert aq.lead_trail_ms(render("ठीक है।")[0], SR)[0] < 1  # old code never padded a lead


def test_soft_limiter_prevents_clipping_on_overshoot(pipe):
    pipe(Fake({"a": 1.4}), voice_gain_db="")
    wav = np.concatenate(render("ठीक है। क्या हुआ?"))
    assert np.abs(wav).max() > 0.9 and aq.clipped_fraction(wav) == 0  # peaks bent below full scale, not flattened
    assert abs(float(wav.mean())) < 0.01


def test_static_gain_equalises_voices_and_does_not_pump_within_one(pipe):
    pipe(voice_gain_db="")
    text = "आपकी किस्त की तारीख पंद्रह अगस्त है, इसलिए कृपया उससे पहले भुगतान कर दीजिए, नहीं तो खाते पर शुल्क लगेगा।"
    a, b = (np.concatenate(render(text, v)) for v in "ab")
    assert abs(aq.lufs(a, SR) - aq.lufs(b, SR)) > 8  # 0.3 vs 0.1 is 9.5 dB
    pipe(voice_gain_db=f"a:0,b:{aq.lufs(a, SR) - aq.lufs(b, SR):.2f}")
    cb = render(text, "b")
    assert abs(aq.lufs(np.concatenate(render(text, "a")), SR) - aq.lufs(np.concatenate(cb), SR)) < 0.3
    assert aq.utterance_stats(cb, SR)["chunk_lufs_spread_db"] < 1.0  # one gain per voice: chunks keep their own dynamics


def test_voice_gain_lookup_and_bad_settings(monkeypatch):
    monkeypatch.setattr(settings, "voice_gain_db", "m:spk:-6,x:3")
    assert abs(tts.voice_gain("m:spk") - 10 ** (-6 / 20)) < 1e-9 and tts.voice_gain("x:other") == pytest.approx(10 ** (3 / 20)) and tts.voice_gain("y") == 1.0
    monkeypatch.setattr(settings, "pause_plan", "comma:100,bogus:1")
    with pytest.raises(ValueError):
        tts.check_audio_settings()


def test_fade_only_at_a_hard_cut_never_on_a_natural_onset(pipe):
    pipe(Fake(hard_cut=True), voice_gain_db="")
    on = render("ठीक है। क्या हुआ?")
    pipe(Fake(hard_cut=True), voice_gain_db="", fade_ms=0)
    off = render("ठीक है। क्या हुआ?")
    lead = settings.lead_silence_ms * SR // 1000
    for c in on:  # engine output stops mid-wave: with the fade the first/last samples ease to ~0
        assert abs(c[lead]) < 0.05 and abs(c[-1]) < 0.01
    assert max(abs(off[0][lead]), abs(off[0][-1])) > 0.1  # without it: a 0.2-sized step
    pipe(Fake(), voice_gain_db="")  # natural onset after silence: byte-identical with and without the fade
    n1 = render("ठीक है।")
    pipe(Fake(), voice_gain_db="", fade_ms=0)
    assert np.array_equal(n1[0], render("ठीक है।")[0])


def test_no_discontinuity_at_seams(pipe):
    pipe(Fake(hard_cut=True), voice_gain_db="")
    chunks = render(SENT)
    wav, starts = np.concatenate(chunks), np.cumsum([0] + [len(c) for c in chunks])
    for s in starts[1:-1]:
        st = aq.seam_stats(wav, SR, int(s))
        assert st["jump"] < 0.01 and st["flux_spike"] < 2.0, st


@pytest.mark.parametrize("rate", [8000, 16000])
def test_telephony_length_level_and_aliasing(pipe, rate):
    pipe(cache_size=64)
    native = np.concatenate(render(SENT))
    out = np.concatenate(render(SENT, rate=rate))  # second render: same cached chunks, so same samples
    st = aq.telephony_stats(native, SR, out, rate)
    assert abs(st["dur_ratio"] - 1) < 0.005 and abs(st["level_change_db"]) < 0.5
    assert st["guard_db"] <= 1.0 and abs(st["pass_db"]) < 0.5 and st["clip_frac"] == 0


def test_cancelling_a_stream_stops_the_look_ahead(pipe):
    pipe()
    Fake.calls = 0

    async def go():
        gen = tts.stream(SENT * 3, "a", 1.0, SR, frame_ms=0)
        await gen.__anext__()
        await gen.aclose()
        await asyncio.sleep(0.2)
    asyncio.run(go())
    assert Fake.calls <= 2 and tts.stats["streams_active"] == 0  # first piece + at most the one look-ahead


REAL = Path(__file__).resolve().parent.parent / "voices" / "hi_IN-custom-medium.onnx"


@pytest.mark.skipif(not REAL.exists(), reason="custom Piper voice not present")
def test_real_custom_voice(monkeypatch):
    from app.services.piper_engine import PiperEngine

    monkeypatch.setattr(settings, "models_dir", REAL.parent)
    monkeypatch.setattr(settings, "models_extra", "")
    monkeypatch.setattr(settings, "cache_size", 64)
    eng = PiperEngine()
    eng.load()
    monkeypatch.setattr(tts, "engine", eng)
    monkeypatch.setattr(tts, "scheduler", Scheduler(2))
    v = "hi_IN-custom-medium"
    sr = eng.sample_rate(v)
    text = "नमस्ते, मैं श्रेया बोल रही हूँ। क्या आप अभी बात कर सकते हैं? ठीक है।"

    async def go(rate):
        return [aq.pcm(b) async for b in tts.stream(text, v, 1.0, rate, frame_ms=0)]
    chunks = asyncio.run(go(sr))
    st = aq.utterance_stats(chunks, sr)
    assert 20 <= st["lead_ms"] <= 31 and st["clip_frac"] == 0 and st["true_peak_db"] < -1.0
    assert -23 < st["lufs"] < -17 and abs(st["dc"]) < 0.005 and st["edge_db"] < -50
    assert st["chunk_lufs_spread_db"] < 4 and all(s["jump"] < 0.01 for s in st["seams"])
    for rate in (8000, 16000):
        out = np.concatenate(asyncio.run(go(rate)))
        t = aq.telephony_stats(np.concatenate(chunks), sr, out, rate)
        assert abs(t["dur_ratio"] - 1) < 0.005 and abs(t["level_change_db"]) < 1.0 and t["guard_db"] < 1.0
