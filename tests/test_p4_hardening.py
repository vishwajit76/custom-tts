"""P4 speaker/cloning hardening: each test pins a defect that was real (race, orphan file, 500, stale cache, unbounded cache)."""
import asyncio
import sys
import threading

import numpy as np
import pytest
import soundfile as sf

from app.core.config import settings
from app.services import speaker_encoder as se
from app.services import speaker_registry as sr
from app.services import tts
from tests.test_speakers import GRANT, A, client, eng, reg, wav_bytes  # noqa: F401  (fixtures)

METRICS = {"sha256": "h0", "duration_s": 4.0, "sample_rate": 16000, "rms_dbfs": -20.0, "clip_fraction": 0.0, "speech_fraction": 0.9, "quality": 0.8}


def _add(reg, sid="asha", n=0, emb=None, backend=None, **kw):
    w = np.zeros(16000 * 4, np.float32) + 0.01
    return reg.add_reference(sid, "o", w, 16000, {**METRICS, "sha256": f"h{n}"}, None, emb, backend, **kw)


def _files(reg, sid="asha"):
    return sorted(str(p.relative_to(reg.root / sid)) for p in (reg.root / sid).rglob("*") if p.is_file())


# ---- default encoder without librosa ----------------------------------------------------------------------------
def test_upload_with_default_mfcc_encoder_but_no_librosa_is_not_a_500(client, monkeypatch):
    """requirements.txt has no librosa, yet SPEAKER_ENCODER defaults to mfcc (needs it): the ImportError escaped as HTTP 500
    after the audio was validated. It is now a BackendUnavailable -> the reference is stored, embedding skipped with a warning."""
    monkeypatch.setitem(sys.modules, "librosa", None)  # `import librosa` raises ImportError
    monkeypatch.setattr(se, "_default", {})
    h = A
    assert client.post("/v1/speakers", json={"id": "asha", "display_name": "A", "consent": GRANT}, headers=h).status_code == 201
    r = client.post("/v1/speakers/asha/references", files={"audio": ("a.wav", wav_bytes(), "audio/wav")}, headers=h)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["raw_retained"] and any("embedding skipped" in w and "librosa" in w for w in body["warnings"])


# ---- blocking work off the event loop ---------------------------------------------------------------------------
def test_reference_ingest_runs_off_the_event_loop(client, monkeypatch):
    """add_reference is async but did decode + encoder + fsync + flock inline, stalling every live WebSocket stream."""
    seen = []
    real = sr.analyze_reference

    def spy(*a, **k):
        try:
            asyncio.get_running_loop()
            seen.append("on-loop")
        except RuntimeError:
            seen.append("off-loop")
        return real(*a, **k)

    monkeypatch.setattr(sr, "analyze_reference", spy)
    client.post("/v1/speakers", json={"id": "asha", "display_name": "A"}, headers=A)
    assert client.post("/v1/speakers/asha/references", files={"audio": ("a.wav", wav_bytes(), "audio/wav")}, headers=A).status_code == 201
    assert seen == ["off-loop"]


# ---- registry: races, partial writes, failed deletion -----------------------------------------------------------
def test_bind_engine_is_atomic_under_concurrency(reg):
    """The legacy upload mirror did get() then update(engine_bindings={**old, new}) outside the lock: concurrent binders lost each other."""
    reg.create("o", "asha", "Asha")
    errs = []

    def bind(i):
        try:
            reg.bind_engine("asha", "o", f"eng{i}", f"v{i}")
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=bind, args=(i,)) for i in range(24)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and reg.get("asha", "o").engine_bindings == {f"eng{i}": f"v{i}" for i in range(24)}


def test_bind_engine_checks_owner(reg):
    reg.create("o", "asha", "Asha")
    with pytest.raises(sr.SpeakerForbidden):
        reg.bind_engine("asha", "other", "piper", "v")


def test_failed_reference_write_leaves_no_orphan_audio(reg, monkeypatch):
    """A failure after the raw clip was written (here: centroid update) left refs/<id>.wav on disk with no record pointing at it."""
    reg.create("o", "asha", "Asha")
    emb = np.ones(4, np.float32)
    monkeypatch.setattr(sr.SpeakerRegistry, "_update_centroid", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        _add(reg, emb=emb, backend="mfcc")
    assert _files(reg) == ["speaker.json"] and reg.get("asha", "o").references == []


def test_embeddings_of_two_backends_are_never_mixed(reg):
    """A second reference embedded by another backend (other dimension) crashed np.mean in _update_centroid (HTTP 500) and
    left the new clip orphaned. It is now stored without an embedding."""
    reg.create("o", "asha", "Asha")
    _add(reg, n=1, emb=np.ones(4, np.float32), backend="mfcc")
    r2 = _add(reg, n=2, emb=np.ones(7, np.float32), backend="resemblyzer")
    sp = reg.get("asha", "o")
    assert r2.embedding_path is None and sp.embedding_backend == "mfcc" and len(sp.references) == 2
    assert np.load(reg.resolve_path("asha", sp.embedding_path)).shape == (4,)


def test_delete_failing_midway_can_be_retried_and_leaves_no_orphans(reg, monkeypatch):
    """speaker.json used to be unlinked first: a failure on a later file left unreachable raw audio behind a speaker that
    could no longer be found or deleted. The record now goes last, so the delete is retryable."""
    reg.create("o", "asha", "Asha")
    _add(reg, n=1, emb=np.ones(4, np.float32), backend="mfcc")
    _add(reg, n=2, emb=np.ones(4, np.float32), backend="mfcc")
    real, calls = sr.secure_unlink, {"n": 0}

    def flaky(path):
        calls["n"] += 1
        if calls["n"] == 2:
            raise PermissionError("disk hiccup")
        return real(path)

    monkeypatch.setattr(sr, "secure_unlink", flaky)
    with pytest.raises(PermissionError):
        reg.delete("asha", "o")
    assert reg.get("asha", "o").id == "asha"  # still reachable: retry possible
    monkeypatch.setattr(sr, "secure_unlink", real)
    reg.delete("asha", "o")
    assert not (reg.root / "asha").exists()


def test_revocation_is_durable_even_if_purge_fails_and_restart_finishes_it(reg, monkeypatch):
    """Purge ran before the revoked record was saved: a failure there left consent 'granted' with half the files gone."""
    reg.create("o", "asha", "Asha")
    reg.set_consent("asha", "o", sr.Consent(**GRANT))
    _add(reg, n=1, emb=np.ones(4, np.float32), backend="mfcc")
    real = sr.secure_unlink
    monkeypatch.setattr(sr, "secure_unlink", lambda p: (_ for _ in ()).throw(PermissionError("x")))
    with pytest.raises(PermissionError):
        reg.set_consent("asha", "o", sr.Consent(status="revoked"))
    monkeypatch.setattr(sr, "secure_unlink", real)
    sp = reg.get("asha", "o")
    assert sp.consent.status == "revoked" and sp.references  # revoked and blocked, purge pending
    with pytest.raises(sr.SpeakerForbidden):
        reg.check_use(sp, "tts")
    reg.purge_expired()  # what startup runs
    sp = reg.get("asha", "o")
    assert sp.references == [] and _files(reg) == ["speaker.json"]


# ---- stale synthesis cache --------------------------------------------------------------------------------------
class SlowEngine:
    supports_cloning = False
    max_workers = 1
    ready = True

    def __init__(self):
        self.started, self.release = threading.Event(), threading.Event()

    def load(self):
        pass

    def voices(self):
        return [{"voice_id": "v", "sample_rate": 24000}]

    def has_voice(self, v):
        return v == "v"

    def sample_rate(self, v):
        return 24000

    def synth(self, text, voice, speed, ref=None, ref_text=None):
        self.started.set()
        self.release.wait(5)
        return np.full(2400, 0.3, np.float32)


def test_synthesis_in_flight_during_clear_cache_is_not_cached():
    """clear_cache(voice) after a voice was replaced/deleted only dropped existing entries; a chunk still being synthesized
    from the OLD clip then landed in the cache and was served for the new voice."""
    e = SlowEngine()
    saved = (tts.engine, tts.scheduler, settings.cache_size)
    tts.engine, tts.scheduler, settings.cache_size = e, tts.Scheduler(1), 8
    tts._cache.clear()

    async def go():
        task = asyncio.create_task(tts.synthesize("नमस्ते।", "v", sample_rate=24000))
        await asyncio.to_thread(e.started.wait, 5)
        tts.clear_cache("v")  # the voice was just replaced
        e.release.set()
        await task

    try:
        asyncio.run(go())
        assert len(tts._cache) == 0
    finally:
        tts.engine, tts.scheduler, settings.cache_size = saved
        tts._cache.clear()


# ---- speaker encoder --------------------------------------------------------------------------------------------
def test_encoder_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(se, "CACHE_MAX", 5)
    enc = se.SpeakerEncoder("mfcc")
    for seed in range(12):
        t = np.arange(16000 * 2) / 16000
        w = (0.3 * np.sin(2 * np.pi * (150 + seed * 7) * t) + np.random.default_rng(seed).normal(0, 0.01, t.shape)).astype(np.float32)
        enc.embed(w, 16000)
    assert len(enc._cache) == 5


def test_similarity_kinds_are_labelled_precisely():
    assert se.SpeakerEncoder("mfcc").similarity_kind == "mfcc_statistics_cosine"
    assert se._ResemblyzerBackend.kind == se._SpeechbrainBackend.kind == "neural_embedding_cosine"
    assert "verification" not in se.SpeakerEncoder("mfcc").similarity_kind
    c = se.SpeakerEncoder("mfcc").consistency([np.ones(3), np.ones(3)])
    assert c["calibrated"] is False and c["kind"] == "mfcc_statistics_cosine"


# ---- qwen engine: atomic clip writes and prompt-cache epoch ---------------------------------------------------------
class FakeQwen:
    def __init__(self):
        self.prompts = 0

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode):
        self.prompts += 1
        return (ref_audio, ref_text)


@pytest.fixture()
def qwen(tmp_path, monkeypatch):
    from app.services.qwen_engine import QwenEngine

    monkeypatch.setattr(settings, "voices_dir", tmp_path / "voices")
    settings.voices_dir.mkdir()
    e = QwenEngine()
    e.model = FakeQwen()
    return e


def test_qwen_save_leaves_no_temp_files_and_replaces_atomically(qwen):
    qwen.save_voice("v1", np.zeros(16000, np.float32), 16000, "नमस्ते")
    qwen.save_voice("v1", np.ones(16000, np.float32) * 0.1, 16000, None)
    assert sorted(p.name for p in settings.voices_dir.iterdir()) == ["v1.wav"]  # transcript removed, no .tmp leftovers
    assert sf.read(settings.voices_dir / "v1.wav")[0].max() > 0.05


def test_qwen_prompt_built_from_old_clip_is_not_cached_after_replace(qwen, monkeypatch):
    """_voice_prompt read the clip and cached the prompt with no coordination with save/delete: a replace during prompt
    building left the OLD voice cached forever."""
    qwen.save_voice("v1", np.zeros(16000, np.float32), 16000, None)
    real = qwen._make_prompt

    def slow(audio, transcript):
        qwen.save_voice("v1", np.ones(16000, np.float32) * 0.1, 16000, None)  # replaced while the prompt is being built
        return real(audio, transcript)

    monkeypatch.setattr(qwen, "_make_prompt", slow)
    qwen._voice_prompt("v1")
    assert "v1" not in qwen._prompts
    monkeypatch.setattr(qwen, "_make_prompt", real)
    qwen._voice_prompt("v1")
    assert "v1" in qwen._prompts
    assert qwen.delete_voice("v1") and "v1" not in qwen._prompts and not list(settings.voices_dir.iterdir())
