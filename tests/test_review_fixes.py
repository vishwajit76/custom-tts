"""Regression tests for the adversarial review of the speaker registry / API / DSP / training tools."""
import asyncio
import http.client
import io
import json
import threading
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
import soundfile as sf

from app.core.config import settings
from app.services import dsp, tts
from app.services import speaker_registry as sr
from tests.test_speakers import A, B, GRANT, client, eng, reg, wav_bytes  # noqa: F401 (fixtures)
from training import annotate, data_rights, split


def test_compressed_audio_bomb_rejected_before_decode(monkeypatch):
    # 16 kHz mono silence-ish FLAC that decodes to far more than ref_max_seconds
    b = io.BytesIO()
    sf.write(b, np.zeros(16000 * 600, np.float32), 16000, format="FLAC")
    assert len(b.getvalue()) < settings.ref_max_bytes
    real = sf.read
    monkeypatch.setattr(sf, "read", lambda *a, **k: pytest.fail("decoded a too-long file"))
    with pytest.raises(sr.ReferenceRejected):
        sr.analyze_reference(b.getvalue())
    monkeypatch.setattr(sf, "read", real)


def test_add_reference_refused_when_revoked_at_registry_level(reg):
    reg.create("o", "s1", "S")
    reg.set_consent("s1", "o", sr.Consent(status="revoked"))
    raw = wav_bytes(dur=4.0)
    wav, rate, m = sr.analyze_reference(raw)
    with pytest.raises(sr.SpeakerForbidden):
        reg.add_reference("s1", "o", wav, rate, m)


def test_legacy_voice_api_cannot_touch_other_owners_speaker(client, reg, eng, monkeypatch):
    reg.create(sr.owner_id("ka"), "alice", "Alice")
    called = []
    monkeypatch.setattr(eng, "supports_cloning", True, raising=False)
    eng.save_voice = lambda *a, **k: called.append("save")
    eng.delete_voice = lambda v: called.append("delete") or True
    r = client.post("/v1/voices", files={"audio": ("a.wav", wav_bytes(dur=3.0), "audio/wav")}, data={"voice_id": "alice"}, headers=B)
    assert r.status_code == 409
    assert client.delete("/v1/voices/alice", headers=B).status_code == 409
    assert called == []


def test_dsp_runs_off_the_event_loop(monkeypatch):
    seen = {}

    def fake(wav, sr_, *a, **k):
        seen["thread"] = threading.current_thread()
        return wav

    monkeypatch.setattr(dsp, "apply_prosody", fake)
    from tests.test_conditioning import FakeEngine

    monkeypatch.setattr(tts, "engine", FakeEngine())
    monkeypatch.setattr(settings, "cache_size", 0)
    tts.scheduler = tts.Scheduler(2)

    async def go():
        async for _ in tts.stream("नमस्ते", "fake", sample_rate=8000, dsp_controls={"pitch": 2.0}):
            pass
        return threading.current_thread()

    main = asyncio.run(go())
    assert seen["thread"] is not main


def test_split_key_keeps_devanagari_marks():
    assert split._key("कि") != split._key("की") != split._key("क")
    assert split._key("नमस्ते, दुनिया!") == "नमस्ते दुनिया"


def test_rights_string_fields_rejected(tmp_path):
    base = {"rights_id": "r1", "source": "s", "licence": "l", "consent_record_id": "c", "speaker_authorization": True,
            "speaker_ids": ["a"], "permitted_uses": ["tts_training"], "vendor_generated": False}
    def load(**over):
        p = tmp_path / "r.jsonl"
        p.write_text(json.dumps({**base, **over}) + "\n")
        return data_rights.load_rights(p)
    assert load()
    for bad in ({"permitted_uses": "no_tts_training_here"}, {"speaker_ids": "abc"}, {"speaker_authorization": "yes"}):
        with pytest.raises(data_rights.RightsError):
            load(**bad)
    p = tmp_path / "d.jsonl"
    p.write_text((json.dumps(base) + "\n") * 2)
    with pytest.raises(data_rights.RightsError):
        data_rights.load_rights(p)


def test_annotate_rejects_cross_site_and_rebinding(tmp_path):
    m = tmp_path / "m.jsonl"
    m.write_text(json.dumps({"id": "x", "audio": "a.wav", "text": "t", "speaker_id": "s", "language": "hi"}) + "\n")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = srv.server_address[1]
    srv.RequestHandlerClass = annotate.make_handler(m, "me", port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    body = json.dumps({"id": "x", "emotion": "sad"})

    def post(headers):
        c = http.client.HTTPConnection("127.0.0.1", port)
        c.request("POST", "/api/update", body, headers)
        return c.getresponse().status

    try:
        assert post({"Content-Type": "text/plain"}) == 403  # simple cross-site form/fetch
        assert post({"Content-Type": "application/json", "Origin": "http://evil.example"}) == 403
        assert post({"Content-Type": "application/json", "Host": "evil.example"}) == 403  # DNS rebinding
        assert "human_verified" not in m.read_text()
        assert post({"Content-Type": "application/json"}) == 200
        c = http.client.HTTPConnection("127.0.0.1", port)
        c.request("GET", "/api/items", headers={"Host": "evil.example"})
        assert c.getresponse().status == 403
    finally:
        srv.shutdown()


# ---------- round 2 ----------
def _bound(reg, client, eng, monkeypatch, uses):
    monkeypatch.setattr(eng, "supports_cloning", True, raising=False)
    eng.has_voice = lambda v: v == "fake"
    name = type(eng).__name__.removesuffix("Engine").lower()
    reg.create(sr.owner_id("ka"), "bob", "Bob", engine_bindings={name: "fake"})
    reg.set_consent("bob", sr.owner_id("ka"), sr.Consent(status="granted", consent_record_id="c", granted_by="g", permitted_uses=uses))


def test_bound_cloned_voice_needs_cloning_use(client, reg, eng, monkeypatch):
    _bound(reg, client, eng, monkeypatch, ["tts"])
    body = {"input": "नमस्ते", "condition": {"speaker_id": "bob"}}
    assert client.post("/v1/audio/speech", json=body, headers=A).status_code == 403
    reg.set_consent("bob", sr.owner_id("ka"), sr.Consent(status="granted", consent_record_id="c", granted_by="g", permitted_uses=["tts", "cloning"]))
    assert client.post("/v1/audio/speech", json=body, headers=A).status_code == 200


def test_retention_expiry_purges_engine_copy_and_marks_not_clone_capable(client, reg, eng, monkeypatch):
    from datetime import datetime, timedelta, timezone
    deleted = []
    eng.delete_voice = lambda v: deleted.append(v) or True
    _bound(reg, client, eng, monkeypatch, ["tts", "cloning"])
    o = sr.owner_id("ka")
    reg.update("bob", o, retention=sr.Retention(delete_after_days=1))
    wav, rate, m = sr.analyze_reference(wav_bytes(dur=4.0))
    reg.add_reference("bob", o, wav, rate, m)
    sp = reg._load("bob")
    sp.references[0].created_at = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    reg._save(sp)
    sp = reg.get("bob", o)
    assert deleted == ["fake"] and sp.clone_capable is False
    r = client.post("/v1/audio/speech", json={"input": "नमस्ते", "condition": {"speaker_id": "bob"}}, headers=A)
    assert r.status_code == 422  # binding ignored, no reference left


def test_upload_body_cap_413(client, monkeypatch):
    monkeypatch.setattr(settings, "upload_max_bytes", 5000)
    reg_ = sr.get_registry()
    reg_.create(sr.owner_id("ka"), "big", "Big")
    r = client.post("/v1/speakers/big/references", files={"audio": ("a.wav", b"x" * 20000, "audio/wav")}, headers=A)
    assert r.status_code == 413

    def chunks():  # no Content-Length: counted while streaming
        for _ in range(10):
            yield b"x" * 2000
    r = client.post("/v1/speakers/big/references", content=chunks(), headers={**A, "Content-Type": "multipart/form-data; boundary=zz"})
    assert r.status_code == 413
    assert client.get("/v1/speakers", headers=A).status_code == 200


def test_legacy_decode_audio_header_check(monkeypatch):
    from app.services import audio_utils
    b = io.BytesIO()
    sf.write(b, np.zeros(16000 * 600, np.float32), 16000, format="FLAC")
    monkeypatch.setattr(sf, "read", lambda *a, **k: pytest.fail("decoded"))
    with pytest.raises(ValueError):
        audio_utils.decode_audio(b.getvalue())


def test_heldout_only_actual_artifacts(tmp_path):
    (tmp_path / "testimonials.jsonl").write_text('{"id":"1"}\n')
    split.assert_not_heldout(tmp_path / "testimonials.jsonl")
    (tmp_path / "test.jsonl").write_text('{"id":"1"}\n')
    with pytest.raises(split.HeldOutError):
        split.assert_not_heldout(tmp_path / "test.jsonl")
    (tmp_path / "x.jsonl").write_text('{"id":"1","heldout":true}\n')
    with pytest.raises(split.HeldOutError):
        split.assert_not_heldout(tmp_path / "x.jsonl")


def test_conflicting_rights_entries_raise():
    def e(rid, ok):
        return {"rights_id": rid, "source": "s", "licence": "l", "consent_record_id": "c", "speaker_authorization": ok,
                "speaker_ids": ["a"], "permitted_uses": ["tts_training"], "vendor_generated": False}
    row = {"id": "1", "speaker_id": "a"}
    with pytest.raises(data_rights.RightsError):
        data_rights.check_rows([row], [e("r1", True), e("r2", False)])
    assert data_rights.check_rows([row], [e("r1", True), e("r2", True)]) == []
    assert data_rights.check_rows([{**row, "rights_id": "r1"}], [e("r1", True), e("r2", False)]) == []
