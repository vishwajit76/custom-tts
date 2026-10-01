"""Speaker registry, reference validation, consent, ownership, speaker_id resolution, legacy clone endpoints."""
import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.api.speech import prepare_ex
from app.core.config import settings
from app.models.schemas import SpeechRequest
from app.services import speaker_registry as sr
from app.services import tts
from app.services.conditioning import EngineCapabilities, VoiceCondition, validate_condition
from tests.test_conditioning import FakeEngine
from tests.test_speaker_encoder import voice

A, B = {"Authorization": "Bearer ka"}, {"Authorization": "Bearer kb"}


def wav_bytes(f0=120, dur=4.0, sr_=16000, seed=0, amp=None):
    w, _ = voice(f0, dur, sr_, seed)
    if amp:
        w = np.clip(w * amp, -1, 1)
    b = io.BytesIO()
    sf.write(b, w, sr_, format="WAV")
    return b.getvalue()


@pytest.fixture()
def reg(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "speakers_dir", tmp_path / "spk")
    return sr.get_registry()


@pytest.fixture()
def eng(monkeypatch, reg):
    e = FakeEngine()
    monkeypatch.setattr(tts, "engine", e)
    monkeypatch.setattr(settings, "default_voice", "fake")
    monkeypatch.setattr(settings, "api_keys", "ka,kb")
    monkeypatch.setattr(settings, "cache_size", 0)
    return e


@pytest.fixture()
def client(eng):
    from app.main import app

    with TestClient(app) as c:
        yield c


GRANT = {"status": "granted", "consent_record_id": "c-1", "granted_by": "Asha", "permitted_uses": ["tts", "cloning"]}


# ---------- registry ----------
@pytest.mark.parametrize("bad", ["../x", "a/b", "", ".hidden", "a" * 65, "x y", "..", "a\\b", "a\x00b"])
def test_invalid_ids(reg, bad):
    with pytest.raises(sr.SpeakerError):
        reg.create("o", bad, "n")


def test_owner_id_stable_and_opaque():
    assert sr.owner_id("secret") == sr.owner_id("secret") != sr.owner_id("other")
    assert "secret" not in sr.owner_id("secret") and sr.owner_id("") == "anonymous"


def test_create_persist_and_ownership(reg):
    reg.create("o1", "asha", "Asha", ["hi"], gender="f")
    with pytest.raises(sr.SpeakerExists):
        reg.create("o1", "asha", "again")
    assert sr.SpeakerRegistry(reg.root).get("asha", "o1").display_name == "Asha"  # reloaded from disk
    for op in (lambda: reg.get("asha", "o2"), lambda: reg.update("asha", "o2", display_name="x"), lambda: reg.delete("asha", "o2"),
               lambda: reg.set_consent("asha", "o2", sr.Consent(status="revoked"))):
        with pytest.raises(sr.SpeakerForbidden):
            op()
    assert reg.list_for("o2") == [] and len(reg.list_for("o1")) == 1


def test_path_confinement(reg):
    reg.create("o", "s", "S")
    for rel in ("../../etc/passwd", "/etc/passwd", "refs/../../x"):
        with pytest.raises(sr.SpeakerError):
            reg.resolve_path("s", rel)


def test_consent_rules(reg):
    sp = reg.create("o", "s", "S")
    with pytest.raises(sr.SpeakerForbidden):
        reg.check_use(sp)  # pending
    with pytest.raises(sr.SpeakerError):
        reg.set_consent("s", "o", sr.Consent(status="granted"))  # incomplete
    sp = reg.set_consent("s", "o", sr.Consent(**GRANT))
    reg.check_use(sp, "tts")
    with pytest.raises(sr.SpeakerForbidden):
        reg.check_use(sp, "training")


def test_reference_validation():
    for raw, ctype, msg in [(b"", None, "empty"), (b"notaudio" * 100, None, "decode"), (wav_bytes(), "text/plain", "content-type"),
                            (wav_bytes(dur=1.0), None, "short"), (wav_bytes(dur=40), None, "long"),
                            (wav_bytes(sr_=8000), None, "sample rate"), (wav_bytes(amp=20), None, "clipped")]:
        with pytest.raises(sr.ReferenceRejected, match=msg):
            sr.analyze_reference(raw, ctype)
    quiet = io.BytesIO()
    sf.write(quiet, np.zeros(64000, np.float32), 16000, format="WAV")
    with pytest.raises(sr.ReferenceRejected, match="quiet|silen"):
        sr.analyze_reference(quiet.getvalue())
    w, rate, m = sr.analyze_reference(wav_bytes(), "audio/wav")
    assert rate == 16000 and 0 < m["quality"] <= 1 and len(m["sha256"]) == 64


def test_size_limit(monkeypatch):
    monkeypatch.setattr(settings, "ref_max_bytes", 1000)
    with pytest.raises(sr.ReferenceRejected, match="large"):
        sr.analyze_reference(wav_bytes())


def test_secure_delete_removes_everything(reg):
    reg.create("o", "s", "S")
    w, rate, m = sr.analyze_reference(wav_bytes())
    ref = reg.add_reference("s", "o", w, rate, m, embedding=np.ones(4, np.float32) / 2, embedding_backend="mfcc")
    files = [p for p in (reg.root / "s").rglob("*") if p.is_file()]
    assert len(files) >= 4
    reg.delete("s", "o")
    assert not (reg.root / "s").exists() and ref.path


def test_secure_unlink_overwrites(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"A" * 100)
    fd = open(p, "rb")  # keep the inode alive to inspect it
    sr.secure_unlink(p)
    assert not p.exists() and fd.read() != b"A" * 100
    fd.close()


def test_revoke_purges_and_retention(reg, monkeypatch):
    reg.create("o", "s", "S", retention=sr.Retention(keep_raw_reference=False))
    w, rate, m = sr.analyze_reference(wav_bytes())
    ref = reg.add_reference("s", "o", w, rate, m)
    assert ref.path is None and not list((reg.root / "s" / "refs").glob("*.wav"))  # raw not retained
    reg.create("o", "t", "T", retention=sr.Retention(delete_after_days=1))
    ref = reg.add_reference("t", "o", w, rate, m)
    p = reg.resolve_path("t", ref.path)
    assert p.exists()
    sp = reg._load("t")
    sp.references[0].created_at = "2000-01-01T00:00:00+00:00"
    reg._save(sp)
    assert reg.purge_expired() == 1 and not p.exists() and reg.get("t", "o").references[0].path is None
    reg.create("o", "u", "U")
    reg.add_reference("u", "o", w, rate, m)
    reg.set_consent("u", "o", sr.Consent(status="revoked"))
    assert reg.get("u", "o").references == [] and not list((reg.root / "u" / "refs").glob("*"))


def test_best_reference_and_duplicates(reg):
    reg.create("o", "s", "S")
    for f0, dur in ((120, 3.2), (125, 8.0)):
        w, rate, m = sr.analyze_reference(wav_bytes(f0, dur))
        reg.add_reference("s", "o", w, rate, m)
    sp = reg.get("s", "o")
    assert reg.best_reference(sp).duration_s == pytest.approx(8.0, abs=0.01)
    with pytest.raises(sr.SpeakerExists):
        reg.add_reference("s", "o", w, rate, m)


# ---------- REST ----------
def mk(client, headers=A, sid="asha", **extra):
    return client.post("/v1/speakers", json={"id": sid, "display_name": "Asha", "languages": ["hi"], **extra}, headers=headers)


def upload(client, sid="asha", headers=A, raw=None, ctype="audio/wav"):
    return client.post(f"/v1/speakers/{sid}/references", files={"audio": ("r.wav", raw or wav_bytes(), ctype)}, data={"transcript": "नमस्ते"}, headers=headers)


def test_rest_lifecycle_and_ownership(client):
    assert mk(client).status_code == 201 and mk(client).status_code == 409
    assert mk(client, sid="../x").status_code in (400, 404, 422)
    assert client.get("/v1/speakers", headers=B).json()["data"] == []
    assert client.get("/v1/speakers/asha", headers=B).status_code == 404  # no existence leak
    assert client.delete("/v1/speakers/asha", headers=B).status_code == 404
    assert client.patch("/v1/speakers/asha", json={"display_name": "x"}, headers=B).status_code == 404
    assert upload(client, headers=B).status_code == 404
    r = upload(client)
    body = r.json()
    assert r.status_code == 201 and body["raw_retained"] and body["encoder"] == {"name": "mfcc", "neural": False, "similarity_kind": "mfcc_statistics_cosine"} and "path" not in body
    r2 = upload(client, raw=wav_bytes(125, 5.0, seed=3)).json()
    assert r2["similarity_to_existing"] is not None and any("uncalibrated" in w for w in r2["warnings"])
    sp = client.get("/v1/speakers/asha", headers=A).json()
    assert "owner" not in sp and len(sp["references"]) == 2 and sp["embedding_path"] == "embedding.npy"
    assert client.delete(f"/v1/speakers/asha/references/{body['ref_id']}", headers=A).status_code == 200
    assert client.get("/v1/speakers/asha", headers=A).json()["references"][0]["ref_id"] == r2["ref_id"]
    assert client.delete("/v1/speakers/asha", headers=A).json() == {"deleted": "asha"}
    assert client.get("/v1/speakers/asha", headers=A).status_code == 404


def test_rest_upload_rejections(client):
    mk(client)
    assert upload(client, raw=b"junk").status_code == 422
    assert upload(client, raw=wav_bytes(dur=1)).status_code == 422
    assert upload(client, raw=wav_bytes(), ctype="text/plain").status_code == 422
    assert upload(client, raw=wav_bytes(amp=30)).status_code == 422
    assert client.get("/v1/speakers/asha", headers=A).json()["references"] == []


def test_rest_consent_and_bindings(client):
    mk(client)
    assert client.patch("/v1/speakers/asha", json={"engine_bindings": {"fake": "nope"}}, headers=A).status_code == 422
    assert client.patch("/v1/speakers/asha", json={"engine_bindings": {"fake": "fake"}}, headers=A).json()["engine_bindings"] == {"fake": "fake"}
    assert client.put("/v1/speakers/asha/consent", json={"status": "granted"}, headers=A).status_code == 400
    assert client.put("/v1/speakers/asha/consent", json=GRANT, headers=A).json()["consent"]["status"] == "granted"


# ---------- synthesis via speaker_id ----------
def speak(client, cond, headers=A, **kw):
    return client.post("/v1/audio/speech", json={"input": "नमस्ते।", "condition": cond, **kw}, headers=headers)


def test_speaker_id_requires_consent_and_use(client, eng):
    mk(client)
    client.patch("/v1/speakers/asha", json={"engine_bindings": {"fake": "fake"}}, headers=A)
    assert speak(client, {"speaker_id": "asha"}).status_code == 403  # pending
    client.put("/v1/speakers/asha/consent", json={**GRANT, "permitted_uses": ["evaluation"]}, headers=A)
    assert speak(client, {"speaker_id": "asha"}).status_code == 403  # tts not permitted
    client.put("/v1/speakers/asha/consent", json=GRANT, headers=A)
    r = speak(client, {"speaker_id": "asha"})
    assert r.status_code == 200 and r.headers["x-tts-applied-controls"] == "speaker_id" and eng.calls
    client.put("/v1/speakers/asha/consent", json={"status": "revoked"}, headers=A)
    assert speak(client, {"speaker_id": "asha"}).status_code == 403
    assert speak(client, {"speaker_id": "asha"}, headers=B).status_code == 404
    assert speak(client, {"speaker_id": "ghost"}).status_code == 404


def test_speaker_without_binding(client, eng):
    mk(client, consent=GRANT)
    r = speak(client, {"speaker_id": "asha"})
    assert r.status_code == 422 and r.json()["detail"]["unsupported"] == ["speaker_id"]
    r = speak(client, {"speaker_id": "asha", "fallback": "ignore"})
    assert r.status_code == 200 and r.headers["x-tts-ignored-controls"] == "speaker_id" and r.headers["x-tts-applied-controls"] == ""


def test_speaker_reference_used_for_cloning_engine(client, monkeypatch):
    seen = {}

    class Cloner(FakeEngine):
        supports_cloning = True
        capabilities = EngineCapabilities(cloning=True, speed=True, streaming="sentence")

        def synth(self, text, voice, speed, ref=None, ref_text=None):
            seen.update(ref=ref, ref_text=ref_text)
            return super().synth(text, voice, speed, ref, ref_text)

    monkeypatch.setattr(tts, "engine", Cloner())
    mk(client, consent=GRANT)
    upload(client)
    r = speak(client, {"speaker_id": "asha"})
    assert r.status_code == 200 and seen["ref"][1] == 16000 and seen["ref_text"] == "नमस्ते"
    client.put("/v1/speakers/asha/consent", json={**GRANT, "permitted_uses": ["tts"]}, headers=A)
    assert speak(client, {"speaker_id": "asha"}).status_code == 403  # cloning use not permitted


def test_speaker_id_with_reference_audio_is_400(client):
    mk(client, consent=GRANT)
    assert speak(client, {"speaker_id": "asha", "reference_audio": "AAAA"}).status_code == 400


def test_validate_condition_speaker_gate():
    caps = EngineCapabilities(speed=True)
    with pytest.raises(Exception):
        validate_condition(VoiceCondition(speaker_id="s"), caps)
    _, applied, _ = validate_condition(VoiceCondition(speaker_id="s"), caps, speaker_ok=True)
    assert applied == ["speaker_id"]


def test_prepare_ex_default_owner_without_auth(monkeypatch, reg):
    monkeypatch.setattr(tts, "engine", FakeEngine())
    monkeypatch.setattr(settings, "default_voice", "fake")
    reg.create(sr.owner_id(""), "s", "S", engine_bindings={"fake": "fake"})
    reg.set_consent("s", sr.owner_id(""), sr.Consent(**GRANT))
    kw, applied, _ = prepare_ex(SpeechRequest(input="x", condition=VoiceCondition(speaker_id="s")))
    assert kw["voice"] == "fake" and applied == ["speaker_id"]


# ---------- legacy clone endpoints ----------
class LegacyCloner(FakeEngine):
    supports_cloning = True
    capabilities = EngineCapabilities(cloning=True, speed=True, streaming="sentence")

    def __init__(self):
        super().__init__()
        self.saved = {}

    def voices(self):
        return [{"voice_id": v, "sample_rate": SR_} for v in self.saved] + [{"voice_id": "fake", "sample_rate": SR_}]

    def has_voice(self, v):
        return v == "fake" or v in self.saved

    def save_voice(self, vid, wav, rate, transcript):
        self.saved[vid] = transcript

    def delete_voice(self, vid):
        return self.saved.pop(vid, "x") != "x" if vid in self.saved else False


SR_ = 22050


def test_legacy_endpoints_keep_contract_and_mirror(client, monkeypatch, reg):
    monkeypatch.setattr(tts, "engine", LegacyCloner())
    r = client.post("/v1/voices", files={"audio": ("a.wav", wav_bytes(dur=3.0), "audio/wav")}, data={"voice_id": "myvoice", "transcript": "hi"}, headers=A)
    assert r.status_code == 200 and r.json() == {"voice_id": "myvoice", "has_transcript": True}
    assert client.post("/v1/voices", files={"audio": ("a.wav", b"junk")}, data={"voice_id": "bad"}, headers=A).status_code == 400
    sp = client.get("/v1/speakers/myvoice", headers=A).json()  # mirrored, consent pending
    assert sp["consent"]["status"] == "pending" and sp["engine_bindings"] == {"legacycloner": "myvoice"}
    assert speak(client, {"speaker_id": "myvoice"}).status_code == 403
    assert client.post("/v1/audio/speech", json={"input": "x", "voice": "myvoice"}, headers=A).status_code == 200  # legacy path unchanged
    assert client.delete("/v1/voices/myvoice", headers=A).json() == {"deleted": "myvoice"}
    assert client.get("/v1/speakers/myvoice", headers=A).status_code == 404
    assert client.delete("/v1/voices/myvoice", headers=A).status_code == 404


def test_legacy_upload_with_consent_fields(client, monkeypatch):
    monkeypatch.setattr(tts, "engine", LegacyCloner())
    client.post("/v1/voices", files={"audio": ("a.wav", wav_bytes(dur=3.0), "audio/wav")},
                data={"voice_id": "v2", "consent_record_id": "c9", "granted_by": "Ravi"}, headers=A)
    assert client.get("/v1/speakers/v2", headers=A).json()["consent"]["status"] == "granted"
    assert speak(client, {"speaker_id": "v2"}).status_code == 200


def test_registry_json_is_valid(reg):
    reg.create("o", "s", "S")
    d = json.loads((reg.root / "s" / "speaker.json").read_text())
    assert d["owner"] == "o" and d["consent"]["status"] == "pending"
