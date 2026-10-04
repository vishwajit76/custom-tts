"""Voice catalog (voices/catalog.json): validation, planned/missing handling, legacy ids, multi-speaker Piper serving.

The multi-speaker model here is a tiny ONNX graph built with the `onnx` helper API. It takes Piper's inputs (input,
input_lengths, scales, sid) and outputs a constant 0.1 s tone whose amplitude is 0.1 * (sid + 1), so a test can read
back which speaker the engine really ran.
"""
import json
import logging
from pathlib import Path

import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper, numpy_helper
from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP

from app.core.config import settings
from app.services import piper_engine, tts, voice_catalog
from app.services.conditioning import EngineCapabilities

SR = 22050
SHIPPED = Path(__file__).resolve().parent.parent / "voices" / "catalog.json"
SPEAKERS = {"f_neutral": 0, "f_happy": 1, "m_neutral": 2}


# ---------- helpers ----------
def tiny_model(directory: Path, stem: str, speakers: dict[str, int] | None = None) -> None:
    """<stem>.onnx + <stem>.onnx.json. Output = constant tone, amplitude 0.1*(sid+1) (0.5 for a single-speaker model)."""
    inputs = [helper.make_tensor_value_info("input", TensorProto.INT64, [1, "n"]),
              helper.make_tensor_value_info("input_lengths", TensorProto.INT64, [1]),
              helper.make_tensor_value_info("scales", TensorProto.FLOAT, [3])]
    nodes, inits = [], [numpy_helper.from_array(np.ones((1, SR // 10), np.float32), "ones"),
                        numpy_helper.from_array(np.array([0.1], np.float32), "tenth")]
    if speakers:
        inputs.append(helper.make_tensor_value_info("sid", TensorProto.INT64, [1]))
        inits.append(numpy_helper.from_array(np.array([0.1], np.float32), "base"))
        nodes += [helper.make_node("Cast", ["sid"], ["sid_f"], to=TensorProto.FLOAT),
                  helper.make_node("Mul", ["sid_f", "tenth"], ["scaled"]),
                  helper.make_node("Add", ["scaled", "base"], ["amp"])]
    else:
        inits.append(numpy_helper.from_array(np.array([0.5], np.float32), "amp"))
    nodes.append(helper.make_node("Mul", ["ones", "amp"], ["output"]))
    graph = helper.make_graph(nodes, "tiny", inputs, [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, SR // 10])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.onnx").write_bytes(model.SerializeToString())
    cfg = {"audio": {"sample_rate": SR}, "espeak": {"voice": "hi"}, "phoneme_type": "espeak", "num_symbols": 256,
           "num_speakers": len(speakers) if speakers else 1, "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
           "phoneme_id_map": DEFAULT_PHONEME_ID_MAP}
    if speakers:
        cfg["speaker_id_map"] = speakers
    (directory / f"{stem}.onnx.json").write_text(json.dumps(cfg), "utf-8")


def entry(voice_id, **kw):
    return {"voice_id": voice_id, "engine": "piper", "model": "hi_IN-v7-medium", "status": "experimental", **kw}


V7 = [
    entry("hi-IN-young-female", speaker="f_neutral", default_style="neutral", gender="F", age_group="young_adult",
          age_evidence="listener screening, 3 native listeners, 2026-10", style="conversational, warm", family="piper-v7", language="hi",
          styles={"neutral": "f_neutral", "happy": "f_happy"}, aliases=["young-female"]),
    entry("hi-IN-young-male", speaker="m_neutral", gender="M", family="piper-v7"),
]


def write_catalog(tmp_path: Path, voices: list[dict]) -> Path:
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps({"version": 1, "voices": voices}), "utf-8")
    return p


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Empty models dir + a catalog path the test fills in; the shipped catalog is not involved."""
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(settings, "models_dir", models)
    monkeypatch.setattr(settings, "models_extra", "")
    monkeypatch.setattr(settings, "voice_catalog", tmp_path / "catalog.json")
    monkeypatch.setattr(settings, "api_keys", "")
    monkeypatch.setattr(settings, "cache_size", 0)
    monkeypatch.setattr(settings, "default_voice", "hi-IN-young-female")
    return models


def load_engine() -> piper_engine.PiperEngine:
    e = piper_engine.PiperEngine()
    e.load()
    return e


def amp(wav: np.ndarray) -> float:
    return round(float(np.median(wav)), 3)


# ---------- catalog validation ----------
def load(tmp_path, voices):
    return voice_catalog._load(str(write_catalog(tmp_path, voices)))


@pytest.mark.parametrize("bad", [
    {"gender": "X"}, {"gender": "female"}, {"age_group": "teen", "age_evidence": "x"}, {"status": "beta"}, {"unknown_field": 1},
    {"age_group": "adult"}, {"age_group": "adult", "age_evidence": "  "},  # needs evidence
    {"styles": {"furious": "f_happy"}, "speaker": "f_neutral"},  # not a value a request can carry
    {"styles": {"happy": "f_happy"}},  # no default speaker
    {"styles": {"happy": "f_happy"}, "default_style": "calm"}, {"speaker": "a", "styles": {"happy": "b"}, "default_style": "happy"},
    {"model": None, "speaker": "a"}, {"aliases": ["has space"]}, {"aliases": ["v"]}, {"voice_id": "../evil"},
])
def test_bad_entries_are_rejected(tmp_path, bad):
    with pytest.raises(voice_catalog.CatalogError):
        load(tmp_path, [{**entry("v"), **bad}])


def test_piper_entry_needs_a_model_and_ids_are_unique(tmp_path):
    with pytest.raises(voice_catalog.CatalogError):
        load(tmp_path, [{"voice_id": "v", "engine": "piper", "status": "production"}])
    with pytest.raises(voice_catalog.CatalogError, match="duplicate"):
        load(tmp_path, [entry("a"), entry("b", aliases=["a"])])
    with pytest.raises(voice_catalog.CatalogError, match="duplicate"):
        load(tmp_path, [entry("a"), entry("a")])


@pytest.mark.parametrize("doc", ["not json", "[]", '{"voices": {}}', '{"voices": [], "extra": 1}'])
def test_malformed_files_are_rejected(tmp_path, doc):
    p = tmp_path / "c.json"
    p.write_text(doc)
    with pytest.raises(voice_catalog.CatalogError):
        voice_catalog._load(str(p))


def test_age_group_with_evidence_is_accepted_and_defaults_are_unspecified(tmp_path):
    c = load(tmp_path, V7)
    v = c.get("hi-IN-young-female")
    assert v.age_group == "young_adult" and "screening" in v.age_evidence
    assert c.get("hi-IN-young-male").age_group == "unspecified" and c.get("hi-IN-young-male").age_evidence is None


def test_missing_catalog_file_is_empty_not_fatal(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        c = voice_catalog._load(str(tmp_path / "nope.json"))
    assert c.voices == () and "voice catalog not found" in caplog.text


def test_shipped_catalog_is_valid_and_has_evidence_for_every_age_label():
    c = voice_catalog._load(str(SHIPPED))
    assert len(c.voices) == len({v.voice_id for v in c.voices})
    for v in c.voices:
        assert v.age_group == "unspecified" or v.age_evidence
        if v.status == "planned":
            assert v.age_group == "unspecified"  # no age claim before a trained voice and its evidence exist
    assert {v.voice_id for v in c.voices if v.status == "planned"} == {"hi-IN-young-female", "hi-IN-young-male"}


# ---------- legacy ids unchanged ----------
OLD_GENDER = {"rohan": "M", "pratham": "M", "priyamvada": "F", "custom": "F"}  # piper_engine.GENDER before the catalog


def test_legacy_piper_genders_and_names_pinned(env, monkeypatch):
    monkeypatch.setattr(settings, "voice_catalog", SHIPPED)
    for who in (*OLD_GENDER, "other"):
        tiny_model(env, f"hi_IN-{who}-medium")
    e = load_engine()
    got = {v["voice_id"]: v for v in e.voices()}
    assert list(got) == [f"hi_IN-{w}-medium" for w in sorted((*OLD_GENDER, "other"))]
    for who, g in OLD_GENDER.items():
        v = got[f"hi_IN-{who}-medium"]
        assert v["gender"] == g and v["name"] == f"Piper {who.title()}" and v["engine"] == "piper" and v["speaker_id"] is None
        assert v["language"] == "hi" and v["sample_rate"] == SR
    assert got["hi_IN-other-medium"]["gender"] is None and got["hi_IN-other-medium"]["name"] == "Piper Other"  # not catalogued: nothing claimed
    assert amp(e.synth("नमस्ते।", "hi_IN-rohan-medium", 1.0)) == 0.5
    assert all(e.capabilities_for(v) is e.capabilities for v in got)  # unchanged capabilities


def test_legacy_ids_still_resolve(env, monkeypatch):
    monkeypatch.setattr(settings, "voice_catalog", SHIPPED)
    monkeypatch.setattr(settings, "default_voice", "hi_IN-rohan-medium")
    tiny_model(env, "hi_IN-rohan-medium")
    tiny_model(env, "hi_IN-custom-medium")
    monkeypatch.setattr(tts, "engine", load_engine())
    assert tts.resolve_voice("default") == "hi_IN-rohan-medium"
    assert tts.resolve_voice("hi_IN-custom-medium") == "hi_IN-custom-medium"
    with pytest.raises(KeyError):
        tts.resolve_voice("hi_IN-pratham-medium")  # catalogued but its model is absent: not routable


def test_shipped_catalog_pins_kokoro_and_supertonic_genders():
    """Both engines used to read the gender off the id (kokoro:hf_alpha -> F, supertonic:M2 -> M); same values, now declared."""
    for n, g in (("hf_alpha", "F"), ("hf_beta", "F"), ("hm_omega", "M"), ("hm_psi", "M")):
        assert voice_catalog._load(str(SHIPPED)).get(f"kokoro:{n}").gender == g == n[1].upper()
    for g in "MF":
        for i in range(1, 6):
            assert voice_catalog._load(str(SHIPPED)).get(f"supertonic:{g}{i}").gender == g


def test_gender_is_never_inferred_from_the_id(env, tmp_path):
    write_catalog(tmp_path, [entry("hi-IN-young-female"), entry("female-voice-f", model="hi_IN-v7-medium")])
    assert voice_catalog.gender_of("hi-IN-young-female") is None and voice_catalog.gender_of("female-voice-f") is None
    assert voice_catalog.gender_of("not-in-catalog") is None
    assert voice_catalog.metadata("hi-IN-young-female")["gender"] is None
    tiny_model(env, "hi_IN-v7-medium")
    e = load_engine()
    assert {v["voice_id"]: v["gender"] for v in e.voices()}["hi-IN-young-female"] is None


# ---------- planned / missing ----------
def test_planned_voice_is_hidden_and_not_routable_even_with_its_model(env, tmp_path, monkeypatch):
    write_catalog(tmp_path, [*V7[:1], entry("hi-IN-young-male", speaker="m_neutral", status="planned", aliases=["young-male"])])
    tiny_model(env, "hi_IN-v7-medium", SPEAKERS)
    e = load_engine()
    monkeypatch.setattr(tts, "engine", e)
    assert [v["voice_id"] for v in e.voices()] == ["hi-IN-young-female"]
    assert not e.has_voice("hi-IN-young-male")
    for name in ("hi-IN-young-male", "young-male"):
        with pytest.raises(KeyError):
            tts.resolve_voice(name)


def test_entry_whose_model_is_absent_is_skipped_with_a_warning(env, tmp_path, caplog):
    write_catalog(tmp_path, [*V7, entry("hi-IN-adult-female", model="hi_IN-v8-medium", speaker="x"),
                             entry("hi-IN-planned", model="hi_IN-v9-medium", status="planned")])
    tiny_model(env, "hi_IN-v7-medium", SPEAKERS)
    with caplog.at_level(logging.WARNING):
        e = load_engine()
    assert {v["voice_id"] for v in e.voices()} == {"hi-IN-young-female", "hi-IN-young-male"}
    skipped = [r for r in caplog.records if "model file absent" in r.getMessage()]
    assert len(skipped) == 1 and skipped[0].extra_fields["voices"] == ["hi-IN-adult-female"]  # planned ones are not reported


def test_only_absent_models_in_the_catalog_still_fails_loudly(env, tmp_path):
    write_catalog(tmp_path, V7)
    with pytest.raises(RuntimeError, match="no Piper voices"):
        load_engine()


def test_entry_that_does_not_fit_its_model_is_skipped(env, tmp_path, caplog):
    write_catalog(tmp_path, [entry("a", speaker="nobody"), entry("b", model="hi_IN-single-medium", speaker="f_neutral"),
                             entry("c", speaker="m_neutral")])
    tiny_model(env, "hi_IN-v7-medium", SPEAKERS)
    tiny_model(env, "hi_IN-single-medium")
    with caplog.at_level(logging.WARNING):
        e = load_engine()
    assert sorted(v["voice_id"] for v in e.voices()) == ["c", "hi_IN-single-medium"]  # the plain stem voice of a single-speaker model stays
    assert "not in the model's speaker_id_map" in caplog.text and "model is single-speaker" in caplog.text


# ---------- multi-speaker serving ----------
def test_voice_id_selects_the_speaker(env, tmp_path):
    write_catalog(tmp_path, V7)
    tiny_model(env, "hi_IN-v7-medium", SPEAKERS)
    tiny_model(env, "hi_IN-rohan-medium")  # a single-speaker model in the same server
    e = load_engine()
    assert amp(e.synth("नमस्ते।", "hi-IN-young-female", 1.0)) == 0.1  # sid 0
    assert amp(e.synth("नमस्ते।", "hi-IN-young-male", 1.0)) == 0.3  # sid 2
    assert amp(e.synth("नमस्ते।", "hi_IN-rohan-medium", 1.0)) == 0.5  # no sid input at all
    got = {v["voice_id"]: v for v in e.voices()}
    assert got["hi-IN-young-female"]["speaker_id"] == 0 and got["hi-IN-young-male"]["speaker_id"] == 2
    assert not any(":" in v for v in got)  # a catalogued multi-speaker model exposes only its catalog voices


def test_uncatalogued_multi_speaker_model_keeps_stem_speaker_ids(env, tmp_path):
    write_catalog(tmp_path, [])
    tiny_model(env, "hi_IN-v7-medium", SPEAKERS)
    e = load_engine()
    assert sorted(v["voice_id"] for v in e.voices()) == ["hi_IN-v7-medium:f_happy", "hi_IN-v7-medium:f_neutral", "hi_IN-v7-medium:m_neutral"]
    assert amp(e.synth("नमस्ते।", "hi_IN-v7-medium:f_happy", 1.0)) == 0.2


def test_style_selects_the_recorded_speaker_of_the_same_voice(env, tmp_path):
    write_catalog(tmp_path, V7)
    tiny_model(env, "hi_IN-v7-medium", SPEAKERS)
    e = load_engine()
    assert amp(e.synth_native("नमस्ते।", "hi-IN-young-female", 1.0, {"emotion": "happy"})) == 0.2
    assert amp(e.synth_native("नमस्ते।", "hi-IN-young-female", 1.0, {"emotion": "neutral"})) == 0.1
    assert amp(e.synth_native("नमस्ते।", "hi-IN-young-female", 1.0, {})) == 0.1  # default
    assert amp(e.synth_native("नमस्ते।", "hi-IN-young-male", 1.0, {"emotion": "happy"})) == 0.3  # that voice has no styles: default speaker


def test_capabilities_name_only_the_recorded_styles(env, tmp_path):
    write_catalog(tmp_path, [*V7[:1], entry("hi-IN-young-male", speaker="m_neutral", styles={"conversational": "m_neutral", "warm": "m_neutral"})])
    tiny_model(env, "hi_IN-v7-medium", SPEAKERS)
    e = load_engine()
    f = e.capabilities_for("hi-IN-young-female")
    assert f.native_emotion and f.emotion_values == ("neutral", "happy") and not f.native_style and f.style_values == ()
    assert f.discrete_styles and f.speed and not (f.pitch or f.energy or f.prompt_emotion or f.cloning)
    m = e.capabilities_for("hi-IN-young-male")
    assert m.native_style and m.style_values == ("conversational", "warm") and not m.native_emotion
    assert e.capabilities.native_emotion is False and e.capabilities_for("hi-IN-young-male") is not e.capabilities


def test_style_voices_work_behind_a_multi_engine(env, tmp_path, monkeypatch):
    class Other:
        supports_cloning, max_workers, ready, capabilities = False, 1, True, EngineCapabilities(speed=True)
        load = lambda self: None  # noqa: E731
        voices = lambda self: [{"voice_id": "other:v", "sample_rate": SR}]  # noqa: E731
        has_voice = lambda self, v: v == "other:v"  # noqa: E731

    write_catalog(tmp_path, V7)
    tiny_model(env, "hi_IN-v7-medium", SPEAKERS)
    multi = tts.MultiEngine([piper_engine.PiperEngine(), Other()])
    multi.load()
    monkeypatch.setattr(tts, "engine", multi)
    assert tts.capabilities_for("hi-IN-young-female").emotion_values == ("neutral", "happy")
    assert tts.capabilities_for("other:v") is Other.capabilities
    assert amp(multi.synth_native("x", "hi-IN-young-female", 1.0, {"emotion": "happy"})) == 0.2


# ---------- API ----------
@pytest.fixture()
def client(env, tmp_path, monkeypatch):
    write_catalog(tmp_path, [*V7, entry("hi-IN-adult-female", speaker="f_neutral", status="planned")])
    tiny_model(env, "hi_IN-v7-medium", SPEAKERS)
    tts._cache.clear()
    monkeypatch.setattr(tts, "engine", piper_engine.PiperEngine())
    from app.main import app
    with TestClient(app) as c:  # the lifespan loads the engine
        yield c


def pcm_amp(r) -> float:
    return amp(np.frombuffer(r.content, "<i2").astype(np.float32) / 32767)


def speak(client, voice="hi-IN-young-female", **condition):
    body = {"input": "नमस्ते।", "voice": voice, "response_format": "pcm", "sample_rate": SR}
    if condition:
        body["condition"] = condition
    return client.post("/v1/audio/speech/stream", json=body)


def test_voices_endpoint_adds_metadata_and_hides_planned(client):
    data = {v["voice_id"]: v for v in client.get("/v1/voices").json()["data"]}
    assert sorted(data) == ["hi-IN-young-female", "hi-IN-young-male"]  # the planned adult voice is not listed
    f = data["hi-IN-young-female"]
    assert (f["gender"], f["age_group"], f["style"], f["family"], f["status"], f["styles"], f["language"]) == (
        "F", "young_adult", "conversational, warm", "piper-v7", "experimental", ["neutral", "happy"], "hi")  # language: the engine's own value wins
    assert f["engine"] == "piper" and f["sample_rate"] == SR and f["name"] == "Piper hi-IN-young-female" and f["speaker_id"] == 0  # old fields intact
    assert f["capabilities"]["emotion_values"] == ["neutral", "happy"] and f["capabilities"]["native_emotion"] is True
    m = data["hi-IN-young-male"]
    assert (m["gender"], m["age_group"], m["styles"], m["status"]) == ("M", "unspecified", [], "experimental")
    assert m["capabilities"]["native_emotion"] is False and m["capabilities"]["emotion_values"] == []
    assert client.get("/v1/capabilities").json()["engines"]["piper"]["voices"] == ["hi-IN-young-female", "hi-IN-young-male"]


def test_api_style_selection_via_emotion(client):
    r = speak(client, emotion="happy")
    assert r.status_code == 200 and pcm_amp(r) == 0.2
    assert r.headers["x-tts-applied-controls"] == "emotion"  # plain name: learned from recordings, not ":dsp" / ":steered"
    assert pcm_amp(speak(client)) == 0.1 and "x-tts-applied-controls" not in speak(client).headers
    assert pcm_amp(speak(client, emotion="neutral")) == 0.1


def test_api_style_selection_via_style_value(client, tmp_path, monkeypatch):
    other = tmp_path / "other"  # the catalog is cached per path
    other.mkdir()
    monkeypatch.setattr(settings, "voice_catalog", write_catalog(other, [entry("v", speaker="f_neutral", styles={"conversational": "f_happy", "neutral": "f_neutral"})]))
    monkeypatch.setattr(tts, "engine", piper_engine.PiperEngine())
    monkeypatch.setattr(settings, "default_voice", "v")
    from app.main import app
    with TestClient(app) as c:
        r = speak(c, voice="v", style="conversational")
        assert r.status_code == 200 and pcm_amp(r) == 0.2 and r.headers["x-tts-applied-controls"] == "style"


def test_api_unknown_style_follows_reject_and_ignore(client):
    r = speak(client, emotion="sad")  # a valid emotion, but this voice was not recorded in it
    assert r.status_code == 422 and r.json()["detail"]["unsupported"] == ["emotion"]
    r = speak(client, emotion="sad", fallback="ignore")
    assert r.status_code == 200 and pcm_amp(r) == 0.1  # default speaker
    assert r.headers["x-tts-applied-controls"] == "" and r.headers["x-tts-ignored-controls"] == "emotion"
    assert speak(client, style="warm").status_code == 422  # this voice lists no style values
    assert speak(client, "hi-IN-young-male", emotion="happy").status_code == 422  # a voice without styles: as today
    r = speak(client, "hi-IN-young-male", emotion="happy", fallback="ignore")
    assert r.status_code == 200 and pcm_amp(r) == 0.3 and r.headers["x-tts-ignored-controls"] == "emotion"


def test_api_discrete_styles_have_no_strength_and_take_one_selector(client):
    r = speak(client, emotion="happy", emotion_strength=0.5)
    assert r.status_code == 422 and r.json()["detail"]["unsupported"] == ["emotion_strength"]  # a recorded speaker has no strength knob
    r = speak(client, emotion="happy", emotion_strength=0.5, fallback="ignore")
    assert r.status_code == 200 and pcm_amp(r) == 0.2
    assert r.headers["x-tts-applied-controls"] == "emotion" and r.headers["x-tts-ignored-controls"] == "emotion_strength"
    r = speak(client, emotion="happy", style="soft", fallback="ignore")  # one recorded speaker per request: emotion wins, style reported
    assert pcm_amp(r) == 0.2 and r.headers["x-tts-ignored-controls"] == "style"


def test_api_alias_and_speed_and_planned(client):
    assert pcm_amp(speak(client, "young-female", emotion="happy")) == 0.2  # alias -> the voice, with its styles
    assert client.post("/v1/audio/speech", json={"input": "x", "voice": "young-female", "speed": 1.2}).status_code == 200
    assert speak(client, "hi-IN-adult-female").status_code == 404  # planned: not routable
    assert speak(client, "hi_IN-v7-medium:f_happy").status_code == 404  # raw speaker ids of a catalogued model are not public


def test_api_websocket_selects_style(client):
    with client.websocket_connect("/v1/audio/ws") as ws:
        ws.send_json({"type": "speak", "id": "a", "text": "नमस्ते।", "voice": "hi-IN-young-female", "sample_rate": SR, "condition": {"emotion": "happy"}})
        start = ws.receive_json()
        assert start["type"] == "start" and start["applied_controls"] == ["emotion"]
        pcm = b""
        while (m := ws.receive())["type"] == "websocket.send" and m.get("bytes"):
            pcm += m["bytes"]
        assert amp(np.frombuffer(pcm, "<i2").astype(np.float32) / 32767) == 0.2


def test_persona_gender_comes_from_the_catalog(client):
    """What the persona code reads: gender_of() and the `gender` of engine.voices() (demo.persona uses the latter)."""
    assert voice_catalog.gender_of("hi-IN-young-female") == "F" and voice_catalog.gender_of("hi-IN-young-male") == "M"
    assert voice_catalog.gender_of("not-a-voice") is None
    assert {v["voice_id"]: v["gender"] for v in tts.engine.voices()} == {"hi-IN-young-female": "F", "hi-IN-young-male": "M"}


def test_pronunciation_rules_pinned_per_voice(tmp_path, monkeypatch):
    path = write_catalog(tmp_path, [entry("hi-IN-young-male", speaker="m_neutral", gender="M", pronunciation_rules="all"),
                                    entry("hi-IN-young-female", speaker="f_neutral", gender="F")])
    monkeypatch.setattr(settings, "voice_catalog", path)
    assert voice_catalog.rules_of("hi-IN-young-male") == "all"
    assert voice_catalog.rules_of("hi-IN-young-female") is None and voice_catalog.rules_of("hi_IN-rohan-medium") is None
    with pytest.raises(voice_catalog.CatalogError):
        (tmp_path / "bad").mkdir()
        voice_catalog._load(str(write_catalog(tmp_path / "bad", [entry("hi-IN-x", pronunciation_rules="bogus")])))
