"""Persona grammar (T2): positive rewrites in both directions, safety negatives, idempotence, API wiring."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import settings
from app.services import hinglish, tts
from app.services.persona_grammar import Persona, apply, llm_hint, normalize_gender, rewrite

F, M, N = Persona("female"), Persona("male"), Persona("neutral")

# (masculine, feminine): m->f under F, f->m under M. Every family the module claims to handle.
PAIRS = [
    # required forms
    ("मैं करता हूँ", "मैं करती हूँ"),
    ("मैं करता था", "मैं करती थी"),
    ("मैं कल करूँगा", "मैं कल करूँगी"),
    ("मैं गया", "मैं गई"),
    ("मैं आया", "मैं आई"),
    ("मैं आपकी मदद कर सकता हूँ", "मैं आपकी मदद कर सकती हूँ"),
    ("मैं आपकी मदद करना चाहता हूँ", "मैं आपकी मदद करना चाहती हूँ"),
    ("मैं बोलता हूँ", "मैं बोलती हूँ"),
    ("मैं देता हूँ", "मैं देती हूँ"),
    ("मैं लेता हूँ", "मैं लेती हूँ"),
    # imperfective / habitual
    ("मैं रोज़ जाता हूँ", "मैं रोज़ जाती हूँ"),
    ("मैं सुनता हूँ", "मैं सुनती हूँ"),
    ("मैं कभी नहीं जाता था", "मैं कभी नहीं जाती थी"),
    ("मैं पीता हूँ", "मैं पीती हूँ"),
    ("मैं होता हूँ", "मैं होती हूँ"),
    ("मैं पाता हूँ", "मैं पाती हूँ"),
    ("मैं बताता हूँ", "मैं बताती हूँ"),
    ("मैं रहता हूँ", "मैं रहती हूँ"),
    # progressive
    ("मैं कर रहा हूँ", "मैं कर रही हूँ"),
    ("मैं कर रहा था", "मैं कर रही थी"),
    ("मैं जा रहा हूँ", "मैं जा रही हूँ"),
    ("मैं देख रहा हूँ", "मैं देख रही हूँ"),
    # perfective / perfect
    ("मैं गया था", "मैं गई थी"),
    ("मैं आया था", "मैं आई थी"),
    ("मैं घर गया", "मैं घर गई"),
    ("मैं आ गया हूँ", "मैं आ गई हूँ"),
    ("मैं समझ गया हूँ", "मैं समझ गई हूँ"),
    ("मैं कर चुका हूँ", "मैं कर चुकी हूँ"),
    ("मैं सुन चुका था", "मैं सुन चुकी थी"),
    ("मैं चला गया", "मैं चली गई"),
    ("मैं बनाया गया हूँ", "मैं बनाई गई हूँ"),
    ("मैं सो गया", "मैं सो गई"),
    ("मैं जाने वाला हूँ", "मैं जाने वाली हूँ"),
    # future
    ("मैं जाऊँगा", "मैं जाऊँगी"),
    ("मैं दूँगा", "मैं दूँगी"),
    ("मैं लूँगा", "मैं लूँगी"),
    ("मैं आऊँगा", "मैं आऊँगी"),
    ("मैं बताऊँगा", "मैं बताऊँगी"),
    ("मैं देखूँगा", "मैं देखूँगी"),
    ("मैं कर सकूँगा", "मैं कर सकूँगी"),
    ("मैं कर पाऊँगा", "मैं कर पाऊँगी"),
    ("मैं कर रहा होऊँगा", "मैं कर रही होऊँगी"),
    ("मैं तैयार रहूँगा", "मैं तैयार रहूँगी"),
    # anusvara / chandrabindu spellings (the writer's spelling is preserved)
    ("मैं करता हूं", "मैं करती हूं"),
    ("मैं करूंगा", "मैं करूंगी"),
    ("मैं जाऊंगा", "मैं जाऊंगी"),
    ("मैं दूंगा", "मैं दूंगी"),
    ("मै जाता हूँ", "मै जाती हूँ"),
    # implicit first person (no मैं), anchored by हूँ
    ("आपकी मदद कर सकता हूँ", "आपकी मदद कर सकती हूँ"),
    ("बताता हूँ", "बताती हूँ"),
    # predicate adjectives and participles
    ("मैं अच्छा हूँ", "मैं अच्छी हूँ"),
    ("मैं नया हूँ", "मैं नई हूँ"),
    ("मैं बहुत थका हुआ हूँ", "मैं बहुत थकी हुई हूँ"),
    ("मैं अकेला रहता हूँ", "मैं अकेली रहती हूँ"),
    ("मैं अकेला महसूस करता हूँ", "मैं अकेली महसूस करती हूँ"),
    ("मैं यहीं बैठा हूँ", "मैं यहीं बैठी हूँ"),
    ("मैं छोटा था", "मैं छोटी थी"),
    # invariant words between agreeing words
    ("मैं यह नहीं करता हूँ", "मैं यह नहीं करती हूँ"),
    ("मैं बोल भी सकता हूँ", "मैं बोल भी सकती हूँ"),
    # was/were with an invariant predicate: only the auxiliary agrees
    ("मैं वहाँ था", "मैं वहाँ थी"),
    ("मैं बीमार था", "मैं बीमार थी"),
    # self-description
    ("मैं आपका सहायक हूँ", "मैं आपकी सहायक हूँ"),
    ("आपका असिस्टेंट हूँ", "आपकी असिस्टेंट हूँ"),
    ("मैं आपका वर्चुअल असिस्टेंट हूँ", "मैं आपकी वर्चुअल असिस्टेंट हूँ"),
    ("मैं आपका assistant हूँ", "मैं आपकी assistant हूँ"),
    ("मैं आपका दोस्त बनूँगा", "मैं आपकी दोस्त बनूँगी"),
    # several clauses / mixed sentences: only the first-person clause changes
    ("मैं समझता हूँ, मैं बताता हूँ", "मैं समझती हूँ, मैं बताती हूँ"),
    ("वह जाता है, लेकिन मैं कल जाऊँगा", "वह जाता है, लेकिन मैं कल जाऊँगी"),
    ("राहुल गया था और मैं आपकी मदद करता हूँ", "राहुल गया था और मैं आपकी मदद करती हूँ"),
    ("मुझे लगता है कि मैं कर सकता हूँ", "मुझे लगता है कि मैं कर सकती हूँ"),
    ("मैं जानता हूँ कि मैं गया था", "मैं जानती हूँ कि मैं गई थी"),
    ("जैसा आपने कहा, मैं वैसा ही करूँगा", "जैसा आपने कहा, मैं वैसा ही करूँगी"),
    ("Hello! मैं आपकी मदद कर सकता हूँ।", "Hello! मैं आपकी मदद कर सकती हूँ।"),
    ("नमस्ते। मैं आपका सहायक हूँ। मैं कल बताऊँगा।", "नमस्ते। मैं आपकी सहायक हूँ। मैं कल बताऊँगी।"),
]


@pytest.mark.parametrize("masc,fem", PAIRS)
def test_masculine_to_feminine(masc, fem):
    assert apply(masc, F) == fem


@pytest.mark.parametrize("masc,fem", PAIRS)
def test_feminine_to_masculine(masc, fem):
    # the alternate spellings गयी/आयी normalise to the standard masculine form
    assert apply(fem, M) == masc


@pytest.mark.parametrize("masc,fem", PAIRS)
def test_idempotent(masc, fem):
    for src, p, out in ((masc, F, fem), (fem, M, masc)):
        once = apply(src, p)
        assert apply(once, p) == once
    assert apply(fem, F) == fem and apply(masc, M) == masc  # already the right gender: untouched


@pytest.mark.parametrize("masc,fem", PAIRS)
def test_neutral_is_a_noop(masc, fem):
    assert apply(masc, N) == masc and apply(fem, N) == fem


# every text below must come back unchanged for BOTH genders
UNCHANGED = [
    # third person
    "वह जाता है", "राहुल गया था", "वो करती है", "वे कर रहे हैं", "वह बोल रहा है और वो कर रहा है", "प्रिया आई थी",
    "सीमा जाती है", "लड़का आया", "ट्रेन चली गई", "बारिश हो रही है",
    # second person
    "आप कर सकते हैं", "आप कर रहे हैं", "तुम जाते हो", "आप कहाँ गए थे", "आप कैसे हैं",
    # plural / royal हम
    "हम जा रहे हैं", "हम आपकी मदद कर सकते हैं", "हम कल आएँगे", "हम आए थे",
    # मुझे-constructions and non-agreeing first person
    "मुझे पता है", "मुझे जाना है", "मुझे लगता है", "मुझे भूख लगी है", "मेरा नाम राहुल है", "मैंने खाना खाया", "मैंने कहा था",
    # nouns and names ending in ता/ती and invariant predicates
    "रास्ता", "मुझे रास्ता नहीं पता", "यह मेरा पता है", "नेता आए", "मैं नेता हूँ", "मैं गीता हूँ", "मैं सीता हूँ", "मैं आरती हूँ",
    "मैं ममता हूँ", "मैं लता हूँ", "मैं तैयार हूँ", "मैं खुश हूँ", "मैं ठीक हूँ", "मैं बीमार हूँ", "मैं व्यस्त हूँ", "मैं आभारी हूँ",
    "मैं अच्छा काम करता हूँ".replace("करता", "करना"),
    # English only
    "I can help you", "I am going to call you", "He said that I will go", "Hello, how are you?",
    # nothing to anchor on
    "नमस्ते", "", "   ", "आज मौसम अच्छा है।",
]


@pytest.mark.parametrize("text", UNCHANGED)
@pytest.mark.parametrize("p", [F, M], ids=["female", "male"])
def test_never_rewritten(text, p):
    assert apply(text, p) == text


# quoted / reported speech: the speaker's own clause may change, the other person's words may not
QUOTED = [
    ('उसने कहा "मैं जाता हूँ"', 'उसने कहा "मैं जाता हूँ"'),
    ("उसने कहा “मैं जाता हूँ”", "उसने कहा “मैं जाता हूँ”"),
    ("उसने कहा ‘मैं जाऊँगा’", "उसने कहा ‘मैं जाऊँगा’"),
    ("उसने कहा 'मैं जाता हूँ'", "उसने कहा 'मैं जाता हूँ'"),
    ('He said "मैं जाता हूँ"', 'He said "मैं जाता हूँ"'),
    ('"मैं कल आऊँगा" उसने कहा', '"मैं कल आऊँगा" उसने कहा'),
    ('उसने कहा: "मैं गया था", और मैं आपकी मदद करता हूँ', None),
    ('"नमस्ते" मैं बोलता हूँ', None),
    ('उसने कहा "ठीक है" और मैं चला गया', None),
    ('उसने कहा "मैं जाता हूँ', 'उसने कहा "मैं जाता हूँ'),  # unbalanced quote: the rest is treated as quoted
    ("उसने कहा कि मैं जाता हूँ", "उसने कहा कि मैं जाता हूँ"),
    ("राहुल ने कहा कि मैं कल आऊँगा", "राहुल ने कहा कि मैं कल आऊँगा"),
    ("वह कहता है कि मैं गया था", "वह कहता है कि मैं गया था"),
    ("उन्होंने बताया कि मैं तैयार हूँ और मैं जाऊँगा", "उन्होंने बताया कि मैं तैयार हूँ और मैं जाऊँगा"),
    ("राहुल ने कहा, मैं कल आऊँगा", "राहुल ने कहा, मैं कल आऊँगा"),
    ("मैं कल आऊँगा, राहुल ने कहा", "मैं कल आऊँगा, राहुल ने कहा"),
    ("वह कहता है मैं जाता हूँ", "वह कहता है मैं जाता हूँ"),
    ("आपने कहा कि मैं गलत था", "आपने कहा कि मैं गलत था"),
    ("He said that मैं जाता हूँ", "He said that मैं जाता हूँ"),
]


@pytest.mark.parametrize("src,keep", QUOTED)
def test_quoted_and_reported_speech(src, keep):
    if keep is not None:
        for p in (F, M):
            assert apply(src, p) == keep
        return
    # mixed: quotes stay byte-identical, the narrator's own clause is rewritten
    out = apply(src, F)
    assert out != src
    quoted = [s for s in ('"मैं गया था"', '"नमस्ते"', '"ठीक है"') if s in src]
    assert all(s in out for s in quoted)


def test_female_persona_keeps_masculine_inside_quotes():
    assert apply('उसने कहा "मैं जाता हूँ"', F) == 'उसने कहा "मैं जाता हूँ"'
    assert apply('उसने कहा "ठीक है" और मैं चला गया', F) == 'उसने कहा "ठीक है" और मैं चली गई'
    assert apply('उसने कहा: "मैं गया था", और मैं आपकी मदद करता हूँ', F) == 'उसने कहा: "मैं गया था", और मैं आपकी मदद करती हूँ'


def test_first_person_reporting_is_the_speakers_own():
    assert apply("मैंने कहा कि मैं जाऊँगा", F) == "मैंने कहा कि मैं जाऊँगी"
    assert apply("मैं कहता हूँ कि मैं जाऊँगा", F) == "मैं कहती हूँ कि मैं जाऊँगी"
    assert apply("मैंने कहा, मैं जाऊँगा", F) == "मैंने कहा, मैं जाऊँगी"


def test_alternate_feminine_spellings_normalise_to_masculine():
    assert apply("मैं आयी थी", M) == "मैं आया था" and apply("मैं गयी थी", M) == "मैं गया था"
    assert apply("मैं थकी हुयी हूँ", M) == "मैं थका हुआ हूँ"


def test_rewrite_lists_the_changes_with_original_offsets():
    text = "वह जाता है, मैं आपका सहायक हूँ और मैं कल करूँगा"
    r = rewrite(text, F)
    assert r.text == "वह जाता है, मैं आपकी सहायक हूँ और मैं कल करूँगी"
    assert [(h.old, h.new, h.rule) for h in r.rewrites] == [("आपका", "आपकी", "possessive"), ("करूँगा", "करूँगी", "future")]
    assert all(text[h.start:h.end] == h.old for h in r.rewrites)
    assert rewrite(text, N).rewrites == () and rewrite("वह जाता है", F).rewrites == ()


def test_age_group_and_persona_label_change_nothing():
    text = "मैं आपकी मदद कर सकता हूँ, मैं कल जाऊँगा"
    base = apply(text, F)
    for age in ("young_adult", "adult", "mature"):
        assert apply(text, Persona("female", "teacher", age)) == base


def test_persona_validation():
    with pytest.raises(ValidationError):
        Persona("robot")
    with pytest.raises(ValidationError):
        Persona("female", age_group="child")
    with pytest.raises(ValidationError):
        Persona("female", persona="x\ny")
    assert Persona(gender="male").persona == "assistant" and Persona("male").age_group == "adult"


def test_normalize_gender_and_hint():
    assert [normalize_gender(g) for g in ("F", "female", " Woman ", "M", "male", "neutral", "", None, "x")] == [
        "female", "female", "female", "male", "male", "neutral", "neutral", "neutral", "neutral"]
    assert "करती हूँ" in llm_hint(F) and "करता हूँ" in llm_hint(M)
    assert llm_hint(N) == ""
    assert "teacher" in llm_hint(Persona("female", "teacher")) and "भूमिका" not in llm_hint(F)


def test_hinglish_wrapper_keeps_old_contract():
    f = hinglish.apply_persona_gender
    assert f("मैं गया था", "female") == "मैं गई थी" and f("आपकी सहायक हूँ", "male") == "आपका सहायक हूँ"
    assert f("मैं कर सकता हूँ", None) == "मैं कर सकता हूँ" and f("मैं कर सकता हूँ", "neutral") == "मैं कर सकता हूँ"


def test_fast_enough():
    import time

    text = "नमस्ते, मैं आपका सहायक हूँ और मैं आपकी हर तरह की मदद कर सकता हूँ। वह कल जाएगा, लेकिन मैं कल आऊँगा। उसने कहा \"ठीक है\"।"
    text = text + " " + text[:60]
    assert 150 < len(text) < 260
    apply(text, F)
    t0 = time.perf_counter()
    for _ in range(200):
        apply(text, F)
    assert (time.perf_counter() - t0) / 200 < 0.002  # budget is 0.5 ms; 4x slack for loaded CI machines


# ---------- API wiring ----------
SR = 22050


class FakeEngine:
    supports_cloning = False
    max_workers = 2
    ready = True

    def __init__(self):
        self.calls = []

    def load(self):
        pass

    def voices(self):
        return [{"voice_id": "fake", "sample_rate": SR}]

    def has_voice(self, v):
        return v == "fake"

    def sample_rate(self, v):
        return SR

    def synth(self, text, voice, speed, ref=None, ref_text=None):
        self.calls.append(text)
        return np.full(SR // 10, 0.5, np.float32)


@pytest.fixture()
def api(monkeypatch):
    e = FakeEngine()
    monkeypatch.setattr(tts, "engine", e)
    monkeypatch.setattr(settings, "default_voice", "fake")
    monkeypatch.setattr(settings, "api_keys", "")
    monkeypatch.setattr(settings, "cache_size", 0)
    from app.main import app

    with TestClient(app) as c:
        yield c, e


TEXT = "मैं आपकी मदद कर सकता हूँ"


@pytest.mark.parametrize("path", ["/v1/audio/speech", "/v1/audio/speech/stream"])
def test_http_persona_is_opt_in(api, path):
    c, e = api
    assert c.post(path, json={"input": TEXT}).status_code == 200
    assert e.calls[-1] == TEXT  # absent = unchanged
    assert c.post(path, json={"input": TEXT, "persona": {"gender": "female"}}).status_code == 200
    assert e.calls[-1] == "मैं आपकी मदद कर सकती हूँ"
    assert c.post(path, json={"input": TEXT, "persona": {"gender": "neutral"}}).status_code == 200
    assert e.calls[-1] == TEXT
    assert c.post(path, json={"input": TEXT, "persona": {"gender": "robot"}}).status_code == 422


def test_ws_persona(api):
    c, e = api
    with c.websocket_connect("/v1/audio/ws") as ws:
        ws.send_json({"type": "speak", "id": "a", "text": TEXT, "persona": {"gender": "female", "age_group": "mature"}})
        while ws.receive().get("text", "").find('"end"') < 0:
            pass
        ws.send_json({"type": "speak", "id": "b", "text": TEXT})
        while ws.receive().get("text", "").find('"end"') < 0:
            pass
    assert e.calls == ["मैं आपकी मदद कर सकती हूँ", TEXT]
