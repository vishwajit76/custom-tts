import json
from pathlib import Path

from app.services import hinglish
from optimizer.core import apply_dict

PRON = json.loads((Path(__file__).parent.parent / "optimizer" / "pron_dict.json").read_text("utf-8"))


def test_schema():
    for k, v in PRON.items():
        assert v["spoken"] and {"reason", "confidence", "iteration", "timestamp", "affected_tests"} <= set(v), k


def test_word_boundary_and_case():
    assert apply_dict("Send OTP now", PRON) == "Send ओ टी पी now"
    assert apply_dict("laptops OTPS", PRON) == "laptops OTPS"  # no partial-word hit
    assert apply_dict("Laptop और laptop", PRON) == "लैपटॉप और लैपटॉप"
    assert apply_dict("it is", PRON) == "it is"  # IT is case-sensitive


def test_names_and_places():
    assert hinglish.convert("मेरा नाम Kshitij है") == "मेरा नाम क्षितिज है"
    assert hinglish.convert("Puducherry") == "पुडुचेरी"


def test_asr_latin_acronym_not_penalised():
    from optimizer.evaluators import asr_metrics
    ref = {"expected_normalized": "आपका ए पी आई अपडेट", "text": "आपका API अपडेट"}
    assert asr_metrics("आपका API अपडेट", ref)["cer"] == 0.0


def test_asr_digits_compared_as_words():
    from optimizer.evaluators import asr_metrics
    ref = {"expected_normalized": "पाँच सौ रुपये", "text": "₹500"}
    assert asr_metrics("500 रुपये", ref)["cer"] < 0.1


def test_number_pauses_not_penalised():
    from optimizer.evaluators import proxies
    aud = {"rate_cps": 12, "issues": [], "pauses": 2, "audio_quality": 1.0, "rms_db": -22}
    s = proxies({"text": "98765 43210"}, {"text_score": 1}, {"cer": 0, "important_acc": 1}, aud, {"rate_cps": [9, 18], "target_rms_db": -22})
    assert s["prosody"] == 1.0
