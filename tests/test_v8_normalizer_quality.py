"""V8 Phase 6/7: deterministic normalization expectations + quality-filter rejection reasons."""
import numpy as np
import pytest
import soundfile as sf

from app.services.text_normalizer import normalize
from training import quality_gates as qg
from training.v8 import quality_filter as qf


@pytest.mark.parametrize("text,expected", [
    ("मुझे आज शाम पाँच बजे कॉल करना है।", "मुझे आज शाम पाँच बजे कॉल करना है।"),
    ("कीमत ₹1,25,000 है", "कीमत एक लाख पच्चीस हज़ार रुपये है"),
    ("₹1,25,00,000", "एक करोड़ पच्चीस लाख रुपये"),
    ("मीटिंग 12/10/2026 को है", "मीटिंग बारह अक्टूबर दो हज़ार छब्बीस को है"),
    ("अपॉइंटमेंट 5:30 PM पर", "अपॉइंटमेंट शाम साढ़े पाँच बजे पर"),
    ("OTP", "ओ टी पी"),
    ("API और CRM और SIP", "ए पी आई और सी आर एम और एस आई पी"),
    ("आपका account successfully verify हो गया है।", "आपका account successfully verify हो गया है।"),
    ("AI calling platform", "ए आई calling platform"),
])
def test_normalize(text, expected):
    assert normalize(text) == expected


def test_quality_filter_reasons(tmp_path):
    rng = np.random.default_rng(0)
    t = np.arange(22050 * 3) / 22050
    good = (0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 3 * t) > 0) + 0.001 * rng.standard_normal(len(t))).astype("float32")
    sf.write(tmp_path / "a.wav", good, 22050)
    sf.write(tmp_path / "b.wav", good, 22050)            # duplicate audio
    sf.write(tmp_path / "c.wav", good[:5000], 22050)     # too short
    sf.write(tmp_path / "d.wav", good, 16000)            # bad sample rate
    (tmp_path / "e.wav").write_bytes(b"junk")            # corrupted
    mk = lambda n, text: {"path": tmp_path / n, "text": text, "gender": "female", "speaker": "s", "source": "x"}  # noqa: E731
    rows = [mk("a.wav", "यह एक अच्छा वाक्य है और काफ़ी लंबा भी है"), mk("b.wav", "दूसरा वाक्य यहाँ पर है और यह भी लंबा है"),
            mk("c.wav", "छोटा"), mk("d.wav", "एक और अलग वाक्य जो यहाँ पर है ठीक"), mk("e.wav", "खराब फ़ाइल यहाँ है")]
    g, bad = qf.process(rows, qg.Thresholds(), 22050)
    reasons = {Path_name: r["reasons"] for r in bad for Path_name in [r["audio_path"][-5:]]}
    assert "duplicate_audio" in reasons["b.wav"]
    assert "too_short" in reasons["c.wav"]
    assert "bad_sample_rate" in reasons["d.wav"]
    assert reasons["e.wav"][0] == "corrupted_audio"


def test_hindi_text_reasons():
    assert qf.hindi_text_reasons("hello world") == ["no_devanagari"]
    assert qf.hindi_text_reasons("नमस्ते 123") == ["bad_normalization"]
    assert qf.hindi_text_reasons("नमस्ते दुनिया।") == []
