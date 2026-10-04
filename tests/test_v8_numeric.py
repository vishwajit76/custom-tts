import pytest
from app.services.text_normalizer import normalize


@pytest.mark.parametrize("text,expected", [
    ("2026-10-04", "चार अक्टूबर दो हज़ार छब्बीस"),
    ("आज 1995-03-12 को", "आज बारह मार्च उन्नीस सौ पचानवे को"),
    ("26 दिसंबर 1995 को", "छब्बीस दिसंबर उन्नीस सौ पचानवे को"),
    ("4 October 2026", "चार अक्टूबर दो हज़ार छब्बीस"),
    ("तारीख 2026-13-45", "तारीख दो हज़ार छब्बीस से तेरह-पैंतालीस"),
])
def test_dates(text, expected):
    assert normalize(text) == expected
