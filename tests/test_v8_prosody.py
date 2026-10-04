import json
from optimizer import core


def test_category_speed_applies():
    a = json.loads(core.ACTIVE.read_text(encoding="utf-8"))
    assert core.settings_for(a, "no_such_category")["speed"] == a["default"]["speed"]
    for cat, cfg in a["category"].items():
        if "speed" in cfg:
            assert core.settings_for(a, cat)["speed"] == cfg["speed"]
            assert 0.8 <= cfg["speed"] <= 1.2
