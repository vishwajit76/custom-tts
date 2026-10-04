from optimizer import g2p


def test_word_and_pattern_fix():
    c = {"words": {"कृपया": "X"}, "patterns": [[r"r\.h", "ɽ"]]}
    assert g2p.fix("कृपया साढ़े", "kɾˈɪpjˌaː sˈaːr.heː", c) == "X sˈaːɽeː"
    assert g2p.fix("जी, कृपया।", "ɟˈi, kɾˈɪpjˌaː", c) == "ɟˈi, X"


def test_count_mismatch_skips_words():
    c = {"words": {"कृपया": "X"}, "patterns": []}
    assert g2p.fix("कृपया check-in", "a b c", c) == "a b c"
