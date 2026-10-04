# Agent B log (english_words, acronyms, abbreviations, hinglish, names, proper_nouns)
Scores are goonj category totals. "before" re-scored with the new evaluator (pronunciation only; baseline audio unchanged).
| category | before (old eval) | before (new eval) | after |
|---|---|---|---|
| english_words | 0.8132 | 0.8192 | 0.8311 |
| acronyms | 0.8325 | 0.8990 | 0.9092 |
| abbreviations | 0.9383 | 0.9490 | 0.9490 |
| hinglish | 0.8729 | 0.8738 | 0.8723 |
| names | 0.8852 | 0.8852 | 0.9012 |
| proper_nouns | 0.9026 | 0.9026 | 0.9146 |
piper_v7a after: english 0.8012, acronyms 0.8948, abbr 0.8838, hinglish 0.8496, names 0.864, proper 0.8744 (no pre-change piper baseline captured).

Experiments
1. +224 pron_dict entries (acronyms letter-by-letter / word-like, ~60 English words in two cases): text_exact english 0.35->1.0, acronyms 0.86->1.0. KEPT.
2. +229 lexicon_hi.tsv name rows (names, cities, brands): names/proper_nouns text_exact -> 1.0. KEPT (whatsapp removed: documented English-route exception, test_normalizer_p3).
3. hinglish regression: generic English words (payment, order, booking, app, ...) in pron_dict broke hinglish bench which expects them in Latin (-0.0085). REVERTED those 10 words; account/update kept (needed by english_words). hinglish flat.
4. Variant sweep (vt.py, Whisper hears the correct name): मुम्बई (0.944->0.987), श्रेय्या (->0.98), चौहाण (->0.98) KEPT. Rejected: लखनऊ/चेन्नई/जयपुर/इंदौर/अभिषेक/सिद्धार्थ/ऐश्वर्या variants (no gain). Caveat: vt must keep raw text = target, or the scorer's raw-text reference inflates results.
5. Evaluator (optimizer/evaluators.py): Latin ASR tokens mapped via pron_dict; ASR digits -> words via text_normalizer; number-group pauses tolerated in prosody. Unit tests in tests/test_v8_lexicon.py.
6. tests/test_optimizer.py: test word WhatsApp -> Zorblax (WhatsApp is now in the lexicon path).

Known regressions inside categories: en-009 software, en-015 account got worse audio (model), net category up.
Bench disputes: hinglish expects English words kept Latin (hg-*), conflicts with english_words expecting the same words in Devanagari.
