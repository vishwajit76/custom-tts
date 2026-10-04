# Evaluation corpora

| version | file | rows | sha256 |
|---|---|---|---|
| v1 | `bench/hi_eval_50.txt` (outside this folder; unchanged) | 50 | see `docs/benchmarks.md` section 10 |
| v2 | `hi_eval_v2.tsv` | 258 | `814850bf4e71617f28355c4898ea54c6832246b68de9164944e77363bfb2f601` |

**Rule: never edit a published corpus version; add v3 instead.** Scores are only comparable on the same text. The loader
(`bench.corpus.load("v2")`) pins the sha256 in `VERSIONS` and raises `CorpusHashMismatch` if the file differs; the hash above and the pin
are cross-checked by `tests/test_corpus.py`. `.gitattributes` marks the TSV `-text` so line-ending conversion cannot change the hash.

**Status: not native-reviewed.** The sentences were drafted by a model and checked only for grammar by the author. Treat them as a benchmark
set, not as reviewed Hindi. A native review that changes a sentence means a new version (v3).

## v2 format and contents

Columns: `id, category, subcategory, text, notes`. `notes` is `key=value; key=value` (`domain=` booking | delivery | banking | appointment |
support; `speaker=f|m` for gendered first-person forms, absent when neutral; `script=mixed|roman` for Hinglish; `focus=` the words that carry a
pronunciation feature; `known_gap=`). Register: assistant and call-centre speech (bookings, delivery, banking, appointments, support), not literary text.
Digits, currency and dates are written as a user or backend would write them (`₹1,250`, `03/11/2026`, `11:45 बजे`); the references used for CER/PER
are `app.services.text_normalizer.normalize(text)`, so a normalizer change moves the reference (the harness records the git commit).
Roman-script Hinglish rows depend on the romanization table in the same way. No sentence is in `bench/hi_eval_50.txt`.

| category | rows | subcategories |
|---|---|---|
| hindi | 100 | statement 30, confirmation 12, request 14, instruction 14, uncertainty 10, apology 10, exclamation 10 |
| hinglish | 50 | statement 14, request 8, instruction 8, confirmation 6, yes_no_question 6, wh_question 4, apology 4 (15 romanized, 35 mixed script) |
| numbers | 32 | currency 8, date 6, time 6, phone_otp 4, percent_quantity 4, ordinal_id 4 |
| pronunciation | 32 | schwa_deletion 4, conjunct 4, nukta 4, anusvara_chandrabindu 3, ri_vowel 3, perso_arabic 3, english_loan 3, indian_name 3, brand 3, city 2 |
| questions | 22 | yes_no_question 11, wh_question 11 |
| expressive | 22 | joy 5, sympathy 5, urgency 4, surprise 4, reassurance 4 |
| **total** | **258** | |

Prosody categories are subcategories that cut across the categories above (`bench.corpus.PROSODY`): statement 44, yes_no_question 17,
wh_question 15, exclamation 10, confirmation 18, uncertainty 10, apology 14, request 22, instruction 22 (each >= 8).

## Contamination check

`python -m bench.check_eval_overlap --eval bench/corpus/hi_eval_v2.tsv --also-eval bench/hi_eval_50.txt --csv <metadata.csv> --csv <test.csv>`
fails on an exact sentence or any shared 5-word run with the given training/test text, and on any sentence that is also in v1. For V7 run it
against the prepared Rasa `metadata.csv` and the V6 IndicTTS `data/hi_f/*.csv` before using v2 for a decision. On the authoring machine the
IndicTTS text (`data/hi_f`) was not present, so overlap with the real training text is **unchecked**; only the no-overlap-with-v1 check was run.
