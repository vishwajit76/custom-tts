# Agent A log (numbers/currency/phone/percent/dates/time)
Baseline goonj: numbers .8371, currency .7067, phone .7535, percent .8163, time .9374, dates .7523. piper_v7a: .7716/.7102/.7389/.8146/.8530/.7256.
Key finding: remaining losses are mostly evaluator artifacts. Whisper writes digits ("24 अप्रैल 2010", "500 रुपिये") and canon() does not map digits to words, so CER/important_acc penalize correct audio. Prosody=0 where text has no pause but audio has a >=120 ms gap (dates/currency/phone). Fix belongs in optimizer/evaluators.py (digits->words via normalizer on ASR hyp), not owned by A.

| # | hypothesis | before | after | decision |
|---|---|---|---|---|
| 1 | "26 दिसंबर 1995" (Hindi month + year) read as "एक हज़ार नौ सौ.." | text_exact dates .8056 | date-011 fixed, dates .7523->.7544 | KEEP (_HI_MONTH_YEAR_RE) |
| 2 | bare ISO 2026-10-04 mis-normalized by app (only worked via optimizer rule iso_date) | "दो हज़ार छब्बीस से दस-शून्य चार" | "चार अक्टूबर दो हज़ार छब्बीस" | KEEP (_ISO_DATE_RE, invalid dates untouched) |
| 3 | phone groups with no comma | phone .7535 | .8028 (prosody artifact), pron .770->.752 | REJECT (pron worse, gaming pause metric) |
| 4 | phone triplets / 3-3-4 / pairs | .7535 | .7515 / .7440 / .7565 | REJECT (noise) |
| 5 | रुपये->रुपए | currency .7067 | .7024 | REJECT |
| 6 | comma after month in dates | dates .7576 | .7345 | REJECT |
| 7 | हज़ार->हजार (post-normalizer) | goonj cur .7067, dates .7544, num .8371 | .7145, .7708, .8365; piper cur +.011 dates +.002 num -.0067 (pron +.013) | REJECT as normalizer change: existing tests/bench expect हज़ार; candidate for a post-normalize option in core.frontend |

## Bench disputes
- पचानवे vs बench पंचानवे (date-001/005/011/018/021/033/034): both spellings valid; canon() does not equate them. Same for तिरसठ/तिरेसठ (num-035), तिरपन/तिरेपन (cur-039).
- pct-016 2.5%: bench "साढ़े दो प्रतिशत"; normalizer "ढाई प्रतिशत" (ढाई is the correct word for 2.5). Add as alternative.
