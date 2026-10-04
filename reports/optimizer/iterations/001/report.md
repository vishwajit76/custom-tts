# Iteration 001: date-002 (dates, priority 1)

- Problem: normalization; text `आपकी अपॉइंटमेंट 2026-05-08 को है।`; expected `आपकी अपॉइंटमेंट आठ मई दो हज़ार छब्बीस को है।`
- Previous: normalized `आपकी अपॉइंटमेंट दो हज़ार छब्बीस से शून्य पाँच-शून्य आठ को है।`; ASR ` आपकी अपॉइंट्मेंट दो हजार चब्वीस से शून्य पांच शून्य आठ को है।`; total 0.574 (pron 0.312, CER 0.735)
- Candidate: rule iso_date; normalized `आपकी अपॉइंटमेंट आठ मई दो हज़ार छब्बीस को है।`; ASR ` आपकी अपोइंटमेंट आठ माई दो हजार चब्वीस को है।`
- Score 0.574 -> 0.750; Laya confidence 1.0; regression (worst category drop) 0.0
- Decision: **rejected** (rtf 0.034->0.374)
- Affected tests: 9
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | rule iso_date | 0.750 | 0.753 | 0.118 | 1.000 | - |
| 2 | baseline (no change) | 0.574 | 0.312 | 0.735 | 0.353 | - |
| 3 | dates.speed = 0.9 | 0.574 | 0.312 | 0.735 | 0.353 | - |
| 4 | dates.speed = 1.1 | 0.574 | 0.312 | 0.735 | 0.353 | - |
| 5 | dates.pause_ms = 60 | 0.574 | 0.312 | 0.735 | 0.353 | - |
| 6 | dates.pause_ms = 250 | 0.574 | 0.312 | 0.735 | 0.353 | - |
