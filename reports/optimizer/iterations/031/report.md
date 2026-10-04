# Iteration 031: date-022 (dates, priority 2)

- Problem: normalization; text `आपकी अपॉइंटमेंट 2025-03-06 को है।`; expected `आपकी अपॉइंटमेंट छह मार्च दो हज़ार पच्चीस को है।`
- Previous: normalized `आपकी अपॉइंटमेंट दो हज़ार पच्चीस से शून्य तीन-शून्य छह को है।`; ASR ` आपकी appointment दो हजार पचीस से शून्य तीन शून्य छे को है।`; total 0.512 (pron 0.157, CER 0.892)
- Candidate: rule iso_date; normalized `आपकी अपॉइंटमेंट छह मार्च दो हज़ार पच्चीस को है।`; ASR ` आपकी अपोइंटमेंट छे मार्च दो हजार पचीस को है।`
- Score 0.512 -> 0.751; Laya confidence 1.0; regression (worst category drop) 0.0
- Decision: **accepted** (gates passed)
- Affected tests: 9
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | rule iso_date | 0.751 | 0.757 | 0.108 | 1.000 | - |
| 2 | dates.speed = 0.9 | 0.560 | 0.276 | 0.595 | 0.378 | - |
| 3 | baseline (no change) | 0.512 | 0.157 | 0.892 | 0.378 | - |
| 4 | dates.speed = 1.1 | 0.512 | 0.157 | 0.892 | 0.378 | - |
| 5 | dates.pause_ms = 60 | 0.512 | 0.157 | 0.892 | 0.378 | - |
| 6 | dates.pause_ms = 250 | 0.512 | 0.157 | 0.892 | 0.378 | - |
