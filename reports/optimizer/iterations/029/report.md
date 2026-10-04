# Iteration 029: date-003 (dates, priority 1)

- Problem: pronunciation; text `आपकी अपॉइंटमेंट 23 जून 2024 को है।`; expected `आपकी अपॉइंटमेंट तेईस जून दो हज़ार चौबीस को है।`
- Previous: normalized `आपकी अपॉइंटमेंट तेईस जून दो हज़ार चौबीस को है।`; ASR ` आपकी अपोइंटमेंट 23 जून 2024 को है।`; total 0.912 (pron 0.785, CER 0.037)
- Candidate: baseline (no change); normalized `आपकी अपॉइंटमेंट तेईस जून दो हज़ार चौबीस को है।`; ASR ` आपकी अपोइंटमेंट 23 जून 2024 को है।`
- Score 0.912 -> 0.912; Laya confidence 0.5; regression (worst category drop) 0.0
- Decision: **rejected** (no candidate beat the current config)
- Affected tests: 0
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | baseline (no change) | 0.912 | 0.785 | 0.037 | 1.000 | - |
| 2 | dates.pause_ms = 60 | 0.912 | 0.785 | 0.037 | 1.000 | - |
| 3 | dates.pause_ms = 250 | 0.912 | 0.785 | 0.037 | 1.000 | - |
| 4 | dates.speed = 0.9 | 0.762 | 0.785 | 0.037 | 1.000 | - |
| 5 | dates.speed = 1.1 | 0.761 | 0.785 | 0.037 | 1.000 | - |
