# Iteration 009: date-000 (dates, priority 1)

- Problem: pronunciation; text `आपकी अपॉइंटमेंट 05/07/2026 को है।`; expected `आपकी अपॉइंटमेंट पाँच जुलाई दो हज़ार छब्बीस को है।`
- Previous: normalized `आपकी अपॉइंटमेंट पाँच जुलाई दो हज़ार छब्बीस को है।`; ASR ` आपकी अपोइंटमेंट 5 जुलाई 2026 को है।`; total 0.724 (pron 0.692, CER 0.269)
- Candidate: baseline (no change); normalized `आपकी अपॉइंटमेंट पाँच जुलाई दो हज़ार छब्बीस को है।`; ASR ` आपकी अपोइंटमेंट 5 जुलाई 2026 को है।`
- Score 0.724 -> 0.724; Laya confidence 0.498; regression (worst category drop) 0.0
- Decision: **rejected** (no candidate beat the current config)
- Affected tests: 0
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | baseline (no change) | 0.724 | 0.692 | 0.269 | 1.000 | - |
| 2 | dates.speed = 0.9 | 0.724 | 0.692 | 0.269 | 1.000 | - |
| 3 | dates.speed = 1.1 | 0.724 | 0.692 | 0.269 | 1.000 | - |
| 4 | dates.pause_ms = 60 | 0.724 | 0.692 | 0.269 | 1.000 | - |
| 5 | dates.pause_ms = 250 | 0.724 | 0.692 | 0.269 | 1.000 | - |
