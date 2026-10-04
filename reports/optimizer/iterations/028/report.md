# Iteration 028: date-001 (dates, priority 1)

- Problem: pronunciation; text `आपकी अपॉइंटमेंट 8 June 1995 को है।`; expected `आपकी अपॉइंटमेंट आठ जून उन्नीस सौ पंचानवे को है।`
- Previous: normalized `आपकी अपॉइंटमेंट आठ जून उन्नीस सौ पचानवे को है।`; ASR ` आपकी अपोइंटमेंट 8 जून 1995 को है।`; total 0.884 (pron 0.718, CER 0.185)
- Candidate: baseline (no change); normalized `आपकी अपॉइंटमेंट आठ जून उन्नीस सौ पचानवे को है।`; ASR ` आपकी अपोइंटमेंट 8 जून 1995 को है।`
- Score 0.884 -> 0.884; Laya confidence 0.5; regression (worst category drop) 0.0
- Decision: **rejected** (no candidate beat the current config)
- Affected tests: 0
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | baseline (no change) | 0.884 | 0.718 | 0.185 | 0.974 | - |
| 2 | dates.pause_ms = 60 | 0.884 | 0.718 | 0.185 | 0.974 | - |
| 3 | dates.pause_ms = 250 | 0.884 | 0.718 | 0.185 | 0.974 | - |
| 4 | dates.speed = 0.9 | 0.734 | 0.718 | 0.185 | 0.974 | - |
| 5 | dates.speed = 1.1 | 0.732 | 0.718 | 0.185 | 0.974 | - |
| 6 | dict June -> जुने | 0.685 | 0.597 | 0.289 | 0.711 | - |
