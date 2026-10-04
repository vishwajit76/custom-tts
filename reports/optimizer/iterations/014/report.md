# Iteration 014: time-019 (time, priority 1)

- Problem: pronunciation; text `मीटिंग का समय 7:30 है।`; expected `मीटिंग का समय साढ़े सात बजे है।`
- Previous: normalized `मीटिंग का समय साढ़े सात बजे है।`; ASR ` मीटिंग का समय साध्य साथ बजे है`; total 0.846 (pron 0.630, CER 0.174)
- Candidate: time.speed = 1.1; normalized `मीटिंग का समय साढ़े सात बजे है।`; ASR ` मीटिंग का समय साढ़े साथ बजे है`
- Score 0.846 -> 0.989; Laya confidence 1.0; regression (worst category drop) 0.0058
- Decision: **rejected** (rtf 0.037->0.101)
- Affected tests: 54
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | time.speed = 1.1 | 0.989 | 0.983 | 0.043 | 1.000 | - |
| 2 | time.speed = 0.9 | 0.847 | 0.630 | 0.174 | 1.000 | - |
| 3 | baseline (no change) | 0.846 | 0.630 | 0.174 | 1.000 | - |
| 4 | time.pause_ms = 60 | 0.846 | 0.630 | 0.174 | 1.000 | - |
| 5 | time.pause_ms = 250 | 0.846 | 0.630 | 0.174 | 1.000 | - |
