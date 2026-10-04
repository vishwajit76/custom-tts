# Iteration 018: time-010 (time, priority 1)

- Problem: pronunciation; text `मीटिंग का समय 4:30 है।`; expected `मीटिंग का समय साढ़े चार बजे है।`
- Previous: normalized `मीटिंग का समय साढ़े चार बजे है।`; ASR ` मीटिंग का समय साध्य चार बजे है।`; total 0.858 (pron 0.648, CER 0.130)
- Candidate: time.speed = 1.1; normalized `मीटिंग का समय साढ़े चार बजे है।`; ASR ` मीटिंग का समय साधे चार बजे है।`
- Score 0.858 -> 0.873; Laya confidence 0.574; regression (worst category drop) 0.0058
- Decision: **rejected** (target +0.015 < 0.05; rtf 0.037->0.101)
- Affected tests: 54
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | time.speed = 1.1 | 0.873 | 0.683 | 0.043 | 1.000 | - |
| 2 | baseline (no change) | 0.858 | 0.648 | 0.130 | 1.000 | - |
| 3 | time.pause_ms = 60 | 0.858 | 0.648 | 0.130 | 1.000 | - |
| 4 | time.pause_ms = 250 | 0.858 | 0.648 | 0.130 | 1.000 | - |
| 5 | time.speed = 0.9 | 0.857 | 0.648 | 0.130 | 1.000 | - |
