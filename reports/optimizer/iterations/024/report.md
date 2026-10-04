# Iteration 024: time-013 (time, priority 1)

- Problem: pronunciation; text `मीटिंग का समय 5:30 है।`; expected `मीटिंग का समय साढ़े पाँच बजे है।`
- Previous: normalized `मीटिंग का समय साढ़े पाँच बजे है।`; ASR ` मीटिंग का समय साधे पांच बजे है।`; total 0.871 (pron 0.683, CER 0.042)
- Candidate: time.speed = 1.1; normalized `मीटिंग का समय साढ़े पाँच बजे है।`; ASR ` मीटिंग का समय साधे पांच बजे है`
- Score 0.871 -> 0.873; Laya confidence 0.51; regression (worst category drop) 0.0058
- Decision: **rejected** (target +0.002 < 0.05)
- Affected tests: 54
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | time.speed = 1.1 | 0.873 | 0.683 | 0.042 | 1.000 | - |
| 2 | baseline (no change) | 0.871 | 0.683 | 0.042 | 1.000 | - |
| 3 | time.pause_ms = 60 | 0.871 | 0.683 | 0.042 | 1.000 | - |
| 4 | time.pause_ms = 250 | 0.871 | 0.683 | 0.042 | 1.000 | - |
| 5 | time.speed = 0.9 | 0.857 | 0.650 | 0.125 | 1.000 | - |
