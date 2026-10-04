# Iteration 026: time-007 (time, priority 1)

- Problem: pronunciation; text `मीटिंग का समय 3:30 है।`; expected `मीटिंग का समय साढ़े तीन बजे है।`
- Previous: normalized `मीटिंग का समय साढ़े तीन बजे है।`; ASR ` मीटिंग का समय साधे तीन बजे है`; total 0.873 (pron 0.683, CER 0.043)
- Candidate: baseline (no change); normalized `मीटिंग का समय साढ़े तीन बजे है।`; ASR ` मीटिंग का समय साधे तीन बजे है`
- Score 0.873 -> 0.873; Laya confidence 0.5; regression (worst category drop) 0.0
- Decision: **rejected** (no candidate beat the current config)
- Affected tests: 0
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | baseline (no change) | 0.873 | 0.683 | 0.043 | 1.000 | - |
| 2 | time.pause_ms = 60 | 0.873 | 0.683 | 0.043 | 1.000 | - |
| 3 | time.pause_ms = 250 | 0.873 | 0.683 | 0.043 | 1.000 | - |
| 4 | time.speed = 0.9 | 0.872 | 0.683 | 0.043 | 1.000 | - |
| 5 | time.speed = 1.1 | 0.872 | 0.683 | 0.043 | 1.000 | - |
