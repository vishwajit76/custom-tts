# Iteration 022: time-022 (time, priority 1)

- Problem: pronunciation; text `मीटिंग का समय 8:30 है।`; expected `मीटिंग का समय साढ़े आठ बजे है।`
- Previous: normalized `मीटिंग का समय साढ़े आठ बजे है।`; ASR ` मीटिंग का समय साधे आठ बजे है`; total 0.870 (pron 0.682, CER 0.045)
- Candidate: baseline (no change); normalized `मीटिंग का समय साढ़े आठ बजे है।`; ASR ` मीटिंग का समय साधे आठ बजे है`
- Score 0.870 -> 0.870; Laya confidence 0.499; regression (worst category drop) 0.0
- Decision: **rejected** (no candidate beat the current config)
- Affected tests: 0
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | baseline (no change) | 0.870 | 0.682 | 0.045 | 1.000 | - |
| 2 | time.speed = 0.9 | 0.870 | 0.682 | 0.045 | 1.000 | - |
| 3 | time.speed = 1.1 | 0.870 | 0.682 | 0.045 | 1.000 | - |
| 4 | time.pause_ms = 60 | 0.870 | 0.682 | 0.045 | 1.000 | - |
| 5 | time.pause_ms = 250 | 0.870 | 0.682 | 0.045 | 1.000 | - |
