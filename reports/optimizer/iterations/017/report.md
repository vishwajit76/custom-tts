# Iteration 017: time-028 (time, priority 1)

- Problem: pronunciation; text `मीटिंग का समय 10:30 है।`; expected `मीटिंग का समय साढ़े दस बजे है।`
- Previous: normalized `मीटिंग का समय साढ़े दस बजे है।`; ASR ` मीटिंग का समय साध्य दस बजे है`; total 0.854 (pron 0.645, CER 0.136)
- Candidate: time.speed = 1.1; normalized `मीटिंग का समय साढ़े दस बजे है।`; ASR ` मीटिंग का समय साधे दस बजे है`
- Score 0.854 -> 0.870; Laya confidence 0.581; regression (worst category drop) 0.0058
- Decision: **rejected** (target +0.016 < 0.05; rtf 0.037->0.101)
- Affected tests: 54
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | time.speed = 1.1 | 0.870 | 0.682 | 0.045 | 1.000 | - |
| 2 | baseline (no change) | 0.854 | 0.645 | 0.136 | 1.000 | - |
| 3 | time.pause_ms = 60 | 0.854 | 0.645 | 0.136 | 1.000 | - |
| 4 | time.pause_ms = 250 | 0.854 | 0.645 | 0.136 | 1.000 | - |
| 5 | time.speed = 0.9 | 0.851 | 0.645 | 0.136 | 1.000 | - |
