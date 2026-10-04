# Iteration 021: time-034 (time, priority 1)

- Problem: pronunciation; text `मीटिंग का समय 12:30 है।`; expected `मीटिंग का समय साढ़े बारह बजे है।`
- Previous: normalized `मीटिंग का समय साढ़े बारह बजे है।`; ASR ` मीटिंग का समय साधे बारा बजे है`; total 0.865 (pron 0.667, CER 0.083)
- Candidate: time.speed = 1.1; normalized `मीटिंग का समय साढ़े बारह बजे है।`; ASR ` मीटिंग का समय साधे बारा बजे है`
- Score 0.865 -> 0.866; Laya confidence 0.503; regression (worst category drop) 0.0058
- Decision: **rejected** (target +0.001 < 0.05; rtf 0.059->0.080)
- Affected tests: 54
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | time.speed = 1.1 | 0.866 | 0.667 | 0.083 | 1.000 | - |
| 2 | baseline (no change) | 0.865 | 0.667 | 0.083 | 1.000 | - |
| 3 | time.speed = 0.9 | 0.865 | 0.667 | 0.083 | 1.000 | - |
| 4 | time.pause_ms = 60 | 0.865 | 0.667 | 0.083 | 1.000 | - |
| 5 | time.pause_ms = 250 | 0.865 | 0.667 | 0.083 | 1.000 | - |
