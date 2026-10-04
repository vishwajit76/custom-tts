# Iteration 015: time-025 (time, priority 1)

- Problem: pronunciation; text `मीटिंग का समय 9:30 है।`; expected `मीटिंग का समय साढ़े नौ बजे है।`
- Previous: normalized `मीटिंग का समय साढ़े नौ बजे है।`; ASR ` मीटिंग का समय साध्य नो बजे है`; total 0.850 (pron 0.627, CER 0.182)
- Candidate: time.speed = 1.1; normalized `मीटिंग का समय साढ़े नौ बजे है।`; ASR ` मीटिंग का समय साधे नो बजे है।`
- Score 0.850 -> 0.865; Laya confidence 0.571; regression (worst category drop) 0.0058
- Decision: **rejected** (target +0.014 < 0.05; rtf 0.037->0.101)
- Affected tests: 54
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | time.speed = 1.1 | 0.865 | 0.664 | 0.091 | 1.000 | - |
| 2 | baseline (no change) | 0.850 | 0.627 | 0.182 | 1.000 | - |
| 3 | time.pause_ms = 60 | 0.850 | 0.627 | 0.182 | 1.000 | - |
| 4 | time.pause_ms = 250 | 0.850 | 0.627 | 0.182 | 1.000 | - |
| 5 | time.speed = 0.9 | 0.845 | 0.627 | 0.182 | 1.000 | - |
