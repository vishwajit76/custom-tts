# Iteration 023: time-004 (time, priority 1)

- Problem: pronunciation; text `मीटिंग का समय 2:30 है।`; expected `मीटिंग का समय ढाई बजे है।`
- Previous: normalized `मीटिंग का समय ढाई बजे है।`; ASR ` मीटिंग का समय धाई बजे है`; total 0.871 (pron 0.679, CER 0.053)
- Candidate: baseline (no change); normalized `मीटिंग का समय ढाई बजे है।`; ASR ` मीटिंग का समय धाई बजे है`
- Score 0.871 -> 0.871; Laya confidence 0.5; regression (worst category drop) 0.0
- Decision: **rejected** (no candidate beat the current config)
- Affected tests: 0
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | baseline (no change) | 0.871 | 0.679 | 0.053 | 1.000 | - |
| 2 | time.speed = 1.1 | 0.871 | 0.679 | 0.053 | 1.000 | - |
| 3 | time.pause_ms = 60 | 0.871 | 0.679 | 0.053 | 1.000 | - |
| 4 | time.pause_ms = 250 | 0.871 | 0.679 | 0.053 | 1.000 | - |
| 5 | time.speed = 0.9 | 0.862 | 0.679 | 0.053 | 1.000 | - |
