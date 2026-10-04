# Iteration 016: time-031 (time, priority 1)

- Problem: pronunciation; text `मीटिंग का समय 11:30 है।`; expected `मीटिंग का समय साढ़े ग्यारह बजे है।`
- Previous: normalized `मीटिंग का समय साढ़े ग्यारह बजे है।`; ASR ` मीटिंग का समय साध्य ग्यारा बजे है`; total 0.853 (pron 0.638, CER 0.154)
- Candidate: time.speed = 1.1; normalized `मीटिंग का समय साढ़े ग्यारह बजे है।`; ASR ` मीटिंग का समय साधी ग्यारा बजे है`
- Score 0.853 -> 0.861; Laya confidence 0.536; regression (worst category drop) 0.0058
- Decision: **rejected** (target +0.007 < 0.05; rtf 0.037->0.101)
- Affected tests: 54
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | time.speed = 1.1 | 0.861 | 0.654 | 0.115 | 1.000 | - |
| 2 | time.speed = 0.9 | 0.854 | 0.638 | 0.154 | 1.000 | - |
| 3 | baseline (no change) | 0.853 | 0.638 | 0.154 | 1.000 | - |
| 4 | time.pause_ms = 60 | 0.853 | 0.638 | 0.154 | 1.000 | - |
| 5 | time.pause_ms = 250 | 0.853 | 0.638 | 0.154 | 1.000 | - |
