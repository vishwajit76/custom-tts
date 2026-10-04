# Iteration 003: cur-027 (currency, priority 1)

- Problem: pronunciation; text `आपको ₹25,000 देने हैं।`; expected `आपको पच्चीस हज़ार रुपये देने हैं।`
- Previous: normalized `आपको पच्चीस हज़ार रुपये देने हैं।`; ASR ` आपको 25,000 रुपिये देने हैं,`; total 0.668 (pron 0.550, CER 0.375)
- Candidate: currency.speed = 1.1; normalized `आपको पच्चीस हज़ार रुपये देने हैं।`; ASR ` आपको 25,000 रुपिये देने हैं,`
- Score 0.668 -> 0.820; Laya confidence 0.504; regression (worst category drop) 0.0
- Decision: **rejected** (critical regression cur-030 pron 0.661->0.559)
- Affected tests: 29
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | currency.speed = 1.1 | 0.820 | 0.550 | 0.375 | 1.000 | - |
| 2 | currency.speed = 0.9 | 0.819 | 0.550 | 0.375 | 1.000 | - |
| 3 | baseline (no change) | 0.668 | 0.550 | 0.375 | 1.000 | - |
| 4 | currency.pause_ms = 60 | 0.668 | 0.550 | 0.375 | 1.000 | - |
| 5 | currency.pause_ms = 250 | 0.668 | 0.550 | 0.375 | 1.000 | - |
| 6 | currency.punct = conj_comma | 0.668 | 0.550 | 0.375 | 1.000 | - |
| 7 | currency.punct = comma_to_danda | 0.668 | 0.550 | 0.375 | 1.000 | - |
