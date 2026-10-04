# Iteration 004: cur-015 (currency, priority 1)

- Problem: pronunciation; text `आपको ₹500 देने हैं।`; expected `आपको पाँच सौ रुपये देने हैं।`
- Previous: normalized `आपको पाँच सौ रुपये देने हैं।`; ASR ` आपको 500 रुपिये देने हैं।`; total 0.676 (pron 0.573, CER 0.318)
- Candidate: currency.speed = 1.1; normalized `आपको पाँच सौ रुपये देने हैं।`; ASR ` आपको 500 रुपिये देने हैं,`
- Score 0.676 -> 0.825; Laya confidence 1.0; regression (worst category drop) 0.0
- Decision: **rejected** (critical regression cur-030 pron 0.661->0.559)
- Affected tests: 29
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | currency.speed = 1.1 | 0.825 | 0.573 | 0.318 | 1.000 | - |
| 2 | baseline (no change) | 0.676 | 0.573 | 0.318 | 1.000 | - |
| 3 | currency.pause_ms = 60 | 0.676 | 0.573 | 0.318 | 1.000 | - |
| 4 | currency.pause_ms = 250 | 0.676 | 0.573 | 0.318 | 1.000 | - |
| 5 | currency.speed = 0.9 | 0.667 | 0.557 | 0.357 | 1.000 | - |
