# Iteration 005: cur-030 (currency, priority 1)

- Problem: pronunciation; text `आपको ₹1,25,000 देने हैं।`; expected `आपको एक लाख पच्चीस हज़ार रुपये देने हैं।`
- Previous: normalized `आपको एक लाख पच्चीस हज़ार रुपये देने हैं।`; ASR ` आपको एक लाख पचीस हजार रुपिये देने हैं।`; total 0.712 (pron 0.661, CER 0.097)
- Candidate: baseline (no change); normalized `आपको एक लाख पच्चीस हज़ार रुपये देने हैं।`; ASR ` आपको एक लाख पचीस हजार रुपिये देने हैं।`
- Score 0.712 -> 0.712; Laya confidence 0.5; regression (worst category drop) 0.0
- Decision: **rejected** (no candidate beat the current config)
- Affected tests: 0
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | baseline (no change) | 0.712 | 0.661 | 0.097 | 1.000 | - |
| 2 | currency.pause_ms = 60 | 0.712 | 0.661 | 0.097 | 1.000 | - |
| 3 | currency.pause_ms = 250 | 0.712 | 0.661 | 0.097 | 1.000 | - |
| 4 | currency.punct = conj_comma | 0.712 | 0.661 | 0.097 | 1.000 | - |
| 5 | currency.punct = comma_to_danda | 0.712 | 0.661 | 0.097 | 1.000 | - |
| 6 | currency.speed = 0.9 | 0.671 | 0.559 | 0.353 | 1.000 | - |
| 7 | currency.speed = 1.1 | 0.671 | 0.559 | 0.353 | 1.000 | - |
