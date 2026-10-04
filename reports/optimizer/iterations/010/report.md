# Iteration 010: num-017 (numbers, priority 1)

- Problem: pronunciation; text `इस गाँव की आबादी 1,500 है।`; expected `इस गाँव की आबादी एक हज़ार पाँच सौ है।`
- Previous: normalized `इस गाँव की आबादी एक हज़ार पाँच सौ है।`; ASR ` इस गांव की आबादी 1500 है।`; total 0.730 (pron 0.700, CER 0.000)
- Candidate: baseline (no change); normalized `इस गाँव की आबादी एक हज़ार पाँच सौ है।`; ASR ` इस गांव की आबादी 1500 है।`
- Score 0.730 -> 0.730; Laya confidence 0.5; regression (worst category drop) 0.0
- Decision: **rejected** (no candidate beat the current config)
- Affected tests: 0
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | baseline (no change) | 0.730 | 0.700 | 0.000 | 1.000 | - |
| 2 | numbers.speed = 0.9 | 0.730 | 0.700 | 0.000 | 1.000 | - |
| 3 | numbers.pause_ms = 60 | 0.730 | 0.700 | 0.000 | 1.000 | - |
| 4 | numbers.pause_ms = 250 | 0.730 | 0.700 | 0.000 | 1.000 | - |
| 5 | numbers.punct = conj_comma | 0.730 | 0.700 | 0.000 | 1.000 | - |
| 6 | numbers.punct = comma_to_danda | 0.730 | 0.700 | 0.000 | 1.000 | - |
| 7 | numbers.speed = 1.1 | 0.729 | 0.700 | 0.000 | 1.000 | - |
