# Iteration 027: num-005 (numbers, priority 1)

- Problem: pronunciation; text `इस गाँव की आबादी 47 है।`; expected `इस गाँव की आबादी सैंतालीस है।`
- Previous: normalized `इस गाँव की आबादी सैंतालीस है।`; ASR ` इस गांव की आबादी 47 है।`; total 0.878 (pron 0.700, CER 0.000)
- Candidate: baseline (no change); normalized `इस गाँव की आबादी सैंतालीस है।`; ASR ` इस गांव की आबादी 47 है।`
- Score 0.878 -> 0.878; Laya confidence 0.5; regression (worst category drop) 0.0
- Decision: **rejected** (no candidate beat the current config)
- Affected tests: 0
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | baseline (no change) | 0.878 | 0.700 | 0.000 | 1.000 | - |
| 2 | numbers.pause_ms = 60 | 0.878 | 0.700 | 0.000 | 1.000 | - |
| 3 | numbers.pause_ms = 250 | 0.878 | 0.700 | 0.000 | 1.000 | - |
| 4 | numbers.speed = 1.1 | 0.876 | 0.700 | 0.000 | 1.000 | - |
| 5 | numbers.speed = 0.9 | 0.727 | 0.700 | 0.000 | 1.000 | - |
