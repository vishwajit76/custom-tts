# Iteration 006: num-024 (numbers, priority 1)

- Problem: pronunciation; text `मेरे पास 125000 किताबें हैं।`; expected `मेरे पास एक लाख पच्चीस हज़ार किताबें हैं।`
- Previous: normalized `मेरे पास एक लाख पच्चीस हज़ार किताबें हैं।`; ASR ` मेरे पास एक लाख पचीस हजार किताबे हैं।`; total 0.713 (pron 0.662, CER 0.094)
- Candidate: baseline (no change); normalized `मेरे पास एक लाख पच्चीस हज़ार किताबें हैं।`; ASR ` मेरे पास एक लाख पचीस हजार किताबे हैं।`
- Score 0.713 -> 0.713; Laya confidence 0.502; regression (worst category drop) 0.0
- Decision: **rejected** (no candidate beat the current config)
- Affected tests: 0
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | baseline (no change) | 0.713 | 0.662 | 0.094 | 1.000 | - |
| 2 | numbers.speed = 1.1 | 0.713 | 0.662 | 0.094 | 1.000 | - |
| 3 | numbers.pause_ms = 60 | 0.713 | 0.662 | 0.094 | 1.000 | - |
| 4 | numbers.pause_ms = 250 | 0.713 | 0.662 | 0.094 | 1.000 | - |
| 5 | numbers.speed = 0.9 | 0.712 | 0.662 | 0.094 | 1.000 | - |
