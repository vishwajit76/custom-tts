# Iteration 012: phone-000 (phone_numbers, priority 1)

- Problem: pronunciation; text `मेरा नंबर 9224491505 है।`; expected `मेरा नंबर नौ दो दो चार चार नौ एक पाँच शून्य पाँच है।`
- Previous: normalized `मेरा नंबर नौ दो दो चार चार, नौ एक पाँच शून्य पाँच है।`; ASR ` मेरा नमबर नो दो दो चार चार नो एक पांच शून्य पांच है।`; total 0.793 (pron 0.869, CER 0.077)
- Candidate: baseline (no change); normalized `मेरा नंबर नौ दो दो चार चार, नौ एक पाँच शून्य पाँच है।`; ASR ` मेरा नमबर नो दो दो चार चार नो एक पांच शून्य पांच है।`
- Score 0.793 -> 0.793; Laya confidence 0.614; regression (worst category drop) 0.0
- Decision: **rejected** (no candidate beat the current config)
- Affected tests: 0
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | baseline (no change) | 0.793 | 0.869 | 0.077 | 1.000 | - |
| 2 | phone_numbers.speed = 1.0 | 0.718 | 0.680 | 0.050 | 1.000 | - |
