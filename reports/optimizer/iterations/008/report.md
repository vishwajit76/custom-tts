# Iteration 008: phone-000 (phone_numbers, priority 1)

- Problem: pronunciation; text `मेरा नंबर 9224491505 है।`; expected `मेरा नंबर नौ दो दो चार चार नौ एक पाँच शून्य पाँच है।`
- Previous: normalized `मेरा नंबर नौ दो दो चार चार, नौ एक पाँच शून्य पाँच है।`; ASR ` मेरा नमबर 9224491505 है।`; total 0.718 (pron 0.680, CER 0.050)
- Candidate: phone_numbers.speed = 0.9; normalized `मेरा नंबर नौ दो दो चार चार, नौ एक पाँच शून्य पाँच है।`; ASR ` मेरा नमबर नो दो दो चार चार नो एक पांच शून्य पांच है।`
- Score 0.718 -> 0.793; Laya confidence 0.614; regression (worst category drop) 0.0
- Decision: **accepted** (gates passed)
- Affected tests: 24
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | phone_numbers.speed = 0.9 | 0.793 | 0.869 | 0.077 | 1.000 | - |
| 2 | baseline (no change) | 0.718 | 0.680 | 0.050 | 1.000 | - |
| 3 | phone_numbers.pause_ms = 60 | 0.718 | 0.680 | 0.050 | 1.000 | - |
| 4 | phone_numbers.pause_ms = 250 | 0.718 | 0.680 | 0.050 | 1.000 | - |
| 5 | phone_numbers.speed = 1.1 | 0.717 | 0.680 | 0.050 | 1.000 | - |
