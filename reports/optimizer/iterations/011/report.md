# Iteration 011: phone-002 (phone_numbers, priority 1)

- Problem: pronunciation; text `मेरा नंबर 67448-49435 है।`; expected `मेरा नंबर छह सात चार चार आठ चार नौ चार तीन पाँच है।`
- Previous: normalized `मेरा नंबर छह सात चार चार आठ, चार नौ चार तीन पाँच है।`; ASR ` मेरा नमबर छे साथ चार चार आठ चार नो चार तीन पांच है।`; total 0.749 (pron 0.758, CER 0.105)
- Candidate: phone_numbers.speed = 1.0; normalized `मेरा नंबर छह सात चार चार आठ, चार नौ चार तीन पाँच है।`; ASR ` मेरा नंबर छे साथ चार चार आठ चार नो चार तीन पांच है।`
- Score 0.749 -> 0.753; Laya confidence 0.497; regression (worst category drop) 0.0102
- Decision: **rejected** (target +0.004 < 0.05; critical regression phone-000 pron 0.869->0.680; category phone_numbers 0.754->0.743)
- Affected tests: 24
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | phone_numbers.speed = 1.0 | 0.753 | 0.768 | 0.079 | 1.000 | - |
| 2 | phone_numbers.speed = 1.1 | 0.753 | 0.768 | 0.079 | 1.000 | - |
| 3 | baseline (no change) | 0.749 | 0.758 | 0.105 | 1.000 | - |
| 4 | rule hyphen_phone | 0.749 | 0.758 | 0.105 | 1.000 | - |
| 5 | phone_numbers.pause_ms = 60 | 0.749 | 0.758 | 0.105 | 1.000 | - |
| 6 | phone_numbers.pause_ms = 250 | 0.749 | 0.758 | 0.105 | 1.000 | - |
