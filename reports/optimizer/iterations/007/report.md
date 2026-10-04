# Iteration 007: phone-001 (phone_numbers, priority 1)

- Problem: pronunciation; text `मेरा नंबर 69983 92161 है।`; expected `मेरा नंबर छह नौ नौ आठ तीन नौ दो एक छह एक है।`
- Previous: normalized `मेरा नंबर छह नौ नौ आठ तीन, नौ दो एक छह एक है।`; ASR ` मेरा नमबर 6998392161 है।`; total 0.716 (pron 0.680, CER 0.050)
- Candidate: baseline (no change); normalized `मेरा नंबर छह नौ नौ आठ तीन, नौ दो एक छह एक है।`; ASR ` मेरा नमबर 6998392161 है।`
- Score 0.716 -> 0.716; Laya confidence 0.496; regression (worst category drop) 0.0
- Decision: **rejected** (no candidate beat the current config)
- Affected tests: 0
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | baseline (no change) | 0.716 | 0.680 | 0.050 | 1.000 | - |
| 2 | phone_numbers.speed = 1.1 | 0.716 | 0.680 | 0.050 | 1.000 | - |
| 3 | phone_numbers.pause_ms = 60 | 0.716 | 0.680 | 0.050 | 1.000 | - |
| 4 | phone_numbers.pause_ms = 250 | 0.716 | 0.680 | 0.050 | 1.000 | - |
| 5 | phone_numbers.speed = 0.9 | 0.708 | 0.680 | 0.050 | 1.000 | - |
