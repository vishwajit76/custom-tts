# Iteration 030: name-000 (names, priority 1)

- Problem: normalization; text `मेरा नाम Vishvajeet है।`; expected `मेरा नाम विश्वजीत है।`
- Previous: normalized `मेरा नाम Vishvajeet है।`; ASR ` मेरा नाम विश्वजीत है।`; total 0.929 (pron 0.824, CER 0.000)
- Candidate: dict Vishvajeet -> विश्वजीत; normalized `मेरा नाम विश्वजीत है।`; ASR ` मेरा नाम विश्वजीत है।`
- Score 0.929 -> 1.000; Laya confidence 0.855; regression (worst category drop) 0.0
- Decision: **accepted** (gates passed)
- Affected tests: 1
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | dict Vishvajeet -> विश्वजीत | 1.000 | 1.000 | 0.000 | 1.000 | - |
| 2 | baseline (no change) | 0.929 | 0.824 | 0.000 | 0.412 | - |
| 3 | names.speed = 0.9 | 0.929 | 0.824 | 0.000 | 0.412 | - |
| 4 | names.pause_ms = 60 | 0.929 | 0.824 | 0.000 | 0.412 | - |
| 5 | names.pause_ms = 250 | 0.929 | 0.824 | 0.000 | 0.412 | - |
| 6 | names.speed = 1.1 | 0.928 | 0.824 | 0.000 | 0.412 | - |
