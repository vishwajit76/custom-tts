# Iteration 020: acr-014 (acronyms, priority 1)

- Problem: pronunciation; text `UPI से जुड़ी समस्या बताइए।`; expected `यू पी आई से जुड़ी समस्या बताइए।`
- Previous: normalized `यू पी आई से जुड़ी समस्या बताइए।`; ASR ` UPI से जोड़ी समस्या बताईए।`; total 0.864 (pron 0.660, CER 0.100)
- Candidate: acronyms.speed = 1.1; normalized `यू पी आई से जुड़ी समस्या बताइए।`; ASR ` UPI से जुड़ी समस्या बताईए।`
- Score 0.864 -> 0.871; Laya confidence 0.538; regression (worst category drop) 0.0001
- Decision: **rejected** (target +0.008 < 0.05)
- Affected tests: 22
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | acronyms.speed = 1.1 | 0.871 | 0.680 | 0.050 | 1.000 | - |
| 2 | baseline (no change) | 0.864 | 0.660 | 0.100 | 1.000 | - |
| 3 | dict UPI -> यू पी आई | 0.864 | 0.660 | 0.100 | 1.000 | - |
| 4 | acronyms.pause_ms = 60 | 0.864 | 0.660 | 0.100 | 1.000 | - |
| 5 | acronyms.pause_ms = 250 | 0.864 | 0.660 | 0.100 | 1.000 | - |
| 6 | acronyms.speed = 0.9 | 0.863 | 0.660 | 0.100 | 1.000 | - |
| 7 | dict UPI -> उपी | 0.816 | 0.548 | 0.250 | 0.826 | - |
