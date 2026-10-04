# Iteration 019: acr-005 (acronyms, priority 1)

- Problem: pronunciation; text `OTP से जुड़ी समस्या बताइए।`; expected `ओ टी पी से जुड़ी समस्या बताइए।`
- Previous: normalized `ओ टी पी से जुड़ी समस्या बताइए।`; ASR ` OTP से जोड़ी समस्या बताईए.`; total 0.864 (pron 0.660, CER 0.100)
- Candidate: baseline (no change); normalized `ओ टी पी से जुड़ी समस्या बताइए।`; ASR ` OTP से जोड़ी समस्या बताईए.`
- Score 0.864 -> 0.864; Laya confidence 0.5; regression (worst category drop) 0.0
- Decision: **rejected** (no candidate beat the current config)
- Affected tests: 0
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | baseline (no change) | 0.864 | 0.660 | 0.100 | 1.000 | - |
| 2 | dict OTP -> ओ टी पी | 0.864 | 0.660 | 0.100 | 1.000 | - |
| 3 | acronyms.speed = 0.9 | 0.864 | 0.660 | 0.100 | 1.000 | - |
| 4 | acronyms.speed = 1.1 | 0.864 | 0.660 | 0.100 | 1.000 | - |
| 5 | acronyms.pause_ms = 60 | 0.864 | 0.660 | 0.100 | 1.000 | - |
| 6 | acronyms.pause_ms = 250 | 0.864 | 0.660 | 0.100 | 1.000 | - |
| 7 | dict OTP -> ओत्प | 0.823 | 0.568 | 0.227 | 0.864 | - |
