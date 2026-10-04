# Iteration 013: acr-000 (acronyms, priority 1)

- Problem: pronunciation; text `कृपया API की जानकारी दीजिए।`; expected `कृपया ए पी आई की जानकारी दीजिए।`
- Previous: normalized `कृपया ए पी आई की जानकारी दीजिए।`; ASR ` क्रिप्याई API की जानकारी दीजिये.`; total 0.823 (pron 0.573, CER 0.318)
- Candidate: acronyms.speed = 1.1; normalized `कृपया ए पी आई की जानकारी दीजिए।`; ASR ` क्रिप्या API की जानकारी दीजिये.`
- Score 0.823 -> 0.831; Laya confidence 0.542; regression (worst category drop) 0.0001
- Decision: **rejected** (target +0.008 < 0.05)
- Affected tests: 22
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | acronyms.speed = 1.1 | 0.831 | 0.591 | 0.273 | 1.000 | - |
| 2 | baseline (no change) | 0.823 | 0.573 | 0.318 | 1.000 | - |
