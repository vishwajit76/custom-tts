# Iteration 002: en-000 (english_words, priority 1)

- Problem: normalization; text `कृपया WhatsApp की जानकारी दीजिए।`; expected `कृपया व्हाट्सऐप की जानकारी दीजिए।`
- Previous: normalized `कृपया WhatsApp की जानकारी दीजिए।`; ASR ` क्रिपिया वर्ट्स एप की जानकारी दीजिये.`; total 0.634 (pron 0.461, CER 0.357)
- Candidate: dict WhatsApp -> व्हाट्सऐप; normalized `कृपया व्हाट्सऐप की जानकारी दीजिए।`; ASR ` कृपियाव हाट सैप की जानकारी दीजिये।`
- Score 0.634 -> 0.695; Laya confidence 0.372; regression (worst category drop) 0.0
- Decision: **accepted** (gates passed)
- Affected tests: 1
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | dict WhatsApp -> व्हाट्सऐप | 0.695 | 0.614 | 0.214 | 1.000 | - |
| 2 | dict WhatsApp -> व्हत्सप्प | 0.689 | 0.598 | 0.148 | 0.857 | - |
| 3 | dict WhatsApp -> वाट्सैप | 0.687 | 0.596 | 0.179 | 0.893 | - |
| 4 | english_words.speed = 0.9 | 0.651 | 0.504 | 0.250 | 0.679 | - |
| 5 | baseline (no change) | 0.634 | 0.461 | 0.357 | 0.679 | - |
| 6 | english_words.pause_ms = 60 | 0.634 | 0.461 | 0.357 | 0.679 | - |
| 7 | english_words.pause_ms = 250 | 0.634 | 0.461 | 0.357 | 0.679 | - |
| 8 | english_words.speed = 1.1 | 0.633 | 0.461 | 0.357 | 0.679 | - |
