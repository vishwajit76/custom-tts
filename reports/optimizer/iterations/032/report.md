# Iteration 032: hg-027 (hinglish, priority 2)

- Problem: pronunciation; text `Form में date of birth वही लिखिए जो Aadhaar card पर है।`; expected `Form में date of birth वही लिखिए जो Aadhaar card पर है।`
- Previous: normalized `Form में date of birth वही लिखिए जो Aadhaar card पर है।`; ASR ` फॉर्म में डेट अव बर्थ वही लिखिये जो अधार कार्ट पर है।`; total 0.634 (pron 0.473, CER 0.674)
- Candidate: hinglish.speed = 1.1; normalized `Form में date of birth वही लिखिए जो Aadhaar card पर है।`; ASR ` फॉर्म में डेट अव बर्थ वही लिखिये जो अधार कार्ट पर है`
- Score 0.634 -> 0.786; Laya confidence 0.534; regression (worst category drop) 0.0
- Decision: **accepted** (gates passed)
- Affected tests: 58
- Laya on chosen: backend off
- Audio: before.wav, after.wav (not committed)

| rank | change | total | pron | CER | text | laya quality / acceptable / failure |
|---|---|---|---|---|---|---|
| 1 | hinglish.speed = 1.1 | 0.786 | 0.473 | 0.674 | 1.000 | - |
| 2 | dict of -> ओफ़ | 0.779 | 0.459 | 0.674 | 0.954 | - |
| 3 | dict date -> दते | 0.775 | 0.445 | 0.674 | 0.907 | - |
| 4 | dict card -> कर्द | 0.774 | 0.445 | 0.674 | 0.907 | - |
| 5 | dict birth -> बिर्थ | 0.774 | 0.438 | 0.674 | 0.884 | - |
| 6 | dict Aadhaar -> आधार | 0.766 | 0.424 | 0.674 | 0.837 | - |
| 7 | hinglish.speed = 0.9 | 0.635 | 0.473 | 0.674 | 1.000 | - |
| 8 | baseline (no change) | 0.634 | 0.473 | 0.674 | 1.000 | - |
| 9 | hinglish.pause_ms = 60 | 0.634 | 0.473 | 0.674 | 1.000 | - |
| 10 | hinglish.pause_ms = 250 | 0.634 | 0.473 | 0.674 | 1.000 | - |
| 11 | dict Form -> फ़ोर्म | 0.620 | 0.438 | 0.674 | 0.884 | - |
