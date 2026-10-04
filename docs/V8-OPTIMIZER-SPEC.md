# V8 Autonomous Hindi TTS Optimization — Spec (from user, 2026-10-04)

Strategy: agent → hypothesis → modify preprocessing/pronunciation/prosody → generate audio → evaluate → Laya ranking/decision → regression → accept/reject → repeat.
GPU training/fine-tuning = LAST RESORT, only after the loop proves a systematic model-level limitation. Do NOT start training. Do not replace the existing model/inference pipeline; build around it.

Maximize: pronunciation, naturalness, prosody, intelligibility, rhythm, punctuation/pauses, numbers/dates/currency, Hinglish/code-switching, proper names, conversational quality, audio consistency, inference speed, concurrency.

## Requirements
1. Audit first: model, inference pipeline, tokenizer, phonemizer, normalization, pronunciation, audio pre/post, eval/bench scripts, model format, V7/V8 changes. Internal architecture report before major changes.
2. Resumable optimization loop (crash at iteration 47 → resume at 47): test set → pick difficult case → candidates → TTS → ASR/pronunciation eval → audio eval → Laya ranking → score → failure analysis → hypothesis → modify rule/config/code → regression → keep or revert.
3. Laya = fast evaluation/ranking/decision component (NOT a generator): candidate ranking, pronunciation-quality classification, confidence, best-variant selection, failure-type classification. Never trusted alone. Abstraction: `class PronunciationEvaluator: evaluate(candidate, reference)`; `class LayaEvaluator(PronunciationEvaluator)`; swappable.
4. Multi-layer eval: (A) text correctness (normalized text, expected pronunciation, phonemes); (B) ASR: WER, CER, token accuracy, important-word accuracy; detect word splits like विश्वजीत → विश्व जीत.
5. Audio metrics: duration, silence/speech ratio, clipping, RMS, loudness, peak, abnormal silence, repeated artifacts, noise, discontinuities → audio quality score. Cheap.
6. Structured benchmark with categories: basic_hindi, difficult_hindi, numbers, currency, dates, time, phone_numbers, names, proper_nouns, english_words, hinglish, abbreviations, acronyms, punctuation, questions, commands, conversational, long_sentences, edge_cases. JSON per case: id, text, expected_normalized, category, priority. Hundreds, programmatic where possible.
7. Controlled candidate generation along dimensions: normalization, phoneme variant, word boundary, pause, prosody, punctuation, stress, speed. Beam-search-like, bounded, not random.
8. Deterministic normalizer: numbers (0..10,00,000 Indian grouping), currency (₹500, ₹25,000, ₹1,25,000, INR 500, Rs. 500), dates (04/10/2026, 4 October 2026, 2026-10-04), time (10:30, 10:30 AM/PM), phone numbers (natural spoken Hindi), percentages, English/tech terms (API, AI, CRM, WhatsApp, SIP, TTS, OTP, URL, HTTP) via configurable dictionary, not blind transliteration.
9. Persistent pronunciation dictionary (e.g. {"WhatsApp": {"spoken": "व्हाट्सऐप", "confidence": 0.97}}); optimizer can add/update; every auto change records reason, score before/after, confidence, affected tests, timestamp, iteration.
10. Configurable prosody layer: sentence/comma/full-stop pauses, question intonation, exclamation, emphasis, splitting, conversational rhythm; variants evaluated automatically.
11. Intelligent chunking aware of commas, conjunctions, questions, quotes, numbers, names, long clauses; output stays one coherent utterance.
12. Weighted score in config: pronunciation 0.40, naturalness 0.25, prosody 0.15, audio_quality 0.10, consistency 0.10. Extensible.
13. Agent role per failure: inspect test/candidates/metrics → root cause → hypothesis → smallest change → targeted test → regression → keep only if improved without regressions.
14. Git checkpoints per accepted optimization (experiment/iteration-NNN branches or commits); checkpoint before, rollback if worse; never destroy the baseline.
15. Regression protection mandatory: accept only if target improvement > threshold AND regression < allowed threshold.
16. Experiment DB (SQLite), queryable: iteration, test_id, change, before/after score, regression score, decision, confidence. Answer: what was tested / failed / worked, repeat-failing words, lowest categories.
17. Learn from history: search similar past failures before new hypotheses; generalize proven rules carefully.
18. CLI: `python optimize.py [--category X] [--test X] [--iterations N] [--resume] [--benchmark]`, `python evaluate.py`, `python benchmark.py`.
19. Human review report per iteration (problem, previous, candidate, score before→after, ASR, Laya confidence, regression, decision) + saved audio for comparison.
20. Training policy: only after normalization, rules, phonemes, dictionary, segmentation, prosody, inference config, audio processing, candidate search all fail repeatedly.
21. Performance: cache audio/TTS/eval by hash(text + pronunciation_config + model_version), dedupe experiments, parallel candidate eval, early rejection, incremental benchmarking.
22. Configurable concurrency (defaults for 16 GB RAM Windows): max_candidates 12, inference_workers 2, evaluation_workers 4. Don't overload GPU.
23. Configurable quality gates: target improvement >= 0.05, no critical pronunciation regression, no category regression > threshold, inference perf degradation < threshold.
24. Lightweight JSON/HTML report: overall/pronunciation/naturalness/prosody/audio/regression scores, top improvements/failures, problem words/categories, recent experiments, accepted/rejected rules.
25. Agent rules: one system per experiment; measurable hypothesis; regression must pass; never delete previous best config; LLM judgment ≠ ground truth; objective metrics first; cache everything expensive; deterministic rules over repeated LLM calls; no training without evidence; reproducible.

## First milestone
Working automated pronunciation evaluation loop: existing model + candidate generation + ASR/audio metrics + Laya ranking + regression testing. Then iterate on preprocessing/pronunciation/prosody/inference.
