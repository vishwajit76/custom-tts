# Agent C log (goonj totals; proxy scores only)
Sweep (reports/agents/c_exp.py): pause_ms 60/250/400, speed .9/.95/1.05, punct conj_comma/comma_to_danda, per category.
- pause_ms: no effect >0.003 (chunks rarely >1/utt; p250 +0.003 long) -> rejected.
- conj_comma: worse everywhere -> rejected. comma_to_danda: punctuation +0.002 but CER 0.069->0.082 -> rejected.
- speed 0.9: worse. ACCEPTED: questions 0.95 (.8855->.9095), conversational 1.05 (.9043->.9093), long_sentences 1.05 (.8845->.8934), basic_hindi 1.05 (.8951->.9110).
- unchanged: punctuation .9451, commands .8684, edge_cases .8259, difficult_hindi .8926.
- piper_v7a check (before->after): questions .8597->.8578, conv .8803->.8805, long .8370->.8384, basic .8575->.8579 (neutral).
- Not done: chunker edits (app chunker already clause/phrase aware; eval pause is flat so no lever), TTFA, Gemini A/B.
Caveat: speed gains partly via rate_cps proxy; unverified by ear.

## Gemini A/B (blind, randomized order, 8 cases/category, old vs new speed; reports/agents/c_judge.py)
questions .95: old 4 / new 4 / tie 0 (non-loss 50%) keep
conversational 1.05: old 4 / new 4 / tie 0 (50%) keep
long_sentences 1.05: old 1 / new 4 / tie 3 (88%) keep
basic_hindi 1.05: old 4 / new 2 / tie 2 (50%) keep (borderline)
Rule >=50% non-loss met for all; questions/conversational/basic are statistical coin flips at n=8, so proxy gain not strongly confirmed. tests/test_v8_prosody.py added (passes).
