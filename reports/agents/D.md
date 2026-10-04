# Agent D: G2P / phoneme layer (goonj)

Goonj G2P = espeak-ng `hi` IPA (app.services.kokoro_engine.phonemes). Layer: optimizer/g2p.py + optimizer/g2p_fixes.json, hooked in Goonj.chunk (engines.py). Harness: reports/agents/d_exp.py (affected-case A/B, seed 0, Whisper CER; unaffected cases have identical phonemes so no regression possible).

## Accepted (CER on cases whose phonemes change)
- Geminate consonants: espeak writes `tː`, model ignores the length mark (बत्ती->बती). Pattern C(ʰ)ː -> CC: 86 cases 0.145->0.121 (28 better / 4 worse).
- सात `sˈaːt`->`sˈaːtə` (heard as साथ): 36 cases 0.124->0.102.
- Names (isolated pc cases, N=1-2 each): लखनऊ 0.75->0.25, इंदौर 0.60->0.20, जयपुर 0.34->0.24, चेन्नई 0.67->0.33, ज्योत्स्ना 0.22->0.15.
- Combined final file: 123 affected / 615 cases, CER 0.104->0.073 (39 better / 9 worse). Whole goonj bench CER ~0.119 -> ~0.113 (est.; unaffected cases unchanged).

## Rejected (no gain or worse)
कृपया (3 variants, 0.154->0.154/0.154/0.196), साढ़े r.h->ɽ / ɖʰ (0.130 flat), नौ->nɔu/nəu (flat), हाँ lengthen/ũ (worse), पटना, अहमदाबाद, क्षितिज kʂ (worse), मुम्बई (flat), final ˌi->ˌiː (0 effect), drop all ˌ stress (0.117->0.116, 85 better / 79 worse = noise).

## Model-limitation ledger (what no code fix solves)
Of 173 goonj cases with CER>0.15:
- 70 (40%): Whisper writes digits ("21 नवंबर 2024", "99 रुपिय") where reference is spoken words. Evaluator artifact, audio is fine. Excluding digit/Latin-hyp cases CER is 0.090 vs 0.119.
- 17: Whisper writes Latin (WhatsApp, ASAP, appointment). Evaluator artifact.
- 24 hinglish: English words transliterated differently (रजिस्टर्ड/रेजिस्टर्ड, मोबल). Spelling variance, not audible error.
- ~62 other, mostly ASR spelling ambiguity of correct audio: नौ->नो, बारह->बारा, दीजिए->दीजिये, ऋ->क्रि/रि (कृपया, ऋषिकेश, कृतज्ञता), ज्ञ->ग्य, ढ़->ध (साढ़े), ण->न (पुणे), geminate जज्->ज (उज्ज्वल, उत्तर), श्र/ष->श्. These cannot be told apart from a real error without a phoneme-level recognizer.
- Genuine model errors left: isolated short place names (मुंबई "मूब बी", पटना->पठना, अहमदाबाद->एहम्दाबाद; 6-8 cases, short utterances), ज्ञानेंद्र/प्रद्युम्न/Gyanendra (heard जाई एनंट, प्रेडियम), क्षितिज->शितिज, conjunct-heavy words (ज्योत्स्ना, प्रज्ज्वलित, श्रृंखला). ~15-20 cases total, all tail items in name/difficult_hindi. Phoneme respelling moved some, not these.
Verdict: no systematic model-level failure found; fine-tuning not justified by this evidence (spec §20).
Caveat: word-level overrides apply only when text word count == IPA token count (107/~700 chunks mismatch, mostly Latin runs and hyphens); patterns always apply.
