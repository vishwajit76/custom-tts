# Blind A/B listening test

Status: **protocol and tooling only. No listening test has been run, and no result exists in this repository.** Every automatic number in
[../training-progress.md](../training-progress.md) (CER, PER, UTMOS predicted MOS, speaker-embedding similarity) is a proxy; this is how to get the human evidence.

## What is compared
Two models on the same sentences: for example the current best milestone against the v4 final, or an annealed milestone against a constant-LR one. Models are identified by
HF folder plus ONNX sha256 (`python -m bench.compare_checkpoints --list`), not by step number alone.

## Listeners and setup
- At least 3 native Hindi listeners who do not know which model is which; headphones, a quiet room, the same playback device for everyone. Not the person who trained the model if avoidable.
- 20 items (the script default; 30 is better) plus 3 hidden repeats (same sentence, A/B sides swapped) that measure each listener's consistency.
- Sentences come from `bench/hi_eval_50.txt` (disjoint from the training text, `python -m bench.check_eval_overlap`), sampled with a fixed seed (recorded in the key).
- One synthesis per model per item with the voice's default inference parameters (noise_scale 0.667, noise_w 0.8, length_scale 1.0). Piper's noise is unseeded, so one draw is one sample: that is why
  there are many items and several listeners. Loudness is RMS-matched per file, A/B side and item order are randomized.

## Procedure
1. Organizer: `python -m bench.listening_pack make --a milestones/step_340000 --b milestones/step_355000 --n 20 --seed 7 --out pack/`
   Give listeners **only** `pack/audio/` and `pack/ratings_sheet.csv`. Keep `pack/KEY_DO_NOT_SHARE.csv` (A/B assignment per item) and `pack/pack.json` (model provenance).
2. Each listener plays `<id>_A.wav` and `<id>_B.wav` as often as they like and fills one row per item with **A**, **B** or **tie** for each question:
   - `naturalness`: which sounds more like a real person speaking?
   - `pronunciation`: which pronounces the words more correctly (wrong sounds, skipped or swallowed syllables)?
   - `speaker_similarity`: which sounds more like the target speaker (an IndicTTS female reference clip can be played first)? This is the listener's judgement, unlike the automatic embedding cosine.
   - `prosody`: which has more natural rhythm, stress and intonation for the sentence?
   - `overall_preference`: which would you rather hear on a phone call?
   Optional `comment`. The `checkpoint_A`/`checkpoint_B` columns stay empty on the listeners' sheets (the template in `ratings_template.csv` has them because the merged output fills them).
3. Organizer, after collecting one csv per listener: `python -m bench.listening_pack merge --key pack/KEY_DO_NOT_SHARE.csv --ratings l1.csv l2.csv l3.csv --out results.csv`
   prints wins per model and tie counts per question, the exact two-sided sign test (ties excluded) and the consistency on the hidden repeats.

## Reading the result
- With n non-tie votes the sign test needs a minimum number of wins for p < 0.05 (two-sided): n=10 -> 9, n=20 -> 15, n=30 -> 21, n=40 -> 27 (`bench.listening_pack.min_wins_for_significance`).
- Votes from several listeners on the same 20 sentences are not independent; the merged p-value is optimistic. Report per-listener counts and check that listeners agree.
- Listener consistency on the hidden repeats below about 70% means the pair is too close to call or the listener is unreliable: say so instead of reporting a winner.
- Decision rule used in the runbook (section 6.2): prefer the later checkpoint only if it wins `overall_preference` significantly or at least 6 of 10 non-tie votes per listener on average, and loses nowhere on `pronunciation`.
- A win here overrides the automatic metrics; a tie means keep the cheaper/earlier checkpoint.

## Files
- `ratings_template.csv`: header of the rating sheet (sample_id, checkpoint_A, checkpoint_B, sentence, naturalness, pronunciation, speaker_similarity, prosody, overall_preference, comment).
- `bench/listening_pack.py`: `make` (stimuli, blank sheet, key) and `merge` (unblind and tally). Tested in `tests/test_listening_pack.py` with synthetic ratings that exist only inside the test.
