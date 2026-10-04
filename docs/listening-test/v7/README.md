# V7 blind listening test

Status: **protocol and tooling only. No listening test has been run; no human result exists in this repository.** Gate 2 of the V7 roadmap
("V7 better than V6 in a blind native-listener test") needs real people and cannot be closed by any script.

**UTMOS is not a human MOS.** The harness (`bench/v7_eval.py`) reports UTMOS22 as *predicted* MOS from an English-trained model; CER/PER are ASR proxies;
speaker similarity is an embedding cosine. None of them says whether a voice sounds natural to a Hindi speaker. Human ratings from this test are recorded
and analysed **separately** (`docs/listening-test/v7/results/`, produced by `analyze-v7`), never merged into `bench/results/v7_eval/` and never averaged with automatic metrics.

## Systems

`systems.json` lists them; an entry with `"spec": null` is a placeholder and is skipped by the generator until the voice exists.

| name | what | status |
|---|---|---|
| `v6` | `voices/hi_IN-custom-medium.onnx` (IndicTTS Hindi female, v6 final step 402504) | available |
| `v7_baseline` | V7 multi-speaker Piper medium, first training result | placeholder |
| `v7_improved` | V7 after the improvement pass (or the medium-vs-high winner) | placeholder |
| `young_female` | `hi-IN-young-female` (Rasa female) | placeholder |
| `young_male` | `hi-IN-young-male` (Rasa male) | placeholder |

A system spec is `NAME=PATH.onnx[@SPEAKER]` or `NAME=voice:ID` (same as `bench.v7_eval`), so a multi-speaker V7 model contributes one system per speaker.

## What listeners do (two tasks, same blinded audio)

Listeners: native Hindi speakers, headphones, quiet room, one playback device for everyone, not the people who trained or tuned the models.
Every file is resampled to 24 kHz and RMS-matched, so neither sample rate nor loudness identifies a model. Sentences are a stratified sample of corpus v2
(assistant/call-centre register; `bench/corpus/README.md`), fixed by `--seed`. Models run with their default inference parameters and the seeded noise of `bench.v7_eval`.

1. **Ratings (MUSHRA-like, single stimulus).** `sheets/listener_NN_rating.csv`, one row per `audio/r_<id>.wav`. Score each 1 (bad) to 5 (excellent) on
   `naturalness` (sounds like a real person), `pronunciation` (correct sounds, no swallowed or wrong syllables), `prosody` (rhythm, stress, intonation fit the sentence and
   its sentence type), `clarity` (easy to understand on a phone line), `conversational_realism` (sounds like someone talking to you, not reading), `speaker_consistency`
   (the same person, same voice quality from start to end of the utterance). One hidden low anchor per sentence (4 kHz band-limited + noise) screens out listeners who do not discriminate.
2. **A/B.** `sheets/listener_NN_ab.csv`, one row per pair `audio/ab_<id>_A.wav` / `_B.wav`, same sentence, random side. For each dimension and `overall_preference`
   write `A`, `B` or `tie` (overall: which would you rather hear if this called you).

Listeners see only blind ids, the sentence text and the audio. Keep `organizer/` away from them.

## How many listeners and items

- **Per pair of systems, A/B:** with n non-tie votes a two-sided sign test needs at least 9/10, 15/20, 21/30, 27/40 wins for p < 0.05 (`bench.listening_pack.min_wins_for_significance`).
  Plan 30 items per pair of interest; with 5 listeners rating the same items the votes are not independent, so treat the pooled p as optimistic and read `per_listener`.
- **Ratings:** 5 or more listeners x 20 items x every system gives ~100 ratings per system and dimension; the CI resamples *items*, which is the honest unit. A mean difference
  of a few tenths of a point is roughly what such a design can resolve (an expectation, not measured here); smaller differences need more items, not more listeners.
- At least 5 native listeners for a decision, 3 for a pilot. Fewer than 20 items or fewer than 3 listeners: report as anecdote.
- Many (pair, dimension) tests are run: `analyze-v7` prints `bonferroni_alpha`; claim a win on one dimension only if it also holds on `overall_preference`.

## Procedure

```bash
# 1. organizer: render (any set of available systems; --systems-file skips placeholders)
python -m bench.listening_pack make-v7 --systems-file docs/listening-test/v7/systems.json --n-items 20 --listeners 5 --seed 7 --out /tmp/v7pack
#    commit organizer/pack_manifest.json (it holds key_sha256) BEFORE the test: it commits to the system assignment, and analyze-v7 refuses a key that differs.
# 2. give listeners /tmp/v7pack/listener_pack only (audio/ + sheets/); each fills their own two CSVs (templates: ratings_template_rating.csv, ratings_template_ab.csv).
# 3. organizer, after collecting:
python -m bench.listening_pack analyze-v7 --key /tmp/v7pack/organizer/KEY_DO_NOT_SHARE.json \
    --rating l1_rating.csv l2_rating.csv ... --ab l1_ab.csv l2_ab.csv ... --out docs/listening-test/v7/results/run1.json
```

Output: per dimension and system the mean rating with a 95% item-bootstrap CI; paired mean differences between systems (`excludes_zero`); per pair and dimension the A/B wins,
ties, exact sign test and per-listener counts; per listener the anchor check (`flag_anchor_not_below_all_systems`: drop or discount that listener).
Cells must be integers 1-5 (ratings) or A/B/tie (A/B); anything else is an error, blank means not rated.

## Reading the result

- A V7 system beats V6 only if `overall_preference` and `naturalness` both favour it with the pooled sign test p < `bonferroni_alpha` *or* every listener's own counts agree, and it does not lose on `pronunciation`.
- A tie keeps the cheaper/earlier system. Disagreement between listeners is a finding, not noise to be averaged away: report it.
- Compare `young_female` / `young_male` with their real Rasa recordings (put a few reference clips in the packet as a labelled "reference voice" for `speaker_consistency`), not with V6.

## Files

- `systems.json`, `ratings_template_rating.csv`, `ratings_template_ab.csv`, this README.
- `bench/listening_pack.py`: `make-v7`, `analyze-v7` (the older `make`/`merge` two-model tool and `../README.md` are unchanged). Tests: `tests/test_listening_pack.py`
  (blinding: no system name in file names, audio bytes, sheets or manifest; one sample rate; key round trip and commitment; analysis on a synthetic sheet that exists only inside the test).
