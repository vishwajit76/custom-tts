# Voice quality research: making the Piper Hindi fine-tune sound natural

Written 2026-09-30. Scope: personal, non-commercial Hindi voice, Piper VITS medium fine-tuned from the hi_IN rohan checkpoint on
IndicTTS Hindi female (7.9 h), Kaggle T4 (batch 24, fp16, ~1.2 global steps/s = 0.83 s/step, ~46k steps per 11 h session (measured v4: 46.9k), ~30 GPU h/week).

## Evidence labels

| Label | Meaning |
|---|---|
| **V-src** | Read this session from the primary source (piper1-gpl `TRAINING.md`, `lightning.py`, HF model card, this repo's code/docs). |
| **V-snip** | Stated in a search-result snippet or forum thread only; the page was not fully read. Treat as likely, not certain. |
| **E** | My estimate or inference (arithmetic on our own numbers, or engineering judgement). Not a measured or cited fact. |
| **U** | Unverified lead. |

Honest caveats: no listening test was done here; nothing below is a measured Hindi MOS. Web search was US-only and several
claims rest on community posts. Numbers for T4 speed of alternative models are not measured by anyone in these sources.

## Summary

1. Our current metric cannot steer training. CER on 3 sentences with whisper-small (0.04-0.10, no trend) is noise; a
   30-50 sentence fixed set with a stronger ASR, plus a small blind A/B on checkpoints, is the cheapest big win.
2. Val mel loss is a weak proxy for naturalness; it falls slowly (0.469 to 0.430) and may keep falling while audio gets
   *smoother/more averaged*. Piper's own logging calls `val_mel` the recommended target for early stopping (V-src), but
   nothing says it tracks perceived naturalness. Use it only to detect divergence/overfit (val rising while train falls).
3. Learning-rate schedule is wrong for our setup (see 2.4, corrected): piper-tts 1.8.0 never steps its scheduler, so LR has been a flat
   1.5258e-4 (the value in the checkpoint), and there is no `lr_final_ratio` in 1.8.0. A low, decaying LR at the end of fine-tuning is the standard way to get a stable, cleaner final model (E).
4. Data hygiene (ASR-verified transcripts, dropping mispronounced/noisy clips, loudness match) usually matters more than more
   steps for a small single-speaker set (E, consistent with general VITS practice; not Piper-specific measured).
5. Inference settings are free to test: try `noise_scale` 0.4-0.7 and `noise_w` 0.6-1.0 with `length_scale` ~1.0-1.1.
6. Architecture change is a real option only for quality, not for CPU real-time. Best-supported Hindi candidates:
   **IndicF5** (MIT card, near-human claims, needs reference clip, GPU) and **Indic Parler-TTS** (Apache-2.0, 0.9B,
   slow). Piper remains 40-100x faster on CPU. Keep Piper as the serving path; use a big model, if at all, offline
   (e.g. to generate/augment data, U for whether that helps).

## 1. Steps/epochs, overfitting, base checkpoint

**Steps to plateau.**
- Community guidance: ~2000 epochs from scratch, roughly +1000 epochs for fine-tuning, "done when loss_disc_all levels off";
  max_epochs in Piper docs examples is 10,000 but 6,000 also works (V-snip, search summary of community guides
  [veralvx/piper-train](https://github.com/veralvx/piper-train), [ssamjh guide](https://ssamjh.nz/create-custom-piper-tts-voice/)). These are for
  small/varied datasets and mostly older `piper` (rhasspy) trainers; no rigorous curve exists in what I found (U for ~8 h).
- Piper1-gpl says a checkpoint "will speed up training a lot, even if from a different language" (V-src,
  [TRAINING.md](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md)).
- Our scale (E): 7.9 h at typical 4-6 s clips is ~5-7k clips, so ~250 steps/epoch at bs 24; measured: 302 steps/epoch at bs 24 (3607 train clips), ~46k steps/session is ~155 epochs.
  We are ~20k+ steps in (~80+ epochs). Community "+1000 epochs" would be ~250k steps, about 7 sessions (~77 GPU h). That is
  a budget upper bound, not a requirement. Expect the audible gain to come early (first 20-50k steps: speaker timbre, pacing)
  and then diminish; later steps mostly polish or start overfitting.
- The slow val_mel fall with no CER trend is consistent with being past the fast phase (E).

**Overfitting signs (E, standard GAN-TTS practice).**
- val_mel/val_kl/val_dur stop improving or rise while train losses keep falling.
- Audible: artifacts on unseen sentences (buzz, metallic sibilants, dropped/repeated syllables) while training-text sentences
  sound great; prosody becomes memorised-flat or oddly identical across sentences.
- Discriminator/feature-matching loss drift; sudden loss spikes.
- Mitigation: keep milestone checkpoints (we already save every 5000 steps and export three samples) and pick by blind
  listening on held-out text, not by last step. Build the test set from sentences NOT in training.
- Note `val_mos` in Piper is a logged perceptual estimate when enabled (V-src, lightning.py); its Hindi validity is unverified.

**Base checkpoint: rohan (male) vs female base vs "high".**
- Repo docs show Piper hi_IN has priyamvada (female) and pratham voices besides rohan (V-src, `docs/benchmarks.md` table). Whether
  priyamvada checkpoints are downloadable for training: U (check the `rhasspy/piper-checkpoints` dataset on HF; the repo docs
  said rohan itself was a fine-tune, so lineage/licensing is inherited).
- A female base likely starts closer in pitch/formants, so it should converge faster and lose fewer "male residue" artifacts
  (E, plausible, not measured). Cost of switching: restart from step 0 of fine-tuning (lose the ~20k steps done). Only worth it
  if listening shows timbre/pitch instability or a male-ish quality that isn't fading. Recommended: try as a *parallel
  short trial* (~10k steps) only if a checkpoint is available, and compare with the blind test in section 3.
- "High" quality: piper1-gpl states "Only `medium` quality checkpoints are supported without tweaking other settings" (V-src).
  High is a bigger model at 22.05 kHz, slower to train and infer, and no Hindi high base exists that I verified (U). With
  ~30 GPU h/week and 7.9 h of data, **do not switch to high**; the bottleneck is data/metric, not capacity (E).

## 2. Practical levers for naturalness

### 2.1 Data cleaning (highest value per effort, E)
- Transcript verification: run Whisper large-v3 (or IndicWhisper) on every clip, compute CER vs text; drop or fix clips with
  CER above ~0.25 (tune on a listened sample). IndicTTS is professionally recorded but has known text/audio slips and
  English loanword spellings (U for rate). `--asr-validate` in `training/prepare_dataset.py` exists; confirm it uses a strong ASR.
- Trim leading/trailing silence to 100-200 ms; cap internal pauses. VITS learns duration from alignment, so long silences or
  breaths inside clips produce weird stops at inference.
- Loudness: normalise per-clip to one target (e.g. -23 to -20 LUFS or RMS match) and peak < -1 dBFS; reject clipping. Keep
  the noise floor consistent; do not over-denoise (spectral gating can add "underwater" artefacts; test a with/without A/B).
- Length: prefer 2-12 s; drop very short (<1 s) and very long clips (memory + alignment errors).
- Punctuation: keep Devanagari danda/commas as in text; the phonemizer/prosody responds to them.

### 2.2 Phonemizer (espeak-ng `hi`)
- Piper uses espeak-ng phonemes (V-src: `--data.espeak_voice`). Hindi has orthographic schwa deletion, which is a known hard
  G2P problem; rule-based/ML models exist to improve it ([Supervised G2P of orthographic schwas, arXiv 2004.10353](https://arxiv.org/pdf/2004.10353), V-snip).
  I found no measured statement of espeak-ng hi's schwa accuracy (U). Practical check: `bench/pronunciation_report.py` on a list
  of ~100 hard words (schwa-deleted words, conjuncts, numerals, English loanwords).
- A fine-tune partly *absorbs* systematic phonemizer mistakes because the model learns "phoneme sequence to audio"; errors
  that are consistent across data become learned, but errors on unseen words remain (E).
- Lever if bad: pre-normalise text (numbers to words, abbreviations, nukta forms) before espeak; add a small override
  dictionary. Replacing the phonemizer means changing the phoneme inventory, which breaks the rohan initialisation (E), so
  avoid it during fine-tuning.

### 2.3 Sample rate
- Piper medium is 22.05 kHz (V-src, `--model.sample_rate` "usually 22050"). IndicTTS source audio is 48 kHz (U, from memory) and
  is downsampled by the pipeline. Staying at 22.05 kHz is required to reuse the checkpoint. No gain from changing.

### 2.4 Learning rate schedule for fine-tuning
- Piper defaults: generator LR 2e-4, discriminator 1e-4, betas (0.8, 0.99); ExponentialLR per epoch; with `max_epochs` set,
  decay is derived so LR ends at `lr_final_ratio=0.05` of initial; `warmup_epochs=0` (V-src,
  [lightning.py](https://raw.githubusercontent.com/OHF-Voice/piper1-gpl/main/src/piper/train/vits/lightning.py)); docs: setting `max_epochs` enables the automatic decay (V-src TRAINING.md).
- **CORRECTION 2026-09-30 (measured, supersedes the two bullets above and summary point 3).** The `lr_final_ratio` / "max_epochs derives the
  decay" behaviour is in piper1-gpl `main`, NOT in the `piper-tts==1.8.0` package the kernel installs. 1.8.0 has `lr_decay=0.999875`,
  `lr_decay_d=0.9999`, `warmup_epochs`, no `lr_final_ratio`, and its manual-optimization `training_step` never calls
  `scheduler.step()` (V-src: `piper/train/vits/lightning.py`, `grep '\.step()'` finds only `opt_g.step()`/`opt_d.step()`). So the LR is **constant
  at whatever the checkpoint's optimizer holds**, and `--trainer.max_epochs` has no effect on it. Read from HF `runs/hi_f/last.ckpt`
  (global_step 310300, epoch 3191, the seed the Kaggle sessions resume from): both optimizers `lr = 1.5258e-4`, `initial_lr = 2e-4`,
  scheduler `gamma = 0.999875`, `last_epoch = 2165` (rohan's own history). Lightning restores optimizer/scheduler state on `--ckpt_path`
  resume, so `--model.learning_rate` is ignored on resume. The LR during v4/v5 was therefore a flat 1.5258e-4 (E for exact value after the
  ckpt was re-saved: nothing steps it, so unchanged). The fix is done in the kernel wrapper (`LR_MODE=anneal`, see the runbook).
- Recommendation (E): set a finite `max_epochs` for the final phase, e.g. a 2-session "anneal" run with LR starting at
  ~1e-4 (or the current value) decaying to 5% (`--model.learning_rate 1e-4`, `--trainer.max_epochs = START_EPOCH + N`, N chosen
  to equal ~1.5 sessions of planned epochs). Annealing generally reduces noisy artifacts in GAN vocoders; not measured for Piper here.

### 2.5 Inference parameters
- Piper defaults: noise_scale 0.667, length_scale 1.0, noise_w (noise_scale_w) 0.8 (V-src: the validation-sample defaults in
  lightning.py are `[0.667, 1.0, 0.8]`). Community docs: noise_w controls duration variability, "higher = more human, lower = more
  mechanical"; noise_scale = 0 sounds robotic (V-snip, [piper-rn-poc README](https://github.com/mmaudet/piper-rn-poc/tree/main)).
- Try first (E): grid noise_scale {0.5, 0.667, 0.8} x noise_w {0.6, 0.8, 1.0} x length_scale {1.0, 1.1}, 5 sentences, 3 seeds.
  Higher noise_scale adds expressiveness but also artefacts; lower is cleaner but flatter. Suggested starting point for a
  natural-but-stable read: noise_scale 0.6, noise_w 0.9, length_scale 1.05. Fix the seed while comparing checkpoints (or average
  over 3 seeds) so noise doesn't confound the A/B.
- Check `config.json` `inference` block of our exported voice and the server's request path so these are settable per request.

## 3. Better objective evaluation (Kaggle/CPU)

**Test set.** 40-50 held-out sentences (not in training text), fixed forever: 12 conversational/short, 12 narrative/long
(15-25 words), 8 numerals/dates/amounts, 8 English loanwords/names, 5 questions/exclamations. Repo has `bench/quality_set.tsv`
(27 sentences) and `bench/sentences.txt` (13): merge/extend to 50, tag by category, commit as versioned file.

**Metrics (all E for how well they track Hindi naturalness unless stated):**
- **CER/WER with a stronger ASR.** Whisper large-v3 in faster-whisper int8 on a Kaggle CPU takes roughly real-time-ish per
  short clip; 50 clips x ~5 s is feasible in minutes on GPU (T4 fp16) and tens of minutes on CPU (U, not measured). IndicWhisper
  (fine-tuned Whisper on 10.7k h across 12 Indian languages; lowest WER in 39/59 Vistaar benchmarks, V-snip,
  [Vistaar](https://arxiv.org/pdf/2305.15386)) is the better Hindi ASR but is HF-hosted and needs conversion for faster-whisper (U effort).
  Use large-v3 first, and always compare checkpoints with the *same* ASR. Normalise text (strip punctuation, unify
  nukta/chandrabindu variants, numerals as words) before scoring; ASR errors on English loanwords inflate CER.
- **UTMOS / SpeechMOS.** `torch.hub.load("tarepan/SpeechMOS:v1.2.0","utmos22_strong",trust_repo=True)`, 16 kHz mono
  (V-src, [SpeechMOS README](https://github.com/tarepan/SpeechMOS)). Trained on English (VoiceMOS 2022 data, V-snip); use only for *relative*
  ranking of our own checkpoints, never as absolute Hindi MOS. Requires GitHub access at runtime (the repo already reports NaN if
  unreachable; on Kaggle with internet on it works). Report mean and per-sentence min; a checkpoint whose worst sentences drop
  is worse than mean suggests.
- **Speaker similarity.** `bench/eval.py` default uses MFCC statistics, not neural (V-src); switch to a neural encoder
  (Resemblyzer or ECAPA/WavLM-SV) to measure similarity to real IndicTTS female clips; expect ~stable, not improving (U).
  It guards against drift, not naturalness.
- **Prosody sanity (cheap, existing):** F0 std in semitones and pause statistics from `bench/eval.py` vs real IndicTTS
  held-out clips; a robotic voice tends to have lower F0 variance (E; noisy per docstring - average over sentences).
- **Blind A/B.** Automated metrics do not replace this. Build a 10-pair page (or a local script): checkpoint X vs Y, same
  sentence, random order; you rate 1-5 or pick. Ten minutes per comparison, and you are the target listener.

**Concrete protocol (E).**
1. Every milestone (5k steps) export ONNX (already done) and synthesise all 50 sentences with fixed noise settings and seed;
   store WAVs on HF next to the milestone.
2. Score: CER (large-v3), UTMOS mean/p10, speaker-sim (neural), val_mel; write one JSON row per milestone to HF.
3. Every 3rd milestone (15k steps) do a blind A/B of the current best vs candidate (10 sentences).
4. **Stopping rule:** stop (and lock the winner) when, over the last 3 milestones (15k steps), (a) CER improvement is below
   the noise band (compute band as std of CER across 3 seeds/ASR beams, likely ~1 pt over 50 sentences) AND (b) UTMOS mean
   changes < ~0.05 AND (c) blind A/B shows no consistent preference for later checkpoints (<= 6/10 wins). Additionally stop and
   roll back if val_mel rises for 2 consecutive milestones while train loss falls. Then run the annealing phase (2.4), evaluate
   once more, and finalize.
5. Budget cap: if no gain by ~150k total steps beyond the start, stop; more steps are unlikely to help (E).

## 4. Different architectures (T4, free Kaggle, personal use)

CPU real-time reference from this repo: Piper hi_IN medium RTF 0.022 at 4 threads (73-86 ms/sentence), Kokoro-82M fp32 0.237
(V-src, `docs/benchmarks.md`). Latency figures for others below are from reports, not our hardware.

| Model | Hindi support | License (personal use OK?) | Fine-tune on T4? | Latency | Verdict |
|---|---|---|---|---|---|
| Piper VITS (current) | fine-tune | code GPL; data CC BY 4.0 (IndicTTS, V-snip) | done, ~1.2 steps/s | RTF 0.022 CPU | keep for serving |
| **IndicF5** (F5-TTS based, 0.4B) | Hindi + 10 langs, trained on 1417 h (Rasa, IndicTTS, LIMMITS, IndicVoices-R) | MIT on card (V-snip/V-src); note data lineage and "clone only with permission" (V-src) | fine-tune not documented (U); needs reference clip + transcript | GPU needed; ~RTF <1 on GPU likely, CPU slow (U) | best quality candidate; **zero-shot with IndicTTS female reference clip** is a cheap try (inference only) |
| **Indic Parler-TTS** (0.9B) | Hindi speakers Rohit, Divya, Aman, Rani; ~107 h Hindi (V-src) | Apache-2.0 (V-src) | training possible but heavy; not for T4 free budget (E) | users report 5-15 s for a 10-15 word sentence without optimisation; team suggests flash-attn/streaming (V-snip) | fixed speakers, style via text description; too slow for interactive |
| StyleTTS2 | multilingual PL-BERT covers 14 languages incl. Hindi (V-snip) | MIT code; pretrained bases English | T4 reported "not very feasible" for high fidelity; needs small batch, stage-2 NaN risk (V-snip) | fast-ish GPU, CPU slower than Piper | tempting for naturalness but high risk/effort on our hardware |
| XTTS-v2 fine-tune | Hindi supported (V-snip) | CPML non-commercial; fine for personal (V-snip) | Feasible on T4 with small batch (U) | GPU near real-time, CPU slow | no clear naturalness win over IndicF5; Coqui is defunct |

Honest trade-off: none of these will match Piper's CPU speed. Realistic use: (a) keep Piper for real-time; (b) if you want
"best possible" offline audio, use IndicF5 with a reference clip from the target speaker on a Kaggle T4. Before any
investment, run the section-3 test set through IndicF5 (zero-shot) and compare with your best Piper checkpoint in a blind
A/B: cost is ~1 GPU-hour and no training. (E)

## Ranked recommendations

| # | Action | Effort | Expected impact | Cost |
|---|---|---|---|---|
| 1 | Replace the 3-sentence CER with a 50-sentence fixed held-out set, whisper large-v3 + UTMOS + neural speaker-sim, per-milestone JSON, plus 10-pair blind A/B | 0.5 day | High (enables all other decisions) | ~0 GPU (CPU, or ~10 min T4) |
| 2 | Fix LR: finite `max_epochs` anneal phase (LR 1e-4 to 5%); log the LR | 1-2 h | Medium-high (cleaner final model) | ~1.5 sessions (~15-17 GPU h) |
| 3 | Data audit: ASR-verify all clips with large-v3, drop high-CER/noisy clips, loudness-match, trim | 0.5-1 day | Medium-high | ~1-2 GPU h; retrain gain shows after ~10-20k steps |
| 4 | Inference grid (noise_scale, noise_w, length_scale), pick defaults and expose per-request | 1-2 h | Medium, immediate | 0 |
| 5 | Zero-shot IndicF5 baseline on the test set to see the ceiling; and, in parallel, optionally a short female-base (priyamvada) trial if checkpoint exists | 0.5 day | Informational / decides whether to change architecture | ~1-3 GPU h |

Not recommended now: "high" quality (medium-only supported, no Hindi high base verified), swapping phonemizer, StyleTTS2/Parler
training on T4.

## Next steps: concrete changes

**`training/kaggle/train_kernel.py`**
1. On load, print the optimizer LR and scheduler state from `ck0` (`ck0["optimizer_states"][0]["param_groups"][0]["lr"]`,
   `ck0["lr_schedulers"]`) and log `lr` in `prog.json` via `tr.optimizers[0].param_groups[0]["lr"]`, so the LR question is answered by
   data first (E).
2. (DONE 2026-09-30, corrected) env `LR_MODE=anneal LR_START LR_D_START LR_FINAL_RATIO ANNEAL_EPOCHS`, applied by a wrapper callback each epoch
   because 1.8.0 has no `--model.lr_final_ratio` and ignores `--model.learning_rate` on resume. Default behaviour unchanged.
3. Replace the 3 `SENTS` with reading the fixed 50-sentence file from the HF repo (or embed) and synthesise all of them per
   milestone with fixed `--noise-scale/--noise-w/--length-scale` and seed; upload WAVs. Also `pip install faster-whisper speechmos`
   (or torch.hub) and compute CER/UTMOS on Kaggle GPU at milestone time; write `metrics.json` per milestone.
4. Keep val_mel/val_kl/val_dur in the heartbeat so overfit detection (val rising) is visible.

**`bench/`**
5. Extend `bench/quality_set.tsv` to ~50 held-out sentences with category tags; keep out of training text (diff against
   `metadata.csv`).
6. `bench/quality_fw.py`: default `WHISPER_MODEL=large-v3` (GPU `float16` on Kaggle, `int8` on CPU), add text normalisation before
   CER, report mean/median plus per-category and a 3-seed spread.
7. `bench/eval.py`: set `SPEAKER_ENCODER` to a neural backend and pass real held-out IndicTTS female clips as `--reference`.
8. Add a small script that builds a randomized A/B HTML page from two milestone folders.

**Decision timeline (E).** Week 1: items 1, 3, 4, 5 plus continue training one more session with better eval. Week 2: anneal
phase from best milestone, final blind A/B, and lock. Only if the A/B says IndicF5 is clearly better, decide whether the offline
quality is worth the latency for your use.

## Sources
- Piper training docs (fine-tuning, medium-only, LR decay, sample rate): https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md (V-src)
- Piper Lightning module defaults, val_mel, noise scales: https://raw.githubusercontent.com/OHF-Voice/piper1-gpl/main/src/piper/train/vits/lightning.py (V-src)
- Community epoch guidance: https://github.com/veralvx/piper-train ; https://ssamjh.nz/create-custom-piper-tts-voice/ (V-snip)
- Piper checkpoints discussion (base checkpoint choice): https://huggingface.co/datasets/rhasspy/piper-checkpoints/discussions/8 (V-src)
- noise_scale/noise_w semantics: https://github.com/mmaudet/piper-rn-poc/tree/main (V-snip)
- SpeechMOS / UTMOS: https://github.com/tarepan/SpeechMOS ; https://arxiv.org/abs/2204.02152 (V-src/V-snip)
- IndicF5 card: https://huggingface.co/ai4bharat/IndicF5 (V-src; license line MIT from search snippet)
- Indic Parler-TTS card: https://huggingface.co/ai4bharat/indic-parler-tts ; speed thread: https://huggingface.co/ai4bharat/indic-parler-tts/discussions/10 (V-src)
- IndicTTS dataset (CC BY 4.0): https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi (V-snip)
- Vistaar / IndicWhisper: https://arxiv.org/pdf/2305.15386 (V-snip)
- Hindi schwa deletion: https://arxiv.org/pdf/2004.10353 (V-snip)
- StyleTTS2 multilingual PL-BERT and T4 notes: https://github.com/yl4579/StyleTTS2/discussions/128 (V-snip)
- XTTS-v2 (Hindi, CPML): https://huggingface.co/coqui/XTTS-v2 (V-snip)
- Repo files read: `docs/model-selection.md`, `docs/benchmarks.md`, `docs/training.md`, `training/kaggle/train_kernel.py`, `bench/eval.py`, `bench/quality_fw.py`
