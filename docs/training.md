# Custom voice training

Pipeline: raw recordings → `training/prepare_dataset.py` → `training/train.py` (Piper VITS fine-tune, auto-resume) →
`training/export.py` (ONNX voice) → drop into `MODELS_DIR` → `bench/quality.py` + `bench/bench.py`.

**Status (2026-09-30):** the generic pipeline is validated end to end by `training/smoke_test.sh` (synthetic data). A
**personal, non-commercial custom Hindi voice is being trained** (IndicTTS Hindi female, 7.9 h, fine-tuned from Piper rohan on a
Kaggle T4 GPU); it is in progress and has had no human listening test. Kaggle runs are now recorded as immutable experiments (section 6). **How it is run today is in
[custom-voice-runbook.md](custom-voice-runbook.md)**; milestone results are in [training-progress.md](training-progress.md).
No commercially clean production voice exists: that needs authorized recordings (see docs/research.md; vendor-API audio is
not an option).

> **Fallback only: CPU long-train path.** `training/run_longtrain.sh` (+ `supervise.sh`, `hf_sync.sh`, `status.sh`,
> `export_latest.sh`) runs the same fine-tune on the 4 vCPU container (~7 s per batch-8 step) with an RSS watchdog and HF
> backup. It is kept as a fallback because the cloud container **pauses when idle** and can be reclaimed, so it cannot train
> unattended. Use the Kaggle GPU path (runbook) instead, and keep only one writer to HF `runs/hi_f/last.ckpt`.

## 1. Environment

```bash
training/setup_env.sh        # creates .venv-train: piper-tts[train], torch<2.9, lightning, builds monotonic_align
```

Separate from the server venv, because piper's training extra pins `librosa<1` and needs torch. `torch<2.9` is
pinned because newer torch defaults `torch.onnx.export` to the dynamo exporter, which fails on VITS.

## 2. Data

Record or obtain audio **only from speakers who consented to TTS training**, under a contract that grants you
commercial rights, or use verified CC BY 4.0 Hindi data such as SYSPIN or LIMMITS (attribution required).

What works well for one voice: 1–3 h minimum, 5–10 h better. One speaker, one room, one mic, 22.05 kHz or
higher, sentences of 2–12 s, read naturally in the calling style you want (greetings, numbers, amounts,
dates, English loan words).

Accepted layouts under `--input` (any mix):

```
raw/metadata.csv            file|text   or   file|speaker|text
raw/clip001.wav + clip001.txt
raw/<speaker_name>/...      one sub-directory per speaker
raw/long_recording.wav      no transcript: use --asr (silence split + local Whisper)
```

```bash
.venv/bin/python -m training.prepare_dataset --input raw --output data/myvoice --denoise [--asr] [--asr-validate]
```

Per clip: mono 22.05 kHz; optional spectral-gating denoise; silence trim with 100 ms pads; rejection of clipped,
< 1 s or > 15 s clips; loudness normalized to -20 dBFS. The transcript goes through **the server's own
normalizer**, so numbers, ₹, dates and Hinglish are spelled out at training exactly as at inference.
Speech-rate sanity check (4–30 chars/s) catches transcripts that don't match their audio. `--asr-validate`
also rejects clips whose Whisper transcript CER exceeds 0.35.

Outputs: `wavs/`, `metadata.csv` (train; Piper holds out its own validation split), `test.csv` (5%, never
trained on, deterministic by clip hash), `report.json` (hours per speaker, every rejection with its reason).

## 2b. Manifest, rights, splits, reports, annotation (multi-speaker / expressive data)

The directory layouts above still work. For multi-speaker or labelled data use a manifest, and **every run needs a
data-rights file**: `prepare_dataset` aborts if any clip has no rights entry, no speaker authorization, no
`tts_training` permission, or is `vendor_generated` without `vendor_generation_permission` (schema:
`training/data_rights.py`). `--allow-unverified-rights` exists only for throwaway smoke runs and is recorded in `report.json`.

Manifest (CSV with header, or JSONL): `audio,text,speaker_id,language[,emotion,style,role,label_source,rights_id]`.
`emotion/style/role` are kept **only** when `label_source=human_verified`; otherwise they are dropped with a warning
(never inferred). Rights entry (one JSON per line): `rights_id, source, licence, consent_record_id,
speaker_authorization, speaker_ids, permitted_uses, vendor_generated[, vendor_generation_permission]`.

```bash
python -m training.manifest --in data/manifest.csv --out data/manifest.jsonl      # validate, add ids + audio hashes
python -m training.annotate --manifest data/manifest.jsonl --annotator NAME        # http://127.0.0.1:8765 (listen, fix text,
                                                                                   #  verify speaker, label, reject; saves human_verified)
python -m training.audio_report --manifest data/manifest.jsonl --out reports/audio # per-file + per-speaker/emotion/style JSON + md
python -m training.split --manifest data/manifest.jsonl --out data/splits --seed 1234 [--speaker-disjoint-test]
python -m training.prepare_dataset --manifest data/manifest.jsonl --rights data/rights.jsonl --output data/myvoice \
    [--seed 1234] [--speaker-disjoint-test] [--denoise]
```

Splits are seeded and deterministic; clips sharing a normalized transcript or audio hash always land in the same split
(with `--speaker-disjoint-test`, whole speakers are held out and any train/val clip duplicating a test transcript/hash is
dropped). The test set is written separately (`test.jsonl` / `test.csv` + `.heldout` flag); `training.train` calls
`assert_not_heldout` and refuses to load it. Audio-report SNR is a rough percentile estimate, and its reject thresholds
(`LIMITS` in `audio_report.py`) are untuned defaults.

Training flags (`training.train`): `--seed` (-> `--seed_everything`), `--precision 16-mixed` (AMP, exercised on the Kaggle T4), resume is automatic,
`--dry-run` prints the `piper.train` command. Multi-speaker: `file|speaker|text` rows set `--model.num_speakers`;
`speaker_map.json` is written to the dataset dir (Piper's own map is in `config.json`). **Gradient accumulation is not
available**: Piper's trainer uses manual optimization and Lightning raises `MisconfigurationException: Automatic gradient
accumulation is not supported for manual optimization` (reproduced 2026-09-29, piper-tts 1.8.0 / lightning 2.x); `training.train`
rejects `--trainer.accumulate_grad_batches` up front, so use a larger `--batch-size`.

**Flags verified against a live `python -m piper.train fit --help`** (piper-tts 1.8.0, 2026-09-29): every flag `training.train` emits exists
(`--seed_everything`, `--ckpt_path`, `--data.{voice_name,csv_path,audio_dir,espeak_voice,cache_dir,config_path,batch_size}`,
`--model.{sample_rate,num_speakers,vocoder_warmstart_ckpt}`, `--trainer.{default_root_dir,accelerator,devices,precision,max_epochs}`).
`--trainer.precision` accepts `16-mixed`/`bf16-mixed` (`16-mixed` used on Kaggle T4). Also available and unused here: `--model.warmstart_ckpt`
(weights-only warm start), `--data.num_workers` (default 1), `--data.trim_silence` (Silero VAD).

**CPU smoke (plumbing only, NOT evidence of quality):** 6 clips of Piper rohan output (synthetic; 22.05 kHz) with the Hindi text,
`piper.train fit` from scratch, batch 2, `--trainer.max_steps 2` on 4 vCPU: exit 0 in 27 s, `last.ckpt` written. Then `python -m training.train
--data <dir> --run <dir> --init <that last.ckpt> --epochs 1 --batch-size 2 --accelerator cpu --trainer.max_steps 4` (exercises `--init` with a
local checkpoint, epoch-offset `max_epochs`, pass-through flags): exit 0, checkpoint written. Caveats: 6 clips leave the validation
set empty (Lightning warns), so `val_mel`/`val_mos` were never logged and no best-checkpoint was saved; UTMOS (torch.hub, GitHub) was
not loaded; `--init rohan|base` and `--warmstart-vocoder` (Hugging Face checkpoints) were not exercised in that smoke (`--init rohan` was used later for the real run, see the runbook). Setup notes: `piper-tts` from
PyPI ships `piper.train` but not the compiled `monotonic_align` extension: build it with `cythonize -i core.pyx` from piper1-gpl and place
`core*.so` in `piper/train/vits/monotonic_align/monotonic_align/` (`training/setup_env.sh` does this). Cython 3 prints `noexcept` warnings at build; the
result imports and runs.

Plumbing test (synthetic sine/noise audio, NOT evidence of quality): `pytest tests/test_training_pipeline.py`.

**Honest status:** the manifest/rights/split/report/annotation tooling is implemented and unit/plumbing tested on
synthetic data only; the annotation UI was not exercised in a browser. The one real fine-tune (IndicTTS Hindi female, plain
`file|text` layout, via `prepare_dataset`) is the custom voice in the runbook; the manifest/rights path was not used for it, and no
expressive/multi-speaker model exists yet.

## 2c. V7 dataset pipeline (ingest, quality gates, review, report)

Code: `training/ingest_hf.py` (Hugging Face parquet to wavs + manifest + rights), `training/quality_gates.py` (measurements and
decisions), `training/prepare_dataset.py` (runs the gates, writes Piper metadata and reports), `training/audio_report.py` (dataset
report). Plumbing is tested on synthetic audio only (`tests/test_quality_gates.py`); the **default thresholds are untuned** and must
be checked against the first real report (listen to a sample of what was rejected and what was kept).

### Ingest a Hugging Face dataset

```bash
export HF_TOKEN=...      # gated datasets: accept the terms on huggingface.co with the account behind this token first
.venv-train/bin/python -m training.ingest_hf --preset rasa --out data/rasa_hi --gender female --split train
.venv-train/bin/python -m training.ingest_hf --preset rasa --out data/rasa_hi --gender male   --split train   # same --out: extends it
```

Output in `--out`: `wavs/<id>.wav` (22.05 kHz mono 16-bit, soxr HQ resample), `metadata.csv` (manifest: `id, audio, text, speaker_id,
language, gender, style, age_group, duration, source_row_id, source_duration, license, source_repo, source_revision, label_source,
rights_id` + every extra mapped field), `rights.jsonl`, `ingest_info.json` (pinned revision sha, filters, counts, hours).

- **Column mapping** instead of hard-coding: `--map FIELD=SOURCE`, SOURCE = parquet column, a template over columns (`rasa_hi_{gender}`) or
  `=literal`. Presets (`--preset rasa|indicvoices-r`) are just default mappings, overridable. Rasa has no speaker column (one female and one
  male speaker per language), so the preset derives `speaker_id` from `gender`. For one Piper speaker id per voice x style, override with
  `--map speaker_id='rasa_hi_{gender}_{style}'` (style values must be filename-safe; check them once the data is visible).
- Filters: `--gender`, `--style` (repeatable), `--split` (shard-name match), `--max-hours`. Shards are read one at a time from the HF cache
  (`hf_hub_download`, revision pinned to the resolved sha), 32 rows per batch, so memory stays small. **Resumable**: a clip whose wav and
  `metadata.csv` row exist is skipped, `--max-hours` counts what is already there.
- Rasa layout seen on the Hub (listing only; the files are gated for us and were never downloaded): `Hindi/train-0000N-of-00025.parquet`
  (0.2 to 1.1 GB each) and `Hindi/test-0000N-of-00003.parquet`. The expected columns (filename, text, language, gender, style, duration,
  wav_path, audio) come from the task brief and are **unverified against real rows**.
- **Rights**: every run creates or extends `rights.jsonl` (schema of `training/data_rights.py` plus `attribution`, `source_url`, `revision`,
  `revisions`): licence CC-BY-4.0 (presets), `permitted_uses` `tts_training, research, commercial_use` (attribution required),
  `consent_record_id = dataset-licence:<repo>@<sha>`, speakers = the ones ingested. Presets assert `speaker_authorization`
  (the corpora are published for TTS under the licence); a bare repo needs `--licence` and `--speaker-authorization`, otherwise
  `prepare_dataset` refuses. `prepare_dataset --manifest` uses the `rights.jsonl` next to the manifest when `--rights` is omitted and
  still aborts (as before) on a missing file, an uncovered speaker, no `tts_training` permission or no speaker authorization.
- Dataset labels (style, emotion) are written with `label_source=human_verified` by default (`--label-source`): they are the
  dataset authors' labels, not model guesses. Change it if you do not trust them; `prepare_dataset` then drops them.
- IndicVoices-R Hindi (`SPRINGLab/IndicVoices-R_Hindi`, 368 speakers, 71.9 h) has at most 0.37 h per speaker: **unsuitable for a single
  production voice**. Preset kept for completeness and for speaker-diversity experiments only. Not smoke-tested: its shards are about
  4.6 GB; the ingest path is tested on synthetic parquet that mimics both layouts.

### Prepare with gates

```bash
.venv-train/bin/python -m training.prepare_dataset --manifest data/rasa_hi/metadata.csv --output data/rasa_hi_prepared \
    [--review data/rasa_hi/review.jsonl] [--workers 8] [--speaker-disjoint-test] [--min-snr-db 20 --max-s 20 ...]
```

Per clip (measured on the raw 22.05 kHz audio, before denoise/trim/normalise; implemented in `quality_gates.measure`):

| reason | rule (default) | method |
|---|---|---|
| `unreadable`, `no_transcript`, `empty_transcript` | file cannot be decoded / no transcript (and no `--asr`) / transcript has no letters | |
| `too_short` / `too_long` | duration outside 1.0 to 15.0 s (`--min-s/--max-s`) | after trim + 100 ms pads, what training sees |
| `rms_low` / `rms_high` | RMS outside -45 to -6 dBFS | whole-clip RMS |
| `loudness_low` / `loudness_high` | LUFS-like outside -48 to -8 | BS.1770 K-weighting, 400 ms blocks, -70 absolute / -10 LU relative gate; mono, approximate |
| `clipping` | samples with abs >= 0.99 above 0.1%, or a run of >= 6 | fraction and longest run |
| `low_snr` | SNR below 15 dB | speech-frame power minus noise power over noise power; noise = quietest 10% of 23 ms frames; speech = VAD frames |
| `silence_ratio` | non-speech frames above 60% | energy VAD: frame dB > max(p10 + 10 dB, p95 - 30 dB) |
| `too_little_speech` | under 0.5 s of speech | same VAD |
| `speech_rate_low` / `speech_rate_high` | letters+digits+matras per second of speech outside 4 to 30 | catches transcripts that do not match the audio |
| `asr_mismatch` | Whisper CER above 0.35 (`--asr-validate`, off by default, slow, single process) | `training/asr.py` |

Dataset level (`quality_gates.dataset_reasons`; first clip by id of a cluster is kept, later ones rejected):

| reason | rule |
|---|---|
| `duplicate_audio` / `speaker_leak_audio` | identical audio (sha256 of the written wav) under one speaker / under two speakers |
| `duplicate_text` | same normalised transcript twice for one speaker |
| `speaker_leak_text` | same transcript under two speakers; **off by default** (reported as `shared_text_across_speakers`; multi-speaker corpora legitimately share prompts), `--reject-cross-speaker-text` |
| `near_duplicate_text` | same speaker, char-trigram cosine >= 0.8 then difflib ratio >= 0.92 (`--text-near-sim`), durations within 10% |
| `near_duplicate_audio` / `speaker_leak_audio` | 16 x 32 log-mel signature, dataset-centred cosine >= 0.97 (`--audio-near-cos`), durations within 10%; across speakers it is leakage. Whole pass is a blocked matrix product (30k clips: seconds) |
| `condition_outlier_<noise_db\|centroid_hz\|bandwidth_hz\|lufs>` | per speaker (>= 20 clips), robust z = (x - median) / (1.4826 MAD) beyond 4.0 (`--condition-z`), with a minimum scale per feature; `--condition-by-style` judges each style separately (expressive styles are legitimately louder/brighter) |
| `no_near_duplicates` flag | `--no-near-duplicates` skips both near-duplicate passes |

Train/val/test overlap is prevented by `training/split.py` (groups by transcript and audio hash) and re-measured into the report
(`leakage.split_overlap` must be 0 for text and audio). Every threshold is a flag generated from `quality_gates.Thresholds`
(`prepare_dataset --help`).

### Human review sidecar

`--review review.jsonl` (JSON object `{id: entry}`, JSON list, or JSONL). One entry per clip id (manifest `id`; for directory input the
generated id, the path, file name or stem also work):

```json
{"id": "clip123", "quality": "approved", "naturalness": 4, "pronunciation": 5, "noise": 4}
```

`quality` is required (`approved | rejected | review`); scores are optional integers 1 to 5 (5 = best, noise 5 = cleanest). Unknown
fields, bad values and duplicate ids abort the run. `rejected` is excluded (`review_rejected`); `review` (undecided) is excluded
(`review_pending`) unless `--allow-review`; scores below `--min-naturalness / --min-pronunciation / --min-noise` (default 0 = off) are excluded
(`review_low_<score>`). A clip without an entry or without that score is not filtered. Review ids that match no clip print a warning.

### Outputs and report

`metadata.csv` / `test.csv` (`id.wav|text`, or `id.wav|speaker|text` when more than one speaker; unchanged), `manifest.jsonl`
(accepted clips with gender, style, source repo/revision), `rejected.jsonl` (id, file, speaker, `reasons`, measured values),
`report.json` (short summary), `dataset_report.json` + `dataset_report.md`: clips and hours (total / accepted / rejected), rejection
reasons with clips and hours (a clip with several reasons counts under each), hours per speaker, hours per category (style, else emotion,
else a `category` column, else `(none)`), average duration, SNR mean / median / p10 / p90, loudness percentiles and a 2 LU histogram, the
leakage numbers and the thresholds used. Rejected clips leave no wav behind. Throughput: the per-clip stage is a process pool
(`--workers`, forced to 1 with ASR); about 35 ms of measurement per 10 s clip plus load/trim/write, so 30k clips are a matter of minutes on an M4
(not timed end to end on real data).

### V7 dataset target

- **At least 10 h of clean Hindi per production voice after the gates; 15 to 20 h preferred for the primary voice.** Count accepted hours
  in `dataset_report.md`, per speaker (and per speaker id if styles become speakers). One speaker, one room, one mic per voice.
- **Prefer conversational over literary read speech.** Rasa is the target corpus (`ai4bharat/Rasa`, Hindi female 27.05 h, male 23.78 h,
  CC-BY-4.0, studio, labelled styles); it is gated for us until the owner accepts the terms. SYSPIN (read speech) stays the fallback.
- Sentence categories and target share of accepted hours (share by primary category; a sentence can belong to several, structured
  categories usually need a short targeted recording or scripted supplement because corpora rarely contain them):

| category | target share | category | target share |
|---|---|---|---|
| conversational | 20% | emotional phrases | 6% |
| questions | 8% | confirmations | 4% |
| answers | 8% | apologies | 2% |
| greetings | 3% | interruptions | 2% |
| numbers | 4% | short responses | 8% |
| currency | 4% | long responses | 7% |
| dates | 4% | Hindi-English code switching | 8% |
| times | 3% | names | 5% |
| addresses | 4% | | |

(sums to 100%). Short = under about 3 s, long = over about 8 s; measure from the report's duration, not by guesswork.

- **Rasa styles to categories.** The per-category table in the report uses the corpus's own `style` values. **TODO (advisor): list the
  real Rasa style values once the data is visible (`--style` takes them verbatim, case-insensitive) and fill in the mapping from each
  style to the categories above; do not guess.** Categories no style covers (numbers, currency, dates, times, addresses, names,
  confirmations, apologies, interruptions) must be tagged by transcript rules or a small supplementary recording set.

## 3. Train

```bash
# commercial voices: warm-start only the vocoder from Piper's CC BY 4.0 _base_model; needs more data and epochs
.venv-train/bin/python -m training.train --data data/myvoice --run runs/myvoice --warmstart-vocoder base --epochs 1000 ...

# experiments only, NOT for commercial voices: fine-tune the Hindi rohan checkpoint. rohan was itself fine-tuned
# from the lessac voice, whose data (Blizzard 2013) is research-only (docs/research.md), so derivatives inherit that.
.venv-train/bin/python -m training.train --data data/myvoice --run runs/myvoice --init rohan --epochs 300 \
    --batch-size 32 --accelerator gpu --precision 16-mixed
```

- **Resumption:** re-run the same command. It picks up `runs/<name>/lightning_logs/version_*/checkpoints/last.ckpt`
  (optimizer, epoch and step included); `--epochs` means "this many more".
- **Checkpoints:** last, top-5 by `val_mel`, top-5 by `val_mos` (UTMOS predicted MOS, the better quality signal).
  Pick the final one by listening plus `val_mos`; mel loss plateaus early while GAN losses keep removing artifacts.
- **Multi-speaker:** use `file|speaker|text` rows. Fine-tuning a single-speaker checkpoint (rohan) into several
  speakers fails on shape mismatch; use `--warmstart-vocoder` for multi-speaker voices.
- Anything else goes straight to `python -m piper.train fit` (e.g. `--model.learning_rate 1e-4`).
- TensorBoard: `.venv-train/bin/tensorboard --logdir runs/myvoice`.

**Hardware.** Measured on Apple M4 (MPS), batch 8: ~7.7 s/step, which is impractical beyond smoke tests.
A single NVIDIA GPU (L4 / A10G / RTX 4090 class) is the realistic target. Plan several hours for a fine-tune
from rohan and a day or more for a vocoder warm start. Measured since: a Kaggle Tesla T4 (fp16-mixed, batch 24) runs about 1.2 global steps/s (0.83 s per global step; the heartbeat field `sps` is
seconds per step, earlier docs misread it as steps per second); 5000 steps take about 70 min. See the runbook.

## 4. Export and evaluate

```bash
.venv-train/bin/python -m training.export --run runs/myvoice --voice hi_IN-myvoice-medium   # -> models/piper/
.venv/bin/python -m bench.quality --voices hi_IN-myvoice-medium --test-csv data/myvoice/test.csv
```

Export writes `<voice>.onnx` + `.onnx.json`, synthesizes a test sentence and fails if the output is silent.
The server loads every `*.onnx` in `MODELS_DIR` at start-up. Set `DEFAULT_VOICE` to make it the default.

## Validation performed (smoke test)

`training/smoke_test.sh` synthesizes 40 clips with an installed voice (with added noise on every 5th clip), plus
one deliberately mismatched clip. It then runs prepare (`--denoise`), trains one epoch from the rohan checkpoint
on MPS, trains again to prove resume, exports, and synthesizes with the new ONNX. Last run:

- prepare: 36 clips kept; 4 rejected: three "जी हाँ" clips under 1 s, and the mismatched transcript
  (313 chars/s).
- train: epoch 3191 from rohan ckpt: `val_mos` 3.74, `val_mel` 0.397. Resume: epoch 3192: `val_mos` 3.61, `val_mel` 0.395.
- export: ONNX voice synthesized 3.25 s of audio at RTF 0.033.

This proves the mechanics only. Training on synthetic audio from an existing voice does not make a new voice.


## 6. Kaggle runs as experiments (hardening pass, 2026-09-30)

Everything below concerns `training/kaggle/train_kernel.py` from the next push on. The running session (v5) uses the older code and is unchanged.

### What changed
- **Explicit resume point.** `RESUME_FROM=<HF path>` or `auto` (default) = the highest `global_step` among `experiments/*/checkpoints/*.ckpt` and `runs/hi_f/*.ckpt`
  (step read from the file name, or from the file for `last.ckpt`). The downloaded file is verified: sha256 against the HF LFS hash, `torch.load`, all weights finite,
  and the name's step equals the file's `global_step`; any mismatch aborts before training. The chosen path, step, sha256 and every candidate considered go into `manifest.json`.
- **Artifact layout, one folder per run.** Everything is written to `experiments/<id>/` (`checkpoints/{last.ckpt,last.meta.json,final_step<N>.ckpt}`, `milestones/step_N/`,
  `samples/step_N/`, `metrics.jsonl`, `heartbeat.txt`, `manifest.json`, `environment.json`, `session.json`, `result.json`, `crash.txt`; evaluations go in `evaluations/`).
  `id` defaults to `hi_f-<KERNEL_VERSION|k>-<UTC start minute>`. `runs/hi_f/last.ckpt` and `milestones/` are **frozen legacy**: the kernel no longer writes them
  (only the legacy heartbeat `runs/hi_f/kaggle_progress.txt`, which the hourly check-in reads, and `kaggle_crash.txt`).
- **Upload guard** (`ExperimentGuard`). All uploads go through it: only under `experiments/<id>/`; refuses to start if the remote `session.json` names another session
  (two sessions can no longer overwrite each other); final checkpoints, milestones, manifest and result are create-once (an existing path is skipped, never overwritten);
  ownership is re-read every 10 min. Not atomic (HF has no compare-and-swap), documented in the class.
- **Checkpoint integrity.** Each periodic upload first copies the checkpoint, loads it (a torn copy made while Lightning writes fails the load and is retried next cycle),
  refuses non-finite weights, refuses to go backwards in step, then uploads and reads back the sha256 from HF.
- **Reproducibility files.** `manifest.json` (git sha, kernel script sha256, resume checkpoint, dataset fingerprint, seed, batch size, LR env, full trainer CLI, UTC/IST start),
  `environment.json` (python, torch, lightning, piper-tts, CUDA runtime, cuDNN, GPU, espeak-ng), `metrics.jsonl` (one line per 20 steps: step, epoch, lr, losses, seconds/step).
  Kaggle script kernels cannot take env vars, so `training/kaggle/push.sh` stamps `GIT_SHA`, `KERNEL_VERSION`, `RESUME_FROM`, `EXPERIMENT_ID` into a temp copy of the script
  (`--dry-run` shows it); a plain `kaggle kernels push -p training/kaggle` still works and records `git_sha: unknown`.
- **Immutable records in git:** `training/experiments/<id>.json` (`python -m training.experiments validate`; create-once writer), results in a separate `<id>.result.json`.
- **LR anneal default corrected:** `ANNEAL_EPOCHS` 160 -> 150 (see Evidence).

### Why
v4 and v5 ran concurrently from the same seed checkpoint, and v5's 20-minute upload re-sent the unchanged 310300 seed to `runs/hi_f/last.ckpt` after v4 had uploaded its
final. Milestone folders `step_320000..345000` were written by both sessions, so a step number did not identify a model, and two rows of the 50-sentence table were labelled with the wrong session.

### Evidence (verified 2026-09-30 06:25 UTC unless stated)
- HF `runs/hi_f/last.ckpt`: `global_step` 310300, epoch 3191, sha256 25b56359... (the stale seed). `runs/hi_f/v4_final_step357212.ckpt`: `global_step` 357212, epoch 3347,
  sha256 2561a9c9..., both optimizers `lr = 1.5257792529e-4`, schedulers `last_epoch = 2165` unchanged: 46.9k steps at a constant LR (downloaded, `torch.load(weights_only=False)`, CPU).
- piper-tts 1.8.0 `piper/train/vits/lightning.py`: `automatic_optimization = False`, `opt_g.step()`/`opt_d.step()` only, no scheduler call anywhere; pinned by `tests/test_lr_anneal.py`.
- Steps per epoch: (357212 - 310300) / (3347 - 3191) = 300.7; piper holds out 10% of `metadata.csv` (4013 rows) as validation and 5 as test, so 3607 clips / 24 = 151 batches x 2 optimizers = 302.
  The earlier "about 336 steps per epoch / 54k steps" was wrong, so 160 epochs = 48.3k steps and 150 epochs = 45.3k steps, about one 11 h session at 1.2 steps/s.
- Milestone provenance on HF now (upload time of the ONNX): 340000 03:31 (v5), 345000 04:41 (v5), 350000 05:50 (v5, v4 had ended at 05:10), 355000 04:39 (v4).
- CPU smoke of the whole kernel script against a local directory (`python -m training.kernel_smoke`): all checks PASS (resume tie-break and probe, manifest, environment, metrics,
  anneal, milestone export+upload, checkpoint upload with read-back, create-once final, result.json, nothing outside `experiments/<id>/`, second session with the same id refused).
  Unit tests: `tests/test_kaggle_kernel.py` (guard with a fake HfApi, resume selection, NaN/torn checkpoints), `tests/test_lr_anneal.py`, `tests/test_experiments.py`.

### Unverified
- The new kernel has not run on Kaggle/GPU (no GPU here, and no push was allowed while v5 runs). The real `HfApi` calls it adds (`file_exists`, `get_paths_info`, `upload_folder`) were
  only exercised read-only against the live repo; the writes were exercised against the local fake.
- Kaggle's `torch`/CUDA versions for v4/v5 were not recorded (environment.json will record them from now on).
- Whether the same seed in every session (piper draws the train/val split from `--seed_everything`, so it must stay 1234) makes each session replay the same batch order at its start is unmeasured.
- The audible effect of the anneal; every number in docs is a proxy (section below and benchmarks.md), no human listening test has been run.

### Next experiment (proposed launch config)
| Item | Value |
|---|---|
| Dataset | `data/hi_f`, metadata.csv sha256 9bf8f152..., 4013 rows (3607 trained, 401 piper-validation, 5 piper-test), 4225 wavs incl. 212 held-out test clips |
| Resume | `RESUME_FROM=auto`; today that resolves to `runs/hi_f/v4_final_step357212.ckpt` (verified step 357212) unless v5's final (`runs/hi_f/last.ckpt` after ~07:25 UTC) verifies higher. Pin it with `push.sh --resume-from runs/hi_f/v4_final_step357212.ckpt` for a fixed baseline |
| Seed / batch | 1234 (do not change) / 24, fp16-mixed, T4, 11 h |
| LR | `LR_MODE=anneal`, 1e-4 -> 5e-6 (ratio 0.05) over `ANNEAL_EPOCHS=150` (about 45k steps), then hold. Reason: 47k steps at constant 1.53e-4 (v4) plateaued on every proxy (340k vs 345k is a tie within the noise band), and the anneal is the standard fine-tune finish. The recorded step from 1.53e-4 to 1e-4 at the start is a 35% drop at once |
| Evaluation | every 5000-step milestone: `python -m bench.compare_checkpoints --milestone <path> --milestone runs-baseline ...` with `--repeats 3`; decision by the paired CIs, then a blind A/B (`docs/listening-test/`) of best-anneal vs v4 final |
| Artifact path | `experiments/<id>/` (record `training/experiments/<id>.json` after launch: `python -m training.experiments new <id> --from-hf`) |

Launch: `training/kaggle/push.sh --kernel-version v6` (after `kaggle kernels status` says the previous session is complete). Dry run of the stamping: `training/kaggle/push.sh --dry-run`.
Local checks before any push: `python -m training.kernel_smoke --ckpt <any real ckpt>` and `python -m training.lr_dryrun --ckpt <ckpt>` (both CPU).
