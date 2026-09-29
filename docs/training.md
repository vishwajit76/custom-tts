# Custom voice training

Pipeline: raw recordings → `training/prepare_dataset.py` → `training/train.py` (Piper VITS fine-tune, auto-resume) →
`training/export.py` (ONNX voice) → drop into `MODELS_DIR` → `bench/quality.py` + `bench/bench.py`.

**Status:** the pipeline is implemented and validated end to end by `training/smoke_test.sh` (synthetic data,
1 epoch + resume + export, see below). **No production voice has been trained yet.** That needs authorized
recordings and an NVIDIA GPU; neither was available here. See docs/research.md for why vendor-API audio is not
an option.

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

Training flags (`training.train`): `--seed` (-> `--seed_everything`), `--precision 16-mixed` (AMP), resume is automatic,
`--dry-run` prints the `piper.train` command. Multi-speaker: `file|speaker|text` rows set `--model.num_speakers`;
`speaker_map.json` is written to the dataset dir (Piper's own map is in `config.json`). **Gradient accumulation is not
wired**: Piper's trainer uses manual optimization, where Lightning rejects `accumulate_grad_batches` (unverified against your
installed version; use a larger `--batch-size`). Flag names were not checked against a live `piper.train --help`
(piper's train extra was not installed here).

Plumbing test (synthetic sine/noise audio, NOT evidence of quality): `pytest tests/test_training_pipeline.py`.

**Honest status:** the manifest/rights/split/report/annotation tooling is implemented and unit/plumbing tested on
synthetic data only. The annotation UI was not exercised in a browser. No production fine-tuning has been performed, no
real data was processed, and no expressive/multi-speaker model exists yet.

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
from rohan and a day or more for a vocoder warm start. This is an estimate, not a measurement: no NVIDIA GPU
was available.

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
