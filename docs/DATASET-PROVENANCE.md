# Dataset provenance (V8 Phase 5)

Speaker age is never inferred from pitch; only provider-supplied metadata is recorded. Audio is referenced by path, not copied.

## 1. AI4Bharat Rasa, Hindi (primary)
| Field | Value |
|---|---|
| Source / URL | `ai4bharat/Rasa`, config `Hindi` - https://huggingface.co/datasets/ai4bharat/Rasa |
| License | CC-BY-4.0 (HF card; attribution to AI4Bharat required) |
| Language | Hindi (`hi`) |
| Speakers / gender | one female and one male speaker, gender as given in the `gender` column; labelled `rasa_hi_female` / `rasa_hi_male`. Age: not provided |
| Styles | provider `style` column (CONV, BOOK, NEWS, WIKI, six emotions, ...); not interpreted |
| Transcript source | provider transcripts (human_verified per `training/ingest_hf.py`) |
| Duration (from V7 ingest, not on disk now) | 44.8 h ingested, 38.2 h accepted (F 20.6, M 17.7) from the train split; test split held out |
| Redistribution | CC-BY-4.0 allows redistribution and commercial use with attribution; we still do not redistribute audio, only manifests |
| Intended usage | TTS training/adaptation and evaluation |
| Local status | HF repo is gated (terms click-through) and no `HF_TOKEN` is configured on this machine, so the download was refused. Only the 90-clip `work/v7/dksmoke` sample (0.14 h; F 0.071 h, M 0.068 h, 22.05 kHz mono) is processed. To fetch: accept terms, set `HF_TOKEN`, then `python -m training.ingest_hf --preset rasa --split train --gender female --out datasets/raw/rasa_hi_f` (and `male`), then `python -m training.v8.quality_filter --src datasets/raw/rasa_hi_f --src datasets/raw/rasa_hi_m` |

Local text-only copies (`work/v7/hf/data/hi_v7/metadata.csv`, 21,438 rows; no audio on disk) derive from this dataset; per-style hours in `source_style_hours.json`.

## 2. IndicTTS Hindi (female)
| Field | Value |
|---|---|
| Source / URL | IIT Madras IndicTTS (https://www.iitm.ac.in/donlab/indictts/) |
| License | personal/research use only per our V7 records; not cleared for commercial use |
| Speakers | one female speaker (gender as provided); age not provided |
| Transcript source | corpus transcripts (`work/v7/hf/data/hi_f/metadata.csv`, 4,013 rows; audio not on disk) |
| Duration | 7.9 h (V7 report) |
| Redistribution | not allowed; keep out of any shipped weights lineage that must be commercial |
| Intended usage | research only; excluded from V8 filtered manifests |

## 3. Smoke sets (`work/v7/msmoke`)
100 clips, speakers `f`/`m`, derived from the Rasa smoke sample (same texts); used for pipeline tests only, not included in manifests (duplicates).

## Not used
IndicVoices-R Hindi (CC-BY-4.0): 71.9 h rejected in V7 for <=0.37 h per speaker.

## Rasa Hindi: full ingest result (local parquet, ai4bharat/Rasa Hindi)

Ingested from `datasets/raw/rasa_parquet` (F 15107 clips / 27.05 h, M 13464 / 23.78 h), then `training.v8.quality_filter`.

Accepted 26323 clips (44.38 h): female 23.53 h, male 20.84 h. Rejected 2248 (too_long 1115, too_little_speech 571, too_short 501, duplicate_transcript 281, silence_ratio 215, clipping 159, bad_normalization 17, speech_rate_low 9). Splits: train 42.65 h, validation 0.91 h, test 0.82 h.

Accepted per style label (clips, hours): WIKI 4998/12.48, CONV 6494/7.47, NEWS 1809/3.75, BOOK 2555/3.67, PROPER NOUN 4253/2.71, FEAR 770/1.84, SURPRISE 749/1.87, HAPPY 746/1.85, SAD 668/1.80, ANGER 722/1.81, DISGUST 711/1.75, INDIC 670/1.72, ALEXA 556/0.47, BB 287/0.47, UMANG 165/0.38, DIGI 170/0.33.
