# Licences of everything used or referenced

Written 2026-09-29. Verdict column: **Yes** = commercial use permitted on the evidence cited; **Conditional** = permitted with
obligations we must meet; **No** = disqualifying for a commercial product; **Unknown** = not established, treat as No
until resolved. Nothing here is legal advice; every "Yes/Conditional" needs counsel sign-off before launch.

Evidence tags: **pip** = read from installed package metadata this session; **gh** = primary GitHub README read this
session; **prior** = recorded in `docs/research.md` (earlier session, not re-verified here); **recollection** = model
memory only, must be checked. **Hugging Face was unreachable this session**, so no HF card or LICENSE file was re-read.
Re-check each row before release and record the date.

## Models, checkpoints, voices

| Item | Where used | Licence | Commercial verdict | Source / evidence | Notes |
|---|---|---|---|---|---|
| Piper Hindi voice `hi_IN-pratham-medium` | `scripts/download_voices.py`, `voices/` | CC BY-NC-SA 4.0 (data) | **No** | https://huggingface.co/rhasspy/piper-voices (MODEL_CARD) - prior; script docstring | Non-commercial + share-alike |
| Piper Hindi voice `hi_IN-priyamvada-medium` | same | CC BY-NC-SA 4.0 (data) | **No** | same - prior | |
| Piper Hindi voice `hi_IN-rohan-medium` (default voice) | default `DEFAULT` in download script; `training/train.py` CKPTS `rohan` | Data: IITM EULA (permissive per prior read); **fine-tuned from lessac**, whose Blizzard 2013 data is research-only | **No (risk); Unknown until lawyer review** | https://huggingface.co/rhasspy/piper-voices - prior | Do not ship as a commercial voice or use as `--init` for commercial training |
| Piper checkpoint `_base_model/base_model.ckpt` (CKPTS `base`) | `training/train.py` | Unknown (checkpoint repo states no clear licence in our records) | **Unknown** | https://huggingface.co/datasets/rhasspy/piper-checkpoints | Base was trained on lessac-derived data per Piper docs (recollection). Verify before commercial fine-tune |
| Piper checkpoint rohan `epoch=3190...ckpt` | `training/train.py` | as rohan voice | **No (risk)** | same | |
| Kokoro-82M v1.0 weights + voices (hf_alpha/beta, hm_omega/psi) | `kokoro_engine.py`, `download_voices.py kokoro` | Apache-2.0 per kokoro-onnx README (gh); training data partly synthetic audio from closed TTS vendors (prior) | **Conditional** (provenance risk on data) | https://github.com/thewh1teagle/kokoro-onnx ; https://huggingface.co/hexgrad/Kokoro-82M (prior) | Model files fetched from kokoro-onnx GitHub release `model-files-v1.0` |
| Supertonic 3 model (`Supertone/supertonic-3`) | `supertonic_engine.py`, download script | BigScience OpenRAIL-M (model), MIT (sample code) (gh, prior) | **Conditional** | https://github.com/supertone-inc/supertonic (archived) ; HF LICENSE (prior) | Flow down Attachment A use restrictions; must disclose machine-generated audio; no impersonation without consent. Upstream archived, no fixes |
| Qwen3-TTS `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `qwen_engine.py`, `config.py` `model_path` | README states no licence (gh) | **Unknown** | https://github.com/QwenLM/Qwen3-TTS | Apache-2.0 is likely (recollection) but unverified. Hindi unsupported (gh) |
| Whisper `openai/whisper-large-v3-turbo` | `training/asr.py`, `bench/quality.py` (evaluation/dataset validation only) | MIT (recollection) | **Yes (unverified)** | https://huggingface.co/openai/whisper-large-v3-turbo | Not shipped in the serving path |
| UTMOS22 via `tarepan/SpeechMOS` v1.2.0 | `bench/quality.py` (torch.hub) | Code MIT (recollection); weights/training data (VCC/BVCC) terms unverified | **Unknown** (eval-only, not shipped) | https://github.com/tarepan/SpeechMOS | Predicted MOS is a proxy, English-trained |
| Speaker encoder `resemblyzer` (GE2E) | `app/services/speaker_encoder.py` optional backend | Code MIT (per module comment; recollection); model trained on VoxCeleb/LibriSpeech | **Unknown** (VoxCeleb data terms are research-oriented) | https://github.com/resemble-ai/Resemblyzer | Optional, off by default |
| Speaker encoder `speechbrain` (ECAPA-TDNN) | same, optional backend | Code Apache-2.0 (recollection); `spkrec-ecapa-voxceleb` weights trained on VoxCeleb | **Unknown** | https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb | Optional; verify weight terms |
| Built-in speaker-encoder fallback (spectral) | `speaker_encoder.py` | Own code | Yes | this repo | Not identity-grade |

## Candidate models referenced only in `docs/model-selection.md` (not used in code)

| Item | Licence | Commercial verdict | Source |
|---|---|---|---|
| F5-TTS pretrained | Code MIT; weights CC-BY-NC (Emilia data) | **No** | https://github.com/SWivid/F5-TTS (gh) |
| IndicF5 | README states none; lineage from F5 unresolved; secondary source says MIT/gated | **Unknown** | https://github.com/AI4Bharat/IndicF5 (gh) |
| Indic Parler-TTS | Apache-2.0 (prior) | Yes (unverified this session) | https://huggingface.co/ai4bharat/indic-parler-tts |
| XTTS-v2 | Coqui Public Model License (non-commercial) (prior) | **No** | https://huggingface.co/coqui/XTTS-v2 |
| StyleTTS2 | Code MIT; pretrained weights: consent/disclosure condition; GPL phonemizer dependency (gh) | **Conditional** (no Hindi weights) | https://github.com/yl4579/StyleTTS2 |
| Fish-Speech / Fish Audio S2 | Fish Audio Research License (gh) | **No** without separate licence | https://github.com/fishaudio/fish-speech |
| CosyVoice 3 | Not stated in README (gh) | **Unknown** | https://github.com/FunAudioLLM/CosyVoice |
| Chatterbox (Multilingual V3 + Hindi) | MIT code (gh); HF "mit" (prior); mandatory Perth watermark | **Yes (unverified weights card)** | https://github.com/resemble-ai/chatterbox |
| NVIDIA Magpie TTS 357M | NVIDIA Open Model License (prior) | Conditional (revocable; NIM paid) | prior |
| Orpheus / svara / Veena | Llama 3.2 lineage; Veena no LICENSE file (prior) | **Unknown** | https://github.com/canopyai/Orpheus-TTS (gh: no licence text) |

## Datasets

| Dataset | Used for | Licence | Commercial verdict | Source |
|---|---|---|---|---|
| IndicTTS (IITM) | lineage of rohan, Indic-TTS | IITM EULA (no NC clause per prior read) | Conditional (read EULA) | https://www.iitm.ac.in/donlab/indictts/ (prior) |
| Blizzard 2013 (lessac) | lineage of Piper base / rohan | "Research purposes only", excludes synthesis products (prior) | **No** | prior |
| Emilia | F5-TTS pretraining | CC-BY-NC | **No** | gh (F5 README) |
| Rasa, LIMMITS, IndicVoices-R | IndicF5 training | Rasa CC BY 4.0 (prior); LIMMITS unverified | Unknown | prior |
| SYSPIN | SYSPIN VITS | CC-BY-4.0 (prior) | Yes with attribution | prior |
| VoxCeleb / LibriSpeech | speaker-encoder training | VoxCeleb research terms (recollection) | **Unknown** | recollection |
| Our own consented recordings | custom training path (`training/`) | Per data-rights manifest | Yes only if manifest shows permitted use | `training/data_rights.py` |

## Vocoders / encoders inside the above

Piper VITS: end-to-end (no separate vocoder). Kokoro: iSTFTNet decoder inside the Apache-2.0 checkpoint. Supertonic:
inside the OpenRAIL-M model. Qwen3-TTS: 12 Hz speech tokenizer shipped with weights, licence as Qwen3-TTS (Unknown).
No other standalone vocoder is used.

## Software dependencies

| Package | Where | Licence | Verdict | Evidence |
|---|---|---|---|---|
| piper-tts 1.8.0 (piper1-gpl) | `requirements.txt`, `indian_english.py` internals | GPL-3.0-or-later | **Conditional**: copyleft. Running it as a network service is fine; distributing images/binaries that include it obliges GPL compliance for the combined work | pip metadata (checked); https://github.com/OHF-Voice/piper1-gpl |
| espeak-ng (via piper-tts, misaki path in Kokoro) | phonemization | GPL-3.0 (recollection) | Conditional (same as above) | recollection |
| supertonic 1.3.1 | `requirements-engines.txt` | MIT sample code (gh); check package metadata | Yes (verify) | gh |
| kokoro-onnx 0.6.1 | same | MIT (gh) | Yes | https://github.com/thewh1teagle/kokoro-onnx |
| qwen-tts, librosa, torch | `requirements-qwen.txt` | Unknown (qwen-tts); ISC (librosa, recollection); BSD-3 (torch, recollection) | qwen-tts Unknown | recollection |
| fastapi, pydantic-settings, python-multipart | `requirements.txt` | MIT / MIT / Apache-2.0 (recollection) | Yes | recollection |
| uvicorn, httpx, websockets, numpy, soundfile | same | BSD-3 (recollection). soundfile bundles libsndfile (LGPL-2.1) | Yes (LGPL dynamic-link) | recollection |
| soxr (libsoxr) | resampling | LGPL-2.1+ (recollection) | Conditional (LGPL) | recollection |
| huggingface_hub, onnx, transformers | requirements | Apache-2.0 (recollection) | Yes | recollection |
| pytest | dev | MIT | Yes | recollection |
| piper-tts[train], cython, torchaudio | `training/requirements.txt` | GPL-3.0 (piper), BSD/Apache | Conditional (GPL) | pip metadata for piper |
| monotonic_align `core.pyx` fetched from piper1-gpl | `training/setup_env.sh` | GPL-3.0 (same repo) | Conditional | https://github.com/OHF-Voice/piper1-gpl |

## Flags to resolve before any commercial launch

1. Replace all shipped Piper Hindi voices with a voice trained from our own consented data (or another cleared model).
2. Resolve the `piper-checkpoints` base-model provenance before any commercial fine-tune.
3. Decide GPL posture for piper-tts/espeak-ng if we ever distribute an image, not just run a service.
4. Confirm Qwen3-TTS, speaker-encoder weights, and UTMOS terms, or keep them out of the shipped product.
5. Kokoro: assess the risk of vendor-synthesised training audio. Supertonic: implement the machine-generated disclosure.
6. Re-read every HF card/LICENSE and stamp the date here (blocked this session).

## Speaker encoder backends (optional, not installed by default)

| Backend | Code licence | Weights / data | Notes |
|---|---|---|---|
| `mfcc` (default) | this repo + librosa (ISC) | none | deterministic baseline, NOT a neural speaker verifier |
| `resemblyzer` | MIT | GE2E encoder shipped in the pip package, trained on VoxCeleb1/2 + LibriSpeech | verify VoxCeleb terms (research use) before commercial deployment |
| `speechbrain` (ECAPA-TDNN `spkrec-ecapa-voxceleb`) | Apache-2.0 | Apache-2.0 model card, trained on VoxCeleb | downloads weights on first use; same VoxCeleb data caveat |

Embeddings are used only for reference QA and evaluation, never to condition synthesis.
