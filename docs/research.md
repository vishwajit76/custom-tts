# Model research and selection

Goal: self-hosted Hindi/Hinglish TTS for real-time AI phone calls: voices that sound human, several male and
female voices, commercial use. Dev hardware: Apple M4 (10 cores, 16 GB), no NVIDIA GPU.

Updated 2026-09-29. Replaces the Sep 28 version, which picked Piper for CPU speed. The priority is now
naturalness and voice choice, with real time as a hard constraint. Sources: six researcher reports, 17 adversarial
verifications against primary sources (HF API and cards, GitHub LICENSE files, papers, and model runs on this M4),
and my own checks, marked "(checked here)".

Provenance labels: **measured on M4** means run on this machine. **verified** means checked against primary
sources. **researcher-only** means claimed in one report and not re-checked. Where a verifier and a researcher
disagree, the verifier's finding is used and the conflict is noted. Nothing here is legal advice. Every
"commercial" verdict below needs counsel's sign-off.

## TL;DR

- **No open Hindi TTS has published human-naturalness evidence.** "Most human-like" stays unproven until our blind
  native-listener test runs (see "Naturalness evaluation"). UTMOS cannot settle it.
- **Tier A: CPU real time (this M4, commodity x86).** Keep the Piper ONNX VITS runtime. Only VITS-class models are
  measured real time on CPU with headroom for concurrent calls (Piper RTF 0.022, SYSPIN ~0.1). **All three Piper
  Hindi voices are commercially blocked**: two carry NC data, and rohan was fine-tuned from the lessac voice, whose
  data licence is research-only. **Measured on M4 (docs/benchmarks.md §7):**
  - **Kokoro-82M (2F+2M, Apache-2.0) is the most natural CPU engine.** UTMOS is 4.13-4.33 against Piper's 3.83-3.91, with intelligibility on par with Piper. It is the recommended default.
  - **Supertonic 3 (5F+5M, OpenRAIL-M with use conditions) is the best on Hinglish.** Its F3, F1, M4, F5 and M1 voices extend the choice.
  - **SYSPIN VITS is not a stopgap after all.** It drops English words (Hinglish PER ~0.5).
  - **Pocket TTS Hindi ships no licensed voice.**
  - **IITM FastSpeech2 (1F+1M, CC BY 4.0) is still unauditioned.**

  Long term, retrain voices on our own consented data. Expect clean TTS, not a human-sounding voice.
- **Tier B: most human-like, needs an NVIDIA GPU.** Run a bake-off between **Chatterbox Multilingual V3 + Hindi pack**
  (MIT; any consented male/female voice via a reference clip) and **NVIDIA Magpie TTS Multilingual 357M** (NVIDIA
  Open Model License, commercial; 5 fixed voices; best published latency, but via NVIDIA's paid NIM container).
  svara-TTS (Orpheus 3B) is the fine-tune route, but only after a Llama 3.2 licence review. Every LLM or
  flow-matching TTS measured on this M4 ran slower than real time (svara RTF ~1.8, Chatterbox ~2), so the Mac is a
  dev box only for this tier.
- **Custom voices** (the only route to many distinct, owned voices): record 2M+2F consented Hindi talents
  (1,000-2,000 lines each) and fine-tune on a rented GPU (see "Fine-tuning path").

## What changed since Sep 28

| Sep 28 statement | Now | Why |
|---|---|---|
| Piper is the engine | Piper is the Tier A runtime only | Its measured quality ceiling is UTMOS ~3.9-4.0 whatever the sampling settings, and the prosody grid gave no real gain (project measurement, Sep 29). The owner hears it as robotic. |
| rohan voice license "unresolved" | IITM EULA read (checked here): permissive. rohan still **not clean** | Its card says "Finetuned from U.S. English lessac voice". lessac's data is the Blizzard 2013 licence: "Research Purposes only", excluding "the development ... of voice synthesis ... products or services" (checked here). |
| Kokoro rejected (RTF 0.24, 10× Piper) | Candidate for voice variety | The priority is now naturalness and voice choice. RTF 0.24 is still 4× faster than real time per stream, and Kokoro offers 4 Apache-2.0 voices. The cost is CPU per stream and 0.4-1 s to first audio. |
| AI4Bharat Indic-TTS blocked by the IndicTTS data license | Candidate (unbenchmarked) | The IITM EULA has no non-commercial clause (checked here). |
| Qwen3-TTS kept as `ENGINE=qwen3` | Experiments only | Official checkpoints list 10 languages and Hindi is not among them (serving report; Supertonic's table also marks it "—" for Hindi). |
| Rasa: "no Hindi" | Rasa has Hindi: 1F 27.05 h + 1M 23.78 h, CC BY 4.0 | The HF language tags omit `hi`, but the card and config include Hindi (3 verifiers read the card; a 4th read only the tags). |
| LIMMITS: CC BY 4.0 | Unverified | The LIMMITS pages state no licence. The CC BY 4.0 claim comes only from the Indic Parler card. |
| IndicF5: unclear because of IndicTTS | Unclear because of **lineage** | The vocab keeps F5-TTS's 1,325 pinyin tokens at their original ids, a sign of warm-starting from the CC-BY-NC F5 weights (verifier). AI4Bharat has not answered HF #43. |
| Veena weights: commercial yes | Unclear | No LICENSE file, and an undisclosed Llama-3.2-architecture base (verifier). |

## Shortlist

Speeds: RTF = compute ÷ audio duration, single stream unless stated. TTFA = time to first audio.

| Model | Params | License verdict (weights / data) | Hindi native | Voices | Streaming | Speed (hardware, source) | M4 feasible | Provenance |
|---|---|---|---|---|---|---|---|---|
| **Piper hi_IN medium** (VITS, ONNX) | ~15M (63 MB fp32) | Code GPL-3.0 (piper1-gpl). pratham/priyamvada data "CC BY-NC-SA 4.0": **no**. rohan data "IITM EULA", lessac lineage: **no (risk)** | yes | rohan M, pratham M, priyamvada F | clause chunking (our server) | RTF 0.022 (4 threads), TTFA p50 74 ms end to end (M4 CPU, benchmarks.md) | yes, the reference | measured on M4 |
| **SYSPIN VITS Hindi** (Coqui) | 36.3M generator | Weights "license: mit" (HF metadata only, no LICENSE file). Data SYSPIN "CC-BY-4.0" (IISc copyright). **Yes** | yes (native speakers, read speech) | Hindi_Female, Hindi_Male | no, chunk by clause | CPU RTF 0.08-0.11, MPS 0.04-0.065, clause of 1.2-3.8 s audio in 0.09-0.35 s (M4 under load, verifier) | yes | measured on M4 |
| **IITM FastSpeech2 HS** (smtiitm) | not published (6 FFT blocks, hidden 256) | Repo "CC BY 4.0" (GitHub API + README, © 2025 Speech Technology Consortium, Bhashini MeitY, IIT Madras). Data not named in README; the New-Models branch ships the IITM EULA. **Yes, with attribution** (checked here) | yes | hindi male/female (22 kHz; 48 kHz `hindi_latest`) + hindi_male/female_bilingual | no | not published, not measured | likely (non-AR + HiFi-GAN), unmeasured | verified (license, voices) |
| **AI4Bharat Indic-TTS** (FastPitch + HiFi-GAN V1) | not published | Repo LICENSE "MIT" (© 2023 AI4Bhārat). Weights are release assets `hi.zip`, `en+hi.zip` (2023-01-18) with no separate licence. Data IndicTTS, IITM EULA. **Yes** (checked here) | yes | 1M + 1F in one model | no | not published, not measured | likely, unmeasured | verified (license) |
| **Kokoro-82M v1.0** | 81.8M | "apache-2.0" (metadata + README, no LICENSE file). Data partly "synthetic audio generated by closed TTS models" from unnamed providers. **Yes** (provenance caveat) | thin: 1-10 h Hindi, 10-100 min per voice, grade C | hf_alpha F, hf_beta F, hm_omega M, hm_psi M | no. Hindi path has no chunker; the 510-phoneme cap truncates | RTF 0.21-0.25 ORT CPU fp32, first audio 400 ms ("जी हाँ।") to ~1 s per sentence, plus ~0.5 s lead silence (M4, verifier) | yes | measured on M4 |
| **Supertonic 3** (Supertone) | ~99M (398 MB ONNX) | "BigScience Open RAIL-M License dated August 18, 2022". **Yes, conditional** (see note 1). Data undisclosed | `hi` token; Hindi CER 5.34 (vendor); naturalness unverified | F1-F5, M1-M5 | no | RTF 0.34-0.44 at 8 steps, 0.19-0.24 at 4 steps; first audio = whole utterance, 0.64-2.2 s (M4 ORT CPU, verifier) | yes | measured on M4 |
| **Pocket TTS Hindi v0.1** (Saryps Labs) | 109.5M | Weights "CC BY 4.0" (MODEL_LICENSE.md). Base kyutai/pocket-tts CC-BY-4.0 (gated; bans "fraudulent calls"). Data IndicVoices + Kathbath CC-BY-4.0. **Yes** | yes (932 h, noisy crowd data) | none shipped; clone from consented prompts | yes | fp32 3.4× real time (RTF ~0.3), TTFA 110 ms; int8 3.7×, 72 ms (AMD EPYC 7V13, 1 thread, card + JSON, verified) | probably (CPU PyTorch), unmeasured | verified |
| **Chatterbox Multilingual V3 + Hindi pack** (Resemble) | ~0.8B stack (T3 520M) | "license: mit" (HF metadata only; GitHub code MIT). Data "0.5M hours", not itemised. **Yes**. PerTh watermark on every output | yes (dedicated Hindi T3). CER 2.55-6.72% (vendor) | none; reference-clip cloning | no official streaming | TTFB 275 ms, RTF ~0.2 on H100 (vendor, unoptimized PyTorch). NIM TTFA 333 ms at 1 stream to 618 ms at 32 (B100). 8.38 s audio in 16-18.5 s (M4 mlx fp16) | no (2× slower than real time) | verified; M4 run by researcher |
| **Magpie TTS Multilingual 357M** (NVIDIA, v2607) | 357M | "NVIDIA Open Model License": "Models are commercially usable." Revocable; Trustworthy-AI terms forbid deceiving people. **Yes** | yes since v2602; v2607 IPA dictionary. CER 1.23% on an internal set | Aria, Sofia, Jason, Leo, John (English-origin speakers) | yes (NeMo-Speech.cpp rolling codec; NIM) | NIM on H100: TTFA 35 ms single stream, 267 ms at 64 streams (English text; NIM needs a paid NVAIE licence to self-host). Open magpie-tts.cpp on Ryzen CPU: RTF 2.6-3.0 | unproven (Swift MLX on M4 Pro: streaming RTF 0.93) | verified |
| **svara-TTS v1** (Kenpath, Orpheus-style) | 3.30B + SNAC | "apache-2.0" tag, but fine-tuned from canopylabs 3b-hi (Llama-3.2-3B base). Data SYSPIN, Rasa, IndicTTS, SPICOR. **Unclear** (Llama 3.2 terms) | yes, ~100 h | Hindi (Female), Hindi (Male) | yes (vLLM server) | M4 mlx 4-bit: RTF ~1.8 non-streaming; streaming first audio 1.5-1.6 s, sustained RTF ~1.5 (verifier). GPU: no svara figure; Orpheus-class 130-200 ms TTFB, 16-25 streams per H100 (vendor, English) | no | measured on M4 |
| **VoxCPM2** (OpenBMB) | 2.29B | "Apache-2.0" (LICENSE file; card: "free for commercial use"). Data 2M+ h, not itemised. **Yes** | in training data but "main weaknesses appear in Arabic and Hindi" (paper). Hindi WER 19.7% (MiniMax-MLS) | none; text voice design or cloning | yes | RTX 4090 Nano-vLLM: TTFB p50 67 ms at c=1, 189 ms at c=64; RTF 0.10 to 0.61 (language unstated). M4 Pro Metal Q8: RTF ~1.76 (vendor) | no | verified |

Note 1, Supertonic 3 (checked here, HF `Supertone/supertonic-3/LICENSE`). The text allows commercial service use.
§2 grants a "royalty-free, irrevocable copyright license". §4 says "You may host for Third Party remote access
purposes (e.g. software-as-a-service)". There is no non-commercial clause. It is not a plain "yes": §4a/§5
require passing the Attachment A use restrictions to our users, and §7 lets the licensor "restrict (remotely or
otherwise) usage of the Model in violation of this License". Attachment A bans use (e) "without expressly and
intelligibly disclaiming that the ... content is machine generated", (g) "to impersonate ... without their
consent", and (h) "for fully automated decision making that adversely impacts an individual's legal rights".
Every call must therefore disclose that the caller is an AI, and flows like automated collections need review. VoiceStudio's docs say only that OpenRAIL-M is outside
*its* blanket commercial statement; they do not contradict this reading. Upstream was archived on 2026-09-09 and
has no fine-tuning code.

Note 2. rohan, Indic-TTS, F5-Hindi and probably IITM FS2 all trace to the same two IndicTTS speakers (1M + 1F,
10.33 h). They are not four different voices.

## Excluded

| Model | Reason |
|---|---|
| Voxtral TTS, rumik-oss-1, OmniVoice, Fish S2 Pro / OpenAudio S1-mini, Higgs TTS 3, Sooktam-2, Spark-TTS, Llasa, XTTS-v2, MMS-TTS-hin, Breeze-TTS-2 | Non-commercial or research-only weights (CC BY-NC, CC BY-NC-SA, CPML, custom research licences). rumik explicitly forbids self-hosted commercial use. |
| SPRING_F5 (IIT Madras, 23 languages) | "apache-2.0" tag, but it fine-tunes F5TTS_v1_Base, which is CC-BY-NC: the vocab's first 2,545 entries are identical to upstream (verifier). **No.** |
| IndicF5 and its fine-tunes (Orato, 190M distilled student, Tharshan code-switch) | Lineage unclear (see the table above). Digits come out garbled. RTF 5.56 on M4 MPS. Tharshan was trained on OpenSLR-104 (CC BY-SA, share-alike). |
| F5-Hindi-24KHz (SPRINGLab) | CC BY 4.0, trained from scratch, but slow: RTF 1.6-2.4 at 16 steps on M4 MPS (verifier). No streaming. Latin letters dropped. Cloning only. |
| Indic Parler-TTS | Apache-2.0, with 4 named Hindi voices, but 0.94B autoregressive. ONNX CPU RTF 4.14 for Hindi. Field reports of 5-15 s per sentence (one on an AWS G5 GPU, one with hardware unstated). Users could not get streaming to work on this checkpoint. Use as an offline voice-design tool only. |
| Orpheus Hindi research release (canopylabs), Veena, Vyom-TTS-Hindi-3B, Coriolis Orpheus-Hindi LoRA | Llama 3.2 lineage with gated or undisclosed bases. Veena: no LICENSE file, voice gender inconsistent, stagnant since 2025-10. Canopy: 1 Hindi voice, known only from third-party docs. |
| MOSS-TTS v1.5 | Hindi was added by continued training only, with zero Hindi evaluation. 4.5-8.5B. Web-sourced data. Output is 48 kHz stereo. |
| ZONOS2 | Hindi is Tier 3, with the worst Hindi WER in its own table (15.5). 7.7B MoE. No training code. zonos2.cpp has no LICENSE. |
| Qwen3-TTS, CosyVoice 3, IndexTTS 2/2.5, Higgs v2, Kani, VibeVoice, Step-Audio, NeuTTS, Dia2, CSM, MeloTTS, OpenVoice, GEPARD, MOSS-TTS-Nano, dots.tts, GPT-SoVITS, Kyutai Pocket TTS (base) | No confirmed Hindi. IndexTTS 2.5 also has a custom licence with user-count limits. |
| Sarvam Bulbul, Gnani Vachana, svara Turbo, Cartesia, ElevenLabs | API-only, not self-hostable. Usable only as listening anchors where their ToS allow it (see "Licensing notes"). |

## Voice inventory (commercially usable today)

| Engine | Voices | Gender and how it was established | Commercial |
|---|---|---|---|
| SYSPIN VITS | Hindi_Female, Hindi_Male | Repo names plus corpus speaker metadata. A pitch check was inconclusive (the female voice may sit low), so a listening check is needed | yes |
| IITM FastSpeech2 | hindi female, hindi male (+ hindi_female/male_bilingual models) | README table and folder names | yes, CC BY 4.0 |
| AI4Bharat Indic-TTS | female, male | Paper and README ("trained jointly on male and female speakers") | yes |
| Kokoro | hf_alpha, hf_beta (F); hm_omega, hm_psi (M) | VOICES.md naming (hf = Hindi female, hm = Hindi male); measured on M4 | yes |
| Supertonic 3 | F1-F5 (F); M1-M5 (M) | supertonic-py docs/voices.md. The styles are language-agnostic, so Hindi naturalness is unverified | conditional (note 1) |
| Magpie Multilingual | Aria, Sofia (F); Jason, Leo, John (M) | **Inferred from first names only**; NVIDIA does not state gender | yes |
| Pocket TTS Hindi, Chatterbox, VoxCPM2 | any voice | From our own consented reference clip | yes |
| svara-TTS | Hindi (Female), Hindi (Male) | Speaker-ID labels. Each is a learned blend of speakers, not one person | unclear |
| Piper | rohan (M), pratham (M), priyamvada (F) | rohan: card ("Hindi Mono Male"). Others inferred from names | **no** |

Open studio data gives 6 real Hindi speakers (3M/3F): SYSPIN, Rasa and IndicTTS each have 1M + 1F. For more
voices that we own, record our own talent (see "Fine-tuning path").

## Serving and optimization

**Tier A (CPU, ONNX Runtime).** Measured on M4 (benchmarks.md):
- 4 intra-op threads: 2.2× lower latency. 2 workers × 4 threads is the best layout. Piper reaches 66× real time
  with 4 concurrent sessions.
- INT8 is slower on conv-dominated VITS (ConvInteger 4.3× slower) and on Kokoro (RTF 0.50 vs 0.24).
- The CoreML EP is slower than CPU. Dynamic batching is not worth it for VITS.
- SYSPIN's ONNX export works only through the legacy exporter and ran slower than PyTorch (RTF ~0.18 vs 0.08-0.11,
  verifier), so serve it through PyTorch or re-export it.
- Supertonic's SDK is ORT CPU only. A sherpa-onnx int8 package exists but is unmeasured.
- Pocket TTS runs at batch size 1. It scales as one stream per thread or process. Its ONNX INT8 path claims 4.7-6×
  on 2 threads (card).
- Streams per core for any engine except Piper: **not measured** (see "Open questions").

**Tier B (GPU).**
- **The M4 cannot run LLM TTS in real time.** SNAC models need ~82-83 tokens per second of audio. The base M4 has
  120 GB/s of bandwidth, which caps a 4-bit 3.3B model at ~48-64 tok/s (derived). Measured: svara at 44-58 tok/s,
  Chatterbox at 2× slower than real time.
- **L4/A10G:** about one 3B stream at best (derived, researcher-only).
- **Sizing:** RTX 4090/L40S for ≤2B models, H100 for 3B Orpheus-class.
- **Serving engines:**
  - vLLM for Orpheus, svara and Veena (plain Llama architecture). svara ships a FastAPI + vLLM server.
  - TensorRT-LLM FP8 for Orpheus: 16-24 streams per H100 MIG 3/7, under 150 ms TTFB on a full H100 (Baseten,
    English, vendor).
  - Nano-vLLM for VoxCPM2 (numbers in the table).
  - NIM for Magpie (NVAIE licence). The open-runtime speed is unproven.
  - Triton + TRT-LLM for F5: RTF 0.04 at concurrency 2, 253 ms latency on an L20 (English).
  - vLLM-Omni serves VoxCPM2, CosyVoice3, Qwen3-TTS and MOSS Delay/Realtime/Nano. SGLang-Omni is the only backend
    for MOSS Local.
- **MLX (mlx-audio):** runs Kokoro, svara, Chatterbox, Orpheus-Hindi and MOSS-Local on M4. Only Kokoro reaches real
  time. It is fine for listening tests.
- **llama.cpp/GGUF:** svara/Orpheus Q4_K_M is 2.09 GB. zonos2.cpp and magpie-tts.cpp exist but are not real time on
  CPU.
- **Quantization:** weight-only INT4 on the Orpheus LM changed UTMOS by ±0.03 and WER by +0.01 at 0.5× latency
  (English, arXiv 2609.28974, researcher-only). Validate on Hindi before using it. Keep codecs in fp16.
- **Concurrency degrades TTFA:** Chatterbox NIM goes from 333 ms to 618 ms (1 to 32 streams, B100), VoxCPM2 from
  67 ms to 189 ms (c=1 to 64, RTX 4090). Size for peak simultaneous *bot-speaking* streams.
- **GPU list prices (RunPod secure, fetched Sep 29, researcher-only):** L4 $0.49/h, RTX 4090 $0.74, L40S $1.09,
  H100 SXM $3.49.

## Fine-tuning path to a custom human-like voice

**Data (Hindi, open):**

| Dataset | License | Commercial | Hindi content | Notes |
|---|---|---|---|---|
| SYSPIN (IISc SPIRE) | CC-BY-4.0 (corpus README) | yes | 1M + 1F; ~55 h validated each | studio 48 kHz read speech; known text/audio errors, so filter first |
| Rasa (AI4Bharat) | CC-BY-4.0 (gated auto) | yes | 1F 27.05 h + 1M 23.78 h | expressive: emotions, commands, conversation. Closest open match to call style |
| IndicTTS (IIT Madras) | IITM EULA (checked here) | yes, with conditions | 1M + 1F, 10.33 h, 48 kHz | Keep the "COPYRIGHT 2016 TTS Consortium, TDIL, Meity ..." notice. §2.2: anyone who *buys* a derivative "shall not be allowed to further sell, lease, license, sub-license". Irrelevant for a hosted service; it matters if we sell voices |
| IndicVoices-R | CC-BY-4.0 (gated auto) | yes | 74.6 h, 399 speakers | restored ASR audio, N-MOS 3.38 vs 4.29 for IndicTTS. Use for diversity or pretraining only |
| LIMMITS '23/'24 | not stated on its pages | unverified | M + F | CC BY 4.0 per the Indic Parler card only |
| OpenSLR-104 (Hindi-English code-switch) | CC BY-SA 4.0 | risky (share-alike) | 95 h | |
| Own recordings | work-for-hire contract | yes | as recorded | the only route to many distinct, owned voices |

The vendor-API distillation rejected on ToS grounds (see "Licensing notes") stays rejected.

**Consent and recording protocol** (Microsoft custom-voice guide and AI4Bharat practice):
- A recorded consent statement from the talent in each session.
- A work-for-hire or ownership clause, plus a **separate AI-training clause**. Also term, territory, pay, a
  revocation and takedown path, and no cloning beyond the persona.
- India context (low confidence, secondary sources):
  - Bombay HC (Oct 2025, Asha Bhosle): cloning a voice without consent prima facie violates personality rights.
  - DPDP Rules 2025: consent is specific to each purpose.
  - IT Amendment Rules 2026: synthetic audio needs a disclosure.
- Studio: 48 kHz/24-bit mono, SNR > 35 dB, noise floor < -70 dB, a "match file" line every page, sessions of
  2-3 h/day.
- Script: ~300 utterances is the minimum; persona voices need 1,000-2,000 (Microsoft). Mix general and call-flow
  lines: greetings, OTP/EMI/amounts, Hinglish, objections, हाँ/अच्छा/hmm backchannels, questions 10-20%.

**Open training code and cost:**

| Model family | Training code | Compute / cost (source) |
|---|---|---|
| Piper VITS | piper1-gpl (GPL-3.0), in repo as `training/` | M4 MPS 7.7 s/step, impractical (measured). Plan on a single 24 GB GPU; hours not published. Start from `_base_model` (LibriTTS-R, CC BY 4.0), **not** `--init rohan` (lessac lineage) |
| Coqui VITS (SYSPIN) | coqui-tts (MPL-2.0). The checkpoint includes optimizer state, so it resumes directly | not published |
| IITM FastSpeech2 | smtiitm/Training_Fastspeech2_HS_Model (GitHub API reports no licence) | not published |
| Pocket TTS | Kyutai, released Aug 2026; Linux + one NVIDIA GPU | batch 16 × accumulation 4 in ~16 GB. 200 h quickstart ("a model that speaks"), 2,000 h for "one that actually sounds good". No documented recipe for adding voices to the Hindi checkpoint (verified) |
| Orpheus / svara | yes: Unsloth 16-bit LoRA r=64, lr 2e-4; quality after ~50 examples, best at ~300 per speaker | LoRA demo on 3 h of data: 1-2 h on T4/L4, ~$0.4-2 (researcher-only) |
| Magpie | NeMo. Align2Speak GRPO recipe used IndicTTS Hindi | not published |
| Chatterbox, Kokoro, Supertonic | no official training code | n/a |
| F5 from scratch (clean base) | F5-TTS code is MIT. **Never fine-tune the SWivid weights** (CC-BY-NC; maintainer: the restriction survives fine-tuning) | 8×A100 for ~1 week ≈ 1,344 A100-h, ~$1.9-2.8k (SPRINGLab compute × list prices) |

Expectation: a VITS fine-tuned on 10-50 h of clean read speech stays "good TTS", not conversational-human
(researcher). The human-like gains come from expressive or conversational data plus an LLM-class model.

## Naturalness evaluation

Automatic metrics catch artifacts and failures. They cannot rank human-likeness for Hindi.

| Signal | Use | Evidence |
|---|---|---|
| UTMOS22 (MIT; in `bench/quality.py`) | Artifact tripwire only | Spearman 0.09 against human ratings on 20 modern TTS systems (English, arXiv 2506.19441). Blind to prosody. Scored native Hindi speakers 1.7-2.4, below TTS (single practitioner). Piper's 3.9-4.0 says nothing about human-likeness |
| UTMOSv2 (MIT), Audiobox Aesthetics (CC-BY 4.0) | Secondary | Audiobox-CE had the best cross-lingual PCC (0.767, Chinese). Neither is validated on Hindi |
| NISQA, TorchAudio-Squim MOS | Evaluation only | Weights are CC BY-NC-SA / CC-BY-NC |
| PER/CER via Whisper large-v3-turbo (in repo) + one CTC judge (IndicConformer-600M, MIT, gated; or omniASR_CTC_300M_v2, Apache-2.0) | Intelligibility gate | Score a natural-speech floor through the same pipeline. Flag sentences where the judges disagree. Add an entity score (amounts, dates, OTPs) |
| Metric calibration | Before trusting any metric | ai4bharat/SpeechArenaBench `hi` (MIT, gated) has human preference pairs. Adopt only metrics that agree with humans ≥ 65% of the time |

**Human protocol (small, repeatable):**
- **Pairwise A/B** against one fixed reference voice. Options: A / B / both good / both bad.
- **Raters and material:** 12-16 native Hindi raters, 40-50 stratified sentences (greetings, numbers/EMI, Hinglish,
  long sentences, backchannels), same-gender pairs.
- **Controls:** blind randomised order, 3 repeated items, 2 control clips. Report Krippendorff α (target > 0.6).
  Label results "preliminary" below 16 raters.
- **Conditions:** run both **wideband and 8 kHz G.711**.
- **Human-or-machine (HFR) test:** add a forced-choice test, ~20 clips per system with real human clips as controls.
- **Precision:** 12 × 40 = 480 judgments resolves only large effects, about ±4.5 points on a win rate (±6.3 with
  clustering; researcher arithmetic).
- **Context:** in AI4Bharat's 120K-judgment Indic study, every commercial system beat open IndicF5 (Bradley-Terry
  805.8 vs 942.8-1128.5). Expressiveness and intelligibility drove preference.

## Real-time calling pipeline

| Stage | Budget (median) | Notes |
|---|---|---|
| Endpointing | 200-300 ms | Silero VAD `stop_secs` 0.2 + Smart Turn v3 (Hindi 93.44%, ~12 ms CPU) |
| STT final | ≤ 150 ms | Measure on 8 kHz Hindi call audio, not FLEURS |
| LLM first clause | ≤ 300 ms TTFT | Prefix caching. Prompt forces a short first clause |
| TTS first audio | ≤ 150 ms (p95 ≤ 250) | Piper measured 74 ms. GPU LLM-TTS claims 130-300 ms (vendor) |
| Return path | 50-100 ms | Co-locate near the carrier PoP (Mumbai) |
| **Voice-to-voice** | ~800 ms median, ~1.2 s p95 | Budget derived by the researcher from Pipecat, Vapi and Exotel guides. Human turn gaps are 0-300 ms |

**Barge-in:**
- Carriers clear only audio they have not yet played: Twilio `clear`/`mark`, Exotel `clear`/`mark`, Plivo
  `clearAudio`/`checkpoint`.
- So pace outbound audio in small chunks, track what actually played, and return played-ms on cancel.
- Our cancel ack is 2.4-4.5 ms (measured).
- Classify barge-ins so that caller backchannels (हाँ, अच्छा, ठीक है) do not cancel the bot.

**Backchannels and fillers:**
- The LLM emits them as text.
- A cache of pre-synthesised हाँ जी / अच्छा / एक मिनट per voice.
- "हाँ" is multifunctional, so its prosody matters.
- Keep disfluency ≤ 5-10%.

**8 kHz telephony:**
- **Carrier formats:** Twilio sends μ-law 8 kHz only. Plivo takes μ-law 8 kHz or L16 8/16 kHz. Exotel takes slin
  8/16/24 kHz.
- **Exotel chunk rules:** chunks must be ≥ 3.2 KB and a multiple of 320 B. Our 40 ms default frame is 640 B at
  8 kHz, **below that minimum**.
- **India trunks:** reportedly G.711 A-law (secondary source).
- **Server gaps:** the server needs μ-law/A-law output and anti-aliased downsampling.
- **Judge quality after the codec:** G.711 cuts content above 3.4 kHz, which carries the fricative cues. Rankings
  can change at 8 kHz (a Sarvam vendor study).

**Chunking:** split on danda। and ॥ (Pipecat does; LiveKit's default tokenizer does not). Never split inside numbers
or English words.

## Licensing notes that still hold

**Runtime:**
- `piper-tts` 1.8 (OHF-Voice/piper1-gpl) is GPL-3.0-or-later and bundles espeak-ng (GPL-3.0). Running it as a
  network service does not trigger source obligations. Distributing the Docker image or binaries does.
- Kokoro Hindi also pulls in espeak-ng and phonemizer-fork (GPL-3.0+).
- coqui-tts (the SYSPIN path) is MPL-2.0.
- ONNX Runtime is MIT. FastAPI/uvicorn/numpy/soundfile are permissive; soxr is LGPL-2.1.

**Output-use clauses that bind an AI calling product.** Each forbids deception or impersonation, so disclose
that the caller is an AI in every call:
- OpenRAIL-M Attachment A (Supertonic)
- NVIDIA Trustworthy-AI terms (Magpie)
- Orpheus card "Model Misuse" ("fraudulent calls")
- kyutai/pocket-tts prohibited uses
- VoxCPM2 usage notice
- Llama 3.2 AUP (svara/Orpheus)

Separately, Chatterbox watermarks all of its output (PerTh).

**Vendor-API distillation: not used** (Sep 28, unchanged). The terms of each vendor were checked on its own pages:

| Vendor | Training a model on its output | Clause |
|---|---|---|
| ElevenLabs | prohibited | "as part of a dataset...for training, fine-tuning...any machine learning" ([use-policy](https://elevenlabs.io/use-policy)) |
| Cartesia | prohibited | "use...any Output to research and develop products, models and services that compete" ([terms](https://cartesia.ai/legal/terms.html)) |
| Sarvam | prohibited | §10.5(a) "develop, train, test, fine-tune...any machine learning algorithm" ([terms](https://www.sarvam.ai/terms-of-service)) |
| Google Cloud / Gemini TTS | prohibited | "use...Generated Output to develop a similar or competing product" ([service terms](https://cloud.google.com/terms/service-terms)) |
| Fish Audio | prohibited | "use output...to develop models that compete" ([terms](https://fish.audio/terms/)) |
| Rime | prohibited | "use Output to train any artificial intelligence or machine learning model" ([terms](https://rime.ai/terms)) |
| AssemblyAI (ASR labels) | prohibited | §2.4(i) "...train...any speech-to-text, text-to-speech..." ([terms](https://www.assemblyai.com/legal/terms-of-service)) |
| OpenAI | unclear | bans only "models that compete with OpenAI" ([usage policies](https://openai.com/policies/usage-policies/)) |
| Inworld | unclear, leans allowed | no customer-side restriction found ([terms](https://www.inworld.ai/terms)) |
| MiniMax | not found | terms page not retrievable |

Sarvam's terms also bar using its output to "test" ML systems, so vendor audio can serve as a listening anchor only
where the vendor's terms allow it. Local Whisper (MIT) is the ASR judge.

## Reference: VoiceStudio

[debpalash/VoiceStudio](https://github.com/debpalash/VoiceStudio) was read at commit eef0e23 (worker-verified).

- **Licence:** AGPL-3.0. We take ideas only and reuse no code in our network service.
- **Its default engine:** OmniVoice, whose weights are CC-BY-NC.
- **What it adds for us:** no Hindi-capable, commercially licensed, CPU-real-time engine that we lack.
- **Engines new to us, none with confirmed Hindi:** dots.tts 2B (Apache-2.0), Confucius4-TTS (too slow on CPU),
  IndexTTS 2.5 (bilibili licence with user-count limits), CosyVoice 3, GPT-SoVITS (MIT), Breeze-TTS-2
  (non-commercial), MOSS-TTS-Nano 100M (Apache-2.0; 20 languages, Hindi not listed).
- **Pocket TTS:** upstream Kyutai Pocket TTS has no Hindi. The Saryps Labs Hindi model in our shortlist is a separate
  derivative.

**Ideas worth adopting:**
- merge chunks that contain no speakable characters;
- sentence-first streaming with an LRU cache of fixed phrases and shared in-flight renders;
- Devanagari-aware lexicon word boundaries (Python's `\b` breaks on matras);
- a synthesis timeout scaled by text length, and one serial slot per engine on MPS/CPU;
- pin HF models to commit SHAs;
- never peak-normalize near-silent audio (below -50 dBFS).

## Open questions (completeness critic)

| # | Gap that matters for the decision | Cheapest way to close it |
|---|---|---|
| 1 | Hindi naturalness: no open model has a human MOS or preference score | The A/B + HFR test from "Naturalness evaluation": 12 raters, the shortlisted voices, one afternoon |
| 2 | Do engine rankings change after 8 kHz G.711? No Hindi data exists | Run the bench output through an ffmpeg μ-law/A-law round trip; include the 8 kHz condition in the same test |
| 3 | Hinglish per engine: char-vocab models **silently drop Latin letters** (SYSPIN and F5-Hindi verified; Indic-TTS per researcher), Kokoro anglicises, Supertonic and FS2-bilingual are untested, and the Praxy paper says (qualitatively) that Chatterbox fails on code-mixing | Put our Devanagari transliteration and normaliser in front of every engine. 20 Hinglish lines per engine, PER plus listening |
| 4 | CPU streams per core: only Piper is measured | `bench/bench.py --concurrency 1,5,10` per CPU engine |
| 5 | IITM FastSpeech2 and Indic-TTS speed and quality on CPU: unmeasured | 30-minute install + RTF/TTFA run each |
| 6 | Streaming: VITS, FS2, Kokoro, Supertonic and Chatterbox are utterance-level. Is first-clause TTFA fine with our chunker? | Measure TTFA with `FIRST_CHUNK_CHARS` at 40 vs 120; also test prosody at chunk seams by ear |
| 7 | Stability of LLM TTS on phone scripts (skips, loops, comma hallucination, digits) | ASR round trip on 200 call sentences per Tier B engine; count insertions and deletions |
| 8 | Speaker drift across sentences for cloning models (Chatterbox, Pocket) | Listen to 10-turn dialogues rendered with one reference clip |
| 9 | Gender and timbre of inferred voices (Magpie, pratham/priyamvada, the low-pitched SYSPIN female) | 5-minute listen + F0 check |
| 10 | Licences: Llama 3.2 flow-down (svara/Orpheus); IndicF5 lineage (HF #43); rohan lessac lineage; Kokoro's synthetic-data provenance; Chatterbox MIT stated only in metadata; IITM EULA §2.2 if voices are ever sold; the IndicTTS download page could not be reached (site timed out), so any extra click-through terms are unseen | One counsel memo covering all of these. Written questions to AI4Bharat and Kenpath |
| 11 | Voice cloning law in India: personality rights, DPDP consent, the IT Rules 2026 disclosure duty for calling services | Indian counsel. Adopt the consent template ("Fine-tuning path") and an AI-disclosure line now |
| 12 | Do UTMOS or Audiobox agree with Hindi human preference? | One-day SpeechArenaBench `hi` meta-evaluation |
| 13 | Tier B throughput on Hindi text (Devanagari token counts, SNAC decode) on 4090/L40S/H100 | One rented-GPU day with our bench harness at concurrency 1/8/16/32 |
| 14 | espeak-ng Hindi schwa-deletion accuracy (affects Piper and Kokoro) | A hand-labelled list of 200 words through `espeak-ng -v hi -x` |
| 15 | Actual SYSPIN hours per speaker; LIMMITS licence text | Download and check `report.json` and the corpus README |
| 16 | Does Chatterbox's PerTh watermark survive, or audibly affect, 8 kHz μ-law? | Encode, decode, run the detector, listen |
| 17 | Speaking rate and filler frequency of real Hindi call-centre agents | Time 10 recorded calls by hand |

## Hands-on results on this machine

See docs/benchmarks.md, section "Engine comparison (Sep 29)".
