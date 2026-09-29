# Expressive model selection (Phase 4)

Written 2026-09-29. Scope: which model(s) could give this platform natural Hindi/Hinglish, several male/female voices,
authorised cloning and emotion/style delivery, and which of them can be used commercially.

## Evidence quality: read this first

| Label | Meaning |
|---|---|
| **V-gh** | Verified this session by reading the project's primary GitHub README (raw.githubusercontent.com). |
| **P** | Taken from `docs/research.md` (earlier session; that document cites HF cards, LICENSE files and M4 runs). **Not re-verified here.** |
| **U** | Unverified. Stated as a lead only. |

**Hugging Face was unreachable from this session** (`EGRESS_BLOCKED` on huggingface.co), so no model card or weights
licence was re-read in this pass. Every weights-licence verdict that rests on an HF card is therefore **P or U**. Before
adopting any model, re-read its card and LICENSE file and record the date in `docs/licenses.md`. No latency, VRAM or
quality number below was measured by this pass; figures are quoted from the cited source, with the hardware it names.
Nothing here is legal advice.

Ground rules used throughout:

- **Prompted emotion is not validated control.** A model that follows "say this sadly" from a text prompt or tag has
  *prompt-based steering*. It becomes *control* only after we measure it (emotion-recognition and native-listener A/B on
  fixed sentences, and speaker-identity preservation). No model below has that evidence for Hindi.
- **CC-BY-NC, CPML, research-only and "non-commercial" licences are disqualifying** for a commercial calling product.
  The licence of the *weights* and of the *training data lineage* both matter (see F5-TTS below).
- Vendor latency numbers are English/Chinese, on datacentre GPUs, and say nothing about Hindi on our hardware.

## Comparison

Columns: weights licence / native Hindi / cloning / emotion-style mechanism / size / latency and VRAM as reported /
streaming. "Native" conditioning means trained-in control inputs; "prompted" means text/tag steering; "none" means no
control other than speaker and speed.

| Model | Weights licence (commercial?) | Native Hindi | Cloning | Emotion / style | Size | Reported latency / VRAM | Streaming | Provenance |
|---|---|---|---|---|---|---|---|---|
| **Piper VITS** (piper1-gpl 1.8.0) | Runtime GPL-3.0-or-later (pip metadata, checked here). Hindi voices: pratham/priyamvada CC BY-NC-SA data = **no**; rohan = IITM EULA but fine-tuned from lessac (research-only Blizzard data) = **no (risk)**. Train-from-scratch on own data = yes | Yes (3 voices) | No | None (speed/noise scales only) | ~15M (63 MB) | RTF 0.022, TTFA p50 74 ms end to end on M4 CPU (our `docs/benchmarks.md`) | Chunk-level (our server) | runtime V-gh (README only, licence from pip); rest P |
| **Supertonic 3** | Model OpenRAIL-M, sample code MIT (V-gh). Commercial use allowed but use-restrictions must be flowed down, incl. disclosing machine-generated audio (P). **Upstream repo archived** (V-gh) | `hi` token; vendor CER 5.34; naturalness unverified (P) | No (10 preset voices F1-F5/M1-M5) | None | ~99M (V-gh) | RTF 0.19-0.44 on M4 ORT CPU (P, our runs) | Whole-utterance only | V-gh + P |
| **Kokoro-82M v1.0** | Apache-2.0 for the model per kokoro-onnx README (V-gh); training data partly synthetic audio from closed TTS vendors, provenance caveat (P) | Thin: 4 Hindi voices, grade C, 1-10 h data (P) | No | None | 82M | RTF 0.21-0.25 on M4 CPU (P, our runs) | Chunk-level; 510-phoneme cap | V-gh + P |
| **SYSPIN VITS Hindi** | HF metadata "mit", data CC-BY-4.0 (P) | Yes, 1M+1F | No | None | 36M | RTF 0.08-0.11 CPU (P) | Chunk-level | P (drops English words, PER ~0.5) |
| **YourTTS** (VITS + speaker encoder) | U (Coqui weights; CPML/other lineage, not re-read) | **No Hindi** in the released model (U: recollection of en/fr/pt) | Yes, zero-shot | None native | ~87M (U) | U | Not native | U. Only useful as an architecture reference |
| **StyleTTS2** | Code MIT; pretrained weights require speaker consent or public disclosure of synthetic audio; inference needs a GPL phonemizer (V-gh) | **No**; pretrained components are English/JP/ZH (V-gh). Would need own Hindi training | Reference-style | Style diffusion (learned, not labelled emotions) (V-gh) | U | U | Not native | V-gh. Training from scratch on Hindi = large project, not a drop-in |
| **F5-TTS (upstream)** | Code MIT; **pretrained models CC-BY-NC** because of Emilia training data (V-gh) = **no** | No | Yes (ref audio + transcript) | Implicit from reference audio | U | RTF 0.039-0.147, latency 253 ms at 2 concurrent, single L20, 16 NFE, Triton/TRT-LLM (V-gh, vendor) | Chunked | V-gh |
| **IndicF5** (AI4Bharat) | **Unclear.** README states no licence; datasets Rasa, IndicTTS, LIMMITS, IndicVoices-R; "inspired by" F5-TTS (V-gh). A web-search snippet says "MIT, gated" (U, secondary). `docs/research.md` flags warm-start from CC-BY-NC F5 weights (vocab evidence) and an unanswered HF issue (P). Treat as **not cleared** | Yes, 11 languages (V-gh) | Yes, needs ref audio + transcript (V-gh) | Copies reference-audio tone (V-gh); no labelled control | ~340M (U) | RTF 5.56 on M4 MPS; digits garbled (P) | Not native | V-gh + P |
| **Indic Parler-TTS** | Apache-2.0 (P; card not re-read) | Yes; 4 Hindi named speakers (P) | No (description-conditioned voices only) | **Prompted**: free-text description of pitch, rate, style (parler README, V-gh, describes the mechanism for English Parler; Indic details P) | 0.94B (P) | Reported ONNX CPU RTF 4.14; 5-15 s per sentence on a G5 GPU (P, third-party) | Users could not get streaming working (P) | mechanism V-gh, rest P |
| **XTTS-v2** | Coqui Public Model License (CPML), non-commercial (P). Coqui the company is gone; the library fork is MPL-2.0 (V-gh) but that does not relicense the weights | Hindi is **not** in the documented 17 languages (P, U for the list) | Yes | Implicit from reference | ~467M (U) | "<200 ms streaming latency" claim in the fork README (V-gh, vendor, GPU unstated) | Yes (claimed) | V-gh + P. **Excluded: licence** |
| **Qwen3-TTS** (0.6B / 1.7B) | Repo README states no licence (V-gh); not verified anywhere else | **Hindi is not among the 10 supported languages** (V-gh) | Yes, 3 s ref + transcript | Instruction-controlled (VoiceDesign / CustomVoice): **prompted** | 0.6B, 1.7B (V-gh) | "97 ms end-to-end" (vendor, hardware unstated) (V-gh); RTF ~2 on M4 (our repo docstring) | Dual-track streaming (claimed) | V-gh. Existing `ENGINE=qwen3` is experimental for non-Hindi cloning |
| **Chatterbox Multilingual V3 + Hindi pack** | GitHub code MIT; HF "mit" (P). Watermark (Perth) on all output (V-gh) | Yes, dedicated Hindi pack (V-gh); vendor CER 2.55-6.72% (P) | Yes (ref clip) | Emotion "exaggeration" knob in earlier versions (U for V3); Turbo/Nano have paralinguistic tags, English (V-gh) | 500M V3 (V-gh) | TTFB 275 ms, RTF ~0.2 H100 unoptimised (P, vendor). ~2x slower than real time on M4 (P) | No official streaming (P) | V-gh + P. **Best-evidenced expressive GPU candidate** |
| **NVIDIA Magpie TTS Multilingual 357M** | NVIDIA Open Model License, "commercially usable"; revocable (P) | Yes (v2602+) (P) | No (5 fixed voices) | None documented (U) | 357M | NIM H100 TTFA 35 ms, 267 ms at 64 streams, English text; NIM needs paid licence (P) | Yes (P) | P |
| **CosyVoice 3** | Not stated in README (V-gh) | **Hindi not listed** (9 languages + dialects) (V-gh) | Yes | Instruction control: emotion, speed, volume (**prompted**) (V-gh) | U | "as low as 150 ms" streaming (vendor) (V-gh) | Yes | V-gh. Not usable for Hindi |
| **Fish-Speech / Fish Audio S2** | **Fish Audio Research License** (V-gh) = not commercial without a separate licence | Hindi not in the listed tiers (V-gh; tier 3 unchecked) | Yes | 15,000+ free-form tags, **prompted** (V-gh) | 4B slow + 400M fast (V-gh) | RTF 0.195, TTFA ~100 ms on H200 (V-gh, vendor) | Yes | V-gh. **Excluded: licence** |
| **Orpheus (Canopy)** and Hindi variants (svara-TTS, Veena, Maya, Vyom, Coriolis LoRA) | Llama-3.2 lineage; svara card says apache-2.0 but base is Llama 3.2 (P); Veena has no LICENSE file (P). Orpheus README here gives no licence (V-gh) | svara ~100 h Hindi, 1F+1M (P); Canopy Hindi is a research preview (V-gh, P) | Zero-shot cloning claimed for base (V-gh) | Emotion **tags** (English) (V-gh); Hindi tags unproven | 3B (V-gh) | svara M4 mlx RTF ~1.5-1.8; Orpheus 130-200 ms TTFB on GPU, English, vendor (P) | Yes (vLLM) | V-gh + P. Needs a Llama-3.2 licence review |
| **Sarvam Bulbul** | API only, not open (P) | Yes | n/a | n/a | n/a | n/a | n/a | P. Listening anchor only where ToS allows |

Excluded on licence per `docs/research.md` (P, not re-read): Voxtral TTS, rumik-oss-1, OmniVoice, Higgs TTS 3, Sooktam-2,
Spark-TTS, Llasa, MMS-TTS-hin, Breeze-TTS-2, SPRING_F5 (fine-tunes CC-BY-NC F5), F5-Hindi-24KHz is CC BY 4.0 but RTF
1.6-2.4 and drops Latin letters.

## What the evidence supports

1. **No candidate has verified, licence-clean, native-Hindi, labelled emotion control.** The models that steer style by
   prompt (Parler, CosyVoice, Fish, Qwen3, Orpheus tags) either lack Hindi, carry non-commercial licences, or are
   English-only for the control vocabulary. Prompt-steered delivery in Hindi must be reported as *prompted, unvalidated*.
2. **Fast CPU path: keep VITS-class + Kokoro/Supertonic as baselines.** Only these are measured real time on CPU with
   headroom for concurrency. Commercially clean voices require **our own consented recordings** trained with Piper
   (`docs/training.md`), because the shipped Piper Hindi voices are NC or tainted. Kokoro (Apache-2.0, 4 voices) and
   Supertonic (OpenRAIL-M, 10 voices, must disclose machine-generated) are the usable pre-trained CPU options; both are
   preset-voice, no cloning, no emotion.
3. **Expressive GPU path: bake off Chatterbox Multilingual V3 + Hindi pack against Magpie 357M**, with svara/Orpheus-Hindi
   as a conditional third only after a Llama-3.2 licence review. Chatterbox gives cloning and a Hindi-specific model
   under an MIT-tagged licence; its emotion control for Hindi is **unverified**. Magpie has the best published latency
   but fixed English-origin voices and a NIM dependency for self-hosting guarantees.
4. **Do not add an expressive engine to the serving path yet.** The plan requires evidence first; the only measurements
   in hand are M4 CPU runs where every LLM/flow model was slower than real time. Add an engine only after the GPU bake-off
   below produces numbers and a native-listener test.
5. **Blocked/unverified items** are listed at the end.

## Reproducible bake-off

Harness pieces already in the repo: `bench/quality.py` (CER/PER via Whisper + espeak, UTMOS22 proxy, signal checks),
`bench/bench.py` (WebSocket TTFA/RTF/underrun at 1/5/10/20 concurrency), `bench/matrix.sh`,
`bench/pronunciation_report.py` (text normalizer categories), `bench/quality_set.tsv` and `bench/sentences.txt` as the
shared inputs. Record `nvidia-smi`, driver, torch version and git SHA next to every result.

```bash
# 0. setup (CPU baselines)
pip install -r requirements.txt -r requirements-dev.txt -r requirements-engines.txt
python scripts/download_voices.py hi_IN-rohan-medium supertonic kokoro

# 1. text-side quality gate (no audio)
python bench/pronunciation_report.py --out bench/results/pronunciation.json

# 2. CPU baselines: intelligibility + naturalness proxy on the identical set
ENGINES=piper,supertonic,kokoro python -m bench.quality --label baseline-cpu
# 2b. serving latency at the concurrencies we can actually run
ENGINES=piper,supertonic,kokoro bench/matrix.sh "WORKERS=2 THREADS_PER_WORKER=4" -- --concurrency 1,5,10,20

# 3. GPU candidates (needs an NVIDIA GPU; not run in this session). Chatterbox example, in a separate venv:
python -m venv .venv-cb && .venv-cb/bin/pip install chatterbox-tts
#   write a thin script that loads the Hindi pack and calls generate(text, audio_prompt_path=<consented clip>) for
#   every line of bench/quality_set.tsv, saving wavs + wall time + torch.cuda.max_memory_allocated();
#   then score the wavs with the same Whisper CER/PER + UTMOS code in bench/quality.py.
# 3b. If a candidate is wrapped as an engine behind the serving interface, reuse bench.bench unchanged:
python -m bench.bench --url ws://localhost:8000/v1/audio/ws --concurrency 1,5,10,20 --label <candidate>-gpu

# 4. Emotion/style: only after 1-3. Fixed sentence x {neutral, happy, sad, calm, apologetic} x same speaker.
#    Objective: an emotion classifier + speaker-similarity to the neutral render (identity preserved?).
#    Decisive: blind native-Hindi listener A/B/MOS. UTMOS is English-trained and cannot settle Hindi naturalness.
```

Acceptance criteria to write down before running: Hinglish PER not worse than Kokoro; TTFA p95 under 500 ms at the
target concurrency on the chosen GPU; speaker similarity to the consented reference above a threshold set from
same-speaker vs different-speaker distributions; emotion classes distinguishable by >= N of M native listeners.

## Recommendation

- **Fast CPU path (production today):** Piper VITS trained on our own consented data (commercially clean, RTF ~0.02),
  with Kokoro as the more natural pre-trained preset option and Supertonic for Hinglish, subject to its disclosure terms.
  No emotion control; expose speed only.
- **Expressive GPU path (evidence-gated):** run the Chatterbox-Hindi vs Magpie-357M bake-off on a rented GPU, only with
  approval for GPU spend. Until a winner has native-listener evidence, API responses must state that style/emotion, where
  offered, is *prompted and unvalidated*.
- **Not recommended:** XTTS-v2, F5-TTS/SPRING_F5, Fish-Speech (licences); Qwen3-TTS and CosyVoice for Hindi (unsupported);
  IndicF5 until AI4Bharat clarifies weights licence and lineage; Parler for real time.

## Unverified claims (do not rely on)

- Every HF-card-derived licence and size marked P or U above (HF blocked this session).
- IndicF5 weights licence (README silent; secondary source says MIT/gated; lineage concern unresolved).
- Qwen3-TTS weights licence (README states none) and any Hindi capability.
- CosyVoice 3, Orpheus, and Magpie licences as stated by their own repos.
- YourTTS language list, size and licence; XTTS-v2 parameter count and language list.
- Chatterbox V3 emotion control in Hindi; Indic Parler Hindi speaker count.
- All GPU latency/VRAM figures: vendor, English/Chinese, datacentre hardware. None were measured here.
- Supertonic: upstream is archived, so no fixes or fine-tuning path.
