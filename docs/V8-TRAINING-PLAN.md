# V8 training plan: Kokoro (goonj) Hindi fine-tune on RTX 3060 6 GB

## Which code actually trains Kokoro weights
- hexgrad released no training code. The goonj checkpoint we ship (`BH-Builds/goonj-1-82M`) states on its model card: "Two-stage StyleTTS2 finetune (via the kikiri-tts recipe)" from Kokoro-82M v1.0, 15 speakers, stage 1 13 epochs bs 8, stage 2 12 epochs bs 6, 1x L40S 48 GB, ~7 h. Same recipe produced Thorsten-Voice German Kokoro.
- So the trainer is [semidark/kikiri-tts](https://github.com/semidark/kikiri-tts) (patched StyleTTS2 submodule `train_first.py` / `train_second.py`, 178-token Kokoro symbol map). Cloned to `exp/kikiri-tts` (gitignored, pinned submodules StyleTTS2 b1956da, kokoro b96fef9).
- Verified compatibility with our checkpoint: kikiri `kokoro_symbols` == `exp/goonj/config.json` vocab (all 114 entries); goonj's 5 modules load into the StyleTTS2 model with 0 missing / 0 unexpected keys. Export back to KModel needs the weight-norm key rename (see EXP-001 README); `kokoro_export.py` asserts the key set equals the base.

## Parameters (counted from the built model)
| module | M params | stage 1 | stage 2 |
|---|---|---|---|
| decoder (ISTFTNet) | 53.28 | train | train |
| predictor | 16.19 | frozen (unused) | train |
| style_encoder / predictor_encoder | 13.85 / 13.85 | train / - | train / train |
| text_encoder | 5.61 | train | train |
| bert + bert_encoder | 6.29 + 0.39 | - | train (bert_lr) |
| text_aligner / pitch_extractor | 7.87 / 5.25 | train | frozen |
| MPD / MSD / WavLM disc | 41.11 / 0.28 / 1.17 | train | train |
| diffusion | 25.33 | - | off (lambda_diff 0) |
| WavLM base+ (SLM, frozen) | ~94 | loaded | loaded |
| inference model (Kokoro) | 81.8 | | |

## VRAM math (fp32 master weights, AdamW)
- Stage 1 trainable 127.3 M x (4 B weight + 4 grad + 8 Adam) = 2.04 GB; frozen 63 M + WavLM 94 M x 4 B = 0.63 GB. Measured static: 2.43 GB after iter 1, 2.91 GB after iter 2 (aligner/pitch Adam states). CUDA context + desktop ~0.5 GB on NVML.
- That leaves ~2.3 GB for activations. Activations scale with the decoder crop (`max_len`) x batch: generator at 24 kHz plus MPD/MSD feature maps for real and fake, plus the WavLM SLM loss; aligner runs on the full clip (<= 10 s).
- bf16 autocast cuts activations ~10-15%; it does not shrink optimizer state. fp16 + GradScaler is **not usable** with this trainer: `optimizer.step("pitch_extractor")` runs on a module with no grads -> `AssertionError: No inf checks were recorded`. Gradient checkpointing and grad accumulation are not implemented upstream; not added (would be untested training changes).
- Stage 2 trains ~152 M (+ predictor, bert, predictor_encoder) -> ~2.45 GB optimizer+weights + same frozen extras: **does not fit 6 GB** (measured OOM at batch 2, max_len 80, bf16, with and without SLM adversarial; batch 1 crashes upstream in `WavLMLoss` because `.squeeze()` drops the batch dim).

## Measured sweep (stage 1, 40 iters each, `exp/v8_kokoro/sweep/*.log`; failure = OOM or NVML > 5.8 GB)
| config | torch peak MiB | NVML peak MiB | s/iter | samples/s | GPU util % | result |
|---|---|---|---|---|---|---|
| b1 l200 fp32 (kikiri default crop) | 5584 | 6032 | - | - | - | OOM |
| b1 l200 fp16 | - | - | - | - | - | GradScaler assertion |
| b1 l200 bf16 | 5340 | 5908 | 1.53 | 0.65 | 49 | NVML > 5.8, reject |
| b1 l140 bf16 | 4623 | 5142 | 1.47 | 0.68 | 52 | ok |
| b1 l100 fp32 | 4841 | 5429 | 1.58 | 0.63 | 49 | ok |
| b1 l100 fp32, 4 workers | 4832 | 5394 | 1.81 | 0.55 | 53 | ok, slower |
| b2 l100 fp32 | 5535 | 6015 | - | - | - | OOM |
| b2 l100 bf16 | 5422 | 5991 | 1.93 | 1.04 | 51 | NVML > 5.8, reject |
| **b2 l80 bf16** | **4955** | **5520** | **1.66** | **1.21** | **49** | **chosen** |
| b4 l100 bf16 | - | - | - | - | - | OOM |

GPU util sits at ~50% in every config and s/iter barely moves with batch, so the step is host-bound, not data-bound (more DataLoader workers made it slower on Windows spawn): per-iteration CPU monotonic alignment search, many `.item()` syncs, and hundreds of small weight-norm parametrization kernels. The GPU also runs at 87-89 C. Raising batch is the only lever that helped throughput, and memory caps it at 2. Not done (would need trainer changes): length-bucketed sampler, cached mels/F0 (mel compute is not the bottleneck at 50% util), CUDA graphs.

## Throughput follow-up (2026-10-04, 40-iter bench, b2 l80 bf16, EXP-001 data)
| config | s/iter | samples/s | GPU util % | NVML peak MiB |
|---|---|---|---|---|
| baseline (cudnn.benchmark on) | 1.461 | 1.37 | 44 | 5412 |
| cudnn.benchmark off (now default; `CUDNN_BENCH=1` restores) | 1.146 | **1.74** (+27%) | 36 | 5498 |

- py-spy profile (`py-spy record --subprocesses`): not data-bound (DataLoader 2.5-6%). With benchmark on, cudnn re-autotuned every step because aligner/style-encoder inputs are variable length. With it off, 35% of host time is `text_aligner` forward: the ASR attention decoder is a teacher-forced Python loop over text tokens (one LSTMCell + attention per token), so the step is bound by kernel launches. Next biggest are the decoder (10%), the SLM/generator losses (16%), the discriminator (7%), and the per-sample s2s CE loop (6%).
- Added but **not benchmarked**: `kokoro_train.py --ckpt-decoder` (gradient checkpointing on the ISTFTNet decoder, to try batch 3-4 at l80) and `kokoro_prep.py --gender female`. A female-only bench run dir is at `exp/v8_kokoro/bench` (1119 clips, 1.5 h).
- Not tried: torch.compile (triton not installed), cached mels or bucketing (data is not the bottleneck), power cap (`nvidia-smi` reports power.limit N/A on this laptop). GPU reached 85-88 C during benches.
- EXP-002 was not started (plan change: GPU training is last-resort only).

## Decision
- **Stage 1 Kokoro fine-tune locally**: batch 2, max_len 80 (1.0 s crop), bf16, TF32, max_audio 10 s, lr 1e-4, save every epoch, resume from newest. EXP-001 proves it: 336 iters, train mel 0.588 -> 0.331, val 0.346 -> 0.261 best, 4969 MiB peak, resume ok, exported KModel is intelligible (Whisper CER equal to base goonj on the check sentences).
- Stage 1 adapts the acoustic side (decoder + style) to the Rasa voice; duration/pitch prosody stays goonj's (predictor untouched; `kokoro_export.py` keeps the base voice's prosodic half).
- **Stage 2 (prosody + SLM adversarial) needs >6 GB**: run it on a larger GPU (goonj used an L40S; kikiri says 10 GB+, bs 4 on 12 GB), e.g. the existing `training/kaggle` route, starting from our local `first_stage.pth`.
- Fallbacks not needed now: Piper/VITS v7a fine-tune remains available (`training/train.py`); F5/IndicF5 LoRA rejected (IndicF5 not realtime here, RTF 4.5-8.5 in fp32, and no upstream LoRA path).

## Full-run estimate and command
Rasa ~20 h F + ~18 h M at ~4 s/clip is ~34k clips; at 1.3 samples/s that is ~7 h per stage-1 epoch (female only ~3.7 h). Checkpoints are 1.7 GB each (one per epoch): budget disk or prune old `epoch_1st_*.pth`.

```powershell
# after the Rasa manifest is built (train/validation/test.jsonl in -Manifest)
.\scripts\windows\train.ps1 -Prep -Manifest datasets\manifest -Run exp\v8_kokoro\EXP-002 -Epochs 10
# rerun the same command without -Prep to resume after a stop; then export:
.venv-v8\Scripts\python.exe training\v8\kokoro_export.py exp\v8_kokoro\EXP-002\ckpt\epoch_1st_00009.pth --out exp\v8_kokoro\EXP-002\export --samples experiments\EXP-002\samples
```
