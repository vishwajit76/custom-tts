# EXP-001: Kokoro (goonj) Hindi stage-1 smoke fine-tune

| field | value |
|---|---|
| model | Kokoro-82M, base = goonj Hindi fine-tune `exp/goonj/kokoro_hindi_final.pth` |
| trainer | kikiri-tts patched StyleTTS2 `train_first.py` (stage 1: text_encoder, style_encoder, decoder, text_aligner, pitch_extractor, MPD/MSD; WavLM SLM loss) via `training/v8/kokoro_train.py` |
| dataset | `datasets/manifest` sample: 84 train / 4 val clips, 0.128 h, Rasa Hindi F+M (2 speakers), resampled 22.05k -> 24 kHz, G2P = `app.services.kokoro_engine.phonemes` (`dataset.json`, `train_list.txt`) |
| steps | 8 epochs x 42 iters = 336 (4 epochs, then resumed from `epoch_1st_00003.pth` to 8) |
| batch / accum | 2 / none (StyleTTS2 has no accumulation) |
| lr | 1e-4 OneCycle (kikiri default), bert_lr unused in stage 1 |
| precision | bf16 autocast, TF32, cudnn.benchmark |
| max_len | 80 mel frames (1.0 s decoder crop); full clip (<=10 s) through aligner |
| GPU | RTX 3060 Laptop 6 GB |
| VRAM | torch peak 4969 MiB, NVML peak 5572 MiB (incl. desktop); no spill |
| speed | 1.47 s/iter, 1.36 samples/s, GPU util 51% during train iters, GPU 89 C max |
| duration | 631 s wall (both runs, incl. load + per-epoch val/TB probe) |
| loss | train mel 0.588 (step 10) -> 0.331 (last); val mel 0.346 (ep0) -> 0.261 best / 0.291 last |
| sample quality | `samples/hi_rasa_f_*.wav` (exported epoch 7, Rasa-F voicepack) vs `samples/base_goonj_meera_*.wav`. Whisper CER on 3 sentences 0.166 for both (errors = Whisper writing "booking/mobile" in Latin + numerals). No listening test. |

Notes
- Proves: loop runs, loss decreases, checkpoints save every epoch, resume restores weights+optimizer+epoch, export -> KModel inference gives intelligible audio.
- Resume re-runs the saved epoch index once (upstream `load_checkpoint` returns the saved epoch as the start epoch): 9 val values for 8 epochs.
- Export bug found and fixed: kikiri's StyleTTS2 saves `parametrizations.weight.original0/1`; kokoro 0.9.4 KModel expects `weight_g/weight_v` and silently loads with strict=False -> noise. `kokoro_export.py` renames and asserts the key set equals the base checkpoint.
- Samples peak near 1.0 (louder than base); check loudness normalisation in the full run.
- Checkpoints are 1.7 GB each (`exp/v8_kokoro/EXP-001/ckpt`, gitignored); epochs 0-5 deleted.
