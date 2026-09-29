#!/usr/bin/env bash
# End-to-end pipeline check on synthetic data: synthesize clips with an installed voice -> prepare_dataset ->
# fine-tune 1 epoch -> export ONNX -> synthesize. Proves the mechanics; the resulting voice is NOT a trained model.
# Usage: training/smoke_test.sh [workdir]   (needs .venv with the server deps and .venv-train from training/setup_env.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
W=${1:-training/runs/smoke}
rm -rf "$W" && mkdir -p "$W/raw"
.venv/bin/python - "$W/raw" <<'PY'
import sys, numpy as np, soundfile as sf
from piper import PiperVoice
from piper.config import SynthesisConfig
from bench.bench import SENTENCES
from app.services.text_normalizer import normalize
out = sys.argv[1]; v = PiperVoice.load("models/piper/hi_IN-rohan-medium.onnx"); rows = []
for i, s in enumerate(SENTENCES * 3):
    ls = (0.9, 1.0, 1.1)[i // len(SENTENCES)]
    wav = np.concatenate([c.audio_float_array for c in v.synthesize(normalize(s), SynthesisConfig(length_scale=ls))])
    if i % 5 == 0:
        wav = wav + np.random.default_rng(i).normal(0, 0.01, len(wav)).astype(np.float32)  # exercise --denoise
    sf.write(f"{out}/utt{i:03d}.wav", wav, 22050); rows.append(f"utt{i:03d}|{s}")
sf.write(f"{out}/bad_mismatch.wav", np.zeros(22050 * 2, np.float32) + 0.1 * np.sin(np.arange(44100) / 9), 22050)
rows.append("bad_mismatch|" + " ".join(SENTENCES))  # 2 s of audio, 700+ chars: must be rejected
open(f"{out}/metadata.csv", "w").write("\n".join(rows) + "\n")
PY
.venv/bin/python -m training.prepare_dataset --input "$W/raw" --output "$W/data" --denoise
.venv-train/bin/python -m training.train --data "$W/data" --run "$W/run" --init rohan --epochs 1 --batch-size 8 \
  --accelerator "${ACCEL:-auto}" --trainer.log_every_n_steps 1
.venv-train/bin/python -m training.train --data "$W/data" --run "$W/run" --epochs 1 --batch-size 8 --accelerator "${ACCEL:-auto}"  # resume
.venv-train/bin/python -m training.export --run "$W/run" --voice hi_IN-smoke-medium --models-dir "$W/models"
echo "smoke test ok"
