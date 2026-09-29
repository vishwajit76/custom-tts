"""Export a trained checkpoint to ONNX as a server voice, then smoke-evaluate it.

  .venv-train/bin/python -m training.export --run runs/myvoice --voice hi_IN-myvoice-medium [--ckpt path] [--models-dir models/piper]

Writes <models-dir>/<voice>.onnx + .onnx.json (the server loads every such pair at start-up).
Evaluate properly afterwards with the server venv:  python -m bench.quality --voice <voice> --test-csv data/myvoice/test.csv
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from training.train import last_checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--voice", required=True, help="e.g. hi_IN-myvoice-medium")
    p.add_argument("--ckpt", type=Path, help="default: the run's last.ckpt")
    p.add_argument("--models-dir", type=Path, default=Path("models/piper"))
    a = p.parse_args()

    ckpt = a.ckpt or last_checkpoint(a.run)
    if not ckpt:
        sys.exit(f"no checkpoint under {a.run}")
    onnx = a.models_dir / f"{a.voice}.onnx"
    a.models_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "piper.train.export_onnx", "--checkpoint", str(ckpt), "--output-file", str(onnx)], check=True)
    cfg = json.loads((a.run / "config.json").read_text("utf-8"))
    cfg.setdefault("language", {"code": "hi_IN"})
    cfg["dataset"] = a.voice
    Path(f"{onnx}.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    from piper import PiperVoice

    v = PiperVoice.load(str(onnx))
    t = time.perf_counter()
    audio = [c.audio_float_array for c in v.synthesize("नमस्ते, मैं आपकी कैसे मदद कर सकती हूँ?")]
    secs = sum(len(x) for x in audio) / v.config.sample_rate
    rtf = (time.perf_counter() - t) / secs
    peak = max(float(abs(x).max()) for x in audio)
    print(json.dumps({"onnx": str(onnx), "from": str(ckpt), "audio_s": round(secs, 2), "rtf": round(rtf, 3), "peak": round(peak, 3)}))
    assert secs > 0.5 and peak > 0.01, "exported voice produced silence"


if __name__ == "__main__":
    main()
