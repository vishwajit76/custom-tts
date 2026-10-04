"""Prepare a Kokoro (StyleTTS2 stage-1/2) fine-tune run from our manifest, reusing kikiri-tts' patched StyleTTS2.

  python training/v8/kokoro_prep.py --manifest datasets/manifest --out exp/v8_kokoro/EXP-001 [--max-sec 10]

Writes <out>/{wavs/*.wav (24 kHz mono), train_list.txt, val_list.txt, base.pth, config.yml}.
Phonemes come from the app's Hindi frontend (app.services.kokoro_engine.phonemes), the same G2P the goonj checkpoint is served with.
Base weights: goonj (exp/goonj/kokoro_hindi_final.pth) converted to StyleTTS2 {'net': {...}} layout (kikiri TRAINING_GUIDE step 3).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
KIKIRI = ROOT / "exp/kikiri-tts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(KIKIRI / "StyleTTS2"))
from kokoro_symbols import dicts  # noqa: E402  (178-token Kokoro map, verified == exp/goonj/config.json vocab)


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="datasets/manifest")
    ap.add_argument("--out", default="exp/v8_kokoro/EXP-001")
    ap.add_argument("--base", default="exp/goonj/kokoro_hindi_final.pth")
    ap.add_argument("--max-sec", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--gender", help="keep only this manifest gender (e.g. female)")
    ap.add_argument("--max-len", type=int, default=80, help="decoder crop in mel frames (80 = 1.0 s @ hop 300); measured best fit on 6 GB")
    a = ap.parse_args()
    from app.services.kokoro_engine import phonemes

    out = (ROOT / a.out).resolve(); (out / "wavs").mkdir(parents=True, exist_ok=True)
    m = ROOT / a.manifest
    spk = {}
    stats = {}
    for split, files in {"train": ["train.jsonl"], "val": ["validation.jsonl", "test.jsonl"]}.items():
        lines, secs = [], 0.0
        for r in sum((rows(m / f) for f in files if (m / f).exists()), []):
            if r["duration"] > a.max_sec or (a.gender and r.get("gender") != a.gender):
                continue
            ps = "".join(c for c in phonemes(r.get("normalized_text") or r["text"]) if c in dicts)
            if len(ps) < 5:
                continue
            name = Path(r["audio_path"]).stem + ".wav"
            dst = out / "wavs" / name
            if not dst.exists():
                w, sr = sf.read(r["audio_path"], dtype="float32", always_2d=True)
                sf.write(dst, soxr.resample(w.mean(1), sr, 24000), 24000, subtype="PCM_16")
            sid = spk.setdefault(r["speaker"], len(spk))
            lines.append(f"{name}|{ps}|{sid}")
            secs += r["duration"]
        (out / f"{split}_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats[split] = {"clips": len(lines), "hours": round(secs / 3600, 4)}

    raw = torch.load(ROOT / a.base, map_location="cpu", weights_only=True)
    net = {k: {n.replace("module.", "", 1): t for n, t in raw[k].items()} for k in ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]}
    torch.save({"net": net}, out / "base.pth")

    cfg = yaml.safe_load(open(KIKIRI / "configs/config_german_ft.yml", encoding="utf-8"))
    cfg.pop("training", None)  # dead block (kikiri issue #14)
    s2 = KIKIRI / "StyleTTS2"
    cfg.update({
        "log_dir": str(out / "ckpt"), "batch_size": a.batch, "epochs": a.epochs, "epochs_1st": a.epochs, "epochs_2nd": a.epochs,
        "save_freq": 1, "max_len": a.max_len, "pretrained_model": str(out / "base.pth"), "load_only_params": True,
        "F0_path": str(s2 / "Utils/JDC/bst.t7"), "ASR_config": str(s2 / "Utils/ASR/config.yml"),
        "ASR_path": str(s2 / "Utils/ASR/epoch_00080.pth"), "PLBERT_dir": str(s2 / "Utils/PLBERT"),
    })
    cfg["data_params"].update({"train_data": str(out / "train_list.txt"), "val_data": str(out / "val_list.txt"),
                               "root_path": str(out / "wavs"), "OOD_data": str(out / "train_list.txt"),
                               "min_length": 30, "num_workers": 0})  # Windows: spawn workers re-import train_first; 0 is safe
    cfg["model_params"]["multispeaker"] = len(spk) > 1
    yaml.safe_dump(cfg, open(out / "config.yml", "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    (out / "dataset.json").write_text(json.dumps({"manifest": str(m), "max_sec": a.max_sec, "speakers": spk, "sr": 24000,
                                                  "g2p": "app.services.kokoro_engine.phonemes", **stats}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(out), "speakers": spk, **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
