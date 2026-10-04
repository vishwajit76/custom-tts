"""StyleTTS2 training checkpoint -> Kokoro KModel dir (model .pth + config.json + voices/<name>.pt) and test samples.

  python training/v8/kokoro_export.py exp/v8_kokoro/EXP-001/ckpt/epoch_1st_00003.pth --out exp/v8_kokoro/EXP-001/export \
      --speaker 0 --base-voice exp/goonj/voices/hi_meera.pt --samples experiments/EXP-001/samples

Stage 1 retrains text_encoder/decoder/style_encoder (style_encoder starts random: Kokoro releases don't ship it), so the
voicepack's acoustic half (ref_s[:, :128], fed to the decoder) is re-extracted with the new style_encoder as the mean over the
speaker's training clips (same recipe as kikiri scripts/extract_voicepack.py). The prosodic half (ref_s[:, 128:], fed to the
predictor) is kept from the base goonj voice because stage 1 does not touch predictor/bert. After stage 2 re-extract both halves.
"""
import argparse, shutil, sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent), str(ROOT / "exp/kikiri-tts/StyleTTS2")]
TEXTS = ["नमस्ते, मैं आपकी कैसे मदद कर सकती हूँ?", "आपकी बुकिंग की पुष्टि हो गई है, टिकट आपके मोबाइल पर भेज दिया गया है।",
         "हमारी शाखा सोमवार से शनिवार तक सुबह दस बजे से शाम चार बजे तक खुली रहती है।"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="exp/goonj/kokoro_hindi_final.pth")
    ap.add_argument("--base-voice", default="exp/goonj/voices/hi_meera.pt")
    ap.add_argument("--speaker", default="0", help="speaker id in train_list.txt")
    ap.add_argument("--name", default="hi_rasa_f")
    ap.add_argument("--samples", help="write test wavs here")
    a = ap.parse_args()
    ck = Path(a.ckpt).resolve(); run = ck.parents[1]; out = (ROOT / a.out).resolve(); (out / "voices").mkdir(parents=True, exist_ok=True)
    net = torch.load(ck, map_location="cpu", weights_only=False)["net"]  # our own checkpoint (has optimizer pickles)
    strip = lambda sd: {k.removeprefix("module."): v for k, v in sd.items()}

    base = torch.load(ROOT / a.base, map_location="cpu", weights_only=True)
    # StyleTTS2 (kikiri) uses torch.nn.utils.parametrizations.weight_norm; kokoro 0.9.4 KModel uses the old weight_norm and
    # falls back to strict=False on mismatch, silently leaving those convs random (output = noise). Rename g/v back.
    old = lambda n: n.replace("parametrizations.weight.original0", "weight_g").replace("parametrizations.weight.original1", "weight_v")
    kw = {k: {"module." + old(n.removeprefix("module.")): t for n, t in (net[k] if k in net else base[k]).items()}
          for k in ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]}
    for k in kw:
        assert set(kw[k]) == set(base[k]), f"{k}: keys differ from base Kokoro layout"
    torch.save(kw, out / "model.pth")
    shutil.copy(ROOT / "exp/goonj/config.json", out / "config.json")

    from models import StyleEncoder
    from meldataset import preprocess  # same mel as training
    se = StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512); se.load_state_dict(strip(net["style_encoder"])); se.eval()
    wavs = [run / "wavs" / l.split("|")[0] for l in (run / "train_list.txt").read_text("utf-8").splitlines() if l.endswith("|" + a.speaker)]
    with torch.no_grad():
        acou = torch.stack([se(preprocess(sf.read(w, dtype="float32")[0]).unsqueeze(1)).squeeze(0) for w in wavs[:200]]).mean(0)
    pack = torch.load(ROOT / a.base_voice, map_location="cpu", weights_only=True).clone()  # [510, 1, 256]
    pack[..., :128] = acou
    torch.save(pack, out / "voices" / f"{a.name}.pt")
    print(f"voice from {len(wavs[:200])} clips, acoustic norm {acou.norm():.3f}")

    if a.samples:
        from bench.kokoro_pt import _kmodel
        from app.services import kokoro_engine
        m = _kmodel()(repo_id="hexgrad/Kokoro-82M", config=str(out / "config.json"), model=str(out / "model.pth")).eval()
        sd = Path(a.samples); sd.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(TEXTS):
            ps = kokoro_engine.phonemes(t)
            with torch.inference_mode():
                w = m(ps, pack[min(len(ps), 509) - 1], speed=1.0).float().numpy()
            assert np.isfinite(w).all() and np.abs(w).max() > 0.01, "silent/NaN output"
            sf.write(sd / f"{a.name}_{i}.wav", w, 24000)
            print(f"sample {i}: {len(w) / 24000:.2f}s peak {np.abs(w).max():.2f}")


if __name__ == "__main__":
    main()
