"""Download voices so the server runs offline afterwards.

Usage: python scripts/download_voices.py [hi_IN-rohan-medium ... supertonic kokoro]
Piper voices come from https://huggingface.co/rhasspy/piper-voices into MODELS_DIR (default models/piper). Check each
voice's MODEL_CARD license before commercial use (docs/research.md): the hi_IN pratham/priyamvada voices are
CC BY-NC-SA (non-commercial).
  supertonic: Supertone/supertonic-3 at the revision the installed package pins, ~400 MB -> SUPERTONIC_DIR
  kokoro:     kokoro-onnx v1.0 release files (fp32 model 326 MB + all voices 28 MB) -> KOKORO_DIR
"""
import shutil
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import settings  # noqa: E402

DEFAULT = ["hi_IN-rohan-medium"]
KOKORO = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"

for name in sys.argv[1:] or DEFAULT:
    if name == "supertonic":
        from supertonic.loader import download_model, has_all_onnx_modules

        if not has_all_onnx_modules(settings.supertonic_dir):
            download_model(settings.supertonic_dir, "supertonic-3")
        print(f"{name} -> {settings.supertonic_dir}")
        continue
    if name == "kokoro":
        settings.kokoro_dir.mkdir(parents=True, exist_ok=True)
        for f in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
            if not (settings.kokoro_dir / f).exists():
                urllib.request.urlretrieve(KOKORO + f, settings.kokoro_dir / f"{f}.part")
                (settings.kokoro_dir / f"{f}.part").rename(settings.kokoro_dir / f)
        print(f"{name} -> {settings.kokoro_dir}")
        continue
    lang, speaker, quality = name.split("-")
    base = f"{lang.split('_')[0]}/{lang}/{speaker}/{quality}"
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    for f, dest in [(f"{name}.onnx", f"{name}.onnx"), (f"{name}.onnx.json", f"{name}.onnx.json"), ("MODEL_CARD", f"{name}.MODEL_CARD")]:
        shutil.copy(hf_hub_download("rhasspy/piper-voices", f"{base}/{f}"), settings.models_dir / dest)
    print(f"{name} -> {settings.models_dir}")
