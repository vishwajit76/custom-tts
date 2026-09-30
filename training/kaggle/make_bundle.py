"""Pack the repo files the in-kernel milestone evaluation needs (Kaggle script kernels upload only train_kernel.py) into one base64 zip string.
push.sh writes it into the `EVAL_BUNDLE_B64 = ...` line of the stamped copy of train_kernel.py; the kernel unzips it under /tmp/w/evalsrc and runs
`python -m bench.kernel_eval` there. Keep FILES in sync with bench/kernel_eval.py's imports (tests/test_kernel_eval.py checks the closure)."""
import base64
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = ["app/services/hinglish.py", "app/services/text_normalizer.py", "app/services/speaker_encoder.py", "app/services/indian_english.py", "app/services/lexicon_hi.tsv",
         "training/asr.py", "bench/compare_checkpoints.py", "bench/milestone_eval.py", "bench/kernel_eval.py", "bench/check_eval_overlap.py", "bench/hi_eval_50.txt"]


def build() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in FILES:
            z.write(ROOT / f, f)
    return base64.b64encode(buf.getvalue()).decode()


if __name__ == "__main__":
    print(build())
