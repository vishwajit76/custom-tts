#!/usr/bin/env bash
# Training environment (separate from the server venv: piper[train] pins librosa<1, pulls torch + lightning).
# The piper-tts wheel ships the training code but not the compiled monotonic_align extension; build it here.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PY=${PY:-python3.12}
[ -d .venv-train ] || uv venv -q --python "$PY" .venv-train
uv pip install -q --python .venv-train/bin/python -r training/requirements.txt
MA=$(.venv-train/bin/python -c "import piper.train.vits, pathlib; print(pathlib.Path(piper.train.vits.__file__).parent / 'monotonic_align')")
if ! ls "$MA"/monotonic_align/core*.so >/dev/null 2>&1; then
  tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/core.pyx" https://raw.githubusercontent.com/OHF-Voice/piper1-gpl/main/src/piper/train/vits/monotonic_align/core.pyx
  (cd "$tmp" && "$ROOT/.venv-train/bin/cythonize" -i core.pyx >/dev/null)
  mkdir -p "$MA/monotonic_align" && mv "$tmp"/core*.so "$MA/monotonic_align/"
fi
.venv-train/bin/python -c "from piper.train.vits.monotonic_align import maximum_path; print('training env ok')"
