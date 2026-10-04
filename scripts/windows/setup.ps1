# Idempotent. Needs uv (Python 3.12 is fetched by uv). CUDA toolkit not needed: torch wheel bundles runtime.
$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot\..\.."
$py = "$root\.venv-v8\Scripts\python.exe"
if (-not (Test-Path $py)) { uv venv "$root\.venv-v8" --python 3.12 --seed }
uv pip install --python $py torch torchaudio --index-url https://download.pytorch.org/whl/cu124
uv pip install --python $py -r "$root\requirements.txt"
foreach ($t in "ffmpeg","git") { if (-not (Get-Command $t -ErrorAction SilentlyContinue)) { Write-Warning "$t missing" } else { "ok: $t" } }
& $py -m pip --version
& $py "$root\scripts\check_gpu.py"
