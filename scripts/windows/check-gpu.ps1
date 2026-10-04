$root = Resolve-Path "$PSScriptRoot\..\.."
& "$root\.venv-v8\Scripts\python.exe" "$root\scripts\check_gpu.py"
