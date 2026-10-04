$root = Resolve-Path "$PSScriptRoot\..\.."
& "$root\.venv-v8\Scripts\python.exe" "$root\scripts\windows\benchmark_gpu.py"
