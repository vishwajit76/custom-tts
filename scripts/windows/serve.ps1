# Local test app on 127.0.0.1:8000. Every 10 min: restart uvicorn only if tracked app/ code or optimizer rule files changed.
$root = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $root
$py = "$root\.venv-v8\Scripts\python.exe"
$env:ENGINES = "goonj,piper"; $env:MODELS_EXTRA = "voices,exp/v7"; $env:DEFAULT_VOICE = "hi_IN-custom-medium"
$env:THREADS_PER_WORKER = "2"; $env:PYTHONUTF8 = "1"
function Get-Sig {
  $files = @(git ls-files app) + @("optimizer/pron_dict.json", "optimizer/active_config.json", "optimizer/core.py") | Where-Object { Test-Path $_ }
  $h = foreach ($f in $files) { (Get-FileHash $f -Algorithm SHA1).Hash }
  ($h -join "") | ForEach-Object { [BitConverter]::ToString([Security.Cryptography.SHA1]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($_))) }
}
$sig = $null; $proc = $null
while ($true) {
  $new = Get-Sig
  if ($new -ne $sig -or $proc.HasExited) {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force; Start-Sleep 2 }
    Write-Host "$(Get-Date -f s) starting uvicorn (code changed or first run)"
    $proc = Start-Process $py -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000" -NoNewWindow -PassThru
    $sig = $new
  }
  Start-Sleep 600
}
