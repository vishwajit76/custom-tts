# Kokoro (goonj) Hindi fine-tune on RTX 3060 6GB. Resumes from the newest checkpoint in <Run>\ckpt automatically.
#   .\scripts\windows\train.ps1                                  # stage 1, EXP-001 config
#   .\scripts\windows\train.ps1 -Stage 2 -Epochs 10
#   .\scripts\windows\train.ps1 -Prep -Manifest datasets\manifest_full -Run exp\v8_kokoro\EXP-002
param([string]$Run = "exp\v8_kokoro\EXP-001", [int]$Stage = 1, [int]$Epochs = 0, [string]$Precision = "bf16",
      [switch]$Prep, [string]$Manifest = "datasets\manifest", [int]$Batch = 2)
$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $root
$py = "$root\.venv-v8\Scripts\python.exe"
$env:PYTHONUTF8 = "1"
if ($Prep -or -not (Test-Path "$Run\config.yml")) {
    & $py training\v8\kokoro_prep.py --manifest $Manifest --out $Run --batch $Batch
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}
$a = @("training\v8\kokoro_train.py", $Run, "--stage", $Stage, "--precision", $Precision)
if ($Epochs) { $a += @("--epochs", $Epochs) }
& $py @a 2>&1 | Tee-Object -Append "$Run\training.log"
exit $LASTEXITCODE
