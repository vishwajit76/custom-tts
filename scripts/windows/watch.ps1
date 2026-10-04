# Live V8 dashboard: GPU, Rasa download, data ingest, training. Ctrl+C to quit.
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pq = "$root\datasets\raw\rasa_parquet\Hindi"
$TOTAL = 28
function Section($t) { Write-Host "`n== $t ==" -ForegroundColor Yellow }
function Bar($done, $total, $w = 40) {
    $f = [int]([math]::Min(1, $done / [math]::Max(1, $total)) * $w)
    "[" + ("#" * $f) + ("." * ($w - $f)) + "]" + (" {0,5:N1}%" -f (100 * $done / [math]::Max(1, $total)))
}
function Cut($s, $n = 150) { if ($s.Length -gt $n) { $s.Substring(0, $n) } else { $s } }

while ($true) {
    Clear-Host
    Write-Host "V8 Hindi TTS  |  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan

    Section "GPU"
    $g = (nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm --format=csv,noheader,nounits) -split ',\s*'
    "util {0}%  |  VRAM {1}/{2} MiB  |  temp {3} C  |  power {4} W  |  clock {5} MHz" -f $g
    "util  " + (Bar ([int]$g[0]) 100 30)
    "VRAM  " + (Bar ([int]$g[1]) ([int]$g[2]) 30)
    if ([int]$g[3] -ge 87) { Write-Host "  HOT: GPU >= 87 C" -ForegroundColor Red }

    Section "Download: Rasa Hindi (aria2)"
    if (Test-Path "$root\datasets\prefetch.done") { Write-Host "  COMPLETE" -ForegroundColor Green }
    $files = Get-ChildItem $pq -Filter *.parquet -ErrorAction SilentlyContinue
    $partial = $files | Where-Object { Test-Path ($_.FullName + '.aria2') }
    $done = $files.Count - $partial.Count
    "files " + (Bar $done $TOTAL) + "  $done/$TOTAL done, $($partial.Count) active"
    $last = (Get-Content "$root\datasets\aria2.log" -Raw -ErrorAction SilentlyContinue) -split "[`r`n]" | Where-Object { $_ -match 'DL:' } | Select-Object -Last 1
    if ($last) {
        if ($last -match 'DL:([^\]]+)') { "speed $($Matches[1])/s" }
        [regex]::Matches($last, '#\w+ ([\d.]+\w+)/([\d.]+\w+)\((\d+)%\)') | ForEach-Object { "  file  " + (Bar ([int]$_.Groups[3].Value) 100 30) + "  $($_.Groups[1].Value)/$($_.Groups[2].Value)" }
    }
    elseif (-not (Get-Process aria2c -ErrorAction SilentlyContinue)) { Write-Host "  aria2 not running" -ForegroundColor DarkGray }

    Section "Ingest + quality filter"
    foreach ($d in 'rasa_hi_f', 'rasa_hi_m') {
        $n = (Get-ChildItem "$root\datasets\raw\$d" -Recurse -Filter *.wav -ErrorAction SilentlyContinue | Measure-Object).Count
        "{0,-10} {1,7} wavs" -f $d, $n
    }
    $tr = "$root\datasets\manifest\train.jsonl"
    if (Test-Path $tr) {
        $h = 0; Get-Content $tr | ForEach-Object { if ($_ -match '"duration":\s*([\d.]+)') { $h += [double]$Matches[1] } }
        "train manifest {0:N2} h  (updated {1:HH:mm})  |  EXP-002 starts at > 20 h" -f ($h / 3600), (Get-Item $tr).LastWriteTime
    }

    Section "Training"
    $log = Get-ChildItem "$root\exp\v8_kokoro", "$root\experiments" -Recurse -Filter *.log -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($log) {
        $age = [int]((Get-Date) - $log.LastWriteTime).TotalMinutes
        Write-Host ("{0}  (updated {1} min ago)" -f $log.FullName.Replace($root, '.'), $age) -ForegroundColor Green
        Get-Content $log.FullName -Tail 12 | ForEach-Object { "  " + (Cut $_) }
    }
    else { "no training logs" }
    Get-ChildItem "$root\exp\v8_kokoro", "$root\experiments" -Recurse -Include *.pth -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne 'base.pth' } | Sort-Object LastWriteTime -Descending | Select-Object -First 2 |
        ForEach-Object { "  ckpt {0}  {1:N0} MB  {2:HH:mm}" -f $_.FullName.Replace($root, '.'), ($_.Length / 1MB), $_.LastWriteTime }

    Section "Processes"
    Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -notmatch 'venv-v8' -and $_.CommandLine -match 'training|bench|datasets' } | ForEach-Object {
        $p = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
        "  pid {0,-6} ram {1,6:N0} MB  {2}" -f $_.ProcessId, ($p.WorkingSet64 / 1MB), (Cut ($_.CommandLine -replace '.*python.exe"?\s*', '') 100)
    }
    $os = Get-CimInstance Win32_OperatingSystem
    "  system RAM {0:N1} / {1:N1} GB  |  disk D: free {2:N0} GB" -f (($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB), ($os.TotalVisibleMemorySize / 1MB), ((Get-PSDrive D).Free / 1GB)
    Start-Sleep 5
}
