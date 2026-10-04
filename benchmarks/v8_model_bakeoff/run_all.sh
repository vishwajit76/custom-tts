#!/bin/bash
# sequential: singles then concurrency, one process at a time (no GPU contention)
cd /d/Github/custom-tts; export PYTHONUTF8=1; PY=.venv-v8/Scripts/python.exe; B=benchmarks/v8_model_bakeoff
E="piper_base_cpu piper_base_cuda piper_v7a_cpu piper_v7a_cuda piper_v7b_cpu piper_v7b_cuda kokoro_goonj f5_hindi_springlab f5_hindi_nfe16 mms_hin indicf5 indicf5_nfe16"  # indicf5: fp32, ~4x slower; REPORT used 10 rows (nfe32) + conc N=1 (nfe16)
for e in $E; do $PY $B/single.py $e 50 > $B/results/log_single_$e.txt 2>&1; done
for e in $E; do $PY $B/concurrency.py $e > $B/results/log_conc_$e.txt 2>&1; done
echo done > $B/results/ALL_DONE
