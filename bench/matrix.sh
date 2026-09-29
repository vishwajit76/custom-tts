#!/usr/bin/env bash
# Server config matrix: restarts the server per config, runs bench.bench, results land in bench/results/.
# Usage: bench/matrix.sh "WORKERS=5 THREADS_PER_WORKER=1" "WORKERS=2 THREADS_PER_WORKER=4" -- --concurrency 1,5,10
set -euo pipefail
PORT=${PORT:-8019}; PY=${PY:-.venv/bin/python}
configs=(); while [[ $# -gt 0 && $1 != "--" ]]; do configs+=("$1"); shift; done; shift || true
for cfg in "${configs[@]}"; do
  label=$(echo "$cfg" | tr ' =' '_-')
  env $cfg CACHE_SIZE=0 API_KEYS= "$PY" -m uvicorn app.main:app --port "$PORT" --log-level warning > "bench/results/$label.server.log" 2>&1 &
  pid=$!
  until curl -sf "localhost:$PORT/health" >/dev/null; do sleep 0.5; done
  echo "## $cfg"
  "$PY" -m bench.bench --url "ws://localhost:$PORT/v1/audio/ws" --label "$label${LABEL_SUFFIX:-}" "$@" | tail -n +1
  kill $pid; wait $pid 2>/dev/null || true
done
