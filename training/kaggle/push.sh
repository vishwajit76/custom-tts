#!/usr/bin/env bash
# Push the Kaggle kernel with the launch settings stamped in (Kaggle script kernels cannot receive env vars).
#   training/kaggle/push.sh [--resume-from <hf path|auto>] [--experiment-id <id>] [--kernel-version v6] [--env KEY=VALUE]... [--slug <kaggle id>] [--dry-run]
# Copies training/kaggle to a temp dir, rewrites the `STAMP = {...}` line of train_kernel.py with GIT_SHA (git rev-parse HEAD, plus "-dirty" if the
# tree differs), KERNEL_VERSION, RESUME_FROM, EXPERIMENT_ID, then runs `kaggle kernels push -p <tmp>` (skipped with --dry-run). Never run while a
# session is still running (the new session would start immediately): check `kaggle kernels status vishwajit76/custom-tts-hindi-train` first.
set -euo pipefail
cd "$(dirname "$0")/../.."
RESUME=auto; EXP=""; KV=""; DRY=0; ENVS=""; SLUG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --resume-from) RESUME="$2"; shift 2;;
    --experiment-id) EXP="$2"; shift 2;;
    --kernel-version) KV="$2"; shift 2;;
    --env) ENVS="$ENVS$2
"; shift 2;;
    --slug) SLUG="$2"; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
SHA=$(git rev-parse --short=12 HEAD)
if ! git diff --quiet -- training/kaggle || ! git diff --cached --quiet -- training/kaggle; then SHA="$SHA-dirty"; fi
TMP=$(mktemp -d); cp training/kaggle/kernel-metadata.json training/kaggle/train_kernel.py "$TMP/"
GIT_SHA="$SHA" RESUME="$RESUME" EXP="$EXP" KV="$KV" ENVS="$ENVS" SLUG="$SLUG" python - "$TMP/train_kernel.py" <<'P'
import json, os, re, sys
p = sys.argv[1]; s = open(p, encoding="utf8").read()
st = {"GIT_SHA": os.environ["GIT_SHA"], "RESUME_FROM": os.environ["RESUME"]}
if os.environ["KV"]: st["KERNEL_VERSION"] = os.environ["KV"]
if os.environ["EXP"]: st["EXPERIMENT_ID"] = os.environ["EXP"]
for kv in filter(None, os.environ["ENVS"].replace("\\n", "\n").split("\n")):  # --env KEY=VALUE: any kernel env setting (VOICE_NAME, INIT_MODE, EXTRA_ARGS, ...)
    k, v = kv.split("=", 1); st[k] = v
if os.environ["SLUG"]:  # a second kernel id lets two experiments run side by side (one session per kernel)
    md = os.path.join(os.path.dirname(p), "kernel-metadata.json"); j = json.load(open(md)); j["id"] = os.environ["SLUG"]; j["title"] = os.environ["SLUG"].split("/")[1]; json.dump(j, open(md, "w"), indent=1)
s2, n = re.subn(r"^STAMP = \{.*$", lambda m: "STAMP = " + json.dumps(st) + "  # stamped by push.sh", s, count=1, flags=re.M)
assert n == 1, "STAMP line not found"
compile(s2, p, "exec"); open(p, "w", encoding="utf8").write(s2)
print("stamped:", st)
P
# vendor the files the in-kernel milestone evaluation needs (script kernels only upload train_kernel.py) as one base64 zip on the EVAL_BUNDLE_B64 line
python - "$TMP/train_kernel.py" <<'Q'
import re, sys
sys.path.insert(0, "training/kaggle")
import make_bundle
p = sys.argv[1]; s = open(p, encoding="utf8").read(); b = make_bundle.build()
s2, n = re.subn(r"^EVAL_BUNDLE_B64 = .*$", lambda m: 'EVAL_BUNDLE_B64 = "' + b + '"  # stamped by push.sh', s, count=1, flags=re.M)
assert n == 1, "EVAL_BUNDLE_B64 line not found"
compile(s2, p, "exec"); open(p, "w", encoding="utf8").write(s2); print("eval bundle:", len(b), "chars base64")
Q
echo "kernel dir: $TMP"
if [ "$DRY" = 1 ]; then grep -n '^STAMP' "$TMP/train_kernel.py" | cut -c1-200; exit 0; fi
export KAGGLE_API_TOKEN=${KAGGLE_API_TOKEN:-$(cat ~/.kaggle/access_token)}
kaggle kernels push -p "$TMP"
