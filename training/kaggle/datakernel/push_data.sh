#!/usr/bin/env bash
# Push the CPU data kernel with the repo code it needs vendored (script kernels upload one file). --dry-run: build only, print the dir.
#   training/kaggle/datakernel/push_data.sh [--dry-run]
set -euo pipefail
cd "$(dirname "$0")/../../.."
TMP=$(mktemp -d); cp training/kaggle/datakernel/kernel-metadata.json training/kaggle/datakernel/data_kernel.py "$TMP/"
.venv/bin/python - "$TMP/data_kernel.py" <<'P'
import base64, io, re, sys, zipfile
from pathlib import Path
FILES = ["app/__init__.py", "app/services/__init__.py", "app/services/text_normalizer.py", "app/services/hinglish.py", "app/services/lexicon_hi.tsv",
         "app/services/persona_grammar.py", *map(str, Path("app/services/pronunciation").rglob("*")), *map(str, Path("training").glob("*.py"))]
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FILES:
        if Path(f).is_file(): z.write(f, f)
p = sys.argv[1]; s = open(p, encoding="utf8").read(); b = base64.b64encode(buf.getvalue()).decode()
s2, n = re.subn(r'^CODE_B64 = .*$', lambda m: f'CODE_B64 = "{b}"  # stamped by push_data.sh', s, count=1, flags=re.M)
assert n == 1; compile(s2, p, "exec"); open(p, "w", encoding="utf8").write(s2); print("code bundle:", len(b), "chars")
P
echo "kernel dir: $TMP"
[ "${1:-}" = "--dry-run" ] && exit 0
export KAGGLE_API_TOKEN=${KAGGLE_API_TOKEN:-$(cat ~/.kaggle/access_token)}
kaggle kernels push -p "$TMP"
