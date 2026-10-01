#!/usr/bin/env bash
# Back up / restore the long-train run to a private HF repo so a reclaimed container can resume.
# Usage: training/hf_sync.sh backup|restore|backup-data   Env: HF_REPO (default <user>/custom-tts-hindi-train), NAME (hi_f)
# Token: ~/.cache/huggingface/token (never committed).
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${NAME:-hi_f}; RUN=training/runs/$NAME
python - "$1" "$NAME" "$RUN" <<'P'
import sys, glob, os, shutil
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
mode, name, run = sys.argv[1:4]
api = HfApi(); repo = os.environ.get("HF_REPO") or f"{api.whoami()['name']}/custom-tts-hindi-train"
if mode == "backup":
    ck = sorted(glob.glob(f"{run}/lightning_logs/version_*/checkpoints/last.ckpt"), key=os.path.getmtime)
    if not ck: sys.exit("no last.ckpt yet")
    api.upload_file(path_or_fileobj=ck[-1], path_in_repo=f"runs/{name}/last.ckpt", repo_id=repo, commit_message=f"ckpt {name}")
    if os.path.exists(f"{run}/config.json"):
        api.upload_file(path_or_fileobj=f"{run}/config.json", path_in_repo=f"runs/{name}/config.json", repo_id=repo)
    print("backed up", ck[-1])
elif mode == "backup-data":
    api.upload_folder(folder_path=f"data/{name}", path_in_repo=f"data/{name}", repo_id=repo, commit_message="dataset")
    print("dataset uploaded")
elif mode == "restore":
    if not os.path.isdir(f"data/{name}/wavs"):
        snapshot_download(repo, allow_patterns=[f"data/{name}/*"], local_dir=".")
    if not glob.glob(f"{run}/lightning_logs/version_*/checkpoints/last.ckpt"):
        dst = f"{run}/lightning_logs/version_restored/checkpoints"; os.makedirs(dst, exist_ok=True)
        p = hf_hub_download(repo, f"runs/{name}/last.ckpt"); shutil.copy(p, f"{dst}/last.ckpt")
        try: shutil.copy(hf_hub_download(repo, f"runs/{name}/config.json"), f"{run}/config.json")
        except Exception: pass
        print("restored ckpt")
    else: print("local ckpt present")
P
