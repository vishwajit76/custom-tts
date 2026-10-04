# Fast Rasa Hindi download: aria2c, 16 connections/file (HF throttles per connection).
# Output: datasets/raw/rasa_parquet/Hindi/*.parquet, then writes datasets/prefetch.done.
import os, subprocess, sys
from pathlib import Path
for line in Path(".env").read_text().splitlines():
    k, _, v = line.partition("=")
    if k.strip() in ("HF_TOKEN", "HUGGINGFACE_ACCESS_TOKEN") and v.strip():
        os.environ.setdefault("HF_TOKEN", v.strip().strip('"\''))
from huggingface_hub import list_repo_files
files = [f for f in list_repo_files("ai4bharat/Rasa", repo_type="dataset") if f.startswith("Hindi/")]
out = Path("datasets/raw/rasa_parquet")
lst = out / "aria2.txt"
out.mkdir(parents=True, exist_ok=True)
lst.write_text("".join(f"https://huggingface.co/datasets/ai4bharat/Rasa/resolve/main/{f}\n  out={f}\n" for f in files))
aria = Path(os.environ["LOCALAPPDATA"]) / "Microsoft/WinGet/Links/aria2c.exe"
cmd = [str(aria), "-i", str(lst), "-d", str(out), "-x16", "-s16", "-k1M", "-j6", "-c",
       "--file-allocation=none", "--max-tries=0", "--retry-wait=5", "--summary-interval=10",
       "--console-log-level=warn", f"--header=Authorization: Bearer {os.environ['HF_TOKEN']}"]
for attempt in range(10):
    if subprocess.call(cmd) == 0:
        Path("datasets/prefetch.done").write_text(str(out.resolve())); print("DONE", len(files), "files"); break
    print("aria2 retry", attempt, flush=True)
else:
    sys.exit(1)
