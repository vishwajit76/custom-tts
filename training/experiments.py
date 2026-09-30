"""Immutable experiment records: training/experiments/<experiment-id>.json (one file per training run, created once, never edited).

  python -m training.experiments validate                 # schema check of every record (CI runs this through tests/test_experiments.py)
  python -m training.experiments new hi_f-v6-0930T0730Z --from-hf   # copy the launch manifest the kernel wrote to HF experiments/<id>/manifest.json

Launch facts go in the record and stay as written ("unknown" where they could not be recovered). Results that only exist after the run ends go
in a SEPARATE append-only file <id>.result.json (never edit the launch record), e.g. final step, final checkpoint sha256, evaluation rows.
The kernel also writes the authoritative manifest.json/environment.json next to the checkpoints on HF (experiments/<id>/), see docs/training.md.
"""
import json
import re
import sys
from pathlib import Path

DIR = Path(__file__).parent / "experiments"
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$")
REQUIRED = ("experiment_id", "status", "kernel", "git_sha", "dataset", "base_checkpoint", "config", "seed", "batch_size", "hardware", "dates", "unknown")
STATUS = ("planned", "launched", "completed", "crashed")


def validate(rec: dict, name: str | None = None) -> list[str]:
    errs = [f"missing key: {k}" for k in REQUIRED if k not in rec]
    if errs:
        return errs
    if not ID_RE.match(rec["experiment_id"]):
        errs.append("bad experiment_id")
    if name and rec["experiment_id"] != name:
        errs.append(f"experiment_id {rec['experiment_id']!r} != file name {name!r}")
    if rec["status"] not in STATUS:
        errs.append(f"status must be one of {STATUS}")
    if rec["seed"] != 1234 and rec["status"] != "planned":
        errs.append("seed differs from 1234: piper's train/val split is drawn from it, changing it leaks validation clips into training")
    if not isinstance(rec["unknown"], list):
        errs.append("unknown must be a list of field names that could not be recovered")
    ds = rec["dataset"]
    if not (isinstance(ds, dict) and ds.get("metadata_sha256") and ds.get("n_wavs")):
        errs.append("dataset needs metadata_sha256 and n_wavs")
    return errs


def write_record(rec: dict, directory: Path = DIR) -> Path:
    """Create-once: refuses to overwrite an existing record (mode 'x')."""
    errs = validate(rec)
    if errs:
        raise ValueError("; ".join(errs))
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"{rec['experiment_id']}.json"
    with open(p, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return p


def validate_all(directory: Path = DIR) -> dict[str, list[str]]:
    out = {}
    for p in sorted(directory.glob("*.json")):
        if p.name.endswith(".result.json"):
            continue
        out[p.name] = validate(json.loads(p.read_text("utf-8")), p.stem)
    return out


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if a[:1] == ["validate"]:
        bad = {k: v for k, v in validate_all().items() if v}
        for k, v in bad.items():
            print(k, v)
        print(f"{len(validate_all())} records, {len(bad)} invalid")
        return 1 if bad else 0
    if len(a) >= 3 and a[0] == "new" and a[2] == "--from-hf":
        from huggingface_hub import hf_hub_download

        m = json.loads(Path(hf_hub_download("vishwajit76/custom-tts-hindi-train", f"experiments/{a[1]}/manifest.json", force_download=True)).read_text("utf-8"))
        rec = {"experiment_id": m["experiment_id"], "status": "launched", "kernel": m["kernel"], "git_sha": m["git_sha"], "dataset": m["dataset"],
               "base_checkpoint": m["resume"], "config": {"lr": m["lr"], "precision": m["precision"], "cli": m["cli"], "max_hours": m["max_hours"]},
               "seed": m["seed"], "batch_size": m["batch_size"], "hardware": "see HF experiments/<id>/environment.json",
               "dates": {"started_utc": m["started_utc"], "started_ist": m["started_ist"]}, "unknown": []}
        print(write_record(rec))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
