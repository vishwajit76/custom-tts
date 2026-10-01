"""Dataset manifest (CSV or JSONL): one row per clip.

Fields: audio (path, relative to the manifest), text, speaker_id, language (e.g. hi, hi-Latn, en-IN, hi-en),
optional emotion / style / role, label_source, rights_id, id, rejected, reject_reason, audio_sha256.

Label policy: emotion/style/role are used as training labels ONLY when label_source == "human_verified".
Otherwise they are dropped (set to None) with a warning. Labels are never inferred or invented here.
"""
import csv
import hashlib
import json
import warnings
from pathlib import Path

LABEL_FIELDS = ("emotion", "style", "role")
FIELDS = ("id", "audio", "text", "speaker_id", "language", "emotion", "style", "role", "label_source",
          "rights_id", "audio_sha256", "rejected", "reject_reason", "verified_by", "verified_at")
HUMAN = "human_verified"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean(v):
    if isinstance(v, str):
        v = v.strip()
    return None if v in ("", None) else v


def read_rows(path: Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Atomic write (tmp + rename) so the annotation tool can never leave a half-written manifest."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def normalize_row(raw: dict, base: Path, index: int, warn=warnings.warn) -> dict:
    row = {k: _clean(raw.get(k)) for k in FIELDS}
    if not row["audio"] or row["text"] is None or not row["speaker_id"]:
        raise ValueError(f"row {index}: audio, text and speaker_id are required")
    if not row["language"]:
        raise ValueError(f"row {index}: language tag is required")
    audio = Path(row["audio"])
    row["audio"] = str(audio if audio.is_absolute() else base / audio)
    row["id"] = row["id"] or hashlib.sha1(f'{row["speaker_id"]}/{row["audio"]}'.encode()).hexdigest()[:12]
    row["rejected"] = str(raw.get("rejected", "")).strip().lower() in ("1", "true", "yes")
    if any(row[k] for k in LABEL_FIELDS) and row["label_source"] != HUMAN:
        warn(f"row {row['id']}: dropping unverified emotion/style/role labels (label_source={row['label_source']!r}; "
             f"only {HUMAN!r} is used)", stacklevel=2)
        for k in LABEL_FIELDS:
            row[k] = None
    return row


def load_manifest(path: Path) -> list[dict]:
    """Validated rows with unverified labels dropped. Rejected rows are kept (flagged); callers skip them."""
    path = Path(path)
    return [normalize_row(r, path.parent, i) for i, r in enumerate(read_rows(path), 1)]


def main() -> None:
    """Convert a CSV/JSONL manifest into a validated JSONL with ids + audio hashes (input for annotate/split)."""
    import argparse
    import sys

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--in", dest="src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = load_manifest(a.src)
    for r in rows:
        if Path(r["audio"]).is_file():
            r["audio_sha256"] = sha256_file(Path(r["audio"]))
        else:
            print(f"warning: missing audio {r['audio']}", file=sys.stderr)
    write_jsonl(a.out, rows)
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
