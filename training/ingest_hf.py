"""Stream a Hugging Face parquet speech dataset into 22.05 kHz mono 16-bit wavs + a manifest CSV + a data-rights entry.

Output (--out DIR):
  wavs/<id>.wav          22.05 kHz mono PCM_16 (soxr HQ resample, no loudness change; training.prepare_dataset normalises)
  metadata.csv           manifest for `prepare_dataset --manifest` (audio,text,speaker_id,language + gender,style,age_group,duration,
                         source_row_id,source_duration,license,source_repo,source_revision,label_source,rights_id + every extra --map field)
  rights.jsonl           data-rights entry (training/data_rights.py schema + attribution/source_url/revision); extended per run
  ingest_info.json       repo, pinned revision sha, filters, counts, hours
Resumable: clips whose wav and metadata row exist are skipped, so a rerun continues (and --max-hours counts what is already there).

Column mapping: --map FIELD=SOURCE where SOURCE is a parquet column, a template like 'rasa_hi_{gender}' (uses the columns), or
'=literal'. Fields: id, text, speaker_id, language, gender, style, age_group, duration + any extra name (kept as a column).
Presets (--preset) fill in repo, config, mapping, licence; flags override. Rasa style / category names are NOT assumed anywhere.

  python -m training.ingest_hf --preset rasa --out data/rasa_hi_f --gender female
  python -m training.ingest_hf ai4bharat/Rasa --config Hindi --map text=text --map speaker_id='rasa_{gender}' --out data/x
REPO may also be a local directory of parquet files (offline use, tests). Needs HF_TOKEN for gated datasets.
"""
import argparse
import csv
import fnmatch
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import data_rights  # noqa: E402

SR = 22050
BASE_COLS = ["id", "audio", "text", "speaker_id", "language", "gender", "style", "age_group", "duration", "source_row_id",
             "license", "source_repo", "source_revision", "label_source", "rights_id", "source_duration"]
PRESETS = {
    "rasa": {
        "repo": "ai4bharat/Rasa", "config": "Hindi", "licence": "CC-BY-4.0",
        # Rasa has one female and one male speaker per language: the speaker id is derived from the gender column
        "map": {"id": "filename", "text": "text", "speaker_id": "rasa_hi_{gender}", "gender": "gender", "style": "style",
                "duration": "duration", "language": "=hi"},
    },
    "indicvoices-r": {
        "repo": "SPRINGLab/IndicVoices-R_Hindi", "config": "", "licence": "CC-BY-4.0",
        "map": {"text": "text", "speaker_id": "ivr_{speaker_id}", "gender": "gender", "age_group": "age_group", "duration": "duration",
                "source_snr": "snr", "language": "=hi"},
    },
}
PERMITTED = ["tts_training", "research", "commercial_use"]  # CC-BY-4.0 allows these with attribution


def parse_map(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        field, _, src = p.partition("=")
        if not field or not src:
            raise SystemExit(f"--map expects FIELD=SOURCE, got {p!r}")
        out[field] = src
    return out


def _columns(src: str) -> set[str]:
    return set() if src.startswith("=") else set(re.findall(r"\{(\w+)\}", src)) or {src}


def _value(src: str, row: dict):
    if src.startswith("="):
        return src[1:]
    if "{" in src:
        return src.format_map({k: ("" if v is None else str(v).lower() if k == "gender" else v) for k, v in row.items()})
    return row.get(src)


def sanitize(s: str) -> str:
    return re.sub(r"[^\w.-]", "_", re.sub(r"\.(wav|flac|mp3|ogg)$", "", Path(str(s)).name, flags=re.I))


def decode(blob: bytes) -> tuple[np.ndarray, int]:
    x, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


def resample(x: np.ndarray, sr: int) -> np.ndarray:
    return np.clip(soxr.resample(x, sr, SR, quality="HQ") if sr != SR else x, -1.0, 1.0)


def list_shards(repo: str, pattern: str, revision: str | None) -> tuple[list[tuple[str, Path | None]], str]:
    """-> ([(shard name, local path or None if it must be downloaded)], revision). Sorted, so reruns see the same order."""
    if Path(repo).is_dir():
        files = sorted(p for p in Path(repo).rglob("*.parquet") if fnmatch.fnmatch(p.relative_to(repo).as_posix(), pattern))
        return [(p.relative_to(repo).as_posix(), p) for p in files], "local"
    from huggingface_hub import HfApi

    api = HfApi()
    sha = api.dataset_info(repo, revision=revision).sha
    return [(f, None) for f in sorted(api.list_repo_files(repo, repo_type="dataset", revision=sha)) if fnmatch.fnmatch(f, pattern)], sha


def update_rights(path: Path, entry: dict) -> None:
    """Create or extend rights.jsonl: same rights_id -> speakers are unioned and the revision recorded."""
    entries = data_rights.load_rights(path) if path.exists() else []
    old = next((e for e in entries if e["rights_id"] == entry["rights_id"]), None)
    if old:
        entry["speaker_ids"] = sorted(set(old["speaker_ids"]) | set(entry["speaker_ids"]))
        entry["revisions"] = sorted({r for r in [*old.get("revisions", []), old.get("revision"), entry["revision"]] if r})
        entries = [e for e in entries if e is not old]
    else:
        entry["revisions"] = [entry["revision"]]
    entries.append(entry)
    path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
    data_rights.load_rights(path)  # re-validate what we wrote


def ingest(repo: str, out: Path, mapping: dict[str, str], *, config: str = "", split: str | None = None, gender: str | None = None,
           styles: list[str] = (), max_hours: float | None = None, revision: str | None = None, pattern: str | None = None,
           licence: str = "", attribution: str = "", source_url: str = "", rights_id: str | None = None,
           speaker_authorization: bool = False, label_source: str = "human_verified", batch_size: int = 32) -> dict:
    import pyarrow.parquet as pq

    if "text" not in mapping or "speaker_id" not in mapping:
        raise SystemExit("--map must define at least text and speaker_id")
    if not licence:
        raise SystemExit("--licence is required: every ingested dataset needs a rights entry")
    out = Path(out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    pattern = pattern or (f"{config}/" if config else "") + "*" + (f"{split}*" if split else "") + ".parquet"
    shards, sha = list_shards(repo, pattern, revision)
    if not shards:
        raise SystemExit(f"no parquet files match {pattern!r} in {repo}")
    slug = Path(repo).name if Path(repo).is_dir() else repo
    rights_id = rights_id or re.sub(r"[^\w-]", "-", f"{slug}-{config}".strip("-")).lower()
    extra = sorted(set(mapping) - set(BASE_COLS))
    meta_path = out / "metadata.csv"
    fields = BASE_COLS + extra
    done: dict[str, float] = {}
    if meta_path.exists():
        with meta_path.open(encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            fields = rd.fieldnames
            done = {r["id"]: float(r["duration"] or 0) for r in rd if (out / r["audio"]).is_file()}
    seconds = sum(done.values())
    need = set().union(*(_columns(s) for s in mapping.values()), {"audio"})
    speakers: set[str] = set()
    n_new = skipped = filtered = 0
    stop = lambda: max_hours is not None and seconds >= max_hours * 3600  # noqa: E731
    with meta_path.open("a", encoding="utf-8", newline="") as mf:
        w = csv.DictWriter(mf, fieldnames=fields, extrasaction="ignore")
        if not done and mf.tell() == 0:
            w.writeheader()
        for name, local in shards:
            if stop():
                break
            if local is None:
                from huggingface_hub import hf_hub_download

                local = Path(hf_hub_download(repo, name, repo_type="dataset", revision=sha))
            pf = pq.ParquetFile(local)
            cols = [c for c in pf.schema_arrow.names if c in need]
            row_no = -1
            for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
                if stop():
                    break
                for row in batch.to_pylist():
                    row_no += 1
                    if stop():
                        break
                    vals = {f: _value(s, row) for f, s in mapping.items()}
                    g = str(vals.get("gender") or "").lower()
                    if (gender and g != gender.lower()) or (styles and str(vals.get("style") or "").lower() not in {s.lower() for s in styles}):
                        filtered += 1
                        continue
                    sid = f"{Path(name).stem}:{row_no}"
                    cid = sanitize(vals["id"]) if vals.get("id") else re.sub(r"[^\w.-]", "_", f"{rights_id}_{Path(name).stem}_{row_no:06d}")
                    speakers.add(str(vals["speaker_id"]))
                    if cid in done:
                        skipped += 1
                        continue
                    audio = row.get("audio") or {}
                    if not audio.get("bytes") or not vals.get("text"):
                        filtered += 1
                        continue
                    try:
                        x, sr = decode(audio["bytes"])
                    except Exception as e:  # noqa: BLE001 - one undecodable clip must not stop a multi-hour ingest
                        print(f"skip {sid}: {e}", file=sys.stderr)
                        filtered += 1
                        continue
                    x = resample(x, sr)
                    sf.write(out / "wavs" / f"{cid}.wav", x, SR, subtype="PCM_16")
                    dur = len(x) / SR
                    w.writerow({**{k: ("" if v is None else v) for k, v in vals.items()}, "id": cid, "audio": f"wavs/{cid}.wav",
                                "gender": g, "duration": round(dur, 3), "source_row_id": sid, "license": licence, "source_repo": repo,
                                "source_revision": sha, "label_source": label_source, "rights_id": rights_id, "source_duration": vals.get("duration") or ""})
                    mf.flush()
                    done[cid] = dur
                    seconds += dur
                    n_new += 1
    if not speakers:
        raise SystemExit("nothing matched the filters")
    url = source_url or (f"https://huggingface.co/datasets/{repo}" if not Path(repo).is_dir() else repo)
    update_rights(out / "rights.jsonl", {
        "rights_id": rights_id, "source": f"{repo} ({config or 'default'} config)", "licence": licence,
        "consent_record_id": f"dataset-licence:{repo}@{sha}", "speaker_authorization": speaker_authorization,
        "speaker_ids": sorted(speakers), "permitted_uses": PERMITTED, "vendor_generated": False, "revision": sha,
        "source_url": url, "attribution": attribution or f"{repo}, licensed {licence}, {url}",
        "notes": "Rights follow the dataset licence; attribution is required. Gated datasets: the terms were accepted by the downloader."})
    info = {"repo": repo, "revision": sha, "pattern": pattern, "filters": {"gender": gender, "styles": list(styles), "split": split,
            "max_hours": max_hours}, "new_clips": n_new, "skipped_existing": skipped, "filtered_out": filtered,
            "clips": len(done), "hours": round(seconds / 3600, 4), "speakers": sorted(speakers)}
    (out / "ingest_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo", nargs="?", help="HF dataset id or local directory of parquet files")
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", help="config sub-folder in the repo, e.g. Hindi")
    p.add_argument("--split", help="only shards whose name contains this (train/validation/test)")
    p.add_argument("--map", action="append", default=[], metavar="FIELD=SOURCE", help="repeatable; overrides the preset's mapping")
    p.add_argument("--gender", help="keep only this (mapped) gender value, case-insensitive")
    p.add_argument("--style", action="append", default=[], help="keep only these style values; repeatable")
    p.add_argument("--max-hours", type=float)
    p.add_argument("--revision", help="dataset revision (branch/tag/sha); the resolved sha is recorded")
    p.add_argument("--pattern", help="override the parquet file glob")
    p.add_argument("--licence")
    p.add_argument("--attribution", default="")
    p.add_argument("--source-url", default="")
    p.add_argument("--rights-id")
    p.add_argument("--label-source", default="human_verified",
                   help="written to metadata.csv; style/emotion are only used by prepare_dataset when this is human_verified "
                        "(dataset-provided labels count; model guesses never do)")
    p.add_argument("--speaker-authorization", action="store_true",
                   help="assert speakers authorised TTS training (implied for presets: published under the dataset licence)")
    a = p.parse_args()
    pre = PRESETS.get(a.preset, {})
    repo = a.repo or pre.get("repo")
    if not repo:
        p.error("give REPO or --preset")
    info = ingest(repo, a.out, {**pre.get("map", {}), **parse_map(a.map)}, config=a.config if a.config is not None else pre.get("config", ""),
                  split=a.split, gender=a.gender, styles=a.style, max_hours=a.max_hours, revision=a.revision, pattern=a.pattern,
                  licence=a.licence or pre.get("licence", ""), attribution=a.attribution, source_url=a.source_url,
                  rights_id=a.rights_id, speaker_authorization=a.speaker_authorization or bool(a.preset), label_source=a.label_source)
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
