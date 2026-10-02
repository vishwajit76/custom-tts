"""Turn raw recordings into a Piper training set.

Input layouts (any mix, per directory):
  metadata.csv  with  file|text  or  file|speaker|text         (LJSpeech / Piper style)
  clip.wav + clip.txt                                          (sidecar transcripts)
  <speaker>/...                                                (one sub-directory per speaker)
  long recordings without transcripts                          (--asr: split on silence + Whisper)

Per clip: load mono @22.05 kHz -> measure the raw audio (training/quality_gates.py: loudness, clipping, SNR, speech/silence,
speech rate, optional ASR CER) -> optional denoise -> trim silence -> loudness normalize -> wavs/<id>.wav. The transcript is
normalized with the server's own normalizer (numbers, Hinglish, ...). Then dataset-level gates: exact/near duplicates, speaker
leakage, per-speaker recording-condition outliers, and the optional human review sidecar (--review).
Output: metadata.csv (train; Piper holds out its own validation split), test.csv (never trained on), rejected.jsonl (every
rejected clip with all its reasons), dataset_report.json/.md (hours, rejection reasons, per speaker/category, SNR, loudness),
report.json (short summary). Every threshold is a flag (see --help; defaults in quality_gates.Thresholds).

Manifest mode: --manifest manifest.(csv|jsonl) replaces directory discovery (fields: audio,text,speaker_id,language,
optional emotion/style/role + label_source; only label_source=human_verified labels are kept, see training/manifest.py).
Rights: --rights rights.jsonl is REQUIRED (default: rights.jsonl next to --manifest, as written by training/ingest_hf.py) (training/data_rights.py); clips without a rights entry, without speaker
authorization/tts_training permission, or vendor-generated without permission make the run abort.
Splits: training/split.py (seeded, no transcript/audio-hash leakage, optional --speaker-disjoint-test).

Usage: python -m training.prepare_dataset (--input data/raw | --manifest m.jsonl) --rights rights.jsonl --output data/myvoice
       [--speaker NAME] [--denoise] [--asr] [--asr-validate] [--review review.jsonl] [--workers N] [--seed N]
       [--speaker-disjoint-test] [--min-snr-db 20 ...]
Only use recordings from speakers who consented to voice cloning / TTS training.
"""
import argparse
import csv
import functools
import hashlib
import json
import multiprocessing
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.text_normalizer import normalize  # noqa: E402
from training import audio_report, data_rights  # noqa: E402
from training import quality_gates as qg  # noqa: E402
from training.manifest import HUMAN, normalize_row, read_rows, sha256_file, write_jsonl  # noqa: E402
from training.split import make_splits, write_splits  # noqa: E402

SR = 22050
AUDIO = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
MAX_SEGMENT_S = 15.0  # --asr: longest piece a long recording is cut into
EXTRA_COLUMNS = ("gender", "age_group", "source_row_id", "source_repo", "source_revision", "license", "category")


def denoise(wav: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """Spectral gating: noise floor per frequency from the quietest 10% of frames, soft mask above it."""
    spec = librosa.stft(wav, n_fft=1024, hop_length=256)
    mag = np.abs(spec)
    frame_energy = mag.sum(axis=0)
    quiet = mag[:, frame_energy <= np.percentile(frame_energy, 10)]
    noise = quiet.mean(axis=1, keepdims=True) if quiet.size else np.zeros((mag.shape[0], 1))
    mask = np.clip((mag - strength * noise) / np.maximum(mag, 1e-8), 0.0, 1.0)
    mask = np.apply_along_axis(lambda r: np.convolve(r, np.ones(3) / 3, "same"), 1, mask)  # smooth over time
    return librosa.istft(spec * mask, hop_length=256, length=len(wav)).astype(np.float32)


def loudness_normalize(wav: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    rms = np.sqrt(np.mean(wav**2)) or 1e-8
    wav = wav * (10 ** (target_dbfs / 20) / rms)
    peak = np.abs(wav).max()
    return wav * (0.95 / peak) if peak > 0.95 else wav


def discover(root: Path, default_speaker: str) -> list[tuple[Path, str | None, str]]:
    """-> [(audio_path, transcript or None, speaker)]"""
    items = []
    for d in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        speaker = default_speaker if d == root else d.relative_to(root).parts[0]
        meta = d / "metadata.csv"
        listed = set()
        if meta.exists():
            with meta.open(encoding="utf-8") as f:
                rows = list(csv.reader(f, delimiter="|"))
            for row in rows:
                if len(row) < 2:
                    continue
                path = next((d / f"{row[0]}{ext}" for ext in ["", *AUDIO] if (d / f"{row[0]}{ext}").is_file()), None)
                if path:
                    listed.add(path)
                    items.append((path, row[-1], row[1] if len(row) >= 3 else speaker))
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in AUDIO and p not in listed:
                txt = p.with_suffix(".txt")
                items.append((p, txt.read_text("utf-8").strip() if txt.exists() else None, speaker))
    return items


def asr_segments(wav: np.ndarray) -> list[np.ndarray]:
    """Split a long recording on silences into 1..15 s pieces."""
    segs, cur = [], None
    for s, e in librosa.effects.split(wav, top_db=35, frame_length=1024, hop_length=256):
        if cur is not None and (e - cur[0]) / SR <= MAX_SEGMENT_S:
            cur[1] = e
        else:
            if cur is not None:
                segs.append(cur)
            cur = [s, e]
    if cur is not None:
        segs.append(cur)
    return [wav[max(0, s - SR // 10): e + SR // 10] for s, e in segs if (e - s) / SR >= 1.0]


def _uid(speaker: str, path: Path, k: int, meta: dict, n_pieces: int) -> str:
    if meta.get("id"):
        base = re.sub(r"[^\w.-]", "_", str(meta["id"]))
        return base if n_pieces == 1 else f"{base}_{k}"
    return hashlib.sha1(f"{speaker}/{path}/{k}".encode()).hexdigest()[:12]


def process(job: tuple, cfg: dict) -> list[dict]:
    """One input file -> one record per clip (several with --asr segmentation). Never raises on bad audio.
    A record has `reasons` ([] = passed the per-clip gates, wav written). Runs in a worker process."""
    path, text, speaker, meta = job
    th: qg.Thresholds = cfg["th"]
    base = {"file": str(path), "speaker": speaker, "meta": meta, "category": meta.get("style") or meta.get("emotion") or meta.get("category"),
            "m": None, "duration_s": 0.0}
    try:
        wav, _ = librosa.load(path, sr=SR, mono=True)
    except Exception as e:  # noqa: BLE001 - bad file: report, keep going
        return [{**base, "id": _uid(speaker, path, 0, meta, 1), "reasons": ["unreadable"], "detail": str(e)}]
    if text is None and not cfg["asr"]:
        return [{**base, "id": _uid(speaker, path, 0, meta, 1), "duration_s": len(wav) / SR, "reasons": ["no_transcript"],
                 "detail": "add .txt / metadata.csv, or pass --asr"}]
    pieces = [(wav, text)] if text is not None else [(seg, None) for seg in asr_segments(wav)]
    out = []
    for k, (seg, txt) in enumerate(pieces):
        uid = _uid(speaker, path, k, meta, len(pieces))
        m = qg.measure(seg, SR, th)  # on the raw audio: denoising or trimming would hide what the gates look for
        if cfg["denoise"]:
            seg = denoise(seg)
        seg, _ = librosa.effects.trim(seg, top_db=40)
        seg = np.pad(seg, SR // 10)  # 100 ms of silence each side
        m["duration_s"] = len(seg) / SR  # what training sees
        if txt is None:
            txt = cfg["transcribe"](seg, SR)
        norm = normalize(txt)
        rec = {**base, "id": uid, "text": norm, "m": m, "duration_s": m["duration_s"], "reasons": qg.clip_reasons(m, norm, th)}
        if not rec["reasons"] and th.asr_validate and text is not None:
            err = cfg["cer"](norm, normalize(cfg["transcribe"](seg, SR)))
            if err > th.max_cer:
                rec["reasons"].append("asr_mismatch")
                rec["detail"] = f"ASR CER {err:.2f} > {th.max_cer}"
        if not rec["reasons"]:
            pcm = (np.clip(loudness_normalize(seg), -1, 1) * 32767).astype(np.int16)
            rec["wav"] = str(cfg["wav_dir"] / f"{uid}.wav")
            sf.write(rec["wav"], pcm, SR, subtype="PCM_16")
            rec["audio_sha256"] = sha256_file(Path(rec["wav"]))
            rec["fp"] = qg.fingerprint(pcm.astype(np.float32) / 32768, SR)
        out.append(rec)
    return out


def _discover_items(a) -> list[tuple]:
    if a.manifest:
        base = a.manifest.parent
        items = []
        for i, raw in enumerate(read_rows(a.manifest), 1):
            row = normalize_row(raw, base, i)
            if not row["rejected"]:
                items.append((Path(row["audio"]), row["text"], row["speaker_id"],
                              {**{k: raw.get(k) or None for k in EXTRA_COLUMNS}, **row}))
        return items
    return [(pa, t, sp, {}) for pa, t, sp in discover(a.input, a.speaker)]


def _progress(it, total: int):
    for i, x in enumerate(it, 1):
        if total >= 200 and i % 500 == 0:
            print(f"  {i}/{total} files", file=sys.stderr, flush=True)
        yield x


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, help="directory of raw recordings (or use --manifest)")
    p.add_argument("--manifest", type=Path, help="CSV/JSONL manifest instead of --input")
    p.add_argument("--rights", type=Path, help="data-rights JSONL (required)")
    p.add_argument("--allow-unverified-rights", action="store_true",
                   help="DANGEROUS: skip the rights check (tests / data whose rights are documented elsewhere); recorded in report.json")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--val-fraction", type=float, default=0.0, help="Piper holds out its own validation split; keep 0 unless you want val.csv")
    p.add_argument("--speaker-disjoint-test", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--speaker", default="speaker0", help="speaker name for files not in a speaker sub-directory")
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--asr", action="store_true", help="transcribe (and segment) recordings that have no transcript")
    p.add_argument("--test-fraction", type=float, default=0.05)
    p.add_argument("--review", type=Path, help="human review sidecar (JSON/JSONL): rejected and below-minimum clips are excluded")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="processes for the per-clip stage (1 with ASR)")
    qg.add_args(p)
    a = p.parse_args()
    if bool(a.input) == bool(a.manifest):
        p.error("give exactly one of --input / --manifest")
    th = qg.from_args(a)
    cfg = {"th": th, "asr": a.asr, "denoise": a.denoise, "wav_dir": a.output / "wavs", "transcribe": None, "cer": None}
    if a.asr or th.asr_validate:
        from training.asr import cer, transcribe
        cfg.update(transcribe=transcribe, cer=cer)
        a.workers = 1  # the Whisper model is loaded once, in this process
    try:
        review = qg.load_review(a.review) if a.review else {}
    except (qg.ReviewError, OSError, ValueError) as e:
        sys.exit(f"invalid --review file: {e}")

    cfg["wav_dir"].mkdir(parents=True, exist_ok=True)
    items = _discover_items(a)
    rights = a.rights or (a.manifest.parent / "rights.jsonl" if a.manifest and (a.manifest.parent / "rights.jsonl").exists() else None)
    if a.allow_unverified_rights:
        print("WARNING: rights check skipped (--allow-unverified-rights)", file=sys.stderr)
    else:
        try:
            data_rights.enforce([{"id": str(pa), "speaker_id": sp, "rights_id": m.get("rights_id")} for pa, _, sp, m in items], rights)
        except data_rights.RightsError as e:
            sys.exit(f"refusing to prepare: {e}")

    work = functools.partial(process, cfg=cfg)
    if a.workers > 1 and len(items) >= 16:
        with multiprocessing.get_context("spawn").Pool(a.workers) as pool:
            recs = [r for rs in _progress(pool.imap(work, items, chunksize=16), len(items)) for r in rs]
    else:
        recs = [r for rs in _progress(map(work, items), len(items)) for r in rs]

    keys = lambda r: [r["id"], r["meta"].get("id") or "", r["file"], Path(r["file"]).name, Path(r["file"]).stem]  # noqa: E731
    for r in recs:  # human review: rejected / below-minimum clips never reach training
        r["reasons"] += qg.review_reasons(keys(r), review, th)
    if unknown := sorted(set(review) - {k for r in recs for k in keys(r)}):
        print(f"WARNING: {len(unknown)} review id(s) match no clip, e.g. {unknown[:3]}", file=sys.stderr)
    by_id = {r["id"]: r for r in recs}
    passed = [{**r, "audio_sha": r["audio_sha256"], "style": r["meta"].get("style")} for r in recs if not r["reasons"]]
    dataset_why, info = qg.dataset_reasons(passed, th)
    for rid, why in dataset_why.items():
        by_id[rid]["reasons"] += why
    for r in recs:
        if r["reasons"] and r.get("wav"):
            Path(r.pop("wav")).unlink(missing_ok=True)

    rows = []
    for r in recs:
        if r["reasons"]:
            continue
        meta = r["meta"]
        rec = {"id": r["id"], "audio": f"wavs/{r['id']}.wav", "text": r["text"], "speaker_id": r["speaker"],
               "language": meta.get("language") or "hi", "rights_id": meta.get("rights_id"),
               "audio_sha256": r["audio_sha256"], "duration_s": round(r["duration_s"], 3), "rejected": False}
        rec.update({k: meta[k] for k in EXTRA_COLUMNS if meta.get(k)})
        if meta.get("label_source") == HUMAN:  # already filtered by normalize_row; never invented
            rec.update({k: meta.get(k) for k in ("emotion", "style", "role")}, label_source=HUMAN)
        rows.append(rec)

    multi = len({r["speaker_id"] for r in rows}) > 1
    splits, dropped = make_splits(rows, a.seed, a.val_fraction, a.test_fraction, a.speaker_disjoint_test)

    def line(r):
        return f"{r['id']}.wav|{r['speaker_id']}|{r['text']}" if multi else f"{r['id']}.wav|{r['text']}"

    for name, fname in (("train", "metadata.csv"), ("val", "val.csv"), ("test", "test.csv")):
        if name == "val" and not splits["val"]:
            continue
        (a.output / fname).write_text("".join(line(r) + "\n" for r in splits[name]), encoding="utf-8")
    (a.output / "test.csv.heldout").write_text("Held-out test set: never train on test.csv.\n", encoding="utf-8")
    write_splits(splits, a.output / "splits", dropped, {"seed": a.seed, "speaker_disjoint_test": a.speaker_disjoint_test})
    write_jsonl(a.output / "manifest.jsonl", rows)
    (a.output / "speaker_map.json").write_text(
        json.dumps({s: i for i, s in enumerate(sorted({r["speaker_id"] for r in splits["train"]}))}, ensure_ascii=False), encoding="utf-8")

    rejected = [{"id": r["id"], "file": r["file"], "speaker": r["speaker"], "reason": ",".join(r["reasons"]), "reasons": r["reasons"],
                 "detail": r.get("detail"), "duration_s": round(r["duration_s"], 3),
                 "values": {k: round(v, 3) for k, v in (r["m"] or {}).items()}} for r in recs if r["reasons"]]
    write_jsonl(a.output / "rejected.jsonl", rejected)
    leakage = {"split_overlap": qg.split_overlap(splits), "dropped_for_leakage": len(dropped), **info,
               "speaker_leak_clips": sum(any(w.startswith("speaker_leak") for w in r["reasons"]) for r in recs)}
    drep = audio_report.dataset_report(recs, th, {"leakage": leakage, "review_file": str(a.review) if a.review else None})
    (a.output / "dataset_report.json").write_text(json.dumps(drep, ensure_ascii=False, indent=2), encoding="utf-8")
    (a.output / "dataset_report.md").write_text(audio_report.dataset_markdown(drep), encoding="utf-8")
    train, test = splits["train"], splits["test"]
    seconds: dict[str, float] = defaultdict(float)
    for r in rows:
        seconds[r["speaker_id"]] += r["duration_s"]
    report = {
        "rights_check": "skipped" if a.allow_unverified_rights else "passed", "seed": a.seed,
        "dropped_for_leakage": len(dropped), "clips": len(rows), "train": len(train), "test": len(test), "multi_speaker": multi,
        "hours": {s: round(v / 3600, 3) for s, v in seconds.items()},
        "rejected": len(rejected), "rejected_by_reason": Counter(w for r in rejected for w in r["reasons"]),
        "rejections": rejected,
    }
    (a.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rejections"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
