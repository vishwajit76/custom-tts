"""V8 Phase 6: quality filter + per-speaker splits over ingest_hf-format sources (metadata.csv + wavs/).

  python -m training.v8.quality_filter --src work/v7/dksmoke/w/rasa_hi [--src DIR ...] --out datasets/manifest

Reuses training.quality_gates (measure/clip_reasons) and text_normalizer. Audio is referenced by path, never copied.
Writes train/validation/test/all.jsonl, rejected.jsonl, data-quality-report.{json,html}.
"""
import argparse
import hashlib
import html
import json
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.services import text_normalizer  # noqa: E402
from training import manifest as tm  # noqa: E402
from training import quality_gates as qg  # noqa: E402

FIELDS = ["audio_path", "text", "normalized_text", "speaker", "gender", "duration", "sr", "source"]
_DEV = re.compile(r"[ऀ-ॿ]")
_LAT = re.compile(r"[A-Za-z]")
_OK = re.compile(r"[ऀ-ॿA-Za-z\s.,?!;:'\"\-–—()।॥]")
SOURCE = "ai4bharat/Rasa:Hindi"  # ponytail: single source; add a --source flag when a second dataset is ingested


def hindi_text_reasons(norm: str) -> list[str]:
    if not norm.strip():
        return ["empty_transcript"]
    r = []
    dev, lat = len(_DEV.findall(norm)), len(_LAT.findall(norm))
    if not dev:
        r.append("no_devanagari")  # language-confidence proxy: Devanagari share of letters
    elif dev / (dev + lat) < 0.5:
        r.append("low_hindi_share")
    if re.search(r"\d", norm) or not all(_OK.fullmatch(c) for c in norm):
        r.append("bad_normalization")  # digits/symbols survived normalisation
    return r


def load(src: Path) -> list[dict]:
    rows = []
    for r in tm.read_rows(src / "metadata.csv"):
        g = (r.get("gender") or "").lower() or "unknown"
        rows.append({"path": (src / r["audio"]).resolve(), "text": r["text"], "gender": g, "speaker": f"rasa_hi_{g}", "source": SOURCE})
    return rows


def process(rows: list[dict], th: qg.Thresholds, expected_sr: int) -> tuple[list[dict], list[dict]]:
    good, bad, seen_a, seen_t = [], [], set(), set()
    for r in sorted(rows, key=lambda r: str(r["path"])):
        reasons, m, sr = [], {}, 0
        try:
            wav, sr = sf.read(r["path"], dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if not np.isfinite(wav).all() or len(wav) == 0:
                raise ValueError("non-finite or empty")
            m = qg.measure(wav, sr, th)
            reasons += qg.clip_reasons(m, r["text"], th)
            if sr != expected_sr:
                reasons.append("bad_sample_rate")
            sha = hashlib.sha256(r["path"].read_bytes()).hexdigest()
            key = (r["speaker"], qg.text_key(r["text"]))
            if sha in seen_a:
                reasons.append("duplicate_audio")
            if key in seen_t:
                reasons.append("duplicate_transcript")
            seen_a.add(sha), seen_t.add(key)
        except Exception:  # corrupt/unreadable audio
            reasons.append("corrupted_audio")
        norm = text_normalizer.normalize(r["text"])
        reasons += hindi_text_reasons(norm)
        n = qg.n_chars(r["text"])
        stats = {"duration": round(m.get("duration_s", 0), 3), "rms_dbfs": m.get("rms_dbfs"), "peak_dbfs": m.get("peak_dbfs"),
                 "clip_frac": m.get("clip_frac"), "silence_ratio": m.get("silence_ratio"), "snr_db": m.get("snr_db"),
                 "text_chars": n, "chars_per_s": round(n / m["speech_s"], 2) if m.get("speech_s") else None}
        rec = {"audio_path": str(r["path"]), "text": r["text"], "normalized_text": norm, "speaker": r["speaker"],
               "gender": r["gender"], "duration": stats["duration"], "sr": sr, "source": r["source"], "stats": stats}
        if reasons:
            bad.append({**rec, "reasons": reasons})
        else:
            good.append(rec)
    return good, bad


def split(good: list[dict], val=0.02, test=0.02) -> dict[str, list[dict]]:
    out = {"train": [], "validation": [], "test": []}
    for r in good:  # deterministic hash, utterance level, applied within each speaker
        h = zlib.crc32(f"{r['speaker']}|{Path(r['audio_path']).name}|{r['text']}".encode()) % 10000 / 10000
        out["test" if h < test else "validation" if h < test + val else "train"].append(r)
    return out


def report(good, bad, splits, total) -> dict:
    hrs = defaultdict(float)
    for r in good:
        hrs[r["speaker"]] += r["duration"] / 3600
    keys = ["duration", "rms_dbfs", "peak_dbfs", "silence_ratio", "snr_db", "text_chars", "chars_per_s"]
    vals = {k: [r["stats"][k] for r in good if r["stats"].get(k) is not None] for k in keys}
    return {"total_clips": total, "accepted": len(good), "rejected": len(bad),
            "accepted_hours_per_speaker": {k: round(v, 4) for k, v in sorted(hrs.items())},
            "reject_reasons": dict(Counter(x for b in bad for x in b["reasons"])),
            "splits": {k: {"clips": len(v), "hours": round(sum(r["duration"] for r in v) / 3600, 4)} for k, v in splits.items()},
            "accepted_stats": {k: {"mean": round(float(np.mean(v)), 3), "min": round(float(np.min(v)), 3), "max": round(float(np.max(v)), 3)}
                               for k, v in vals.items() if v}}


def to_html(rep: dict) -> str:
    def table(d, head):
        rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(json.dumps(v) if isinstance(v, dict) else str(v))}</td></tr>"
                       for k, v in d.items())
        return f"<table border=1 cellpadding=4><tr><th>{head[0]}</th><th>{head[1]}</th></tr>{rows}</table>"
    return ("<!doctype html><meta charset=utf-8><title>Data quality report</title><body style='font-family:sans-serif'>"
            f"<h1>Data quality report</h1><p>{rep['accepted']} accepted / {rep['rejected']} rejected of {rep['total_clips']}</p>"
            f"<h2>Accepted hours per speaker</h2>{table(rep['accepted_hours_per_speaker'], ['speaker', 'hours'])}"
            f"<h2>Reject reasons</h2>{table(rep['reject_reasons'], ['reason', 'clips'])}"
            f"<h2>Splits</h2>{table(rep['splits'], ['split', 'clips/hours'])}"
            f"<h2>Accepted stats</h2>{table(rep['accepted_stats'], ['metric', 'mean/min/max'])}</body>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, default=Path("datasets/manifest"))
    ap.add_argument("--expected-sr", type=int, default=22050)
    ap.add_argument("--val", type=float, default=0.02)
    ap.add_argument("--test", type=float, default=0.02)
    a = ap.parse_args()
    rows = [r for s in a.src for r in load(s)]
    good, bad = process(rows, qg.Thresholds(), a.expected_sr)
    sp = split(good, a.val, a.test)
    a.out.mkdir(parents=True, exist_ok=True)

    def w(name, rs, keep=FIELDS):
        (a.out / name).write_text("".join(json.dumps({k: r[k] for k in keep}, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")

    for k, v in sp.items():
        w(f"{k}.jsonl", v)
    w("all.jsonl", good)
    w("rejected.jsonl", bad, FIELDS + ["reasons"])
    rep = report(good, bad, sp, len(rows))
    (a.out / "data-quality-report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    (a.out / "data-quality-report.html").write_text(to_html(rep), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ["accepted", "rejected", "accepted_hours_per_speaker", "reject_reasons", "splits"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
