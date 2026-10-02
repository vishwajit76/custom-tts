"""Audio quality reports.

audio_report (CLI): per-file stats + per-speaker/emotion/style aggregates for a manifest (JSON + markdown).
dataset_report (used by prepare_dataset): hours, rejection reasons, per-speaker / per-category hours, SNR and loudness
distributions for a prepared dataset. Measurements and thresholds come from training/quality_gates.py (SNR and loudness are
estimates, not calibrated measurements).

  python -m training.audio_report --manifest data/manifest.jsonl --out reports/audio
"""
import argparse
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from training import quality_gates as qg
from training.manifest import load_manifest

MIN_SR = 16000


def analyze(path: Path, th: qg.Thresholds = qg.Thresholds()) -> dict:
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001 - report unreadable files instead of aborting
        return {"file": str(path), "rejects": [f"unreadable: {e}"]}
    m = qg.measure(wav.mean(axis=1), int(sr), th)
    res = {
        "file": str(path), "duration_s": round(m["duration_s"], 3), "sample_rate": int(sr), "clipping_ratio": m["clip_frac"],
        "silence_ratio": m["silence_ratio"], "rms_dbfs": round(m["rms_dbfs"], 2), "snr_db_est": round(m["snr_db"], 2),
        "lufs": round(m["lufs"], 2),
    }
    res["rejects"] = ([] if sr >= MIN_SR else ["sample_rate"]) + qg.clip_reasons(m, None, th)
    return res


def _agg(items: list[dict]) -> dict:
    ok = [i for i in items if "duration_s" in i]
    return {
        "files": len(items), "rejected": sum(bool(i["rejects"]) for i in items),
        "hours": round(sum(i["duration_s"] for i in ok) / 3600, 4),
        "mean_snr_db_est": round(float(np.mean([i["snr_db_est"] for i in ok])), 2) if ok else None,
        "mean_clipping_ratio": round(float(np.mean([i["clipping_ratio"] for i in ok])), 6) if ok else None,
        "mean_silence_ratio": round(float(np.mean([i["silence_ratio"] for i in ok])), 3) if ok else None,
    }


def report(rows: list[dict], th: qg.Thresholds = qg.Thresholds()) -> dict:
    files, groups = [], {"speaker": defaultdict(list), "emotion": defaultdict(list), "style": defaultdict(list)}
    for r in rows:
        a = analyze(Path(r["audio"]), th)
        a["id"] = r["id"]
        files.append(a)
        groups["speaker"][r["speaker_id"]].append(a)
        # emotion/style are populated only for human_verified rows (manifest drops the rest)
        groups["emotion"][r.get("emotion") or "(none)"].append(a)
        groups["style"][r.get("style") or "(none)"].append(a)
    return {"limits": dataclasses.asdict(th), "overall": _agg(files),
            **{f"by_{k}": {name: _agg(v) for name, v in sorted(g.items())} for k, g in groups.items()},
            "files": files}


def to_markdown(rep: dict) -> str:
    out = ["# Audio quality report", "", "SNR is a rough percentile-based estimate.", ""]
    for title, data in [("Overall", {"all": rep["overall"]}), ("By speaker", rep["by_speaker"]),
                        ("By emotion", rep["by_emotion"]), ("By style", rep["by_style"])]:
        out += [f"## {title}", "", "| group | files | rejected | hours | SNR est (dB) | clipping | silence |", "|---|---|---|---|---|---|---|"]
        out += [f"| {k} | {v['files']} | {v['rejected']} | {v['hours']} | {v['mean_snr_db_est']} | "
                f"{v['mean_clipping_ratio']} | {v['mean_silence_ratio']} |" for k, v in data.items()]
        out.append("")
    bad = [f for f in rep["files"] if f["rejects"]]
    out += ["## Rejects", ""] + [f"- `{f['file']}`: {', '.join(f['rejects'])}" for f in bad]
    return "\n".join(out) + "\n"


def _stats(x: list[float]) -> dict | None:
    if not x:
        return None
    a = np.asarray(x)
    return {"mean": round(float(a.mean()), 2), "median": round(float(np.median(a)), 2),
            "p10": round(float(np.percentile(a, 10)), 2), "p90": round(float(np.percentile(a, 90)), 2)}


def _hist(x: list[float], width: float = 2.0) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for v in x:
        lo = np.floor(v / width) * width
        out[f"{lo:.0f}..{lo + width:.0f}"] += 1
    return dict(sorted(out.items(), key=lambda kv: float(kv[0].split("..")[0])))


def _hours(recs: list[dict], key) -> dict[str, dict]:
    out: dict[str, dict] = defaultdict(lambda: {"clips": 0, "accepted_clips": 0, "total_hours": 0.0, "accepted_hours": 0.0})
    for r in recs:
        g = out[key(r)]
        g["clips"] += 1
        g["total_hours"] += r["duration_s"] / 3600
        if not r["reasons"]:
            g["accepted_clips"] += 1
            g["accepted_hours"] += r["duration_s"] / 3600
    return {k: {n: round(v, 4) if isinstance(v, float) else v for n, v in g.items()} for k, g in sorted(out.items())}


def dataset_report(recs: list[dict], th: qg.Thresholds | None = None, extra: dict | None = None) -> dict:
    """recs: one dict per clip with speaker, category, duration_s (what training sees), reasons ([] = accepted), m (measure dict or None)."""
    acc = [r for r in recs if not r["reasons"]]
    rej = [r for r in recs if r["reasons"]]
    hrs = lambda rs: round(sum(r["duration_s"] for r in rs) / 3600, 4)  # noqa: E731
    reasons: dict[str, dict] = defaultdict(lambda: {"clips": 0, "hours": 0.0})
    for r in rej:
        for why in r["reasons"]:  # a clip with several reasons counts under each
            reasons[why]["clips"] += 1
            reasons[why]["hours"] += r["duration_s"] / 3600
    col = lambda rs, k: [r["m"][k] for r in rs if r.get("m")]  # noqa: E731
    lufs = col(acc, "lufs")
    return {
        "clips": {"total": len(recs), "accepted": len(acc), "rejected": len(rej)},
        "hours": {"total": hrs(recs), "accepted": hrs(acc), "rejected": hrs(rej)},
        "average_duration_s": {"accepted": round(float(np.mean([r["duration_s"] for r in acc])), 3) if acc else None,
                               "all": round(float(np.mean([r["duration_s"] for r in recs])), 3) if recs else None},
        "rejection_reasons": {k: {"clips": v["clips"], "hours": round(v["hours"], 4)}
                              for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]["clips"])},
        "by_speaker": _hours(recs, lambda r: r["speaker"]),
        "by_category": _hours(recs, lambda r: r.get("category") or "(none)"),
        "snr_db": {"accepted": _stats(col(acc, "snr_db")), "all": _stats(col(recs, "snr_db"))},
        "loudness_lufs": {"accepted": _stats(lufs), "percentiles": {f"p{q}": round(float(np.percentile(lufs, q)), 2) for q in (5, 25, 50, 75, 95)} if lufs else None,
                          "histogram_2lu_bins": _hist(lufs)},
        **(extra or {}), **({"thresholds": dataclasses.asdict(th)} if th else {}),
    }


def dataset_markdown(rep: dict) -> str:
    c, h = rep["clips"], rep["hours"]
    out = ["# Dataset report", "", f"- clips: {c['accepted']} accepted / {c['rejected']} rejected / {c['total']} total",
           f"- hours: **{h['accepted']} accepted**, {h['rejected']} rejected, {h['total']} total",
           f"- average duration: {rep['average_duration_s']['accepted']} s accepted ({rep['average_duration_s']['all']} s all)",
           f"- SNR dB (estimate), accepted: {rep['snr_db']['accepted']}", f"- loudness LUFS-like, accepted: {rep['loudness_lufs']['accepted']}",
           f"- speaker leakage / overlap: {rep.get('leakage')}", "", "## Rejection reasons (a clip can have several)", "",
           "| reason | clips | hours |", "|---|---|---|"]
    out += [f"| {k} | {v['clips']} | {v['hours']} |" for k, v in rep["rejection_reasons"].items()]
    for title, key in (("Per speaker", "by_speaker"), ("Per category", "by_category")):
        out += ["", f"## {title}", "", "| name | clips | accepted clips | accepted hours | total hours |", "|---|---|---|---|---|"]
        out += [f"| {k} | {v['clips']} | {v['accepted_clips']} | {v['accepted_hours']} | {v['total_hours']} |" for k, v in rep[key].items()]
    out += ["", "## Loudness histogram (accepted, LUFS-like, 2 LU bins)", "", "| bin | clips |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rep["loudness_lufs"]["histogram_2lu_bins"].items()]
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="directory for audio_report.json / .md")
    a = p.parse_args()
    rep = report([r for r in load_manifest(a.manifest) if not r["rejected"]])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "audio_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    (a.out / "audio_report.md").write_text(to_markdown(rep), encoding="utf-8")
    print(json.dumps(rep["overall"]))


if __name__ == "__main__":
    main()
