"""Audio quality report: per-file stats + per-speaker/emotion/style aggregates (JSON + markdown).

Per file: duration, sample rate, clipping ratio, silence ratio, RMS dBFS, SNR estimate, reject reasons.
SNR is a crude estimate (loud-frame vs quiet-frame RMS percentiles), not a calibrated measurement.

  python -m training.audio_report --manifest data/manifest.jsonl --out reports/audio
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from training.manifest import load_manifest

LIMITS = {"min_s": 1.0, "max_s": 15.0, "min_sr": 16000, "max_clip": 0.001, "max_silence": 0.6, "min_snr_db": 15.0}
FRAME = 0.025


def analyze(path: Path, limits: dict = LIMITS) -> dict:
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001 - report unreadable files instead of aborting
        return {"file": str(path), "rejects": [f"unreadable: {e}"]}
    wav = wav.mean(axis=1)
    n = max(1, int(sr * FRAME))
    frames = wav[: len(wav) // n * n].reshape(-1, n) if len(wav) >= n else wav[None, :]
    rms = np.sqrt((frames**2).mean(axis=1)) + 1e-10
    db = 20 * np.log10(rms)
    peak_db = float(np.percentile(db, 95))
    noise_db = float(np.percentile(db, 10))
    res = {
        "file": str(path), "duration_s": round(len(wav) / sr, 3), "sample_rate": int(sr),
        "clipping_ratio": float(np.mean(np.abs(wav) >= 0.999)),
        "silence_ratio": float(np.mean(db < peak_db - 40)),  # frames >40 dB below the loud frames
        "rms_dbfs": round(float(20 * np.log10(np.sqrt(np.mean(wav**2)) + 1e-10)), 2),
        "snr_db_est": round(peak_db - noise_db, 2),
    }
    r = []
    if not limits["min_s"] <= res["duration_s"] <= limits["max_s"]:
        r.append("duration")
    if sr < limits["min_sr"]:
        r.append("sample_rate")
    if res["clipping_ratio"] > limits["max_clip"]:
        r.append("clipping")
    if res["silence_ratio"] > limits["max_silence"]:
        r.append("silence")
    if res["snr_db_est"] < limits["min_snr_db"]:
        r.append("low_snr")
    res["rejects"] = r
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


def report(rows: list[dict], limits: dict = LIMITS) -> dict:
    files, groups = [], {"speaker": defaultdict(list), "emotion": defaultdict(list), "style": defaultdict(list)}
    for r in rows:
        a = analyze(Path(r["audio"]), limits)
        a["id"] = r["id"]
        files.append(a)
        groups["speaker"][r["speaker_id"]].append(a)
        # emotion/style are populated only for human_verified rows (manifest drops the rest)
        groups["emotion"][r.get("emotion") or "(none)"].append(a)
        groups["style"][r.get("style") or "(none)"].append(a)
    return {"limits": limits, "overall": _agg(files),
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
