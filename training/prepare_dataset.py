"""Turn raw recordings into a Piper training set.

Input layouts (any mix, per directory):
  metadata.csv  with  file|text  or  file|speaker|text         (LJSpeech / Piper style)
  clip.wav + clip.txt                                          (sidecar transcripts)
  <speaker>/...                                                (one sub-directory per speaker)
  long recordings without transcripts                          (--asr: split on silence + Whisper)

Per clip: load mono @22.05 kHz -> optional denoise -> trim silence -> reject clipped/too short/too long
-> loudness normalize -> transcript normalized with the server's own normalizer (numbers, Hinglish, ...)
-> sanity checks (speech rate, optional ASR CER) -> wavs/<id>.wav.
Output: metadata.csv (train; Piper holds out its own validation split), test.csv (never trained on),
report.json (hours per speaker, every rejection with its reason).

Manifest mode: --manifest manifest.(csv|jsonl) replaces directory discovery (fields: audio,text,speaker_id,language,
optional emotion/style/role + label_source; only label_source=human_verified labels are kept, see training/manifest.py).
Rights: --rights rights.jsonl is REQUIRED (training/data_rights.py); clips without a rights entry, without speaker
authorization/tts_training permission, or vendor-generated without permission make the run abort.
Splits: training/split.py (seeded, no transcript/audio-hash leakage, optional --speaker-disjoint-test).

Usage: python -m training.prepare_dataset (--input data/raw | --manifest m.jsonl) --rights rights.jsonl --output data/myvoice
       [--speaker NAME] [--denoise] [--asr] [--asr-validate] [--seed N] [--speaker-disjoint-test]
Only use recordings from speakers who consented to voice cloning / TTS training.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.text_normalizer import normalize  # noqa: E402
from training import data_rights  # noqa: E402
from training.manifest import HUMAN, load_manifest, sha256_file, write_jsonl  # noqa: E402
from training.split import make_splits, write_splits  # noqa: E402

SR = 22050
AUDIO = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
MIN_S, MAX_S = 1.0, 15.0
CHARS_PER_S = (4.0, 30.0)  # outside this, transcript and audio probably don't match


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
    """Split a long recording on silences into MIN_S..MAX_S pieces."""
    segs, cur = [], None
    for s, e in librosa.effects.split(wav, top_db=35, frame_length=1024, hop_length=256):
        if cur is not None and (e - cur[0]) / SR <= MAX_S:
            cur[1] = e
        else:
            if cur is not None:
                segs.append(cur)
            cur = [s, e]
    if cur is not None:
        segs.append(cur)
    return [wav[max(0, s - SR // 10): e + SR // 10] for s, e in segs if (e - s) / SR >= MIN_S]


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
    p.add_argument("--asr-validate", action="store_true", help="reject clips whose Whisper transcript differs (CER > --max-cer)")
    p.add_argument("--max-cer", type=float, default=0.35)
    p.add_argument("--test-fraction", type=float, default=0.05)
    a = p.parse_args()
    if bool(a.input) == bool(a.manifest):
        p.error("give exactly one of --input / --manifest")
    if a.asr or a.asr_validate:
        from training.asr import cer, transcribe

    wav_dir = a.output / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    rejected: list[dict] = []
    seconds: dict[str, float] = defaultdict(float)

    def reject(src, reason):
        rejected.append({"file": str(src), "reason": reason})

    if a.manifest:
        items = [(Path(r["audio"]), r["text"], r["speaker_id"], r) for r in load_manifest(a.manifest) if not r["rejected"]]
    else:
        items = [(pa, t, sp, {}) for pa, t, sp in discover(a.input, a.speaker)]
    if a.allow_unverified_rights:
        print("WARNING: rights check skipped (--allow-unverified-rights)", file=sys.stderr)
    else:
        try:
            data_rights.enforce([{"id": str(pa), "speaker_id": sp, "rights_id": m.get("rights_id")} for pa, _, sp, m in items],
                                a.rights)
        except data_rights.RightsError as e:
            sys.exit(f"refusing to prepare: {e}")

    for path, text, speaker, meta in items:
        try:
            wav, _ = librosa.load(path, sr=SR, mono=True)
        except Exception as e:  # noqa: BLE001 - bad file: report, keep going
            reject(path, f"unreadable: {e}")
            continue
        if np.mean(np.abs(wav) >= 0.999) > 0.001:
            reject(path, "clipped")
            continue
        if a.denoise:
            wav = denoise(wav)
        pieces = [(wav, text)]
        if text is None:
            if not a.asr:
                reject(path, "no transcript (add .txt / metadata.csv, or pass --asr)")
                continue
            pieces = [(seg, None) for seg in asr_segments(wav)]
        for k, (seg, txt) in enumerate(pieces):
            seg, _ = librosa.effects.trim(seg, top_db=40)
            seg = np.pad(seg, SR // 10)  # 100 ms of silence each side
            dur = len(seg) / SR
            if not MIN_S <= dur <= MAX_S:
                hint = "too short" if dur < MIN_S else "split long clips, or use --asr"
                reject(path, f"duration {dur:.1f}s outside {MIN_S}-{MAX_S}s ({hint})")
                continue
            if txt is None:
                txt = transcribe(seg, SR)
            norm = normalize(txt)
            rate = len(norm.replace(" ", "")) / dur
            if not norm or not CHARS_PER_S[0] <= rate <= CHARS_PER_S[1]:
                reject(path, f"speech rate {rate:.1f} chars/s: transcript probably does not match audio")
                continue
            if a.asr_validate and text is not None:
                err = cer(norm, normalize(transcribe(seg, SR)))
                if err > a.max_cer:
                    reject(path, f"ASR CER {err:.2f} > {a.max_cer}")
                    continue
            uid = hashlib.sha1(f"{speaker}/{path}/{k}".encode()).hexdigest()[:12]
            sf.write(wav_dir / f"{uid}.wav", loudness_normalize(seg), SR, subtype="PCM_16")
            rec = {"id": uid, "audio": f"wavs/{uid}.wav", "text": norm, "speaker_id": speaker,
                   "language": meta.get("language") or "hi", "rights_id": meta.get("rights_id"),
                   "audio_sha256": sha256_file(wav_dir / f"{uid}.wav"), "rejected": False}
            if meta.get("label_source") == HUMAN:  # already filtered by load_manifest; never invented
                rec.update({k: meta.get(k) for k in ("emotion", "style", "role")}, label_source=HUMAN)
            rows.append(rec)
            seconds[speaker] += dur

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
    train, test = splits["train"], splits["test"]
    report = {
        "rights_check": "skipped" if a.allow_unverified_rights else "passed", "seed": a.seed,
        "dropped_for_leakage": len(dropped), "clips": len(rows), "train": len(train), "test": len(test), "multi_speaker": multi,
        "hours": {s: round(v / 3600, 3) for s, v in seconds.items()},
        "rejected": len(rejected), "rejected_by_reason": Counter(r["reason"].split(":")[0].split(" ")[0] for r in rejected),
        "rejections": rejected,
    }
    (a.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rejections"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
