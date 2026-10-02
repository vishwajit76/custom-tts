"""Dataset quality gates: per-clip measurements, pass/reject decisions with named reasons, dataset-level checks.

Stages (all numpy/scipy, vectorised; the per-clip stage is meant to run in a process pool):
  measure(wav, sr)            loudness, clipping, SNR, speech/silence, spectral features of ONE raw clip
  clip_reasons(m, text, th)   named reasons from one clip's measurements ([] = passes)
  fingerprint(wav, sr)        16 mel bands x 32 time bins log-mel signature (for near-duplicate audio)
  dataset_reasons(recs, th)   duplicates, near-duplicates, speaker leakage, per-speaker recording-condition outliers
  load_review / review_reasons  optional human review sidecar
  split_overlap(splits)       transcript / audio overlap between train, val, test

Methods (all estimates, none is a calibrated measurement; thresholds in `Thresholds` are untuned defaults):
  frames      25 ms (512 samples @22.05 kHz) Hann/rect frames, 10 ms hop. Frame power in dB.
  VAD         speech frame = frame dB > max(noise_db + vad_floor_db, p95_db - vad_rel_db); noise_db = 10th percentile.
  SNR         10*log10((mean power of speech frames - noise power) / noise power); noise power = mean power of the
              quietest 10% of frames. Needs some non-speech frames: a clip trimmed to its last phoneme reads low.
  LUFS-like   ITU-R BS.1770 K-weighting (RBJ shelf + high-pass at the reference corner frequencies), 400 ms blocks,
              75% overlap, absolute gate -70 LUFS, relative gate -10 LU. Mono, not certified.
  clipping    fraction of samples with |x| >= clip_level, and the longest run of such samples.
  rate        letters/digits/matras per second of SPEECH (VAD) time.
  condition   per speaker: robust z = (x - median) / (1.4826 * MAD) on noise floor, spectral centroid, bandwidth, LUFS.
"""
import dataclasses
import difflib
import json
import re
import unicodedata
import zlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter

FRAME, HOP = 512, 220  # samples at 22.05 kHz: 23 ms / 10 ms
FP_MELS, FP_BINS = 16, 32
CONDITION_FEATURES = {"noise_db": 1.0, "centroid_hz": 50.0, "bandwidth_hz": 50.0, "lufs": 1.0}  # feature -> min robust scale
REVIEW_QUALITY = ("approved", "rejected", "review")
REVIEW_SCORES = ("naturalness", "pronunciation", "noise")  # 1-5, 5 = best (noise 5 = cleanest)


@dataclass
class Thresholds:
    min_s: float = 1.0
    max_s: float = 15.0
    min_rms_dbfs: float = -45.0
    max_rms_dbfs: float = -6.0
    min_lufs: float = -48.0
    max_lufs: float = -8.0
    clip_level: float = 0.99
    max_clip_frac: float = 0.001
    max_clip_run: int = 6
    min_snr_db: float = 15.0
    max_silence_ratio: float = 0.6
    min_speech_s: float = 0.5
    vad_floor_db: float = 10.0
    vad_rel_db: float = 30.0
    min_cps: float = 4.0
    max_cps: float = 30.0
    asr_validate: bool = False
    max_cer: float = 0.35
    text_near_sim: float = 0.92
    audio_near_cos: float = 0.97
    near_dur_tol: float = 0.1
    no_near_duplicates: bool = False
    reject_cross_speaker_text: bool = False
    condition_z: float = 4.0
    condition_min_group: int = 20
    condition_by_style: bool = False
    allow_review: bool = False
    min_naturalness: int = 0
    min_pronunciation: int = 0
    min_noise: int = 0


def add_args(parser, skip: tuple[str, ...] = ()) -> None:
    """One --kebab-case flag per Thresholds field (bool -> store_true)."""
    g = parser.add_argument_group("quality gates (see training/quality_gates.py)")
    for f in dataclasses.fields(Thresholds):
        if f.name in skip:
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool:
            g.add_argument(flag, action="store_true", default=f.default)
        else:
            g.add_argument(flag, type=f.type, default=f.default, help=f"default {f.default}")


def from_args(ns) -> Thresholds:
    return Thresholds(**{f.name: getattr(ns, f.name) for f in dataclasses.fields(Thresholds) if hasattr(ns, f.name)})


def text_key(text: str) -> str:
    """Same normalisation as split._key (keeps Devanagari matras)."""
    return re.sub(r"\s+", " ", "".join(c if unicodedata.category(c)[0] in "LNM" else " " for c in text.lower())).strip()


def n_chars(text: str) -> int:
    return sum(unicodedata.category(c)[0] in "LNM" for c in text)


def _frames(wav: np.ndarray) -> np.ndarray:
    if len(wav) < FRAME:
        wav = np.pad(wav, (0, FRAME - len(wav)))
    return sliding_window_view(wav, FRAME)[::HOP]


def _db(p: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(p, 1e-12))


@cache
def _kweight(sr: int) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    A, f0, q = 10 ** (3.999843853973347 / 40), 1681.974450955533, 0.7071752369554196
    w, al = 2 * np.pi * f0 / sr, np.sin(2 * np.pi * f0 / sr) / (2 * q)
    c, s = np.cos(w), 2 * np.sqrt(A) * al
    shelf = (np.array([A * ((A + 1) + (A - 1) * c + s), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - s)]),
             np.array([(A + 1) - (A - 1) * c + s, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - s]))
    w, al = 2 * np.pi * 38.13547087602444 / sr, np.sin(2 * np.pi * 38.13547087602444 / sr) / (2 * 0.5003270373238773)
    c = np.cos(w)
    hp = (np.array([(1 + c) / 2, -(1 + c), (1 + c) / 2]), np.array([1 + al, -2 * c, 1 - al]))
    return shelf, hp


def lufs(wav: np.ndarray, sr: int) -> float:
    """Gated integrated loudness, BS.1770-style (see module docstring)."""
    (b1, a1), (b2, a2) = _kweight(sr)
    z = lfilter(b2, a2, lfilter(b1, a1, wav.astype(np.float64))) ** 2
    n, hop = int(0.4 * sr), int(0.1 * sr)
    if len(z) < n:
        return float(-0.691 + _db(z.mean()))
    c = np.concatenate([[0.0], np.cumsum(z)])
    start = np.arange(0, len(z) - n + 1, hop)
    ms = (c[start + n] - c[start]) / n
    keep = ms[_db(ms) - 0.691 > -70]
    if keep.size:
        keep = keep[_db(keep) > _db(keep.mean()) - 10]
    return float(-0.691 + _db(keep.mean())) if keep.size else -70.0


def _runs(mask: np.ndarray) -> int:
    """Longest run of True."""
    if not mask.any():
        return 0
    edges = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())


def measure(wav: np.ndarray, sr: int, th: Thresholds = Thresholds()) -> dict:
    """Measurements of one raw (not yet trimmed or normalised) mono clip. All values are plain floats."""
    fr = _frames(wav)
    p = np.mean(fr.astype(np.float64) ** 2, axis=1)
    db = _db(p)
    noise_db, peak_db = float(np.percentile(db, 10)), float(np.percentile(db, 95))
    speech = db > max(noise_db + th.vad_floor_db, peak_db - th.vad_rel_db)
    noise_p = np.sort(p)[: max(1, len(p) // 10)].mean()
    snr = float(np.clip(_db(p[speech].mean() - noise_p) - _db(noise_p), 0, 70)) if speech.any() else 0.0
    spec = np.abs(np.fft.rfft(fr[speech if speech.any() else slice(None)] * np.hanning(FRAME), axis=1)) ** 2
    freqs, avg = np.fft.rfftfreq(FRAME, 1 / sr), spec.mean(axis=0)
    centroid = float((freqs * avg).sum() / max(avg.sum(), 1e-12))
    near = np.abs(wav) >= th.clip_level
    return {
        "duration_s": len(wav) / sr, "rms_dbfs": float(_db(np.mean(wav.astype(np.float64) ** 2))),
        "peak_dbfs": float(20 * np.log10(max(np.abs(wav).max(), 1e-6))), "lufs": lufs(wav, sr),
        "clip_frac": float(near.mean()), "clip_run": _runs(near),
        "noise_db": noise_db, "snr_db": snr, "speech_ratio": float(speech.mean()),
        "silence_ratio": float(1 - speech.mean()), "speech_s": float(speech.sum() * HOP / sr),
        "centroid_hz": centroid, "bandwidth_hz": float(np.sqrt(((freqs - centroid) ** 2 * avg).sum() / max(avg.sum(), 1e-12))),
    }


@cache
def _mel(sr: int) -> np.ndarray:
    import librosa

    return librosa.filters.mel(sr=sr, n_fft=FRAME, n_mels=FP_MELS)


def fingerprint(wav: np.ndarray, sr: int) -> np.ndarray:
    """16 x 32 log-mel signature, flattened float32 (time axis averaged into 32 bins so durations compare by shape)."""
    spec = np.abs(np.fft.rfft(_frames(wav) * np.hanning(FRAME), axis=1)) ** 2
    logmel = np.log(spec @ _mel(sr).T + 1e-8)
    edge = np.linspace(0, len(logmel), FP_BINS + 1).astype(int)
    return np.stack([logmel[a:max(b, a + 1)].mean(axis=0) for a, b in zip(edge[:-1], edge[1:])]).astype(np.float32).ravel()


def clip_reasons(m: dict, text: str | None, th: Thresholds) -> list[str]:
    """Named reasons from one clip's measurements. `m['duration_s']` is the duration training would see (after trim)."""
    r = []
    if m["duration_s"] < th.min_s:
        r.append("too_short")
    if m["duration_s"] > th.max_s:
        r.append("too_long")
    if m["rms_dbfs"] < th.min_rms_dbfs:
        r.append("rms_low")
    if m["rms_dbfs"] > th.max_rms_dbfs:
        r.append("rms_high")
    if m["lufs"] < th.min_lufs:
        r.append("loudness_low")
    if m["lufs"] > th.max_lufs:
        r.append("loudness_high")
    if m["clip_frac"] > th.max_clip_frac or m["clip_run"] >= th.max_clip_run:
        r.append("clipping")
    if m["snr_db"] < th.min_snr_db:
        r.append("low_snr")
    if m["silence_ratio"] > th.max_silence_ratio:
        r.append("silence_ratio")
    if m["speech_s"] < th.min_speech_s:
        r.append("too_little_speech")
    if text is not None:
        cps = n_chars(text) / max(m["speech_s"], 1e-3)
        if not n_chars(text):
            r.append("empty_transcript")
        elif cps < th.min_cps:
            r.append("speech_rate_low")
        elif cps > th.max_cps:
            r.append("speech_rate_high")
    return r


# ---- dataset level -------------------------------------------------------------------------------------------

def _pairs(mat: np.ndarray, thresh: float, block: int = 1024):
    """Yield (i, j), i < j, with cosine(mat[i], mat[j]) >= thresh. mat rows must be L2-normalised."""
    for s in range(0, len(mat), block):
        sim = mat[s:s + block] @ mat.T
        for i, j in np.argwhere(sim >= thresh):
            if j > s + i:
                yield s + int(i), int(j)


def _normalised(x: np.ndarray, center: bool) -> np.ndarray:
    x = x.astype(np.float32)
    if center:
        x = x - x.mean(axis=0)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def _trigram_vec(texts: list[str], dim: int = 1024) -> np.ndarray:
    v = np.zeros((len(texts), dim), np.float32)
    for k, t in enumerate(texts):
        t = f"  {t} "
        for i in range(len(t) - 2):
            v[k, zlib.crc32(t[i:i + 3].encode()) % dim] += 1
    return v


def dataset_reasons(recs: list[dict], th: Thresholds) -> tuple[dict[str, list[str]], dict]:
    """recs: locally accepted clips: id, speaker, text, audio_sha, fp, duration_s, m (measure dict), style.
    -> ({id: [reason]}, info). The first clip (by id) of a duplicate cluster is kept, later ones are rejected.
    info: shared_text_across_speakers (texts read by 2+ speakers: informational unless reject_cross_speaker_text)."""
    recs = sorted(recs, key=lambda r: r["id"])
    out: dict[str, list[str]] = {}

    def add(r, reason):
        out.setdefault(r["id"], []).append(reason)

    alive = lambda r: r["id"] not in out  # noqa: E731
    seen_a: dict[str, dict] = {}
    seen_t: dict[tuple, dict] = {}
    by_text: dict[str, set] = {}
    for r in recs:
        r["_key"] = text_key(r["text"])
        by_text.setdefault(r["_key"], set()).add(r["speaker"])
        if (first := seen_a.get(r["audio_sha"])) is not None:
            add(r, "duplicate_audio" if first["speaker"] == r["speaker"] else "speaker_leak_audio")
        else:
            seen_a[r["audio_sha"]] = r
    for r in filter(alive, recs):
        if (first := seen_t.get((r["speaker"], r["_key"]))) is not None:
            add(r, "duplicate_text")
        else:
            seen_t[(r["speaker"], r["_key"])] = r
    shared = sorted(k for k, s in by_text.items() if len(s) > 1)
    if th.reject_cross_speaker_text:
        first_by_key: dict[str, dict] = {}
        for r in filter(alive, recs):
            if (first := first_by_key.setdefault(r["_key"], r)) is not r and first["speaker"] != r["speaker"]:
                add(r, "speaker_leak_text")
    if not th.no_near_duplicates:
        live = [r for r in recs if alive(r)]
        close = lambda a, b: abs(a["duration_s"] - b["duration_s"]) <= th.near_dur_tol * max(a["duration_s"], b["duration_s"])  # noqa: E731
        groups: dict[str, list[dict]] = {}
        for r in live:
            groups.setdefault(r["speaker"], []).append(r)
        for g in groups.values():
            if len(g) < 2:
                continue
            for i, j in _pairs(_normalised(_trigram_vec([r["_key"] for r in g]), False), 0.8):
                a, b = g[i], g[j]
                if alive(a) and alive(b) and close(a, b) and a["_key"] != b["_key"] \
                        and difflib.SequenceMatcher(None, a["_key"], b["_key"]).ratio() >= th.text_near_sim:
                    add(b, "near_duplicate_text")
        live = [r for r in live if alive(r)]
        if len(live) > 1:
            for i, j in _pairs(_normalised(np.stack([r["fp"] for r in live]), True), th.audio_near_cos):
                a, b = live[i], live[j]
                if alive(a) and alive(b) and close(a, b):
                    add(b, "near_duplicate_audio" if a["speaker"] == b["speaker"] else "speaker_leak_audio")
    # recording-condition outliers, judged against the clips that survived the checks above
    groups = {}
    for r in filter(alive, recs):
        groups.setdefault((r["speaker"], r.get("style") or "") if th.condition_by_style else (r["speaker"],), []).append(r)
    for g in groups.values():
        if len(g) < th.condition_min_group:
            continue
        for feat, floor in CONDITION_FEATURES.items():
            x = np.array([r["m"][feat] for r in g])
            med = np.median(x)
            z = (x - med) / max(1.4826 * np.median(np.abs(x - med)), floor)
            for r, zi in zip(g, z):
                if abs(zi) > th.condition_z:
                    add(r, f"condition_outlier_{feat}")
    return out, {"shared_text_across_speakers": len(shared)}


def split_overlap(splits: dict[str, list[dict]]) -> dict[str, int]:
    """Number of transcripts / audio hashes that occur in more than one split."""
    def overlap(keyf) -> int:
        sets = [{keyf(r) for r in rows if keyf(r)} for rows in splits.values()]
        return len({k for i, a in enumerate(sets) for b in sets[i + 1:] for k in a & b})
    return {"text": overlap(lambda r: text_key(r["text"])), "audio": overlap(lambda r: r.get("audio_sha256"))}


# ---- human review sidecar ------------------------------------------------------------------------------------

class ReviewError(ValueError):
    pass


def load_review(path: Path) -> dict[str, dict]:
    """JSON ({id: entry} or [{"id": ..., ...}]) or JSONL. Entry: quality (approved|rejected|review) and optional
    naturalness / pronunciation / noise as integers 1-5. Raises ReviewError on anything else."""
    text = Path(path).read_text("utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    entries = data if isinstance(data, list) else [{"id": k, **v} if isinstance(v, dict) else {"id": k, "bad": v} for k, v in data.items()]
    out: dict[str, dict] = {}
    for e in entries:
        cid = e.get("id") if isinstance(e, dict) else None
        if not isinstance(cid, str) or not cid:
            raise ReviewError(f"review entry without a string id: {e!r}")
        if cid in out:
            raise ReviewError(f"duplicate review id {cid!r}")
        if extra := set(e) - {"id", "quality", *REVIEW_SCORES}:
            raise ReviewError(f"review {cid!r}: unknown fields {sorted(extra)}")
        if e.get("quality") not in REVIEW_QUALITY:
            raise ReviewError(f"review {cid!r}: quality must be one of {REVIEW_QUALITY}, got {e.get('quality')!r}")
        for k in REVIEW_SCORES:
            if k in e and not (isinstance(e[k], int) and not isinstance(e[k], bool) and 1 <= e[k] <= 5):
                raise ReviewError(f"review {cid!r}: {k} must be an integer 1-5, got {e[k]!r}")
        out[cid] = {k: v for k, v in e.items() if k != "id"}
    return out


def review_reasons(keys: list[str], review: dict[str, dict], th: Thresholds) -> list[str]:
    """Reasons from the first of `keys` (clip id, path, name, stem) that has a review entry. No entry = no opinion."""
    e = next((review[k] for k in keys if k in review), None)
    if e is None:
        return []
    r = []
    if e["quality"] == "rejected":
        r.append("review_rejected")
    elif e["quality"] == "review" and not th.allow_review:
        r.append("review_pending")
    r += [f"review_low_{k}" for k in REVIEW_SCORES if k in e and e[k] < getattr(th, f"min_{k}")]
    return r
