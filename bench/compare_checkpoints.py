"""Compare Piper milestones on the fixed Hindi eval set with repeat variance and bootstrap confidence intervals. One JSON file out.

  python -m bench.compare_checkpoints --milestone milestones/step_340000 --milestone milestones/step_350000 --repeats 3
  python -m bench.compare_checkpoints --milestone experiments/<id>/milestones/step_5000 --milestone milestones/step_355000 --asr large-v3
  python -m bench.compare_checkpoints --list                        # every milestone on HF (legacy milestones/ and experiments/*/milestones/) with upload time
  python -m bench.compare_checkpoints --onnx a.onnx --onnx b.onnx   # local files (provenance fields become "unknown")

Why this exists (what milestone_eval.py did not give): (1) every sentence is synthesized `--repeats` times (default 3) because Piper's ONNX graph draws its
own noise, unseedable from Python, so one pass is one random sample; (2) CER and PER are separate metrics (PER = espeak-ng phoneme error rate, script-neutral);
(3) UTMOS and speaker similarity are computed on EVERY repeat, not only the first; (4) 95% bootstrap CIs resample sentences AND repeats (two-level), and
checkpoints are compared with a PAIRED bootstrap on the same sentences; (5) each row carries the sha256 of the exact ONNX file and the HF commit that wrote it,
because legacy names milestones/step_N were overwritten by two concurrent sessions (v4/v5), so "step_350000" alone does not identify a model.

Metric labels (read these before quoting a number):
  cer, per                 cer counts Devanagari vowel signs (fixed 2026-09-30; `cer_legacy_skeleton` = old consonant-only metric). faster-whisper ASR (int8, CPU, beam 5, hi) vs the normalized reference: intelligibility proxy, inflated by the ASR (loanwords, years).
  utmos                    UTMOS22-strong PREDICTED MOS (English-trained). Relative ranking of our own checkpoints only; not a listening-test MOS.
  speaker_similarity_neural  cosine of Resemblyzer GE2E d-vectors to the centroid of real held-out clips of the dataset speaker. Embedding similarity only: it is
                           NOT speaker verification and says nothing about naturalness. `real_vs_real_loo` is the ceiling (each real clip vs the others).
  speaker_similarity_mfcc  cosine of handcrafted MFCC statistics (app.services.speaker_encoder mfcc backend). Weak, content/channel sensitive, drift check only.
  telephony.8k / .16k      CER after a simulated channel: 8k = 8 kHz G.711 mu-law round trip (PSTN narrowband); 16k = 16 kHz, 8-bit mu-law companding round trip
                           (wideband PCM channel). Synthetic channel models, not a real carrier/codec. Plain `cer` is the clean 16 kHz path.
Nothing is uploaded; the HF token is read from ~/.cache/huggingface/token.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from bench import milestone_eval as me

HERE = Path(__file__).parent
ROOT = HERE.parent
REPO = me.REPO
SCHEMA = "compare_checkpoints/v1"


# ---------------------------------------------------------------- statistics (pure numpy; unit-tested)
def two_level_bootstrap(mat, b=2000, seed=0, stat=np.mean):
    """mat: (n_sentences, n_repeats). Resample sentences with replacement, then repeats with replacement inside each picked sentence; return the
    distribution of the corpus mean. Captures sentence-sampling AND synthesis-noise uncertainty. Deterministic for a given seed."""
    m = np.asarray(mat, float)
    n, r = m.shape
    rng = np.random.default_rng(seed)
    si = rng.integers(0, n, size=(b, n))
    ri = rng.integers(0, r, size=(b, n))
    return stat(m[si, ri], axis=1)


def summarize(mat, b=2000, seed=0) -> dict:
    """Point estimate = mean over sentences of the per-sentence repeat mean. repeat_std = std (ddof=1) of the per-repeat corpus means (pure
    synthesis+ASR noise for the same sentences); ci95 = percentile interval of the two-level bootstrap."""
    m = np.asarray(mat, float)
    per_rep = m.mean(axis=0)
    d = two_level_bootstrap(m, b, seed)
    return {"mean": round(float(m.mean()), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "repeat_means": [round(float(x), 4) for x in per_rep], "repeat_std": round(float(per_rep.std(ddof=1)), 4) if m.shape[1] > 1 else None,
            "n_sentences": int(m.shape[0]), "n_repeats": int(m.shape[1]), "median_sentence": round(float(np.median(m.mean(axis=1))), 4),
            "p90_sentence": round(float(np.percentile(m.mean(axis=1), 90)), 4)}


def paired_delta(a, b_, b=2000, seed=0) -> dict:
    """B minus A on the same sentences, same resampling indices for both (paired). `excludes_zero` = the 95% interval does not contain 0."""
    a, b_ = np.asarray(a, float), np.asarray(b_, float)
    assert a.shape[0] == b_.shape[0], "different sentence sets"
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    si = rng.integers(0, n, size=(b, n))
    da = a.mean(axis=1)[si].mean(axis=1)
    db = b_.mean(axis=1)[si].mean(axis=1)
    d = db - da
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"delta_mean": round(float(b_.mean() - a.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)], "excludes_zero": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------- telephony channel models
def _mulaw(x, mu=255.0):
    x = np.clip(x, -1, 1)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((y + 1) / 2 * 255)  # 8-bit code
    y = q / 255 * 2 - 1
    return np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu


def channel(wav: np.ndarray, sr: int, kind: str) -> tuple[np.ndarray, int]:
    """kind '8k': resample to 8 kHz, mu-law 8-bit round trip, return at 8 kHz. '16k': resample to 16 kHz, mu-law 8-bit round trip. Returns (wav, sr)."""
    target = {"8k": 8000, "16k": 16000}[kind]
    w = soxr.resample(wav, sr, target) if sr != target else wav
    return _mulaw(w.astype(np.float64)).astype(np.float32), target


# ---------------------------------------------------------------- model resolution and provenance
def _api():
    from huggingface_hub import HfApi

    return HfApi()


def list_milestones() -> list[dict]:
    api = _api()
    files = api.list_repo_files(REPO)
    dirs = sorted({f.rsplit("/", 1)[0] for f in files if f.endswith("/hi_IN-custom-medium.onnx") and (f.startswith("milestones/") or "/milestones/" in f)})
    out = []
    for d in dirs:
        i = api.get_paths_info(REPO, [f"{d}/hi_IN-custom-medium.onnx"], expand=True)[0]
        m = re.search(r"step_(\d+)", d)
        out.append({"path": d, "step": int(m.group(1)) if m else None, "experiment_id": d.split("/")[1] if d.startswith("experiments/") else "unknown (legacy path, shared by v4/v5)",
                    "uploaded_utc": i.last_commit.date.strftime("%Y-%m-%dT%H:%M:%SZ"), "commit_title": i.last_commit.title, "onnx_sha256": i.lfs.sha256})
    return out


def resolve_hf(path: str) -> dict:
    from huggingface_hub import hf_hub_download

    path = path.strip("/")
    api = _api()
    onnx = hf_hub_download(REPO, f"{path}/hi_IN-custom-medium.onnx")
    hf_hub_download(REPO, f"{path}/hi_IN-custom-medium.onnx.json")
    info = api.get_paths_info(REPO, [f"{path}/hi_IN-custom-medium.onnx"], expand=True)[0]
    sess = {}
    try:
        sess = json.loads(Path(hf_hub_download(REPO, f"{path}/session.json", force_download=True)).read_text("utf-8"))
    except Exception:  # noqa: BLE001
        pass
    m = re.search(r"step_(\d+)", path)
    legacy = not path.startswith("experiments/")
    return {"path": path, "onnx": Path(onnx), "global_step": int(m.group(1)) if m else sess.get("step"), "onnx_sha256": info.lfs.sha256 if info.lfs else sha256(onnx),
            "hf_commit": {"date_utc": info.last_commit.date.strftime("%Y-%m-%dT%H:%M:%SZ"), "title": info.last_commit.title, "id": info.last_commit.oid},
            "experiment_id": sess.get("experiment_id") or (path.split("/")[1] if not legacy else "unknown (legacy milestones/step_N: v4 and v5 wrote the same names; see hf_commit.date_utc)"),
            "git_sha": sess.get("git_sha", "unknown"), "session": sess.get("session", "unknown")}


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def dataset_info() -> dict:
    meta = ROOT / "data/hi_f/metadata.csv"
    d = {"name": "IndicTTS Hindi female (data/hi_f), private HF repo " + REPO, "metadata_sha256": sha256(meta) if meta.exists() else "unknown (data/hi_f/metadata.csv not present locally)"}
    return d


def eval_set_info(path) -> dict:
    sents = me.load_sentences(path)
    info = {"file": str(Path(path).name), "sha256": sha256(path), "n": len(sents)}
    try:
        from bench import check_eval_overlap as ov

        csvs = [ROOT / "data/hi_f/metadata.csv", ROOT / "data/hi_f/test.csv"]
        if all(c.exists() for c in csvs):
            grams, exact = set(), set()
            for c in csvs:
                for line in c.read_text("utf-8").splitlines():
                    w = ov.words(line.split("|", 1)[-1])
                    exact.add(" ".join(w))
                    grams.update(tuple(w[i:i + 5]) for i in range(len(w) - 4))
            bad = 0
            for s in sents:
                w = ov.words(s)
                bad += (" ".join(w) in exact) or any(tuple(w[i:i + 5]) in grams for i in range(len(w) - 4))
            info["overlap_with_training_and_test_text"] = {"criterion": "exact sentence or any shared 5-word run", "n_overlapping": int(bad)}
        else:
            info["overlap_with_training_and_test_text"] = "not checked (data/hi_f csv not present)"
    except Exception as e:  # noqa: BLE001
        info["overlap_with_training_and_test_text"] = f"check failed: {e!r}"[:120]
    return info


# ---------------------------------------------------------------- evaluation of one checkpoint
def evaluate_checkpoint(onnx: Path, sents, params, repeats, asr, utmos, refs, telephony=True, log=print) -> dict:
    """Returns raw matrices (n_sentences x repeats) for every metric plus transcripts of repeat 0."""
    from app.services.text_normalizer import normalize
    from training.asr import cer, per

    synth = me.make_synth(onnx, params)
    n = len(sents)
    M = {k: np.zeros((n, repeats)) for k in ("cer", "cer_legacy", "per", "utmos", "spk_neural", "spk_mfcc", "cer_8k", "cer_16k")}
    hyps, refs_txt, durs = [], [], []
    enc = _neural_encoder()
    cent_n, cent_m, loo = _centroids(refs, enc)
    for i, text in enumerate(sents):
        ref = normalize(text)
        refs_txt.append(ref)
        for r in range(repeats):
            wav, sr = synth(ref)
            hyp = me.transcribe(asr, wav, sr)
            nh = normalize(hyp)
            M["cer"][i, r], M["per"][i, r], M["cer_legacy"][i, r] = cer(ref, nh), per(ref, nh), cer(ref, nh, keep_marks=False)
            if utmos is not None and utmos.fn:
                M["utmos"][i, r] = utmos(wav, sr)
            if enc is not None:
                M["spk_neural"][i, r] = float(np.dot(enc(wav, sr), cent_n))
            M["spk_mfcc"][i, r] = _mfcc_sim(wav, sr, cent_m)
            if telephony:
                for k in ("8k", "16k"):
                    w2, s2 = channel(wav, sr, k)
                    M["cer_" + k][i, r] = cer(ref, normalize(me.transcribe(asr, w2, s2)))
            if r == 0:
                hyps.append(hyp); durs.append(len(wav) / sr)
        log(f"  {i + 1:2d}/{n} cer={M['cer'][i].mean():.3f} per={M['per'][i].mean():.3f} utmos={M['utmos'][i].mean():.2f} {ref[:36]}")
    return {"M": M, "hyps": hyps, "refs": refs_txt, "durs": durs, "loo": loo}


_SPK = {}


def _neural_encoder():
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav

        enc = _SPK.setdefault("enc", VoiceEncoder("cpu", verbose=False))
        return lambda w, sr: _unit(enc.embed_utterance(preprocess_wav(soxr.resample(w, sr, 16000) if sr != 16000 else w, source_sr=16000)))
    except Exception:  # noqa: BLE001
        return None


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def _mfcc_vec(w, sr):
    from app.services.speaker_encoder import SpeakerEncoder

    enc = _SPK.setdefault("mfcc", SpeakerEncoder("mfcc"))
    return enc.embed(w, sr)


def _mfcc_sim(w, sr, cent):
    if cent is None:
        return 0.0
    return float(np.dot(_unit(_mfcc_vec(w, sr)), cent))


def _centroids(refs, enc):
    if not refs:
        return None, None, None
    wavs = [sf.read(str(p), dtype="float32") for p in refs]
    ne = [enc(w, sr) for w, sr in wavs] if enc else []
    cn = _unit(np.mean(ne, axis=0)) if ne else None
    loo = None
    if len(ne) > 1:
        loo = round(float(np.mean([np.dot(e, _unit(np.mean([x for j, x in enumerate(ne) if j != i], axis=0))) for i, e in enumerate(ne)])), 4)
    try:  # raw (unstandardized) MFCC-statistics vectors, cosine to the centroid of the unit reference vectors: same definition as app.services.speaker_encoder
        cm = _unit(np.mean([_unit(_mfcc_vec(w, sr)) for w, sr in wavs], axis=0))
    except Exception:  # noqa: BLE001
        cm = None
    return cn, cm, loo


def build_entry(meta: dict, raw: dict, params, repeats, asr, asr_name, utmos_note, n_refs, b, seed) -> dict:
    M = raw["M"]
    e = {"checkpoint": {"hf_path": meta.get("path"), "global_step": meta.get("global_step"), "onnx_sha256": meta.get("onnx_sha256"), "hf_commit": meta.get("hf_commit")},
         "experiment_id": meta.get("experiment_id", "unknown"), "git_sha": meta.get("git_sha", "unknown"), "session": meta.get("session", "unknown"),
         "dataset": dataset_info(), "inference_params": params, "repeats": repeats, "asr": asr_name,
         "cer": summarize(M["cer"], b, seed), "per": summarize(M["per"], b, seed),
         "cer_legacy_skeleton": {"label": "CER on consonants only (vowel signs dropped): the metric of every row before 2026-09-30, kept only to compare with them", **summarize(M["cer_legacy"], b, seed)}}
    e["utmos"] = ({**summarize(M["utmos"], b, seed), "label": "UTMOS22-strong PREDICTED MOS (English-trained); relative ranking only", "backend": utmos_note}
                  if np.any(M["utmos"]) else {"skipped": utmos_note})
    sn = {"label": "Resemblyzer GE2E d-vector cosine to centroid of real held-out clips; embedding similarity, NOT speaker verification, NOT naturalness",
          "n_refs": n_refs, "real_vs_real_loo": raw["loo"]}
    sn.update(summarize(M["spk_neural"], b, seed) if np.any(M["spk_neural"]) else {"skipped": "resemblyzer unavailable"})
    e["speaker_similarity"] = {"neural": sn, "mfcc": {"label": "handcrafted MFCC-statistics cosine (app.services.speaker_encoder mfcc backend); saturates near 0.99 for any same-language speech, weak drift check only",
                                                      **(summarize(M["spk_mfcc"], b, seed) if np.any(M["spk_mfcc"]) else {"skipped": "no mfcc"})}}
    e["telephony"] = {"8k": {"label": "8 kHz G.711 mu-law round trip (synthetic channel)", **summarize(M["cer_8k"], b, seed)} if np.any(M["cer_8k"]) else "skipped",
                      "16k": {"label": "16 kHz 8-bit mu-law companding round trip (synthetic channel)", **summarize(M["cer_16k"], b, seed)} if np.any(M["cer_16k"]) else "skipped"}
    e["ci"] = {"method": "two-level percentile bootstrap (sentences, then repeats), 95%", "resamples": b, "seed": seed}
    e["per_sentence_worst_cer"] = [{"i": int(i) + 1, "cer": round(float(M["cer"][i].mean()), 3), "per": round(float(M["per"][i].mean()), 3), "ref": raw["refs"][i], "hyp_repeat0": raw["hyps"][i]}
                                   for i in np.argsort(-M["cer"].mean(axis=1))[:5]]
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--milestone", action="append", default=[], help="HF folder, e.g. milestones/step_340000 or experiments/<id>/milestones/step_N (repeatable)")
    ap.add_argument("--onnx", action="append", default=[], help="local .onnx (repeatable; provenance unknown)")
    ap.add_argument("--list", action="store_true", help="list milestones on HF with upload time and exit")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--asr", default="small")
    ap.add_argument("--sentences", default=str(me.DEFAULT_SENTENCES))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--n-refs", type=int, default=12)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument("--no-telephony", action="store_true")
    ap.add_argument("--no-utmos", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if a.list:
        for m in list_milestones():
            print(f"{m['uploaded_utc']}  step {m['step']}  {m['path']}  [{m['experiment_id']}]  {m['onnx_sha256'][:12]}")
        return
    if a.repeats < 3:
        print("WARNING: repeats < 3: the repeat variance cannot be estimated properly", file=sys.stderr)
    metas = [resolve_hf(p) for p in a.milestone] + [{"path": None, "onnx": Path(p), "global_step": None, "onnx_sha256": sha256(p), "hf_commit": None} for p in a.onnx]
    if not metas:
        ap.error("give at least one --milestone/--onnx")
    sents = me.load_sentences(a.sentences)[:a.limit]
    params = dict(me.DEFAULT_PARAMS)
    utmos = None if a.no_utmos else me.Utmos()
    refs = me.pick_refs(a.n_refs, None)
    out = {"schema": SCHEMA, "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "eval_set": eval_set_info(a.sentences),
           "asr": a.asr, "inference_params": params, "repeats": a.repeats, "n_reference_clips": len(refs), "results": [], "paired_vs_first": []}
    mats = []
    t0 = time.time()
    for m in metas:
        print(f"== {m.get('path') or m['onnx']} step {m.get('global_step')} sha {m['onnx_sha256'][:12]}", flush=True)
        raw = evaluate_checkpoint(m["onnx"], sents, params, a.repeats, a.asr, utmos, refs, telephony=not a.no_telephony)
        mats.append(raw["M"])
        out["results"].append(build_entry(m, raw, params, a.repeats, a.asr, a.asr, getattr(utmos, "note", None), len(refs), a.bootstrap, a.seed))
    for i in range(1, len(mats)):
        d = {"baseline": out["results"][0]["checkpoint"]["hf_path"] or "onnx0", "candidate": out["results"][i]["checkpoint"]["hf_path"] or f"onnx{i}"}
        for k, name in (("cer", "cer"), ("per", "per"), ("utmos", "utmos"), ("spk_neural", "speaker_similarity_neural"), ("cer_8k", "telephony_8k_cer")):
            if np.any(mats[i][k]) and np.any(mats[0][k]):
                d[name] = paired_delta(mats[0][k], mats[i][k], a.bootstrap, a.seed)
        out["paired_vs_first"].append(d)
    out["eval_seconds"] = round(time.time() - t0)
    path = Path(a.out or HERE / "results" / f"compare_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")
    print("wrote", path)
    for r in out["results"]:
        c, p_, u = r["cer"], r["per"], r["utmos"]
        print(f"{r['checkpoint']['hf_path']}: CER {c['mean']} {c['ci95']} (repeat std {c['repeat_std']}) PER {p_['mean']} {p_['ci95']} UTMOS {u.get('mean')} {u.get('ci95')}")


if __name__ == "__main__":
    main()
