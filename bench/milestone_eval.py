"""Score one Piper milestone on the fixed 50-sentence Hindi eval set and append one JSON row to bench/results/milestones.jsonl.

Usage (from the repo root):
  python -m bench.milestone_eval --hf-step 345000                       # downloads milestones/step_345000/ from the private HF repo
  python -m bench.milestone_eval --hf-step 345000 --hf-folder milestones/step_345000_kv6-0930T1305   # session-suffixed folder
  python -m bench.milestone_eval --onnx voices/hi_IN-custom-medium.onnx --step 345000 --session v4
  python -m bench.milestone_eval --hf-step 345000 --asr large-v3        # stronger ASR (about 3 GB download, minutes on 4 CPUs)

Pipeline: text -> app.services.text_normalizer.normalize (same as the server; --no-normalize to skip) -> Piper ONNX with FIXED
inference params (noise_scale 0.667, noise_w 0.8, length_scale 1.0 = the voice defaults) -> faster-whisper (int8, CPU, beam 5,
language hi) -> training.asr.cer against the normalized reference, with the hypothesis normalized the same way (digits -> words).
Also: UTMOS22 (English-trained MOS predictor; only for ranking our own checkpoints) via the ONNX export
TigreGotico/utmos-onnx on the HF hub (falls back to torch.hub, else skipped with a note), and neural speaker similarity
(Resemblyzer, cosine to the mean embedding of a few real held-out dataset clips; the real-vs-real value is stored as the ceiling).

Noise: Piper's ONNX graph draws its own random noise (not seedable from Python), so re-scoring the same file moves CER: three runs of
milestone 345000 (small ASR) gave mean CER 0.1355 / 0.1411 (2 repeats) / 0.1453, a spread of 0.010 (std about 0.005), so treat differences below
about 0.01 as noise (use --repeats N to average).
Rows carry provenance (onnx sha256 + HF commit time). For repeat variance, CIs, telephony and paired comparisons use bench/compare_checkpoints.py.
Token: HF token is read from ~/.cache/huggingface/token (or HF_TOKEN); nothing is uploaded.
"""
import argparse
import datetime as dt
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

HERE = Path(__file__).parent
ROOT = HERE.parent
REPO = "vishwajit76/custom-tts-hindi-train"
DEFAULT_SENTENCES = HERE / "hi_eval_50.txt"
OUT = HERE / "results" / "milestones.jsonl"
DETAILS = HERE / "results" / "milestone_details"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
DEFAULT_PARAMS = {"noise_scale": 0.667, "noise_w": 0.8, "length_scale": 1.0}


def load_sentences(path) -> list[str]:
    return [l.strip() for l in Path(path).read_text(encoding="utf8").splitlines() if l.strip() and not l.startswith("#")]


def pct(xs, q):  # nearest-rank-with-interpolation percentile, no numpy dependency on empty input
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def now_stamps() -> dict:
    t = dt.datetime.now(dt.timezone.utc)
    return {"timestamp_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "timestamp_ist": t.astimezone(IST).strftime("%Y-%m-%dT%H:%M:%S+05:30")}


# ---------------------------------------------------------------- model resolution / synthesis
def resolve_hf(step: int, folder: str | None) -> tuple[Path, str]:
    from huggingface_hub import HfApi, hf_hub_download

    if folder is None:
        files = HfApi().list_repo_files(REPO)
        # legacy milestones/step_N[_session] and the experiment layout experiments/<id>/milestones/step_N
        names = sorted({f.rsplit("/", 1)[0] for f in files if f.endswith("/hi_IN-custom-medium.onnx") and (f.startswith("milestones/") or "/milestones/" in f)})
        cands = [n for n in names if re.fullmatch(rf"(milestones/step_{step}(_.+)?|experiments/[^/]+/milestones/step_{step})", n)]
        if not cands:
            sys.exit(f"no milestone for step {step} on {REPO}; have: {names}")
        exact = [n for n in cands if n == f"milestones/step_{step}"]
        if len(cands) > 1 and not exact:
            sys.exit(f"several experiments/sessions for step {step}: {cands}; pass --hf-folder <one of them>")
        if len(cands) > 1:
            print(f"NOTE: step {step} exists in {cands}; using the legacy {exact[0]} (pass --hf-folder to choose)", file=sys.stderr)
        folder = (exact or cands)[0]
    onnx = hf_hub_download(REPO, f"{folder}/hi_IN-custom-medium.onnx")
    hf_hub_download(REPO, f"{folder}/hi_IN-custom-medium.onnx.json")
    return Path(onnx), folder


def provenance(folder: str, onnx: Path) -> dict:
    """sha256 of the evaluated ONNX + the HF commit that wrote it: legacy milestones/step_N were overwritten by two concurrent sessions, so the step
    number alone does not identify the model that was scored."""
    import hashlib

    from huggingface_hub import HfApi

    out = {"onnx_sha256": hashlib.sha256(Path(onnx).read_bytes()).hexdigest(), "hf_folder": folder}
    try:
        i = HfApi().get_paths_info(REPO, [f"{folder}/hi_IN-custom-medium.onnx"], expand=True)[0]
        out["hf_commit_utc"], out["hf_commit_title"] = i.last_commit.date.strftime("%Y-%m-%dT%H:%M:%SZ"), i.last_commit.title
    except Exception as e:  # noqa: BLE001
        out["hf_commit_error"] = repr(e)[:80]
    return out


def make_synth(onnx: Path, params: dict):
    from piper import PiperVoice, SynthesisConfig

    voice = PiperVoice.load(str(onnx), config_path=str(onnx) + ".json")
    cfg = SynthesisConfig(noise_scale=params["noise_scale"], noise_w_scale=params["noise_w"], length_scale=params["length_scale"])

    def synth(text: str) -> tuple[np.ndarray, int]:
        chunks = list(voice.synthesize(text, syn_config=cfg))
        return np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32), chunks[0].sample_rate

    return synth


# ---------------------------------------------------------------- metrics
_ASR = {}


def transcribe(model_size: str, wav: np.ndarray, sr: int) -> str:
    if model_size not in _ASR:
        from faster_whisper import WhisperModel

        _ASR[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    w = soxr.resample(wav, sr, 16000) if sr != 16000 else wav
    segs, _ = _ASR[model_size].transcribe(w, language="hi", beam_size=5, condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segs)


class Utmos:
    """UTMOS22-strong. Backends: ONNX from the HF hub (works offline once cached), torch.hub (needs GitHub)."""

    def __init__(self):
        self.note = None
        self.fn = None
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            sess = ort.InferenceSession(hf_hub_download("TigreGotico/utmos-onnx", "utmos22_strong.onnx"),
                                        providers=["CPUExecutionProvider"])
            self.fn = lambda w16: float(sess.run(None, {"wave": w16[None].astype(np.float32)})[0].reshape(-1)[0])
            self.note = "utmos22_strong via TigreGotico/utmos-onnx (onnxruntime)"
        except Exception as e:  # noqa: BLE001
            try:
                import torch

                m = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()

                def f(w16):
                    with torch.inference_mode():
                        return float(m(torch.from_numpy(w16.astype(np.float32))[None], 16000))

                self.fn, self.note = f, "utmos22_strong via torch.hub"
            except Exception as e2:  # noqa: BLE001
                self.note = f"skipped: onnx {repr(e)[:80]}; torch.hub {repr(e2)[:80]}"

    def __call__(self, wav, sr):
        return self.fn(soxr.resample(wav, sr, 16000) if sr != 16000 else wav) if self.fn else float("nan")


def pick_refs(n: int, ref_dir: str | None) -> list[Path]:
    """A few real held-out dataset clips (3-10 s): --ref-dir with wavs, else the first suitable rows of data/hi_f/test.csv from HF."""
    if ref_dir:
        return sorted(Path(ref_dir).glob("*.wav"))[:n]
    from huggingface_hub import hf_hub_download

    rows = [l.split("|", 1)[0] for l in (ROOT / "data/hi_f/test.csv").read_text(encoding="utf8").splitlines() if l.strip()]
    out = []
    for fn in rows:
        p = Path(hf_hub_download(REPO, f"data/hi_f/wavs/{fn}"))
        d = sf.info(p).duration
        if 3.0 <= d <= 10.0:
            out.append(p)
        if len(out) == n:
            break
    return out


def spk_sim(wavs: list[tuple[np.ndarray, int]], refs: list[Path]) -> dict:
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except Exception as e:  # noqa: BLE001
        return {"skipped": f"resemblyzer unavailable: {repr(e)[:100]}"}
    enc = VoiceEncoder("cpu", verbose=False)

    def emb(w, sr):
        return enc.embed_utterance(preprocess_wav(soxr.resample(w, sr, 16000) if sr != 16000 else w, source_sr=16000))

    ref_e = [emb(*sf.read(str(p), dtype="float32")) for p in refs]
    cent = np.mean(ref_e, axis=0)
    cent /= np.linalg.norm(cent)
    sims = [float(np.dot(emb(w, sr), cent)) for w, sr in wavs]
    loo = []  # real-vs-real ceiling: each reference clip against the centroid of the others
    for i, e in enumerate(ref_e):
        c = np.mean([x for j, x in enumerate(ref_e) if j != i], axis=0)
        loo.append(float(np.dot(e, c / np.linalg.norm(c))))
    return {"mean": round(statistics.mean(sims), 4), "min": round(min(sims), 4), "n_refs": len(refs),
            "real_vs_real_mean": round(statistics.mean(loo), 4) if len(loo) > 1 else None, "per_sentence": [round(s, 4) for s in sims]}


def rescore_per(details_json: str, out: Path):
    """Backfill 'per' into rows written before PER existed: recompute from the hypotheses stored in milestone_details/."""
    from app.services.text_normalizer import normalize
    from training.asr import per

    dp = Path(details_json)
    d = json.loads(dp.read_text(encoding="utf8"))
    ps = [per(r["ref"], normalize(r["hyp"])) for r in d["sentences"]]
    for r, p in zip(d["sentences"], ps):
        r["per"] = round(p, 4)
    pp = [r["per"] for r in d["sentences"]]
    per_row = {"mean": round(statistics.mean(pp), 4), "median": round(statistics.median(pp), 4), "p90": round(pct(pp, 90), 4),
               "note": "backfilled from stored first-repeat hypotheses"}
    d["row"]["per"] = per_row
    ts = d["row"]["timestamp_utc"]
    lines = [json.loads(l) for l in out.read_text(encoding="utf8").splitlines() if l.strip()]
    for l in lines:
        if l.get("timestamp_utc") == ts and l["step"] == d["row"]["step"] and l["asr"] == d["row"]["asr"]:
            l["per"] = per_row
    out.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines), encoding="utf8")
    dp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf8")
    print(d["row"]["step"], d["row"]["asr"], "per", per_row)


# ---------------------------------------------------------------- main
def evaluate(synth, sents, asr, normalize_text=True, utmos=None, refs=None, save_wavs=None, repeats=1, log=print) -> dict:
    from app.services.text_normalizer import normalize
    from training.asr import cer, per

    rows, wavs = [], []
    for i, text in enumerate(sents, 1):
        ref = normalize(text) if normalize_text else text
        cs, ps, hyps, cl = [], [], [], []
        for r in range(repeats):
            wav, sr = synth(ref)
            if r == 0:
                wavs.append((wav, sr))
                if save_wavs:
                    Path(save_wavs).mkdir(parents=True, exist_ok=True)
                    sf.write(Path(save_wavs) / f"s{i:02d}.wav", wav, sr)
            hyp = transcribe(asr, wav, sr)
            hyps.append(hyp)
            nh = normalize(hyp) if normalize_text else hyp
            cs.append(cer(ref, nh))
            cl.append(cer(ref, nh, keep_marks=False))
            ps.append(per(ref, nh))
        row = {"i": i, "ref": ref, "hyp": hyps[0], "cer": round(statistics.mean(cs), 4), "cer_legacy": round(statistics.mean(cl), 4), "per": round(statistics.mean(ps), 4), "dur_s": round(len(wavs[-1][0]) / wavs[-1][1], 2)}
        if utmos:
            row["utmos"] = round(utmos(*wavs[-1]), 3)
        rows.append(row)
        log(f"  {i:2d}/{len(sents)} cer={row['cer']:.3f} utmos={row.get('utmos', float('nan')):.2f} {ref[:40]}")
    c = [r["cer"] for r in rows]
    res = {"cer": {"mean": round(statistics.mean(c), 4), "median": round(statistics.median(c), 4), "p90": round(pct(c, 90), 4),
                   "max": round(max(c), 4), "n": len(c)}}
    res["cer_legacy_skeleton"] = {"mean": round(statistics.mean(r["cer_legacy"] for r in rows), 4),
                                  "note": "consonant-only CER = the metric of rows before 2026-09-30 (vowel signs were dropped by a \\w regex bug); `cer` now counts them"}
    pp = [r["per"] for r in rows]
    res["per"] = {"mean": round(statistics.mean(pp), 4), "median": round(statistics.median(pp), 4), "p90": round(pct(pp, 90), 4),
                  "note": "phoneme error rate (espeak-ng, English runs in English IPA): script-neutral, so ASR writing 'online order' or 1988 costs little"}
    if utmos:
        u = [r["utmos"] for r in rows]
        res["utmos"] = ({"mean": round(statistics.mean(u), 3), "min": round(min(u), 3), "p10": round(pct(u, 10), 3), "note": utmos.note,
                         "label": "UTMOS22 PREDICTED MOS (English-trained), first repeat only; ranking aid, not a listening-test MOS"}
                        if utmos.fn else {"skipped": utmos.note})
    if refs:
        res["spk_sim"] = spk_sim(wavs, refs)
        res["spk_sim"]["label"] = "Resemblyzer d-vector cosine to real held-out clips (embedding similarity; not speaker verification, not naturalness)"
    return res, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--onnx", help="local milestone .onnx (its .onnx.json must sit next to it)")
    src.add_argument("--hf-step", type=int, help="fetch milestones/step_N from the private HF repo")
    ap.add_argument("--hf-folder", help="explicit HF folder, e.g. milestones/step_345000_kv6-0930T1305")
    ap.add_argument("--step", type=int, help="step label for --onnx runs")
    ap.add_argument("--session", help="session label (default: suffix of the HF folder name, else 'unrecorded')")
    ap.add_argument("--asr", default="small", help="faster-whisper size: small (default, comparable to older rows) or large-v3")
    ap.add_argument("--sentences", default=str(DEFAULT_SENTENCES))
    ap.add_argument("--noise-scale", type=float, default=DEFAULT_PARAMS["noise_scale"])
    ap.add_argument("--noise-w", type=float, default=DEFAULT_PARAMS["noise_w"])
    ap.add_argument("--length-scale", type=float, default=DEFAULT_PARAMS["length_scale"])
    ap.add_argument("--repeats", type=int, default=1, help="synthesize+score each sentence N times and average (noise band)")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--no-utmos", action="store_true")
    ap.add_argument("--no-spk", action="store_true")
    ap.add_argument("--n-refs", type=int, default=5)
    ap.add_argument("--ref-dir", help="dir of real reference wavs (default: held-out clips from data/hi_f/test.csv via HF)")
    ap.add_argument("--save-wavs", help="write s01.wav.. here")
    ap.add_argument("--limit", type=int, help="only the first N sentences (smoke test)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--rescore-per", metavar="DETAILS_JSON", help="add the PER field to an existing row from its stored hypotheses (no synthesis/ASR) and exit")
    ap.add_argument("--no-write", action="store_true", help="print the row, do not append to the jsonl")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if a.rescore_per:
        return rescore_per(a.rescore_per, Path(a.out))
    if not (a.onnx or a.hf_step):
        ap.error("one of --onnx / --hf-step is required")

    prov = {}
    if a.hf_step:
        onnx, folder = resolve_hf(a.hf_step, a.hf_folder)
        step, source = a.hf_step, f"hf:{REPO}/{folder}"
        session = a.session or (folder.split("_", 2)[2] if folder.count("_") >= 2 else "unrecorded")
        prov = provenance(folder, onnx)
    else:
        onnx, step, source = Path(a.onnx), a.step, f"file:{a.onnx}"
        session = a.session or "unrecorded"
    params = {"noise_scale": a.noise_scale, "noise_w": a.noise_w, "length_scale": a.length_scale, "normalize_text": not a.no_normalize}
    sents = load_sentences(a.sentences)[:a.limit]
    print(f"step {step} session {session} | {len(sents)} sentences | asr {a.asr} | params {params}", flush=True)

    t0 = time.time()
    utmos = None if a.no_utmos else Utmos()
    refs = None if a.no_spk else pick_refs(a.n_refs, a.ref_dir)
    res, rows = evaluate(make_synth(onnx, params), sents, a.asr, not a.no_normalize, utmos, refs, a.save_wavs, a.repeats)
    row = {"step": step, "session": session, "source": source, "asr": a.asr, "params": {k: v for k, v in params.items() if k != "normalize_text"},
           "normalize_text": not a.no_normalize, "repeats": a.repeats, "sentence_set": Path(a.sentences).name, **({"provenance": prov} if prov else {}), **res,
           "eval_s": round(time.time() - t0), **now_stamps()}
    row["spk_sim"].pop("per_sentence", None) if "spk_sim" in row else None
    print(json.dumps(row, ensure_ascii=False))
    if not a.no_write:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a", encoding="utf8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        DETAILS.mkdir(parents=True, exist_ok=True)
        (DETAILS / f"{step}_{session}_{a.asr}.json").write_text(json.dumps({"row": row, "sentences": rows}, ensure_ascii=False, indent=1), encoding="utf8")
        print("appended to", a.out)


if __name__ == "__main__":
    main()
