"""V7 evaluation harness: one or more voices x one hash-pinned corpus version -> one JSON. Model-agnostic (Piper medium, high, multi-speaker).

  python -m bench.v7_eval --system v6=voices/hi_IN-custom-medium.onnx --quick --asr small
  python -m bench.v7_eval --system med=exp/medium.onnx@young_female --system high=exp/high.onnx@young_female --asr large-v3-turbo   # paired A/B, same corpus + params
  python -m bench.v7_eval --system rohan=voice:hi_IN-rohan-medium --quick                                                       # a server voice id

System spec: NAME=PATH.onnx (single speaker), NAME=PATH.onnx@SPEAKER (speaker name or numeric sid of a multi-speaker model) or NAME=voice:VOICE_ID
(an id the server would serve, resolved through voices/catalog.json and the model directories).

Path under test: bench.corpus.load(version) -> app.services.tts.stream (normalize, chunking, scheduler, lead/tail trim, PCM16) on a real PiperEngine.
Reference text for CER/PER is text_normalizer.normalize(text); the hypothesis is normalized the same way. CER is vowel-aware (training.asr.cer keeps matras).

Per system the JSON holds: CER and PER (overall and per category, bootstrap 95% CI, same ASR model name), UTMOS22 as PREDICTED MOS (never a MOS; skipped with a
reason when unavailable), speaker similarity (Resemblyzer cosine to a --ref-dir reference set when given, else leave-one-out self-consistency across the system's own
utterances: a drift check, not similarity to a person), TTFA/RTF (warm, single stream, N repeats, p50/p95), RSS, total audio and seconds per character, clipping
fraction, and telephony variants: the same waveform pushed through the server's resampler (soxr ResampleStream HQ + to_pcm16, as in tts.stream) at 8 and 16 kHz,
CER at each. With two or more systems it adds paired bootstrap deltas against the first (bench.compare_checkpoints.paired_delta).

Reproducibility: corpus sha256 is verified; the Piper ONNX graph draws noise with RandomNormalLike that has no seed, so the harness writes an in-memory copy of
the graph with a fixed `seed` attribute on those nodes (`--no-seed-noise` keeps the original; the model's sha256 is always the ORIGINAL file's). With the
seed, the same model, corpus order and --seed give identical audio in a fresh process; timing, RSS and creation time are the only fields allowed to differ
(`strip_timing`). Bootstrap and --quick sampling use --seed too.

Caveats: ASR errors are inflated by loanwords/numbers (read PER too); UTMOS is English-trained; RSS is process-wide (ru_maxrss never goes down), so the
exact per-voice peak needs one --system per process; the 8/16 kHz variants are resampling only (no codec). Nothing here says how natural a voice sounds:
that needs the listening test (docs/listening-test/v7/README.md).
"""
import argparse
import asyncio
import datetime as dt
import gc
import hashlib
import importlib.metadata as md
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soxr

from bench import compare_checkpoints as cc
from bench import corpus as corpus_mod

HERE = Path(__file__).parent
ROOT = HERE.parent
SCHEMA = "v7_eval/v1"
TELEPHONY = (8000, 16000)
PAIRED = ("cer", "per", "cer_8k", "cer_16k", "utmos", "spk_self")
QUICK_ROWS = 30


# ---------------------------------------------------------------- small helpers
def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def pct(xs, q):
    return round(float(np.percentile(xs, q)), 3) if len(xs) else None


def rss_mb() -> dict:
    """current RSS (ps) and the process high-water mark (ru_maxrss: bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20 if sys.platform == "darwin" else 1 << 10)
    try:
        cur = int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout.strip()) / 1024
    except (OSError, ValueError):
        cur = None
    return {"current_mb": round(cur, 1) if cur else None, "process_peak_mb": round(peak, 1)}


def server_resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """What tts.stream does to a chunk when the client asks for another rate: soxr ResampleStream (HQ), then to_pcm16; returned as float."""
    from app.services import audio_utils

    w = wav
    if sr_out != sr_in:
        w = soxr.ResampleStream(sr_in, sr_out, 1, dtype="float32", quality="HQ").resample_chunk(np.asarray(wav, np.float32), last=True)
    return np.frombuffer(audio_utils.to_pcm16(w), dtype="<i2").astype(np.float32) / 32767


def environment() -> dict:
    def ver(p):
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return None

    cpu = platform.processor()
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or cpu
        if not cpu and Path("/proc/cpuinfo").exists():
            cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name")), "")
    except OSError:
        pass
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    except OSError:
        git, dirty = "", None
    return {"cpu": cpu, "cpu_count": os.cpu_count(), "platform": platform.platform(), "python": platform.python_version(),
            "git_commit": git or None, "git_dirty_tracked": dirty,
            "packages": {p: ver(p) for p in ("numpy", "onnxruntime", "piper-tts", "soxr", "faster-whisper", "ctranslate2", "resemblyzer", "onnx")}}


# ---------------------------------------------------------------- systems (engine + voice id)
class System:
    def __init__(self, name: str, engine, voice: str, model_path: Path | None = None, params: dict | None = None):
        self.name, self.engine, self.voice, self.model_path, self.params = name, engine, voice, model_path, params or {}

    @property
    def rules(self) -> str | None:
        """Pronunciation rule groups the server applies to this voice (catalog `pronunciation_rules`; None = settings default). The CER/PER
        reference is normalized with the same rules, so V7 voices are scored on what they are actually asked to say."""
        from app.services import voice_catalog

        return voice_catalog.rules_of(self.voice)

    def info(self) -> dict:
        return {"name": self.name, "pronunciation_rules": self.rules, "voice_id": self.voice, "sample_rate": self.engine.sample_rate(self.voice),
                "model_file": self.model_path.name if self.model_path else None,
                "model_sha256": sha256_file(self.model_path) if self.model_path else None, "inference_params": self.params}


def seeded_make_session(seed: int):
    """Drop-in for piper_engine.make_session: the same session, but the graph copy has `seed` on every RandomNormalLike (see module docstring)."""
    import onnx
    import onnxruntime as ort
    from onnx import helper

    def make(model: Path, threads: int, use_cuda: bool):
        m = onnx.load(str(model))
        for i, n in enumerate(x for x in m.graph.node if x.op_type == "RandomNormalLike"):
            n.attribute.append(helper.make_attribute("seed", float(seed + i)))
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])

    return make


def resolve_model(spec_path: str) -> tuple[Path, str | None]:
    """'voice:ID' -> (model file, speaker or None) through the catalog and model dirs; otherwise PATH[@SPEAKER]."""
    from app.core.config import settings
    from app.services import piper_engine, voice_catalog

    if spec_path.startswith("voice:"):
        vid = spec_path[6:]
        entry = voice_catalog.get().get(vid)
        stem = entry.model if entry else vid.split(":")[0]
        dirs = [settings.models_dir, *(Path(d.strip()) for d in settings.models_extra.split(",") if d.strip()), ROOT / "voices"]
        found = next((p for p in piper_engine.model_files(dirs) if p.stem == stem), None)
        if not found:
            raise SystemExit(f"voice {vid!r}: model {stem!r} not found in {dirs}")
        return found, (vid.split(":", 1)[1] if ":" in vid else None)
    path, _, spk = spec_path.partition("@")
    return Path(path), spk or None


def build_piper_systems(specs: list[str], seed_noise: bool, seed: int, noise_scale=None, noise_w=None) -> list[System]:
    """One PiperEngine per distinct model file, loaded from a temp dir that holds only that model (the server's own load path, so the voice id,
    sample rate, speaker id handling and warm-up are the server's). Per-system RSS is therefore not polluted by the other bundled voices."""
    from app.core.config import settings
    from app.services import piper_engine

    if noise_scale is not None:
        settings.noise_scale = noise_scale
    if noise_w is not None:
        settings.noise_w = noise_w
    if seed_noise:
        piper_engine.make_session = seeded_make_session(seed)
    engines: dict[Path, tuple] = {}
    out = []
    for spec in specs:
        name, _, rest = spec.partition("=")
        if not rest:
            raise SystemExit(f"--system {spec!r}: expected NAME=PATH[@SPEAKER] or NAME=voice:ID")
        path, spk = resolve_model(rest)
        path = path.resolve()
        if path not in engines:
            stage = Path(tempfile.mkdtemp(prefix="v7eval_"))
            for suffix in ("", ".json"):
                (stage / (path.name + suffix)).symlink_to(str(path) + suffix)
            old = (settings.models_dir, settings.models_extra)
            settings.models_dir, settings.models_extra = stage, ""
            try:
                eng = piper_engine.PiperEngine()
                eng.load()
            finally:
                settings.models_dir, settings.models_extra = old
            engines[path] = (eng, json.loads(Path(f"{path}.json").read_text("utf-8")))
        eng, cfg = engines[path]
        have = [v["voice_id"] for v in eng.voices()]
        want = path.stem if spk is None and path.stem in have else None
        if want is None:
            hits = [v["voice_id"] for v in eng.voices() if spk is not None and (v["voice_id"] == f"{path.stem}:{spk}" or str(v.get("speaker_id")) == spk or v["voice_id"] == spk)]
            if not hits:
                raise SystemExit(f"{name}: {path.name} has voices {have}; pick one with @SPEAKER")
            want = hits[0]
        inf = cfg.get("inference", {})
        out.append(System(name, eng, want, path, {"noise_scale": settings.noise_scale if settings.noise_scale is not None else inf.get("noise_scale"),
                                                  "noise_w": settings.noise_w if settings.noise_w is not None else inf.get("noise_w"),
                                                  "length_scale": 1.0, "source": "overrides" if noise_scale is not None or noise_w is not None else "voice config.json inference block",
                                                  "noise_seeded": bool(seed_noise), "noise_seed": seed if seed_noise else None}))
    return out


# ---------------------------------------------------------------- phase A: synthesis + latency
async def _once(tts, text: str, voice: str):
    t0 = time.perf_counter()
    first, parts = None, []
    async for b in tts.stream(text, voice, 1.0, None):
        if first is None:
            first = time.perf_counter() - t0
        parts.append(b)
    return b"".join(parts), first, time.perf_counter() - t0


def synthesize_all(system: System, rows: list[dict], repeats: int, log=print) -> dict:
    """Warm up, then for each row stream `repeats` times through tts.stream. Audio of repeat 0 is kept (int16) for scoring; every repeat is timed."""
    from app.core.config import settings
    from app.services import tts
    from app.services.scheduler import Scheduler

    old = (tts.engine, tts.scheduler, settings.cache_size)
    tts.engine, tts.scheduler, settings.cache_size = system.engine, Scheduler(1), 0  # 1 worker: chunks run in order, so the seeded noise stream is reproducible; no phrase cache: every call is a real synthesis
    try:
        for w in rows[:2]:
            asyncio.run(_once(tts, w["text"], system.voice))
        sr = system.engine.sample_rate(system.voice)
        audio, ttfa, rtf, per_row_t = [], [], [], []
        t_all = time.perf_counter()
        for i, r in enumerate(rows):
            for k in range(repeats):
                pcm, first, total = asyncio.run(_once(tts, r["text"], system.voice))
                a = np.frombuffer(pcm, dtype="<i2")
                if k == 0:
                    audio.append(a.copy())
                if len(a):
                    ttfa.append(first * 1000)
                    rtf.append(total / (len(a) / sr))
            if (i + 1) % 25 == 0:
                log(f"  [{system.name}] synthesized {i + 1}/{len(rows)}")
        return {"audio": audio, "sr": sr, "ttfa_ms": ttfa, "rtf": rtf, "synth_wall_s": time.perf_counter() - t_all,
                "threads": {"workers": 1, "threads_per_worker": settings.threads_per_worker}}
    finally:
        tts.engine, tts.scheduler, settings.cache_size = old


# ---------------------------------------------------------------- phase B: scoring
def score_system(system: System, rows: list[dict], syn: dict, asr, asr_name: str, utmos, encoder, ref_emb, log=print) -> list[dict]:
    from app.services.text_normalizer import normalize
    from training.asr import cer, per

    sr = syn["sr"]
    out, embs = [], []
    for i, r in enumerate(rows):
        a = syn["audio"][i]
        wav = a.astype(np.float32) / 32767
        ref = normalize(r["text"], system.rules)
        dur = len(wav) / sr
        row = {"id": r["id"], "category": r["category"], "subcategory": r["subcategory"], "ref": ref, "dur_s": round(dur, 3), "audio_sha256": hashlib.sha256(a.tobytes()).hexdigest()[:16],
               "s_per_char": round(dur / max(1, len(ref.replace(" ", ""))), 4), "clip_fraction": float(np.mean(np.abs(wav) >= 0.999)) if len(wav) else 0.0,
               "silent": bool(not len(wav) or np.max(np.abs(wav)) < 1e-3)}
        if asr is not None and not row["silent"]:
            hyp = asr(wav, sr)
            nh = normalize(hyp, system.rules)
            row |= {"hyp": hyp, "cer": cer(ref, nh), "per": per(ref, nh)}
            for t in TELEPHONY:
                w2 = server_resample(wav, sr, t)
                row[f"cer_{t // 1000}k"] = cer(ref, normalize(asr(w2, t), system.rules))
        elif asr is not None:
            row |= {"hyp": "", "cer": 1.0, "per": 1.0, "cer_8k": 1.0, "cer_16k": 1.0}
        if utmos is not None and not row["silent"]:
            row["utmos"] = float(utmos(wav, sr))
        if encoder is not None and not row["silent"]:
            try:
                e = encoder.embed(wav, sr)
                embs.append((i, np.asarray(e, np.float64)))
                if ref_emb is not None:
                    row["spk_ref"] = float(np.dot(e, ref_emb) / (np.linalg.norm(e) * np.linalg.norm(ref_emb)))
            except Exception:  # noqa: BLE001 - too short / clipped: left out, counted via n
                pass
        out.append(row)
        if (i + 1) % 25 == 0:
            log(f"  [{system.name}] scored {i + 1}/{len(rows)}")
    if len(embs) >= 3:  # leave-one-out centroid cosine, within this system only
        tot = np.sum([e for _, e in embs], axis=0)
        for i, e in embs:
            c = tot - e
            out[i]["spk_self"] = float(np.dot(e, c) / (np.linalg.norm(e) * np.linalg.norm(c)))
    return out


def stat(vals, b, seed) -> dict:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {"n": 0}
    s = cc.summarize(v[:, None], b, seed)
    return {"n": int(len(v)), "mean": s["mean"], "ci95": s["ci95"], "median": s["median_sentence"], "p90": s["p90_sentence"]}


def aggregate(rows: list[dict], syn: dict, b: int, seed: int) -> dict:
    sr = syn["sr"]
    by = defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    metric = lambda rs, k: stat([r.get(k) for r in rs], b, seed)
    keys = ("cer", "per", "cer_8k", "cer_16k", "utmos", "spk_ref", "spk_self")
    present = [k for k in keys if any(k in r for r in rows)]
    tot_dur, tot_chars = sum(r["dur_s"] for r in rows), sum(len(r["ref"].replace(" ", "")) for r in rows)
    return {
        "overall": {k: metric(rows, k) for k in present},
        "per_category": {c: {k: metric(rs, k) for k in present if k in ("cer", "per", "cer_8k", "cer_16k", "utmos")} | {"n": len(rs)} for c, rs in sorted(by.items())},
        "audio": {"total_s": round(tot_dur, 2), "seconds_per_char": round(tot_dur / max(1, tot_chars), 4), "chars_per_s": round(tot_chars / max(tot_dur, 1e-9), 2),
                  "silent_outputs": sum(r["silent"] for r in rows), "clipping_fraction_mean": float(np.mean([r["clip_fraction"] for r in rows])),
                  "clipping_fraction_max": float(max(r["clip_fraction"] for r in rows)), "rows_with_clipping": sum(r["clip_fraction"] > 0 for r in rows)},
        "timing": {"ttfa_ms": {"p50": pct(syn["ttfa_ms"], 50), "p95": pct(syn["ttfa_ms"], 95), "n_streams": len(syn["ttfa_ms"])},
                   "rtf": {"p50": pct(syn["rtf"], 50), "p95": pct(syn["rtf"], 95)}, "synth_wall_s": round(syn["synth_wall_s"], 1),
                   "note": "warm (2 warm-up streams), single stream at a time, phrase cache off, through app.services.tts.stream on this machine; localhost-free, no network", **syn["threads"]},
    }


def paired(per_system_rows: dict[str, list[dict]], b: int, seed: int) -> list[dict]:
    names = list(per_system_rows)
    out = []
    for other in names[1:]:
        d = {"baseline": names[0], "candidate": other, "note": "candidate minus baseline on the same corpus rows; excludes_zero = 95% paired-bootstrap CI excludes 0"}
        for k in PAIRED:
            a = [r.get(k) for r in per_system_rows[names[0]]]
            c = [r.get(k) for r in per_system_rows[other]]
            keep = [i for i in range(len(a)) if a[i] is not None and c[i] is not None]
            if len(keep) >= 5:
                d[k] = cc.paired_delta(np.array([a[i] for i in keep], float)[:, None], np.array([c[i] for i in keep], float)[:, None], b, seed)
        out.append(d)
    return out


def strip_timing(rep: dict) -> dict:
    """The report minus fields that legitimately differ between identical runs (wall-clock timing, RSS, creation time)."""
    rep = json.loads(json.dumps(rep))
    for k in ("created_utc", "elapsed_s"):
        rep.pop(k, None)
    for s in rep["systems"].values():
        s.pop("timing", None)
        s.pop("rss", None)
        s.get("aggregate", {}).pop("timing", None)
    return rep


# ---------------------------------------------------------------- orchestration
def run(systems: list[System], rows: list[dict], *, corpus_version: str, asr=None, asr_name: str | None = None, utmos=None, utmos_note: str = "skipped: not requested",
        encoder=None, encoder_name: str | None = None, ref_emb=None, latency_repeats: int = 3, seed: int = 0, bootstrap: int = 2000, quick: bool = False, log=print) -> dict:
    t0 = time.time()
    syn, rss = {}, {}
    for s in systems:  # phase A for every system first: ASR/UTMOS/encoder are not loaded yet, so RSS here is the synthesis side only
        syn[s.name] = synthesize_all(s, rows, latency_repeats, log)
        rss[s.name] = rss_mb()
    per_rows, systems_rep = {}, {}
    for s in systems:
        log(f"scoring {s.name}")
        per_rows[s.name] = score_system(s, rows, syn[s.name], asr, asr_name, utmos, encoder, ref_emb, log)
        agg = aggregate(per_rows[s.name], syn[s.name], bootstrap, seed)
        systems_rep[s.name] = {"system": s.info(), "aggregate": agg, "timing": agg["timing"], "rss": rss[s.name] | {
            "note": "after this system's synthesis phase, before ASR/UTMOS load; ru_maxrss is process-wide so exact per-voice peak needs one --system per process"},
            "rows": per_rows[s.name]}
    return {
        "schema": SCHEMA, "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "elapsed_s": round(time.time() - t0, 1),
        "mode": "quick" if quick else "full", "seed": seed, "environment": environment(),
        "corpus": {"version": corpus_version, "sha256": corpus_mod.VERSIONS[corpus_version][1] if corpus_version in corpus_mod.VERSIONS else None,
                   "n_rows": len(rows), "row_ids": [r["id"] for r in rows], "counts": corpus_mod.counts(rows)["category"]},
        "asr": {"model": asr_name, "engine": "faster-whisper int8 cpu beam 5 language hi" if asr_name else None, "skipped": None if asr else "not requested"},
        "cer": {"definition": "training.asr.cer (Levenshtein/len(ref), punctuation/space/nukta dropped, Devanagari vowel signs counted); ref = text_normalizer.normalize(text), hyp normalized the same way"},
        "per": {"definition": "training.asr.per (espeak-ng phoneme error rate, English runs in English IPA), script-neutral"},
        "utmos": {"label": "UTMOS22 PREDICTED MOS (English-trained); ranking aid, NOT a human MOS" if utmos else None, "backend": utmos_note if utmos else None,
                  "skipped": None if utmos else utmos_note},
        "speaker_similarity": {"encoder": encoder_name, "reference": "given clips (spk_ref)" if ref_emb is not None else None,
                               "self_consistency": "spk_self: leave-one-out cosine to the centroid of the system's other utterances (drift check, not similarity to a person)",
                               "skipped": None if encoder else "no neural encoder available"},
        "telephony": {"variants_hz": list(TELEPHONY), "method": "same waveform -> soxr ResampleStream HQ -> to_pcm16 (server path); ASR on the result; no codec"},
        "latency": {"repeats_per_row": latency_repeats},
        "paired_vs_first": paired(per_rows, bootstrap, seed) if len(systems) > 1 else [],
        "ci": {"method": "percentile bootstrap over corpus rows, 95%", "resamples": bootstrap, "seed": seed},
        "systems": systems_rep,
        "labels": "CER/PER = ASR intelligibility proxies; UTMOS = predicted MOS; none of this is a listening test",
    }


def build_deps(a, log=print):
    """ASR, UTMOS, encoder and reference embedding from CLI args. Each is optional and reports why it was skipped."""
    from bench import milestone_eval as me

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    asr = None if a.no_asr else (lambda w, sr: me.transcribe(a.asr, w, sr))
    utmos, note = None, "skipped: --no-utmos"
    if not a.no_utmos:
        u = me.Utmos()
        utmos, note = (u, u.note) if u.fn else (None, u.note)
    encoder, ref_emb, ename = None, None, None
    if not a.no_spk:
        try:
            from app.services.speaker_encoder import SpeakerEncoder

            encoder = SpeakerEncoder("resemblyzer")
            encoder.embed(np.random.default_rng(0).normal(0, 0.05, 32000).astype(np.float32), 16000)  # load now: fail here, not mid-run
            ename = "resemblyzer"
        except Exception as e:  # noqa: BLE001
            log(f"speaker encoder unavailable: {e!r}"[:160])
            encoder = None
        if encoder is not None and a.ref_dir:
            import soundfile as sf

            es = []
            for p in sorted(Path(a.ref_dir).glob("*.wav")):
                w, sr = sf.read(p, dtype="float32")
                es.append(encoder.embed(w.mean(axis=1) if w.ndim > 1 else w, sr))
            ref_emb = np.mean(es, axis=0) if es else None
    return asr, utmos, note, encoder, ename, ref_emb


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", required=True, help="NAME=PATH.onnx[@SPEAKER] | NAME=voice:ID (repeatable)")
    ap.add_argument("--corpus", default="v2")
    ap.add_argument("--quick", action="store_true", help=f"~{QUICK_ROWS} stratified rows, 2 latency repeats")
    ap.add_argument("--limit", type=int, help="first N rows (smoke test)")
    ap.add_argument("--asr", default=None, help="faster-whisper model: small | large-v3-turbo | large-v3 (default: small with --quick, else large-v3-turbo)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latency-repeats", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--noise-scale", type=float)
    ap.add_argument("--noise-w", type=float)
    ap.add_argument("--no-seed-noise", action="store_true")
    ap.add_argument("--ref-dir", help="directory of real reference wavs of the target speaker (spk_ref)")
    ap.add_argument("--no-asr", action="store_true")
    ap.add_argument("--no-utmos", action="store_true")
    ap.add_argument("--no-spk", action="store_true")
    ap.add_argument("--label")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    a.asr = a.asr or ("small" if a.quick else "large-v3-turbo")
    rows = corpus_mod.load(a.corpus)
    if a.quick:
        rows = corpus_mod.stratified(rows, QUICK_ROWS, a.seed)
    rows = rows[:a.limit]
    systems = build_piper_systems(a.system, not a.no_seed_noise, a.seed, a.noise_scale, a.noise_w)
    asr, utmos, note, enc, ename, ref_emb = build_deps(a)
    rep = run(systems, rows, corpus_version=a.corpus, asr=asr, asr_name=None if a.no_asr else a.asr, utmos=utmos, utmos_note=note, encoder=enc, encoder_name=ename,
              ref_emb=ref_emb, latency_repeats=a.latency_repeats or (2 if a.quick else 3), seed=a.seed, bootstrap=a.bootstrap, quick=a.quick)
    label = a.label or "_".join([*(s.name for s in systems), a.corpus, rep["mode"], a.asr])
    out = Path(a.out or HERE / "results" / "v7_eval" / f"{label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", "utf-8")
    for n, s in rep["systems"].items():
        o = s["aggregate"]["overall"]
        print(f"{n}: CER {o.get('cer', {}).get('mean')} {o.get('cer', {}).get('ci95')} PER {o.get('per', {}).get('mean')} UTMOS(pred) {o.get('utmos', {}).get('mean')} "
              f"TTFA p50/p95 {s['timing']['ttfa_ms']['p50']}/{s['timing']['ttfa_ms']['p95']} ms RTF p50 {s['timing']['rtf']['p50']}")
    print("wrote", out, f"({rep['elapsed_s']} s)")


if __name__ == "__main__":
    main()
