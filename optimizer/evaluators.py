"""Evaluators (spec §3-5). All return plain dicts of floats in [0, 1] unless named otherwise."""
import re
import unicodedata

import numpy as np

STOP = set("है हैं था थी थे का की के को में से पर ने और या तो भी ही यह वह एक हो कर जी आप मैं हम".split())


def canon(s: str) -> str:
    """Spelling-variant-insensitive form: NFC, no nukta, chandrabindu = anusvara, no punctuation, single spaces."""
    s = unicodedata.normalize("NFD", s or "").replace("़", "")
    s = unicodedata.normalize("NFC", s).replace("ँ", "ं").lower()
    s = "".join(" " if unicodedata.category(c)[0] in "PSZC" else c for c in s)  # \w misses matras/virama, so by category
    return " ".join(s.split())


def lev(a, b) -> int:
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        c = [i]
        for j, y in enumerate(b, 1):
            c.append(min(p[j] + 1, c[-1] + 1, p[j - 1] + (x != y)))
        p = c
    return p[-1]


def err(ref, hyp) -> float:
    return min(1.0, lev(ref, hyp) / max(1, len(ref)))


class PronunciationEvaluator:
    """evaluate(candidate, reference) -> dict. candidate: {'normalized','audio','sr','asr',...}; reference: a bench case."""

    def evaluate(self, candidate: dict, reference: dict) -> dict:
        raise NotImplementedError


class TextEvaluator(PronunciationEvaluator):
    def evaluate(self, candidate, reference):
        got = canon(candidate["normalized"])
        refs = [canon(r) for r in [reference["expected_normalized"], *reference.get("alternatives", [])]]
        e = min(err(r.replace(" ", ""), got.replace(" ", "")) for r in refs)
        return {"text_score": round(1 - e, 4), "text_exact": float(got in refs)}


def latin_to_dev(hyp: str) -> str:
    """Whisper writes API/Google in Latin even when the audio says ए पी आई/गूगल: map Latin tokens through pron_dict.json."""
    import json
    import re
    from pathlib import Path
    d = json.loads((Path(__file__).parent / "pron_dict.json").read_text("utf-8"))
    return re.sub(r"[A-Za-z0-9]+", lambda m: d[m[0]]["spoken"] if m[0] in d else m[0], hyp or "")


def digits_to_words(hyp: str) -> str:
    """Whisper writes '500 रुपये'/'24 अप्रैल'; the reference is spoken words, so run the transcript through the app normalizer."""
    from app.services import text_normalizer
    return text_normalizer.normalize(hyp or "") if re.search(r"\d", hyp or "") else hyp


def asr_metrics(hyp: str, reference: dict) -> dict:
    """CER/WER vs the best of (expected spoken form, alternatives, raw text: Whisper may write English words in Latin)."""
    refs = [reference["expected_normalized"], *reference.get("alternatives", []), reference["text"]]
    hs = list(dict.fromkeys(canon(x) for x in (hyp, latin_to_dev(hyp), digits_to_words(hyp), digits_to_words(latin_to_dev(hyp)))))
    best, h = min(((canon(r), x) for r in refs for x in hs), key=lambda p: err(p[0].replace(" ", ""), p[1].replace(" ", "")))
    rw, hw = best.split(), h.split()
    wer = min(1.0, lev(rw, hw) / max(1, len(rw)))
    important = [canon(w) for w in reference.get("important") or [w for w in rw if w not in STOP and len(w) > 2]]
    joined = h.replace(" ", "")
    hits, splits = 0, []
    for w in important:
        if f" {w} " in f" {h} ":
            hits += 1
        elif " " not in w and w in joined:  # heard, but with a word break inside (विश्वजीत -> विश्व जीत)
            splits.append(w)
    return {"cer": round(err(best.replace(" ", ""), joined), 4), "wer": round(wer, 4), "token_acc": round(1 - wer, 4),
            "important_acc": round(hits / len(important), 4) if important else 1.0, "splits": splits, "hyp": hyp}


class ASREvaluator(PronunciationEvaluator):
    """Whisper large-v3-turbo fp16 on GPU (HF cache). transcribe() batches; evaluate() scores one transcript."""

    def __init__(self, model="openai/whisper-large-v3-turbo", batch_size=8):
        self.model, self.batch_size, self.pipe = model, batch_size, None

    def transcribe(self, wavs_sr: list) -> list[str]:
        import soxr
        import torch
        from transformers import pipeline
        if self.pipe is None:
            torch.cuda.set_per_process_memory_fraction(0.9)
            self.pipe = pipeline("automatic-speech-recognition", model=self.model, dtype=torch.float16, device="cuda")
        xs = [{"raw": soxr.resample(w, sr, 16000) if sr != 16000 else w, "sampling_rate": 16000} for w, sr in wavs_sr]
        empty = [len(x["raw"]) < 1600 for x in xs]
        out = self.pipe([x for x, e in zip(xs, empty) if not e], batch_size=self.batch_size,
                        generate_kwargs={"language": "hi", "task": "transcribe"}) if not all(empty) else []
        it = iter(out)
        return ["" if e else next(it)["text"] for e in empty]

    def evaluate(self, candidate, reference):
        return asr_metrics(candidate["asr"], reference)


class AudioEvaluator(PronunciationEvaluator):
    """Spec §5, numpy only. 20 ms frames; speech = frame RMS above -40 dBFS."""

    def evaluate(self, candidate, reference=None):
        return audio_metrics(candidate["audio"], candidate["sr"], len(canon(candidate.get("normalized", "")).replace(" ", "")))


def audio_metrics(w: np.ndarray, sr: int, n_chars: int = 0) -> dict:
    if len(w) < sr // 20:
        return {"duration": len(w) / sr, "audio_quality": 0.0, "speech_ratio": 0.0, "pauses": 0, "rate_cps": 0.0,
                "rms_db": -120.0, "peak": 0.0, "clip_ratio": 0.0, "max_silence": 0.0, "discontinuities": 0, "repeat_score": 0.0,
                "noise_floor_db": -120.0, "issues": ["empty"]}
    hop = sr // 50
    fr = w[: len(w) // hop * hop].reshape(-1, hop)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    sp = db > -40
    idx = np.flatnonzero(sp)
    inner = sp[idx[0]: idx[-1] + 1] if len(idx) else sp
    runs, cur = [], 0  # internal silence runs (frames)
    for s in inner:
        if not s:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    speech_s = sp.sum() / 50
    peak = float(np.abs(w).max())
    env = db[sp] if sp.any() else db
    rep = 0.0
    if len(db) > 150:  # repeated artifact: energy envelope strongly self-similar at a 0.4-2 s lag
        e = (db - db.mean()) / (db.std() + 1e-9)
        rep = max(float(np.dot(e[:-L], e[L:]) / (len(e) - L)) for L in range(20, min(100, len(e) // 2)))
    m = {"duration": round(len(w) / sr, 3), "rms_db": round(float(env.mean()), 2), "peak": round(peak, 4),
         "clip_ratio": round(float((np.abs(w) > 0.99).mean()), 5), "speech_ratio": round(float(sp.mean()), 3),
         "max_silence": round(max(runs, default=0) / 50, 3), "pauses": sum(r >= 6 for r in runs),
         "lead_silence": round((idx[0] if len(idx) else len(sp)) / 50, 3),
         "noise_floor_db": round(float(np.percentile(db, 10)), 2),
         "discontinuities": int((np.abs(np.diff(w)) > 0.5).sum()), "repeat_score": round(rep, 3),
         "rate_cps": round(n_chars / speech_s, 2) if speech_s else 0.0}
    issues = []
    if m["clip_ratio"] > 0.001: issues.append("clipping")
    if m["max_silence"] > 1.2: issues.append("abnormal_silence")
    if m["discontinuities"] > 5: issues.append("discontinuity")
    if rep > 0.9: issues.append("repetition")
    if m["noise_floor_db"] > -50 and m["speech_ratio"] < 0.95: issues.append("noise")
    if m["rms_db"] < -35 or peak < 0.05: issues.append("too_quiet")
    m["issues"] = issues
    m["audio_quality"] = round(max(0.0, 1 - 0.25 * len(issues)), 3)
    return m


def proxies(case, text_m, asr_m, aud_m, cfg) -> dict:
    """Spec §12 dimension scores. pronunciation is objective; the rest are cheap audio proxies (not MOS)."""
    lo, hi = cfg["rate_cps"]
    r = aud_m.get("rate_cps", 0)
    rate_ok = 1.0 if lo <= r <= hi else max(0.0, 1 - min(abs(r - lo), abs(r - hi)) / lo)
    natural = 0.6 * rate_ok + 0.4 * (0.0 if "abnormal_silence" in aud_m["issues"] or "repetition" in aud_m["issues"] else 1.0)
    want = len(re.findall(r"[,;।.?!—]+\s+\S", case["text"]))  # pause points inside the utterance
    got = aud_m.get("pauses", 0)
    if got > want:  # a pause between number groups (phone, date, currency) is natural, not a prosody fault
        got = max(want, got - len(re.findall(r"\d+", case["text"])))
    prosody = 1.0 if want == got == 0 else max(0.0, 1 - abs(got - want) / max(want, got, 1))
    consistency = max(0.0, 1 - abs(aud_m.get("rms_db", -20) - cfg["target_rms_db"]) / 20)
    pron = 0.3 * text_m["text_score"] + 0.4 * (1 - asr_m["cer"]) + 0.3 * asr_m["important_acc"]
    return {"pronunciation": round(pron, 4), "naturalness": round(natural, 4), "prosody": round(prosody, 4),
            "audio_quality": aud_m["audio_quality"], "consistency": round(consistency, 4)}


def classify(s: dict) -> str:
    """Failure type, first that applies (Laya's classification output)."""
    if s.get("text_score", 1) < 0.95:
        return "normalization"
    if s.get("splits"):
        return "word_boundary"
    if s.get("cer", 0) > 0.15 or s.get("important_acc", 1) < 0.75:
        return "pronunciation"
    if s.get("audio_quality", 1) < 0.75:
        return "audio"
    if s.get("prosody", 1) < 0.5:
        return "prosody"
    return "ok"


class LayaEvaluator(PronunciationEvaluator):
    """Ranking/decision component (spec §3). Never trusted alone: it ranks on objective scores first.

    config `laya.backend`:
      laya          - convaiinnovations/laya (pip `laya`, text-only System-1 classifier, multilingual checkpoint for Hindi):
                      `score` = pronunciation-quality ordinal (blended at laya.weight), `noul` = acceptability sanity
                      check, `choice` = best candidate pick + failure type. Logged, never the sole authority.
      deterministic - weighted score, tie-break pronunciation then smaller change; confidence from margin + metric agreement
      gemini        - deterministic + training/v8/gemini audio judge naturalness blended at laya.gemini_weight (top-k)
    Laya's shipped multilingual checkpoint has no fitted temperatures: its confidence is over-confident as shipped.
    """
    _agent = None

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def evaluate(self, candidate, reference=None):
        s = candidate["scores"]
        return {"failure_type": classify({**s, **candidate.get("metrics", {})}), "score": s["total"]}

    def rank(self, cands: list[dict]) -> tuple[list[dict], float]:
        """cands: [{'scores': {...,'total'}, 'metrics': {...}, 'change': {...}, 'wav': path}] -> (best first, confidence)."""
        if self.cfg.get("backend") == "gemini":
            self._gemini(cands[: self.cfg.get("gemini_top_k", 3)])
        laya = self._laya(cands) if self.cfg.get("backend") == "laya" else None
        key = lambda c: (round(c["scores"]["total"], 3), c["scores"]["pronunciation"], -len(str(c.get("change"))))
        ranked = sorted(cands, key=key, reverse=True)
        if len(ranked) < 2:
            return ranked, 0.5
        a, b = ranked[0], ranked[1]
        margin = a["scores"]["total"] - b["scores"]["total"]
        agree = (a["metrics"].get("text_score", 0) >= b["metrics"].get("text_score", 0)) == (a["metrics"].get("cer", 1) <= b["metrics"].get("cer", 1))
        conf = min(1.0, 0.5 + 5 * margin) * (1.0 if agree else 0.7)
        if laya:  # Laya agreeing with the objective winner keeps confidence; a confident veto halves it
            la = a.get("laya", {})
            conf *= 1.0 if laya.get("pick") == ranked.index(a) or laya.get("pick") == cands.index(a) else 0.85
            if la.get("acceptable") is False and la.get("acceptable_p", 0) > 0.8:
                conf *= 0.5
        return ranked, round(conf, 3)

    LAYA_Q = {
        "quality": {"type": "score", "instructions": "How closely does `asr_heard` match `expected_speech` (same words, same pronunciation)?",
                    "criteria": ["completely different", "many words wrong", "one or two words wrong", "identical"]},
        "acceptable": {"type": "noul", "instructions": "Is `spoken_text` a correct way to read `input_text` aloud in Hindi?"},
        "failure": {"type": "choice", "instructions": "What is the main problem with `asr_heard` compared with `expected_speech`?",
                    "criteria": {"normalization": "numbers, dates, symbols or abbreviations read wrongly or not expanded",
                                 "word_boundary": "a word was split into two or merged", "pronunciation": "a word was mispronounced or misheard",
                                 "prosody": "pauses or rhythm are wrong", "ok": "no problem"}}}

    def _load(self):
        if LayaEvaluator._agent is None:
            import laya
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            LayaEvaluator._agent = laya.load(self.cfg.get("model", "ml"), device=dev)
        return LayaEvaluator._agent

    @staticmethod
    def _state(c):
        m = c["metrics"]
        return {"input_text": c["text"], "spoken_text": c["scores"].get("normalized", ""), "expected_speech": c.get("expected", ""),
                "asr_heard": m.get("hyp", ""), "cer": m.get("cer"), "wer": m.get("wer"), "split_words": m.get("splits", []),
                "audio_issues": m.get("a_issues", [])}

    def _laya(self, cands):
        """Annotates each cand with c['laya'] and blends quality into total; returns {'pick': index} from a `choice`."""
        try:
            agent = self._load()
            rs = agent.predict_batch([self._state(c) for c in cands], self.LAYA_Q)
            w = self.cfg.get("weight", 0.15)
            for c, r in zip(cands, rs):
                ans, prob = r.get("answers", {}), r.get("probabilities", {})
                qp = prob.get("quality") or {}
                levels = self.LAYA_Q["quality"]["criteria"]
                q = sum(levels.index(k) * float(v) for k, v in qp.items() if k in levels) / 3 if qp else None
                acc = ans.get("acceptable")
                c["laya"] = {"quality": q, "quality_answer": ans.get("quality"), "acceptable": acc if isinstance(acc, bool) else str(acc).lower() == "true",
                             "acceptable_p": max(map(float, (prob.get("acceptable") or {"x": 0}).values())),
                             "failure": ans.get("failure"), "confidence": r.get("confidence")}
                if q is not None:
                    c["scores"]["laya_quality"] = round(q, 4)
                    c["scores"]["total"] = round((1 - w) * c["scores"]["total"] + w * q, 4)
            if len(cands) > 1:
                opts = {f"c{i}": f"heard '{c['metrics'].get('hyp', '')}' for spoken '{c['scores'].get('normalized', '')}'" for i, c in enumerate(cands)}
                st = {"expected_speech": cands[0].get("expected", ""), "candidates": opts}
                r = agent.predict_batch([st], {"best": {"type": "choice", "instructions": "Which candidate's `heard` text matches `expected_speech` best?", "criteria": opts}})[0]
                best = r.get("answers", {}).get("best")
                return {"pick": int(best[1:]) if isinstance(best, str) and best[1:].isdigit() else None, "pick_p": (r.get("probabilities", {}).get("best") or {}).get(best)}
            return {}
        except Exception as e:  # optional model; objective ranking stands
            for c in cands:
                c.setdefault("laya", {})["error"] = f"{type(e).__name__}: {e}"[:200]
            return {}

    def _gemini(self, cands):
        import sys
        from optimizer.bench import ROOT
        sys.path.insert(0, str(ROOT / "training/v8/gemini"))
        import gem
        from audio_judge import part
        w = self.cfg.get("gemini_weight", 0.1)
        for c in cands:
            try:
                o = gem.ask_json(gem.PRO, f'Strict Hindi TTS judge. Intended text: "{c["text"]}". Return JSON {{"naturalness": 1-5}}.',
                                 [part(c["wav"])], extra=c["wav_hash"])
                n = (float(o["naturalness"]) - 1) / 4
                c["scores"]["laya_gemini"] = n
                c["scores"]["total"] = (1 - w) * c["scores"]["total"] + w * n
            except Exception as e:  # judge is optional; objective ranking stands
                c["scores"]["laya_gemini_error"] = type(e).__name__
