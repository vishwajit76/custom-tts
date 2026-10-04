"""Frontend, cached evaluation, scoring, gates, experiment DB, candidates, git checkpoints, reports (spec §2, §7-24)."""
import copy
import datetime as dt
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from optimizer import evaluators as ev
from optimizer.bench import ROOT

HERE = ROOT / "optimizer"
STATE = HERE / ".state"            # gitignored: DB, audio cache, private git index
REPORTS = ROOT / "reports/optimizer"
PRON = HERE / "pron_dict.json"
ACTIVE = HERE / "active_config.json"
DEFAULT_ACTIVE = {"rules": [], "pron_rules": "off", "default": {"pause_ms": 120, "speed": 1.0, "punct": None}, "category": {}}


def config(path=HERE / "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_json(p, default):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else copy.deepcopy(default)


def save_json(p, obj):
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")


# ---------------- frontend: library rules -> optimizer dictionary -> app normalizer -> punctuation -> chunks ----------
RULES = {  # small rewrites into a form the normalizer already reads correctly; candidates only, accepted by the gates
    "iso_date": (r"\b(\d{4})-(\d{2})-(\d{2})\b", lambda m: f"{int(m[3])}/{int(m[2])}/{m[1]}"),
    "hyphen_phone": (r"\b(\d{5})-(\d{5})\b", r"\1 \2"),
}
_B = r"\wऀ-ॿ"


def apply_rules(text, names):
    for n in names:
        pat, rep = RULES[n]
        text = re.sub(pat, rep, text)
    return text


def apply_dict(text, pron: dict):
    for src in sorted(pron, key=len, reverse=True):
        text = re.sub(rf"(?<![{_B}]){re.escape(src)}(?![{_B}])", pron[src]["spoken"], text)
    return text


def punct(text, mode):
    if mode == "comma_to_danda":
        return text.replace(",", "।")
    if mode == "conj_comma":
        return re.sub(r"(?<![,।])\s+(और|लेकिन|क्योंकि|तो|या)\s", r", \1 ", text)
    return text


def settings_for(active, category):
    return {**active["default"], **active["category"].get(category, {})}


def frontend(text, category, active, pron):
    from app.services import text_normalizer
    from app.services.tts import split_for_stream
    s = settings_for(active, category)
    norm = text_normalizer.normalize(apply_dict(apply_rules(text, active["rules"]), pron), rules=active["pron_rules"])
    norm = punct(norm, s["punct"])
    return norm, split_for_stream(norm) if norm else [], {"pause_ms": s["pause_ms"], "speed": s["speed"]}


def key_of(engine_version, chunks, params) -> str:
    """§21 cache key: hash(text after pronunciation config + synthesis params + model version)."""
    return hashlib.sha256(json.dumps([engine_version, chunks, params], ensure_ascii=False).encode()).hexdigest()[:24]


# ---------------- DB (§16) + cache (§21) + resumable state (§2) ----------------
class DB:
    def __init__(self, path=None):
        path = path or STATE / "optimizer.db"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.c = sqlite3.connect(path)
        self.c.executescript("""
        CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS state(k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE IF NOT EXISTS results(run TEXT, case_id TEXT, engine TEXT, category TEXT, priority INT, total REAL,
            scores TEXT, PRIMARY KEY(run, case_id, engine));
        CREATE TABLE IF NOT EXISTS experiments(id INTEGER PRIMARY KEY, iteration INT, test_id TEXT, engine TEXT, change TEXT,
            before REAL, after REAL, regression REAL, decision TEXT, confidence REAL, failure_type TEXT, reason TEXT, ts TEXT);
        """)
        try:
            self.c.execute("ALTER TABLE experiments ADD COLUMN laya TEXT")  # Laya answers + calibrated confidence
        except sqlite3.OperationalError:
            pass

    def get(self, key):
        r = self.c.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, key, value):
        self.c.execute("INSERT OR REPLACE INTO cache VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False)))
        self.c.commit()

    def state(self, k, default=None):
        r = self.c.execute("SELECT v FROM state WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_state(self, k, v):
        self.c.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (k, json.dumps(v)))
        self.c.commit()

    def save_results(self, run, engine, results):
        self.c.execute("DELETE FROM results WHERE run=? AND engine=?", (run, engine))
        self.c.executemany("INSERT INTO results VALUES(?,?,?,?,?,?,?)", [
            (run, r["id"], engine, r["category"], r["priority"], r["total"], json.dumps(r, ensure_ascii=False)) for r in results])
        self.c.commit()

    def results(self, run, engine) -> dict:
        return {cid: json.loads(s) for cid, s in self.c.execute(
            "SELECT case_id, scores FROM results WHERE run=? AND engine=?", (run, engine))}

    def log(self, **kw):
        kw = {"ts": dt.datetime.now().isoformat(timespec="seconds"), **kw}
        kw["change"] = json.dumps(kw.get("change"), ensure_ascii=False)
        cols = ",".join(kw)
        self.c.execute(f"INSERT INTO experiments({cols}) VALUES({','.join('?' * len(kw))})", list(kw.values()))
        self.c.commit()

    def q(self, sql, *a):
        return self.c.execute(sql, a).fetchall()


# ---------------- scoring (§12) + gates (§15, §23) ----------------
def total(scores: dict, weights: dict) -> float:
    return round(sum(w * scores.get(k, 0.0) for k, w in weights.items()) / sum(weights.values()), 4)


def gates(target_before, target_after, old: dict, new: dict, cases: dict, g: dict, rtf_old=None, rtf_new=None):
    """old/new: case_id -> result over the full benchmark. Returns (ok, reasons, regression = worst category drop)."""
    why = []
    if target_after - target_before < g["min_target_improvement"]:
        why.append(f"target +{target_after - target_before:.3f} < {g['min_target_improvement']}")
    for cid, n in new.items():
        if cases[cid]["priority"] == 1 and cid in old and old[cid]["pronunciation"] - n["pronunciation"] > g["critical_regression"]:
            why.append(f"critical regression {cid} pron {old[cid]['pronunciation']:.3f}->{n['pronunciation']:.3f}")
    worst = 0.0
    for cat in {cases[c]["category"] for c in new}:
        ids = [c for c in old if cases[c]["category"] == cat]
        o = np.mean([old[c]["total"] for c in ids])
        m = np.mean([new.get(c, old[c])["total"] for c in ids])
        worst = max(worst, o - m)
        if o - m > g["max_category_regression"]:
            why.append(f"category {cat} {o:.3f}->{m:.3f}")
    if rtf_old and rtf_new and (rtf_new - rtf_old) / rtf_old > g["max_rtf_degradation"] and rtf_new - rtf_old > g.get("min_rtf_delta", 0.02):
        why.append(f"rtf {rtf_old:.3f}->{rtf_new:.3f}")
    return not why, why, round(float(worst), 4)


# ---------------- evaluation with cache ----------------
class Evaluator:
    def __init__(self, cfg, db, engines: dict, asr=None):
        self.cfg, self.db, self.engines = cfg, db, engines
        self.asr = asr or ev.ASREvaluator(batch_size=cfg["concurrency"]["asr_batch"])
        self.text, self.laya = ev.TextEvaluator(), ev.LayaEvaluator(cfg["laya"])
        (STATE / "audio").mkdir(parents=True, exist_ok=True)

    def run(self, cases, engine, active, pron) -> list[dict]:
        import soundfile as sf
        eng = self.engines[engine]
        fronts = [frontend(c["text"], c["category"], active, pron) for c in cases]
        keys = [key_of(eng.version, ch, p) for _, ch, p in fronts]
        todo = {}
        for k, (norm, ch, p) in zip(keys, fronts):
            if k not in todo and self.db.get(k) is None:
                todo[k] = (norm, ch, p)
        new = []
        if todo and hasattr(eng, "ensure"):
            eng.ensure()
        for k, (norm, ch, p) in todo.items():  # synth serially per engine (one GPU model)
            t = time.perf_counter()
            wav, sr = eng.synth(ch, p["speed"], p["pause_ms"])
            dt_s = time.perf_counter() - t
            path = STATE / "audio" / f"{k}.wav"
            sf.write(path, wav, sr)
            new.append((k, norm, wav, sr, dt_s, str(path)))
        if new:
            hyps = self.asr.transcribe([(w, sr) for _, _, w, sr, _, _ in new])
            with ThreadPoolExecutor(self.cfg["concurrency"]["evaluation_workers"]) as pool:
                auds = list(pool.map(lambda x: ev.audio_metrics(x[2], x[3], len(ev.canon(x[1]).replace(" ", ""))), new))
            for (k, norm, wav, sr, dt_s, path), h, a in zip(new, hyps, auds):
                self.db.put(k, {"hyp": h, "audio": a, "synth_s": round(dt_s, 4), "wav": path,
                                "rtf": round(dt_s / max(a["duration"], 1e-3), 4)})
        return [self.score(c, norm, k, self.db.get(k)) for c, (norm, _, _), k in zip(cases, fronts, keys)]

    def time_rtf(self, cases, engine, active, pron, n=5):
        """Uncached mean RTF over up to n cases: old and new configs are timed back to back, so GPU state is comparable."""
        eng, rs = self.engines[engine], []
        for c in cases[:n]:
            _, ch, p = frontend(c["text"], c["category"], active, pron)
            t = time.perf_counter()
            wav, sr = eng.synth(ch, p["speed"], p["pause_ms"])
            rs.append((time.perf_counter() - t) / max(len(wav) / sr, 1e-3))
        return float(np.mean(rs)) if rs else None

    def score(self, case, norm, key, cached):
        t = self.text.evaluate({"normalized": norm}, case)
        a = ev.asr_metrics(cached["hyp"], case)
        s = ev.proxies(case, t, a, cached["audio"], self.cfg["proxy"])
        s["total"] = total(s, self.cfg["weights"])
        m = {**t, **{k: v for k, v in a.items()}, **{f"a_{k}": v for k, v in cached["audio"].items()}}
        out = {"id": case["id"], "category": case["category"], "priority": case["priority"], "normalized": norm, "key": key,
               "wav": cached["wav"], "rtf": cached["rtf"], **s, "metrics": m}
        out["failure_type"] = ev.classify({**s, **m, "audio_quality": s["audio_quality"]})
        return out


def failing(r, thr) -> bool:
    return r["total"] < thr or r["failure_type"] != "ok"


# ---------------- candidates (§7) ----------------
LETTERS = dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "ए बी सी डी ई एफ जी एच आई जे के एल एम एन ओ पी क्यू आर एस टी यू वी डब्ल्यू एक्स वाई ज़ेड".split()))


def token_variants(tok, case):
    from app.services import hinglish
    from app.services.pronunciation import lexical
    out = []
    if tok in case.get("terms", {}):
        out.append(case["terms"][tok])
    if tok.isupper() and tok.isalpha() and len(tok) <= 6:
        out.append(" ".join(LETTERS[c] for c in tok))
    for f in (lambda t: hinglish.brand(t), lambda t: hinglish.roman_to_devanagari(t.lower()),
              lambda t: lexical.tables(lexical.parse("all"))[1].get(t.lower())):
        try:
            v = f(tok)
        except Exception:
            v = None
        if v and v != tok and not re.search("[A-Za-z]", v):
            out.append(v)
    return list(dict.fromkeys(out))


def candidates(case, result, active, pron, loop_cfg, limit) -> list[dict]:
    """Bounded, ordered by the failure type. Each candidate is ONE change (one system per experiment, §25)."""
    from app.services.pronunciation import phonological
    ft, out = result["failure_type"], []
    dict_c = []
    for tok in dict.fromkeys(re.findall(r"[A-Za-z][A-Za-z']*", case["text"]) + list(case.get("terms", {}))):
        for v in token_variants(tok, case):
            if pron.get(tok, {}).get("spoken") != v:
                dict_c.append({"type": "dict", "source": tok, "spoken": v})
    for w in result["metrics"].get("splits", []) + [w for w in case.get("important", []) if " " not in w]:
        for v in {phonological.word(w)} - {w}:
            if re.search("[ऀ-ॿ]", w) and w in case["text"]:
                dict_c.append({"type": "dict", "source": w, "spoken": v})
    rule_c = [{"type": "rule", "name": n} for n, (pat, _) in RULES.items() if n not in active["rules"] and re.search(pat, case["text"])]
    s = settings_for(active, case["category"])
    set_c = [{"type": "setting", "category": case["category"], "key": "pause_ms", "value": v} for v in loop_cfg["pause_ms"] if v != s["pause_ms"]]
    set_c += [{"type": "setting", "category": case["category"], "key": "punct", "value": v} for v in ("conj_comma", "comma_to_danda", None)
              if v != s["punct"] and (v is None or re.search(r"[,]|\s(और|लेकिन|क्योंकि|तो|या)\s", case["text"]))]
    set_c += [{"type": "setting", "category": case["category"], "key": "speed", "value": v} for v in loop_cfg["speed"] if v != s["speed"]]
    order = {"normalization": rule_c + dict_c + set_c, "word_boundary": dict_c + set_c + rule_c,
             "pronunciation": dict_c + rule_c + set_c, "prosody": set_c + dict_c + rule_c}.get(ft, set_c + dict_c + rule_c)
    seen = []
    for c in order:
        if c not in seen:
            seen.append(c)
    return seen[:limit]


def apply_change(active, pron, ch, meta=None):
    a, p = copy.deepcopy(active), copy.deepcopy(pron)
    if ch["type"] == "dict":
        p[ch["source"]] = {"spoken": ch["spoken"], **(meta or {})}
    elif ch["type"] == "rule":
        a["rules"] = [*a["rules"], ch["name"]]
    elif ch["type"] == "setting":
        a["category"].setdefault(ch["category"], {})[ch["key"]] = ch["value"]
    return a, p


def describe(ch):
    return {"dict": lambda: f"dict {ch.get('source')} -> {ch.get('spoken')}", "rule": lambda: f"rule {ch.get('name')}",
            "setting": lambda: f"{ch.get('category')}.{ch.get('key')} = {ch.get('value')}",
            "none": lambda: "baseline (no change)"}[ch["type"]]()


# ---------------- git checkpoint (§14): commit to a branch through a private index; HEAD/working tree untouched ---------
COMMIT_PATHS = ["optimizer/__init__.py", "optimizer/bench.py", "optimizer/core.py", "optimizer/engines.py", "optimizer/evaluators.py",
                "optimizer/loop.py", "optimizer/config.yaml", "optimizer/pron_dict.json", "optimizer/active_config.json",
                "optimizer/.gitignore", "optimize.py", "evaluate.py", "benchmark.py", "tests/test_optimizer.py",
                "tests/pronunciation_bench", "reports/optimizer/.gitignore", "reports/optimizer/report.json",
                "reports/optimizer/report.html", "reports/optimizer/iterations"]


def git(*a, env=None):
    return subprocess.run(["git", *a], cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8")


def git_commit(branch, msg, paths=COMMIT_PATHS) -> str | None:
    env = {**os.environ, "GIT_INDEX_FILE": str(STATE / "git-index")}
    r = git("rev-parse", "--verify", "-q", f"refs/heads/{branch}")
    parent = r.stdout.strip() or git("rev-parse", "HEAD").stdout.strip()
    git("read-tree", parent, env=env)
    existing = [p for p in paths if (ROOT / p).exists()]
    git("add", "--", *existing, env=env)  # .gitignore rules apply: wavs, .state never enter
    tree = git("write-tree", env=env).stdout.strip()
    if tree == git("rev-parse", f"{parent}^{{tree}}").stdout.strip():
        return None
    c = git("commit-tree", tree, "-p", parent, "-m", msg + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", env=env).stdout.strip()
    git("update-ref", f"refs/heads/{branch}", c)
    return c


# ---------------- reports (§19, §24) ----------------
def summarize(results: dict) -> dict:
    rs = list(results.values())
    if not rs:
        return {}
    dims = ["total", "pronunciation", "naturalness", "prosody", "audio_quality", "consistency"]
    cats = {}
    for r in rs:
        cats.setdefault(r["category"], []).append(r)
    return {"n": len(rs), **{d: round(float(np.mean([r[d] for r in rs])), 4) for d in dims},
            "cer": round(float(np.mean([r["metrics"]["cer"] for r in rs])), 4),
            "text_exact": round(float(np.mean([r["metrics"]["text_exact"] for r in rs])), 4),
            "categories": {c: {"n": len(v), "total": round(float(np.mean([r["total"] for r in v])), 4),
                               "pronunciation": round(float(np.mean([r["pronunciation"] for r in v])), 4),
                               "text_exact": round(float(np.mean([r["metrics"]["text_exact"] for r in v])), 4),
                               "cer": round(float(np.mean([r["metrics"]["cer"] for r in v])), 4)} for c, v in sorted(cats.items())}}


def write_report(db: DB, engines: list, pron: dict, active: dict):
    REPORTS.mkdir(parents=True, exist_ok=True)
    data = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "engines": {}, "active_config": active,
            "dictionary": pron}
    for e in engines:
        base, cur = db.results(f"baseline:{e}", e), db.results(f"current:{e}", e)
        if not cur:
            continue
        worst = sorted(cur.values(), key=lambda r: r["total"])[:15]
        words = {}
        for r in cur.values():
            for w in r["metrics"].get("splits", []):
                words[w] = words.get(w, 0) + 1
        data["engines"][e] = {"baseline": summarize(base), "current": summarize(cur),
                              "top_failures": [{k: r[k] for k in ("id", "category", "total", "failure_type", "normalized")} |
                                               {"hyp": r["metrics"]["hyp"]} for r in worst],
                              "split_words": words}
    data["experiments"] = [dict(zip(("iteration", "test_id", "change", "before", "after", "regression", "decision", "confidence",
                                     "failure_type", "reason"), r)) for r in db.q(
        "SELECT iteration,test_id,change,before,after,regression,decision,confidence,failure_type,reason FROM experiments "
        "WHERE decision IN ('accepted','rejected') ORDER BY id DESC LIMIT 50")]
    data["repeat_failing_tests"] = db.q("SELECT test_id, COUNT(*) n FROM experiments WHERE decision='rejected' GROUP BY test_id HAVING n>1 ORDER BY n DESC LIMIT 20")
    save_json(REPORTS / "report.json", data)
    (REPORTS / "report.html").write_text(_html(data), encoding="utf-8")
    return data


def _html(d):
    import html
    esc = lambda x: html.escape(str(x))
    rows = []
    for e, v in d["engines"].items():
        b, c = v["baseline"], v["current"]
        rows.append(f"<h2>{esc(e)}</h2><table><tr><th>category</th><th>n</th><th>baseline total</th><th>current total</th>"
                    f"<th>text exact</th><th>CER</th></tr>")
        for cat, s in c.get("categories", {}).items():
            bt = b.get("categories", {}).get(cat, {}).get("total", "")
            rows.append(f"<tr><td>{esc(cat)}</td><td>{s['n']}</td><td>{bt}</td><td>{s['total']}</td><td>{s['text_exact']}</td><td>{s['cer']}</td></tr>")
        rows.append(f"<tr><th>all</th><th>{c['n']}</th><th>{b.get('total', '')}</th><th>{c['total']}</th><th>{c['text_exact']}</th><th>{c['cer']}</th></tr></table>")
        rows.append("<h3>Top failures</h3><table><tr><th>id</th><th>type</th><th>total</th><th>normalized</th><th>ASR</th></tr>" + "".join(
            f"<tr><td>{esc(r['id'])}</td><td>{esc(r['failure_type'])}</td><td>{r['total']}</td><td>{esc(r['normalized'])}</td><td>{esc(r['hyp'])}</td></tr>"
            for r in v["top_failures"]) + "</table>")
    rows.append("<h2>Recent experiments</h2><table><tr><th>it</th><th>test</th><th>change</th><th>before</th><th>after</th><th>regr</th><th>decision</th><th>conf</th><th>reason</th></tr>" + "".join(
        f"<tr><td>{x['iteration']}</td><td>{esc(x['test_id'])}</td><td>{esc(x['change'])}</td><td>{x['before']}</td><td>{x['after']}</td><td>{x['regression']}</td><td>{esc(x['decision'])}</td><td>{x['confidence']}</td><td>{esc(x['reason'])}</td></tr>"
        for x in d["experiments"]) + "</table>")
    return ("<!doctype html><meta charset=utf-8><title>V8 optimizer</title><style>body{font:14px system-ui;margin:16px;background:#fff;color:#111}"
            "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 6px;text-align:left}"
            "@media(prefers-color-scheme:dark){body{background:#111;color:#eee}td,th{border-color:#444}}</style>"
            f"<h1>V8 optimizer report</h1><p>{esc(d['generated'])} &middot; active config <code>{esc(json.dumps(d['active_config'], ensure_ascii=False))}</code>"
            f" &middot; dictionary entries: {len(d['dictionary'])}</p>" + "".join(rows))
