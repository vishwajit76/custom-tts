"""The optimization loop (spec §2, §13). One iteration = one target case, one accepted-or-rejected change."""
import datetime as dt
import shutil

from optimizer import bench, core
from optimizer.engines import make


class Session:
    def __init__(self, cfg=None, db=None, engines=None, asr=None):
        self.cfg = cfg or core.config()
        self.db = db or core.DB()
        self.cases = {c["id"]: c for c in bench.load()}
        self.engines = engines or {n: make(n) for n in self.cfg["engines"]}
        self.ev = core.Evaluator(self.cfg, self.db, self.engines, asr)
        self.active = core.load_json(core.ACTIVE, core.DEFAULT_ACTIVE)
        self.pron = core.load_json(core.PRON, {})

    def benchmark(self, engines=None, label=None):
        """Full bench on each engine under the active config -> runs baseline:<e> (first time), current:<e>, bench:<ts>:<e>."""
        out = {}
        for e in engines or self.cfg["engines"]:
            res = {r["id"]: r for r in self.ev.run(list(self.cases.values()), e, self.active, self.pron)}
            base = self.db.results(f"baseline:{e}", e)  # first score of each case is its baseline (new cases join later)
            self.db.save_results(f"baseline:{e}", e, [*base.values(), *(r for i, r in res.items() if i not in base)])
            self.db.save_results(f"current:{e}", e, res.values())
            self.db.save_results(label or f"bench:{dt.datetime.now():%Y%m%d-%H%M%S}", e, res.values())
            out[e] = core.summarize(res)
        core.write_report(self.db, self.cfg["engines"], self.pron, self.active)
        return out

    def current(self, e):
        cur = self.db.results(f"current:{e}", e)
        if len(cur) < len(self.cases):
            self.benchmark([e])
            cur = self.db.results(f"current:{e}", e)
        return cur

    def targets(self, cur, category=None, test=None):
        lc = self.cfg["loop"]
        tried = dict(self.db.q("SELECT test_id, COUNT(*) FROM experiments WHERE decision IN ('accepted','rejected') GROUP BY test_id"))
        tried.update({t: 99 for (t,) in self.db.q("SELECT DISTINCT test_id FROM experiments WHERE decision='exhausted'")})
        ts = [r for r in cur.values() if (test and r["id"] == test) or (not test and core.failing(r, lc["fail_threshold"])
              and tried.get(r["id"], 0) < lc["max_attempts_per_case"] and (not category or r["category"] == category))]
        return sorted(ts, key=lambda r: (r["priority"], r["total"], r["id"]))

    def iterate(self, it, category=None, test=None) -> dict | None:
        e = self.cfg["primary_engine"]
        cur = self.current(e)
        for before in self.targets(cur, category, test):  # first target with untried candidates
            case = self.cases[before["id"]]
            tried = {c for (c,) in self.db.q("SELECT change FROM experiments WHERE test_id=?", case["id"])}
            cands = core.candidates(case, before, self.active, self.pron, self.cfg["loop"], self.cfg["concurrency"]["max_candidates"])
            cands = [c for c in cands if core.json.dumps(c, ensure_ascii=False) not in tried]  # dedupe (§21)
            if cands:
                break
            self.db.log(iteration=it, test_id=case["id"], engine=e, change={"type": "none"}, decision="exhausted",
                        reason="all candidates already tried")  # not an attempt: next target, same iteration
        else:
            return None
        evald = [{"change": {"type": "none"}, "scores": before, "metrics": before["metrics"], "text": case["text"], "expected": case["expected_normalized"], "wav": before["wav"],
                  "wav_hash": before["key"]}]
        for ch in cands:
            a, p = core.apply_change(self.active, self.pron, ch)
            r = self.ev.run([case], e, a, p)[0]
            evald.append({"change": ch, "scores": r, "metrics": r["metrics"], "text": case["text"], "expected": case["expected_normalized"], "wav": r["wav"], "wav_hash": r["key"]})
            self.db.log(iteration=it, test_id=case["id"], engine=e, change=ch, before=before["total"], after=r["total"],
                        decision="candidate", failure_type=before["failure_type"])
        ranked, conf = self.ev.laya.rank(evald)
        best = ranked[0]
        rec = {"iteration": it, "case": case, "before": before, "best": best, "confidence": conf, "candidates": ranked,
               "failure_type": before["failure_type"]}
        if best["change"]["type"] == "none" or not cands:
            reason = "no candidate beat the current config" if cands else "no candidates"
            return self._decide(rec, False, [reason], 0.0, None)
        # regression: only cases whose frontend output changes need new audio; the rest are identical by cache key
        a, p = core.apply_change(self.active, self.pron, best["change"])
        eng = self.engines[e]
        affected = [c for c in self.cases.values()
                    if core.key_of(eng.version, *core.frontend(c["text"], c["category"], a, p)[1:]) != cur[c["id"]]["key"]]
        new = {r["id"]: r for r in self.ev.run(affected, e, a, p)}
        ok, why, regr = core.gates(before["total"], best["scores"]["total"], cur, new, self.cases, self.cfg["gates"],
                                   self.ev.time_rtf(affected, e, self.active, self.pron), self.ev.time_rtf(affected, e, a, p))
        rec["affected"] = sorted(new)
        return self._decide(rec, ok, why, regr, (a, p, new, cur))

    def _decide(self, rec, ok, why, regr, apply):
        it, case, before, best = rec["iteration"], rec["case"], rec["before"], rec["best"]
        decision = "accepted" if ok else "rejected"
        commit = None
        if ok:
            a, p, new, cur = apply
            ch = best["change"]
            if ch["type"] == "dict":  # §9 change metadata
                p[ch["source"]].update({"confidence": rec["confidence"], "reason": f"{rec['failure_type']} on {case['id']}",
                                        "score_before": before["total"], "score_after": best["scores"]["total"],
                                        "affected_tests": rec["affected"], "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                                        "iteration": it})
            self.active, self.pron = a, p
            core.save_json(core.ACTIVE, a)
            core.save_json(core.PRON, p)
            cur.update(new)
            self.db.save_results(f"current:{self.cfg['primary_engine']}", self.cfg["primary_engine"], cur.values())
        self.db.log(iteration=it, test_id=case["id"], engine=self.cfg["primary_engine"], change=best["change"],
                    before=before["total"], after=best["scores"]["total"], regression=regr, decision=decision,
                    confidence=rec["confidence"], failure_type=rec["failure_type"], reason="; ".join(why) or "gates passed",
                    laya=core.json.dumps(best.get("laya"), ensure_ascii=False))
        self._report(rec, decision, why, regr)
        core.write_report(self.db, self.cfg["engines"], self.pron, self.active)
        if ok:
            commit = core.git_commit(self.cfg["git"]["branch"], f"optimizer iteration {it:03d}: {core.describe(best['change'])} "
                                     f"({case['id']} {before['total']:.3f} -> {best['scores']['total']:.3f})")
        return {"iteration": it, "test": case["id"], "change": core.describe(best["change"]), "before": before["total"],
                "after": best["scores"]["total"], "regression": regr, "decision": decision, "why": why, "commit": commit,
                "confidence": rec["confidence"]}

    def _report(self, rec, decision, why, regr):
        d = core.REPORTS / "iterations" / f"{rec['iteration']:03d}"
        d.mkdir(parents=True, exist_ok=True)
        b, best = rec["before"], rec["best"]
        shutil.copy(b["wav"], d / "before.wav")
        shutil.copy(best["wav"], d / "after.wav")
        lines = [f"# Iteration {rec['iteration']:03d}: {rec['case']['id']} ({rec['case']['category']}, priority {rec['case']['priority']})", "",
                 f"- Problem: {rec['failure_type']}; text `{rec['case']['text']}`; expected `{rec['case']['expected_normalized']}`",
                 f"- Previous: normalized `{b['normalized']}`; ASR `{b['metrics']['hyp']}`; total {b['total']:.3f} (pron {b['pronunciation']:.3f}, CER {b['metrics']['cer']:.3f})",
                 f"- Candidate: {core.describe(best['change'])}; normalized `{best['scores']['normalized']}`; ASR `{best['metrics']['hyp']}`",
                 f"- Score {b['total']:.3f} -> {best['scores']['total']:.3f}; Laya confidence {rec['confidence']}; regression (worst category drop) {regr}",
                 f"- Decision: **{decision}** ({'; '.join(why) or 'gates passed'})", f"- Affected tests: {len(rec.get('affected', []))}",
                 f"- Laya on chosen: {best.get('laya', 'backend off')}", "- Audio: before.wav, after.wav (not committed)", "",
                 "| rank | change | total | pron | CER | text | laya quality / acceptable / failure |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {i} | {core.describe(c['change'])} | {c['scores']['total']:.3f} | {c['scores']['pronunciation']:.3f} | "
                  f"{c['metrics']['cer']:.3f} | {c['metrics']['text_score']:.3f} | {_laya(c)} |" for i, c in enumerate(rec["candidates"], 1)]
        (d / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def run(self, iterations, resume=False, category=None, test=None, stop_after=None):
        """Resumable: state 'next_iteration' advances only after an iteration is fully decided and logged."""
        start = self.db.state("next_iteration", 1)  # history is never overwritten: a plain run also continues numbering
        if resume and self.db.state("in_progress") == start:
            print(f"resuming interrupted iteration {start}", flush=True)
        out = []
        for it in range(start, start + iterations):
            if stop_after is not None and len(out) >= stop_after:
                break
            self.db.set_state("in_progress", it)
            r = self.iterate(it, category, test)
            if r is None:
                break
            self.db.set_state("next_iteration", it + 1)
            out.append(r)
            print(f"[{it:03d}] {r['test']}: {r['change']} {r['before']:.3f}->{r['after']:.3f} {r['decision']} {'; '.join(r['why'])}", flush=True)
        return out


def _laya(c):
    la = c.get("laya") or {}
    return "-" if not la or "error" in la else f"{la.get('quality')!s:.5} / {la.get('acceptable')} / {la.get('failure')}"


def _mean(xs):
    return sum(xs) / len(xs) if xs else None
