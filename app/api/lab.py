"""Local test lab on the demo page: synth with a visible normalized text, A/B baseline vs V8, optimizer 'what improved'.

All routes are loopback-only (no API key). V8 = the optimizer's frontend (pron_dict.json + active_config.json, re-read
when their mtime changes, so accepted changes apply without a restart). Files under reports/optimizer/iterations are
served read-only through a fixed name whitelist.
"""
import asyncio
import io
import json
import subprocess
import wave
from pathlib import Path
from urllib.parse import quote

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from starlette.requests import HTTPConnection

from app.services import text_normalizer, tts

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/optimizer"
BENCH = ROOT / "tests/pronunciation_bench"


def _local(conn: HTTPConnection) -> None:
    host = conn.client.host if conn.client else ""
    if host not in ("127.0.0.1", "::1") or "x-forwarded-for" in conn.headers:
        raise HTTPException(404, "Not found")


router = APIRouter(prefix="/demo/lab", dependencies=[Depends(_local)])
_json_cache: dict[Path, tuple[float, object]] = {}


def _jload(p: Path, default):
    """json file re-read when its mtime changes; missing/half-written file -> last good value or default."""
    try:
        m = p.stat().st_mtime
        if p not in _json_cache or _json_cache[p][0] != m:
            _json_cache[p] = (m, json.loads(p.read_text("utf-8")))
        return _json_cache[p][1]
    except (OSError, ValueError):
        return _json_cache[p][1] if p in _json_cache else default


def normalize(text: str, mode: str) -> str:
    if mode == "baseline":  # V7: normalizer as shipped, no pronunciation rule groups
        return text_normalizer.normalize(text, rules="off")
    try:
        from optimizer import core
    except Exception:  # optimizer not there / mid-edit: app rules only
        return text_normalizer.normalize(text, rules="all")
    active = {**core.DEFAULT_ACTIVE, **_jload(core.ACTIVE, {})}
    pron = _jload(core.PRON, {})
    norm = text_normalizer.normalize(core.apply_dict(core.apply_rules(text, active["rules"]), pron), rules=active["pron_rules"])
    return core.punct(norm, core.settings_for(active, "")["punct"])


class Speak(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    voice: str = Field(max_length=128)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    mode: str = Field("v8", pattern="^(baseline|v8)$")


_lock = asyncio.Lock()  # one synth at a time: keeps the shared GPU/CPU use small


def _synth(norm: str, voice: str, speed: float) -> bytes:
    sr = tts.engine.sample_rate(voice)
    gap = np.zeros(int(sr * 0.12), np.float32)
    parts = []
    for c in tts.split_for_stream(norm):
        parts += [np.asarray(tts.engine.synth(c, voice, speed), np.float32), gap]
    pcm = (np.clip(np.concatenate(parts) if parts else gap, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr), w.writeframes(pcm.tobytes())
    return buf.getvalue()


@router.post("/speak")
async def speak(req: Speak):
    if not tts.engine.ready or not tts.engine.has_voice(req.voice):
        raise HTTPException(404, "voice not available")
    norm = normalize(req.text, req.mode)
    async with _lock:
        wav = await asyncio.to_thread(_synth, norm, req.voice, req.speed)
    return Response(wav, media_type="audio/wav", headers={"X-Normalized": quote(norm), "Access-Control-Expose-Headers": "X-Normalized"})


@router.get("/info")
def info():
    presets: dict[str, list[str]] = {}
    for f in sorted(BENCH.glob("*/cases.json")):
        try:
            cases = json.loads(f.read_text("utf-8"))
        except ValueError:
            continue
        presets[f.parent.name] = [c["text"] for c in cases[:5] if "text" in c]
    return {"voices": [{"id": v["voice_id"], "name": v.get("name", v["voice_id"]), "engine": v.get("engine", "")} for v in tts.engine.voices()],
            "presets": presets}


def _git(*a: str) -> list[str]:
    try:
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=10)
        return r.stdout.splitlines() if r.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        return []


@router.get("/improved")
def improved():
    rep = _jload(REPORTS / "report.json", None)
    exps = []
    for e in (rep or {}).get("experiments", []):
        d = REPORTS / "iterations" / f"{int(e.get('iteration', 0)):03d}"
        e = {**e, "audio": {n: f"/demo/lab/file/{d.name}/{n}.wav" for n in ("before", "after") if (d / f"{n}.wav").is_file()}}
        exps.append(e)
    return {"generated": (rep or {}).get("generated"),
            "engines": {k: {"baseline": v.get("baseline"), "current": v.get("current")} for k, v in (rep or {}).get("engines", {}).items()},
            "experiments": exps, "commits": _git("log", "v8-optimizer", "--oneline", "-20"),
            "docs": _git("log", "-8", "--format=%h %ad %s", "--date=short", "--", "docs")}


@router.get("/file/{it}/{name}")
def file(it: str, name: str):
    if not (it.isdigit() and len(it) == 3 and name in ("before.wav", "after.wav")):
        raise HTTPException(404, "Not found")
    p = REPORTS / "iterations" / it / name
    if not p.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(p, media_type="audio/wav")
