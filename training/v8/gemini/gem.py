"""Shared Gemini helper: disk cache by hash, backoff, JSON parse, usage/cost tally. Key loaded in-process only."""
import hashlib, json, os, re, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "training/v8/gemini/.cache"   # gitignored-style local cache
FLASH = "gemini-3.5-flash"        # bulk text
PRO = "gemini-3.1-pro-preview"    # audio judging
# USD per 1M tokens (in, out) -- assumed list prices; estimate only
PRICE = {FLASH: (0.30, 2.50), PRO: (2.00, 12.00)}
USAGE = {"in": 0, "out": 0, "usd": 0.0, "calls": 0}


def parse_json(text):
    """Extract first JSON object/array from model text (handles ``` fences)."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", t, re.S)
        if not m:
            raise
        return json.loads(m.group(1))


def key_for(model, prompt, extra=""):
    return hashlib.sha256(f"{model}\0{prompt}\0{extra}".encode()).hexdigest()


def cached(model, prompt, extra, fn, cache_dir=CACHE):
    """Return parsed JSON from cache or fn() (which returns raw text)."""
    p = Path(cache_dir) / f"{key_for(model, prompt, extra)}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    out = parse_json(fn())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def _client():
    from dotenv import load_dotenv
    from google import genai
    load_dotenv(ROOT / ".env")
    return genai.Client(api_key=os.environ["PLATFORM_GEMINI_API_KEY"])


_C = None
_LOCK = __import__('threading').Lock()


def call(model, prompt, parts=(), tries=6):
    """Raw text from Gemini with exponential backoff on 429/5xx."""
    global _C
    from google.genai import types
    with _LOCK:
        _C = _C or _client()
    for i in range(tries):
        try:
            r = _C.models.generate_content(
                model=model, contents=[*parts, prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.4))
            u = r.usage_metadata
            ti, to = (u.prompt_token_count or 0), ((u.candidates_token_count or 0) + (u.thoughts_token_count or 0))
            pi, po = PRICE[model]
            USAGE["in"] += ti; USAGE["out"] += to; USAGE["calls"] += 1
            USAGE["usd"] += (ti * pi + to * po) / 1e6
            return r.text
        except Exception as e:  # noqa
            s = str(e)
            if i == tries - 1 or not re.search(r"429|500|502|503|504|RESOURCE_EXHAUSTED|UNAVAILABLE|overloaded|disconnected|Timeout|timed out|ConnectError|RemoteProtocol", s):
                raise
            time.sleep(min(60, 2 ** i * 2))


def ask_json(model, prompt, parts=(), extra=""):
    return cached(model, prompt, extra, lambda: call(model, prompt, parts))
