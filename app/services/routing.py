"""Capability- and latency-aware engine selection (`routing_policy`).

Policies (only consulted when the caller did not pick a voice, i.e. voice == "default", and gave no reference/speaker):
  fast        lowest latency tier among engines that can honour every requested control
  balanced    prefers the "balanced" tier, then fast, then slow (quality-first engines that still stream in real time on CPU)
  expressive  requires an engine with native/steered emotion, style or role; never satisfied by DSP
  clone       requires an engine with zero-shot cloning

Latency tiers are configuration (settings.engine_latency_tiers), taken from measurements in docs/benchmarks.md, not
guessed by this code: fast = real-time at several concurrent streams on CPU, balanced = about 2 real-time streams per
4 cores, slow = needs GPU or is not real-time. An engine with no tier entry counts as "slow".

An incapable engine is never chosen silently: with fallback="reject" the request fails (422); with fallback="ignore" it is
routed as `balanced` and the caller is told ("routing_policy" is listed as an ignored control, along with whatever the
engine then drops). Explicit voice / speaker_id / reference always wins over the policy.
"""
from dataclasses import dataclass

from app.services.conditioning import DEFAULT_CAPABILITIES, EngineCapabilities, UnsupportedControl, VoiceCondition, validate_condition

TIER_ORDER = {"fast": 0, "balanced": 1, "slow": 2}
POLICIES = ("fast", "balanced", "expressive", "clone")


class NoCapableEngine(Exception):
    def __init__(self, policy: str, missing: list[str]) -> None:
        self.policy, self.missing = policy, missing
        super().__init__(f"no loaded engine satisfies routing_policy={policy!r} with the requested controls"
                         + (f" ({', '.join(missing)})" if missing else "") + "; use fallback=\"ignore\" to route without them")


@dataclass
class Route:
    engine: str
    voice: str
    policy: str
    degraded: bool = False  # policy could not be met and fallback="ignore" routed it as balanced


def parse_tiers(spec: str) -> dict[str, str]:
    out = {}
    for item in spec.split(","):
        if ":" in item:
            name, tier = (x.strip() for x in item.split(":", 1))
            if tier not in TIER_ORDER:
                raise ValueError(f"unknown latency tier {tier!r} for {name!r}; use fast|balanced|slow")
            out[name] = tier
    return out


def _tier(name: str, tiers: dict[str, str]) -> int:
    return TIER_ORDER[tiers.get(name, "slow")]


def _capable(cond: VoiceCondition, caps: EngineCapabilities, dsp: bool) -> list[str]:
    """Controls in `cond` this engine cannot honour ([] = fully capable). speaker_id/reference are resolved before routing."""
    probe = cond.model_copy(update={"fallback": "reject", "speaker_id": None, "reference_audio": None, "reference_text": None})
    try:
        validate_condition(probe, caps, dsp=dsp)
    except UnsupportedControl as e:
        return e.controls
    return []


def _default_voice(engine, default_voice: str) -> str | None:
    voices = [v["voice_id"] for v in engine.voices()]
    if not voices:
        return None
    return default_voice if default_voice in voices else voices[0]


def route(policy: str, cond: VoiceCondition | None, engines: list[tuple[str, object]], tiers: dict[str, str],
          default_voice: str, dsp: bool = False) -> Route:
    """Pick (engine, voice) for `policy`. `engines` = [(name, engine)]. Raises NoCapableEngine."""
    cond = cond or VoiceCondition()
    fallback_ok = cond.fallback == "ignore"
    rows = []
    for name, e in engines:
        caps = getattr(e, "capabilities", DEFAULT_CAPABILITIES)
        voice = _default_voice(e, default_voice)
        if voice is not None:
            rows.append((name, caps, voice, _capable(cond, caps, dsp)))

    def pick(order: list[str], ok) -> Route | None:
        cands = [(n, c, v, miss) for n, c, v, miss in rows if ok(c, miss)]
        if not cands:
            return None
        cands.sort(key=lambda r: (order.index(tiers.get(r[0], "slow")), r[0]))
        n, _, v, _ = cands[0]
        return Route(n, v, policy)

    fast_order, bal_order = ["fast", "balanced", "slow"], ["balanced", "fast", "slow"]
    if policy == "fast":
        r = pick(fast_order, lambda c, miss: not miss)
        req_missing = sorted({m for _, _, _, miss in rows for m in miss})
    elif policy == "balanced":
        r = pick(bal_order, lambda c, miss: not miss)
        req_missing = sorted({m for _, _, _, miss in rows for m in miss})
    elif policy == "expressive":
        r = pick(bal_order, lambda c, miss: not miss and (c.native_emotion or c.native_style or c.role or c.prompt_emotion))
        req_missing = ["expressive_engine"]
    elif policy == "clone":
        r = pick(bal_order, lambda c, miss: not miss and c.cloning)
        req_missing = ["cloning_engine"]
    else:
        raise ValueError(f"unknown routing_policy {policy!r}")
    if r is not None:
        return r
    if not fallback_ok:
        raise NoCapableEngine(policy, req_missing)
    # explicit fallback: best-latency-balanced engine regardless of controls (the caller sees what was dropped)
    r = pick(bal_order, lambda c, miss: True)
    if r is None:
        raise NoCapableEngine(policy, ["no engine with a voice"])
    r.degraded = True
    return r
