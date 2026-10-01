"""Generate listening previews ONLY for (voice x control) combinations the serving engine really supports.

Usage: [ENGINES=...] [DSP_PROSODY=1] python -m bench.previews [--out bench/results/previews] [--voices a,b] [--speakers id1,id2]
Every file is named and labelled by the control AND how it is realised (native | steered | dsp). Nothing is generated for
emotion/style/role unless an engine declares them; DSP pitch/energy is labelled as such and never as emotion. A speaker id
goes through the consent-checked registry path (condition.speaker_id); combos it cannot serve are skipped and listed.
manifest.json lists what was generated and what was skipped and why. These are listening aids, not a quality score.
"""
import argparse
import json
from pathlib import Path

from app.services.conditioning import EngineCapabilities, VoiceCondition, control_kinds, validate_condition

TEXT = "नमस्ते, मैं आपकी मदद के लिए यहाँ हूँ। आपका ऑर्डर कल शाम तक पहुँच जाएगा।"


def supported_combos(caps: EngineCapabilities, dsp: bool = False) -> list[dict]:
    """[{name, condition, kinds}] for neutral + every single control the engine honours (checked with validate_condition)."""
    candidates = [("neutral", {}), ("speed-0.85", {"speed": 0.85}), ("speed-1.15", {"speed": 1.15}),
                  ("pitch+2", {"pitch": 2.0}), ("pitch-2", {"pitch": -2.0}), ("energy-1.3", {"energy": 1.3}), ("energy-0.7", {"energy": 0.7})]
    candidates += [(f"emotion-{e}", {"emotion": e}) for e in ("happy", "sad", "calm", "empathetic", "apologetic")]
    candidates += [(f"style-{s}", {"style": s}) for s in ("conversational", "warm", "narration")]
    candidates += [(f"role-{r}", {"role": r}) for r in ("customer_support", "narrator")]
    kinds = control_kinds(caps, dsp)
    out = []
    for name, c in candidates:
        try:
            _, applied, _ = validate_condition(VoiceCondition(**c, fallback="reject"), caps, dsp=dsp)
        except ValueError:
            continue
        key = next(iter(c), None)
        out.append({"name": name, "condition": c or None, "applied": applied, "kind": "speed" if key == "speed" else kinds.get(key, "n/a")})
    return out


def label(voice: str, combo: dict) -> str:
    kind = f" [{combo['kind']}]" if combo["kind"] not in ("n/a",) and combo["name"] != "neutral" else ""
    return f"{voice} | {combo['name']}{kind}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "previews")
    ap.add_argument("--voices", default="")
    ap.add_argument("--speakers", default="", help="registry speaker ids (need consent for tts)")
    ap.add_argument("--text", default=TEXT)
    a = ap.parse_args()
    import soundfile as sf

    from app.core.config import settings
    from app.services import tts
    from bench.eval import default_synth

    tts.load()
    a.out.mkdir(parents=True, exist_ok=True)
    synth = default_synth()
    targets = [("voice", v) for v in (a.voices.split(",") if a.voices else [x["voice_id"] for x in tts.engine.voices()])]
    targets += [("speaker", s) for s in a.speakers.split(",") if s]
    manifest = {"disclaimer": "Listening aids only. Labels state the mechanism (native/steered/dsp); DSP is not emotion.",
                "dsp_prosody": settings.dsp_prosody, "generated": [], "skipped": []}
    for kind, ident in targets:
        voice = ident if kind == "voice" else "default"
        caps = tts.capabilities_for(tts.resolve_voice(voice))
        for combo in supported_combos(caps, settings.dsp_prosody):
            cond = dict(combo["condition"] or {})
            if kind == "speaker":
                cond["speaker_id"] = ident
            try:
                wav, sr, applied, ignored = synth(a.text, voice, cond or None)
            except ValueError as e:
                manifest["skipped"].append({"target": ident, "combo": combo["name"], "reason": str(e)[:200]})
                continue
            fn = f"{ident.replace(':', '_')}__{combo['name']}.wav".replace("+", "p")
            sf.write(a.out / fn, wav, sr)
            manifest["generated"].append({"file": fn, "label": label(ident, combo), "applied": applied, "ignored": ignored})
    (a.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(manifest['generated'])} previews, {len(manifest['skipped'])} skipped -> {a.out}")


if __name__ == "__main__":
    main()
