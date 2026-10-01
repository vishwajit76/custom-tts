"""Same-speaker vs different-speaker cosine + embed latency for the speaker-encoder backends.

Usage: ENGINES=piper,kokoro,supertonic python -m bench.spk_sim --voices v1,v2,... [--backends mfcc,resemblyzer] [--label spk-sim]
Each voice speaks the same N sentences; 'same' = one voice, different sentences; 'diff' = different voices, same sentence.
"""
import argparse
import itertools
import json
import statistics as st
import time
from pathlib import Path

import numpy as np

from app.services import tts
from app.services.speaker_encoder import SpeakerEncoder, to_16k
from app.services.text_normalizer import normalize

HERE = Path(__file__).parent
SENT = ["नमस्ते, मैं आपकी बैंक से बात कर रही हूँ, क्या आप अभी दो मिनट बात कर सकते हैं?",
        "हमारी टीम कल सुबह आपके घर पर डिलीवरी के लिए आएगी, कृपया घर पर रहिएगा।",
        "आपका भुगतान सफलतापूर्वक हो गया है, धन्यवाद, आपका दिन शुभ हो।",
        "मैं समझ सकती हूँ, कोई बात नहीं, मैं आपको शाम को दोबारा फ़ोन करूँगी।"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--voices", required=True)
    p.add_argument("--backends", default="mfcc,resemblyzer")
    p.add_argument("--label", default="spk-sim")
    a = p.parse_args()
    tts.engine.load()
    voices = a.voices.split(",")
    audio = {v: [(tts.engine.synth(normalize(s), v, 1.0), tts.engine.sample_rate(v)) for s in SENT] for v in voices}
    rep = {"label": a.label, "voices": voices, "n_sentences": len(SENT), "backends": {}}
    for b in a.backends.split(","):
        enc = SpeakerEncoder(b)
        enc.embed(*audio[voices[0]][0])  # load
        lat, E = [], {}
        for v in voices:
            for i, (w, sr) in enumerate(audio[v]):
                t = time.perf_counter()
                E[v, i] = enc._backend.embed(to_16k(w, sr))
                lat.append((time.perf_counter() - t) * 1000)
                E[v, i] = E[v, i] / np.linalg.norm(E[v, i])
        same = [float(E[v, i] @ E[v, j]) for v in voices for i, j in itertools.combinations(range(len(SENT)), 2)]
        diff = [float(E[v, i] @ E[w, i]) for v, w in itertools.combinations(voices, 2) for i in range(len(SENT))]
        pair = {f"{v}|{w}": round(st.mean(float(E[v, i] @ E[w, i]) for i in range(len(SENT))), 3) for v, w in itertools.combinations(voices, 2)}
        rep["backends"][b] = {"neural": enc.neural, "embed_ms_median": round(st.median(lat), 1), "embed_ms_max": round(max(lat), 1),
                              "audio_s_median": round(st.median(len(w) / sr for v in voices for w, sr in audio[v]), 2),
                              "same_mean": round(st.mean(same), 3), "same_min": round(min(same), 3), "n_same": len(same),
                              "diff_mean": round(st.mean(diff), 3), "diff_max": round(max(diff), 3), "n_diff": len(diff), "pairs": pair}
        print(b, json.dumps({k: v for k, v in rep["backends"][b].items() if k != "pairs"}))
    (HERE / "results" / f"{a.label}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    main()
