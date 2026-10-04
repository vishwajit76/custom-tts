"""Run AFTER the TTS benchmarks. Whisper large-v3-turbo (fp16, GPU) transcribes samples/<engine>/*.wav; CER vs corpus text (rough proxy)."""
import re, unicodedata
from common import *
import soundfile as sf, soxr
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", dtype=torch.float16, device="cuda")
text_of = {r["id"]: r["text"] for r in corpus(258)}


def norm(s):
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"[^\w]|_", "", s)  # drop punctuation + whitespace


def lev(a, b):
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        c = [i]
        for j, y in enumerate(b, 1):
            c.append(min(p[j] + 1, c[-1] + 1, p[j - 1] + (x != y)))
        p = c
    return p[-1]


res = {}
for d in sorted((HERE / "samples").iterdir()):
    per = []
    for w in sorted(d.glob("*.wav")):
        a, sr = sf.read(w, dtype="float32")
        a = soxr.resample(a, sr, 16000) if sr != 16000 else a
        hyp = pipe({"raw": a, "sampling_rate": 16000}, generate_kwargs={"language": "hi", "task": "transcribe"})["text"]
        ref = text_of[w.stem]
        per.append({"id": w.stem, "cer": round(lev(norm(ref), norm(hyp)) / max(1, len(norm(ref))), 4), "hyp": hyp})
    if per:
        res[d.name] = {"cer_mean": round(float(np.mean([p["cer"] for p in per])), 4), "n": len(per), "per": per}
        print(d.name, res[d.name]["cer_mean"], len(per), flush=True)
(HERE / "results" / "asr_cer.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print("max_alloc_mib", torch.cuda.max_memory_allocated() / 2**20)
