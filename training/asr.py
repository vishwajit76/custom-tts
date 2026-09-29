"""Whisper ASR for transcript validation (dataset prep) and intelligibility scoring (bench/quality.py).

Offline once the model is cached: HF_HUB_OFFLINE=1. Model: ASR_MODEL env, default openai/whisper-large-v3-turbo.
"""
import functools
import os
import re
import unicodedata

import numpy as np

ASR_MODEL = os.environ.get("ASR_MODEL", "openai/whisper-large-v3-turbo")


@functools.cache
def _pipe():
    import torch
    from transformers import pipeline

    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return pipeline("automatic-speech-recognition", model=ASR_MODEL, device=device,
                    torch_dtype=torch.float16 if device != "cpu" else torch.float32)


def transcribe(wav: np.ndarray, sr: int) -> str:
    if sr != 16000:
        import soxr

        wav = soxr.resample(wav, sr, 16000)
    out = _pipe()({"raw": wav.astype(np.float32), "sampling_rate": 16000}, generate_kwargs={"language": "hindi", "task": "transcribe"})
    return out["text"].strip()


def _clean(s: str) -> str:
    s = unicodedata.normalize("NFC", s.lower())
    s = s.replace("़", "")  # nukta: ASR and scripts disagree on ज़/ज
    return re.sub(r"[^\w]|_", "", s)  # drop spaces and punctuation; CER is about sounds, not spacing


def _edit_rate(r, h) -> float:
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(r)


def cer(ref: str, hyp: str) -> float:
    """Character error rate (Levenshtein / len(ref)) after dropping punctuation, spaces and nukta."""
    return _edit_rate(_clean(ref), _clean(hyp))


def phonemes(text: str) -> list[str]:
    """espeak-ng Hindi IPA, English runs in Indian-English IPA (so 'loan' and 'लोन' match), without stress,
    length marks and punctuation."""
    from app.services.indian_english import _ENGLISH_RUN, english_ipa, espeak

    out, pos = [], 0
    for m in [*_ENGLISH_RUN.finditer(text), None]:
        hindi = text[pos:m.start() if m else len(text)]
        out += [p for sent in espeak("hi", hindi) for p in sent] if hindi.strip() else []
        if m:
            out += list(english_ipa(m[0]))
            pos = m.end()
    return [p for p in out if p not in "ˈˌː ,.?!;:-।\"'"]


def per(ref: str, hyp: str) -> float:
    """Phoneme error rate: script-neutral, so ASR writing 'लोन' for 'loan' or '12' for 'बारह' (after
    normalizing hyp) costs little. Pass both sides through the text normalizer first."""
    return _edit_rate(phonemes(ref), phonemes(hyp))


if __name__ == "__main__":
    assert cer("नमस्ते, जी।", "नमस्ते जी") == 0.0
    assert abs(cer("abcd", "abxd") - 0.25) < 1e-9
    assert cer("ज़रूर", "जरूर") == 0.0
    assert per("आपका loan", "आपका लोन") < 0.3 and per("नमस्ते", "नमस्ते") == 0.0
    print("ok")
