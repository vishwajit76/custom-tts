"""Drop-in for resemble-ai/monotonic_align (1-step topology only), used by StyleTTS2 utils.py."""
import numpy as np
import torch
from .core import maximum_path_c


def mask_from_len(lens, max_len=None):
    if max_len is None:
        max_len = lens.max()
    return torch.arange(max_len).to(lens).view(1, -1) < lens.unsqueeze(1)


def mask_from_lens(similarity, symbol_lens, mel_lens):
    _, S, T = similarity.size()
    return (mask_from_len(symbol_lens, S).unsqueeze(2) * mask_from_len(mel_lens, T).unsqueeze(1)).to(similarity)


def maximum_path(value, mask=None):
    if mask is None:
        mask = torch.ones_like(value)
    device, dtype = value.device, value.dtype
    v = np.ascontiguousarray((value * mask).data.cpu().numpy().astype(np.float32))
    m = mask.data.cpu().numpy()
    path = np.zeros_like(v).astype(np.int32)
    maximum_path_c(path, v, m.sum(1)[:, 0].astype(np.int32), m.sum(2)[:, 0].astype(np.int32))
    return torch.from_numpy(path).to(device=device, dtype=dtype)
