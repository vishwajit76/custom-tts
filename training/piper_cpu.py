"""piper.train entry point for long CPU runs: swaps Piper's default callbacks (top-5 val_mel + top-5 val_mos, ~850 MB each)
for a single last.ckpt written every 500 optimizer steps (Lightning global_step counts BOTH GAN optimizers: 2 per batch)."""
import os
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from piper.train import __main__ as m

_T = int(os.environ.get("PIPER_THREADS", os.cpu_count() or 4))
torch.set_num_threads(_T)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
m._DEFAULT_CALLBACKS[:] = [ModelCheckpoint(save_last=True, save_top_k=0, every_n_train_steps=100)]
if __name__ == "__main__":
    m.main()
