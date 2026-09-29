"""piper.train entry point for long CPU runs: swaps Piper's default callbacks (top-5 val_mel + top-5 val_mos, ~850 MB each)
for a single last.ckpt written every 500 optimizer steps (Lightning global_step counts BOTH GAN optimizers: 2 per batch)."""
from lightning.pytorch.callbacks import ModelCheckpoint
from piper.train import __main__ as m

m._DEFAULT_CALLBACKS[:] = [ModelCheckpoint(save_last=True, save_top_k=0, every_n_train_steps=100)]
if __name__ == "__main__":
    m.main()
