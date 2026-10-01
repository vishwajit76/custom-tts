"""LR anneal regression test (CPU, seconds, no data, no checkpoint download).

Runs the EXACT callback source the Kaggle kernel embeds (WRAP_SRC in training/kaggle/train_kernel.py) inside a tiny manual-optimization
Lightning module that mimics piper's VitsModel (two optimizers, two ExponentialLR schedulers that are never stepped). Also pins the fact the
anneal depends on: piper-tts 1.8.0 never steps its schedulers. For the real-checkpoint check run `python -m training.lr_dryrun --ckpt <ckpt>`.
"""
import inspect
import math
import os
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pl = pytest.importorskip("lightning.pytorch")
pytest.importorskip("piper.train.__main__")

ROOT = Path(__file__).resolve().parent.parent
KERNEL = ROOT / "training/kaggle/train_kernel.py"


@pytest.fixture(scope="module")
def wrap(tmp_path_factory):
    """Exec the embedded wrapper (without running piper's CLI) and hand back its namespace."""
    from piper.train import __main__ as m

    saved = list(m._DEFAULT_CALLBACKS)
    src = re.search(r"WRAP_SRC = r'''(.*?)'''\n", KERNEL.read_text("utf-8"), re.S).group(1)
    os.environ.update(W_DIR=str(tmp_path_factory.mktemp("w")), LAST_CKPT="/nonexistent", DEADLINE="9999999999", CKPT_DIR="/nonexistent")
    ns = {"__name__": "wrap_under_test"}
    exec(compile(src, "wrap.py", "exec"), ns)
    m._DEFAULT_CALLBACKS[:] = saved
    return ns


class Tiny(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.g, self.d = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        self.automatic_optimization = False

    def configure_optimizers(self):
        og = torch.optim.AdamW(self.g.parameters(), lr=2e-4)
        od = torch.optim.AdamW(self.d.parameters(), lr=2e-4)
        return [og, od], [torch.optim.lr_scheduler.ExponentialLR(og, 0.999875), torch.optim.lr_scheduler.ExponentialLR(od, 0.999875)]

    def training_step(self, batch, _):
        og, od = self.optimizers()
        for o, m in ((og, self.g), (od, self.d)):  # like piper: two optimizer steps per batch, scheduler.step() never called
            o.zero_grad(); m(batch[0]).sum().backward(); o.step()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(8, 2)), batch_size=4)


class Rec(pl.Callback):
    def __init__(self):
        self.lrs = []

    def on_train_epoch_start(self, tr, _):  # runs after LrCtl (registered first) so it sees the applied value
        self.lrs.append((tr.current_epoch, [o.param_groups[0]["lr"] for o in tr.optimizers]))


def fit(wrap, tmp, env, epochs, ckpt=None):
    for k in ("LR_MODE", "LR_START", "LR_D_START", "LR_FINAL_RATIO", "ANNEAL_EPOCHS"):
        os.environ.pop(k, None)
    os.environ.update(env)
    ctl, rec = wrap["LrCtl"](), Rec()
    tr = pl.Trainer(max_epochs=epochs, accelerator="cpu", devices=1, callbacks=[ctl, rec], logger=False, enable_checkpointing=False,
                    enable_progress_bar=False, enable_model_summary=False)
    tr.fit(Tiny(), ckpt_path=ckpt)
    out = Path(tmp) / f"c{len(list(Path(tmp).glob('c*.ckpt')))}.ckpt"
    tr.save_checkpoint(out)
    return rec.lrs, out, ctl


ANNEAL = dict(LR_MODE="anneal", LR_START="1e-4", LR_FINAL_RATIO="0.05", ANNEAL_EPOCHS="4")


def want(k, n=4, start=1e-4, ratio=0.05):
    return start * ratio ** (min(k, n) / n)


def test_keep_mode_leaves_the_learning_rate_untouched(wrap, tmp_path):
    lrs, ck, _ = fit(wrap, tmp_path, {}, 3)
    assert [l[1] for l in lrs] == [[2e-4, 2e-4]] * 3
    lrs2, _, _ = fit(wrap, tmp_path, {}, 5, str(ck))  # resume: restored optimizer LR is kept, the scheduler is not stepped by anyone
    assert all(math.isclose(g, 2e-4) for _, l in lrs2 for g in l)


def test_anneal_overrides_restored_lr_decays_then_holds(wrap, tmp_path):
    _, ck, _ = fit(wrap, tmp_path, {}, 1)  # seed checkpoint with lr 2e-4 in the optimizer state
    lrs, _, _ = fit(wrap, tmp_path, ANNEAL, 8, str(ck))  # epochs 1..7
    for i, (ep, l) in enumerate(lrs):
        assert math.isclose(l[0], want(i), rel_tol=1e-9) and math.isclose(l[1], want(i), rel_tol=1e-9), (ep, l)
    assert math.isclose(lrs[-1][1][0], 5e-6, rel_tol=1e-9)  # held at LR_START * ratio
    assert lrs[0][1][0] < 2e-4  # the restored value was overridden


def test_resume_same_settings_continues_changed_settings_restart(wrap, tmp_path):
    _, seed, _ = fit(wrap, tmp_path, {}, 1)
    lrs_b, ck_b, _ = fit(wrap, tmp_path, ANNEAL, 4, str(seed))  # 3 epochs, e0 = 1
    first = lrs_b[0][0]
    lrs_c, _, ctl = fit(wrap, tmp_path, ANNEAL, 7, str(ck_b))  # same settings -> continues from epoch 4
    assert ctl.e0 == first
    assert math.isclose(lrs_c[0][1][0], want(lrs_c[0][0] - first), rel_tol=1e-9) and lrs_c[0][1][0] < lrs_b[-1][1][0] * 1.0001
    lrs_d, _, ctl2 = fit(wrap, tmp_path, dict(ANNEAL, ANNEAL_EPOCHS="10"), 6, str(ck_b))  # changed settings -> restart from LR_START
    assert ctl2.e0 == lrs_d[0][0] and math.isclose(lrs_d[0][1][0], 1e-4, rel_tol=1e-9)


def test_piper_never_steps_its_schedulers():
    """The anneal exists because of this. If a piper upgrade starts stepping the schedulers, LrCtl would fight it: review before upgrading."""
    from piper.train.vits import lightning as L

    src = inspect.getsource(L)
    assert "automatic_optimization = False" in src
    assert not re.search(r"lr_schedulers\(\)|lr_scheduler_step|sch\w*\.step\(\)|scheduler\w*\.step\(\)", src)
    assert "ExponentialLR" in src  # they are created (so the checkpoint carries gamma/last_epoch) but never advanced


def test_kernel_defaults_anneal_to_five_percent():
    src = KERNEL.read_text("utf-8")
    assert 'LR_MODE="anneal"' in src and 'LR_FINAL_RATIO="0.05"' in src and 'LR_START="1e-4"' in src
