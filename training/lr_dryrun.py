"""CPU dry-run of the Kaggle trainer wrapper (LR control) on a tiny dataset. Uses the EXACT wrapper source embedded in
training/kaggle/train_kernel.py (WRAP_SRC), resumes a real Piper checkpoint on CPU with batch 2 and reports the learning rate per epoch.

  python -m training.lr_dryrun --ckpt /path/to/last.ckpt [--clips 24] [--epochs 3] [--work /tmp/lr_dryrun]

Checks: (A) default (LR_MODE unset) keeps the checkpoint LR untouched, (B) LR_MODE=anneal overrides the restored LR and decays it,
(C) resuming from B's checkpoint with the same settings CONTINUES the anneal, (D) changed settings restart it. Exit 1 on failure.
Data: first N rows of data/hi_f/metadata.csv; missing wavs are fetched from the private HF dataset repo (needs the HF token). Nothing is uploaded.
"""
import argparse
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "vishwajit76/custom-tts-hindi-train"


def wrap_src() -> str:
    src = (ROOT / "training/kaggle/train_kernel.py").read_text(encoding="utf8")
    return re.search(r"WRAP_SRC = r'''(.*?)'''\n", src, re.S).group(1)


def tiny_data(work: Path, n: int) -> Path:
    d = work / "data"
    if (d / "metadata.csv").exists():
        return d
    from huggingface_hub import hf_hub_download

    (d / "wavs").mkdir(parents=True, exist_ok=True)
    rows = [r for r in (ROOT / "data/hi_f/metadata.csv").read_text("utf-8").splitlines() if r.strip()][:n]
    for r in rows:
        shutil.copy(hf_hub_download(REPO, f"data/hi_f/wavs/{r.split('|')[0]}"), d / "wavs" / r.split("|")[0])
    (d / "metadata.csv").write_text("\n".join(rows) + "\n", "utf-8")
    return d


def run(work: Path, tag: str, ckpt: Path, data: Path, epochs: int, env_extra: dict, first_epoch: int) -> dict:
    run_dir = work / tag
    shutil.rmtree(run_dir, ignore_errors=True)
    ck_dir = run_dir / "ckpt"  # same layout as the kernel: seed elsewhere, new checkpoints in their own dir
    ck_dir.mkdir(parents=True)
    (run_dir / "ms").mkdir()
    cfg = work / "config.json"
    if not cfg.exists():
        from huggingface_hub import hf_hub_download

        shutil.copy(hf_hub_download(REPO, "runs/hi_f/config.json"), cfg)
    shutil.copy(cfg, run_dir / "config.json")
    (run_dir / "wrap.py").write_text(wrap_src())
    cmd = [sys.executable, str(run_dir / "wrap.py"), "fit", "--seed_everything", "1234", "--data.voice_name", "hi_f",
           "--data.csv_path", str(data / "metadata.csv"), "--data.audio_dir", str(data / "wavs"), "--data.espeak_voice", "hi",
           "--data.cache_dir", str(work / "cache"), "--data.config_path", str(run_dir / "config.json"), "--data.batch_size", "2",
           "--data.num_workers", "0", "--model.sample_rate", "22050", "--model.num_speakers", "1", "--trainer.default_root_dir", str(run_dir),
           "--trainer.accelerator", "cpu", "--trainer.devices", "1", "--trainer.max_epochs", str(first_epoch + epochs),
           "--trainer.log_every_n_steps", "50", "--trainer.enable_progress_bar", "false", "--ckpt_path", str(ckpt)]  # resumed in place (no copy: checkpoints are 850 MB); the run writes its own last.ckpt
    env = dict(os.environ, W_DIR=str(run_dir), LAST_CKPT=str(ck_dir / "last.ckpt"), CKPT_DIR=str(ck_dir), DEADLINE="9999999999", CKPT_EVERY="4", MS_EVERY="100000000",
               PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="", **env_extra)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=ROOT)
    (work / f"{tag}.log").write_text(p.stdout + p.stderr)
    if p.returncode:
        print((p.stdout + p.stderr)[-3000:])
        sys.exit(f"{tag}: trainer failed rc={p.returncode}")
    out = p.stdout + p.stderr
    if tag not in ("B_anneal",):  # keep only B's checkpoint (C and D resume from it); free disk
        (ck_dir / "last.ckpt").unlink(missing_ok=True)
    return {"epochs": [(int(m[1]), int(m[2]), eval(m[3])) for m in re.finditer(r"^epoch (\d+) step (\d+) lr (\[.*?\])", out, re.M)],
            "start": re.findall(r"^LR at train start.*$", out, re.M), "anneal": re.findall(r"^LR anneal .*$", out, re.M),
            "ckpt": ck_dir / "last.ckpt"}  # noqa


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--clips", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--work", type=Path, default=Path("/tmp/lr_dryrun"))
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    import torch

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    lr0, e0 = ck["optimizer_states"][0]["param_groups"][0]["lr"], int(ck["epoch"])
    del ck
    data = tiny_data(a.work, a.clips)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    A = run(a.work, "A_default", a.ckpt, data, a.epochs, {}, e0)
    lrs = [l[0] for _, _, l in A["epochs"]]
    print("A default lrs per epoch:", lrs, A["start"])
    check("A: default keeps the checkpoint LR constant (piper 1.8.0 never steps the scheduler)", lrs and all(math.isclose(x, lr0, rel_tol=1e-9) for x in lrs), f"ckpt lr {lr0:.6g}")

    env = dict(LR_MODE="anneal", LR_START="1e-4", LR_FINAL_RATIO="0.05", ANNEAL_EPOCHS="4")
    B = run(a.work, "B_anneal", a.ckpt, data, a.epochs, env, e0)
    b = [l[0] for _, _, l in B["epochs"]]
    first = B["epochs"][0][0]
    want = [1e-4 * 0.05 ** (min(k, 4) / 4) for k in range(len(b))]
    print("B anneal lrs per epoch:", b, "expected", want, B["anneal"])
    check("B: anneal starts at LR_START, overriding the restored LR, and decays", b and all(math.isclose(x, w, rel_tol=1e-6) for x, w in zip(b, want)) and b[0] < lr0)
    check("B: discriminator LR follows", all(math.isclose(l[1], w, rel_tol=1e-6) for (_, _, l), w in zip(B["epochs"], want)))
    check("B: log says STARTS", any("STARTS" in x for x in B["anneal"]))

    C = run(a.work, "C_continue", B["ckpt"], data, a.epochs, env, first + len(b))
    c = [l[0] for _, _, l in C["epochs"]]
    print("C resume-same-settings lrs per epoch:", c, C["anneal"])
    check("C: resume with same settings continues the anneal (not restarted) and holds the final LR",
          any("CONTINUES" in x for x in C["anneal"]) and c and c[0] <= b[-1] * (1 + 1e-6) and math.isclose(c[-1], 1e-4 * 0.05, rel_tol=1e-6),
          f"first epoch lr {c[0] if c else None}, last {c[-1] if c else None}")

    D = run(a.work, "D_restart", B["ckpt"], data, 1, dict(env, ANNEAL_EPOCHS="10"), first + len(b))
    d = [l[0] for _, _, l in D["epochs"]]
    check("D: changed settings restart the anneal from LR_START", any("STARTS" in x for x in D["anneal"]) and d and math.isclose(d[0], 1e-4, rel_tol=1e-6), f"lr {d}")
    print("ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
