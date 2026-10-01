"""Fine-tune a Piper (VITS) voice. Resumes automatically from the run's last checkpoint.

  .venv-train/bin/python -m training.train --data data/myvoice --run runs/myvoice --init rohan --epochs 200

--init rohan              fine-tune from the Piper hi_IN rohan checkpoint (Hindi; fastest; its data license is unresolved,
                          see docs/research.md)
--init <path.ckpt>        any Piper medium checkpoint with the same speaker count as your data
--warmstart-vocoder base  commercially clean start: copy only the vocoder from Piper's _base_model (LibriTTS-R, CC BY 4.0);
                          text encoder trains from scratch, so this needs several hours of audio and many more epochs

Also: --seed N (default 1234, -> Lightning --seed_everything), --precision 16-mixed (AMP), --dry-run.
Refuses to run on a held-out test set (training.split.assert_not_heldout).
Gradient accumulation: NOT available. Piper's VITS trainer uses manual optimization (GAN, two optimizers), and Lightning
raises MisconfigurationException for --trainer.accumulate_grad_batches (verified with piper-tts 1.8.0 + lightning 2.x,
2026-09-29); this script rejects it up front. Raise --batch-size instead.
Multi-speaker: rows file|speaker|text set --model.num_speakers; Piper writes speaker_id_map into config.json.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from training.split import assert_not_heldout

CKPTS = {  # rhasspy/piper-checkpoints (dataset repo)
    "rohan": "hi/hi_IN/rohan/medium/epoch=3190-step=309852.ckpt",
    "base": "_base_model/base_model.ckpt",
}


def resolve(ckpt: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download("rhasspy/piper-checkpoints", CKPTS[ckpt], repo_type="dataset") if ckpt in CKPTS else ckpt


def last_checkpoint(run: Path) -> Path | None:
    found = sorted(run.glob("lightning_logs/version_*/checkpoints/last.ckpt"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def epoch_of(ckpt: str) -> int:
    import torch

    # Lightning checkpoints hold pickled hyper-parameters, so weights_only=True fails; Lightning itself unpickles
    # the same file on resume. Only point --init/--ckpt at checkpoints you trust (official repo or your own runs).
    # a checkpoint saved during epoch N resumes at N + 1
    return int(torch.load(ckpt, map_location="cpu", weights_only=False).get("epoch", -1)) + 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="output dir of training.prepare_dataset")
    p.add_argument("--run", type=Path, required=True, help="checkpoints, logs and config.json go here")
    p.add_argument("--init", default="rohan", help="rohan | base | path to .ckpt (ignored when resuming)")
    p.add_argument("--warmstart-vocoder", help="base | path: train from scratch with a pretrained vocoder instead of --init")
    p.add_argument("--epochs", type=int, default=200, help="epochs to train in this invocation")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accelerator", default="auto", help="auto | gpu | mps | cpu")
    p.add_argument("--precision", default="32-true", help="e.g. 16-mixed on NVIDIA GPUs")
    p.add_argument("--name", default="custom")
    p.add_argument("--seed", type=int, default=1234, help="passed as --seed_everything (deterministic seeding of python/numpy/torch)")
    p.add_argument("--dry-run", action="store_true", help="print the piper.train command and exit (no downloads, no training)")
    a, extra = p.parse_known_args()  # anything else goes straight to `piper.train fit`

    if any(x.split("=")[0] == "--trainer.accumulate_grad_batches" for x in extra):
        p.error("--trainer.accumulate_grad_batches is rejected by Piper's manual-optimization VITS trainer; raise --batch-size")
    assert_not_heldout(a.data)  # never train on a held-out test set
    assert_not_heldout(a.data / "metadata.csv")
    rows = [r for r in (a.data / "metadata.csv").read_text("utf-8").splitlines() if r.strip()]
    speakers = len({r.split("|")[1] for r in rows}) if rows[0].count("|") == 2 else 1
    a.run.mkdir(parents=True, exist_ok=True)

    resume = last_checkpoint(a.run)
    if a.dry_run:
        ckpt = str(resume) if resume else None if a.warmstart_vocoder else f"<{a.init}>"
        start = 0
    else:
        ckpt = str(resume) if resume else None if a.warmstart_vocoder else resolve(a.init)
        start = epoch_of(ckpt) if ckpt else 0
    cmd = [
        sys.executable, "-m", os.environ.get("PIPER_TRAIN_MODULE", "piper.train"), "fit", "--seed_everything", str(a.seed),
        "--data.voice_name", a.name, "--data.csv_path", str(a.data / "metadata.csv"), "--data.audio_dir", str(a.data / "wavs"),
        "--data.espeak_voice", "hi", "--data.cache_dir", str(a.run / "cache"), "--data.config_path", str(a.run / "config.json"),
        "--data.batch_size", str(a.batch_size), "--model.sample_rate", "22050", "--model.num_speakers", str(speakers),
        "--trainer.default_root_dir", str(a.run), "--trainer.accelerator", a.accelerator, "--trainer.devices", "1",
        "--trainer.precision", a.precision, "--trainer.max_epochs", str(start + a.epochs),
    ]
    if ckpt:
        cmd += ["--ckpt_path", ckpt]
    if a.warmstart_vocoder and not resume:
        cmd += ["--model.vocoder_warmstart_ckpt", a.warmstart_vocoder if a.dry_run else resolve(a.warmstart_vocoder)]
    cmd += extra
    print(("resuming from " if resume else "starting from ") + str(ckpt or "scratch + warm vocoder"), f"at epoch {start}", flush=True)
    if a.dry_run:
        print(" ".join(cmd))
        return
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
