"""CPU smoke test of the WHOLE Kaggle kernel script (training/kaggle/train_kernel.py) against a local directory instead of Hugging Face.

  python -m training.kernel_smoke --ckpt /path/to/a/real/ckpt.ckpt [--work /tmp/kernel_smoke] [--clips 24] [--seconds 150]

Nothing is uploaded and no network is needed except to fetch the tiny wav subset from the private HF repo (first run only; token from
~/.cache/huggingface/token). The kernel runs with SMOKE_LOCAL_DIR set: LocalApi replaces HfApi, CPU trainer, batch 2, checkpoint every 4 steps,
milestone every 8, deadline after --seconds, so the resume resolver, sha256 verification, manifest/environment, LR anneal, metrics.jsonl, milestone
export, checkpoint upload with read-back, final checkpoint, result.json and the collision guard all run for real. Exit 1 on any failed check.
The resume checkpoint is offered twice (legacy runs/hi_f/last.ckpt with no step in its name, and experiments/seed/checkpoints/final_step<N>.ckpt)
to exercise the probe and the tie-break. Disk: hard links are used, but plan for about 3 GB free.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = ROOT / "training/kaggle/train_kernel.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--work", type=Path, default=Path("/tmp/kernel_smoke"))
    ap.add_argument("--clips", type=int, default=24)
    ap.add_argument("--seconds", type=int, default=150)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    import torch
    from training.lr_dryrun import tiny_data

    step = int(torch.load(a.ckpt, map_location="cpu", weights_only=False)["global_step"])
    shutil.rmtree(a.work, ignore_errors=True)
    hub = a.work / "hub"
    (hub / "runs/hi_f").mkdir(parents=True)
    data = tiny_data(a.work, a.clips)
    shutil.copytree(data, hub / "data/hi_f")
    from huggingface_hub import hf_hub_download

    shutil.copy(hf_hub_download("vishwajit76/custom-tts-hindi-train", "runs/hi_f/config.json"), hub / "runs/hi_f/config.json")
    os.symlink(a.ckpt.resolve(), hub / "runs/hi_f/last.ckpt")
    (hub / "experiments/seed/checkpoints").mkdir(parents=True)
    os.symlink(a.ckpt.resolve(), hub / f"experiments/seed/checkpoints/final_step{step}.ckpt")
    env = dict(os.environ, SMOKE_LOCAL_DIR=str(hub), W_ROOT=str(a.work / "w"), BS="2", CKPT_EVERY="4", MS_EVERY="8", UPLOAD_EVERY_S="20", HB_EVERY_S="5",
               METRICS_EVERY_S="20", MAX_HOURS=str(a.seconds / 3600), KERNEL_VERSION="smoke", GIT_SHA="smoketest", ANNEAL_EPOCHS="3",
               SMOKE_EPOCHS="6", METRICS_EVERY_STEPS="4", CUDA_VISIBLE_DEVICES="")
    env.pop("HF_TOKEN", None)
    log = a.work / "kernel.log"
    with open(log, "w") as f:
        rc = subprocess.run([sys.executable, str(KERNEL)], env=env, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    out = log.read_text()
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    check("kernel exit code 0", rc == 0, f"rc={rc}")
    exps = sorted((hub / "experiments").glob("hi_f-smoke-*"))
    check("one experiment folder created", len(exps) == 1, str(exps))
    if not exps:
        print(out[-3000:]); sys.exit(1)
    e = exps[0]
    man = json.loads((e / "manifest.json").read_text())
    check("resume explicit+verified: experiments/ preferred on a step tie", man["resume"]["path"].startswith("experiments/seed/") or man["resume"]["path"].endswith(f"step{step}.ckpt"), man["resume"]["path"])
    check("manifest: step, sha256, dataset fingerprint, git sha, seed, cli", man["resume"]["global_step"] == step and len(man["resume"]["sha256"]) == 64
          and man["dataset"]["metadata_rows"] == a.clips and man["git_sha"] == "smoketest" and man["seed"] == 1234 and "--seed_everything" in man["cli"])
    check("manifest: probed legacy last.ckpt step", any(c["path"] == "runs/hi_f/last.ckpt" and c["step"] == step for c in man["resume"]["considered"]))
    envj = json.loads((e / "environment.json").read_text())
    check("environment.json has versions", envj["packages"].get("torch") and envj["packages"].get("piper-tts") and envj["python"], str(envj["packages"].get("piper-tts")))
    lines = [json.loads(l) for l in (e / "metrics.jsonl").read_text().splitlines()] if (e / "metrics.jsonl").exists() else []
    check("metrics.jsonl has rows with step/lr/t_utc", bool(lines) and all({"step", "lr", "t_utc"} <= set(r) for r in lines), f"{len(lines)} rows")
    last = json.loads((e / "checkpoints/last.meta.json").read_text()) if (e / "checkpoints/last.meta.json").exists() else {}
    fin = sorted((e / "checkpoints").glob("final_step*.ckpt"))
    check("last.ckpt + meta uploaded, step advanced", last.get("global_step", 0) > step and (e / "checkpoints/last.ckpt").exists(), str(last.get("global_step")))
    check("create-once final checkpoint matches last step", len(fin) == 1 and fin[0].name == f"final_step{last.get('global_step')}.ckpt", str([p.name for p in fin]))
    res = json.loads((e / "result.json").read_text()) if (e / "result.json").exists() else {}
    check("result.json written", res.get("exit_code") == 0 and res.get("final"), str({k: res.get(k) for k in ("exit_code", "last_upload")}))
    check("LR anneal ran from the kernel defaults", "LR anneal STARTS" in out and "lr [" in out)
    check("LR lowered below the restored value", any(r["lr"] and r["lr"][0] < 1.5e-4 for r in lines) or "now [0.0001, 0.0001]" in out)
    ms = sorted((e / "milestones").glob("step_*/hi_IN-custom-medium.onnx"))
    check("milestone ONNX exported and uploaded under experiments/<id>/milestones", bool(ms), str([str(p.relative_to(e)) for p in ms]))
    sp = json.loads((e / "val_split.json").read_text()) if (e / "val_split.json").exists() else {}
    check("val_split.json: piper train/val/test split fingerprint recorded", sp.get("val_n", 0) > 0 and len(sp.get("val_sha256", "")) == 64, str({k: v for k, v in sp.items() if k.endswith("_n")}))
    check("heartbeat under the experiment", (e / "heartbeat.txt").exists())
    foreign = [p for p in hub.rglob("*") if p.is_file() and not p.is_symlink() and "experiments/hi_f-smoke" not in str(p) and str(p.relative_to(hub)) not in
               ("runs/hi_f/config.json", "runs/hi_f/kaggle_progress.txt") and not str(p.relative_to(hub)).startswith(("data/", "experiments/seed/"))]
    check("nothing written outside experiments/<id>/ except the legacy heartbeat", not foreign, str(foreign[:3]))
    # second launch with the SAME experiment id but a new session must be refused by the guard
    exp_id = e.name
    env2 = dict(env, EXPERIMENT_ID=exp_id, SESSION_ID="other-session", MAX_HOURS="0.001")
    with open(a.work / "kernel2.log", "w") as f:
        rc2 = subprocess.run([sys.executable, str(KERNEL)], env=env2, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    out2 = (a.work / "kernel2.log").read_text()
    check("same experiment id from another session is refused", rc2 != 0 and "CollisionError" in out2, f"rc={rc2}")
    check("...and left the owner's files untouched", json.loads((e / "session.json").read_text())["owner_session"] != "other-session")
    if not a.keep:
        shutil.rmtree(a.work / "w", ignore_errors=True)
    print("ALL PASS" if ok else f"FAILED, see {log}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
