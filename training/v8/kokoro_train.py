"""Run kikiri-tts' patched StyleTTS2 trainer (stage 1 = train_first.py, stage 2 = train_second.py) on a run dir from kokoro_prep.py,
resuming from the newest checkpoint in <run>/ckpt. Caps VRAM at 95% (WDDM spills silently past 6 GB) and logs peak VRAM.

  python training/v8/kokoro_train.py exp/v8_kokoro/EXP-001 --stage 1 [--epochs N] [--precision no|bf16|fp16]
"""
import argparse, json, os, re, sys, threading, time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
S2 = ROOT / "exp/kikiri-tts/StyleTTS2"


def latest(ckpt_dir, stage):
    pat = re.compile(r"epoch_1st_(\d+)\.pth" if stage == 1 else r"epoch_2nd_(\d+)\.pth")
    c = sorted((int(m[1]), p) for p in Path(ckpt_dir).glob("*.pth") if (m := pat.fullmatch(p.name)))
    return c[-1][1] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--precision", default="no", choices=["no", "bf16", "fp16"])  # accelerate autocast; fp16 adds GradScaler
    ap.add_argument("--batch", type=int)
    ap.add_argument("--max-len", type=int, help="decoder crop, mel frames (80 per second)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--log-dir", help="override checkpoint dir (throughput sweeps)")
    ap.add_argument("--max-iters", type=int, default=0, help="sweep: stop after N train iterations")
    ap.add_argument("--ckpt-decoder", action="store_true", help="gradient checkpointing on the ISTFTNet decoder")
    a = ap.parse_args()
    run = (ROOT / a.run).resolve()
    cfg = yaml.safe_load(open(run / "config.yml", encoding="utf-8"))
    if a.epochs:
        cfg["epochs_1st" if a.stage == 1 else "epochs_2nd"] = a.epochs
    if a.batch: cfg["batch_size"] = a.batch
    if a.max_len: cfg["max_len"] = a.max_len
    if a.workers is not None: cfg["data_params"]["num_workers"] = a.workers
    if a.log_dir: cfg["log_dir"] = str((ROOT / a.log_dir).resolve())
    ck = Path(cfg["log_dir"])
    if a.stage == 2:  # stage 2 starts from stage-1 output (kikiri: second_stage_load_pretrained=false)
        cfg["first_stage_path"] = str(ck / "first_stage.pth")
    prev = latest(ck, a.stage)
    if prev:  # resume: weights + optimizer + epoch/iters
        cfg.update(pretrained_model=str(prev), load_only_params=False, second_stage_load_pretrained=True)
        print(f"resuming from {prev}", flush=True)
    rc = run / f"_stage{a.stage}.yml"
    yaml.safe_dump(cfg, open(rc, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)

    # StyleTTS2 utility checkpoints (ASR/JDC/PLBERT, own optimizer states) are full pickles; torch 2.6 defaults to weights_only.
    # ponytail: global opt-out is acceptable only because every file loaded here is ours or from the pinned kikiri submodule.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.environ["ACCELERATE_MIXED_PRECISION"] = a.precision
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for l in open(ROOT / ".env", encoding="utf-8"):  # HF token for WavLM/PLBERT downloads; never printed
        k, _, v = l.strip().partition("=")
        if k in ("HF_TOKEN", "HUGGINGFACE_ACCESS_TOKEN") and v.strip().strip('"'):
            os.environ.setdefault("HF_TOKEN", v.strip().strip('"'))
    import torch, pynvml
    torch.cuda.set_per_process_memory_fraction(0.95)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = os.environ.get("CUDNN_BENCH") == "1"  # variable-length aligner/style inputs re-autotune every step
    pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
    peak = {"nvml_mib": 0.0}
    t0 = time.time()
    steps, utils = [], []  # decoder optimizer steps (1 per train iteration) and (t, gpu util %)

    def sample():
        while True:
            peak["nvml_mib"] = max(peak["nvml_mib"], pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20)
            peak["torch_mib"] = torch.cuda.max_memory_allocated() / 2**20
            peak["temp_c"] = max(peak.get("temp_c", 0), pynvml.nvmlDeviceGetTemperature(h, 0))
            peak["wall_s"] = round(time.time() - t0, 1)
            utils.append((time.time(), pynvml.nvmlDeviceGetUtilizationRates(h).gpu))
            if len(steps) > 11:  # steady state: skip first 10 iterations (cudnn autotune, warmup)
                s = steps[10:]; win = [u for t, u in utils if s[0] <= t <= s[-1]]
                peak.update(train_iters=len(steps), s_per_iter=round((s[-1] - s[0]) / (len(s) - 1), 3),
                            samples_per_s=round(cfg["batch_size"] * (len(s) - 1) / (s[-1] - s[0]), 2),
                            gpu_util_train_mean=round(sum(win) / max(1, len(win)), 1))
            (ck / f"perf_stage{a.stage}.json").write_text(json.dumps(peak))
            time.sleep(1)
    ck.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=sample, daemon=True).start()

    sys.path[:0] = [str(Path(__file__).parent), str(S2)]  # monotonic_align numba shim, StyleTTS2 modules
    os.chdir(S2)
    import optimizers
    _step = optimizers.MultiOptimizer.step

    def step(self, key=None, scaler=None):
        if key == "decoder":  # ponytail: per-iteration timing via the decoder step; includes the sync it forces
            steps.append(time.time())
            if a.max_iters and len(steps) > a.max_iters:
                raise KeyboardInterrupt(f"max-iters {a.max_iters} reached")
            if len(steps) <= 3:
                print(f"iter {len(steps)} alloc {torch.cuda.memory_allocated() / 2**20:.0f} MiB peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)
        return _step(self, key, scaler)
    optimizers.MultiOptimizer.step = step
    import kokoro_tb_utils  # TensorBoard-only voicepack probe: 200 clips/epoch costs minutes; 20 is enough to monitor
    _ev = kokoro_tb_utils.extract_voicepack
    kokoro_tb_utils.extract_voicepack = lambda *x, n_samples=200, **k: _ev(*x, n_samples=min(n_samples, 20), **k)
    if a.ckpt_decoder:  # recompute decoder activations in backward: frees VRAM for a bigger batch (step is launch-bound)
        import torch.utils.checkpoint as tuc
        from Modules import istftnet
        _fwd = istftnet.Decoder.forward
        istftnet.Decoder.forward = lambda self, *x: (tuc.checkpoint(_fwd, self, *x, use_reentrant=False)
                                                     if torch.is_grad_enabled() else _fwd(self, *x))
    mod = __import__("train_first" if a.stage == 1 else "train_second")
    rc_ = 0
    try:
        mod.main.main(["--config_path", str(rc)], standalone_mode=False)
    except BaseException:
        import traceback; traceback.print_exc(); rc_ = 1
    print("PERF", json.dumps(peak), flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(rc_)  # dataloader/daemon threads can keep a crashed run alive holding VRAM


if __name__ == "__main__":
    main()
