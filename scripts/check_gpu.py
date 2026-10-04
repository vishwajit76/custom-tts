import platform, subprocess, sys
try:
    import torch
except ImportError:
    torch = None
def smi(q):
    try:
        return subprocess.check_output(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], text=True).splitlines()[0].strip()
    except Exception:
        return None
name, mem, drv = smi("name"), smi("memory.total"), smi("driver_version")
cuda = bool(torch and torch.cuda.is_available())
if cuda and not name:
    name = torch.cuda.get_device_name(0)
print(f"GPU: {name or 'none'}")
print(f"VRAM: ~{round(int(mem) / 1024) if mem else 0} GB")
print(f"CUDA: {'available' if name else 'unavailable'}")
print(f"PyTorch CUDA: {'available' if cuda else 'unavailable'}")
print(f"Driver: {drv}")
print(f"torch: {torch.__version__ if torch else None} (cuda runtime {torch.version.cuda if torch else None})")
try:
    import torchaudio; print(f"torchaudio: {torchaudio.__version__}")
except Exception as e:
    print(f"torchaudio: missing ({type(e).__name__})")
print(f"Python: {sys.version.split()[0]}")
print(f"Windows: {platform.platform()}")
