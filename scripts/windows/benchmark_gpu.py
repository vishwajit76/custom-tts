import time, torch
assert torch.cuda.is_available(), "CUDA unavailable"
d = "cuda"
print("GPU:", torch.cuda.get_device_name(0))
def tflops(dt, n=4096, it=30):
    a = torch.randn(n, n, device=d, dtype=dt); b = torch.randn(n, n, device=d, dtype=dt)
    for _ in range(3): a @ b
    torch.cuda.synchronize(); t = time.time()
    for _ in range(it): a @ b
    torch.cuda.synchronize(); return 2 * n**3 * it / (time.time() - t) / 1e12
print(f"fp32 matmul: {tflops(torch.float32):.2f} TFLOPS")
print(f"fp16 matmul: {tflops(torch.float16):.2f} TFLOPS")
x = torch.empty(256 * 1024 * 1024, dtype=torch.uint8).pin_memory(); y = torch.empty_like(x, device=d)
y.copy_(x); torch.cuda.synchronize(); t = time.time()
for _ in range(10): y.copy_(x, non_blocking=True)
torch.cuda.synchronize(); print(f"H2D bandwidth: {2.5 / (time.time() - t):.1f} GB/s (pinned)")
del x, y; torch.cuda.empty_cache(); blocks = []
try:
    while True: blocks.append(torch.empty(256 * 1024**2, dtype=torch.uint8, device=d))
except torch.OutOfMemoryError:
    print(f"Peak VRAM alloc: {len(blocks) * 0.25:.2f} GB (of {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB)")
