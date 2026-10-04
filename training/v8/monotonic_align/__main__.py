import torch
from monotonic_align import maximum_path
v = torch.full((1, 3, 3), -10.0); v[0, [0, 1, 2], [0, 1, 2]] = 0
assert maximum_path(v).squeeze().equal(torch.eye(3)), maximum_path(v)
print("ok")
