"""Numba port of resemble-ai/monotonic_align core.pyx (1-step MAS); avoids needing MSVC for the Cython build on Windows."""
import numba
import numpy as np


@numba.njit(cache=True)
def _each(path, value, t_x, t_y, max_neg_val):
    index = t_x - 1
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = max_neg_val if x == y else value[x, y - 1]
            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg_val
            else:
                v_prev = value[x - 1, y - 1]
            value[x, y] = max(v_cur, v_prev) + value[x, y]
    for y in range(t_y - 1, -1, -1):
        path[index, y] = 1
        if index != 0 and (index == y or value[index, y - 1] < value[index - 1, y - 1]):
            index = index - 1


@numba.njit(cache=True)
def maximum_path_c(paths, values, t_xs, t_ys, max_neg_val=-1e9):
    for i in range(values.shape[0]):
        _each(paths[i], values[i], t_xs[i], t_ys[i], np.float32(max_neg_val))
