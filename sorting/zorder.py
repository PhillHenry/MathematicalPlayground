import numpy as np


def interleave(x: int, y: int) -> int:
    x = np.binary_repr(x)
    y = np.binary_repr(y)
    pad = max(len(x), len(y))
    a = x.zfill(pad)
    b = y.zfill(pad)

    c = np.empty(pad * 2, dtype=np.int)
    c[0::2] = list(a)
    c[1::2] = list(b)

    return c.dot(1 << np.arange(c.shape[-1] - 1, -1, -1))
