# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Leonilson Toffoli Pedrini
"""TOF3D — normalised density filter on a masked 3D grid by FFT convolution on all CPU threads.

Same filter as the direct one in the core (Bruns and Tortorelli 2001; Bourdin 2001): with a symmetric kernel w
(cone of radius r, or ball) and the mask m of active elements,

    fwd(x) = m * (w * (m x)) / W,      W = w * m   (W = 1 where it vanishes),
    adj(g) = m * (w * (m g / W)),

where * is the correlation with zero padding outside the grid. The kernel spectrum is computed once; each
application is one real FFT, a product and one inverse FFT on a grid padded by the kernel radius (no wrap-around).

    filt = FiltFFT(mask, r, kind="cone", workers=-1)
    xt = filt.fwd(x);  gx = filt.adj(gt)
"""
import math

import numpy as np
from scipy import fft as sfft


def kernel3d(r, kind="cone"):
    """Cone max(0, r - d) on the offsets with d < r, or ball 1[d <= r]; identical to the core's kernel3d."""
    m = int(math.ceil(r)) - (1 if kind == "cone" else 0)
    m = max(m, 0)
    o = np.arange(-m, m + 1)
    X, Y, Z = np.meshgrid(o, o, o, indexing="ij")
    d = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    if kind == "cone":
        return np.maximum(0.0, r - d)
    return (d <= r + 1e-9).astype(np.float64)


class FiltFFT:
    def __init__(self, mask, r, kind="cone", workers=-1):
        self.m = np.asarray(mask, dtype=np.float64)
        self.shape = self.m.shape
        self.workers = workers
        k = kernel3d(r, kind)
        h = k.shape[0] // 2
        self.h = h
        self.fs = tuple(sfft.next_fast_len(s + h, real=True) for s in self.shape)
        kp = np.zeros(self.fs)
        o = np.arange(-h, h + 1)
        ix = [np.mod(o, n) for n in self.fs]
        kp[np.ix_(*ix)] = k            # kernel centred on the origin (offsets taken modulo the padded size)
        self.K = sfft.rfftn(kp, s=self.fs, workers=workers)
        W = self._conv(self.m)
        W[W <= 1e-12] = 1.0
        self.W = W

    def _conv(self, a):
        A = sfft.rfftn(a, s=self.fs, workers=self.workers)
        A *= self.K
        out = sfft.irfftn(A, s=self.fs, workers=self.workers)
        return np.ascontiguousarray(out[:self.shape[0], :self.shape[1], :self.shape[2]])

    def fwd(self, x):
        return self._conv(x * self.m) / self.W * self.m

    def adj(self, g):
        return self._conv(g * self.m / self.W) * self.m
