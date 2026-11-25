#!/usr/bin/env python3
"""check_filters.py — (from confere_filtro.py, 06/11/2026). Conferência de tof3d_filtro.FiltFFT contra o filtro direto do núcleo
(tof3d_core.Filt, ndimage.correlate), lido por ast sem importar o Taichi.

F1  igualdade fwd e adj com o filtro direto: cone r = 3 (consola), cone r = 2,5 com máscara em L, bola R = 6
    (volume local), malha 96 x 48 x 48 e 96 x 96 x 32; limite 1e-12 relativo ao máximo
F2  adjunto exato: <fwd(x), g> = <x, adj(g)> para x e g aleatórios; limite 1e-12
F3  tempo de fwd e adj nas malhas do artigo, com o raio escalado com a malha (r = 3 x nx/96 e R = 6 x nx/96),
    em todos os fios do CPU
Saída: confere_filtro.json; código de saída 0 só se F1 e F2 passarem."""
import ast
import json
import math
import os
import sys
import time

import numpy as np
from scipy import ndimage, signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import tof3d_filters as TF  # noqa: E402


def core_filt():
    src = open(os.path.join(HERE, "..", "tof3d_core.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    keep = [nd for nd in tree.body if isinstance(nd, (ast.FunctionDef, ast.ClassDef)) and nd.name in ("kernel3d", "Filt")]
    ns = {"np": np, "math": math, "ndimage": ndimage, "signal": signal}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "tof3d_core_filt", "exec"), ns)
    return ns["Filt"]


def main():
    Filt = core_filt()
    rng = np.random.default_rng(20261106)
    out = {"F1": {}, "F2": {}}
    ok = True
    L = np.ones((96, 96, 32), bool)
    a = int(round(0.4 * 96))
    I, J = np.meshgrid(np.arange(96), np.arange(96), indexing="ij")
    L &= ((I < a) | (J < a))[..., None]
    cases = [("cone3_consola", np.ones((96, 48, 48), bool), 3.0, "cone"),
             ("cone2.5_L", L, 2.5, "cone"),
             ("bola6_consola", np.ones((96, 48, 48), bool), 6.0, "ball")]
    for name, mask, r, kind in cases:
        fd = Filt(mask, r, kind=kind, fft=False)
        ff = TF.FiltFFT(mask, r, kind=kind)
        x = rng.uniform(0, 1, mask.shape)
        g = rng.normal(size=mask.shape)
        e1 = float(np.abs(fd.fwd(x) - ff.fwd(x)).max() / np.abs(fd.fwd(x)).max())
        e2 = float(np.abs(fd.adj(g) - ff.adj(g)).max() / np.abs(fd.adj(g)).max())
        lhs, rhs = float(np.sum(ff.fwd(x) * g)), float(np.sum(x * ff.adj(g)))
        e3 = abs(lhs - rhs) / max(abs(lhs), 1e-300)
        out["F1"][name] = {"fwd": e1, "adj": e2}
        out["F2"][name] = e3
        ok &= e1 < 1e-12 and e2 < 1e-12 and e3 < 1e-12
        print(name, e1, e2, e3, flush=True)
    out["passou"] = bool(ok)
    out["F3"] = {}
    for dims in ((256, 128, 128), (256, 256, 256), (512, 256, 256)):
        mask = np.ones(dims, bool)
        for kind, r0 in (("cone", 3.0), ("ball", 6.0)):
            r = r0 * dims[0] / 96
            t0 = time.perf_counter()
            ff = TF.FiltFFT(mask, r, kind=kind)
            ts = time.perf_counter() - t0
            x = rng.uniform(0, 1, dims)
            t0 = time.perf_counter(); y = ff.fwd(x); tf = time.perf_counter() - t0
            t0 = time.perf_counter(); z = ff.adj(y); ta = time.perf_counter() - t0
            out["F3"][f"{dims[0]}x{dims[1]}x{dims[2]}_{kind}{r:.1f}"] = {"s_preparar": ts, "s_fwd": tf, "s_adj": ta,
                                                                        "grelha_fft": list(ff.fs)}
            print(dims, kind, round(r, 1), round(ts, 2), round(tf, 2), round(ta, 2), ff.fs, flush=True)
            del ff, x, y, z
    with open(os.path.join(HERE, "check_filters.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("FIM passou", out["passou"], flush=True)
    sys.exit(0 if out["passou"] else 1)


if __name__ == "__main__":
    main()
