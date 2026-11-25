#!/usr/bin/env python3
"""check_mma_gpu.py — (from confere_mma_gpu.py, 06/11/2026). Conferência de tof3d_mma_gpu.py (MMA com subproblema pelo dual, Taichi/GPU)
contra tof3d_mma.py (numpy, ponto interior primal-dual; conferido em confere_mma.json). O Taichi arranca como nas
corridas (float32 por omissão), para apanhar arredondamentos de constantes.

T10 passos contra o numpy: n = 1000, m = 1, 2, 4 (como T2), 1 e 5 passos, guardando x em float64 e em float32.
    Referência: o ponto interior do numpy com eps_min = 1e-12 («apertado»). Com o eps_min = 1e-7 de omissão o
    ponto interior deixa x até 1,2e-4 dentro dos limites fracamente ativos (diag_mma_dual.json, 06/11), por isso
    a diferença a ele só se regista, não decide
T11 problemas com solução conhecida: brinquedo (T3, contra o SLSQP de confere_mma.json), mínimos quadrados com
    restrição linear (T4, analítico) e min-max (T5, a = 1, analítico)
T12 tempo e memória da GPU por passo (m = 4, como T6) com n = 4 194 304, 16 777 216 e 33 554 432, três passos
    seguidos (o 2.º e o 3.º arrancam do lambda anterior)
Saída: confere_mma_gpu.json; código de saída 0 só se T10 e T11 passarem."""
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import tof3d_core as C  # noqa: E402

C.init("cuda", mem_gb=11.0)
import taichi as ti  # noqa: E402
import tof3d_mma as N  # noqa: E402
import tof3d_mma_gpu as G  # noqa: E402

DEF = N.solve_subproblem.__defaults__


def gpu_mib():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=20).stdout.strip())
    except Exception:
        return None


def t10(rng):
    out = {}
    for store, tag in ((ti.f64, "f64"), (ti.f32, "f32")):
        w1, w5, wd, nw = 0.0, 0.0, 0.0, []
        for m in (1, 2, 4):
            n = 1000
            target = rng.uniform(0, 1, n)
            W = rng.uniform(0.5, 1.5, (m, n))
            a = np.zeros(m); a[0] = 1.0 if m > 1 else 0.0

            def funcs(x):
                return (float(np.sum((x - target) ** 2)), 2 * (x - target), (W @ x) / (0.4 * n) - 1.0, W / (0.4 * n))
            on = N.MMA(n, m, np.zeros(n), np.ones(n), move=0.2, a=a)
            od = N.MMA(n, m, np.zeros(n), np.ones(n), move=0.2, a=a)
            og = G.MMAGPU(n, m, move=0.2, a=a, store=store)
            xn = xd = xg = np.full(n, 0.5)
            for k in range(5):
                N.solve_subproblem.__defaults__ = (1e-12, 2000)
                xn = on.step(xn, *funcs(xn))
                N.solve_subproblem.__defaults__ = DEF
                xd = od.step(xd, *funcs(xd))
                xg = og.step(xg, *funcs(xg))
                nw.append((og.last_newton, og.last_passes))
                dif = float(np.max(np.abs(xn - xg)))
                wd = max(wd, float(np.max(np.abs(xd - xg))))
                if k == 0:
                    w1 = max(w1, dif)
                w5 = max(w5, dif)
        lim1, lim5 = (1e-5, 1e-4) if tag == "f64" else (1e-4, 1e-3)
        out[tag] = {"max_dif_um_passo": w1, "max_dif_cinco_passos": w5, "newton_passagens": nw,
                    "max_dif_ao_numpy_eps_1e-7_so_registo": wd,
                    "limites": [lim1, lim5], "passa": w1 < lim1 and w5 < lim5}
        print("T10", tag, json.dumps(out[tag]), flush=True)
    return out


def t11(rng):
    from scipy.optimize import minimize
    cons = lambda v: np.array([np.sum((v - [5, 2, 1]) ** 2) - 9.0, np.sum((v - [3, 4, 3]) ** 2) - 9.0])
    sl = minimize(lambda v: float(np.sum(v ** 2)), np.array([4.0, 3.0, 2.0]), method="SLSQP", bounds=[(1, 5)] * 3,
                  constraints=[{"type": "ineq", "fun": lambda v: -cons(v)}], options={"ftol": 1e-14})
    ref = {"T3_brinquedo": {"x_slsqp": sl.x.tolist()}}
    # brinquedo: 3 variáveis em [1, 5], 2 restrições
    opt = G.MMAGPU(3, 2, move=1.0, xmin=1.0, xmax=5.0, store=ti.f64)
    y = np.array([4.0, 3.0, 2.0])
    for _ in range(60):
        fc = np.array([np.sum((y - [5, 2, 1]) ** 2) - 9.0, np.sum((y - [3, 4, 3]) ** 2) - 9.0])
        dfc = np.vstack([2 * (y - [5, 2, 1]), 2 * (y - [3, 4, 3])])
        y = opt.step(y, float(np.sum(y ** 2)), 2 * y, fc, dfc)
    d3 = float(np.max(np.abs(y - np.array(ref["T3_brinquedo"]["x_slsqp"]))))
    # mínimos quadrados com restrição linear
    n, b_ = 40, 12.0
    a_ = rng.uniform(-0.2, 1.2, n)
    opt = G.MMAGPU(n, 1, move=0.5, store=ti.f64)
    x = np.full(n, 0.5)
    for _ in range(80):
        x = opt.step(x, float(np.sum((x - a_) ** 2)), 2 * (x - a_), np.array([x.sum() / b_ - 1.0]), np.ones((1, n)) / b_)
    lo, hi = -2.0, 2.0
    for _ in range(200):
        nu = 0.5 * (lo + hi)
        if np.clip(a_ - nu, 0, 1).sum() > b_:
            lo = nu
        else:
            hi = nu
    xa = np.clip(a_ - nu, 0, 1) if np.clip(a_, 0, 1).sum() > b_ else np.clip(a_, 0, 1)
    d4 = float(np.max(np.abs(x - xa)))
    # min-max
    n = 30
    u, w = rng.uniform(0.1, 0.4, n), rng.uniform(0.6, 0.9, n)
    opt = G.MMAGPU(n, 2, move=0.2, a0=1.0, a=np.ones(2), store=ti.f64)
    x = rng.uniform(0, 1, n)
    for _ in range(150):
        f = np.array([np.sum((x - u) ** 2), np.sum((x - w) ** 2)])
        x = opt.step(x, 0.0, np.zeros(n), f, np.vstack([2 * (x - u), 2 * (x - w)]))
    d5 = float(np.max(np.abs(x - 0.5 * (u + w))))
    out = {"brinquedo_max_dif_slsqp": d3, "quadratico_linear_max_dif": d4, "minmax_max_dif": d5,
           "passa": d3 < 1e-4 and d4 < 1e-4 and d5 < 1e-3}
    print("T11", json.dumps(out), flush=True)
    return out


def t12(rng):
    out = {"mem_antes_MiB": gpu_mib()}
    m = 4
    for n in (4194304, 16777216, 33554432):
        x = rng.uniform(0.2, 0.8, n)
        a = np.array([1.0, 1.0, 1.0, 0.0])
        opt = G.MMAGPU(n, m, move=0.1, a0=1.0, a=a, store=ti.f32)
        r = {"passos": []}
        for k in range(3):
            df0 = np.zeros(n)
            df = rng.normal(0, 1e-3, (m, n))
            f = np.array([-0.1, -0.1, -0.1, 0.01])
            ti.sync()
            t0 = time.perf_counter()
            x = opt.step(x, 0.0, df0, f, df)
            ti.sync()
            r["passos"].append({"s": time.perf_counter() - t0, "newton": opt.last_newton, "passagens": opt.last_passes})
        r["mem_MiB"] = gpu_mib()
        out[str(n)] = r
        print("T12", n, json.dumps(r), flush=True)
    return out


def main():
    rng = np.random.default_rng(20261106)
    out = {"T10_contra_numpy": t10(rng), "T11_solucoes_conhecidas": t11(rng)}
    out["passou"] = all(v["passa"] for v in out["T10_contra_numpy"].values()) and out["T11_solucoes_conhecidas"]["passa"]
    out["T12_tempo"] = t12(rng)
    with open(os.path.join(HERE, "check_mma_gpu.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("FIM passou", out["passou"], flush=True)
    sys.exit(0 if out["passou"] else 1)


if __name__ == "__main__":
    main()
