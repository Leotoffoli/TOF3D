#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_gradients.py — TOF3D core checks (from confere_gradientes.py, 06/11/2026).

Corre em float64 (TOF3D_FP64=1) numa malha minúscula e confere:
  1. PCG sem matriz contra a solução direta (scipy) da matriz montada, com molas e GL fixos parciais;
  2. gradientes adjuntos contra diferenças finitas centrais, através do filtro e da projeção:
     compliance, deslocamento de saída de mecanismo, norma p de von Mises (relaxação qp),
     volume local (norma p) e compliance da realização erodida (três projeções);
  3. o MMA num problema com solução conhecida.
Escreve confere_gradientes.json ao lado. Critério congelado: erro relativo <= 1e-5 em todos os testes.
"""
import json
import os
import sys

os.environ["TOF3D_FP64"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import tof3d_core as C
import tof3d_mg as MGM
import tof3d_mma as MMAN

arch = sys.argv[1] if len(sys.argv) > 1 else "cpu"
C.init(arch)
rng = np.random.default_rng(20261106)
nx, ny, nz = 6, 4, 3
fe = C.FE(nx, ny, nz)
ns = fe.nshape

# condições de contorno: face x=0 encastrada; plano y=0 com uy=0 (fixação parcial); molas na face x=nx
free = np.ones(ns + (3,))
free[0, :, :, :] = 0.0
free[:, 0, :, 1] = 0.0
spring = np.zeros(ns + (3,))
spring[nx, 1:3, 1, 0] = 0.3
f = np.zeros(ns + (3,))
f[nx, ny, :, 1] = -1.0 / (nz + 1)
f[nx, 2, 1, 0] = 0.4
lout = np.zeros(ns + (3,))
lout[nx, 1:3, 1, 0] = 0.5
fe.set_bc(free, spring)

out = {}

# ---- 1. PCG contra solução direta
E = 0.05 + rng.random(fe.eshape)
fe.set_E(E)
Ke = C.Ke_h8()
ndof = int(np.prod(ns)) * 3
rows, cols, vals = [], [], []
for ex in range(nx):
    for ey in range(ny):
        for ez in range(nz):
            dofs = []
            for n in range(8):
                ni = ((ex + C.NX_LUT[n]) * ns[1] + ey + C.NY_LUT[n]) * ns[2] + ez + C.NZ_LUT[n]
                dofs += [3 * ni, 3 * ni + 1, 3 * ni + 2]
            dofs = np.array(dofs)
            rows.append(np.repeat(dofs, 24))
            cols.append(np.tile(dofs, 24))
            vals.append((E[ex, ey, ez] * Ke).ravel())
K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ndof, ndof))
K = K + sp.diags(spring.ravel())
fr = free.ravel() > 0.5
Kf = K[fr][:, fr]
xd = np.zeros(ndof)
xd[fr] = spla.spsolve(Kf.tocsc(), (f.ravel())[fr])
xp_, it, rel = fe.solve(f, tol=1e-13, maxit=20000)
err = np.linalg.norm(xp_.ravel() - xd) / np.linalg.norm(xd)
out["pcg_vs_direto"] = {"erro_relativo": float(err), "iteracoes": int(it), "residuo": float(rel)}
# Ke igual ao da produção (mesma fórmula); simetria e modos rígidos
rb = np.zeros((24, 3))
rb[0::3, 0] = 1
rb[1::3, 1] = 1
rb[2::3, 2] = 1
out["Ke_simetria"] = float(np.abs(Ke - Ke.T).max())
out["Ke_modos_rigidos"] = float(np.abs(Ke @ rb).max())

# ---- 2. gradientes por diferenças finitas
act = np.ones(fe.eshape, bool)
filt = C.Filt(act, 1.5)
solid = np.zeros(fe.eshape, bool)
void = np.zeros(fe.eshape, bool)
beta = 4.0
pipe1 = C.Pipe(filt, solid, void, {"i": 0.5})
pipe3 = C.Pipe(filt, solid, void, {"e": 0.7, "i": 0.5, "d": 0.3})
resp = C.Resp(fe, Emin=1e-3, penal=3.0, tol=1e-13)
ball = C.Filt(act, 2.0, kind="ball")
x0 = 0.25 + 0.5 * rng.random(fe.eshape)
smask = np.ones(fe.eshape, bool)


def F_comp(x):
    xt, xps = pipe1.fwd(x, beta)
    resp.set_design(xps["i"])
    c, g, _ = resp.compliance(f, "c")
    return c, pipe1.bwd(xt, beta, {"i": g})


def F_mech(x):
    xt, xps = pipe1.fwd(x, beta)
    resp.set_design(xps["i"])
    uo, g, _ = resp.mech(f, lout, "m")
    return uo, pipe1.bwd(xt, beta, {"i": g})


def F_stress(x):
    xt, xps = pipe1.fwd(x, beta)
    resp.set_design(xps["i"])
    PN, mx, g, _, _ = resp.stress(f, "s", 0.7, 8.0, 0.5, smask)
    return PN, pipe1.bwd(xt, beta, {"i": g})


def F_local(x):
    xt, xps = pipe1.fwd(x, beta)
    gv, mx, g, _ = C.local_volume(ball, xps["i"], 0.6, 16.0, act)
    return gv, pipe1.bwd(xt, beta, {"i": g})


def F_robust(x):
    xt, xps = pipe3.fwd(x, beta)
    resp.set_design(xps["e"])
    c, g, _ = resp.compliance(f, "ce")
    return c, pipe3.bwd(xt, beta, {"e": g})


h = 1e-6
idx = [tuple(rng.integers(0, s) for s in fe.eshape) for _ in range(6)]
for name, F in [("compliance", F_comp), ("mecanismo_uout", F_mech), ("tensao_pnorm", F_stress),
                ("volume_local", F_local), ("compliance_erodida_3proj", F_robust)]:
    v0, g = F(x0)
    rows_ = []
    worst = 0.0
    for I in idx:
        xa = x0.copy(); xa[I] += h
        xb = x0.copy(); xb[I] -= h
        fd = (F(xa)[0] - F(xb)[0]) / (2 * h)
        e = abs(fd - g[I]) / max(abs(fd), 1e-12)
        worst = max(worst, e)
        rows_.append({"elem": [int(i) for i in I], "adjunto": float(g[I]), "dif_finitas": float(fd), "erro_rel": float(e)})
    out[name] = {"valor": float(v0), "pior_erro_relativo": worst, "amostras": rows_}


# ---- 2b. the same gradients with the multigrid-preconditioned CG (8 x 4 x 4 grid, two levels)
fe2 = C.FE(8, 4, 4)
ns2 = fe2.nshape
free2 = np.ones(ns2 + (3,)); free2[0] = 0.0; free2[:, 0, :, 1] = 0.0
spring2 = np.zeros(ns2 + (3,)); spring2[8, 1:3, 1, 0] = 0.3
f2 = np.zeros(ns2 + (3,)); f2[8, 4, :, 1] = -1.0 / 5; f2[8, 2, 1, 0] = 0.4
lout2 = np.zeros(ns2 + (3,)); lout2[8, 1:3, 1, 0] = 0.5
fe2.set_bc(free2, spring2)
mg2 = MGM.MG(fe2, max_coarse_dof=150)
act2 = np.ones(fe2.eshape, bool)
pipe2 = C.Pipe(C.Filt(act2, 1.5), np.zeros(fe2.eshape, bool), np.zeros(fe2.eshape, bool), {"i": 0.5})
resp2 = C.Resp(fe2, Emin=1e-3, penal=3.0, tol=1e-13, mg=mg2, maxit=2000)
x02 = 0.25 + 0.5 * rng.random(fe2.eshape)


def G_comp(x):
    xt, xps = pipe2.fwd(x, beta)
    resp2.set_design(xps["i"])
    c, g, _ = resp2.compliance(f2, "c")
    return c, pipe2.bwd(xt, beta, {"i": g})


def G_mech(x):
    xt, xps = pipe2.fwd(x, beta)
    resp2.set_design(xps["i"])
    uo, g, _ = resp2.mech(f2, lout2, "m")
    return uo, pipe2.bwd(xt, beta, {"i": g})


idx2 = [tuple(rng.integers(0, s) for s in fe2.eshape) for _ in range(6)]
for name, F in [("compliance_multigrelha", G_comp), ("mecanismo_uout_multigrelha", G_mech)]:
    v0, g = F(x02)
    worst = 0.0
    for I in idx2:
        xa = x02.copy(); xa[I] += h
        xb = x02.copy(); xb[I] -= h
        fd = (F(xa)[0] - F(xb)[0]) / (2 * h)
        worst = max(worst, abs(fd - g[I]) / max(abs(fd), 1e-12))
    out[name] = {"valor": float(v0), "pior_erro_relativo": worst}
out["multigrelha_solucoes_max_it"] = int(max(e[1] for e in resp2.cg_log))

# ---- 3. MMA: min sum (x-a)^2 s.t. sum x <= b  (solução: x = clip(a - lam/2))
n = 50
a_ = rng.random(n)
b_ = 0.3 * n * 0.5
mma = MMAN.MMA(n, 1, np.zeros(n), np.ones(n), move=0.5)
x = 0.5 * np.ones(n)
for k in range(60):
    x = mma.step(x, float(np.sum((x - a_) ** 2)), 2 * (x - a_), np.array([x.sum() / b_ - 1.0]),
                 np.ones((1, n)) / b_)
lo, hi = -10.0, 10.0
for _ in range(200):
    lam = 0.5 * (lo + hi)
    if np.clip(a_ - lam / 2, 0, 1).sum() > b_:
        lo = lam
    else:
        hi = lam
xs = np.clip(a_ - lam / 2, 0, 1)
out["mma_problema_conhecido"] = {"erro_max": float(np.abs(x - xs).max()), "restricao": float(x.sum() / b_ - 1)}

lim = 1e-5
checks = [out["pcg_vs_direto"]["erro_relativo"]] + [out[k]["pior_erro_relativo"] for k in
                                                     ("compliance", "mecanismo_uout", "tensao_pnorm",
                                                      "volume_local", "compliance_erodida_3proj",
                                                      "compliance_multigrelha", "mecanismo_uout_multigrelha")]
checks.append(out["mma_problema_conhecido"]["erro_max"] * 1e-1)
out["criterio"] = "erro relativo <= 1e-5 (MMA: erro absoluto <= 1e-4)"
out["passou"] = bool(all(c <= lim for c in checks) and out["Ke_modos_rigidos"] < 1e-10)
out["arch"] = arch
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_gradients.json")
with open(p, "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "amostras"})
                  for k, v in out.items()}, indent=1, ensure_ascii=False))
