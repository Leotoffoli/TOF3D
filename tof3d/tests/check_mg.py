#!/usr/bin/env python3
"""check_mg.py — TOF3D multigrid checks (from confere_mg.py, 06/11/2026; Emenda 3: G1 at level 1 against
the auxiliary operator when the domain has removed regions, flag checked; G4 L-shaped domain with the pass-1 design).

Modo «exato» (correr com TOF3D_FP64=1):
  G1  operadores grossos = Galerkin exato: S_l x contra P^T A_{l-1} P x, vetores aleatórios, níveis 1 e 2,
      em três problemas (consola com densidades SIMP, suporte em L com elementos inativos, inversor com molas)
  G2  solução da multigrelha + CG contra o PCG de Jacobi do núcleo, ambos com resíduo 1e-10
Modo «iter» (float32, como nas corridas):
  G3  iterações e tempo até resíduo 1e-5 em malhas 24, 48 e 96 (consola, projeto suave com contraste SIMP) e no
      projeto real C_std do passo 1; escolha de omega e nu numa malha de 48
Uso: python confere_mg.py exato|iter [cpu|cuda]  →  confere_mg_<modo>.json"""
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import tof3d_core as C  # noqa: E402

MODE = sys.argv[1] if len(sys.argv) > 1 else "exato"
ARCH = sys.argv[2] if len(sys.argv) > 2 else "cpu"
if MODE == "exato" and not C.FP64:
    sys.exit("modo exato pede TOF3D_FP64=1")
C.init(ARCH, mem_gb=6.0)
import tof3d_examples as X  # noqa: E402
import tof3d_mg as M  # noqa: E402

EMIN, PENAL = 1e-4, 3.0


def smooth_design(shape, seed, vf=0.3):
    rng = np.random.default_rng(seed)
    g = ndimage.gaussian_filter(rng.normal(size=shape), sigma=max(1.0, shape[0] / 16))
    thr = np.quantile(g, 1 - vf)
    return 1.0 / (1.0 + np.exp(-(g - thr) / (0.05 * g.std())))


def problem(kind, dims, seed=0):
    if kind == "L":
        n, nz = dims
        fe = C.FE(n, n, nz)
        g = X.geo_L(fe, n, 1.0)
    else:
        fe = C.FE(*dims)
        g = X.geo_cantilever(fe, 1.0) if kind == "cant" else X.geo_inverter(fe, 1.0)
    fe.set_bc(g["free"], g["spring"])
    rho = smooth_design(fe.eshape, seed)
    E = np.where(g["act"], EMIN + (1 - EMIN) * rho ** PENAL, 0.0)
    fe.set_E(E)
    load = g["loads"].get("f", g["loads"].get("fin"))
    return fe, load


def g1_galerkin(mg, rng):
    out = {}
    for l in range(1, min(3, mg.L)):
        shape = mg.e[l].shape + (3,)
        x = rng.normal(size=shape)
        mg.e[l].from_numpy(x.astype(C.NPR))
        mg.apply_stencil(mg.S[l], mg.e[l], mg.t[l])
        y_st = mg.t[l].to_numpy()
        src, dst = (mg.fe.d, mg.t0) if l == 1 else (mg.e[l - 1], mg.t[l - 1])
        mg.e[l].from_numpy(x.astype(C.NPR))
        mg.prolong(mg.e[l], src, 0)
        if l == 1:
            mg.apply_fine(src, dst, 1 if mg.aux_used else 0)   # Emenda 3: Galerkin of the auxiliary operator
        else:
            mg.apply(l - 1, src, dst)
        mg.restrict(dst, mg.r[l])
        y_ref = mg.r[l].to_numpy()
        # linhas exatamente nulas de P^T A P (nós fora do domínio ativo) passam a identidade em fix_zero_rows
        nulas = (y_ref == 0.0)
        out[f"nivel{l}"] = float(np.linalg.norm((y_st - y_ref)[~nulas]) / np.linalg.norm(y_ref[~nulas]))
        out[f"nivel{l}_linhas_identidade"] = int(nulas.sum())
        out[f"nivel{l}_identidade_erro"] = float(np.abs((y_st - x)[nulas]).max()) if nulas.any() else 0.0
        # simetria do denso do nível
    A = mg.dense(mg.L - 1)
    out["simetria_grosso"] = float(np.abs(A - A.T).max() / np.abs(A).max())
    out["min_autovalor_grosso_rel"] = float(np.linalg.eigvalsh(A).min() / np.abs(A).max())
    return out


def run_exato():
    rng = np.random.default_rng(7)
    res = {}
    for kind, dims in (("cant", (32, 16, 16)), ("L", (32, 8)), ("inv", (32, 16, 16))):
        fe, load = problem(kind, dims)
        xj, itj, relj = fe.solve(load, tol=1e-10, maxit=200000, check=50)
        r = {}
        for pen in (1.0, 0.0):
            mg = M.MG(fe, pen_support=pen)
            mg.setup()
            xm, itm, relm = mg.solve(load, tol=1e-10, maxit=2000)
            r[f"pen{pen:g}"] = {"niveis": [list(s) for s in mg.shapes], "aux": bool(mg.aux_used),
                                "nos_fora": int(mg.n_out[None]), "G1": g1_galerkin(mg, rng),
                                "G2": {"it_jacobi": itj, "it_mg": itm, "res_jacobi": relj, "res_mg": relm,
                                       "dif_rel_solucao": float(np.linalg.norm(xm - xj) / np.linalg.norm(xj))}}
        res[kind] = r
        print(kind, json.dumps(r), flush=True)
    ok = all(v["G1"]["nivel1"] < 1e-12 and v["G1"].get("nivel2", 0.0) < 1e-12 and v["G1"]["simetria_grosso"] < 1e-12
             and v["G1"]["nivel1_identidade_erro"] < 1e-12
             and v["G1"]["min_autovalor_grosso_rel"] > 0 and v["G2"]["dif_rel_solucao"] < 1e-8
             for r in res.values() for v in r.values())
    # Emenda 3: the auxiliary operator is used exactly when the domain has removed regions
    ok = ok and all(v["aux"] == (kind == "L") and (v["nos_fora"] > 0) == (kind == "L")
                    for kind, r in res.items() for v in r.values())
    res["passou"] = bool(ok)
    return res


def timed(fn):
    t0 = time.time()
    out = fn()
    return out, time.time() - t0


def run_iter():
    res = {"G3": {}, "omega_nu": {}}
    fe, load = problem("cant", (48, 24, 24), seed=1)
    for om in (0.5, 0.6, 0.7):
        for nu in (1, 2, 3):
            mg = M.MG(fe, omega=om, nu=nu)
            mg.setup()
            (x, it, rel), t = timed(lambda: mg.solve(load, tol=1e-5))
            res["omega_nu"][f"w{om}_nu{nu}"] = {"it": it, "s": t, "rel": rel}
            print("omega_nu", om, nu, it, round(t, 3), flush=True)
    best = min(res["omega_nu"].items(), key=lambda kv: kv[1]["s"])[0]
    om, nu = float(best.split("_")[0][1:]), int(best.split("nu")[1])
    res["escolha"] = {"omega": om, "nu": nu}
    cases = [("cant24", (24, 12, 12), None), ("cant48", (48, 24, 24), None), ("cant96", (96, 48, 48), None),
             ("C_std_passo1", (96, 48, 48), os.environ.get("TOF3D_CSTD_DESIGN", os.path.join(HERE, "C_std_desenho.npz")))]
    # Emenda 3: L-shaped domain with the pass-1 design, coarse levels from the auxiliary operator (void_eps 1e-6)
    # against the release behaviour (void_eps 0: coarse levels from the true operator, cut coarse cells)
    res["G4_L"] = {}
    zl = np.load(os.path.join(HERE, "L_comp_desenho.npz"))
    for n_, nz_ in ((96, 32),):
        fe, load = problem("L", (n_, nz_), seed=2)
        act = zl["act"]
        fe.set_E(np.where(act, EMIN + (1 - EMIN) * zl["xp_i"].astype(np.float64) ** PENAL, 0.0))
        (xj, itj, relj), tj = timed(lambda: fe.solve(load, tol=1e-5, maxit=100000))
        for eps in (1e-6, 0.0):
            # production defaults (omega 0.6, nu 2, as in tof3d_examples): the sweep above picks by wall time and
            # can pick omega 0.7, which is unstable here (lambda_max(D^-1 A) up to 3.2 > 2/0.7)
            mg = M.MG(fe, void_eps=eps)
            _, ts = timed(mg.setup)
            (xm, itm, relm), tm = timed(lambda: mg.solve(load, tol=1e-5))
            res["G4_L"][f"void_eps{eps:g}"] = {"aux": bool(mg.aux_used), "it_mg": itm, "s_mg": tm, "s_setup": ts,
                                               "res_mg": relm, "it_jacobi": itj, "s_jacobi": tj,
                                               "dif_rel": float(np.linalg.norm(xm - xj) / np.linalg.norm(xj))}
            print("G4_L", eps, json.dumps(res["G4_L"][f"void_eps{eps:g}"]), flush=True)
    for name, dims, path in cases:
        fe, load = problem("cant", dims, seed=2)
        if path and os.path.exists(path):
            xp = np.load(path)["xp_i"].astype(np.float64)
            fe.set_E(EMIN + (1 - EMIN) * xp ** PENAL)
        elif path:
            res["G3"][name] = "sem ficheiro"
            continue
        mg = M.MG(fe, omega=om, nu=nu)
        _, ts = timed(mg.setup)
        (xm, itm, relm), tm = timed(lambda: mg.solve(load, tol=1e-5))
        (xj, itj, relj), tj = timed(lambda: fe.solve(load, tol=1e-5, maxit=100000))
        res["G3"][name] = {"dims": list(dims), "niveis": len(mg.shapes), "it_mg": itm, "s_mg": tm, "s_setup": ts,
                           "it_jacobi": itj, "s_jacobi": tj, "res_mg": relm, "res_jacobi": relj,
                           "dif_rel": float(np.linalg.norm(xm - xj) / np.linalg.norm(xj))}
        print(name, json.dumps(res["G3"][name]), flush=True)
    return res


if __name__ == "__main__":
    out = run_exato() if MODE == "exato" else run_iter()
    out["arch"] = ARCH
    out["fp64"] = bool(C.FP64)
    with open(os.path.join(HERE, f"check_mg_{MODE}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("FIM", MODE, out.get("passou", ""))
