#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Leonilson Toffoli Pedrini
"""TOF3D examples: four classes of topology optimisation problems in 3D (eight cases), on one GPU.

  L_comp  L-bracket, minimum compliance, V <= 0.30                                     (reference)
  L_tens  L-bracket, minimum compliance, V <= 0.30, p-norm of von Mises <= 0.70 x peak of L_comp
  C_std   cantilever, minimum compliance, V <= 0.12, one projection (eta 0.5)          (reference)
  C_rob   cantilever, robust formulation (eta 0.75/0.5/0.25): eroded compliance, volume on the dilated design
  M_std   force inverter, maximum output displacement, V <= 0.25                      (reference)
  M_rob   force inverter, min-max over the three projections (eta 0.7/0.5/0.3), volume on the dilated design
  I_inf   cantilever, bone-like infill: minimum compliance, local volume <= 0.6 (R = 6, p = 16)
  I_ref   cantilever, minimum compliance, V <= final volume of I_inf, same filter      (reference)

Sizes: --size N sets the largest side of the grid. Cantilever and inverter: N x N/2 x N/2; L-bracket:
N x N x nz with nz close to N/3 and suited to the multigrid. Every length of the problem (filter radii, load
patches, supports, the zone left out of the stress measure) is scaled by s = N/96, so that N = 96 reproduces the
reference problems exactly.

Solver: conjugate gradient with the geometric multigrid of tof3d_mg (block-Jacobi CG for grids that cannot be
coarsened); optimiser: MMA with the dual subproblem solver on the GPU (tof3d_mma_gpu); filter by FFT
(tof3d_filters).

Usage:  python tof3d_examples.py CASE [--size N] [--iters K] [--out DIR] [--mg-policy design|iteration]
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import ndimage

import tof3d_core as C

HERE = os.path.dirname(os.path.abspath(__file__))

EMIN, PENAL, NU = 1e-4, 3.0, 0.3
CG_TOL = 1e-5
ITERS = 300
REF_SIZE = 96
CASES = {
    "L_comp": dict(geo="L", r=2.5, vf=0.30, kind="comp", etas={"i": 0.5}, betas=[1, 2, 4, 8], every=50, move=0.1),
    "L_tens": dict(geo="L", r=2.5, vf=0.30, kind="tens", etas={"i": 0.5}, betas=[1, 2, 4, 8], every=50, move=0.1,
                   P=10.0, q=0.5, frac_lim=0.70, ref="L_comp"),
    "C_std": dict(geo="C", r=3.0, vf=0.12, kind="comp", etas={"i": 0.5}, betas=[1, 2, 4, 8, 16, 32], every=50,
                  move=0.2),
    "C_rob": dict(geo="C", r=3.0, vf=0.12, kind="rob_comp", etas={"e": 0.75, "i": 0.5, "d": 0.25},
                  betas=[1, 2, 4, 8, 16, 32], every=50, move=0.2),
    "M_std": dict(geo="M", r=2.5, vf=0.25, kind="mech", etas={"i": 0.5}, betas=[1, 2, 4, 8, 16, 32], every=50,
                  move=0.1),
    "M_rob": dict(geo="M", r=2.5, vf=0.25, kind="mech", etas={"e": 0.7, "i": 0.5, "d": 0.3},
                  betas=[1, 2, 4, 8, 16, 32], every=50, move=0.1),
    "I_inf": dict(geo="I", r=1.5, kind="infill", etas={"i": 0.5}, betas=[1, 2, 4, 8, 16], every=60, move=0.2,
                  R=6.0, alpha=0.6, pl=16.0, vf0=0.5),
    "I_ref": dict(geo="I", r=1.5, kind="comp", etas={"i": 0.5}, betas=[1, 2, 4, 8, 16], every=60, move=0.2,
                  ref="I_inf"),
}
VD_EVERY = 20          # volume of the dilated design refreshed every 20 iterations (Wang, Lazarov and Sigmund 2011)
MECH_SHIFT = 500.0     # |u_out| <= f/(2 sqrt(k_in k_out)) bounds the normalised value near -330


def beta_at(it, every, betas):
    return betas[min(it // every, len(betas) - 1)]


def coarse_dofs(dims, max_coarse_dof=4500):
    """Unknowns of the coarsest multigrid level for these element counts (rule of tof3d_mg.MG)."""
    d = list(dims)
    while (d[0] + 1) * (d[1] + 1) * (d[2] + 1) * 3 > max_coarse_dof and not (d[0] % 2 or d[1] % 2 or d[2] % 2):
        d = [v // 2 for v in d]
    return (d[0] + 1) * (d[1] + 1) * (d[2] + 1) * 3


def grid(geo, N):
    if geo == "L":
        target = N / 3.0
        if N == REF_SIZE:
            return dict(n=N, nz=32)
        cands = [z for z in range(max(2, int(0.8 * target)), int(1.2 * target) + 1) if z % 2 == 0]
        ok = [z for z in cands if coarse_dofs((N, N, z)) <= 6000]
        nz = min(ok or cands, key=lambda z: abs(z - target))
        return dict(n=N, nz=nz)
    return dict(nx=N, ny=N // 2, nz=N // 2)


# ------------------------------------------------------------------ geometries (s = N/96)
def geo_L(fe, n, s):
    a = int(round(0.4 * n))
    I, J, K = C.elem_grid(fe)
    act = (I < a) | (J < a)
    In, Jn, Kn = C.node_grid(fe)
    nact = C.nodes_touching(fe, act)
    free = np.repeat(nact[..., None], 3, axis=-1).astype(float)
    sup = (Jn == fe.ny) & (In <= a)
    free[sup] = 0.0
    h = max(2, int(round(0.05 * n)))
    ld = (In == fe.nx) & (Jn >= a - h) & (Jn <= a)
    f = np.zeros(fe.nshape + (3,))
    f[ld, 1] = -1.0 / ld.sum()
    solid = C.elems_touching(fe, ld) & act
    near = ndimage.binary_dilation(C.elems_touching(fe, ld) | C.elems_touching(fe, sup),
                                   iterations=max(1, int(round(3 * s))))
    smask = act & ~near
    return dict(act=act, free=free, spring=None, loads={"f": f}, solid=solid, void=~act, smask=smask,
                info=dict(a=a, h=h, n_ativos=int(act.sum())))


def geo_cantilever(fe, s):
    In, Jn, Kn = C.node_grid(fe)
    free = np.ones(fe.nshape + (3,))
    free[0] = 0.0
    zc = fe.nz // 2
    w = int(round(2 * s))
    ld = (In == fe.nx) & (Jn <= w) & (np.abs(Kn - zc) <= w)
    f = np.zeros(fe.nshape + (3,))
    f[ld, 1] = -1.0 / ld.sum()
    flat = np.zeros(fe.nshape + (3,))
    flat[ld, 2] = 1.0 / ld.sum()
    solid = C.elems_touching(fe, ld)
    act = np.ones(fe.eshape, bool)
    return dict(act=act, free=free, spring=None, loads={"f": f, "f_lateral": flat}, solid=solid,
                void=np.zeros(fe.eshape, bool), smask=None, info=dict(n_carga=int(ld.sum())))


def geo_inverter(fe, s):
    """Half domain with symmetry at y = 0 (uy = 0); input at x = 0, output at x = nx, both on the symmetry plane."""
    In, Jn, Kn = C.node_grid(fe)
    free = np.ones(fe.nshape + (3,))
    free[:, 0, :, 1] = 0.0
    sp = max(2, fe.ny // 10)
    sup = (In == 0) & (Jn >= fe.ny - sp)
    free[sup] = 0.0
    z0, z1 = fe.nz // 3, fe.nz - fe.nz // 3
    patch = (Kn >= z0) & (Kn <= z1) & (Jn <= max(1, int(round(s))))
    nin = (In == 0) & patch
    nout = (In == fe.nx) & patch
    k_tot = 0.1 * (z1 - z0)
    spring = np.zeros(fe.nshape + (3,))
    spring[nin, 0] = k_tot / nin.sum()
    spring[nout, 0] = k_tot / nout.sum()
    fin = np.zeros(fe.nshape + (3,))
    fin[nin, 0] = 1.0 / nin.sum()
    lout = np.zeros(fe.nshape + (3,))
    lout[nout, 0] = 1.0 / nout.sum()
    solid = C.elems_touching(fe, nin) | C.elems_touching(fe, nout)
    act = np.ones(fe.eshape, bool)
    return dict(act=act, free=free, spring=spring, loads={"fin": fin, "lout": lout}, solid=solid,
                void=np.zeros(fe.eshape, bool), smask=None,
                info=dict(k_in=k_tot, k_out=k_tot, n_in=int(nin.sum()), n_out=int(nout.sum()), apoio=int(sp)))


def build(case, N):
    cfg = CASES[case]
    g = cfg["geo"]
    dims = grid(g, N)
    s = N / REF_SIZE
    if g == "L":
        fe = C.FE(dims["n"], dims["n"], dims["nz"], NU)
        geo = geo_L(fe, dims["n"], s)
    elif g == "M":
        fe = C.FE(dims["nx"], dims["ny"], dims["nz"], NU)
        geo = geo_inverter(fe, s)
    else:
        fe = C.FE(dims["nx"], dims["ny"], dims["nz"], NU)
        geo = geo_cantilever(fe, s)
    fe.set_bc(geo["free"], geo["spring"])
    return cfg, fe, geo, s


def load_ref(outroot, name):
    with open(os.path.join(outroot, name, "metricas.json")) as fh:
        return json.load(fh)


def volfrac(xp, act):
    return float(xp[act].sum() / act.sum())


def run(case, N, iters, outroot, mg_policy="design", store="auto", smoke=False):
    cfg = CASES[case]
    outdir = os.path.join(outroot, case + ("_smoke" if smoke else ""))
    os.makedirs(outdir, exist_ok=True)
    t_start = time.time()
    C.init("cuda", mem_gb=11.0)
    import taichi as ti
    import tof3d_filters as TF
    import tof3d_mg as MGM
    import tof3d_mma_gpu as MG_MMA
    cfg, fe, geo, s = build(case, N)
    act, solid, void = geo["act"], geo["solid"], geo["void"]
    filt = TF.FiltFFT(act, cfg["r"] * s)
    pipe = C.Pipe(filt, solid, void, cfg["etas"])
    try:
        mg = MGM.MG(fe)
    except ValueError:
        mg = None
    resp = C.Resp(fe, EMIN, PENAL, CG_TOL, active=act, mg=mg, mg_policy=mg_policy)
    dv = act & ~solid & ~void
    ndv = int(dv.sum())
    nact = int(act.sum())
    kind = cfg["kind"]
    ref = load_ref(outroot, cfg["ref"]) if cfg.get("ref") and not smoke else None

    vf = cfg.get("vf")
    if case == "I_ref":
        vf = ref["volume_i"] if ref else 0.35
    if kind == "infill":
        vf = cfg["vf0"]
    sig_lim = None
    if kind == "tens":
        sig_lim = cfg["frac_lim"] * (ref["sigma_max"] if ref else 1.0)
    x = np.zeros(fe.eshape)
    x[act] = vf
    x[solid] = 1.0

    m = {"comp": 1, "tens": 2, "rob_comp": 1, "infill": 1}.get(kind, len(cfg["etas"]) + 1)
    st = (ti.f64 if ndv <= 2_000_000 else ti.f32) if store == "auto" else (ti.f64 if store == "f64" else ti.f32)
    if kind == "mech":
        a = np.array([1.0] * (m - 1) + [0.0])
        mma = MG_MMA.MMAGPU(ndv, m, move=cfg["move"], a0=1.0, a=a, c=1000.0 * np.ones(m), d=np.ones(m), store=st)
    else:
        mma = MG_MMA.MMAGPU(ndv, m, move=cfg["move"], store=st)
    ball = TF.FiltFFT(act, cfg["R"] * s, kind="ball") if kind == "infill" else None

    hist = []
    c_ref = None
    vd_star = None
    ck = 1.0
    beta_prev = None
    u_scale = None
    vol_key = "d" if "d" in cfg["etas"] else "i"
    t_mma = 0.0
    alerts = []
    for it in range(iters):
        t0 = time.time()
        nlog0 = len(resp.cg_log)
        resp.new_iteration()
        beta = beta_at(it, cfg["every"], cfg["betas"])
        if beta_prev is not None and beta != beta_prev:
            mma.reset()
        beta_prev = beta
        xt, xps = pipe.fwd(x, beta)
        vols = {k: volfrac(v, act) for k, v in xps.items()}
        dvol = {k: act / nact for k in xps}
        rec = dict(it=it, beta=beta, vol=vols)
        if vol_key == "d" and (vd_star is None or (it % VD_EVERY == 0)):
            vd_star = vols["d"] / vols["i"] * vf
        if kind in ("comp", "tens", "infill"):
            resp.set_design(xps["i"])
            c, gc, u = resp.compliance(geo["loads"]["f"], "u")
            if c_ref is None:
                c_ref = c
            f0 = 10.0 * c / c_ref
            df0 = pipe.bwd(xt, beta, {"i": 10.0 * gc / c_ref})
            fv, dfv = [], []
            if kind in ("comp", "tens"):
                fv.append(vols["i"] / vf - 1.0)
                dfv.append(pipe.bwd(xt, beta, {"i": dvol["i"] / vf}))
            if kind == "tens":
                PN, mx, gs, _, ratio = resp.stress(geo["loads"]["f"], "u", sig_lim, cfg["P"], cfg["q"], geo["smask"])
                ck_new = mx / PN
                ck = ck_new if it == 0 else 0.5 * ck_new + 0.5 * ck
                fv.append(ck * PN - 1.0)
                dfv.append(pipe.bwd(xt, beta, {"i": ck * gs}))
                rec.update(PN=PN, ratio_max=mx, ck=ck)
            if kind == "infill":
                gl, lmax, gg, _ = C.local_volume(ball, xps["i"], cfg["alpha"], cfg["pl"], act)
                fv.append(gl)
                dfv.append(pipe.bwd(xt, beta, {"i": gg}))
                rec.update(local_pnorm=gl, local_max=lmax)
            rec.update(c=c, f0=f0)
        elif kind == "rob_comp":
            resp.set_design(xps["e"])
            c, gc, u = resp.compliance(geo["loads"]["f"], "ue")
            if c_ref is None:
                c_ref = c
            f0 = 10.0 * c / c_ref
            df0 = pipe.bwd(xt, beta, {"e": 10.0 * gc / c_ref})
            fv = [vols["d"] / vd_star - 1.0]
            dfv = [pipe.bwd(xt, beta, {"d": dvol["d"] / vd_star})]
            rec.update(c_e=c, f0=f0, vd_star=vd_star)
        elif kind == "mech":
            f0 = 0.0
            df0 = np.zeros(fe.eshape)
            fv, dfv = [], []
            uo, gms = {}, {}
            for k in cfg["etas"]:
                resp.set_design(xps[k])
                uo[k], gms[k], _ = resp.mech(geo["loads"]["fin"], geo["loads"]["lout"], "m" + k)
            if u_scale is None:
                u_scale = abs(uo["i"]) / 10.0
            for k in cfg["etas"]:
                fv.append(uo[k] / u_scale + MECH_SHIFT)
                dfv.append(pipe.bwd(xt, beta, {k: gms[k] / u_scale}))
            if vol_key == "d":
                fv.append(vols["d"] / vd_star - 1.0)
                dfv.append(pipe.bwd(xt, beta, {"d": dvol["d"] / vd_star}))
            else:
                fv.append(vols["i"] / vf - 1.0)
                dfv.append(pipe.bwd(xt, beta, {"i": dvol["i"] / vf}))
            rec.update(u_out=uo, vd_star=vd_star, u_scale=u_scale)
        fv = np.array(fv, float)
        dfv = np.array([g[dv] for g in dfv])
        tm = time.time()
        xnew = mma.step(x[dv], f0, df0[dv], fv, dfv)
        t_mma += time.time() - tm
        change = float(np.abs(xnew - x[dv]).max())
        x[dv] = xnew
        sol = resp.cg_log[nlog0:]
        bad = [e for e in sol if e[2] > CG_TOL]
        if bad:
            alerts.append({"it": it, "solucoes_sem_convergir": bad})
        rec.update(g=[float(v) for v in fv], change=change, cg=[int(e[1]) for e in sol],
                   mma_newton=mma.last_newton, mma_passagens=mma.last_passes, t=time.time() - t0)
        hist.append(rec)
        if it % 10 == 0 or it == iters - 1:
            print(json.dumps(rec, default=float), flush=True)
        if not smoke and (it % 50 == 49):
            np.savez_compressed(os.path.join(outdir, "checkpoint.npz"), x=x.astype(np.float32), it=it)

    # ---------------- final evaluation (final beta, design unchanged)
    beta = beta_at(iters - 1, cfg["every"], cfg["betas"])
    met = dict(caso=case, iteracoes=iters, beta_final=beta, dims=list(fe.eshape), tamanho=N, escala=s,
               n_ativos=nact, n_variaveis=ndv, cfg={k: v for k, v in cfg.items()}, geo=geo["info"], emin=EMIN,
               penal=PENAL, cg_tol=CG_TOL, sig_lim=sig_lim, vf_alvo=vf,
               solver="multigrelha+CG" if mg is not None else "Jacobi+CG", mg_politica=mg_policy,
               mg_niveis=[list(v) for v in mg.shapes] if mg is not None else None,
               mma="dual na GPU", mma_armazenamento="f64" if st == ti.f64 else "f32", filtro="FFT")
    resp.new_iteration()
    xt, xps = pipe.fwd(x, beta)
    met["volume_i"] = volfrac(xps["i"], act)
    met["volumes"] = {k: volfrac(v, act) for k, v in xps.items()}
    met["cinza_i"] = float(np.mean(4 * xps["i"][act] * (1 - xps["i"][act])))
    resp.warm = {}
    if cfg["geo"] in ("L", "C", "I"):
        evals = {"e": 0.75, "i": 0.5, "d": 0.25} if cfg["geo"] == "C" else {"i": 0.5}
        ev = C.Pipe(filt, solid, void, evals)
        _, xev = ev.fwd(x, beta)
        met["compliance"] = {}
        for k, xp in xev.items():
            resp.set_design(xp)
            c, _, u = resp.compliance(geo["loads"]["f"], "ev" + k)
            met["compliance"][k] = c
            if k == "i":
                vms = fe.vm(u)
                xq = np.maximum(xp, 1e-3)
                if geo["smask"] is not None:
                    sm = geo["smask"]
                    sig = xq ** 0.5 * vms
                    met["sigma_max"] = float(sig[sm].max())
                    met["sigma_p999"] = float(np.quantile(sig[sm], 0.999))
                    solid_sm = sm & (xp > 0.5)
                    met["sigma_max_solido"] = float(vms[solid_sm].max()) if solid_sm.any() else None
                    np.save(os.path.join(outdir, "sigma.npy"), sig.astype(np.float32))
        if "f_lateral" in geo["loads"]:
            resp.set_design(xev["i"])
            cl, _, _ = resp.compliance(geo["loads"]["f_lateral"], "lat")
            met["compliance_lateral"] = cl
        if kind == "infill" or case == "I_ref":
            ballm = TF.FiltFFT(act, CASES["I_inf"]["R"] * s, kind="ball")
            gl, lmax, _, rb = C.local_volume(ballm, xev["i"], CASES["I_inf"]["alpha"], CASES["I_inf"]["pl"], act)
            met["volume_local_max"] = lmax
            met["volume_local_pnorm"] = gl
    if cfg["geo"] == "M":
        ev = C.Pipe(filt, solid, void, {"e": 0.7, "i": 0.5, "d": 0.3})
        _, xev = ev.fwd(x, beta)
        met["u_out"] = {}
        for k, xp in xev.items():
            resp.set_design(xp)
            u_out, _, _ = resp.mech(geo["loads"]["fin"], geo["loads"]["lout"], "ev" + k)
            met["u_out"][k] = u_out
    met["u_scale"] = u_scale
    met["tempo_total_s"] = time.time() - t_start
    its = [e[1] for e in resp.cg_log]
    met["solucoes"] = len(its)
    met["cg_total"] = int(sum(its))
    met["cg_mediana"] = float(np.median(its)) if its else None
    met["cg_max"] = int(max(its)) if its else None
    met["mg_montagens"] = resp.n_setup
    met["t_solve_s"] = (mg.t_solve + mg.t_setup) if mg is not None else fe.t_solve
    met["t_mg_montagem_s"] = mg.t_setup if mg is not None else 0.0
    met["t_mma_s"] = t_mma
    met["alertas"] = alerts
    np.savez_compressed(os.path.join(outdir, "desenho.npz"), x=x.astype(np.float32), xt=xt.astype(np.float32),
                        act=act, solid=solid, beta=beta,
                        **{"xp_" + k: v.astype(np.float16) for k, v in xps.items()})
    with open(os.path.join(outdir, "historico.json"), "w") as fh:
        json.dump(hist, fh, default=float)
    with open(os.path.join(outdir, "metricas.json"), "w") as fh:
        json.dump(met, fh, indent=1, default=float, ensure_ascii=False)
    print("FIM", json.dumps({k: met[k] for k in met if k not in ("cfg",)}, default=float), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=list(CASES))
    ap.add_argument("--size", type=int, default=REF_SIZE)
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--mg-policy", choices=["design", "iteration"], default="design")
    ap.add_argument("--store", choices=["auto", "f32", "f64"], default="auto")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(a.case, a.size, a.iters, a.out, a.mg_policy, a.store, a.smoke)
