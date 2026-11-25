#!/usr/bin/env python3
"""check_mma.py — checks of tof3d_mma.py (from confere_mma.py, 06/11/2026; T2, the comparison with the old pass-1 MMA, is not part of the package).

T1  subproblema contra o SciPy (trust-constr, independente; o SLSQP viola as restrições nestas instâncias e foi
    posto de parte em 06/11, ver diário) em instâncias aleatórias pequenas
T2  um passo e cinco passos contra o MMA antigo de prototipo_core.py (lido por ast, sem importar o Taichi)
T3  problema de brinquedo de três variáveis e duas restrições, contra o SLSQP no problema original
T4  mínimos quadrados com restrição linear (o teste de confere_gradientes.py), contra a solução analítica
T5  forma min-max (a0 = 1, a = 1): ponto médio entre dois alvos, solução analítica
T6  tempo de um passo com n = 221 184 e m = 4 (tamanho da malha da consola)
Saída: confere_mma.json; código de saída 0 só se tudo passar."""
import ast
import json
import os
import sys
import time

import warnings

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint, minimize

# o numpy do Mac (Accelerate) emite avisos falsos de overflow em matmul; não afetam os valores
warnings.filterwarnings("ignore", message=".*encountered in matmul")
warnings.filterwarnings("ignore", category=UserWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import tof3d_mma as N  # noqa: E402


def old_mma_namespace():
    src = open(os.path.join(HERE, "prototipo_core.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    keep = [nd for nd in tree.body if isinstance(nd, (ast.FunctionDef, ast.ClassDef)) and nd.name in
            ("MMA", "mmasub", "subsolv")]
    ns = {"np": np}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "prototipo_core_mma", "exec"), ns)
    return ns


def sub_objective(L, U, p0, q0, a0, c, d, n, m):
    def f(v):
        x, y, z = v[:n], v[n:n + m], v[n + m]
        return float(np.sum(p0 / (U - x) + q0 / (x - L)) + a0 * z + np.sum(c * y + 0.5 * d * y * y))
    return f


def t1(rng):
    worst_x, worst_f, worst_v, sem_ref = 0.0, -np.inf, 0.0, 0
    for trial in range(40):
        n, m = int(rng.integers(3, 9)), int(rng.integers(1, 4))
        x0 = rng.uniform(0.2, 0.8, n)
        L, U = x0 - rng.uniform(0.2, 0.6, n), x0 + rng.uniform(0.2, 0.6, n)
        alpha = np.maximum(0.0, L + 0.1 * (x0 - L)); beta = np.minimum(1.0, U - 0.1 * (U - x0))
        p0, q0 = rng.uniform(0, 1, n) * (U - x0) ** 2, rng.uniform(0, 1, n) * (x0 - L) ** 2
        p, q = rng.uniform(0, 1, (m, n)) * (U - x0) ** 2, rng.uniform(0, 1, (m, n)) * (x0 - L) ** 2
        g0 = p @ (1 / (U - x0)) + q @ (1 / (x0 - L))
        b = g0 * rng.uniform(0.7, 1.1, m)
        a0 = 1.0
        a = (rng.uniform(0, 1, m) > 0.5).astype(float) if trial % 2 else np.zeros(m)
        c, d = np.full(m, 1000.0), np.ones(m)
        sol = N.solve_subproblem(L, U, alpha, beta, p0, q0, p, q, b, a0, a, c, d)
        f = sub_objective(L, U, p0, q0, a0, c, d, n, m)

        def viol(v):
            x = v[:n]
            return p @ (1 / (U - x)) + q @ (1 / (x - L)) - a * v[n + m] - v[n:n + m] - b
        lo = np.concatenate([alpha, np.zeros(m + 1)]); hi = np.concatenate([beta, np.full(m + 1, np.inf)])
        v0 = np.concatenate([0.5 * (alpha + beta), np.ones(m), [1.0]])
        vn = np.concatenate([sol["x"], sol["y"], [sol["z"]]])
        cands = []
        for start in (v0, vn):
            r = minimize(f, start, method="trust-constr", bounds=Bounds(lo, hi),
                         constraints=[NonlinearConstraint(viol, -np.inf, 0.0)],
                         options={"gtol": 1e-12, "xtol": 1e-14, "maxiter": 20000})
            cands.append(r.x)
            r = minimize(f, start, method="SLSQP", bounds=list(zip(lo, [None if np.isinf(h) else h for h in hi])),
                         constraints=[{"type": "ineq", "fun": lambda v: -viol(v)}],
                         options={"ftol": 1e-15, "maxiter": 5000})
            cands.append(r.x)
        feas = [v for v in cands if viol(v).max() <= 1e-8 and np.all(v >= lo - 1e-12) and np.all(v <= hi + 1e-12)]
        best = min(feas, key=f) if feas else None
        fn = f(vn)
        worst_v = max(worst_v, float(viol(vn).max()))
        if best is None:
            sem_ref += 1
            continue
        # o nosso não pode ficar pior do que o melhor ponto viável encontrado pelo SciPy
        worst_f = max(worst_f, (fn - f(best)) / max(1.0, abs(f(best))))
        worst_x = max(worst_x, float(np.max(np.abs(sol["x"] - best[:n]))))
    return {"instancias": 40, "sem_referencia_viavel": sem_ref,
            "referencia": "melhor ponto viável de trust-constr e SLSQP, cada um a partir do centro e do nosso ponto",
            "max_excesso_obj_rel": worst_f, "max_dif_x_ao_melhor": worst_x, "max_violacao_nossa": worst_v,
            "passa": worst_f < 1e-5 and worst_v < 1e-6 and sem_ref == 0}


def t2(rng, old):
    worst1, worst5 = 0.0, 0.0
    for m in (1, 2, 4):
        n = 1000
        target = rng.uniform(0, 1, n)
        W = rng.uniform(0.5, 1.5, (m, n))
        a = np.zeros(m); a[0] = 1.0 if m > 1 else 0.0

        def funcs(x):
            f0 = float(np.sum((x - target) ** 2)); df0 = 2 * (x - target)
            fv = (W @ x) / (0.4 * n) - 1.0; dfv = W / (0.4 * n)
            return f0, df0, fv, dfv
        newo = N.MMA(n, m, np.zeros(n), np.ones(n), move=0.2, a=a)
        oldo = old["MMA"](n, m, np.zeros(n), np.ones(n), move=0.2, a=a)
        x_n = x_o = np.full(n, 0.5)
        for k in range(5):
            x_n = newo.step(x_n, *funcs(x_n))
            x_o = oldo.step(x_o, *funcs(x_o))
            dif = float(np.max(np.abs(x_n - x_o)))
            if k == 0:
                worst1 = max(worst1, dif)
            worst5 = max(worst5, dif)
    return {"n": 1000, "m": [1, 2, 4], "max_dif_um_passo": worst1, "max_dif_cinco_passos": worst5,
            "passa": worst1 < 1e-5 and worst5 < 1e-4}


def t3():
    def f0(x): return float(np.sum(x ** 2)), 2 * x
    def cons(x):
        f = np.array([np.sum((x - [5, 2, 1]) ** 2) - 9.0, np.sum((x - [3, 4, 3]) ** 2) - 9.0])
        df = np.vstack([2 * (x - [5, 2, 1]), 2 * (x - [3, 4, 3])])
        return f, df
    opt = N.MMA(3, 2, np.ones(3), 5 * np.ones(3), move=1.0)
    x = np.array([4.0, 3.0, 2.0])
    for _ in range(60):
        x = opt.step(x, *f0(x), *cons(x))
    ref = minimize(lambda v: float(np.sum(v ** 2)), np.array([4.0, 3.0, 2.0]), method="SLSQP",
                   bounds=[(1, 5)] * 3, constraints=[{"type": "ineq", "fun": lambda v: -cons(v)[0]}],
                   options={"ftol": 1e-14})
    dif = float(np.max(np.abs(x - ref.x)))
    return {"x_mma": x.tolist(), "x_slsqp": ref.x.tolist(), "max_dif": dif, "viol_max": float(max(cons(x)[0].max(), 0)),
            "passa": dif < 1e-4}


def t4(rng):
    n, b_ = 40, 12.0
    a_ = rng.uniform(-0.2, 1.2, n)
    opt = N.MMA(n, 1, np.zeros(n), np.ones(n), move=0.5)
    x = np.full(n, 0.5)
    for _ in range(80):
        x = opt.step(x, float(np.sum((x - a_) ** 2)), 2 * (x - a_), np.array([x.sum() / b_ - 1.0]),
                     np.ones((1, n)) / b_)
    lo, hi = -2.0, 2.0
    for _ in range(200):
        nu = 0.5 * (lo + hi)
        if np.clip(a_ - nu, 0, 1).sum() > b_:
            lo = nu
        else:
            hi = nu
    xa = np.clip(a_ - nu, 0, 1) if np.clip(a_, 0, 1).sum() > b_ else np.clip(a_, 0, 1)
    dif = float(np.max(np.abs(x - xa)))
    return {"max_dif": dif, "passa": dif < 1e-4}


def t5(rng):
    n = 30
    u, w = rng.uniform(0.1, 0.4, n), rng.uniform(0.6, 0.9, n)
    opt = N.MMA(n, 2, np.zeros(n), np.ones(n), move=0.2, a0=1.0, a=np.ones(2))
    x = rng.uniform(0, 1, n)
    for _ in range(150):
        f = np.array([np.sum((x - u) ** 2), np.sum((x - w) ** 2)])
        df = np.vstack([2 * (x - u), 2 * (x - w)])
        x = opt.step(x, 0.0, np.zeros(n), f, df)
    xs = 0.5 * (u + w)
    dif = float(np.max(np.abs(x - xs)))
    return {"max_dif": dif, "passa": dif < 1e-3}


def t6(rng):
    n, m = 96 * 48 * 48, 4
    opt = N.MMA(n, m, np.zeros(n), np.ones(n), move=0.1, a0=1.0, a=np.array([1.0, 1.0, 1.0, 0.0]))
    x = rng.uniform(0.2, 0.8, n)
    df0 = np.zeros(n); df = rng.normal(0, 1e-3, (m, n)); f = np.array([-0.1, -0.1, -0.1, 0.01])
    t0 = time.time()
    opt.step(x, 0.0, df0, f, df)
    return {"n": n, "m": m, "s_por_passo": time.time() - t0, "passos_newton": opt.last_newton}


def main():
    rng = np.random.default_rng(20261106)
    out = {"T1_subproblema_vs_scipy": t1(rng), "T3_brinquedo": t3(),
           "T4_quadratico_linear": t4(rng), "T5_minmax": t5(rng), "T6_tempo": t6(rng)}
    out["passou"] = all(v.get("passa", True) for v in out.values() if isinstance(v, dict))
    with open(os.path.join(HERE, "check_mma.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1, default=float)[:3000])
    sys.exit(0 if out["passou"] else 1)


if __name__ == "__main__":
    main()
