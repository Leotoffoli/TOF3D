# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Leonilson Toffoli Pedrini
"""TOF3D — Method of Moving Asymptotes for very many design variables, on the GPU (Taichi), dual subproblem solver.

Same problem, approximations, asymptote rules and constants as `tof3d_mma.py` (Svanberg 1987, 2007):

    minimize    f0(x) + a0*z + sum_i (c_i*y_i + 0.5*d_i*y_i**2)
    subject to  f_i(x) - a_i*z - y_i <= 0,  xmin <= x <= xmax,  y >= 0,  z >= 0.

Only the subproblem solver differs. Instead of the primal-dual interior-point method in all variables, the convex
separable subproblem is solved through its dual in the m multipliers lambda (Svanberg 1987, section 4; Aage and
Lazarov 2013 for the parallel large-scale setting): for fixed lambda the minimiser in x is explicit,

    x_j(lambda) = clip( (sqrt(P_j) L_j + sqrt(Q_j) U_j) / (sqrt(P_j) + sqrt(Q_j)), alpha_j, beta_j ),
    P_j = p0_j + sum_i lambda_i p_ij,   Q_j = q0_j + sum_i lambda_i q_ij,

y_i(lambda) = max(0, (lambda_i - c_i)/d_i), and z enters only through the dual constraint a^T lambda <= a0. The
concave dual is maximised over lambda >= 0, a^T lambda <= a0 by a log-barrier Newton method on the host (m <= 8
unknowns). Each evaluation of the dual value, gradient and Hessian is ONE fused pass over the n variables on the
GPU (p, q, alpha, beta are rebuilt on the fly from x, L, U and the gradients; sums in float64), so the memory is
(5 + 1 + m) n numbers and a step costs a few tens of passes.

    K. Svanberg (1987) Int J Numer Meth Eng 24:359-373. doi:10.1002/nme.1620240207
    K. Svanberg (2007) MMA and GCMMA - two methods for nonlinear optimization. Technical note, KTH.
    N. Aage, B.S. Lazarov (2013) Parallel framework for topology optimization using the method of moving
        asymptotes. Struct Multidisc Optim 47:493-505. doi:10.1007/s00158-012-0869-2

    opt = MMAGPU(n, m, move=0.2, a0=1.0, a=None, c=None, d=None, xmin=0.0, xmax=1.0, store=ti.f32)
    x_new = opt.step(x, f0, df0, f, df)       # numpy in, numpy (float64) out; df: (m, n)
Taichi must be initialised (ti.init) before the object is created.
"""
import math

import numpy as np
import taichi as ti

from tof3d_mma import ALBEFA, ASY_DECR, ASY_INCR, ASY_INIT, ASY_MAX, ASY_MIN, RAA0

MAXM = 8
F64 = ti.f64
NTRI = MAXM * (MAXM + 1) // 2
NACC = 2 + MAXM + NTRI          # phi, number of free x, g (MAXM), lower triangle of H (NTRI)
TRI = [(i, k) for i in range(MAXM) for k in range(i + 1)]


@ti.data_oriented
class MMAGPU:
    def __init__(self, n, m, move=0.2, a0=1.0, a=None, c=None, d=None, xmin=0.0, xmax=1.0, store=ti.f32,
                 mu0=0.1, mu_warm=1e-4, mu_min=1e-11, newton_tol=1e-10):
        if not 1 <= m <= MAXM:
            raise ValueError(f"1 <= m <= {MAXM}")
        self.n, self.m = int(n), int(m)
        self.move, self.a0 = float(move), float(a0)
        self.a = np.zeros(m) if a is None else np.asarray(a, dtype=np.float64)
        self.c = np.full(m, 1000.0) if c is None else np.asarray(c, dtype=np.float64)
        self.d = np.ones(m) if d is None else np.asarray(d, dtype=np.float64)
        if np.any(self.d <= 0.0) or self.a0 <= 0.0 or np.any(self.a < 0.0):
            raise ValueError("need d_i > 0, a0 > 0, a_i >= 0")
        self.xmin, self.xmax = float(xmin), float(xmax)
        self.span = self.xmax - self.xmin
        self.reg = RAA0 / max(self.span, 1e-12)
        self.mu0, self.mu_warm = float(mu0), float(mu_warm)
        self.mu_min, self.newton_tol = float(mu_min), float(newton_tol)
        self.store = store
        self.cdt = F64            # arithmetic per variable in float64 (float32 arithmetic made the line search noisy)
        # constants in a float64 field: Python literals in a kernel take Taichi's default float type (float32 in the
        # solver), which would round 1.001, 1.2, 0.7, ... differently from the numpy reference
        self.cst = ti.field(F64, 16)
        cst = np.zeros(16)
        cst[:12] = [1.001, 0.001, self.reg, ALBEFA, self.move * self.span, self.xmin, self.xmax,
                    ASY_INIT * self.span, ASY_INCR, ASY_DECR, ASY_MIN * self.span, ASY_MAX * self.span]
        self.cst.from_numpy(cst)
        self.npstore = np.float32 if store == ti.f32 else np.float64
        self.x, self.x1, self.x2, self.L, self.U, self.df0 = (ti.field(store, self.n) for _ in range(6))
        self.df = ti.field(store, (m, self.n))
        self.lam = ti.field(F64, MAXM)
        # two-stage reductions: NB threads, each sums a strided share of the n variables into its own column of
        # `part`; a second kernel adds the columns (no atomic operations on shared addresses)
        self.NB = int(min(65536, self.n))
        self.part = ti.field(F64, (NACC, self.NB))
        self.red = ti.field(F64, NACC)
        self.k = 0
        self.lam_prev = None
        self.last_passes = 0
        self.last_newton = 0

    def reset(self):
        """Forget the previous iterates; the next two steps use the initial asymptotes."""
        self.k = 0
        self.lam_prev = None

    # ---------------------------------------------------------------- GPU passes
    @ti.func
    def _pq(self, ux, xl, g):
        gp = ti.max(g, 0.0)
        gm = ti.max(-g, 0.0)
        hi = ti.cast(self.cst[0], self.cdt)
        lo = ti.cast(self.cst[1], self.cdt)
        reg = ti.cast(self.cst[2], self.cdt)
        return ux * ux * (hi * gp + lo * gm + reg), xl * xl * (lo * gp + hi * gm + reg)

    @ti.kernel
    def _asymptotes(self, first: ti.i32):
        for j in self.x:
            x = ti.cast(self.x[j], F64)
            L = x - self.cst[7]
            U = x + self.cst[7]
            if first == 0:
                x1 = ti.cast(self.x1[j], F64)
                x2 = ti.cast(self.x2[j], F64)
                trend = (x - x1) * (x1 - x2)
                gamma = ti.cast(1.0, F64)
                if trend > 0.0:
                    gamma = self.cst[8]
                elif trend < 0.0:
                    gamma = self.cst[9]
                L = x - gamma * (x1 - ti.cast(self.L[j], F64))
                U = x + gamma * (ti.cast(self.U[j], F64) - x1)
                L = ti.min(ti.max(L, x - self.cst[11]), x - self.cst[10])
                U = ti.min(ti.max(U, x + self.cst[10]), x + self.cst[11])
            self.L[j] = ti.cast(L, self.store)
            self.U[j] = ti.cast(U, self.store)

    @ti.kernel
    def _reduce(self):
        for k in range(NACC):
            acc = ti.cast(0.0, F64)
            for t in range(self.NB):
                acc += self.part[k, t]
            self.red[k] = acc

    @ti.kernel
    def _gsum_part(self):
        """part[2+i, t] = share of sum_j p_ij/(U_j - x_j) + q_ij/(x_j - L_j) at the current point."""
        for t in range(self.NB):
            acc = ti.Vector.zero(F64, MAXM)
            j = t
            while j < self.n:
                x = ti.cast(self.x[j], F64)
                L = ti.cast(self.L[j], F64)
                U = ti.cast(self.U[j], F64)
                ux, xl = U - x, x - L
                rux, rxl = 1.0 / ux, 1.0 / xl
                for i in ti.static(range(MAXM)):
                    if ti.static(i < self.m):
                        p, q = self._pq(ux, xl, ti.cast(self.df[i, j], F64))
                        acc[i] += p * rux + q * rxl
                j += self.NB
            for k in ti.static(range(NACC)):
                self.part[k, t] = 0.0
            for i in ti.static(range(MAXM)):
                self.part[2 + i, t] = acc[i]

    def _gsum_at_x(self):
        self._gsum_part()
        self._reduce()
        return self.red.to_numpy()[2:2 + self.m]

    @ti.func
    def _xstar(self, j, lam):
        """Primal minimiser x_j(lambda) and the quantities the dual needs at it (arithmetic in self.cdt)."""
        x = ti.cast(self.x[j], self.cdt)
        L = ti.cast(self.L[j], self.cdt)
        U = ti.cast(self.U[j], self.cdt)
        ux, xl = U - x, x - L
        P, Q = self._pq(ux, xl, ti.cast(self.df0[j], self.cdt))
        pv = ti.Vector.zero(self.cdt, MAXM)
        qv = ti.Vector.zero(self.cdt, MAXM)
        for i in ti.static(range(MAXM)):
            if ti.static(i < self.m):
                p, q = self._pq(ux, xl, ti.cast(self.df[i, j], self.cdt))
                pv[i] = p
                qv[i] = q
                P += lam[i] * p
                Q += lam[i] * q
        alb, mv = ti.cast(self.cst[3], self.cdt), ti.cast(self.cst[4], self.cdt)
        alpha = ti.max(ti.max(ti.cast(self.cst[5], self.cdt), L + alb * (x - L)), x - mv)
        beta = ti.min(ti.min(ti.cast(self.cst[6], self.cdt), U - alb * (U - x)), x + mv)
        sp, sq = ti.sqrt(P), ti.sqrt(Q)
        xs = (sp * L + sq * U) / (sp + sq)
        inside = 1
        if xs <= alpha:
            xs = alpha
            inside = 0
        elif xs >= beta:
            xs = beta
            inside = 0
        return xs, L, U, P, Q, pv, qv, inside

    @ti.func
    def _lam(self):
        lam = ti.Vector.zero(self.cdt, MAXM)
        for i in ti.static(range(MAXM)):
            lam[i] = ti.cast(self.lam[i], self.cdt)
        return lam

    @ti.kernel
    def _dual_part(self):
        """Shares of phi_x = sum_j P/(U-x)+Q/(x-L), g_i = sum_j p_ij/(U-x)+q_ij/(x-L) and
        H = sum_{x_j free} G G^T / D at x(lambda); float64 throughout."""
        for t in range(self.NB):
            lam = self._lam()
            a_phi = ti.cast(0.0, F64)
            a_free = ti.cast(0.0, F64)
            a_g = ti.Vector.zero(F64, MAXM)
            a_H = ti.Vector.zero(F64, NTRI)
            j = t
            while j < self.n:
                xs, L, U, P, Q, pv, qv, inside = self._xstar(j, lam)
                ru, rl = 1.0 / (U - xs), 1.0 / (xs - L)
                a_phi += P * ru + Q * rl
                for i in ti.static(range(MAXM)):
                    if ti.static(i < self.m):
                        a_g[i] += pv[i] * ru + qv[i] * rl
                if inside == 1:
                    a_free += 1.0
                    ru2, rl2 = ru * ru, rl * rl
                    rD = 1.0 / (2.0 * P * ru2 * ru + 2.0 * Q * rl2 * rl)
                    G = ti.Vector.zero(F64, MAXM)
                    for i in ti.static(range(MAXM)):
                        if ti.static(i < self.m):
                            G[i] = pv[i] * ru2 - qv[i] * rl2
                    for h in ti.static(range(NTRI)):
                        if ti.static(TRI[h][0] < self.m):
                            a_H[h] += G[TRI[h][0]] * G[TRI[h][1]] * rD
                j += self.NB
            self.part[0, t] = a_phi
            self.part[1, t] = a_free
            for i in ti.static(range(MAXM)):
                self.part[2 + i, t] = a_g[i]
            for h in ti.static(range(NTRI)):
                self.part[2 + MAXM + h, t] = a_H[h]

    @ti.kernel
    def _finish(self):
        """x2 <- x1, x1 <- x, x <- x(lambda*)."""
        for j in self.x:
            xs, L, U, P, Q, pv, qv, inside = self._xstar(j, self._lam())
            self.x2[j] = self.x1[j]
            self.x1[j] = self.x[j]
            self.x[j] = ti.cast(xs, self.store)

    # ---------------------------------------------------------------- dual on the host
    def _eval(self, lam):
        lp = np.zeros(MAXM)
        lp[:self.m] = lam
        self.lam.from_numpy(lp)
        self._dual_part()
        self._reduce()
        self.last_passes += 1
        m = self.m
        red = self.red.to_numpy()
        phix = float(red[0])
        self.last_nfree = int(round(red[1]))
        g = red[2:2 + m]
        H = np.zeros((m, m))
        for h, (i, k) in enumerate(TRI):
            if i < m:
                H[i, k] = H[k, i] = red[2 + MAXM + h]
        y = np.maximum(0.0, (lam - self.c) / self.d)
        phi = phix - lam @ self.b + float(np.sum((self.c - lam) * y + 0.5 * self.d * y * y))
        grad = g - self.b - y
        Hd = -H - np.diag((lam > self.c) / self.d)
        return phi, grad, Hd

    def _barrier(self, lam, phi, mu):
        v = -phi - mu * float(np.sum(np.log(lam)))
        if self.has_a:
            v -= mu * math.log(self.a0 - float(self.a @ lam))
        return v

    def _direction(self, lam, grad, Hd, mu):
        """Newton step for the barrier problem  min -phi(lam) - mu sum log lam - mu log(a0 - a.lam)."""
        s0 = self.a0 - float(self.a @ lam) if self.has_a else 1.0
        G = -grad - mu / lam
        Hs = -Hd + np.diag(mu / lam ** 2)
        if self.has_a:
            G = G + mu * self.a / s0
            Hs = Hs + mu * np.outer(self.a, self.a) / s0 ** 2
        try:
            dl = -np.linalg.solve(Hs, G)
        except np.linalg.LinAlgError:
            # Hs exactly singular in floating point (fix of 06/11/2026): in min-max problems the barrier term of
            # a.lam < a0 grows as mu a a^T / s0^2 and, when the realisations coincide (a design with no load path),
            # the rest of the block is lost in rounding. Minimum-norm Newton step; never taken when Hs is regular,
            # so runs that never hit this case are unchanged.
            dl = -np.linalg.lstsq(Hs, G, rcond=None)[0]
            self.singular_steps = getattr(self, "singular_steps", 0) + 1
        return dl, float(-G @ dl), s0

    def _solve_dual(self):
        """Barrier method on the m multipliers. mu falls tenfold whenever the iterate is centred (Newton decrement
        squared <= 0.1 mu), possibly several times without a new pass; one GPU pass per trial point."""
        m = self.m
        self.has_a = bool(np.any(self.a > 0.0))
        warm = self.lam_prev is not None
        lam = np.maximum(self.lam_prev, 1e-6) if warm else np.ones(m)
        if self.has_a and float(self.a @ lam) >= 0.9 * self.a0:
            lam = lam * (0.9 * self.a0 / float(self.a @ lam))
        phi, grad, Hd = self._eval(lam)
        scale = max(1.0, abs(phi))
        mu = (self.mu_warm if warm else self.mu0) * scale
        mu_end = self.mu_min * scale
        newton = 0
        for _ in range(300):
            dl, dec2, s0 = self._direction(lam, grad, Hd, mu)
            while dec2 <= 0.1 * mu and mu > mu_end:
                mu = max(0.1 * mu, mu_end)
                dl, dec2, s0 = self._direction(lam, grad, Hd, mu)
            if mu <= mu_end and dec2 <= self.newton_tol * scale:
                break
            t = 1.0
            neg = dl < 0.0
            if np.any(neg):
                t = min(t, 0.99 * float(np.min(-lam[neg] / dl[neg])))
            if self.has_a:
                ad = float(self.a @ dl)
                if ad > 0.0:
                    t = min(t, 0.99 * s0 / ad)
            psi0 = self._barrier(lam, phi, mu)
            accepted = False
            for _halving in range(30):
                trial = lam + t * dl
                phi_t, grad_t, Hd_t = self._eval(trial)
                if self._barrier(trial, phi_t, mu) <= psi0 - 1e-4 * t * dec2:
                    accepted = True
                    break
                t *= 0.5
            newton += 1
            if not accepted:
                # no decrease within the rounding of the sums: lambda is as good as this mu allows
                if mu <= mu_end:
                    break
                mu = max(0.1 * mu, mu_end)
                continue
            lam, phi, grad, Hd = trial, phi_t, grad_t, Hd_t
        self.last_newton = newton
        return lam

    def step(self, x, f0, df0, f, df):
        n, m = self.n, self.m
        f = np.atleast_1d(np.asarray(f, dtype=np.float64))
        df = np.atleast_2d(np.asarray(df))
        self.x.from_numpy(np.ascontiguousarray(x, dtype=self.npstore))
        self.df0.from_numpy(np.ascontiguousarray(df0, dtype=self.npstore))
        self.df.from_numpy(np.ascontiguousarray(df.reshape(m, n), dtype=self.npstore))
        self.last_passes = 0
        self._asymptotes(1 if self.k < 2 else 0)
        self.b = self._gsum_at_x() - f
        lam = self._solve_dual()
        self.lam_prev = lam.copy()
        lp = np.zeros(MAXM)
        lp[:m] = lam
        self.lam.from_numpy(lp)
        self._finish()
        self.k += 1
        self.last_lambda = lam
        return self.x.to_numpy().astype(np.float64)
