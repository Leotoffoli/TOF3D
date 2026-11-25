# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Leonilson Toffoli Pedrini
"""TOF3D — Method of Moving Asymptotes (MMA) in numpy.

Problem (Svanberg 1987; Svanberg 2007):

    minimize    f0(x) + a0*z + sum_i (c_i*y_i + 0.5*d_i*y_i**2)
    subject to  f_i(x) - a_i*z - y_i <= 0,        i = 1..m
                xmin <= x <= xmax,   y >= 0,   z >= 0

Each outer iteration replaces f0 and f_i by separable convex approximations built around the current point
with moving asymptotes L < x < U, and solves the resulting subproblem with a primal-dual interior-point
(Newton) method. In every Newton step the design variables x and the elastic variables y are eliminated
analytically, so the step costs O(m*n) operations plus one dense (m+1)x(m+1) solve for the multipliers and z.

The module was written from the equations in the two references below; it does not reuse code from any other
MMA implementation. The numerical constants are the values recommended by Svanberg (2007).

    K. Svanberg (1987) The method of moving asymptotes - a new method for structural optimization.
        International Journal for Numerical Methods in Engineering 24:359-373. doi:10.1002/nme.1620240207
    K. Svanberg (2007) MMA and GCMMA - two methods for nonlinear optimization. Technical note, KTH, Stockholm.
"""
import numpy as np

ASY_INIT = 0.5      # first two iterations: distance of the asymptotes, as a fraction of xmax - xmin
ASY_INCR = 1.2      # asymptotes move away when a variable keeps the same direction
ASY_DECR = 0.7      # asymptotes move closer when a variable oscillates
ASY_MIN = 0.01      # bounds on |x - asymptote|, as fractions of xmax - xmin
ASY_MAX = 10.0
ALBEFA = 0.1        # the subproblem box stays this fraction away from the asymptotes
RAA0 = 1e-5         # regularisation of the approximations


class MMA:
    """MMA optimizer with memory of the last two iterates.

    opt = MMA(n, m, xmin, xmax, move=0.2, a0=1.0, a=None, c=None, d=None)
    x_new = opt.step(x, f0, df0, f, df)     # f: (m,), df: (m, n)
    opt.reset()                             # restart the asymptotes (e.g. after a continuation step)

    Defaults a = 0, c = 1000, d = 1 give the usual form "minimize f0 subject to f_i <= 0" with elastic y.
    """

    def __init__(self, n, m, xmin, xmax, move=0.2, a0=1.0, a=None, c=None, d=None):
        self.n, self.m = int(n), int(m)
        self.xmin = np.asarray(xmin, dtype=np.float64).copy()
        self.xmax = np.asarray(xmax, dtype=np.float64).copy()
        if self.xmin.shape != (self.n,) or self.xmax.shape != (self.n,):
            raise ValueError("xmin and xmax must have shape (n,)")
        self.move = float(move)
        self.a0 = float(a0)
        self.a = np.zeros(self.m) if a is None else np.asarray(a, dtype=np.float64)
        self.c = np.full(self.m, 1000.0) if c is None else np.asarray(c, dtype=np.float64)
        self.d = np.ones(self.m) if d is None else np.asarray(d, dtype=np.float64)
        if np.any(self.d <= 0.0):
            raise ValueError("all d_i must be positive")
        if self.a0 <= 0.0:
            raise ValueError("a0 must be positive")
        self.last_newton = 0
        self.reset()

    def reset(self):
        """Forget the previous iterates; the next two steps use the initial asymptotes."""
        self.prev = []
        self.L = None
        self.U = None

    # ---------------------------------------------------------------- approximation
    def _asymptotes(self, x):
        span = self.xmax - self.xmin
        if len(self.prev) < 2:
            return x - ASY_INIT * span, x + ASY_INIT * span
        x1, x2 = self.prev[-1], self.prev[-2]
        trend = (x - x1) * (x1 - x2)
        gamma = np.ones_like(x)
        gamma[trend > 0.0] = ASY_INCR
        gamma[trend < 0.0] = ASY_DECR
        L = np.clip(x - gamma * (x1 - self.L), x - ASY_MAX * span, x - ASY_MIN * span)
        U = np.clip(x + gamma * (self.U - x1), x + ASY_MIN * span, x + ASY_MAX * span)
        return L, U

    def _pq(self, x, L, U, grad):
        """Coefficients of the convex approximation for gradient rows `grad` (k x n)."""
        span = self.xmax - self.xmin
        gp = np.maximum(grad, 0.0)
        gm = np.maximum(-grad, 0.0)
        reg = RAA0 / np.maximum(span, 1e-12)
        p = (U - x) ** 2 * (1.001 * gp + 0.001 * gm + reg)
        q = (x - L) ** 2 * (0.001 * gp + 1.001 * gm + reg)
        return p, q

    def step(self, x, f0, df0, f, df):
        x = np.asarray(x, dtype=np.float64)
        f = np.atleast_1d(np.asarray(f, dtype=np.float64))
        df = np.atleast_2d(np.asarray(df, dtype=np.float64))
        df0 = np.asarray(df0, dtype=np.float64)
        span = self.xmax - self.xmin
        L, U = self._asymptotes(x)
        alpha = np.maximum.reduce([self.xmin, L + ALBEFA * (x - L), x - self.move * span])
        beta = np.minimum.reduce([self.xmax, U - ALBEFA * (U - x), x + self.move * span])
        p0, q0 = self._pq(x, L, U, df0[None, :])
        p, q = self._pq(x, L, U, df)
        b = p @ (1.0 / (U - x)) + q @ (1.0 / (x - L)) - f
        sol = solve_subproblem(L, U, alpha, beta, p0[0], q0[0], p, q, b, self.a0, self.a, self.c, self.d)
        self.last_newton = sol["newton"]
        self.L, self.U = L, U
        self.prev = (self.prev + [x.copy()])[-2:]
        return sol["x"]


# -------------------------------------------------------------------- subproblem
def _kkt(v, eps, data):
    """Residuals of the perturbed optimality conditions of the subproblem, and shared intermediate terms."""
    x, y, z, lam, xsi, eta, mu, zet, s = v
    L, U, alpha, beta, p0, q0, p, q, b, a0, a, c, d = data
    ux, xl = U - x, x - L
    P = p0 + lam @ p
    Q = q0 + lam @ q
    dpsi = P / ux ** 2 - Q / xl ** 2
    g = p @ (1.0 / ux) + q @ (1.0 / xl)
    r = [dpsi - xsi + eta,                      # stationarity in x
         c + d * y - lam - mu,                  # stationarity in y
         np.atleast_1d(a0 - a @ lam - zet),     # stationarity in z
         g - a * z - y + s - b,                 # constraints with slack s
         xsi * (x - alpha) - eps,               # complementarity
         eta * (beta - x) - eps,
         mu * y - eps,
         np.atleast_1d(zet * z - eps),
         lam * s - eps]
    return np.concatenate(r), (ux, xl, P, Q, dpsi, g)


def _fraction_to_boundary(pairs):
    t = np.inf
    for v, dv in pairs:
        neg = dv < 0.0
        if np.any(neg):
            t = min(t, float(np.min(-v[neg] / dv[neg])))
    return t


def solve_subproblem(L, U, alpha, beta, p0, q0, p, q, b, a0, a, c, d, eps_min=1e-7, max_newton=200):
    """Primal-dual interior-point solution of the MMA subproblem.

    The barrier parameter eps decreases tenfold from 1 to eps_min; for each value Newton steps are taken until
    the largest residual falls below 0.9*eps. Step lengths keep all bounded quantities strictly positive
    (fraction to the boundary 0.99) and are halved until the residual norm decreases.
    """
    m = b.size
    data = (L, U, alpha, beta, p0, q0, p, q, b, a0, a, c, d)
    x = 0.5 * (alpha + beta)
    y = np.ones(m)
    z = 1.0
    lam = np.ones(m)
    s = np.ones(m)
    xsi = np.maximum(1.0, 1.0 / (x - alpha))
    eta = np.maximum(1.0, 1.0 / (beta - x))
    mu = np.maximum(1.0, 0.5 * c)
    zet = 1.0
    v = [x, y, np.float64(z), lam, xsi, eta, mu, np.float64(zet), s]
    newton = 0
    eps = 1.0
    while True:
        for _ in range(max_newton):
            res, (ux, xl, P, Q, dpsi, g) = _kkt(v, eps, data)
            if np.max(np.abs(res)) < 0.9 * eps:
                break
            x, y, z, lam, xsi, eta, mu, zet, s = v
            xa, bx = x - alpha, beta - x
            # diagonal blocks after eliminating the bound multipliers and slacks
            Dx = 2.0 * P / ux ** 3 + 2.0 * Q / xl ** 3 + xsi / xa + eta / bx
            hx = -dpsi + eps / xa - eps / bx
            Dy = d + mu / y
            hy = -(c + d * y - lam) + eps / y
            Dz = zet / z
            hz = -(a0 - a @ lam) + eps / z
            hl = -(g - a * z - y - b) - eps / lam
            G = p / ux ** 2 - q / xl ** 2                       # d g_i / d x_j
            GD = G / Dx
            K = np.empty((m + 1, m + 1))
            K[:m, :m] = GD @ G.T + np.diag(1.0 / Dy + s / lam)
            K[:m, m] = a
            K[m, :m] = -a
            K[m, m] = Dz
            rhs = np.concatenate([G @ (hx / Dx) - hy / Dy - hl, [hz]])
            sol = np.linalg.solve(K, rhs)
            dlam, dz = sol[:m], sol[m]
            dx = (hx - G.T @ dlam) / Dx
            dy = (hy + dlam) / Dy
            dxsi = eps / xa - xsi - xsi / xa * dx
            deta = eps / bx - eta + eta / bx * dx
            dmu = eps / y - mu - mu / y * dy
            dzet = eps / z - zet - zet / z * dz
            ds = eps / lam - s - s / lam * dlam
            dv = [dx, dy, dz, dlam, dxsi, deta, dmu, dzet, ds]
            tmax = _fraction_to_boundary([(xa, dx), (bx, -dx), (y, dy), (np.atleast_1d(z), np.atleast_1d(dz)),
                                          (lam, dlam), (xsi, dxsi), (eta, deta), (mu, dmu),
                                          (np.atleast_1d(zet), np.atleast_1d(dzet)), (s, ds)])
            t = min(1.0, 0.99 * tmax)
            norm0 = np.linalg.norm(res)
            for _halving in range(50):
                trial = [vi + t * di for vi, di in zip(v, dv)]
                if np.linalg.norm(_kkt(trial, eps, data)[0]) < norm0:
                    break
                t *= 0.5
            v = trial
            newton += 1
        if eps <= eps_min * 1.0001:
            break
        eps *= 0.1
    x, y, z, lam = v[0], v[1], float(v[2]), v[3]
    return {"x": x, "y": y, "z": z, "lam": lam, "newton": newton}
