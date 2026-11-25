# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Leonilson Toffoli Pedrini
"""TOF3D — geometric multigrid preconditioner for the matrix-free Hex8 operator of the TOF3D core (FE class).

Fine level (level 0): the matrix-free operator of FE, extended on the supports,
    A0 = F (K + S) F + W,
where F is the diagonal mask of free degrees of freedom, S the springs and W a diagonal penalty equal to
`pen_support` on fixed degrees of freedom of nodes that touch an active element (true supports) and zero on
nodes outside the active domain. Vectors of the conjugate gradient method live in the free subspace, so W only
shapes the coarse operators: it keeps them positive definite near supports without stiffening the boundary of
removed regions. Coarse rows that are exactly zero (outside the active domain) are replaced by the identity.

Removed regions (nodes that touch no element with E > 0, e.g. the cut-out of an L-shaped domain): the coarse
operators are built from an auxiliary fine operator in which the removed elements have the small modulus
`void_eps` and their nodes are free, so that no coarse cell is cut by the boundary of the domain. The conjugate
gradient method and the fine-level smoother keep the true operator (E = 0 outside the domain, those nodes fixed), so
the problem solved is unchanged; only the preconditioner differs. The V-cycle stays symmetric positive definite on
the free subspace (convergent smoother, symmetric positive definite coarse operators). Without removed regions the
auxiliary operator is the true one and nothing changes. (Amendment 3, 06/11/2026: in the L-shaped example the
cut coarse cells cost about 140 iterations per solve against 20 for the full box.)

Coarse levels l >= 1: Galerkin operators A_l = P_l^T A_{l-1} P_l with trilinear prolongation P_l on a grid with
half the elements in every direction. A_l is a 27-point stencil of 3x3 blocks; it is built exactly by probing
(27 colours x 3 components) with the operator of the level above, and only the 14 blocks of one half of the
stencil are stored (A[I, I-o] = A[I-o, I]^T).

Smoother: damped 3x3 block Jacobi, nu sweeps before and after the coarse correction (symmetric V-cycle).
Coarsest level: dense Cholesky factorisation on the host in float64.

Usage:
    mg = MG(fe)              # after fe.set_bc
    mg.setup()               # after every fe.set_E (new design)
    x, it, rel = mg.solve(b, x0, tol)
"""
import itertools
import math
import time

import numpy as np
import scipy.linalg as sla
import taichi as ti

import tof3d_core as C

REAL, ACC, NPR = C.REAL, C.ACC, C.NPR
NX_LUT, NY_LUT, NZ_LUT, LOCAL_LUT = C.NX_LUT, C.NY_LUT, C.NZ_LUT, C.LOCAL_LUT

# half stencil: centre first, then the 13 offsets (ox, oy, oz) with (oz, oy, ox) > (0, 0, 0)
HALF = [(0, 0, 0)] + sorted([o for o in itertools.product((-1, 0, 1), repeat=3) if (o[2], o[1], o[0]) > (0, 0, 0)],
                            key=lambda o: (o[2], o[1], o[0]))
HX = tuple(o[0] for o in HALF)
HY = tuple(o[1] for o in HALF)
HZ = tuple(o[2] for o in HALF)
NH = len(HALF)  # 14
CODE2HALF = [-1] * 27   # (ox+1) + 3 (oy+1) + 9 (oz+1)  ->  index in HALF, or -1 for the other half
for h, o in enumerate(HALF):
    CODE2HALF[(o[0] + 1) + 3 * (o[1] + 1) + 9 * (o[2] + 1)] = h


@ti.data_oriented
class MG:
    def __init__(self, fe, omega=0.6, nu=2, max_coarse_dof=4500, pen_support=1.0, void_eps=1e-6):
        self.fe = fe
        self.void_eps = float(void_eps)
        self.Ke = fe.Ke
        self.E, self.free, self.spring, self.dinv0 = fe.E, fe.free, fe.spring, fe.dinv
        self.omega, self.nu = float(omega), int(nu)
        self.pen_support = float(pen_support)
        shapes = [tuple(fe.eshape)]
        while True:
            ex, ey, ez = shapes[-1]
            if (ex + 1) * (ey + 1) * (ez + 1) * 3 <= max_coarse_dof or ex % 2 or ey % 2 or ez % 2:
                break
            shapes.append((ex // 2, ey // 2, ez // 2))
        if len(shapes) < 2:
            raise ValueError("grid cannot be coarsened (odd number of elements or already small)")
        self.shapes = shapes
        self.L = len(shapes)
        V3 = lambda ns: ti.Vector.field(3, REAL, ns)
        ns0 = fe.nshape
        self.t0, self.pen = V3(ns0), V3(ns0)
        self.outside = ti.field(ti.u8, ns0)     # 1 on nodes that touch no element with E > 0
        self.n_out = ti.field(ti.i32, ())
        self.aux_used = False
        self.S, self.Dc, self.r, self.e, self.t = [None], [None], [None], [None], [None]
        for l in range(1, self.L):
            ns = tuple(s + 1 for s in shapes[l])
            self.S.append(ti.Matrix.field(3, 3, REAL, ns + (NH,)))
            self.Dc.append(ti.Matrix.field(3, 3, REAL, ns))
            self.r.append(V3(ns)); self.e.append(V3(ns)); self.t.append(V3(ns))
        self.code2half = ti.field(ti.i32, 27)
        self.code2half.from_numpy(np.array(CODE2HALF, dtype=np.int32))
        self.s_a = ti.field(ACC, ())
        self.s_b = ti.field(ACC, ())
        self.s_ff = ti.field(ACC, ())
        self.s_rz = ti.field(ACC, ())
        self.chol = None
        self.n_solves = 0
        self.n_it = 0
        self.t_solve = 0.0
        self.t_setup = 0.0

    # ------------------------------------------------------------------ operators
    def apply_fine(self, src, dst, aux=0):
        """dst = F (K + S) F src + W src on the fine grid; aux = 1: the auxiliary operator of the removed regions
        (E = void_eps on elements with E = 0, nodes outside the domain free), used only to build the coarse levels."""
        self._apply_fine(src, dst, aux)

    @ti.kernel
    def _apply_fine(self, src: ti.template(), dst: ti.template(), aux: ti.template()):
        for i, j, k in dst:
            res = ti.Vector([0.0, 0.0, 0.0])
            fr = self.free[i, j, k]
            if ti.static(aux == 1):
                if self.outside[i, j, k] == 1:
                    fr = ti.Vector([1.0, 1.0, 1.0], REAL)
            si = src[i, j, k] * fr
            for ddx in ti.static(range(2)):
                for ddy in ti.static(range(2)):
                    for ddz in ti.static(range(2)):
                        ex, ey, ez = i - ddx, j - ddy, k - ddz
                        if 0 <= ex < self.fe.nx and 0 <= ey < self.fe.ny and 0 <= ez < self.fe.nz:
                            Ee = self.E[ex, ey, ez]
                            if ti.static(aux == 1):
                                if Ee <= 0.0:
                                    Ee = ti.cast(self.void_eps, REAL)
                            ln = ti.static(LOCAL_LUT[ddz][ddy][ddx])
                            for n in ti.static(range(8)):
                                if ti.static(n != ln):
                                    ni, nj, nk = ex + NX_LUT[n], ey + NY_LUT[n], ez + NZ_LUT[n]
                                    frn = self.free[ni, nj, nk]
                                    if ti.static(aux == 1):
                                        if self.outside[ni, nj, nk] == 1:
                                            frn = ti.Vector([1.0, 1.0, 1.0], REAL)
                                    # Emenda 8: diferenças do vetor mascarado (as linhas de Ke somam zero)
                                    uv = src[ni, nj, nk] * frn - si
                                    for rr in ti.static(range(3)):
                                        tt = 0.0
                                        for cc in ti.static(range(3)):
                                            kv = ti.static(self.Ke[(ln * 3 + rr) * 24 + n * 3 + cc])
                                            if ti.static(kv != 0.0):
                                                tt += kv * uv[cc]
                                        res[rr] += tt * Ee
            sv = src[i, j, k]
            res += self.spring[i, j, k] * sv * fr
            dst[i, j, k] = res * fr + self.pen[i, j, k] * sv

    @ti.kernel
    def apply_stencil(self, S: ti.template(), src: ti.template(), dst: ti.template()):
        """dst = A src for a coarse level stored as a half 27-point stencil."""
        for i, j, k in dst:
            res = S[i, j, k, 0] @ src[i, j, k]
            for h in ti.static(range(1, NH)):
                ox, oy, oz = ti.static(HX[h]), ti.static(HY[h]), ti.static(HZ[h])
                a, b, c = i + ox, j + oy, k + oz
                if 0 <= a < dst.shape[0] and 0 <= b < dst.shape[1] and 0 <= c < dst.shape[2]:
                    res += S[i, j, k, h] @ src[a, b, c]
                a2, b2, c2 = i - ox, j - oy, k - oz
                if 0 <= a2 < dst.shape[0] and 0 <= b2 < dst.shape[1] and 0 <= c2 < dst.shape[2]:
                    res += S[a2, b2, c2, h].transpose() @ src[a2, b2, c2]
            dst[i, j, k] = res

    def apply(self, l, src, dst):
        if l == 0:
            self.apply_fine(src, dst)
        else:
            self.apply_stencil(self.S[l], src, dst)

    # ------------------------------------------------------------------ transfers
    @ti.kernel
    def prolong(self, ec: ti.template(), ef: ti.template(), mode: ti.template()):
        """mode 0: ef = P ec;  mode 1: ef += F P ec (fine level);  mode 2: ef += P ec."""
        for i, j, k in ef:
            acc = ti.Vector([0.0, 0.0, 0.0])
            for a in ti.static(range(2)):
                for b in ti.static(range(2)):
                    for c in ti.static(range(2)):
                        wa = 0.5 if i % 2 == 1 else (1.0 if a == 0 else 0.0)
                        wb = 0.5 if j % 2 == 1 else (1.0 if b == 0 else 0.0)
                        wc = 0.5 if k % 2 == 1 else (1.0 if c == 0 else 0.0)
                        w = wa * wb * wc
                        if w != 0.0:
                            acc += w * ec[(i + a) // 2, (j + b) // 2, (k + c) // 2]
            if ti.static(mode == 0):
                ef[i, j, k] = acc
            elif ti.static(mode == 1):
                ef[i, j, k] += acc * self.free[i, j, k]
            else:
                ef[i, j, k] += acc

    @ti.kernel
    def restrict(self, ef: ti.template(), ec: ti.template()):
        """ec = P^T ef (full weighting)."""
        for I, J, K in ec:
            acc = ti.Vector([0.0, 0.0, 0.0])
            for dx in ti.static(range(-1, 2)):
                for dy in ti.static(range(-1, 2)):
                    for dz in ti.static(range(-1, 2)):
                        a, b, c = 2 * I + dx, 2 * J + dy, 2 * K + dz
                        w = ti.static((1.0 if dx == 0 else 0.5) * (1.0 if dy == 0 else 0.5) * (1.0 if dz == 0 else 0.5))
                        if 0 <= a < ef.shape[0] and 0 <= b < ef.shape[1] and 0 <= c < ef.shape[2]:
                            acc += w * ef[a, b, c]
            ec[I, J, K] = acc

    # ------------------------------------------------------------------ smoother pieces
    @ti.kernel
    def smooth_first(self, Dinv: ti.template(), r: ti.template(), e: ti.template()):
        for I in ti.grouped(e):
            e[I] = self.omega * (Dinv[I] @ r[I])

    @ti.kernel
    def smooth(self, Dinv: ti.template(), r: ti.template(), t: ti.template(), e: ti.template()):
        for I in ti.grouped(e):
            e[I] += self.omega * (Dinv[I] @ (r[I] - t[I]))

    @ti.kernel
    def residual_inplace(self, r: ti.template(), t: ti.template()):
        for I in ti.grouped(t):
            t[I] = r[I] - t[I]

    # ------------------------------------------------------------------ setup by probing
    @ti.kernel
    def set_probe(self, e: ti.template(), cx: ti.i32, cy: ti.i32, cz: ti.i32, comp: ti.i32):
        for i, j, k in e:
            v = ti.Vector([0.0, 0.0, 0.0])
            if i % 3 == cx and j % 3 == cy and k % 3 == cz:
                v[comp] = 1.0
            e[i, j, k] = v

    @ti.kernel
    def harvest(self, S: ti.template(), y: ti.template(), cx: ti.i32, cy: ti.i32, cz: ti.i32, comp: ti.i32):
        for i, j, k in y:
            ox = (cx - i % 3 + 3) % 3
            oy = (cy - j % 3 + 3) % 3
            oz = (cz - k % 3 + 3) % 3
            ox = -1 if ox == 2 else ox
            oy = -1 if oy == 2 else oy
            oz = -1 if oz == 2 else oz
            h = self.code2half[(ox + 1) + 3 * (oy + 1) + 9 * (oz + 1)]
            a, b, c = i + ox, j + oy, k + oz
            if h >= 0 and 0 <= a < y.shape[0] and 0 <= b < y.shape[1] and 0 <= c < y.shape[2]:
                yv = y[i, j, k]
                for rr in ti.static(range(3)):
                    S[i, j, k, h][rr, comp] = yv[rr]

    @ti.kernel
    def zero_stencil(self, S: ti.template()):
        for I in ti.grouped(S):
            S[I] = ti.Matrix.zero(REAL, 3, 3)

    @ti.kernel
    def build_dinv_c(self, S: ti.template(), Dc: ti.template()):
        for i, j, k in Dc:
            Dc[i, j, k] = S[i, j, k, 0].inverse()

    @ti.kernel
    def build_pen(self, w: REAL):
        self.n_out[None] = 0
        for i, j, k in self.pen:
            touch = 0
            for ddx in ti.static(range(2)):
                for ddy in ti.static(range(2)):
                    for ddz in ti.static(range(2)):
                        ex, ey, ez = i - ddx, j - ddy, k - ddz
                        if 0 <= ex < self.fe.nx and 0 <= ey < self.fe.ny and 0 <= ez < self.fe.nz:
                            if self.E[ex, ey, ez] > 0.0:
                                touch = 1
            self.pen[i, j, k] = (1.0 - self.free[i, j, k]) * (w if touch == 1 else 0.0)
            self.outside[i, j, k] = ti.cast(1 - touch, ti.u8)
            if touch == 0:
                ti.atomic_add(self.n_out[None], 1)

    @ti.kernel
    def fix_zero_rows(self, S: ti.template()):
        """Rows of a coarse operator that are exactly zero belong to nodes outside the active domain: identity."""
        for i, j, k in ti.ndrange(S.shape[0], S.shape[1], S.shape[2]):
            for rr in ti.static(range(3)):
                if S[i, j, k, 0][rr, rr] == 0.0:
                    S[i, j, k, 0][rr, rr] = 1.0

    def setup(self):
        """Build every coarse operator for the current E field (call after fe.set_E)."""
        t0 = time.time()
        self.build_pen(self.pen_support)
        aux = 1 if (self.void_eps > 0.0 and int(self.n_out[None]) > 0) else 0
        self.aux_used = aux == 1
        for l in range(1, self.L):
            S = self.S[l]
            self.zero_stencil(S)
            if l == 1:
                src, dst = self.fe.d, self.t0      # the CG vectors of FE are free while the coarse levels are built
            else:
                src, dst = self.e[l - 1], self.t[l - 1]
            for cz, cy, cx in itertools.product(range(3), repeat=3):
                for comp in range(3):
                    self.set_probe(self.e[l], cx, cy, cz, comp)
                    self.prolong(self.e[l], src, 0)
                    if l == 1:
                        self.apply_fine(src, dst, aux)
                    else:
                        self.apply(l - 1, src, dst)
                    self.restrict(dst, self.r[l])
                    self.harvest(S, self.r[l], cx, cy, cz, comp)
            self.fix_zero_rows(S)
            self.build_dinv_c(S, self.Dc[l])
        self.chol = sla.cho_factor(self.dense(self.L - 1), lower=True, check_finite=False)
        self.t_setup += time.time() - t0

    def dense(self, l):
        """Dense matrix of level l (float64), assembled from the half stencil."""
        S = self.S[l].to_numpy().astype(np.float64)        # (nx, ny, nz, 14, 3, 3)
        ns = S.shape[:3]
        N = int(np.prod(ns))
        A = np.zeros((3 * N, 3 * N))
        idx = np.arange(N).reshape(ns)
        I, J, K = np.meshgrid(*[np.arange(s) for s in ns], indexing="ij")
        for h, (ox, oy, oz) in enumerate(HALF):
            a, b, c = I + ox, J + oy, K + oz
            ok = (a >= 0) & (a < ns[0]) & (b >= 0) & (b < ns[1]) & (c >= 0) & (c < ns[2])
            rows = idx[ok]
            cols = idx[a[ok], b[ok], c[ok]]
            blk = S[..., h, :, :][ok]
            for p in range(3):
                for q in range(3):
                    A[3 * rows + p, 3 * cols + q] += blk[:, p, q]
                    if h > 0:
                        A[3 * cols + q, 3 * rows + p] += blk[:, p, q]
        return A

    # ------------------------------------------------------------------ V-cycle
    def _level(self, l):
        if l == 0:
            return self.fe.r, self.fe.z, self.t0, self.dinv0
        return self.r[l], self.e[l], self.t[l], self.Dc[l]

    def vcycle(self, l=0):
        if l == self.L - 1:
            rhs = self.r[l].to_numpy().astype(np.float64).reshape(-1)
            sol = sla.cho_solve(self.chol, rhs, check_finite=False)
            self.e[l].from_numpy(sol.reshape(self.e[l].shape + (3,)).astype(NPR))
            return
        r, e, t, Dinv = self._level(l)
        for s in range(self.nu):
            if s == 0:
                self.smooth_first(Dinv, r, e)
            else:
                self.apply(l, e, t)
                self.smooth(Dinv, r, t, e)
        self.apply(l, e, t)
        self.residual_inplace(r, t)
        self.restrict(t, self.r[l + 1])
        self.vcycle(l + 1)
        self.prolong(self.e[l + 1], e, 1 if l == 0 else 2)
        for _ in range(self.nu):
            self.apply(l, e, t)
            self.smooth(Dinv, r, t, e)

    # ------------------------------------------------------------------ preconditioned CG
    @ti.kernel
    def _start(self):
        self.s_ff[None] = 0.0
        self.s_b[None] = 0.0
        for I in ti.grouped(self.fe.r):
            fr = self.free[I]
            bb = self.fe.b[I] * fr
            rv = (bb - self.fe.q[I]) * fr
            self.fe.r[I] = rv
            self.s_ff[None] += ti.cast(bb.dot(bb), ACC)
            self.s_b[None] += ti.cast(rv.dot(rv), ACC)

    @ti.kernel
    def _rz(self):
        self.s_a[None] = 0.0
        for I in ti.grouped(self.fe.r):
            self.s_a[None] += ti.cast(self.fe.r[I].dot(self.fe.z[I]), ACC)

    @ti.kernel
    def _dir(self, beta: ACC):
        for I in ti.grouped(self.fe.d):
            self.fe.d[I] = self.fe.z[I] + ti.cast(beta, REAL) * self.fe.d[I]

    @ti.kernel
    def _update(self, alpha: ACC):
        self.s_b[None] = 0.0
        al = ti.cast(alpha, REAL)
        for I in ti.grouped(self.fe.xs):
            self.fe.xs[I] += al * self.fe.d[I]
            rv = self.fe.r[I] - al * self.fe.q[I]
            self.fe.r[I] = rv
            self.s_b[None] += ti.cast(rv.dot(rv), ACC)

    def solve(self, b, x0=None, tol=1e-5, maxit=1000):
        """Solve K x = b with multigrid-preconditioned CG. Returns (x, iterations, relative residual)."""
        fe = self.fe
        t0 = time.time()
        fe.b.from_numpy(np.ascontiguousarray(b, dtype=NPR))
        if x0 is None:
            fe.xs.fill(0.0)
        else:
            fe.xs.from_numpy(np.ascontiguousarray(x0, dtype=NPR))
        fe._x_to_d()
        fe._apply_K()
        self._start()
        ff = float(self.s_ff[None])
        if ff <= 0.0:
            fe.xs.fill(0.0)
            return fe.xs.to_numpy().astype(np.float64), 0, 0.0
        rel = math.sqrt(max(float(self.s_b[None]), 0.0) / ff)
        it = 0
        if rel > tol:
            self.vcycle()
            self._rz()
            rz = float(self.s_a[None])
            self._dir(0.0)
            while it < maxit:
                fe._apply_K()
                fe._cg_dq()
                dq = float(fe.s_dq[None])
                self._update(rz / dq if dq != 0.0 else 0.0)
                it += 1
                rel = math.sqrt(max(float(self.s_b[None]), 0.0) / ff)
                if rel <= tol:
                    break
                self.vcycle()
                self._rz()
                rzn = float(self.s_a[None])
                self._dir(rzn / rz if rz != 0.0 else 0.0)
                rz = rzn
        x = fe.xs.to_numpy().astype(np.float64)
        self.n_solves += 1
        self.n_it += it
        self.t_solve += time.time() - t0
        return x, it, rel
