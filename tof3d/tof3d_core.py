# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Leonilson Toffoli Pedrini
"""TOF3D core: matrix-free Hex8 finite elements on the GPU (Taichi), density filter, Heaviside projection and the
responses and adjoint sensitivities used by the examples.

Physics: trilinear 8-node hexahedron with full (2x2x2) integration, unit cube elements, E0 = 1, Poisson ratio nu.
The product K u is computed without assembling K, by gathering at each node the contributions of its 8 elements.
Rectangular grid nx x ny x nz with a per-element Young modulus E (zero outside the active domain), a mask of free
degrees of freedom and optional springs. Solvers: CG with 3x3 block Jacobi preconditioner (FE.solve) or with the
geometric multigrid of tof3d_mg (Resp with mg=...).

Python side (numpy/scipy): density filter, Heaviside projection (one or three realisations), compliance,
mechanism output displacement, p-norm of the relaxed von Mises stress, local volume (p-norm) and their gradients.

Precision: TOF3D_FP64=1 runs everything in float64 (gradient checks); otherwise float32 on the GPU with float64
reductions. The element products use displacement differences inside each element (Emenda 8, 07/11/2026): the rows
of Ke sum to zero, so rigid translations never pass through the rounded single-precision element matrix.
"""
import math
import os
import time

import numpy as np
import taichi as ti
from scipy import ndimage, signal

FP64 = os.environ.get("TOF3D_FP64", "0") == "1"
REAL = ti.f64 if FP64 else ti.f32
NPR = np.float64 if FP64 else np.float32
ACC = ti.f64

NX_LUT = (0, 1, 1, 0, 0, 1, 1, 0)
NY_LUT = (0, 0, 1, 1, 0, 0, 1, 1)
NZ_LUT = (0, 0, 0, 0, 1, 1, 1, 1)
LOCAL_LUT = (((0, 1), (3, 2)), ((4, 5), (7, 6)))  # [dz][dy][dx] -> nó local
NODE_COORDS = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)


def init(arch="cuda", mem_gb=2.0):
    kw = dict(default_fp=REAL, random_seed=0, offline_cache=True, log_level=ti.WARN)
    if arch == "cuda":
        kw["device_memory_GB"] = mem_gb
    ti.init(arch=getattr(ti, arch), **kw)


# ---------------------------------------------------------------- elemento H8 (igual ao da produção)
def D_matrix(nu=0.3):
    D = np.zeros((6, 6))
    c = 1.0 / ((1 + nu) * (1 - 2 * nu))
    D[0, 0] = D[1, 1] = D[2, 2] = c * (1 - nu)
    D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = c * nu
    D[3, 3] = D[4, 4] = D[5, 5] = c * 0.5 * (1 - 2 * nu)
    return D


def B_at(xi, eta, zeta):
    dN = np.zeros((8, 3))
    for i in range(8):
        s = NODE_COORDS[i]
        dN[i, 0] = 0.125 * s[0] * (1 + s[1] * eta) * (1 + s[2] * zeta)
        dN[i, 1] = 0.125 * s[1] * (1 + s[0] * xi) * (1 + s[2] * zeta)
        dN[i, 2] = 0.125 * s[2] * (1 + s[0] * xi) * (1 + s[1] * eta)
    dN *= 2.0  # elemento unitário
    B = np.zeros((6, 24))
    for i in range(8):
        B[0, 3 * i] = dN[i, 0]
        B[1, 3 * i + 1] = dN[i, 1]
        B[2, 3 * i + 2] = dN[i, 2]
        B[3, 3 * i] = dN[i, 1]; B[3, 3 * i + 1] = dN[i, 0]
        B[4, 3 * i + 1] = dN[i, 2]; B[4, 3 * i + 2] = dN[i, 1]
        B[5, 3 * i] = dN[i, 2]; B[5, 3 * i + 2] = dN[i, 0]
    return B


def Ke_h8(nu=0.3):
    D = D_matrix(nu)
    g = 1.0 / math.sqrt(3.0)
    K = np.zeros((24, 24))
    for xi in (-g, g):
        for eta in (-g, g):
            for zeta in (-g, g):
                B = B_at(xi, eta, zeta)
                K += B.T @ D @ B * 0.125
    return K


# ---------------------------------------------------------------- elementos finitos na GPU
@ti.data_oriented
class FE:
    def __init__(self, nx, ny, nz, nu=0.3):
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.eshape = (self.nx, self.ny, self.nz)
        self.nshape = (self.nx + 1, self.ny + 1, self.nz + 1)
        Ke = Ke_h8(nu)
        D = D_matrix(nu)
        DB0 = D @ B_at(0.0, 0.0, 0.0)
        self.Ke_np, self.DB0_np = Ke, DB0
        self.Ke = tuple(float(v) for v in Ke.flatten())
        self.DB0 = tuple(float(v) for v in DB0.flatten())
        es, ns = self.eshape, self.nshape
        self.E = ti.field(REAL, es)
        self.ew = ti.field(REAL, es)
        self.eo = ti.field(REAL, es)
        V3 = lambda: ti.Vector.field(3, REAL, ns)
        self.free, self.spring = V3(), V3()
        self.dinv = ti.Matrix.field(3, 3, REAL, ns)
        self.b, self.xs, self.r, self.z, self.d, self.q = V3(), V3(), V3(), V3(), V3(), V3()
        self.s_rz = ti.field(ACC, ())
        self.s_rzn = ti.field(ACC, ())
        self.s_dq = ti.field(ACC, ())
        self.s_rr = ti.field(ACC, ())
        self.s_ff = ti.field(ACC, ())
        self.n_solves = 0
        self.n_cg = 0
        self.t_solve = 0.0

    # ---- dados
    def set_E(self, E):
        self.E.from_numpy(np.ascontiguousarray(E, dtype=NPR))
        self._build_dinv()

    def set_bc(self, free, spring=None):
        self.free.from_numpy(np.ascontiguousarray(free, dtype=NPR))
        sp = np.zeros(self.nshape + (3,)) if spring is None else spring
        self.spring.from_numpy(np.ascontiguousarray(sp, dtype=NPR))

    # ---- núcleos
    @ti.kernel
    def _build_dinv(self):
        """Pré-condicionador de Jacobi por blocos 3x3 (como na produção), restrito aos GL livres."""
        for i, j, k in self.dinv:
            blk = ti.Matrix.zero(REAL, 3, 3)
            for ddx in ti.static(range(2)):
                for ddy in ti.static(range(2)):
                    for ddz in ti.static(range(2)):
                        ex, ey, ez = i - ddx, j - ddy, k - ddz
                        if 0 <= ex < self.nx and 0 <= ey < self.ny and 0 <= ez < self.nz:
                            Ee = self.E[ex, ey, ez]
                            ln = ti.static(LOCAL_LUT[ddz][ddy][ddx])
                            for rr in ti.static(range(3)):
                                for cc in ti.static(range(3)):
                                    blk[rr, cc] += Ee * ti.static(self.Ke[(ln * 3 + rr) * 24 + ln * 3 + cc])
            sp = self.spring[i, j, k]
            fr = self.free[i, j, k]
            for rr in ti.static(range(3)):
                blk[rr, rr] += sp[rr]
            for rr in ti.static(range(3)):
                if fr[rr] < 0.5:
                    for cc in ti.static(range(3)):
                        blk[rr, cc] = 0.0
                        blk[cc, rr] = 0.0
                    blk[rr, rr] = 1.0
            inv = blk.inverse()
            for rr in ti.static(range(3)):
                for cc in ti.static(range(3)):
                    inv[rr, cc] *= fr[rr] * fr[cc]
            self.dinv[i, j, k] = inv

    @ti.kernel
    def _apply_K(self):
        """q = K d (recolha nos 8 elementos do nó; molas na diagonal; GL fixos anulados)."""
        for i, j, k in self.q:
            res = ti.Vector([0.0, 0.0, 0.0])
            di = self.d[i, j, k]
            for ddx in ti.static(range(2)):
                for ddy in ti.static(range(2)):
                    for ddz in ti.static(range(2)):
                        ex, ey, ez = i - ddx, j - ddy, k - ddz
                        if 0 <= ex < self.nx and 0 <= ey < self.ny and 0 <= ez < self.nz:
                            Ee = self.E[ex, ey, ez]
                            ln = ti.static(LOCAL_LUT[ddz][ddy][ddx])
                            for n in ti.static(range(8)):
                                if ti.static(n != ln):
                                    # Emenda 8: diferenças u_n - u_i (as linhas de Ke somam zero); a translação não
                                    # passa pela matriz arredondada a 32 bits
                                    uv = self.d[ex + NX_LUT[n], ey + NY_LUT[n], ez + NZ_LUT[n]] - di
                                    for rr in ti.static(range(3)):
                                        t = 0.0
                                        for cc in ti.static(range(3)):
                                            kv = ti.static(self.Ke[(ln * 3 + rr) * 24 + n * 3 + cc])
                                            if ti.static(kv != 0.0):
                                                t += kv * uv[cc]
                                        res[rr] += t * Ee
            res += self.spring[i, j, k] * self.d[i, j, k]
            self.q[i, j, k] = res * self.free[i, j, k]

    @ti.kernel
    def _x_to_d(self):
        for I in ti.grouped(self.xs):
            v = self.xs[I] * self.free[I]
            self.xs[I] = v
            self.d[I] = v

    @ti.kernel
    def _cg_init(self):
        self.s_rz[None] = 0.0
        self.s_rr[None] = 0.0
        self.s_ff[None] = 0.0
        for I in ti.grouped(self.r):
            fr = self.free[I]
            bb = self.b[I] * fr
            rv = (bb - self.q[I]) * fr
            zv = self.dinv[I] @ rv
            self.r[I] = rv
            self.z[I] = zv
            self.d[I] = zv
            self.s_rz[None] += ti.cast(rv.dot(zv), ACC)
            self.s_rr[None] += ti.cast(rv.dot(rv), ACC)
            self.s_ff[None] += ti.cast(bb.dot(bb), ACC)

    @ti.kernel
    def _cg_dq(self):
        self.s_dq[None] = 0.0
        for I in ti.grouped(self.d):
            self.s_dq[None] += ti.cast(self.d[I].dot(self.q[I]), ACC)

    @ti.kernel
    def _cg_update(self):
        alpha = 0.0
        if self.s_dq[None] != 0.0:
            alpha = ti.cast(self.s_rz[None] / self.s_dq[None], REAL)
        self.s_rzn[None] = 0.0
        self.s_rr[None] = 0.0
        for I in ti.grouped(self.xs):
            self.xs[I] += alpha * self.d[I]
            rv = self.r[I] - alpha * self.q[I]
            zv = self.dinv[I] @ rv
            self.r[I] = rv
            self.z[I] = zv
            self.s_rzn[None] += ti.cast(rv.dot(zv), ACC)
            self.s_rr[None] += ti.cast(rv.dot(rv), ACC)

    @ti.kernel
    def _cg_dir(self):
        beta = 0.0
        if self.s_rz[None] != 0.0:
            beta = ti.cast(self.s_rzn[None] / self.s_rz[None], REAL)
        self.s_rz[None] = self.s_rzn[None]
        for I in ti.grouped(self.d):
            self.d[I] = self.z[I] + beta * self.d[I]

    def solve(self, b, x0=None, tol=1e-5, maxit=6000, check=25):
        """Resolve K x = b (b e x0 em numpy, forma nshape + (3,)). Devolve (x, iterações, resíduo relativo)."""
        t0 = time.time()
        self.b.from_numpy(np.ascontiguousarray(b, dtype=NPR))
        if x0 is None:
            self.xs.fill(0.0)
        else:
            self.xs.from_numpy(np.ascontiguousarray(x0, dtype=NPR))
        self._x_to_d()
        self._apply_K()
        self._cg_init()
        ff = float(self.s_ff[None])
        if ff <= 0.0:
            self.xs.fill(0.0)
            return self.xs.to_numpy().astype(np.float64), 0, 0.0
        rel = math.sqrt(max(float(self.s_rr[None]), 0.0) / ff)
        it = 0
        while rel > tol and it < maxit:
            for _ in range(check):
                self._apply_K()
                self._cg_dq()
                self._cg_update()
                self._cg_dir()
            it += check
            rel = math.sqrt(max(float(self.s_rr[None]), 0.0) / ff)
        x = self.xs.to_numpy().astype(np.float64)
        self.n_solves += 1
        self.n_cg += it
        self.t_solve += time.time() - t0
        return x, it, rel

    @ti.kernel
    def _elem_aKb(self):
        for ex, ey, ez in self.eo:
            ua = ti.Matrix.zero(REAL, 8, 3)
            ub = ti.Matrix.zero(REAL, 8, 3)
            a0 = self.d[ex, ey, ez]
            b0 = self.q[ex, ey, ez]
            for n in ti.static(range(8)):
                # Emenda 8: diferenças ao nó 0 do elemento (Ke 1 = 0)
                va = self.d[ex + NX_LUT[n], ey + NY_LUT[n], ez + NZ_LUT[n]] - a0
                vb = self.q[ex + NX_LUT[n], ey + NY_LUT[n], ez + NZ_LUT[n]] - b0
                for c in ti.static(range(3)):
                    ua[n, c] = va[c]
                    ub[n, c] = vb[c]
            s = 0.0
            for i in ti.static(range(8)):
                for j in ti.static(range(8)):
                    for r in ti.static(range(3)):
                        for c in ti.static(range(3)):
                            kv = ti.static(self.Ke[(i * 3 + r) * 24 + j * 3 + c])
                            if ti.static(kv != 0.0):
                                s += ua[i, r] * kv * ub[j, c]
            self.eo[ex, ey, ez] = s

    def aKb(self, a, b):
        """Por elemento: a_e^T Ke b_e (sem o fator E)."""
        self.d.from_numpy(np.ascontiguousarray(a, dtype=NPR))
        self.q.from_numpy(np.ascontiguousarray(b, dtype=NPR))
        self._elem_aKb()
        return self.eo.to_numpy().astype(np.float64)

    @ti.func
    def _stress_solid(self, ex, ey, ez):
        ue = ti.Vector.zero(REAL, 24)
        v0 = self.d[ex, ey, ez]
        for n in ti.static(range(8)):
            v = self.d[ex + NX_LUT[n], ey + NY_LUT[n], ez + NZ_LUT[n]] - v0  # Emenda 8: D B0 1 = 0
            for c in ti.static(range(3)):
                ue[n * 3 + c] = v[c]
        s = ti.Vector.zero(REAL, 6)
        for r in ti.static(range(6)):
            acc = 0.0
            for c in ti.static(range(24)):
                kv = ti.static(self.DB0[r * 24 + c])
                if ti.static(kv != 0.0):
                    acc += kv * ue[c]
            s[r] = acc
        return s

    @ti.kernel
    def _elem_vm(self):
        for ex, ey, ez in self.eo:
            s = self._stress_solid(ex, ey, ez)
            vm2 = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2] - s[0] * s[1] - s[1] * s[2] - s[0] * s[2]
                   + 3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))
            self.eo[ex, ey, ez] = ti.sqrt(ti.max(vm2, 0.0))

    def vm(self, u):
        """Tensão de von Mises no centro do elemento, com o material cheio (E = 1)."""
        self.d.from_numpy(np.ascontiguousarray(u, dtype=NPR))
        self._elem_vm()
        return self.eo.to_numpy().astype(np.float64)

    @ti.kernel
    def _stress_rhs(self):
        for I in ti.grouped(self.q):
            self.q[I] = ti.Vector([0.0, 0.0, 0.0])
        for ex, ey, ez in self.ew:
            we = self.ew[ex, ey, ez]
            if we != 0.0:
                s = self._stress_solid(ex, ey, ez)
                vm2 = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2] - s[0] * s[1] - s[1] * s[2] - s[0] * s[2]
                       + 3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))
                vm = ti.sqrt(ti.max(vm2, 1e-30))
                g = ti.Vector([(2.0 * s[0] - s[1] - s[2]) / (2.0 * vm), (2.0 * s[1] - s[0] - s[2]) / (2.0 * vm),
                               (2.0 * s[2] - s[0] - s[1]) / (2.0 * vm), 3.0 * s[3] / vm, 3.0 * s[4] / vm,
                               3.0 * s[5] / vm])
                for n in ti.static(range(8)):
                    f = ti.Vector([0.0, 0.0, 0.0])
                    for c in ti.static(range(3)):
                        acc = 0.0
                        for r in ti.static(range(6)):
                            kv = ti.static(self.DB0[r * 24 + n * 3 + c])
                            if ti.static(kv != 0.0):
                                acc += kv * g[r]
                        f[c] = acc * we
                    self.q[ex + NX_LUT[n], ey + NY_LUT[n], ez + NZ_LUT[n]] += f
        for I in ti.grouped(self.q):
            self.q[I] = self.q[I] * self.free[I]

    def stress_rhs(self, u, w):
        """Lado direito do adjunto da tensão: soma_e w_e (D B0)^T d(vm)/d(sigma)."""
        self.d.from_numpy(np.ascontiguousarray(u, dtype=NPR))
        self.ew.from_numpy(np.ascontiguousarray(w, dtype=NPR))
        self._stress_rhs()
        return self.q.to_numpy().astype(np.float64)


# ---------------------------------------------------------------- filtros e projeção (numpy)
def kernel3d(r, kind="cone"):
    m = int(math.ceil(r)) - (1 if kind == "cone" else 0)
    m = max(m, 0)
    o = np.arange(-m, m + 1)
    X, Y, Z = np.meshgrid(o, o, o, indexing="ij")
    d = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    if kind == "cone":
        return np.maximum(0.0, r - d)
    return (d <= r + 1e-9).astype(np.float64)


class Filt:
    """Filtro de densidade normalizado, restrito aos elementos ativos (máscara)."""

    def __init__(self, mask, r, kind="cone", fft=False):
        self.m = mask.astype(np.float64)
        self.k = kernel3d(r, kind)
        self.fft = fft
        W = self._conv(self.m)
        W[W <= 1e-12] = 1.0
        self.W = W

    def _conv(self, a):
        if self.fft:
            return signal.fftconvolve(a, self.k, mode="same")
        return ndimage.correlate(a, self.k, mode="constant", cval=0.0)

    def fwd(self, x):
        return self._conv(x * self.m) / self.W * self.m

    def adj(self, g):
        return self._conv(g * self.m / self.W) * self.m


def proj(x, beta, eta):
    if beta <= 0.0:
        return x.copy()
    tb = math.tanh(beta * eta)
    den = tb + math.tanh(beta * (1.0 - eta))
    return (tb + np.tanh(beta * (x - eta))) / den


def dproj(x, beta, eta):
    if beta <= 0.0:
        return np.ones_like(x)
    den = math.tanh(beta * eta) + math.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (x - eta)) ** 2) / den


class Pipe:
    """x (desenho) -> xt (filtrado) -> xp (projetado, uma por realização), com regiões passivas."""

    def __init__(self, filt, solid, void, etas):
        self.f, self.solid, self.void, self.etas = filt, solid, void, dict(etas)

    def fwd(self, x, beta):
        xt = self.f.fwd(x)
        out = {}
        for k, eta in self.etas.items():
            xp = proj(xt, beta, eta)
            xp[self.solid] = 1.0
            xp[self.void] = 0.0
            out[k] = np.clip(xp, 0.0, 1.0)
        return xt, out

    def bwd(self, xt, beta, grads):
        gt = np.zeros_like(xt)
        for k, g in grads.items():
            g = g.copy()
            g[self.solid] = 0.0
            g[self.void] = 0.0
            gt += g * dproj(xt, beta, self.etas[k])
        return self.f.adj(gt)


# ---------------------------------------------------------------- respostas e sensibilidades
class Resp:
    """Responses and adjoint sensitivities on top of FE.

    mg: a tof3d_mg.MG built on the same FE, or None for the block-Jacobi CG of FE.solve.
    mg_policy: "design" rebuilds the coarse operators after every set_design; "iteration" rebuilds them only at the
    first solve after new_iteration() (the conjugate gradient stays exact; only the preconditioner is older).
    Every solve is logged in cg_log as (slot, iterations, relative residual)."""

    def __init__(self, fe, Emin=1e-4, penal=3.0, tol=1e-5, active=None, mg=None, mg_policy="design", maxit=None):
        self.fe, self.Emin, self.p, self.tol = fe, Emin, penal, tol
        self.act = np.ones(fe.eshape) if active is None else active.astype(np.float64)
        self.warm = {}
        self.cg_log = []
        self.mg = mg
        self.mg_policy = mg_policy
        self.maxit = maxit if maxit is not None else (500 if mg is not None else 6000)
        self._rebuild = True
        self.n_setup = 0

    def new_iteration(self):
        if self.mg_policy == "iteration":
            self._rebuild = True

    def set_design(self, xp):
        """E = (Emin + xp^p (1 - Emin)) on active elements; E = 0 outside the domain (nodes fixed)."""
        self.xp = xp
        self.fe.set_E((self.Emin + xp ** self.p * (1.0 - self.Emin)) * self.act)
        self.dE = self.p * np.maximum(xp, 1e-9) ** (self.p - 1.0) * (1.0 - self.Emin) * self.act
        if self.mg_policy == "design":
            self._rebuild = True

    def _solve(self, b, slot):
        if self.mg is None:
            x, it, rel = self.fe.solve(b, self.warm.get(slot), tol=self.tol, maxit=self.maxit)
        else:
            if self._rebuild:
                self.mg.setup()
                self.n_setup += 1
                self._rebuild = False
            x, it, rel = self.mg.solve(b, self.warm.get(slot), tol=self.tol, maxit=self.maxit)
            if not rel <= self.tol:
                # safety net: finish with the block-Jacobi CG from where the multigrid stopped, and log both
                self.cg_log.append((slot + "_mg_parou", it, rel))
                x, it, rel = self.fe.solve(b, x, tol=self.tol, maxit=20000)
                slot = slot + "_jacobi"
                self.cg_log.append((slot, it, rel))
                self.warm[slot[:-len("_jacobi")]] = x
                return x
        self.warm[slot] = x
        self.cg_log.append((slot, it, rel))
        return x

    def compliance(self, f, slot):
        u = self._solve(f, slot)
        c = float(np.sum(f * u))
        g = -self.dE * self.fe.aKb(u, u)
        return c, g, u

    def mech(self, fin, lout, slot):
        u = self._solve(fin, slot + "_u")
        mu = self._solve(lout, slot + "_mu")
        uout = float(np.sum(lout * u))
        g = -self.dE * self.fe.aKb(mu, u)
        return uout, g, u

    def stress(self, f, slot, sig_lim, P, q, smask):
        u = self._solve(f, slot)
        vms = self.fe.vm(u)
        xq = np.maximum(self.xp, 1e-3)
        ratio = np.where(smask, xq ** q * vms / sig_lim, 0.0)
        S = float(np.sum(ratio ** P))
        PN = S ** (1.0 / P)
        dPN = np.where(smask, PN ** (1.0 - P) * ratio ** (P - 1.0), 0.0)
        w = dPN * xq ** q / sig_lim
        rhs = self.fe.stress_rhs(u, w)
        lam = self._solve(rhs, slot + "_adj")
        expl = dPN * q * xq ** (q - 1.0) * vms / sig_lim
        expl = np.where(self.xp > 1e-3, expl, 0.0)
        impl = -self.dE * self.fe.aKb(lam, u)
        return PN, float(ratio.max()), expl + impl, u, ratio


def local_volume(ball, xp, alpha, p, mask):
    rb = ball.fwd(xp)
    v = rb[mask]
    n = v.size
    M = float(np.mean(v ** p))
    g = M ** (1.0 / p) / alpha - 1.0
    d = np.zeros_like(rb)
    d[mask] = (1.0 / alpha) * M ** (1.0 / p - 1.0) * v ** (p - 1.0) / n
    return g, float(v.max()), ball.adj(d), rb


# ---------------------------------------------------------------- grid helpers
def node_grid(fe):
    I, J, K = np.meshgrid(np.arange(fe.nx + 1), np.arange(fe.ny + 1), np.arange(fe.nz + 1), indexing="ij")
    return I, J, K


def elem_grid(fe):
    I, J, K = np.meshgrid(np.arange(fe.nx), np.arange(fe.ny), np.arange(fe.nz), indexing="ij")
    return I, J, K


def nodes_touching(fe, emask):
    """Máscara dos nós que pertencem a algum elemento de emask."""
    m = np.zeros(fe.nshape, bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                m[dx:dx + fe.nx, dy:dy + fe.ny, dz:dz + fe.nz] |= emask
    return m


def elems_touching(fe, node_sel):
    """Máscara dos elementos que têm algum nó em node_sel."""
    m = np.zeros(fe.eshape, bool)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                m |= node_sel[dx:dx + fe.nx, dy:dy + fe.ny, dz:dz + fe.nz]
    return m
