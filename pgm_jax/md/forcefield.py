"""pGM + Lennard-Jones force field for molecular dynamics: periodic box, smooth PME, pair list.

Energy (kJ/mol) with induced dipoles mu (Wei et al. JCP 153, 114116 (2020) Sec. II D for PME):

  E = KE [ U_dir + U_rec + U_self + U_bg + sum |mu|^2 / (2 alpha) ] + E_LJ,     d = p(R) + mu
  U_dir  = sum_{pairs < rc} q_i q_j G0 + (q_i d_j.x - q_j d_i.x) G1 - G2 (d_i.x)(d_j.x) + G1 d_i.d_j
           with G_n = B_n[erf(b_ij r)/r] - B_n[erf(b0 r)/r]  (all pairs, no masking; x = r_i - r_j)
  U_rec  = smooth PME of charges and dipoles with Ewald coefficient b0          (pme.py)
  U_self = -(b0/sqrt(pi)) sum q^2 - (2 b0^3 / (3 sqrt(pi))) sum |d|^2;  U_bg = -pi Q^2 / (2 V b0^2)
  E_LJ   = sum over intermolecular pairs < rc of eps (s^12 - 2 s^6), s = (R*_i + R*_j)/r  [+ tail]

The induced dipoles minimise E (a quadratic in mu); they are solved each step as in pmemd-pgm:
  * right-hand side: the permanent field b = -dU/dd at d = p (direct + PME + self);
  * initial guess: pmemd-pgm's multi-order least-squares extrapolation (dipole_scf_init = 3):
    coefficients c fitted so that sum_j c_j (alpha b)_{n-j} reproduces the current alpha b, then
    applied to the last converged dipoles (order 1), to their prediction errors (order 2) and to
    the errors of those (order 3);
  * conjugate gradients preconditioned by a few inner CG iterations on the short-range
    (< local_cut) dipole tensor plus 1/alpha (scf_local_cut, scf_local_niter; flexible PCG),
    converged when max|alpha r| / mean|alpha b| <= tol (pmemd-pgm's criterion), then one peek
    step mu += omega alpha r (scf_sor_coefficient).
Forces are -dE/dR at fixed mu (E is variational in mu).  Pair terms run over the rows of a dense,
full neighbour list (every pair appears in both rows): per-atom sums with no scatter-adds, and the
pair forces are row sums of the analytic gradient of the pair energy with respect to the row
displacement (F_i = -sum_k de_ik/dx_ik, using grad_x G_n = -G_{n+1} x).  The dipole derivatives dE/dd_i (row sums, PME, self term) are carried
through the covalent-dipole frames by one vector-Jacobian product.  PME forces come from autodiff
through the splines.  The molecular virial is the strain derivative at fixed mu (molecular
centres of mass scaled with the box), by autodiff of the whole energy.

Precision: 'mixed' evaluates pair kernels, PME and CG vectors in float32 and accumulates energies,
dot products, positions and forces in float64; 'double' uses float64 throughout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..lj import lj_long_range
from ..system import System
from ..units import KE
from ._jaxmd import dataclasses
from .box import min_image, volume
from .kernels import erf_kernels
from .pme import PME, grid_size

_SQRT_PI = math.sqrt(math.pi)


@dataclass(frozen=True)
class MDSettings:
    """Nonbonded and induction settings; names in comments are the pmemd-pgm equivalents."""
    cutoff: float = 0.9               # nm; cut, ee_dsum_cut (direct space and LJ)
    skin: float = 0.1                 # nm; skinnb
    ewald_beta: float = 4.0           # nm^-1; ew_coeff (0.4 A^-1)
    pme_grid: tuple | None = None     # nfft1..3; None: from pme_spacing
    pme_spacing: float = 0.05         # nm
    pme_order: int = 8                # order
    lj_lrc: bool = True               # vdwmeth = 1
    dipole_tol: float = 1e-5          # dipole_scf_tol (max|alpha r| / mean|alpha b|)
    max_iter: int = 50                # scf_cg_niter
    local_cut: float = 0.3            # nm; scf_local_cut
    local_niter: int = 0              # scf_local_niter; 0: Jacobi (fastest on GPU, see README)
    peek: float = 0.65                # scf_sor_coefficient (0: no peek step)
    extrap_order: int = 3             # dipole_scf_init_order (0: start from alpha b)
    extrap_steps: int = 2             # dipole_scf_init_step
    precision: str = "mixed"          # "mixed" | "double"

    @property
    def dtype(self):
        return jnp.float32 if self.precision == "mixed" else jnp.float64


@dataclasses.dataclass
class InductionState:
    """Converged dipoles and the extrapolation history (newest first)."""
    mu: jnp.ndarray          # (N, 3) e nm
    rec: jnp.ndarray         # (4, S, N, 3): alpha b, mu, mu - pred1, mu - pred2
    pred: jnp.ndarray        # (2, N, 3): order-1 and order-2 predictions of the last step
    count: jnp.ndarray       # (4,) int32


class Result(NamedTuple):
    energy: dict             # kJ/mol: elec, vdw, total
    forces: jnp.ndarray      # (N, 3) kJ/mol/nm, float64
    induction: InductionState
    iterations: jnp.ndarray
    residual: jnp.ndarray    # final max|alpha r| / mean|alpha b| (before the peek step)
    overflow: jnp.ndarray    # row capacity exceeded (results invalid; driver re-sizes and repeats)


def _dot(a, b):
    return jnp.sum((a * b).astype(jnp.float64))


def _push(stack, x):
    return jnp.concatenate([x[None].astype(stack.dtype), stack[:-1]], 0)


class PGMForceField:
    def __init__(self, sys: System, H, settings: MDSettings = MDSettings(), short_capacity: int = 48,
                 row_capacity: int | None = None):
        self.sys, self.s = sys, settings
        self.cd = settings.dtype
        self.n = sys.n
        self.b0 = float(settings.ewald_beta)
        self.c_self = 4.0 * self.b0 ** 3 / (3.0 * _SQRT_PI)
        grid = settings.pme_grid or grid_size(H, settings.pme_spacing)
        self.pme = PME(grid, settings.pme_order, self.b0, self.cd)
        self.mol = jnp.asarray(sys.mol)
        self.cov_i, self.cov_j = jnp.asarray(sys.cov_i), jnp.asarray(sys.cov_j)
        self.masses = jnp.asarray(sys.masses)
        self.S = max(1, int(settings.extrap_steps))
        self.ms = int(short_capacity)
        self.mc = row_capacity            # pairs kept per row after compaction to the cutoff (None: no compaction)

    # ------------------------------------------------------------------ building blocks
    def _atoms(self, params):
        P = self.sys.expand(params)
        return {k: jnp.asarray(v, jnp.float64) for k, v in P.items()}

    def perm_dipoles(self, pos, H, cov_c):
        if len(self.sys.cov_i) == 0:
            return jnp.zeros((self.n, 3))
        v = min_image(pos[self.cov_j] - pos[self.cov_i], H)
        u = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
        return jnp.zeros((self.n, 3)).at[self.cov_i].add(cov_c[:, None] * u)

    def _rows(self, pos, H, idx):
        """Row neighbours k (padding -> 0), displacements x_ik and masks.  Displacements are
        computed in the compute dtype (float32 in mixed precision, as OpenMM's mixed mode).  With a
        row capacity set, each row is compacted to its pairs inside the cutoff, so that everything
        downstream (kernels, CG products, forces) touches only those.  Returns also the overflow
        flag (a row with more pairs inside the cutoff than the capacity)."""
        N, cd = self.n, self.cd
        valid = idx < N
        k = jnp.where(valid, idx, 0)
        posc, Hc = pos.astype(cd), H.astype(cd)
        x = min_image(posc[:, None, :] - posc[k], Hc)
        r2 = jnp.sum(x * x, -1)
        within = valid & (r2 < self.s.cutoff ** 2)
        overflow = jnp.zeros((), bool)
        if self.mc is not None:
            mc = self.mc
            slot = jnp.cumsum(within, axis=1) - 1
            count = slot[:, -1] + 1
            tgt = jnp.where(within & (slot < mc), slot, mc)
            rows = jnp.broadcast_to(jnp.arange(N)[:, None], k.shape)
            k = jnp.zeros((N, mc + 1), k.dtype).at[rows, tgt].set(k)[:, :mc]
            x = jnp.zeros((N, mc + 1, 3), x.dtype).at[rows, tgt].set(x)[:, :mc]
            within = jnp.arange(mc)[None, :] < jnp.minimum(count, mc)[:, None]
            r2 = jnp.where(within, jnp.sum(x * x, -1), 1.0)
            overflow = jnp.max(count) > mc
        inter = within & (self.mol[:, None] != self.mol[k])
        return k, x, r2, within, inter, overflow

    def row_counts(self, pos, H, idx):
        """Largest number of pairs inside the cutoff in any row (to size the row capacity)."""
        N = self.n
        valid = idx < N
        k = jnp.where(valid, idx, 0)
        x = min_image(pos[:, None, :] - pos[k], H)
        return jnp.max(jnp.sum(valid & (jnp.sum(x * x, -1) < self.s.cutoff ** 2), axis=1))

    def _kernels(self, x, within, a, nmax: int = 3):
        cd = self.cd
        r = jnp.sqrt(jnp.where(within, jnp.sum(x * x, -1), 1.0))
        A = erf_kernels(a, r, nmax)
        B = erf_kernels(jnp.asarray(self.b0, cd), r, nmax)
        w = within.astype(cd)
        return (r,) + tuple((u - v) * w for u, v in zip(A, B))

    def geometry(self, pos, H, idx, P, forces: bool = False):
        """Row geometry and direct-space kernels (masked beyond the cutoff), compute dtype; with
        `forces`, also G3 and the LJ pair parameters for the analytic row forces."""
        cd = self.cd
        k, x, r2, within, inter, overflow = self._rows(pos, H, idx)
        R = P["radius"]
        a = (1.0 / jnp.sqrt(2.0 * (R[:, None] ** 2 + R[k] ** 2))).astype(cd)
        xc = x.astype(cd)
        r, G0, G1, G2, *G3 = self._kernels(xc, within, a, 4 if forces else 3)
        g = {"k": k, "x": xc, "G0": G0, "G1": G1, "G2": G2, "overflow": overflow}
        if forces:
            g.update(r=r, G3=G3[0], inter=inter,
                     rmin=(P["lj_rmin_half"][:, None] + P["lj_rmin_half"][k]).astype(cd),
                     eps=(P["lj_sqrt_eps"][:, None] * P["lj_sqrt_eps"][k]).astype(cd))
        if self.s.local_niter > 0:
            g["short"] = self._short_rows(k, xc, G1, G2, within & (r2 < self.s.local_cut ** 2))
        return g

    def _short_rows(self, k, x, G1, G2, short):
        """Compact the short-range entries of each row into (N, ms) for the preconditioner."""
        N, ms = self.n, self.ms
        slot = jnp.cumsum(short, axis=1) - 1
        tgt = jnp.where(short & (slot < ms), slot, ms)
        rows = jnp.broadcast_to(jnp.arange(N)[:, None], k.shape)

        def pack(v, fill):
            shape = (N, ms + 1) + v.shape[2:]
            return jnp.full(shape, fill, v.dtype).at[rows, tgt].set(v)[:, :ms]

        return {"k": pack(k, 0), "x": pack(x, 0.0), "G1": pack(G1, 0.0), "G2": pack(G2, 0.0)}

    @staticmethod
    def _row_field(g, q, d):
        """sum_k de_ik/dd_i: minus the direct-space field at each atom (charges q may be None)."""
        k, x, G1, G2 = g["k"], g["x"], g["G1"], g["G2"]
        dk = d[k]
        dkx = jnp.sum(dk * x, -1)
        c = -G2 * dkx if q is None else -q[k] * G1 - G2 * dkx
        return jnp.sum(c[..., None] * x + G1[..., None] * dk, axis=1)

    def _rec_grad(self, S, Gk, q, d):
        return jax.grad(lambda dd: self.pme.energy(S, Gk, q, dd))(d.astype(self.cd)).astype(self.cd)

    # ------------------------------------------------------------------ induction
    def _extrapolate(self, st: InductionState, new):
        S, order = self.S, self.s.extrap_order
        rec1 = st.rec[0].astype(jnp.float64)
        M = jnp.einsum("snd,tnd->st", rec1, rec1)
        bv = jnp.einsum("snd,nd->s", rec1, new.astype(jnp.float64))
        ridge = 1e-12 * jnp.trace(M) + 1e-300
        c = jnp.linalg.solve(M + ridge * jnp.eye(S), bv)
        lin = lambda R: jnp.einsum("s,snd->nd", c, R.astype(jnp.float64))
        p1 = lin(st.rec[1])
        p2 = p1 + lin(st.rec[2])
        p3 = p2 + lin(st.rec[3])
        have = st.count[0] >= S
        use1 = have & (st.count[1] >= S) & (order >= 1)
        use2 = use1 & (st.count[2] >= S) & (order >= 2)
        use3 = use2 & (st.count[3] >= S) & (order >= 3)
        guess = jnp.where(use3, p3, jnp.where(use2, p2, jnp.where(use1, p1, new.astype(jnp.float64))))
        pred = jnp.stack([jnp.where(use1, p1, 0.0), jnp.where(use2, p2, 0.0)]).astype(st.pred.dtype)
        st = st.set(rec=st.rec.at[0].set(_push(st.rec[0], new)), pred=pred, count=st.count.at[0].add(1))
        return guess.astype(self.cd), st

    def _record(self, st: InductionState, mu):
        S = self.S
        rec = st.rec.at[1].set(_push(st.rec[1], mu))
        c1 = st.count[1] + 1
        up2 = c1 > S
        rec = rec.at[2].set(jnp.where(up2, _push(rec[2], mu - st.pred[0]), rec[2]))
        c2 = st.count[2] + up2.astype(jnp.int32)
        up3 = c2 > S
        rec = rec.at[3].set(jnp.where(up3, _push(rec[3], mu - st.pred[1]), rec[3]))
        c3 = st.count[3] + up3.astype(jnp.int32)
        return st.set(mu=mu.astype(st.mu.dtype), rec=rec, count=jnp.stack([st.count[0], c1, c2, c3]))

    def init_induction(self) -> InductionState:
        dt = jnp.float64
        return InductionState(mu=jnp.zeros((self.n, 3), dt), rec=jnp.zeros((4, self.S, self.n, 3), dt),
                              pred=jnp.zeros((2, self.n, 3), dt), count=jnp.zeros(4, jnp.int32))

    def _solve(self, g, S, Gk, alpha, b, x0):
        cd, s = self.cd, self.s
        inv_a = (1.0 / alpha).astype(cd)[:, None]
        a_c = alpha.astype(cd)[:, None]
        c_self = jnp.asarray(self.c_self, cd)
        zq = jnp.zeros(self.n, cd)

        def A(v):
            return v * inv_a + self._row_field(g, None, v) + self._rec_grad(S, Gk, zq, v) - c_self * v

        def precond(r):
            z = r * a_c
            if s.local_niter <= 0:
                return z
            gs = g["short"]
            A_loc = lambda v: v * inv_a + self._row_field(gs, None, v)
            rr = r - A_loc(z)
            zz = rr * a_c
            rz = _dot(rr, zz)

            def inner(_, c):
                z, rr, p, rz = c
                Ap = A_loc(p)
                al = (rz / jnp.maximum(_dot(p, Ap), 1e-300)).astype(cd)
                z = z + al * p
                rr = rr - al * Ap
                zz = rr * a_c
                rz_new = _dot(rr, zz)
                p = zz + (rz_new / jnp.maximum(rz, 1e-300)).astype(cd) * p
                return z, rr, p, rz_new

            z, *_ = jax.lax.fori_loop(0, s.local_niter, inner, (z, rr, zz, rz))
            return z

        b = b.astype(cd)
        bnorm = jnp.mean(jnp.abs(b * a_c).astype(jnp.float64)) + 1e-300
        err_of = lambda r: jnp.max(jnp.abs(r * a_c).astype(jnp.float64)) / bnorm
        x = x0.astype(cd)
        r = b - A(x)
        z = precond(r)

        def cond(c):
            return (c[6] > s.dipole_tol) & (c[5] < s.max_iter)

        def body(c):
            x, r, z, p, rz, it, _ = c
            Ap = A(p)
            al = (rz / jnp.maximum(_dot(p, Ap), 1e-300)).astype(cd)
            x = x + al * p
            r_new = r - al * Ap
            z_new = precond(r_new)
            beta = (_dot(z_new, r_new - r) / jnp.maximum(rz, 1e-300)).astype(cd)       # flexible (Polak-Ribiere)
            p = z_new + beta * p
            return x, r_new, z_new, p, _dot(r_new, z_new), it + 1, err_of(r_new)

        x, r, _, _, _, it, err = jax.lax.while_loop(cond, body, (x, r, z, z, _dot(r, z), jnp.zeros((), jnp.int32), err_of(r)))
        if s.peek:
            x = x + jnp.asarray(s.peek, cd) * r * a_c
        return x.astype(jnp.float64), it, err

    def _field(self, g, S, Gk, P, p):
        cd = self.cd
        qc = P["q"].astype(cd)
        pc = p.astype(cd)
        return -(self._row_field(g, qc, pc) + self._rec_grad(S, Gk, qc, pc) - jnp.asarray(self.c_self, cd) * pc)

    # ------------------------------------------------------------------ energy terms
    def _nonpair(self, pos, H, d, mu, P):
        """PME + self + background + polarisation energies (kJ/mol), as a function of (pos, d)."""
        q = P["q"]
        u_rec = self.pme.energy(self.pme.setup(pos, H), self.pme.influence(H), q, d)
        u_self = -(self.b0 / _SQRT_PI) * jnp.sum(q * q) - 0.5 * self.c_self * jnp.sum(d * d)
        u_bg = -jnp.pi * jnp.sum(q) ** 2 / (2.0 * volume(H) * self.b0 ** 2)
        u_pol = jnp.sum(mu * mu / (2.0 * P["alpha"][:, None]))
        return KE * (u_rec + u_self + u_bg + u_pol)

    def _pair_sum(self, x, di, dk, qi, qk, a, within, inter, rminp, epsp):
        """sum over row entries of KE e_elec + e_LJ (each pair twice), float64."""
        r, G0, G1, G2 = self._kernels(x, within, a)
        dix, dkx = jnp.sum(di * x, -1), jnp.sum(dk * x, -1)
        e = qi * qk * G0 + (qi * dkx - qk * dix) * G1 - G2 * dix * dkx + G1 * jnp.sum(di * dk, -1)
        s6 = (rminp / r) ** 6
        elj = jnp.where(inter, epsp * (s6 * s6 - 2.0 * s6), 0.0)
        se, sl = jnp.sum(e.astype(jnp.float64)), jnp.sum(elj.astype(jnp.float64))
        return KE * se + sl, (KE * se, sl)

    def _row_inputs(self, pos, H, idx, P, d):
        cd = self.cd
        k, x, r2, within, inter, _ = self._rows(pos, H, idx)
        R, q = P["radius"], P["q"]
        a = (1.0 / jnp.sqrt(2.0 * (R[:, None] ** 2 + R[k] ** 2))).astype(cd)
        dc = d.astype(cd)
        di = jnp.broadcast_to(dc[:, None, :], x.shape)
        rminp = (P["lj_rmin_half"][:, None] + P["lj_rmin_half"][k]).astype(cd)
        epsp = (P["lj_sqrt_eps"][:, None] * P["lj_sqrt_eps"][k]).astype(cd)
        consts = (dc[k], q.astype(cd)[:, None], q.astype(cd)[k], a, within, inter, rminp, epsp)
        return x.astype(cd), di, consts

    def energy_fixed_mu(self, pos, H, mu, idx, P):
        """Total energy (kJ/mol, float64) and components with the induced dipoles held at mu;
        differentiable in positions and box (used for virials and Monte Carlo trials)."""
        p = self.perm_dipoles(pos, H, P["cov"])
        d = p + mu
        x, di, (dk, qi, qk, a, within, inter, rminp, epsp) = self._row_inputs(pos, H, idx, P, d)
        spair, (se, sl) = self._pair_sum(x, di, dk, qi, qk, a, within, inter, rminp, epsp)
        e_elec = 0.5 * se + self._nonpair(pos, H, d, mu, P)
        e_lj = 0.5 * sl + (lj_long_range(P, volume(H), self.s.cutoff) if self.s.lj_lrc else 0.0)
        return e_elec + e_lj, {"elec": e_elec, "vdw": e_lj}

    @staticmethod
    def _row_terms(g, q, d):
        """Pair energies (each pair counted in both rows) and the row sums of de_ik/dx_ik, from
        the kernels G0..G3 (grad_x G_n = -G_{n+1} x); charges q and total dipoles d, compute dtype."""
        k, x = g["k"], g["x"]
        G0, G1, G2, G3 = g["G0"], g["G1"], g["G2"], g["G3"]
        qi, qk = q[:, None], q[k]
        di, dk = d[:, None, :], d[k]
        dix, dkx = jnp.sum(di * x, -1), jnp.sum(dk * x, -1)
        didk = jnp.sum(di * dk, -1)
        t = qi * dkx - qk * dix
        e = qi * qk * G0 + t * G1 - G2 * dix * dkx + G1 * didk
        radial = -qi * qk * G1 - t * G2 + G3 * dix * dkx - G2 * didk
        gx = (radial[..., None] * x + (qi * G1)[..., None] * dk - (qk * G1)[..., None] * di
              - G2[..., None] * (di * dkx[..., None] + dk * dix[..., None]))
        s6 = (g["rmin"] / g["r"]) ** 6
        elj = jnp.where(g["inter"], g["eps"] * (s6 * s6 - 2.0 * s6), 0.0)
        glj = jnp.where(g["inter"], g["eps"] * 12.0 * (s6 - s6 * s6) / (g["r"] * g["r"]), 0.0)
        return (jnp.sum(e.astype(jnp.float64)), jnp.sum(elj.astype(jnp.float64)),
                jnp.sum(gx, axis=1).astype(jnp.float64), jnp.sum(glj[..., None] * x, axis=1).astype(jnp.float64))

    def _energy_forces(self, pos, H, mu, g, P):
        """Energy and forces at fixed mu: analytic row forces for the pair terms (no scatter-adds),
        autodiff for PME, one vector-Jacobian product through the covalent-dipole frames."""
        cd = self.cd
        p, vjp_p = jax.vjp(lambda y: self.perm_dipoles(y, H, P["cov"]), pos)
        d = p + mu
        qc, dc = P["q"].astype(cd), d.astype(cd)
        se, sl, gx_el, gx_lj = self._row_terms(g, qc, dc)
        dEdd = KE * self._row_field(g, qc, dc).astype(jnp.float64)
        e_np, (gpos_np, gd_np) = jax.value_and_grad(self._nonpair, argnums=(0, 2))(pos, H, d, mu, P)
        forces = -(KE * gx_el + gx_lj + gpos_np + vjp_p(dEdd + gd_np)[0])
        e_elec = 0.5 * KE * se + e_np
        e_lj = 0.5 * sl + (lj_long_range(P, volume(H), self.s.cutoff) if self.s.lj_lrc else 0.0)
        return {"elec": e_elec, "vdw": e_lj, "total": e_elec + e_lj}, forces

    # ------------------------------------------------------------------ public
    def compute(self, pos, H, idx, ind: InductionState, params=None) -> Result:
        """Solve the induced dipoles (extrapolated guess), then energy and forces."""
        pos, H = jnp.asarray(pos, jnp.float64), jnp.asarray(H, jnp.float64)
        P = self._atoms(params)
        g = self.geometry(pos, H, idx, P, forces=True)
        p = self.perm_dipoles(pos, H, P["cov"])
        S = self.pme.setup(pos, H)
        Gk = self.pme.influence(H)
        b = self._field(g, S, Gk, P, p)
        alpha = P["alpha"]
        guess, ind = self._extrapolate(ind, alpha[:, None] * b.astype(jnp.float64))
        mu, it, err = self._solve(g, S, Gk, alpha, b, guess)
        ind = self._record(ind, mu)
        energy, forces = self._energy_forces(pos, H, mu, g, P)
        return Result(energy, forces, ind, it, err, g["overflow"])

    def energy(self, pos, H, idx, ind: InductionState, params=None):
        """Energy only (Monte Carlo barostat trials), dipoles solved from ind.mu; returns
        (total, InductionState with the new mu, iterations, row-capacity overflow flag)."""
        pos, H = jnp.asarray(pos, jnp.float64), jnp.asarray(H, jnp.float64)
        P = self._atoms(params)
        g = self.geometry(pos, H, idx, P)
        p = self.perm_dipoles(pos, H, P["cov"])
        S = self.pme.setup(pos, H)
        Gk = self.pme.influence(H)
        b = self._field(g, S, Gk, P, p)
        mu, it, err = self._solve(g, S, Gk, P["alpha"], b, ind.mu)
        e, _ = self.energy_fixed_mu(pos, H, mu, idx, P)
        return e, ind.set(mu=mu), it, g["overflow"]

    def strain_derivative(self, pos, H, idx, mu, params=None, molecular: bool = True):
        """dE/d eps (3, 3) at fixed mu, plus the LJ tail impulse term (-E_lrc I) if lj_lrc."""
        pos, H = jnp.asarray(pos, jnp.float64), jnp.asarray(H, jnp.float64)
        P = self._atoms(params)
        if molecular:
            w = self.masses
            com = jax.ops.segment_sum(w[:, None] * pos, self.mol, self.sys.nmol) / jax.ops.segment_sum(w, self.mol, self.sys.nmol)[:, None]

        def e(eps):
            F = jnp.eye(3) + eps
            x = pos + ((com @ eps.T)[self.mol] if molecular else pos @ eps.T)
            return self.energy_fixed_mu(x, H @ F.T, mu, idx, P)[0]

        W = jax.grad(e)(jnp.zeros((3, 3)))
        if self.s.lj_lrc:
            W = W - lj_long_range(P, volume(H), self.s.cutoff) * jnp.eye(3)
        return W
