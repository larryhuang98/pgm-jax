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
  * initial guess (`predictor`): "mu4", cubic extrapolation of the converged dipoles
    4 mu_1 - 6 mu_2 + 4 mu_3 - mu_4 (pmemd-pgm GPU PGM_GPU_PRED=mu4; 3-4 CG iterations at
    tol 1e-4 in Langevin water at 1 fs); "mu3" quadratic; "ls", pmemd-pgm CPU's multi-order least-squares extrapolation
    (dipole_scf_init = 3); "none", alpha b;
  * fused initial residual (`fused`, with mu3/mu4): the guess depends only on history, so the
    permanent-field sweep is done for d = p + x0 and gives r0 = field(q, p + x0) - x0/alpha
    directly (pmemd-pgm PGM_FUSED); the normaliser mean|alpha b| of the convergence test is taken
    from the last unfused step (refreshed every `norm_refresh` steps);
  * conjugate gradients, Jacobi-preconditioned, or with a few inner CG iterations on the
    short-range (< local_cut) tensor (scf_local_cut, scf_local_niter; flexible PCG), converged when
    max|alpha r| / mean|alpha b| <= tol (pmemd-pgm's criterion), then one peek step
    mu += omega alpha r (scf_sor_coefficient).
Forces are -dE/dR at fixed mu (E is variational in mu).  Pair terms run over rows: for each atom,
its special partners (a fixed table from md/topology.py: the rest of a small molecule, the nearby
heavy-atom groups of a large one, each with a van der Waals weight) followed by its candidates
from the neighbour list (van der Waals weight 1), compacted every step to the pairs inside the
cutoff (fixed capacity; the driver re-sizes and repeats on overflow).  Rows are stored as a structure of arrays (index, x, y,
z, G0..G3): the row kernels are memory bound and read each component with unit stride.
Intramolecular displacements come from offsets within the molecule (exact in float32).  Every pair
appears in both rows: per-atom sums with no scatter-adds, and the pair forces are row sums of the
analytic gradient of the pair energy with respect to the row displacement (F_i = -sum_k
de_ik/dx_ik, grad_x G_n = -G_{n+1} x).  The dipole derivatives dE/dd_i (row sums, PME, self term)
are carried through the covalent-dipole frames by one vector-Jacobian product.  The PME dipole
gradient in each CG iteration is computed directly (one r2c/c2r FFT pair, spline derivatives);
PME forces come from autodiff through the splines.  The molecular virial is the strain
derivative at fixed mu (molecular centres of mass scaled with the box), by autodiff of the whole
energy.

Differentiability: E(pos, H, theta) at fixed mu is differentiable throughout (forces, virial,
dE/dtheta by Hellmann-Feynman).  With `differentiable=True`, compute() also returns forces and
dipoles with exact derivatives: the dipole solve is a jax.custom_vjp whose backward pass solves
A lam = mu_bar by CG (A symmetric) and pulls lam back through b - A mu at the solution.

Precision: 'mixed' evaluates pair kernels, PME and CG vectors in float32 and accumulates energies,
dot products, positions and forces in float64; 'double' uses float64 throughout.  float32 matrix
products are requested at full precision (NVIDIA GPUs otherwise use TF32, ~1e-3 relative error).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..lj import lj_long_range
from .topology import MDTopology
from ..options import check_vdw, elec_flags
from ..system import System
from ..vdw import gvdw_long_range, gvdw_pair
from ..units import KE
from ._jaxmd import dataclasses
from .box import min_image, volume
from .kernels import erf_kernels, erf_kernels_closed
from .pme import PME, grid_size

_SQRT_PI = math.sqrt(math.pi)


@dataclass(frozen=True)
class MDSettings:
    """Nonbonded and induction settings; names in comments are the pmemd-pgm equivalents."""
    cutoff: float = 0.9               # nm; cut, ee_dsum_cut (direct space and LJ)
    skin: float = 0.1                 # nm; skinnb
    ewald_beta: float = 4.0           # nm^-1; ew_coeff (0.4 A^-1)
    pme_grid: tuple | None = None     # nfft1..3; None: from pme_spacing
    pme_spacing: float = 0.08         # nm
    pme_order: int = 6                # order
    lj_lrc: bool = True               # vdwmeth = 1 (the r^-6 tail of LJ, or of the GVDW dispersion)
    dipole_tol: float = 1e-5          # dipole_scf_tol (max|alpha r| / mean|alpha b|); 1e-4: ~20 % faster, NVE drift 0.02 kT/ns/dof
    max_iter: int = 50                # scf_cg_niter
    predictor: str = "mu4"            # mu4 | mu3 | ls | none
    fused: bool = True                # fused initial residual (mu3/mu4)
    norm_refresh: int = 1000          # steps between unfused steps (convergence normaliser)
    local_cut: float = 0.3            # nm; scf_local_cut
    local_niter: int = 0              # scf_local_niter; 0: Jacobi (fastest on GPU)
    peek: float = 0.65                # scf_sor_coefficient (0: no peek step)
    extrap_order: int = 3             # dipole_scf_init_order (predictor "ls")
    extrap_steps: int = 2             # dipole_scf_init_step (predictor "ls")
    precision: str = "mixed"          # "mixed" | "double"
    differentiable: bool = False      # forces and induced dipoles differentiable (reverse mode) in
                                      # parameters, positions and box: implicit differentiation of
                                      # the dipole solve, one adjoint CG per gradient
    adjoint_tol: float = 1e-6         # adjoint CG: max|alpha r| / mean|alpha rhs|
    elec: str = "qpi"                 # "q" charges | "qp" + permanent dipoles | "qi" charges + induction |
                                      # "qpi" pGM (options.py); quadrupoles are not in the MD engine yet
    vdw: str = "lj"                   # "lj" | "gvdw" (vdw.py; pmemd-pgm igvdw=1) | "none"
    gvdw_rep: str = "gauss"           # GVDW repulsion: "gauss" (gvdw_rep_form=0) | "slater" (=1)

    @property
    def perm_dipoles(self) -> bool:
        return elec_flags(self.elec)[0]

    @property
    def induction(self) -> bool:
        return elec_flags(self.elec)[1]

    @property
    def dtype(self):
        return jnp.float32 if self.precision == "mixed" else jnp.float64


_PRED = {"mu3": (3.0, -3.0, 1.0), "mu4": (4.0, -6.0, 4.0, -1.0)}


@dataclasses.dataclass
class InductionState:
    """Converged dipoles, predictor history (newest first) and the convergence normaliser."""
    mu: jnp.ndarray          # (N, 3) e nm
    hist: jnp.ndarray        # (4, N, 3) converged dipoles of the last steps
    count: jnp.ndarray       # () int32: steps recorded
    norm: jnp.ndarray        # () mean|alpha b| of the last unfused step
    rec: jnp.ndarray         # (4, S, N, 3) "ls" records: alpha b, mu, mu - pred1, mu - pred2
    pred: jnp.ndarray        # (2, N, 3) "ls" order-1 and order-2 predictions
    lscount: jnp.ndarray     # (4,) int32


class Result(NamedTuple):
    energy: dict             # kJ/mol: elec, vdw, total
    forces: jnp.ndarray      # (N, 3) kJ/mol/nm, float64
    induction: InductionState
    iterations: jnp.ndarray
    residual: jnp.ndarray    # final max|alpha r| / mean|alpha b| (before the peek step)
    overflow: jnp.ndarray    # row capacity exceeded (results invalid; driver re-sizes and repeats)


def _zero_cotangent(x):
    x = jnp.asarray(x)
    if jnp.issubdtype(x.dtype, jnp.inexact):
        return jnp.zeros_like(x)
    return np.zeros(x.shape, dtype=jax.dtypes.float0)


def _dot(a, b):
    return jnp.sum((a * b).astype(jnp.float64))


def _push(stack, x):
    return jnp.concatenate([x[None].astype(stack.dtype), stack[:-1]], 0)


class PGMForceField:
    def __init__(self, sys: System, H, settings: MDSettings = MDSettings(), short_capacity: int = 48,
                 row_capacity: int | None = None, topology: MDTopology | None = None):
        self.sys, self.s = sys, settings
        if settings.predictor not in ("mu4", "mu3", "ls", "none"):
            raise ValueError(f"unknown predictor {settings.predictor!r}")
        check_vdw(settings.vdw, settings.gvdw_rep)
        self.pd, self.ind = elec_flags(settings.elec)
        if any(len(m.quad) for m in sys.molecules):
            import warnings
            warnings.warn("quadrupole terms are ignored by the MD engine (gas phase only for now)")
        self.cd = settings.dtype
        self.n = sys.n
        self.b0 = float(settings.ewald_beta)
        self.c_self = 4.0 * self.b0 ** 3 / (3.0 * _SQRT_PI)
        grid = settings.pme_grid or grid_size(H, settings.pme_spacing)
        self.pme = PME(grid, settings.pme_order, self.b0, self.cd)
        mol = np.asarray(sys.mol)
        self.mol = jnp.asarray(mol)
        self.cov_i, self.cov_j = jnp.asarray(sys.cov_i), jnp.asarray(sys.cov_j)
        self.masses = jnp.asarray(sys.masses)
        self.S = max(1, int(settings.extrap_steps))
        self.ms = int(short_capacity)
        self.mc = row_capacity            # intermolecular pairs kept per row (None: no compaction)
        # special partners of every atom (fixed table with van der Waals weights; md/topology.py)
        self.topology = MDTopology.rigid(sys) if topology is None else topology
        self.special = jnp.asarray(self.topology.special)
        self.special_w = jnp.asarray(self.topology.special_w, jnp.float64)
        self.gid = jnp.asarray(self.topology.group, jnp.int32)
        self.sg = jnp.asarray(self.topology.special_groups)
        first = np.searchsorted(mol, np.arange(sys.nmol)) if np.all(np.diff(mol) >= 0) else \
            np.array([int(np.nonzero(mol == k)[0][0]) for k in range(sys.nmol)])
        self.first = jnp.asarray(first)

    # ------------------------------------------------------------------ building blocks
    def _atoms(self, params):
        P = self.sys.expand(params)
        return {k: jnp.asarray(v, jnp.float64) for k, v in P.items()}

    def perm_dipoles(self, pos, H, cov_c):
        if len(self.sys.cov_i) == 0 or not self.pd:
            return jnp.zeros((self.n, 3))
        v = min_image(pos[self.cov_j] - pos[self.cov_i], H)
        u = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
        return jnp.zeros((self.n, 3)).at[self.cov_i].add(cov_c[:, None] * u)

    @staticmethod
    def _displacements(p, k, H):
        """Minimum-image p_i - p_k as three (N, C) arrays (structure of arrays: the row kernels are
        memory bound and read components with unit stride); p (N, 3) in the compute dtype."""
        pk = p[k]
        x = [p[:, c][:, None] - pk[..., c] for c in range(3)]
        for c in (2, 1, 0):                                   # sequential reduction, reduced box
            n = jnp.round(x[c] / H[c, c])
            x = [x[j] - n * H[c, j] if j <= c else x[j] for j in range(3)]
        return x

    @property
    def intra(self):
        """The special-partner table (the name of the rigid engine's intramolecular table)."""
        return self.special

    def _intra_exact(self, pos, H, k, x, cd):
        """Replace the special entries (first columns; same molecule) by differences of offsets
        from each molecule's first atom: exact in float32 whatever the absolute coordinates."""
        ni = self.special.shape[1]
        off = (pos - pos[self.first][self.mol]).astype(cd)
        xi = self._displacements(off, self.special, H)
        hit = (k[:, :ni] == self.special) & (self.special < self.n)
        return tuple(xc.at[:, :ni].set(jnp.where(hit, xic, xc[:, :ni])) for xc, xic in zip(x, xi))

    def _rows(self, pos, H, idx):
        """Rows = [special partners | candidates from the neighbour list], masked to the cutoff and,
        with a row capacity set, compacted to the pairs inside it.  List candidates in the atom's
        special groups are dropped (those pairs come from the table).  Returns k, x = (x, y, z)
        components (compute dtype), the within mask, van der Waals weights (0 off `within`) and
        the overflow flag."""
        N, cd = self.n, self.cd
        ni = self.special.shape[1]
        cand = jnp.concatenate([self.special, idx.astype(self.special.dtype)], axis=1)
        valid = cand < N
        k = jnp.where(valid, cand, 0)
        from_list = jnp.arange(cand.shape[1])[None, :] >= ni
        in_special = jnp.any(self.gid[k][:, :, None] == self.sg[:, None, :], axis=-1)
        keep = valid & ~(from_list & in_special)
        wv = jnp.concatenate([self.special_w.astype(cd), jnp.ones(idx.shape, cd)], axis=1)
        p = pos.astype(cd)
        Hc = H.astype(cd)
        x = self._displacements(p, k, Hc)
        within = keep & (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] < self.s.cutoff ** 2)
        overflow = jnp.zeros((), bool)
        if self.mc is not None:
            mc = self.mc
            slot = jnp.cumsum(within, axis=1) - 1
            count = slot[:, -1] + 1
            tgt = jnp.where(within & (slot < mc), slot, mc)
            rows = jnp.broadcast_to(jnp.arange(N)[:, None], k.shape)
            k = jnp.zeros((N, mc + 1), k.dtype).at[rows, tgt].set(k)[:, :mc]
            wv = jnp.zeros((N, mc + 1), wv.dtype).at[rows, tgt].set(wv)[:, :mc]
            within = jnp.arange(mc)[None, :] < jnp.minimum(count, mc)[:, None]
            x = self._displacements(p, k, Hc)                  # recompute on the compacted rows
            overflow = jnp.max(count) > mc
        x = self._intra_exact(pos, Hc, k, x, cd)
        return k, x, within, jnp.where(within, wv, 0.0), overflow

    def row_counts(self, pos, H, idx):
        """Largest number of pairs (special + list pairs inside the cutoff) in any row."""
        saved, self.mc = self.mc, None
        try:
            return jnp.max(jnp.sum(self._rows(pos, H, idx)[2], axis=1))
        finally:
            self.mc = saved

    def _pair_a(self, R, k):
        return 1.0 / jnp.sqrt(2.0 * (R[:, None] ** 2 + R[k] ** 2))

    def _kernels(self, x, within, a, nmax: int = 3, series: bool = True):
        dt = x[0].dtype
        r = jnp.sqrt(jnp.where(within, x[0] * x[0] + x[1] * x[1] + x[2] * x[2], 1.0))
        kern = erf_kernels if series else erf_kernels_closed
        A = kern(a.astype(dt), r, nmax)
        B = kern(jnp.asarray(self.b0, dt), r, nmax)
        w = within.astype(dt)
        return (r,) + tuple((u - v) * w for u, v in zip(A, B))

    def geometry(self, pos, H, idx, P, forces: bool = False):
        """Row displacements and kernels G0..G2 (G3 and LJ pair parameters with `forces`)."""
        cd = self.cd
        nmax = 4 if forces else 3
        k, x, within, wv, overflow = self._rows(pos, H, idx)
        r, *G = self._kernels(x, within, self._pair_a(P["radius"].astype(cd), k), nmax)
        g = {"k": k, "x": x, "overflow": overflow}
        for n in range(nmax):
            g[f"G{n}"] = G[n]
        if forces:
            g.update(r=r, wv=wv, vp=self._vdw_params(P, k))
        if self.s.local_niter > 0:
            short = within & (r < self.s.local_cut)
            g["short"] = self._short_rows(k, x, g["G1"], g["G2"], short)
        return g

    def _short_rows(self, k, x, G1, G2, short):
        """Compact the short-range entries of each row into (N, ms) for the preconditioner."""
        N, ms = self.n, self.ms
        slot = jnp.cumsum(short, axis=1) - 1
        tgt = jnp.where(short & (slot < ms), slot, ms)
        rows = jnp.broadcast_to(jnp.arange(N)[:, None], k.shape)

        def pack(v, fill):
            return jnp.full((N, ms + 1), fill, v.dtype).at[rows, tgt].set(v)[:, :ms]

        return {"k": pack(k, 0), "x": tuple(pack(c, 0.0) for c in x), "G1": pack(G1, 0.0), "G2": pack(G2, 0.0)}

    @staticmethod
    def _row_field(g, q, d):
        """sum_k de_ik/dd_i: minus the direct-space field at each atom (charges q may be None)."""
        k, x, G1, G2 = g["k"], g["x"], g["G1"], g["G2"]
        dk = d[k]
        dk = (dk[..., 0], dk[..., 1], dk[..., 2])
        dkx = dk[0] * x[0] + dk[1] * x[1] + dk[2] * x[2]
        c = -G2 * dkx if q is None else -q[k] * G1 - G2 * dkx
        return jnp.stack([jnp.sum(c * x[j] + G1 * dk[j], axis=1) for j in range(3)], -1)

    def _rec_grad(self, S, Gk, q, d):
        return self.pme.grad_dipoles(S, Gk, q, d.astype(self.cd)).astype(self.cd)

    def _field(self, g, S, Gk, q, d):
        """Total field -dU/dd (direct + PME + self) of charges q and dipoles d, compute dtype."""
        cd = self.cd
        dc = d.astype(cd)
        return -(self._row_field(g, q, dc) + self._rec_grad(S, Gk, q, dc) - jnp.asarray(self.c_self, cd) * dc)

    # ------------------------------------------------------------------ induction
    def init_induction(self) -> InductionState:
        dt = jnp.float64
        z = jnp.zeros((self.n, 3), dt)
        return InductionState(mu=z, hist=jnp.zeros((4, self.n, 3), dt), count=jnp.zeros((), jnp.int32),
                              norm=jnp.ones((), dt), rec=jnp.zeros((4, self.S, self.n, 3), dt),
                              pred=jnp.zeros((2, self.n, 3), dt), lscount=jnp.zeros(4, jnp.int32))

    def _extrapolate_ls(self, st: InductionState, new):
        """pmemd-pgm CPU multi-order least-squares extrapolation (dipole_scf_init = 3)."""
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
        cnt = st.lscount
        use1 = (cnt[0] >= S) & (cnt[1] >= S) & (order >= 1)
        use2 = use1 & (cnt[2] >= S) & (order >= 2)
        use3 = use2 & (cnt[3] >= S) & (order >= 3)
        guess = jnp.where(use3, p3, jnp.where(use2, p2, jnp.where(use1, p1, new.astype(jnp.float64))))
        pred = jnp.stack([jnp.where(use1, p1, 0.0), jnp.where(use2, p2, 0.0)]).astype(st.pred.dtype)
        st = st.set(rec=st.rec.at[0].set(_push(st.rec[0], new)), pred=pred, lscount=cnt.at[0].add(1))
        return guess, st

    def _record_ls(self, st: InductionState, mu):
        S = self.S
        rec = st.rec.at[1].set(_push(st.rec[1], mu))
        c1 = st.lscount[1] + 1
        up2 = c1 > S
        rec = rec.at[2].set(jnp.where(up2, _push(rec[2], mu - st.pred[0]), rec[2]))
        c2 = st.lscount[2] + up2.astype(jnp.int32)
        up3 = c2 > S
        rec = rec.at[3].set(jnp.where(up3, _push(rec[3], mu - st.pred[1]), rec[3]))
        c3 = st.lscount[3] + up3.astype(jnp.int32)
        return st.set(rec=rec, lscount=jnp.stack([st.lscount[0], c1, c2, c3]))

    def _operator(self, g, S, Gk, alpha):
        cd = self.cd
        inv_a = (1.0 / alpha).astype(cd)[:, None]
        zq = jnp.zeros(self.n, cd)
        return lambda v: v * inv_a - self._field(g, S, Gk, zq, v)

    def _cg(self, g, A, alpha, x, r, norm, tol=None, peek=None):
        """Preconditioned CG from (x0, r0 = b - A x0); returns mu (float64), iterations, residual."""
        cd, s = self.cd, self.s
        tol = s.dipole_tol if tol is None else tol
        peek = s.peek if peek is None else peek
        inv_a = (1.0 / alpha).astype(cd)[:, None]
        a_c = alpha.astype(cd)[:, None]

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

        err_of = lambda r: jnp.max(jnp.abs(r * a_c).astype(jnp.float64)) / norm
        x, r = x.astype(cd), r.astype(cd)
        z = precond(r)

        def cond(c):
            return (c[6] > tol) & (c[5] < s.max_iter)

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
        if peek:
            x = x + jnp.asarray(peek, cd) * r * a_c
        return x.astype(jnp.float64), it, err

    def _residual(self, g, S, Gk, alpha, q, p, mu):
        """b - A mu = field(q, p + mu) - mu / alpha (compute dtype): zero at the induced dipoles."""
        cd = self.cd
        return self._field(g, S, Gk, q.astype(cd), p + mu) - (mu / alpha[:, None]).astype(cd)

    def _solve(self, g, S, Gk, P, p, ind: InductionState, fused_ok: bool = True):
        """Induced dipoles; returns mu, iterations, residual, updated InductionState.  With
        settings.differentiable, mu carries exact derivatives (implicit function theorem):
        A mu = b(theta)  =>  mu_bar . dmu = lam . d(b - A mu)|_mu  with  A lam = mu_bar  (A is
        symmetric, so the adjoint is one more CG with the same operator)."""
        if not self.s.differentiable:
            return self._solve_core(g, S, Gk, P["alpha"], P["q"], p, ind, fused_ok)
        cd = self.cd

        @jax.custom_vjp
        def run(g, S, Gk, alpha, q, p, ind):
            return self._solve_core(g, S, Gk, alpha, q, p, ind, fused_ok)

        def fwd(g, S, Gk, alpha, q, p, ind):
            out = self._solve_core(g, S, Gk, alpha, q, p, ind, fused_ok)
            return out, (g, S, Gk, alpha, q, p, out[0], ind)

        def bwd(res, cot):
            g, S, Gk, alpha, q, p, mu, ind = res
            mu_bar = cot[0]
            A = self._operator(g, S, Gk, alpha)
            rhs = mu_bar.astype(cd)
            norm = jnp.mean(jnp.abs(alpha[:, None] * mu_bar)) + 1e-300
            lam, _, _ = self._cg(g, A, alpha, jnp.zeros_like(mu_bar), rhs, norm, tol=self.s.adjoint_tol, peek=0.0)
            _, vjp = jax.vjp(lambda g, S, Gk, alpha, q, p: self._residual(g, S, Gk, alpha, q, p, mu), g, S, Gk, alpha, q, p)
            return (*vjp(lam.astype(cd)), jax.tree_util.tree_map(_zero_cotangent, ind))

        run.defvjp(fwd, bwd)
        mu, it, err, ind_new = run(g, S, Gk, P["alpha"], P["q"], p, jax.lax.stop_gradient(ind))
        # derivatives flow through mu only; the predictor history is data for the next step
        return mu, it, err, jax.lax.stop_gradient(ind_new).set(mu=mu)

    def _solve_core(self, g, S, Gk, alpha, q, p, ind: InductionState, fused_ok: bool = True):
        """Initial guess + residual (fused when possible), CG; returns mu, iterations, residual,
        updated InductionState."""
        cd = self.cd
        a64 = alpha[:, None]
        qc = q.astype(cd)
        A = self._operator(g, S, Gk, alpha)
        pred = self.s.predictor

        def plain(x0_hist, have):
            b = self._field(g, S, Gk, qc, p)
            ab = a64 * b.astype(jnp.float64)
            x0 = jnp.where(have, x0_hist, ab)
            r0 = b - A(x0.astype(cd))
            return x0, r0, jnp.mean(jnp.abs(ab)) + 1e-300

        if pred in _PRED:
            c = _PRED[pred]
            K = len(c)
            have = ind.count >= K
            x0h = sum(ci * ind.hist[j] for j, ci in enumerate(c))
            if self.s.fused and fused_ok:
                use_fused = have & (ind.count % self.s.norm_refresh != 0)

                def fused(_):
                    r0 = self._field(g, S, Gk, qc, p + x0h) - (x0h / a64).astype(cd)
                    return x0h, r0, ind.norm

                x0, r0, norm = jax.lax.cond(use_fused, fused, lambda _: plain(x0h, have), None)
            else:
                x0, r0, norm = plain(x0h, have)
        elif pred == "ls":
            b = self._field(g, S, Gk, qc, p)
            ab = a64 * b.astype(jnp.float64)
            x0, ind = self._extrapolate_ls(ind, ab)
            r0 = b - A(x0.astype(cd))
            norm = jnp.mean(jnp.abs(ab)) + 1e-300
        else:
            x0, r0, norm = plain(jnp.zeros((self.n, 3)), jnp.zeros((), bool))
        mu, it, err = self._cg(g, A, alpha, x0, r0, norm)
        if pred == "ls":
            ind = self._record_ls(ind, mu)
        ind = ind.set(mu=mu, hist=_push(ind.hist, mu), count=ind.count + 1, norm=norm)
        return mu, it, err, ind

    # ------------------------------------------------------------------ energy terms
    def _nonpair(self, pos, H, d, mu, P):
        """PME + self + background + polarisation energies (kJ/mol), as a function of (pos, d)."""
        q = P["q"]
        u_rec = self.pme.energy(self.pme.setup(pos, H), self.pme.influence(H), q, d)
        u_self = -(self.b0 / _SQRT_PI) * jnp.sum(q * q) - 0.5 * self.c_self * jnp.sum(d * d)
        u_bg = -jnp.pi * jnp.sum(q) ** 2 / (2.0 * volume(H) * self.b0 ** 2)
        u_pol = jnp.sum(mu * mu / (2.0 * P["alpha"][:, None])) if self.ind else 0.0
        return KE * (u_rec + u_self + u_bg + u_pol)

    # ------------------------------------------------------------------ van der Waals rows
    def _vdw_params(self, P, k):
        """Row pair parameters of the van der Waals form (tuple of (N, C) arrays)."""
        cd = self.cd
        if self.s.vdw == "lj":
            rh, se = P["lj_rmin_half"].astype(cd), P["lj_sqrt_eps"].astype(cd)
            return (rh[:, None] + rh[k], se[:, None] * se[k])
        if self.s.vdw == "gvdw":
            sa, sc, b = (P[n].astype(cd) for n in ("gvdw_sqrt_a", "gvdw_sqrt_c6", "gvdw_b"))
            return (sa[:, None] * sa[k], sc[:, None] * sc[k], 0.5 * (b[:, None] + b[k]),
                    self._pair_a(P["radius"].astype(cd), k))
        return ()

    def _vdw_rows(self, r, vp, wv, grad: bool = False):
        """Row pair energies (and (1/r) dU/dr) of the van der Waals form times the pair weights wv
        (0 for excluded pairs and outside the cutoff)."""
        if self.s.vdw == "lj":
            rminp, epsp = vp
            s6 = (rminp / r) ** 6
            e = epsp * (s6 * s6 - 2.0 * s6)
            d = epsp * 12.0 * (s6 - s6 * s6) / (r * r)
        elif self.s.vdw == "gvdw":
            A, C6, B, a = vp
            e, d = gvdw_pair(r, a, A, C6, B, self.s.gvdw_rep, grad=True)
        else:
            e = d = jnp.zeros_like(r)
        on = wv != 0
        e, d = jnp.where(on, e * wv, 0.0), jnp.where(on, d * wv, 0.0)
        return (e, d) if grad else e

    def _vdw_tail(self, P, H):
        if not self.s.lj_lrc or self.s.vdw == "none":
            return 0.0
        f = lj_long_range if self.s.vdw == "lj" else gvdw_long_range
        return f(P, volume(H), self.s.cutoff)

    def _pair_sum(self, x, di, dk, qi, qk, a, within, wv, vp):
        """sum over row entries of KE e_elec + e_vdW (each pair twice), float64 (autodiff path).
        x, di, dk: component triples of (N, C) arrays."""
        r, G0, G1, G2 = self._kernels(x, within, a)
        dix = di[0] * x[0] + di[1] * x[1] + di[2] * x[2]
        dkx = dk[0] * x[0] + dk[1] * x[1] + dk[2] * x[2]
        didk = di[0] * dk[0] + di[1] * dk[1] + di[2] * dk[2]
        e = qi * qk * G0 + (qi * dkx - qk * dix) * G1 - G2 * dix * dkx + G1 * didk
        elj = self._vdw_rows(r, vp, wv)
        se = jnp.sum(jnp.sum(e, axis=1).astype(jnp.float64))
        sl = jnp.sum(jnp.sum(elj, axis=1).astype(jnp.float64))
        return KE * se + sl, (KE * se, sl)

    def _row_inputs(self, pos, H, idx, P, d):
        """Rows for the differentiable (autodiff) energy."""
        cd = self.cd
        k, x, within, wv, _ = self._rows(pos, H, idx)
        R, q = P["radius"], P["q"]
        a = self._pair_a(R.astype(cd), k)
        dc = d.astype(cd)
        di = tuple(dc[:, j][:, None] for j in range(3))
        dk = tuple(dc[:, j][k] for j in range(3))
        consts = (dk, q.astype(cd)[:, None], q.astype(cd)[k], a, within, wv, self._vdw_params(P, k))
        return x, di, consts

    def energy_fixed_mu(self, pos, H, mu, idx, P):
        """Total energy (kJ/mol, float64) and components with the induced dipoles held at mu;
        differentiable in positions and box (used for virials and Monte Carlo trials)."""
        p = self.perm_dipoles(pos, H, P["cov"])
        d = p + mu
        x, di, (dk, qi, qk, a, within, wv, vp) = self._row_inputs(pos, H, idx, P, d)
        spair, (se, sl) = self._pair_sum(x, di, dk, qi, qk, a, within, wv, vp)
        e_elec = 0.5 * se + self._nonpair(pos, H, d, mu, P)
        e_lj = 0.5 * sl + self._vdw_tail(P, H)
        return e_elec + e_lj, {"elec": e_elec, "vdw": e_lj}

    def _row_terms(self, g, q, d):
        """Pair energies (each pair counted in both rows) and the row sums of de_ik/dx_ik, from
        the kernels G0..G3 (grad_x G_n = -G_{n+1} x); charges q and total dipoles d, compute dtype."""
        k, x = g["k"], g["x"]
        G0, G1, G2, G3 = g["G0"], g["G1"], g["G2"], g["G3"]
        qi, qk = q[:, None], q[k]
        dkk = d[k]
        dk = (dkk[..., 0], dkk[..., 1], dkk[..., 2])
        di = tuple(d[:, j][:, None] for j in range(3))
        dix = di[0] * x[0] + di[1] * x[1] + di[2] * x[2]
        dkx = dk[0] * x[0] + dk[1] * x[1] + dk[2] * x[2]
        didk = di[0] * dk[0] + di[1] * dk[1] + di[2] * dk[2]
        t = qi * dkx - qk * dix
        e = qi * qk * G0 + t * G1 - G2 * dix * dkx + G1 * didk
        radial = -qi * qk * G1 - t * G2 + G3 * dix * dkx - G2 * didk
        elj, glj = self._vdw_rows(g["r"], g["vp"], g["wv"], grad=True)
        qiG1, qkG1 = qi * G1, qk * G1
        gx = jnp.stack([jnp.sum(radial * x[j] + qiG1 * dk[j] - qkG1 * di[j] - G2 * (di[j] * dkx + dk[j] * dix), axis=1)
                        for j in range(3)], -1)
        glx = jnp.stack([jnp.sum(glj * x[j], axis=1) for j in range(3)], -1)
        rowsum = lambda v: jnp.sum(jnp.sum(v, axis=1).astype(jnp.float64))   # rows in compute dtype, total in float64
        return rowsum(e), rowsum(elj), gx.astype(jnp.float64), glx.astype(jnp.float64)

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
        e_lj = 0.5 * sl + self._vdw_tail(P, H)
        return {"elec": e_elec, "vdw": e_lj, "total": e_elec + e_lj}, forces

    # ------------------------------------------------------------------ public
    def rows_for(self, pos, H):
        """Candidate rows (every atom within the cutoff) for a fixed frame, built on the host: for
        single points and parameter fitting outside MD (which keeps its own neighbour list)."""
        from .neighbors import AtomNeighbors
        H = jnp.asarray(H, jnp.float64)
        return AtomNeighbors(self.n, H, self.s.cutoff, 0.0).allocate(jnp.asarray(pos, jnp.float64), None, H).idx

    def compute(self, pos, H, idx, ind: InductionState, params=None) -> Result:
        """Solve the induced dipoles (predicted guess), then energy and forces.  idx: candidate
        rows (N, C) from a neighbour list (padding N).  With settings.differentiable, energy,
        forces and Result.induction.mu can be differentiated (jax.grad / vjp) in params, pos, H."""
        pos, H = jnp.asarray(pos, jnp.float64), jnp.asarray(H, jnp.float64)
        P = self._atoms(params)
        g = self.geometry(pos, H, idx, P, forces=True)
        p = self.perm_dipoles(pos, H, P["cov"])
        S = self.pme.setup(pos, H)
        Gk = self.pme.influence(H)
        if self.ind:
            mu, it, err, ind = self._solve(g, S, Gk, P, p, ind)
        else:                                                  # no induced dipoles ("q", "qp")
            mu, it, err = jnp.zeros((self.n, 3)), jnp.zeros((), jnp.int32), jnp.zeros(())
        energy, forces = self._energy_forces(pos, H, mu, g, P)
        return Result(energy, forces, ind, it, err, g["overflow"])

    def energy(self, pos, H, idx, ind: InductionState, params=None):
        """Energy only (Monte Carlo barostat trials): dipoles solved from the last converged ones
        (no history update); returns (total, InductionState with mu, iterations, overflow)."""
        pos, H = jnp.asarray(pos, jnp.float64), jnp.asarray(H, jnp.float64)
        P = self._atoms(params)
        g = self.geometry(pos, H, idx, P)
        p = self.perm_dipoles(pos, H, P["cov"])
        S = self.pme.setup(pos, H)
        Gk = self.pme.influence(H)
        cd = self.cd
        alpha = P["alpha"]
        if self.ind:
            b = self._field(g, S, Gk, P["q"].astype(cd), p)
            A = self._operator(g, S, Gk, alpha)
            x0 = ind.mu
            mu, it, err = self._cg(g, A, alpha, x0, b - A(x0.astype(cd)), jnp.mean(jnp.abs(alpha[:, None] * b)) + 1e-300)
        else:
            mu, it = jnp.zeros((self.n, 3)), jnp.zeros((), jnp.int32)
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
        return W - self._vdw_tail(P, H) * jnp.eye(3)
