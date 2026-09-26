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

Separate electrostatics cutoff (`elec_cutoff`; pmemd's es_cutoff next to vdw_cutoff): the real-space
Ewald sum converges as erfc(b0 r)/r, which a larger b0 makes short-ranged, while the van der Waals
term keeps the cutoff (and long-range correction) its parameters were fitted with.  With
elec_cutoff < cutoff every row is split into two compact structures of arrays: the electrostatic
rows (pairs inside elec_cutoff, special partners first, with their van der Waals weights; capacity
mc_e) and the van der Waals rows (pairs between elec_cutoff and cutoff with a nonzero van der Waals
weight; capacity mc - mc_e).  The CG field sweeps, the permanent and induced energies, their forces,
the virial and the differentiable path use only the electrostatic rows, and the van der Waals term
sums over both.  The dipole matvec, which is memory bound, then streams (elec_cutoff / cutoff)^3 of
the bytes (0.47 at 0.7 / 0.9 nm).  ewald_beta and the PME grid must be chosen for the shorter cutoff
(`elec_cutoff_settings`).  elec_cutoff = None or = cutoff is the single-cutoff engine, unchanged.

Charge flux (`flux=`, md/flux.py): charges q(R) and covalent-dipole strengths c(R) that depend on
bond lengths.  Every energy evaluation takes them at its own positions (charges_at); the forces add
-phi . dq/dR - (dE/dc) . dc/dR, with the potential phi = dE/dq from one more row sum and the PME
charge gradient, pulled back through the bond-local flux map by one vector-Jacobian product
(_energy_forces_flux).  Without flux none of this code runs.

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
    cutoff: float = 0.9               # nm; cut / vdw_cutoff (van der Waals; electrostatics too unless elec_cutoff)
    elec_cutoff: float | None = None  # nm; es_cutoff: real-space electrostatics (None: cutoff).  Shorter than
                                      # cutoff: set ewald_beta and the PME grid for it (elec_cutoff_settings)
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

    @property
    def elec_rc(self) -> float:
        """Real-space electrostatics cutoff (nm)."""
        return float(self.cutoff) if self.elec_cutoff is None else float(self.elec_cutoff)

    @property
    def pair_cutoff(self) -> float:
        """Cutoff of the pair rows and of the neighbour list (nm): the larger of the electrostatics
        and van der Waals cutoffs (the latter only with a van der Waals term)."""
        return self.elec_rc if self.vdw == "none" else max(float(self.cutoff), self.elec_rc)

    def describe_cutoffs(self) -> str:
        """Cutoffs for log headers."""
        if self.elec_cutoff is None or self.elec_rc == float(self.cutoff):
            return f"cutoff {self.cutoff} nm"
        return f"cutoff {self.cutoff} nm (electrostatics {self.elec_rc} nm, Ewald {self.ewald_beta:.4g} /nm)"


# Direct-sum tolerance, in Amber's convention (ewald_beta_for), of the default pair 0.9 nm / 4.0 nm^-1.
DSUM_TOL = 3.95e-8


def ewald_beta_for(elec_cutoff: float, dsum_tol: float = DSUM_TOL) -> float:
    """Ewald coefficient (nm^-1) for the real-space cutoff elec_cutoff (nm) and a direct-sum
    tolerance in Amber's convention (sander / pmemd `dsum_tol`): erfc(beta rc) / rc = dsum_tol with
    rc in Angstrom, solved by bisection as Amber does.  Amber's default dsum_tol = 1e-5 gives the
    ew_coeff of its outputs (0.34864 A^-1 at 8 A, 0.30768 at 9 A).  The pGM-JAX default pair
    0.9 nm / 4.0 nm^-1 (erfc(3.6) = 3.6e-7; pmemd-pgm's ew_coeff 0.4 A^-1 at 9 A) is
    dsum_tol = 3.95e-8 (DSUM_TOL); at that tolerance 0.8 nm needs 4.52 nm^-1, 0.7 nm 5.19 and 0.6 nm
    6.09.  The rule bounds the charge-charge term; the dipole terms of pGM decay with higher powers
    of beta, so the measured real-space force error grows as the cutoff shrinks (ubiquitin: 4e-6 at
    0.9 nm, 8e-6 at 0.7, 3e-5 at 0.6; pGM water 2e-5 at 0.7, 1e-4 at 0.6)."""
    rc = float(elec_cutoff)
    if not rc > 0.0:
        raise ValueError(f"elec_cutoff must be positive, got {elec_cutoff}")
    f = lambda b: math.erfc(b * rc) / (10.0 * rc) - dsum_tol
    if not (0.0 < dsum_tol and f(0.0) > 0.0):
        raise ValueError(f"dsum_tol must be in (0, 1/rc_A) = (0, {1.0 / (10.0 * rc):.3g}), got {dsum_tol}")
    lo, hi = 0.0, 1.0
    while f(hi) > 0.0:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


def elec_cutoff_settings(elec_cutoff: float, dsum_tol: float = DSUM_TOL, exponent: float = 1.6) -> dict:
    """MDSettings arguments for a real-space electrostatics cutoff: ewald_beta from ewald_beta_for and
    the PME grid spacing h = 0.08 nm (4.0 / beta)^exponent, scaled from the default pair
    (4.0 nm^-1, 0.08 nm).

        MDSettings(cutoff=0.9, **elec_cutoff_settings(0.7))   # LJ at 0.9 nm, electrostatics at 0.7

    At a fixed spline order the PME force error of pGM (charges and dipoles) grows roughly as
    beta^9.5 h^6 (measured, order 6, ubiquitin in water): keeping beta x h fixed (exponent 1: 0.7 nm,
    5.19 nm^-1, 0.0616 nm) is the cheaper grid but multiplies the force error by 2.5 at 0.7 nm; exponent
    1.6 (the default: 0.7 nm, 5.19 nm^-1, 0.0527 nm) keeps the error of the default settings (3e-5
    relative for ubiquitin, 7e-5 for pGM water at 0.8 and 0.7 nm) for 25 % more PME work per CG
    iteration (8 % per step for ubiquitin).  Measurements: docs/protein_ff.md (What limits the speed)."""
    b = ewald_beta_for(elec_cutoff, dsum_tol)
    return {"elec_cutoff": float(elec_cutoff), "ewald_beta": b, "pme_spacing": 0.08 * (4.0 / b) ** exponent}


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


def _div_alpha(x, alpha, mask: bool):
    """x / alpha; with `mask`, 0 where alpha = 0.  Atoms (or virtual sites) with zero polarizability
    are not polarizable: their induced dipole stays 0 (the Jacobi-preconditioned CG, z = alpha r,
    never moves it, and every initial guess is 0 there) and their mu^2 / (2 alpha) is 0, with no 0/0
    in values or gradients (the derivative with respect to such an alpha is taken as 0).  The mask
    is static (PGMForceField.alpha_mask: some alpha of the system's parameter table is 0), so that
    systems without such atoms run the plain division, bit for bit."""
    if not mask:
        return x / alpha
    pol = alpha != 0
    return jnp.where(pol, x / jnp.where(pol, alpha, 1.0), 0.0)


class PGMForceField:
    def __init__(self, sys: System, H, settings: MDSettings = MDSettings(), short_capacity: int = 48,
                 row_capacity: int | None = None, topology: MDTopology | None = None,
                 elec_capacity: int | None = None, flux=None):
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
        self.has_vsites = any(getattr(m, "vsites", None) for m in sys.molecules)   # md/vsites.py
        # atoms with alpha = 0 in the parameter table: induced dipoles masked (_div_alpha).  Parameters
        # passed at call time with new zeros need a force field built from a table with zeros
        self.alpha_mask = bool(np.any(np.asarray(sys.expand()["alpha"]) == 0.0))
        self.masses = jnp.asarray(sys.masses)
        self.S = max(1, int(settings.extrap_steps))
        self.ms = int(short_capacity)
        self.mc = row_capacity            # pairs kept per row (None: no compaction)
        # cutoffs (nm): electrostatics, van der Waals, rows.  With elec_cutoff < cutoff (and a van der
        # Waals term) the rows are split: mc_e electrostatic entries, then mc - mc_e van der Waals ones
        self.rc_e, self.rc_v, self.rc_pair = settings.elec_rc, float(settings.cutoff), settings.pair_cutoff
        if not (self.rc_e > 0.0 and self.rc_v > 0.0):
            raise ValueError(f"cutoffs must be positive (cutoff {settings.cutoff}, elec_cutoff {settings.elec_cutoff})")
        self.split = settings.vdw != "none" and self.rc_e < self.rc_v
        if elec_capacity is not None and not self.split:
            raise ValueError("elec_capacity applies only to split rows (elec_cutoff < cutoff with a van der Waals term)")
        self.mc_e = elec_capacity
        # special partners of every atom (fixed table with van der Waals weights; md/topology.py)
        self.topology = MDTopology.rigid(sys) if topology is None else topology
        self.special = jnp.asarray(self.topology.special)
        self.special_w = jnp.asarray(self.topology.special_w, jnp.float64)
        self.gid = jnp.asarray(self.topology.group, jnp.int32)
        self.sg = jnp.asarray(self.topology.special_groups)
        first = np.searchsorted(mol, np.arange(sys.nmol)) if np.all(np.diff(mol) >= 0) else \
            np.array([int(np.nonzero(mol == k)[0][0]) for k in range(sys.nmol)])
        self.first = jnp.asarray(first)
        self.flux = flux                  # md/flux.py ChargeFlux (geometry-dependent q and c) or None
        if flux is not None and (flux.n_atoms != sys.n or len(flux.cov_bond) != len(sys.cov_i)):
            raise ValueError(f"charge flux for {flux.n_atoms} atoms / {len(flux.cov_bond)} covalent dipoles; the "
                             f"system has {sys.n} / {len(sys.cov_i)}")

    # ------------------------------------------------------------------ building blocks
    def _atoms(self, params):
        P = self.sys.expand(params)
        P = {k: jnp.asarray(v, jnp.float64) for k, v in P.items()}
        if self.flux is not None:                     # flux parameters ride along; charges_at applies them
            P["flux"] = self.flux.theta(params)
        elif isinstance(params, dict) and "flux" in params:
            raise ValueError("the parameters have charge-flux values but the force field has no charge flux")
        return P

    def charges_at(self, pos, H, P):
        """P (from _atoms) with the charges q and covalent-dipole strengths cov of the geometry pos
        (charge flux, md/flux.py); P itself without flux or when already applied."""
        if "flux" not in P:
            return P
        q, cov = self.flux.charges(pos, H, P["q"], P["cov"], P["flux"])
        out = {k: v for k, v in P.items() if k != "flux"}
        out.update(q=q, cov=cov)
        return out

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

    def _special_exact(self, pos, H, k, x, sp):
        """Replace the displacements of the special pairs, the entries of the first sp.shape[1]
        columns where sp, by differences of offsets within the molecule (as _intra_exact)."""
        w = sp.shape[1]
        if w == 0:
            return x
        off = (pos - pos[self.first][self.mol]).astype(self.cd)
        xi = self._displacements(off, k[:, :w], H)
        return tuple(xc.at[:, :w].set(jnp.where(sp, xic, xc[:, :w])) for xc, xic in zip(x, xi))

    def _compact_parts(self, k, wv, masks, widths):
        """Compact every row into consecutive parts: the entries where masks[j] go, in column order, to
        part j (widths[j] columns).  One scatter of the partner indices places all parts (the slots
        of two parts come from one scan, one count per 16 bits); the scatter of anything else over
        the candidate rows is avoided: list entries have van der Waals weight 1, and the special
        entries, which lead each part, take their weights from the small special block.  Returns,
        per part, (k, within, weights, special-entry mask of the first min(S, width) columns), and
        the overflow flag."""
        N, ni, C = self.n, self.special.shape[1], k.shape[1]
        if len(masks) == 1:
            slots = [jnp.cumsum(masks[0].astype(jnp.int32), axis=1) - 1]
        else:
            if C >= 1 << 15:
                raise ValueError(f"candidate rows of {C} entries: at most {(1 << 15) - 1} for the packed counters")
            cs = jnp.cumsum(masks[0].astype(jnp.int32) + (masks[1].astype(jnp.int32) << 16), axis=1)
            slots = [(cs & 0xFFFF) - 1, (cs >> 16) - 1]
        starts = [int(o) for o in np.cumsum([0] + list(widths))]
        tgt = jnp.full(k.shape, starts[-1], jnp.int32)                  # past the end: dropped
        for m, sl, w, o in reversed(list(zip(masks, slots, widths, starts))):
            tgt = jnp.where(m & (sl < w), o + sl, tgt)
        rows = jnp.broadcast_to(jnp.arange(N, dtype=jnp.int32)[:, None], k.shape)
        kk = jnp.zeros((N, starts[-1] + 1), k.dtype).at[rows, tgt].set(k)
        parts, overflow = [], jnp.zeros((), bool)
        for m, sl, w, o in zip(masks, slots, widths, starts):
            count = sl[:, -1] + 1
            within = jnp.arange(w)[None, :] < jnp.minimum(count, w)[:, None]
            ws = min(ni, w)
            sp = jnp.arange(ws)[None, :] < jnp.sum(m[:, :ni], axis=1)[:, None]
            t = jnp.where(m[:, :ni] & (sl[:, :ni] < ws), sl[:, :ni], ws)
            r = jnp.broadcast_to(jnp.arange(N, dtype=jnp.int32)[:, None], t.shape)
            wsp = jnp.zeros((N, ws + 1), wv.dtype).at[r, t].set(wv[:, :ni])[:, :ws]
            wp = jnp.concatenate([jnp.where(sp, wsp, 1.0), jnp.ones((N, w - ws), wv.dtype)], axis=1)
            parts.append((kk[:, o:o + w], within, jnp.where(within, wp, 0.0), sp))
            overflow = overflow | (jnp.max(count) > w)
        return parts, overflow

    def _rows(self, pos, H, idx):
        """Rows = [special partners | candidates from the neighbour list], masked to the pair cutoff
        and, with a row capacity set, compacted to the pairs inside it.  List candidates in the atom's
        special groups are dropped (those pairs come from the table).  Returns k, x = (x, y, z)
        components (compute dtype), the within mask, van der Waals weights (0 off `within` and
        beyond the van der Waals cutoff), the overflow flag, and the van der Waals rows (k, x,
        within, weights) of split rows (_split_rows; None otherwise, when k, x, ... hold every pair)."""
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
        r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
        if self.split:
            return self._split_rows(pos, p, Hc, k, x, r2, keep, wv)
        within = keep & (r2 < self.rc_pair ** 2)
        overflow = jnp.zeros((), bool)
        if self.mc is not None:
            ((k, within, wv, _),), overflow = self._compact_parts(k, wv, (within,), (self.mc,))
            x = self._displacements(p, k, Hc)                  # recompute on the compacted rows
            r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
        if self.rc_v < self.rc_pair:                           # van der Waals cut before electrostatics
            wv = jnp.where(r2 < self.rc_v ** 2, wv, 0.0)
        x = self._intra_exact(pos, Hc, k, x, cd)
        return k, x, within, jnp.where(within, wv, 0.0), overflow, None

    def _split_rows(self, pos, p, H, k, x, r2, keep, wv):
        """Rows split at the electrostatics cutoff (elec_cutoff < cutoff): electrostatic rows (pairs
        inside elec_cutoff, with their van der Waals weights) and van der Waals rows (pairs between
        elec_cutoff and cutoff with a nonzero van der Waals weight), compacted to their own
        capacities (mc_e and mc - mc_e) into separate arrays, so that the CG streams only the
        electrostatic ones (_compact_parts).  Compaction keeps the column order, so each part starts
        with its special partners, whose exact displacements are then found by count.  Without a
        capacity (single points) both parts span the candidate rows, masked."""
        N = self.n
        ni = self.special.shape[1]
        ein = keep & (r2 < self.rc_e ** 2)
        vin = keep & ~ein & (r2 < self.rc_v ** 2) & (wv != 0)
        if self.mc is None:
            sp = self.special < N
            xe = self._special_exact(pos, H, k, x, sp & ein[:, :ni])
            xv = self._special_exact(pos, H, k, x, sp & vin[:, :ni])
            return (k, xe, ein, jnp.where(ein, wv, 0.0), jnp.zeros((), bool),
                    (k, xv, vin, jnp.where(vin, wv, 0.0)))
        if self.mc_e is None or not 0 <= self.mc_e <= self.mc:
            raise ValueError(f"split rows need 0 <= mc_e <= mc (got mc {self.mc}, mc_e {self.mc_e}); see size_rows")
        ((ke, ein, wve, spe), (kv, vin, wvv, spv)), overflow = self._compact_parts(
            k, wv, (ein, vin), (self.mc_e, self.mc - self.mc_e))
        xe = self._special_exact(pos, H, ke, self._displacements(p, ke, H), spe)
        xv = self._special_exact(pos, H, kv, self._displacements(p, kv, H), spv)
        return ke, xe, ein, wve, overflow, (kv, xv, vin, wvv)

    def pair_counts(self, pos, H, idx):
        """Largest numbers of pairs in any row: (electrostatic rows, van der Waals rows); the rows hold
        every pair inside the pair cutoff and the second count is 0 unless the rows are split."""
        saved, self.mc = self.mc, None
        try:
            _, _, within, _, _, vrows = self._rows(pos, H, idx)
        finally:
            self.mc = saved
        ce = jnp.max(jnp.sum(within, axis=1))
        cv = jnp.zeros_like(ce) if vrows is None else jnp.max(jnp.sum(vrows[2], axis=1))
        return jnp.stack([ce, cv])

    def row_counts(self, pos, H, idx):
        """Largest number of pairs (special + list pairs inside the cutoffs) in any row."""
        saved, self.mc = self.mc, None
        try:
            _, _, within, _, _, vrows = self._rows(pos, H, idx)
        finally:
            self.mc = saved
        n = jnp.sum(within, axis=1)
        return jnp.max(n if vrows is None else n + jnp.sum(vrows[2], axis=1))

    @property
    def capacity(self) -> tuple:
        """(mc, mc_e): pairs kept per row and, for split rows, how many of them are electrostatic."""
        return self.mc, self.mc_e

    def size_rows(self, pos, H, idx, factor: float = 1.2):
        """Set the row capacities (static shapes: re-jit afterwards) from the largest pair counts at
        pos, with half the neighbour list's head-room (pair counts inside a sphere fluctuate by a few
        per cent), in multiples of 8, at most the candidate width; split rows size each part."""
        ce, cv = (int(c) for c in jax.jit(self.pair_counts)(jnp.asarray(pos), jnp.asarray(H), idx))
        width = int(idx.shape[1]) + int(self.special.shape[1])
        cap = lambda c: min(int(np.ceil((c * (1.0 + 0.5 * (factor - 1.0)) + 8) / 8.0) * 8), width)
        self.mc_e = cap(ce) if self.split else None
        self.mc = self.mc_e + cap(cv) if self.split else cap(ce)
        return self.capacity

    def grow_rows(self, old):
        """After a row overflow at capacities `old` (`capacity`): every part of the rows at least 8
        wider than it was (the driver re-sizes where the block started, where the rows fit)."""
        mc, mc_e = old
        if self.mc is None or mc is None:
            return
        if self.split and self.mc_e is not None and mc_e is not None:
            e = max(self.mc_e, mc_e + 8)
            self.mc, self.mc_e = e + max(self.mc - self.mc_e, mc - mc_e + 8), e
        else:
            self.mc = max(self.mc, mc + 8)

    def fit_rows(self, caps):
        """Capacities that fit every one of `caps` (`capacity` tuples, e.g. one per replica): the
        largest of each part of the rows (static shapes: re-jit afterwards)."""
        caps = [c for c in caps if c is not None and c[0] is not None]
        if not caps:
            return
        if self.split and all(c[1] is not None for c in caps):
            e = max(c[1] for c in caps)
            self.mc, self.mc_e = e + max(c[0] - c[1] for c in caps), e
        else:
            self.mc = max(c[0] for c in caps)

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
        """Displacements and kernels G0..G2 of the electrostatic rows (with `forces`: G3, distances,
        van der Waals weights and pair parameters, and for split rows the van der Waals rows under
        "vdw_rows")."""
        cd = self.cd
        nmax = 4 if forces else 3
        k, x, within, wv, overflow, vrows = self._rows(pos, H, idx)
        r, *G = self._kernels(x, within, self._pair_a(P["radius"].astype(cd), k), nmax)
        g = {"k": k, "x": x, "overflow": overflow}
        for n in range(nmax):
            g[f"G{n}"] = G[n]
        if forces:
            g.update(r=r, wv=wv, vp=self._vdw_params(P, k))
            if vrows is not None:
                kv, xv, vin, wvv = vrows
                rv = jnp.sqrt(jnp.where(vin, xv[0] * xv[0] + xv[1] * xv[1] + xv[2] * xv[2], 1.0))
                g["vdw_rows"] = {"k": kv, "x": xv, "r": rv, "wv": wvv, "vp": self._vdw_params(P, kv)}
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

    @staticmethod
    def _row_potential(g, q, d):
        """sum_k de_ik/dq_i = sum_k q_k G0 + (d_k . x_ik) G1: the direct-space potential at each atom
        (charge flux)."""
        k, x, G0, G1 = g["k"], g["x"], g["G0"], g["G1"]
        dk = d[k]
        dkx = dk[..., 0] * x[0] + dk[..., 1] * x[1] + dk[..., 2] * x[2]
        return jnp.sum(q[k] * G0 + dkx * G1, axis=1)

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
        inv_a = _div_alpha(1.0, alpha, self.alpha_mask).astype(cd)[:, None]
        zq = jnp.zeros(self.n, cd)
        return lambda v: v * inv_a - self._field(g, S, Gk, zq, v)

    def _cg(self, g, A, alpha, x, r, norm, tol=None, peek=None):
        """Preconditioned CG from (x0, r0 = b - A x0); returns mu (float64), iterations, residual."""
        cd, s = self.cd, self.s
        tol = s.dipole_tol if tol is None else tol
        peek = s.peek if peek is None else peek
        inv_a = _div_alpha(1.0, alpha, self.alpha_mask).astype(cd)[:, None]
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
        return self._field(g, S, Gk, q.astype(cd), p + mu) - _div_alpha(mu, alpha[:, None], self.alpha_mask).astype(cd)

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
                    r0 = self._field(g, S, Gk, qc, p + x0h) - _div_alpha(x0h, a64, self.alpha_mask).astype(cd)
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
        u_pol = jnp.sum(_div_alpha(mu * mu, 2.0 * P["alpha"][:, None], self.alpha_mask)) if self.ind else 0.0
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

    def _vdw_sum(self, x, within, wv, vp):
        """sum over row entries of e_vdW (each pair twice), float64 (autodiff path, van der Waals rows)."""
        r = jnp.sqrt(jnp.where(within, x[0] * x[0] + x[1] * x[1] + x[2] * x[2], 1.0))
        return jnp.sum(jnp.sum(self._vdw_rows(r, vp, wv), axis=1).astype(jnp.float64))

    def _row_inputs(self, pos, H, idx, P, d):
        """Rows for the differentiable (autodiff) energy; the last item holds the van der Waals rows
        (x, within, weights, pair parameters) of split rows, else None."""
        cd = self.cd
        k, x, within, wv, _, vrows = self._rows(pos, H, idx)
        R, q = P["radius"], P["q"]
        a = self._pair_a(R.astype(cd), k)
        dc = d.astype(cd)
        di = tuple(dc[:, j][:, None] for j in range(3))
        dk = tuple(dc[:, j][k] for j in range(3))
        consts = (dk, q.astype(cd)[:, None], q.astype(cd)[k], a, within, wv, self._vdw_params(P, k))
        if vrows is not None:
            vrows = (vrows[1], vrows[2], vrows[3], self._vdw_params(P, vrows[0]))
        return x, di, consts, vrows

    def energy_fixed_mu(self, pos, H, mu, idx, P):
        """Total energy (kJ/mol, float64) and components with the induced dipoles held at mu;
        differentiable in positions and box (used for virials and Monte Carlo trials).  With charge
        flux, q and c are taken at pos."""
        P = self.charges_at(pos, H, P)
        p = self.perm_dipoles(pos, H, P["cov"])
        d = p + mu
        x, di, (dk, qi, qk, a, within, wv, vp), vrows = self._row_inputs(pos, H, idx, P, d)
        spair, (se, sl) = self._pair_sum(x, di, dk, qi, qk, a, within, wv, vp)
        if vrows is not None:
            sl = sl + self._vdw_sum(*vrows)
        e_elec = 0.5 * se + self._nonpair(pos, H, d, mu, P)
        e_lj = 0.5 * sl + self._vdw_tail(P, H)
        return e_elec + e_lj, {"elec": e_elec, "vdw": e_lj}

    def _row_terms(self, g, q, d):
        """Pair energies (each pair counted in both rows) and the row sums of de_ik/dx_ik, from
        the kernels G0..G3 (grad_x G_n = -G_{n+1} x); charges q and total dipoles d, compute dtype.
        Electrostatics over the electrostatic rows, van der Waals over those and the van der Waals
        rows of split rows."""
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
        glx = jnp.stack([jnp.sum(glj * x[j], axis=1) for j in range(3)], -1).astype(jnp.float64)
        rowsum = lambda v: jnp.sum(jnp.sum(v, axis=1).astype(jnp.float64))   # rows in compute dtype, total in float64
        sl = rowsum(elj)
        if "vdw_rows" in g:                                    # split rows: van der Waals beyond elec_cutoff
            t = g["vdw_rows"]
            elt, glt = self._vdw_rows(t["r"], t["vp"], t["wv"], grad=True)
            glx = glx + jnp.stack([jnp.sum(glt * t["x"][j], axis=1) for j in range(3)], -1).astype(jnp.float64)
            sl = sl + rowsum(elt)
        return rowsum(e), sl, gx.astype(jnp.float64), glx

    def _energy_forces(self, pos, H, mu, g, P, flux_pull=None):
        """Energy and forces at fixed mu: analytic row forces for the pair terms (no scatter-adds),
        autodiff for PME, one vector-Jacobian product through the covalent-dipole frames.  With
        charge flux (flux_pull: the pull-back of the flux map at pos; P holds q(R), c(R)):
        _energy_forces_flux."""
        if flux_pull is not None:
            return self._energy_forces_flux(pos, H, mu, g, P, flux_pull)
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

    def _energy_forces_flux(self, pos, H, mu, g, P, flux_pull):
        """_energy_forces with charge flux; P holds q(R) and c(R).  F = -dE/dR|_{q,c,mu}
        - phi . dq/dR - (dE/dc) . dc/dR: the potential phi = dE/dq (rows, and PME, self and
        background terms from the autodiff of _nonpair with q among the arguments) and dE/dc (the
        covalent-frame pull-back taken with respect to c as well) go through flux_pull, the
        jax.vjp of the bond-local map R -> (q, c)."""
        cd = self.cd
        p, vjp_p = jax.vjp(lambda y, c: self.perm_dipoles(y, H, c), pos, P["cov"])
        d = p + mu
        qc, dc = P["q"].astype(cd), d.astype(cd)
        se, sl, gx_el, gx_lj = self._row_terms(g, qc, dc)
        dEdd = KE * self._row_field(g, qc, dc).astype(jnp.float64)
        phi = KE * self._row_potential(g, qc, dc).astype(jnp.float64)
        e_np, (gpos_np, gd_np, gq_np) = jax.value_and_grad(
            lambda y, dd, q: self._nonpair(y, H, dd, mu, dict(P, q=q)), argnums=(0, 1, 2))(pos, d, P["q"])
        gpos_p, gcov = vjp_p(dEdd + gd_np)
        forces = -(KE * gx_el + gx_lj + gpos_np + gpos_p + flux_pull((phi + gq_np, gcov))[0])
        e_elec = 0.5 * KE * se + e_np
        e_lj = 0.5 * sl + self._vdw_tail(P, H)
        return {"elec": e_elec, "vdw": e_lj, "total": e_elec + e_lj}, forces

    # ------------------------------------------------------------------ public
    def rows_for(self, pos, H):
        """Candidate rows (every atom within the cutoff) for a fixed frame, built on the host: for
        single points and parameter fitting outside MD (which keeps its own neighbour list)."""
        from .neighbors import AtomNeighbors
        H = jnp.asarray(H, jnp.float64)
        return AtomNeighbors(self.n, H, self.rc_pair, 0.0).allocate(jnp.asarray(pos, jnp.float64), None, H).idx

    def compute(self, pos, H, idx, ind: InductionState, params=None) -> Result:
        """Solve the induced dipoles (predicted guess), then energy and forces.  idx: candidate
        rows (N, C) from a neighbour list (padding N).  With settings.differentiable, energy,
        forces and Result.induction.mu can be differentiated (jax.grad / vjp) in params, pos, H."""
        pos, H = jnp.asarray(pos, jnp.float64), jnp.asarray(H, jnp.float64)
        P = self._atoms(params)
        pull = None
        if self.flux is not None:                              # q(R), c(R) and the flux map's pull-back
            (q, cov), pull = jax.vjp(lambda y: self.flux.charges(y, H, P["q"], P["cov"], P["flux"]), pos)
            P = {**{k: v for k, v in P.items() if k != "flux"}, "q": q, "cov": cov}
        g = self.geometry(pos, H, idx, P, forces=True)
        p = self.perm_dipoles(pos, H, P["cov"])
        S = self.pme.setup(pos, H)
        Gk = self.pme.influence(H)
        if self.ind:                                           # the solve sees the electrostatic rows only
            ge = {key: v for key, v in g.items() if key != "vdw_rows"}
            mu, it, err, ind = self._solve(ge, S, Gk, P, p, ind)
        else:                                                  # no induced dipoles ("q", "qp")
            mu, it, err = jnp.zeros((self.n, 3)), jnp.zeros((), jnp.int32), jnp.zeros(())
        energy, forces = self._energy_forces(pos, H, mu, g, P, pull)
        return Result(energy, forces, ind, it, err, g["overflow"])

    def energy(self, pos, H, idx, ind: InductionState, params=None):
        """Energy only (Monte Carlo barostat trials): dipoles solved from the last converged ones
        (no history update); returns (total, InductionState with mu, iterations, overflow)."""
        pos, H = jnp.asarray(pos, jnp.float64), jnp.asarray(H, jnp.float64)
        P = self.charges_at(pos, H, self._atoms(params))
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
        """dE/d eps (3, 3) at fixed mu, plus the LJ tail impulse term (-E_lrc I) if lj_lrc.
        molecular: molecules translated with their centres of mass (virtual sites move with them);
        otherwise every position is scaled affinely, which virtual sites do not follow (refused)."""
        if not molecular and self.has_vsites:
            raise NotImplementedError("atomic (affine) strain derivative with virtual sites: the sites would have to be "
                                      "rebuilt from the deformed parents; use molecular=True")
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
