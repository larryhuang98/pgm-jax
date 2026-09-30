"""pGM + Lennard-Jones force field for molecular dynamics: periodic box, smooth PME, pair list.

Contents: the settings (`MDSettings` with its groups `Terms`, `Cutoffs`, `NeighborList`,
`PMESettings`, `Induction`, `ExtendedLagrangian`; `ewald_beta_for`, `elec_cutoff_settings`), the
state and result types (`InductionState`, `Result`), the force field `PGMForceField` (pair rows,
PME, induced-dipole solvers, energies, forces, strain derivative) and `full_strain_derivative`.

Energy (kJ/mol) with induced dipoles mu ([1]_ Sec. II D for PME):

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
cutoff (fixed capacity; the driver re-sizes and repeats on overflow).  Rows are stored as a structure of arrays
(index, x, y,
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

Extended-Lagrangian dipoles (`iel`, docs/iel.md; [2]_): auxiliary dipoles x ride along with the
atoms, x_{n+1} = 2 x_n - x_{n-1} + kappa (mu_n - x_n) + a sum_k c_k x_{n-k} (Niklasson's
dissipative Verlet [3]_), and replace the predictor + CG:
  * "0scf" (iEL/0-SCF): one field sweep, r = field(q, p + x) - x/alpha, mu = x + delta with
    delta = alpha r, and the shadow energy U~(R, x) = U(R, x) - sum alpha |r|^2 / 2
    = U(R, x + delta) - U_es(0, delta), which is stationary in delta, so its exact forces are the
    fixed-dipole forces at mu minus the fixed-dipole forces of U_es(0, delta) (the dipole-dipole
    energy of delta alone: rows, PME, self, in the same passes as the forces).  E_kin + U~ is conserved;
    U~ - U* is second order in the error of x.
  * "scf" (iEL/SCF): iel_iter CG iterations started from x, forces at fixed mu.
The first steps (and those after an accepted Monte Carlo volume move) are solved to dipole_tol.

Differentiability: E(pos, H, theta) at fixed mu is differentiable throughout (forces, virial,
dE/dtheta by Hellmann-Feynman).  With `differentiable=True`, compute() also returns forces and
dipoles with exact derivatives: the dipole solve is a jax.custom_vjp whose backward pass solves
A lam = mu_bar by CG (A symmetric) and pulls lam back through b - A mu at the solution.

External field (`efield=(E, offset)` in compute / energy / energy_fixed_mu / strain_derivative;
md/efield.py): a uniform field E (V/nm) adds -E . M to the energy, M = sum q r + sum d + offset (e nm;
offset: the dipole of re-wrapped charged molecules), q_i E to the forces and the torques on the
covalent dipoles, and E to the right-hand side of the induction equations.  Constant electric
displacement (`efield=(D, offset, "D")`, D/eps0 in V/nm): the field is F(M) = D - M / (eps0 V) and
the energy V eps0 |F(M)|^2 / 2; the induction operator gains the all-to-all term (4 pi / V) sum mu
(md/efield.py).  efield=None is the field-free code path, unchanged.

Precision: 'mixed' evaluates pair kernels, PME and CG vectors in float32 and accumulates energies,
dot products, positions and forces in float64; 'double' uses float64 throughout.  float32 matrix
products are requested at full precision (NVIDIA GPUs otherwise use TF32, ~1e-3 relative error).

Symbols: KE the Coulomb constant (units.py) [kJ/mol nm/e^2], b0 = ewald_beta [1/nm], b_ij the
Gaussian screening of the pair (1 / sqrt(2 (R_i^2 + R_j^2)), R the pGM radii) [1/nm], B_n the
radial kernels of md/kernels.py, p the permanent (covalent) dipoles, alpha the polarizabilities
[nm^3], Q the net charge, V the volume.  Internal field units e/nm^2 (energy KE M . F).

Units: nm, e, e nm, nm^3, kJ/mol (forces kJ/mol/nm).

References
----------
.. [1] H. Wei, R. Qi, J. Wang, P. Cieplak, Y. Duan, R. Luo, J. Chem. Phys. 153, 114116 (2020).
.. [2] A. Albaugh, A. M. N. Niklasson, T. Head-Gordon, J. Phys. Chem. Lett. 8, 1714 (2017).
.. [3] A. M. N. Niklasson, P. Steneteg, A. Odell, N. Bock, M. Challacombe, C. J. Tymczak,
   E. Holmstrom, G. Zheng, V. Weber, J. Chem. Phys. 130, 214109 (2009).
"""

from __future__ import annotations

import dataclasses as _dc
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..de import DE_ALPHA, DE_BETA, de_groups, de_long_range, de_pair_grad, de_tail_impulse
from ..lj import lj_long_range
from ..options import check_vdw, elec_flags
from ..system import System
from ..units import KE
from ..vdw import gvdw_long_range, gvdw_pair
from ._jaxmd import dataclasses
from .box import centers_of_mass, min_image, volume
from .kernels import erf_kernels, erf_kernels_closed
from .pme import PME, grid_size
from .topology import MDTopology

if TYPE_CHECKING:
    from jax.typing import ArrayLike, DTypeLike

    from .flux import ChargeFlux

_SQRT_PI = math.sqrt(math.pi)


@_dc.dataclass(frozen=True)
class Terms:
    """The interactions of the force field (immutable).

    Parameters
    ----------
    elec : {"q", "qp", "qi", "qpi"}
        Electrostatics: "q" charges, "qp" charges + permanent dipoles, "qi" charges + induction,
        "qpi" pGM (charges, permanent and induced dipoles; options.py).  Quadrupoles are not in
        the MD engine.
    vdw : {"lj", "de", "gvdw", "none"}
        Van der Waals term: "lj" (Lennard-Jones), "de" (double exponential of DEGAUSS, de.py, with the
        LJ well depth and minimum), "gvdw" (vdw.py; pmemd-pgm igvdw=1) or "none".
    gvdw_rep : {"gauss", "slater"}
        GVDW repulsion: "gauss" (gvdw_rep_form=0) or "slater" (=1).
    lj_lrc : bool
        Long-range correction of the r^-6 tail of LJ or of the GVDW dispersion, or of the DE tail
        (vdwmeth = 1).
    de_alpha, de_beta : float
        Repulsive and attractive exponents of the DE form (alpha > beta > 0; vdw = "de").
    """

    elec: str = "qpi"
    vdw: str = "lj"
    gvdw_rep: str = "gauss"
    lj_lrc: bool = True
    de_alpha: float = DE_ALPHA
    de_beta: float = DE_BETA


@_dc.dataclass(frozen=True)
class Cutoffs:
    """Cutoffs of the pair terms and of the neighbour list (immutable).

    Parameters
    ----------
    cutoff : float
        Van der Waals cutoff [nm], and the electrostatics cutoff unless elec_cutoff is set
        (pmemd: cut / vdw_cutoff).
    elec_cutoff : float or None
        Real-space electrostatics cutoff [nm] (pmemd: es_cutoff); None: `cutoff`.  Shorter than
        cutoff: set the Ewald coefficient and the PME grid for it (elec_cutoff_settings).
    """

    cutoff: float = 0.9
    elec_cutoff: float | None = None


@_dc.dataclass(frozen=True)
class NeighborList:
    """The Verlet neighbour list of the engines (immutable).

    Parameters
    ----------
    skin : float
        Skin added to the pair cutoff [nm] (pmemd: skinnb); the list is rebuilt when an atom (or a
        group centre) has moved by half of it.
    mode : {"auto", "molecule", "atom"}
        "auto" (a list of molecule / group centres when the box is large enough for the groups'
        radius, else of atoms), "molecule" or "atom".
    """

    skin: float = 0.1
    mode: str = "auto"


@_dc.dataclass(frozen=True)
class PMESettings:
    """Smooth particle-mesh Ewald (immutable).

    Parameters
    ----------
    ewald_beta : float
        Ewald coefficient [1/nm] (pmemd: ew_coeff; 4.0 /nm = 0.4 /A).
    grid : tuple of 3 int or None
        Grid points per box vector (pmemd: nfft1..3); None: from `spacing` and the box.
    spacing : float
        Largest grid spacing [nm] when grid is None.
    order : int
        B-spline order (pmemd: order).
    """

    ewald_beta: float = 4.0
    grid: tuple | None = None
    spacing: float = 0.08
    order: int = 6


@_dc.dataclass(frozen=True)
class ExtendedLagrangian:
    """Extended-Lagrangian induced dipoles (docs/iel.md; immutable).

    Auxiliary dipoles x are propagated by Niklasson's time-reversible Verlet with dissipation,
    x_{n+1} = 2 x_n - x_{n-1} + kappa (mu_n - x_n) + a sum_k c_k x_{n-k}.

    Parameters
    ----------
    scheme : {"none", "0scf", "scf"}
        "none": SCF from the predictor (Induction); "0scf": iEL/0-SCF, no CG, mu = x + alpha r(x)
        with the exact forces of the shadow energy; "scf": iEL/SCF, CG started from x.
    iterations : int
        "scf": CG iterations per step (0: to Induction.tol).
    order : int
        Niklasson dissipation order K (3..9; 0: none, exactly time-reversible).
    kappa : float or None
        kappa = (omega dt)^2; None: Niklasson's value for K (1.0 for K = 0).
    alpha : float or None
        Dissipation strength a; None: Niklasson's value for K.
    precond : {"block", "jacobi"}
        "0scf": delta = omega alpha r ("jacobi") or omega M^-1 r with M = 1/alpha + the
        intramolecular row blocks of molecules of <= 8 atoms ("block").
    omega : float
        "0scf": mu = x + omega alpha r(x) (omega < 1: damped Jacobi step).
    shadow : bool
        "0scf": energy and forces of the shadow potential (exact, conserved); False: fixed-dipole
        forces at mu (error second order in mu - x).
    """

    scheme: str = "none"
    iterations: int = 1
    order: int = 7
    kappa: float | None = None
    alpha: float | None = None
    precond: str = "block"
    omega: float = 1.0
    shadow: bool = True


@_dc.dataclass(frozen=True)
class Induction:
    """The induced-dipole solver (preconditioned CG, pmemd-pgm's scheme) and its predictor (immutable).

    Parameters
    ----------
    tol : float
        Convergence: max |alpha r| / mean |alpha b| (pmemd: dipole_scf_tol); 1e-4 is about 20 %
        faster with an NVE drift of 0.02 kT/ns/dof.
    max_iter : int
        Largest number of CG iterations (pmemd: scf_cg_niter).
    predictor : {"mu4", "mu3", "ls", "none"}
        Initial guess from earlier steps: "mu4", "mu3", "ls" (least squares) or "none".
    fused : bool
        Fused initial residual of the mu3 / mu4 predictors.
    norm_refresh : int
        Steps between unfused steps (they refresh the convergence normaliser).
    local_cut : float
        Cutoff of the local preconditioner [nm] (pmemd: scf_local_cut).
    local_niter : int
        Iterations of the local preconditioner (pmemd: scf_local_niter); 0: Jacobi (fastest on GPU).
    peek : float
        Coefficient of the peek step after the CG (pmemd: scf_sor_coefficient; 0: none).
    extrap_order : int
        Order of the "ls" predictor (pmemd: dipole_scf_init_order).
    extrap_steps : int
        Step of the "ls" predictor (pmemd: dipole_scf_init_step).
    iel : ExtendedLagrangian
        Extended-Lagrangian dipoles (scheme "none": off, the solver above from the predictor).
    """

    tol: float = 1e-5
    max_iter: int = 50
    predictor: str = "mu4"
    fused: bool = True
    norm_refresh: int = 1000
    local_cut: float = 0.3
    local_niter: int = 0
    peek: float = 0.65
    extrap_order: int = 3
    extrap_steps: int = 2
    iel: ExtendedLagrangian = _dc.field(default_factory=ExtendedLagrangian)


# flat names of the settings (the pmemd-like names of the single-level MDSettings of pgm_jax up to
# commit e72c57c, still accepted by MDSettings.replace) -> path of groups and field
FLAT_SETTINGS = {
    "elec": ("terms", "elec"),
    "vdw": ("terms", "vdw"),
    "gvdw_rep": ("terms", "gvdw_rep"),
    "lj_lrc": ("terms", "lj_lrc"),
    "de_alpha": ("terms", "de_alpha"),
    "de_beta": ("terms", "de_beta"),
    "cutoff": ("cutoffs", "cutoff"),
    "elec_cutoff": ("cutoffs", "elec_cutoff"),
    "skin": ("neighbors", "skin"),
    "neighbor_list": ("neighbors", "mode"),
    "ewald_beta": ("pme", "ewald_beta"),
    "pme_grid": ("pme", "grid"),
    "pme_spacing": ("pme", "spacing"),
    "pme_order": ("pme", "order"),
    "dipole_tol": ("induction", "tol"),
    "max_iter": ("induction", "max_iter"),
    "predictor": ("induction", "predictor"),
    "fused": ("induction", "fused"),
    "norm_refresh": ("induction", "norm_refresh"),
    "local_cut": ("induction", "local_cut"),
    "local_niter": ("induction", "local_niter"),
    "peek": ("induction", "peek"),
    "extrap_order": ("induction", "extrap_order"),
    "extrap_steps": ("induction", "extrap_steps"),
    "iel": ("induction", "iel", "scheme"),
    "iel_iter": ("induction", "iel", "iterations"),
    "iel_order": ("induction", "iel", "order"),
    "iel_kappa": ("induction", "iel", "kappa"),
    "iel_alpha": ("induction", "iel", "alpha"),
    "iel_precond": ("induction", "iel", "precond"),
    "iel_omega": ("induction", "iel", "omega"),
    "iel_shadow": ("induction", "iel", "shadow"),
}


def _set_path(obj: Any, path: tuple, value: Any) -> Any:
    """Return a copy of the frozen dataclass `obj` with the field at `path` (nested field names) set."""
    if len(path) == 1:
        return _dc.replace(obj, **{path[0]: value})
    return _dc.replace(obj, **{path[0]: _set_path(getattr(obj, path[0]), path[1:], value)})


_GROUPS = {
    "terms": Terms,
    "cutoffs": Cutoffs,
    "neighbors": NeighborList,
    "pme": PMESettings,
    "induction": Induction,
}


@_dc.dataclass(frozen=True)
class MDSettings:
    """Force-field and solver settings of the MD engines, in groups.

        MDSettings()                                           # the defaults
        MDSettings(induction=Induction(tol=1e-6), precision="double")
        MDSettings().replace(dipole_tol=1e-6, cutoff=0.8, iel="0scf")   # flat (pmemd-like) names

    Frozen and hashable (compiled functions are cached on it); change it with `replace`.  Every
    setting is static for the compiled steps (a change recompiles).

    Parameters
    ----------
    terms : Terms
        Electrostatics and van der Waals terms.
    cutoffs : Cutoffs
        Cutoffs of the pair terms [nm].
    neighbors : NeighborList
        Neighbour-list skin [nm] and kind.
    pme : PMESettings
        Ewald coefficient and PME grid.
    induction : Induction
        Induced-dipole solver, predictor and extended-Lagrangian dipoles.
    precision : {"mixed", "double"}
        "mixed" (float32 kernels, float64 accumulation where it matters) or "double".
    differentiable : bool
        Forces and induced dipoles differentiable (reverse mode) in parameters, positions and
        box: implicit differentiation of the dipole solve, one adjoint CG per gradient.
    adjoint_tol : float
        Convergence of the adjoint CG: max |alpha r| / mean |alpha rhs|.
    """

    terms: Terms = _dc.field(default_factory=Terms)
    cutoffs: Cutoffs = _dc.field(default_factory=Cutoffs)
    neighbors: NeighborList = _dc.field(default_factory=NeighborList)
    pme: PMESettings = _dc.field(default_factory=PMESettings)
    induction: Induction = _dc.field(default_factory=Induction)
    precision: str = "mixed"
    differentiable: bool = False
    adjoint_tol: float = 1e-6

    def replace(self, **changes: Any) -> MDSettings:
        """Return a copy with some settings changed.

        Parameters
        ----------
        **changes
            Top-level fields (precision, differentiable, adjoint_tol), whole groups (terms=Terms(...),
            cutoffs, neighbors, pme, induction), or single settings under their flat names
            (FLAT_SETTINGS: dipole_tol, cutoff, skin, pme_grid, iel="0scf", iel_iter, ...).

        Returns
        -------
        MDSettings

        Raises
        ------
        TypeError
            An unknown name.
        """
        out = self
        for k, v in changes.items():
            if k in _GROUPS or k in ("precision", "differentiable", "adjoint_tol"):
                out = _dc.replace(out, **{k: v})
            elif k in FLAT_SETTINGS:
                out = _set_path(out, FLAT_SETTINGS[k], v)
            else:
                raise TypeError(f"MDSettings.replace: unknown setting {k!r}")
        return out

    @property
    def perm_dipoles(self) -> bool:
        """Whether the electrostatics have permanent dipoles."""
        return elec_flags(self.terms.elec)[0]

    @property
    def has_induction(self) -> bool:
        """Whether the electrostatics have induced dipoles."""
        return elec_flags(self.terms.elec)[1]

    @property
    def dtype(self) -> DTypeLike:
        """Floating-point type of the kernels (float32 in mixed precision)."""
        return jnp.float32 if self.precision == "mixed" else jnp.float64

    @property
    def elec_rc(self) -> float:
        """Real-space electrostatics cutoff (nm)."""
        c = self.cutoffs
        return float(c.cutoff) if c.elec_cutoff is None else float(c.elec_cutoff)

    @property
    def pair_cutoff(self) -> float:
        """Cutoff of the pair rows and of the neighbour list [nm].

        The larger of the electrostatics and van der Waals cutoffs (the latter only with a van der
        Waals term).
        """
        return self.elec_rc if self.terms.vdw == "none" else max(float(self.cutoffs.cutoff), self.elec_rc)

    def describe_induction(self) -> str:
        """Return the induced-dipole scheme for log headers."""
        ind = self.induction
        x = ind.iel
        if x.scheme == "none":
            return f"predictor {ind.predictor}{' (fused)' if ind.fused else ''}, dipole tol {ind.tol:g}"
        kap, a, _ = _XL[x.order]
        kap = kap if x.kappa is None else x.kappa
        a = a if x.alpha is None else x.alpha
        what = (
            ("iEL/0-SCF" if x.shadow else "iEL/0-SCF (fixed-dipole forces)")
            if x.scheme == "0scf"
            else f"iEL/SCF ({x.iterations} CG iterations)"
            if x.iterations > 0
            else "iEL/SCF (CG to tol)"
        )
        om = f", omega {x.omega:g}" if (x.scheme == "0scf" and x.omega != 1.0) else ""
        om += ", block preconditioner" if (x.scheme == "0scf" and x.precond == "block") else ""
        return (
            f"{what} extended-Lagrangian dipoles (K {x.order}, kappa {kap:g}, a {a:g}{om}), "
            f"dipole tol {ind.tol:g} (warm-up)"
        )

    def describe_cutoffs(self) -> str:
        """Return the cutoffs for log headers."""
        c = self.cutoffs
        if c.elec_cutoff is None or self.elec_rc == float(c.cutoff):
            return f"cutoff {c.cutoff} nm"
        return f"cutoff {c.cutoff} nm (electrostatics {self.elec_rc} nm, Ewald {self.pme.ewald_beta:.4g} /nm)"


# Direct-sum tolerance, in Amber's convention (ewald_beta_for), of the default pair 0.9 nm / 4.0 nm^-1.
DSUM_TOL = 3.95e-8


def ewald_beta_for(elec_cutoff: float, dsum_tol: float = DSUM_TOL) -> float:
    """Return the Ewald coefficient [1/nm] for a real-space cutoff and an Amber direct-sum tolerance.

    erfc(beta rc) / rc = dsum_tol with rc in Angstrom (sander / pmemd `dsum_tol`), solved by
    bisection as Amber does.  Amber's default dsum_tol = 1e-5 gives the ew_coeff of its outputs
    (0.34864 A^-1 at 8 A, 0.30768 at 9 A).  The pGM-JAX default pair 0.9 nm / 4.0 nm^-1
    (erfc(3.6) = 3.6e-7; pmemd-pgm's ew_coeff 0.4 A^-1 at 9 A) is dsum_tol = 3.95e-8 (DSUM_TOL); at
    that tolerance 0.8 nm needs 4.52 nm^-1, 0.7 nm 5.19 and 0.6 nm 6.09.  The rule bounds the
    charge-charge term; the dipole terms of pGM decay with higher powers of beta, so the measured
    real-space force error grows as the cutoff shrinks (ubiquitin: 4e-6 at 0.9 nm, 8e-6 at 0.7, 3e-5
    at 0.6; pGM water 2e-5 at 0.7, 1e-4 at 0.6).

    Parameters
    ----------
    elec_cutoff : float
        Real-space cutoff [nm].
    dsum_tol : float
        Direct-sum tolerance (Amber's convention, per Angstrom).

    Returns
    -------
    float
        beta [1/nm].

    Raises
    ------
    ValueError
        A non-positive cutoff, or dsum_tol outside (0, 1/rc_A).
    """
    rc = float(elec_cutoff)
    if not rc > 0.0:
        raise ValueError(f"elec_cutoff must be positive, got {elec_cutoff}")

    def f(b: float) -> float:  # erfc(b rc) / rc_A - dsum_tol (rc_A = 10 rc)
        return math.erfc(b * rc) / (10.0 * rc) - dsum_tol

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
    """Return MDSettings arguments for a real-space electrostatics cutoff.

    ewald_beta from `ewald_beta_for` and the PME grid spacing h = 0.08 nm (4.0 / beta)^exponent,
    scaled from the default pair (4.0 nm^-1, 0.08 nm).

        MDSettings().replace(cutoff=0.9, **elec_cutoff_settings(0.7))   # LJ at 0.9 nm, electrostatics at 0.7

    At a fixed spline order the PME force error of pGM (charges and dipoles) grows roughly as
    beta^9.5 h^6 (measured, order 6, ubiquitin in water): keeping beta x h fixed (exponent 1: 0.7 nm,
    5.19 nm^-1, 0.0616 nm) is the cheaper grid but multiplies the force error by 2.5 at 0.7 nm;
    exponent 1.6 (the default: 0.7 nm, 5.19 nm^-1, 0.0527 nm) keeps the error of the default
    settings (3e-5 relative for ubiquitin, 7e-5 for pGM water at 0.8 and 0.7 nm) for 25 % more PME
    work per CG iteration (8 % per step for ubiquitin).  Measurements: docs/protein_ff.md (What limits
    the speed).

    Parameters
    ----------
    elec_cutoff : float
        Real-space electrostatics cutoff [nm].
    dsum_tol : float
        Direct-sum tolerance (`ewald_beta_for`).
    exponent : float
        Exponent of the grid-spacing rule.

    Returns
    -------
    dict
        {"elec_cutoff" [nm], "ewald_beta" [1/nm], "pme_spacing" [nm]} (flat MDSettings names).
    """
    b = ewald_beta_for(elec_cutoff, dsum_tol)
    return {"elec_cutoff": float(elec_cutoff), "ewald_beta": b, "pme_spacing": 0.08 * (4.0 / b) ** exponent}


# polynomial extrapolation coefficients of the predictors, applied to the history newest first
_PRED = {"mu3": (3.0, -3.0, 1.0), "mu4": (4.0, -6.0, 4.0, -1.0)}


def full_strain_derivative(
    energy: Callable[..., jax.Array],
    pos: jax.Array,
    H: jax.Array,
    mol: jax.Array | None = None,
    com: jax.Array | None = None,
    vectors: tuple = (),
) -> jax.Array:
    """Return dE/d eps (3, 3) of energy(x, H, *vectors) under a homogeneous strain.

    The energy is evaluated by code that assumes a lower-triangular box (the row displacements
    subtract lattice vectors component by component).  The strain maps x -> x + (com eps^T)[mol]
    (molecular scaling; com None: x -> x (1 + eps)^T), H -> H (1 + eps)^T, with the vectors held
    fixed (induced dipoles, an external field: arrays of rows (..., 3)).

    Parameters
    ----------
    energy : Callable
        energy(x, H, *vectors) [kJ/mol], differentiable.
    pos : jax.Array (N, 3)
        Positions [nm].
    H : jax.Array (3, 3)
        Lower-triangular box [nm].
    mol : jax.Array (N,) int, optional
        Molecule of every atom (with com).
    com : jax.Array (M, 3), optional
        Molecular centres of mass [nm] (None: atomic scaling).
    vectors : tuple of jax.Array
        Row-vector arrays held fixed.

    Returns
    -------
    jax.Array (3, 3)
        dE/d eps [kJ/mol].

    Notes
    -----
    Strains eps_ab with a < b and the diagonal keep H lower triangular and are differentiated
    directly; the lower components follow from rotation invariance of energy(x R^T, H R^T,
    v R^T, ...): the atomic strain derivative W_at = W + G, G_ab = sum_i dE/dx_ia (x_i - c_i)_b (zero
    for atomic scaling), satisfies W_at - W_at^T = T^T - T with T_ab = sum_v sum_k dE/dv_ka v_kb (zero
    for converged induced dipoles and no field).
    """

    def e(eps: jax.Array, dx: jax.Array, *vs: jax.Array) -> jax.Array:  # energy at strain eps, shift dx
        x = pos + dx + ((com @ eps.T)[mol] if com is not None else pos @ eps.T)
        return energy(x, H @ (jnp.eye(3) + eps).T, *vs)

    grads = jax.grad(e, argnums=tuple(range(2 + len(vectors))))(jnp.zeros((3, 3)), jnp.zeros_like(pos), *vectors)
    W, g = grads[0], grads[1]
    G = jnp.zeros((3, 3)) if com is None else g.T @ (pos - com[mol])
    T = jnp.zeros((3, 3))
    for v, gv in zip(vectors, grads[2:]):
        T = T + jnp.reshape(gv, (-1, 3)).T @ jnp.reshape(v, (-1, 3))
    return jnp.triu(W) + jnp.tril(W.T + G.T - G + T.T - T, -1)


# Niklasson's dissipative extended-Lagrangian Verlet (Niklasson et al., JCP 130, 214109 (2009),
# Table I): K -> (kappa, a, c_0..c_K); x_{n+1} = 2 x_n - x_{n-1} + kappa (mu_n - x_n) + a sum_k c_k x_{n-k}
_XL = {
    0: (1.0, 0.0, (0.0,)),
    3: (1.69, 0.150, (-2, 3, 0, -1)),
    4: (1.75, 0.057, (-3, 6, -2, -2, 1)),
    5: (1.82, 0.018, (-6, 14, -8, -3, 4, -1)),
    6: (1.84, 0.0055, (-14, 36, -27, -2, 12, -6, 1)),
    7: (1.86, 0.0016, (-36, 99, -88, 11, 32, -25, 8, -1)),
    8: (1.88, 0.00044, (-99, 286, -286, 78, 78, -90, 42, -10, 1)),
    9: (1.89, 0.00012, (-286, 858, -936, 364, 168, -300, 184, -63, 12, -1)),
}


@dataclasses.dataclass
class InductionState:
    """Induced-dipole solver state: converged dipoles, predictor history, convergence normaliser.

    A JAX-MD dataclass (pytree), part of MDState; float64 unless noted.  S = extrap_steps, K1 =
    PGMForceField.xl_len.

    Parameters
    ----------
    mu : jax.Array (N, 3)
        Converged induced dipoles of the last solve [e nm].
    hist : jax.Array (4, N, 3)
        Converged dipoles of the last steps, newest first [e nm] (anchored MTS: mu - mu_fast).
    count : jax.Array () int32
        Steps recorded (kept unbatched in stacked replica states).
    norm : jax.Array ()
        mean |alpha b| of the last unfused step (convergence normaliser) [e nm].
    rec : jax.Array (4, S, N, 3)
        "ls" records: alpha b, mu, mu - pred1, mu - pred2.
    pred : jax.Array (2, N, 3)
        "ls" order-1 and order-2 predictions.
    lscount : jax.Array (4,) int32
        "ls" record counts.
    xl : jax.Array (K1, N, 3), optional
        Extended-Lagrangian auxiliary dipoles x_{n+1}, x_n, ... (induction.iel; None otherwise).
    """

    mu: jnp.ndarray  # (N, 3) e nm
    hist: jnp.ndarray  # (4, N, 3) converged dipoles of the last steps
    count: jnp.ndarray  # () int32: steps recorded
    norm: jnp.ndarray  # () mean|alpha b| of the last unfused step
    rec: jnp.ndarray  # (4, S, N, 3) "ls" records: alpha b, mu, mu - pred1, mu - pred2
    pred: jnp.ndarray  # (2, N, 3) "ls" order-1 and order-2 predictions
    lscount: jnp.ndarray  # (4,) int32
    xl: jnp.ndarray = None  # (K1, N, 3) extended-Lagrangian auxiliary dipoles x_{n+1}, x_n, ... (induction.iel)


class Result(NamedTuple):
    """Result of PGMForceField.compute (a NamedTuple, i.e. a pytree).

    Parameters
    ----------
    energy : dict
        "elec", "vdw", "total" (and "field" with an external field) [kJ/mol].
    forces : jax.Array (N, 3) float64
        Forces [kJ/mol/nm].
    induction : InductionState
        Updated solver state.
    iterations : jax.Array () int32
        CG iterations.
    residual : jax.Array ()
        Final max|alpha r| / mean|alpha b| (before the peek step).
    overflow : jax.Array () bool
        Row capacity exceeded (results invalid; the driver re-sizes and repeats).
    geometry : dict, optional
        compute(keep_geometry=True): the electrostatic rows (md/mts.py).
    dipole : jax.Array (3,), optional
        With an external field: M = sum q r + sum d + offset [e nm].
    """

    energy: dict  # kJ/mol: elec, vdw, total
    forces: jnp.ndarray  # (N, 3) kJ/mol/nm, float64
    induction: InductionState
    iterations: jnp.ndarray
    residual: jnp.ndarray  # final max|alpha r| / mean|alpha b| (before the peek step)
    overflow: jnp.ndarray  # row capacity exceeded (results invalid; driver re-sizes and repeats)
    geometry: dict | None = None  # compute(keep_geometry=True): the electrostatic rows (md/mts.py)
    dipole: jnp.ndarray | None = None  # with an external field: M = sum q r + sum d + offset (e nm)


def _zero_cotangent(x: Any) -> Any:
    """Return a zero cotangent for x (float0 for integer leaves, as custom_vjp requires)."""
    x = jnp.asarray(x)
    if jnp.issubdtype(x.dtype, jnp.inexact):
        return jnp.zeros_like(x)
    return np.zeros(x.shape, dtype=jax.dtypes.float0)


def _dot(a: jax.Array, b: jax.Array) -> jax.Array:
    """Return sum(a b), accumulated in float64."""
    return jnp.sum((a * b).astype(jnp.float64))


def _push(stack: jax.Array, x: jax.Array) -> jax.Array:
    """Return the history stack with x prepended and the oldest entry dropped (newest first)."""
    return jnp.concatenate([x[None].astype(stack.dtype), stack[:-1]], 0)


def _div_alpha(x: jax.Array | float, alpha: jax.Array, mask: bool) -> jax.Array:
    """Return x / alpha; with `mask`, 0 where alpha = 0.

    Atoms (or virtual sites) with zero polarizability are not polarizable: their induced dipole
    stays 0 (the Jacobi-preconditioned CG, z = alpha r, never moves it, and every initial guess is 0
    there) and their mu^2 / (2 alpha) is 0, with no 0/0 in values or gradients (the derivative with
    respect to such an alpha is taken as 0).  The mask is static (PGMForceField.alpha_mask: some
    alpha of the system's parameter table is 0), so that systems without such atoms run the plain
    division, bit for bit.
    """
    if not mask:
        return x / alpha
    pol = alpha != 0
    return jnp.where(pol, x / jnp.where(pol, alpha, 1.0), 0.0)


class PGMForceField:
    """The pGM + van der Waals force field of an MD system (module docstring for the physics).

    Built on the host for one system, box shape and settings; its methods are JAX functions of
    positions, box, parameters and solver state, called inside the engines' compiled steps.  The row
    capacities (`mc`, `mc_e`) are static sizes set on the host (`size_rows`, `grow_rows`,
    `fit_rows`): a change requires re-jitting the callers.  Not a pytree.

        ff = PGMForceField(system, H, MDSettings())
        idx = ff.rows_for(pos, H)                        # candidate rows of a single frame
        res = ff.compute(pos, H, idx, ff.init_induction())
        res.energy["total"], res.forces                  # [kJ/mol], [kJ/mol/nm]

    Parameter pytree `params` (all methods): a dict of the system's parameter table (System.expand;
    None: the system's initial values), optionally with "flux" (md/flux.py).

    Attributes
    ----------
    sys : System
        The system.
    s : MDSettings
        Settings.
    pd, ind : bool
        Permanent dipoles, induced dipoles (from the electrostatics level).
    cd : jnp.dtype
        Compute dtype of the kernels (float32 in mixed precision).
    n : int
        Number of atoms N.
    b0 : float
        Ewald coefficient [1/nm].
    c_self : float
        4 b0^3 / (3 sqrt(pi)), the dipole self-energy coefficient [1/nm^3].
    pme : PME
        Reciprocal space (md/pme.py).
    mol, cov_i, cov_j : jax.Array int
        Molecule of every atom; covalent-dipole atom pairs.
    masses : jax.Array (N,)
        Masses [amu] (set to the repartitioned masses by FlexibleSimulation).
    alpha_mask : bool
        Some polarizability of the table is 0 (static; masks the divisions by alpha).
    mc, mc_e : int or None
        Row capacity (pairs kept per row) and, for split rows, its electrostatic part.
    ms : int
        Capacity of the short-range rows of the local preconditioner.
    rc_e, rc_v, rc_pair : float
        Electrostatics, van der Waals and row cutoffs [nm].
    split : bool
        Rows split at the electrostatics cutoff (elec_cutoff < cutoff with van der Waals).
    topology : MDTopology
        Pair topology (special partners, groups).
    special, special_w, gid, sg : jax.Array
        Special partners (N, S), their weights, group of every atom, special groups (N, Gs).
    first : jax.Array (M,) int
        First atom of every molecule.
    flux : ChargeFlux or None
        Charge flux.
    """

    def __init__(
        self,
        sys: System,
        H: ArrayLike,
        settings: MDSettings = MDSettings(),
        short_capacity: int = 48,
        row_capacity: int | None = None,
        topology: MDTopology | None = None,
        elec_capacity: int | None = None,
        flux: ChargeFlux | None = None,
    ) -> None:
        """Set up the force field of `sys` for boxes like H.

        Parameters
        ----------
        sys : System
            The system (quadrupoles are ignored with a warning).
        H : ArrayLike (3, 3)
            Box [nm] (sets the PME grid when settings.pme.grid is None).
        settings : MDSettings
            Settings.
        short_capacity : int
            Entries per row of the local preconditioner's short-range rows.
        row_capacity : int, optional
            Pairs kept per row (None: no compaction; the engines size it with `size_rows`).
        topology : MDTopology, optional
            Pair topology (None: MDTopology.rigid, the rigid-body engine).
        elec_capacity : int, optional
            Electrostatic part of the row capacity (split rows only).
        flux : ChargeFlux, optional
            Charge flux (md/flux.py).

        Raises
        ------
        ValueError
            An unknown predictor, van der Waals form or iel scheme, invalid iel settings (or iel with
            differentiable=True), non-positive cutoffs, elec_capacity without split rows, or a flux of
            another system.
        """
        self.sys, self.s = sys, settings
        P0 = sys.expand(None) if settings.terms.vdw == "de" else None  # concrete: the groups feed jitted code
        self._de_groups_cache = (
            de_groups(np.asarray(P0["lj_rmin_half"]), np.asarray(P0["lj_sqrt_eps"])) if P0 is not None else None
        )
        if settings.induction.predictor not in ("mu4", "mu3", "ls", "none"):
            raise ValueError(f"unknown predictor {settings.induction.predictor!r}")
        check_vdw(settings.terms.vdw, settings.terms.gvdw_rep)
        self.pd, self.ind = elec_flags(settings.terms.elec)
        if settings.induction.iel.scheme not in ("none", "0scf", "scf"):
            raise ValueError(f"unknown iel {settings.induction.iel.scheme!r} (none | 0scf | scf)")
        if settings.induction.iel.scheme != "none":
            if settings.induction.iel.order not in _XL:
                raise ValueError(f"iel_order must be one of {sorted(_XL)}, got {settings.induction.iel.order}")
            if settings.differentiable:
                raise ValueError("iel (extended-Lagrangian dipoles) and differentiable=True are exclusive")
            if settings.induction.iel.iterations < 0:
                raise ValueError("iel_iter must be >= 0")
            if not 0.0 < settings.induction.iel.omega < 2.0:
                raise ValueError("iel_omega must be in (0, 2)")
            if settings.induction.iel.precond not in ("jacobi", "block"):
                raise ValueError(f"iel_precond must be jacobi or block, got {settings.induction.iel.precond!r}")
        if any(len(m.quad) for m in sys.molecules):
            import warnings

            warnings.warn("quadrupole terms are ignored by the MD engine (gas phase only for now)", stacklevel=2)
        self.cd = settings.dtype
        self.n = sys.n
        self.b0 = float(settings.pme.ewald_beta)
        self.c_self = 4.0 * self.b0**3 / (3.0 * _SQRT_PI)
        grid = settings.pme.grid or grid_size(H, settings.pme.spacing)
        self.pme = PME(grid, settings.pme.order, self.b0, self.cd)
        mol = np.asarray(sys.mol)
        self.mol = jnp.asarray(mol)
        self.cov_i, self.cov_j = jnp.asarray(sys.cov_i), jnp.asarray(sys.cov_j)
        self.has_vsites = any(getattr(m, "vsites", None) for m in sys.molecules)  # md/vsites.py
        # atoms with alpha = 0 in the parameter table: induced dipoles masked (_div_alpha).  Parameters
        # passed at call time with new zeros need a force field built from a table with zeros
        self.alpha_mask = bool(np.any(np.asarray(sys.expand()["alpha"]) == 0.0))
        self.masses = jnp.asarray(sys.masses)
        self.S = max(1, int(settings.induction.extrap_steps))
        self.ms = int(short_capacity)
        self.mc = row_capacity  # pairs kept per row (None: no compaction)
        # cutoffs (nm): electrostatics, van der Waals, rows.  With elec_cutoff < cutoff (and a van der
        # Waals term) the rows are split: mc_e electrostatic entries, then mc - mc_e van der Waals ones
        self.rc_e, self.rc_v, self.rc_pair = settings.elec_rc, float(settings.cutoffs.cutoff), settings.pair_cutoff
        if not (self.rc_e > 0.0 and self.rc_v > 0.0):
            raise ValueError(
                f"cutoffs must be positive (cutoff {settings.cutoffs.cutoff}, "
                f"elec_cutoff {settings.cutoffs.elec_cutoff})"
            )
        self.split = settings.terms.vdw != "none" and self.rc_e < self.rc_v
        if elec_capacity is not None and not self.split:
            raise ValueError(
                "elec_capacity applies only to split rows (elec_cutoff < cutoff with a van der Waals term)"
            )
        self.mc_e = elec_capacity
        # special partners of every atom (fixed table with van der Waals weights; md/topology.py)
        self.topology = MDTopology.rigid(sys) if topology is None else topology
        self.special = jnp.asarray(self.topology.special)
        self.special_w = jnp.asarray(self.topology.special_w, jnp.float64)
        self.gid = jnp.asarray(self.topology.group, jnp.int32)
        self.sg = jnp.asarray(self.topology.special_groups)
        first = (
            np.searchsorted(mol, np.arange(sys.nmol))
            if np.all(np.diff(mol) >= 0)
            else np.array([int(np.nonzero(mol == k)[0][0]) for k in range(sys.nmol)])
        )
        self.first = jnp.asarray(first)
        self._blocks = None
        if settings.induction.iel.scheme == "0scf" and settings.induction.iel.precond == "block":
            self._blocks = self._block_layout(mol, first, sys.nmol)
        self.flux = flux  # md/flux.py ChargeFlux (geometry-dependent q and c) or None
        if flux is not None and (flux.n_atoms != sys.n or len(flux.cov_bond) != len(sys.cov_i)):
            raise ValueError(
                f"charge flux for {flux.n_atoms} atoms / {len(flux.cov_bond)} covalent dipoles; the "
                f"system has {sys.n} / {len(sys.cov_i)}"
            )

    # ------------------------------------------------------------------ building blocks
    def _atoms(self, params: dict | None) -> dict[str, jax.Array]:
        """Return the per-atom parameters (System.expand, float64), with "flux" when there is charge flux.

        Raises
        ------
        ValueError
            Flux parameters for a force field without charge flux.
        """
        P = self.sys.expand(params)
        P = {k: jnp.asarray(v, jnp.float64) for k, v in P.items()}
        if self.flux is not None:  # flux parameters ride along; charges_at applies them
            P["flux"] = self.flux.theta(params)
        elif isinstance(params, dict) and "flux" in params:
            raise ValueError("the parameters have charge-flux values but the force field has no charge flux")
        return P

    def charges_at(self, pos: jax.Array, H: jax.Array, P: dict[str, jax.Array]) -> dict[str, jax.Array]:
        """Return P with the charges and covalent-dipole strengths of the geometry pos (charge flux).

        P comes from `_atoms`; without flux (or when already applied: no "flux" key) P itself.  pos
        (N, 3) [nm], H (3, 3) [nm].
        """
        if "flux" not in P:
            return P
        q, cov = self.flux.charges(pos, H, P["q"], P["cov"], P["flux"])
        out = {k: v for k, v in P.items() if k != "flux"}
        out.update(q=q, cov=cov)
        return out

    def perm_dipoles(self, pos: jax.Array, H: jax.Array, cov_c: jax.Array) -> jax.Array:
        """Return the permanent dipoles p_i = sum_m c_m unit(r_j - r_i) (N, 3) [e nm].

        Summed over the covalent dipoles m = (i, j) (minimum image); zero without permanent
        dipoles.  cov_c (n_cov,) the strengths [e nm].
        """
        if len(self.sys.cov_i) == 0 or not self.pd:
            return jnp.zeros((self.n, 3))
        v = min_image(pos[self.cov_j] - pos[self.cov_i], H)
        u = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
        return jnp.zeros((self.n, 3)).at[self.cov_i].add(cov_c[:, None] * u)

    @staticmethod
    def _displacements(p: jax.Array, k: jax.Array, H: jax.Array) -> list[jax.Array]:
        """Return the minimum-image displacements p_i - p_k as three (N, C) arrays (x, y, z).

        Structure of arrays: the row kernels are memory bound and read components with unit stride.
        p (N, 3) positions in the compute dtype [nm]; k (N, C) partners; H (3, 3) reduced box [nm].
        """
        pk = p[k]
        x = [p[:, c][:, None] - pk[..., c] for c in range(3)]
        for c in (2, 1, 0):  # sequential reduction, reduced box
            n = jnp.round(x[c] / H[c, c])
            x = [x[j] - n * H[c, j] if j <= c else x[j] for j in range(3)]
        return x

    @property
    def intra(self) -> jax.Array:
        """The special-partner table (the name of the rigid engine's intramolecular table)."""
        return self.special

    def _intra_exact(self, pos: jax.Array, H: jax.Array, k: jax.Array, x: Sequence[jax.Array], cd: DTypeLike) -> tuple:
        """Return x with the special entries replaced by differences of within-molecule offsets.

        The special entries (first columns; same molecule) take differences of offsets from each
        molecule's first atom: exact in float32 whatever the absolute coordinates.
        """
        ni = self.special.shape[1]
        off = (pos - pos[self.first][self.mol]).astype(cd)
        xi = self._displacements(off, self.special, H)
        hit = (k[:, :ni] == self.special) & (self.special < self.n)
        return tuple(xc.at[:, :ni].set(jnp.where(hit, xic, xc[:, :ni])) for xc, xic in zip(x, xi))

    def _special_exact(
        self, pos: jax.Array, H: jax.Array, k: jax.Array, x: Sequence[jax.Array], sp: jax.Array
    ) -> tuple:
        """Return x with the special pairs' displacements replaced by within-molecule offset differences.

        The entries of the first sp.shape[1] columns where sp (as _intra_exact).
        """
        w = sp.shape[1]
        if w == 0:
            return x
        off = (pos - pos[self.first][self.mol]).astype(self.cd)
        xi = self._displacements(off, k[:, :w], H)
        return tuple(xc.at[:, :w].set(jnp.where(sp, xic, xc[:, :w])) for xc, xic in zip(x, xi))

    def _compact_parts(
        self, k: jax.Array, wv: jax.Array, masks: Sequence[jax.Array], widths: Sequence[int]
    ) -> tuple[list, jax.Array]:
        """Compact every row into consecutive parts; return the parts and the overflow flag.

        The entries where masks[j] go, in column order, to part j (widths[j] columns).  One scatter of
        the partner indices places all parts (the slots of two parts come from one scan, one count per 16
        bits); the scatter of anything else over the candidate rows is avoided: list entries have van der
        Waals weight 1, and the special entries, which lead each part, take their weights from the small
        special block.

        Returns
        -------
        parts : list of tuple
            Per part (k, within, weights, special-entry mask of the first min(S, width) columns).
        overflow : jax.Array () bool
            Some row has more entries than its part's width.

        Raises
        ------
        ValueError
            Two parts with candidate rows of 2^15 or more entries (the packed counters).
        """
        N, ni, C = self.n, self.special.shape[1], k.shape[1]
        if len(masks) == 1:
            slots = [jnp.cumsum(masks[0].astype(jnp.int32), axis=1) - 1]
        else:
            if C >= 1 << 15:
                raise ValueError(f"candidate rows of {C} entries: at most {(1 << 15) - 1} for the packed counters")
            cs = jnp.cumsum(masks[0].astype(jnp.int32) + (masks[1].astype(jnp.int32) << 16), axis=1)
            slots = [(cs & 0xFFFF) - 1, (cs >> 16) - 1]
        starts = [int(o) for o in np.cumsum([0] + list(widths))]
        tgt = jnp.full(k.shape, starts[-1], jnp.int32)  # past the end: dropped
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
            parts.append((kk[:, o : o + w], within, jnp.where(within, wp, 0.0), sp))
            overflow = overflow | (jnp.max(count) > w)
        return parts, overflow

    def _rows(self, pos: jax.Array, H: jax.Array, idx: jax.Array) -> tuple:
        """Return the pair rows: [special partners | candidates from the neighbour list], masked and compacted.

        Masked to the pair cutoff and, with a row capacity set, compacted to the pairs inside it.  List
        candidates in the atom's special groups are dropped (those pairs come from the table).

        Parameters
        ----------
        pos : jax.Array (N, 3)
            Positions [nm] (float64).
        H : jax.Array (3, 3)
            Box [nm].
        idx : jax.Array (N, C) int
            Candidate rows from the neighbour list (padding N).

        Returns
        -------
        k : jax.Array (N, W) int
            Partners.
        x : tuple of 3 jax.Array (N, W)
            Displacement components x_i - x_k [nm] (compute dtype).
        within : jax.Array (N, W) bool
            Valid pairs.
        wv : jax.Array (N, W)
            Van der Waals weights (0 off `within` and beyond the van der Waals cutoff).
        overflow : jax.Array () bool
            Row capacity exceeded.
        vrows : tuple or None
            Split rows: the van der Waals rows (k, x, within, weights) (_split_rows); None otherwise,
            when k, x, ... hold every pair.
        """
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
        within = keep & (r2 < self.rc_pair**2)
        overflow = jnp.zeros((), bool)
        if self.mc is not None:
            ((k, within, wv, _),), overflow = self._compact_parts(k, wv, (within,), (self.mc,))
            x = self._displacements(p, k, Hc)  # recompute on the compacted rows
            r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
        if self.rc_v < self.rc_pair:  # van der Waals cut before electrostatics
            wv = jnp.where(r2 < self.rc_v**2, wv, 0.0)
        x = self._intra_exact(pos, Hc, k, x, cd)
        return k, x, within, jnp.where(within, wv, 0.0), overflow, None

    def _split_rows(
        self,
        pos: jax.Array,
        p: jax.Array,
        H: jax.Array,
        k: jax.Array,
        x: Sequence[jax.Array],
        r2: jax.Array,
        keep: jax.Array,
        wv: jax.Array,
    ) -> tuple:
        """Return the rows split at the electrostatics cutoff (elec_cutoff < cutoff), as `_rows`.

        Electrostatic rows (pairs inside elec_cutoff, with their van der Waals weights) and van der
        Waals rows (pairs between elec_cutoff and cutoff with a nonzero van der Waals weight), compacted
        to their own capacities (mc_e and mc - mc_e) into separate arrays, so that the CG streams only
        the electrostatic ones (_compact_parts).  Compaction keeps the column order, so each part starts
        with its special partners, whose exact displacements are then found by count.  Without a
        capacity (single points) both parts span the candidate rows, masked.

        Raises
        ------
        ValueError
            Capacities with mc_e outside [0, mc].
        """
        N = self.n
        ni = self.special.shape[1]
        ein = keep & (r2 < self.rc_e**2)
        vin = keep & ~ein & (r2 < self.rc_v**2) & (wv != 0)
        if self.mc is None:
            sp = self.special < N
            xe = self._special_exact(pos, H, k, x, sp & ein[:, :ni])
            xv = self._special_exact(pos, H, k, x, sp & vin[:, :ni])
            return (k, xe, ein, jnp.where(ein, wv, 0.0), jnp.zeros((), bool), (k, xv, vin, jnp.where(vin, wv, 0.0)))
        if self.mc_e is None or not 0 <= self.mc_e <= self.mc:
            raise ValueError(f"split rows need 0 <= mc_e <= mc (got mc {self.mc}, mc_e {self.mc_e}); see size_rows")
        ((ke, ein, wve, spe), (kv, vin, wvv, spv)), overflow = self._compact_parts(
            k, wv, (ein, vin), (self.mc_e, self.mc - self.mc_e)
        )
        xe = self._special_exact(pos, H, ke, self._displacements(p, ke, H), spe)
        xv = self._special_exact(pos, H, kv, self._displacements(p, kv, H), spv)
        return ke, xe, ein, wve, overflow, (kv, xv, vin, wvv)

    def pair_counts(self, pos: jax.Array, H: jax.Array, idx: jax.Array) -> jax.Array:
        """Return the largest numbers of pairs in any row: (electrostatic rows, van der Waals rows) (2,) int.

        Evaluated without compaction; the second count is 0 unless the rows are split.
        """
        saved, self.mc = self.mc, None
        try:
            _, _, within, _, _, vrows = self._rows(pos, H, idx)
        finally:
            self.mc = saved
        ce = jnp.max(jnp.sum(within, axis=1))
        cv = jnp.zeros_like(ce) if vrows is None else jnp.max(jnp.sum(vrows[2], axis=1))
        return jnp.stack([ce, cv])

    def row_counts(self, pos: jax.Array, H: jax.Array, idx: jax.Array) -> jax.Array:
        """Return the largest number of pairs (special + list pairs inside the cutoffs) in any row."""
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

    def size_rows(self, pos: ArrayLike, H: ArrayLike, idx: jax.Array, factor: float = 1.2) -> tuple:
        """Set the row capacities from the largest pair counts at pos (host; static shapes: re-jit after).

        Half the neighbour list's head-room (pair counts inside a sphere fluctuate by a few per cent),
        in multiples of 8, at most the candidate width; split rows size each part.  Returns `capacity`.
        """
        ce, cv = (int(c) for c in jax.jit(self.pair_counts)(jnp.asarray(pos), jnp.asarray(H), idx))
        width = int(idx.shape[1]) + int(self.special.shape[1])

        def cap(c: int) -> int:  # count with head-room, a multiple of 8, at most the width
            return min(int(np.ceil((c * (1.0 + 0.5 * (factor - 1.0)) + 8) / 8.0) * 8), width)

        self.mc_e = cap(ce) if self.split else None
        self.mc = self.mc_e + cap(cv) if self.split else cap(ce)
        return self.capacity

    def grow_rows(self, old: tuple) -> None:
        """Widen every part of the rows by at least 8 after an overflow at capacities `old` (`capacity`).

        The driver re-sizes where the block started, where the rows fit; this keeps the capacities
        above what overflowed.
        """
        mc, mc_e = old
        if self.mc is None or mc is None:
            return
        if self.split and self.mc_e is not None and mc_e is not None:
            e = max(self.mc_e, mc_e + 8)
            self.mc, self.mc_e = e + max(self.mc - self.mc_e, mc - mc_e + 8), e
        else:
            self.mc = max(self.mc, mc + 8)

    def fit_rows(self, caps: Sequence[tuple | None]) -> None:
        """Set capacities that fit every one of `caps` (`capacity` tuples, e.g. one per replica).

        The largest of each part of the rows (static shapes: re-jit afterwards).
        """
        caps = [c for c in caps if c is not None and c[0] is not None]
        if not caps:
            return
        if self.split and all(c[1] is not None for c in caps):
            e = max(c[1] for c in caps)
            self.mc, self.mc_e = e + max(c[0] - c[1] for c in caps), e
        else:
            self.mc = max(c[0] for c in caps)

    def _pair_a(self, R: jax.Array, k: jax.Array) -> jax.Array:
        """Return the Gaussian screening a_ik = 1 / sqrt(2 (R_i^2 + R_k^2)) [1/nm] of the row pairs."""
        return 1.0 / jnp.sqrt(2.0 * (R[:, None] ** 2 + R[k] ** 2))

    def _kernels(
        self, x: Sequence[jax.Array], within: jax.Array, a: jax.Array, nmax: int = 3, series: bool = True
    ) -> tuple:
        """Return r and the kernels G_n = B_n[erf(a r)/r] - B_n[erf(b0 r)/r] (masked by `within`).

        (r, G0, ..., G_{nmax-1}) of the rows, in the dtype of x; series: md/kernels.py erf_kernels
        (accurate for small a r) or the closed form.
        """
        dt = x[0].dtype
        r = jnp.sqrt(jnp.where(within, x[0] * x[0] + x[1] * x[1] + x[2] * x[2], 1.0))
        kern = erf_kernels if series else erf_kernels_closed
        A = kern(a.astype(dt), r, nmax)
        B = kern(jnp.asarray(self.b0, dt), r, nmax)
        w = within.astype(dt)
        return (r,) + tuple((u - v) * w for u, v in zip(A, B))

    def geometry(self, pos: jax.Array, H: jax.Array, idx: jax.Array, P: dict, forces: bool = False) -> dict:
        """Return the row geometry: displacements and kernels G0..G2 of the electrostatic rows.

        Parameters
        ----------
        pos : jax.Array (N, 3)
            Positions [nm].
        H : jax.Array (3, 3)
            Box [nm].
        idx : jax.Array (N, C) int
            Candidate rows.
        P : dict
            Per-atom parameters (`_atoms` / `charges_at`).
        forces : bool
            Also G3, distances "r", van der Waals weights "wv", pair parameters "vp" and "within", and
            for split rows the van der Waals rows under "vdw_rows".

        Returns
        -------
        dict
            "k", "x", "overflow", "G0".. and (local preconditioner) "short".
        """
        cd = self.cd
        nmax = 4 if forces else 3
        k, x, within, wv, overflow, vrows = self._rows(pos, H, idx)
        r, *G = self._kernels(x, within, self._pair_a(P["radius"].astype(cd), k), nmax)
        g = {"k": k, "x": x, "overflow": overflow}
        for n in range(nmax):
            g[f"G{n}"] = G[n]
        if forces:
            g.update(r=r, wv=wv, vp=self._vdw_params(P, k), within=within)
            if vrows is not None:
                kv, xv, vin, wvv = vrows
                rv = jnp.sqrt(jnp.where(vin, xv[0] * xv[0] + xv[1] * xv[1] + xv[2] * xv[2], 1.0))
                g["vdw_rows"] = {"k": kv, "x": xv, "r": rv, "wv": wvv, "vp": self._vdw_params(P, kv)}
        if self.s.induction.local_niter > 0:
            short = within & (r < self.s.induction.local_cut)
            g["short"] = self._short_rows(k, x, g["G1"], g["G2"], short)
        return g

    def _short_rows(self, k: jax.Array, x: Sequence[jax.Array], G1: jax.Array, G2: jax.Array, short: jax.Array) -> dict:
        """Return the short-range entries of each row compacted into (N, ms) for the preconditioner."""
        N, ms = self.n, self.ms
        slot = jnp.cumsum(short, axis=1) - 1
        tgt = jnp.where(short & (slot < ms), slot, ms)
        rows = jnp.broadcast_to(jnp.arange(N)[:, None], k.shape)

        def pack(v: jax.Array, fill: float) -> jax.Array:  # scatter the kept entries to their slots
            return jnp.full((N, ms + 1), fill, v.dtype).at[rows, tgt].set(v)[:, :ms]

        return {"k": pack(k, 0), "x": tuple(pack(c, 0.0) for c in x), "G1": pack(G1, 0.0), "G2": pack(G2, 0.0)}

    @staticmethod
    def _row_field(g: dict, q: jax.Array | None, d: jax.Array) -> jax.Array:
        """Return sum_k de_ik/dd_i (N, 3): minus the direct-space field at each atom (q may be None)."""
        k, x, G1, G2 = g["k"], g["x"], g["G1"], g["G2"]
        dk = d[k]
        dk = (dk[..., 0], dk[..., 1], dk[..., 2])
        dkx = dk[0] * x[0] + dk[1] * x[1] + dk[2] * x[2]
        c = -G2 * dkx if q is None else -q[k] * G1 - G2 * dkx
        return jnp.stack([jnp.sum(c * x[j] + G1 * dk[j], axis=1) for j in range(3)], -1)

    @staticmethod
    def _row_potential(g: dict, q: jax.Array, d: jax.Array) -> jax.Array:
        """Return sum_k de_ik/dq_i = sum_k q_k G0 + (d_k . x_ik) G1 (N,), the direct-space potential.

        Used by charge flux.
        """
        k, x, G0, G1 = g["k"], g["x"], g["G0"], g["G1"]
        dk = d[k]
        dkx = dk[..., 0] * x[0] + dk[..., 1] * x[1] + dk[..., 2] * x[2]
        return jnp.sum(q[k] * G0 + dkx * G1, axis=1)

    def _rec_grad(self, S: dict, Gk: jax.Array, q: jax.Array, d: jax.Array) -> jax.Array:
        """Return dU_rec/dd (N, 3) in the compute dtype (PME.grad_dipoles)."""
        return self.pme.grad_dipoles(S, Gk, q, d.astype(self.cd)).astype(self.cd)

    def _field(self, g: dict, S: dict, Gk: jax.Array, q: jax.Array, d: jax.Array) -> jax.Array:
        """Return the total field -dU/dd (N, 3) (direct + PME + self) of charges q and dipoles d [e/nm^2]."""
        cd = self.cd
        dc = d.astype(cd)
        return -(self._row_field(g, q, dc) + self._rec_grad(S, Gk, q, dc) - jnp.asarray(self.c_self, cd) * dc)

    # ------------------------------------------------------------------ induction
    def init_induction(self) -> InductionState:
        """Return a fresh solver state (zero dipoles and history, normaliser 1)."""
        dt = jnp.float64
        z = jnp.zeros((self.n, 3), dt)
        xl = jnp.zeros((self.xl_len, self.n, 3), dt) if self.iel else None
        return InductionState(
            mu=z,
            hist=jnp.zeros((4, self.n, 3), dt),
            count=jnp.zeros((), jnp.int32),
            norm=jnp.ones((), dt),
            rec=jnp.zeros((4, self.S, self.n, 3), dt),
            pred=jnp.zeros((2, self.n, 3), dt),
            lscount=jnp.zeros(4, jnp.int32),
            xl=xl,
        )

    # ------------------------------------------------------------------ extended Lagrangian (iEL)
    @property
    def iel(self) -> bool:
        """Extended-Lagrangian induced dipoles (settings.induction.iel.scheme != "none", with induction)."""
        return self.s.induction.iel.scheme != "none" and self.ind

    @property
    def shadow(self) -> bool:
        """Whether the energy and forces are those of the iEL/0-SCF shadow potential U~(R, x).

        Not the converged U*(R); the barostat then evaluates U* at both volumes.
        """
        return self.iel and self.s.induction.iel.scheme == "0scf"

    @property
    def xl_coefficients(self) -> tuple[float, float, tuple[float, ...]]:
        """(kappa, a, c_0..c_K) of the auxiliary-dipole recurrence."""
        kap, a, c = _XL[self.s.induction.iel.order]
        kap = kap if self.s.induction.iel.kappa is None else float(self.s.induction.iel.kappa)
        a = a if self.s.induction.iel.alpha is None else float(self.s.induction.iel.alpha)
        return kap, a, tuple(float(v) for v in c)

    @property
    def xl_len(self) -> int:
        """Auxiliary dipoles kept: x_{n+1} and x_n ... x_{n-K} after a step (at least 3)."""
        return max(len(_XL[self.s.induction.iel.order][2]) + 1, 3)

    @property
    def xl_warmup(self) -> int:
        """Steps solved to dipole_tol at the start (and after an accepted volume move).

        They fill the auxiliary history with converged dipoles.
        """
        return max(self.s.induction.iel.order, 2) + 1

    BLOCK_MAX = 8  # atoms per molecule in the block preconditioner (larger ones: Jacobi)

    def _block_layout(self, mol: np.ndarray, first: np.ndarray, nmol: int) -> dict | None:
        """Return the static layout of the block preconditioner (host), or None without small molecules.

        Molecules of 2..BLOCK_MAX atoms each get a block; per atom its block ("blk", nblk for none) and
        local index ("loc"), per block slot its atom ("slot", n for padding), with "pad", "nblk", "nb"
        (largest block size) and "inb" (atom in a block).
        """
        n = self.n
        sizes = np.bincount(mol, minlength=nmol)
        small = (sizes >= 2) & (sizes <= self.BLOCK_MAX)
        nblk = int(small.sum())
        if nblk == 0:
            return None
        blk_of_mol = np.full(nmol, nblk)
        blk_of_mol[small] = np.arange(nblk)
        nb = int(sizes[small].max())
        blk = blk_of_mol[mol]
        loc = np.arange(n) - np.asarray(first)[mol]
        loc = np.where(blk < nblk, loc, 0)
        slot = np.full((nblk + 1, nb), n)
        inb = blk < nblk
        slot[blk[inb], loc[inb]] = np.nonzero(inb)[0]
        return {
            "blk": jnp.asarray(blk, jnp.int32),
            "loc": jnp.asarray(loc, jnp.int32),
            "slot": jnp.asarray(slot, jnp.int32),
            "pad": jnp.asarray(slot == n),
            "nblk": nblk,
            "nb": nb,
            "inb": jnp.asarray(inb),
        }

    def _block_mask(self, k: jax.Array) -> jax.Array:
        """Return the (N, C) mask of the row entries (special columns) pairing two atoms of one block."""
        L = self._blocks
        ni = self.special.shape[1]
        kk = k[:, :ni]
        b = L["blk"]
        m = (b[kk] == b[:, None]) & (b[:, None] < L["nblk"]) & (kk != jnp.arange(self.n)[:, None])
        return jnp.concatenate([m, jnp.zeros((self.n, k.shape[1] - ni), bool)], axis=1)

    def _block_solve(self, g: dict, alpha: jax.Array, r: jax.Array) -> jax.Array:
        """Return delta = M^-1 r (N, 3), float64: the block-preconditioned iEL/0-SCF step.

        M = diag(1/alpha) + the same-molecule row tensors G1 I - G2 x x^T of the special columns, one
        dense (3 nb)^2 block per small molecule (float64, jnp.linalg.solve); alpha r elsewhere.
        """
        L = self._blocks
        ni = self.special.shape[1]
        nblk, nb = L["nblk"], L["nb"]
        k = g["k"][:, :ni]
        m = self._block_mask(g["k"])[:, :ni].astype(jnp.float64)
        G1 = g["G1"][:, :ni].astype(jnp.float64) * m
        G2 = g["G2"][:, :ni].astype(jnp.float64) * m
        x = jnp.stack([c[:, :ni].astype(jnp.float64) for c in g["x"]], -1)  # (N, ni, 3)
        T = G1[..., None, None] * jnp.eye(3) - G2[..., None, None] * x[..., :, None] * x[..., None, :]
        b, l = L["blk"], L["loc"]
        M = jnp.zeros((nblk + 1, nb, 3, nb, 3))
        M = M.at[jnp.broadcast_to(b[:, None], k.shape), jnp.broadcast_to(l[:, None], k.shape), :, l[k], :].add(T)
        inv_a = _div_alpha(1.0, alpha, self.alpha_mask)
        M = M.at[b, l, :, l, :].add(jnp.where(L["inb"], inv_a, 0.0)[:, None, None] * jnp.eye(3))
        bi = jnp.broadcast_to(jnp.arange(nblk + 1)[:, None], L["slot"].shape)
        li = jnp.broadcast_to(jnp.arange(nb)[None, :], L["slot"].shape)
        M = M.at[bi, li, :, li, :].add(L["pad"][..., None, None] * jnp.eye(3))
        M = M.reshape(nblk + 1, 3 * nb, 3 * nb)
        rp = jnp.concatenate([r.astype(jnp.float64), jnp.zeros((1, 3))])[L["slot"]].reshape(nblk + 1, 3 * nb)
        d = jnp.linalg.solve(M, rp[..., None])[..., 0].reshape(nblk + 1, nb, 3)[b, l]
        return jnp.where(L["inb"][:, None], d, alpha[:, None] * r.astype(jnp.float64))

    def _solve_iel(
        self,
        g: dict,
        S: dict,
        Gk: jax.Array,
        alpha: jax.Array,
        q: jax.Array,
        p: jax.Array,
        ind: InductionState,
        ext: tuple | None = None,
    ) -> tuple:
        """Solve the extended-Lagrangian dipoles of one step and propagate the auxiliary dipoles.

        x = ind.xl[0] are the auxiliary dipoles of this step (propagated at the last one).  "0scf":
        mu = x + omega delta with delta = alpha r(x) (or M^-1 r(x)), r(x) = field(q, p + x) - x / alpha,
        one field sweep; "scf": iel_iter CG iterations from x (to dipole_tol if 0).  The first
        xl_warmup steps solve to dipole_tol and put the solution in place of x.  Then
        x_{n+1} = 2 x_n - x_{n-1} + kappa (mu - x_n) + a sum_k c_k x_{n-k}.

        Parameters
        ----------
        g : dict
            Row geometry (electrostatic rows).
        S, Gk : dict, jax.Array
            PME setup and influence function.
        alpha : jax.Array (N,)
            Polarizabilities [nm^3].
        q : jax.Array (N,)
            Charges [e].
        p : jax.Array (N, 3)
            Permanent dipoles [e nm].
        ind : InductionState
            Solver state with the auxiliary dipoles.
        ext : tuple, optional
            A uniform external field as in _solve_core (the field on the right-hand side, r(x) with the
            field at x; constant displacement also puts its kappa term into the operator).

        Returns
        -------
        mu : jax.Array (N, 3)
            Dipoles [e nm].
        delta : jax.Array (N, 3)
            The shadow displacement mu - x ("0scf"; 0 in the warm-up) [e nm].
        iterations : jax.Array () int32
            CG iterations (0 for "0scf" after the warm-up).
        residual : jax.Array ()
            Relative residual (at x for "0scf": max|delta| / mean|alpha b|).
        ind : InductionState
            New state.
        """
        cd = self.cd
        a64 = alpha[:, None]
        qc = q.astype(cd)
        A = self._operator(g, S, Gk, alpha, ext)
        add_ext = (lambda b, x: b) if ext is None else (lambda b, x: b + self._ext_at(ext, x)[None, :])
        X = ind.xl
        x = X[0]
        first = ind.count == 0
        warm = ind.count < self.xl_warmup

        def converged(_: None) -> tuple:
            """Warm-up: solve to dipole_tol by CG (from ind.mu, alpha b, or x)."""
            b = add_ext(self._field(g, S, Gk, qc, p), jnp.zeros((1, 3), cd))
            ab = a64 * b.astype(jnp.float64)
            # the first step starts from ind.mu when set (a volume move's converged dipoles), else alpha b
            x0 = jnp.where(first, jnp.where(jnp.any(ind.mu != 0.0), ind.mu, ab), x)
            norm = jnp.mean(jnp.abs(ab)) + 1e-300
            mu, it, err = self._cg(g, A, alpha, x0, b - A(x0.astype(cd)), norm)
            return mu, jnp.zeros_like(mu), it, err, norm

        def extended(_: None) -> tuple:
            """Take the extended step: the 0-SCF update from x, or iel_iter CG iterations from x."""
            r0 = add_ext(self._field(g, S, Gk, qc, p + x), x) - _div_alpha(x, a64, self.alpha_mask).astype(cd)
            norm = ind.norm
            if self.s.induction.iel.scheme == "0scf":
                d = a64 * r0.astype(jnp.float64) if self._blocks is None else self._block_solve(g, alpha, r0)
                d = float(self.s.induction.iel.omega) * d
                return x + d, d, jnp.zeros((), jnp.int32), jnp.max(jnp.abs(d)) / norm, norm
            k = int(self.s.induction.iel.iterations)
            mu, it, err = self._cg(
                g, A, alpha, x, r0, norm, tol=(0.0 if k > 0 else None), max_iter=(k if k > 0 else None)
            )
            return mu, mu - x, it, err, norm

        mu, d, it, err, norm = jax.lax.cond(warm, converged, extended, None)
        kap, a, c = self.xl_coefficients
        Xh = jnp.where(first, jnp.broadcast_to(mu, X.shape), jnp.where(warm, X.at[0].set(mu), X))
        dx = jnp.where(warm, 0.0, d)
        x_new = 2.0 * Xh[0] - Xh[1] + kap * dx
        if a != 0.0:
            x_new = x_new + a * sum(ck * Xh[k] for k, ck in enumerate(c) if ck != 0.0)
        ind = ind.set(mu=mu, xl=_push(Xh, x_new), hist=_push(ind.hist, mu), count=ind.count + 1, norm=norm)
        return mu, dx, it, err, ind

    def _extrapolate_ls(self, st: InductionState, new: jax.Array) -> tuple[jax.Array, InductionState]:
        """Return pmemd-pgm CPU's multi-order least-squares extrapolated guess and the updated state.

        dipole_scf_init = 3: coefficients c from the last S records of alpha b (ridge-regularised
        normal equations) applied to the records of mu and of the order-1, order-2 corrections, up to
        extrap_order once enough records exist.  new: alpha b of this step (N, 3).
        """
        S, order = self.S, self.s.induction.extrap_order
        rec1 = st.rec[0].astype(jnp.float64)
        M = jnp.einsum("snd,tnd->st", rec1, rec1)
        bv = jnp.einsum("snd,nd->s", rec1, new.astype(jnp.float64))
        ridge = 1e-12 * jnp.trace(M) + 1e-300
        c = jnp.linalg.solve(M + ridge * jnp.eye(S), bv)

        def lin(R: jax.Array) -> jax.Array:  # sum_s c_s R_s
            return jnp.einsum("s,snd->nd", c, R.astype(jnp.float64))

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

    def _record_ls(self, st: InductionState, mu: jax.Array) -> InductionState:
        """Record the converged mu and its deviations from the order-1 / order-2 predictions ("ls")."""
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

    def _operator(
        self, g: dict, S: dict, Gk: jax.Array, alpha: jax.Array, ext: tuple | None = None
    ) -> Callable[[jax.Array], jax.Array]:
        """Return the induction operator A: v -> alpha^-1 v - T v (compute dtype).

        With a constant displacement (ext[2] = kappa = 4 pi / V) also + kappa sum v.  T v is minus
        the field of the dipoles v (rows, PME, self).
        """
        cd = self.cd
        inv_a = _div_alpha(1.0, alpha, self.alpha_mask).astype(cd)[:, None]
        zq = jnp.zeros(self.n, cd)
        if ext is None or ext[2] is None:
            return lambda v: v * inv_a - self._field(g, S, Gk, zq, v)
        kappa = ext[2]
        return lambda v: (
            v * inv_a
            - self._field(g, S, Gk, zq, v)
            + (kappa * jnp.sum(v.astype(jnp.float64), axis=0)).astype(cd)[None, :]
        )

    def _cg(
        self,
        g: dict,
        A: Callable[[jax.Array], jax.Array],
        alpha: jax.Array,
        x: jax.Array,
        r: jax.Array,
        norm: jax.Array,
        tol: float | None = None,
        peek: float | None = None,
        max_iter: int | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Solve A mu = b by preconditioned CG from (x0, r0 = b - A x0).

        Jacobi preconditioner z = alpha r, or with local_niter > 0 a few inner CG iterations on the
        short-range tensor (flexible PCG, Polak-Ribiere beta).  Converged when max|alpha r| / norm <= tol
        (pmemd-pgm's criterion) or after max_iter iterations (a lax.while_loop); then the peek step
        mu += peek alpha r.

        Parameters
        ----------
        g : dict
            Row geometry (the local preconditioner's short rows).
        A : Callable
            The operator (`_operator`).
        alpha : jax.Array (N,)
            Polarizabilities [nm^3].
        x : jax.Array (N, 3)
            Initial guess [e nm].
        r : jax.Array (N, 3)
            Initial residual b - A x [e/nm^2].
        norm : jax.Array ()
            Normaliser mean|alpha b| [e nm].
        tol, peek, max_iter : optional
            Overrides of settings.induction (None: the settings).

        Returns
        -------
        mu : jax.Array (N, 3) float64
            Solution [e nm].
        iterations : jax.Array () int32
            CG iterations.
        residual : jax.Array ()
            Final max|alpha r| / norm (before the peek step).
        """
        cd, s = self.cd, self.s
        tol = s.induction.tol if tol is None else tol
        peek = s.induction.peek if peek is None else peek
        max_iter = s.induction.max_iter if max_iter is None else int(max_iter)
        inv_a = _div_alpha(1.0, alpha, self.alpha_mask).astype(cd)[:, None]
        a_c = alpha.astype(cd)[:, None]

        def precond(r: jax.Array) -> jax.Array:
            """Return z = alpha r, refined by local_niter inner CG iterations on the short-range tensor."""
            z = r * a_c
            if s.induction.local_niter <= 0:
                return z
            gs = g["short"]

            def A_loc(v: jax.Array) -> jax.Array:  # short-range operator alpha^-1 v - T_short v
                return v * inv_a + self._row_field(gs, None, v)

            rr = r - A_loc(z)
            zz = rr * a_c
            rz = _dot(rr, zz)

            def inner(_: int, c: tuple) -> tuple:
                """Run one inner CG iteration; carry (z, residual, direction, r.z)."""
                z, rr, p, rz = c
                Ap = A_loc(p)
                al = (rz / jnp.maximum(_dot(p, Ap), 1e-300)).astype(cd)
                z = z + al * p
                rr = rr - al * Ap
                zz = rr * a_c
                rz_new = _dot(rr, zz)
                p = zz + (rz_new / jnp.maximum(rz, 1e-300)).astype(cd) * p
                return z, rr, p, rz_new

            z, *_ = jax.lax.fori_loop(0, s.induction.local_niter, inner, (z, rr, zz, rz))
            return z

        def err_of(r: jax.Array) -> jax.Array:  # max|alpha r| / norm
            return jnp.max(jnp.abs(r * a_c).astype(jnp.float64)) / norm

        x, r = x.astype(cd), r.astype(cd)
        z = precond(r)

        def cond(c: tuple) -> jax.Array:  # carry (x, r, z, p, r.z, it, err): not converged, iterations left
            return (c[6] > tol) & (c[5] < max_iter)

        def body(c: tuple) -> tuple:
            """Run one preconditioned CG iteration."""
            x, r, z, p, rz, it, _ = c
            Ap = A(p)
            al = (rz / jnp.maximum(_dot(p, Ap), 1e-300)).astype(cd)
            x = x + al * p
            r_new = r - al * Ap
            z_new = precond(r_new)
            beta = (_dot(z_new, r_new - r) / jnp.maximum(rz, 1e-300)).astype(cd)  # flexible (Polak-Ribiere)
            p = z_new + beta * p
            return x, r_new, z_new, p, _dot(r_new, z_new), it + 1, err_of(r_new)

        x, r, _, _, _, it, err = jax.lax.while_loop(
            cond, body, (x, r, z, z, _dot(r, z), jnp.zeros((), jnp.int32), err_of(r))
        )
        if peek:
            x = x + jnp.asarray(peek, cd) * r * a_c
        return x.astype(jnp.float64), it, err

    def _ext_at(self, ext: tuple, x: jax.Array) -> jax.Array:
        """Return the external field (3,) (compute dtype) acting when the induced dipoles are x.

        F0 for a constant field, F0 - kappa (m0 + sum x) for constant displacement;
        ext = (F0 [e/nm^2], m0 [e nm], kappa [1/nm^3] or None).
        """
        F0, m0, kappa = ext
        if kappa is None:
            return F0.astype(self.cd)
        return (F0 - kappa * (m0 + jnp.sum(x.astype(jnp.float64), axis=0))).astype(self.cd)

    def _residual(
        self,
        g: dict,
        S: dict,
        Gk: jax.Array,
        alpha: jax.Array,
        q: jax.Array,
        p: jax.Array,
        mu: jax.Array,
        ext: tuple | None = None,
    ) -> jax.Array:
        """Return the residual b - A mu = field(q, p + mu) (+ external field) - mu / alpha (compute dtype).

        Zero at the induced dipoles.
        """
        cd = self.cd
        r = self._field(g, S, Gk, q.astype(cd), p + mu) - _div_alpha(mu, alpha[:, None], self.alpha_mask).astype(cd)
        return r if ext is None else r + self._ext_at(ext, mu)[None, :]

    def _solve(
        self,
        g: dict,
        S: dict,
        Gk: jax.Array,
        P: dict,
        p: jax.Array,
        ind: InductionState,
        fused_ok: bool = True,
        ext: tuple | None = None,
    ) -> tuple:
        """Return the induced dipoles, iterations, residual and updated InductionState.

        With settings.differentiable, mu carries exact derivatives (implicit function theorem).

        Notes
        -----
        A jax.custom_vjp around `_solve_core`: the forward pass saves (g, S, Gk, alpha, q, p, ext, mu);
        from A mu = b(theta), mu_bar . dmu = lam . d(b - A mu)|_mu with A lam = mu_bar.  A is symmetric,
        so the adjoint is one more CG with the same operator, to adjoint_tol (no peek); lam is pulled
        back through `_residual` at fixed mu by jax.vjp.  The predictor history gets no gradient
        (stop_gradient): it is data for the next step.
        """
        if not self.s.differentiable:
            return self._solve_core(g, S, Gk, P["alpha"], P["q"], p, ind, fused_ok, ext)
        cd = self.cd

        @jax.custom_vjp
        def run(
            g: dict, S: dict, Gk: jax.Array, alpha: jax.Array, q: jax.Array, p: jax.Array, ext: Any, ind: InductionState
        ) -> tuple:  # the solve with the custom backward pass
            return self._solve_core(g, S, Gk, alpha, q, p, ind, fused_ok, ext)

        def fwd(
            g: dict, S: dict, Gk: jax.Array, alpha: jax.Array, q: jax.Array, p: jax.Array, ext: Any, ind: InductionState
        ) -> tuple:  # forward: the solve, residuals for bwd
            out = self._solve_core(g, S, Gk, alpha, q, p, ind, fused_ok, ext)
            return out, (g, S, Gk, alpha, q, p, ext, out[0], ind)

        def bwd(res: tuple, cot: tuple) -> tuple:
            """Solve the adjoint A lam = mu_bar and pull lam back through the residual at mu."""
            g, S, Gk, alpha, q, p, ext, mu, ind = res
            mu_bar = cot[0]
            A = self._operator(g, S, Gk, alpha, ext)
            rhs = mu_bar.astype(cd)
            norm = jnp.mean(jnp.abs(alpha[:, None] * mu_bar)) + 1e-300
            lam, _, _ = self._cg(g, A, alpha, jnp.zeros_like(mu_bar), rhs, norm, tol=self.s.adjoint_tol, peek=0.0)
            _, vjp = jax.vjp(
                lambda g, S, Gk, alpha, q, p, ext: self._residual(g, S, Gk, alpha, q, p, mu, ext),
                g,
                S,
                Gk,
                alpha,
                q,
                p,
                ext,
            )
            return (*vjp(lam.astype(cd)), jax.tree_util.tree_map(_zero_cotangent, ind))

        run.defvjp(fwd, bwd)
        mu, it, err, ind_new = run(g, S, Gk, P["alpha"], P["q"], p, ext, jax.lax.stop_gradient(ind))
        # derivatives flow through mu only; the predictor history is data for the next step
        return mu, it, err, jax.lax.stop_gradient(ind_new).set(mu=mu)

    def _solve_core(
        self,
        g: dict,
        S: dict,
        Gk: jax.Array,
        alpha: jax.Array,
        q: jax.Array,
        p: jax.Array,
        ind: InductionState,
        fused_ok: bool = True,
        ext: tuple | None = None,
    ) -> tuple:
        """Solve the induced dipoles: initial guess and residual (fused when possible), then CG.

        Parameters
        ----------
        g : dict
            Row geometry (electrostatic rows).
        S, Gk : dict, jax.Array
            PME setup and influence function.
        alpha : jax.Array (N,)
            Polarizabilities [nm^3].
        q : jax.Array (N,)
            Charges [e].
        p : jax.Array (N, 3)
            Permanent dipoles [e nm].
        ind : InductionState
            Solver state (history).
        fused_ok : bool
            Allow the fused initial residual (static).
        ext : tuple, optional
            A uniform external field (F0, m0, kappa) in internal units (_ext_at): F0 added to the
            permanent field; for constant displacement also -kappa (m0 + sum mu), with the kappa term in
            the operator.

        Returns
        -------
        mu : jax.Array (N, 3) float64
            Induced dipoles [e nm].
        iterations : jax.Array () int32
            CG iterations.
        residual : jax.Array ()
            Final relative residual.
        ind : InductionState
            State with mu pushed onto the history.
        """
        cd = self.cd
        a64 = alpha[:, None]
        qc = q.astype(cd)
        A = self._operator(g, S, Gk, alpha, ext)
        pred = self.s.induction.predictor
        zero = jnp.zeros((1, 3), cd)
        add_ext = (lambda b, x: b) if ext is None else (lambda b, x: b + self._ext_at(ext, x)[None, :])

        def plain(x0_hist: jax.Array, have: jax.Array) -> tuple:
            """Unfused start: the permanent field b, x0 (history guess if `have`, else alpha b), r0, norm."""
            b = add_ext(self._field(g, S, Gk, qc, p), zero)
            ab = a64 * b.astype(jnp.float64)
            x0 = jnp.where(have, x0_hist, ab)
            r0 = b - A(x0.astype(cd))
            return x0, r0, jnp.mean(jnp.abs(ab)) + 1e-300

        if pred in _PRED:
            c = _PRED[pred]
            K = len(c)
            have = ind.count >= K
            x0h = sum(ci * ind.hist[j] for j, ci in enumerate(c))
            if self.s.induction.fused and fused_ok:
                use_fused = have & (ind.count % self.s.induction.norm_refresh != 0)

                def fused(_: None) -> tuple:  # r0 from one field sweep at d = p + x0 (pmemd-pgm PGM_FUSED)
                    r0 = add_ext(self._field(g, S, Gk, qc, p + x0h), x0h) - _div_alpha(
                        x0h, a64, self.alpha_mask
                    ).astype(cd)
                    return x0h, r0, ind.norm

                x0, r0, norm = jax.lax.cond(use_fused, fused, lambda _: plain(x0h, have), None)
            else:
                x0, r0, norm = plain(x0h, have)
        elif pred == "ls":
            b = add_ext(self._field(g, S, Gk, qc, p), zero)
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
    def _nonpair(
        self, pos: jax.Array, H: jax.Array, d: jax.Array, mu: jax.Array, P: dict, delta: jax.Array | None = None
    ) -> jax.Array:
        """Return the PME + self + background + polarisation energies [kJ/mol] as a function of (pos, d).

        KE [U_rec(q, d) + U_self + U_bg + sum |mu|^2 / (2 alpha)].  With delta (iEL/0-SCF), minus the
        PME and self energies of the dipoles delta alone (and, for omega != 1, the constant
        -(1 - 1/omega) |delta|^2 / (2 alpha)).  Differentiated by autodiff for the forces, the dipole
        gradient and (charge flux) the charge gradient.
        """
        q = P["q"]
        S, G = self.pme.setup(pos, H), self.pme.influence(H)
        u_rec = self.pme.energy(S, G, q, d)
        u_self = -(self.b0 / _SQRT_PI) * jnp.sum(q * q) - 0.5 * self.c_self * jnp.sum(d * d)
        if delta is not None:
            u_rec = u_rec - self.pme.energy(S, G, jnp.zeros_like(q), delta)
            u_self = u_self + 0.5 * self.c_self * jnp.sum(delta * delta)
            c = 1.0 - 1.0 / float(self.s.induction.iel.omega)  # damped Jacobi step: - c |delta|^2 / (2 alpha)
            if c != 0.0:
                u_self = u_self - c * jnp.sum(_div_alpha(delta * delta, 2.0 * P["alpha"][:, None], self.alpha_mask))
        u_bg = -jnp.pi * jnp.sum(q) ** 2 / (2.0 * volume(H) * self.b0**2)
        u_pol = jnp.sum(_div_alpha(mu * mu, 2.0 * P["alpha"][:, None], self.alpha_mask)) if self.ind else 0.0
        return KE * (u_rec + u_self + u_bg + u_pol)

    @staticmethod
    def _ext(efield: tuple | None, H: jax.Array) -> tuple | None:
        """Return the external-field tuple (F0, offset, kappa) of an efield argument, or None.

        F0 the field in internal units (3,) [e/nm^2], the dipole offset (3,) [e nm], kappa = 4 pi / V
        [1/nm^3] for constant displacement or None; efield = (E in V/nm, offset) [constant field] or
        (D/eps0 in V/nm, offset, "D") [constant displacement].

        Raises
        ------
        ValueError
            An unknown field kind.
        """
        if efield is None:
            return None
        from .efield import internal

        E, off = efield[0], efield[1]
        disp = len(efield) > 2 and efield[2] == "D"
        if len(efield) > 2 and efield[2] not in ("E", "D"):
            raise ValueError(f"field kind {efield[2]!r}: 'E' (constant field) or 'D' (constant displacement)")
        return (
            internal(E),
            jnp.zeros(3) if off is None else jnp.asarray(off, jnp.float64),
            4.0 * jnp.pi / volume(H) if disp else None,
        )

    @staticmethod
    def _ext_terms(ext: tuple, M: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the external field at the total dipole M (internal units) and its energy [kJ/mol].

        Constant field F0: -KE F0 . M; constant displacement F = F0 - kappa M: KE |F|^2 / (2 kappa)
        (= V eps0 |E|^2 / 2).
        """
        F0, _, kappa = ext
        if kappa is None:
            return F0, -KE * jnp.dot(F0, M)
        F = F0 - kappa * M
        return F, KE * jnp.dot(F, F) / (2.0 * kappa)

    @staticmethod
    def field_dipole(pos: jax.Array, q: jax.Array, d: jax.Array, off: jax.Array | None = None) -> jax.Array:
        """Return M = sum q r + sum d (+ offset) (3,) [e nm], the dipole a uniform field acts on.

        Molecules must be whole.
        """
        M = jnp.sum(q[:, None] * pos + jnp.asarray(d, jnp.float64), axis=0)  # one reduction (GPU: one kernel)
        return M if off is None else M + off

    # ------------------------------------------------------------------ van der Waals rows
    def _vdw_params(self, P: dict, k: jax.Array) -> tuple:
        """Return the row pair parameters of the van der Waals form (tuple of (N, C) arrays).

        LJ and DE: (rmin_ij [nm], eps_ij [kJ/mol]); GVDW: (A_ij, C6_ij, b_ij, a_ij); none: ().
        """
        cd = self.cd
        if self.s.terms.vdw in ("lj", "de"):
            rh, se = P["lj_rmin_half"].astype(cd), P["lj_sqrt_eps"].astype(cd)
            return (rh[:, None] + rh[k], se[:, None] * se[k])
        if self.s.terms.vdw == "gvdw":
            sa, sc, b = (P[n].astype(cd) for n in ("gvdw_sqrt_a", "gvdw_sqrt_c6", "gvdw_b"))
            return (
                sa[:, None] * sa[k],
                sc[:, None] * sc[k],
                0.5 * (b[:, None] + b[k]),
                self._pair_a(P["radius"].astype(cd), k),
            )
        return ()

    def _vdw_rows(
        self, r: jax.Array, vp: tuple, wv: jax.Array, grad: bool = False
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Return the row pair energies (and (1/r) dU/dr) of the van der Waals form times the weights.

        The weights wv are 0 for excluded pairs and outside the cutoff.  LJ:
        e = eps (s^12 - 2 s^6), s = rmin / r; DE: de.de_pair.  r [nm]; energies [kJ/mol],
        (1/r) dU/dr [kJ/mol/nm^2].
        """
        if self.s.terms.vdw == "lj":
            rminp, epsp = vp
            s6 = (rminp / r) ** 6
            e = epsp * (s6 * s6 - 2.0 * s6)
            d = epsp * 12.0 * (s6 - s6 * s6) / (r * r)
        elif self.s.terms.vdw == "de":
            rminp, epsp = vp
            e, d = de_pair_grad(r, rminp, epsp, self.s.terms.de_alpha, self.s.terms.de_beta)
        elif self.s.terms.vdw == "gvdw":
            A, C6, B, a = vp
            e, d = gvdw_pair(r, a, A, C6, B, self.s.terms.gvdw_rep, grad=True)
        else:
            e = d = jnp.zeros_like(r)
        on = wv != 0
        e, d = jnp.where(on, e * wv, 0.0), jnp.where(on, d * wv, 0.0)
        return (e, d) if grad else e

    def _vdw_tail(self, P: dict, H: jax.Array) -> jax.Array | float:
        """Return the long-range correction of the van der Waals term [kJ/mol] (0 without lj_lrc)."""
        if not self.s.terms.lj_lrc or self.s.terms.vdw == "none":
            return 0.0
        if self.s.terms.vdw == "de":
            t = self.s.terms
            return de_long_range(P, volume(H), self.s.cutoffs.cutoff, self._de_groups(), t.de_alpha, t.de_beta)
        f = lj_long_range if self.s.terms.vdw == "lj" else gvdw_long_range
        return f(P, volume(H), self.s.cutoffs.cutoff)

    def _de_groups(self) -> tuple:
        """Return the distinct (R*, sqrt(eps)) parameter sets with their populations (de_groups; built in __init__)."""
        return self._de_groups_cache

    def _vdw_tail_impulse(self, P: dict, H: jax.Array) -> jax.Array | float:
        """Return the scalar X of the cutoff impulse -X I of the van der Waals tail in the strain derivative [kJ/mol].

        X = E_lrc for the power-law tails of LJ and GVDW (`_vdw_tail`); for DE, X = 2 pi rc^3 / (3 V) sum_ij u_ij(rc)
        (de.de_tail_impulse).  Zero without lj_lrc.
        """
        if self.s.terms.vdw == "de" and self.s.terms.lj_lrc:
            t = self.s.terms
            return de_tail_impulse(P, volume(H), self.s.cutoffs.cutoff, self._de_groups(), t.de_alpha, t.de_beta)
        return self._vdw_tail(P, H)

    def _pair_sum(
        self,
        x: Sequence[jax.Array],
        di: Sequence[jax.Array],
        dk: Sequence[jax.Array],
        qi: jax.Array,
        qk: jax.Array,
        a: jax.Array,
        within: jax.Array,
        wv: jax.Array,
        vp: tuple,
    ) -> tuple:
        """Return the row sum of KE e_elec + e_vdW (each pair twice), float64, and its two parts.

        The autodiff path (energy_fixed_mu); x, di, dk: component triples of (N, C) arrays.
        """
        r, G0, G1, G2 = self._kernels(x, within, a)
        dix = di[0] * x[0] + di[1] * x[1] + di[2] * x[2]
        dkx = dk[0] * x[0] + dk[1] * x[1] + dk[2] * x[2]
        didk = di[0] * dk[0] + di[1] * dk[1] + di[2] * dk[2]
        e = qi * qk * G0 + (qi * dkx - qk * dix) * G1 - G2 * dix * dkx + G1 * didk
        elj = self._vdw_rows(r, vp, wv)
        se = jnp.sum(jnp.sum(e, axis=1).astype(jnp.float64))
        sl = jnp.sum(jnp.sum(elj, axis=1).astype(jnp.float64))
        return KE * se + sl, (KE * se, sl)

    def _vdw_sum(self, x: Sequence[jax.Array], within: jax.Array, wv: jax.Array, vp: tuple) -> jax.Array:
        """Return the row sum of e_vdW (each pair twice), float64 (autodiff path, van der Waals rows)."""
        r = jnp.sqrt(jnp.where(within, x[0] * x[0] + x[1] * x[1] + x[2] * x[2], 1.0))
        return jnp.sum(jnp.sum(self._vdw_rows(r, vp, wv), axis=1).astype(jnp.float64))

    def _row_inputs(self, pos: jax.Array, H: jax.Array, idx: jax.Array, P: dict, d: jax.Array) -> tuple:
        """Return the rows for the differentiable (autodiff) energy: x, di, constants, van der Waals rows.

        The last item holds the van der Waals rows (x, within, weights, pair parameters) of split rows,
        else None.
        """
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

    def energy_fixed_mu(
        self, pos: jax.Array, H: jax.Array, mu: jax.Array, idx: jax.Array, P: dict, efield: tuple | None = None
    ) -> tuple[jax.Array, dict]:
        """Return the total energy and its components with the induced dipoles held at mu.

        Differentiable in positions, box and parameters (used for virials, Monte Carlo trials and
        parameter gradients).  With charge flux, q and c are taken at pos.

        Parameters
        ----------
        pos : jax.Array (N, 3)
            Positions [nm].
        H : jax.Array (3, 3)
            Box [nm].
        mu : jax.Array (N, 3)
            Induced dipoles [e nm].
        idx : jax.Array (N, C) int
            Candidate rows.
        P : dict
            Per-atom parameters (`_atoms`).
        efield : tuple, optional
            (E in V/nm, dipole offset) or with "D": adds the field energy ("field").

        Returns
        -------
        energy : jax.Array () float64
            Total energy [kJ/mol].
        parts : dict
            "elec", "vdw" (and "field") [kJ/mol].
        """
        P = self.charges_at(pos, H, P)
        p = self.perm_dipoles(pos, H, P["cov"])
        d = p + mu
        x, di, (dk, qi, qk, a, within, wv, vp), vrows = self._row_inputs(pos, H, idx, P, d)
        spair, (se, sl) = self._pair_sum(x, di, dk, qi, qk, a, within, wv, vp)
        if vrows is not None:
            sl = sl + self._vdw_sum(*vrows)
        e_elec = 0.5 * se + self._nonpair(pos, H, d, mu, P)
        e_lj = 0.5 * sl + self._vdw_tail(P, H)
        if efield is None:
            return e_elec + e_lj, {"elec": e_elec, "vdw": e_lj}
        ext = self._ext(efield, H)
        _, e_f = self._ext_terms(ext, self.field_dipole(pos, P["q"], d, ext[1]))
        return e_elec + e_lj + e_f, {"elec": e_elec, "vdw": e_lj, "field": e_f}

    def _row_terms(self, g: dict, q: jax.Array, d: jax.Array, delta: jax.Array | None = None) -> tuple:
        """Return the pair energies and the row sums of de_ik/dx_ik from the kernels G0..G3.

        Each pair is counted in both rows (grad_x G_n = -G_{n+1} x); charges q and total dipoles d in
        the compute dtype.  Electrostatics over the electrostatic rows, van der Waals over those and the
        van der Waals rows of split rows.  With delta (iEL/0-SCF shadow energy), minus the pair energies
        of the dipoles delta alone and their gradient, in the same pass over the rows (inside a block
        of the block preconditioner scaled by 1 - 1/omega).

        Returns
        -------
        e_elec : jax.Array ()
            Row sum of the electrostatic pair energies (units of KE; float64).
        e_vdw : jax.Array ()
            Row sum of the van der Waals energies [kJ/mol].
        g_elec : jax.Array (N, 3)
            sum_k de_ik/dx_ik of the electrostatics (units of KE).
        g_vdw : jax.Array (N, 3)
            sum_k de_ik/dx_ik of the van der Waals term [kJ/mol/nm].
        """
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
        dd = dix * dkx
        if delta is None:  # as before iEL (same rounding: SCF runs unchanged)
            e = qi * qk * G0 + t * G1 - G2 * dix * dkx + G1 * didk
            radial = -qi * qk * G1 - t * G2 + G3 * dix * dkx - G2 * didk
            cross = [di[j] * dkx + dk[j] * dix for j in range(3)]
        else:  # minus the delta-delta pair terms
            Dkk = delta[k]
            Dk = (Dkk[..., 0], Dkk[..., 1], Dkk[..., 2])
            Di = tuple(delta[:, j][:, None] for j in range(3))
            if self._blocks is not None:  # block preconditioner: M / omega holds these pairs
                w_blk = 1.0 - 1.0 / float(self.s.induction.iel.omega)
                keep = jnp.where(self._block_mask(k), w_blk, 1.0).astype(x[0].dtype)
                Di = tuple(Di[j] * keep for j in range(3))
            Dix = Di[0] * x[0] + Di[1] * x[1] + Di[2] * x[2]
            Dkx = Dk[0] * x[0] + Dk[1] * x[1] + Dk[2] * x[2]
            dd = dd - Dix * Dkx
            didk = didk - (Di[0] * Dk[0] + Di[1] * Dk[1] + Di[2] * Dk[2])
            e = qi * qk * G0 + t * G1 - G2 * dd + G1 * didk
            radial = -qi * qk * G1 - t * G2 + G3 * dd - G2 * didk
            cross = [di[j] * dkx + dk[j] * dix - Di[j] * Dkx - Dk[j] * Dix for j in range(3)]
        elj, glj = self._vdw_rows(g["r"], g["vp"], g["wv"], grad=True)
        qiG1, qkG1 = qi * G1, qk * G1
        gx = jnp.stack(
            [jnp.sum(radial * x[j] + qiG1 * dk[j] - qkG1 * di[j] - G2 * cross[j], axis=1) for j in range(3)], -1
        )
        glx = jnp.stack([jnp.sum(glj * x[j], axis=1) for j in range(3)], -1).astype(jnp.float64)

        def rowsum(v: jax.Array) -> jax.Array:
            return jnp.sum(jnp.sum(v, axis=1).astype(jnp.float64))  # rows in compute dtype, total in float64

        sl = rowsum(elj)
        if "vdw_rows" in g:  # split rows: van der Waals beyond elec_cutoff
            t = g["vdw_rows"]
            elt, glt = self._vdw_rows(t["r"], t["vp"], t["wv"], grad=True)
            glx = glx + jnp.stack([jnp.sum(glt * t["x"][j], axis=1) for j in range(3)], -1).astype(jnp.float64)
            sl = sl + rowsum(elt)
        return rowsum(e), sl, gx.astype(jnp.float64), glx

    def _energy_forces(
        self,
        pos: jax.Array,
        H: jax.Array,
        mu: jax.Array,
        g: dict,
        P: dict,
        flux_pull: Callable | None = None,
        ext: tuple | None = None,
        M: jax.Array | None = None,
        delta: jax.Array | None = None,
    ) -> tuple[dict, jax.Array]:
        """Return the energy parts and forces at fixed mu.

        Analytic row forces for the pair terms (no scatter-adds), autodiff for PME, one vector-Jacobian
        product through the covalent-dipole frames.  With charge flux (flux_pull: the pull-back of the
        flux map at pos; P holds q(R), c(R)): _energy_forces_flux.

        Parameters
        ----------
        pos : jax.Array (N, 3)
            Positions [nm].
        H : jax.Array (3, 3)
            Box [nm].
        mu : jax.Array (N, 3)
            Induced dipoles [e nm].
        g : dict
            Row geometry (geometry(forces=True)).
        P : dict
            Per-atom parameters.
        flux_pull : Callable, optional
            VJP of the flux map R -> (q, c).
        ext : tuple, optional
            `_ext` output: the field F at M = sum q r + sum d + offset adds its energy (_ext_terms),
            KE q F to the forces and -KE F to dE/dd (torques).
        M : jax.Array (3,), optional
            Cell dipole [e nm] (None: computed here).
        delta : jax.Array (N, 3), optional
            iEL/0-SCF shadow displacement.

        Returns
        -------
        energy : dict
            "elec", "vdw", "total" (and "field") [kJ/mol].
        forces : jax.Array (N, 3) float64
            Forces [kJ/mol/nm].
        """
        if flux_pull is not None:
            return self._energy_forces_flux(pos, H, mu, g, P, flux_pull, ext, M, delta)
        cd = self.cd
        p, vjp_p = jax.vjp(lambda y: self.perm_dipoles(y, H, P["cov"]), pos)
        d = p + mu
        qc, dc = P["q"].astype(cd), d.astype(cd)
        se, sl, gx_el, gx_lj = self._row_terms(g, qc, dc, None if delta is None else delta.astype(cd))
        dEdd = KE * self._row_field(g, qc, dc).astype(jnp.float64)
        if ext is not None:
            Fx, e_f = self._ext_terms(ext, self.field_dipole(pos, P["q"], d, ext[1]) if M is None else M)
            if delta is not None and ext[2] is not None:  # iEL/0-SCF at constant D: minus kappa |sum delta|^2 / 2
                sd = jnp.sum(delta.astype(jnp.float64), axis=0)
                e_f = e_f - 0.5 * KE * ext[2] * jnp.dot(sd, sd)
            dEdd = dEdd - KE * Fx[None, :]
        e_np, (gpos_np, gd_np) = jax.value_and_grad(self._nonpair, argnums=(0, 2))(pos, H, d, mu, P, delta)
        forces = -(KE * gx_el + gx_lj + gpos_np + vjp_p(dEdd + gd_np)[0])
        e_elec = 0.5 * KE * se + e_np
        e_lj = 0.5 * sl + self._vdw_tail(P, H)
        if ext is None:
            return {"elec": e_elec, "vdw": e_lj, "total": e_elec + e_lj}, forces
        forces = forces + KE * P["q"][:, None] * Fx[None, :]
        return {"elec": e_elec, "vdw": e_lj, "field": e_f, "total": e_elec + e_lj + e_f}, forces

    def _energy_forces_flux(
        self,
        pos: jax.Array,
        H: jax.Array,
        mu: jax.Array,
        g: dict,
        P: dict,
        flux_pull: Callable,
        ext: tuple | None = None,
        M: jax.Array | None = None,
        delta: jax.Array | None = None,
    ) -> tuple[dict, jax.Array]:
        """Return the energy parts and forces as `_energy_forces`, with charge flux (P holds q(R), c(R)).

        F = -dE/dR|_{q,c,mu} - phi . dq/dR - (dE/dc) . dc/dR: the potential phi = dE/dq (rows, and PME,
        self and background terms from the autodiff of _nonpair with q among the arguments) and dE/dc
        (the covalent-frame pull-back taken with respect to c as well) go through flux_pull, the
        jax.vjp of the bond-local map R -> (q, c).
        """
        cd = self.cd
        p, vjp_p = jax.vjp(lambda y, c: self.perm_dipoles(y, H, c), pos, P["cov"])
        d = p + mu
        qc, dc = P["q"].astype(cd), d.astype(cd)
        se, sl, gx_el, gx_lj = self._row_terms(g, qc, dc, None if delta is None else delta.astype(cd))
        dEdd = KE * self._row_field(g, qc, dc).astype(jnp.float64)
        phi = KE * self._row_potential(g, qc, dc).astype(jnp.float64)
        if ext is not None:  # the external potential -F . r at each atom
            Fx, e_f = self._ext_terms(ext, self.field_dipole(pos, P["q"], d, ext[1]) if M is None else M)
            if delta is not None and ext[2] is not None:  # iEL/0-SCF at constant D: minus kappa |sum delta|^2 / 2
                sd = jnp.sum(delta.astype(jnp.float64), axis=0)
                e_f = e_f - 0.5 * KE * ext[2] * jnp.dot(sd, sd)
            dEdd = dEdd - KE * Fx[None, :]
            phi = phi - KE * (pos @ Fx)
        e_np, (gpos_np, gd_np, gq_np) = jax.value_and_grad(
            lambda y, dd, q: self._nonpair(y, H, dd, mu, dict(P, q=q), delta), argnums=(0, 1, 2)
        )(pos, d, P["q"])
        gpos_p, gcov = vjp_p(dEdd + gd_np)
        forces = -(KE * gx_el + gx_lj + gpos_np + gpos_p + flux_pull((phi + gq_np, gcov))[0])
        e_elec = 0.5 * KE * se + e_np
        e_lj = 0.5 * sl + self._vdw_tail(P, H)
        if ext is None:
            return {"elec": e_elec, "vdw": e_lj, "total": e_elec + e_lj}, forces
        forces = forces + KE * P["q"][:, None] * Fx[None, :]
        return {"elec": e_elec, "vdw": e_lj, "field": e_f, "total": e_elec + e_lj + e_f}, forces

    # ------------------------------------------------------------------ public
    def rows_for(self, pos: ArrayLike, H: ArrayLike) -> jax.Array:
        """Return candidate rows (every atom within the pair cutoff) for a fixed frame, built on the host.

        For single points and parameter fitting outside MD (which keeps its own neighbour list): an
        AtomNeighbors list without skin.  pos (N, 3) [nm], H (3, 3) [nm]; result (N, C) int (padding N).
        """
        from .neighbors import AtomNeighbors

        H = jnp.asarray(H, jnp.float64)
        return AtomNeighbors(self.n, H, self.rc_pair, 0.0).allocate(jnp.asarray(pos, jnp.float64), None, H).idx

    def compute(
        self,
        pos: ArrayLike,
        H: ArrayLike,
        idx: jax.Array,
        ind: InductionState,
        params: dict | None = None,
        keep_geometry: bool = False,
        efield: tuple | None = None,
    ) -> Result:
        """Solve the induced dipoles (predicted guess), then return energy and forces.

        Parameters
        ----------
        pos : ArrayLike (N, 3)
            Positions [nm] (molecules need not be whole).
        H : ArrayLike (3, 3)
            Box, reduced lower triangular [nm].
        idx : jax.Array (N, C) int
            Candidate rows from a neighbour list (padding N).
        ind : InductionState
            Solver state of the previous step (predictor history).
        params : dict, optional
            Parameter pytree (None: the system's initial values).
        keep_geometry : bool
            Also return the row geometry (Result.geometry: partner indices k, displacements x,
            distances r, `within` mask, van der Waals weights wv of the electrostatic rows), from which
            multiple time stepping builds its short-range pair list (md/mts.py).
        efield : tuple, optional
            (E (3,) V/nm, dipole offset (3,) e nm or None[, "D"]): a uniform external field
            (md/efield.py); energy["field"] and Result.dipole = M.

        Returns
        -------
        Result
            Energies [kJ/mol], forces [kJ/mol/nm], new solver state, CG statistics, overflow.

        Notes
        -----
        With settings.differentiable, energy, forces and Result.induction.mu can be differentiated
        (jax.grad / vjp) in params, pos and H (`_solve`).  With iEL/0-SCF and iel.shadow the energy and
        forces are those of the shadow potential.  Meant to be called inside the caller's jit.
        """
        pos, H = jnp.asarray(pos, jnp.float64), jnp.asarray(H, jnp.float64)
        ext = self._ext(efield, H)
        P = self._atoms(params)
        pull = None
        if self.flux is not None:  # q(R), c(R) and the flux map's pull-back
            (q, cov), pull = jax.vjp(lambda y: self.flux.charges(y, H, P["q"], P["cov"], P["flux"]), pos)
            P = {**{k: v for k, v in P.items() if k != "flux"}, "q": q, "cov": cov}
        g = self.geometry(pos, H, idx, P, forces=True)
        p = self.perm_dipoles(pos, H, P["cov"])
        S = self.pme.setup(pos, H)
        Gk = self.pme.influence(H)
        delta = None
        if self.ind:  # the solve sees the electrostatic rows only
            ge = {key: v for key, v in g.items() if key != "vdw_rows"}
            ext_s = None if ext is None else (ext[0], self.field_dipole(pos, P["q"], p, ext[1]), ext[2])
            if self.iel:  # extended-Lagrangian dipoles
                if ind.xl is None:  # a state from an SCF run: start the history
                    ind = ind.set(xl=jnp.zeros((self.xl_len, self.n, 3)), count=jnp.zeros_like(ind.count))
                mu, delta, it, err, ind = self._solve_iel(ge, S, Gk, P["alpha"], P["q"], p, ind, ext=ext_s)
            else:
                mu, it, err, ind = self._solve(ge, S, Gk, P, p, ind, ext=ext_s)
        else:  # no induced dipoles ("q", "qp")
            mu, it, err = jnp.zeros((self.n, 3)), jnp.zeros((), jnp.int32), jnp.zeros(())
        M = None if ext is None else self.field_dipole(pos, P["q"], p + mu, ext[1])
        # iEL/0-SCF: U~ = U(mu) - U_es(0, delta) and its exact forces (iel_shadow), in the same passes
        energy, forces = self._energy_forces(
            pos, H, mu, g, P, pull, ext, M, delta if (self.shadow and self.s.induction.iel.shadow) else None
        )
        return Result(energy, forces, ind, it, err, g["overflow"], g if keep_geometry else None, M)

    def energy(
        self,
        pos: ArrayLike,
        H: ArrayLike,
        idx: jax.Array,
        ind: InductionState,
        params: dict | None = None,
        efield: tuple | None = None,
    ) -> tuple:
        """Return the energy only (Monte Carlo barostat trials), the dipoles solved from the last ones.

        No predictor history update (iEL: the auxiliary dipoles restart at the new geometry).

        Parameters
        ----------
        pos : ArrayLike (N, 3)
            Positions [nm].
        H : ArrayLike (3, 3)
            Box [nm].
        idx : jax.Array (N, C) int
            Candidate rows.
        ind : InductionState
            Solver state (its mu is the initial guess).
        params : dict, optional
            Parameter pytree.
        efield : tuple, optional
            As in `compute`.

        Returns
        -------
        energy : jax.Array ()
            Total energy [kJ/mol].
        ind : InductionState
            State with the converged mu.
        iterations : jax.Array () int32
            CG iterations.
        overflow : jax.Array () bool
            Row capacity exceeded.
        """
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
            ext = None
            if efield is not None:
                e = self._ext(efield, H)
                ext = (e[0], self.field_dipole(pos, P["q"], p, e[1]), e[2])
                b = b + self._ext_at(ext, jnp.zeros((1, 3)))[None, :]
            A = self._operator(g, S, Gk, alpha, ext)
            x0 = ind.mu
            mu, it, err = self._cg(
                g, A, alpha, x0, b - A(x0.astype(cd)), jnp.mean(jnp.abs(alpha[:, None] * b)) + 1e-300
            )
        else:
            mu, it = jnp.zeros((self.n, 3)), jnp.zeros((), jnp.int32)
        e, _ = self.energy_fixed_mu(pos, H, mu, idx, P, efield)
        if self.iel:  # auxiliary dipoles restart at the new geometry
            return e, ind.set(mu=mu, count=jnp.zeros_like(ind.count)), it, g["overflow"]
        return e, ind.set(mu=mu), it, g["overflow"]

    def strain_derivative(
        self,
        pos: ArrayLike,
        H: ArrayLike,
        idx: jax.Array,
        mu: ArrayLike,
        params: dict | None = None,
        molecular: bool = True,
        efield: tuple | None = None,
    ) -> jax.Array:
        """Return dE/d eps (3, 3) [kJ/mol] at fixed mu, plus the tail impulse term (-E_lrc I) with lj_lrc.

        The full tensor: the components that would take the box out of lower-triangular form come from
        rotation invariance (full_strain_derivative, with mu, the field and its dipole offset as the
        vectors held fixed).

        Parameters
        ----------
        pos : ArrayLike (N, 3)
            Positions [nm] (molecules whole).
        H : ArrayLike (3, 3)
            Box [nm].
        idx : jax.Array (N, C) int
            Candidate rows.
        mu : ArrayLike (N, 3)
            Induced dipoles [e nm].
        params : dict, optional
            Parameter pytree.
        molecular : bool
            Molecules translated with their centres of mass (virtual sites move with them); otherwise
            every position is scaled affinely, which virtual sites do not follow (refused).
        efield : tuple, optional
            The external-field term is included (zero for neutral molecules under molecular scaling;
            md/efield.py).

        Raises
        ------
        NotImplementedError
            molecular=False with virtual sites.
        """
        if not molecular and self.has_vsites:
            raise NotImplementedError(
                "atomic (affine) strain derivative with virtual sites: the sites would have to be "
                "rebuilt from the deformed parents; use molecular=True"
            )
        pos, H = jnp.asarray(pos, jnp.float64), jnp.asarray(H, jnp.float64)
        P = self._atoms(params)
        com = None
        if molecular:
            w = self.masses
            com = centers_of_mass(pos, w, self.mol, self.sys.nmol)
        mu = jnp.asarray(mu, jnp.float64)
        if efield is None:
            W = full_strain_derivative(
                lambda x, h, m: self.energy_fixed_mu(x, h, m, idx, P)[0], pos, H, self.mol, com, (mu,)
            )
        else:
            kind, off = tuple(efield[2:]), efield[1]
            vecs = (mu, jnp.asarray(efield[0], jnp.float64)) + (() if off is None else (jnp.asarray(off, jnp.float64),))
            W = full_strain_derivative(
                lambda x, h, m, E, *o: self.energy_fixed_mu(x, h, m, idx, P, (E, o[0] if o else None) + kind)[0],
                pos,
                H,
                self.mol,
                com,
                vecs,
            )
        return W - self._vdw_tail_impulse(P, H) * jnp.eye(3)
