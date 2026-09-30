"""Alchemical free energies with pGM: lambda-dependent Hamiltonians of one solute molecule.

Contents: `alchemical_system` and `standard_schedule` (setup), `Alchemy` (the Hamiltonian at
lambda, bound to a force field), `GasPhaseLeg` and `lone_solute` (the gas-phase leg of the
solvation cycle), `LambdaWindows` (the lambda windows, batched on the GPU; a remd.MDReplicas)
and `FreeEnergyRun` (sampling for TI / BAR / MBAR with optional Hamiltonian replica exchange;
the estimators are in pgm_jax/fit/free_energy.py).

Thermodynamic cycle (hydration or solvation free energy of a solute A)

    A(gas)      --- Delta G_gas(1 -> 0) --->  A(gas, no electrostatics)
      |                                             | 0 (ideal gas either way)
    A(solution) --- Delta G_solv(1 -> 0) -->  A(solution, decoupled)

    Delta G_hyd = Delta G_gas(1 -> 0) - Delta G_solv(1 -> 0)

with the same number density of A in both phases (Ben-Naim's standard state, the one of tabulated
experimental hydration free energies).  In solution the solute is taken through two stages
(`standard_schedule`): its electrostatics is switched off with its van der Waals on (lambda_elec
1 -> 0 at lambda_vdw = 1), then its van der Waals with the environment (lambda_vdw 1 -> 0 at
lambda_elec = 0), so that charges never sit on atoms that other atoms can reach.

What lambda does to each term (Alchemy)

  lambda_elec (annihilation of the solute's electrostatics, as the ele-lambda of AMOEBA free
  energies in Tinker: permanent multipoles and polarizabilities of the mutated atoms scaled,
  intramolecular terms included [1]_, [2]_, [3]_):
      q_s     -> lambda_e q_s               (Gaussian charges)
      c_s     -> lambda_e c_s               (covalent-dipole strengths, i.e. permanent dipoles p_s)
      alpha_s -> alpha_s [eps + (1 - eps) lambda_e],  eps = alpha_floor (1e-8)
  Every electrostatic interaction of the solute is scaled: with the environment, with its own
  periodic images and inside the molecule (pGM couples every pair, and the solute's own induction).
  At lambda_e = 0 the solute carries no electrostatics; the gas-phase leg removes the same
  intramolecular energy in vacuum (GasPhaseLeg: exact from one configuration for a rigid solute;
  a flexible solute's is sampled with the same engine: lone_solute).
  intramolecular="keep" instead adds E_gas(x_s; lambda_e = 1) - E_gas(x_s; lambda_e), the solute's
  electrostatics in vacuum (dense pGM of the lone solute at its current geometry, float64) at full
  minus reduced coupling: the solute keeps its whole gas-phase intramolecular electrostatics at every
  lambda and only its coupling to the environment and its periodic images is switched off, so the
  decoupled state is the gas-phase molecule and Delta G_hyd = -Delta G_solv(1 -> 0), no separate
  gas leg.  Flexible pGM solutes need it: their bonded terms were fitted together with every
  intramolecular pair's electrostatics, and without them the molecule is held by its bonded terms
  alone (methanol at lambda_e = 0 in vacuum blew apart within 14 steps of 2 fs).  For a rigid solute
  the correction is a constant at each lambda and both modes give the same free energy; with it,
  dU/dlambda_e is the gas-subtracted integrand, which TI integrates with a smaller quadrature error.
  The polarizability floor: the induced dipoles minimise |mu|^2 / (2 alpha) - mu.E + mu T mu / 2;
  the Jacobi-preconditioned operator of the CG is I - alpha^1/2 T alpha^1/2, whose coupling only
  shrinks with alpha_s, so the solve stays well posed as alpha_s -> 0, but alpha_s = 0 would put
  1/0 into the operator and the polarization energy.  Tinker masks the solute's induced dipoles at
  lambda = 0 (douind = .false.); here alpha_s(0) = eps alpha_s instead: U(lambda) is smooth up to
  the endpoint, dU/dlambda_e at lambda_e = 0 is the exact limit (-(1 - eps) alpha_s |E|^2 / 2 per
  site plus the permanent terms, finite), and the endpoint differs from masked dipoles by eps
  times the solute's induction energy (~1e-7 kJ/mol).  The Gaussian radii are not scaled (they
  are damping widths, not interaction strengths).

  lambda_vdw (decoupling of the solute-environment van der Waals, Beutler soft core [4]_; alpha
  0.5 and power 1 as in [5]_):
      U_ij = lambda_v eps_ij [1 / w^2 - 2 / w],   w = (r / rmin_ij)^6 + (sc_alpha / 2)(1 - lambda_v)
  i.e. 4 eps lambda_v [1/y^2 - 1/y] with y = sc_alpha (1 - lambda_v) + (r / sigma)^6, sigma^6 =
  rmin^6 / 2, in Amber's rmin form: Lennard-Jones at lambda_v = 1, finite energy and force at r = 0
  for lambda_v < 1 (the force vanishes there: w depends on r^6).  Only solute-environment pairs
  are soft; the intramolecular van der Waals of a flexible solute (pairs >= lj_min_sep bonds
  apart, scaled 1-4 pairs) is kept at full strength at every lambda, as its bonded terms;
  the long-range correction of those pairs is scaled by lambda_v (beyond the cutoff the soft core
  differs from lambda_v x LJ by (sc_alpha / 2)(1 - lambda_v)(rmin / rc)^6 < 1e-3 of the term).
  Van der Waals forms: "lj" and "none"; GVDW (finite at overlap, would need no soft core) is
  refused for now.

Hamiltonian at lambda = (lambda_e, lambda_v): the ordinary force field (forcefield.py) with the
solute's parameters at lambda_e and its van der Waals parameters set to zero, so the ordinary pair
rows never see a solute pair's van der Waals, plus the soft-core solute-environment term evaluated
here in a dedicated small row set (each solute atom's candidates from the neighbour list, n_solute
x C pairs, forces by autodiff) and the solute's intramolecular van der Waals pairs (from the
special-pair table of md/topology.py, unscaled).  No pair kernel of the engine changes, and without an alchemical
region the engine runs exactly as before.  The solute's tied parameters must be its own
(`alchemical_system` gives the molecule its own keys).  At lambda = (1, 1) the Hamiltonian is the
original one (tests/test_alchemy.py: 1e-10 in float64).

Derivatives.  The energy is variational in the induced dipoles, so dU/dlambda is the partial
derivative at the converged dipoles (Hellmann-Feynman): jax.grad of the fixed-mu energy with
respect to lambda (`Alchemy.dudl`), checked against finite differences with the dipoles re-solved.

Windows (LambdaWindows).  lambda is a traced (2,) value of MDState (`MDState.lam`), like kT for
temperature replica exchange, so every window shares one compiled step and all windows are one
stacked state advanced by jax.vmap (remd.MDReplicas, whose resizing, checkpoints and exchange
bookkeeping they reuse).  Samples (`LambdaWindows.sample`, every `sample_every` steps):
  u[k, n] = beta U_k(x_n), the configuration of window n in the Hamiltonian of window k, for every
  pair (k, n): U_k = E_ff(x; lambda_e(k)) + E_sc(x; lambda_v(k)); E_ff needs the induced dipoles
  re-solved at lambda_e(k) (one CG from the configuration's own dipoles per distinct lambda_e; the
  van der Waals stage shares lambda_e = 0, so it adds one solve), E_sc is a small row sum;
  dU/dlambda (2,) at each window's own lambda, at its converged dipoles.
u_n(x_n) is beta times the step's potential energy (restraints and bonded terms included); P V
under NPT, a per-configuration constant, cancels in every estimator and in the exchange criterion
and is left out.

Hamiltonian replica exchange (FreeEnergyRun, exchange_every > 0): neighbouring windows try to swap
configurations with the Metropolis test of remd.metropolis on the sampled u (no extra energy
evaluation); an accepted swap re-evaluates forces and dipoles in the new Hamiltonian, books the
energy change as heat and restarts the dipole predictor from the new dipoles.

Engines: Simulation (rigid molecules) and FlexibleSimulation (a flexible solute, e.g. with X-H
constraints, among rigid waters).  Limits: one solute molecule; a net-charged solute would need
finite-size corrections [6]_ and is refused; the cell dipole recorder does not scale the
solute's charges (refused with an alchemical region).

Units: nm, ps, kJ/mol, K, e.

References
----------
.. [1] P. Ren, J. W. Ponder, J. Comput. Chem. 23, 1497 (2002).
.. [2] P. Ren, J. W. Ponder, J. Phys. Chem. B 107, 5933 (2003).
.. [3] Y. Shi, C. Wu, J. W. Ponder, P. Ren, J. Comput. Chem. 32, 967 (2011).
.. [4] T. C. Beutler, A. E. Mark, R. C. van Schaik, P. R. Gerber, W. F. van Gunsteren, Chem.
   Phys. Lett. 222, 529 (1994).
.. [5] M. R. Shirts, V. S. Pande, J. Chem. Phys. 122, 134508 (2005).
.. [6] G. J. Rocklin, D. L. Mobley, K. A. Dill, P. H. Hunenberger, J. Chem. Phys. 139, 184103
   (2013).

See also docs/free_energy.md, docs/fe_gradients.md; md/fe_grad.py (parameter gradients).
"""

from __future__ import annotations

import dataclasses as _dc
import json
import logging
from typing import TYPE_CHECKING, Any, TextIO

import jax
import jax.numpy as jnp
import numpy as np

from ..lj import lj_long_range
from ..system import ATOM_QUANTITIES, QUANTITIES, System
from ..units import KB
from .box import centers_of_mass, min_image, volume
from .driver import LogTable, Stopwatch, block_length, read_checkpoint, write_checkpoint
from .engine import OPTIONAL_STATE
from .forcefield import full_strain_derivative
from .io import write_restart
from .remd import ExchangeStatistics, MDReplicas, _nocount, _stack, _take, exchange_pairs, metropolis

if TYPE_CHECKING:
    from jax.typing import ArrayLike

    from .fe_grad import ParameterGradients
    from .forcefield import InductionState, PGMForceField, Result
    from .integrate import MDState

PREFIX = "alch:"  # tying-key prefix of an alchemical molecule's own parameters
LEGACY_FORMAT = "pgm_jax free energy 1"  # the "format" entry of legacy pickle checkpoints
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- setup
def alchemical_system(sys: System, solute: int, params: dict | None = None) -> tuple[System, dict[str, jax.Array]]:
    """Return `sys` with the solute's tied parameters made its own, and the parameters mapped to it.

    Molecule `solute` is replaced by a copy whose tied parameters are its own (every tying key
    prefixed "alch:"), so lambda can scale them without touching identical molecules of the
    environment.

    Parameters
    ----------
    sys : System
        The system.
    solute : int
        Index of the solute molecule.
    params : dict, optional
        Parameter pytree of sys.table (None: its initial values sys.params0).

    Returns
    -------
    system : System
        The new system.
    params : dict of str to jax.Array
        `params` mapped to the new table by key (the solute copy takes the values of the keys it
        came from).

    Raises
    ------
    ValueError
        If `solute` is out of range.
    """
    k = int(solute)
    if not 0 <= k < sys.nmol:
        raise ValueError(f"solute molecule {solute} out of range (0..{sys.nmol - 1})")
    m = sys.molecules[k]
    keys = m.tying_keys()
    new = _dc.replace(m, keys={qn: [PREFIX + key for key in keys[qn]] for qn in QUANTITIES})
    mols = list(sys.molecules)
    mols[k] = new
    out = System(mols)
    P0 = sys.params0 if params is None else params
    vals = {qn: dict(zip(sys.table.keys[qn], np.asarray(P0[qn], float).tolist())) for qn in QUANTITIES}
    P = {
        qn: jnp.asarray(
            np.array(
                [vals[qn][key[len(PREFIX) :] if key.startswith(PREFIX) else key] for key in out.table.keys[qn]], float
            )
        )
        for qn in QUANTITIES
    }
    return out, P


def standard_schedule(n_elec: int = 8, vdw: ArrayLike | None = None) -> np.ndarray:
    """Return the two-stage lambda schedule: electrostatics 1 -> 0, then van der Waals 1 -> 0.

    The electrostatics goes 1 -> 0 in `n_elec` evenly spaced windows at lambda_vdw = 1, then the
    van der Waals 1 -> 0 at lambda_elec = 0.  The window (0, 1) ends the first stage and starts the
    second.

    Parameters
    ----------
    n_elec : int
        Windows of the electrostatics stage (>= 2, both ends included).
    vdw : ArrayLike, optional
        lambda_vdw values of the second stage, decreasing, below 1 and ending at 0 (None: 0.9, 0.8,
        ..., 0.1, 0.05, 0, denser towards 0, where the soft-core integrand bends).

    Returns
    -------
    np.ndarray (K, 2)
        Windows (lambda_elec, lambda_vdw), K = n_elec + len(vdw).

    Raises
    ------
    ValueError
        If n_elec < 2 or `vdw` is not as described.
    """
    if int(n_elec) < 2:
        raise ValueError("n_elec >= 2 (both ends of the electrostatics stage)")
    lv = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0]) if vdw is None else np.asarray(vdw, float)
    if lv.ndim != 1 or np.any(np.diff(lv) >= 0) or lv[0] >= 1.0 or lv[-1] != 0.0:
        raise ValueError("vdw: decreasing lambda_vdw values below 1, ending at 0")
    le = np.linspace(1.0, 0.0, int(n_elec))
    return np.concatenate([np.stack([le, np.ones_like(le)], 1), np.stack([np.zeros_like(lv), lv], 1)])


# ----------------------------------------------------------------------------- Hamiltonian
class Alchemy:
    """Lambda-dependent Hamiltonian of one solute molecule (module docstring for the physics).

        sys, P = alchemical_system(sys, solute=0)
        alch = Alchemy(sys, solute=0)
        sim = Simulation(sys, pos, H, settings, params=P, alchemy=alch, thermostat="bussi")

    `lam` is the coupling (lambda_elec, lambda_vdw) of states whose MDState.lam is None (the
    default single simulation); LambdaWindows gives each window its own.  The cutoff, long-range
    correction, van der Waals form and electrostatics level are taken from the force field the
    first time the Hamiltonian is bound to one (`check`, called by the integrator).  A mutable host
    object (not a pytree); its methods are traceable functions of positions, box, parameters and
    lambda.

    Attributes
    ----------
    sys : System
        The system (with the solute's own parameter keys).
    solute : int
        Index of the solute molecule.
    sc_alpha : float
        Soft-core alpha (dimensionless).
    alpha_floor : float
        eps of the polarizability scaling.
    intramolecular : {"annihilate", "keep"}
        Treatment of the solute's intramolecular electrostatics.
    lam : jax.Array (2,) float64
        Default (lambda_elec, lambda_vdw).
    atoms : jax.Array (n_s,) int32
        Global indices of the solute's atoms (atoms_np: the same as numpy).
    is_solute : jax.Array (N,) bool
        Solute mask.
    params0 : dict
        Initial parameter table sys.params0.
    mask : dict of str to jax.Array
        Per table quantity: which entries are the solute's own.
    bound : tuple or None
        (vdw form, van der Waals cutoff [nm], long-range correction on, elec level) of the bound
        force field.
    """

    def __init__(
        self,
        sys: System,
        solute: int,
        lam: ArrayLike = (1.0, 1.0),
        sc_alpha: float = 0.5,
        alpha_floor: float = 1e-8,
        intramolecular: str = "annihilate",
    ) -> None:
        """Set up the alchemical region of molecule `solute`.

        Parameters
        ----------
        sys : System
            The system, built with `alchemical_system` so that the solute's parameters are its own.
        solute : int
            Index of the solute molecule.
        lam : ArrayLike (2,)
            Default coupling (lambda_elec, lambda_vdw), each in [0, 1].
        sc_alpha : float
            Soft-core alpha (> 0; module docstring).
        alpha_floor : float
            eps of the polarizability scaling alpha_s [eps + (1 - eps) lambda_e], in (0, 1e-3).
        intramolecular : {"annihilate", "keep"}
            "annihilate": the solute's intramolecular electrostatics goes with lambda_elec (gas-phase
            leg separately, e.g. GasPhaseLeg); "keep": kept by the gas-phase correction (flexible pGM
            solutes).

        Raises
        ------
        ValueError
            Invalid options or lambda, a solute index out of range, or a solute that shares parameters
            with other molecules.
        NotImplementedError
            A net-charged solute.
        """
        k = int(solute)
        if intramolecular not in ("annihilate", "keep"):
            raise ValueError("intramolecular: 'annihilate' (with a gas-phase leg) or 'keep' (gas-phase correction)")
        self.intramolecular = intramolecular
        if not 0 <= k < sys.nmol:
            raise ValueError(f"solute molecule {solute} out of range (0..{sys.nmol - 1})")
        if not sc_alpha > 0.0:
            raise ValueError("sc_alpha must be positive (0 is plain Lennard-Jones: singular at overlap)")
        if not 0.0 < alpha_floor < 1e-3:
            raise ValueError("alpha_floor must be in (0, 1e-3)")
        self.sys, self.solute = sys, k
        self.sc_alpha, self.alpha_floor = float(sc_alpha), float(alpha_floor)
        self.lam = self._check_lam(lam)
        n = sys.n
        atoms = np.arange(int(sys.offsets[k]), int(sys.offsets[k + 1]))
        env = np.setdiff1d(np.arange(n), atoms)
        self.atoms_np = atoms
        self.atoms = jnp.asarray(atoms, jnp.int32)
        is_sol = np.zeros(n, bool)
        is_sol[atoms] = True
        self.is_solute = jnp.asarray(is_sol)
        self.params0 = sys.params0
        # the solute's own parameter keys (masks over the table)
        self.mask = {}
        for qn in ATOM_QUANTITIES:
            own = np.unique(sys.idx[qn][atoms])
            shared = np.intersect1d(own, sys.idx[qn][env])
            if shared.size:
                names = ", ".join(sys.table.keys[qn][i] for i in shared[:4])
                raise ValueError(
                    f"the solute shares its {qn} parameters ({names}) "
                    f"with other molecules: build the system with alchemical_system(sys, {k})"
                )
            m = np.zeros(len(sys.table.keys[qn]), bool)
            m[own] = True
            self.mask[qn] = jnp.asarray(m)
        terms = np.isin(sys.cov_i, atoms)
        own, other = np.unique(sys.idx["cov"][terms]), sys.idx["cov"][~terms]
        if np.intersect1d(own, other).size:
            raise ValueError(
                f"the solute shares covalent-dipole parameters with other molecules: build the system "
                f"with alchemical_system(sys, {k})"
            )
        m = np.zeros(len(sys.table.keys["cov"]), bool)
        m[own] = True
        self.mask["cov"] = jnp.asarray(m)
        charge = float(np.sum(np.asarray(sys.expand(None)["q"])[atoms]))
        if abs(charge) > 1e-6:
            raise NotImplementedError(
                f"the solute carries a net charge ({charge:+.4f} e): charged solutes need "
                "finite-size corrections of the periodic electrostatics, not implemented"
            )
        self.bound = None  # (vdw form, van der Waals cutoff, long-range correction, elec)
        self._intra = None  # the solute's intramolecular van der Waals pairs (i, j, weight)
        self._gas = None  # gas-phase electrostatics of the solute ("keep")

    # ------------------------------------------------------------------ binding / checks
    @staticmethod
    def _check_lam(lam: ArrayLike) -> jnp.ndarray:
        """Return lam as a float64 (2,) array after checking that it is (lambda_elec, lambda_vdw) in [0, 1]."""
        a = np.asarray(lam, float)
        if a.shape != (2,) or np.any(a < 0.0) or np.any(a > 1.0):
            raise ValueError(f"lambda = (lambda_elec, lambda_vdw) in [0, 1], got {lam}")
        return jnp.asarray(a, jnp.float64)

    def check(self, ff: PGMForceField) -> None:
        """Bind the Hamiltonian to a force field and refuse what it does not implement.

        Takes the van der Waals form and cutoff, the long-range correction and the electrostatics level
        from the force field, the solute's intramolecular van der Waals pairs from its topology
        (special pairs with a nonzero weight; the ordinary rows lose them with the solute's parameters,
        so they are evaluated here, unscaled) and, for intramolecular="keep", builds the dense gas-phase
        energy of the lone solute.

        Parameters
        ----------
        ff : PGMForceField
            The force field of the integrator.

        Raises
        ------
        ValueError
            Another system, or settings different from those it is already bound to.
        NotImplementedError
            A van der Waals form other than "lj" / "none", or charge flux.
        """
        s = ff.s
        if ff.sys.fingerprint() != self.sys.fingerprint():
            raise ValueError("the alchemical region was built for another System")
        if s.terms.vdw not in ("lj", "none"):
            raise NotImplementedError(
                f"soft-core van der Waals is implemented for vdw='lj' (and 'none'), not {s.terms.vdw!r}"
            )
        if getattr(ff, "flux", None) is not None:
            raise NotImplementedError(
                "an alchemical region with charge flux: lambda scales the fixed charges, "
                "not the flux terms, so the decoupled state would keep charges"
            )
        b = (s.terms.vdw, float(ff.rc_v), bool(s.terms.lj_lrc), s.terms.elec)
        if self.bound is not None and self.bound != b:
            raise ValueError(f"the alchemical region is bound to other settings {self.bound}, not {b}")
        self.bound = b
        # intramolecular van der Waals pairs of the solute (flexible molecules: special pairs with a
        # nonzero weight); the ordinary rows lose them with the solute's parameters, so they are
        # evaluated here, unscaled
        sp, w = np.asarray(ff.topology.special), np.asarray(ff.topology.special_w)
        pairs = [
            (int(i), int(j), float(w[i, c]))
            for i in self.atoms_np
            for c, j in enumerate(sp[i])
            if i < j < self.sys.n and w[i, c] != 0.0
        ]
        if pairs and s.terms.vdw != "none":
            ii, jj, ww = (np.array(x) for x in zip(*pairs))
            self._intra = (jnp.asarray(ii, jnp.int32), jnp.asarray(jj, jnp.int32), jnp.asarray(ww, jnp.float64))
        else:
            self._intra = None
        if self.intramolecular == "keep" and self._gas is None:  # the solute alone in vacuum (dense pGM)
            from ..channels import ElecChannel
            from ..model import Model

            sub, _ = self.sys.sub((self.solute,))
            self._gas = Model([ElecChannel.level(s.terms.elec)]).energy_fn(sub)

    @property
    def vdw(self) -> str:
        """Van der Waals form of the bound force field ("lj" or "none"); RuntimeError if not bound."""
        if self.bound is None:
            raise RuntimeError("Alchemy is not bound to a force field yet (Alchemy.check(ff))")
        return self.bound[0]

    def describe(self) -> str:
        """Return one line for the log header (solute, lambda, soft-core alpha, floor, intramolecular mode)."""
        m = self.sys.molecules[self.solute]
        return (
            f"solute molecule {self.solute} ({m.name}, {m.n} atoms), lambda (elec, vdw) = "
            f"({float(self.lam[0]):g}, {float(self.lam[1]):g}), soft core alpha {self.sc_alpha:g}, "
            f"polarizability floor {self.alpha_floor:g}, intramolecular electrostatics: {self.intramolecular}"
        )

    def _lam(self, lam: ArrayLike | None) -> jax.Array:
        """Return lam as a float64 array, or the default `self.lam` for None."""
        return self.lam if lam is None else jnp.asarray(lam, jnp.float64)

    # ------------------------------------------------------------------ parameters at lambda
    def params(self, params: dict | None, lam: ArrayLike | None) -> dict[str, jax.Array]:
        """Return the parameter table of the ordinary force field at lambda.

        The solute's charges and covalent-dipole strengths times lambda_e, its polarizabilities times
        eps + (1 - eps) lambda_e, its van der Waals parameters zero (its van der Waals is
        `vdw_energy`); every other entry unchanged.  Differentiable in `params` and `lam`.

        Parameters
        ----------
        params : dict, optional
            Parameter pytree of sys.table (None: params0).
        lam : ArrayLike (2,), optional
            (lambda_elec, lambda_vdw) (None: the default); only lambda_elec is used.

        Returns
        -------
        dict of str to jax.Array
            The scaled table.
        """
        P = dict(self.params0 if params is None else params)
        le = self._lam(lam)[0]
        a = self.alpha_floor + (1.0 - self.alpha_floor) * le
        P["q"] = jnp.where(self.mask["q"], le * P["q"], P["q"])
        P["cov"] = jnp.where(self.mask["cov"], le * P["cov"], P["cov"])
        P["alpha"] = jnp.where(self.mask["alpha"], a * P["alpha"], P["alpha"])
        for qn in ("lj_sqrt_eps", "gvdw_sqrt_a", "gvdw_sqrt_c6"):
            P[qn] = jnp.where(self.mask[qn], 0.0, P[qn])
        return P

    def _tail(self, P: dict[str, jax.Array], H: jax.Array) -> jax.Array:
        """Return the solute's share of the r^-6 long-range correction [kJ/mol].

        The full tail minus the tail with the solute's LJ switched off, for per-atom parameters P
        (unscaled by lambda) and box H [nm].
        """
        rc = self.bound[1]
        off = dict(P, lj_sqrt_eps=jnp.where(self.is_solute, 0.0, P["lj_sqrt_eps"]))
        V = volume(H)
        return lj_long_range(P, V, rc) - lj_long_range(off, V, rc)

    def softcore_energy(
        self, pos: jax.Array, H: jax.Array, cand: jax.Array, params: dict | None, lam_v: float | jax.Array
    ) -> jax.Array:
        """Return the soft-core van der Waals energy of the solute with its environment.

        Pairs of each solute atom's candidate row with environment atoms inside the van der Waals
        cutoff (U_ij = lambda_v eps_ij [1/w^2 - 2/w], module docstring), plus lambda_v times the
        solute's long-range correction.  Differentiable (forces by autodiff).

        Parameters
        ----------
        pos : jax.Array (N, 3)
            Positions [nm].
        H : jax.Array (3, 3)
            Box, lattice vectors as rows [nm].
        cand : jax.Array (N, C) int
            Candidate rows from the neighbour list (padding N).
        params : dict, optional
            Parameter pytree (None: params0).
        lam_v : float or jax.Array ()
            lambda_vdw.

        Returns
        -------
        jax.Array () float64
            Energy [kJ/mol] (0 for vdw="none").
        """
        if self.vdw == "none":
            return jnp.zeros((), jnp.float64)
        N = self.sys.n
        P = self.sys.expand(self.params0 if params is None else params)
        P = {q: jnp.asarray(P[q], jnp.float64) for q in ("lj_rmin_half", "lj_sqrt_eps")}
        i = self.atoms
        k = cand[i]
        kk = jnp.where(k < N, k, 0)
        valid = (k < N) & ~self.is_solute[kk]
        x = min_image(pos[kk] - pos[i][:, None, :], H)
        r2 = jnp.sum(x * x, axis=-1)
        rmin = P["lj_rmin_half"][i][:, None] + P["lj_rmin_half"][kk]
        eps = P["lj_sqrt_eps"][i][:, None] * P["lj_sqrt_eps"][kk]
        on = valid & (r2 < self.bound[1] ** 2) & (eps != 0.0) & (rmin > 0.0)
        s = jnp.where(on, r2, 1.0) / jnp.where(on, rmin * rmin, 1.0)  # (r / rmin)^2; masked pairs 1
        w = s * s * s + 0.5 * self.sc_alpha * (1.0 - lam_v)  # w = (r / rmin)^6 + (sc_alpha / 2)(1 - lambda_v)
        e = jnp.sum(jnp.where(on, lam_v * eps * (1.0 / (w * w) - 2.0 / w), 0.0))
        if self.bound[2]:
            e = e + lam_v * self._tail(P, H)
        return e

    def intra_energy(self, pos: jax.Array, H: jax.Array, params: dict | None) -> jax.Array:
        """Return the solute's intramolecular van der Waals energy, unscaled by lambda.

        Flexible molecules: pairs at least lj_min_sep bonds apart and scaled 1-4 pairs, with the weights
        of md/topology.py, inside the van der Waals cutoff; the decoupling keeps it.  Zero for rigid
        molecules.

        Parameters
        ----------
        pos : jax.Array (N, 3)
            Positions [nm].
        H : jax.Array (3, 3)
            Box, lattice vectors as rows [nm].
        params : dict, optional
            Parameter pytree (None: params0).

        Returns
        -------
        jax.Array () float64
            Energy [kJ/mol].
        """
        if self._intra is None:
            return jnp.zeros((), jnp.float64)
        i, j, w = self._intra
        P = self.sys.expand(self.params0 if params is None else params)
        x = min_image(pos[j] - pos[i], H)
        r2 = jnp.sum(x * x, axis=-1)
        rmin = jnp.asarray(P["lj_rmin_half"], jnp.float64)
        se = jnp.asarray(P["lj_sqrt_eps"], jnp.float64)
        s6 = ((rmin[i] + rmin[j]) ** 2 / r2) ** 3
        return jnp.sum(jnp.where(r2 < self.bound[1] ** 2, w * se[i] * se[j] * (s6 * s6 - 2.0 * s6), 0.0))

    def correction_energy(self, pos: jax.Array, params: dict | None, lam_e: float | jax.Array) -> jax.Array:
        """Return the gas-phase correction E_gas(x_s; lambda_e = 1) - E_gas(x_s; lambda_e) ("keep" only).

        The solute's electrostatics in vacuum (dense pGM, float64) at full coupling minus at lambda_e,
        so that the solute keeps its whole gas-phase intramolecular electrostatics at every lambda and
        only its coupling to the environment (and to its periodic images) is switched off.

        Parameters
        ----------
        pos : jax.Array (N, 3)
            Positions [nm] (the solute's atoms are used).
        params : dict, optional
            Parameter pytree (None: params0).
        lam_e : float or jax.Array ()
            lambda_elec.

        Returns
        -------
        jax.Array ()
            Energy [kJ/mol]; 0 with intramolecular="annihilate".

        Raises
        ------
        RuntimeError
            If not bound to a force field.
        """
        if self.intramolecular != "keep":
            return jnp.zeros((), jnp.float64)
        if self._gas is None:
            raise RuntimeError("Alchemy is not bound to a force field yet (Alchemy.check(ff))")
        x = pos[self.atoms]
        one = self._gas(x, self.params(params, jnp.ones(2)))["total"]
        return one - self._gas(x, self.params(params, jnp.stack([jnp.asarray(lam_e, jnp.float64), 1.0])))["total"]

    def extra_energy(
        self, pos: jax.Array, H: jax.Array, cand: jax.Array, params: dict | None, lam: jax.Array
    ) -> jax.Array:
        """Return everything the scaled ordinary force field leaves out [kJ/mol].

        `vdw_energy` at lambda_vdw plus `correction_energy` at lambda_elec; arguments as there, lam the
        (2,) coupling.
        """
        return self.vdw_energy(pos, H, cand, params, lam[1]) + self.correction_energy(pos, params, lam[0])

    def vdw_energy(
        self, pos: jax.Array, H: jax.Array, cand: jax.Array, params: dict | None, lam_v: float | jax.Array
    ) -> jax.Array:
        """Return all van der Waals energy of the solute (the ordinary rows see none of it) [kJ/mol].

        The soft-core solute-environment pairs and their long-range correction at lambda_vdw
        (`softcore_energy`) plus its intramolecular pairs (`intra_energy`).
        """
        return self.softcore_energy(pos, H, cand, params, lam_v) + self.intra_energy(pos, H, params)

    # ------------------------------------------------------------------ engine interface
    def compute(
        self,
        ff: PGMForceField,
        pos: jax.Array,
        H: jax.Array,
        cand: jax.Array,
        ind: InductionState,
        params: dict | None,
        lam: ArrayLike | None,
    ) -> Result:
        """Return the energies, forces and dipoles of the Hamiltonian at lam.

        The ordinary force field at the scaled parameters (PGMForceField.compute) plus `extra_energy`,
        whose energy is added to "vdw" and "total" and whose forces come from autodiff.

        Parameters
        ----------
        ff : PGMForceField
            The bound force field.
        pos : jax.Array (N, 3)
            Positions [nm].
        H : jax.Array (3, 3)
            Box, lattice vectors as rows [nm].
        cand : jax.Array (N, C) int
            Candidate rows (padding N).
        ind : InductionState
            Induction state (predictor history) of the previous step.
        params : dict, optional
            Parameter pytree (None: params0).
        lam : ArrayLike (2,), optional
            (lambda_elec, lambda_vdw) (None: the default).

        Returns
        -------
        forcefield.Result
            As PGMForceField.compute, energies [kJ/mol], forces [kJ/mol/nm].
        """
        lam = self._lam(lam)
        res = ff.compute(pos, H, cand, ind, self.params(params, lam))
        if self.vdw == "none" and self.intramolecular != "keep":
            return res
        e, g = jax.value_and_grad(self.extra_energy)(pos, H, cand, params, lam)
        energy = dict(res.energy, vdw=res.energy["vdw"] + e, total=res.energy["total"] + e)
        return res._replace(energy=energy, forces=res.forces - g)

    def energy(
        self,
        ff: PGMForceField,
        pos: jax.Array,
        H: jax.Array,
        cand: jax.Array,
        ind: InductionState,
        params: dict | None,
        lam: ArrayLike | None,
    ) -> tuple[jax.Array, InductionState, jax.Array, jax.Array]:
        """Return the energy at lam with the dipoles solved, as PGMForceField.energy.

        Arguments as `compute`.

        Returns
        -------
        energy : jax.Array ()
            Total energy [kJ/mol] (`extra_energy` included).
        induction : InductionState
            State with the converged dipoles.
        iterations : jax.Array () int
            CG iterations.
        overflow : jax.Array () bool
            Row capacity exceeded.
        """
        lam = self._lam(lam)
        e, ind, it, ovf = ff.energy(pos, H, cand, ind, self.params(params, lam))
        return e + self.extra_energy(pos, H, cand, params, lam), ind, it, ovf

    def energy_fixed_mu(
        self,
        ff: PGMForceField,
        pos: jax.Array,
        H: jax.Array,
        cand: jax.Array,
        mu: jax.Array,
        params: dict | None,
        lam: ArrayLike | None,
    ) -> jax.Array:
        """Return the total energy at lam with the induced dipoles held at mu [kJ/mol].

        Differentiable in positions, box, parameters and lam; arguments as `compute`, with
        mu (N, 3) the induced dipoles [e nm] instead of the induction state.
        """
        lam = self._lam(lam)
        e, _ = ff.energy_fixed_mu(pos, H, mu, cand, ff._atoms(self.params(params, lam)))
        return e + self.extra_energy(pos, H, cand, params, lam)

    def dudl(
        self,
        ff: PGMForceField,
        pos: jax.Array,
        H: jax.Array,
        cand: jax.Array,
        mu: jax.Array,
        params: dict | None,
        lam: ArrayLike | None,
    ) -> jax.Array:
        """Return (dU/dlambda_elec, dU/dlambda_vdw) (2,) [kJ/mol] at the converged dipoles mu.

        jax.grad of `energy_fixed_mu` with respect to lam (Hellmann-Feynman: the energy is stationary
        in mu, so no derivative of the solve is needed); arguments as `energy_fixed_mu`.
        """
        return jax.grad(lambda l: self.energy_fixed_mu(ff, pos, H, cand, mu, params, l))(self._lam(lam))

    def strain_derivative(
        self,
        ff: PGMForceField,
        pos: jax.Array,
        H: jax.Array,
        cand: jax.Array,
        mu: jax.Array,
        params: dict | None,
        lam: ArrayLike | None,
        molecular: bool = True,
    ) -> jax.Array:
        """Return dE/d eps (3, 3) [kJ/mol] at fixed mu of the Hamiltonian at lam.

        As PGMForceField.strain_derivative at the scaled parameters, plus the strain derivative of
        `extra_energy` (forcefield.full_strain_derivative; molecular scaling of the centres of mass
        when `molecular`) and the tail impulse term -lambda_v E_tail I of the soft-core long-range
        correction.  Arguments as `energy_fixed_mu`.
        """
        lam = self._lam(lam)
        W = ff.strain_derivative(pos, H, cand, mu, self.params(params, lam), molecular)
        if self.vdw == "none" and self.intramolecular != "keep":
            return W
        com = None
        if molecular:
            w = ff.masses
            com = centers_of_mass(pos, w, ff.mol, self.sys.nmol)
        W = W + full_strain_derivative(lambda x, h: self.extra_energy(x, h, cand, params, lam), pos, H, ff.mol, com)
        if self.bound[2]:
            P = self.sys.expand(self.params0 if params is None else params)
            W = W - lam[1] * self._tail(P, H) * jnp.eye(3)
        return W


# ----------------------------------------------------------------------------- gas-phase leg
class GasPhaseLeg:
    """The solute alone in vacuum with the alchemical parameters at lambda_elec.

    The gas-phase leg of the cycle with intramolecular="annihilate" (with "keep" the leg is part of
    the Hamiltonian).  E_gas(lambda_e) is the pGM energy of the isolated molecule (every pair,
    induced dipoles by a dense solve: channels.ElecChannel, the kernels and Coulomb constant of the
    MD engine) at the solute's rigid geometry `xyz`.  For a rigid solute the gas-phase energy does
    not depend on the configuration (only on orientation and position, to which it is invariant),
    so the free energy of annihilating its electrostatics is exact from one configuration:

        Delta G_gas(1 -> 0) = E_gas(0) - E_gas(1)      (-kT ln <exp(-beta dU)> of a constant dU).

    dudl(lambda_e) = dE_gas/dlambda_e is what TI subtracts from the solution-phase integrand
    (free_energy.estimate): the large intramolecular part of pGM's electrostatics then drops out
    before the quadrature.

    Attributes
    ----------
    alchemy : Alchemy
        The alchemical region.
    xyz : jax.Array (n_s, 3)
        Solute geometry [nm].
    """

    def __init__(self, alchemy: Alchemy, xyz: ArrayLike, elec: str = "qpi") -> None:
        """Build the jitted gas-phase energy and its lambda derivative.

        Parameters
        ----------
        alchemy : Alchemy
            The alchemical region (its solute and parameter scaling).
        xyz : ArrayLike (n_s, 3)
            Solute geometry [nm].
        elec : {"q", "qp", "qi", "qpi"}
            Electrostatics level (channels.ElecChannel.level): charges, permanent dipoles, induction.
        """
        from ..channels import ElecChannel
        from ..model import Model

        self.alchemy = alchemy
        sub, _ = alchemy.sys.sub((alchemy.solute,))
        self.xyz = jnp.asarray(np.asarray(xyz, float).reshape(sub.n, 3))
        f = Model([ElecChannel.level(elec)]).energy_fn(sub)
        self._e = jax.jit(lambda le, params: f(self.xyz, alchemy.params(params, jnp.stack([le, 1.0])))["total"])
        self._g = jax.jit(
            jax.grad(lambda le, params: f(self.xyz, alchemy.params(params, jnp.stack([le, 1.0])))["total"])
        )

    def energy(self, lam_e: float, params: dict | None = None) -> float:
        """Return E_gas(lambda_e) [kJ/mol] (params: parameter pytree, None: initial values)."""
        return float(self._e(jnp.asarray(float(lam_e)), params))

    def dudl(self, lam_e: float, params: dict | None = None) -> float:
        """Return dE_gas/dlambda_e [kJ/mol] (params: parameter pytree, None: initial values)."""
        return float(self._g(jnp.asarray(float(lam_e)), params))

    def delta_g(self, params: dict | None = None) -> float:
        """Return Delta G_gas(lambda_e 1 -> 0) = E_gas(0) - E_gas(1) [kJ/mol] (rigid solute: exact)."""
        return self.energy(0.0, params) - self.energy(1.0, params)


def lone_solute(sys: System, solute: int, pos: ArrayLike, box_nm: float = 4.0) -> tuple[System, np.ndarray, np.ndarray]:
    """Return the solute molecule alone, centred in a cubic box: the gas-phase leg of a flexible solute.

    The leg is sampled with the same engine at the lambda_elec windows (FlexibleSimulation +
    LambdaWindows + FreeEnergyRun, as the solution leg).  The sub-system shares sys's parameter
    table, so the same params and Alchemy(sub, 0) apply.  The periodic images of a neutral
    molecule contribute ~1e-3 kJ/mol at 4 nm (tests); run it with
    MDSettings().replace(lj_lrc=False) (there is no continuum of other atoms), a PME grid for this
    box and an atom neighbour list (MDSettings neighbors.mode "atom": a molecule list has no other
    molecule to list).

    Parameters
    ----------
    sys : System
        The system.
    solute : int
        Index of the solute molecule.
    pos : ArrayLike (N, 3)
        Positions of sys [nm], the solute whole.
    box_nm : float
        Edge of the cubic box [nm].

    Returns
    -------
    system : System
        The one-molecule sub-system.
    positions : np.ndarray (n_s, 3)
        Solute positions [nm], centred (by the mean of the atoms) in the box.
    box : np.ndarray (3, 3)
        box_nm times the identity [nm].
    """
    sub, idx = sys.sub((int(solute),))
    x = np.asarray(pos, float)[idx]
    x = x - x.mean(axis=0) + 0.5 * float(box_nm)
    return sub, x, np.eye(3) * float(box_nm)


# ----------------------------------------------------------------------------- lambda windows
def _select(mask: ArrayLike, a: MDState, b: MDState) -> MDState:
    """Stacked states: slot k from a where mask[k], else from b (the unbatched step counter from a)."""
    m = jnp.asarray(mask)
    A, B = _nocount(a), _nocount(b)
    out = jax.tree_util.tree_map(lambda x, y: jnp.where(m.reshape((-1,) + (1,) * (jnp.ndim(x) - 1)), x, y), A, B)
    return out.set(induction=out.induction.set(count=a.induction.count))


class LambdaWindows(MDReplicas):
    """The lambda windows of one simulation with an alchemical region, at one temperature.

    The engine of FreeEnergyRun, built on remd.MDReplicas (resizing, shared static sizes and
    checkpoints).  Every window starts from the current configuration of `sim` with momenta drawn
    from its own random stream (`seed`) and forces at its own lambda.

    batched=True (NVT): the windows are one stacked state advanced by jax.vmap of the step (lambda
    traced in MDState.lam), one program for all windows; batched=False advances them one after the
    other through the driver of `sim` (NPT, or systems that fill the GPU alone).

    Attributes
    ----------
    alchemy : Alchemy
        The integrator's alchemical region.
    lambdas : np.ndarray (K, 2)
        (lambda_elec, lambda_vdw) of the windows.
    lam_e : np.ndarray
        Distinct lambda_elec values (one dipole re-solve each per sample).
    group : np.ndarray (K,) int
        Index of each window's lambda_elec in `lam_e`.

    Other attributes as MDReplicas (`temperatures` all equal).
    """

    def __init__(self, sim: Any, lambdas: ArrayLike, batched: bool = True, seed: int = 0) -> None:
        """Build the window states from the current configuration of `sim`.

        Parameters
        ----------
        sim : Simulation or FlexibleSimulation
            Created with alchemy=Alchemy(...) and a thermostat.
        lambdas : ArrayLike (K, 2)
            Distinct windows (lambda_elec, lambda_vdw) in [0, 1], K >= 2 (e.g. `standard_schedule`).
        batched : bool
            One vmapped program for all windows (NVT only) or one after the other.
        seed : int
            Seed of the windows' momenta and thermostat streams.

        Raises
        ------
        ValueError
            No alchemical region, no thermostat, batched NPT, or invalid / repeated lambdas.
        """
        integ = sim.integ
        if getattr(integ, "alchemy", None) is None:
            raise ValueError("the simulation has no alchemical region: Simulation(..., alchemy=Alchemy(...))")
        if integ.thermostat is None:
            raise ValueError("lambda windows need a thermostat (create the simulation with thermostat=...)")
        if batched and sim.ensemble == "npt":
            raise ValueError(
                "batched windows run NVT only (under vmap the barostat's trial energy would be "
                "evaluated every step); use batched=False for NPT"
            )
        L = np.asarray(lambdas, float)
        if L.ndim != 2 or L.shape[1] != 2 or len(L) < 2 or np.any(L < 0.0) or np.any(L > 1.0):
            raise ValueError("lambdas: (K >= 2, 2) array of (lambda_elec, lambda_vdw) in [0, 1]")
        if len({tuple(r) for r in L.tolist()}) != len(L):
            raise ValueError("lambda windows must be distinct")
        self.sim, self.integ, self.batched = sim, integ, bool(batched)
        self.alchemy = integ.alchemy
        self.lambdas = L
        self.temperatures = np.full(len(L), integ.kT / KB)  # one temperature (MDReplicas interface)
        self.n, self.dt = len(L), float(sim.dt)
        self.pressure = float(integ.pressure) if sim.ensemble == "npt" else None
        self.time_ps = 0.0
        base = sim.state
        self._template = base.nbr
        states = []
        for lam, key in zip(L, jax.random.split(jax.random.PRNGKey(int(seed)), self.n)):
            st = integ.init(base.dyn.position, base.box, key)
            st = integ.forces(st.set(lam=jnp.asarray(lam, jnp.float64), nbr=base.nbr), False)
            states.append(st)
        self._exchange_seq = jax.jit(self._exchange_one)
        self._switch_seq = jax.jit(self._switch_one)
        # distinct lambda_elec values: one dipole solve each per sample
        self.lam_e = np.unique(L[:, 0])
        self.group = np.searchsorted(self.lam_e, L[:, 0])
        self._samplers = {}
        if self.batched:
            self.S = _stack(states)
            self._build()
        else:
            self.states = states

    def _build(self) -> None:
        """Build MDReplicas' vmapped entry points and the vmapped `_switch_one`."""
        super()._build()
        from .remd import _axes

        ax = _axes(self.S)
        self._switch = jax.jit(jax.vmap(self._switch_one, in_axes=(ax,), out_axes=ax))

    # ------------------------------------------------------------------ samples
    def _sample_one(
        self, st: MDState, lam_e: jax.Array, group: jax.Array, lam_v: jax.Array, lam_e_all: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return the energies of one configuration in every window's Hamiltonian.

        Parameters
        ----------
        st : MDState
            State of one window.
        lam_e : jax.Array (L,)
            Distinct lambda_elec values (`lam_e`): one dipole re-solve each (lax.map).
        group : jax.Array (K,) int
            Index of each window's lambda_elec in `lam_e`.
        lam_v : jax.Array (K,)
            lambda_vdw of every window.
        lam_e_all : jax.Array (K,)
            lambda_elec of every window.

        Returns
        -------
        U : jax.Array (K,)
            U_k(x) [kJ/mol] for every window k (force field at lambda_elec(k), `extra_energy`, and the
            lambda-independent restraint and bonded energies, so that U_n(x_n) is the step's potential
            energy).
        dudl : jax.Array (2,)
            dU/dlambda [kJ/mol] at the window's own lambda and converged dipoles.
        iterations : jax.Array () int
            Largest CG iteration count of the re-solves.
        overflow : jax.Array () bool
            Row capacity exceeded.
        """
        integ, ff, alch = self.integ, self.sim.ff, self.alchemy
        params = integ.params
        if hasattr(integ, "flex"):  # flexible engine: atoms, neighbour-list group centres
            pos = st.dyn.position
            centers = integ.flex.list_centers(pos)
        else:  # rigid bodies
            pos = self.sim.rigid.positions(st.dyn.position)
            centers = st.dyn.position.center
        cand, ovf0 = integ.nb.candidates(st.nbr, centers, st.box, pos)

        def e_ff(le: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:  # force-field energy at lambda_elec le
            e, _, it, ovf = ff.energy(pos, st.box, cand, st.induction, alch.params(params, jnp.stack([le, 1.0])))
            return e, it, ovf

        E, it, ovf = jax.lax.map(e_ff, lam_e)
        esc = jax.vmap(lambda le, lv: alch.extra_energy(pos, st.box, cand, params, jnp.stack([le, lv])))(
            lam_e_all, lam_v
        )
        g = alch.dudl(ff, pos, st.box, cand, st.induction.mu, params, st.lam)
        # lambda-independent terms (restraints, bonded energy of flexible molecules): the same in every
        # window, kept so that u_n(x_n) = beta U of the step
        const = integ._restraint_energy(pos, st.box, st.bias) + (
            integ.flex.energy(pos) if hasattr(integ, "flex") else 0.0
        )
        return E[group] + esc + const, g, jnp.max(it), jnp.any(ovf) | ovf0

    def _sampler(self) -> Any:
        """Return the jitted (and, batched, vmapped) `_sample_one`, cached by static sizes and layout."""
        # the stacked state's structure too: a loaded checkpoint or a resize can change the neighbour
        # list's static layout at the same row capacities
        key = (self._sizes(), jax.tree_util.tree_structure(self.S) if self.batched else None)
        if key not in self._samplers:
            if self.batched:
                from .remd import _axes

                f = jax.vmap(self._sample_one, in_axes=(_axes(self.S), None, None, None, None))
            else:
                f = self._sample_one
            self._samplers = {key: jax.jit(f)}
        return self._samplers[key]

    def sample(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Return the reduced energies and dU/dlambda of every window's configuration.

        Returns
        -------
        u : np.ndarray (K, K)
            u[k, n] = beta U_k(x_n), the configuration of window n in the Hamiltonian of window k (the
            potential energy of the step at k = n; P V left out under NPT), dimensionless.
        dudl : np.ndarray (K, 2)
            dU/dlambda [kJ/mol] of each window at its own lambda.
        iterations : int
            Largest CG iteration count of the re-solves.

        Raises
        ------
        RuntimeError
            If the row capacity is exceeded (the configuration's rows should fit).
        """
        f = self._sampler()
        args = (
            jnp.asarray(self.lam_e),
            jnp.asarray(self.group),
            jnp.asarray(self.lambdas[:, 1]),
            jnp.asarray(self.lambdas[:, 0]),
        )
        if self.batched:
            U, g, it, ovf = f(self.S, *args)
        else:
            outs = [f(s, *args) for s in self.states]
            U, g, it, ovf = (jnp.stack([o[j] for o in outs]) for j in range(4))
        if bool(np.any(np.asarray(ovf))):
            raise RuntimeError("row capacity exceeded while sampling (the configuration's rows should fit)")
        beta = 1.0 / float(self.integ.kT)
        return beta * np.asarray(U, float).T, np.asarray(g, float), int(np.max(np.asarray(it)))

    # ------------------------------------------------------------------ Hamiltonian exchange
    def _switch_one(self, st: MDState) -> MDState:
        """Return a swapped-in state re-evaluated in the slot's Hamiltonian.

        Forces, energies and dipoles at the slot's lambda; the energy change is booked as heat; the
        dipole predictor restarts from the new dipoles (its history was recorded in another
        Hamiltonian).
        """
        e0 = st.epot
        new = self.integ._state_forces(st, True)
        ind = new.induction
        return new.set(
            induction=ind.set(hist=jnp.broadcast_to(ind.mu, ind.hist.shape).astype(ind.hist.dtype)),
            heat=new.heat + (new.epot - e0),
        )

    def permute(self, src: ArrayLike) -> None:
        """Move the configuration of slot src[k] to slot k, re-evaluated at slot k's lambda."""
        src = np.asarray(src, int)
        changed = src != np.arange(self.n)
        if not changed.any():
            return
        if self.batched:
            old = self.S
            moved = self._exchange(old, _take(old, jnp.asarray(src)), jnp.ones(self.n))
            self.S = _select(changed, self._switch(moved), old)
        else:
            old = list(self.states)
            self.states = [
                self._switch_seq(self._exchange_seq(old[k], old[src[k]], 1.0)) if changed[k] else old[k]
                for k in range(self.n)
            ]

    # ------------------------------------------------------------------ outputs / checkpoints
    def write_restarts(self, prefix: str) -> None:
        """Write an Amber NetCDF restart of every window: prefix_Lkk.rst7."""
        for k in range(self.n):
            sim = self._on(k)
            write_restart(
                f"{prefix}_L{k:02d}.rst7",
                sim.positions() * 10.0,
                sim.velocities() * 10.0,
                np.asarray(sim.state.box) * 10.0,
                self.time_ps,
                title=f"pgm_jax lambda window {k}: {self.lambdas[k].tolist()}",
            )

    def state_dict(self) -> dict:
        """Return MDReplicas.state_dict with the lambdas."""
        d = super().state_dict()
        d["lambdas"] = self.lambdas.copy()
        return d

    def load_state_dict(self, d: dict) -> None:
        """Load window states from a checkpoint (MDReplicas.load_state_dict); ValueError if lambdas differ."""
        if (
            "lambdas" not in d
            or np.shape(d["lambdas"]) != self.lambdas.shape
            or not np.allclose(d["lambdas"], self.lambdas)
        ):
            raise ValueError("checkpoint lambda windows differ from these")
        super().load_state_dict(d)


# ----------------------------------------------------------------------------- driver
class FreeEnergyRun:
    """Lambda windows with samples for the estimators and optional Hamiltonian replica exchange.

        system, P = alchemical_system(system, 0)
        sim = Simulation(system, positions, box, settings, params=P, alchemy=Alchemy(system, 0),
                         thermostat="bussi", dt=0.002)
        fe = FreeEnergyRun(LambdaWindows(sim, standard_schedule()), sample_every=500, exchange_every=500)
        fe.run(1000000, prefix="wat", report_every=5000, checkpoint_every=50000)   # 2 ns per window
        free_energy.estimate(np.load("wat_fe.npz"), discard_ps=200)    # TI, BAR, MBAR

    Every `sample_every` steps: u[k, n] and dU/dlambda of every window (LambdaWindows.sample);
    every `exchange_every` steps (a multiple of sample_every; 0: no exchanges) neighbouring windows
    try to swap configurations, even and odd pairs alternately, on the sampled u.  Outputs:
    prefix_fe.npz (samples: u (S, K, K), dudl (S, K, 2), step, time_ps, replica (S, K), epot (S, K),
    lambdas, kT, meta), prefix_fe.log (one line per report: temperatures, CG iterations, acceptance,
    speed), prefix_fe.json (acceptance matrix, round trips, speed), prefix.fe.chk (checkpoint:
    windows, samples, statistics, random state; `load_checkpoint`) and prefix_Lkk.rst7.

    Attributes
    ----------
    windows : LambdaWindows
        The windows.
    n : int
        Number of windows K.
    samples : dict of str to list
        Per sample: "u", "dudl", "step", "time_ps", "replica", "epot", "cg" (and "dudp").
    stats : ExchangeStatistics
        Exchange statistics.
    step : int
        Steps done.
    """

    def __init__(
        self,
        windows: LambdaWindows,
        sample_every: int = 500,
        exchange_every: int = 0,
        seed: int = 0,
        log: TextIO | None = None,
        meta: dict | None = None,
        param_grad: ParameterGradients | None = None,
    ) -> None:
        """Set up sampling (and exchanges) over the lambda windows.

        Parameters
        ----------
        windows : LambdaWindows
            The windows.
        sample_every : int
            Steps between samples of the reduced energies and dU/dlambda.
        exchange_every : int
            Steps between exchange attempts (a multiple of sample_every; 0: none).
        seed : int
            Seed of the exchange random numbers.
        log : text stream or None
            Receives the rows of the log table of `run` too (the setup is logged to the logger
            "pgm_jax.md.alchemy").
        meta : dict, optional
            JSON-serializable metadata stored with the samples.
        param_grad : ParameterGradients, optional
            Also sample parameter gradients of the end states (md/fe_grad.py).

        Raises
        ------
        ValueError
            Invalid intervals, or param_grad of other windows.
        """
        self.windows = windows
        self.param_grad = param_grad
        self.n = windows.n
        self.sample_every, self.exchange_every = int(sample_every), int(exchange_every)
        if self.sample_every < 1:
            raise ValueError("sample_every must be >= 1")
        if self.exchange_every < 0 or (self.exchange_every and self.exchange_every % self.sample_every):
            raise ValueError(
                "exchange_every must be 0 or a multiple of sample_every (exchanges use the sampled energies)"
            )
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), 0xA1C4])))
        self.stats = ExchangeStatistics(self.n)
        self.step = 0
        self.log = log
        self.meta = dict(meta or {})
        self.samples = {k: [] for k in ("u", "dudl", "step", "time_ps", "replica", "epot", "cg")}
        if param_grad is not None:  # parameter gradients of the end states (fe_grad.py)
            if param_grad.windows is not windows:
                raise ValueError("param_grad was built for other windows")
            self.samples["dudp"] = []
            self.meta.update(param_grad.meta())
        mode = "batched" if windows.batched else "sequential"
        logger.info(
            f"lambda windows: {self.n} ({mode}), T = {windows.temperatures[0]:.2f} K, samples every "
            f"{self.sample_every} steps ({self.sample_every * windows.dt:g} ps), "
            + (f"Hamiltonian exchange every {self.exchange_every} steps" if self.exchange_every else "no exchanges")
        )
        logger.info("(lambda_elec, lambda_vdw): " + " ".join(f"({a:g},{b:g})" for a, b in windows.lambdas))

    # ------------------------------------------------------------------ one sample / exchange
    def _sample(self) -> np.ndarray:
        """Take one sample of every window (and the parameter gradients) and return its u (K, K)."""
        w = self.windows
        u, g, it = w.sample()
        self.samples["u"].append(u)
        self.samples["dudl"].append(g)
        self.samples["step"].append(self.step)
        self.samples["time_ps"].append(w.time_ps)
        self.samples["replica"].append(self.stats.replica.copy())
        self.samples["epot"].append(w.potentials())
        self.samples["cg"].append(it)
        if self.param_grad is not None:
            self.samples["dudp"].append(self.param_grad.sample())
        return u

    def _exchange(self, u: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
        """Attempt one Hamiltonian exchange round on the sampled u; return (pairs, accepted)."""
        pairs = exchange_pairs(self.n, self.stats.n_exchanges)
        acc, src = metropolis(u, pairs, self.rng.random(len(pairs)))
        self.windows.permute(src)
        self.stats.record(pairs, acc, src)
        return pairs, acc

    # ------------------------------------------------------------------ running
    def run(self, nsteps: int, *, prefix: str | None = "fe", report_every: int = 0, checkpoint_every: int = 0) -> dict:
        """Advance every window nsteps with samples every sample_every steps (and exchanges).

        Parameters
        ----------
        nsteps : int
            Steps.
        prefix : str or None
            Path prefix of the files (None: no files).
        report_every : int
            Steps between rows of the log table prefix_fe.log (mean, minimum and maximum
            temperature of the windows [K], mean CG iterations per step and of the last sample,
            neighbour acceptance, ns/day per window); 0: none.
        checkpoint_every : int
            Steps between checkpoints prefix.fe.chk with the samples prefix_fe.npz and restarts
            prefix_Lkk.rst7 (0: only at the end).

        Returns
        -------
        dict
            `summary` of the run.
        """
        report, restart = report_every, checkpoint_every
        w = self.windows
        block = block_length(nsteps, self.sample_every, report, restart)
        files = prefix is not None
        table = None
        if report:
            title = ["lambda (elec, vdw): " + " ".join(f"({a:g},{b:g})" for a, b in w.lambdas)]
            path = f"{prefix}_fe.log" if files else None
            table = LogTable(path, append=bool(self.step), title=title, echo=self.log)
        clock = Stopwatch(self.step, w.dt)
        done = 0
        while done < nsteps:
            m = min(block, nsteps - done)
            w.advance(m)
            done += m
            self.step += m
            if self.step % self.sample_every == 0:
                u = self._sample()
                if self.exchange_every and self.step % self.exchange_every == 0:
                    self._exchange(u)
            if table is not None and self.step % report == 0:
                obs = [w.observables(k) for k in range(self.n)]
                T = np.array([o["temp_K"] for o in obs])
                row = {
                    "step": self.step,
                    "time_ps": w.time_ps,
                    "T_mean_K": float(T.mean()),
                    "T_min_K": float(T.min()),
                    "T_max_K": float(T.max()),
                    "cg_mean": float(np.mean([o["cg_mean"] for o in obs])),
                    "cg_sample": int(self.samples["cg"][-1]) if self.samples["cg"] else 0,
                }
                if self.exchange_every:
                    acc = self.stats.neighbour_acceptance()
                    row.update({f"acc_{k}_{k + 1}": float(a) for k, a in enumerate(acc)})
                row["ns_per_day"] = clock.ns_per_day(self.step)
                table.write(row)
            if files and restart and self.step % restart == 0:
                self._write_checkpoint_files(prefix)
        speed = clock.ns_per_day(self.step)
        if table is not None:
            table.close()
        summary = self.summary(ns_per_day=speed)
        if files:
            self._write_checkpoint_files(prefix)
            with open(f"{prefix}_fe.json", "w") as fh:
                json.dump(summary, fh, indent=1)
        return summary

    def summary(self, ns_per_day: float | None = None) -> dict:
        """Return the windows, temperature, counts, acceptance, round trips and (optionally) speed.

        Parameters
        ----------
        ns_per_day : float, optional
            Speed per window [ns/day] to include (with the aggregate speed).

        Returns
        -------
        dict
            JSON-serializable summary (the content of prefix_fe.json).
        """
        st = self.stats
        out = {
            "lambdas": self.windows.lambdas.tolist(),
            "temperature_K": float(self.windows.temperatures[0]),
            "steps": self.step,
            "time_ps": self.windows.time_ps,
            "samples": len(self.samples["u"]),
            "sample_every": self.sample_every,
            "exchange_every": self.exchange_every,
            "exchanges": st.n_exchanges,
            "batched": bool(self.windows.batched),
        }
        if self.exchange_every:
            out["neighbour_acceptance"] = [None if np.isnan(a) else float(a) for a in st.neighbour_acceptance()]
            out["round_trips_total"] = int(st.round_trips.sum())
        if ns_per_day is not None:
            out["ns_per_day_per_window"] = ns_per_day
            out["ns_per_day_aggregate"] = ns_per_day * self.n
        return out

    # ------------------------------------------------------------------ outputs / checkpoints
    def arrays(self) -> dict[str, Any]:
        """Return the samples as arrays (the content of prefix_fe.npz; class docstring).

        With parameter gradients also "dudp" (S, T, K, M): dU_{targets[t]}/dP at the
        configuration of window k (md/fe_grad.py).
        """
        S, w = self.samples, self.windows
        K = self.n
        return {
            "u": np.array(S["u"], float).reshape(-1, K, K),
            "dudl": np.array(S["dudl"], float).reshape(-1, K, 2),
            "step": np.array(S["step"], int),
            "time_ps": np.array(S["time_ps"], float),
            "replica": np.array(S["replica"], int).reshape(-1, K),
            "epot": np.array(S["epot"], float).reshape(-1, K),
            "cg": np.array(S["cg"], int),
            "lambdas": w.lambdas.copy(),
            "kT": float(w.integ.kT),
            "temperature": float(w.temperatures[0]),
            "dt": w.dt,
            "sample_every": self.sample_every,
            "meta": json.dumps(self.meta),
        } | (
            {"dudp": np.array(S["dudp"], float).reshape(len(S["u"]), -1, K, len(self.meta["dudp_names"]))}
            if S.get("dudp") is not None and "dudp_names" in self.meta
            else {}
        )

    def _write_checkpoint_files(self, prefix: str) -> None:
        """Write the files of run's checkpoints: prefix_fe.npz, prefix.fe.chk and prefix_Lkk.rst7."""
        self.save_samples(f"{prefix}_fe.npz")
        self.save_checkpoint(prefix + ".fe.chk")
        self.windows.write_restarts(prefix)

    def save_samples(self, path: str) -> None:
        """Write the samples (`arrays`) to an .npz file for free_energy.estimate / fe.load."""
        np.savez(path, **self.arrays())

    def save_checkpoint(self, path: str) -> None:
        """Write a checkpoint of the run (driver.write_checkpoint, kind "free-energy").

        Parameters
        ----------
        path : str
            The file (prefix.fe.chk in `run`).

        Notes
        -----
        The checkpoint holds every window state, the samples, the exchange statistics and random
        state, the step and the metadata; `load_checkpoint` continues the run bitwise on the CPU.
        """
        content = {
            "lambdas": self.windows.lambdas,
            "step": self.step,
            "rng": self.rng.bit_generator.state,
            "stats": self.stats.to_dict(),
            "samples": {k: np.asarray(v) for k, v in self.samples.items()},
            "meta": self.meta,
            "windows": self.windows.state_dict(),
        }
        write_checkpoint(path, "free-energy", content)

    def _read(self, path: str) -> dict:
        """Return the content of a checkpoint (current or legacy .fe.chk), samples as lists per sample."""
        d = read_checkpoint(
            path, "free-energy", self.windows.state_template(), OPTIONAL_STATE, legacy_format=LEGACY_FORMAT
        )
        d["samples"] = {k: list(v) for k, v in d["samples"].items()}
        return d

    def load_checkpoint(self, path: str) -> None:
        """Continue from a checkpoint written by `save_checkpoint` (or a legacy pickle .fe.chk).

        Legacy pickle checkpoints are those of pgm_jax up to commit e72c57c; the system, settings
        and windows must be the same.

        Parameters
        ----------
        path : str
            The checkpoint file.

        Raises
        ------
        ValueError
            Another kind of checkpoint, other windows, or samples without the parameter gradients
            this run collects.
        """
        d = self._read(path)
        if self.param_grad is not None and "dudp" not in d["samples"] and d["samples"]["u"]:
            raise ValueError(
                f"{path}: its samples have no parameter gradients (continue without param_grad, "
                "or start new samples with load_windows)"
            )
        self.windows.load_state_dict(d["windows"])
        self.step = int(d["step"])
        self.rng.bit_generator.state = d["rng"]
        self.stats = ExchangeStatistics.from_dict(d["stats"])
        self.samples = d["samples"]
        if self.param_grad is not None:
            self.samples.setdefault("dudp", [])
        self.meta.update(d.get("meta", {}))
        if self.param_grad is not None:
            self.meta.update(self.param_grad.meta())

    def load_windows(self, path: str) -> None:
        """Start from the window configurations of a checkpoint with no samples, step 0 and time 0.

        The checkpoint (written by `save_checkpoint`) may e.g. be equilibrated at other
        parameters.  Its samples, statistics and random state are not taken over.  Windows are matched by
        lambda when the checkpoint has other windows.

        Parameters
        ----------
        path : str
            The checkpoint file (current or legacy format).

        Raises
        ------
        ValueError
            Another kind of checkpoint, or no window at one of this run's lambdas.
        """
        d = self._read(path)
        wd = dict(d["windows"])
        L, mine = np.asarray(wd["lambdas"], float), self.windows.lambdas
        if L.shape != mine.shape or not np.allclose(L, mine):  # windows matched by lambda
            pick = []
            for lam in mine:
                j = np.nonzero(np.all(np.abs(L - lam) < 1e-12, axis=1))[0]
                if not len(j):
                    raise ValueError(f"{path}: no window at lambda {lam.tolist()}")
                pick.append(int(j[0]))
            wd.update(
                lambdas=mine.copy(),
                temperatures=np.asarray(wd["temperatures"])[pick],
                states=[wd["states"][j] for j in pick],
            )
        self.windows.load_state_dict(wd)
        self.windows.time_ps = 0.0
        self.step = 0
        self.stats = ExchangeStatistics(self.n)
        self.samples = {k: [] for k in self.samples}
