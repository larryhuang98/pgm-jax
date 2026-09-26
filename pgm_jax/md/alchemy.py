"""Alchemical free energies with pGM: lambda-dependent Hamiltonians for one solute molecule,
lambda windows batched on the GPU, Hamiltonian replica exchange, samples for TI / BAR / MBAR
(estimators in free_energy.py), and the gas-phase leg of the solvation cycle.

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
  intramolecular terms included; Ren & Ponder, JCC 23, 1497 (2002) and JPC B 107, 5933 (2003);
  Shi, Ren, Schnieders & Ponder, Methods Mol. Biol. 819, 215 (2012)):
      q_s     -> lambda_e q_s               (Gaussian charges)
      c_s     -> lambda_e c_s               (covalent-dipole strengths, i.e. permanent dipoles p_s)
      alpha_s -> alpha_s [eps + (1 - eps) lambda_e],  eps = alpha_floor (1e-8)
  Every electrostatic interaction of the solute is scaled: with the environment, with its own
  periodic images and inside the molecule (pGM couples every pair, and the solute's own induction).
  At lambda_e = 0 the solute carries no electrostatics; the gas-phase leg removes the same
  intramolecular energy in vacuum (GasPhaseLeg: exact from one configuration for a rigid solute).
  The polarizability floor: the induced dipoles minimise |mu|^2 / (2 alpha) - mu.E + mu T mu / 2;
  the Jacobi-preconditioned operator of the CG is I - alpha^1/2 T alpha^1/2, whose coupling only
  shrinks with alpha_s, so the solve stays well posed as alpha_s -> 0, but alpha_s = 0 would put
  1/0 into the operator and the polarization energy.  Tinker masks the solute's induced dipoles at
  lambda = 0 (douind = .false.); here alpha_s(0) = eps alpha_s instead: U(lambda) is smooth up to
  the endpoint, dU/dlambda_e at lambda_e = 0 is the exact limit (-(1 - eps) alpha_s |E|^2 / 2 per
  site plus the permanent terms, finite), and the endpoint differs from masked dipoles by eps
  times the solute's induction energy (~1e-7 kJ/mol).  The Gaussian radii are not scaled (they
  are damping widths, not interaction strengths).

  lambda_vdw (decoupling of the solute-environment van der Waals, Beutler soft core; Beutler et
  al., CPL 222, 529 (1994); alpha 0.5 and power 1 as in Shirts & Pande, JCP 122, 134508 (2005)):
      U_ij = lambda_v eps_ij [1 / w^2 - 2 / w],   w = (r / rmin_ij)^6 + (sc_alpha / 2)(1 - lambda_v)
  i.e. 4 eps lambda_v [1/y^2 - 1/y] with y = sc_alpha (1 - lambda_v) + (r / sigma)^6, sigma^6 =
  rmin^6 / 2, in Amber's rmin form: Lennard-Jones at lambda_v = 1, finite energy and force at r = 0
  for lambda_v < 1 (the force vanishes there: w depends on r^6).  Only solute-environment pairs
  are soft (intramolecular van der Waals of the solute, zero for rigid molecules, is not touched);
  the long-range correction of those pairs is scaled by lambda_v (beyond the cutoff the soft core
  differs from lambda_v x LJ by (sc_alpha / 2)(1 - lambda_v)(rmin / rc)^6 < 1e-3 of the term).
  Van der Waals forms: "lj" and "none"; GVDW (finite at overlap, would need no soft core) is
  refused for now.

Hamiltonian at lambda = (lambda_e, lambda_v): the ordinary force field (forcefield.py) with the
solute's parameters at lambda_e and its van der Waals parameters set to zero, so the ordinary pair
rows never see a solute pair's van der Waals, plus the soft-core solute-environment term evaluated
here in a dedicated small row set (each solute atom's candidates from the neighbour list, n_solute
x C pairs, forces by autodiff).  No pair kernel of the engine changes, and without an alchemical
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
Per-configuration constants (restraint energies, P V under NPT) cancel in every estimator and in
the exchange criterion and are left out of u.

Hamiltonian replica exchange (FreeEnergyRun, exchange_every > 0): neighbouring windows try to swap
configurations with the Metropolis test of remd.metropolis on the sampled u (no extra energy
evaluation); an accepted swap re-evaluates forces and dipoles in the new Hamiltonian, books the
energy change as heat and restarts the dipole predictor from the new dipoles.

Limits: one solute molecule, rigid (the rigid-molecule engine, Simulation); a net-charged solute
would need finite-size corrections (Rocklin et al., JCP 139, 184103 (2013)) and is refused;
the cell dipole recorder does not scale the solute's charges (refused with an alchemical region).

Units: nm, ps, kJ/mol, K, e."""
from __future__ import annotations

import dataclasses as _dc
import json
import math
import pickle
import sys as _sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..lj import lj_long_range
from ..system import ATOM_QUANTITIES, QUANTITIES, System
from .box import min_image, volume
from .integrate import KB
from .io import write_restart
from .remd import ExchangeStatistics, MDReplicas, _nocount, _stack, _take, exchange_pairs, metropolis

PREFIX = "alch:"                 # tying-key prefix of an alchemical molecule's own parameters
KCAL = 4.184                     # kJ per kcal
FORMAT = "pgm_jax free energy 1"


# ----------------------------------------------------------------------------- setup
def alchemical_system(sys: System, solute: int, params=None):
    """(System, params): `sys` with molecule `solute` replaced by a copy whose tied parameters are its
    own (every tying key prefixed 'alch:'), so lambda can scale them without touching identical
    molecules of the environment.  `params` (a pytree of sys.table, or None for its initial values)
    is mapped to the new table by key (the solute copy takes the values of the keys it came from)."""
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
    P = {qn: jnp.asarray(np.array([vals[qn][key[len(PREFIX):] if key.startswith(PREFIX) else key]
                                   for key in out.table.keys[qn]], float)) for qn in QUANTITIES}
    return out, P


def standard_schedule(n_elec: int = 8, vdw=None) -> np.ndarray:
    """(K, 2) windows (lambda_elec, lambda_vdw): electrostatics 1 -> 0 in n_elec evenly spaced
    windows at lambda_vdw = 1, then van der Waals 1 -> 0 at lambda_elec = 0 (default points denser
    towards 0, where the soft-core integrand bends: 0.9, 0.8, ..., 0.1, 0.05, 0).  The window
    (0, 1) ends the first stage and starts the second."""
    if int(n_elec) < 2:
        raise ValueError("n_elec >= 2 (both ends of the electrostatics stage)")
    lv = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0]) if vdw is None else np.asarray(vdw, float)
    if lv.ndim != 1 or np.any(np.diff(lv) >= 0) or lv[0] >= 1.0 or lv[-1] != 0.0:
        raise ValueError("vdw: decreasing lambda_vdw values below 1, ending at 0")
    le = np.linspace(1.0, 0.0, int(n_elec))
    return np.concatenate([np.stack([le, np.ones_like(le)], 1), np.stack([np.zeros_like(lv), lv], 1)])


# ----------------------------------------------------------------------------- Hamiltonian
class Alchemy:
    """lambda-dependent Hamiltonian of one solute molecule (module docstring for the physics).

        sys, P = alchemical_system(sys, solute=0)
        alch = Alchemy(sys, solute=0)
        sim = Simulation(sys, pos, H, settings, params=P, alchemy=alch, ensemble="nvt", thermostat="bussi")

    `lam` is the coupling (lambda_elec, lambda_vdw) of states whose MDState.lam is None (the
    default single simulation); LambdaWindows gives each window its own.  The cutoff, long-range
    correction and van der Waals form are taken from the force field the first time the
    Hamiltonian is bound to one (`check`, called by the integrator)."""

    def __init__(self, sys: System, solute: int, lam=(1.0, 1.0), sc_alpha: float = 0.5, alpha_floor: float = 1e-8):
        k = int(solute)
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
                raise ValueError(f"the solute shares its {qn} parameters ({', '.join(sys.table.keys[qn][i] for i in shared[:4])}) "
                                 f"with other molecules: build the system with alchemical_system(sys, {k})")
            m = np.zeros(len(sys.table.keys[qn]), bool)
            m[own] = True
            self.mask[qn] = jnp.asarray(m)
        terms = np.isin(sys.cov_i, atoms)
        own, other = np.unique(sys.idx["cov"][terms]), sys.idx["cov"][~terms]
        if np.intersect1d(own, other).size:
            raise ValueError(f"the solute shares covalent-dipole parameters with other molecules: build the system "
                             f"with alchemical_system(sys, {k})")
        m = np.zeros(len(sys.table.keys["cov"]), bool)
        m[own] = True
        self.mask["cov"] = jnp.asarray(m)
        charge = float(np.sum(np.asarray(sys.expand(None)["q"])[atoms]))
        if abs(charge) > 1e-6:
            raise NotImplementedError(f"the solute carries a net charge ({charge:+.4f} e): charged solutes need "
                                      "finite-size corrections of the periodic electrostatics, not implemented")
        self.bound = None                          # (vdw form, van der Waals cutoff, long-range correction)

    # ------------------------------------------------------------------ binding / checks
    @staticmethod
    def _check_lam(lam) -> jnp.ndarray:
        a = np.asarray(lam, float)
        if a.shape != (2,) or np.any(a < 0.0) or np.any(a > 1.0):
            raise ValueError(f"lambda = (lambda_elec, lambda_vdw) in [0, 1], got {lam}")
        return jnp.asarray(a, jnp.float64)

    def check(self, ff):
        """Bind to a force field (cutoff, long-range correction, van der Waals form); refuse what the
        Hamiltonian does not implement."""
        s = ff.s
        if ff.sys.fingerprint() != self.sys.fingerprint():
            raise ValueError("the alchemical region was built for another System")
        if s.vdw not in ("lj", "none"):
            raise NotImplementedError(f"soft-core van der Waals is implemented for vdw='lj' (and 'none'), not {s.vdw!r}")
        b = (s.vdw, float(ff.rc_v), bool(s.lj_lrc))
        if self.bound is not None and self.bound != b:
            raise ValueError(f"the alchemical region is bound to other settings {self.bound}, not {b}")
        self.bound = b

    @property
    def vdw(self) -> str:
        if self.bound is None:
            raise RuntimeError("Alchemy is not bound to a force field yet (Alchemy.check(ff))")
        return self.bound[0]

    def describe(self) -> str:
        m = self.sys.molecules[self.solute]
        return (f"solute molecule {self.solute} ({m.name}, {m.n} atoms), lambda (elec, vdw) = "
                f"({float(self.lam[0]):g}, {float(self.lam[1]):g}), soft core alpha {self.sc_alpha:g}, "
                f"polarizability floor {self.alpha_floor:g}")

    def _lam(self, lam):
        return self.lam if lam is None else jnp.asarray(lam, jnp.float64)

    # ------------------------------------------------------------------ parameters at lambda
    def params(self, params, lam):
        """Parameter table (pytree of sys.table) of the ordinary force field at lambda: the solute's
        charges and covalent dipoles times lambda_e, polarizabilities times eps + (1 - eps) lambda_e,
        van der Waals parameters zero (its van der Waals with the environment is softcore_energy)."""
        P = dict(self.params0 if params is None else params)
        le = self._lam(lam)[0]
        a = self.alpha_floor + (1.0 - self.alpha_floor) * le
        P["q"] = jnp.where(self.mask["q"], le * P["q"], P["q"])
        P["cov"] = jnp.where(self.mask["cov"], le * P["cov"], P["cov"])
        P["alpha"] = jnp.where(self.mask["alpha"], a * P["alpha"], P["alpha"])
        for qn in ("lj_sqrt_eps", "gvdw_sqrt_a", "gvdw_sqrt_c6"):
            P[qn] = jnp.where(self.mask[qn], 0.0, P[qn])
        return P

    def _tail(self, P, H):
        """The solute's share of the r^-6 long-range correction (kJ/mol): the full tail minus the tail
        with the solute's LJ switched off (P per atom, unscaled)."""
        rc = self.bound[1]
        off = dict(P, lj_sqrt_eps=jnp.where(self.is_solute, 0.0, P["lj_sqrt_eps"]))
        V = volume(H)
        return lj_long_range(P, V, rc) - lj_long_range(off, V, rc)

    def softcore_energy(self, pos, H, cand, params, lam_v):
        """Soft-core van der Waals of the solute with its environment (kJ/mol, float64): pairs of each
        solute atom's candidate row (neighbour list, padding N) with environment atoms inside the
        van der Waals cutoff, plus lambda_v times the solute's long-range correction."""
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
        s = jnp.where(on, r2, 1.0) / jnp.where(on, rmin * rmin, 1.0)
        w = s * s * s + 0.5 * self.sc_alpha * (1.0 - lam_v)
        e = jnp.sum(jnp.where(on, lam_v * eps * (1.0 / (w * w) - 2.0 / w), 0.0))
        if self.bound[2]:
            e = e + lam_v * self._tail(P, H)
        return e

    # ------------------------------------------------------------------ engine interface
    def compute(self, ff, pos, H, cand, ind, params, lam):
        """forcefield.Result of the Hamiltonian at lam (None: self.lam): the ordinary force field at
        the scaled parameters plus the soft-core term (energy added to 'vdw' and 'total')."""
        lam = self._lam(lam)
        res = ff.compute(pos, H, cand, ind, self.params(params, lam))
        if self.vdw == "none":
            return res
        e, g = jax.value_and_grad(self.softcore_energy)(pos, H, cand, params, lam[1])
        energy = dict(res.energy, vdw=res.energy["vdw"] + e, total=res.energy["total"] + e)
        return res._replace(energy=energy, forces=res.forces - g)

    def energy(self, ff, pos, H, cand, ind, params, lam):
        """(energy, InductionState, CG iterations, overflow) as PGMForceField.energy, at lam."""
        lam = self._lam(lam)
        e, ind, it, ovf = ff.energy(pos, H, cand, ind, self.params(params, lam))
        return e + self.softcore_energy(pos, H, cand, params, lam[1]), ind, it, ovf

    def energy_fixed_mu(self, ff, pos, H, cand, mu, params, lam):
        """Total energy (kJ/mol) at lam with the induced dipoles held at mu; differentiable in lam."""
        lam = self._lam(lam)
        e, _ = ff.energy_fixed_mu(pos, H, mu, cand, ff._atoms(self.params(params, lam)))
        return e + self.softcore_energy(pos, H, cand, params, lam[1])

    def dudl(self, ff, pos, H, cand, mu, params, lam):
        """(dU/dlambda_elec, dU/dlambda_vdw) in kJ/mol at the converged dipoles mu (Hellmann-Feynman:
        the energy is stationary in mu)."""
        return jax.grad(lambda l: self.energy_fixed_mu(ff, pos, H, cand, mu, params, l))(self._lam(lam))

    def strain_derivative(self, ff, pos, H, cand, mu, params, lam, molecular: bool = True):
        """dE/d eps (3, 3) at fixed mu (as PGMForceField.strain_derivative) of the Hamiltonian at lam,
        with the tail impulse term of the soft-core long-range correction."""
        lam = self._lam(lam)
        W = ff.strain_derivative(pos, H, cand, mu, self.params(params, lam), molecular)
        if self.vdw == "none":
            return W
        if molecular:
            w = ff.masses
            com = jax.ops.segment_sum(w[:, None] * pos, ff.mol, self.sys.nmol) / \
                jax.ops.segment_sum(w, ff.mol, self.sys.nmol)[:, None]

        def e(eps):
            F = jnp.eye(3) + eps
            x = pos + ((com @ eps.T)[ff.mol] if molecular else pos @ eps.T)
            return self.softcore_energy(x, H @ F.T, cand, params, lam[1])

        W = W + jax.grad(e)(jnp.zeros((3, 3)))
        if self.bound[2]:
            P = self.sys.expand(self.params0 if params is None else params)
            W = W - lam[1] * self._tail(P, H) * jnp.eye(3)
        return W


# ----------------------------------------------------------------------------- gas-phase leg
class GasPhaseLeg:
    """The solute alone in vacuum with the alchemical parameters at lambda_elec: the gas-phase leg
    of the cycle.  E_gas(lambda_e) is the pGM energy of the isolated molecule (every pair, induced
    dipoles by a dense solve: channels.ElecChannel, the kernels and Coulomb constant of the MD
    engine) at the solute's rigid geometry `xyz` (nm).  For a rigid solute the gas-phase energy
    does not depend on the configuration (only on orientation and position, which it is invariant
    to), so the free energy of annihilating its electrostatics is exact from one configuration:
        Delta G_gas(1 -> 0) = E_gas(0) - E_gas(1)      (-kT ln <exp(-beta dU)> of a constant dU).
    dudl(lambda_e) = dE_gas/dlambda_e is what TI subtracts from the solution-phase integrand
    (free_energy.estimate): the large intramolecular part of pGM's electrostatics then drops out
    before the quadrature."""

    def __init__(self, alchemy: Alchemy, xyz, elec: str = "qpi"):
        from ..channels import ElecChannel
        from ..model import Model
        self.alchemy = alchemy
        sub, _ = alchemy.sys.sub((alchemy.solute,))
        self.xyz = jnp.asarray(np.asarray(xyz, float).reshape(sub.n, 3))
        f = Model([ElecChannel.level(elec)]).energy_fn(sub)
        self._e = jax.jit(lambda le, params: f(self.xyz, alchemy.params(params, jnp.stack([le, 1.0])))["total"])
        self._g = jax.jit(jax.grad(lambda le, params: f(self.xyz, alchemy.params(params, jnp.stack([le, 1.0])))["total"]))

    def energy(self, lam_e: float, params=None) -> float:
        """E_gas(lambda_e), kJ/mol."""
        return float(self._e(jnp.asarray(float(lam_e)), params))

    def dudl(self, lam_e: float, params=None) -> float:
        """dE_gas/dlambda_e, kJ/mol."""
        return float(self._g(jnp.asarray(float(lam_e)), params))

    def delta_g(self, params=None) -> float:
        """Delta G_gas(lambda_e 1 -> 0) = E_gas(0) - E_gas(1), kJ/mol (rigid solute: exact)."""
        return self.energy(0.0, params) - self.energy(1.0, params)


# ----------------------------------------------------------------------------- lambda windows
def _select(mask, a, b):
    """Stacked states: slot k from a where mask[k], else from b (the unbatched step counter from a)."""
    m = jnp.asarray(mask)
    A, B = _nocount(a), _nocount(b)
    out = jax.tree_util.tree_map(lambda x, y: jnp.where(m.reshape((-1,) + (1,) * (jnp.ndim(x) - 1)), x, y), A, B)
    return out.set(induction=out.induction.set(count=a.induction.count))


class LambdaWindows(MDReplicas):
    """The lambda windows of one Simulation with an alchemical region, at one temperature: the
    engine of FreeEnergyRun.  `lambdas` (K, 2) are the (lambda_elec, lambda_vdw) of the windows
    (`standard_schedule`).  Every window starts from the current configuration of `sim` with
    momenta drawn from its own random stream (`seed`) and forces at its own lambda.

    batched=True (NVT): the windows are one stacked state advanced by jax.vmap of the step (lambda
    traced in MDState.lam), one program for all windows; batched=False advances them one after the
    other through the driver of `sim` (NPT, or systems that fill the GPU alone).  Resizing, the
    shared static sizes and checkpoints are those of remd.MDReplicas."""

    def __init__(self, sim, lambdas, batched: bool = True, seed: int = 0):
        integ = sim.integ
        if getattr(integ, "alchemy", None) is None:
            raise ValueError("the simulation has no alchemical region: Simulation(..., alchemy=Alchemy(...))")
        if integ.thermostat is None:
            raise ValueError("lambda windows need a thermostat (ensemble nvt or npt)")
        if batched and sim.ensemble == "npt":
            raise ValueError("batched windows run NVT only (under vmap the barostat's trial energy would be "
                             "evaluated every step); use batched=False for NPT")
        L = np.asarray(lambdas, float)
        if L.ndim != 2 or L.shape[1] != 2 or len(L) < 2 or np.any(L < 0.0) or np.any(L > 1.0):
            raise ValueError("lambdas: (K >= 2, 2) array of (lambda_elec, lambda_vdw) in [0, 1]")
        if len({tuple(r) for r in L.tolist()}) != len(L):
            raise ValueError("lambda windows must be distinct")
        self.sim, self.integ, self.batched = sim, integ, bool(batched)
        self.alchemy = integ.alchemy
        self.lambdas = L
        self.temperatures = np.full(len(L), integ.kT / KB)          # one temperature (MDReplicas interface)
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

    def _build(self):
        super()._build()
        from .remd import _axes
        ax = _axes(self.S)
        self._switch = jax.jit(jax.vmap(self._switch_one, in_axes=(ax,), out_axes=ax))

    # ------------------------------------------------------------------ samples
    def _sample_one(self, st, lam_e, group, lam_v):
        """For one configuration: U_k(x) (K,) up to a per-configuration constant, dU/dlambda at its own
        lambda (2,), the largest CG iteration count, overflow."""
        integ, ff, alch = self.integ, self.sim.ff, self.alchemy
        params = integ.params
        body = st.dyn.position
        pos = self.sim.rigid.positions(body)
        cand, ovf0 = integ.nb.candidates(st.nbr, body.center, st.box, pos)

        def e_ff(le):
            e, _, it, ovf = ff.energy(pos, st.box, cand, st.induction, alch.params(params, jnp.stack([le, 1.0])))
            return e, it, ovf

        E, it, ovf = jax.lax.map(e_ff, lam_e)
        esc = jax.vmap(lambda lv: alch.softcore_energy(pos, st.box, cand, params, lv))(lam_v)
        g = alch.dudl(ff, pos, st.box, cand, st.induction.mu, params, st.lam)
        return E[group] + esc, g, jnp.max(it), jnp.any(ovf) | ovf0

    def _sampler(self):
        key = self._sizes()
        if key not in self._samplers:
            if self.batched:
                from .remd import _axes
                f = jax.vmap(self._sample_one, in_axes=(_axes(self.S), None, None, None))
            else:
                f = self._sample_one
            self._samplers = {key: jax.jit(f)}
        return self._samplers[key]

    def sample(self):
        """u (K, K): u[k, n] = beta U_k(x_n), the configuration of window n in the Hamiltonian of window
        k (per-configuration constants dropped); dudl (K, 2): dU/dlambda (kJ/mol) of each window at its
        own lambda; the largest CG iteration count of the re-solves."""
        f = self._sampler()
        args = (jnp.asarray(self.lam_e), jnp.asarray(self.group), jnp.asarray(self.lambdas[:, 1]))
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
    def _switch_one(self, st):
        """After a swap: forces, energies and dipoles in the slot's Hamiltonian; the energy change is
        booked as heat; the dipole predictor restarts from the new dipoles (its history was recorded
        in another Hamiltonian)."""
        e0 = st.epot
        new = self.integ._state_forces(st, True)
        ind = new.induction
        return new.set(induction=ind.set(hist=jnp.broadcast_to(ind.mu, ind.hist.shape).astype(ind.hist.dtype)),
                       heat=new.heat + (new.epot - e0))

    def permute(self, src):
        """Slot k receives the configuration of slot src[k], re-evaluated at slot k's lambda."""
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
            self.states = [self._switch_seq(self._exchange_seq(old[k], old[src[k]], 1.0)) if changed[k] else old[k]
                           for k in range(self.n)]

    # ------------------------------------------------------------------ outputs / checkpoints
    def write_restarts(self, prefix: str):
        """Amber NetCDF restart of every window: prefix_Lkk.rst7."""
        for k in range(self.n):
            sim = self._on(k)
            write_restart(f"{prefix}_L{k:02d}.rst7", sim.positions_nm() * 10.0, sim.velocities_nm_ps() * 10.0,
                          np.asarray(sim.state.box) * 10.0, self.time_ps,
                          title=f"pgm_jax lambda window {k}: {self.lambdas[k].tolist()}")

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["lambdas"] = self.lambdas.copy()
        return d

    def load_state_dict(self, d: dict):
        if "lambdas" not in d or np.shape(d["lambdas"]) != self.lambdas.shape or not np.allclose(d["lambdas"], self.lambdas):
            raise ValueError("checkpoint lambda windows differ from these")
        super().load_state_dict(d)


# ----------------------------------------------------------------------------- driver
class FreeEnergyRun:
    """Lambda windows with samples for the estimators and optional Hamiltonian replica exchange.

        sys, P = alchemical_system(sys, 0)
        sim = Simulation(sys, pos, H, settings, params=P, alchemy=Alchemy(sys, 0), ensemble="nvt",
                         thermostat="bussi", dt=0.002)
        fe = FreeEnergyRun(LambdaWindows(sim, standard_schedule()), sample_every=500, exchange_every=500)
        fe.run(1000000, prefix="wat", report=5000, restart=50000)       # 2 ns per window
        free_energy.estimate(np.load("wat_fe.npz"), discard_ps=200)    # TI, BAR, MBAR

    Every `sample_every` steps: u[k, n] and dU/dlambda of every window (LambdaWindows.sample);
    every `exchange_every` steps (a multiple of sample_every; 0: no exchanges) neighbouring windows
    try to swap configurations, even and odd pairs alternately, on the sampled u.  Outputs:
    prefix_fe.npz (samples: u (S, K, K), dudl (S, K, 2), step, time_ps, replica (S, K), epot (S, K),
    lambdas, kT, meta), prefix_fe.log (one line per report: temperatures, CG iterations, acceptance,
    speed), prefix_fe.json (acceptance matrix, round trips, speed), prefix.fe.chk (checkpoint:
    windows, samples, statistics, random state; `load`) and prefix_Lkk.rst7."""

    def __init__(self, windows: LambdaWindows, sample_every: int = 500, exchange_every: int = 0, seed: int = 0,
                 log=_sys.stdout, meta: dict | None = None):
        self.windows = windows
        self.n = windows.n
        self.sample_every, self.exchange_every = int(sample_every), int(exchange_every)
        if self.sample_every < 1:
            raise ValueError("sample_every must be >= 1")
        if self.exchange_every < 0 or (self.exchange_every and self.exchange_every % self.sample_every):
            raise ValueError("exchange_every must be 0 or a multiple of sample_every (exchanges use the sampled energies)")
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), 0xA1C4])))
        self.stats = ExchangeStatistics(self.n)
        self.step = 0
        self.log = log
        self.meta = dict(meta or {})
        self.samples = {k: [] for k in ("u", "dudl", "step", "time_ps", "replica", "epot", "cg")}
        mode = "batched" if windows.batched else "sequential"
        self._print(f"# lambda windows: {self.n} ({mode}), T = {windows.temperatures[0]:.2f} K, samples every "
                    f"{self.sample_every} steps ({self.sample_every * windows.dt:g} ps), "
                    + (f"Hamiltonian exchange every {self.exchange_every} steps" if self.exchange_every else "no exchanges"))
        self._print("# (lambda_elec, lambda_vdw): " + " ".join(f"({a:g},{b:g})" for a, b in windows.lambdas))

    def _print(self, s):
        if self.log is not None:
            print(s, file=self.log, flush=True)

    # ------------------------------------------------------------------ one sample / exchange
    def _sample(self):
        w = self.windows
        u, g, it = w.sample()
        self.samples["u"].append(u)
        self.samples["dudl"].append(g)
        self.samples["step"].append(self.step)
        self.samples["time_ps"].append(w.time_ps)
        self.samples["replica"].append(self.stats.replica.copy())
        self.samples["epot"].append(w.potentials())
        self.samples["cg"].append(it)
        return u

    def _exchange(self, u):
        pairs = exchange_pairs(self.n, self.stats.n_exchanges)
        acc, src = metropolis(u, pairs, self.rng.random(len(pairs)))
        self.windows.permute(src)
        self.stats.record(pairs, acc, src)
        return pairs, acc

    # ------------------------------------------------------------------ running
    def run(self, nsteps: int, prefix: str | None = "fe", report: int = 0, restart: int = 0):
        """nsteps steps of every window with samples every sample_every steps (and exchanges);
        a log line every `report` steps, checkpoint + samples every `restart` (prefix None: no files)."""
        w = self.windows
        block = int(np.gcd.reduce([x for x in (self.sample_every, report, restart, nsteps) if x > 0]))
        files = prefix is not None
        logf = open(f"{prefix}_fe.log", "a" if self.step else "w") if (files and report) else None
        if logf is not None and not self.step:
            logf.write("# lambda (elec, vdw): " + " ".join(f"({a:g},{b:g})" for a, b in w.lambdas) + "\n"
                       "#       step    time_ps  T_mean_K  T_min_K  T_max_K  cg_mean  cg_samp  acceptance (pairs)"
                       "   ns/day/window\n")
        t0, s0, done = time.time(), self.step, 0
        while done < nsteps:
            m = min(block, nsteps - done)
            w.advance(m)
            done += m
            self.step += m
            if self.step % self.sample_every == 0:
                u = self._sample()
                if self.exchange_every and self.step % self.exchange_every == 0:
                    self._exchange(u)
            speed = (self.step - s0) * w.dt / 1000.0 / max(time.time() - t0, 1e-9) * 86400.0
            if report and self.step % report == 0:
                obs = [w.observables(k) for k in range(self.n)]
                T = np.array([o["temp_K"] for o in obs])
                cg = np.mean([o["cg_mean"] for o in obs])
                acc = self.stats.neighbour_acceptance() if self.exchange_every else np.array([])
                cgs = self.samples["cg"][-1] if self.samples["cg"] else 0
                line = (f"  {self.step:10d} {w.time_ps:10.2f} {T.mean():9.2f} {T.min():8.2f} {T.max():8.2f} "
                        f"{cg:8.2f} {cgs:8d}  " + " ".join("  -  " if np.isnan(a) else f"{a:.3f}" for a in acc)
                        + f"   {speed:.1f}")
                self._print(line)
                if logf is not None:
                    logf.write(line + "\n")
                    logf.flush()
            if files and restart and self.step % restart == 0:
                self.save(prefix)
        el = time.time() - t0
        if logf is not None:
            logf.close()
        summary = self.summary(ns_per_day=(self.step - s0) * w.dt / 1000.0 / max(el, 1e-9) * 86400.0)
        if files:
            self.save(prefix)
            with open(f"{prefix}_fe.json", "w") as fh:
                json.dump(summary, fh, indent=1)
        return summary

    def summary(self, ns_per_day: float | None = None) -> dict:
        st = self.stats
        out = {"lambdas": self.windows.lambdas.tolist(), "temperature_K": float(self.windows.temperatures[0]),
               "steps": self.step, "time_ps": self.windows.time_ps, "samples": len(self.samples["u"]),
               "sample_every": self.sample_every, "exchange_every": self.exchange_every,
               "exchanges": st.n_exchanges, "batched": bool(self.windows.batched)}
        if self.exchange_every:
            out["neighbour_acceptance"] = [None if np.isnan(a) else float(a) for a in st.neighbour_acceptance()]
            out["round_trips_total"] = int(st.round_trips.sum())
        if ns_per_day is not None:
            out["ns_per_day_per_window"] = ns_per_day
            out["ns_per_day_aggregate"] = ns_per_day * self.n
        return out

    # ------------------------------------------------------------------ outputs / checkpoints
    def arrays(self) -> dict:
        """The samples as arrays (the content of prefix_fe.npz)."""
        S, w = self.samples, self.windows
        K = self.n
        return {"u": np.array(S["u"], float).reshape(-1, K, K), "dudl": np.array(S["dudl"], float).reshape(-1, K, 2),
                "step": np.array(S["step"], int), "time_ps": np.array(S["time_ps"], float),
                "replica": np.array(S["replica"], int).reshape(-1, K), "epot": np.array(S["epot"], float).reshape(-1, K),
                "cg": np.array(S["cg"], int), "lambdas": w.lambdas.copy(), "kT": float(w.integ.kT),
                "temperature": float(w.temperatures[0]), "dt": w.dt, "sample_every": self.sample_every,
                "meta": json.dumps(self.meta)}

    def save(self, prefix: str):
        np.savez(f"{prefix}_fe.npz", **self.arrays())
        d = {"format": FORMAT, "lambdas": self.windows.lambdas.copy(), "step": self.step,
             "rng": self.rng.bit_generator.state, "stats": self.stats.to_dict(), "samples": self.samples,
             "meta": self.meta, "windows": self.windows.state_dict()}
        with open(prefix + ".fe.chk", "wb") as fh:
            pickle.dump(d, fh)
        self.windows.write_restarts(prefix)

    def load(self, path: str):
        """Continue from a checkpoint written by `save` (same system, settings and windows)."""
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        if d.get("format") != FORMAT:
            raise ValueError(f"{path}: not a {FORMAT!r} checkpoint")
        self.windows.load_state_dict(d["windows"])
        self.step = int(d["step"])
        self.rng.bit_generator.state = d["rng"]
        self.stats = ExchangeStatistics.from_dict(d["stats"])
        self.samples = d["samples"]
        self.meta.update(d.get("meta", {}))
