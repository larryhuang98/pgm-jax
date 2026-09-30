"""Multiple time stepping (r-RESPA) for pGM MD: force groups integrated with their own time steps.

Contents: the settings `MTS`, the state `MTSState` (part of MDState), the step `_MTSMixin` with
its two integrators `MTSIntegrator` (rigid bodies) and `MTSFlexibleIntegrator` (atoms), the
switching function `switch`, and `mts_stats` (diagnostics).

The force is split into groups, F = F_slow + F_fast (+ F_bonded), and the Liouville propagator is
factorised as in r-RESPA [1]_:

    exp(iL dt) ~ B_slow(dt/2) [B_fast(h/2) A(h) B_fast(h/2)]^n B_slow(dt/2),     h = dt / n,

with B_g(t): p += t F_g (the kicks of one group; RATTLE after each kick with constraints) and A the
drift (SHAKE for constrained atoms, the NO_SQUISH free rotor for rigid bodies).  Every group is a
function of the positions only, so the scheme is symplectic and time-reversible.  `dt` of the
simulation is the outer step; the fast forces are evaluated n times per outer step, the slow ones
(with the induced-dipole solve and the PME) once.

Splits (`MTS.split`):
  "bonded"  F_fast = bonded terms + restraints, F_slow = everything nonbonded (flexible engine).
  "special" F_fast = bonded terms + restraints + the pGM and van der Waals interactions of the
            special pairs (md/topology.py: the rest of a small molecule, the nearby heavy-atom
            groups of a large one), unswitched, with the kernel erf(a r)/r and the fast induction
            model below; F_slow = F_full - F_fast.  pGM has no electrostatic exclusions, so the
            1-2 and 1-3 electrostatics vibrate with the bonds and angles (flexible engine).
  "short"   F_fast = a cheap short-range nonbonded model (below) + the bonded terms + restraints;
            F_slow = F_full - F_fast, with F_full the ordinary force (converged induced dipoles,
            PME, cutoffs) at the same positions.  The groups sum exactly to the full force, and
            with n = 1 the integrator is the ordinary one.
  bonded > 1 adds a third, innermost level for the bonded terms (+ restraints), stepped `bonded`
  times per fast step (the short-range forces then form the middle level, as in Tinker-HP's RESPA1).

The fast nonbonded model is a conservative potential of the positions alone, evaluated over the
pairs closer than r_short (smoothly switched off from r_short - switch_width):
  U_fast = 1/2 sum_i sum_k S(r_ik) [KE e_ik + w_ik e_vdW(r_ik)] + KE U_ind,
  e_ik   = the pGM pair energy of the Gaussian charges and dipoles with the kernel
           erf(a_ik r)/r - erf(beta_short r)/r: the Gaussian-screened Coulomb minus its smooth
           long-range part (the Ewald real-space form, with beta_short chosen so that
           erfc(beta_short r_short) = 1e-3).  The screening makes the pair terms negligible at the
           switch; the bare Coulomb switched atom by atom would cut through neutral molecules and
           leave monopole forces 30 times the real ones.
  S(r)   = 1 - t^3 (10 - 15 t + 6 t^2), t = (r - r_on) / (r_short - r_on) in [0, 1] (C2),
  U_ind  = polarization "mutual" (default): -1/2 E.mu1 with nu0 = alpha E, mu1 = nu0 + alpha T nu0 (E
           the switched permanent field, T the switched dipole tensor: one mutual iteration, as the
           OPT/ExPT expansions of Simmonett et al. 2015); "direct": -1/2 E.alpha E; "none": 0.
           The forces are the exact gradients: -mu1.dE/dx - 1/2 nu0 dT/dx nu0 (three sweeps over the
           pair list, no solve).
The fast pairs come from a list of the pairs closer than r_short + buffer, compacted from the
electrostatic rows of the full evaluation at every outer step (Result.geometry).  At every fast
evaluation the list is checked (sum of the two largest atomic displacements since it was built <
buffer) and rebuilt from the neighbour list if needed (a lax.cond: rare), so no pair inside r_short
is ever missed.  Displacements are taken in float64 (exact for bonded pairs in mixed precision).

Thermostats (`o_step`): "outer" puts one O step of length dt in the middle of the outer step
(BAOAB-RESPA of Tinker-HP [2]_): for n = 1 this is BAOAB; for n
even the O step sits between two fast steps, for n odd in the middle of the middle one's drift
(recursively with three levels).  "inner": BAOAB at the innermost level (an O step of length h in
every innermost drift).  The heat of every O step is booked, so observables()["econs"] stays the
conserved effective energy.

Induced-dipole predictor: the dipole history lives on the outer steps (one full solve per outer
step).  With `anchor` (default with fast induced dipoles) the history holds mu - mu_fast, whose
short-range part is removed, and the guess is mu_fast(x_new) + extrapolation(mu - mu_fast): mu_fast
is known at the new positions before the solve (the last fast evaluation), so the fused initial
residual of the solver still applies (ubiquitin, 8 fs outer step: 18.1 CG iterations, 17.4 with
the quadratic predictor mu3, against 16.9 per step at 4 fs without MTS).  MDState.induction.mu stays the converged
dipoles; only the history (and so the checkpoint's induction state) is in the anchored form.

Barostat: the Monte Carlo barostat runs at outer steps (its trial energy is the full energy); an
accepted move re-evaluates every group at the scaled positions.  Replica exchange (remd.py) does
not support MTS yet.

    from pgm_jax.md.mts import MTS
    # a solvated protein: bonded terms and special pairs every 7/3 fs, the rest every 7 fs
    sim = FlexibleSimulation(sys, templates, pos, H, MDSettings().replace(predictor="mu3", **elec_cutoff_settings(0.7)),
                             dt=0.007, mts=MTS(inner=3, split="special"), constraints="h-bonds", hmr=3.024,
                             thermostat="bussi")

Measurements, stability limits and recommended settings: docs/mts.md.

Units: nm, ps, amu, kJ/mol, K, e.

References
----------
.. [1] M. Tuckerman, B. J. Berne, G. J. Martyna, J. Chem. Phys. 97, 1990 (1992).
.. [2] L. Lagardere, F. Aviat, J.-P. Piquemal, J. Phys. Chem. Lett. 10, 2593 (2019).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from ..units import KE
from ._jaxmd import dataclasses, simulate
from .flexible import FlexibleIntegrator
from .forcefield import _PRED
from .integrate import Integrator, MDState
from .kernels import erf_kernels_closed

if TYPE_CHECKING:
    from jax.typing import ArrayLike

    from .forcefield import Result
    from .integrate import Dynamics

BETA_R = 2.3268  # erfc(BETA_R) = 1e-3: default beta_short = BETA_R / r_short


@dataclass(frozen=True)
class MTS:
    """Multiple time stepping settings (the simulation's dt is the outer step).

    Immutable (frozen dataclass); read by the integrator when it builds its step (static).

    Parameters
    ----------
    inner : int
        Fast steps per outer step n.
    split : {"short", "special", "bonded"}
        Fast group (module docstring): "short" (short-range pGM model + bonded), "special" (the
        topologically close pairs + bonded) or "bonded" (bonded terms).
    r_short : float
        The fast nonbonded pairs are switched off at r_short [nm].
    switch_width : float
        The switch starts at r_short - switch_width [nm].
    buffer : float
        The fast pair list holds the pairs closer than r_short + buffer [nm].
    beta_short : float, optional
        Screening of the fast electrostatics, erfc(beta_short r) [1/nm] (None: erfc = 1e-3 at
        r_short, BETA_R / r_short).
    polarization : {"auto", "mutual", "direct", "none"}
        Fast induced dipoles ("auto": mutual with induction).
    bonded : int
        Bonded (+ restraint) steps per fast step; > 1: a third level.
    o_step : {"outer", "inner"}
        Thermostat step: "outer" (BAOAB-RESPA) or "inner" (every innermost step).
    anchor : bool, optional
        Predictor on mu - mu_fast (None: when the fast level has induced dipoles and the
        predictor is mu3 / mu4).
    """

    inner: int = 2  # fast steps per outer step
    split: str = "short"  # fast group: "short" (short-range pGM model + bonded) | "special" (the
    # topologically close pairs + bonded) | "bonded" (bonded terms)
    r_short: float = 0.5  # nm: the fast nonbonded pairs are switched off at r_short
    switch_width: float = 0.1  # nm: the switch starts at r_short - switch_width
    buffer: float = 0.1  # nm: the fast pair list holds the pairs closer than r_short + buffer
    beta_short: float | None = None  # nm^-1: fast electrostatics screened by erfc(beta_short r); None: erfc = 1e-3
    # at r_short (BETA_R / r_short)
    polarization: str = "auto"  # fast induced dipoles: "mutual" | "direct" | "none" | "auto" (mutual with induction)
    bonded: int = 1  # bonded (+ restraint) steps per fast step; > 1: a third level
    o_step: str = "outer"  # thermostat step: "outer" (BAOAB-RESPA) | "inner" (every innermost step)
    anchor: bool | None = None  # predictor on mu - mu_fast (None: when the fast level has induced dipoles)

    def levels(self) -> list:
        """Return the levels as (groups, steps per step of the enclosing level), outermost first."""
        if self.split == "bonded":
            return [(("slow",), 1), (("bonded",), int(self.inner))]
        if int(self.bonded) > 1:
            return [(("slow",), 1), ((self.split,), int(self.inner)), (("bonded",), int(self.bonded))]
        return [(("slow",), 1), ((self.split, "bonded"), int(self.inner))]


def switch(r: jax.Array, r_on: float, r_off: float) -> tuple[jax.Array, jax.Array]:
    """Return the switching function S(r) and S'(r) / r.

    S(r) = 1 - t^3 (10 - 15 t + 6 t^2), t = (r - r_on) / (r_off - r_on) clipped to [0, 1] (C2: the
    force and its derivative are continuous).

    Parameters
    ----------
    r : jax.Array
        Distances [nm], r > 0.
    r_on, r_off : float
        Start and end of the switch [nm].

    Returns
    -------
    S : jax.Array
        Switch (dimensionless), 1 below r_on, 0 above r_off.
    dS_over_r : jax.Array
        S'(r) / r [1/nm^2].
    """
    w = r_off - r_on
    t = jnp.clip((r - r_on) / w, 0.0, 1.0)
    S = 1.0 - t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
    dS = -30.0 * t * t * (1.0 - t) * (1.0 - t) / w
    return S, dS / r


@dataclasses.dataclass
class MTSState:
    """Multiple-time-stepping part of MDState (a JAX-MD dataclass, i.e. a pytree).

    The force of every level at the current positions, the fast-level induced dipoles and energy,
    and the short-range pair list (None for the "bonded" and "special" splits).

    Parameters
    ----------
    forces : tuple
        Force of every level, outermost (slow) first, in engine form (atomic forces (N, 3) or
        rigid-body forces) [kJ/mol/nm].
    mu : jax.Array (N, 3)
        Fast-level induced dipoles [e nm].
    efast : jax.Array ()
        Fast-level nonbonded energy [kJ/mol].
    ks : jax.Array (N, ms) int
        Partners in the short-range list (compact rows).
    ws : jax.Array (N, ms)
        Van der Waals weights of the entries.
    within : jax.Array (N, ms) bool
        Valid entries.
    ref : jax.Array (N, 3)
        Atom positions when the list was built [nm].
    rebuilds : jax.Array () int32
        Rebuilds of the list at fast steps (list too old).
    overflow : jax.Array () bool
        The list exceeded its capacity (block repeated).
    """

    forces: tuple
    mu: jnp.ndarray  # (N, 3) fast-level induced dipoles (e nm)
    efast: jnp.ndarray  # () fast-level nonbonded energy (kJ/mol)
    ks: jnp.ndarray  # (N, ms) partners in the short-range list (compact rows)
    ws: jnp.ndarray  # (N, ms) van der Waals weights of the entries
    within: jnp.ndarray  # (N, ms) valid entries
    ref: jnp.ndarray  # (N, 3) atom positions when the list was built (nm)
    rebuilds: jnp.ndarray  # () int32: rebuilds of the list at fast steps (list too old)
    overflow: jnp.ndarray  # () bool: the list exceeded its capacity (block repeated)


def _tree_sub(a: Any, b: Any) -> Any:
    """Return the leafwise difference a - b of two pytrees."""
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


class _MTSMixin:
    """r-RESPA step for the rigid-body (Integrator) and the flexible (FlexibleIntegrator) engines.

    Mixed in before the engine's integrator class, which provides the kicks, drifts, O steps and
    the full force evaluation; the subclasses provide `_atoms`, `_centers` and `_engine_forces`.
    The state carries MDState.mts (MTSState); `ms`, the capacity of the short-range list, is a
    static size (a change recompiles).

    Attributes
    ----------
    mts : MTS
        Settings.
    ms : int or None
        Capacity of the short-range list (sized in `init`).
    pol : {"mutual", "direct", "none"}
        Fast induction model.
    anchor : bool
        Anchored predictor.
    short : bool
        split == "short".
    r_on, r_off, r_list : float
        Switch start, switch end (r_short) and list radius r_short + buffer [nm] ("short").
    beta_s : float
        Fast screening parameter [1/nm] ("short").
    levels : list
        MTS.levels().
    nlev : int
        Number of levels.
    o_outer : bool
        o_step == "outer".
    """

    keep_geometry = True  # the full evaluation returns its electrostatic rows (short list)

    def __init__(self, *args: Any, mts: MTS, **kw: Any) -> None:
        """Set up the engine's integrator (args, kw) with multiple time stepping `mts`.

        Raises
        ------
        TypeError
            mts is not an MTS.
        NotImplementedError
            Extended-Lagrangian dipoles, or the combinations refused by `_configure`.
        ValueError
            Invalid settings (`_configure`).
        """
        if getattr(args[0], "ips", False):
            raise ValueError("multiple time stepping is not implemented for long_range='ips'")
        if not isinstance(mts, MTS):
            raise TypeError("mts must be an MTS instance")
        self.mts = mts
        self.ms = None  # capacity of the short-range list (sized in init)
        ff = args[0] if args else kw.get("ff")
        if ff is not None and getattr(ff, "iel", False):
            raise NotImplementedError("extended-Lagrangian dipoles (iel) with multiple time stepping")
        super().__init__(*args, **kw)
        self._configure()

    # ------------------------------------------------------------------ settings
    def _configure(self) -> None:
        """Check the settings against the engine and derive the level structure and fast model.

        Raises
        ------
        ValueError
            Invalid inner / bonded / split / o_step / polarization / anchor / switch settings, a
            split the rigid engine cannot use, or r_short + buffer beyond the cutoffs.
        NotImplementedError
            An alchemical region, virtual sites in the flexible engine, charge flux with a pair
            split, or zero polarizabilities with fast induction.
        """
        m, ff, s = self.mts, self.ff, self.ff.s
        if int(m.inner) != m.inner or m.inner < 1 or int(m.bonded) != m.bonded or m.bonded < 1:
            raise ValueError(f"MTS inner and bonded must be positive integers (got {m.inner}, {m.bonded})")
        if m.split not in ("short", "special", "bonded"):
            raise ValueError(f"MTS split must be 'short', 'special' or 'bonded', got {m.split!r}")
        if m.o_step not in ("outer", "inner"):
            raise ValueError(f"MTS o_step must be 'outer' or 'inner', got {m.o_step!r}")
        flexible = isinstance(self, FlexibleIntegrator)
        if not flexible and (m.split != "short" or m.bonded > 1):
            raise ValueError(
                "rigid bodies have no bonded terms and no internal motion: use MTS(split='short', bonded=1)"
            )
        if m.split == "bonded" and m.bonded > 1:
            raise ValueError("MTS(split='bonded') has two levels: bonded must be 1")
        self.short = m.split == "short"
        pairs = m.split != "bonded"  # a fast pair model (short-range list or special pairs)
        pol = m.polarization
        if pol == "auto":
            pol = "mutual" if (pairs and ff.ind) else "none"
        if pol not in ("direct", "mutual", "none"):
            raise ValueError(f"MTS polarization must be 'direct', 'mutual', 'none' or 'auto', got {m.polarization!r}")
        if pol != "none" and not (pairs and ff.ind):
            raise ValueError(
                f"MTS polarization {pol!r} needs split 'short' or 'special' and induced dipoles (elec 'qi' or 'qpi')"
            )
        self.pol = pol
        # combinations the fast models do not implement (explicit, no silent fallback)
        if getattr(self, "alchemy", None) is not None:
            raise NotImplementedError(
                "multiple time stepping with an alchemical region: the fast pair models "
                "do not scale the solute (run the lambda windows without MTS)"
            )
        if flexible and getattr(self, "vsites", None) is not None:
            raise NotImplementedError(
                "multiple time stepping of the flexible engine with virtual sites: the RESPA "
                "drifts do not rebuild the sites (use the rigid engine for rigid molecules "
                "with sites, or no MTS)"
            )
        if getattr(ff, "flux", None) is not None and pairs:
            raise NotImplementedError(
                "charge flux with MTS(split='short' | 'special'): the fast pair models use "
                "fixed charges; use split='bonded'"
            )
        if pol != "none" and getattr(ff, "alpha_mask", False):
            raise NotImplementedError(
                "the fast induction model divides by the polarizabilities; atoms with "
                "alpha = 0 (e.g. virtual sites) need MTS(polarization='none')"
            )
        anchor = m.anchor
        if anchor is None:
            anchor = pol != "none" and s.induction.predictor in _PRED
        if anchor and not (pol != "none" and s.induction.predictor in _PRED):
            raise ValueError(
                "the anchored predictor needs fast induced dipoles (polarization 'mutual' or 'direct') "
                f"and predictor mu3 or mu4 (got {pol!r}, {s.induction.predictor!r})"
            )
        self.anchor = bool(anchor)
        if self.short:
            r_on = m.r_short - m.switch_width
            if not (m.switch_width > 0 and r_on > 0 and m.buffer > 0):
                raise ValueError(
                    f"MTS needs 0 < switch_width < r_short and buffer > 0 (got {m.r_short}, "
                    f"{m.switch_width}, {m.buffer})"
                )
            if m.r_short + m.buffer > ff.rc_e + 1e-12:
                raise ValueError(
                    f"MTS r_short + buffer = {m.r_short + m.buffer:g} nm exceeds the electrostatics "
                    f"cutoff {ff.rc_e:g} nm (the short-range list is taken from its rows)"
                )
            if s.terms.vdw != "none" and m.r_short > ff.rc_v + 1e-12:
                raise ValueError(f"MTS r_short {m.r_short:g} nm exceeds the van der Waals cutoff {ff.rc_v:g} nm")
            self.r_on, self.r_off, self.r_list = r_on, float(m.r_short), float(m.r_short + m.buffer)
            self.beta_s = float(BETA_R / m.r_short if m.beta_short is None else m.beta_short)
            if not self.beta_s > 0:
                raise ValueError(f"MTS beta_short must be positive, got {m.beta_short}")
        if m.split == "special":  # rows of the special-pair model: flexible molecules
            flex_rows = [np.asarray(r).ravel() for _, r in self.flex.groups]
            if not flex_rows:
                raise ValueError("MTS(split='special') needs flexible molecules (FlexibleTemplate)")
            self._flex_rows = jnp.asarray(np.sort(np.concatenate(flex_rows)).astype(np.int32))
        self.levels = m.levels()
        self.nlev = len(self.levels)
        self.o_outer = m.o_step == "outer"

    def describe_mts(self) -> str:
        """Return one line for log headers (levels with their steps, fast model, O step, predictor)."""
        m = self.mts
        hs, steps = [], self.dt
        for groups, n in self.levels:
            steps = steps / n
            hs.append(f"{'+'.join(groups)} {steps * 1000:g} fs")
        what = f"; fast pairs: the special pairs, {self.pol} induction" if self.mts.split == "special" else ""
        if self.short:
            what = (
                f"; fast pairs < {self.r_off:g} nm (switch from {self.r_on:g}), "
                f"{self.pol} induction, list buffer {m.buffer:g} nm "
                f"(capacity {self.ms})"
            )
        return (
            f"multiple time stepping (r-RESPA): {', '.join(hs)}{what}; O step {m.o_step}; "
            f"predictor {'anchored on the fast dipoles' if self.anchor else 'on the outer steps'}"
        )

    # ------------------------------------------------------------------ engine-specific pieces
    def _atoms(self, x: Any) -> jax.Array:
        """Return the atom positions (N, 3) [nm] of the engine's position variable."""
        raise NotImplementedError

    def _centers(self, x: Any, pos: jax.Array) -> jax.Array:
        """Return the neighbour-list centres [nm]."""
        raise NotImplementedError

    def _engine_forces(self, x: Any, F: jax.Array) -> Any:
        """Return atomic forces F (N, 3) mapped to the engine's force variable (linear)."""
        raise NotImplementedError

    def _kick_by(self, dyn: Dynamics, F: Any, h: float) -> Dynamics:
        """Return dyn after p += h F (RATTLE with constraints); dyn.force is kept."""
        d = dyn.set(force=F)
        d = self._kick(d, h) if getattr(self, "cons", None) is not None else simulate.momentum_step(d, h)
        return d.set(force=dyn.force)

    def _drift_by(self, dyn: Dynamics, h: float) -> Dynamics:
        """Return dyn after a drift of h [ps] (SHAKE / RATTLE, or the free rigid-body rotor)."""
        if getattr(self, "cons", None) is not None:
            return self._drift(dyn, h)
        return simulate.position_step(dyn, self.shift, h)

    def _bonded_forces(self, pos: jax.Array, box: jax.Array) -> jax.Array:
        """Return the atomic forces (N, 3) [kJ/mol/nm] of the bonded group: bonded terms and restraints."""
        F = jnp.zeros_like(pos)
        flex = getattr(self, "flex", None)
        if flex is not None and flex.groups:
            F = F - jax.grad(flex.energy)(pos)
        if self.restraints is not None:
            F = F - jax.grad(self.restraints.energy)(pos, box)
        return F

    # ------------------------------------------------------------------ fast nonbonded model
    def _short_nonbonded(
        self,
        pos: jax.Array,
        H: jax.Array,
        ks: jax.Array,
        ws: jax.Array,
        within: jax.Array,
        special: bool = False,
        rows: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return the energy, atomic forces and induced dipoles of the fast nonbonded model.

        Over the short-range list (module docstring), or with `special` over the special pairs
        (unswitched, the Gaussian-screened Coulomb without the long-range part removed).  Row sums
        as in the force field: every pair is in both rows, F_i = -sum_k de_ik/dx_ik (+ the
        covalent-dipole frames, by the VJP of the permanent dipoles).  Displacements are float64
        (sequential minimum image), the pair algebra in the force field's compute dtype.

        Parameters
        ----------
        pos : jax.Array (N, 3)
            Atom positions [nm].
        H : jax.Array (3, 3)
            Box [nm].
        ks : jax.Array (R, m) int
            Partners of every row atom.
        ws : jax.Array (R, m)
            Van der Waals weights.
        within : jax.Array (R, m) bool
            Valid entries.
        special : bool
            The special-pair model (static).
        rows : jax.Array (R,) int, optional
            The atoms of the rows (None: all; every partner of a row atom must be a row atom, as
            within the flexible molecules).

        Returns
        -------
        energy : jax.Array ()
            Fast nonbonded energy (pairs and induction) [kJ/mol].
        forces : jax.Array (N, 3)
            Its exact negative gradient [kJ/mol/nm].
        mu : jax.Array (N, 3)
            Fast induced dipoles [e nm] (0 without fast induction).
        """
        ff = self.ff
        cd = ff.cd
        N = ff.n
        P = ff._atoms(self.params)
        p, vjp_p = jax.vjp(lambda y: ff.perm_dipoles(y, H, P["cov"]), pos)
        at = (lambda v: v) if rows is None else (lambda v: v[rows])
        full = (lambda v: v) if rows is None else (lambda v: jnp.zeros((N,) + v.shape[1:], v.dtype).at[rows].set(v))
        pr = at(pos)
        x = [pr[:, c][:, None] - pos[:, c][ks] for c in range(3)]  # float64 differences, minimum image
        for c in (2, 1, 0):
            n = jnp.round(x[c] / H[c, c])
            x = [x[j] - n * H[c, j] if j <= c else x[j] for j in range(3)]
        x = [c.astype(cd) for c in x]
        w = within.astype(cd)
        r = jnp.sqrt(jnp.where(within, x[0] * x[0] + x[1] * x[1] + x[2] * x[2], 1.0))
        R = P["radius"].astype(cd)
        A = erf_kernels_closed(1.0 / jnp.sqrt(2.0 * (at(R)[:, None] ** 2 + R[ks] ** 2)), r, 4)
        if special:
            A0, A1, A2, A3 = (u * w for u in A)
            S, dS = w, jnp.zeros_like(w)
        else:
            B = erf_kernels_closed(jnp.asarray(self.beta_s, cd), r, 4)
            A0, A1, A2, A3 = ((u - v) * w for u, v in zip(A, B))
            S, dS = switch(r, self.r_on, self.r_off)
            S, dS = S * w, dS * w
        SA1, SA2 = S * A1, S * A2
        q = P["q"].astype(cd)
        qi, qk = at(q)[:, None], q[ks]
        pc = p.astype(cd)
        pk = [pc[:, c][ks] for c in range(3)]
        pi = [at(pc)[:, c][:, None] for c in range(3)]

        def rowsum(v: jax.Array) -> jax.Array:  # sum over each row's partners
            return jnp.sum(v, axis=1)

        alpha = at(P["alpha"])[:, None]
        corr = 0.0  # polarization energy not in the pair sum (units of KE)
        if self.pol != "none":  # nu0 = alpha E_short, E_short = -dU_pair/dd at d = p
            pkx = pk[0] * x[0] + pk[1] * x[1] + pk[2] * x[2]
            cf = -qk * SA1 - SA2 * pkx
            gp = jnp.stack([rowsum(cf * x[c] + SA1 * pk[c]) for c in range(3)], -1).astype(jnp.float64)
            mu = nu0 = -alpha * gp  # (row atoms, 3)
            corr = 0.5 * jnp.sum(nu0 * nu0 / alpha)
        if self.pol == "mutual":  # nu1 = alpha T nu0 (one mutual iteration); mu = nu0 + nu1
            n0 = nu0.astype(cd)
            n0f = full(n0)
            nk = [n0f[:, c][ks] for c in range(3)]
            ni = [n0[:, c][:, None] for c in range(3)]
            nkx = nk[0] * x[0] + nk[1] * x[1] + nk[2] * x[2]
            gm = jnp.stack([rowsum(-SA2 * nkx * x[c] + SA1 * nk[c]) for c in range(3)], -1).astype(jnp.float64)
            mu = nu0 - alpha * gm  # T nu0 = -gm
            corr = corr - jnp.sum(nu0 * gm)
        if self.pol != "none":
            mc = mu.astype(cd)
            mcf = full(mc)
            mk = [mcf[:, c][ks] for c in range(3)]
            mi = [mc[:, c][:, None] for c in range(3)]
            dk = [pk[c] + mk[c] for c in range(3)]
            di = [pi[c] + mi[c] for c in range(3)]
        else:
            mu = jnp.zeros_like(pr)
            dk, di = pk, pi
        dix = di[0] * x[0] + di[1] * x[1] + di[2] * x[2]
        dkx = dk[0] * x[0] + dk[1] * x[1] + dk[2] * x[2]
        didk = di[0] * dk[0] + di[1] * dk[1] + di[2] * dk[2]
        t = qi * dkx - qk * dix
        e = qi * qk * A0 + t * A1 - A2 * dix * dkx + A1 * didk
        rad = -qi * qk * A1 - t * A2 + A3 * dix * dkx - A2 * didk
        non = [A1 * (qi * dk[c] - qk * di[c]) - A2 * (di[c] * dkx + dk[c] * dix) for c in range(3)]
        if self.pol != "none":  # no mu_i-mu_k term (the induction model's own dipole terms below)
            mix = mi[0] * x[0] + mi[1] * x[1] + mi[2] * x[2]
            mkx = mk[0] * x[0] + mk[1] * x[1] + mk[2] * x[2]
            mimk = mi[0] * mk[0] + mi[1] * mk[1] + mi[2] * mk[2]
            e = e + A2 * mix * mkx - A1 * mimk
            rad = rad - A3 * mix * mkx + A2 * mimk
            non = [non[c] + A2 * (mi[c] * mkx + mk[c] * mix) for c in range(3)]
        if self.pol == "mutual":  # + the nu0_i-nu0_k pair term (-1/2 nu0 T nu0)
            nix = ni[0] * x[0] + ni[1] * x[1] + ni[2] * x[2]
            ninj = ni[0] * nk[0] + ni[1] * nk[1] + ni[2] * nk[2]
            e = e - A2 * nix * nkx + A1 * ninj
            rad = rad + A3 * nix * nkx - A2 * ninj
            non = [non[c] - A2 * (ni[c] * nkx + nk[c] * nix) for c in range(3)]
        elj, glj = ff._vdw_rows(r, self._vdw_params(P, ks, at), ws.astype(cd), grad=True)
        E = KE * e + elj
        radial = S * (KE * rad + glj) + dS * E
        SK = S * KE
        g = jnp.stack([rowsum(radial * x[c] + SK * non[c]) for c in range(3)], -1).astype(jnp.float64)
        cfd = -qk * SA1 - SA2 * dkx  # dU/dd_i (total dipoles) for the covalent-dipole frames
        gd = KE * jnp.stack([rowsum(cfd * x[c] + SA1 * dk[c]) for c in range(3)], -1).astype(jnp.float64)
        forces = -(full(g) + vjp_p(full(gd))[0])
        energy = 0.5 * jnp.sum(rowsum(S * E).astype(jnp.float64)) + KE * corr
        return energy, forces, full(mu)

    def _vdw_params(self, P: dict, k: jax.Array, at: Callable[[jax.Array], jax.Array]) -> tuple:
        """Return the van der Waals pair parameters for rows at(.) and partners k.

        PGMForceField._vdw_params with a subset of row atoms.

        Raises
        ------
        ValueError
            An unknown van der Waals form.
        """
        cd, vdw = self.ff.cd, self.ff.s.terms.vdw
        if vdw in ("lj", "de"):
            rh, se = P["lj_rmin_half"].astype(cd), P["lj_sqrt_eps"].astype(cd)
            return (at(rh)[:, None] + rh[k], at(se)[:, None] * se[k])
        if vdw == "gvdw":
            sa, sc, b = (P[n].astype(cd) for n in ("gvdw_sqrt_a", "gvdw_sqrt_c6", "gvdw_b"))
            R = P["radius"].astype(cd)
            return (
                at(sa)[:, None] * sa[k],
                at(sc)[:, None] * sc[k],
                0.5 * (at(b)[:, None] + b[k]),
                1.0 / jnp.sqrt(2.0 * (at(R)[:, None] ** 2 + R[k] ** 2)),
            )
        if vdw == "none":
            return ()
        raise ValueError(f"multiple time stepping: unknown van der Waals form {vdw!r}")

    def _special(self) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return the special-pair rows of the flexible molecules as a pair list.

        (partners (padding 0), van der Waals weights, valid mask, row atoms).  The special pairs of
        rigid molecules only exert internal forces, which their constraints remove.
        """
        rows = self._flex_rows
        sp = self.ff.special[rows]
        valid = sp < self.ff.n
        return jnp.where(valid, sp, 0), jnp.where(valid, self.ff.special_w[rows], 0.0), valid, rows

    def short_energy(self, pos: jax.Array, H: jax.Array, st: MDState) -> jax.Array:
        """Return the fast nonbonded energy [kJ/mol] at atom positions pos (for tests: forces = -grad).

        Over the state's short-range list, or the special pairs.
        """
        if self.mts.split == "special":
            ks, ws, valid, rows = self._special()
            return self._short_nonbonded(pos, H, ks, ws, valid, special=True, rows=rows)[0]
        m = st.mts
        return self._short_nonbonded(pos, H, m.ks, m.ws, m.within)[0]

    # ------------------------------------------------------------------ short-range list
    def _compact(
        self, k: jax.Array, r: jax.Array, within: jax.Array, wv: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return the short-range list compacted from force-field rows, and an overflow flag.

        The entries of rows (k, distances r [nm], valid mask, van der Waals weights) closer than
        r_short + buffer, compacted in column order to the capacity ms: (ks, ws, valid,
        overflow).
        """
        N, ms = self.ff.n, self.ms
        m = within & (r < self.r_list)
        slot = jnp.cumsum(m.astype(jnp.int32), axis=1) - 1
        count = slot[:, -1] + 1
        tgt = jnp.where(m & (slot < ms), slot, ms)
        rows = jnp.broadcast_to(jnp.arange(N, dtype=jnp.int32)[:, None], k.shape)
        ks = jnp.zeros((N, ms + 1), k.dtype).at[rows, tgt].set(k)[:, :ms]
        ws = jnp.zeros((N, ms + 1), wv.dtype).at[rows, tgt].set(wv)[:, :ms]
        valid = jnp.arange(ms)[None, :] < jnp.minimum(count, ms)[:, None]
        return ks, ws, valid, jnp.max(count) > ms

    def _rebuild(self, st: MDState, pos: jax.Array, m: MTSState) -> MTSState:
        """Return m with the short-range list rebuilt at a fast step from the neighbour list.

        The list's candidates reach the pair cutoff, far beyond r_short + buffer, so they are
        still complete.
        """
        cand, ovf = self.nb.candidates(st.nbr, self._centers(st.dyn.position, pos), st.box, pos)
        k, x, within, wv, ovf2, _ = self.ff._rows(pos, st.box, cand)
        r = jnp.sqrt(jnp.where(within, x[0] * x[0] + x[1] * x[1] + x[2] * x[2], 1.0))
        ks, ws, valid, ovf3 = self._compact(k, r, within, wv)
        return m.set(
            ks=ks, ws=ws, within=valid, ref=pos, rebuilds=m.rebuilds + 1, overflow=m.overflow | ovf | ovf2 | ovf3
        )

    def _fresh(self, st: MDState, pos: jax.Array, m: MTSState) -> MTSState:
        """Return m with the list rebuilt (lax.cond) if a pair may have moved inside r_short.

        That is when the two largest atomic displacements since the build sum to at least the
        buffer.
        """
        d2 = jnp.sum((pos - m.ref) ** 2, axis=1)
        i = jnp.argmax(d2)
        d1 = d2[i]
        stale = jnp.sqrt(d1) + jnp.sqrt(jnp.max(d2.at[i].set(0.0))) >= self.mts.buffer
        return jax.lax.cond(stale, lambda m: self._rebuild(st, pos, m), lambda m: m, m)

    def _size_list(self, x: Any, box: ArrayLike) -> None:
        """Set the capacity ms of the short-range list from the pair counts at x (host; "short" only).

        10 % + 8 head-room, a multiple of 8, at most the width of the electrostatic rows it is
        taken from; recompiles when it changes.
        """
        if not self.short:
            return
        ff = self.ff
        box = jnp.asarray(box, jnp.float64)
        pos = self._atoms(x)
        centers = self._centers(x, pos)
        nbr = self.nb.allocate(pos, centers, box)
        cand = self.nb.candidates(nbr, centers, box, pos)[0]

        def count(pos: jax.Array, H: jax.Array, cand: jax.Array) -> jax.Array:
            """Return the largest number of partners within r_short + buffer (rows without capacity)."""
            saved = ff.mc
            ff.mc = None
            try:
                k, xx, within, _, _, _ = ff._rows(pos, H, cand)
            finally:
                ff.mc = saved
            r2 = xx[0] * xx[0] + xx[1] * xx[1] + xx[2] * xx[2]
            return jnp.max(jnp.sum(within & (r2 < self.r_list**2), axis=1))

        c = int(jax.jit(count)(pos, box, cand))
        ms = int(np.ceil((c * 1.1 + 8) / 8.0) * 8)
        width = ff.mc_e if ff.mc_e is not None else ff.mc
        if width is not None:
            ms = min(ms, int(width))
        if ms != self.ms:
            self.ms = ms
            self.compile()

    def check_block(self, st: MDState) -> None:
        """Grow the short-range list's capacity after a block in which it overflowed (host).

        The driver then repeats the block, since MDState.overflow is set too.
        """
        m = getattr(st, "mts", None)
        if m is not None and m.overflow is not None and bool(m.overflow):
            width = self.ff.mc_e if self.ff.mc_e is not None else self.ff.mc
            self.ms = int(self.ms + max(8, int(np.ceil(0.1 * self.ms / 8.0)) * 8))
            if width is not None:
                self.ms = min(self.ms, int(width))
            self.compile()

    # ------------------------------------------------------------------ force evaluations
    def _eval_fast(self, st: MDState, j: int) -> MDState:
        """Return the state with the forces of level j >= 1 at its positions (list refreshed if needed)."""
        x = st.dyn.position
        pos = self._atoms(x)
        m = st.mts
        groups = self.levels[j][0]
        F = jnp.zeros_like(pos)
        mu, efast = m.mu, m.efast
        if "short" in groups:
            m = self._fresh(st, pos, m)
            efast, Fs, mu = self._short_nonbonded(pos, st.box, m.ks, m.ws, m.within)
            F = F + Fs
        if "special" in groups:
            ks, ws, valid, rows = self._special()
            efast, Fs, mu = self._short_nonbonded(pos, st.box, ks, ws, valid, special=True, rows=rows)
            F = F + Fs
        if "bonded" in groups:
            F = F + self._bonded_forces(pos, st.box)
        forces = m.forces[:j] + (self._engine_forces(x, F),) + m.forces[j + 1 :]
        return st.set(mts=m.set(forces=forces, mu=mu, efast=efast))

    def _with_result(self, st: MDState, F: Any, res: Result, nbr: Any, fast: tuple | None = None) -> MDState:
        """Return the state after a full evaluation, with the MTS state rebuilt.

        The base bookkeeping (dyn.force = the full force), then the short-range list from the
        evaluation's rows, the fast levels (evaluated here unless given: `fast`, evaluated with
        the previous list at the same positions), and F_slow = F_full - sum(fast).

        Raises
        ------
        ValueError
            The result has no row geometry (keep_geometry).
        """
        geo = res.geometry
        st = super()._with_result(st, F, res._replace(geometry=None), nbr)
        x = st.dyn.position
        pos = self._atoms(x)
        z = jnp.zeros((), bool)
        old = st.mts
        rebuilds = jnp.zeros((), jnp.int32) if old is None else old.rebuilds
        if self.short:
            if geo is None:
                raise ValueError("multiple time stepping needs the force field's row geometry (keep_geometry)")
            ks, ws, valid, ovf = self._compact(geo["k"], geo["r"], geo["within"], geo["wv"])
        else:
            ks = ws = valid = None
            ovf = z
        placeholder = tuple(jax.tree_util.tree_map(jnp.zeros_like, F) for _ in range(self.nlev))
        m = MTSState(
            forces=placeholder,
            mu=jnp.zeros_like(pos) if old is None else old.mu,
            efast=jnp.zeros((), jnp.float64) if old is None else old.efast,
            ks=ks,
            ws=ws,
            within=valid,
            ref=pos,
            rebuilds=rebuilds,
            overflow=ovf if old is None else (old.overflow | ovf),
        )
        if fast is None:
            s = st.set(mts=m)
            for j in range(self.nlev - 1, 0, -1):
                s = self._eval_fast(s, j)
            m = s.mts
            fast = m.forces[1:]
            if self.anchor:  # the solve just pushed mu: keep the history as mu - mu_fast
                ind = st.induction
                st = st.set(induction=ind.set(hist=ind.hist.at[0].add(-m.mu)))
        slow = F
        for f in fast:
            slow = _tree_sub(slow, f)
        m = m.set(forces=(slow,) + tuple(fast))
        return st.set(mts=m, overflow=st.overflow | m.overflow)

    def _eval_outer(self, st: MDState) -> MDState:
        """Return the state after the full evaluation at the end of an outer step.

        The fast levels were just evaluated at the same positions with the previous list; with the
        anchored predictor mu_fast is added to the history before the solve and removed after.
        """
        m = st.mts
        ind = st.induction
        if self.anchor:
            ind = ind.set(hist=ind.hist + m.mu[None])
        F, res, nbr = self._forces(
            st.dyn.position, st.box, ind, st.nbr, bias=st.bias, field=self.field_at(st, st.step + 1)
        )
        if self.anchor:
            ri = res.induction
            res = res._replace(induction=ri.set(hist=ri.hist - m.mu[None]))
        return self._with_result(st, F, res, nbr, fast=m.forces[1:])

    # ------------------------------------------------------------------ the step
    def _o(self, st: MDState, h: float) -> MDState:
        """Return the state after a thermostat step of length h [ps]."""
        dyn, aux, heat = self._o_step(st.dyn, st.aux, st.heat, h, self.thermostat_kT(st))
        return st.set(dyn=dyn, aux=aux, heat=heat)

    def _level(self, st: MDState, j: int, h: float, o_mid: bool) -> MDState:
        """Return the state after one step of level j (length h [ps], j and h static).

        Kick, the sub-steps of level j + 1 (lax.fori_loop) or the innermost drift, the forces of
        level j at the new positions, kick.  o_mid: the outer O step (length dt) goes in the
        middle of this step (between the two halves of the sub-steps for even n, recursively in
        the middle sub-step for odd n).
        """
        st = st.set(dyn=self._kick_by(st.dyn, st.mts.forces[j], 0.5 * h))
        if j == self.nlev - 1:
            thermo = self.thermostat is not None and (o_mid or not self.o_outer)
            if thermo:
                st = st.set(dyn=self._drift_by(st.dyn, 0.5 * h))
                st = self._o(st, self.dt if o_mid else h)
                st = st.set(dyn=self._drift_by(st.dyn, 0.5 * h))
            else:
                st = st.set(dyn=self._drift_by(st.dyn, h))
        else:
            n = self.levels[j + 1][1]
            hs = h / n

            def loop(s: MDState, count: int) -> MDState:  # `count` sub-steps of level j + 1
                return (
                    s if count == 0 else jax.lax.fori_loop(0, count, lambda _, c: self._level(c, j + 1, hs, False), s)
                )

            if o_mid and n % 2 == 0:
                st = self._o(loop(st, n // 2), self.dt)
                st = loop(st, n // 2)
            elif o_mid:
                st = loop(self._level(loop(st, n // 2), j + 1, hs, True), n // 2)
            else:
                st = loop(st, n)
        st = self._eval_outer(st) if j == 0 else self._eval_fast(st, j)
        return st.set(dyn=self._kick_by(st.dyn, st.mts.forces[j], 0.5 * h))

    def _step(self, st: MDState) -> MDState:
        """Advance one outer step (and the barostat every `interval` steps)."""
        o_mid = self.thermostat is not None and self.o_outer
        st = self._level(st, 0, self.dt, o_mid).set(step=st.step + 1)
        if self.ensemble == "npt":
            st = jax.lax.cond(st.step % self.interval == 0, self._barostat, lambda s: s, st)
        return st

    def _run(self, st: MDState, n: int | jax.Array) -> MDState:
        """Advance n outer steps; a state without MTS part gets its split forces first.

        E.g. a checkpoint of an ordinary run: the forces are split and the predictor restarted.
        """
        if st.mts is None:  # e.g. a checkpoint of an ordinary run: split forces, restart the predictor
            s = self._state_forces(st, False)
            st = s.set(induction=s.induction.set(count=jnp.ones_like(s.induction.count)))
        return super()._run(st, n)

    def init(self, x: Any, box: ArrayLike, key: jax.Array, momentum: Any = None, bias: Any = None) -> MDState:
        """Return a new state as the engine's init, after sizing the short-range list."""
        self._size_list(x, box)
        return super().init(x, box, key, momentum, bias=bias)


class MTSIntegrator(_MTSMixin, Integrator):
    """r-RESPA for rigid molecules (NO_SQUISH rigid bodies): Integrator with `mts=MTS(...)`."""

    def _atoms(self, x: Any) -> jax.Array:
        """Return the atom positions of the rigid bodies x."""
        return self.rigid.positions(x)

    def _centers(self, x: Any, pos: jax.Array) -> jax.Array:
        """Return the centres of mass of the bodies (the list groups)."""
        return x.center

    def _engine_forces(self, x: Any, F: jax.Array) -> Any:
        """Return the body forces of atomic forces F."""
        return self.rigid.forces(x, F)


class MTSFlexibleIntegrator(_MTSMixin, FlexibleIntegrator):
    """r-RESPA for atoms with constraints (g-BAOAB kicks and drifts): FlexibleIntegrator + `mts`."""

    def _atoms(self, x: jax.Array) -> jax.Array:
        """Return x (the positions are the atoms)."""
        return x

    def _centers(self, x: jax.Array, pos: jax.Array) -> jax.Array:
        """Return the neighbour-list group centres."""
        return self.flex.list_centers(pos)

    def _engine_forces(self, x: jax.Array, F: jax.Array) -> jax.Array:
        """Return F (atomic forces are the engine's)."""
        return F


def mts_stats(sim: Any) -> dict:
    """Return diagnostics of a simulation with multiple time stepping ({} without).

    "list_rebuilds" (at fast steps), "list_capacity" (ms) and "efast" (the fast-level energy of
    the current state [kJ/mol]).
    """
    st = sim.state
    m = getattr(st, "mts", None)
    if m is None:
        return {}
    return {"list_rebuilds": int(m.rebuilds), "list_capacity": sim.integ.ms, "efast": float(m.efast)}


# ----------------------------------------------------------------------------- command lines
