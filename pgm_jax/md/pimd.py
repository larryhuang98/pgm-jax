"""Path-integral molecular dynamics: PIMD (PILE-L / PILE-G), thermostatted RPMD and RPMD.

Contents: the ring polymer (`normal_modes`, `contraction_matrix`, `RingPolymer`), the
thermostat (`PILE` settings, `as_pile`, `PILEStep`), the state `PIMDState` and the integrator
`PIMDIntegrator`, the force engines `PotentialEngine` (any JAX potential) and `PGMBeads` (the
pGM force field on every bead, state `PGMBeadState`), and the driver `PIMDSimulation`.

The quantum canonical partition function of the nuclei is sampled with a ring polymer of P beads
per atom (imaginary-time path integral, Trotter factorisation):

    H_P(q, p) = sum_k sum_i [ |p_i^k|^2 / (2 m_i) + m_i omega_P^2 |q_i^k - q_i^(k+1)|^2 / 2 ] + U(q),
    U(q)      = sum_k V(q^k),         omega_P = P kB T / hbar,         bead index k = 0..P-1 cyclic,

sampled classically at P T (beta_P = beta / P), with the physical masses on every bead.  Quantum
averages of position-dependent observables are bead averages; <V> = <U> / P.

Integrator ([1]_, in the BAOAB order of [2]_):

    B(dt/2)  p^k += dt/2 f^k                                (physical forces, one evaluation per step)
    A(dt/2)  free ring polymer, exactly, in normal modes    (harmonic mode frequencies omega_k)
    O(dt)    thermostat in normal modes
    A(dt/2), forces, B(dt/2).

NVE (RPMD) is B A(dt) B.  Normal modes are real and orthonormal, ordered by frequency,
q~_l = sum_j C_jl q_j with C = [1/sqrt(P), sqrt(2/P) cos(2 pi j l/P), sqrt(2/P) sin(2 pi j l/P), ...,
(-1)^j/sqrt(P) for even P], omega_l = 2 omega_P sin(pi l / P).  The free ring-polymer step is the
exact harmonic rotation ("exact") or its Cayley transform ([3]_; "cayley", the default):

    p' = [(1 - a^2) p - m w^2 h q] / (1 + a^2),   q' = [h p / m + (1 - a^2) q] / (1 + a^2),
    a = w h / 2.

Both conserve the free ring-polymer energy of every mode (Cayley = implicit midpoint on
a quadratic Hamiltonian), so both sample the free ring polymer exactly; Cayley is strongly stable
(no resonance of stiff modes with the physical forces as P grows).

Thermostats (the O step, at kB T_P = P kB T, heat booked so that econs = H_P - heat is conserved):
  "pimd"   PILE: Langevin on every internal mode with gamma_l = 2 lam omega_l (lam = 1: critical
           damping of the free modes), and on the centroid either Langevin with gamma_0 =
           1/tau_centroid (PILE-L, thermostat=PILE("l")) or Bussi's global stochastic rescaling
           with time constant tau_centroid (PILE-G, PILE("g"): gentle on the centroid dynamics
           and the dipole predictor).
  "trpmd"  thermostatted RPMD [4]_: the
           internal modes as above with lam = 1/2 by default, no thermostat on the centroid, whose
           dynamics estimates Kubo-transformed correlation functions (e.g. diffusion).
  "rpmd"   no thermostat (NVE ring polymer [5]_).

Estimators (per atom, then summed or averaged by element):
  primitive        K_i = 3 P kB T / 2 - (1/P) sum_k m_i omega_P^2 |q_i^k - q_i^(k+1)|^2 / 2
  centroid virial  K_i = 3 kB T / 2 - (1 / 2P) sum_k (q_i^k - qbar_i) . f_i^k
Both have the same average (for a harmonic oscillator exactly <K> = <V> = omega^2/beta sum_l
1/(omega_l^2 + omega^2) / 2 at any P); the centroid virial one has a P-independent variance.

Engines.  `PotentialEngine` wraps any JAX potential V(x, box) (tests, model systems).  `PGMBeads`
runs the pGM force field of a `FlexibleSimulation` on every bead (vmapped: one program for all
beads), each bead with its own induced dipoles and predictor history (the step counter of the
predictor is shared, so its fused / unfused switch stays a branch).  One neighbour list, built from
the centroid, serves every bead: the list radius is enlarged by `bead_margin`, the largest
distance of a bead atom from its centroid (checked after every block).
Ring-polymer contraction ([6]_; `contract=P'`): the
potential is split as V = V_mono + (V - V_mono), where V_mono is the sum of the gas-phase monomer
energies of the fitted flexible templates (bonded terms + intramolecular pGM + intramolecular van
der Waals: exactly the model the bonded terms were fitted with, cheap and stiff), evaluated on all
P beads, and the intermolecular remainder (PME, induction, van der Waals: expensive and smooth on
the scale of the ring polymer) on P' beads obtained by truncating the normal modes,
q' = T q with T = sqrt(P'/P) C'_{jl} C_{kl} over the P' lowest modes; the forces return by T^T
and U = sum_k V_mono(q^k) + (P/P') sum_k' [V - V_mono](q'^k').  P' = 1 puts the intermolecular
forces on the centroid.

NPT: isotropic Monte Carlo barostat, every bead of a molecule translated with the centroid's
molecular centre of mass (PIMDIntegrator).  Force beads are evaluated in lax.map chunks of 8
vmapped beads by default (bead_chunk; twice as fast as one vmap over 32 beads of 512 waters).

Rigid bodies and constraints are not supported: a rigid-rotor path integral is not a ring polymer
of atoms.  Quantum water needs flexible molecules (FlexibleTemplate;
pgm_jax.models.water.flexible_water builds a flexible pGM water with the q-TIP4P/F monomer
surface; docs/pimd.md).

Units: nm, ps, amu, kJ/mol, K.

References
----------
.. [1] M. Ceriotti, M. Parrinello, T. E. Markland, D. E. Manolopoulos, J. Chem. Phys. 133,
   124104 (2010).
.. [2] J. Liu, D. Li, X. Liu, J. Chem. Phys. 145, 024103 (2016).
.. [3] R. Korol, N. Bou-Rabee, T. F. Miller III, J. Chem. Phys. 151, 124103 (2019).
.. [4] M. Rossi, M. Ceriotti, D. E. Manolopoulos, J. Chem. Phys. 140, 234116 (2014).
.. [5] I. R. Craig, D. E. Manolopoulos, J. Chem. Phys. 121, 3368 (2004).
.. [6] T. E. Markland, D. E. Manolopoulos, J. Chem. Phys. 129, 024105 (2008).
"""

from __future__ import annotations

import dataclasses as _dc
import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TextIO

import jax
import jax.numpy as jnp
import numpy as np

from ..units import AMU_NM3_TO_G_CM3, BAR_PER_KJMOL_NM3, DEBYE_E_NM, HBAR_KJMOL_PS, KB, KJMOL_TO_MEV
from ._jaxmd import dataclasses
from .barostats import MonteCarloBarostat
from .box import inv3, max_cutoff, volume
from .driver import (
    LogTable,
    Stopwatch,
    advance_with_rebuilds,
    block_length,
    device_tree,
    finite_or_raise,
    read_checkpoint,
    retry_block,
    write_checkpoint,
)
from .io import NetCDFTrajectory, write_restart
from .neighbors import AtomNeighbors, MoleculeNeighbors
from .thermostats import Bussi

if TYPE_CHECKING:
    from jax.typing import ArrayLike

    from ._jaxmd import partition
    from .flexible import FlexibleSimulation
    from .forcefield import InductionState

logger = logging.getLogger(__name__)

LEGACY_FORMAT = "pgm_jax pimd 1"  # the "format" entry of legacy pickle checkpoints


# ----------------------------------------------------------------------------- ring polymer
def normal_modes(P: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the real orthonormal normal-mode matrix of a P-bead ring and the mode frequency indices.

    Parameters
    ----------
    P : int
        Number of beads.

    Returns
    -------
    C : np.ndarray (P, P)
        C[j, l]: bead j, mode column l, ordered by frequency: centroid 1/sqrt(P), cos 1, sin 1,
        cos 2, sin 2, ... (sqrt(2/P) cos / sin(2 pi j l / P)), and (-1)^j / sqrt(P) for even P.
    idx : np.ndarray (P,) int
        Frequency index of every column (omega = 2 omega_P sin(pi idx / P)).
    """
    P = int(P)
    j = np.arange(P)
    cols, idx = [np.full(P, 1.0 / math.sqrt(P))], [0]
    for l in range(1, (P - 1) // 2 + 1):
        cols.append(math.sqrt(2.0 / P) * np.cos(2.0 * math.pi * j * l / P))
        idx.append(l)
        cols.append(math.sqrt(2.0 / P) * np.sin(2.0 * math.pi * j * l / P))
        idx.append(l)
    if P % 2 == 0 and P > 1:
        cols.append((-1.0) ** j / math.sqrt(P))
        idx.append(P // 2)
    return np.stack(cols, 1), np.array(idx, int)


def contraction_matrix(P: int, Pc: int) -> np.ndarray:
    """Return the (Pc, P) ring-polymer contraction matrix T [6]_.

    The P' = Pc lowest normal modes of the P-bead polymer, rescaled by sqrt(Pc / P), on Pc beads:
    q' = T q.  The centroid is kept, and Pc = P gives the identity.

    Raises
    ------
    ValueError
        Unless 1 <= Pc <= P.
    """
    if not 1 <= Pc <= P:
        raise ValueError("need 1 <= P' <= P")
    C, _ = normal_modes(P)
    Cc, _ = normal_modes(Pc)
    return math.sqrt(Pc / P) * Cc @ C[:, :Pc].T


class RingPolymer:
    """Normal modes and free ring-polymer propagation of P beads at temperature T.

    Attributes
    ----------
    P : int
        Number of beads.
    T : float
        Physical temperature [K].
    kT, kT_P : float
        kB T and P kB T [kJ/mol].
    omega_P : float
        P kB T / hbar [rad/ps].
    C : np.ndarray (P, P)
        Normal-mode matrix (`normal_modes`).
    omega : np.ndarray (P,)
        Free ring-polymer mode frequencies 2 omega_P sin(pi idx / P) [rad/ps] (centroid 0).
    """

    def __init__(self, beads: int, temperature: float) -> None:
        """Set up a ring polymer of `beads` beads at `temperature` [K] (omega_P = P kB T / hbar).

        Raises
        ------
        ValueError
            Fewer than one bead.
        """
        self.P = int(beads)
        if self.P < 1:
            raise ValueError("at least one bead")
        self.T = float(temperature)
        self.kT = KB * self.T
        self.kT_P = self.P * self.kT
        self.omega_P = self.P * self.kT / HBAR_KJMOL_PS
        C, idx = normal_modes(self.P)
        self.C = C
        self.omega = 2.0 * self.omega_P * np.sin(np.pi * idx / self.P)  # (P,) rad/ps
        self.omega[0] = 0.0

    def to_nm(self, x: jax.Array) -> jax.Array:
        """Return the normal-mode coordinates C^T x (P, ...) of a bead array x (P, ...)."""
        return jnp.tensordot(jnp.asarray(self.C.T), x, axes=1)

    def from_nm(self, y: jax.Array) -> jax.Array:
        """Return the bead array C y (P, ...) of normal-mode coordinates y (P, ...)."""
        return jnp.tensordot(jnp.asarray(self.C), y, axes=1)

    def spring(self, q: jax.Array, mass: jax.Array) -> jax.Array:
        """Return the spring energy sum_k m_i omega_P^2 |q_i^k - q_i^(k+1)|^2 / 2 per atom (N,) [kJ/mol].

        q (P, N, 3) [nm], mass (N, 1) [amu].
        """
        d = q - jnp.roll(q, -1, axis=0)
        return 0.5 * self.omega_P**2 * mass[:, 0] * jnp.sum(d * d, axis=(0, 2))

    def propagator(
        self, h: float, mass: ArrayLike, kind: str = "cayley"
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Return the coefficients of the free ring-polymer step of length h for every mode and atom.

        p' = a p + b q, q' = c p + d q in normal modes (module docstring; the centroid drifts
        freely).

        Parameters
        ----------
        h : float
            Step [ps].
        mass : ArrayLike (N, 1)
            Masses [amu].
        kind : {"cayley", "exact"}
            Cayley transform or exact harmonic rotation.

        Returns
        -------
        a, b, c, d : jax.Array (P, N, 1)
            Coefficients (b [amu/ps^2], c [ps/amu], a and d dimensionless).

        Raises
        ------
        ValueError
            An unknown kind.
        """
        w = self.omega[:, None, None]
        m = np.asarray(mass, float)[None]
        if kind == "exact":
            wh = w * h
            cs, sn = np.cos(wh), np.sin(wh)
            safe = np.where(w > 0, w, 1.0)
            a, d = cs + 0 * m, cs + 0 * m
            b = -m * w * sn
            c = np.where(w > 0, sn / (m * safe), h / m)
        elif kind == "cayley":
            x = (0.5 * w * h) ** 2
            den = 1.0 + x
            a = d = (1.0 - x) / den + 0 * m
            b = -m * w * w * h / den
            c = h / m / den + 0 * w
        else:
            raise ValueError("propagator: 'cayley' | 'exact'")
        return tuple(jnp.asarray(np.broadcast_to(v, (self.P,) + m.shape[1:]).copy()) for v in (a, b, c, d))

    def sample_free(self, key: jax.Array, centroid: ArrayLike, mass: ArrayLike) -> jax.Array:
        """Return beads (P, N, 3) [nm] drawn from the free ring polymer around a centroid (N, 3) [nm].

        At T_P: internal mode l ~ N(0, kT_P / (m omega_l^2)); mass (N, 1) [amu].
        """
        if self.P == 1:
            return jnp.asarray(centroid)[None]
        m = jnp.asarray(mass)[None]
        w = jnp.asarray(self.omega[1:])[:, None, None]
        z = jax.random.normal(key, (self.P - 1,) + jnp.shape(centroid), jnp.float64)
        y = jnp.concatenate([math.sqrt(self.P) * jnp.asarray(centroid)[None], z * jnp.sqrt(self.kT_P / m) / w], 0)
        return self.from_nm(y)


@_dc.dataclass(frozen=True)
class PILE:
    """Settings of the path-integral Langevin thermostat [1]_ (immutable).

    Parameters
    ----------
    kind : str
        Centroid thermostat: "l" (Langevin with friction 1/tau_centroid, PILE-L) or "g" (Bussi's
        stochastic rescaling with time constant tau_centroid, PILE-G).
    tau_centroid : float
        Time constant of the centroid thermostat [ps].
    lam : float or None
        Internal modes get the friction 2 lam omega_l; None: 1 (critical damping) for mode
        "pimd", 1/2 for "trpmd".

    Raises
    ------
    ValueError
        An unknown kind or a non-positive tau_centroid.
    """

    kind: str = "l"
    tau_centroid: float = 0.2
    lam: float | None = None

    def __post_init__(self) -> None:
        """Check and normalize the settings."""
        kind = str(self.kind).lower().removeprefix("pile-")
        if kind not in ("l", "g"):
            raise ValueError(f"PILE: kind must be 'l' (Langevin centroid) or 'g' (Bussi centroid), not {self.kind!r}")
        if float(self.tau_centroid) <= 0.0:
            raise ValueError(f"PILE: tau_centroid must be > 0 ({self.tau_centroid!r} ps)")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "tau_centroid", float(self.tau_centroid))


def as_pile(spec: PILE | str) -> PILE:
    """Return a PILE object from a PILE, or from the names "pile-l" / "pile-g" (default settings).

    Parameters
    ----------
    spec : PILE or str
        The thermostat settings.

    Returns
    -------
    PILE

    Raises
    ------
    ValueError
        Anything else.
    """
    if isinstance(spec, PILE):
        return spec
    if isinstance(spec, str) and spec.lower() in ("pile-l", "pile-g"):
        return PILE(spec)
    raise ValueError(f"thermostat: a PILE object or 'pile-l' / 'pile-g', not {spec!r}")


class PILEStep:
    """The O step of the path-integral thermostat in normal modes for one mode of dynamics.

    mode "pimd": internal modes gamma_l = 2 lam omega_l, centroid Langevin 1/tau_centroid (PILE("l"))
    or Bussi rescaling with time constant tau_centroid (PILE("g")); "trpmd": internal modes only
    (lam = 1/2 by default); "rpmd": none.

    Attributes
    ----------
    gamma : np.ndarray (P,)
        Friction of every normal mode [1/ps] (centroid 0 with Bussi or without a thermostat).
    lam : float
        Internal-mode damping factor.
    centroid_bussi : bool
        The centroid is thermostatted by Bussi rescaling.
    """

    def __init__(self, ring: RingPolymer, mode: str = "pimd", pile: PILE = PILE()) -> None:
        """Set the friction of every normal mode for `mode` ("pimd" | "trpmd" | "rpmd") and `pile`.

        Raises
        ------
        ValueError
            An unknown mode.
        """
        mode = mode.lower()
        if mode not in ("pimd", "trpmd", "rpmd"):
            raise ValueError(f"mode: 'pimd' | 'trpmd' | 'rpmd', not {mode!r}")
        self.ring, self.mode, self.pile = ring, mode, pile
        self.kind, self.tau0 = "pile-" + pile.kind, pile.tau_centroid
        lam = pile.lam
        self.lam = (0.5 if mode == "trpmd" else 1.0) if lam is None else float(lam)
        g = 2.0 * self.lam * ring.omega
        g[0] = 1.0 / self.tau0 if (mode == "pimd" and pile.kind == "l") else 0.0
        if mode == "rpmd":
            g[:] = 0.0
        self.gamma = g
        self.centroid_bussi = mode == "pimd" and pile.kind == "g"
        self._bussi = Bussi(self.tau0)

    @property
    def active(self) -> bool:
        """Whether the O step does anything (not RPMD)."""
        return self.mode != "rpmd"

    def describe(self) -> str:
        """One line for the log header."""
        if self.mode == "rpmd":
            return "RPMD (no thermostat)"
        c = {"pile-l": f"centroid Langevin {1.0 / self.tau0:g}/ps", "pile-g": f"centroid Bussi {self.tau0:g} ps"}
        cen = c[self.kind] if self.mode == "pimd" else "centroid free"
        return f"{'PILE' if self.mode == 'pimd' else 'TRPMD'} (internal modes gamma = {2 * self.lam:g} omega_l, {cen})"

    def apply(
        self, pn: jax.Array, mass: jax.Array, key: jax.Array, h: float, dof: float
    ) -> tuple[jax.Array, jax.Array]:
        """Apply the O step of length h [ps] to the normal-mode momenta pn (P, N, 3) [amu nm/ps].

        Exact Langevin (Ornstein-Uhlenbeck) step at kT_P for every mode, c = exp(-gamma h); with
        PILE-G the centroid is rescaled by Bussi's step (dof: its degrees of freedom).  mass
        (N, 1) [amu].

        Returns
        -------
        pn : jax.Array (P, N, 3)
            New momenta.
        heat : jax.Array ()
            Kinetic energy change (heat taken up) [kJ/mol].
        """
        k1, k2 = jax.random.split(key)
        m = mass[None]
        e0 = 0.5 * jnp.sum(pn * pn / m)
        c = jnp.asarray(np.exp(-self.gamma * h))[:, None, None]
        s = jnp.sqrt((1.0 - c * c) * self.ring.kT_P * m)
        out = c * pn + s * jax.random.normal(k1, pn.shape, pn.dtype)
        if self.centroid_bussi:
            v = pn[0] / jnp.sqrt(mass)
            v, _ = self._bussi.apply(v, None, k2, h, self.ring.kT_P, dof, lambda u: u, None)
            out = out.at[0].set(v * jnp.sqrt(mass))
        return out, 0.5 * jnp.sum(out * out / m) - e0


# ----------------------------------------------------------------------------- state and integrator
@dataclasses.dataclass
class PIMDState:
    """State of a ring-polymer trajectory (a JAX-MD dataclass, i.e. a pytree).

    Parameters
    ----------
    q : jax.Array (P, N, 3)
        Bead positions [nm] (ring polymers whole).
    p : jax.Array (P, N, 3)
        Bead momenta [amu nm/ps].
    f : jax.Array (P, N, 3)
        Forces -dU/dq [kJ/mol/nm].
    upot : jax.Array ()
        U = sum_k V(q^k) (with contraction: the contracted U) [kJ/mol].
    box : jax.Array (3, 3)
        Box [nm].
    eng : object
        Engine state (PGMBeadState: induced dipoles, neighbour list, ...; PotentialEngine: 0).
    rng : jax.Array
        PRNG key.
    heat : jax.Array ()
        Heat taken up by the thermostat since the start [kJ/mol].
    step : jax.Array () int32
        Step counter.
    mc : jax.Array (4,) int32, optional
        Barostat (tries, accepts, window tries, window accepts).
    mc_dv : jax.Array (), optional
        Current maximum volume change [nm^3].
    """

    q: jnp.ndarray  # (P, N, 3) bead positions, nm (ring polymers whole)
    p: jnp.ndarray  # (P, N, 3) bead momenta, amu nm / ps
    f: jnp.ndarray  # (P, N, 3) forces -dU/dq, kJ/mol/nm
    upot: jnp.ndarray  # () U = sum_k V(q^k) (with contraction: the contracted U), kJ/mol
    box: jnp.ndarray  # (3, 3) nm
    eng: object  # engine state (induced dipoles, neighbour list, ...)
    rng: jnp.ndarray
    heat: jnp.ndarray  # () heat taken up by the thermostat since the start, kJ/mol
    step: jnp.ndarray
    mc: jnp.ndarray = None  # barostat (tries, accepts, window tries, window accepts)
    mc_dv: jnp.ndarray = None  # current maximum volume change (nm^3)


class PIMDIntegrator:
    """BAOAB ring-polymer integrator for an engine with init(q, box) -> eng and compute(q, box, eng).

    compute returns (forces (P, N, 3), U, eng).  `run(state, n)` and `forces(state)` are jitted
    entry points (re-created by `set_thermostat`).

    With a barostat (MonteCarloBarostat): isotropic Monte Carlo moves every `barostat.every` steps
    (engines with molecules: scale(q, s), energy(q, box, eng), nmol).  Every bead of a molecule is
    translated with the molecular centre of mass of the centroid, which leaves the springs and the
    intramolecular terms unchanged; acceptance on (U' - U) / P + p dV - N_mol kT ln(V'/V) (the
    ring polymer isomorphism at beta_P = beta / P), step size adapted to 25-75 % acceptance.

    Attributes
    ----------
    engine : PotentialEngine or PGMBeads
        Force engine.
    mass : jax.Array (N, 1)
        Masses [amu].
    ring : RingPolymer
        Normal modes and free propagation.
    P : int
        Beads.
    dt : float
        Time step [ps].
    pressure : float
        Barostat target [kJ/mol/nm^3].
    thermo : PILEStep
        The O step.
    mode : {"pimd", "trpmd", "rpmd"}
        Dynamics.
    dof : float
        3 N (degrees of freedom of the centroid, for Bussi).
    """

    def __init__(
        self,
        engine: Any,
        masses: ArrayLike,
        beads: int,
        temperature: float,
        dt: float,
        mode: str = "pimd",
        thermostat: PILE | str = PILE(),
        propagator: str = "cayley",
        barostat: MonteCarloBarostat | None = None,
    ) -> None:
        """Set up the ring-polymer integrator.

        Parameters
        ----------
        engine : PotentialEngine or PGMBeads
            Force engine.
        masses : ArrayLike (N,)
            Atom masses [amu].
        beads : int
            Number of beads P.
        temperature : float
            Temperature [K] (the ring polymer runs at P T).
        dt : float
            Time step [ps].
        mode : str
            "pimd", "trpmd" or "rpmd" (see the module docstring).
        thermostat : PILE or str
            Path-integral thermostat settings (as_pile).
        propagator : str
            Free ring-polymer step: "cayley" or "exact".
        barostat : MonteCarloBarostat or None
            Isotropic Monte Carlo barostat (None: constant volume).

        Raises
        ------
        ValueError
            A barostat with an engine without molecules, or invalid settings.
        """
        if barostat is not None and not hasattr(engine, "scale"):
            raise ValueError("the barostat needs an engine with molecules (PGMBeads)")
        self.barostat = barostat
        self.ensemble = "npt" if barostat is not None else "nvt"
        pressure = barostat.pressure if barostat is not None else 1.0
        self.pressure = float(pressure) / BAR_PER_KJMOL_NM3  # bar -> kJ/mol/nm^3
        self.interval = int(barostat.every if barostat is not None else 100)
        self.engine = engine
        self.mass = jnp.asarray(np.asarray(masses, float).reshape(-1, 1))
        self.n = int(self.mass.shape[0])
        self.ring = RingPolymer(beads, temperature)
        self.P = self.ring.P
        self.dt = float(dt)
        self.propagator = propagator
        self.set_thermostat(mode, thermostat)

    def set_thermostat(self, mode: str = "pimd", thermostat: PILE | str = PILE()) -> None:
        """(Re)configure the thermostat (e.g. PIMD equilibration, then TRPMD or RPMD) and re-jit.

        Parameters
        ----------
        mode : str
            "pimd", "trpmd" or "rpmd".
        thermostat : PILE or str
            Path-integral thermostat settings (as_pile).
        """
        self.thermo = PILEStep(self.ring, mode, as_pile(thermostat))
        self.mode = self.thermo.mode
        self.dof = 3.0 * self.n
        h = self.dt if self.mode == "rpmd" else 0.5 * self.dt
        self._A = self.ring.propagator(h, np.asarray(self.mass), self.propagator)
        self.run = jax.jit(self._run)
        self.forces = jax.jit(self._forces)

    # ------------------------------------------------------------------ pieces
    def _free(self, qn: jax.Array, pn: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return (q', p') in normal modes after the free ring-polymer step `_A`."""
        a, b, c, d = self._A
        return c * pn + d * qn, a * pn + b * qn

    def _forces(self, st: PIMDState) -> PIMDState:
        """Return the state with the engine's forces and U at its beads (jitted as `forces`)."""
        f, U, eng = self.engine.compute(st.q, st.box, st.eng)
        return st.set(f=f, upot=U, eng=eng)

    def init(
        self, q: ArrayLike, box: ArrayLike, key: jax.Array, momenta: ArrayLike | None = None, spread: bool = True
    ) -> PIMDState:
        """Return a new state with forces (host).

        Parameters
        ----------
        q : ArrayLike (N, 3) or (P, N, 3)
            Centroid positions (beads drawn from the free ring polymer if `spread`, else all on
            the centroid) or beads [nm].
        box : ArrayLike (3, 3)
            Box [nm].
        key : jax.Array
            PRNG key.
        momenta : ArrayLike (P, N, 3), optional
            Momenta [amu nm/ps] (None: drawn at kT_P).
        spread : bool
            Spread centroid positions over the free ring polymer.

        Returns
        -------
        PIMDState
        """
        q = jnp.asarray(q, jnp.float64)
        box = jnp.asarray(box, jnp.float64)
        k1, k2, k3 = jax.random.split(key, 3)
        if q.ndim == 2:
            q = self.ring.sample_free(k1, q, self.mass) if spread else jnp.broadcast_to(q, (self.P,) + q.shape)
        if momenta is None:
            p = jnp.sqrt(self.ring.kT_P * self.mass)[None] * jax.random.normal(k2, q.shape, jnp.float64)
        else:
            p = jnp.asarray(momenta, jnp.float64)
        z = jnp.zeros((), jnp.float64)
        st = PIMDState(
            q=q,
            p=p,
            f=jnp.zeros_like(q),
            upot=z,
            box=box,
            eng=self.engine.init(q, box),
            rng=k3,
            heat=z,
            step=jnp.zeros((), jnp.int32),
            mc=jnp.zeros(4, jnp.int32),
            mc_dv=jnp.asarray(0.01 * float(volume(box)), jnp.float64),
        )
        return self.forces(st)

    def _step(self, st: PIMDState) -> PIMDState:
        """Advance one BAOAB step (RPMD: B A(dt) B); the barostat every `interval` steps."""
        dt = self.dt
        p = st.p + 0.5 * dt * st.f
        qn, pn = self.ring.to_nm(st.q), self.ring.to_nm(p)
        rng, heat = st.rng, st.heat
        if self.mode == "rpmd":
            qn, pn = self._free(qn, pn)
        else:
            qn, pn = self._free(qn, pn)
            rng, k = jax.random.split(rng)
            pn, dh = self.thermo.apply(pn, self.mass, k, dt, self.dof)
            heat = heat + dh
            qn, pn = self._free(qn, pn)
        st = st.set(q=self.ring.from_nm(qn), p=self.ring.from_nm(pn), rng=rng, heat=heat)
        st = self._forces(st)
        st = st.set(p=st.p + 0.5 * dt * st.f, step=st.step + 1)
        if self.ensemble == "npt":
            st = jax.lax.cond(st.step % self.interval == 0, self._barostat, lambda s: s, st)
        return st

    def _barostat(self, st: PIMDState) -> PIMDState:
        """Try one Monte Carlo volume move (class docstring) and adapt the step size as Integrator._barostat."""
        eng = self.engine
        key, k1, k2 = jax.random.split(st.rng, 3)
        V = volume(st.box)
        dV = (2.0 * jax.random.uniform(k1, dtype=jnp.float64) - 1.0) * st.mc_dv
        Vn = V + dV
        s = jnp.cbrt(jnp.maximum(Vn, 1e-12) / V)
        qn, Hn = eng.scale(st.q, s), st.box * s
        Un, en = eng.energy(qn, Hn, st.eng)
        kT = self.ring.kT
        w = (Un - st.upot) / self.P + self.pressure * dV - eng.nmol * kT * jnp.log(jnp.maximum(Vn, 1e-12) / V)
        accept = (Vn > 0) & (jnp.log(jax.random.uniform(k2, dtype=jnp.float64)) < -w / kT)
        st = st.set(rng=key, eng=eng.flag(st.eng, en))
        st = jax.lax.cond(accept, lambda s: self._forces(s.set(q=qn, box=Hn, eng=en)), lambda s: s, st)
        mc = st.mc + jnp.array([1, 0, 1, 0], jnp.int32) + accept.astype(jnp.int32) * jnp.array([0, 1, 0, 1], jnp.int32)
        adapt = mc[2] >= 10
        rate = mc[3] / jnp.maximum(mc[2], 1)
        dv = jnp.where(
            adapt & (rate < 0.25), st.mc_dv / 1.1, jnp.where(adapt & (rate > 0.75), st.mc_dv * 1.1, st.mc_dv)
        )
        dv = jnp.minimum(dv, 0.3 * volume(st.box))
        mc = jnp.where(adapt, mc.at[2].set(0).at[3].set(0), mc)
        return st.set(mc=mc, mc_dv=dv)

    def _run(self, st: PIMDState, n: int | jax.Array) -> PIMDState:
        """Advance n steps (lax.fori_loop; jitted as `run`), resetting the engine's block maxima."""
        if hasattr(self.engine, "reset_block"):
            st = st.set(eng=self.engine.reset_block(st.eng))
        return jax.lax.fori_loop(0, n, lambda _, s: self._step(s), st)

    # ------------------------------------------------------------------ estimators
    def estimators(self, st: PIMDState) -> dict:
        """Return the estimators of a state (traceable).

        Returns
        -------
        dict
            "prim", "cv": per-atom primitive and centroid-virial kinetic energies (N,) [kJ/mol];
            "ke_beads", "spring", "hamiltonian" (H_P), "econs" (H_P - heat) [kJ/mol];
            "t_beads" (kinetic temperature of the beads divided by P), "t_modes" (P,) and
            "t_centroid" [K]; "epot" = U / P [kJ/mol].
        """
        ring, m = self.ring, self.mass
        q, p, f = st.q, st.p, st.f
        qc = jnp.mean(q, 0)
        spring_i = ring.spring(q, m)
        prim = 1.5 * ring.P * ring.kT - spring_i / ring.P
        cv = 1.5 * ring.kT - 0.5 / ring.P * jnp.sum((q - qc[None]) * f, axis=(0, 2))
        ke = 0.5 * jnp.sum(p * p / m[None])
        pn = ring.to_nm(p)
        t_mode = jnp.sum(pn * pn / m[None], axis=(1, 2)) / (3.0 * self.n * KB * ring.P)
        return {
            "prim": prim,
            "cv": cv,
            "ke_beads": ke,
            "spring": jnp.sum(spring_i),
            "hamiltonian": ke + jnp.sum(spring_i) + st.upot,
            "econs": ke + jnp.sum(spring_i) + st.upot - st.heat,
            "t_beads": 2.0 * ke / (3.0 * self.n * ring.P * KB) / ring.P,
            "t_modes": t_mode,
            "t_centroid": t_mode[0],
            "epot": st.upot / ring.P,
        }


# ----------------------------------------------------------------------------- engines
class PotentialEngine:
    """Any potential V(x (N, 3), box) [kJ/mol] on every bead (vmapped value_and_grad).

    With `soft` and `contract` = P', the potential is V + soft, and soft is evaluated on P'
    contracted beads (U = sum_k V(q^k) + (P/P') sum_k' soft(q'^k'), as for the pGM engine).  For
    tests and model systems; no barostat (no molecules).
    """

    def __init__(
        self,
        energy_fn: Callable[[jax.Array, jax.Array], jax.Array],
        soft: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        contract: int | None = None,
    ) -> None:
        """Set up the vmapped value-and-gradient functions of V (and soft) [kJ/mol]."""
        self.energy_fn, self.soft, self.contract = energy_fn, soft, contract
        self._vg = jax.vmap(jax.value_and_grad(energy_fn), in_axes=(0, None))
        self._sg = None if soft is None else jax.vmap(jax.value_and_grad(soft), in_axes=(0, None))

    def init(self, q: jax.Array, box: jax.Array) -> jax.Array:
        """Return the (empty) engine state, a zero scalar."""
        return jnp.zeros(())

    def compute(self, q: jax.Array, box: jax.Array, eng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return the forces (P, N, 3) [kJ/mol/nm], U [kJ/mol] and the engine state at beads q (P, N, 3)."""
        V, g = self._vg(q, box)
        U, f = jnp.sum(V), -g
        if self.soft is not None:
            P = q.shape[0]
            Pc = P if self.contract is None else min(int(self.contract), P)
            Tm = jnp.asarray(contraction_matrix(P, Pc))
            Vs, gs = self._sg(jnp.tensordot(Tm, q, axes=1), box)
            U = U + P / Pc * jnp.sum(Vs)
            f = f - P / Pc * jnp.tensordot(Tm.T, gs, axes=1)
        return f, U, eng


@dataclasses.dataclass
class PGMBeadState:
    """Engine state of PGMBeads (a JAX-MD dataclass, i.e. a pytree).

    Parameters
    ----------
    induction : InductionState
        Batched over the force beads (leading axis), the predictor step counter shared.
    nbr : partition.NeighborList
        Neighbour list of the centroid (shared by every bead).
    iters : jax.Array () int32
        CG iterations of the last solve (largest over the beads).
    max_iters : jax.Array () int32
        Largest since the block started.
    resid : jax.Array ()
        Largest final CG residual since the block started.
    overflow : jax.Array () bool
        Row or list capacity exceeded.
    cg_total : jax.Array ()
        CG iterations (largest over the beads) summed over the steps.
    elec, vdw : jax.Array ()
        Force-bead averages of the force field's parts [kJ/mol].
    """

    induction: object  # InductionState batched over the force beads (count shared)
    nbr: object  # neighbour list of the centroid (shared by every bead)
    iters: jnp.ndarray  # CG iterations of the last solve (largest over the beads)
    max_iters: jnp.ndarray  # largest since the block started
    resid: jnp.ndarray
    overflow: jnp.ndarray
    cg_total: jnp.ndarray  # CG iterations (largest over the beads) summed over the steps
    elec: jnp.ndarray  # force-bead averages of the force field's parts (kJ/mol)
    vdw: jnp.ndarray


def _nocount(ind: InductionState) -> InductionState:
    """Return the induction state without its step counter (kept unbatched over the beads)."""
    return ind.set(count=None)


class PGMBeads:
    """The pGM force field of a FlexibleSimulation on every bead of a ring polymer.

    Every bead (or contracted bead) keeps its own induced dipoles and predictor history; one
    neighbour list of the centroid serves all beads (its radius is enlarged by `bead_margin` nm,
    the largest allowed distance of a bead atom from its centroid).  contract = P' < P: ring-polymer
    contraction of the intermolecular part (module docstring).  bead_chunk: force beads per vmapped
    chunk, the chunks run one after the other in a lax.map ("auto": 8 when that divides more than 8
    beads; None: all at once); results are identical.

    Attributes
    ----------
    P : int
        Beads.
    Pc : int or None
        Contracted beads (None: no contraction).
    nf : int
        Force beads (P or Pc).
    chunk : int or None
        Force beads per vmapped chunk.
    bead_margin : float
        Allowed distance of a bead atom from its centroid [nm].
    nb : AtomNeighbors or MoleculeNeighbors
        The centroid's neighbour-list object.
    r_list : float
        Group radius of the molecule list including the margin [nm].
    T : jax.Array (Pc, P)
        Contraction matrix (with contraction).
    """

    def __init__(
        self,
        sim: FlexibleSimulation,
        beads: int,
        contract: int | None = None,
        bead_margin: float = 0.08,
        bead_chunk: int | str | None = "auto",
    ) -> None:
        """Set up the bead engine of a FlexibleSimulation.

        Parameters
        ----------
        sim : FlexibleSimulation
            The simulation (force field, settings, molecules).
        beads : int
            Beads P.
        contract : int, optional
            Contract the intermolecular part to this many beads (None or >= P: off).
        bead_margin : float
            Allowed distance of a bead atom from its centroid [nm].
        bead_chunk : int, "auto" or None
            Force beads per vmapped chunk.

        Raises
        ------
        ValueError
            Constraints; with contraction, a molecule that is not a FlexibleTemplate.
        NotImplementedError
            Virtual sites, mts, alchemy, restraints, extended-Lagrangian dipoles, biases or an
            external field.
        """
        integ = sim.integ
        if getattr(integ, "cons", None) is not None:
            raise ValueError(
                "path integrals need flexible molecules without constraints (constraints='none', no RigidTemplate)"
            )
        if getattr(integ, "vsites", None) is not None:
            raise NotImplementedError("virtual sites are not supported with path integrals yet")
        if getattr(integ, "mts", None) is not None or integ.alchemy is not None or integ.restraints is not None:
            raise NotImplementedError("path integrals with mts / alchemy / restraints are not supported yet")
        if getattr(sim.ff, "iel", False):
            raise NotImplementedError(
                "path integrals with extended-Lagrangian dipoles (MDSettings.induction.iel): every bead "
                "would need its own auxiliary dipoles; use iel='none'"
            )
        if getattr(integ, "bias", None) is not None or getattr(integ, "efield", None) is not None:
            raise NotImplementedError(
                "path integrals with biases on collective variables (bias=) or an external "
                "electric field (efield=) are not supported yet"
            )
        self.sim, self.ff, self.flex, self.integ = sim, sim.ff, sim.flex, integ
        self.P = int(beads)
        self.Pc = None if (contract is None or int(contract) >= self.P) else int(contract)
        self.nf = self.P if self.Pc is None else self.Pc
        if bead_chunk == "auto":  # 512 waters, P = 32: 15.2 ms/step vmapped at once, 8.0 in chunks of 8
            bead_chunk = 8 if (self.nf > 8 and self.nf % 8 == 0) else None
        self.chunk = None if not bead_chunk else int(bead_chunk)
        self.bead_margin = float(bead_margin)
        self.params = integ.params
        if self.Pc is not None:
            self.T = jnp.asarray(contraction_matrix(self.P, self.Pc))
            self._mono = self._monomer_groups()
        self.make_neighbors(np.asarray(sim.state.box))

    # ------------------------------------------------------------------ monomer reference (contraction)
    def _monomer_groups(self) -> list[tuple[Callable[[jax.Array], jax.Array], jax.Array]]:
        """Return (gas-phase intramolecular nonbonded energy of one copy, atom rows) per template.

        Raises
        ------
        ValueError
            Some molecule is not a FlexibleTemplate.
        """
        groups = []
        covered = 0
        for tpl, rows in self.flex.groups:
            model = tpl.model
            P = jax.tree_util.tree_map(jnp.asarray, tpl.P)

            def fn(R: jax.Array, model: Any = model, P: dict = P, idx: int = tpl.index) -> jax.Array:
                """Return the fitted model's intramolecular nonbonded energy [kJ/mol] of one copy."""
                return model.nonbonded(idx, R, P)[0]

            groups.append((fn, rows))
            covered += int(np.size(rows))
        if covered != self.flex.n:
            raise ValueError(
                "ring-polymer contraction needs every molecule to be a FlexibleTemplate "
                "(its gas-phase model is the stiff reference)"
            )
        return groups

    def monomer_nonbonded(self, x: jax.Array) -> jax.Array:
        """Return the gas-phase intramolecular pGM + van der Waals energy of all molecules [kJ/mol].

        Each template's fitted model at positions x (N, 3) [nm] (contraction only).
        """
        e = 0.0
        for fn, rows in self._mono:
            e = e + jnp.sum(jax.vmap(fn)(x[rows]))
        return e

    def reference(self, x: jax.Array) -> jax.Array:
        """Return the stiff reference on one bead [kJ/mol]: bonded + monomer nonbonded (contraction), or bonded."""
        e = self.flex.energy(x)
        if self.Pc is not None:
            e = e + self.monomer_nonbonded(x)
        return e

    # ------------------------------------------------------------------ neighbour lists
    def make_neighbors(self, H: ArrayLike) -> None:
        """Create the centroid's neighbour-list object for box H [nm], radii enlarged by the bead margin.

        A molecule list with the group radius + bead_margin, or an atom list with the pair cutoff +
        2 bead_margin (the skin shrunk to fit the box).

        Raises
        ------
        ValueError
            A box too small for the bead margin (atom list skin below 0.02 nm).
        """
        sim, s = self.sim, self.sim.settings
        self.r_list = float(sim.r_list) + self.bead_margin
        mode = s.neighbors.mode
        if mode == "auto":
            mode = "molecule" if MoleculeNeighbors.fits(H, s.pair_cutoff, s.neighbors.skin, self.r_list) else "atom"
        if mode == "molecule":
            self.nb = MoleculeNeighbors(
                sim.topology.group, sim.topology.n_group, self.r_list, H, s.pair_cutoff, s.neighbors.skin
            )
        else:  # pairs of bead atoms within the cutoff: centroid atoms within cutoff + 2 margins
            rc = s.pair_cutoff + 2.0 * self.bead_margin
            skin = min(s.neighbors.skin, max_cutoff(H) - rc - 0.002)
            if skin < 0.02:
                raise ValueError(
                    f"box too small for the bead margin: cutoff {s.pair_cutoff} + 2 x {self.bead_margin} "
                    f"leaves a list skin of {skin:.3f} nm (half the box height {max_cutoff(H):.3f} nm)"
                )
            self.nb = AtomNeighbors(sim.sys.n, H, rc, skin)
        self._nb_volume = float(volume(jnp.asarray(H)))

    def _centers(self, qc: jax.Array) -> jax.Array:
        """Return the neighbour-list group centres of the centroid qc (N, 3) [nm]."""
        return self.flex.list_centers(qc)

    def force_positions(self, q: jax.Array) -> jax.Array:
        """Return the positions (nf, N, 3) [nm] of the force beads: all beads, or the contracted ones."""
        return q if self.Pc is None else jnp.tensordot(self.T, q, axes=1)

    def extent(self, q: jax.Array) -> jax.Array:
        """Return the largest distance [nm] of a bead atom from what the list is built on.

        Its centroid list group's centre (molecule list) or its centroid (atom list).
        """
        qc = jnp.mean(q, 0)
        if self.nb.kind == "molecule":
            c = self._centers(qc)[self.flex.group]
            return jnp.max(jnp.linalg.norm(q - c[None], axis=-1))
        return jnp.max(jnp.linalg.norm(q - qc[None], axis=-1))

    def limit(self) -> float:
        """Return the largest allowed `extent` [nm]: r_list (molecule list) or bead_margin (atom list)."""
        return self.r_list if self.nb.kind == "molecule" else self.bead_margin

    def size(
        self, q: jax.Array, box: ArrayLike, factor: float = 1.2, nbr: partition.NeighborList | None = None
    ) -> partition.NeighborList:
        """Set the static sizes (molecule-list width, row capacities) for the largest force bead (host).

        Returns the centroid's neighbour list (allocated unless `nbr` is given).
        """
        qc = jnp.mean(q, 0)
        c = self._centers(qc)
        nbr = self.nb.allocate(qc, c, box) if nbr is None else nbr
        x = self.force_positions(q)
        if self.nb.kind == "molecule":
            caps = [self.nb.size(nbr, c, box, x[k], factor) for k in range(x.shape[0])]
            self.nb.cap = max(caps)
        ff = self.ff
        counts = np.array(
            [
                np.asarray(jax.jit(ff.pair_counts)(x[k], box, self.nb.candidates(nbr, c, box, x[k])[0]))
                for k in range(x.shape[0])
            ]
        )
        caps = []
        for k in {int(np.argmax(counts[:, 0])), int(np.argmax(counts[:, 1]))}:
            ff.size_rows(x[k], box, self.nb.candidates(nbr, c, box, x[k])[0], factor)
            caps.append(ff.capacity)
        ff.fit_rows(caps)
        return nbr

    # ------------------------------------------------------------------ forces
    def init(self, q: jax.Array, box: jax.Array) -> PGMBeadState:
        """Return the initial engine state: fresh induction for every force bead, the centroid's list."""
        ind0 = self.ff.init_induction()
        ind = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (self.nf,) + jnp.shape(x)), _nocount(ind0))
        ind = ind.set(count=ind0.count)
        qc = jnp.mean(q, 0)
        nbr = self.nb.allocate(qc, self._centers(qc), box)
        z, zi = jnp.zeros((), jnp.float64), jnp.zeros((), jnp.int32)
        return PGMBeadState(
            induction=ind,
            nbr=nbr,
            iters=zi,
            max_iters=zi,
            resid=z,
            overflow=jnp.zeros((), bool),
            cg_total=z,
            elec=z,
            vdw=z,
        )

    def reset_block(self, e: PGMBeadState) -> PGMBeadState:
        """Return e with the block maxima (CG iterations, residual, overflow) reset."""
        return e.set(max_iters=jnp.zeros((), jnp.int32), resid=jnp.zeros((), jnp.float64), overflow=jnp.zeros((), bool))

    def _ind_axes(self, ind: InductionState) -> Any:
        """Return the vmap axes of a batched induction state (0, the counter unbatched)."""
        return jax.tree_util.tree_map(lambda _: 0, _nocount(ind))

    def _over_beads(self, fn: Callable, x: jax.Array, ind: InductionState, n_out: int, k_ind: int) -> tuple:
        """Return fn(x_k, ind_k) for every force bead (n_out outputs; output k_ind an InductionState).

        jax.vmap over all of them, or with `bead_chunk` a lax.map over chunks of vmapped beads
        (less memory traffic per kernel for many beads).  The predictor step counter stays
        unbatched.
        """
        ax = self._ind_axes(ind)
        out_ax = tuple(ax if i == k_ind else 0 for i in range(n_out))
        nf = x.shape[0]
        c = self.chunk
        if c is None or c >= nf or nf % c:
            return jax.vmap(fn, in_axes=(0, ax), out_axes=out_ax)(x, ind)
        count = ind.count

        def split(a: jax.Array) -> jax.Array:  # (nf, ...) -> (chunks, c, ...)
            return a.reshape((nf // c, c) + a.shape[1:])

        xs = split(x)
        inds = jax.tree_util.tree_map(split, _nocount(ind))
        vf = jax.vmap(fn, in_axes=(0, ax), out_axes=out_ax)

        def body(args: tuple) -> tuple:
            """Evaluate one chunk; return its outputs (counter removed) and the new counter."""
            xc, ic = args
            out = vf(xc, ic.set(count=count))
            return tuple(o.set(count=None) if i == k_ind else o for i, o in enumerate(out)), out[k_ind].count

        out, counts = jax.lax.map(body, (xs, inds))

        def merge(a: jax.Array) -> jax.Array:  # (chunks, c, ...) -> (nf, ...)
            return a.reshape((nf,) + a.shape[2:])

        return tuple(
            jax.tree_util.tree_map(merge, o).set(count=counts[0]) if i == k_ind else merge(o) for i, o in enumerate(out)
        )

    def compute(self, q: jax.Array, box: jax.Array, e: PGMBeadState) -> tuple[jax.Array, jax.Array, PGMBeadState]:
        """Return the forces (P, N, 3) [kJ/mol/nm], U [kJ/mol] and the engine state at beads q (P, N, 3).

        The force field on every force bead with its own dipoles (the centroid's list updated),
        plus the bonded terms on every bead; with contraction U = sum_k ref(q^k) +
        (P/P') sum_k' [V - V_mono](q'^k'), forces returned through T^T.
        """
        qc = jnp.mean(q, 0)
        c = self._centers(qc)
        nbr = self.nb.update(e.nbr, qc, c, box)
        x = self.force_positions(q)
        ff, params, nb = self.ff, self.params, self.nb

        def one(xk: jax.Array, ind: InductionState) -> tuple:
            """Return the force field's energies, forces, induction and statistics on one bead."""
            cand, ovf = nb.candidates(nbr, c, box, xk)
            res = ff.compute(xk, box, cand, ind, params)
            return (
                res.energy["total"],
                res.energy["elec"],
                res.energy["vdw"],
                res.forces,
                res.induction,
                res.iterations,
                res.residual,
                res.overflow | ovf,
            )

        E, El, Ev, F, ind, it, err, ovf = self._over_beads(one, x, e.induction, 8, 4)
        ref_vg = jax.vmap(jax.value_and_grad(self.reference))
        if self.Pc is None:
            eb, gb = ref_vg(q)  # bonded terms on every bead
            f = F - gb
            U = jnp.sum(E) + jnp.sum(eb)
        else:
            em, gm = jax.vmap(jax.value_and_grad(self.monomer_nonbonded))(x)
            dE, dF = E - em, F + gm  # intermolecular remainder on the contracted beads
            er, gr = ref_vg(q)  # stiff reference on every bead
            scale = self.P / self.Pc
            f = scale * jnp.tensordot(self.T.T, dF, axes=1) - gr
            U = scale * jnp.sum(dE) + jnp.sum(er)
        itm = jnp.max(it)
        e = e.set(
            induction=ind,
            nbr=nbr,
            iters=itm,
            max_iters=jnp.maximum(e.max_iters, itm),
            resid=jnp.maximum(e.resid, jnp.max(err)),
            overflow=e.overflow | jnp.any(ovf),
            cg_total=e.cg_total + itm,
            elec=jnp.mean(El),
            vdw=jnp.mean(Ev),
        )
        return f, U, e

    # ------------------------------------------------------------------ barostat
    @property
    def nmol(self) -> int:
        """Number of molecules."""
        return self.flex.nmol

    def scale(self, q: jax.Array, s: jax.Array) -> jax.Array:
        """Return beads translated by (s - 1) times the centroid's molecular centre of mass."""
        com = self.flex.centers(jnp.mean(q, 0))
        return q + ((s - 1.0) * com)[self.flex.mol][None]

    def flag(self, e: PGMBeadState, trial: PGMBeadState) -> PGMBeadState:
        """Keep the overflow flag of a trial evaluation (the block is repeated if it overflowed)."""
        return e.set(overflow=e.overflow | trial.overflow)

    def energy(self, q: jax.Array, box: jax.Array, e: PGMBeadState) -> tuple[jax.Array, PGMBeadState]:
        """Return U [kJ/mol] at (q, box) and the engine state, the dipoles solved from the last ones.

        No predictor history update (Monte Carlo trials); the neighbour list is rebuilt.
        """
        qc = jnp.mean(q, 0)
        c = self._centers(qc)
        nbr = self.nb.update(e.nbr, qc, c, box, True)
        x = self.force_positions(q)
        ff, params, nb = self.ff, self.params, self.nb

        def one(xk: jax.Array, ind: InductionState) -> tuple:  # energy on one bead
            cand, ovf = nb.candidates(nbr, c, box, xk)
            E, ind, it, ovf2 = ff.energy(xk, box, cand, ind, params)
            return E, ind, ovf | ovf2

        E, ind, ovf = self._over_beads(one, x, e.induction, 3, 1)
        ref = jax.vmap(self.reference)
        if self.Pc is None:
            U = jnp.sum(E) + jnp.sum(ref(q))
        else:
            U = self.P / self.Pc * jnp.sum(E - jax.vmap(self.monomer_nonbonded)(x)) + jnp.sum(ref(q))
        return U, e.set(nbr=nbr, induction=ind, overflow=e.overflow | jnp.any(ovf))

    # ------------------------------------------------------------------ pressure
    def strain_derivative(self, q: jax.Array, box: jax.Array, e: PGMBeadState) -> jax.Array:
        """Return (1/P) dU/d eps (3, 3) [kJ/mol], the molecular centroid virial.

        Every bead of a molecule is translated with the molecular centre of mass of the centroid
        (the intramolecular reference does not change under such a strain); at fixed dipoles, with
        the van der Waals tail impulse term; the mean over the force beads.
        """
        ff, flex = self.ff, self.flex
        qc = jnp.mean(q, 0)
        com = flex.centers(qc)
        c = self._centers(qc)
        x = self.force_positions(q)
        mu = e.induction.mu
        P = ff._atoms(self.params)

        def W(xk: jax.Array, muk: jax.Array) -> jax.Array:
            """Return dE/d eps (3, 3) on one force bead at its dipoles muk."""
            cand, _ = self.nb.candidates(e.nbr, c, box, xk)

            def en(eps: jax.Array) -> jax.Array:  # energy at strain eps
                Fm = jnp.eye(3) + eps
                return ff.energy_fixed_mu(xk + (com @ eps.T)[flex.mol], box @ Fm.T, muk, cand, P)[0]

            return jax.grad(en)(jnp.zeros((3, 3))) - ff._vdw_tail_impulse(P, box) * jnp.eye(3)

        return jnp.mean(jax.vmap(W)(x, mu), 0)


# ----------------------------------------------------------------------------- driver
class PIMDSimulation:
    """Path-integral MD of a FlexibleSimulation's system (flexible molecules, NVT or NPT).

        sim = FlexibleSimulation(system, [tpl] * n, positions, box, MDSettings(), dt=0.00025, temperature=298)
        pi = PIMDSimulation(sim, beads=32, mode="pimd", thermostat=PILE("g", tau_centroid=0.5))
        pi.run(40000, report_every=400, traj_every=400, prefix="qwater")   # centroid trajectory qwater.nc
        pi.set_mode("trpmd"); pi.run(...)                                  # dynamics from the PIMD ensemble

    dt, temperature and settings come from `sim` (its own integrator and thermostat are not used).

    Attributes
    ----------
    engine : PGMBeads
        The bead engine.
    integ : PIMDIntegrator
        The integrator.
    state : PIMDState
        Current state.
    P : int
        Beads.
    dt : float
        Time step [ps].
    T0 : float
        Temperature [K].
    ensemble : {"nvt", "npt"}
        Ensemble.
    time_ps : float
        Simulation time [ps].
    """

    def __init__(
        self,
        sim: FlexibleSimulation,
        beads: int = 32,
        mode: str = "pimd",
        thermostat: PILE | str = PILE(),
        propagator: str = "cayley",
        contract: int | None = None,
        barostat: MonteCarloBarostat | None = None,
        bead_margin: float = 0.08,
        bead_chunk: int | str | None = "auto",
        spread: bool = True,
        seed: int = 0,
        dt: float | None = None,
        log: TextIO | None = None,
    ) -> None:
        """Set up the ring polymers of `sim`'s current configuration.

        Parameters
        ----------
        sim : FlexibleSimulation
            The system, force field, settings, temperature and (default) time step; flexible
            molecules without constraints.
        beads : int
            Number of beads P.
        mode : str
            "pimd" (PILE thermostat), "trpmd" (thermostatted RPMD) or "rpmd" (NVE).
        thermostat : PILE or str
            Path-integral thermostat settings (PILE; "pile-l" / "pile-g" for the defaults).
        propagator : str
            Free ring-polymer step: "cayley" (default) or "exact".
        contract : int or None
            Ring-polymer contraction of the intermolecular forces to this many beads (None: off).
        barostat : MonteCarloBarostat or None
            Isotropic Monte Carlo barostat (None: constant volume).
        bead_margin : float
            Largest distance of a bead atom from its centroid [nm] (enlarges the neighbour lists).
        bead_chunk : int, "auto" or None
            Force beads per vmapped chunk ("auto": 8 for more than 8 beads; None: all at once).
        spread : bool
            Draw the beads from the free ring-polymer distribution around the configuration
            (False: all beads on it).
        seed : int
            Seed of the random stream (bead spread, momenta, thermostat, barostat).
        dt : float or None
            Time step [ps] (None: sim.dt).
        log : text stream or None
            Receives the rows of the log table of `run` (diagnostics go to the logger
            "pgm_jax.md.pimd").
        """
        self.sim, self.log = sim, log
        self.engine = PGMBeads(sim, beads, contract, bead_margin, bead_chunk)
        self.P = int(beads)
        self.dt = float(sim.dt if dt is None else dt)
        self.T0 = float(sim.T0)
        self.integ = PIMDIntegrator(
            self.engine,
            np.asarray(sim.flex.masses),
            beads,
            self.T0,
            self.dt,
            mode,
            thermostat,
            propagator,
            barostat,
        )
        self.ensemble = self.integ.ensemble
        self.elements = np.array(sim.sys.elements)
        H = jnp.asarray(sim.state.box)
        pos = sim.state.dyn.position
        key = jax.random.PRNGKey(int(seed))
        k1, k2 = jax.random.split(key)
        q = (
            self.integ.ring.sample_free(k1, pos, self.integ.mass)
            if spread
            else jnp.broadcast_to(pos, (self.P,) + pos.shape)
        )
        self._size(q, H)
        self.state = self.integ.init(q, H, k2)
        self.time_ps = 0.0
        e = self.engine
        logger.info(
            f"pgm_jax PIMD: {sim.sys.nmol} molecules, {sim.sys.n} atoms, {self.P} beads"
            f"{'' if e.Pc is None else f' (intermolecular forces contracted to {e.Pc})'}, "
            f"{self.integ.thermo.describe()}, {self.ensemble.upper()}"
            f"{f' ({barostat.describe()})' if barostat is not None else ''}, "
            f"T {self.T0:g} K, dt {self.dt * 1000:g} fs, "
            f"{propagator} free ring-polymer step, {e.nb.kind} neighbour list of the centroid "
            f"(bead margin {e.bead_margin:g} nm), {sim.settings.precision} precision, device {jax.devices()[0]}"
        )

    def set_mode(self, mode: str, thermostat: PILE | str | None = None) -> None:
        """Switch between "pimd", "trpmd" and "rpmd" (same state; the conserved quantity restarts).

        Parameters
        ----------
        mode : str
            The new mode.
        thermostat : PILE, str or None
            New thermostat settings; None keeps the centroid thermostat (kind and tau_centroid)
            with the internal-mode friction of the new mode (lam = 1 for "pimd", 1/2 for "trpmd").
        """
        if thermostat is None:
            pile = self.integ.thermo.pile
            thermostat = PILE(pile.kind, pile.tau_centroid)
        self.integ.set_thermostat(mode, thermostat)
        logger.info(f"thermostat: {self.integ.thermo.describe()}")

    # ------------------------------------------------------------------ sizes and blocks
    def _size(
        self, q: jax.Array, H: ArrayLike, factor: float = 1.2, nbr: partition.NeighborList | None = None
    ) -> partition.NeighborList:
        """Size the engine for beads q (host), re-jit the integrator and return the neighbour list."""
        nbr = self.engine.size(q, H, factor, nbr)
        self._w_jit = None
        self.integ.run = jax.jit(self.integ._run)
        self.integ.forces = jax.jit(self.integ._forces)
        return nbr

    def _wrap(self, st: PIMDState) -> PIMDState:
        """Return the state with whole ring polymers shifted so that the molecular centres lie in the cell.

        The centroid's molecular centres of mass define the shift of every bead of a molecule.
        """
        flex = self.sim.flex
        H = st.box
        qc = jnp.mean(st.q, 0)
        fr = jnp.matmul(flex.centers(qc), inv3(H), precision=jax.lax.Precision.HIGHEST)
        shift = jnp.matmul(jnp.floor(fr), H, precision=jax.lax.Precision.HIGHEST)[flex.mol]
        return st.set(q=st.q - shift[None])

    def _rebuild_neighbors(self) -> None:
        """Build new neighbour lists for the current box, re-size and recompute the forces."""
        H = np.asarray(self.state.box)
        logger.info(f"step {int(self.state.step)}: neighbour lists rebuilt for volume {float(volume(H)):.3f} nm^3")
        self.engine.make_neighbors(H)
        nbr = self._size(self.state.q, H)
        redo = self.integ.forces(self.state.set(eng=self.state.eng.set(nbr=nbr)))
        self.state = redo.set(eng=redo.eng.set(induction=self.state.eng.induction))

    def advance(self, n: int) -> None:
        """Advance n steps without writing files.

        Under NPT the lists are rebuilt when the volume has drifted by 10 %, and a block that keeps
        overflowing (a box shrinking fast) is split in halves with rebuilds in between
        (driver.advance_with_rebuilds).

        Parameters
        ----------
        n : int
            Steps.
        """
        advance_with_rebuilds(
            n,
            self._advance_block,
            self._rebuild_neighbors,
            lambda: float(volume(self.state.box)) / self.engine._nb_volume,
        )

    def _run_block(self, start: PIMDState, n: int) -> PIMDState:
        """One compiled block of n ring-polymer steps from `start` (waits for the result)."""
        new = self.integ.run(start, n)
        jax.block_until_ready(new.upot)
        return new

    def _resize(self, start: PIMDState, n: int, list_bad: bool, rows_bad: bool) -> PIMDState:
        """Enlarge the capacities after an overflow in the block of n steps from `start`.

        Parameters
        ----------
        start : PIMDState
            State at the start of the failed block.
        n : int
            Steps of the block (for the log line).
        list_bad, rows_bad : bool
            Whether the neighbour list / the pair rows overflowed.

        Returns
        -------
        PIMDState
            `start` with a new neighbour list and its forces evaluated at the new sizes (the
            induced-dipole history is kept: no second history entry).
        """
        e = self.engine
        old = (e.ff.capacity, getattr(e.nb, "cap", None))
        nbr = self._size(start.q, start.box, 1.3, None if list_bad else start.eng.nbr)
        if rows_bad:  # never shrink below what overflowed
            e.ff.grow_rows(old[0])
            if getattr(e.nb, "cap", None) is not None and old[1] is not None:
                e.nb.cap = max(e.nb.cap, old[1] + 4)
            self.integ.run = jax.jit(self.integ._run)
            self.integ.forces = jax.jit(self.integ._forces)
        logger.info(
            f"{'neighbour list' if list_bad else 'row capacity'} overflow in steps {int(start.step)}-"
            f"{int(start.step) + n}: resized, repeating"
        )
        redo = self.integ.forces(start.set(eng=start.eng.set(nbr=nbr)))
        return redo.set(eng=redo.eng.set(induction=start.eng.induction))

    def _advance_block(self, n: int) -> None:
        """Advance n steps as one compiled block, re-wrap, and check the bead spread.

        driver.retry_block handles overflows; then the ring polymers are re-wrapped into the box
        and the bead spread checked against the list margin.

        Raises
        ------
        RuntimeError
            A bead atom beyond the neighbour-list margin (increase bead_margin), or a block that
            keeps overflowing.
        FloatingPointError
            A non-finite energy.
        """
        e = self.engine
        new = retry_block(
            lambda s: self._run_block(s, n),
            self.state,
            lambda s: (e.nb.failed(s.eng.nbr), bool(s.eng.overflow)),
            lambda s, lb, rb: self._resize(s, n, lb, rb),
        )
        new = self._wrap(new)
        ext = float(jax.jit(e.extent)(new.q))
        if ext > e.limit():
            raise RuntimeError(
                f"a bead atom is {ext:.3f} nm from its centroid reference, beyond the list margin "
                f"{e.limit():.3f} nm; increase bead_margin"
            )
        finite_or_raise(new.upot, new.step)
        self.state = new
        self.time_ps += n * self.dt

    # ------------------------------------------------------------------ observables
    def _est(self) -> dict:
        """Return the estimators of the current state (jitted once)."""
        if getattr(self, "_est_jit", None) is None:
            self._est_jit = jax.jit(self.integ.estimators)
        return self._est_jit(self.state)

    def observables(self) -> dict:
        """Return the observables of the current state (one row of the log table).

        step, time_ps, temperatures [K] (beads / P, centroid), epot = U / P, the primitive and
        centroid-virial kinetic energies, econs, elec, vdw [kJ/mol], volume [nm^3], density
        [g/cm^3], barostat acceptance, per-element kinetic energies [meV], molecular dipoles [D]
        (bead-averaged and per bead) and CG statistics.
        """
        st, eng = self.state, self.state.eng
        est = {k: np.asarray(v) for k, v in self._est().items()}
        out = {
            "step": int(st.step),
            "time_ps": self.time_ps,
            "temp_K": float(est["t_beads"]),
            "temp_centroid": float(est["t_centroid"]),
            "epot": float(est["epot"]),
            "ekin_prim": float(est["prim"].sum()),
            "ekin_cv": float(est["cv"].sum()),
            "econs": float(est["econs"]),
            "elec": float(eng.elec),
            "vdw": float(eng.vdw),
        }
        V = float(volume(st.box))
        out["volume_nm3"] = V
        out["density_g_cm3"] = float(np.sum(self.sim.sys.masses)) / V * AMU_NM3_TO_G_CM3
        if self.ensemble == "npt":
            out["mc_accept"] = int(st.mc[1]) / max(int(st.mc[0]), 1)
        for el in sorted(set(self.elements.tolist())):
            sel = self.elements == el
            out[f"ke_{el}_cv_meV"] = float(est["cv"][sel].mean()) * KJMOL_TO_MEV
            out[f"ke_{el}_prim_meV"] = float(est["prim"][sel].mean()) * KJMOL_TO_MEV
        M, mb = self.molecular_dipoles()
        out["dipole_D"] = float(np.mean(np.linalg.norm(M, axis=-1))) / DEBYE_E_NM
        out["dipole_bead_D"] = mb / DEBYE_E_NM
        out.update(
            {
                "cg_iter": int(eng.iters),
                "cg_iter_max": int(eng.max_iters),
                "cg_resid_max": float(eng.resid),
                "cg_mean": float(eng.cg_total) / max(int(st.step), 1),
            }
        )
        return out

    def pressure(self) -> float:
        """Return the molecular centroid-virial pressure [bar]: N_mol kT / V - tr((1/P) dU/d eps) / (3 V)."""
        if getattr(self, "_w_jit", None) is None:
            self._w_jit = jax.jit(self.engine.strain_derivative)
        st = self.state
        W = self._w_jit(st.q, st.box, st.eng)
        V = float(volume(st.box))
        return (self.sim.sys.nmol * KB * self.T0 - float(jnp.trace(W)) / 3.0) / V * BAR_PER_KJMOL_NM3

    def molecular_dipoles(self) -> tuple[np.ndarray, float]:
        """Return the bead-averaged molecular dipoles (nmol, 3) and the mean |mu_mol| over beads [e nm].

        Charges, covalent and induced dipoles of every force bead (the contracted beads with
        contraction).
        """
        if getattr(self, "_dip_jit", None) is None:
            from .dipoles import CellDipole

            cd = CellDipole(self.engine.ff)
            params = self.engine.params

            def f(q: jax.Array, box: jax.Array, mu: jax.Array) -> tuple[jax.Array, jax.Array]:
                """Return the bead-averaged molecular dipoles and the mean molecular |dipole|."""
                x = self.engine.force_positions(q)
                M = jax.vmap(lambda xk, mk: cd.molecular(xk, box, mk, params))(x, mu)
                return jnp.mean(M, 0), jnp.mean(jnp.linalg.norm(M, axis=-1))

            self._dip_jit = jax.jit(f)
        st = self.state
        M, m = self._dip_jit(st.q, st.box, st.eng.induction.mu)
        return np.asarray(M), float(m)

    def centroid(self) -> np.ndarray:
        """Return the centroid positions (N, 3) [nm] of the current state."""
        return np.asarray(jnp.mean(self.state.q, 0))

    def beads(self) -> np.ndarray:
        """Return the bead positions (P, N, 3) [nm] of the current state (ring polymers whole)."""
        return np.asarray(self.state.q)

    def box(self) -> np.ndarray:
        """Return the box (3, 3) [nm] of the current state (lattice vectors as rows)."""
        return np.asarray(self.state.box)

    # ------------------------------------------------------------------ running
    def run(
        self,
        nsteps: int,
        *,
        prefix: str = "pimd",
        report_every: int = 100,
        traj_every: int = 0,
        beads_traj_every: int = 0,
        checkpoint_every: int = 0,
        report_pressure: bool = False,
        append: bool = False,
    ) -> None:
        """Advance nsteps with output files.

        Parameters
        ----------
        nsteps : int
            Steps.
        prefix : str
            Path prefix of the files.
        report_every : int
            Steps between rows of the log table prefix.log (estimators, energies [kJ/mol], volume,
            dipoles [D], CG statistics, speed); 0: none.
        traj_every : int
            Steps between centroid frames of prefix.nc (0: none).
        beads_traj_every : int
            Steps between frames of every bead in prefix_beads.nc (P x N atoms, bead-major; 0: none).
        checkpoint_every : int
            Steps between checkpoints prefix.pimd.chk plus Amber restarts of the centroid
            prefix.rst7 (0: none; with checkpoints also at the end).
        report_pressure : bool
            Add the centroid-virial pressure [bar] to every log row.
        append : bool
            Append to existing files (a continuation).
        """
        report, traj, beads_traj, restart = report_every, traj_every, beads_traj_every, checkpoint_every
        block = block_length(nsteps, report, traj, beads_traj, restart)
        n = self.sim.sys.n
        tfile = NetCDFTrajectory(prefix + ".nc", n, append=append) if traj else None
        bfile = NetCDFTrajectory(prefix + "_beads.nc", n * self.P, append=append) if beads_traj else None
        table = LogTable(prefix + ".log", append=append, echo=self.log)
        clock = Stopwatch(int(self.state.step), self.dt)
        done = 0
        while done < nsteps:
            m = min(block, nsteps - done)
            self.advance(m)
            done += m
            step = int(self.state.step)
            if report and step % report == 0:
                obs = self.observables()
                if report_pressure:
                    obs["press_bar"] = self.pressure()
                obs["ns_per_day"] = clock.ns_per_day(step)
                table.write(obs)
            box_A = np.asarray(self.state.box) * 10.0
            if tfile is not None and step % traj == 0:
                tfile.write(self.time_ps, self.centroid() * 10.0, box_A)
            if bfile is not None and step % beads_traj == 0:
                bfile.write(self.time_ps, self.beads().reshape(-1, 3) * 10.0, box_A)
            if restart and step % restart == 0:
                self._write_checkpoint_files(prefix)
        table.close()
        if restart:
            self._write_checkpoint_files(prefix)

    # ------------------------------------------------------------------ checkpoints
    def _write_checkpoint_files(self, prefix: str) -> None:
        """Write the files of run's checkpoints: prefix.pimd.chk and the centroid restart prefix.rst7."""
        self.save_checkpoint(prefix + ".pimd.chk")
        self.write_restart(prefix + ".rst7")

    def write_restart(self, path: str) -> None:
        """Write an Amber NetCDF restart of the centroid (positions, centroid velocities, box) to `path`."""
        st = self.state
        vc = np.asarray(jnp.mean(st.p, 0) / self.integ.mass)
        write_restart(
            path,
            self.centroid() * 10.0,
            vc * 10.0,
            np.asarray(st.box) * 10.0,
            self.time_ps,
            title=f"pgm_jax PIMD centroid, {self.P} beads",
        )

    def save_checkpoint(self, path: str) -> None:
        """Write a checkpoint of the complete state (driver.write_checkpoint, kind "pimd").

        Parameters
        ----------
        path : str
            The file (prefix.pimd.chk in `run`).

        Notes
        -----
        The checkpoint holds beads, momenta, forces, induced dipoles and predictor history of
        every bead, thermostat and barostat state and the random key; continuing from it
        reproduces the run bitwise on the CPU.
        """
        st = self.state
        write_checkpoint(
            path, "pimd", {"beads": self.P, "time_ps": self.time_ps, "state": st.set(eng=st.eng.set(nbr=None))}
        )

    def load_checkpoint(self, path: str) -> None:
        """Continue from a checkpoint written by `save_checkpoint` (or a legacy pickle .pimd.chk).

        Legacy pickle checkpoints are those of pgm_jax up to commit e72c57c; the system, settings
        and number of beads must be the same.

        Parameters
        ----------
        path : str
            The checkpoint file.

        Raises
        ------
        ValueError
            Another kind of checkpoint, another number of beads, or another system.
        """
        st = self.state
        d = read_checkpoint(path, "pimd", st.set(eng=st.eng.set(nbr=None)), legacy_format=LEGACY_FORMAT)
        if int(d["beads"]) != self.P:
            raise ValueError(f"{path}: a {int(d['beads'])}-bead checkpoint, not a {self.P}-bead one")
        st = device_tree(d["state"])
        if st.mc is None:  # legacy checkpoints from before the barostat
            st = st.set(mc=jnp.zeros(4, jnp.int32), mc_dv=jnp.asarray(0.01 * float(volume(st.box)), jnp.float64))
        self.engine.make_neighbors(np.asarray(st.box))
        nbr = self._size(st.q, st.box)
        self.state = st.set(eng=st.eng.set(nbr=nbr))
        self.time_ps = float(d["time_ps"])
