"""Thermostats: the stochastic (O) part of the BAOAB-type steps of both integrators.

Contents: the base class `Thermostat` (the O-step interface), `Langevin`, `Bussi` and `GLE`
(with the `GLE.band` and `GLE.lowpass` kernels), and `make_thermostat`, which turns the names of
the command line into objects.

A thermostat acts at fixed positions on the mass-scaled momenta v = M^-1/2 p (and on its own
auxiliary momenta s, same shape as v).  Each O step is exact for its own dynamics and leaves
exp(-beta (|v|^2 + |s|^2) / 2) invariant, so together with the Hamiltonian steps the scheme
samples the canonical ensemble (up to the usual O(dt^2) splitting error).

  Langevin(friction) white noise on every degree of freedom:
                    dv = -friction v dt + sqrt(2 friction kT) dW.
  Bussi(tau)        stochastic velocity rescaling [1]_: one random factor per step scales all
                    momenta, so the total kinetic energy follows the canonical distribution
                    with relaxation time tau.
  GLE(A)            generalized Langevin equation per degree of freedom, Markovian embedding [3]_
                    d(v, s) = -A (v, s) dt + B dW with B B^T = kT (A + A^T), i.e.
                    fluctuation-dissipation holds pointwise.  With A_vv = 0 the noise reaches v only
                    through the auxiliaries, and the trajectories stay smooth in time.
                    GLE.band() is a slow-band kernel with three variables (v, s, q): friction and
                    noise act on s (and, weakly, on q), K(w) ~ a^2 g w^2 / ((w0^2 - w^2)^2 + g^2 w^2),
                    peaked at w0, small at w = 0 and at the fast (librational) frequencies.

Why this matters for pGM: the induced-dipole predictor extrapolates the dipoles along the
trajectory, and per-atom white noise makes the trajectory rough (predictor error ~
sqrt(friction) dt^1.5).  Measured on 4096 pGM waters at tol 1e-5 (docs/thermostat_ideas.md):

  thermostat        CG it. 1 fs   H~ drift 2 fs (kT/ns/dof)   D vs NVE   T 1 ps after 146 K start
  Langevin 1/ps        6.0            +0.005                   -17 %        276 K
  Bussi 1 ps           4.0            +0.003 (as NVE)        small (1)      global
  GLE.band()           4.0            +0.005                   -7.5 %        277 K

(1) Bussi & Parrinello [2]_: diffusion nearly unchanged.

Bussi is the fastest choice and perturbs dynamics least (recommended).  Use GLE.band() when every
degree of freedom should be coupled to the bath (local control: heterogeneous heating,
equilibration).  Langevin 1/ps is the engines' default (thermostat="langevin"), kept so that
results of existing inputs do not change.

The engines take a thermostat object (or one of the names of `make_thermostat` for the default
settings of each kind, or None for NVE) and a barostat object (md/barostats.py) or None.

The integrators also book the heat each O step exchanges, so H~ = E_tot + |s|^2/2 - heat
("econs" in Simulation.observables()) is conserved up to integration and induction errors, as
in NVE.

Rule for new thermostats: A and B may depend on the configuration only, never on momenta, dipole
rates, predictor errors or solver history.  The dipoles must not become thermal variables:
thermalized dipoles add (kT/2) ln det(alpha^-1 - T(x)) to the free energy.

Units: mass-scaled momenta v = p / sqrt(m) [sqrt(kJ/mol)] (|v|^2 / 2 is a kinetic energy
[kJ/mol]), kT [kJ/mol], times [ps], frictions [1/ps], frequencies [rad/ps].

References
----------
.. [1] G. Bussi, D. Donadio, M. Parrinello, J. Chem. Phys. 126, 014101 (2007).
.. [2] G. Bussi, M. Parrinello, Comput. Phys. Commun. 179, 26 (2008).
.. [3] M. Ceriotti, G. Bussi, M. Parrinello, J. Chem. Theory Comput. 6, 1170 (2010).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from jax.typing import ArrayLike


class Thermostat:
    """Base class of the thermostats: the interface of one exact O step.

    Subclasses set the class attributes `name` (the log and command-line name) and `n_aux` (number
    of auxiliary momenta per degree of freedom; 0 for Langevin and Bussi) and implement `apply`.
    Instances are plain Python objects, not pytrees: the integrators read their settings when they
    build the compiled step (a change recompiles), and only kT may be traced.

    Attributes
    ----------
    name : str
        Thermostat kind ("none" for the base class).
    n_aux : int
        Number of auxiliary momenta per degree of freedom.
    """

    name = "none"
    n_aux = 0

    def init_aux(self, key: jax.Array, shape: Sequence[int], kT: float) -> jax.Array:
        """Draw the auxiliary momenta from their stationary distribution N(0, kT).

        Parameters
        ----------
        key : jax.Array
            PRNG key.
        shape : Sequence[int]
            Shape of the mass-scaled momenta v.
        kT : float
            kB T [kJ/mol].

        Returns
        -------
        jax.Array (n_aux, *shape) float64
            Auxiliary momenta [sqrt(kJ/mol)] (not yet projected or masked; the integrators do that).
        """
        return jnp.sqrt(kT) * jax.random.normal(key, (self.n_aux,) + tuple(shape), jnp.float64)

    def apply(
        self,
        v: jax.Array,
        aux: jax.Array,
        key: jax.Array,
        h: float,
        kT: float | jax.Array,
        dof: float,
        project: Callable[[jax.Array], jax.Array],
        mask: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        """Advance the mass-scaled momenta and the auxiliaries by one exact O step of length h.

        Parameters
        ----------
        v : jax.Array
            Mass-scaled momenta v = M^-1/2 p [sqrt(kJ/mol)], any shape (the rigid integrator stacks
            translation and rotation: (2, M, 3)).
        aux : jax.Array (n_aux, *v.shape)
            Auxiliary momenta [sqrt(kJ/mol)].
        key : jax.Array
            PRNG key.
        h : float
            Length of the O step [ps] (static: a Python float, part of the compiled step).
        kT : float or jax.Array ()
            kB T [kJ/mol]; a float or a traced scalar (replica exchange: one per replica).
        dof : float
            Number of degrees of freedom N_f (used by Bussi).
        project : Callable[[jax.Array], jax.Array]
            Projection of a mass-scaled vector onto the constraint tangent space (identity without
            constraints).
        mask : jax.Array or None
            1 for real degrees of freedom, 0 for padding (e.g. the rotations of single atoms), shape of
            `v`; None: no padding.

        Returns
        -------
        v : jax.Array
            New mass-scaled momenta, shape of `v`.
        aux : jax.Array
            New auxiliary momenta, shape of `aux`.

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        raise NotImplementedError

    def describe(self) -> str:
        """Return the name for the log header (subclasses give their settings too)."""
        return self.name


class Langevin(Thermostat):
    """Langevin thermostat: white noise and friction on every degree of freedom.

        dv = -friction v dt + sqrt(2 friction kT) dW

    Attributes
    ----------
    friction : float
        Friction coefficient gamma [1/ps].
    """

    name = "langevin"

    def __init__(self, friction: float = 1.0) -> None:
        """Set up a Langevin thermostat.

        Parameters
        ----------
        friction : float
            Friction coefficient gamma [1/ps] (the inverse of the momentum relaxation time).

        Raises
        ------
        ValueError
            A negative friction.
        """
        if float(friction) < 0.0:
            raise ValueError(f"Langevin: friction must be >= 0 ({friction!r} 1/ps)")
        self.friction = float(friction)

    def apply(
        self,
        v: jax.Array,
        aux: jax.Array,
        key: jax.Array,
        h: float,
        kT: float | jax.Array,
        dof: float,
        project: Callable[[jax.Array], jax.Array],
        mask: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        """Advance v by the exact O step v -> c v + sqrt(kT (1 - c^2)) xi, c = exp(-friction h).

        The result is then projected onto the constraint tangent space and multiplied by the padding
        mask; `aux` (empty) and `dof` are not used.  Parameters and returns as in Thermostat.apply.

        Parameters
        ----------
        v : jax.Array
            Mass-scaled momenta [sqrt(kJ/mol)].
        aux : jax.Array
            Auxiliary momenta (none; passed through).
        key : jax.Array
            PRNG key.
        h : float
            Step length [ps] (static).
        kT : float or jax.Array ()
            kB T [kJ/mol].
        dof : float
            Not used.
        project : Callable[[jax.Array], jax.Array]
            Projection onto the constraint tangent space.
        mask : jax.Array or None
            Padding mask.

        Returns
        -------
        v : jax.Array
            New mass-scaled momenta.
        aux : jax.Array
            `aux`, unchanged.
        """
        c = np.exp(-self.friction * h)
        v = c * v + jnp.sqrt(kT * (1.0 - c * c)) * jax.random.normal(key, v.shape, v.dtype)
        v = project(v)
        return (v if mask is None else v * mask), aux

    def describe(self) -> str:
        """Return the name for the log header, e.g. "Langevin 1/ps"."""
        return f"Langevin {self.friction:g}/ps"


class Bussi(Thermostat):
    """Stochastic velocity rescaling [1]_ (module docstring): one random factor scales all momenta.

    The total kinetic energy K relaxes to the canonical distribution with time constant tau; the
    single factor leaves the direction of v unchanged, so constraints and padding are respected
    without a projection.

    Attributes
    ----------
    tau : float
        Relaxation time of the kinetic energy [ps].
    """

    name = "bussi"

    def __init__(self, tau: float = 1.0) -> None:
        """Set up a Bussi thermostat.

        Parameters
        ----------
        tau : float
            Relaxation time of the kinetic energy [ps].

        Raises
        ------
        ValueError
            A non-positive tau.
        """
        if float(tau) <= 0.0:
            raise ValueError(f"Bussi: tau must be > 0 ({tau!r} ps)")
        self.tau = float(tau)

    def apply(
        self,
        v: jax.Array,
        aux: jax.Array,
        key: jax.Array,
        h: float,
        kT: float | jax.Array,
        dof: float,
        project: Callable[[jax.Array], jax.Array],
        mask: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        """Rescale v by the exact Bussi factor for a step of length h.

        Parameters
        ----------
        v : jax.Array
            Mass-scaled momenta [sqrt(kJ/mol)].
        aux : jax.Array
            Auxiliary momenta (none; passed through).
        key : jax.Array
            PRNG key.
        h : float
            Step length [ps] (static).
        kT : float or jax.Array ()
            kB T [kJ/mol].
        dof : float
            Number of degrees of freedom N_f of v.
        project : Callable[[jax.Array], jax.Array]
            Not used (a scaling stays in the tangent space).
        mask : jax.Array or None
            Not used (padding stays zero).

        Returns
        -------
        v : jax.Array
            alpha v.
        aux : jax.Array
            `aux`, unchanged.

        Notes
        -----
        With K = |v|^2 / 2, K0 = N_f kT / 2, c = exp(-h / tau), R1 ~ N(0, 1) and S ~ chi^2(N_f - 1)
        (drawn as twice a Gamma((N_f - 1)/2) variate), the new kinetic energy is [1]_

            K' / K = alpha^2 = c + (1 - c) (K0 / (N_f K)) (R1^2 + S) + 2 R1 sqrt(c (1 - c) K0 / (N_f K)),

        and alpha takes the sign of R1 + sqrt(c N_f K / ((1 - c) K0)).  K is floored at 1e-30 to avoid
        0/0 for a zero start.
        """
        k1, k2 = jax.random.split(key)
        K = jnp.maximum(0.5 * jnp.sum(v * v), 1e-30)
        c = np.exp(-h / self.tau)
        f = 0.5 * dof * kT / (dof * K)  # K0 / (N_f K)
        r1 = jax.random.normal(k1, (), jnp.float64)
        rest = 2.0 * jax.random.gamma(k2, (dof - 1.0) / 2.0, dtype=jnp.float64)  # chi^2 with N_f - 1 dof
        a2 = c + (1.0 - c) * f * (r1 * r1 + rest) + 2.0 * r1 * jnp.sqrt(c * (1.0 - c) * f)
        alpha = jnp.sign(r1 + jnp.sqrt(c / ((1.0 - c) * f))) * jnp.sqrt(a2)
        return alpha * v, aux

    def describe(self) -> str:
        """Return the name for the log header, e.g. "Bussi tau 1 ps"."""
        return f"Bussi tau {self.tau:g} ps"


class GLE(Thermostat):
    """Markovian generalized Langevin thermostat with drift matrix A [3]_.

    Every degree of freedom has its momentum v and n_aux auxiliary momenta s, and (v, s) follow
    d(v, s) = -A (v, s) dt + B dW with B B^T = kT (A + A^T) (fluctuation-dissipation), so the O step
    leaves exp(-(|v|^2 + |s|^2) / (2 kT)) invariant.  The friction kernel seen by v is
    K(w) = Re[A_vv - A_vs (i w + A_ss)^-1 A_sv] (`kernel`).

        th = GLE.band()                  # slow-band kernel (module docstring)
        th = GLE.lowpass(friction=1.0)   # Langevin-like below 50 rad/ps, smooth noise

    Attributes
    ----------
    A : np.ndarray (1 + n_aux, 1 + n_aux)
        Drift matrix [1/ps], first index = the momentum.
    n_aux : int
        Number of auxiliary momenta per degree of freedom.
    label : str
        Description for the log header.
    """

    name = "gle"

    def __init__(self, A: ArrayLike, label: str = "gle") -> None:
        """Set up a GLE thermostat from its drift matrix.

        Parameters
        ----------
        A : ArrayLike (1 + n_aux, 1 + n_aux)
            Drift matrix [1/ps], first index = the momentum; the noise follows from
            fluctuation-dissipation.
        label : str
            Description for the log header.

        Raises
        ------
        ValueError
            If A is not square with at least one auxiliary, or A + A^T is not positive semidefinite
            (smallest eigenvalue below -1e-12 of max |A|).
        """
        A = np.asarray(A, float)
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 2:
            raise ValueError("A must be square with at least one auxiliary")
        if np.linalg.eigvalsh(A + A.T).min() < -1e-12 * np.abs(A).max():
            raise ValueError("A + A^T must be positive semidefinite (fluctuation-dissipation)")
        self.A, self.n_aux, self.label = A, A.shape[0] - 1, label
        self._cache = {}  # float(h) -> (T, S), filled at trace time by _propagator

    @classmethod
    def band(cls, peak: float = 3.0, center: float = 20.0, width: float = 30.0, floor: float = 0.1) -> GLE:
        """Return the slow-band GLE with three variables (v, s, q) per degree of freedom.

        K(w) ~ a^2 g w^2 / ((w0^2 - w^2)^2 + g^2 w^2) peaks at `center` (w0) at about `peak`, with
        width `width` (g) and K ~ peak width / w^2 at high frequency; `floor` sets K(0).  Friction and
        noise act on s and, weakly (the q-q drift element), on q.

        - floor = 0 gives the pure band-pass.  A is then singular: center v + a q is conserved by the
          thermostat, so zero-frequency motion (the total momentum) is never thermalized.
        - The default 0.1/ps keeps the scheme ergodic at a small cost in diffusion.

        With the defaults, pGM water thermalizes as fast as with Langevin 1/ps, the predictor keeps its
        NVE accuracy and diffusion is perturbed by about 7 %.

        Parameters
        ----------
        peak : float
            Height of the friction peak [1/ps].
        center : float
            Frequency of the peak w0 [rad/ps].
        width : float
            Width of the peak g [rad/ps].
        floor : float
            Zero-frequency friction K(0) [1/ps].

        Returns
        -------
        GLE
            The thermostat, A = [[0, a, 0], [-a, width, center], [0, -center, gq]] with
            a^2 = peak width and gq = center^2 / (a^2 / floor - width) (0 for floor = 0).

        Raises
        ------
        ValueError
            Unless 0 <= floor < peak.
        """
        a2 = peak * width
        if floor < 0 or floor >= a2 / width:
            raise ValueError("need 0 <= floor < peak")
        # gq solves K(0) = a^2 gq / (width gq + center^2) = floor
        gq = 0.0 if floor == 0 else center**2 / (a2 / floor - width)
        a = np.sqrt(a2)
        return cls(
            [[0.0, a, 0.0], [-a, width, center], [0.0, -center, gq]],
            label=f"GLE band (peak {peak:g}/ps at {center:g} rad/ps, width {width:g}, floor {floor:g}/ps)",
        )

    @classmethod
    def lowpass(cls, friction: float = 1.0, cutoff: float = 50.0) -> GLE:
        """Return the low-pass GLE with kernel K(w) = friction cutoff^2 / (cutoff^2 + w^2).

        Langevin-like friction below the cutoff frequency, smooth noise.

        Parameters
        ----------
        friction : float
            Zero-frequency friction K(0) [1/ps].
        cutoff : float
            Cutoff frequency [rad/ps].

        Returns
        -------
        GLE
            The thermostat (one auxiliary momentum per degree of freedom;
            A = [[0, a], [-a, cutoff]], a^2 = friction cutoff).
        """
        a = np.sqrt(friction * cutoff)
        return cls([[0.0, a], [-a, cutoff]], label=f"GLE low-pass ({friction:g}/ps below {cutoff:g} rad/ps)")

    def kernel(self, omega: ArrayLike) -> np.ndarray:
        """Return the friction spectrum K(w) = Re K^(i w) (host, numpy).

        Parameters
        ----------
        omega : ArrayLike
            Angular frequencies w [rad/ps] (scalar or 1-d).

        Returns
        -------
        np.ndarray (len(omega),)
            Re[A_vv - A_vs (i w + A_ss)^-1 A_sv] [1/ps].
        """
        A = self.A
        out = []
        for w in np.atleast_1d(omega):
            Kz = A[0, 0] - A[0, 1:] @ np.linalg.solve(1j * w * np.eye(self.n_aux) + A[1:, 1:], A[1:, 0])
            out.append(np.real(Kz))
        return np.array(out)

    def _propagator(self, h: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the drift propagator T = exp(-A h) and the noise factor S at unit kT (host, cached).

        S S^T = I - T T^T, from the eigen-decomposition (negative round-off eigenvalues clipped to 0).
        The noise scales as sqrt(kT), so kT may be a traced value (e.g. one per replica) while T and S
        are numpy constants computed once per step length at trace time and cached by `float(h)`.

        Parameters
        ----------
        h : float
            Step length [ps] (a Python float).

        Returns
        -------
        T : np.ndarray (1 + n_aux, 1 + n_aux)
            exp(-A h).
        S : np.ndarray (1 + n_aux, 1 + n_aux)
            Noise factor at kT = 1.
        """
        key = float(h)
        if key not in self._cache:
            T = scipy.linalg.expm(-self.A * h)
            w, V = np.linalg.eigh(np.eye(len(T)) - T @ T.T)
            S = V @ np.diag(np.sqrt(np.maximum(w, 0.0)))  # S S^T = I - T T^T
            self._cache[key] = (T, S)
        return self._cache[key]

    def apply(
        self,
        v: jax.Array,
        aux: jax.Array,
        key: jax.Array,
        h: float,
        kT: float | jax.Array,
        dof: float,
        project: Callable[[jax.Array], jax.Array],
        mask: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        """Advance (v, s) by the exact O step (v, s) -> T (v, s) + sqrt(kT) S xi.

        Afterwards every component (v and each auxiliary) is projected onto the constraint tangent
        space (vmapped over the leading axis) and multiplied by the padding mask; `dof` is not used.

        Parameters
        ----------
        v : jax.Array
            Mass-scaled momenta [sqrt(kJ/mol)].
        aux : jax.Array (n_aux, *v.shape)
            Auxiliary momenta [sqrt(kJ/mol)].
        key : jax.Array
            PRNG key.
        h : float
            Step length [ps] (static; T and S are compiled in).
        kT : float or jax.Array ()
            kB T [kJ/mol].
        dof : float
            Not used.
        project : Callable[[jax.Array], jax.Array]
            Projection onto the constraint tangent space.
        mask : jax.Array or None
            Padding mask, shape of `v`.

        Returns
        -------
        v : jax.Array
            New mass-scaled momenta.
        aux : jax.Array
            New auxiliary momenta.
        """
        T, S = self._propagator(h)
        y = jnp.concatenate([v[None], aux], 0)
        xi = jax.random.normal(key, y.shape, y.dtype)
        y = jnp.tensordot(jnp.asarray(T), y, axes=1) + jnp.sqrt(kT) * jnp.tensordot(jnp.asarray(S), xi, axes=1)
        y = jax.vmap(project)(y)
        if mask is not None:
            y = y * mask[None]
        return y[0], y[1:]

    def describe(self) -> str:
        """Return the label for the log header."""
        return self.label


THERMOSTAT_NAMES = ("langevin", "bussi", "gle", "gle-lowpass")


def make_thermostat(spec: Thermostat | str | None) -> Thermostat | None:
    """Return the thermostat object an engine uses.

    Parameters
    ----------
    spec : Thermostat, str or None
        A Thermostat instance (used as it is), None (no thermostat: NVE), or a name for the
        default settings of a kind (case-insensitive): "langevin" (Langevin(friction=1.0)), "bussi"
        (aliases "csvr", "v-rescale"; Bussi(tau=1.0)), "gle" (aliases "gle-band", "band";
        GLE.band()), "gle-lowpass" (alias "lowpass"; GLE.lowpass()).

    Returns
    -------
    Thermostat or None
        The thermostat (None for NVE).

    Raises
    ------
    ValueError
        An unknown name.
    """
    if spec is None or isinstance(spec, Thermostat):
        return spec
    s = str(spec).lower()
    if s == "langevin":
        return Langevin()
    if s in ("bussi", "csvr", "v-rescale"):
        return Bussi()
    if s in ("gle", "gle-band", "band"):
        return GLE.band()
    if s in ("gle-lowpass", "lowpass"):
        return GLE.lowpass()
    raise ValueError(
        f"thermostat: unknown name {spec!r}; use one of {', '.join(THERMOSTAT_NAMES)}, a Thermostat object "
        "(Langevin(friction), Bussi(tau), GLE...) or None (NVE)"
    )
