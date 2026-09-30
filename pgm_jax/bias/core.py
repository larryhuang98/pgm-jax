"""Define biases on collective variables and the set of biases a simulation carries.

Contents: the base class `Bias`; static biases `StaticBias` (V = fn(s)), `Harmonic` (umbrella
restraint with its centre and force constants in the state, `HarmonicState`), `UpperWall` /
`LowerWall` (PLUMED walls); well-tempered metadynamics `MetaD` (state `MetaDState`, optional hill
grid `HillGrid`); `OPES` (OPES_METAD, state `OPESState`); `BiasSet`, the biases of one simulation
with a COLVAR buffer (state `BiasState`); `as_bias_set`, which normalizes the `bias=` argument of
the engines.

A bias is a potential V(s) of the CVs s = s(pos, H) (cv.py) with a state (a pytree of device
arrays: hills, kernels, normalisation) that the MD loop updates every `pace` steps, inside the
compiled block (no host round-trips).  Forces are -dV/dpos by autodiff through V(s(pos, H)).

    StaticBias(cvs, fn)                     V = fn(s), fixed (umbrella windows, walls, model potentials)
    Harmonic(cvs, at, kappa)                V = sum_k kappa_k / 2 (s_k - at_k)^2 (PLUMED RESTRAINT)
    UpperWall / LowerWall(cvs, at, kappa)   V = kappa ((s - at) / eps)^exp beyond `at` (PLUMED walls)
    MetaD(cvs, sigma, height, pace, biasfactor)
        well-tempered metadynamics [1]_: Gaussian hills of fixed width sigma in a fixed-size
        device buffer, the height of each new hill w = height * exp(-V(s) / (kB (gamma - 1) T))
        (biasfactor gamma; None: standard metadynamics [2]_, constant height).
        F(s) = -gamma / (gamma - 1) V(s, t) + c(t).
    OPES(cvs, sigma, pace, barrier, biasfactor)
        OPES_METAD [3]_, as in PLUMED 2.8+: weighted kernel density estimate P(s) of the unbiased
        distribution from compressed truncated Gaussian kernels, bias
        V(s) = (1 - 1/gamma) kB T log(P(s) / Z + eps).  F(s) = -V(s) / (1 - 1/gamma).

Periodic CVs (Dihedral, CV.period) wrap every difference to the nearest image (cv.wrap), and
hill / kernel centres are stored in the canonical interval of the CV (CVSet.canonical).

The biases of one simulation form a `BiasSet` (the `bias=` argument of Simulation /
FlexibleSimulation, bias/toy.py), whose state (`BiasState`) also holds a COLVAR buffer (the CVs
and bias energies every `colvar` steps, written to prefix.colvar by the drivers) and the work
done on the system by bias updates (booked as heat by the engines, so econs stays conserved).
The drivers call `BiasSet.reserve` between compiled blocks (the only place where buffer shapes
change), `BiasSet.record` / `BiasSet.deposit` after the steps inside a block, and
`BiasSet.drain` at the end of a block.

    metad = MetaD([cv.Dihedral(4, 6, 8, 14)], sigma=0.35, height=1.2, pace=250, biasfactor=6.0)
    bias = BiasSet([metad, UpperWall(cv.Distance(0, 7), at=1.2, kappa=500.0)], colvar=100)

Units: energies kJ/mol, temperatures K, CV values in the CVs' units (nm, rad); steps are MD
steps.

References
----------
.. [1] A. Barducci, G. Bussi, M. Parrinello, Phys. Rev. Lett. 100, 020603 (2008).
   doi:10.1103/PhysRevLett.100.020603
.. [2] A. Laio, M. Parrinello, Proc. Natl. Acad. Sci. USA 99, 12562 (2002).
   doi:10.1073/pnas.202427399
.. [3] M. Invernizzi, M. Parrinello, J. Phys. Chem. Lett. 11, 2731 (2020).
   doi:10.1021/acs.jpclett.0c00497

See also docs/enhanced_sampling.md; related modules: bias/cv.py (CVs), bias/walkers.py
(several walkers in one program), bias/analysis.py (FES and reweighting), md/integrate.py
(`_bias_post`, where the engines call the bias after a step).
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..units import KB
from .cv import CVSet, wrap

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .cv import CV


def _vec(x: ArrayLike, d: int, name: str) -> np.ndarray:
    """Return a per-CV parameter as a float vector of length `d`.

    Parameters
    ----------
    x : ArrayLike
        A scalar (broadcast to every CV) or one value per CV.
    d : int
        Number of CVs.
    name : str
        Argument name, for the error message.

    Returns
    -------
    np.ndarray (d,)
        The values as float64.

    Raises
    ------
    ValueError
        If `x` is neither a scalar nor of size `d`.
    """
    a = np.asarray(x, float).reshape(-1)
    if a.size == 1:
        a = np.full(d, float(a[0]))
    if a.shape != (d,):
        raise ValueError(f"{name}: a scalar or one value per CV ({d})")
    return a


class Bias:
    """Base class of the biases: a potential V(s) of a set of CVs with a device-side state.

    A subclass defines `potential(state, s)` and, if it changes during the run, `init`, `update`
    (called every `pace` steps inside the compiled MD loop) and the host-side buffer management
    (`reserve`, `buffer_size`, `grow_to`).  The state is a pytree of device arrays (() for static
    biases); the bias object itself holds only static settings (changing them recompiles).  Bias
    objects are mutable only through `bind` (the temperature).

    Attributes
    ----------
    kind : str
        Short name of the bias type (class attribute; used in COLVAR column names and in saved
        states).
    pace : int
        Steps between updates [steps] (class attribute; 0: static bias).
    cvs : CVSet
        The CVs the bias acts on.
    d : int
        Number of CVs.
    temperature : float or None
        Temperature of the bias [K]; None until `bind` sets the simulation's.
    """

    kind = "bias"
    pace = 0

    def __init__(self, cvs: CV | Sequence[CV] | CVSet, temperature: float | None = None) -> None:
        """Set up the CVs and the bias temperature.

        Parameters
        ----------
        cvs : CV, sequence of CV, or CVSet
            The collective variables (cv.py).
        temperature : float, optional
            Temperature of the bias [K]; None: the simulation's, set by `bind`.
        """
        self.cvs = CVSet(cvs)
        self.d = len(self.cvs)
        self.temperature = None if temperature is None else float(temperature)

    @property
    def kT(self) -> float:
        """Thermal energy kB T of the bias [kJ/mol].

        Raises
        ------
        ValueError
            If the temperature is not known (neither `temperature=` nor `bind`).
        """
        if self.temperature is None:
            raise ValueError(f"{self.kind}: temperature unknown (pass temperature= or bind())")
        return KB * self.temperature

    def bind(self, temperature: float) -> None:
        """Set the temperature to the simulation's unless the bias has its own (called by the drivers).

        Parameters
        ----------
        temperature : float
            Temperature of the simulation [K].
        """
        if self.temperature is None:
            self.temperature = float(temperature)

    # -- device side
    def init(self) -> Any:
        """Return the initial state (a pytree of device arrays; () for a stateless bias)."""
        return ()

    def potential(self, state: Any, s: jax.Array) -> jax.Array:
        """Return V(s) for the CV vector `s` (to be defined by subclasses).

        Parameters
        ----------
        state : pytree
            The bias state (`init`).
        s : jax.Array (d,)
            CV values [CV units].

        Returns
        -------
        jax.Array ()
            Bias energy [kJ/mol].

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        raise NotImplementedError

    def update(self, state: Any, s: jax.Array, step: ArrayLike) -> Any:
        """Return the state after an update at the CV value `s` (called every `pace` steps; default: unchanged).

        Parameters
        ----------
        state : pytree
            The bias state.
        s : jax.Array (d,)
            CV values at the current positions [CV units].
        step : ArrayLike
            MD step counter at the update (int32 scalar).

        Returns
        -------
        pytree
            The new state (same structure and shapes: it is traced inside the MD loop).
        """
        return state

    def energy(self, state: Any, pos: ArrayLike, H: ArrayLike | None = None) -> jax.Array:
        """Return V(s(pos, H)) for atom positions (differentiable in `pos` and `H`).

        Parameters
        ----------
        state : pytree
            The bias state.
        pos : ArrayLike (N, 3)
            Atom positions [nm].
        H : ArrayLike (3, 3), optional
            Box, lattice vectors as rows [nm]; None: no periodic boundaries (no minimum images).

        Returns
        -------
        jax.Array ()
            Bias energy [kJ/mol].
        """
        return self.potential(state, self.cvs.values(pos, H))

    # -- host side
    def reserve(self, state: Any, n_updates: int) -> Any:
        """Return the state with room for `n_updates` more updates (host side; may change shapes).

        Parameters
        ----------
        state : pytree
            The bias state.
        n_updates : int
            Number of updates in the next compiled block.

        Returns
        -------
        pytree
            The state (the base class has no buffers and returns it unchanged).
        """
        return state

    def buffer_size(self, state: Any) -> int:
        """Return the number of slots in the state's buffers (0: no buffers)."""
        return 0

    def grow_to(self, state: Any, size: int) -> Any:
        """Return the state with its buffers padded to `size` slots (host; base class: unchanged)."""
        return state

    def describe(self) -> str:
        """Return a one-line description of the bias for the log."""
        return f"{self.kind} on {', '.join(self.cvs.names)}"

    def info(self, state: Any) -> dict:
        """Return scalars for the log (host side; empty in the base class)."""
        return {}


# ----------------------------------------------------------------------------- static biases
class StaticBias(Bias):
    """Fixed bias V = fn(s) with a JAX function `fn` of the CV vector.

    Used for umbrella windows with a fixed centre, walls and analytic model potentials.  The state
    is () and the bias never updates.

    Attributes
    ----------
    fn : callable
        fn(s) with s a jax.Array (d,) [CV units], returning a scalar [kJ/mol].
    kind : str
        The `name` given to the constructor.
    """

    kind = "static"

    def __init__(
        self,
        cvs: CV | Sequence[CV] | CVSet,
        fn: Callable[[jax.Array], ArrayLike],
        temperature: float | None = None,
        name: str = "static",
    ) -> None:
        """Set up the bias.

        Parameters
        ----------
        cvs : CV, sequence of CV, or CVSet
            The collective variables.
        fn : callable
            fn(s) -> V, a differentiable JAX function of the CV vector s (d,) [CV units], returning a
            scalar [kJ/mol].
        temperature : float, optional
            Temperature of the bias [K] (unused by the potential; None: the simulation's).
        name : str
            Name of the bias in the log and the COLVAR columns (`kind`).
        """
        super().__init__(cvs, temperature)
        self.fn = fn
        self.kind = name

    def potential(self, state: Any, s: jax.Array) -> jax.Array:
        """Return fn(s) as a float64 scalar [kJ/mol] (`state` is unused)."""
        return jnp.asarray(self.fn(s), jnp.float64)


class HarmonicState(NamedTuple):
    """State of a `Harmonic` bias (a NamedTuple pytree; both leaves are traced).

    Attributes
    ----------
    at : jax.Array (d,)
        Centre of the restraint [CV units].
    kappa : jax.Array (d,)
        Force constants [kJ/mol per CV unit^2].
    """

    at: jax.Array  # (d,) centre (CV units)
    kappa: jax.Array  # (d,) kJ/mol per CV unit^2


class Harmonic(Bias):
    """Harmonic restraint on CVs, V = sum_k kappa_k / 2 (s_k - at_k)^2.

    Differences are wrapped to the nearest image for periodic CVs.  This is PLUMED's RESTRAINT
    convention; note the factor 1/2, unlike md/restraints.py's Amber form k x^2.  Centre and force
    constants are state variables (`HarmonicState`), so umbrella windows share one compiled step
    (walkers.py, `bias_states=`) and a centre can be moved between blocks (steered MD) without
    recompiling.  The bias itself never updates (pace 0).

        h = Harmonic(cv.Dihedral(4, 6, 8, 14), at=0.0, kappa=150.0)   # kJ/mol/rad^2
        states = [h.state(at=c) for c in centres]                      # one window per walker

    Attributes
    ----------
    at : np.ndarray (d,)
        Default centre [CV units] (the state of `init`).
    kappa : np.ndarray (d,)
        Default force constants [kJ/mol per CV unit^2].
    """

    kind = "harmonic"

    def __init__(
        self,
        cvs: CV | Sequence[CV] | CVSet,
        at: ArrayLike,
        kappa: ArrayLike,
        temperature: float | None = None,
    ) -> None:
        """Set up the restraint.

        Parameters
        ----------
        cvs : CV, sequence of CV, or CVSet
            The collective variables.
        at : ArrayLike
            Centre [CV units]: a scalar (every CV) or one value per CV.
        kappa : ArrayLike
            Force constant [kJ/mol per CV unit^2]: a scalar or one value per CV.
        temperature : float, optional
            Temperature of the bias [K] (unused by the potential; None: the simulation's).

        Raises
        ------
        ValueError
            If `at` or `kappa` has neither 1 nor d values.
        """
        super().__init__(cvs, temperature)
        self.at, self.kappa = _vec(at, self.d, "at"), _vec(kappa, self.d, "kappa")

    def init(self) -> HarmonicState:
        """Return the state with the constructor's centre and force constants."""
        return HarmonicState(jnp.asarray(self.at), jnp.asarray(self.kappa))

    def state(self, at: ArrayLike | None = None, kappa: ArrayLike | None = None) -> HarmonicState:
        """Return a state with another centre and / or other force constants.

        Parameters
        ----------
        at : ArrayLike, optional
            Centre [CV units] (scalar or one per CV); None: the constructor's.
        kappa : ArrayLike, optional
            Force constants [kJ/mol per CV unit^2]; None: the constructor's.

        Returns
        -------
        HarmonicState
            The state (device arrays), e.g. for one umbrella window.

        Raises
        ------
        ValueError
            If `at` or `kappa` has neither 1 nor d values.
        """
        return HarmonicState(
            jnp.asarray(self.at if at is None else _vec(at, self.d, "at")),
            jnp.asarray(self.kappa if kappa is None else _vec(kappa, self.d, "kappa")),
        )

    def potential(self, state: HarmonicState, s: jax.Array) -> jax.Array:
        """Return sum_k kappa_k / 2 ds_k^2 [kJ/mol], ds = s - at wrapped for periodic CVs."""
        ds = self.cvs.diff(s, state.at)
        return 0.5 * jnp.sum(state.kappa * ds * ds)


class UpperWall(StaticBias):
    """Upper wall, V = sum_k kappa_k ((s_k - at_k) / eps_k)^exp for s_k > at_k, 0 below (PLUMED UPPER_WALLS).

    A `StaticBias` whose `fn` is the wall.  The subclass `LowerWall` flips the sign (`sign`, class
    attribute: +1 upper, -1 lower).  The difference s - at is not wrapped, so walls are meant for
    non-periodic CVs.

    Attributes
    ----------
    at : np.ndarray (d,)
        Wall positions [CV units].
    kappa : np.ndarray (d,)
        Energy scale [kJ/mol] (the energy at (s - at) = eps).
    exp : float
        Exponent (dimensionless).
    eps : np.ndarray (d,)
        Length scale of the wall [CV units].
    """

    sign = 1.0

    def __init__(
        self,
        cvs: CV | Sequence[CV] | CVSet,
        at: ArrayLike,
        kappa: ArrayLike,
        exp: float = 2.0,
        eps: ArrayLike = 1.0,
        temperature: float | None = None,
    ) -> None:
        """Set up the wall.

        Parameters
        ----------
        cvs : CV, sequence of CV, or CVSet
            The collective variables.
        at : ArrayLike
            Wall position [CV units]: a scalar (every CV) or one value per CV.
        kappa : ArrayLike
            Energy scale [kJ/mol]: a scalar or one value per CV.
        exp : float
            Exponent of the wall (dimensionless; 2: harmonic).
        eps : ArrayLike
            Length scale [CV units]: a scalar or one value per CV.
        temperature : float, optional
            Temperature of the bias [K] (unused by the potential; None: the simulation's).

        Raises
        ------
        ValueError
            If `at`, `kappa` or `eps` has neither 1 nor d values.
        """
        Bias.__init__(self, cvs, temperature)
        self.at, self.kappa = _vec(at, self.d, "at"), _vec(kappa, self.d, "kappa")
        self.exp, self.eps = float(exp), _vec(eps, self.d, "eps")
        self.kind = "upper_wall" if self.sign > 0 else "lower_wall"

    def fn(self, s: jax.Array) -> jax.Array:
        """Return the wall energy sum_k kappa_k max(sign (s_k - at_k), 0)^exp / eps_k^exp [kJ/mol]."""
        x = jnp.maximum(self.sign * (s - jnp.asarray(self.at)), 0.0) / jnp.asarray(self.eps)
        return jnp.sum(jnp.asarray(self.kappa) * x**self.exp)


class LowerWall(UpperWall):
    """Lower wall, V = sum_k kappa_k ((at_k - s_k) / eps_k)^exp for s_k < at_k (PLUMED LOWER_WALLS).

    Same constructor and attributes as `UpperWall`, with `sign` = -1.
    """

    sign = -1.0


# ----------------------------------------------------------------------------- metadynamics
class MetaDState(NamedTuple):
    """State of a `MetaD` bias (a NamedTuple pytree; every leaf is traced).

    M is the buffer capacity (grown on the host between blocks, which re-traces the MD step).

    Attributes
    ----------
    centers : jax.Array (M, d)
        Hill centres [CV units], periodic components in their canonical interval.
    heights : jax.Array (M,)
        Hill heights [kJ/mol] (0 for unused slots).
    steps : jax.Array (M,) int32
        MD step of each deposition (-1 for unused slots).
    n : jax.Array () int32
        Number of hills deposited (the next free slot).
    grid : jax.Array (2^d, *nodes) or None
        With a grid: V and its derivatives on the grid nodes (see `HillGrid`); None otherwise.
    """

    centers: jax.Array  # (M, d) hill centres (periodic components canonical)
    heights: jax.Array  # (M,) kJ/mol (0 for unused slots)
    steps: jax.Array  # (M,) int32 step of deposition (-1 unused)
    n: jax.Array  # hills deposited (int32)
    grid: jax.Array | None = None  # (2^d, *nodes) V and its derivatives on the grid nodes (grid=...), else None


def _h(t: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return the cubic Hermite basis on [0, 1] at `t`: (h00, h01, h10, h11).

    h00, h01 weight the values at 0 and 1, h10, h11 the slopes at 0 and 1 (slopes in units of the
    cell, i.e. derivative times the cell width):

        h00 = 2 t^3 - 3 t^2 + 1,  h01 = -2 t^3 + 3 t^2,  h10 = t^3 - 2 t^2 + t,  h11 = t^3 - t^2
    """
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1, -2 * t3 + 3 * t2, t3 - 2 * t2 + t, t3 - t2)


class HillGrid:
    """Hills summed on a grid of 1 or 2 CVs and read back by (bi)cubic Hermite interpolation.

    The grid stores, at every node, V and dV/ds_k (and d2V/ds1ds2 for 2 CVs); each new hill adds its
    exact values and derivatives there (`add`).  `value` interpolates with cubic Hermite polynomials
    per cell (a tensor product in 2D), which is C1: the forces are continuous and exactly the
    gradient of the interpolated V, so a static grid bias conserves the energy.  A periodic CV spans
    its period (from its `lo`, default -period/2; `hi` is ignored) with `bins` nodes; a non-periodic
    one has bins + 1 nodes from lo to hi and a flat bias (zero force) beyond them (keep the CV inside
    with a wall).  The grid layout is PLUMED's GRID_MIN / GRID_MAX / GRID_BIN.

    Attributes
    ----------
    d : int
        Number of CVs (1 or 2).
    periods : np.ndarray (d,)
        Period of each CV [CV units] (0: not periodic).
    bins : np.ndarray (d,) int
        Number of cells per CV.
    lo : np.ndarray (d,)
        Position of the first node [CV units].
    dx : np.ndarray (d,)
        Node spacing [CV units].
    nodes : np.ndarray (d,) int
        Number of nodes per CV (bins if periodic, bins + 1 otherwise).
    points : np.ndarray (G, d)
        Coordinates of the G = prod(nodes) nodes [CV units], in C order of `shape`.
    shape : tuple of int
        Grid shape (nodes per CV).
    """

    def __init__(self, cvs: CVSet, lo: ArrayLike, hi: ArrayLike, bins: ArrayLike) -> None:
        """Build the grid nodes.

        Parameters
        ----------
        cvs : CVSet
            The CVs of the bias (their periods and lower bounds are used).
        lo, hi : ArrayLike
            Range of each non-periodic CV [CV units] (scalars or one per CV; ignored for periodic CVs).
        bins : ArrayLike
            Number of cells per CV (scalar or one per CV).

        Raises
        ------
        ValueError
            If there are not 1 or 2 CVs, or if a node spacing is not positive (hi <= lo).
        """
        d = len(cvs)
        if d not in (1, 2):
            raise ValueError("grids for 1 or 2 CVs (more CVs: hills without a grid)")
        self.d = d
        self.periods = np.asarray(cvs.periods, float)
        lo, hi, bins = (np.broadcast_to(np.asarray(x, float), (d,)).copy() for x in (lo, hi, bins))
        self.bins = bins.astype(int)
        per = self.periods > 0
        lows = np.where(np.isnan(cvs.lows), -0.5 * self.periods, cvs.lows)
        self.lo = np.where(per, lows, lo)
        self.dx = np.where(per, self.periods / self.bins, (hi - lo) / self.bins)
        if np.any(self.dx <= 0):
            raise ValueError("grid: hi must exceed lo")
        self.nodes = np.where(per, self.bins, self.bins + 1).astype(int)
        axes = [self.lo[k] + self.dx[k] * np.arange(self.nodes[k]) for k in range(d)]
        g = np.meshgrid(*axes, indexing="ij")
        self.points = np.stack([x.reshape(-1) for x in g], 1)  # (G, d)
        self.shape = tuple(int(n) for n in self.nodes)

    def zeros(self) -> jax.Array:
        """Return an empty grid, jax.Array (2^d, *shape) float64 (V and derivatives all zero)."""
        return jnp.zeros((2**self.d,) + self.shape, jnp.float64)

    def add(self, grid: jax.Array, c: jax.Array, w: ArrayLike, sigma: ArrayLike) -> jax.Array:
        """Add the hill w exp(-|(s - c) / sigma|^2 / 2) to the node values and derivatives.

        Parameters
        ----------
        grid : jax.Array (2^d, *shape)
            Node values: [V, dV/ds1] (1 CV) or [V, dV/ds1, dV/ds2, d2V/ds1ds2] (2 CVs).
        c : jax.Array (d,)
            Hill centre [CV units].
        w : ArrayLike
            Hill height [kJ/mol].
        sigma : ArrayLike (d,)
            Hill widths [CV units].

        Returns
        -------
        jax.Array (2^d, *shape)
            The updated grid.

        Notes
        -----
        With u_k = (x_k - c_k) / sigma_k^2 (x_k - c_k wrapped for periodic CVs) and g the hill value at
        the node, dg/ds_k = -g u_k and d2g/ds1ds2 = g u_1 u_2 (the width is the same for every hill,
        so only these terms are needed).
        """
        X = jnp.asarray(self.points)
        dd = wrap(X - c[None, :], self.periods)
        u = dd / jnp.asarray(sigma) ** 2
        g = w * jnp.exp(-0.5 * jnp.sum(dd * u, 1))
        parts = [g, -g * u[:, 0]] if self.d == 1 else [g, -g * u[:, 0], -g * u[:, 1], g * u[:, 0] * u[:, 1]]
        return grid + jnp.stack(parts).reshape(grid.shape)

    def value(self, grid: jax.Array, s: jax.Array) -> jax.Array:
        """Return the interpolated V(s) [kJ/mol] (differentiable in `s`).

        Parameters
        ----------
        grid : jax.Array (2^d, *shape)
            Node values and derivatives (see `add`).
        s : jax.Array (d,)
            CV values [CV units].

        Returns
        -------
        jax.Array ()
            V at s.

        Notes
        -----
        For each CV the fractional node index f = (s - lo) / dx is split into a cell index i0 and a
        position t in [0, 1).  Periodic CVs take f modulo `bins` and wrap the upper node to 0;
        non-periodic ones clip f to [0, bins], so V is constant (flat, zero force) outside the grid.
        The derivatives stored per unit CV are multiplied by dx to become slopes per cell.
        """
        f = (s - jnp.asarray(self.lo)) / jnp.asarray(self.dx)
        per = self.periods > 0
        idx, ts = [], []
        for k in range(self.d):
            if per[k]:
                fk = jnp.mod(f[k], float(self.bins[k]))
                i0 = jnp.clip(jnp.floor(fk).astype(jnp.int32), 0, self.bins[k] - 1)
                i1 = jnp.mod(i0 + 1, self.bins[k])
                t = fk - i0
            else:
                fk = jnp.clip(f[k], 0.0, float(self.bins[k]))
                i0 = jnp.clip(jnp.floor(fk).astype(jnp.int32), 0, self.bins[k] - 1)
                i1 = i0 + 1
                t = fk - i0
            idx.append((i0, i1))
            ts.append(t)
        dx = self.dx
        if self.d == 1:
            h00, h01, h10, h11 = _h(ts[0])
            ((i0, i1),) = idx
            V, D = grid[0], grid[1] * dx[0]
            return h00 * V[i0] + h01 * V[i1] + h10 * D[i0] + h11 * D[i1]
        hx, hy = _h(ts[0]), _h(ts[1])
        out = 0.0
        for a in (0, 1):
            for b in (0, 1):
                i, j = idx[0][a], idx[1][b]
                out = out + (
                    hx[a] * hy[b] * grid[0, i, j]
                    + hx[2 + a] * hy[b] * dx[0] * grid[1, i, j]
                    + hx[a] * hy[2 + b] * dx[1] * grid[2, i, j]
                    + hx[2 + a] * hy[2 + b] * dx[0] * dx[1] * grid[3, i, j]
                )
        return out


class MetaD(Bias):
    """(Well-tempered) metadynamics with Gaussian hills.

        V(s) = sum_i w_i exp(-sum_k d_ik^2 / (2 sigma_k^2)),  d_ik = s_k - c_ik (nearest image)

    One hill every `pace` steps at the current s, with height w = height * exp(-V(s) / (kB dT)),
    dT = (biasfactor - 1) T (well-tempered [1]_; biasfactor None: standard metadynamics [2]_,
    w = height).  At long times F(s) = -gamma / (gamma - 1) V(s, t) + c(t) (`fes_factor`).

    Hills live in a device buffer of `capacity` slots, enlarged by the drivers between blocks
    (`reserve`).  Without a grid every step evaluates all slots (O(M d); unused slots have height
    0).  With grid=(lo, hi, bins) (1 or 2 CVs) the hills are also summed on a grid (`HillGrid`,
    cubic Hermite interpolation) and V is read from it, O(1) per step whatever the number of hills;
    use bins with a spacing <= sigma / 4 (interpolation error ~1e-3 of the hill height).  The hill
    list is kept for output and for c(t) (analysis.py).

    Attributes
    ----------
    grid : HillGrid or None
        The hill grid, or None (V summed over the hill list).
    sigma : np.ndarray (d,)
        Hill widths [CV units].
    height : float
        Initial hill height [kJ/mol].
    pace : int
        Steps between hills [steps].
    biasfactor : float or None
        gamma = (T + dT) / T (dimensionless); None: standard metadynamics.
    capacity : int
        Initial number of hill slots.

    References
    ----------
    .. [1] A. Barducci, G. Bussi, M. Parrinello, Phys. Rev. Lett. 100, 020603 (2008).
       doi:10.1103/PhysRevLett.100.020603
    .. [2] A. Laio, M. Parrinello, Proc. Natl. Acad. Sci. USA 99, 12562 (2002).
       doi:10.1073/pnas.202427399
    """

    kind = "metad"

    def __init__(
        self,
        cvs: CV | Sequence[CV] | CVSet,
        sigma: ArrayLike,
        height: float,
        pace: int,
        biasfactor: float | None = 10.0,
        temperature: float | None = None,
        capacity: int = 1024,
        grid: tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
    ) -> None:
        """Set up the metadynamics bias.

        Parameters
        ----------
        cvs : CV, sequence of CV, or CVSet
            The collective variables.
        sigma : ArrayLike
            Hill widths [CV units]: a scalar (every CV) or one value per CV.
        height : float
            Initial hill height [kJ/mol].
        pace : int
            Steps between depositions [steps].
        biasfactor : float, optional
            Well-tempered bias factor gamma > 1 (dimensionless); None: standard metadynamics.
        temperature : float, optional
            Temperature of the bias [K]; None: the simulation's (`bind`).
        capacity : int
            Initial number of hill slots in the device buffer.
        grid : tuple (lo, hi, bins), optional
            Hill grid for 1 or 2 CVs (see `HillGrid`: lo, hi [CV units], bins); None: no grid.

        Raises
        ------
        ValueError
            If sigma, height or pace is not positive, if biasfactor <= 1, or if the grid is invalid.
        """
        super().__init__(cvs, temperature)
        self.grid = None if grid is None else HillGrid(self.cvs, *grid)
        self.sigma = _vec(sigma, self.d, "sigma")
        if np.any(self.sigma <= 0) or height <= 0 or int(pace) < 1:
            raise ValueError("sigma, height and pace must be positive")
        if biasfactor is not None and biasfactor <= 1.0:
            raise ValueError("biasfactor must be > 1 (None: standard metadynamics)")
        self.height, self.pace = float(height), int(pace)
        self.biasfactor = None if biasfactor is None else float(biasfactor)
        self.capacity = int(capacity)

    def init(self) -> MetaDState:
        """Return an empty state with `capacity` hill slots (and an empty grid if there is one)."""
        M, d = self.capacity, self.d
        return MetaDState(
            jnp.zeros((M, d), jnp.float64),
            jnp.zeros(M, jnp.float64),
            jnp.full(M, -1, jnp.int32),
            jnp.zeros((), jnp.int32),
            None if self.grid is None else self.grid.zeros(),
        )

    def potential(self, state: MetaDState, s: jax.Array) -> jax.Array:
        """Return V(s) [kJ/mol]: interpolated from the grid if there is one, else the sum over the hills."""
        if self.grid is not None:
            return self.grid.value(state.grid, s)
        return self.hill_sum(state, s)

    def hill_sum(self, state: MetaDState, s: jax.Array) -> jax.Array:
        """Return V(s) summed over the hills (exact; the grid's reference) [kJ/mol]."""
        ds = wrap(s[None, :] - state.centers, self.cvs.periods) / jnp.asarray(self.sigma)
        return jnp.sum(state.heights * jnp.exp(-0.5 * jnp.sum(ds * ds, 1)))

    def hill_height(self, state: MetaDState, s: jax.Array) -> jax.Array:
        """Return the height of a hill deposited at `s` [kJ/mol].

        height * exp(-V(s) / (kT (gamma - 1))) for well-tempered metadynamics, `height` for the
        standard one.  V is read the same way as the potential (grid or hill sum).
        """
        if self.biasfactor is None:
            return jnp.asarray(self.height, jnp.float64)
        return self.height * jnp.exp(-self.potential(state, s) / (self.kT * (self.biasfactor - 1.0)))

    def update(self, state: MetaDState, s: jax.Array, step: ArrayLike) -> MetaDState:
        """Deposit a hill at `s` (in the next free slot, and on the grid).

        Parameters
        ----------
        state : MetaDState
            The current state.
        s : jax.Array (d,)
            CV values [CV units] (stored in canonical form).
        step : ArrayLike
            MD step of the deposition (stored as int32).

        Returns
        -------
        MetaDState
            The state with one more hill.

        Notes
        -----
        Traced inside the MD loop, so the buffer cannot grow here: a hill beyond the buffer is dropped
        by the out-of-bounds scatter while `n` still counts it, and `reserve` raises afterwards.  The
        grid (if any) receives every hill.
        """
        w = self.hill_height(state, s)
        n = state.n
        c = self.cvs.canonical(s)
        grid = None if self.grid is None else self.grid.add(state.grid, c, w, self.sigma)
        return MetaDState(
            state.centers.at[n].set(c),
            state.heights.at[n].set(w),
            state.steps.at[n].set(jnp.asarray(step, jnp.int32)),
            n + 1,
            grid,
        )

    def reserve(self, state: MetaDState, n_updates: int) -> MetaDState:
        """Return the state with room for `n_updates` more hills (host side).

        The buffer grows to max(2 M, n + n_updates + 16) slots when it is too small.

        Parameters
        ----------
        state : MetaDState
            The current state.
        n_updates : int
            Hills that the next block can deposit.

        Returns
        -------
        MetaDState
            The state, padded if needed (a new shape re-traces the MD step).

        Raises
        ------
        RuntimeError
            If more hills were deposited than the buffer holds (hills were lost).
        """
        n, M = int(state.n), state.heights.shape[0]
        if n + n_updates <= M:
            return state
        if n > M:  # n counts every deposition; writes beyond the buffer were dropped by the scatter
            raise RuntimeError("metadynamics hill buffer overflowed (hills lost)")
        return self.grow_to(state, max(2 * M, n + n_updates + 16))  # at least double, 16 spare slots

    def buffer_size(self, state: MetaDState) -> int:
        """Return the number of hill slots M."""
        return int(state.heights.shape[0])

    def grow_to(self, state: MetaDState, size: int) -> MetaDState:
        """Return the state with its hill buffer padded to `size` slots (host; unchanged if already that large)."""
        pad = int(size) - state.heights.shape[0]
        if pad <= 0:
            return state
        return MetaDState(
            jnp.concatenate([state.centers, jnp.zeros((pad, self.d))]),
            jnp.concatenate([state.heights, jnp.zeros(pad)]),
            jnp.concatenate([state.steps, jnp.full(pad, -1, jnp.int32)]),
            state.n,
            state.grid,
        )

    def fes_factor(self) -> float:
        """Return the factor in F(s) = -factor V(s) + const at long times.

        gamma / (gamma - 1) for well-tempered metadynamics, 1 for standard metadynamics (whose bias
        converges to -F only on average).
        """
        return 1.0 if self.biasfactor is None else self.biasfactor / (self.biasfactor - 1.0)

    def hills(self, state: MetaDState) -> dict[str, np.ndarray]:
        """Return the deposited hills as host arrays.

        Parameters
        ----------
        state : MetaDState
            The state.

        Returns
        -------
        dict
            "step" (n,) int32 deposition steps, "center" (n, d) centres [CV units], "height" (n,)
            heights [kJ/mol], "sigma" (d,) widths [CV units].
        """
        n = int(state.n)
        return {
            "step": np.asarray(state.steps[:n]),
            "center": np.asarray(state.centers[:n]),
            "height": np.asarray(state.heights[:n]),
            "sigma": self.sigma.copy(),
        }

    def info(self, state: MetaDState) -> dict[str, float]:
        """Return the number of hills and the last hill height [kJ/mol] for the log."""
        n = int(state.n)
        return {"hills": n, "last_height": float(state.heights[n - 1]) if n else 0.0}

    def describe(self) -> str:
        """Return a one-line description (type, height, sigma, pace, temperature)."""
        g = "standard" if self.biasfactor is None else f"well-tempered, biasfactor {self.biasfactor:g}"
        return (
            f"metadynamics on {', '.join(self.cvs.names)} ({g}, height {self.height:g} kJ/mol, sigma "
            f"{', '.join(f'{x:g}' for x in self.sigma)}, pace {self.pace}, T {self.temperature})"
        )


# ----------------------------------------------------------------------------- OPES
class OPESState(NamedTuple):
    """State of an `OPES` bias (a NamedTuple pytree; every leaf is traced).

    K is the kernel buffer capacity (grown on the host between blocks).

    Attributes
    ----------
    centers : jax.Array (K, d)
        Kernel centres [CV units], periodic components canonical.
    sigmas : jax.Array (K, d)
        Kernel widths [CV units] (1 in unused slots, to avoid division by zero).
    heights : jax.Array (K,)
        Kernel heights (weights, dimensionless; 0 in unused slots).
    nk : jax.Array () int32
        Number of kernels in use (slots 0 .. nk - 1).
    sum_w : jax.Array ()
        Sum of the deposition weights, starting at eps^(1 - 1/gamma) as in PLUMED.
    sum_w2 : jax.Array ()
        Sum of the squared deposition weights (starting at the square of the initial sum_w).
    zed : jax.Array ()
        Normalisation Z: the mean of P over the kernel centres (1 before the first deposition).
    counter : jax.Array () int32
        Number of depositions.
    merged : jax.Array () int32
        Depositions merged into an existing kernel.
    forced : jax.Array () int32 or None
        Merges forced by a full buffer (0 unless the capacity was too small); None in states
        saved before the field existed (treated as 0).
    """

    centers: jax.Array  # (K, d)
    sigmas: jax.Array  # (K, d)
    heights: jax.Array  # (K,) (0: unused slot)
    nk: jax.Array  # kernels in use (int32)
    sum_w: jax.Array  # sum of the deposition weights (incl. the initial eps^(1 - 1/gamma))
    sum_w2: jax.Array  # sum of the squared deposition weights
    zed: jax.Array  # Z: mean of P over the kernel centres
    counter: jax.Array  # depositions (int32)
    merged: jax.Array  # depositions merged into an existing kernel (int32)
    forced: jax.Array | None = None  # merges forced by a full buffer (int32; 0 unless the capacity was too small)


class OPES(Bias):
    """OPES_METAD: on-the-fly probability enhanced sampling [1]_, with the algorithm of PLUMED's OPES_METAD.

        P(s) = sum_k G_k(s) / sum_w,  G_k(s) = h_k [exp(-d_k^2 / 2) - exp(-cut^2 / 2)] for d_k < cut,
        d_k = |(s - c_k) / sigma_k|,  V(s) = (1 - 1/gamma) kB T log(P(s) / Z + eps)

    Every `pace` steps a kernel is deposited at the current s with weight w = exp(V(s) / kB T)
    (sum_w += w), width sigma = sigma0 (N_eff (d + 2) / 4)^(-1 / (d + 4)) (bandwidth rescaling;
    N_eff = (1 + sum_w)^2 / (1 + sum_w2); fixed_sigma=True keeps sigma0; never below sigma_min),
    and height w prod(sigma0 / sigma).  It is merged into the nearest kernel when that is closer
    than `compression` in units of its sigma (compression 0: off; recursive=True, PLUMED's default:
    the merged kernel is merged again while another kernel is within the threshold).  Z = mean over
    the kernel centres of P, recomputed after each deposition (O(K^2)).

    `barrier` is the expected barrier Delta E [kJ/mol]; the defaults follow PLUMED: biasfactor
    gamma = Delta E / kB T, eps = exp(-Delta E / ((1 - 1/gamma) kB T)), kernel cutoff
    sqrt(2 Delta E / ((1 - 1/gamma) kB T)).  The bias is bounded below by -Delta E; sum_w starts at
    eps^(1 - 1/gamma) (PLUMED).  The temperature-dependent parameters (`biasfactor`, `prefactor`,
    `epsilon`, `cutoff`) are properties, available once the temperature is known (`bind`).  At long
    times F(s) = -V(s) / (1 - 1/gamma) (`fes_factor`).

    Attributes
    ----------
    sigma0 : np.ndarray (d,)
        Initial kernel widths [CV units].
    sigma_min : np.ndarray (d,)
        Lower bound of the rescaled widths [CV units].
    pace : int
        Steps between depositions [steps].
    barrier : float
        Expected barrier Delta E [kJ/mol].
    compression : float
        Merge threshold in units of the kernel's sigma (0: no compression).
    recursive : bool
        Recursive merging (PLUMED's default).
    fixed_sigma : bool
        Keep sigma0 (no bandwidth rescaling).
    capacity : int
        Initial number of kernel slots.

    References
    ----------
    .. [1] M. Invernizzi, M. Parrinello, J. Phys. Chem. Lett. 11, 2731 (2020).
       doi:10.1021/acs.jpclett.0c00497
    """

    kind = "opes"

    def __init__(
        self,
        cvs: CV | Sequence[CV] | CVSet,
        sigma: ArrayLike,
        pace: int,
        barrier: float,
        biasfactor: float | None = None,
        temperature: float | None = None,
        epsilon: float | None = None,
        kernel_cutoff: float | None = None,
        compression: float = 1.0,
        sigma_min: ArrayLike | None = None,
        fixed_sigma: bool = False,
        capacity: int = 512,
        recursive: bool = True,
    ) -> None:
        """Set up the OPES bias.

        Parameters
        ----------
        cvs : CV, sequence of CV, or CVSet
            The collective variables.
        sigma : ArrayLike
            Initial kernel widths sigma0 [CV units]: a scalar (every CV) or one per CV.
        pace : int
            Steps between depositions [steps].
        barrier : float
            Expected barrier Delta E [kJ/mol].
        biasfactor : float, optional
            gamma > 1 (dimensionless); None: Delta E / kB T.
        temperature : float, optional
            Temperature of the bias [K]; None: the simulation's (`bind`).
        epsilon : float, optional
            Regularisation eps of the logarithm (dimensionless); None: exp(-Delta E / ((1 - 1/gamma) kT)).
        kernel_cutoff : float, optional
            Kernel truncation radius in units of sigma; None: sqrt(2 Delta E / ((1 - 1/gamma) kT)).
        compression : float
            Merge threshold in units of sigma (0: no merging).
        sigma_min : ArrayLike, optional
            Lower bound of the rescaled widths [CV units]; None: 0.
        fixed_sigma : bool
            Keep sigma0 (no bandwidth rescaling).
        capacity : int
            Initial number of kernel slots.
        recursive : bool
            Merge recursively, as PLUMED does by default.

        Raises
        ------
        ValueError
            If sigma, pace or barrier is not positive, or `sigma` / `sigma_min` has neither 1 nor d
            values.
        """
        super().__init__(cvs, temperature)
        self.sigma0 = _vec(sigma, self.d, "sigma")
        self.sigma_min = np.zeros(self.d) if sigma_min is None else _vec(sigma_min, self.d, "sigma_min")
        if np.any(self.sigma0 <= 0) or int(pace) < 1 or barrier <= 0:
            raise ValueError("sigma, pace and barrier must be positive")
        self.pace, self.barrier = int(pace), float(barrier)
        self._biasfactor, self._epsilon, self._cutoff = biasfactor, epsilon, kernel_cutoff
        self.compression = float(compression)
        self.recursive = bool(recursive)
        self.fixed_sigma = bool(fixed_sigma)
        self.capacity = int(capacity)

    # parameters that depend on the temperature (known after bind)
    @property
    def biasfactor(self) -> float:
        """Bias factor gamma (dimensionless): the given one, else Delta E / kB T.

        Raises
        ------
        ValueError
            If gamma <= 1.
        """
        g = self.barrier / self.kT if self._biasfactor is None else float(self._biasfactor)
        if g <= 1.0:
            raise ValueError("OPES biasfactor must be > 1")
        return g

    @property
    def prefactor(self) -> float:
        """Prefactor 1 - 1/gamma of the bias (dimensionless)."""
        return 1.0 - 1.0 / self.biasfactor

    @property
    def epsilon(self) -> float:
        """Regularisation eps (dimensionless): the given one, else exp(-Delta E / ((1 - 1/gamma) kT))."""
        if self._epsilon is not None:
            return float(self._epsilon)
        return float(np.exp(-self.barrier / (self.prefactor * self.kT)))

    @property
    def cutoff(self) -> float:
        """Kernel cutoff in units of sigma: the given one, else sqrt(2 Delta E / ((1 - 1/gamma) kT))."""
        if self._cutoff is not None:
            return float(self._cutoff)
        return float(np.sqrt(2.0 * self.barrier / (self.prefactor * self.kT)))

    def init(self) -> OPESState:
        """Return an empty state with `capacity` kernel slots.

        sum_w starts at eps^(1 - 1/gamma) and sum_w2 at its square (PLUMED); Z = 1.
        """
        K, d = self.capacity, self.d
        w0 = self.epsilon**self.prefactor
        z = jnp.zeros((), jnp.int32)
        return OPESState(
            jnp.zeros((K, d)),
            jnp.ones((K, d)),
            jnp.zeros(K),
            z,
            jnp.asarray(w0, jnp.float64),
            jnp.asarray(w0 * w0, jnp.float64),
            jnp.ones((), jnp.float64),
            z,
            z,
            z,
        )

    def _kernels(self, st: OPESState, s: jax.Array) -> jax.Array:
        """Return the truncated kernels G_k(s) of all slots, jax.Array (K,) (0 beyond the cutoff, unused slots)."""
        ds = wrap(s[None, :] - st.centers, self.cvs.periods) / st.sigmas
        n2 = jnp.sum(ds * ds, 1)
        c2 = self.cutoff**2
        # truncated Gaussian: shifted to 0 at the cutoff; the minimum keeps exp finite (and its gradient
        # zero) beyond the cutoff, where the jnp.where below zeroes the kernel anyway
        g = jnp.exp(-0.5 * jnp.minimum(n2, c2)) - np.exp(-0.5 * c2)
        return jnp.where(n2 < c2, st.heights * g, 0.0)

    def probability(self, st: OPESState, s: jax.Array) -> jax.Array:
        """Return P(s) = sum_k G_k(s) / sum_w (unnormalised KDE of the unbiased distribution; V divides it by Z)."""
        return jnp.sum(self._kernels(st, s)) / st.sum_w

    def potential(self, st: OPESState, s: jax.Array) -> jax.Array:
        """Return V(s) = (1 - 1/gamma) kT log(P(s) / Z + eps) [kJ/mol]."""
        return self.prefactor * self.kT * jnp.log(self.probability(st, s) / st.zed + self.epsilon)

    def update(self, st: OPESState, s: jax.Array, step: ArrayLike) -> OPESState:
        """Deposit a kernel at `s`: add it or merge it into the nearest kernel, then recompute Z.

        Parameters
        ----------
        st : OPESState
            The current state.
        s : jax.Array (d,)
            CV values [CV units] (mapped to canonical form first).
        step : ArrayLike
            MD step (unused: OPES does not record deposition steps).

        Returns
        -------
        OPESState
            The state after the deposition.

        Notes
        -----
        1. w = exp(V(s) / kT) with the bias before the deposition; sum_w += w, sum_w2 += w^2.
        2. Unless `fixed_sigma`: sigma = max(sigma0 (N_eff (d + 2) / 4)^(-1/(d + 4)), sigma_min);
           the height h = w prod(sigma0 / sigma) keeps the kernel's integral proportional to w.
        3. The nearest active kernel k (distance in units of sigma_k) is merged with the new one when
           it is within `compression`; with a full buffer the new kernel is merged into it anyway
           (counted in `forced`).  With `recursive`, the merged kernel is merged again (`_recursive`),
           except after a forced merge.
        4. Z = (1 / N_k) sum_k P(c_k) over the active kernels, O(K^2).
        Everything is traced (fixed shapes; branches by jnp.where / lax.while_loop).
        """
        s = self.cvs.canonical(s)
        V = self.potential(st, s)
        w = jnp.exp(V / self.kT)
        sum_w, sum_w2 = st.sum_w + w, st.sum_w2 + w * w
        counter = st.counter + 1
        sigma = jnp.asarray(self.sigma0)
        if not self.fixed_sigma:
            neff = (1.0 + sum_w) ** 2 / (1.0 + sum_w2)
            sigma = sigma * (neff * (self.d + 2.0) / 4.0) ** (-1.0 / (4.0 + self.d))
            sigma = jnp.maximum(sigma, jnp.asarray(self.sigma_min))
        h = w * jnp.prod(jnp.asarray(self.sigma0) / sigma)
        K = st.heights.shape[0]
        active = jnp.arange(K) < st.nk
        # nearest existing kernel in units of its own sigma (compression)
        ds = wrap(s[None, :] - st.centers, self.cvs.periods)
        n2 = jnp.where(active, jnp.sum((ds / st.sigmas) ** 2, 1), jnp.inf)
        k = jnp.argmin(n2)
        merge = (self.compression > 0) & (n2[k] < self.compression**2)
        full = st.nk >= K  # no free slot: merge into the nearest kernel (counted in `forced`)
        forced = full & ~merge
        merge = merge | full
        cm, sm, hm = self._merge(st.centers[k], st.sigmas[k], st.heights[k], s, sigma, h)
        slot = jnp.where(merge, k, st.nk)
        centers = st.centers.at[slot].set(jnp.where(merge, cm, s))
        sigmas = st.sigmas.at[slot].set(jnp.where(merge, sm, sigma))
        heights = st.heights.at[slot].set(jnp.where(merge, hm, h))
        nk = st.nk + jnp.where(merge, 0, 1).astype(jnp.int32)
        if self.recursive:
            centers, sigmas, heights, nk = self._recursive(centers, sigmas, heights, nk, slot, merge & ~forced)
        new = OPESState(
            centers,
            sigmas,
            heights,
            nk,
            sum_w,
            sum_w2,
            st.zed,
            counter,
            st.merged + merge.astype(jnp.int32),
            (jnp.zeros((), jnp.int32) if st.forced is None else st.forced) + forced.astype(jnp.int32),
        )
        # Z = (1 / N_k) sum_k P(c_k)
        act = jnp.arange(K) < nk
        P = jax.vmap(lambda c: self.probability(new, c))(centers)
        zed = jnp.sum(jnp.where(act, P, 0.0)) / jnp.maximum(nk, 1)
        return new._replace(zed=zed)

    def _merge(
        self, c1: jax.Array, s1: jax.Array, h1: jax.Array, c2: jax.Array, s2: jax.Array, h2: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return kernel 2 merged into kernel 1: (centre, sigma, height).

        Heights add; the centre is the height-weighted mean and sigma^2 the weighted second moment
        minus the squared mean, both computed about c1 so that periodic CVs are safe:

            h = h1 + h2,  a = h2 / h,  dc = wrap(c2 - c1),  c = c1 + a dc,
            sigma^2 = (h1 s1^2 + h2 (s2^2 + dc^2)) / h - (a dc)^2

        Centres and widths are in CV units, heights dimensionless; per-CV arrays have shape (d,).
        """
        hm = h1 + h2
        dc = wrap(c2 - c1, self.cvs.periods)
        a = h2 / hm
        cm = self.cvs.canonical(c1 + a * dc)
        # second moment about c1 minus the squared mean shift; 1e-300 keeps the sqrt (and its gradient) finite
        sm = jnp.sqrt(jnp.maximum((h1 * s1**2 + h2 * (s2**2 + dc**2)) / hm - (a * dc) ** 2, 1e-300))
        return cm, sm, hm

    def _mergeable(
        self, centers: jax.Array, sigmas: jax.Array, nk: jax.Array, g: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Return the nearest active kernel j != g to kernel g's centre and whether it is within the threshold.

        The distance is measured in units of sigma_j; the threshold is `compression`.

        Returns
        -------
        j : jax.Array () int
            Index of the nearest other kernel.
        near : jax.Array () bool
            True if its distance is below `compression`.
        """
        K = centers.shape[0]
        ds = wrap(centers[g][None, :] - centers, self.cvs.periods) / sigmas
        idx = jnp.arange(K)
        n2 = jnp.where((idx < nk) & (idx != g), jnp.sum(ds * ds, 1), jnp.inf)
        j = jnp.argmin(n2)
        return j, n2[j] < self.compression**2

    def _recursive(
        self,
        centers: jax.Array,
        sigmas: jax.Array,
        heights: jax.Array,
        nk: jax.Array,
        g: jax.Array,
        go: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Merge kernel g recursively, as PLUMED does, while another kernel is within the threshold.

        While the kernel g is within `compression` of another kernel t, g is merged into t and deleted
        (the last active kernel moves into its slot), and t becomes the new g.

        Parameters
        ----------
        centers, sigmas : jax.Array (K, d)
            Kernel centres and widths [CV units].
        heights : jax.Array (K,)
            Kernel heights.
        nk : jax.Array () int32
            Kernels in use.
        g : jax.Array () int
            Slot of the kernel that was just merged into.
        go : jax.Array () bool
            Whether to start (False: no merge happened, or it was forced by a full buffer).

        Returns
        -------
        centers, sigmas, heights, nk
            The kernel arrays after the merges.
        """
        t, near = self._mergeable(centers, sigmas, nk, g)

        def body(c: tuple) -> tuple:
            """Merge g into t, delete g, and find the next partner of the merged kernel (while_loop body)."""
            centers, sigmas, heights, nk, g, t, _ = c
            cm, sm, hm = self._merge(centers[t], sigmas[t], heights[t], centers[g], sigmas[g], heights[g])
            centers, sigmas, heights = centers.at[t].set(cm), sigmas.at[t].set(sm), heights.at[t].set(hm)
            last = nk - 1  # delete g: the last kernel takes its slot
            centers, sigmas, heights = (
                centers.at[g].set(centers[last]),
                sigmas.at[g].set(sigmas[last]),
                heights.at[g].set(heights[last]),
            )
            heights = heights.at[last].set(0.0)
            sigmas = sigmas.at[last].set(1.0)
            t = jnp.where(t == last, g, t)
            nk = nk - 1
            g = t
            t, near = self._mergeable(centers, sigmas, nk, g)
            return centers, sigmas, heights, nk, g, t, near

        # carry (centers, sigmas, heights, nk, g, t, near); loop while the merged kernel g has a partner t
        c = jax.lax.while_loop(lambda c: c[-1], body, (centers, sigmas, heights, nk, g, t, go & near))
        return c[0], c[1], c[2], c[3]

    def reserve(self, st: OPESState, n_updates: int) -> OPESState:
        """Return the state with room for the kernels of the next block (host side).

        Kernels are compressed, so the buffer is not sized for every update: it grows (at least
        doubles) when the next updates could bring it above 80 % (at most 64 new kernels assumed per
        block).  Should it fill up within a block, new kernels merge into their nearest neighbour
        (`OPESState.forced`).

        Parameters
        ----------
        st : OPESState
            The current state.
        n_updates : int
            Depositions in the next block.

        Returns
        -------
        OPESState
            The state, padded if needed (a new shape re-traces the MD step).
        """
        nk, K = int(st.nk), st.heights.shape[0]
        # at most 64 new kernels per block assumed, buffer kept below 80 % full
        if nk + min(n_updates, 64) <= 0.8 * K:
            return st
        return self.grow_to(st, max(2 * K, int((nk + min(n_updates, 64)) / 0.8) + 16))

    def buffer_size(self, st: OPESState) -> int:
        """Return the number of kernel slots K."""
        return int(st.heights.shape[0])

    def grow_to(self, st: OPESState, size: int) -> OPESState:
        """Return the state with its kernel buffer padded to `size` slots (host; unchanged if already that large)."""
        pad = int(size) - st.heights.shape[0]
        if pad <= 0:
            return st
        return st._replace(
            centers=jnp.concatenate([st.centers, jnp.zeros((pad, self.d))]),
            sigmas=jnp.concatenate([st.sigmas, jnp.ones((pad, self.d))]),
            heights=jnp.concatenate([st.heights, jnp.zeros(pad)]),
        )

    def fes_factor(self) -> float:
        """Return the factor in F(s) = -factor V(s) + const: 1 / (1 - 1/gamma)."""
        return 1.0 / self.prefactor

    def neff(self, st: OPESState) -> float:
        """Return the effective sample size N_eff = (1 + sum_w)^2 / (1 + sum_w2) (host)."""
        return float((1.0 + st.sum_w) ** 2 / (1.0 + st.sum_w2))

    def info(self, st: OPESState) -> dict[str, float]:
        """Return scalars for the log (host).

        Returns
        -------
        dict
            "kernels" (in use), "forced" (merges forced by a full buffer), "zed" (Z), "neff"
            (N_eff), "rct" = kT log(sum_w / counter) [kJ/mol] (PLUMED's rct; 0 before the first
            deposition).
        """
        c = int(st.counter)
        return {
            "kernels": int(st.nk),
            "forced": int(0 if st.forced is None else st.forced),
            "zed": float(st.zed),
            "neff": self.neff(st),
            "rct": float(self.kT * np.log(float(st.sum_w) / max(c, 1))) if c else 0.0,
        }

    def describe(self) -> str:
        """Return a one-line description (barrier, biasfactor, sigma0, pace, compression, temperature)."""
        return (
            f"OPES_METAD on {', '.join(self.cvs.names)} (barrier {self.barrier:g} kJ/mol, biasfactor "
            f"{self.biasfactor:.3g}, sigma0 {', '.join(f'{x:g}' for x in self.sigma0)}, pace {self.pace}, "
            f"compression {self.compression:g}{', fixed sigma' if self.fixed_sigma else ''}, T {self.temperature})"
        )


# ----------------------------------------------------------------------------- the set of biases
class BiasState(NamedTuple):
    """State of a `BiasSet` (a NamedTuple pytree, the `bias` field of the MD state; every leaf is traced).

    Attributes
    ----------
    parts : tuple
        One state per bias, in the order of `BiasSet.biases`.
    log : jax.Array (C, ncol)
        COLVAR buffer: rows (step, CVs of each bias, V of each bias [kJ/mol]); C grows on the host.
    nlog : jax.Array () int32
        Rows filled since the last `BiasSet.drain`.
    work : jax.Array ()
        Work done on the system by the bias updates, sum of V_new(x) - V_old(x) at the updates
        [kJ/mol] (accumulated by the engines, md/integrate.py).
    """

    parts: tuple  # one state per bias
    log: jax.Array  # (C, 1 + sum_d + n_bias) COLVAR rows: step, CVs of each bias, V of each bias
    nlog: jax.Array  # rows filled since the last drain (int32)
    work: jax.Array  # sum of V_new(x) - V_old(x) at the updates (kJ/mol)


class BiasSet:
    """The biases of one simulation, with a COLVAR buffer on the device.

    energy(state, pos, H) is the sum of the biases.  After each MD step (or each stride, see
    `stride`) the engines call `record` (a COLVAR row every `colvar` steps; 0: none) and, when
    `due`, `deposit` (the updates due at this step, step % pace == 0).  The drivers call `reserve`
    between blocks (buffers sized for the block) and `drain` to collect the COLVAR rows.  The
    object holds only static settings; the state (`BiasState`) is a pytree.

        bias = BiasSet([MetaD(phi, sigma=0.35, height=1.2, pace=250), UpperWall(d, 1.2, 500.0)], colvar=100)
        sim = Simulation(..., bias=bias)

    Attributes
    ----------
    biases : list of Bias
        The biases.
    colvar : int
        Steps between COLVAR rows [steps] (0: none).
    ncol : int
        Columns of a COLVAR row: 1 + (number of CVs of all biases) + (number of biases).
    """

    def __init__(self, biases: Bias | Sequence[Bias] | BiasSet, colvar: int = 0) -> None:
        """Set up the bias set.

        Parameters
        ----------
        biases : Bias, sequence of Bias, or BiasSet
            The biases (a BiasSet is copied, keeping its `colvar` unless one is given).
        colvar : int
            Steps between COLVAR rows [steps] (0: none).

        Raises
        ------
        ValueError
            If there are no biases.
        TypeError
            If an element is not a `Bias`.
        """
        if isinstance(biases, BiasSet):
            colvar = colvar or biases.colvar
            biases = biases.biases
        if isinstance(biases, Bias):
            biases = [biases]
        self.biases = list(biases)
        if not self.biases:
            raise ValueError("no biases")
        for b in self.biases:
            if not isinstance(b, Bias):
                raise TypeError(f"not a bias: {b!r}")
        self.colvar = int(colvar)
        self.ncol = 1 + sum(b.d for b in self.biases) + len(self.biases)  # step, CVs of each bias, V of each bias

    def bind(self, temperature: float) -> None:
        """Bind every bias to the simulation's temperature [K] (see `Bias.bind`)."""
        for b in self.biases:
            b.bind(temperature)

    def check(self, n_atoms: int) -> None:
        """Check that every CV atom index is below `n_atoms` (raises ValueError, cv.CVSet.check)."""
        for b in self.biases:
            b.cvs.check(n_atoms)

    @property
    def dynamic(self) -> bool:
        """Whether any bias updates during the run (pace > 0)."""
        return any(b.pace > 0 for b in self.biases)

    @property
    def stride(self) -> int:
        """Steps between possible COLVAR rows or updates: gcd of `colvar` and the paces (0: none).

        The engines run plain inner loops of `stride` steps and call the bias between them
        (md/integrate.strided_loop), so the steps in between carry no conditional.
        """
        import math

        s = 0
        for x in [self.colvar] + [b.pace for b in self.biases]:
            if x > 0:
                s = math.gcd(s, int(x))
        return s

    def init(self, log_rows: int = 64) -> BiasState:
        """Return the initial state: each bias's `init`, an empty COLVAR buffer of `log_rows` rows, zero work."""
        return BiasState(
            tuple(b.init() for b in self.biases),
            jnp.zeros((int(log_rows), self.ncol), jnp.float64),
            jnp.zeros((), jnp.int32),
            jnp.zeros((), jnp.float64),
        )

    # -- device side
    def cv_values(self, pos: ArrayLike, H: ArrayLike | None = None) -> tuple[jax.Array, ...]:
        """Return the CV vector of each bias, a tuple of jax.Array (d_b,) [CV units].

        Parameters
        ----------
        pos : ArrayLike (N, 3)
            Atom positions [nm].
        H : ArrayLike (3, 3), optional
            Box, lattice vectors as rows [nm]; None: no periodic boundaries.
        """
        return tuple(b.cvs.values(pos, H) for b in self.biases)

    def energy(self, state: BiasState, pos: ArrayLike, H: ArrayLike | None = None) -> jax.Array:
        """Return the total bias energy sum_b V_b(s_b(pos, H)) [kJ/mol] (differentiable in `pos` and `H`).

        Parameters
        ----------
        state : BiasState
            The state.
        pos : ArrayLike (N, 3)
            Atom positions [nm].
        H : ArrayLike (3, 3), optional
            Box, lattice vectors as rows [nm]; None: no periodic boundaries.
        """
        e = jnp.zeros((), jnp.float64)
        for b, p in zip(self.biases, state.parts):
            e = e + b.energy(p, pos, H)
        return e

    def energies(self, state: BiasState, pos: ArrayLike, H: ArrayLike | None = None) -> jax.Array:
        """Return the energy of each bias, jax.Array (n_bias,) [kJ/mol] (arguments as `energy`)."""
        return jnp.stack([b.energy(p, pos, H) for b, p in zip(self.biases, state.parts)])

    def due(self, step: ArrayLike) -> jax.Array:
        """Return whether any bias updates at this step (a traced bool; step % pace == 0).

        `step` is the step counter after it was incremented by the step just taken.
        """
        d = jnp.zeros((), bool)
        for b in self.biases:
            if b.pace > 0:
                d = d | (step % b.pace == 0)
        return d

    def record(self, state: BiasState, pos: ArrayLike, H: ArrayLike | None, step: ArrayLike) -> BiasState:
        """Append a COLVAR row (step, CVs, bias energies) if step % colvar == 0.

        Parameters
        ----------
        state : BiasState
            The state.
        pos : ArrayLike (N, 3)
            Atom positions [nm].
        H : ArrayLike (3, 3) or None
            Box, lattice vectors as rows [nm]; None: no periodic boundaries.
        step : ArrayLike
            Step counter (traced int).

        Returns
        -------
        BiasState
            The state, with the row at index `nlog` and `nlog` incremented if a row was due.  The
            energies are those of the current bias states (before any update at this step).

        Notes
        -----
        Traced: the condition is a `lax.cond`.  The buffer must have room (`reserve`); a row beyond it
        is dropped by the scatter.
        """
        if self.colvar <= 0:
            return state

        def rec(st: BiasState) -> BiasState:
            """Write the COLVAR row at index nlog and increment nlog."""
            S = self.cv_values(pos, H)
            V = [b.potential(p, s) for b, p, s in zip(self.biases, st.parts, S)]
            row = jnp.concatenate([jnp.asarray(step, jnp.float64)[None]] + list(S) + [jnp.stack(V)])
            return st._replace(log=st.log.at[st.nlog].set(row), nlog=st.nlog + 1)

        return jax.lax.cond(step % self.colvar == 0, rec, lambda st: st, state)

    def deposit(self, state: BiasState, pos: ArrayLike, H: ArrayLike | None, step: ArrayLike) -> BiasState:
        """Apply the updates due at this step (each bias with its own pace, at its CVs of `pos`).

        Parameters
        ----------
        state : BiasState
            The state.
        pos : ArrayLike (N, 3)
            Atom positions [nm].
        H : ArrayLike (3, 3) or None
            Box, lattice vectors as rows [nm]; None: no periodic boundaries.
        step : ArrayLike
            Step counter (traced int).

        Returns
        -------
        BiasState
            The state with the updated parts (`work` is not changed here; the engines book it).

        Notes
        -----
        One `lax.cond` per dynamic bias.
        """
        parts = list(state.parts)
        for k, b in enumerate(self.biases):
            if b.pace > 0:
                s = b.cvs.values(pos, H)
                parts[k] = jax.lax.cond(
                    step % b.pace == 0, lambda p, s=s, b=b: b.update(p, s, step), lambda p: p, parts[k]
                )
        return state._replace(parts=tuple(parts))

    # -- host side
    def reserve(self, state: BiasState, nsteps: int, step0: int = 0) -> BiasState:
        """Return the state with buffers large enough for `nsteps` more steps (host side).

        Each dynamic bias reserves nsteps // pace + 1 updates, and the COLVAR buffer grows (at least
        doubles) when it cannot hold nsteps // colvar + 1 more rows.

        Parameters
        ----------
        state : BiasState
            The state.
        nsteps : int
            Steps in the next block [steps].
        step0 : int
            Unused.

        Returns
        -------
        BiasState
            The state, padded if needed (new shapes re-trace the MD step).
        """
        parts = tuple(
            b.reserve(p, (nsteps // b.pace + 1) if b.pace > 0 else 0) for b, p in zip(self.biases, state.parts)
        )
        state = state._replace(parts=parts)
        if self.colvar > 0:
            need = int(state.nlog) + nsteps // self.colvar + 1
            C = state.log.shape[0]
            if need > C:
                state = state._replace(log=jnp.concatenate([state.log, jnp.zeros((max(need, 2 * C) - C, self.ncol))]))
        return state

    def reserve_many(self, states: Sequence[BiasState], nsteps: int) -> list[BiasState]:
        """Reserve for several states (walkers), then pad them to common buffer sizes so they can be stacked.

        Parameters
        ----------
        states : sequence of BiasState
            One state per walker (unbatched).
        nsteps : int
            Steps in the next block [steps] (per walker).

        Returns
        -------
        list of BiasState
            The states with equal buffer shapes (hill / kernel buffers and COLVAR buffer).
        """
        states = [self.reserve(st, nsteps) for st in states]
        parts = []
        for k, b in enumerate(self.biases):
            size = max(b.buffer_size(st.parts[k]) for st in states)
            parts.append([b.grow_to(st.parts[k], size) for st in states])
        C = max(st.log.shape[0] for st in states)
        out = []
        for w, st in enumerate(states):
            log = (
                st.log
                if st.log.shape[0] == C
                else jnp.concatenate([st.log, jnp.zeros((C - st.log.shape[0], self.ncol))])
            )
            out.append(st._replace(parts=tuple(p[w] for p in parts), log=log))
        return out

    def drain(self, state: BiasState) -> tuple[np.ndarray, BiasState]:
        """Return the COLVAR rows since the last drain and the state with an empty buffer (host side).

        Returns
        -------
        rows : np.ndarray (n, ncol)
            The rows (step, CVs of each bias, V of each bias [kJ/mol]); columns as `columns`.
        state : BiasState
            The state with nlog = 0.
        """
        n = int(state.nlog)
        rows = np.asarray(state.log[:n])
        return rows, state._replace(nlog=jnp.zeros((), jnp.int32))

    def columns(self) -> list[str]:
        """Return the COLVAR column names: "step", the CV names of each bias, "bias{k}_{kind}" per bias."""
        cols = ["step"]
        for _k, b in enumerate(self.biases):
            cols += [f"{n}" for n in b.cvs.names]
        cols += [f"bias{k}_{b.kind}" for k, b in enumerate(self.biases)]
        return cols

    def describe(self) -> str:
        """Return the descriptions of the biases, joined by "; "."""
        return "; ".join(b.describe() for b in self.biases)

    def info(self, state: BiasState) -> dict[str, float]:
        """Return the log scalars of the biases (keys numbered by bias index only when several biases report)."""
        infos = [(k, b.info(p)) for k, (b, p) in enumerate(zip(self.biases, state.parts))]
        infos = [(k, i) for k, i in infos if i]
        out = {}
        for k, i in infos:  # keys numbered by bias only when several biases report
            for key, v in i.items():
                out[key if len(infos) == 1 else f"{key}{k}"] = v
        return out

    def save(self, state: BiasState, path: str) -> None:
        """Write the bias state (hills, kernels, normalisation) to a file (host arrays, pickle).

        The COLVAR buffer is not saved.  The file holds {"kinds": [bias kinds], "state": BiasState}.

        Parameters
        ----------
        state : BiasState
            The state.
        path : str
            Output file (conventionally prefix.bias).
        """
        host = jax.tree_util.tree_map(np.asarray, state._replace(log=np.zeros((0, self.ncol)), nlog=np.int32(0)))
        with open(path, "wb") as fh:
            pickle.dump({"kinds": [b.kind for b in self.biases], "state": host}, fh)

    def load(self, path: str) -> BiasState:
        """Return a state written by `save` (same biases); the COLVAR buffer starts empty (64 rows).

        Parameters
        ----------
        path : str
            File written by `save` (a pickle: load only trusted files).

        Returns
        -------
        BiasState
            The state as device arrays.

        Raises
        ------
        ValueError
            If the file's bias kinds differ from this set's.
        """
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        if d["kinds"] != [b.kind for b in self.biases]:
            raise ValueError(f"{path}: biases {d['kinds']} differ from {[b.kind for b in self.biases]}")
        st = jax.tree_util.tree_map(jnp.asarray, d["state"])
        return st._replace(log=jnp.zeros((64, self.ncol)), nlog=jnp.zeros((), jnp.int32))


def as_bias_set(x: Bias | Sequence[Bias] | BiasSet | None, colvar: int = 0) -> BiasSet | None:
    """Return `x` as a BiasSet (the `bias=` argument of the engines), or None.

    Parameters
    ----------
    x : Bias, sequence of Bias, BiasSet, or None
        The biases.  A BiasSet is returned as it is (not copied); its `colvar` is set to `colvar`
        if it had none.
    colvar : int
        Steps between COLVAR rows [steps] for a new set (0: none).

    Returns
    -------
    BiasSet or None
        None if `x` is None.
    """
    if x is None:
        return None
    if isinstance(x, BiasSet):
        if colvar and not x.colvar:
            x.colvar = int(colvar)
        return x
    return BiasSet(x, colvar)
