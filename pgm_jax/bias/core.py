"""Biases on collective variables: static potentials (umbrellas, walls, model potentials),
well-tempered metadynamics and OPES, and the set of biases a simulation carries.

A bias is a potential V(s) of the CVs s = s(pos, H) (cv.py) with a state (a pytree of device
arrays: hills, kernels, normalisation) that the MD loop updates every `pace` steps, inside the
compiled block (no host round-trips).  Forces are -dV/dpos by autodiff through V(s(pos, H)).

  StaticBias(cvs, fn)   V = fn(s), fixed (umbrella windows, walls, analytic model potentials)
  Harmonic(cvs, at, kappa)                V = sum_k kappa_k / 2 (s_k - at_k)^2 (PLUMED RESTRAINT)
  UpperWall / LowerWall(cvs, at, kappa)   V = kappa ((s - at) / eps)^exp beyond `at` (PLUMED walls)
  MetaD(cvs, sigma, height, pace, biasfactor)
      well-tempered metadynamics (Barducci, Bussi, Parrinello, PRL 100, 020603 (2008)): Gaussian
      hills of fixed width sigma in a fixed-size device buffer, the height of each new hill
      w = height * exp(-V(s) / (kB (gamma - 1) T)) (biasfactor gamma; None: standard metadynamics,
      constant height).  F(s) = -gamma / (gamma - 1) V(s, t) + c(t).
  OPES(cvs, sigma, pace, barrier, biasfactor)
      OPES_METAD (Invernizzi & Parrinello, JPCL 11, 2731 (2020)), as in PLUMED 2.8+: weighted
      kernel density estimate P(s) of the unbiased distribution from compressed truncated Gaussian
      kernels, bias V(s) = (1 - 1/gamma) kB T log(P(s) / Z + eps).  F(s) = -V(s) / (1 - 1/gamma).

Periodic CVs (Dihedral, CV.period) wrap every difference to the nearest image.  Energies kJ/mol,
temperatures K; CV units are the CVs' (nm, rad).

The biases of one simulation form a `BiasSet` (the `bias=` argument of Simulation /
FlexibleSimulation, bias/toy.py), whose state (`BiasState`) also holds a COLVAR buffer (the CVs and
bias energies every `colvar` steps, written to prefix.colvar by the drivers) and the work done on
the system by bias updates (booked as heat, so econs stays conserved)."""
from __future__ import annotations

import pickle
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .cv import CVSet, wrap

KB = 0.0083144626181532                  # kJ/mol/K


def _vec(x, d, name):
    a = np.asarray(x, float).reshape(-1)
    if a.size == 1:
        a = np.full(d, float(a[0]))
    if a.shape != (d,):
        raise ValueError(f"{name}: a scalar or one value per CV ({d})")
    return a


class Bias:
    """Base class.  cvs: a CV, a list of CVs or a CVSet.  pace: steps between updates (0: static).
    temperature: of the bias (K; None: the simulation's, set by bind())."""
    kind = "bias"
    pace = 0

    def __init__(self, cvs, temperature=None):
        self.cvs = CVSet(cvs)
        self.d = len(self.cvs)
        self.temperature = None if temperature is None else float(temperature)

    @property
    def kT(self):
        if self.temperature is None:
            raise ValueError(f"{self.kind}: temperature unknown (pass temperature= or bind())")
        return KB * self.temperature

    def bind(self, temperature: float) -> None:
        """Called by the drivers: the simulation's temperature unless the bias has its own."""
        if self.temperature is None:
            self.temperature = float(temperature)

    # -- device side
    def init(self):
        """Initial state (pytree)."""
        return ()

    def potential(self, state, s):
        """V(s) kJ/mol for the CV vector s (d,)."""
        raise NotImplementedError

    def update(self, state, s, step):
        """State after an update at CV value s (called every `pace` steps)."""
        return state

    def energy(self, state, pos, H=None):
        return self.potential(state, self.cvs.values(pos, H))

    # -- host side
    def reserve(self, state, n_updates: int):
        """State with room for n_updates more updates (host; may change shapes)."""
        return state

    def buffer_size(self, state) -> int:
        """Size of the state's buffers (0: none)."""
        return 0

    def grow_to(self, state, size: int):
        """The state with its buffers padded to `size` slots (host)."""
        return state

    def describe(self) -> str:
        return f"{self.kind} on {', '.join(self.cvs.names)}"

    def info(self, state) -> dict:
        """Scalars for the log (host)."""
        return {}


# ----------------------------------------------------------------------------- static biases
class StaticBias(Bias):
    """V = fn(s) with fn a JAX function of the CV vector s (d,) -> kJ/mol."""
    kind = "static"

    def __init__(self, cvs, fn, temperature=None, name: str = "static"):
        super().__init__(cvs, temperature)
        self.fn = fn
        self.kind = name

    def potential(self, state, s):
        return jnp.asarray(self.fn(s), jnp.float64)


class HarmonicState(NamedTuple):
    at: jnp.ndarray                 # (d,) centre (CV units)
    kappa: jnp.ndarray              # (d,) kJ/mol per CV unit^2


class Harmonic(Bias):
    """V = sum_k kappa_k / 2 (s_k - at_k)^2, differences wrapped for periodic CVs (PLUMED RESTRAINT;
    note the factor 1/2, unlike md/restraints.py's Amber form k x^2).  Centre and force constants
    are state variables (HarmonicState): umbrella windows can share one compiled step (walkers.py)
    and a centre can be moved between blocks (steered MD) without recompiling."""
    kind = "harmonic"

    def __init__(self, cvs, at, kappa, temperature=None):
        super().__init__(cvs, temperature)
        self.at, self.kappa = _vec(at, self.d, "at"), _vec(kappa, self.d, "kappa")

    def init(self):
        return HarmonicState(jnp.asarray(self.at), jnp.asarray(self.kappa))

    def state(self, at=None, kappa=None) -> HarmonicState:
        """A state with another centre and / or force constants."""
        return HarmonicState(jnp.asarray(self.at if at is None else _vec(at, self.d, "at")),
                             jnp.asarray(self.kappa if kappa is None else _vec(kappa, self.d, "kappa")))

    def potential(self, state, s):
        ds = self.cvs.diff(s, state.at)
        return 0.5 * jnp.sum(state.kappa * ds * ds)


class UpperWall(StaticBias):
    """V = sum_k kappa_k ((s_k - at_k) / eps_k)^exp for s_k > at_k (PLUMED UPPER_WALLS)."""
    sign = 1.0

    def __init__(self, cvs, at, kappa, exp: float = 2.0, eps: float = 1.0, temperature=None):
        Bias.__init__(self, cvs, temperature)
        self.at, self.kappa = _vec(at, self.d, "at"), _vec(kappa, self.d, "kappa")
        self.exp, self.eps = float(exp), _vec(eps, self.d, "eps")
        self.kind = "upper_wall" if self.sign > 0 else "lower_wall"

    def fn(self, s):
        x = jnp.maximum(self.sign * (s - jnp.asarray(self.at)), 0.0) / jnp.asarray(self.eps)
        return jnp.sum(jnp.asarray(self.kappa) * x ** self.exp)


class LowerWall(UpperWall):
    """V = sum_k kappa_k ((at_k - s_k) / eps_k)^exp for s_k < at_k (PLUMED LOWER_WALLS)."""
    sign = -1.0


# ----------------------------------------------------------------------------- metadynamics
class MetaDState(NamedTuple):
    centers: jnp.ndarray            # (M, d) hill centres (periodic components canonical)
    heights: jnp.ndarray            # (M,) kJ/mol (0 for unused slots)
    steps: jnp.ndarray              # (M,) int32 step of deposition (-1 unused)
    n: jnp.ndarray                  # hills deposited (int32)
    grid: jnp.ndarray = None        # (2^d, *nodes) V and its derivatives on the grid nodes (grid=...), else None


def _h(t):
    """Cubic Hermite basis on [0, 1]: (h00, h01, h10, h11) for the values at 0, 1 and the slopes at
    0, 1 (slopes in units of the cell)."""
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1, -2 * t3 + 3 * t2, t3 - 2 * t2 + t, t3 - t2)


class HillGrid:
    """Hills summed on a grid of 1 or 2 CVs: V, dV/ds_k (and d2V/ds1ds2) at the nodes, evaluated by
    (bi)cubic Hermite interpolation (C1: forces continuous, and exactly the gradient of the
    interpolated V, so a static bias conserves the energy).  lo, hi, bins per CV; a periodic CV spans
    its period (lo, bins; hi ignored), non-periodic ones have bins + 1 nodes from lo to hi and a flat
    bias (zero force) beyond them (keep the CV inside with a wall)."""

    def __init__(self, cvs, lo, hi, bins):
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
        self.points = np.stack([x.reshape(-1) for x in g], 1)          # (G, d)
        self.shape = tuple(int(n) for n in self.nodes)

    def zeros(self):
        return jnp.zeros((2 ** self.d,) + self.shape, jnp.float64)

    def add(self, grid, c, w, sigma):
        """Add the hill w exp(-|(s - c) / sigma|^2 / 2) to the node values and derivatives."""
        X = jnp.asarray(self.points)
        dd = wrap(X - c[None, :], self.periods)
        u = dd / jnp.asarray(sigma) ** 2
        g = w * jnp.exp(-0.5 * jnp.sum(dd * u, 1))
        parts = [g, -g * u[:, 0]] if self.d == 1 else [g, -g * u[:, 0], -g * u[:, 1], g * u[:, 0] * u[:, 1]]
        return grid + jnp.stack(parts).reshape(grid.shape)

    def value(self, grid, s):
        """Interpolated V(s)."""
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
            (i0, i1), = idx
            V, D = grid[0], grid[1] * dx[0]
            return h00 * V[i0] + h01 * V[i1] + h10 * D[i0] + h11 * D[i1]
        hx, hy = _h(ts[0]), _h(ts[1])
        out = 0.0
        for a in (0, 1):
            for b in (0, 1):
                i, j = idx[0][a], idx[1][b]
                out = out + (hx[a] * hy[b] * grid[0, i, j] + hx[2 + a] * hy[b] * dx[0] * grid[1, i, j]
                             + hx[a] * hy[2 + b] * dx[1] * grid[2, i, j]
                             + hx[2 + a] * hy[2 + b] * dx[0] * dx[1] * grid[3, i, j])
        return out


class MetaD(Bias):
    """(Well-tempered) metadynamics with Gaussian hills
        V(s) = sum_i w_i exp(-sum_k d_ik^2 / (2 sigma_k^2)),  d_ik = s_k - c_ik (nearest image),
    one hill every `pace` steps at the current s with height w = height * exp(-V(s) / (kB dT)),
    dT = (biasfactor - 1) T (biasfactor None: standard metadynamics, w = height).  Hills live in a
    device buffer of `capacity` slots, enlarged by the drivers between blocks (reserve).  Every step
    evaluates all slots (O(M d); unused slots have height 0).
    sigma (CV units), height (kJ/mol), temperature (K; default the simulation's).
    grid=(lo, hi, bins) (per CV, or scalars; 1 or 2 CVs): the hills are also summed on a grid
    (HillGrid, cubic Hermite interpolation) and V is read from it, O(1) per step whatever the
    number of hills (PLUMED's GRID_MIN / GRID_MAX / GRID_BIN); use bins with a spacing <= sigma / 4
    (interpolation error ~1e-3 of the hill height).  The hill list is kept for output and c(t)."""
    kind = "metad"

    def __init__(self, cvs, sigma, height: float, pace: int, biasfactor: float | None = 10.0,
                 temperature=None, capacity: int = 1024, grid=None):
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

    def init(self):
        M, d = self.capacity, self.d
        return MetaDState(jnp.zeros((M, d), jnp.float64), jnp.zeros(M, jnp.float64),
                          jnp.full(M, -1, jnp.int32), jnp.zeros((), jnp.int32),
                          None if self.grid is None else self.grid.zeros())

    def potential(self, state, s):
        if self.grid is not None:
            return self.grid.value(state.grid, s)
        return self.hill_sum(state, s)

    def hill_sum(self, state, s):
        """V(s) summed over the hills (exact; the grid's reference)."""
        ds = wrap(s[None, :] - state.centers, self.cvs.periods) / jnp.asarray(self.sigma)
        return jnp.sum(state.heights * jnp.exp(-0.5 * jnp.sum(ds * ds, 1)))

    def hill_height(self, state, s):
        if self.biasfactor is None:
            return jnp.asarray(self.height, jnp.float64)
        return self.height * jnp.exp(-self.potential(state, s) / (self.kT * (self.biasfactor - 1.0)))

    def update(self, state, s, step):
        w = self.hill_height(state, s)
        n = state.n
        c = self.cvs.canonical(s)
        grid = None if self.grid is None else self.grid.add(state.grid, c, w, self.sigma)
        return MetaDState(state.centers.at[n].set(c), state.heights.at[n].set(w),
                          state.steps.at[n].set(jnp.asarray(step, jnp.int32)), n + 1, grid)

    def reserve(self, state, n_updates: int):
        n, M = int(state.n), state.heights.shape[0]
        if n + n_updates <= M:
            return state
        if n > M:
            raise RuntimeError("metadynamics hill buffer overflowed (hills lost)")
        return self.grow_to(state, max(2 * M, n + n_updates + 16))

    def buffer_size(self, state) -> int:
        return int(state.heights.shape[0])

    def grow_to(self, state, size: int):
        pad = int(size) - state.heights.shape[0]
        if pad <= 0:
            return state
        return MetaDState(jnp.concatenate([state.centers, jnp.zeros((pad, self.d))]),
                          jnp.concatenate([state.heights, jnp.zeros(pad)]),
                          jnp.concatenate([state.steps, jnp.full(pad, -1, jnp.int32)]), state.n, state.grid)

    def fes_factor(self) -> float:
        """F(s) = -factor V(s) + const at long times: gamma / (gamma - 1) (1 for standard metaD,
        whose bias converges to -F only on average)."""
        return 1.0 if self.biasfactor is None else self.biasfactor / (self.biasfactor - 1.0)

    def hills(self, state) -> dict:
        """Deposited hills (host): step, center (n, d), height, sigma (d,)."""
        n = int(state.n)
        return {"step": np.asarray(state.steps[:n]), "center": np.asarray(state.centers[:n]),
                "height": np.asarray(state.heights[:n]), "sigma": self.sigma.copy()}

    def info(self, state):
        n = int(state.n)
        return {"hills": n, "last_height": float(state.heights[n - 1]) if n else 0.0}

    def describe(self):
        g = "standard" if self.biasfactor is None else f"well-tempered, biasfactor {self.biasfactor:g}"
        return (f"metadynamics on {', '.join(self.cvs.names)} ({g}, height {self.height:g} kJ/mol, sigma "
                f"{', '.join(f'{x:g}' for x in self.sigma)}, pace {self.pace}, T {self.temperature})")


# ----------------------------------------------------------------------------- OPES
class OPESState(NamedTuple):
    centers: jnp.ndarray            # (K, d)
    sigmas: jnp.ndarray             # (K, d)
    heights: jnp.ndarray            # (K,) (0: unused slot)
    nk: jnp.ndarray                 # kernels in use (int32)
    sum_w: jnp.ndarray              # sum of the deposition weights (incl. the initial eps^(1 - 1/gamma))
    sum_w2: jnp.ndarray
    zed: jnp.ndarray                # Z: mean of P over the kernel centres
    counter: jnp.ndarray            # depositions (int32)
    merged: jnp.ndarray             # depositions merged into an existing kernel (int32)
    forced: jnp.ndarray = None      # merges forced by a full buffer (int32; 0 unless the capacity was too small)


class OPES(Bias):
    """OPES_METAD (on-the-fly probability enhanced sampling, Invernizzi & Parrinello 2020; the
    algorithm of PLUMED's OPES_METAD):
        P(s) = sum_k G_k(s) / sum_w,  G_k(s) = h_k [exp(-d_k^2 / 2) - exp(-cut^2 / 2)] for d_k < cut,
        d_k = |(s - c_k) / sigma_k|,  V(s) = (1 - 1/gamma) kB T log(P(s) / Z + eps),
    every `pace` steps a kernel at the current s with weight w = exp(V(s) / kB T) (sum_w += w), width
    sigma = sigma0 (N_eff (d + 2) / 4)^(-1 / (d + 4)) (bandwidth rescaling; N_eff = (1 + sum_w)^2 /
    (1 + sum_w2); fixed_sigma=True keeps sigma0), height w prod(sigma0 / sigma); merged into the
    nearest kernel when that is closer than `compression` in units of its sigma (compression: 0 = off;
    recursive=True, PLUMED's default: the merged kernel is merged again while another kernel is
    within the threshold); Z = mean over kernel centres of P,
    recomputed after each deposition (O(K^2)).
    barrier: expected barrier Delta E (kJ/mol); defaults as PLUMED: biasfactor gamma = Delta E /
    kB T, eps = exp(-Delta E / ((1 - 1/gamma) kB T)), kernel cutoff sqrt(2 Delta E / ((1 - 1/gamma)
    kB T)).  The bias is bounded by -Delta E below; sum_w starts at eps^(1 - 1/gamma) (PLUMED)."""
    kind = "opes"

    def __init__(self, cvs, sigma, pace: int, barrier: float, biasfactor: float | None = None,
                 temperature=None, epsilon: float | None = None, kernel_cutoff: float | None = None,
                 compression: float = 1.0, sigma_min=None, fixed_sigma: bool = False, capacity: int = 512,
                 recursive: bool = True):
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
    def biasfactor(self):
        g = self.barrier / self.kT if self._biasfactor is None else float(self._biasfactor)
        if g <= 1.0:
            raise ValueError("OPES biasfactor must be > 1")
        return g

    @property
    def prefactor(self):
        return 1.0 - 1.0 / self.biasfactor

    @property
    def epsilon(self):
        if self._epsilon is not None:
            return float(self._epsilon)
        return float(np.exp(-self.barrier / (self.prefactor * self.kT)))

    @property
    def cutoff(self):
        if self._cutoff is not None:
            return float(self._cutoff)
        return float(np.sqrt(2.0 * self.barrier / (self.prefactor * self.kT)))

    def init(self):
        K, d = self.capacity, self.d
        w0 = self.epsilon ** self.prefactor
        z = jnp.zeros((), jnp.int32)
        return OPESState(jnp.zeros((K, d)), jnp.ones((K, d)), jnp.zeros(K), z, jnp.asarray(w0, jnp.float64),
                         jnp.asarray(w0 * w0, jnp.float64), jnp.ones((), jnp.float64), z, z, z)

    def _kernels(self, st, s):
        """G_k(s) for every slot (K,)."""
        ds = wrap(s[None, :] - st.centers, self.cvs.periods) / st.sigmas
        n2 = jnp.sum(ds * ds, 1)
        c2 = self.cutoff ** 2
        g = jnp.exp(-0.5 * jnp.minimum(n2, c2)) - np.exp(-0.5 * c2)
        return jnp.where(n2 < c2, st.heights * g, 0.0)

    def probability(self, st, s):
        """P(s) (unnormalised KDE of the unbiased distribution; divide by Z for V)."""
        return jnp.sum(self._kernels(st, s)) / st.sum_w

    def potential(self, st, s):
        return self.prefactor * self.kT * jnp.log(self.probability(st, s) / st.zed + self.epsilon)

    def update(self, st, s, step):
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
        merge = (self.compression > 0) & (n2[k] < self.compression ** 2)
        full = st.nk >= K                     # no free slot: merge into the nearest kernel (counted in `forced`)
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
        new = OPESState(centers, sigmas, heights, nk, sum_w, sum_w2, st.zed, counter,
                        st.merged + merge.astype(jnp.int32),
                        (jnp.zeros((), jnp.int32) if st.forced is None else st.forced) + forced.astype(jnp.int32))
        # Z = (1 / N_k) sum_k P(c_k)
        act = jnp.arange(K) < nk
        P = jax.vmap(lambda c: self.probability(new, c))(centers)
        zed = jnp.sum(jnp.where(act, P, 0.0)) / jnp.maximum(nk, 1)
        return new._replace(zed=zed)

    def _merge(self, c1, s1, h1, c2, s2, h2):
        """Kernel 2 merged into kernel 1: summed heights, weighted mean and second moment (moments
        about c1, so periodic CVs are safe)."""
        hm = h1 + h2
        dc = wrap(c2 - c1, self.cvs.periods)
        a = h2 / hm
        cm = self.cvs.canonical(c1 + a * dc)
        sm = jnp.sqrt(jnp.maximum((h1 * s1 ** 2 + h2 * (s2 ** 2 + dc ** 2)) / hm - (a * dc) ** 2, 1e-300))
        return cm, sm, hm

    def _mergeable(self, centers, sigmas, nk, g):
        """Nearest kernel j != g to kernel g's centre in units of sigma_j, and whether it is below
        the compression threshold."""
        K = centers.shape[0]
        ds = wrap(centers[g][None, :] - centers, self.cvs.periods) / sigmas
        idx = jnp.arange(K)
        n2 = jnp.where((idx < nk) & (idx != g), jnp.sum(ds * ds, 1), jnp.inf)
        j = jnp.argmin(n2)
        return j, n2[j] < self.compression ** 2

    def _recursive(self, centers, sigmas, heights, nk, g, go):
        """PLUMED's recursive merging: while the merged kernel g is within the threshold of another
        kernel t, merge g into t and delete g (the last kernel moves into its slot)."""
        t, near = self._mergeable(centers, sigmas, nk, g)

        def body(c):
            centers, sigmas, heights, nk, g, t, _ = c
            cm, sm, hm = self._merge(centers[t], sigmas[t], heights[t], centers[g], sigmas[g], heights[g])
            centers, sigmas, heights = centers.at[t].set(cm), sigmas.at[t].set(sm), heights.at[t].set(hm)
            last = nk - 1                                         # delete g: the last kernel takes its slot
            centers, sigmas, heights = (centers.at[g].set(centers[last]), sigmas.at[g].set(sigmas[last]),
                                        heights.at[g].set(heights[last]))
            heights = heights.at[last].set(0.0)
            sigmas = sigmas.at[last].set(1.0)
            t = jnp.where(t == last, g, t)
            nk = nk - 1
            g = t
            t, near = self._mergeable(centers, sigmas, nk, g)
            return centers, sigmas, heights, nk, g, t, near

        c = jax.lax.while_loop(lambda c: c[-1], body, (centers, sigmas, heights, nk, g, t, go & near))
        return c[0], c[1], c[2], c[3]

    def reserve(self, st, n_updates: int):
        """Kernels are compressed, so the buffer is not sized for every update: it doubles when the
        next updates could bring it above 80 % (at most 64 new kernels assumed per block).  Should it
        fill up within a block, new kernels merge into their nearest neighbour (OPESState.forced)."""
        nk, K = int(st.nk), st.heights.shape[0]
        if nk + min(n_updates, 64) <= 0.8 * K:
            return st
        return self.grow_to(st, max(2 * K, int((nk + min(n_updates, 64)) / 0.8) + 16))

    def buffer_size(self, st) -> int:
        return int(st.heights.shape[0])

    def grow_to(self, st, size: int):
        pad = int(size) - st.heights.shape[0]
        if pad <= 0:
            return st
        return st._replace(centers=jnp.concatenate([st.centers, jnp.zeros((pad, self.d))]),
                           sigmas=jnp.concatenate([st.sigmas, jnp.ones((pad, self.d))]),
                           heights=jnp.concatenate([st.heights, jnp.zeros(pad)]))

    def fes_factor(self) -> float:
        return 1.0 / self.prefactor

    def neff(self, st) -> float:
        return float((1.0 + st.sum_w) ** 2 / (1.0 + st.sum_w2))

    def info(self, st):
        c = int(st.counter)
        return {"kernels": int(st.nk), "forced": int(0 if st.forced is None else st.forced), "zed": float(st.zed), "neff": self.neff(st),
                "rct": float(self.kT * np.log(float(st.sum_w) / max(c, 1))) if c else 0.0}

    def describe(self):
        return (f"OPES_METAD on {', '.join(self.cvs.names)} (barrier {self.barrier:g} kJ/mol, biasfactor "
                f"{self.biasfactor:.3g}, sigma0 {', '.join(f'{x:g}' for x in self.sigma0)}, pace {self.pace}, "
                f"compression {self.compression:g}{', fixed sigma' if self.fixed_sigma else ''}, T {self.temperature})")


# ----------------------------------------------------------------------------- the set of biases
class BiasState(NamedTuple):
    parts: tuple                    # one state per bias
    log: jnp.ndarray                # (C, 1 + sum_d + n_bias) COLVAR rows: step, CVs of each bias, V of each bias
    nlog: jnp.ndarray               # rows filled since the last drain (int32)
    work: jnp.ndarray               # sum of V_new(x) - V_old(x) at the updates (kJ/mol)


class BiasSet:
    """The biases of one simulation.  energy(state, pos, H) = sum of the biases; update(state, pos,
    H, step) after each MD step: a COLVAR row every `colvar` steps (0: none), then the updates due
    at this step (step % pace == 0).  The drivers call reserve() between blocks (buffers sized for
    the block) and drain() to collect the COLVAR rows."""

    def __init__(self, biases, colvar: int = 0):
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
        self.ncol = 1 + sum(b.d for b in self.biases) + len(self.biases)

    def bind(self, temperature: float) -> None:
        for b in self.biases:
            b.bind(temperature)

    def check(self, n_atoms: int) -> None:
        for b in self.biases:
            b.cvs.check(n_atoms)

    @property
    def dynamic(self) -> bool:
        return any(b.pace > 0 for b in self.biases)

    def init(self, log_rows: int = 64) -> BiasState:
        return BiasState(tuple(b.init() for b in self.biases), jnp.zeros((int(log_rows), self.ncol), jnp.float64),
                         jnp.zeros((), jnp.int32), jnp.zeros((), jnp.float64))

    # -- device side
    def cv_values(self, pos, H=None):
        """Tuple of the CV vectors of each bias."""
        return tuple(b.cvs.values(pos, H) for b in self.biases)

    def energy(self, state: BiasState, pos, H=None):
        e = jnp.zeros((), jnp.float64)
        for b, p in zip(self.biases, state.parts):
            e = e + b.energy(p, pos, H)
        return e

    def energies(self, state: BiasState, pos, H=None):
        """(n_bias,) energies of each bias."""
        return jnp.stack([b.energy(p, pos, H) for b, p in zip(self.biases, state.parts)])

    def due(self, step):
        """Does any bias update at this step (after the step counter was incremented)?"""
        d = jnp.zeros((), bool)
        for b in self.biases:
            if b.pace > 0:
                d = d | (step % b.pace == 0)
        return d

    def record(self, state: BiasState, pos, H, step) -> BiasState:
        """COLVAR row (step, CVs, bias energies) if step % colvar == 0."""
        if self.colvar <= 0:
            return state

        def rec(st):
            S = self.cv_values(pos, H)
            V = [b.potential(p, s) for b, p, s in zip(self.biases, st.parts, S)]
            row = jnp.concatenate([jnp.asarray(step, jnp.float64)[None]] + list(S) + [jnp.stack(V)])
            return st._replace(log=st.log.at[st.nlog].set(row), nlog=st.nlog + 1)

        return jax.lax.cond(step % self.colvar == 0, rec, lambda st: st, state)

    def deposit(self, state: BiasState, pos, H, step) -> BiasState:
        """The updates due at this step (each bias with its own pace, at its CVs of pos)."""
        parts = list(state.parts)
        for k, b in enumerate(self.biases):
            if b.pace > 0:
                s = b.cvs.values(pos, H)
                parts[k] = jax.lax.cond(step % b.pace == 0, lambda p, s=s, b=b: b.update(p, s, step),
                                        lambda p: p, parts[k])
        return state._replace(parts=tuple(parts))

    # -- host side
    def reserve(self, state: BiasState, nsteps: int, step0: int = 0) -> BiasState:
        """Buffers large enough for nsteps more steps starting after step0 (host)."""
        parts = tuple(b.reserve(p, (nsteps // b.pace + 1) if b.pace > 0 else 0)
                      for b, p in zip(self.biases, state.parts))
        state = state._replace(parts=parts)
        if self.colvar > 0:
            need = int(state.nlog) + nsteps // self.colvar + 1
            C = state.log.shape[0]
            if need > C:
                state = state._replace(log=jnp.concatenate([state.log, jnp.zeros((max(need, 2 * C) - C, self.ncol))]))
        return state

    def reserve_many(self, states, nsteps: int) -> list:
        """reserve() for several states (walkers), then padded to common buffer sizes so that they
        can be stacked."""
        states = [self.reserve(st, nsteps) for st in states]
        parts = []
        for k, b in enumerate(self.biases):
            size = max(b.buffer_size(st.parts[k]) for st in states)
            parts.append([b.grow_to(st.parts[k], size) for st in states])
        C = max(st.log.shape[0] for st in states)
        out = []
        for w, st in enumerate(states):
            log = st.log if st.log.shape[0] == C else jnp.concatenate([st.log, jnp.zeros((C - st.log.shape[0], self.ncol))])
            out.append(st._replace(parts=tuple(p[w] for p in parts), log=log))
        return out

    def drain(self, state: BiasState):
        """(COLVAR rows since the last drain as a numpy array, state with an empty buffer)."""
        n = int(state.nlog)
        rows = np.asarray(state.log[:n])
        return rows, state._replace(nlog=jnp.zeros((), jnp.int32))

    def columns(self) -> list:
        cols = ["step"]
        for k, b in enumerate(self.biases):
            cols += [f"{n}" for n in b.cvs.names]
        cols += [f"bias{k}_{b.kind}" for k, b in enumerate(self.biases)]
        return cols

    def describe(self) -> str:
        return "; ".join(b.describe() for b in self.biases)

    def info(self, state: BiasState) -> dict:
        infos = [(k, b.info(p)) for k, (b, p) in enumerate(zip(self.biases, state.parts))]
        infos = [(k, i) for k, i in infos if i]
        out = {}
        for k, i in infos:                     # keys numbered by bias only when several biases report
            for key, v in i.items():
                out[key if len(infos) == 1 else f"{key}{k}"] = v
        return out

    def save(self, state: BiasState, path: str) -> None:
        """The bias state (hills, kernels, normalisation) to a file (host arrays, pickle)."""
        host = jax.tree_util.tree_map(np.asarray, state._replace(log=np.zeros((0, self.ncol)), nlog=np.int32(0)))
        with open(path, "wb") as fh:
            pickle.dump({"kinds": [b.kind for b in self.biases], "state": host}, fh)

    def load(self, path: str) -> BiasState:
        """A state written by save() (same biases); the COLVAR buffer starts empty."""
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        if d["kinds"] != [b.kind for b in self.biases]:
            raise ValueError(f"{path}: biases {d['kinds']} differ from {[b.kind for b in self.biases]}")
        st = jax.tree_util.tree_map(jnp.asarray, d["state"])
        return st._replace(log=jnp.zeros((64, self.ncol)), nlog=jnp.zeros((), jnp.int32))


def as_bias_set(x, colvar: int = 0) -> BiasSet | None:
    if x is None:
        return None
    if isinstance(x, BiasSet):
        if colvar and not x.colvar:
            x.colvar = int(colvar)
        return x
    return BiasSet(x, colvar)
