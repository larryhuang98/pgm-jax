"""Run Langevin dynamics of particles in analytic potentials with biases, W walkers at once.

Contents: `ToyLangevin` (the engine; state `ToyState`) and the model potentials
`double_well`, `mueller_brown` and `ring`, whose free energy surfaces are known exactly (each
returned potential carries it as its `fes` attribute).  Used to validate the biases of core.py
and the analysis of analysis.py (scripts/bias/validate_toy.py).

The W walkers are advanced together by jax.vmap: independent runs (shared=False: each walker
has its own bias state, batched over a leading axis W) or multiple-walker metadynamics / OPES
(shared=True: one bias state; every walker deposits into it and feels all hills).

    from pgm_jax.bias import MetaD, cv
    from pgm_jax.bias.toy import ToyLangevin, double_well
    U = double_well(barrier=20.0)                              # U(pos) kJ/mol, pos (1, 3) nm
    b = MetaD(cv.Component(0, 0), sigma=0.1, height=1.0, pace=100, biasfactor=10)
    sim = ToyLangevin(U, [[-1.0, 0, 0]], mass=40.0, temperature=300.0, dt=0.005, gamma=5.0,
                      bias=b, walkers=8, shared=False)
    out = sim.run(200000, sample=100)                          # CVs and bias energies every 100 steps

Integrator: BAOAB [1]_ with the exact O step.  Each step: B (half kick), A (half drift), O
(p <- c1 p + c2 sqrt(m kT) xi, c1 = exp(-gamma dt), c2 = sqrt(1 - c1^2)), A, then the bias
update if due (at the new positions, so the forces of the step already include the new hill),
forces, B.  The samples hold the bias energy before that step's update (as the MD drivers'
COLVAR).  Cartesian axes not in `active_axes` are frozen (momenta and forces masked).

Units: nm, ps, amu, kJ/mol, K; friction `gamma` 1/ps.

References
----------
.. [1] B. Leimkuhler, C. Matthews, Appl. Math. Res. Express 2013, 34 (2013). doi:10.1093/amrx/abs010
.. [2] K. Mueller, L. D. Brown, Theor. Chim. Acta 53, 75 (1979).

See also docs/enhanced_sampling.md (validation 1).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..units import KB
from .core import as_bias_set

if TYPE_CHECKING:
    from .core import Bias, BiasSet


class ToyState(NamedTuple):
    """State of `ToyLangevin` (a NamedTuple pytree; every leaf is traced).

    Attributes
    ----------
    pos : jax.Array (W, N, 3)
        Positions [nm].
    mom : jax.Array (W, N, 3)
        Momenta [amu nm/ps].
    force : jax.Array (W, N, 3)
        Forces -d(U + V)/dpos [kJ/mol/nm] (masked on frozen axes).
    epot : jax.Array (W,)
        Potential energy U + V [kJ/mol].
    key : jax.Array (W, 2) uint32
        PRNG key per walker.
    step : jax.Array () int32
        Step counter (shared by the walkers).
    bias : BiasState or None
        One BiasState (shared walkers), a BiasState batched over a leading axis W (independent
        walkers), or None without a bias.
    """

    pos: jax.Array  # (W, N, 3)
    mom: jax.Array  # (W, N, 3)
    force: jax.Array  # (W, N, 3)
    epot: jax.Array  # (W,) U + V
    key: jax.Array  # (W, 2)
    step: jax.Array  # () int32
    bias: object  # BiasState (shared) or batched BiasState (independent walkers) or None


class ToyLangevin:
    """BAOAB Langevin dynamics of W walkers of N particles in an analytic potential, with biases.

    See the module docstring for the integrator and the walker modes.

    Attributes
    ----------
    U : callable
        The potential U(pos) [kJ/mol] of one walker, pos (N, 3) [nm].
    W, N : int
        Number of walkers and of particles per walker.
    m : np.ndarray (N, 1)
        Masses [amu].
    kT : float
        Thermal energy [kJ/mol].
    T : float
        Temperature [K].
    dt : float
        Time step [ps].
    gamma : float
        Friction [1/ps].
    mask : np.ndarray (3,)
        1 for active Cartesian axes, 0 for frozen ones.
    bias : BiasSet or None
        The biases (bound to T).
    shared : bool
        One bias for all walkers.
    state : ToyState
        The current state.
    """

    def __init__(
        self,
        potential: Callable[[jax.Array], jax.Array],
        pos0: ArrayLike,
        mass: ArrayLike = 40.0,
        temperature: float = 300.0,
        dt: float = 0.005,
        gamma: float = 5.0,
        bias: Bias | Sequence[Bias] | BiasSet | None = None,
        walkers: int = 1,
        shared: bool = False,
        seed: int = 0,
        active_axes: Sequence[int] = (0, 1, 2),
    ) -> None:
        """Set up the walkers, draw Maxwell-Boltzmann momenta and compute the initial forces.

        Parameters
        ----------
        potential : callable
            U(pos) -> scalar [kJ/mol], a JAX function of the positions (N, 3) [nm] of one walker.
        pos0 : ArrayLike (N, 3) or (W, N, 3)
            Initial positions [nm] (the same for every walker, or one set per walker).
        mass : ArrayLike
            Mass [amu]: a scalar or one per particle.
        temperature : float
            Temperature [K].
        dt : float
            Time step [ps].
        gamma : float
            Langevin friction [1/ps].
        bias : Bias, sequence of Bias, or BiasSet, optional
            The biases on CVs of the positions (H None: no periodic boundaries); None: unbiased.
        walkers : int
            Number of walkers W (used when `pos0` is (N, 3)).
        shared : bool
            One bias shared by the walkers (multiple walkers) instead of one per walker.
        seed : int
            Seed of the momenta and the noise.
        active_axes : sequence of int
            Cartesian axes that move (e.g. (0, 1) for a 2D model).
        """
        self.U = potential
        pos0 = np.asarray(pos0, float)
        if pos0.ndim == 2:
            pos0 = np.broadcast_to(pos0, (int(walkers),) + pos0.shape).copy()
        self.W, self.N = pos0.shape[0], pos0.shape[1]
        self.m = np.broadcast_to(np.asarray(mass, float).reshape(-1, 1), (self.N, 1)).copy()
        self.kT = KB * float(temperature)
        self.T = float(temperature)
        self.dt, self.gamma = float(dt), float(gamma)
        self.mask = np.zeros(3)
        self.mask[list(active_axes)] = 1.0  # frozen Cartesian axes (2D models)
        self.bias = as_bias_set(bias)
        if self.bias is not None:
            self.bias.bind(self.T)
        self.shared = bool(shared)
        keys = jax.random.split(jax.random.PRNGKey(seed), self.W + 1)
        pos = jnp.asarray(pos0)
        mom = jnp.sqrt(jnp.asarray(self.m) * self.kT) * jax.random.normal(keys[0], pos.shape) * self.mask
        bstate = None
        if self.bias is not None:
            b0 = self.bias.init(log_rows=1)
            bstate = b0 if self.shared else jax.tree_util.tree_map(lambda a: jnp.stack([a] * self.W), b0)
        st = ToyState(pos, mom, jnp.zeros_like(pos), jnp.zeros(self.W), keys[1:], jnp.zeros((), jnp.int32), bstate)
        self.state = self._with_forces(st)
        self._runs = {}

    # -- energies
    def _total(self, pos: jax.Array, bstate: Any) -> jax.Array:
        """Return U(pos) + V(pos) [kJ/mol] of one walker (`bstate` is that walker's or the shared bias state)."""
        e = self.U(pos)
        if self.bias is not None:
            e = e + self.bias.energy(bstate, pos, None)
        return e

    def _with_forces(self, st: ToyState) -> ToyState:
        """Return the state with forces -grad(U + V) (masked) and epot of every walker (vmapped over W)."""
        f = jax.value_and_grad(self._total)
        if self.bias is None or self.shared:
            e, g = jax.vmap(f, in_axes=(0, None))(st.pos, st.bias)
        else:
            e, g = jax.vmap(f)(st.pos, st.bias)
        return st._replace(force=-g * self.mask, epot=e)

    def _deposit(self, st: ToyState) -> ToyState:
        """Return the state after the bias updates due at this step.

        Shared bias: the walkers deposit one after the other into the one state (walker w sees the
        hills of walkers < w).  Independent walkers: vmapped over the batched bias states.
        """
        b = self.bias
        if self.shared:  # walkers deposit one after the other into one bias
            bs = st.bias
            for w in range(self.W):
                bs = b.deposit(bs, st.pos[w], None, st.step)
            return st._replace(bias=bs)
        return st._replace(bias=jax.vmap(lambda bs, x: b.deposit(bs, x, None, st.step))(st.bias, st.pos))

    def _step(self, st: ToyState, want_sample: bool = False) -> ToyState | tuple[ToyState, dict]:
        """Advance all walkers by one BAOAB step (and the bias update if due).

        Parameters
        ----------
        st : ToyState
            The state.
        want_sample : bool
            Also return the sample of the step (`_sample`, taken after the drift and before the bias
            update).

        Returns
        -------
        ToyState, or (ToyState, dict)
            The new state, with the sample if `want_sample`.
        """
        m, h = jnp.asarray(self.m), self.dt
        c1 = np.exp(-self.gamma * h)  # O step over a full dt: p <- c1 p + c2 sqrt(m kT) xi (exact OU solution)
        c2 = np.sqrt(1.0 - c1 * c1)
        p = st.mom + 0.5 * h * st.force
        x = st.pos + 0.5 * h * p / m
        ks = jax.vmap(jax.random.split)(st.key)  # (W, 2, 2): new key and noise key per walker
        noise = jax.vmap(lambda k: jax.random.normal(k, st.pos.shape[1:]))(ks[:, 1])
        p = (c1 * p + c2 * jnp.sqrt(m * self.kT) * noise) * self.mask
        x = x + 0.5 * h * p / m
        st = st._replace(pos=x, mom=p, key=ks[:, 0], step=st.step + 1)
        smp = self._sample(st) if want_sample else None
        if self.bias is not None and self.bias.dynamic:
            st = jax.lax.cond(self.bias.due(st.step), self._deposit, lambda s: s, st)
        st = self._with_forces(st)
        st = st._replace(mom=st.mom + 0.5 * h * st.force)
        return (st, smp) if want_sample else st

    def _sample(self, st: ToyState) -> dict[str, jax.Array]:
        """Return the sample of the current state: step, positions, CVs and bias energies.

        Returns
        -------
        dict
            "step" () int32, "x" (W, N, 3) [nm]; with a bias also "cv" (W, sum d) [CV units] and "bias"
            (W, n_bias) [kJ/mol] (the bias state before the update of the step).
        """
        out = {"step": st.step, "x": st.pos}
        b = self.bias
        if b is None:
            return out
        S = jax.vmap(lambda x: jnp.concatenate(b.cv_values(x, None)))(st.pos)
        if self.shared:
            V = jax.vmap(lambda x: b.energies(st.bias, x, None))(st.pos)
        else:
            V = jax.vmap(lambda bs, x: b.energies(bs, x, None))(st.bias, st.pos)
        out["cv"], out["bias"] = S, V
        return out

    def run(self, nsteps: int, sample: int = 100) -> dict[str, np.ndarray]:
        """Advance `nsteps` steps, sampling every `sample` steps.

        Parameters
        ----------
        nsteps : int
            Steps to run [steps] (a multiple of `sample`).
        sample : int
            Steps between samples [steps].

        Returns
        -------
        dict
            "step" (F,), "x" (F, W, N, 3) [nm], and with a bias "cv" (F, W, d) [CV units] and "bias"
            (F, W, n_bias) [kJ/mol], F = nsteps / sample; each sample is taken after its step, with
            the bias before that step's update.

        Raises
        ------
        ValueError
            If `nsteps` is not a multiple of `sample`.

        Notes
        -----
        The bias buffers are grown on the host first (`BiasSet.reserve`, `_reserve_batched`); then one
        jitted `lax.scan` runs the block.  The compiled program is cached for the last (nchunk, sample)
        only; new buffer shapes re-trace it.
        """
        nsteps, sample = int(nsteps), int(sample)
        if nsteps % sample:
            raise ValueError("nsteps must be a multiple of sample")
        nchunk = nsteps // sample
        if self.bias is not None:
            if self.shared:
                bs = self.bias.reserve(self.state.bias, nsteps * self.W)
            else:
                bs = self._reserve_batched(self.state.bias, nsteps)
            self.state = self.state._replace(bias=bs)
        key = (nchunk, sample)
        if key not in self._runs:
            self._runs = {key: jax.jit(lambda st: self._scan(st, nchunk, sample))}  # keep only the latest program
        self.state, out = self._runs[key](self.state)
        return {k: np.asarray(v) for k, v in out.items()}

    def _reserve_batched(self, bs: Any, nsteps: int) -> Any:
        """Return the batched bias state with the buffers of every walker grown for `nsteps` steps (host)."""
        per = self.bias.reserve_many([jax.tree_util.tree_map(lambda a: a[w], bs) for w in range(self.W)], nsteps)
        return jax.tree_util.tree_map(lambda *a: jnp.stack(a), *per)

    def _scan(self, st: ToyState, nchunk: int, sample: int) -> tuple[ToyState, dict]:
        """Return the state after nchunk x sample steps and the stacked samples (`lax.scan` over chunks)."""

        def body(st: ToyState, _: None) -> tuple[ToyState, dict]:
            """Run sample - 1 plain steps and one sampled step (scan body: carry the state, output the sample)."""
            st = jax.lax.fori_loop(0, sample - 1, lambda _, s: self._step(s), st)
            return self._step(st, True)

        return jax.lax.scan(body, st, None, length=nchunk)


# ----------------------------------------------------------------------------- model potentials
def double_well(
    barrier: float = 20.0, x0: float = 1.0, k_perp: float = 500.0, tilt: float = 0.0
) -> Callable[[jax.Array], jax.Array]:
    """Return a 1-particle double well U = barrier ((x/x0)^2 - 1)^2 + tilt x / x0 + k_perp / 2 (y^2 + z^2).

    Parameters
    ----------
    barrier : float
        Barrier height at x = 0 [kJ/mol] (untilted).
    x0 : float
        Position of the minima [nm].
    k_perp : float
        Force constant in y and z [kJ/mol/nm^2].
    tilt : float
        Linear tilt [kJ/mol] (energy difference between x = x0 and 0 is tilt).

    Returns
    -------
    callable
        U(pos) [kJ/mol] for pos (1, 3) [nm], with `U.fes(x)` the exact FES of the CV x
        (U(x, 0, 0), numpy) [kJ/mol].
    """

    def U(pos: jax.Array) -> jax.Array:
        """Return the double-well energy [kJ/mol] of pos (1, 3) [nm]."""
        x = pos[0, 0] / x0
        return barrier * (x * x - 1.0) ** 2 + tilt * x + 0.5 * k_perp * (pos[0, 1] ** 2 + pos[0, 2] ** 2)

    U.fes = lambda x: barrier * ((np.asarray(x) / x0) ** 2 - 1.0) ** 2 + tilt * np.asarray(x) / x0
    return U


# Mueller-Brown parameters (Mueller and Brown 1979): U = sum_i A_i exp(a_i dx^2 + b_i dx dy + c_i dy^2),
# dx = x - x0_i, dy = y - y0_i (x, y in nm here, A in units of `scale` kJ/mol)
MB_A = np.array([-200.0, -100.0, -170.0, 15.0])
MB_a = np.array([-1.0, -1.0, -6.5, 0.7])
MB_b = np.array([0.0, 0.0, 11.0, 0.6])
MB_c = np.array([-10.0, -10.0, -6.5, 0.7])
MB_x0 = np.array([1.0, 0.0, -0.5, -1.0])
MB_y0 = np.array([0.0, 0.5, 1.5, 1.0])


def mueller_brown(scale: float = 0.1, k_perp: float = 500.0) -> Callable[[jax.Array], jax.Array]:
    """Return the Mueller-Brown potential [2]_ (x, y in nm) times `scale` [kJ/mol], harmonic in z.

    Parameters
    ----------
    scale : float
        Energy scale [kJ/mol per Mueller-Brown unit].
    k_perp : float
        Force constant in z [kJ/mol/nm^2].

    Returns
    -------
    callable
        U(pos) [kJ/mol] for pos (1, 3) [nm], with `U.fes(x, y)` the exact FES of the CVs (x, y)
        (the scaled potential, numpy) [kJ/mol].
    """

    def mb(x: ArrayLike, y: ArrayLike, xp: Any = jnp) -> Any:
        """Return the scaled Mueller-Brown energy at (x, y) [nm] with array module `xp` (jnp or np)."""
        d = xp.asarray(x)[..., None] - MB_x0
        e = xp.asarray(y)[..., None] - MB_y0
        return scale * xp.sum(MB_A * xp.exp(MB_a * d * d + MB_b * d * e + MB_c * e * e), -1)

    def U(pos: jax.Array) -> jax.Array:
        """Return the Mueller-Brown energy [kJ/mol] of pos (1, 3) [nm]."""
        return mb(pos[0, 0], pos[0, 1]) + 0.5 * k_perp * pos[0, 2] ** 2

    U.fes = lambda x, y: mb(x, y, np)
    return U


def ring(
    coeffs: Sequence[tuple[int, float]] = ((1, 6.0), (2, -4.0), (3, 3.0)),
    radius: float = 1.0,
    k_r: float = 2000.0,
    k_z: float = 500.0,
) -> Callable[[jax.Array], jax.Array]:
    """Return a particle on a ring: U = sum_n a_n cos(n theta) + k_r / 2 (r - R)^2 + k_z / 2 z^2.

    theta = atan2(y, x) and r = |(x, y)|.  The exact FES of the periodic CV theta is
    sum_n a_n cos(n theta): the radial integral does not depend on theta.

    Parameters
    ----------
    coeffs : sequence of (int, float)
        (n, a_n) pairs [a_n in kJ/mol].
    radius : float
        Ring radius R [nm].
    k_r : float
        Radial force constant [kJ/mol/nm^2].
    k_z : float
        Force constant in z [kJ/mol/nm^2].

    Returns
    -------
    callable
        U(pos) [kJ/mol] for pos (1, 3) [nm], with `U.fes(theta)` the exact FES [kJ/mol].
    """
    coeffs = [(int(n), float(a)) for n, a in coeffs]

    def ang(t: ArrayLike, xp: Any = jnp) -> Any:
        """Return sum_n a_n cos(n t) with array module `xp` (jnp or np)."""
        return sum(a * xp.cos(n * t) for n, a in coeffs)

    def U(pos: jax.Array) -> jax.Array:
        """Return the ring energy [kJ/mol] of pos (1, 3) [nm]."""
        x, y, z = pos[0, 0], pos[0, 1], pos[0, 2]
        r = jnp.sqrt(x * x + y * y)
        return ang(jnp.arctan2(y, x)) + 0.5 * k_r * (r - radius) ** 2 + 0.5 * k_z * z * z

    U.fes = lambda t: ang(np.asarray(t), np)
    return U
