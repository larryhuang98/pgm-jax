"""Langevin dynamics of particles in an analytic potential with biases (model systems whose free
energy surfaces are known exactly), with W walkers advanced together by jax.vmap: independent runs
(each walker its own bias) or multiple-walker metadynamics / OPES (one shared bias: every walker
deposits into it and feels all hills).

    from pgm_jax.bias import MetaD, cv
    from pgm_jax.bias.toy import ToyLangevin, double_well
    U = double_well(barrier=20.0)                              # U(pos) kJ/mol, pos (1, 3) nm
    b = MetaD(cv.Component(0, 0), sigma=0.1, height=1.0, pace=100, biasfactor=10)
    sim = ToyLangevin(U, [[-1.0, 0, 0]], mass=40.0, temperature=300.0, dt=0.005, gamma=5.0,
                      bias=b, walkers=8, shared=False)
    out = sim.run(200000, sample=100)                          # CVs and bias energies every 100 steps

BAOAB (Leimkuhler-Matthews) with the exact O step; each step: B, A, O, A, the bias update if
due (at the new positions, so the forces of the step already include the new hill), forces, B.
The samples hold the bias energy before that step's update (as the MD drivers' COLVAR).
Units nm, ps, amu, kJ/mol, K."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..units import KB
from .core import as_bias_set


class ToyState(NamedTuple):
    pos: jnp.ndarray  # (W, N, 3)
    mom: jnp.ndarray
    force: jnp.ndarray
    epot: jnp.ndarray  # (W,) U + V
    key: jnp.ndarray  # (W, 2)
    step: jnp.ndarray
    bias: object  # BiasState (shared) or batched BiasState (independent walkers) or None


class ToyLangevin:
    def __init__(
        self,
        potential,
        pos0,
        mass=40.0,
        temperature: float = 300.0,
        dt: float = 0.005,
        gamma: float = 5.0,
        bias=None,
        walkers: int = 1,
        shared: bool = False,
        seed: int = 0,
        active_axes=(0, 1, 2),
    ):
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
    def _total(self, pos, bstate):
        e = self.U(pos)
        if self.bias is not None:
            e = e + self.bias.energy(bstate, pos, None)
        return e

    def _with_forces(self, st: ToyState) -> ToyState:
        f = jax.value_and_grad(self._total)
        if self.bias is None or self.shared:
            e, g = jax.vmap(f, in_axes=(0, None))(st.pos, st.bias)
        else:
            e, g = jax.vmap(f)(st.pos, st.bias)
        return st._replace(force=-g * self.mask, epot=e)

    def _deposit(self, st: ToyState) -> ToyState:
        b = self.bias
        if self.shared:  # walkers deposit one after the other into one bias
            bs = st.bias
            for w in range(self.W):
                bs = b.deposit(bs, st.pos[w], None, st.step)
            return st._replace(bias=bs)
        return st._replace(bias=jax.vmap(lambda bs, x: b.deposit(bs, x, None, st.step))(st.bias, st.pos))

    def _step(self, st: ToyState, want_sample: bool = False):
        m, h = jnp.asarray(self.m), self.dt
        c1 = np.exp(-self.gamma * h)
        c2 = np.sqrt(1.0 - c1 * c1)
        p = st.mom + 0.5 * h * st.force
        x = st.pos + 0.5 * h * p / m
        ks = jax.vmap(jax.random.split)(st.key)
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

    def _sample(self, st: ToyState):
        """Positions, CVs (W, sum d) and bias energies (W, n_bias) before the update of the step."""
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

    def run(self, nsteps: int, sample: int = 100):
        """Advance nsteps (a multiple of `sample`); returns {"step" (F,), "cv" (F, W, d), "bias" (F, W, n_bias),
        "x" (F, W, N, 3)} sampled every `sample` steps (after the step; bias before its update)."""
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
            self._runs = {key: jax.jit(lambda st: self._scan(st, nchunk, sample))}
        self.state, out = self._runs[key](self.state)
        return {k: np.asarray(v) for k, v in out.items()}

    def _reserve_batched(self, bs, nsteps):
        """Grow the buffers of every walker's bias (host)."""
        per = self.bias.reserve_many([jax.tree_util.tree_map(lambda a: a[w], bs) for w in range(self.W)], nsteps)
        return jax.tree_util.tree_map(lambda *a: jnp.stack(a), *per)

    def _scan(self, st, nchunk, sample):
        def body(st, _):
            st = jax.lax.fori_loop(0, sample - 1, lambda _, s: self._step(s), st)
            return self._step(st, True)

        return jax.lax.scan(body, st, None, length=nchunk)


# ----------------------------------------------------------------------------- model potentials
def double_well(barrier: float = 20.0, x0: float = 1.0, k_perp: float = 500.0, tilt: float = 0.0):
    """U = barrier ((x/x0)^2 - 1)^2 + tilt x / x0 + k_perp / 2 (y^2 + z^2) for one particle (pos (1, 3));
    the exact FES of the CV x is U(x, 0, 0)."""

    def U(pos):
        x = pos[0, 0] / x0
        return barrier * (x * x - 1.0) ** 2 + tilt * x + 0.5 * k_perp * (pos[0, 1] ** 2 + pos[0, 2] ** 2)

    U.fes = lambda x: barrier * ((np.asarray(x) / x0) ** 2 - 1.0) ** 2 + tilt * np.asarray(x) / x0
    return U


MB_A = np.array([-200.0, -100.0, -170.0, 15.0])
MB_a = np.array([-1.0, -1.0, -6.5, 0.7])
MB_b = np.array([0.0, 0.0, 11.0, 0.6])
MB_c = np.array([-10.0, -10.0, -6.5, 0.7])
MB_x0 = np.array([1.0, 0.0, -0.5, -1.0])
MB_y0 = np.array([0.0, 0.5, 1.5, 1.0])


def mueller_brown(scale: float = 0.1, k_perp: float = 500.0):
    """Mueller-Brown potential (x, y in nm) times `scale` (kJ/mol), harmonic in z; exact FES of the
    CVs (x, y) is the scaled potential."""

    def mb(x, y, xp=jnp):
        d = xp.asarray(x)[..., None] - MB_x0
        e = xp.asarray(y)[..., None] - MB_y0
        return scale * xp.sum(MB_A * xp.exp(MB_a * d * d + MB_b * d * e + MB_c * e * e), -1)

    def U(pos):
        return mb(pos[0, 0], pos[0, 1]) + 0.5 * k_perp * pos[0, 2] ** 2

    U.fes = lambda x, y: mb(x, y, np)
    return U


def ring(coeffs=((1, 6.0), (2, -4.0), (3, 3.0)), radius: float = 1.0, k_r: float = 2000.0, k_z: float = 500.0):
    """One particle on a ring: U = sum_n a_n cos(n theta) + k_r / 2 (r - R)^2 + k_z / 2 z^2 with
    theta = atan2(y, x), r = |(x, y)|.  The exact FES of the periodic CV theta is sum_n a_n cos(n theta)
    (the radial integral does not depend on theta)."""
    coeffs = [(int(n), float(a)) for n, a in coeffs]

    def ang(t, xp=jnp):
        return sum(a * xp.cos(n * t) for n, a in coeffs)

    def U(pos):
        x, y, z = pos[0, 0], pos[0, 1], pos[0, 2]
        r = jnp.sqrt(x * x + y * y)
        return ang(jnp.arctan2(y, x)) + 0.5 * k_r * (r - radius) ** 2 + 0.5 * k_z * z * z

    U.fes = lambda t: ang(np.asarray(t), np)
    return U
