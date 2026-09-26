"""Thermostats: the stochastic (O) part of the BAOAB-type steps of both integrators.

A thermostat acts at fixed positions on the mass-scaled momenta v = M^-1/2 p (and on its own
auxiliary momenta s, same shape as v).  Each O step is exact for its own dynamics and leaves
exp(-beta (|v|^2 + |s|^2) / 2) invariant, so together with the Hamiltonian steps the scheme
samples the canonical ensemble (up to the usual O(dt^2) splitting error).

  Langevin(gamma)   white noise on every degree of freedom:
                    dv = -gamma v dt + sqrt(2 gamma kT) dW.
  Bussi(tau)        stochastic velocity rescaling (Bussi, Donadio & Parrinello, JCP 126, 014101,
                    2007): one random factor per step scales all momenta, so the total kinetic
                    energy follows the canonical distribution with relaxation time tau.
  GLE(A)            generalized Langevin equation per degree of freedom, Markovian embedding
                    d(v, s) = -A (v, s) dt + B dW with B B^T = kT (A + A^T), i.e.
                    fluctuation-dissipation holds pointwise.  With A_vv = 0 the noise reaches v only
                    through the auxiliaries, and the trajectories stay smooth in time.
                    GLE.band() is a slow-band kernel with three variables (v, s, q): friction and
                    noise act on s (and, weakly, on q), K(w) ~ a^2 g w^2 / ((w0^2 - w^2)^2 + g^2 w^2),
                    peaked at w0, small at w = 0 and at the fast (librational) frequencies.

Why this matters for pGM: the induced-dipole predictor extrapolates the dipoles along the
trajectory, and per-atom white noise makes the trajectory rough (predictor error ~ sqrt(gamma)
dt^1.5).  Measured on 4096 pGM waters at tol 1e-5 (docs/thermostat_ideas.md):

  thermostat        CG it. 1 fs   H~ drift 2 fs (kT/ns/dof)   D vs NVE   T 1 ps after 146 K start
  Langevin 1/ps        6.0            +0.005                   -17 %        276 K
  Bussi 1 ps           4.0            +0.003 (as NVE)        small (1)      global
  GLE.band()           4.0            +0.005                   -7.5 %        277 K

(1) Bussi & Parrinello, CPC 179, 26 (2008): diffusion nearly unchanged.

Bussi is the fastest choice and perturbs dynamics least.  Use GLE.band() when every degree of
freedom should be coupled to the bath (local control: heterogeneous heating, equilibration).
Langevin is kept for compatibility.

The integrators also book the heat each O step exchanges, so H~ = E_tot + |s|^2/2 - heat
("econs" in Simulation.observables()) is conserved up to integration and induction errors, as
in NVE.

Rule for new thermostats: A and B may depend on the configuration only, never on momenta, dipole
rates, predictor errors or solver history.  The dipoles must not become thermal variables:
thermalized dipoles add (kT/2) ln det(alpha^-1 - T(x)) to the free energy.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg


class Thermostat:
    """Base class.  n_aux auxiliary momenta per degree of freedom."""
    name = "none"
    n_aux = 0

    def init_aux(self, key, shape, kT: float):
        """Auxiliary momenta drawn from their stationary distribution N(0, kT)."""
        return jnp.sqrt(kT) * jax.random.normal(key, (self.n_aux,) + tuple(shape), jnp.float64)

    def apply(self, v, aux, key, h: float, kT: float, dof: float, project, mask):
        """One exact O step of length h.  v: mass-scaled momenta; aux: (n_aux,) + v.shape;
        kT: kB T (kJ/mol), a float or a traced scalar (replica exchange: one per replica);
        project: projection of a mass-scaled vector onto the constraint tangent space; mask:
        1 for real degrees of freedom, 0 for padding (or None)."""
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class Langevin(Thermostat):
    name = "langevin"

    def __init__(self, gamma: float = 1.0):
        self.gamma = float(gamma)

    def apply(self, v, aux, key, h, kT, dof, project, mask):
        c = np.exp(-self.gamma * h)
        v = c * v + jnp.sqrt(kT * (1.0 - c * c)) * jax.random.normal(key, v.shape, v.dtype)
        v = project(v)
        return (v if mask is None else v * mask), aux

    def describe(self):
        return f"Langevin {self.gamma:g}/ps"


class Bussi(Thermostat):
    name = "bussi"

    def __init__(self, tau: float = 1.0):
        self.tau = float(tau)

    def apply(self, v, aux, key, h, kT, dof, project, mask):
        k1, k2 = jax.random.split(key)
        K = jnp.maximum(0.5 * jnp.sum(v * v), 1e-30)
        c = np.exp(-h / self.tau)
        f = 0.5 * dof * kT / (dof * K)                               # K0 / (N_f K)
        r1 = jax.random.normal(k1, (), jnp.float64)
        rest = 2.0 * jax.random.gamma(k2, (dof - 1.0) / 2.0, dtype=jnp.float64)   # chi^2 with N_f - 1 dof
        a2 = c + (1.0 - c) * f * (r1 * r1 + rest) + 2.0 * r1 * jnp.sqrt(c * (1.0 - c) * f)
        alpha = jnp.sign(r1 + jnp.sqrt(c / ((1.0 - c) * f))) * jnp.sqrt(a2)
        return alpha * v, aux

    def describe(self):
        return f"Bussi tau {self.tau:g} ps"


class GLE(Thermostat):
    """Markovian GLE with drift matrix A ((1 + n_aux) square, first index = the momentum).  The
    noise matrix follows from fluctuation-dissipation; A + A^T must be positive semidefinite."""
    name = "gle"

    def __init__(self, A, label: str = "gle"):
        A = np.asarray(A, float)
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 2:
            raise ValueError("A must be square with at least one auxiliary")
        if np.linalg.eigvalsh(A + A.T).min() < -1e-12 * np.abs(A).max():
            raise ValueError("A + A^T must be positive semidefinite (fluctuation-dissipation)")
        self.A, self.n_aux, self.label = A, A.shape[0] - 1, label
        self._cache = {}

    @classmethod
    def band(cls, peak: float = 3.0, center: float = 20.0, width: float = 30.0, floor: float = 0.1) -> "GLE":
        """Slow-band kernel: K(w) peaks at `center` (rad/ps) at about `peak` (1/ps), with width
        `width` (rad/ps) and K ~ peak width / w^2 at high frequency.  `floor` = K(0) (1/ps).
        - floor = 0 gives the pure band-pass. A is then singular: center v + a q is conserved
          by the thermostat, so zero-frequency motion (the total momentum) is never
          thermalized.
        - The default 0.1/ps keeps the scheme ergodic at a small cost in diffusion.
        With the defaults, pGM water thermalizes as fast as with Langevin 1/ps, the predictor
        keeps its NVE accuracy and diffusion is perturbed by about 7 %."""
        a2 = peak * width
        if floor < 0 or floor >= a2 / width:
            raise ValueError("need 0 <= floor < peak")
        gq = 0.0 if floor == 0 else center ** 2 / (a2 / floor - width)
        a = np.sqrt(a2)
        return cls([[0.0, a, 0.0], [-a, width, center], [0.0, -center, gq]],
                   label=f"GLE band (peak {peak:g}/ps at {center:g} rad/ps, width {width:g}, floor {floor:g}/ps)")

    @classmethod
    def lowpass(cls, gamma0: float = 1.0, cutoff: float = 50.0) -> "GLE":
        """Low-pass kernel K(w) = gamma0 cutoff^2 / (cutoff^2 + w^2): Langevin-like friction gamma0
        below `cutoff` (rad/ps), smooth noise."""
        a = np.sqrt(gamma0 * cutoff)
        return cls([[0.0, a], [-a, cutoff]], label=f"GLE low-pass ({gamma0:g}/ps below {cutoff:g} rad/ps)")

    def kernel(self, omega):
        """Friction spectrum K(w) = Re K^(i w), 1/ps, for w in rad/ps."""
        A = self.A
        out = []
        for w in np.atleast_1d(omega):
            Kz = A[0, 0] - A[0, 1:] @ np.linalg.solve(1j * w * np.eye(self.n_aux) + A[1:, 1:], A[1:, 0])
            out.append(np.real(Kz))
        return np.array(out)

    def _propagator(self, h):
        """Drift propagator T = exp(-A h) and noise factor S at unit kT (the noise scales as sqrt(kT), so
        kT may be a traced value, e.g. one per replica)."""
        key = float(h)
        if key not in self._cache:
            T = scipy.linalg.expm(-self.A * h)
            w, V = np.linalg.eigh(np.eye(len(T)) - T @ T.T)
            S = V @ np.diag(np.sqrt(np.maximum(w, 0.0)))               # S S^T = I - T T^T
            self._cache[key] = (T, S)
        return self._cache[key]

    def apply(self, v, aux, key, h, kT, dof, project, mask):
        T, S = self._propagator(h)
        y = jnp.concatenate([v[None], aux], 0)
        xi = jax.random.normal(key, y.shape, y.dtype)
        y = jnp.tensordot(jnp.asarray(T), y, axes=1) + jnp.sqrt(kT) * jnp.tensordot(jnp.asarray(S), xi, axes=1)
        y = jax.vmap(project)(y)
        if mask is not None:
            y = y * mask[None]
        return y[0], y[1:]

    def describe(self):
        return self.label


def make_thermostat(spec, gamma: float = 1.0, tau: float = 1.0) -> Thermostat:
    """"langevin" (friction gamma), "bussi" (time constant tau), "gle" / "gle-band" (slow-band
    GLE), "gle-lowpass", or a Thermostat instance."""
    if isinstance(spec, Thermostat):
        return spec
    s = str(spec).lower()
    if s == "langevin":
        return Langevin(gamma)
    if s in ("bussi", "csvr", "v-rescale"):
        return Bussi(tau)
    if s in ("gle", "gle-band", "band"):
        return GLE.band()
    if s in ("gle-lowpass", "lowpass"):
        return GLE.lowpass(gamma)
    raise ValueError(f"unknown thermostat {spec!r}: langevin | bussi | gle | gle-lowpass | Thermostat")
