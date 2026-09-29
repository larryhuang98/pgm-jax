"""Compute ensemble averages and their parameter gradients by reweighting stored frames.

This is the top-down step of force-field refinement (J couplings, helicities, populations from
MD against experiment).  Contents: Reweighting (weights, averages, n_eff, chi^2 and its
gradient), karplus and KARPLUS (3J couplings), backbone_torsions, in_region and ALPHA_BOX
(helical-basin populations).

Frames R_k sampled with parameters theta0 are reweighted to parameters theta,

    w_k(theta) = exp(-beta [U(R_k; theta) - U(R_k; theta0)]) / sum_l (...),
    <O>_theta  = sum_k w_k O(R_k),

where U is the part of the energy that depends on theta (bonded terms, a backbone map, the
weights of the neural bonded network through its frozen coefficients).  Everything is JAX, so
d<O>/dtheta (at theta0: -beta (<O dU/dtheta> - <O><dU/dtheta>)) and the gradient of any loss
of the averages come from autodiff.  The Kish effective sample size n_eff = (sum w)^2 / sum w^2
says when theta has moved too far from the sampled ensemble (resample).

    frames, _, _ = read_trajectory("prod.nc", atoms=protein_atoms)          # A
    X = frames * 0.1
    rw = Reweighting(lambda th, R: terms.bonded_energy(0, R, with_cmap(P, th)), th0, X, temperature=298.0)
    phi, psi = backbone_torsions(X, terms.mols[0].top)
    J = karplus(phi, *KARPLUS["3J_HNHA_Vogeli2007"])                         # (frames, residues)
    loss, grad = rw.chi2_and_grad(th, [(J, J_exp, 0.5)])

The reweighting is exact for the energy function given (no linearisation), but its variance
grows as theta leaves theta0; watch n_eff.

Units: kJ/mol, K, rad, Hz, nm.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

from ..bonded import terms as T
from ..units import KB

# Karplus relations J = A cos^2(theta) + B cos(theta) + C with theta = phi + delta (rad).
# Only sets whose provenance is stated; add others as (A, B, C, delta) from the paper you use.
KARPLUS = {
    "3J_HNHA_Vogeli2007": (7.97, -1.26, 0.63, -np.pi / 3),  # Vogeli, Ying, Grishaev, Bax, JACS 129, 9377 (2007)
}


def karplus(phi: ArrayLike, A: float, B: float, C: float, delta: float) -> jax.Array:
    """Return the Karplus J coupling A cos^2(phi + delta) + B cos(phi + delta) + C [Hz].

    Parameters
    ----------
    phi : ArrayLike
        Backbone torsions [rad].
    A, B, C : float
        Karplus coefficients [Hz] (e.g. KARPLUS["3J_HNHA_Vogeli2007"]).
    delta : float
        Phase offset [rad].

    Returns
    -------
    jax.Array
        J couplings [Hz], same shape as `phi`.
    """
    c = jnp.cos(phi + delta)
    return A * c * c + B * c + C


def backbone_torsions(frames: ArrayLike, top: Any) -> tuple[jax.Array, jax.Array]:
    """Return the backbone torsions phi, psi of every residue that has both.

    Parameters
    ----------
    frames : ArrayLike (F, N, 3)
        Positions [nm] (any length unit works; torsions are scale-free).
    top : Topology
        Bonded topology whose `cmaps` rows (the five atoms C-N-CA-C-N of each residue with phi and
        psi) define the torsions (bonded/topology.py).

    Returns
    -------
    phi, psi : jax.Array (F, residues)
        Torsions [rad] in the IUPAC sign convention (bonded.terms.phi_psi, vmapped over frames).
    """
    q = np.asarray(top.cmaps)
    return jax.vmap(lambda R: T.phi_psi(R, q))(jnp.asarray(frames))


def in_region(
    phi: ArrayLike, psi: ArrayLike, phi_range: tuple[float, float], psi_range: tuple[float, float]
) -> jax.Array:
    """Return 1.0 where (phi, psi) lies in a box of the Ramachandran plot, else 0.0.

    `phi`, `psi` in rad; the ranges (lower, upper) in degrees, lower <= x < upper (e.g. ALPHA_BOX).
    The indicator is piecewise constant, so its reweighted average is differentiable in theta only
    through the weights.
    """
    p, s = jnp.degrees(phi), jnp.degrees(psi)
    return ((p >= phi_range[0]) & (p < phi_range[1]) & (s >= psi_range[0]) & (s < psi_range[1])).astype(float)


ALPHA_BOX = ((-100.0, -30.0), (-67.0, -7.0))  # a common helical basin definition (degrees)


class Reweighting:
    """Reweight stored frames from parameters theta0 to theta (see the module docstring).

    The frames are held on the device; energies are recomputed at every theta with `energy_fn`
    (vmapped over frames, or lax.map in chunks).  Every method is a JAX function of theta, so
    jax.grad of a loss of the averages is the reweighting gradient.  Not a pytree.

    Attributes
    ----------
    f : callable
        energy_fn(theta, R) [kJ/mol].
    X : jax.Array (F, N, 3)
        Frames [nm].
    beta : float
        1 / (kB T) [mol/kJ].
    chunk : int or None
        Frames per lax.map batch (None: one vmap over all frames).
    u0 : jax.Array (F,)
        Energies at theta0 [kJ/mol] (stop_gradient).
    """

    def __init__(
        self,
        energy_fn: Callable[[Any, jax.Array], jax.Array],
        theta0: Any,
        frames: ArrayLike,
        temperature: float = 298.0,
        chunk: int | None = None,
    ) -> None:
        """Store the frames and their energies at theta0.

        Parameters
        ----------
        energy_fn : callable
            energy_fn(theta, R) -> the theta-dependent energy [kJ/mol] of frame R (N, 3) [nm].
        theta0 : pytree
            Parameters the frames were sampled with.
        frames : ArrayLike (F, N, 3)
            Frames [nm].
        temperature : float
            Temperature of the sampling [K].
        chunk : int, optional
            Evaluate the energies with lax.map in batches of `chunk` frames (bounded memory); None: one
            vmap.
        """
        self.f = energy_fn
        self.X = jnp.asarray(frames)
        self.beta = 1.0 / (KB * float(temperature))
        self.chunk = chunk
        self.u0 = jax.lax.stop_gradient(self.energies(theta0))

    def energies(self, theta: Any) -> jax.Array:
        """Return the theta-dependent energies (F,) [kJ/mol] of all frames."""

        def one(R: jax.Array) -> jax.Array:
            return self.f(theta, R)

        if self.chunk is None:
            return jax.vmap(one)(self.X)
        return jax.lax.map(one, self.X, batch_size=self.chunk)

    def log_weights(self, theta: Any) -> jax.Array:
        """Return the normalised log weights (F,) -beta (U(theta) - U(theta0)) - logsumexp(...)."""
        lw = -self.beta * (self.energies(theta) - self.u0)
        return lw - logsumexp(lw)

    def weights(self, theta: Any) -> jax.Array:
        """Return the normalised frame weights w_k(theta) (F,), summing to 1."""
        return jnp.exp(self.log_weights(theta))

    def average(self, theta: Any, values: ArrayLike) -> jax.Array:
        """Return <O>_theta = sum_k w_k O_k for per-frame values (F, ...) (shape (...), units of O)."""
        w = self.weights(theta)
        v = jnp.asarray(values)
        return jnp.tensordot(w, v, axes=(0, 0))

    def n_eff(self, theta: Any) -> jax.Array:
        """Return Kish's effective sample size 1 / sum w_k^2 (between 1 and F) of the weights at theta."""
        w = self.weights(theta)
        return 1.0 / jnp.sum(w * w)

    def chi2(self, theta: Any, observables: Sequence[tuple[ArrayLike, ArrayLike, float]]) -> jax.Array:
        """Return sum over observables of mean(((<O>_theta - target) / sigma)^2).

        Parameters
        ----------
        theta : pytree
            Parameters.
        observables : sequence of (values, target, sigma)
            values (F, ...) per-frame observable, target (...) experimental value, sigma its
            uncertainty (same units); the mean runs over the components of each observable.

        Returns
        -------
        jax.Array ()
            chi^2 (dimensionless).
        """
        tot = 0.0
        for values, target, sigma in observables:
            tot = tot + jnp.mean(((self.average(theta, values) - jnp.asarray(target)) / sigma) ** 2)
        return tot

    def chi2_and_grad(
        self, theta: Any, observables: Sequence[tuple[ArrayLike, ArrayLike, float]]
    ) -> tuple[jax.Array, Any]:
        """Return (chi2, d chi2/d theta) by jax.value_and_grad (arguments as in `chi2`)."""
        return jax.value_and_grad(self.chi2)(theta, observables)
