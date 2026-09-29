"""Ensemble averages and their parameter gradients by reweighting: the top-down step of force-field
refinement (J couplings, helicities, populations from MD against experiment).

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

Units: kJ/mol, K, rad, Hz."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from .bonded import terms as T

KB = 0.0083144626181532  # kJ/mol/K

# Karplus relations J = A cos^2(theta) + B cos(theta) + C with theta = phi + delta (rad).
# Only sets whose provenance is stated; add others as (A, B, C, delta) from the paper you use.
KARPLUS = {
    "3J_HNHA_Vogeli2007": (7.97, -1.26, 0.63, -np.pi / 3),  # Vogeli, Ying, Grishaev, Bax, JACS 129, 9377 (2007)
}


def karplus(phi, A, B, C, delta):
    """J coupling (Hz) of backbone torsions phi (rad)."""
    c = jnp.cos(phi + delta)
    return A * c * c + B * c + C


def backbone_torsions(frames, top):
    """phi, psi (rad) of every residue with both torsions (Topology.cmaps), frames (F, N, 3):
    two (F, residues) arrays in the IUPAC sign convention."""
    q = np.asarray(top.cmaps)
    return jax.vmap(lambda R: T.phi_psi(R, q))(jnp.asarray(frames))


def in_region(phi, psi, phi_range, psi_range):
    """1 where (phi, psi) lies in the box (degrees, lower <= x < upper), else 0."""
    p, s = jnp.degrees(phi), jnp.degrees(psi)
    return ((p >= phi_range[0]) & (p < phi_range[1]) & (s >= psi_range[0]) & (s < psi_range[1])).astype(float)


ALPHA_BOX = ((-100.0, -30.0), (-67.0, -7.0))  # a common helical basin definition (degrees)


class Reweighting:
    def __init__(self, energy_fn, theta0, frames, temperature: float = 298.0, chunk: int | None = None):
        """energy_fn(theta, R) -> the theta-dependent energy (kJ/mol) of frame R (n, 3) nm."""
        self.f = energy_fn
        self.X = jnp.asarray(frames)
        self.beta = 1.0 / (KB * float(temperature))
        self.chunk = chunk
        self.u0 = jax.lax.stop_gradient(self.energies(theta0))

    def energies(self, theta):
        def one(R):
            return self.f(theta, R)

        if self.chunk is None:
            return jax.vmap(one)(self.X)
        return jax.lax.map(one, self.X, batch_size=self.chunk)

    def log_weights(self, theta):
        lw = -self.beta * (self.energies(theta) - self.u0)
        return lw - logsumexp(lw)

    def weights(self, theta):
        return jnp.exp(self.log_weights(theta))

    def average(self, theta, values):
        """<O>_theta for per-frame values (F, ...)."""
        w = self.weights(theta)
        v = jnp.asarray(values)
        return jnp.tensordot(w, v, axes=(0, 0))

    def n_eff(self, theta) -> float:
        w = self.weights(theta)
        return 1.0 / jnp.sum(w * w)

    def chi2(self, theta, observables):
        """sum over observables (values (F, ...), target (...), sigma) of mean(((<O> - target)/sigma)^2)."""
        tot = 0.0
        for values, target, sigma in observables:
            tot = tot + jnp.mean(((self.average(theta, values) - jnp.asarray(target)) / sigma) ** 2)
        return tot

    def chi2_and_grad(self, theta, observables):
        return jax.value_and_grad(self.chi2)(theta, observables)
