"""Linear stability and response spectrum of the extended-Lagrangian induced dipoles.

Diagnostics for `MDSettings.induction.iel` (the engine is in md/forcefield.py, the command-line
options in cli/args.py): `spectral_radius`, the damping factor per step of the auxiliary-dipole
recurrence, and `response_spectrum`, the extreme eigenvalues of the preconditioned response
W A at one configuration.

    MDSettings().replace(iel="0scf")                    # iEL/0-SCF: one field sweep per step, no CG
    MDSettings().replace(iel="scf", iel_iter=2)         # iEL/SCF: two CG iterations from the auxiliary dipoles

The auxiliary dipoles x follow Niklasson's dissipative Verlet recurrence [1]_

    x_{n+1} = 2 x_n - x_{n-1} + kappa (mu_n - x_n) + a sum_{k=0..K} c_k x_{n-k},

with (kappa, a, c) from md/forcefield.py `_XL` for the order K.  For iEL/0-SCF [2]_ the dipoles
of step n are mu_n = x_n + omega W r(x_n), with r = b - A mu the residual, A = 1/alpha - T the
dipole Hessian and W = alpha (Jacobi) or M^-1 (block preconditioner); near the solution mu*
this is a linear response mu - x = -lam (x - mu*) with lam an eigenvalue of omega W A.

docs/iel.md has the equations, the validation and the speed.

Units: dimensionless (eigenvalues, spectral radii); positions in nm.

References
----------
.. [1] A. M. N. Niklasson, P. Steneteg, A. Odell, N. Bock, M. Challacombe, C. J. Tymczak,
   E. Holmstrom, G. Zheng, V. Weber, J. Chem. Phys. 130, 214109 (2009).
.. [2] A. Albaugh, A. M. N. Niklasson, T. Head-Gordon, J. Phys. Chem. Lett. 8, 1714 (2017).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from jax.typing import ArrayLike

from .forcefield import _XL

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax

    from .forcefield import PGMForceField


def spectral_radius(lam: float, order: int = 5, kappa: float | None = None, alpha: float | None = None) -> float:
    """Return the largest |root| of the linearized error recurrence of the auxiliary dipoles.

    The error e = x - mu* of the auxiliary dipoles obeys

        e_{n+1} = (2 - kappa lam) e_n - e_{n-1} + a sum_k c_k e_{n-k}

    for a response mu - x = -lam (x - mu*).  Below 1 the auxiliary dipoles are stable and their
    errors decay by this factor per step.

    Parameters
    ----------
    lam : float
        Eigenvalue of the response (of omega W A for iEL/0-SCF, e.g. alpha A for Jacobi with
        omega = 1; 1 for an exact solve), dimensionless.
    order : int
        Dissipation order K (a key of `_XL`: 0, 3-9).
    kappa : float, optional
        kappa = (omega_x dt)^2 (None: Niklasson's value for K).
    alpha : float, optional
        Dissipation strength a (None: Niklasson's value for K).

    Returns
    -------
    float
        Spectral radius of the companion matrix of the recurrence (dimensionless).

    Notes
    -----
    pGM water: alpha A has eigenvalues 0.68-1.85, where K = 5 gives at most 0.968; M^-1 A has
    0.71-1.56.
    """
    k0, a0, c = _XL[order]
    kappa = k0 if kappa is None else kappa
    alpha = a0 if alpha is None else alpha
    n = max(len(c), 2)
    M = np.zeros((n, n))
    row = np.zeros(n)
    row[: len(c)] = alpha * np.asarray(c, float)
    row[0] += 2.0 - kappa * lam
    row[1] += -1.0
    M[0] = row  # companion matrix: first row the recurrence, sub-diagonal shifts the history
    M[np.arange(1, n), np.arange(n - 1)] = 1.0
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def response_spectrum(
    ff: PGMForceField,
    pos: ArrayLike,
    H: ArrayLike,
    idx: jax.Array,
    params: dict[str, jax.Array] | None = None,
    iters: int = 60,
    precond: str | None = None,
) -> tuple[float, float]:
    """Return the smallest and largest eigenvalue of W A, the linear response of the iEL/0-SCF step.

    The response is mu - x = -omega W A (x - mu*) with W = alpha (Jacobi), or M^-1 for the block
    preconditioner; the eigenvalues are found by power iteration at one configuration (float64
    host loop, the operator jitted).  The auxiliary modes carry energy of the sign of
    1 - omega lambda: with omega lambda_max < 1 every mode is positive and the propagation is stable
    without dissipation (docs/iel.md).  pGM water: lambda in 0.68-1.85 (Jacobi), 0.71-1.56 (block).

    Parameters
    ----------
    ff : PGMForceField
        The force field (md/forcefield.py).  With precond="block" its block layout is built and
        cached on `ff` if missing.
    pos : ArrayLike (N, 3)
        Positions [nm].
    H : ArrayLike (3, 3)
        Box, lattice vectors as rows [nm].
    idx : jax.Array
        Candidate neighbour rows, as for PGMForceField.compute.
    params : dict of str to jax.Array, optional
        Parameter pytree (None: the force field's own parameters).
    iters : int
        Power iterations for each of the two eigenvalues.
    precond : {"jacobi", "block"}, optional
        Preconditioner W (None: the force field's `induction.iel.precond`).

    Returns
    -------
    lmin : float
        Smallest eigenvalue of W A (dimensionless), by power iteration on 1.02 lmax - W A.
    lmax : float
        Largest eigenvalue of W A (dimensionless).
    """
    import jax
    import jax.numpy as jnp

    pos, H = jnp.asarray(pos, jnp.float64), jnp.asarray(H, jnp.float64)
    precond = ff.s.induction.iel.precond if precond is None else precond
    P = ff._atoms(params)
    g = ff.geometry(pos, H, idx, P)
    S, Gk = ff.pme.setup(pos, H), ff.pme.influence(H)
    alpha = P["alpha"]
    A = ff._operator(g, S, Gk, alpha)
    if precond == "block":
        if ff._blocks is None:
            ff._blocks = ff._block_layout(np.asarray(ff.sys.mol), np.asarray(ff.first), ff.sys.nmol)

        def W(v: jax.Array) -> jax.Array:  # M^-1 v
            return ff._block_solve(g, alpha, v)
    else:

        def W(v: jax.Array) -> jax.Array:  # alpha v
            return alpha[:, None] * v

    WA = jax.jit(lambda v: W(A(v.astype(ff.cd)).astype(jnp.float64)))

    def power(op: Callable[[jax.Array], jax.Array], v: jax.Array) -> float:
        """Return the Rayleigh quotient of `op` after `iters` normalized power iterations from v."""
        lam = 0.0
        for _ in range(iters):
            w = op(v)
            lam = float(jnp.vdot(v, w) / jnp.vdot(v, v))
            v = w / jnp.linalg.norm(w)
        return lam

    v0 = jax.random.normal(jax.random.PRNGKey(0), (ff.n, 3), jnp.float64)
    lmax = power(WA, v0)
    shift = 1.02 * lmax  # largest eigenvalue of shift - W A is shift - lmin
    lmin = shift - power(lambda v: shift * v - WA(v), v0)
    return lmin, lmax
