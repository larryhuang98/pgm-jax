"""Extended-Lagrangian induced dipoles (MDSettings.induction.iel; the engine is in forcefield.py): the linear
stability of the auxiliary-dipole recurrence and the response spectrum (command-line options:
cli/args.py).

    MDSettings().replace(iel="0scf")                    # iEL/0-SCF: one field sweep per step, no CG
    MDSettings().replace(iel="scf", iel_iter=2)         # iEL/SCF: two CG iterations from the auxiliary dipoles

docs/iel.md has the equations, the validation and the speed."""

from __future__ import annotations

import numpy as np

from .forcefield import _XL


def spectral_radius(lam, order: int = 5, kappa: float | None = None, alpha: float | None = None) -> float:
    """Largest |root| of the error recurrence e_{n+1} = 2 e_n - e_{n-1} - kappa lam e_n + a sum_k c_k e_{n-k}
    for a response mu - x = -lam (x - mu*) (lam: an eigenvalue of alpha A for iEL/0-SCF, 1 for an exact
    solve).  Below 1: the auxiliary dipoles are stable and their errors decay by this factor per step
    (pGM water: alpha A has eigenvalues 0.68-1.85, where K = 5 gives at most 0.968; M^-1 A 0.71-1.56)."""
    k0, a0, c = _XL[order]
    kappa = k0 if kappa is None else kappa
    alpha = a0 if alpha is None else alpha
    n = max(len(c), 2)
    M = np.zeros((n, n))
    row = np.zeros(n)
    row[: len(c)] = alpha * np.asarray(c, float)
    row[0] += 2.0 - kappa * lam
    row[1] += -1.0
    M[0] = row
    M[np.arange(1, n), np.arange(n - 1)] = 1.0
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def response_spectrum(ff, pos, H, idx, params=None, iters: int = 60, precond: str | None = None) -> tuple[float, float]:
    """Smallest and largest eigenvalue of W A, the linear response of the iEL/0-SCF step
    (mu - x = -omega W A (x - mu*) with W = alpha, or M^-1 for the block preconditioner), by power
    iteration at one configuration (pos, H; candidate rows idx as for PGMForceField.compute).

    The auxiliary modes carry energy of the sign of 1 - omega lambda: with omega lambda_max < 1 every
    mode is positive and the propagation is stable without dissipation (docs/iel.md); pGM water:
    lambda in 0.68-1.85 (Jacobi), 0.71-1.56 (block)."""
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

        def W(v):
            return ff._block_solve(g, alpha, v)
    else:

        def W(v):
            return alpha[:, None] * v

    WA = jax.jit(lambda v: W(A(v.astype(ff.cd)).astype(jnp.float64)))

    def power(op, v):
        lam = 0.0
        for _ in range(iters):
            w = op(v)
            lam = float(jnp.vdot(v, w) / jnp.vdot(v, v))
            v = w / jnp.linalg.norm(w)
        return lam

    v0 = jax.random.normal(jax.random.PRNGKey(0), (ff.n, 3), jnp.float64)
    lmax = power(WA, v0)
    shift = 1.02 * lmax
    lmin = shift - power(lambda v: shift * v - WA(v), v0)
    return lmin, lmax
