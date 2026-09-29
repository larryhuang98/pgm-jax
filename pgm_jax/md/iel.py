"""Extended-Lagrangian induced dipoles (MDSettings.iel; the engine is in forcefield.py): command-line
options for the scripts and the linear stability of the auxiliary-dipole recurrence.

    MDSettings(iel="0scf")                    # iEL/0-SCF: one field sweep per step, no CG
    MDSettings(iel="scf", iel_iter=2)         # iEL/SCF: two CG iterations from the auxiliary dipoles

docs/iel.md has the equations, the validation and the speed."""

from __future__ import annotations

import numpy as np

from .forcefield import _XL


def add_iel_arguments(ap):
    """--iel, --iel-iter, --iel-order, --iel-kappa, --iel-alpha."""
    ap.add_argument(
        "--iel",
        default="none",
        choices=["none", "0scf", "scf"],
        help="extended-Lagrangian induced dipoles: 0scf (iEL/0-SCF, no CG, shadow forces) or scf "
        "(--iel-iter CG iterations from the auxiliary dipoles); none: predictor + CG to tolerance",
    )
    ap.add_argument("--iel-iter", type=int, default=1, help="--iel scf: CG iterations per step (0: to tolerance)")
    ap.add_argument("--iel-order", type=int, default=7, help="Niklasson dissipation order K (0, 3..9)")
    ap.add_argument("--iel-kappa", type=float, default=None, help="kappa = (omega dt)^2 (default: Niklasson's for K)")
    ap.add_argument("--iel-alpha", type=float, default=None, help="dissipation strength (default: Niklasson's for K)")
    ap.add_argument("--iel-omega", type=float, default=1.0, help="--iel 0scf: mu = x + omega alpha r(x)")
    ap.add_argument(
        "--iel-precond",
        default="block",
        choices=["jacobi", "block"],
        help="--iel 0scf: delta = alpha r (jacobi) or M^-1 r with the intramolecular blocks (block)",
    )
    ap.add_argument(
        "--iel-no-shadow",
        action="store_true",
        help="--iel 0scf: fixed-dipole forces at mu = x + alpha r instead of the exact forces of the shadow energy",
    )


def iel_settings(a) -> dict:
    """MDSettings keyword arguments from add_iel_arguments' options."""
    return dict(
        iel=a.iel,
        iel_iter=a.iel_iter,
        iel_order=a.iel_order,
        iel_kappa=a.iel_kappa,
        iel_alpha=a.iel_alpha,
        iel_shadow=not a.iel_no_shadow,
        iel_omega=a.iel_omega,
        iel_precond=a.iel_precond,
    )


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
    precond = ff.s.iel_precond if precond is None else precond
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
