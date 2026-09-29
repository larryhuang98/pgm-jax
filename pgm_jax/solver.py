"""Variational solvers for the response variables (induced dipoles now; widths and
charge transfer later).

The energy of a polarizable model is E = min_x F(x; R, theta).  At the minimum,
  * forces are dE/dR = dF/dR at fixed x (Hellmann-Feynman), and
  * parameter gradients dE/dtheta = dF/dtheta at fixed x,
so neither needs dx/dR.  JAX gets both automatically as long as the solve is either a
differentiable linear solve (quadratic F) or wrapped in `lax.custom_root` (general F).
`variational` makes the first derivatives cost one solve while keeping higher derivatives exact.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def solve_linear_induction(T, alpha, F):
    """Quadratic functional  sum |mu|^2/2a - mu.F + 1/2 sum_{i!=j} mu_i T_ij mu_j.
    T: (n, n, 3, 3) with zero diagonal blocks; alpha: (n,); F: (n, 3).  Returns mu (n, 3)."""
    n = alpha.shape[0]
    A = T.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n) + jnp.diag(jnp.repeat(1.0 / alpha, 3))
    return jnp.linalg.solve(A, F.reshape(-1)).reshape(n, 3)


def minimize_newton(F, x0, *args, iters: int = 30, tol: float = 1e-10):
    """Minimise a smooth convex functional F(x, *args) over a flat vector x by Newton's method.
    Differentiable in args via implicit differentiation (custom_root on grad F = 0).
    Used for non-quadratic self-energies (breathing widths, gapped charge transfer)."""
    g = jax.grad(F, 0)

    def solve(fun, x):
        def body(state):
            x, k, _ = state
            H = jax.hessian(F, 0)(x, *args)
            dx = jnp.linalg.solve(H, g(x, *args))
            return x - dx, k + 1, jnp.max(jnp.abs(dx))

        def cond(state):
            return (state[1] < iters) & (state[2] > tol)

        x, _, _ = jax.lax.while_loop(cond, body, (x, 0, jnp.inf))
        return x

    def tangent_solve(f, y):
        J = jax.jacobian(f)(jnp.zeros_like(y))
        return jnp.linalg.solve(J, y)

    return jax.lax.custom_root(lambda x: g(x, *args), x0, solve, tangent_solve)


def variational(G, solve):
    """E(theta) = G(x*(theta), theta), with x* = solve(theta) the stationary point of G(., theta).

    First derivatives use stationarity, dE/dtheta = dG/dtheta at fixed x* (one solve, no
    derivative of the solve: Hellmann-Feynman forces, virials and parameter gradients).  Higher
    derivatives (force matching, Hessians, gradients of forces or virials with respect to
    parameters) differentiate this rule itself, which includes dx*/dtheta through `solve`, so
    `solve` must be differentiable (a dense solve, or lax.custom_linear_solve such as
    jax.scipy.sparse.linalg.cg).  theta may be any pytree of float arrays."""

    @jax.custom_jvp
    def E(theta):
        return G(solve(theta), theta)

    @E.defjvp
    def _E_jvp(primals, tangents):
        (theta,), (dtheta,) = primals, tangents
        x = solve(theta)
        return jax.jvp(lambda t: G(x, t), (theta,), (dtheta,))

    return E
