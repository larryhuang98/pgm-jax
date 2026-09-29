"""Solve for the response variables of a polarizable model (induced dipoles now).

Contents: solve_linear_induction (dense direct solve of the quadratic induction functional),
minimize_newton (Newton minimisation of a general convex functional, implicitly differentiable),
variational (wrap an energy E(theta) = G(x*(theta), theta) so that first derivatives use
stationarity).  Widths and charge transfer as response variables are planned, not implemented.

The energy of a polarizable model is E = min_x F(x; R, theta).  At the minimum,

  * forces are dE/dR = dF/dR at fixed x (Hellmann-Feynman), and
  * parameter gradients dE/dtheta = dF/dtheta at fixed x,

so neither needs dx/dR.  JAX gets both automatically as long as the solve is either a
differentiable linear solve (quadratic F) or wrapped in `lax.custom_root` (general F).
`variational` makes the first derivatives cost one solve while keeping higher derivatives exact.

Units: those of the caller (induction: e nm, nm^3, e/nm^2; see channels.py).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


def solve_linear_induction(T: jax.Array, alpha: jax.Array, F: jax.Array) -> jax.Array:
    """Return the induced dipoles that minimise the quadratic induction functional (dense solve).

    The functional is

        G(mu) = sum_i |mu_i|^2 / (2 alpha_i) - sum_i mu_i . F_i + 1/2 sum_{i != j} mu_i T_ij mu_j,

    and its minimum solves the 3n x 3n linear system (T + diag(1/alpha)) mu = F.

    Parameters
    ----------
    T : jax.Array (n, n, 3, 3)
        Dipole-dipole interaction tensors with zero diagonal blocks [1/nm^3], T_ij = d^2 phi_ij /
        dr_i dr_j of the pair kernel, so that the field at i of a dipole mu_j is -T_ij mu_j (without
        the Coulomb constant).
    alpha : jax.Array (n,)
        Isotropic atomic polarizabilities [nm^3]; must be nonzero.
    F : jax.Array (n, 3)
        External field on each atom (from the permanent multipoles and any applied field), without
        the Coulomb constant [e/nm^2].

    Returns
    -------
    jax.Array (n, 3)
        Induced dipoles [e nm].

    Notes
    -----
    One dense LU solve of size 3n (O(n^3)); meant for molecules and small clusters.  Differentiable
    in all arguments (jnp.linalg.solve).
    """
    n = alpha.shape[0]
    A = T.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n) + jnp.diag(jnp.repeat(1.0 / alpha, 3))
    return jnp.linalg.solve(A, F.reshape(-1)).reshape(n, 3)


def minimize_newton(
    F: Callable[..., jax.Array], x0: jax.Array, *args: Any, iters: int = 30, tol: float = 1e-10
) -> jax.Array:
    """Minimise a smooth convex functional F(x, *args) over a flat vector x by Newton's method.

    Meant for non-quadratic self-energies (breathing widths, gapped charge transfer); the result is
    differentiable in `args` by implicit differentiation (lax.custom_root on grad F = 0), not by
    differentiating the iterations.

    Parameters
    ----------
    F : callable
        F(x, *args) -> scalar; twice differentiable in x, with a positive-definite Hessian.
    x0 : jax.Array (m,)
        Starting point.
    *args
        Further arguments of F (differentiable).
    iters : int
        Maximum number of Newton steps.
    tol : float
        Stop when the largest component of the Newton step |dx| is at most `tol` (units of x).

    Returns
    -------
    jax.Array (m,)
        The minimiser (the last iterate if `iters` is reached; no error is raised).

    Notes
    -----
    Each step solves H dx = grad F with the full Hessian H = jax.hessian(F) (dense, O(m^3)).  The
    tangent solve of custom_root uses the Jacobian of grad F (= H) evaluated at zero, which is exact
    for the linear map custom_root passes it.  The loop is a lax.while_loop, so the function can be
    jitted; reverse-mode derivatives come from the implicit-function theorem.
    """
    g = jax.grad(F, 0)

    def solve(fun: Callable[[jax.Array], jax.Array], x: jax.Array) -> jax.Array:
        """Run the Newton iterations from `x` (the `solve` of lax.custom_root; `fun` is unused)."""

        def body(state: tuple[jax.Array, jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
            """Take one Newton step x -> x - H^-1 grad F and record the largest step component."""
            # carry: (iterate x, step count, max |Newton step| of the last step)
            x, k, _ = state
            H = jax.hessian(F, 0)(x, *args)
            dx = jnp.linalg.solve(H, g(x, *args))
            return x - dx, k + 1, jnp.max(jnp.abs(dx))

        def cond(state: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
            return (state[1] < iters) & (state[2] > tol)

        x, _, _ = jax.lax.while_loop(cond, body, (x, 0, jnp.inf))
        return x

    def tangent_solve(f: Callable[[jax.Array], jax.Array], y: jax.Array) -> jax.Array:
        # f is linear here (the JVP of grad F at the root), so its Jacobian at 0 is the Hessian
        J = jax.jacobian(f)(jnp.zeros_like(y))
        return jnp.linalg.solve(J, y)

    return jax.lax.custom_root(lambda x: g(x, *args), x0, solve, tangent_solve)


def variational(G: Callable[[Any, Any], jax.Array], solve: Callable[[Any], Any]) -> Callable[[Any], jax.Array]:
    """Return E(theta) = G(x*(theta), theta), with x* = solve(theta) the stationary point of G(., theta).

    First derivatives use stationarity, dE/dtheta = dG/dtheta at fixed x* (one solve, no
    derivative of the solve: Hellmann-Feynman forces, virials and parameter gradients).  Higher
    derivatives (force matching, Hessians, gradients of forces or virials with respect to
    parameters) differentiate this rule itself, which includes dx*/dtheta through `solve`, so
    `solve` must be differentiable (a dense solve, or lax.custom_linear_solve such as
    jax.scipy.sparse.linalg.cg).

    Parameters
    ----------
    G : callable
        G(x, theta) -> scalar energy functional, stationary in x at x*(theta).
    solve : callable
        solve(theta) -> x*, the stationary point (e.g. the induced dipoles).

    Returns
    -------
    callable
        E(theta) -> scalar, a jax.custom_jvp function.  theta may be any pytree of float arrays.

    Notes
    -----
    The custom JVP rule computes x = solve(theta) and returns jax.jvp of G(x, .) at fixed x.  The
    first derivative is exact only if solve converges (an iterative solve with tolerance tol gives
    an error of order tol^2 in E and of order tol in dE/dtheta).
    """

    @jax.custom_jvp
    def E(theta: Any) -> jax.Array:
        """Energy at the stationary point: G(solve(theta), theta)."""
        return G(solve(theta), theta)

    @E.defjvp
    def _E_jvp(primals: tuple[Any], tangents: tuple[Any]) -> tuple[jax.Array, jax.Array]:
        """JVP of E by stationarity: the tangent of G(x*, .) at fixed x* = solve(theta)."""
        (theta,), (dtheta,) = primals, tangents
        x = solve(theta)
        return jax.jvp(lambda t: G(x, t), (theta,), (dtheta,))

    return E
