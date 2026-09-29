"""Compute periodic pGM electrostatics by Ewald summation (triclinic boxes).

Everything is differentiable in the coordinates, the parameters and the box.  Contents:
neighbor_list (pairs and integer images within a cutoff, numpy), kvector_indices (the
reciprocal-space index set, numpy), _pair_tensors (direct-space kernel and its derivatives) and
PeriodicPGM (the model: energy, forces, induced dipoles).  PeriodicModel (periodic.py) adds
van der Waals, virial and pressure.

Same model as `channels.ElecChannel` (Gaussian charges + covalent dipoles + induced Gaussian
dipoles, all pairs, no masking), under periodic boundary conditions, following Wei et al.
[1]_, Sec. II D, with a plain Ewald reciprocal sum instead of PME:

  pair kernel  erf(b_ij r)/r = [erf(b_ij r) - erf(b0 r)]/r   (direct, short range, r < rc)
                             +  erf(b0 r)/r                   (reciprocal, all pairs and images)
  self term    -(b0/sqrt(pi)) sum q^2 - (2 b0^3/(3 sqrt(pi))) sum |d|^2
  background   -pi Q^2 / (2 V b0^2) for a net charge Q (zero for neutral systems; keeps the
               energy independent of b0 while fitted charges drift)
  tin-foil boundary (no surface term), as in Amber.

with b_ij the Gaussian pair exponent [1/nm] (densities.gauss_bij), b0 = ewald_beta [1/nm], and d
the total dipole of each atom [e nm].  In reciprocal space each atom is a point charge and dipole
smeared with the Ewald Gaussian: the energy is

    U_rec = (2 pi / V) sum_{k != 0} exp(-k^2 / (4 b0^2)) / k^2 |S(k)|^2,
    S(k) = sum_i (q_i + i k.d_i) exp(i k.r_i),

summed over half of k-space and doubled.  The direct part is truncated at the cutoff with a hard
mask (no switching), the reciprocal part at exp(-k^2 / (4 b0^2)) < k_tol.

With d = p + mu (all dipoles), U(q, d) is quadratic in d.  The induced dipoles minimise
G(mu) = U(q, p + mu) + sum |mu|^2/(2 alpha); G(mu*) is the total electrostatic energy (EELEC in
Amber).  mu* comes from conjugate gradients with exact Hessian-vector products; the energy is
wrapped with `solver.variational`, so first derivatives (forces, virial, parameter gradients)
cost one solve and higher derivatives differentiate through the solve.

Box: H with lattice vectors as ROWS (nm).  The neighbour list holds integer image vectors n
(displacement = r_i - r_j + n.H) and the reciprocal sum holds integer indices m
(k = 2 pi m.H^-T), both chosen once at a reference geometry and box; displacements and
k-vectors are computed from H inside JAX, so H can be differentiated.  Molecules must be whole
(covalent dipoles use plain coordinate differences).  The neighbour list contains only pairs
i < j and searches the 27 cells around the minimum image, so the cutoff plus skin should not
exceed half the shortest box width (an atom's interaction with its own images is not in the
direct sum).

    model = PeriodicPGM(system, box, positions, ewald_beta=3.8, cutoff=1.0)
    energies, aux = model.energy(positions, params, box)   # {"perm", "ind", "total"}

Units: nm, e, e nm, nm^3, kJ/mol.

References
----------
.. [1] H. Wei, R. Qi, J. Wang, P. Cieplak, Y. Duan, R. Luo, J. Chem. Phys. 153, 114116 (2020).

See also md/pme.py (the PME version used by the MD engine).
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf
from jax.typing import ArrayLike

from .channels import perm_dipoles
from .densities import gauss_bij
from .solver import variational
from .system import System
from .units import KE

SQRT_PI = 1.7724538509055159  # sqrt(pi)


def neighbor_list(
    pos: np.ndarray, H: np.ndarray, rc: float, chunk: int = 200_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return all pairs i < j and lattice images n with |r_i - r_j + n.H| < rc (host numpy).

    Parameters
    ----------
    pos : np.ndarray (N, 3)
        Positions [nm].
    H : np.ndarray (3, 3)
        Box, lattice vectors as rows [nm].
    rc : float
        Cutoff [nm] (cutoff plus skin); at most half the shortest box width for a complete list.
    chunk : int
        Number of pairs processed at once (memory bound: chunk x 27 x 3 floats).

    Returns
    -------
    i, j : np.ndarray (P,) int
        Atom indices, i < j; a pair appears once per image within the cutoff.
    n : np.ndarray (P, 3) int32
        Integer image vectors: the displacement is r_i - r_j + n.H.

    Notes
    -----
    Each displacement is first reduced to the image nearest the origin (rounding its fractional
    coordinates), then the 27 neighbouring images are tested.  O(N^2) work; meant for reference
    models and tests, not for MD (md/neighbors.py).
    """
    pos, H = np.asarray(pos, float), np.asarray(H, float)
    n = len(pos)
    ii, jj = np.triu_indices(n, k=1)
    cells = np.array([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], float)
    shifts = cells @ H
    # bring each displacement to the reduced cell first, then test the 27 neighbours
    Hinv = np.linalg.inv(H)
    out_i, out_j, out_n = [], [], []
    for s in range(0, len(ii), chunk):
        a, b = ii[s : s + chunk], jj[s : s + chunk]
        d = pos[a] - pos[b]
        base = -np.round(d @ Hinv)  # integer: d + base.H is near the origin
        dd = (d + base @ H)[:, None, :] + shifts[None, :, :]  # candidate displacements
        r = np.linalg.norm(dd, axis=-1)
        k, m = np.nonzero(r < rc)
        out_i.append(a[k])
        out_j.append(b[k])
        out_n.append(base[k] + cells[m])
    i = np.concatenate(out_i)
    j = np.concatenate(out_j)
    img = np.rint(np.concatenate(out_n)).astype(np.int32)
    return i, j, img


def kvector_indices(H: np.ndarray, kcut: float) -> np.ndarray:
    """Return the integer indices m (half space) of the reciprocal vectors k = 2 pi m.H^-T with 0 < |k| < kcut.

    Parameters
    ----------
    H : np.ndarray (3, 3)
        Box, lattice vectors as rows [nm].
    kcut : float
        Reciprocal-space cutoff [1/nm].

    Returns
    -------
    np.ndarray (K, 3) int32
        One m of each pair (m, -m): m_x > 0, or m_x = 0 and m_y > 0, or m_x = m_y = 0 and m_z > 0.

    Notes
    -----
    Since H_i . k = 2 pi m_i, |m_i| <= kcut |H_i| / (2 pi); the search cube uses the longest
    lattice vector for all three indices, plus one.
    """
    H = np.asarray(H, float)
    B = 2 * np.pi * np.linalg.inv(H).T  # H[i] . B[j] = 2 pi delta_ij
    # integer range: |m_i| <= kcut * |H_i| / (2 pi), bounded generously for skewed cells
    Mx = int(np.ceil(kcut * np.max(np.linalg.norm(H, axis=1)) / (2 * np.pi))) + 1
    rng = range(-Mx, Mx + 1)
    m = np.array([[a, b, c] for a in rng for b in rng for c in rng], float)
    half = (m[:, 0] > 0) | ((m[:, 0] == 0) & (m[:, 1] > 0)) | ((m[:, 0] == 0) & (m[:, 1] == 0) & (m[:, 2] > 0))
    m = m[half]
    kk = np.linalg.norm(m @ B, axis=1)
    return m[(kk > 0) & (kk < kcut)].astype(np.int32)


def _pair_tensors(x: jax.Array, bij: jax.Array, b0: float) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the direct-space kernel g(|x|) = (erf(b_ij r) - erf(b0 r))/r, its gradient and Hessian in x.

    Parameters
    ----------
    x : jax.Array (3,)
        Pair displacement r_i - r_j + n.H [nm], nonzero.
    bij : jax.Array ()
        Gaussian pair exponent [1/nm].
    b0 : float
        Ewald coefficient [1/nm].

    Returns
    -------
    g : jax.Array () [1/nm]
    dg : jax.Array (3,) [1/nm^2]
    d2g : jax.Array (3, 3) [1/nm^3]
    """

    def g(v: jax.Array) -> jax.Array:
        return (erf(bij * jnp.linalg.norm(v)) - erf(b0 * jnp.linalg.norm(v))) / jnp.linalg.norm(v)

    return g(x), jax.grad(g)(x), jax.hessian(g)(x)


class PeriodicPGM:
    """pGM electrostatic energy, forces and induced dipoles of one System in a periodic box (Ewald).

    The reference box and positions fix the neighbour list and the set of k-vectors; energies take
    any positions, parameters and box (default: the reference box).  Pairs of the list farther
    apart than the cutoff are masked, so a list built with a skin (rc + skin) stays valid for small
    displacements.  Not a pytree; methods are pure functions of their arguments and can be jitted
    or differentiated by the caller.

    Attributes
    ----------
    sys : System
        The system.
    H : np.ndarray (3, 3)
        Reference box [nm], lattice vectors as rows.
    b0, rc : float
        Ewald coefficient [1/nm] and direct-space cutoff [nm].
    cg_tol : float
        Relative residual of the induced-dipole CG.
    pd, ind : bool
        Permanent dipoles and induction switched on (from `elec`).
    pi, pj, img : np.ndarray
        Neighbour list (P,), (P,), (P, 3) (neighbor_list).
    m : np.ndarray (K, 3) int32
        Reciprocal-space indices (kvector_indices).
    """

    def __init__(
        self,
        system: System,
        box: np.ndarray,
        positions_ref: np.ndarray,
        ewald_beta: float = 3.8,
        cutoff: float = 1.0,
        skin: float = 0.0,
        k_tol: float = 1e-12,
        dipole_tol: float = 1e-12,
        nlist: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        elec: str = "qpi",
    ) -> None:
        """Build the Ewald model.

        Parameters
        ----------
        system : System
            The molecules (whole, not wrapped atom by atom).
        box : np.ndarray (3, 3)
            Reference box [nm], lattice vectors as rows (fixes the k-vector set).
        positions_ref : np.ndarray (N, 3)
            Reference positions [nm] of the neighbour list.
        ewald_beta : float
            Ewald coefficient [1/nm].
        cutoff : float
            Real-space cutoff [nm] (pairs of the list beyond it are masked).
        skin : float
            Neighbour-list skin [nm].
        k_tol : float
            Reciprocal-space truncation: k-vectors with exp(-k^2 / (4 beta^2)) below k_tol are dropped
            (kcut = 2 beta sqrt(-ln k_tol)).
        dipole_tol : float
            Relative residual of the induced-dipole CG (jax.scipy.sparse.linalg.cg).
        nlist : tuple of np.ndarray, optional
            A neighbour list (i, j, image) to share (PeriodicModel); None builds one with cutoff + skin.
        elec : {"q", "qp", "qi", "qpi"}
            Electrostatics level (options.py).

        Raises
        ------
        ValueError
            If `elec` is unknown.
        """
        from .options import elec_flags

        self.pd, self.ind = elec_flags(elec)
        self.sys, self.H = system, np.asarray(box, float)
        b0, rc = ewald_beta, cutoff
        self.b0, self.rc, self.cg_tol = b0, rc, dipole_tol
        self.pi, self.pj, self.img = nlist if nlist is not None else neighbor_list(positions_ref, self.H, rc + skin)
        kcut = 2 * b0 * np.sqrt(-np.log(k_tol))
        self.m = kvector_indices(self.H, kcut)
        self._E = variational(self._G, self._solve)

    # -------------------------------------------------------------- energy U(q, d)
    def _box(self, H: ArrayLike | None) -> jax.Array:
        """Return H as a JAX array, or the reference box if H is None."""
        return jnp.asarray(self.H if H is None else H)

    def _direct_tensors(self, pos: jax.Array, R: jax.Array, H: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return the masked direct-space kernel, gradient and Hessian of every listed pair.

        Parameters
        ----------
        pos : jax.Array (N, 3)
            Positions [nm].
        R : jax.Array (N,)
            pGM radii [nm].
        H : jax.Array (3, 3)
            Box [nm], rows.

        Returns
        -------
        tuple of jax.Array
            (P,), (P, 3), (P, 3, 3): _pair_tensors per pair, zero for pairs beyond the cutoff.

        Notes
        -----
        The cutoff mask is wrapped in stop_gradient: it is a step function, so it contributes no
        derivative (the energy has a small jump when a pair crosses the cutoff).
        """
        x = pos[self.pi] - pos[self.pj] + jnp.asarray(self.img, float) @ H
        bij = gauss_bij(R[self.pi], R[self.pj])
        f, gx, Hx = jax.vmap(lambda v, b: _pair_tensors(v, b, self.b0))(x, bij)
        w = jax.lax.stop_gradient(jnp.where(jnp.sum(x * x, -1) < self.rc**2, 1.0, 0.0))
        return f * w, gx * w[:, None], Hx * w[:, None, None]

    def _U_dir(self, tens: tuple[jax.Array, jax.Array, jax.Array], q: jax.Array, d: jax.Array) -> jax.Array:
        """Return the direct-space energy of charges q (N,) [e] and dipoles d (N, 3) [e nm] [e^2/nm].

        `tens` comes from _direct_tensors; with x = r_i - r_j + n.H the pair energy is
        q_i q_j g - q_i d_j . grad g + q_j d_i . grad g - d_i . (grad grad g) . d_j.
        """
        f, gx, Hx = tens
        i, j = self.pi, self.pj
        e = (
            q[i] * q[j] * f
            - q[i] * jnp.sum(d[j] * gx, -1)
            + q[j] * jnp.sum(d[i] * gx, -1)
            - jnp.einsum("pa,pab,pb->p", d[i], Hx, d[j])
        )
        return jnp.sum(e)

    def _U_rec(self, pos: jax.Array, q: jax.Array, d: jax.Array, H: jax.Array) -> jax.Array:
        """Return the reciprocal-space energy [e^2/nm] of charges q [e] and dipoles d [e nm].

        U_rec = (4 pi / V) sum_{half space} exp(-k^2/(4 b0^2)) / k^2 |S(k)|^2 with
        S(k) = sum_i (q_i + i k.d_i) exp(i k.r_i) (module docstring); pos [nm], H [nm] (rows).
        """
        B = 2 * jnp.pi * jnp.linalg.inv(H).T
        k = jnp.asarray(self.m, float) @ B  # (K, 3)
        V = jnp.abs(jnp.linalg.det(H))
        k2 = jnp.sum(k**2, 1)
        kfac = (4 * jnp.pi / V) * jnp.exp(-k2 / (4 * self.b0**2)) / k2  # half-space x 2
        ph = pos @ k.T  # (n, K)
        kd = d @ k.T
        c, s = jnp.cos(ph), jnp.sin(ph)
        A = jnp.sum(q[:, None] * c - kd * s, 0)
        Bs = jnp.sum(q[:, None] * s + kd * c, 0)
        return jnp.sum(kfac * (A * A + Bs * Bs))

    def _U_self(self, q: jax.Array, d: jax.Array) -> jax.Array:
        """Return the Ewald self energy -(b0/sqrt(pi)) sum q^2 - (2 b0^3/(3 sqrt(pi))) sum |d|^2 [e^2/nm]."""
        b0 = self.b0
        return -(b0 / SQRT_PI) * jnp.sum(q**2) - (2 * b0**3 / (3 * SQRT_PI)) * jnp.sum(d * d)

    def _U_bg(self, q: jax.Array, H: jax.Array) -> jax.Array:
        """Return the neutralising-background energy -pi Q^2 / (2 V b0^2) of the net charge Q [e^2/nm]."""
        return -jnp.pi * jnp.sum(q) ** 2 / (2 * jnp.abs(jnp.linalg.det(H)) * self.b0**2)

    def U(
        self,
        pos: jax.Array,
        q: jax.Array,
        d: jax.Array,
        R: jax.Array,
        H: jax.Array,
        tens: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    ) -> jax.Array:
        """Return the periodic electrostatic energy of Gaussian charges and dipoles.

        U(q, d) = KE (U_dir + U_rec + U_self + U_bg): the energy of charges q and total dipoles d
        (permanent plus induced), without the polarization self-energy.

        Parameters
        ----------
        pos : jax.Array (N, 3)
            Positions [nm].
        q : jax.Array (N,)
            Charges [e].
        d : jax.Array (N, 3)
            Dipoles [e nm].
        R : jax.Array (N,)
            pGM radii [nm].
        H : jax.Array (3, 3)
            Box [nm], lattice vectors as rows.
        tens : tuple of jax.Array, optional
            Precomputed _direct_tensors(pos, R, H); None computes them.

        Returns
        -------
        jax.Array ()
            Energy [kJ/mol].
        """
        tens = self._direct_tensors(pos, R, H) if tens is None else tens
        return KE * (self._U_dir(tens, q, d) + self._U_rec(pos, q, d, H) + self._U_self(q, d) + self._U_bg(q, H))

    # ----------------------------------------------------------------- induction --
    def _p(self, pos: jax.Array, P: Mapping[str, jax.Array]) -> jax.Array:
        """Return the permanent dipoles (N, 3) [e nm] from the per-atom parameters P, or zeros without them."""
        return perm_dipoles(pos, self.sys, P["cov"]) if self.pd else jnp.zeros((self.sys.n, 3))

    def _G(self, mu: jax.Array, theta: tuple[jax.Array, dict[str, jax.Array], jax.Array]) -> jax.Array:
        """Return the induction functional G(mu) = U(q, p + mu) + KE sum |mu|^2 / (2 alpha) [kJ/mol].

        `mu` (N, 3) are induced dipoles [e nm]; `theta` = (positions [nm], per-atom parameters from
        System.expand, box [nm]).  G is quadratic in mu; its minimum is the total electrostatic energy.
        """
        pos, P, H = theta
        p = self._p(pos, P)
        return self.U(pos, P["q"], p + mu, P["radius"], H) + KE * jnp.sum(mu * mu / (2 * P["alpha"][:, None]))

    def _solve(self, theta: tuple[jax.Array, dict[str, jax.Array], jax.Array]) -> jax.Array:
        """Return the induced dipoles mu* (N, 3) [e nm] that minimise _G at `theta`, by conjugate gradients.

        Notes
        -----
        G is quadratic, so grad G(mu) = g0 + A mu with a constant Hessian A; jax.linearize at mu = 0
        gives g0 and the Hessian-vector product, and CG solves A mu = -g0 to the relative residual
        `cg_tol` (at most 2000 iterations; convergence is not checked).  jax.scipy.sparse.linalg.cg is
        differentiable through lax.custom_linear_solve (implicit differentiation).
        """
        z = jnp.zeros((self.sys.n, 3))

        def gradG(mu: jax.Array) -> jax.Array:
            return jax.grad(self._G)(mu, theta)

        g0, hvp = jax.linearize(gradG, z)  # G is quadratic: the Hessian is constant
        mu, _ = jax.scipy.sparse.linalg.cg(hvp, -g0, tol=self.cg_tol, maxiter=2000)
        return mu

    def _theta(
        self, pos: ArrayLike, params: Mapping[str, ArrayLike] | None, H: ArrayLike | None
    ) -> tuple[jax.Array, dict[str, jax.Array], jax.Array]:
        """Return (positions, per-atom parameters, box) as the argument of _G, _solve and the variational energy."""
        return (jnp.asarray(pos), self.sys.expand(params), self._box(H))

    def induced_dipoles(
        self, pos: ArrayLike, params: Mapping[str, ArrayLike] | None = None, H: ArrayLike | None = None
    ) -> jax.Array:
        """Return the induced dipoles mu* (N, 3) [e nm].

        `pos` (N, 3) [nm], `params` the parameter pytree (None: initial values), `H` (3, 3) the box
        [nm] (None: reference box).  Differentiable (implicit differentiation of the CG solve).  The
        dipoles are computed even for an electrostatics level without induction.
        """
        return self._solve(self._theta(pos, params, H))

    def energy(
        self, pos: ArrayLike, params: Mapping[str, ArrayLike] | None = None, H: ArrayLike | None = None
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        """Return the electrostatic energy components and the permanent dipoles of one configuration.

        Parameters
        ----------
        pos : ArrayLike (N, 3)
            Positions [nm] (molecules whole).
        params : Mapping of str to ArrayLike, optional
            Parameter pytree (system.py); None: the table's initial values.
        H : ArrayLike (3, 3), optional
            Box [nm], lattice vectors as rows; None: the reference box.

        Returns
        -------
        energies : dict of str to jax.Array ()
            "perm": U of the permanent multipoles alone; "total": G(mu*) (EELEC in Amber), equal to
            "perm" without induction; "ind" = total - perm [kJ/mol].
        aux : dict of str to jax.Array
            "p": permanent dipoles (N, 3) [e nm].

        Notes
        -----
        The total goes through solver.variational, so forces, strain derivatives and parameter
        gradients cost one CG solve and do not differentiate the solve.
        """
        theta = self._theta(pos, params, H)
        pos, P, H = theta
        p = self._p(pos, P)
        e_perm = self.U(pos, P["q"], p, P["radius"], H)
        e_tot = self._E(theta) if self.ind else e_perm
        return {"perm": e_perm, "ind": e_tot - e_perm, "total": e_tot}, {"p": p}

    def forces(
        self, pos: ArrayLike, params: Mapping[str, ArrayLike] | None = None, H: ArrayLike | None = None
    ) -> jax.Array:
        """Return the forces -dE_total/dpos (N, 3) [kJ/mol/nm]; arguments as in `energy`."""
        return -jax.grad(lambda x: self.energy(x, params, H)[0]["total"])(jnp.asarray(pos))
