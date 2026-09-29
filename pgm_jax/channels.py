"""Compute pGM electrostatics of a gas-phase system: a channel maps (coordinates, System, params) to energies.

Everything is a JAX function of the coordinates *and* of the parameter pytree `params`
(see system.py; `None` means the table's initial values), so energies, forces, induced dipoles
and polarizabilities can be differentiated with respect to both, to any order.

Contents:

  ElecChannel                pGM permanent Gaussian multipoles (charges + covalent dipoles,
                             optionally quadrupoles) and linear induced Gaussian dipoles, all
                             pairs, no masking.  Matches sander/pmemd-pgm (tests/test_elec.py,
                             scripts/validate_amber.py).
  elec_decomposition         SAPT-like elst / ind split of the intermolecular pGM energy.
  molecular_polarizability   pGM polarizability tensor of a whole system.
  perm_dipoles               atomic permanent dipoles from the covalent dipoles.
  quadrupole_field           field of Gaussian quadrupoles (right-hand side of the induction).
  _pair_perm, _field_at_i, _dipole_tensor, _dipole_matrix
                             pair energies, fields and dipole tensors by automatic
                             differentiation of the Gaussian Coulomb kernel (also used by
                             bonded/model.py).

Physics.  Atom i carries a Gaussian charge q_i, a permanent dipole p_i = sum_j c_ij u_ij (u_ij
the unit vector from i to its covalent partner j) and an induced dipole mu_i, all with the
Gaussian width of the atom's pGM radius.  With phi_ij(r) = erf(b_ij r)/r (densities.py) the
pair interaction of charges and dipoles is obtained by differentiating phi_ij with respect to
the two positions; T_ij = d^2 phi_ij / dr_i dr_j is the dipole-dipole tensor, so that the field
at i of a dipole mu_j at j is -T_ij mu_j.  The induced dipoles minimise

    G(mu) = sum_i |mu_i|^2 / (2 alpha_i) - sum_i mu_i . F_i + 1/2 sum_{i != j} mu_i T_ij mu_j,

with F the field of the permanent multipoles of all other atoms (no exclusions), so
mu = (T + 1/alpha)^-1 F and the induction energy is G(mu) = -mu . F / 2.  All fields and
kernels here are without the Coulomb constant (e/nm^2, e^2/nm); energies are multiplied by
KE (units.py) at the end.

Implementation: dense all-pairs kernels (O(N^2) pairs, O(N^3) induction solve) for molecules
and clusters; the periodic and MD code paths are ewald.py and md/.

Other channels (pair terms, dispersion, breathing widths, charge transfer) follow the same
interface: an object with `name` and `energy(pos, sys, params) -> (dict, aux)`.

Units: nm, e, e nm, nm^3, kJ/mol; external fields V/nm.

References
----------
.. [1] H. Wei, R. Qi, J. Wang, P. Cieplak, Y. Duan, R. Luo, J. Chem. Phys. 153, 114116 (2020).

See also docs/model_options.md, docs/efield.md.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .densities import DENSITIES
from .solver import solve_linear_induction
from .system import System
from .units import KE

# ================================================================== electrostatics ==


def perm_dipoles(pos: jax.Array, sys: System, cov_c: jax.Array) -> jax.Array:
    """Return the atomic permanent dipoles built from the covalent dipoles of one geometry.

    p_i = sum over the covalent dipoles (i, j) of c_ij (r_j - r_i) / |r_j - r_i|.

    Parameters
    ----------
    pos : jax.Array (N, 3)
        Atom positions [nm] (molecules whole; no minimum image).
    sys : System
        System (cov_i, cov_j index arrays).
    cov_c : jax.Array (C,)
        Covalent-dipole strengths [e nm], e.g. sys.expand(params)["cov"].

    Returns
    -------
    jax.Array (N, 3)
        Permanent dipoles [e nm]; zeros if the system has no covalent dipoles.
    """
    if len(sys.cov_i) == 0:
        return jnp.zeros((sys.n, 3))
    v = pos[sys.cov_j] - pos[sys.cov_i]
    u = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    return jnp.zeros((sys.n, 3)).at[sys.cov_i].add(cov_c[:, None] * u)


def _pair_perm(
    ri: jax.Array,
    rj: jax.Array,
    qi: jax.Array,
    pi: jax.Array,
    qj: jax.Array,
    pj: jax.Array,
    b: jax.Array,
    phi: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """Return the energy of one pair of Gaussian charges and dipoles [e^2/nm], by autodiff of the kernel.

    Parameters
    ----------
    ri, rj : jax.Array (3,)
        Positions of the two atoms [nm] (distinct).
    qi, qj : jax.Array ()
        Charges [e].
    pi, pj : jax.Array (3,)
        Dipoles [e nm].
    b : jax.Array ()
        Pair exponent b_ij [1/nm].
    phi : callable
        Kernel phi(r, b) [1/nm] (densities.DENSITIES[...]["coulomb"]).

    Returns
    -------
    jax.Array ()
        q_i q_j phi + q_i p_j . d_j phi + q_j p_i . d_i phi + p_i . (d_i d_j phi) . p_j, with d_i the
        gradient with respect to r_i.
    """

    def f(a: jax.Array, c: jax.Array) -> jax.Array:
        return phi(jnp.linalg.norm(a - c), b)

    gi = jax.grad(f, 0)(ri, rj)
    gj = jax.grad(f, 1)(ri, rj)
    H = jax.jacfwd(jax.grad(f, 0), 1)(ri, rj)
    return qi * qj * f(ri, rj) + qi * pj @ gj + qj * pi @ gi + pi @ H @ pj


def _field_at_i(
    ri: jax.Array,
    rj: jax.Array,
    qj: jax.Array,
    pj: jax.Array,
    b: jax.Array,
    phi: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """Return the field (3,) at r_i of the Gaussian charge qj [e] and dipole pj [e nm] at r_j [e/nm^2].

    The field is -grad V with V(x) = q_j phi(|x - r_j|, b) + p_j . d_j phi(|x - r_j|, b); b is the
    pair exponent [1/nm] and phi the kernel as in _pair_perm.
    """

    def f(a: jax.Array, c: jax.Array) -> jax.Array:
        return phi(jnp.linalg.norm(a - c), b)

    def V(x: jax.Array) -> jax.Array:
        return qj * f(x, rj) + pj @ jax.grad(f, 1)(x, rj)

    return -jax.grad(V)(ri)


def quadrupole_field(x: jax.Array, a: jax.Array, Tj: jax.Array) -> jax.Array:
    """Return the field at atom i of the Gaussian quadrupoles of atom j, for P pairs.

    The quadrupole part of multipole.multipole_field: -(2/3) Th_j x B2 + (1/3)(x Th_j x) x B3.

    Parameters
    ----------
    x : jax.Array (P, 3)
        Separation r_i - r_j [nm], nonzero.
    a : jax.Array (P,)
        Pair exponent b_ij [1/nm].
    Tj : jax.Array (P, 3, 3)
        Traceless quadrupoles of j [e nm^2].

    Returns
    -------
    jax.Array (P, 3)
        Field [e/nm^2] (without the Coulomb constant).
    """
    from .md.kernels import erf_kernels

    r = jnp.linalg.norm(x, axis=-1)
    _, _, B2, B3 = erf_kernels(a, r, 4)
    Tjx = jnp.einsum("pab,pb->pa", Tj, x)
    xTx = jnp.sum(x * Tjx, -1)
    return -(2.0 / 3.0) * Tjx * B2[:, None] + (xTx * B3 / 3.0)[:, None] * x


def _dipole_tensor(
    ri: jax.Array, rj: jax.Array, b: jax.Array, phi: Callable[[jax.Array, jax.Array], jax.Array]
) -> jax.Array:
    """Return T_ij = d^2 phi(|r_i - r_j|, b) / dr_i dr_j (3, 3) [1/nm^3] by forward-over-reverse autodiff."""

    def f(a: jax.Array, c: jax.Array) -> jax.Array:
        return phi(jnp.linalg.norm(a - c), b)

    return jax.jacfwd(jax.grad(f, 0), 1)(ri, rj)


def _dipole_matrix(
    pos: jax.Array, sys: System, phi: Callable[[jax.Array, jax.Array], jax.Array], b_pair: jax.Array
) -> jax.Array:
    """Return the dipole-dipole tensors T_ij of all atom pairs (N, N, 3, 3) [1/nm^3], zero diagonal blocks.

    `b_pair` (P,) holds the pair exponents [1/nm] of the pairs sys.pair_i < sys.pair_j; the
    lower blocks are the transposes, T_ji = T_ij^T.
    """
    ii, jj = sys.pair_i, sys.pair_j
    T_pair = jax.vmap(lambda a, c, bb: _dipole_tensor(a, c, bb, phi))(pos[ii], pos[jj], b_pair)  # (P,3,3), i<j
    return jnp.zeros((sys.n, sys.n, 3, 3)).at[ii, jj].set(T_pair).at[jj, ii].set(jnp.swapaxes(T_pair, 1, 2))


def molecular_polarizability(
    pos: jax.Array, sys: System, params: Mapping[str, ArrayLike] | None = None, density: str = "gaussian"
) -> jax.Array:
    """Return the pGM polarizability tensor of the whole system.

    alpha_mol = d(sum_i mu_i)/dF_ext = sum_ij [(T + diag(1/alpha))^-1]_ij (3 x 3 blocks summed), the
    response of the total induced dipole to a uniform external field.

    Parameters
    ----------
    pos : jax.Array (N, 3)
        Atom positions [nm].
    sys : System
        System.
    params : Mapping of str to ArrayLike, optional
        Parameter pytree (system.py); None: the table's initial values.
    density : str
        Density name, a key of densities.DENSITIES.

    Returns
    -------
    jax.Array (3, 3)
        Polarizability tensor [nm^3].

    Notes
    -----
    Inverts the dense 3N x 3N matrix (O(N^3)); differentiable in `pos` and `params`.
    """
    dens = DENSITIES[density]
    P = sys.expand(params)
    R = P["radius"]
    b_pair = dens["pair_exponent"](R[sys.pair_i], R[sys.pair_j])
    T = _dipole_matrix(pos, sys, dens["coulomb"], b_pair)
    n = sys.n
    A = T.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n) + jnp.diag(jnp.repeat(1.0 / P["alpha"], 3))
    B = jnp.linalg.inv(A).reshape(n, 3, n, 3)
    return B.sum(axis=(0, 2))


@dataclass
class ElecChannel:
    """Gaussian electrostatics of a gas-phase system, every pair interacting (module docstring).

    A mutable dataclass (not a pytree) holding the options; `energy` does the work.  Use
    ElecChannel.level("q" | "qp" | "qi" | "qpi", quadrupoles=...) for the named levels (options.py);
    the defaults are pGM (charges, permanent and induced dipoles).

    Parameters
    ----------
    density : str
        Density name, a key of densities.DENSITIES.
    polarizable : bool
        Include induced dipoles.
    perm_dipoles : bool
        Include the covalent permanent dipoles.
    quadrupoles : bool
        Include the covalent quadrupole basis (multipole.py).
    name : str
        Channel name (key of the channel in a Model).
    efield : tuple of 3 float, optional
        Uniform external electric field [V/nm] (pgm_jax/md/efield.py); None: no field.

    Notes
    -----
    With `efield`, the channel adds "field" = -E . M (kJ/mol) with M = sum q r + sum p + sum mu, and
    E to the right-hand side of the induction equations, so that the induced dipoles respond with the
    molecular polarizability: sum mu(E) - sum mu(0) = alpha_mol E and the total energy is
    E(0) - E . M(0) - E . alpha_mol E / 2 exactly.  "ind" is then the rest of the induction energy,
    (mu . E - mu . F_perm) / 2 (kJ/mol), so that perm + ind is the model's energy at the
    field-polarized dipoles (as "elec" of the MD engine).  sum q r depends on the origin for a
    charged system.
    """

    density: str = "gaussian"
    polarizable: bool = True
    perm_dipoles: bool = True
    quadrupoles: bool = False
    name: str = "elec"
    efield: tuple | None = None

    @classmethod
    def level(cls, elec: str = "qpi", quadrupoles: bool = False, **kw: Any) -> ElecChannel:
        """Return the channel of a named electrostatics level.

        Parameters
        ----------
        elec : {"q", "qp", "qi", "qpi"}
            Electrostatics level (options.py).
        quadrupoles : bool
            Include the covalent quadrupole basis.
        **kw
            Further fields of ElecChannel (density, name, efield).

        Returns
        -------
        ElecChannel

        Raises
        ------
        ValueError
            If `elec` is unknown (options.elec_flags).
        """
        from .options import elec_flags

        pd, ind = elec_flags(elec)
        return cls(polarizable=ind, perm_dipoles=pd, quadrupoles=quadrupoles, **kw)

    def energy(
        self, pos: jax.Array, sys: System, params: Mapping[str, ArrayLike] | None = None
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        """Return the electrostatic energy components and the multipoles of one configuration.

        Parameters
        ----------
        pos : jax.Array (N, 3)
            Atom positions [nm] (all pairs, no periodic images).
        sys : System
            System.
        params : Mapping of str to ArrayLike, optional
            Parameter pytree (system.py); None: the table's initial values.

        Returns
        -------
        energies : dict of str to jax.Array ()
            "perm": energy of the permanent multipoles (all pairs) [kJ/mol]; "ind": induction energy
            -mu . F / 2 (or (mu . E - mu . F) / 2 with a field) [kJ/mol], if polarizable; "field":
            -E . M [kJ/mol], if `efield` is set.
        aux : dict of str to jax.Array
            "p" (N, 3) permanent dipoles [e nm]; "mu" (N, 3) induced dipoles [e nm], if polarizable;
            "Theta" (N, 3, 3) quadrupoles [e nm^2], if quadrupoles.

        Notes
        -----
        Differentiable in `pos` and `params` to any order (dense direct induction solve).  The pair
        energies and fields are computed by automatic differentiation of the kernel per pair (vmap
        over pairs); the induced-dipole field sums over ordered pairs i != j, the energy over i < j.
        """
        dens = DENSITIES[self.density]
        phi = dens["coulomb"]
        P = sys.expand(params)
        R, q = P["radius"], P["q"]
        p = perm_dipoles(pos, sys, P["cov"]) if self.perm_dipoles else jnp.zeros((sys.n, 3))
        ii, jj = sys.pair_i, sys.pair_j
        b_pair = dens["pair_exponent"](R[ii], R[jj])
        e_perm = jnp.sum(
            jax.vmap(lambda a, c, qa, pa, qc, pc, bb: _pair_perm(a, c, qa, pa, qc, pc, bb, phi))(
                pos[ii], pos[jj], q[ii], p[ii], q[jj], p[jj], b_pair
            )
        )
        aux = {"p": p}
        if self.quadrupoles:
            from .md.kernels import erf_kernels
            from .multipole import quadrupole_pair_terms, quadrupoles

            Th = quadrupoles(pos, sys, P["quad"])
            x = pos[ii] - pos[jj]
            B = erf_kernels(b_pair, jnp.linalg.norm(x, axis=-1), 5)
            e_perm = e_perm + jnp.sum(quadrupole_pair_terms(x, B, q[ii], p[ii], Th[ii], q[jj], p[jj], Th[jj]))
            aux["Theta"] = Th
        out = {"perm": KE * e_perm}
        Ext = None
        if self.efield is not None:
            from .md.efield import VNM_TO_INTERNAL

            # V/nm -> e/nm^2 (field without the Coulomb constant, as F)
            Ext = jnp.asarray(self.efield, jnp.float64).reshape(3) * VNM_TO_INTERNAL
            out["field"] = -KE * jnp.dot(Ext, jnp.sum(q[:, None] * pos, axis=0) + jnp.sum(p, axis=0))
        if self.polarizable:
            n = sys.n
            # ordered pairs i != j only (no self terms: they would put NaNs into the gradients)
            oi, oj = np.nonzero(~np.eye(n, dtype=bool))
            b_ord = dens["pair_exponent"](R[oi], R[oj])
            F_ord = jax.vmap(lambda a, c, qc, pc, bb: _field_at_i(a, c, qc, pc, bb, phi))(
                pos[oi], pos[oj], q[oj], p[oj], b_ord
            )
            if self.quadrupoles:
                F_ord = F_ord + quadrupole_field(pos[oi] - pos[oj], b_ord, Th[oj])
            F = jnp.zeros((n, 3)).at[oi].add(F_ord)
            T = _dipole_matrix(pos, sys, phi, b_pair)
            if Ext is None:
                mu = solve_linear_induction(T, P["alpha"], F)
                out["ind"] = KE * (-0.5 * jnp.sum(mu * F))
            else:
                mu = solve_linear_induction(T, P["alpha"], F + Ext[None, :])
                out["ind"] = KE * 0.5 * (jnp.sum(mu * Ext[None, :]) - jnp.sum(mu * F))
                out["field"] = out["field"] - KE * jnp.dot(Ext, jnp.sum(mu, axis=0))
            aux["mu"] = mu
        return out, aux


def elec_decomposition(
    pos: jax.Array,
    sys: System,
    params: Mapping[str, ArrayLike] | None = None,
    density: str = "gaussian",
    inter_point: bool = False,
) -> dict[str, jax.Array]:
    """Return a SAPT-like split of the pGM intermolecular electrostatic energy, in one pass.

    In pGM every atom pair interacts, so each isolated monomer already carries induced dipoles
    mu0 from its own permanent multipoles; its "gas-phase charge distribution" is q + p + mu0
    (this is what py_resp fits to the QM ESP).  Hence:

        elst = interaction of the isolated, self-polarized monomers (mu frozen at mu0)
               -> compare with SAPT elst (monomer densities, frozen);
        ind  = E(all mu relaxed) - E(mu frozen at mu0) <= 0 (variational)
               -> compare with SAPT ind.

    The "perm"/"ind" split of ElecChannel.energy is NOT this: its interaction "ind" contains the
    electrostatics of mu0 and can be large and positive.

    Parameters
    ----------
    pos : jax.Array (N, 3)
        Atom positions [nm] of the complex (several molecules).
    sys : System
        System; molecules are the monomers.
    params : Mapping of str to ArrayLike, optional
        Parameter pytree (system.py); None: the table's initial values.
    density : str
        Density name, a key of densities.DENSITIES.
    inter_point : bool
        Intermolecular pairs use point multipoles (Gaussian exponent x 1e4), a diagnostic for how
        much the Gaussian overlap (charge penetration) contributes.

    Returns
    -------
    dict of str to jax.Array ()
        "elst", "ind" and their sum "elec" = E(complex) - sum E(monomers) [kJ/mol].

    Notes
    -----
    With G the induction functional of the complex (module docstring), mu the relaxed dipoles of
    the complex, mu0 the dipoles of each monomer in its own field F0, and E_perm,inter the
    intermolecular permanent-multipole energy:

        elst = E_perm,inter + G(mu0) - G0(mu0),   G0(mu0) = -mu0 . F0 / 2 (sum over monomers)
        ind  = G(mu) - G(mu0),                    G(mu) = -mu . F / 2

    Always uses charges and permanent dipoles (no level switch), no quadrupoles and no external
    field.  Differentiable in `pos` and `params`.
    """
    dens = DENSITIES[density]
    phi = dens["coulomb"]
    P = sys.expand(params)
    R, q, al = P["radius"], P["q"], P["alpha"]
    p = perm_dipoles(pos, sys, P["cov"])
    n = sys.n
    ii, jj = sys.pair_i, sys.pair_j
    b_pair = dens["pair_exponent"](R[ii], R[jj])
    oi, oj = np.nonzero(~np.eye(n, dtype=bool))
    b_ord = dens["pair_exponent"](R[oi], R[oj])
    if inter_point:
        b_pair = b_pair * jnp.where(jnp.asarray(sys.pair_inter), 1e4, 1.0)
        b_ord = b_ord * jnp.where(jnp.asarray(sys.mol[oi] != sys.mol[oj]), 1e4, 1.0)
    e_pair = jax.vmap(lambda a, c, qa, pa, qc, pc, bb: _pair_perm(a, c, qa, pa, qc, pc, bb, phi))(
        pos[ii], pos[jj], q[ii], p[ii], q[jj], p[jj], b_pair
    )
    intra_pair = jnp.asarray(~sys.pair_inter, float)
    F_ord = jax.vmap(lambda a, c, qc, pc, bb: _field_at_i(a, c, qc, pc, bb, phi))(pos[oi], pos[oj], q[oj], p[oj], b_ord)
    same = jnp.asarray(sys.mol[oi] == sys.mol[oj], float)[:, None]
    F = jnp.zeros((n, 3)).at[oi].add(F_ord)
    F0 = jnp.zeros((n, 3)).at[oi].add(F_ord * same)
    T = _dipole_matrix(pos, sys, phi, b_pair)
    # 1 on the diagonal molecule blocks: T restricted to intramolecular pairs gives the monomers' mu0
    same_blk = jnp.asarray(sys.mol[:, None] == sys.mol[None, :], float)[:, :, None, None]
    mu = solve_linear_induction(T, al, F)
    mu0 = solve_linear_induction(T * same_blk, al, F0)
    Tm = T.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)

    def G(m: jax.Array) -> jax.Array:
        return jnp.sum(m * m / (2 * al[:, None])) - jnp.sum(m * F) + 0.5 * m.reshape(-1) @ Tm @ m.reshape(-1)

    e_perm_int = jnp.sum(e_pair * (1 - intra_pair))
    e_mono_ind = -0.5 * jnp.sum(mu0 * F0)
    e_frozen_ind = G(mu0)
    e_relaxed_ind = -0.5 * jnp.sum(mu * F)
    return {
        "elst": KE * (e_perm_int + e_frozen_ind - e_mono_ind),
        "ind": KE * (e_relaxed_ind - e_frozen_ind),
        "elec": KE * (e_perm_int + e_relaxed_ind - e_mono_ind),
    }
