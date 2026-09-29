"""pGM electrostatics.  A channel maps (coordinates, System, parameters) -> energy components.

Everything is a JAX function of the coordinates *and* of the parameter pytree `params`
(see system.py; `None` means the table's initial values), so energies, forces, induced dipoles
and polarizabilities can be differentiated with respect to both, to any order.

  ElecChannel          pGM permanent Gaussian multipoles (charges + covalent dipoles) and linear
                       induced Gaussian dipoles, all pairs, no masking.  Matches sander/pmemd-pgm
                       (tests/test_elec.py, scripts/validate_amber.py).
  elec_decomposition   SAPT-like elst / ind split of the intermolecular pGM energy.
  molecular_polarizability   pGM polarizability tensor of a whole system.

Other channels (pair terms, dispersion, breathing widths, charge transfer) follow the same
interface: an object with `name` and `energy(pos, sys, params) -> (dict, aux)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .kernels import DENSITIES
from .solver import solve_linear_induction
from .system import System
from .units import KE

# ================================================================== electrostatics ==


def perm_dipoles(pos, sys: System, cov_c):
    """Covalent dipoles -> atomic permanent dipoles (n, 3) for one geometry (n, 3).
    cov_c: per-covalent-dipole strengths (e nm), e.g. sys.expand(params)["cov"]."""
    if len(sys.cov_i) == 0:
        return jnp.zeros((sys.n, 3))
    v = pos[sys.cov_j] - pos[sys.cov_i]
    u = v / jnp.linalg.norm(v, axis=-1, keepdims=True)
    return jnp.zeros((sys.n, 3)).at[sys.cov_i].add(cov_c[:, None] * u)


def _pair_perm(ri, rj, qi, pi, qj, pj, b, phi):
    def f(a, c):
        return phi(jnp.linalg.norm(a - c), b)

    gi = jax.grad(f, 0)(ri, rj)
    gj = jax.grad(f, 1)(ri, rj)
    H = jax.jacfwd(jax.grad(f, 0), 1)(ri, rj)
    return qi * qj * f(ri, rj) + qi * pj @ gj + qj * pi @ gi + pi @ H @ pj


def _field_at_i(ri, rj, qj, pj, b, phi):
    def f(a, c):
        return phi(jnp.linalg.norm(a - c), b)

    def V(x):
        return qj * f(x, rj) + pj @ jax.grad(f, 1)(x, rj)

    return -jax.grad(V)(ri)


def quadrupole_field(x, a, Tj):
    """Field (P, 3) at i of the quadrupoles Tj (P, 3, 3) of j, x = r_i - r_j (multipole.py)."""
    from .md.kernels import erf_kernels

    r = jnp.linalg.norm(x, axis=-1)
    _, _, B2, B3 = erf_kernels(a, r, 4)
    Tjx = jnp.einsum("pab,pb->pa", Tj, x)
    xTx = jnp.sum(x * Tjx, -1)
    return -(2.0 / 3.0) * Tjx * B2[:, None] + (xTx * B3 / 3.0)[:, None] * x


def _dipole_tensor(ri, rj, b, phi):
    def f(a, c):
        return phi(jnp.linalg.norm(a - c), b)

    return jax.jacfwd(jax.grad(f, 0), 1)(ri, rj)


def _dipole_matrix(pos, sys: System, phi, b_pair):
    """(n, n, 3, 3) dipole-dipole tensors, zero diagonal blocks."""
    ii, jj = sys.pair_i, sys.pair_j
    T_pair = jax.vmap(lambda a, c, bb: _dipole_tensor(a, c, bb, phi))(pos[ii], pos[jj], b_pair)  # (P,3,3), i<j
    return jnp.zeros((sys.n, sys.n, 3, 3)).at[ii, jj].set(T_pair).at[jj, ii].set(jnp.swapaxes(T_pair, 1, 2))


def molecular_polarizability(pos, sys: System, params=None, density: str = "gaussian"):
    """pGM polarizability tensor of the whole system (3, 3), nm^3: d(sum mu)/dF_ext = sum_ij [(1/a + T)^-1]_ij."""
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
    """Gaussian electrostatics, every pair interacting.  Options (options.py): `perm_dipoles`
    (covalent dipoles), `polarizable` (induced dipoles), `quadrupoles` (covalent quadrupole basis,
    multipole.py).  Defaults: pGM (charges, permanent and induced dipoles).  Use
    ElecChannel.level("q" | "qp" | "qi" | "qpi", quadrupoles=...) for the named levels.

    efield: a uniform external electric field (three numbers, V/nm; pgm_jax/md/efield.py).  It adds
    "field" = -E . M (kJ/mol) with M = sum q r + sum p + sum mu, and E to the right-hand side of the
    induction equations, so that the induced dipoles respond with the molecular polarizability:
    sum mu(E) - sum mu(0) = alpha_mol E and the total energy is E(0) - E . M(0) - E . alpha_mol E / 2
    exactly.  "ind" is then the rest of the induction energy, (mu . E - mu . F_perm) / 2 (kJ/mol), so
    that perm + ind is the model's energy at the field-polarized dipoles (as "elec" of the MD engine)."""

    density: str = "gaussian"
    polarizable: bool = True
    perm_dipoles: bool = True
    quadrupoles: bool = False
    name: str = "elec"
    efield: tuple | None = None

    @classmethod
    def level(cls, elec: str = "qpi", quadrupoles: bool = False, **kw):
        from .options import elec_flags

        pd, ind = elec_flags(elec)
        return cls(polarizable=ind, perm_dipoles=pd, quadrupoles=quadrupoles, **kw)

    def energy(self, pos, sys: System, params=None):
        """pos (n, 3), params (pytree or None) -> dict(perm, ind) in kJ/mol and aux (mu, p, Theta)."""
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


def elec_decomposition(pos, sys: System, params=None, density: str = "gaussian", inter_point: bool = False):
    """SAPT-like split of the pGM intermolecular electrostatic energy (kJ/mol), in one pass.

    In pGM every atom pair interacts, so each isolated monomer already carries induced dipoles
    mu0 from its own permanent multipoles; its 'gas-phase charge distribution' is q + p + mu0
    (this is what py_resp fits to the QM ESP).  Hence:
      elst = interaction of the isolated, self-polarized monomers (mu frozen at mu0)
             -> compare with SAPT elst (monomer densities, frozen);
      ind  = E(all mu relaxed) - E(mu frozen at mu0) <= 0 (variational)
             -> compare with SAPT ind.
    The 'perm'/'ind' split of ElecChannel.energy is NOT this: its interaction 'ind' contains the
    electrostatics of mu0 and can be large and positive.

    inter_point=True: intermolecular pairs use point multipoles (Gaussian exponent x 1e4), a
    diagnostic for how much the Gaussian overlap (charge penetration) contributes."""
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
    same_blk = jnp.asarray(sys.mol[:, None] == sys.mol[None, :], float)[:, :, None, None]
    mu = solve_linear_induction(T, al, F)
    mu0 = solve_linear_induction(T * same_blk, al, F0)
    Tm = T.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)

    def G(m):
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
