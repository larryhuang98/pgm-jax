"""Build a flexible pGM water for path-integral MD (docs/pimd.md).

The bonded terms are fitted so that the gas-phase monomer (bonded terms + intramolecular pGM +
van der Waals) reproduces the q-TIP4P/F intramolecular surface [1]_.  Contents: QTIP4PF (the
target's parameters), qtip4pf_intra (the target potential), water_geometry, harmonic_frequencies
and flexible_water (the fit), WATER_FAMILIES (the default bonded families).

Units: nm, degrees (angles of geometries), kJ/mol, amu, cm^-1 (frequencies).

References
----------
.. [1] S. Habershon, T. E. Markland, D. E. Manolopoulos, J. Chem. Phys. 131, 024501 (2009).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..units import C_CM_PS, KCAL

if TYPE_CHECKING:
    from ..md.flexible import FlexibleTemplate
    from ..system import Molecule

QTIP4PF = {"D": 116.09 * KCAL, "alpha": 22.87, "r_eq": 0.09419, "k_theta": 87.85 * KCAL, "theta_eq": 107.4}
"""q-TIP4P/F intramolecular potential (Habershon, Markland & Manolopoulos, JCP 131, 024501 (2009)):
quartic Morse O-H bonds D [a^2 dr^2 - a^3 dr^3 + 7/12 a^4 dr^4] (D kJ/mol, alpha 1/nm, r_eq nm) and a
harmonic bend k_theta (theta - theta_eq)^2 / 2 (kJ/mol/rad^2, degrees)."""

WATER_FAMILIES = ("bond_quartic", "angle_harm", "angle_cubic", "bond_bond", "bond_angle")


def qtip4pf_intra(R: jax.Array, p: dict = QTIP4PF) -> jax.Array:
    """Return the q-TIP4P/F intramolecular energy [kJ/mol] of one water.

    Parameters
    ----------
    R : jax.Array (3, 3)
        Positions of O, H, H [nm].
    p : dict
        Parameters (QTIP4PF: D [kJ/mol], alpha [1/nm], r_eq [nm], k_theta [kJ/mol/rad^2],
        theta_eq [degrees]).

    Returns
    -------
    jax.Array ()
        sum over the two O-H bonds of D [x^2 - x^3 + 7/12 x^4], x = alpha (r - r_eq), plus
        k_theta (theta - theta_eq)^2 / 2.  Differentiable.
    """
    u, v = R[1] - R[0], R[2] - R[0]
    r1, r2 = jnp.linalg.norm(u), jnp.linalg.norm(v)
    th = jnp.arccos(jnp.clip(jnp.dot(u, v) / (r1 * r2), -1.0, 1.0))
    a = p["alpha"]

    def morse4(r: jax.Array) -> jax.Array:
        x = a * (r - p["r_eq"])
        return p["D"] * (x * x - x**3 + 7.0 / 12.0 * x**4)

    return morse4(r1) + morse4(r2) + 0.5 * p["k_theta"] * (th - math.radians(p["theta_eq"])) ** 2


def water_geometry(r1: float, r2: float, theta_deg: float) -> np.ndarray:
    """Return O, H, H coordinates (3, 3) [nm] with O at the origin, the first H on +x, in the xy plane.

    `r1`, `r2` are the O-H lengths [nm] and `theta_deg` the H-O-H angle [degrees].  (A different
    orientation from models/toy.water_geometry.)
    """
    t = math.radians(theta_deg)
    return np.array([[0.0, 0.0, 0.0], [r1, 0.0, 0.0], [r2 * math.cos(t), r2 * math.sin(t), 0.0]])


def harmonic_frequencies(energy_fn: Callable[[jax.Array], jax.Array], R: ArrayLike, masses: ArrayLike) -> np.ndarray:
    """Return the harmonic vibrational wavenumbers [cm^-1] of a molecule.

    Parameters
    ----------
    energy_fn : callable
        energy_fn(R (n, 3) [nm]) -> energy [kJ/mol]; twice differentiable.
    R : ArrayLike (n, 3)
        Geometry [nm] (a minimum for meaningful frequencies).
    masses : ArrayLike (n,)
        Masses [amu].

    Returns
    -------
    np.ndarray (3n - 6,)
        The 3n - 6 largest eigenvalues of the mass-weighted Hessian as wavenumbers, ascending
        (negative eigenvalues clipped to 0).

    Notes
    -----
    The eigenvalues are omega^2 in kJ/mol/nm^2/amu = 1/ps^2; omega / (2 pi c) with c in cm/ps gives
    cm^-1.
    """
    R = jnp.asarray(R, jnp.float64)
    n = R.shape[0]
    Hs = np.asarray(jax.hessian(lambda x: energy_fn(x.reshape(n, 3)))(R.reshape(-1)))
    m = np.repeat(np.asarray(masses, float), 3)
    w2 = np.linalg.eigvalsh(Hs / np.sqrt(np.outer(m, m)))
    w = np.sqrt(np.clip(np.sort(w2)[-(3 * n - 6) :], 0.0, None))  # rad/ps
    return w / (2.0 * math.pi * C_CM_PS)  # 1/ps -> cm^-1


def flexible_water(
    molecule: Molecule,
    target: Callable[[jax.Array], jax.Array] = qtip4pf_intra,
    families: Sequence[str] = WATER_FAMILIES,
    n_samples: int = 2000,
    sigma_r: float = 0.008,
    sigma_theta: float = 9.0,
    force_weight: float = 1e-4,
    seed: int = 0,
) -> tuple[FlexibleTemplate, dict[str, Any]]:
    """Fit the bonded terms of a flexible pGM water to a target intramolecular potential.

    The gas-phase monomer potential (bonded terms + the all-pair intramolecular pGM electrostatics
    and induction of `molecule`) is fitted to `target` (default: the q-TIP4P/F intramolecular
    potential).

    Parameters
    ----------
    molecule : Molecule
        The water (atoms O, H, H) whose nonbonded parameters are kept.
    target : callable
        target(R (3, 3) [nm]) -> energy [kJ/mol].
    families : sequence of str
        Bonded families (pgm_jax.bonded; quartic bonds, harmonic and cubic bend, bond-bond and
        bond-angle couplings by default).  Only "r0" may be a nonlinear parameter besides the
        reference geometry.
    n_samples : int
        Number of sampled geometries.
    sigma_r : float
        Standard deviation of the sampled bond lengths [nm].
    sigma_theta : float
        Standard deviation of the sampled angles [degrees].
    force_weight : float
        Weight of the force residuals relative to the energy residuals [nm^2].
    seed : int
        Seed of the sampled geometries.

    Returns
    -------
    template : FlexibleTemplate
        The fitted flexible water (md/flexible.py).
    report : dict
        rms_energy_kJmol, rms_force_kJmol_nm (per component), rms_target_force, the minimum of the
        fitted monomer (minimum_bonds_nm, minimum_angle_deg), harmonic frequencies of the fitted and
        target monomers (freq_fit_cm, freq_target_cm), the fitted params and the Nelder-Mead message.

    Raises
    ------
    ValueError
        If the atoms are not O, H, H or a family has other nonlinear parameters.

    Notes
    -----
    Geometries are drawn around the target minimum (bonds r_eq + sigma_r N(0, 1), angle
    theta_eq + sigma_theta N(0, 1)).  The force constants, on which the energy is linear, are
    found by linear least squares (energies with their mean removed, and forces, columns
    normalised) inside a Nelder-Mead search over the reference values (b0, theta0; r0 of a
    Urey-Bradley term).
    """
    import scipy.optimize

    from ..bonded import terms as TT
    from ..bonded.model import BondedModel, BondedSettings, MolSpec
    from ..md.flexible import FlexibleTemplate

    if list(molecule.elements) != ["O", "H", "H"]:
        raise ValueError("atoms must be O, H, H")
    p = QTIP4PF
    x0 = water_geometry(p["r_eq"], p["r_eq"], p["theta_eq"])
    spec = MolSpec(molecule.name, ["O", "H", "H"], [(0, 1), (0, 2)], [1, 1], 0, x0, molecule)
    model = BondedModel([spec], BondedSettings(families=tuple(families)))
    P0 = jax.tree_util.tree_map(np.asarray, model.init_params())
    lin = [(f, k) for f in model.fams for k in TT.REGISTRY[f].linear]
    sizes = [int(np.size(P0[f][k])) for f, k in lin]
    nonlin = [("ref", "b0"), ("ref", "th0")] + [(f, "r0") for f in model.fams if "r0" in P0[f]]
    for f in model.fams:
        extra = [k for k in P0[f] if k not in TT.REGISTRY[f].linear and k != "r0"]
        if extra:
            raise ValueError(f"family {f}: nonlinear parameters {extra} are not fitted here")

    def build(theta: jax.Array, nl: Sequence[float]) -> dict:
        """Return the bonded parameter pytree with linear parameters `theta` and nonlinear ones `nl`."""
        P = {f: dict(v) for f, v in P0.items()}
        o = 0
        for (f, k), n in zip(lin, sizes):
            P[f][k] = jnp.reshape(theta[o : o + n], np.shape(P0[f][k]))
            o += n
        for (f, k), v in zip(nonlin, nl):
            P[f][k] = jnp.full(np.shape(P0[f][k]), v)
        return P

    rng = np.random.default_rng(seed)
    r = p["r_eq"] + sigma_r * rng.standard_normal((n_samples, 2))
    th = p["theta_eq"] + sigma_theta * rng.standard_normal(n_samples)
    X = jnp.asarray(np.stack([water_geometry(a, b, t) for (a, b), t in zip(r, th)]))

    def enb(R: jax.Array) -> jax.Array:
        return model.nonbonded(0, R, None)[0]

    yE = np.asarray(jax.vmap(target)(X) - jax.vmap(enb)(X))
    yF = np.asarray(-jax.vmap(jax.grad(target))(X) + jax.vmap(jax.grad(enb))(X))
    L = sum(sizes)
    wF = math.sqrt(force_weight)

    @jax.jit
    def design(nl: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the Jacobians of the bonded energies (S, L) and forces (S, 3, 3, L) in the linear parameters.

        Exact (the energy is linear in them), at the nonlinear parameters `nl`; jitted.
        """

        def eb(theta: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Return the bonded energies (S,) [kJ/mol] and forces (S, 3, 3) [kJ/mol/nm] of all samples."""
            P = build(theta, nl)
            E = jax.vmap(lambda R: model.bonded_energy(0, R, P))(X)
            F = -jax.vmap(jax.grad(lambda R: model.bonded_energy(0, R, P)))(X)
            return E, F

        return jax.jacfwd(eb)(jnp.zeros(L))  # energies are linear in theta: exact

    def solve(nl: ArrayLike) -> tuple[np.ndarray, float]:
        """Return the least-squares linear parameters at nonlinear parameters `nl` and the residual sum of squares.

        Energy residuals have their mean removed (the zero of energy is free); force residuals are
        weighted by sqrt(force_weight); columns are normalised before np.linalg.lstsq.
        """
        JE, JF = (np.asarray(a) for a in design(jnp.asarray(nl)))
        A = np.concatenate([(JE - JE.mean(0)) / math.sqrt(len(yE)), wF * JF.reshape(-1, L) / math.sqrt(len(yE))])
        b = np.concatenate([(yE - yE.mean()) / math.sqrt(len(yE)), wF * yF.reshape(-1) / math.sqrt(len(yE))])
        scale = np.maximum(np.linalg.norm(A, axis=0), 1e-30)
        theta = np.linalg.lstsq(A / scale, b, rcond=None)[0] / scale
        return theta, float(np.sum((A @ theta - b) ** 2))

    t0 = [p["r_eq"], math.radians(p["theta_eq"])] + [float(np.mean(P0[f]["r0"])) for f, _ in nonlin[2:]]
    steps = np.array([0.002, 0.05] + [0.005] * (len(t0) - 2))
    opt = scipy.optimize.minimize(
        lambda nl: solve(nl)[1],
        np.array(t0),
        method="Nelder-Mead",
        options={
            "xatol": 1e-9,
            "fatol": 1e-12,
            "maxiter": 4000,
            "initial_simplex": np.vstack([t0, np.array(t0) + np.diag(steps)]),
        },
    )
    theta, loss = solve(opt.x)
    P = jax.tree_util.tree_map(jnp.asarray, build(jnp.asarray(theta), opt.x))
    tpl = FlexibleTemplate.from_fit(model, P)

    def model_E(R: jax.Array) -> jax.Array:
        return model.energy(0, R, P)[0]

    E = jax.vmap(model_E)(X)
    Et = jax.vmap(target)(X)
    dE = (E - Et) - jnp.mean(E - Et)
    dF = -jax.vmap(jax.grad(model_E))(X) + jax.vmap(jax.grad(target))(X)
    m = np.asarray(molecule.masses, float)
    e_fit = jax.jit(model_E)
    g_fit = jax.jit(jax.grad(lambda x: model_E(x.reshape(3, 3))))
    mn = scipy.optimize.minimize(
        lambda x: float(e_fit(jnp.asarray(x).reshape(3, 3))),
        x0.reshape(-1),
        jac=lambda x: np.asarray(g_fit(jnp.asarray(x)), float),
        method="BFGS",
        options={"gtol": 1e-8},
    )
    Rm = mn.x.reshape(3, 3)
    b = np.linalg.norm(Rm[1:] - Rm[0], axis=1)
    ang = math.degrees(math.acos(np.dot(Rm[1] - Rm[0], Rm[2] - Rm[0]) / (b[0] * b[1])))
    report = {
        "rms_energy_kJmol": float(jnp.sqrt(jnp.mean(dE**2))),
        "rms_force_kJmol_nm": float(jnp.sqrt(jnp.mean(jnp.sum(dF**2, -1)) / 3.0)),
        "rms_target_force": float(jnp.sqrt(jnp.mean(jnp.sum(jax.vmap(jax.grad(target))(X) ** 2, -1)) / 3.0)),
        "minimum_bonds_nm": b.tolist(),
        "minimum_angle_deg": ang,
        "freq_fit_cm": harmonic_frequencies(model_E, Rm, m).tolist(),
        "freq_target_cm": harmonic_frequencies(target, x0, m).tolist(),
        "params": {f: {k: np.asarray(v).tolist() for k, v in d.items()} for f, d in P.items()},
        "outer": str(opt.message),
    }
    return tpl, report
