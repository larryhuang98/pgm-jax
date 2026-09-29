"""Compute benchmarks in the paper's terms: relaxed torsion scans and single-point scan profiles.

Contents: `kabsch_rmsd`, `relaxed_scan` (force-field minimisation with a torsion restraint,
kept near the reference structures) and `scan_metrics` (maximum energy error of the scan
profile, structure RMSD).

Units: model energies kJ/mol and positions nm; metrics kcal/mol and A (the paper's).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from ...units import KCAL
from ..terms import _dihedral

if TYPE_CHECKING:
    from ..fit import FrameSet
    from ..model import BondedModel


def kabsch_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    """Return the RMSD of A and B (n, 3) after centring and optimal rotation (Kabsch), in the input units."""
    A, B = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))  # -1: reflection, flip the last axis to get a proper rotation
    Rm = U @ np.diag([1, 1, d]) @ Vt
    return float(np.sqrt(np.mean(np.sum((A @ Rm - B) ** 2, 1))))


def relaxed_scan(
    model: BondedModel,
    m: int,
    P: dict,
    X0: ArrayLike,
    torsion: ArrayLike,
    angles_deg: Sequence[float],
    k_restr: float = 1e4,
    maxiter: int = 500,
    box: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimise E + k (phi - phi0)^2 from each reference geometry, keeping every coordinate within +-box nm.

    A local relaxation: class II cross terms are unbounded below far from equilibrium.  phi0 is the
    torsion as measured in the reference geometry.

    Parameters
    ----------
    model : BondedModel
        The model.
    m : int
        Molecule index.
    P : dict
        Parameters.
    X0 : ArrayLike (k, n, 3)
        Reference geometries [nm].
    torsion : ArrayLike (4,)
        Atoms of the scanned torsion.
    angles_deg : sequence of float
        Nominal scan angles [deg] (unused: the restraint uses the measured angle).
    k_restr : float
        Restraint force constant [kJ/mol/rad^2].
    maxiter : int
        L-BFGS-B iterations per frame.
    box : float
        Largest move of any coordinate [nm].

    Returns
    -------
    E : np.ndarray (k,)
        Energies without the restraint [kJ/mol].
    X : np.ndarray (k, n, 3)
        Relaxed geometries [nm].
    hit : np.ndarray (k,) bool
        Whether the box was hit (the model wants to leave the basin).
    """
    t = np.asarray(torsion)

    def efun(X: jax.Array) -> jax.Array:
        """Return the model energy of geometry X [kJ/mol]."""
        return model.energy(m, X, P)[0]

    def obj(x: jax.Array, phi0: float) -> jax.Array:
        """Return the energy plus the torsion restraint for flattened coordinates x."""
        X = x.reshape(-1, 3)
        dphi = _dihedral(X[t[0]], X[t[1]], X[t[2]], X[t[3]]) - phi0
        dphi = jnp.arctan2(jnp.sin(dphi), jnp.cos(dphi))  # wrapped to (-pi, pi]
        return efun(X) + k_restr * dphi**2

    vg = jax.jit(jax.value_and_grad(obj))
    ef = jax.jit(efun)
    Es, Xs, hit = [], [], []
    for X, a in zip(X0, angles_deg):
        x0 = np.asarray(X, float).ravel()
        phi0 = float(_dihedral(*[jnp.asarray(X[i]) for i in t]))  # restrain to the reference angle as measured
        del a

        def f(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Return the objective and its gradient as float64 numpy arrays (for scipy)."""
            return tuple(np.asarray(v, float) for v in vg(jnp.asarray(x), phi0))

        r = minimize(
            f,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(x0 - box, x0 + box)),
            options={"maxiter": maxiter, "gtol": 1e-6},
        )
        Xr = r.x.reshape(-1, 3)
        Es.append(float(ef(jnp.asarray(Xr))))
        Xs.append(Xr)
        hit.append(bool(np.max(np.abs(r.x - x0)) > 0.999 * box))
    return np.array(Es), np.array(Xs), np.array(hit)


def scan_metrics(model: BondedModel, m: int, P: dict, fs: FrameSet, relax: bool = True) -> dict:
    """Return the scan metrics of one torsion scan.

    Parameters
    ----------
    model : BondedModel
        The model.
    m : int
        Molecule index.
    P : dict
        Parameters.
    fs : FrameSet
        The scan frames, with extras "angle" and "torsion".
    relax : bool
        Also run `relaxed_scan`.

    Returns
    -------
    dict
        "sp_max" max |error| of the single-point profile (both shifted to their minimum)
        [kcal/mol], "barrier_ref" [kcal/mol], "profile_sp", "profile_ref" [kcal/mol], "angle";
        with relax: "relax_ok" (finite and no box hit), "relax_hits", "rmsd_A" largest RMSD of
        the relaxed vs the reference structures [A] and, if ok, "relaxed_max" [kcal/mol] and
        "profile_ff".
    """
    E_ref = fs.E - fs.E.min()
    sp = jax.jit(jax.vmap(lambda X: model.energy(m, X, P)[0]))(jnp.asarray(fs.X))
    sp = np.asarray(sp)
    out = {"sp_max": float(np.max(np.abs((sp - sp.min()) - E_ref))) / KCAL, "barrier_ref": float(E_ref.max()) / KCAL}
    out["profile_sp"] = ((sp - sp.min()) / KCAL).tolist()
    out["profile_ref"] = (E_ref / KCAL).tolist()
    out["angle"] = np.asarray(fs.extra["angle"]).tolist()
    if relax:
        E, X, hit = relaxed_scan(model, m, P, fs.X, fs.extra["torsion"][0], fs.extra["angle"])
        rmsd = np.array([kabsch_rmsd(a, b) for a, b in zip(X, fs.X)]) * 10.0
        ok = bool(np.all(np.isfinite(E)) and not hit.any())  # the model's minimum left the basin = failure
        out["relax_ok"] = ok
        out["relax_hits"] = int(hit.sum())
        out["rmsd_A"] = float(rmsd.max())
        if ok:
            out["relaxed_max"] = float(np.max(np.abs((E - E.min()) - E_ref))) / KCAL
            out["profile_ff"] = ((E - E.min()) / KCAL).tolist()
    return out
