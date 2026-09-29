"""Benchmarks in the paper's terms: relaxed torsion scans at force-field level (maximum energy
error, structure RMSD) and single-point scan profiles."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from ...units import KCAL
from ..terms import _dihedral


def kabsch_rmsd(A, B):
    A, B = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ np.diag([1, 1, d]) @ Vt
    return float(np.sqrt(np.mean(np.sum((A @ Rm - B) ** 2, 1))))


def relaxed_scan(model, m, P, X0, torsion, angles_deg, k_restr=1e4, maxiter=500, box=0.03):
    """Minimise E + k (phi - phi0)^2 from each reference geometry, every coordinate kept within
    +-box nm of its start (a local relaxation: class II cross terms are unbounded below far from
    equilibrium).  Returns energies (without the restraint, kJ/mol), geometries and whether the
    box was hit (the model wants to leave the basin)."""
    t = np.asarray(torsion)

    def efun(X):
        return model.energy(m, X, P)[0]

    def obj(x, phi0):
        X = x.reshape(-1, 3)
        dphi = _dihedral(X[t[0]], X[t[1]], X[t[2]], X[t[3]]) - phi0
        dphi = jnp.arctan2(jnp.sin(dphi), jnp.cos(dphi))
        return efun(X) + k_restr * dphi**2

    vg = jax.jit(jax.value_and_grad(obj))
    ef = jax.jit(efun)
    Es, Xs, hit = [], [], []
    for X, a in zip(X0, angles_deg):
        x0 = np.asarray(X, float).ravel()
        phi0 = float(_dihedral(*[jnp.asarray(X[i]) for i in t]))  # restrain to the reference angle as measured
        del a

        def f(x):
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


def scan_metrics(model, m, P, fs, relax=True):
    """Max |error| of the scan profile (both shifted to their minimum), kcal/mol; RMSD of the
    relaxed structures vs the reference ones (A)."""
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
