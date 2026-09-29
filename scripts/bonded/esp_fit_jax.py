"""Multi-conformer pGM ESP fit in JAX (py_resp's multi-molecule mode fails for pGM-perm).

Charges and covalent-dipole strengths (tied by symmetry, as py_resp) are fitted to the
B3LYP/aug-cc-pVTZ ESP of several conformers at once; the model potential includes the induced
dipoles of each conformer (pGM, all pairs), like py_resp ipol=5.  Start: the single-conformer
py_resp parameters (data/bonded/params/<name>.json).  Restraint: lam * sum (theta - theta_start)^2
(weak) plus --lam0 times a restraint towards zero charges and covalent dipoles, and a penalty
keeping the total charge.

Usage:

    python scripts/bonded/esp_fit_jax.py [names] [--lam0 0.001] [--out params2]
    python scripts/bonded/esp_fit_jax.py --help

Inputs: runs/bonded/pgm2/<name>/esp_*.dat (scripts/bonded/qm_esp.py), data/bonded/params/<name>.json.
Outputs: data/bonded/<out>/<name>.json (param.save_molecule); a printed line per molecule.
Units: atomic units (ESP), bohr in the .dat files, e and e nm in the molecule files.
Runtime: CPU, a minute per molecule.  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import glob
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

from pgm_jax.channels import ElecChannel
from pgm_jax.param import load_molecule, save_molecule
from pgm_jax.paths import repo_path
from pgm_jax.system import System
from pgm_jax.units import BOHR_NM

jax.config.update("jax_enable_x64", True)


def read_esp(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return atoms (N, 3) [bohr], ESP points (M, 3) [bohr] and ESP values (M,) [a.u.] of a py_resp esp.dat."""
    lines = open(path).read().splitlines()
    n, m = int(lines[0][:5]), int(lines[0][5:10])
    X = np.array([[float(v) for v in (ln[18:34], ln[34:50], ln[50:66])] for ln in lines[1 : 1 + n]])  # bohr
    P = np.array([[float(v) for v in (ln[:16], ln[16:32], ln[32:48], ln[48:64])] for ln in lines[1 + n : 1 + n + m]])
    return X, P[:, 1:], P[:, 0]  # atoms bohr, points bohr, esp a.u.


def fit(name: str, lam: float = 1e-3, lam0: float = 0.0, out: str = "params2", verbose: bool = True) -> None:
    """Fit charges and covalent dipoles of one molecule to its conformers' ESP; save data/bonded/<out>/<name>.json.

    Parameters
    ----------
    name : str
        Molecule.
    lam : float
        Weight of the restraint to the start parameters.
    lam0 : float
        Weight of the restraint towards zero (charges [e], covalent dipoles in units of 0.01 e nm).
    out : str
        Output directory below data/bonded.
    verbose : bool
        Print the relative RMS error before and after and the charges.
    """
    wd = repo_path("runs", "bonded", "pgm2", name)
    m0 = load_molecule(repo_path("data", "bonded", "params", f"{name}.json"))
    confs = [read_esp(p) for p in sorted(glob.glob(os.path.join(wd, "esp_*.dat")))]
    sys_ = System([m0])
    th0 = sys_.params0
    Qtot = float(np.sum(m0.q))
    chan = ElecChannel()

    def potential(th, Xb, Pb):
        """Return the model ESP [a.u.] at the points Pb [bohr] of the conformer Xb [bohr] (induced dipoles solved)."""
        P = sys_.expand(th)
        q = P["q"]
        X = jnp.asarray(Xb) * BOHR_NM
        _, aux = chan.energy(X, sys_, th)  # induced dipoles for this conformer
        d = aux["p"] + aux["mu"]  # e nm
        r = jnp.asarray(Pb)[:, None, :] * BOHR_NM - X[None]  # nm
        dist = jnp.linalg.norm(r, axis=-1)
        V = jnp.sum(q[None] / dist, 1) + jnp.sum(jnp.sum(d[None] * r, -1) / dist**3, 1)  # e / nm
        return V * BOHR_NM  # a.u. (e / bohr)

    free = {"q": th0["q"], "cov": th0["cov"]}
    z0, unravel = ravel_pytree(free)
    ssv = sum(float(np.sum(v**2)) for _, _, v in confs)

    def loss(z):
        """Return the relative squared ESP error plus the restraints and the total-charge penalty."""
        th = dict(th0)
        th.update(unravel(z))
        L = 0.0
        for Xb, Pb, v in confs:
            L = L + jnp.sum((potential(th, Xb, Pb) - v) ** 2)
        Pq = sys_.expand(th)["q"]
        u = unravel(z)
        return (
            L / ssv
            + lam * jnp.sum((z - z0) ** 2)
            + lam0 * (jnp.sum(u["q"] ** 2) + jnp.sum((u["cov"] / 0.01) ** 2))
            + 10.0 * (jnp.sum(Pq) - Qtot) ** 2
        )

    vg = jax.jit(jax.value_and_grad(loss))

    def f(z):
        """Return the loss and its gradient as numpy arrays (for scipy)."""
        return tuple(np.asarray(t, float) for t in vg(jnp.asarray(z)))

    rr0 = float(np.sqrt(loss(z0) - 0.0))
    res = minimize(f, np.asarray(z0), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    th = dict(th0)
    th.update(unravel(jnp.asarray(res.x)))
    P = sys_.expand(th)
    q = np.asarray(P["q"] - (jnp.sum(P["q"]) - Qtot) / sys_.n)
    m = load_molecule(repo_path("data", "bonded", "params", f"{name}.json"))
    m.q = q
    m.cov = [(i, j, float(c)) for (i, j, _), c in zip(m.cov, np.asarray(P["cov"]))]
    os.makedirs(repo_path("data", "bonded", out), exist_ok=True)
    save_molecule(m, repo_path("data", "bonded", out, f"{name}.json"))
    Ls = float(sum(jnp.sum((potential(th, Xb, Pb) - v) ** 2) for Xb, Pb, v in confs) / ssv)
    if verbose:
        print(
            f"{name:20s} lam0 {lam0:g}: RRMSE over {len(confs)} conformers {rr0:.3f} -> {np.sqrt(Ls):.3f}   q "
            f"{' '.join(f'{x:+.2f}' for x in q)}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and fit every molecule (failures are printed; see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="molecules (default: every directory of runs/bonded/pgm2)")
    ap.add_argument("--lam0", type=float, default=0.0, help="weight of the restraint towards zero")
    ap.add_argument("-o", "--out", default="params2", help="output directory below data/bonded")
    a = ap.parse_args(argv)
    names = a.names or [os.path.basename(p) for p in sorted(glob.glob(repo_path("runs", "bonded", "pgm2", "*")))]
    for n in names:
        try:
            fit(n, lam0=a.lam0, out=a.out)
        except Exception as exc:
            print(n, "failed", repr(exc)[:200], flush=True)


if __name__ == "__main__":
    main()
