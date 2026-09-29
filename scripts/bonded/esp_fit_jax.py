"""Multi-conformer pGM ESP fit in JAX (py_resp's multi-molecule mode fails for pGM-perm).

Charges and covalent-dipole strengths (tied by symmetry, as py_resp) are fitted to the
B3LYP/aug-cc-pVTZ ESP of several conformers at once; the model potential includes the induced
dipoles of each conformer (pGM, all pairs), like py_resp ipol=5.  Start: the single-conformer
py_resp parameters.  Restraint: lam * sum (q - q_start)^2 on nothing but the free charges (weak).

    python scripts/bonded/esp_fit_jax.py [names]     # runs/bonded/pgm2/<name>/esp_*.dat -> data/bonded/params2/
"""

import glob
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from pgm_jax.channels import ElecChannel  # noqa: E402
from pgm_jax.param import load_molecule, save_molecule  # noqa: E402
from pgm_jax.system import System  # noqa: E402

BOHR_NM = 0.052917721067
KE_AU = 1.0 / 138.935458  # (e^2 / nm) in kJ/mol -> we work in a.u. below


def read_esp(path):
    lines = open(path).read().splitlines()
    n, m = int(lines[0][:5]), int(lines[0][5:10])
    X = np.array([[float(v) for v in (ln[18:34], ln[34:50], ln[50:66])] for ln in lines[1 : 1 + n]])  # bohr
    P = np.array([[float(v) for v in (ln[:16], ln[16:32], ln[32:48], ln[48:64])] for ln in lines[1 + n : 1 + n + m]])
    return X, P[:, 1:], P[:, 0]  # atoms bohr, points bohr, esp a.u.


def fit(name, lam=1e-3, lam0=0.0, out="params2", verbose=True):
    wd = os.path.join(ROOT, "runs/bonded/pgm2", name)
    m0 = load_molecule(os.path.join(ROOT, "data/bonded/params", f"{name}.json"))
    confs = [read_esp(p) for p in sorted(glob.glob(os.path.join(wd, "esp_*.dat")))]
    sys_ = System([m0])
    th0 = sys_.params0
    Qtot = float(np.sum(m0.q))
    chan = ElecChannel()

    def potential(th, Xb, Pb):
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
        return tuple(np.asarray(t, float) for t in vg(jnp.asarray(z)))

    rr0 = float(np.sqrt(loss(z0) - 0.0))
    res = minimize(f, np.asarray(z0), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
    th = dict(th0)
    th.update(unravel(jnp.asarray(res.x)))
    P = sys_.expand(th)
    q = np.asarray(P["q"] - (jnp.sum(P["q"]) - Qtot) / sys_.n)
    m = load_molecule(os.path.join(ROOT, "data/bonded/params", f"{name}.json"))
    m.q = q
    m.cov = [(i, j, float(c)) for (i, j, _), c in zip(m.cov, np.asarray(P["cov"]))]
    os.makedirs(os.path.join(ROOT, "data/bonded", out), exist_ok=True)
    save_molecule(m, os.path.join(ROOT, "data/bonded", out, f"{name}.json"))
    Ls = float(sum(jnp.sum((potential(th, Xb, Pb) - v) ** 2) for Xb, Pb, v in confs) / ssv)
    if verbose:
        print(
            f"{name:20s} lam0 {lam0:g}: RRMSE over {len(confs)} conformers {rr0:.3f} -> {np.sqrt(Ls):.3f}   q "
            f"{' '.join(f'{x:+.2f}' for x in q)}",
            flush=True,
        )


if __name__ == "__main__":
    lam0 = float(os.environ.get("LAM0", "0"))
    out = os.environ.get("OUT", "params2")
    names = sys.argv[1:] or [os.path.basename(p) for p in sorted(glob.glob(os.path.join(ROOT, "runs/bonded/pgm2/*")))]
    for n in names:
        try:
            fit(n, lam0=lam0, out=out)
        except Exception as exc:
            print(n, "failed", repr(exc)[:200], flush=True)
