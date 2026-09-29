"""CG residual structure of the induced-dipole solve in a solvated protein: per iteration the
largest residual (max|alpha r| / mean|alpha b|) in the protein, the water and the ions, and the
atom that holds it.  Shows whether the predictor start or the convergence rate is the problem.

    python scripts/protein/cg_diag.py runs/protein/ubq.prmtop runs/protein/ubq.inpcrd [--dt 0.002]
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.md.flexible import FlexibleSimulation
from pgm_jax.md.forcefield import _PRED, MDSettings
from pgm_jax.protein import amber_template, load_amber

jax.config.update("jax_enable_x64", True)
ap = argparse.ArgumentParser()
ap.add_argument("prmtop")
ap.add_argument("inpcrd")
ap.add_argument("--dt", type=float, default=0.002)
ap.add_argument("--iterations", type=int, default=20)
args = ap.parse_args()
pr, rs, dt = args.prmtop, args.inpcrd, args.dt
asys = load_amber(pr, rs)
prot = [k for k, m in enumerate(asys.molecules) if m.kind == "protein"]
tpl = {k: amber_template(asys.molecules[k], pr) for k in prot}
sim = FlexibleSimulation(
    asys.system(),
    asys.templates(tpl),
    asys.system_positions(),
    asys.box,
    MDSettings(dipole_tol=1e-5),
    dt=dt,
    constraints="h-bonds",
    hmr=3.024,
)
sim.minimize(300)
sim._advance(300)
S0 = sim.state
I, ff = sim.integ, sim.ff
S1 = I.run(S0, 1)
print("engine iterations at this step:", int(S1.iters))
pos, H = S1.dyn.position, S1.box
c = sim.flex.list_centers(pos)
idx = sim.nb.candidates(S1.nbr, c, H, pos)[0]
P = ff._atoms(I.params)
g = ff.geometry(pos, H, idx, P)
p = ff.perm_dipoles(pos, H, P["cov"])
Sp, Gk = ff.pme.setup(pos, H), ff.pme.influence(H)
cd = ff.cd
A = jax.jit(ff._operator(g, Sp, Gk, P["alpha"]))
b = ff._field(g, Sp, Gk, P["q"].astype(cd), p.astype(cd))
a = P["alpha"][:, None]
ab = a * b.astype(jnp.float64)
norm = float(jnp.mean(jnp.abs(ab)))
x0 = sum(ci * S0.induction.hist[j] for j, ci in enumerate(_PRED["mu4"]))
sysm = sim.sys
mol = np.asarray(sysm.mol)
offs = np.asarray(sysm.offsets)
kind = np.array([asys.molecules[m].kind for m in mol])


def label(i):
    m = mol[i]
    L = asys.molecules[m]
    j = i - offs[m]
    return f"{L.kind}:{L.residue_names[j]}{int(L.residue_index[j])}:{L.atom_names[j]} alpha={float(P['alpha'][i]):.3g}"


groups = {k: np.nonzero(kind == k)[0] for k in ("protein", "water", "ion")}
print("atoms:", {k: len(v) for k, v in groups.items()}, " mean|alpha b|", norm)
aab = np.abs(np.asarray(ab)).max(1)
for k, v in groups.items():
    if len(v):
        print(f"  |alpha b| {k}: mean {aab[v].mean() / norm:.2f}  max {aab[v].max() / norm:.1f} (x mean over all)")

x = x0.astype(cd)
r = b - A(x)
z = r * a.astype(cd)
pp = z
rz = jnp.sum(r * z)


def report(it, r):
    e = np.abs(np.asarray(r * a.astype(cd), np.float64)).max(1) / norm
    parts = "  ".join(f"{k} {e[v].max():.1e}" for k, v in groups.items() if len(v))
    i = int(np.argmax(e))
    print(f"it {it:2d}: max {e.max():.2e} rms {np.sqrt(np.mean(e**2)):.1e} | {parts} | argmax {label(i)}")


report(0, r)
for it in range(1, args.iterations + 1):
    Ap = A(pp)
    al = rz / jnp.sum(pp * Ap)
    x = x + al * pp
    r_new = r - al * Ap
    z_new = r_new * a.astype(cd)
    beta = jnp.sum(z_new * (r_new - r)) / rz
    pp = z_new + beta * pp
    r, rz = r_new, jnp.sum(r_new * z_new)
    report(it, r)

# the worst atoms at the start
e0 = np.abs(np.asarray((b - A(x0.astype(cd))) * a.astype(cd), np.float64)).max(1) / norm
print("largest initial residuals:")
for i in np.argsort(-e0)[:8]:
    print(f"  {e0[i]:.2e}  {label(i)}")
