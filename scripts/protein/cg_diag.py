"""CG residual structure of the induced-dipole solve in a solvated protein.

Per iteration of a preconditioned CG solve (Jacobi preconditioner alpha, started from the mu4
predictor of the engine's dipole history): the largest residual max|alpha r| / mean|alpha b| in the
protein, the water and the ions, and the atom that holds it.  Shows whether the predictor start or
the convergence rate is the problem.  The system is minimised (300 steps) and run for 300 steps
(flexible engine, X-H constraints, HMR 3.024 amu, dipole tolerance 1e-5) before the analysed step.
Uses internals of PGMForceField (_atoms, _operator, _field) and md.forcefield._PRED.

Usage:

    python scripts/protein/cg_diag.py runs/protein/ubq.prmtop runs/protein/ubq.inpcrd [--dt-fs 2]
    python scripts/protein/cg_diag.py --help

Inputs: a tleap prmtop and coordinates (placeholder electrostatics).
Outputs: printed per-iteration residuals and the atoms with the largest initial residuals.
Units: --dt-fs fs; residuals relative to mean|alpha b| (dimensionless).
Runtime: GPU or CPU, minutes (compilation dominates).  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.cli.args import add_dt_arg, setup_logging
from pgm_jax.md.flexible import FlexibleSimulation
from pgm_jax.md.forcefield import _PRED, MDSettings
from pgm_jax.protein import amber_template, load_amber

jax.config.update("jax_enable_x64", True)


def build_sim(prmtop: str, inpcrd: str, dt: float) -> tuple[FlexibleSimulation, object]:
    """Return the minimised and briefly run simulation of the tleap system and the Amber system (dt [ps])."""
    asys = load_amber(prmtop, inpcrd)
    prot = [k for k, m in enumerate(asys.molecules) if m.kind == "protein"]
    tpl = {k: amber_template(asys.molecules[k], prmtop) for k in prot}
    sim = FlexibleSimulation(
        asys.system(),
        asys.templates(tpl),
        asys.system_positions(),
        asys.box,
        MDSettings().replace(dipole_tol=1e-5),
        dt=dt,
        constraints="h-bonds",
        hmr=3.024,
    )
    sim.minimize(300)
    sim.advance(300)
    return sim, asys


def diagnose(sim: FlexibleSimulation, asys: object, iterations: int) -> None:
    """Run the CG iterations of the next step's dipole solve and print the residual structure (module docstring)."""
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
    A = jax.jit(ff._operator(g, Sp, Gk, P["alpha"]))  # mu -> T mu (the induced-dipole matrix)
    b = ff._field(g, Sp, Gk, P["q"].astype(cd), p.astype(cd))  # field of charges and permanent dipoles
    a = P["alpha"][:, None]
    ab = a * b.astype(jnp.float64)
    norm = float(jnp.mean(jnp.abs(ab)))
    x0 = sum(ci * S0.induction.hist[j] for j, ci in enumerate(_PRED["mu4"]))  # predictor start
    sysm = sim.sys
    mol = np.asarray(sysm.mol)
    offs = np.asarray(sysm.offsets)
    kind = np.array([asys.molecules[m].kind for m in mol])

    def label(i):
        """Return a label (kind:residue:atom, alpha) of atom i."""
        m = mol[i]
        L = asys.molecules[m]
        j = i - offs[m]
        return (
            f"{L.kind}:{L.residue_names[j]}{int(L.residue_index[j])}:{L.atom_names[j]} alpha={float(P['alpha'][i]):.3g}"
        )

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
        """Print the largest relative residual per group and overall, and the atom that holds it."""
        e = np.abs(np.asarray(r * a.astype(cd), np.float64)).max(1) / norm
        parts = "  ".join(f"{k} {e[v].max():.1e}" for k, v in groups.items() if len(v))
        i = int(np.argmax(e))
        print(f"it {it:2d}: max {e.max():.2e} rms {np.sqrt(np.mean(e**2)):.1e} | {parts} | argmax {label(i)}")

    report(0, r)
    for it in range(1, iterations + 1):  # preconditioned CG (Polak-Ribiere form of beta)
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


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the diagnosis (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prmtop", help="tleap prmtop")
    ap.add_argument("inpcrd", help="tleap coordinates")
    add_dt_arg(ap, 2.0)
    ap.add_argument("--iterations", type=int, default=20, help="CG iterations to trace")
    a = ap.parse_args(argv)
    setup_logging()
    sim, asys = build_sim(a.prmtop, a.inpcrd, a.dt_fs / 1000)
    diagnose(sim, asys, a.iterations)


if __name__ == "__main__":
    main()
