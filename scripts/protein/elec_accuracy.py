"""Electrostatic accuracy of real-space cutoffs (MDSettings.elec_cutoff), van der Waals kept at
--cut, against a tight reference: 0.9 nm, beta 4.0 nm^-1, PME spacing 0.04 nm, order 6, dipole
tol 1e-9, float64.  Every case has the same van der Waals term, so force differences are
electrostatic; they are reported as rms (and max) force difference / rms electrostatic force,
with the electrostatic energy difference and the rms induced-dipole difference.

    python scripts/protein/elec_accuracy.py runs/protein/ubq.prmtop runs/protein/ubq.inpcrd --frame eq.npz
    python scripts/protein/elec_accuracy.py --water                   # 512 pGM3P-25 waters

--frame: an .npz with `pos` and `box` (nm), e.g. an equilibrated frame (the tleap coordinates have
clashes that dominate the forces).  Cases: the default (0.9 nm / 4.0 / 0.08 nm) and, for every
--elec-cut, elec_cutoff_settings with the grid exponents of --exponents; also the reference itself
against a longer real-space cutoff (1.2 nm or the largest the box allows, order 8, 0.03 nm).
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.md.box import max_cutoff
from pgm_jax.md.forcefield import MDSettings, PGMForceField, elec_cutoff_settings
from pgm_jax.paths import resource

jax.config.update("jax_enable_x64", True)
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("prmtop", nargs="?")
ap.add_argument("inpcrd", nargs="?")
ap.add_argument("--water", action="store_true", help="the 512-water pGM3P-25 box of the tests instead of a protein")
ap.add_argument("--frame", help=".npz with pos and box (nm)")
ap.add_argument("--library", default=None, help="pGM residue library (default: placeholder electrostatics)")
ap.add_argument("--cut", type=float, default=0.9, help="van der Waals cutoff (nm)")
ap.add_argument("--elec-cut", type=float, nargs="+", default=[0.8, 0.7, 0.6])
ap.add_argument("--exponents", type=float, nargs="+", default=[1.6, 1.0], help="grid rules of elec_cutoff_settings")
ap.add_argument("--precision", nargs="+", default=["mixed"])
a = ap.parse_args()
top = None
if a.water:
    from pgm_jax.md.box import box_from_cell
    from pgm_jax.md.io import read_coordinates
    from pgm_jax.system import System

    TOP = resource("gvdw_data", "topology/rayl_512_v2.prmtop")
    RST = resource("gvdw_data", "inputs/lj/inpcrd.restrt")
    system = System.from_prmtop(TOP)
    xyz, _, cell = read_coordinates(RST)
    pos, H = xyz * 0.1, box_from_cell(*cell) * 0.1
else:
    from pgm_jax.md.topology import MDTopology
    from pgm_jax.protein import ResidueLibrary, amber_template, load_amber

    lib = ResidueLibrary.load(a.library) if a.library else "placeholder"
    asys = load_amber(a.prmtop, a.inpcrd, electrostatics=lib)
    prot = [k for k, m in enumerate(asys.molecules) if m.kind == "protein"]
    temps = asys.templates({k: amber_template(asys.molecules[k], a.prmtop) for k in prot})
    system = asys.system()
    rules = {id(t): t.md_rule("h-bonds") for t in {id(t): t for t in temps}.values()}
    top = MDTopology.build(system, [rules[id(t)] for t in temps])  # special pairs as FlexibleSimulation
    pos, H = asys.system_positions(), asys.box
if a.frame:
    f = np.load(a.frame)
    pos, H = f["pos"], f["box"]
pos, H = jnp.asarray(pos, jnp.float64), jnp.asarray(H, jnp.float64)
far = min(1.2, np.floor((max_cutoff(np.asarray(H)) - 0.002) * 100) / 100)
print(
    f"{system.n} atoms, box heights up to {max_cutoff(np.asarray(H)) * 2:.3f} nm, device {jax.devices()[0]}", flush=True
)


def run(label, **kw):
    base = dict(
        cutoff=a.cut,
        skin=0.1,
        ewald_beta=4.0,
        pme_spacing=0.08,
        pme_order=6,
        dipole_tol=1e-9,
        max_iter=200,
        precision="double",
        lj_lrc=True,
    )
    s = MDSettings(**{**base, **kw})
    t = time.time()
    ff = PGMForceField(system, H, s, topology=top)
    idx = ff.rows_for(pos, H)
    ff.size_rows(pos, H, idx)
    r = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
    jax.block_until_ready(r.forces)
    if bool(r.overflow):
        raise RuntimeError("row capacity overflow")
    print(
        f"  [{label}: grid {ff.pme.K}, rows {ff.capacity}, {int(r.iterations)} CG iterations, {time.time() - t:.1f} s]",
        flush=True,
    )
    return r


ref = run("reference", pme_spacing=0.04)
rmsF = float(jnp.sqrt(jnp.mean(run("reference, no van der Waals", pme_spacing=0.04, vdw="none").forces ** 2)))
Eel = float(ref.energy["elec"])
rmsmu = float(jnp.sqrt(jnp.mean(ref.induction.mu**2)))
print(f"rms electrostatic force {rmsF:.2f} kJ/mol/nm, E_elec {Eel:.2f} kJ/mol", flush=True)
print(f"{'case':58s} {'F rms':>9s} {'F max':>9s} {'dE kJ/mol':>10s} {'dE rel':>8s} {'mu rms':>8s}", flush=True)


def report(label, r):
    dF = r.forces - ref.forces
    print(
        f"{label:58s} {float(jnp.sqrt(jnp.mean(dF**2))) / rmsF:9.2e} {float(jnp.max(jnp.abs(dF))) / rmsF:9.2e} "
        f"{float(r.energy['elec']) - Eel:10.4f} {abs(float(r.energy['elec']) - Eel) / abs(Eel):8.1e} "
        f"{float(jnp.sqrt(jnp.mean((r.induction.mu - ref.induction.mu) ** 2))) / rmsmu:8.1e}",
        flush=True,
    )


report(
    f"reference vs {far} nm / 4.0 / 0.03, order 8", run("longer cutoff", elec_cutoff=far, pme_spacing=0.03, pme_order=8)
)
for prec in a.precision:
    tol = 1e-9 if prec == "double" else 1e-6
    report(f"0.9 / 4.00 / 0.0800 (default), {prec}", run("default", precision=prec, dipole_tol=tol))
    for rc in a.elec_cut:
        for ex in a.exponents:
            kw = elec_cutoff_settings(rc, exponent=ex)
            report(
                f"{rc} / {kw['ewald_beta']:.2f} / {kw['pme_spacing']:.4f} (exponent {ex:g}), {prec}",
                run(f"{rc}, exponent {ex:g}", precision=prec, dipole_tol=tol, **kw),
            )
