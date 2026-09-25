"""Check the "amber" term set with GAFF parameters against cpptraj's Amber bonded energies
(bond + angle + dihedral incl. impropers) on DFT test frames, as energy differences between
frames (torsion constants differ by convention).
    python scripts/bonded/check_amber_import.py methanol formamide alanine_dipeptide ..."""
import os, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts/bonded"))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from experiments import load
from pgm_jax.bonded import terms as T
from pgm_jax.bonded.amber import init_from_prmtop, with_amber_impropers
from pgm_jax.bonded.model import BondedModel, BondedSettings

CPPTRAJ = os.path.expanduser("~/amber25/bin/cpptraj")
for name in sys.argv[1:]:
    specs, data = load([name], with_scans=False)
    prm = os.path.join(ROOT, "runs/bonded/pgm", name, "gaff.prmtop")
    with_amber_impropers(specs[0], prm)
    model = BondedModel(specs, BondedSettings(families=T.AMBER, typing="amber", lj14_scale=0.5))
    P = init_from_prmtop(model, model.init_params(), {0: prm})
    X = np.asarray(data[0]["test"].X)[:20]                         # nm
    ours = np.array([float(model.bonded_energy(0, jnp.asarray(x), P)) for x in X]) / 4.184
    from pgm_jax.md.io import NetCDFTrajectory
    with tempfile.TemporaryDirectory() as wd:
        tr = NetCDFTrajectory(os.path.join(wd, "t.nc"), X.shape[1])
        for k, x in enumerate(X):
            tr.write(float(k), x * 10.0, np.eye(3) * 100.0)
        del tr
        open(os.path.join(wd, "in"), "w").write(f"parm {prm}\ntrajin t.nc\nenergy E bond angle dihedral out e.dat\nrun\n")
        subprocess.run([CPPTRAJ, "-i", "in"], cwd=wd, check=True, stdout=subprocess.DEVNULL)
        e = np.loadtxt(os.path.join(wd, "e.dat"))
    fam = {}
    for f in T.AMBER:
        m1 = BondedModel(specs, BondedSettings(families=(f,), typing="amber", lj14_scale=0.5))
        P1 = {"ref": P["ref"], f: P[f]}
        fam[f] = np.array([float(m1.bonded_energy(0, jnp.asarray(x), P1)) for x in X]) / 4.184
    c = lambda v: v - v.mean()
    print(name, "bond", np.abs(c(fam["bond_harm"]) - c(e[:, 1])).max(), "angle", np.abs(c(fam["angle_harm"]) - c(e[:, 2])).max(),
          "dih", np.abs(c(fam["torsion_amber"] + fam["improper_amber"]) - c(e[:, 3])).max(),
          "imp range", np.ptp(fam["improper_amber"]))
    amb = e[:, 1:4].sum(1) if e.ndim == 2 else e[1:4].sum()
    d_ours, d_amb = ours - ours.mean(), amb - amb.mean()
    print(f"{name:20s} frames {len(X)}  Amber bonded range {np.ptp(amb):8.3f} kcal/mol  "
          f"max |diff| {np.abs(d_ours - d_amb).max():.2e}", flush=True)
