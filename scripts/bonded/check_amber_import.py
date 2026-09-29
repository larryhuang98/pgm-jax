"""Check the "amber" term set with GAFF parameters against cpptraj's Amber bonded energies.

Bond + angle + dihedral (incl. impropers) energies of the model initialised from the GAFF prmtop
(runs/bonded/pgm/<name>/gaff.prmtop, scripts/bonded/gaff_prmtop.py) against cpptraj on 20 DFT
test frames, compared as energy differences between frames (torsion constants differ by
convention), per family and in total.

Usage:

    python scripts/bonded/check_amber_import.py methanol formamide alanine_dipeptide ...
    python scripts/bonded/check_amber_import.py --help

Inputs: the study data and GAFF prmtops; cpptraj of AMBERHOME (pgm_jax.paths).
Outputs: printed maximum differences.
Units: kcal/mol.
Runtime: seconds per molecule.  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.bonded import terms as T
from pgm_jax.bonded.amber import init_from_prmtop, with_amber_impropers
from pgm_jax.bonded.model import BondedModel, BondedSettings
from pgm_jax.bonded.study.data import load
from pgm_jax.md.io import NetCDFTrajectory
from pgm_jax.paths import repo_path, resource
from pgm_jax.units import KCAL

jax.config.update("jax_enable_x64", True)
CPPTRAJ = resource("amberhome", "bin/cpptraj")


def check(name: str) -> None:
    """Compare the model's bonded energies of one molecule with cpptraj's and print the differences."""
    specs, data = load([name], with_scans=False)
    prm = repo_path("runs", "bonded", "pgm", name, "gaff.prmtop")
    with_amber_impropers(specs[0], prm)
    model = BondedModel(specs, BondedSettings(families=T.AMBER, typing="amber", lj14_scale=0.5))
    P = init_from_prmtop(model, model.init_params(), {0: prm})
    X = np.asarray(data[0]["test"].X)[:20]  # nm
    ours = np.array([float(model.bonded_energy(0, jnp.asarray(x), P)) for x in X]) / KCAL
    with tempfile.TemporaryDirectory() as wd:
        tr = NetCDFTrajectory(os.path.join(wd, "t.nc"), X.shape[1])
        for k, x in enumerate(X):
            tr.write(float(k), x * 10.0, np.eye(3) * 100.0)
        del tr
        with open(os.path.join(wd, "in"), "w") as fh:
            fh.write(f"parm {prm}\ntrajin t.nc\nenergy E bond angle dihedral out e.dat\nrun\n")
        subprocess.run([CPPTRAJ, "-i", "in"], cwd=wd, check=True, stdout=subprocess.DEVNULL)
        e = np.loadtxt(os.path.join(wd, "e.dat"))
    fam = {}
    for f in T.AMBER:
        m1 = BondedModel(specs, BondedSettings(families=(f,), typing="amber", lj14_scale=0.5))
        P1 = {"ref": P["ref"], f: P[f]}
        fam[f] = np.array([float(m1.bonded_energy(0, jnp.asarray(x), P1)) for x in X]) / KCAL

    def c(v):
        """Return v minus its mean (the energy zero differs by convention)."""
        return v - v.mean()

    print(
        name,
        "bond",
        np.abs(c(fam["bond_harm"]) - c(e[:, 1])).max(),
        "angle",
        np.abs(c(fam["angle_harm"]) - c(e[:, 2])).max(),
        "dih",
        np.abs(c(fam["torsion_amber"] + fam["improper_amber"]) - c(e[:, 3])).max(),
        "imp range",
        np.ptp(fam["improper_amber"]),
    )
    amb = e[:, 1:4].sum(1) if e.ndim == 2 else e[1:4].sum()
    d_ours, d_amb = ours - ours.mean(), amb - amb.mean()
    print(
        f"{name:20s} frames {len(X)}  Amber bonded range {np.ptp(amb):8.3f} kcal/mol  "
        f"max |diff| {np.abs(d_ours - d_amb).max():.2e}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and check every molecule (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="molecule names")
    for name in ap.parse_args(argv).names:
        check(name)


if __name__ == "__main__":
    main()
