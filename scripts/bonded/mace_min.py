"""Find the MACE-OFF minimum of every molecule (lowest over its RDKit conformers).

Each conformer of data/bonded/molecules/<name>.json is minimised with ASE's BFGS (fmax 0.005
eV/A) on the MACE-OFF23 medium model (CPU, float64, 2 torch threads); the lowest minimum is saved.

Usage:

    python scripts/bonded/mace_min.py methanol ethanol ...
    python scripts/bonded/mace_min.py --help

Inputs: data/bonded/molecules/<name>.json; data/bonded/mace/MACE-OFF23_medium.model; torch, ASE and
mace installed.
Outputs: data/bonded/frames/<name>_min.npz (minima (1, N, 3) [A], minima_E (1,) [eV]); printed
energies.
Units: Angstrom, eV (ASE's).
Runtime: CPU, minutes per molecule.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from pgm_jax.paths import repo_path


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and minimise every molecule (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="molecule names")
    a = ap.parse_args(argv)
    import torch
    from ase import Atoms
    from ase.optimize import BFGS
    from mace.calculators import mace_off

    torch.set_num_threads(2)
    calc = mace_off(
        model=repo_path("data", "bonded", "mace", "MACE-OFF23_medium.model"), device="cpu", default_dtype="float64"
    )
    for name in a.names:
        with open(repo_path("data", "bonded", "molecules", f"{name}.json")) as fh:
            d = json.load(fh)
        best = None
        for x in d["conformers"]:
            at = Atoms(d["elements"], positions=np.array(x))
            at.calc = calc
            BFGS(at, logfile=None).run(fmax=0.005, steps=2000)
            e = at.get_potential_energy()
            if best is None or e < best[0]:
                best = (e, at.get_positions().copy())
        np.savez(
            os.path.join(repo_path("data", "bonded", "frames"), f"{name}_min.npz"),
            minima=best[1][None],
            minima_E=np.array([best[0]]),
        )
        print(name, best[0], flush=True)


if __name__ == "__main__":
    main()
