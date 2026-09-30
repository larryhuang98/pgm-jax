"""Check zero-field identity with the code in another directory (e.g. master).

The same short runs (rigid and flexible engines, NVT Bussi, mixed precision, 300 steps of 2 fs) of
a small pGM box (tests/test_md.small_box of that tree) are run by the code in --code (a checkout;
default: this repository), and positions and energies are saved to --out (.npz); --compare prints
the largest differences of two such files and whether they are bitwise identical.  Written for the
external-field work (docs/efield.md): with no field, the new code must reproduce the old one.

This script puts the package and tests of --code first on sys.path on purpose (it compares two code
versions), one of the two deliberate sys.path edits of the repository (docs/api_design.md, 9.1; the
other is the subprocess script of scripts/validation/validate_vsites.py identical).

Usage:

    git archive master | tar -x -C /tmp/master
    python scripts/validation/efield_identical.py --code /tmp/master --out m.npz
    python scripts/validation/efield_identical.py --out b.npz
    python scripts/validation/efield_identical.py --compare m.npz b.npz
    python scripts/validation/efield_identical.py --help

Inputs: the code tree (--code) with tests/test_md.py.
Outputs: the .npz (--out: rigid_pos, rigid_epot, flex_pos, flex_epot); printed sums or differences.
Units: nm, kJ/mol.
Runtime: CPU or GPU, a minute.  Sets jax_enable_x64 (after the path change).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # this checkout


def compare(a_path: str, b_path: str) -> None:
    """Print the largest absolute difference of every array of two result files, and bitwise identity."""
    x, y = np.load(a_path), np.load(b_path)
    for k in x.files:
        d = np.abs(x[k] - y[k]).max()
        print(f"{k:14s} max |difference| {d:.3e}  {'bitwise identical' if np.array_equal(x[k], y[k]) else ''}")


def run(code: str, out_path: str) -> None:
    """Run the two short simulations with the code of the tree `code` and save the results to out_path."""
    # the code under test is the tree given by --code (e.g. an older commit): its package and tests
    # are put first on the path on purpose, so this script compares two code versions
    sys.path.insert(0, code)
    sys.path.insert(0, os.path.join(code, "tests"))
    import jax

    jax.config.update("jax_enable_x64", True)
    from test_md import small_box

    from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate
    from pgm_jax.md.forcefield import MDSettings
    from pgm_jax.md.simulation import Simulation

    sys_, pos, H = small_box(0, nm=0)
    s = MDSettings().replace(cutoff=0.6, skin=0.05, pme_grid=(32, 32, 32), dipole_tol=1e-5, precision="mixed")
    out = {}
    sim = Simulation(sys_, pos, H, s, dt=0.002, thermostat="bussi", log=None, seed=7)
    sim.advance(300)
    out["rigid_pos"], out["rigid_epot"] = sim.positions(), np.array(float(sim.state.epot))
    tpl = [RigidTemplate(m, pos[sys_.atom_slice(k)]) for k, m in enumerate(sys_.molecules)]
    sim = FlexibleSimulation(sys_, tpl, pos, H, s, dt=0.002, thermostat="bussi", log=None, seed=7)
    sim.advance(300)
    out["flex_pos"], out["flex_epot"] = sim.positions(), np.array(float(sim.state.epot))
    np.savez(out_path, **out)
    print("saved", out_path, {k: float(np.sum(v)) for k, v in out.items()})


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run or compare (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", default=REPO, help="code tree to run (default: this checkout)")
    ap.add_argument("-o", "--out", help="result file (.npz)")
    ap.add_argument("--compare", nargs=2, help="two result files to compare")
    a = ap.parse_args(argv)
    if a.compare:
        compare(*a.compare)
    elif a.out:
        run(a.code, a.out)
    else:
        ap.error("--out or --compare")


if __name__ == "__main__":
    main()
