"""Validate GVDW against pmemd-pgm's implementation (Huang, Luo, Duan; `igvdw = 1`) on the 512-water box.

pmemd-pgm is run with the settings of the LJ reference single point (data/validation/amber_ref/
pmemd_pbc: PME 72^3, order 8, 10 A cutoff, dipole tol 1e-9, vdwmeth = 0) and the GVDW water
parameters of the GVDW manuscript (O-O only: pgm_jax.vdw.PGM3P_GVDW), with the Slater and the
Gaussian repulsion.  The electrostatics of the runs are identical, so the GVDW forces are compared
as differences to the LJ run:
    F_gvdw(amber) - F_lj(amber)  vs  F_gvdw(ours) - F_lj(ours),
and the VDWAALS energies directly (docs/model_options.md).

Usage:

    python scripts/validation/validate_gvdw.py run       # pmemd-pgm, ~minutes per run (CPU)
    python scripts/validation/validate_gvdw.py compare
    python scripts/validation/validate_gvdw.py --help

Inputs: the LJ reference run and the box (as scripts/validation/validate_amber.py); pmemd-pgm
(PGM_PMEMD_CPU).
Outputs: runs/validate_gvdw/<rep>/ (pmemd runs), data/validation/validate_gvdw.json; printed results.
Units: kcal/mol, kcal/mol/A.
Runtime: CPU, minutes.  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
from validate_amber import PMEMD, REF, RST, TOP, mdout_step0, read_nc_frames, read_restart

from pgm_jax.lj import PeriodicLJ
from pgm_jax.md.box import box_from_cell
from pgm_jax.param import read_prmtop_molecules
from pgm_jax.paths import repo_path
from pgm_jax.system import System
from pgm_jax.units import KCAL
from pgm_jax.vdw import PGM3P_GVDW, PeriodicGVDW, set_gvdw

jax.config.update("jax_enable_x64", True)
OUT = repo_path("runs", "validate_gvdw")
RESULT = repo_path("data", "validation", "validate_gvdw.json")
PMEMD_GVDW = {  # pmemd-pgm &pol_gauss settings per repulsion form
    "slater": "igvdw=1, gvdw_rep_form=1, b_rep_scale=4.52, gvdw_arep=87500.0, gvdw_c6=594.825035",
    "gauss": "igvdw=1, gvdw_rep_form=0, b_rep_scale=0.9453, gvdw_arep=422.0, gvdw_c6=594.825035",
}


def run() -> None:
    """Run pmemd-pgm with GVDW (both repulsion forms) at the LJ reference settings."""
    base = open(os.path.join(REF, "pmemd_pbc", "mdin")).read()
    for rep, nml in PMEMD_GVDW.items():
        wd = os.path.join(OUT, rep)
        os.makedirs(wd, exist_ok=True)
        mdin = base.replace("dipole_print=1,", f"dipole_print=0,\n  {nml},")
        assert nml in mdin
        open(os.path.join(wd, "mdin"), "w").write(mdin.replace("single point", f"GVDW ({rep}) single point"))
        subprocess.run(
            [PMEMD, "-O", "-i", "mdin", "-c", RST, "-p", TOP, "-o", "mdout", "-x", "mdcrd", "-frc", "mdfrc"],
            cwd=wd,
            check=True,
        )
        print(rep, mdout_step0(os.path.join(wd, "mdout")), flush=True)


def compare() -> None:
    """Compare our GVDW energies and force differences with pmemd-pgm's and write the JSON."""
    xyz, cell = read_restart(RST)
    H = box_from_cell(cell[0], cell[1]) * 0.1
    pos = jnp.asarray(xyz * 0.1)
    mols = read_prmtop_molecules(TOP)

    def to_kcal_A(F):
        """Convert forces from kJ/mol/nm to kcal/mol/A."""
        return np.asarray(F) / KCAL / 10.0

    lj_amb = mdout_step0(os.path.join(REF, "pmemd_pbc", "mdout"))
    f_lj_amb = read_nc_frames(os.path.join(REF, "pmemd_pbc", "mdfrc"), "forces")[0]
    sys_lj = System(mols)
    ljp = PeriodicLJ(sys_lj, H, pos, rc=1.0)
    e_lj = float(ljp.energy(pos)[0]["vdw"]) / KCAL
    f_lj = to_kcal_A(-jax.grad(lambda x: ljp.energy(x)[0]["vdw"])(pos))
    res = {"LJ": {"VDWAALS_amber": lj_amb["VDWAALS"], "VDWAALS_ours": e_lj}}
    for rep in PMEMD_GVDW:
        wd = os.path.join(OUT, rep)
        amb = mdout_step0(os.path.join(wd, "mdout"))
        f_amb = read_nc_frames(os.path.join(wd, "mdfrc"), "forces")[0]
        par = PGM3P_GVDW[rep]
        gm = {id(m): set_gvdw(m, {"OW": par["OW"]}) for m in mols}
        sys_g = System([gm[id(m)] for m in mols])
        gp = PeriodicGVDW(sys_g, H, pos, rc=1.0, rep=par["rep"])
        e_g = float(gp.energy(pos)[0]["vdw"]) / KCAL
        f_g = to_kcal_A(-jax.grad(lambda x: gp.energy(x)[0]["vdw"])(pos))
        d_amb, d_ours = f_amb - f_lj_amb, f_g - f_lj
        res[rep] = {
            "VDWAALS_amber": amb["VDWAALS"],
            "VDWAALS_ours": e_g,
            "dE": e_g - amb["VDWAALS"],
            "EELEC_amber": amb["EELEC"],
            "EELEC_amber_LJ_run": lj_amb["EELEC"],
            "dF_rms_amber": float(np.sqrt(np.mean(d_amb**2))),
            "dF_rmsd": float(np.sqrt(np.mean((d_ours - d_amb) ** 2))),
            "dF_maxdev": float(np.abs(d_ours - d_amb).max()),
        }
        print(rep, json.dumps(res[rep], indent=1), flush=True)
    with open(RESULT, "w") as fh:
        json.dump(res, fh, indent=1)


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run one step (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["run", "compare"], help="run pmemd-pgm, or compare")
    {"run": run, "compare": compare}[ap.parse_args(argv).step]()


if __name__ == "__main__":
    main()
