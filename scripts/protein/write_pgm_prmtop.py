"""Write the MD engine's model of a tleap system as a pmemd-pgm prmtop and mdin (`pgm-jax write-prmtop`).

For production MD with pmemd.pgm.cuda (pgm_jax/protein/pmemd.py describes what is written;
docs/protein_ff.md).

Electrostatics: --library (ResidueLibrary JSON), --placeholder (Amber charges, pGM-pol
polarizabilities: pipeline tests only) or --from-prmtop (the input is a pGM prmtop).  --water
replaces the water model (Molecule JSON, param.save_molecule; atoms in the prmtop's order).
Bonded terms of the flexible molecules: --bonded template (default: amber_template, the
ff19SB-form terms with Fourier CMAP that FlexibleSimulation runs), --template file.flex (a saved
FlexibleTemplate: typed or neural fit; one flexible molecule), or --bonded prmtop (the input's own
terms, e.g. ff19SB with its CMAP grids, with --lj14 as the 1-4 scale).  --hmr: hydrogen masses as
FlexibleSimulation(hmr=...) (--hmr-amu).  --mdin PREFIX writes three pmemd inputs with the nonbonded model
of the given settings (cutoff, Ewald coefficient, PME grid of pmemd_grid(spacing), order, LJ
tail, dipole tolerance): PREFIX.min.in (500 minimisation steps: tleap structures have clashes),
PREFIX.heat.in (2 ps at 0.5 fs from 0 K: pmemd's tempi would start a constrained system ~1.5x
too hot) and PREFIX.md.in (the run, continuing from the heating restart; --thermostat none is
NVE, --barostat mc NPT).

Usage:

    python scripts/protein/write_pgm_prmtop.py sys.prmtop sys.inpcrd sys_pgm.prmtop --library lib.json
        --mdin sys --nstlim 500000 --dt-fs 2
    P=pmemd.pgm.cuda_SPFP
    $P -O -i sys.min.in -p sys_pgm.prmtop -c sys.inpcrd -o min.out -r min.rst7
    $P -O -i sys.heat.in -p sys_pgm.prmtop -c min.rst7 -o heat.out -r heat.rst7
    $P -O -i sys.md.in -p sys_pgm.prmtop -c heat.rst7 -o md.out -r md.rst7 -x md.nc
    python scripts/protein/write_pgm_prmtop.py --help

Inputs: the tleap prmtop and coordinates; --library / --water / --template files.
Outputs: the pmemd-pgm prmtop, with --mdin PREFIX.{min,heat,md}.in; the export summary (JSON) is
printed.
Units: --dt-fs fs, --temperature-K K, --pressure-bar bar, --friction-per-ps 1/ps, --tau-ps ps,
--cutoff-nm and --pme-spacing-nm nm, --ewald-beta-per-nm 1/nm, --hmr-amu amu (written in Amber's
units).
Runtime: seconds to a minute.  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json

import jax
import numpy as np

from pgm_jax.cli.args import (
    add_barostat_args,
    add_cutoff_arg,
    add_dipole_tol_arg,
    add_dt_arg,
    add_temperature_arg,
    add_thermostat_args,
)
from pgm_jax.md.flexible import FlexibleTemplate
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.param import load_molecule
from pgm_jax.protein import (
    ResidueLibrary,
    amber_template,
    load_amber,
    pmemd_grid,
    pmemd_mdin,
    write_pgm_prmtop,
)

jax.config.update("jax_enable_x64", True)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prmtop", help="tleap prmtop")
    ap.add_argument("inpcrd", help="tleap coordinates")
    ap.add_argument("out", help="output pmemd-pgm prmtop")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--library", help="ResidueLibrary JSON")
    g.add_argument("--placeholder", action="store_true", help="placeholder electrostatics (default)")
    g.add_argument("--from-prmtop", action="store_true", help="the input is a pGM prmtop")
    ap.add_argument("--water", help="water Molecule JSON (param.save_molecule)")
    ap.add_argument("--bonded", choices=("template", "prmtop"), default="template", help="source of the bonded terms")
    ap.add_argument("--template", help="saved FlexibleTemplate of the (single) flexible molecule")
    ap.add_argument("--lj14", type=float, default=0.5, help="1-4 LJ scale with --bonded prmtop")
    ap.add_argument("--hmr-amu", type=float, default=None, help="hydrogen mass [amu] (default: unchanged)")
    g = ap.add_argument_group("pmemd inputs (--mdin)")
    g.add_argument("--mdin", help="also write pmemd-pgm inputs PREFIX.{min,heat,md}.in")
    g.add_argument("--nstlim", type=int, default=500000, help="steps of the md run (pmemd nstlim)")
    add_dt_arg(g, 2.0, help="time step of the md run [fs]")
    add_temperature_arg(g, 298.0)
    add_thermostat_args(ap, default="langevin", choices=("none", "langevin", "bussi"))
    add_barostat_args(ap, default="none")
    g = ap.add_argument_group("nonbonded model of the mdin")
    add_cutoff_arg(g, 0.9)
    g.add_argument("--ewald-beta-per-nm", type=float, default=4.0, help="Ewald coefficient [1/nm]")
    g.add_argument("--pme-spacing-nm", type=float, default=0.08, help="PME grid spacing [nm] (pmemd_grid)")
    g.add_argument("--order", type=int, default=6, help="PME order")
    add_dipole_tol_arg(g, 1e-5, help="dipole_scf_tol")
    g.add_argument("--lrc", type=int, default=1, help="LJ long-range correction (vdwmeth)")
    return ap


def write_mdin(a: argparse.Namespace, box: np.ndarray) -> None:
    """Write PREFIX.min.in, PREFIX.heat.in and PREFIX.md.in (see the module docstring).

    Parameters
    ----------
    a : argparse.Namespace
        Options.
    box : np.ndarray (3, 3)
        The system's box [nm].
    """
    st = MDSettings().replace(
        cutoff=a.cutoff_nm,
        ewald_beta=a.ewald_beta_per_nm,
        pme_grid=pmemd_grid(box, a.pme_spacing_nm),
        pme_order=a.order,
        lj_lrc=bool(a.lrc),
        dipole_tol=a.dipole_tol,
    )
    ensemble = "nve" if a.thermostat == "none" else ("npt" if a.barostat == "mc" else "nvt")
    thermostat = "langevin" if a.thermostat == "none" else a.thermostat
    kw = dict(
        ensemble=ensemble,
        thermostat=thermostat,
        temperature=a.temperature_K,
        gamma=a.friction_per_ps,
        tau_t=a.tau_ps,
        pressure=a.pressure_bar,
    )
    with open(a.mdin + ".min.in", "w") as fh:
        fh.write(pmemd_mdin(st, box, maxcyc=500, ntpr=100))
    with open(a.mdin + ".heat.in", "w") as fh:
        fh.write(
            pmemd_mdin(st, box, nstlim=4000, dt=0.0005, tempi=0.0, ntpr=500, ntwr=4000, **{**kw, "ensemble": "nvt"})
        )
    with open(a.mdin + ".md.in", "w") as fh:
        fh.write(
            pmemd_mdin(st, box, nstlim=a.nstlim, dt=a.dt_fs / 1000, irest=1, ntpr=1000, ntwx=5000, ntwr=50000, **kw)
        )
    print(
        f"mdin: {a.mdin}.min.in, .heat.in, .md.in (engine settings: MDSettings(cutoff={a.cutoff_nm}, "
        f"ewald_beta={a.ewald_beta_per_nm}, pme_grid={st.pme.grid}, pme_order={a.order}, lj_lrc={bool(a.lrc)}, "
        f"dipole_tol={a.dipole_tol}))"
    )


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and write the prmtop (and the mdin files) (see the module docstring)."""
    a = build_parser().parse_args(argv)
    elec = ResidueLibrary.load(a.library) if a.library else ("prmtop" if a.from_prmtop else "placeholder")
    asys = load_amber(a.prmtop, a.inpcrd, electrostatics=elec, water=load_molecule(a.water) if a.water else None)
    flex = [k for k, m in enumerate(asys.molecules) if m.kind not in ("water", "ion")]
    if a.bonded == "prmtop":
        if a.template:
            raise SystemExit("--template needs --bonded template")
        info = write_pgm_prmtop(asys, a.out, lj14_scale=a.lj14, hmr=a.hmr_amu)
    else:
        if a.template:
            if len(flex) != 1:
                raise SystemExit(f"--template: the system has {len(flex)} flexible molecules")
            tpls = {flex[0]: FlexibleTemplate.load(a.template)}
        else:
            tpls = {k: amber_template(asys.molecules[k], a.prmtop) for k in flex}
        info = write_pgm_prmtop(asys, a.out, asys.templates(tpls), hmr=a.hmr_amu)
    info["exported"] = {str(k): v for k, v in info["exported"].items()}
    print(json.dumps(info, indent=1, default=float))
    if a.mdin:
        write_mdin(a, asys.box)


if __name__ == "__main__":
    main()
