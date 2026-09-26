"""Write the MD engine's model of a tleap system as a pmemd-pgm prmtop (and a matching mdin), for
production MD with pmemd.pgm.cuda (protein/pmemd.py describes what is written).

    python scripts/protein/write_pgm_prmtop.py sys.prmtop sys.inpcrd sys_pgm.prmtop --library lib.json \
        --mdin sys --nstlim 500000 --dt 0.002
    P=pmemd.pgm.cuda_SPFP
    $P -O -i sys.min.in -p sys_pgm.prmtop -c sys.inpcrd -o min.out -r min.rst7
    $P -O -i sys.heat.in -p sys_pgm.prmtop -c min.rst7 -o heat.out -r heat.rst7
    $P -O -i sys.md.in -p sys_pgm.prmtop -c heat.rst7 -o md.out -r md.rst7 -x md.nc

Electrostatics: --library (ResidueLibrary JSON), --placeholder (Amber charges, pGM-pol
polarizabilities: pipeline tests only) or --from-prmtop (the input is a pGM prmtop).  --water
replaces the water model (Molecule JSON, param.save_molecule; atoms in the prmtop's order).
Bonded terms of the flexible molecules: --bonded template (default: amber_template, the
ff19SB-form terms with Fourier CMAP that FlexibleSimulation runs), --template file.flex (a saved
FlexibleTemplate: typed or neural fit; one flexible molecule), or --bonded prmtop (the input's own
terms, e.g. ff19SB with its CMAP grids, with --lj14 as the 1-4 scale).  --hmr: hydrogen masses as
FlexibleSimulation(hmr=...).  --mdin PREFIX writes three pmemd inputs with the nonbonded model
of the given settings (cutoff, Ewald coefficient, PME grid of pmemd_grid(spacing), order, LJ
tail, dipole tolerance): PREFIX.min.in (500 minimisation steps: tleap structures have clashes),
PREFIX.heat.in (2 ps at 0.5 fs from 0 K: pmemd's tempi would start a constrained system ~1.5x
too hot) and PREFIX.md.in (the run, continuing from the heating restart).
"""
import argparse
import json
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pgm_jax.md.flexible import FlexibleTemplate  # noqa: E402
from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.param import load_molecule  # noqa: E402
from pgm_jax.protein import ResidueLibrary, amber_template, load_amber, pmemd_grid, pmemd_mdin, write_pgm_prmtop  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("prmtop")
ap.add_argument("inpcrd")
ap.add_argument("out")
g = ap.add_mutually_exclusive_group()
g.add_argument("--library", help="ResidueLibrary JSON")
g.add_argument("--placeholder", action="store_true", help="placeholder electrostatics (default)")
g.add_argument("--from-prmtop", action="store_true", help="the input is a pGM prmtop")
ap.add_argument("--water", help="water Molecule JSON (param.save_molecule)")
ap.add_argument("--bonded", choices=("template", "prmtop"), default="template")
ap.add_argument("--template", help="saved FlexibleTemplate of the (single) flexible molecule")
ap.add_argument("--lj14", type=float, default=0.5, help="1-4 LJ scale with --bonded prmtop")
ap.add_argument("--hmr", type=float, default=None, help="hydrogen mass (amu)")
ap.add_argument("--mdin", help="also write pmemd-pgm inputs PREFIX.{min,heat,md}.in")
ap.add_argument("--nstlim", type=int, default=500000)
ap.add_argument("--dt", type=float, default=0.002, help="ps")
ap.add_argument("--ensemble", default="nvt", choices=("nve", "nvt", "npt"))
ap.add_argument("--thermostat", default="langevin", choices=("langevin", "bussi"))
ap.add_argument("--temp", type=float, default=298.0)
ap.add_argument("--cut", type=float, default=0.9, help="nm")
ap.add_argument("--beta", type=float, default=4.0, help="Ewald coefficient (nm^-1)")
ap.add_argument("--spacing", type=float, default=0.08, help="PME grid spacing (nm)")
ap.add_argument("--order", type=int, default=6)
ap.add_argument("--tol", type=float, default=1e-5, help="dipole_scf_tol")
ap.add_argument("--lrc", type=int, default=1, help="LJ long-range correction (vdwmeth)")
a = ap.parse_args()

elec = ResidueLibrary.load(a.library) if a.library else ("prmtop" if a.from_prmtop else "placeholder")
asys = load_amber(a.prmtop, a.inpcrd, electrostatics=elec, water=load_molecule(a.water) if a.water else None)
flex = [k for k, m in enumerate(asys.molecules) if m.kind not in ("water", "ion")]
if a.bonded == "prmtop":
    if a.template:
        raise SystemExit("--template needs --bonded template")
    info = write_pgm_prmtop(asys, a.out, lj14_scale=a.lj14, hmr=a.hmr)
else:
    if a.template:
        if len(flex) != 1:
            raise SystemExit(f"--template: the system has {len(flex)} flexible molecules")
        tpls = {flex[0]: FlexibleTemplate.load(a.template)}
    else:
        tpls = {k: amber_template(asys.molecules[k], a.prmtop) for k in flex}
    info = write_pgm_prmtop(asys, a.out, asys.templates(tpls), hmr=a.hmr)
info["exported"] = {str(k): v for k, v in info["exported"].items()}
print(json.dumps(info, indent=1, default=float))
if a.mdin:
    st = MDSettings(cutoff=a.cut, ewald_beta=a.beta, pme_grid=pmemd_grid(asys.box, a.spacing), pme_order=a.order,
                    lj_lrc=bool(a.lrc), dipole_tol=a.tol)
    kw = dict(ensemble=a.ensemble, thermostat=a.thermostat, temperature=a.temp)
    open(a.mdin + ".min.in", "w").write(pmemd_mdin(st, asys.box, maxcyc=500, ntpr=100))
    open(a.mdin + ".heat.in", "w").write(pmemd_mdin(st, asys.box, nstlim=4000, dt=0.0005, tempi=0.0, ntpr=500,
                                                    ntwr=4000, **{**kw, "ensemble": "nvt"}))
    open(a.mdin + ".md.in", "w").write(pmemd_mdin(st, asys.box, nstlim=a.nstlim, dt=a.dt, irest=1, ntpr=1000,
                                                  ntwx=5000, ntwr=50000, **kw))
    print(f"mdin: {a.mdin}.min.in, .heat.in, .md.in (engine settings: MDSettings(cutoff={a.cut}, ewald_beta={a.beta}, "
          f"pme_grid={st.pme_grid}, pme_order={a.order}, lj_lrc={bool(a.lrc)}, dipole_tol={a.tol}))")
