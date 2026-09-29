"""Validation runs for the static dielectric constant of pGM water, with the cell dipole recorded.

The 512-water pGM3P box of the README (optionally replicated n x n x n) or its TIP3P control is
run with the cell dipole recorded (<out>.dip; analyse with scripts/dielectric/dielectric.py;
docs/dielectric.md).

The prmtop (PGM_GVDW_DATA/topology/rayl_512_v2.prmtop) has the pGM3P-25 electrostatics (charges,
covalent dipoles, radii, polarizabilities of Wu et al., JCTC 21, 3563 (2025)) on TIP3P's geometry
(0.9572 A, 104.52 deg) and Lennard-Jones.  --model pgm3p25 uses the paper's geometry (0.9745 A,
103.64 deg; the hydrogens are rebuilt about each oxygen) and Lennard-Jones (sigma 3.18156 A, epsilon
0.14473 kcal/mol).  --model tip3p keeps the box, geometry and Lennard-Jones of the prmtop and
replaces the pGM electrostatics by TIP3P point charges (q_O = -0.834 e; Gaussian radii of 1e-4 nm,
no covalent or induced dipoles, MDSettings(elec="q")): an end-to-end check of the recording and of
the fluctuation formula against a model with a well-known eps.  Default: NPT at 298 K and 1 bar,
Bussi thermostat (1 ps), Monte Carlo barostat (every 100 steps), PME 48^3 per replica (order 6),
0.9 nm cutoff, LJ tail.  Configurational properties do not depend on the masses, so --hmr-amu
(heavier water hydrogens, constraints engine) with 4 fs is legitimate for eps (not for the IR
spectrum).

Usage:

    python scripts/dielectric/water_dielectric.py --time-ns 20 -o runs/eps/pgm                 # rigid bodies, 2 fs
    python scripts/dielectric/water_dielectric.py --time-ns 10 --engine constraints --hmr-amu 4.0 --dt-fs 4
        -o runs/eps/pgm_hmr
    python scripts/dielectric/water_dielectric.py --time-ns 10 --model tip3p -o runs/eps/tip3p  # point charges
    python scripts/dielectric/water_dielectric.py --time-ns 10 --model pgm3p25 -o runs/eps/p25  # paper's geometry
    python scripts/dielectric/water_dielectric.py --time-ns 0.2 --barostat none --dipoles-every 1 -o runs/eps/ir  # IR
    python scripts/dielectric/water_dielectric.py --help

Inputs: --prmtop and --coords (default: PGM_GVDW_DATA's 512-water box, pgm_jax.paths).
Outputs: <out>.log, <out>.dip (cell dipole every --dipoles-every steps), <out>.chk / <out>.rst7
every --checkpoint-every steps; --continue-from appends to them.
Units: --time-ns ns, --dt-fs fs, --temperature-K K, --pressure-bar bar, --tau-ps ps, --hmr-amu amu.
Runtime: GPU for production runs (ns); a few steps run on a CPU.  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

import jax
import numpy as np

from pgm_jax.cli.args import (
    add_barostat_args,
    add_dipole_tol_arg,
    add_dt_arg,
    add_iel_args,
    add_output_args,
    add_seed_arg,
    add_temperature_arg,
    add_thermostat_args,
    coupling_from_args,
    iel_settings,
    setup_logging,
)
from pgm_jax.md.box import box_from_cell
from pgm_jax.md.forcefield import MDSettings
from pgm_jax.md.io import read_coordinates
from pgm_jax.md.simulation import Simulation
from pgm_jax.param import read_prmtop_molecules
from pgm_jax.paths import pgm3p25_files
from pgm_jax.system import System
from pgm_jax.units import KCAL

jax.config.update("jax_enable_x64", True)
TOP, RST = pgm3p25_files()


def paper_geometry(xyz: np.ndarray, l_oh: float, theta: float, elements: list[list[str]]) -> np.ndarray:
    """Rebuild the hydrogens of every water at a given bond length and angle.

    The oxygen, the HOH plane and the bisector of each molecule are kept.

    Parameters
    ----------
    xyz : np.ndarray (N, 3)
        Coordinates of a box of waters in the order O, H, H [A].
    l_oh : float
        O-H bond length [A].
    theta : float
        H-O-H angle [deg].
    elements : list of list of str
        Elements of each molecule (must all be O, H, H).

    Returns
    -------
    np.ndarray (N, 3)
        The new coordinates [A].

    Raises
    ------
    ValueError
        A molecule that is not O, H, H.
    """
    if any(e != ["O", "H", "H"] for e in elements):
        raise ValueError("expected a box of water only")
    x = np.asarray(xyz, float).reshape(-1, 3, 3).copy()
    o, h1, h2 = x[:, 0], x[:, 1], x[:, 2]
    b = (h1 - o) + (h2 - o)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    u = h1 - h2
    u -= np.sum(u * b, 1, keepdims=True) * b
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    c, s = np.cos(np.radians(theta / 2)), np.sin(np.radians(theta / 2))
    x[:, 1] = o + l_oh * (c * b + s * u)
    x[:, 2] = o + l_oh * (c * b - s * u)
    return x.reshape(-1, 3)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--model",
        default="pgm",
        choices=["pgm", "pgm3p25", "tip3p"],
        help="pgm: the prmtop as it is; pgm3p25: the paper's geometry and LJ; tip3p: TIP3P point charges",
    )
    ap.add_argument("--engine", default="rigid", choices=["rigid", "constraints"], help="MD engine")
    ap.add_argument(
        "--hmr-amu", type=float, default=None, help="water hydrogen mass [amu] (constraints engine; default: as is)"
    )
    ap.add_argument("--time-ns", type=float, default=10.0, help="run length [ns]")
    add_dt_arg(ap, 2.0)
    add_temperature_arg(ap, 298.0)
    add_thermostat_args(ap, default="bussi", tau=1.0)
    add_barostat_args(ap, default="mc")
    ap.add_argument("--replicate", type=int, default=1, help="n: replicate the box n x n x n")
    ap.add_argument("--dipoles-every", type=int, default=25, help="steps between cell-dipole samples")
    add_dipole_tol_arg(ap)
    add_seed_arg(ap)
    ap.add_argument("--prmtop", default=TOP, help="pGM water prmtop (--model pgm/tip3p; default: the pGM3P box)")
    ap.add_argument("--coords", default=RST, help="restart matching --prmtop")
    add_output_args(ap, out=None, report_every=5000, checkpoint_every=250000, continue_from=True)
    add_iel_args(ap)
    return ap


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, build the box and run it (see the module docstring)."""
    a = build_parser().parse_args(argv)
    setup_logging()
    mols = read_prmtop_molecules(a.prmtop)
    elec = "qpi"
    if a.model == "tip3p":
        if any(list(m.elements) != ["O", "H", "H"] for m in mols):
            raise ValueError("--model tip3p expects a box of water only")
        tip = {
            id(m): dataclasses.replace(
                m, name="TIP3", q=np.array([-0.834, 0.417, 0.417]), radius=np.full(3, 1e-4), cov=[]
            )
            for m in mols
        }
        mols = [tip[id(m)] for m in mols]
        elec = "q"
    xyz, vel, box = read_coordinates(a.coords)
    if a.model == "pgm3p25":
        xyz = paper_geometry(xyz, 0.9745, 103.64, [list(m.elements) for m in mols])
        sig, eps = 3.18156, 0.14473  # A, kcal/mol (A = 622716.4, B = 600.41)
        rh = np.array([2 ** (1 / 6) * sig / 2 * 0.1, 0.0, 0.0])
        se = np.array([np.sqrt(eps * KCAL), 0.0, 0.0])
        new = {id(m): dataclasses.replace(m, lj_rmin_half=rh, lj_sqrt_eps=se) for m in mols}
        mols = [new[id(m)] for m in mols]
    H = box_from_cell(*box) * 0.1
    n = a.replicate
    shifts = [i * H[0] + j * H[1] + k * H[2] for i in range(n) for j in range(n) for k in range(n)]
    pos = np.concatenate([xyz * 0.1 + s for s in shifts])
    v = None if vel is None else np.concatenate([vel * 0.1] * len(shifts))
    sys_ = System(mols * len(shifts))
    st = MDSettings().replace(
        cutoff=0.9,
        skin=0.1,
        ewald_beta=4.0,
        pme_grid=(48 * n,) * 3,
        pme_order=6,
        lj_lrc=True,
        dipole_tol=a.dipole_tol,
        precision="mixed",
        elec=elec,
        **iel_settings(a),
    )
    thermostat, barostat = coupling_from_args(a)
    kw = dict(
        settings=st,
        temperature=a.temperature_K,
        thermostat=thermostat,
        barostat=barostat,
        dt=a.dt_fs / 1000,
        log=sys.stdout,
        seed=a.seed,
    )
    if a.engine == "rigid":
        sim = Simulation(sys_, pos, H * n, velocities=v, **kw)
    else:
        from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate

        tpl = {id(m): RigidTemplate(m, pos[sys_.atom_slice(k)]) for k, m in enumerate(sys_.molecules)}
        sim = FlexibleSimulation(sys_, [tpl[id(m)] for m in sys_.molecules], pos, H * n, hmr=a.hmr_amu, **kw)
    if a.continue_from:
        sim.load_checkpoint(a.continue_from)
    nsteps = int(round(a.time_ns * 1e6 / a.dt_fs))
    nsteps -= nsteps % a.report_every
    sim.run(
        nsteps,
        report_every=a.report_every,
        checkpoint_every=a.checkpoint_every,
        prefix=a.out,
        dipoles_every=a.dipoles_every,
        append=bool(a.continue_from),
    )


if __name__ == "__main__":
    main()
