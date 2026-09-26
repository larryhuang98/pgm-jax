"""Validation runs for the static dielectric constant: pGM water (the 512-water pGM3P box of the
README, optionally replicated n x n x n) or its TIP3P control, with the cell dipole recorded
(prefix.dip; analyse with scripts/dielectric.py).

    python scripts/water_dielectric.py --ns 20 -o runs/eps/pgm                       # rigid bodies, 2 fs
    python scripts/water_dielectric.py --ns 10 --engine constraints --hmr 4.0 --dt 4 -o runs/eps/pgm_hmr
    python scripts/water_dielectric.py --ns 10 --model tip3p -o runs/eps/tip3p        # fixed point charges
    python scripts/water_dielectric.py --ns 10 --model pgm3p25 -o runs/eps/pgm3p25    # geometry + LJ of the paper
    python scripts/water_dielectric.py --ns 0.2 --ensemble nvt --dipoles 1 -o runs/eps/ir   # M every step (IR)

The prmtop (~/pgm-gvdw-data/topology/rayl_512_v2.prmtop) has the pGM3P-25 electrostatics (charges,
covalent dipoles, radii, polarizabilities of Wu et al., JCTC 21, 3563 (2025)) on TIP3P's geometry
(0.9572 A, 104.52 deg) and Lennard-Jones.  --model pgm3p25 uses the paper's geometry (0.9745 A,
103.64 deg; the hydrogens are rebuilt about each oxygen) and Lennard-Jones (sigma 3.18156 A, epsilon
0.14473 kcal/mol).  --model tip3p keeps the box, geometry and Lennard-Jones of the prmtop and
replaces the pGM electrostatics by TIP3P point charges (q_O = -0.834 e; Gaussian radii of 1e-4 nm,
no covalent or induced dipoles, MDSettings(elec="q")): an end-to-end check of the recording and of
the fluctuation formula against a model with a well-known eps.  NPT at 298 K and 1 bar, Bussi
thermostat, Monte Carlo barostat, PME 48^3 per replica (order 6), 0.9 nm cutoff, LJ tail.
Configurational properties do not depend on the masses, so --hmr (heavier water hydrogens,
constraints engine) with 4 fs is legitimate for eps (not for the IR spectrum).
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pgm_jax.md.forcefield import MDSettings  # noqa: E402
from pgm_jax.md.io import box_from_cell, read_coordinates  # noqa: E402
from pgm_jax.md.simulation import Simulation, _dedupe  # noqa: E402
from pgm_jax.param import read_prmtop_pgm  # noqa: E402
from pgm_jax.system import System  # noqa: E402

TOP = os.path.expanduser("~/pgm-gvdw-data/topology/rayl_512_v2.prmtop")
RST = os.path.expanduser("~/pgm-gvdw-data/inputs/lj/inpcrd.restrt")


def paper_geometry(xyz, l_oh, theta, elements):
    """Rebuild the hydrogens of every water (O, H, H; Angstrom) at bond length l_oh (A) and angle
    theta (deg), keeping the oxygen, the HOH plane and the bisector."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", required=True, help="output prefix")
    ap.add_argument("--model", default="pgm", choices=["pgm", "pgm3p25", "tip3p"])
    ap.add_argument("--engine", default="rigid", choices=["rigid", "constraints"])
    ap.add_argument("--hmr", type=float, default=None, help="water hydrogen mass (amu), constraints engine")
    ap.add_argument("--ns", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=2.0, help="fs")
    ap.add_argument("--ensemble", default="npt", choices=["nvt", "npt"])
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--dipoles", type=int, default=25, help="steps between cell-dipole samples")
    ap.add_argument("--report", type=int, default=5000)
    ap.add_argument("--restart", type=int, default=250000)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", help="continue from this .chk (appends to the outputs)")
    a = ap.parse_args()
    mols = _dedupe(read_prmtop_pgm(TOP, first_residue_only=False))
    elec = "qpi"
    if a.model == "tip3p":
        if any(list(m.elements) != ["O", "H", "H"] for m in mols):
            raise ValueError("--model tip3p expects a box of water only")
        tip = {id(m): dataclasses.replace(m, name="TIP3", q=np.array([-0.834, 0.417, 0.417]), radius=np.full(3, 1e-4),
                                          cov=[]) for m in mols}
        mols = [tip[id(m)] for m in mols]
        elec = "q"
    xyz, vel, box = read_coordinates(RST)
    if a.model == "pgm3p25":
        xyz = paper_geometry(xyz, 0.9745, 103.64, [list(m.elements) for m in mols])
        sig, eps = 3.18156, 0.14473                              # A, kcal/mol (A = 622716.4, B = 600.41)
        rh = np.array([2 ** (1 / 6) * sig / 2 * 0.1, 0.0, 0.0])
        se = np.array([np.sqrt(eps * 4.184), 0.0, 0.0])
        new = {id(m): dataclasses.replace(m, lj_rmin_half=rh, lj_sqrt_eps=se) for m in mols}
        mols = [new[id(m)] for m in mols]
    H = box_from_cell(*box) * 0.1
    n = a.replicate
    shifts = [i * H[0] + j * H[1] + k * H[2] for i in range(n) for j in range(n) for k in range(n)]
    pos = np.concatenate([xyz * 0.1 + s for s in shifts])
    v = np.concatenate([vel * 0.1] * len(shifts))
    sys_ = System(mols * len(shifts))
    st = MDSettings(cutoff=0.9, skin=0.1, ewald_beta=4.0, pme_grid=(48 * n,) * 3, pme_order=6, lj_lrc=True,
                    dipole_tol=a.tol, precision="mixed", elec=elec)
    kw = dict(settings=st, ensemble=a.ensemble, temperature=298.0, pressure=1.0, barostat_interval=100,
              dt=a.dt / 1000, log=sys.stdout, thermostat="bussi", tau_t=1.0, seed=a.seed)
    if a.engine == "rigid":
        sim = Simulation(sys_, pos, H * n, vel_nm_ps=v, **kw)
    else:
        from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate
        tpl = {id(m): RigidTemplate(m, pos[sys_.atom_slice(k)]) for k, m in enumerate(sys_.molecules)}
        sim = FlexibleSimulation(sys_, [tpl[id(m)] for m in sys_.molecules], pos, H * n, hmr=a.hmr, **kw)
    if a.checkpoint:
        sim.load(a.checkpoint)
    nsteps = int(round(a.ns * 1e6 / a.dt))
    nsteps -= nsteps % a.report
    sim.run(nsteps, report=a.report, restart=a.restart, prefix=a.out, dipoles=a.dipoles, append=bool(a.checkpoint))


if __name__ == "__main__":
    main()
