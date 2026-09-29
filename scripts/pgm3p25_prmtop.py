"""pGM3P-25 water with its published geometry and Lennard-Jones as a pmemd-pgm topology (+ start
coordinates and mdin files), for runs with pmemd.pgm / pmemd.pgm.cuda.

    python scripts/pgm3p25_prmtop.py -o runs/p25/p25                  # 512 waters: p25.prmtop, p25.rst7
    python scripts/pgm3p25_prmtop.py -o runs/p25/p25_4096 --replicate 2 --mdin   # 4,096 waters (pmemd.pgm.cuda)

Source: ~/pgm-gvdw-data/topology/rayl_512_v2.prmtop and inputs/lj/inpcrd.restrt, which carry the
pGM3P-25 electrostatics (Wu et al., JCTC 21, 3563 (2025); github.com/yxwu21/pGM3P-25: charges,
covalent dipoles, Gaussian radii, polarizabilities) on TIP3P's geometry and Lennard-Jones.  Changed:
- OW-OW Lennard-Jones: A = 622716.376 kcal A^12/mol, B = 600.412 kcal A^6/mol (sigma 3.18156 A,
  epsilon 0.14473 kcal/mol);
- rigid geometry 0.9745 A, 103.64 deg: the O-H and H-H bond lengths SETTLE uses, and the hydrogens
  of the coordinates rebuilt about each oxygen (plane and bisector kept);
- CHARGE: 18.2223 x the pGM monopoles (e).  pmemd-pgm ignores CHARGE (it reads
  POL_GAUSS_MONOPOLES_LIST, in e), but analysis tools (cpptraj, MDAnalysis, ParmEd) read CHARGE:
  a tleap water topology keeps TIP3P's -0.834 / +0.417 e there, and a dipole computed from those
  point charges is not the pGM model's (it is 2.413 D for every molecule at this geometry).
The published README lists the monopoles divided by 18.2223 (-0.112 / +0.056); in e they are
-2.04056 / +1.02028, as in the POL_GAUSS_MONOPOLES_LIST section (unit e) pmemd-pgm runs.

--replicate n: n x n x n supercell with cpptraj (replicatecell) and scripts/pgm_supercell.py (pGM
sections); the periodic box, which cpptraj drops from the combined topology, is restored
(IFBOX, SOLVENT_POINTERS, ATOMS_PER_MOLECULE, BOX_DIMENSIONS).  pmemd.pgm.cuda needs at least three
neighbour-list cells across the box: the 512-water truncated octahedron (27.2 A) is too small for a
9 A cutoff, 2 x 2 x 2 is fine.  --mdin writes <out>.eq.in (100 ps NPT from the coordinates) and
<out>.md.in (NPT, Langevin 1/ps, 298 K, 1 bar, Monte Carlo barostat, SETTLE, 2 fs, frames every
1 ps; the settings of docs/dielectric.md).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import numpy as np
from water_dielectric import RST, TOP, paper_geometry

from pgm_jax.md.io import read_coordinates
from pgm_jax.paths import resource
from pgm_jax.prmtop import Prmtop

L_OH, THETA = 0.9745, 103.64  # A, deg
LJ_A, LJ_B = 622716.376, 600.412  # kcal A^12/mol, kcal A^6/mol
AMBER_CHARGE = 18.2223
CPPTRAJ = resource("pmemd_pgm_bin", "cpptraj")


def pgm3p25_topology(src: str = TOP) -> Prmtop:
    """rayl_512_v2.prmtop with pGM3P-25's Lennard-Jones, SETTLE lengths and a consistent CHARGE."""
    pt = Prmtop.read(src)
    names = pt.get("AMBER_ATOM_TYPE")
    if set(names) != {"OW", "HW"}:
        raise ValueError(f"{src}: expected a box of water only (atom types OW, HW), got {sorted(set(names))}")
    nt = pt.pointers["NTYPES"]
    tidx, nbi = pt.get("ATOM_TYPE_INDEX"), pt.get("NONBONDED_PARM_INDEX")
    to = int(tidx[names.index("OW")])
    k = int(nbi[(to - 1) * nt + to - 1]) - 1
    ac, bc = pt.get("LENNARD_JONES_ACOEF").copy(), pt.get("LENNARD_JONES_BCOEF").copy()
    ac[k], bc[k] = LJ_A, LJ_B
    pt.set("LENNARD_JONES_ACOEF", ac)
    pt.set("LENNARD_JONES_BCOEF", bc)
    beq = pt.get("BOND_EQUIL_VALUE").copy()
    l_hh = 2.0 * L_OH * np.sin(np.radians(THETA / 2.0))
    for i, j, t in pt.get("BONDS_INC_HYDROGEN").reshape(-1, 3):
        pair = (names[abs(int(i)) // 3], names[abs(int(j)) // 3])
        beq[int(t) - 1] = l_hh if pair == ("HW", "HW") else L_OH
    pt.set("BOND_EQUIL_VALUE", beq)
    pt.set("CHARGE", AMBER_CHARGE * np.asarray(pt.get("POL_GAUSS_MONOPOLES_LIST"), float))
    return pt


def write_rst7(path: str, xyz, lengths, angles, title: str = "pGM3P-25"):
    with open(path, "w") as fh:
        fh.write(f"{title}\n{len(xyz):6d}\n")
        flat = np.asarray(xyz, float).reshape(-1)
        for i in range(0, len(flat), 6):
            fh.write("".join(f"{v:12.7f}" for v in flat[i : i + 6]) + "\n")
        fh.write("".join(f"{v:12.7f}" for v in list(lengths) + list(angles)) + "\n")


def replicate(prmtop: str, rst7: str, out: str, n: int, lengths, angles):
    """n x n x n supercell of a water box (cpptraj + pgm_supercell), box sections restored."""
    import pgm_supercell

    dirs = " ".join(f"dir {i}{j}{k}" for i in range(n) for j in range(n) for k in range(n))
    with tempfile.TemporaryDirectory() as tmp:
        std = os.path.join(tmp, "std.prmtop")
        inp = f"parm {prmtop}\ntrajin {rst7}\nreplicatecell out {out}.rst7 parmout {std} {dirs}\nrun\n"
        r = subprocess.run([CPPTRAJ], input=inp, text=True, capture_output=True)
        if r.returncode != 0 or not os.path.exists(std):
            raise RuntimeError(f"cpptraj failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        pgm_supercell.main(prmtop, std, out + ".prmtop", n**3)
    pt = Prmtop.read(out + ".prmtop")
    nres = pt.pointers["NRES"]
    L = [float(x) * n for x in lengths]
    pt.set_pointers(IFBOX=2 if np.allclose(angles, 109.4712190, atol=1e-4) else 1)
    for name, vals, fmt, after in (
        ("SOLVENT_POINTERS", [0, nres, 1], "3I8", "IROTAT"),
        ("ATOMS_PER_MOLECULE", [3] * nres, "10I8", "SOLVENT_POINTERS"),
        ("BOX_DIMENSIONS", [float(angles[1])] + L, "5E16.8", "ATOMS_PER_MOLECULE"),
    ):
        pt.set(name, vals, fmt=None if name in pt else fmt, after=after)
    pt.write(out + ".prmtop")
    with open(out + ".rst7") as fh:  # cpptraj writes no box line without a box
        lines = fh.read().splitlines()
    natom = 3 * nres
    if len(lines) == 2 + (3 * natom + 5) // 6:
        with open(out + ".rst7", "a") as fh:
            fh.write("".join(f"{v:12.7f}" for v in L + list(angles)) + "\n")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", required=True, help="output prefix (<out>.prmtop, <out>.rst7)")
    ap.add_argument("--replicate", type=int, default=1, help="n x n x n copies of the 512-water box")
    ap.add_argument("--mdin", action="store_true", help="also write <out>.eq.in and <out>.md.in")
    ap.add_argument("--ns", type=float, default=3.0, help="length of the production run in <out>.md.in")
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    xyz, _, (lengths, angles) = read_coordinates(RST)
    x = paper_geometry(xyz, L_OH, THETA, [["O", "H", "H"]] * (len(xyz) // 3))
    one = a.out if a.replicate == 1 else a.out + "_512"
    pgm3p25_topology().write(one + ".prmtop")
    write_rst7(one + ".rst7", x, lengths, angles, "pGM3P-25, published geometry")
    L = list(lengths)
    if a.replicate > 1:
        L = replicate(one + ".prmtop", one + ".rst7", a.out, a.replicate, lengths, angles)
    print(
        f"{a.out}.prmtop / .rst7: {len(x) // 3 * a.replicate**3} waters, box {L[0]:.4f} A, angles {angles[0]:.4f} deg"
    )
    if a.mdin:
        from pgm_jax.md.box import box_from_cell
        from pgm_jax.md.forcefield import MDSettings
        from pgm_jax.protein.pmemd import pmemd_mdin

        H = box_from_cell(L, angles) * 0.1
        st = MDSettings(
            cutoff=0.9,
            skin=0.1,
            ewald_beta=4.0,
            pme_grid=(48 * a.replicate,) * 3,
            pme_order=6,
            lj_lrc=True,
            dipole_tol=1e-5,
        )
        kw = dict(dt=0.002, ensemble="npt", thermostat="langevin", gamma=1.0)  # pmemd's mdin (Amber names)
        open(a.out + ".eq.in", "w").write(pmemd_mdin(st, H, nstlim=50000, ntpr=5000, ntwr=50000, **kw))
        open(a.out + ".md.in", "w").write(
            pmemd_mdin(st, H, nstlim=int(round(a.ns * 5e5)), irest=1, ntpr=5000, ntwx=500, ntwr=50000, **kw)
        )
        print(
            f"{a.out}.eq.in, {a.out}.md.in: pmemd.pgm.cuda_SPFP -O -i {a.out}.eq.in -p {a.out}.prmtop "
            f"-c {a.out}.rst7 -r eq.rst7 ..."
        )


if __name__ == "__main__":
    main()
