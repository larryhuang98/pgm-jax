"""Sampling with MACE-OFF (the role xTB plays in Abdullah et al.): minima, Langevin MD frames
at 500 K (training) and 298 K (test), relaxed torsion scans.  Labels come from DFT later.

    python scripts/bonded/mace_sample.py NAME [--device cpu] [--what md,scan]
    python scripts/bonded/mace_sample.py alanine_dipeptide --what scan2d --rows 0:4
Writes data/bonded/frames/<name>_<what>.npz (coordinates in Angstrom, MACE energies eV)."""

import argparse
import json
import os
import time

import numpy as np
import torch
from ase import Atoms, units
from ase.constraints import FixInternals
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS
from mace.calculators import mace_off

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--device", default="cpu")
ap.add_argument("--what", default="md,scan")
ap.add_argument("--rows", default=None, help="scan2d: phi rows a:b")
ap.add_argument("--n_train", type=int, default=200)
ap.add_argument("--n_test", type=int, default=400)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.set_num_threads(a.threads)
d = json.load(open(os.path.join(ROOT, "data/bonded/molecules", f"{a.name}.json")))
out_dir = os.path.join(ROOT, "data/bonded/frames")
os.makedirs(out_dir, exist_ok=True)
calc = mace_off(
    model=os.path.join(ROOT, "data/bonded/mace/MACE-OFF23_medium.model"), device=a.device, default_dtype="float64"
)
el = d["elements"]
bonds = [tuple(b) for b in d["bonds"]]
nbr = {i: set() for i in range(len(el))}
for i, j in bonds:
    nbr[i].add(j)
    nbr[j].add(i)
rng = np.random.default_rng(a.seed)


def atoms_of(x):
    at = Atoms(el, positions=np.asarray(x))
    at.calc = calc
    return at


def relax(at, fmax=0.005, steps=2000):
    opt = BFGS(at, logfile=None, maxstep=0.04)
    opt.run(fmax=fmax, steps=steps)
    return at


def side(b, c):
    """Atoms on c's side of bond b-c (moved when the dihedral about b-c is set)."""
    seen, stack = {c}, [c]
    while stack:
        u = stack.pop()
        for v in nbr[u]:
            if v not in seen and not (u == c and v == b):
                seen.add(v)
                stack.append(v)
    mask = np.zeros(len(el), bool)
    mask[list(seen)] = True
    return mask


def fmax_of(at):
    return float(np.max(np.linalg.norm(at.get_forces(), axis=1)))


def rotatable_torsions():
    """One torsion per rotatable bond (single, not in a ring, both ends with other neighbours);
    heavy atoms preferred at the ends."""
    import networkx  # noqa: F401  (ASE depends on it)

    out = []
    ring = set()
    # ring bonds: bond whose removal keeps its ends connected
    for i, j in bonds:
        seen, stack = {i}, [i]
        while stack:
            u = stack.pop()
            for v in nbr[u]:
                if (u, v) in ((i, j), (j, i)) or v in seen:
                    continue
                seen.add(v)
                stack.append(v)
        if j in seen:
            ring.add((i, j))
    for k, (j, kk) in enumerate(bonds):
        if (j, kk) in ring or d["bond_orders"][k] != 1.0:
            continue
        a_ = [x for x in nbr[j] if x != kk]
        b_ = [x for x in nbr[kk] if x != j]
        if not a_ or not b_:
            continue

        def pick(c):
            return sorted(c, key=lambda x: (el[x] == "H", x))[0]

        out.append((pick(a_), j, kk, pick(b_)))
    return out


def md(at, T, n_frames, every, equil, dt=0.5):
    MaxwellBoltzmannDistribution(at, temperature_K=T, rng=rng)
    Stationary(at)
    ZeroRotation(at)
    dyn = Langevin(at, dt * units.fs, temperature_K=T, friction=0.01 / units.fs, rng=rng)
    dyn.run(equil)
    X, E = [], []
    for _ in range(n_frames):
        dyn.run(every)
        X.append(at.get_positions().copy())
        E.append(at.get_potential_energy())
    return np.array(X), np.array(E)


t0 = time.time()
minima = [relax(atoms_of(x)) for x in d["conformers"]]
Emin = np.array([m.get_potential_energy() for m in minima])
# drop duplicate minima (same energy within 1e-4 eV and RMSD < 0.05 A after relaxation)
uniq = []
for i in np.argsort(Emin):
    if all(abs(Emin[i] - Emin[j]) > 1e-4 for j in uniq):
        uniq.append(int(i))
minima = [minima[i] for i in uniq]
Xmin = np.array([m.get_positions() for m in minima])
print(
    f"{a.name}: {len(minima)} minima, E rel (kcal/mol) {np.round((Emin[uniq] - Emin[uniq].min()) * 23.0605, 2)}",
    flush=True,
)

if a.what in ("scan", "scan2d") and os.path.exists(os.path.join(out_dir, f"{a.name}_md.npz")):
    z = np.load(os.path.join(out_dir, f"{a.name}_md.npz"))  # reuse the minima of the MD run
    Xmin = z["minima"]

if "md" in a.what:
    out = {"elements": np.array(el), "minima": Xmin, "minima_E": Emin[uniq]}
    for tag, T, n, every in (("train500", 500.0, a.n_train, 100), ("test298", 298.0, a.n_test, 50)):
        per = [n // len(minima) + (1 if k < n % len(minima) else 0) for k in range(len(minima))]
        Xs, Es, C = [], [], []
        for k, m in enumerate(minima):
            X, E = md(atoms_of(m.get_positions()), T, per[k], every, 2000)
            Xs.append(X)
            Es.append(E)
            C += [k] * len(X)
        out[tag] = np.concatenate(Xs)
        out[tag + "_E"] = np.concatenate(Es)
        out[tag + "_conf"] = np.array(C)
        print(
            f"  {tag}: {len(out[tag])} frames, E range {np.ptp(out[tag + '_E']) * 23.06:.1f} kcal/mol, "
            f"{time.time() - t0:.0f} s",
            flush=True,
        )
    np.savez(os.path.join(out_dir, f"{a.name}_md.npz"), **out)

if "scan" in a.what and "scan2d" not in a.what:
    tors = rotatable_torsions()
    out = {"elements": np.array(el), "torsions": np.array(tors, int).reshape(-1, 4)}
    grid = np.arange(-180.0, 180.0, 15.0)
    for t, dih in enumerate(tors):
        X, E, FM = [], [], []
        at = atoms_of(Xmin[0])
        mask = side(dih[1], dih[2])
        for phi in grid:
            at.set_constraint()
            at.set_dihedral(*dih, phi, mask=mask)
            at.set_constraint(FixInternals(dihedrals_deg=[[phi, list(dih)]]))
            relax(at, fmax=0.005, steps=3000)
            X.append(at.get_positions().copy())
            E.append(at.get_potential_energy())
            FM.append(fmax_of(at))
        at.set_constraint()
        out[f"scan{t}"] = np.array(X)
        out[f"scan{t}_E"] = np.array(E)
        out[f"scan{t}_angle"] = grid
        out[f"scan{t}_fmax"] = np.array(FM)
        print(
            f"  scan {dih}: barrier {np.ptp(E) * 23.06:.2f} kcal/mol, max residual force {max(FM):.3f} eV/A "
            f"(constraint direction included), {time.time() - t0:.0f} s",
            flush=True,
        )
    np.savez(os.path.join(out_dir, f"{a.name}_scan.npz"), **out)

if "scan2d" in a.what:
    # alanine dipeptide: phi = C(0-1)-N-CA-C, psi = N-CA-C-N from the SMILES order CC(=O)N[C@@H](C)C(=O)NC
    phi_idx, psi_idx = [1, 3, 4, 6], [3, 4, 6, 8]
    grid = np.arange(-180.0, 180.0, 15.0)
    lo, hi = (int(v) for v in a.rows.split(":"))
    X, E, ang = [], [], []
    m_phi, m_psi = side(phi_idx[1], phi_idx[2]), side(psi_idx[1], psi_idx[2])
    for p in grid[lo:hi]:
        at = atoms_of(Xmin[0])
        for s in grid:
            at.set_constraint()
            at.set_dihedral(*phi_idx, p, mask=m_phi)
            at.set_dihedral(*psi_idx, s, mask=m_psi)
            at.set_constraint(FixInternals(dihedrals_deg=[[p, phi_idx], [s, psi_idx]]))
            relax(at, fmax=0.005, steps=3000)
            X.append(at.get_positions().copy())
            E.append(at.get_potential_energy())
            ang.append((p, s))
        print(f"  phi {p}: {time.time() - t0:.0f} s", flush=True)
    np.savez(
        os.path.join(out_dir, f"{a.name}_scan2d_{lo}_{hi}.npz"),
        elements=np.array(el),
        X=np.array(X),
        E=np.array(E),
        angles=np.array(ang),
    )
print(f"done {a.name} {time.time() - t0:.0f} s", flush=True)
