"""Sample the bonded-study molecules with MACE-OFF: minima, MD frames and relaxed torsion scans.

MACE-OFF plays the role xTB plays in Abdullah et al.: minima (BFGS from the conformers of
data/bonded/molecules/<name>.json, duplicates by energy dropped), Langevin MD frames at 500 K
(training) and 298 K (test) started from each minimum, relaxed 1D scans (15 degree grid) of one
torsion per rotatable bond, and for alanine dipeptide a relaxed 2D phi/psi scan in row blocks.
Energies are relabelled with DFT later (scripts/bonded/make_dft_tasks.py).

Usage:

    python scripts/bonded/mace_sample.py NAME [--device cpu] [--what md,scan]
    python scripts/bonded/mace_sample.py alanine_dipeptide --what scan2d --rows 0:4
    python scripts/bonded/mace_sample.py --help

Inputs: data/bonded/molecules/<name>.json, data/bonded/mace/MACE-OFF23_medium.model.
Outputs: data/bonded/frames/<name>_md.npz (minima, train500, test298), <name>_scan.npz,
<name>_scan2d_<a>_<b>.npz.
Units: coordinates [A], MACE energies [eV], angles [degree], MD time step 0.5 fs, friction 0.01/fs.
Runtime: CPU, minutes (md) to hours (scan2d).  Needs torch, ASE and mace-torch.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from pgm_jax.paths import repo_path

EV_KCAL = 23.0605  # kcal/mol per eV (the rounded factor 23.06 is used in two progress lines)


class Sampler:
    """MACE-OFF calculator, topology and random generator of one molecule.

    Parameters
    ----------
    d : dict
        The molecule record (elements, bonds, bond_orders, conformers).
    calc : ase calculator
        MACE-OFF.
    seed : int
        Seed of the MD random generator.
    """

    def __init__(self, d: dict, calc, seed: int):
        """Store the molecule, build its neighbour lists and seed the random generator."""
        self.d = d
        self.calc = calc
        self.el = d["elements"]
        self.bonds = [tuple(b) for b in d["bonds"]]
        self.nbr = {i: set() for i in range(len(self.el))}
        for i, j in self.bonds:
            self.nbr[i].add(j)
            self.nbr[j].add(i)
        self.rng = np.random.default_rng(seed)

    def atoms_of(self, x):
        """Return ASE Atoms at the positions x [A] with the MACE calculator attached."""
        from ase import Atoms

        at = Atoms(self.el, positions=np.asarray(x))
        at.calc = self.calc
        return at

    @staticmethod
    def relax(at, fmax: float = 0.005, steps: int = 2000):
        """Relax at in place with BFGS (maximum step 0.04 A) to fmax [eV/A]; return it."""
        from ase.optimize import BFGS

        opt = BFGS(at, logfile=None, maxstep=0.04)
        opt.run(fmax=fmax, steps=steps)
        return at

    def side(self, b: int, c: int) -> np.ndarray:
        """Return the mask of the atoms on c's side of bond b-c (moved when the dihedral about b-c is set)."""
        seen, stack = {c}, [c]
        while stack:
            u = stack.pop()
            for v in self.nbr[u]:
                if v not in seen and not (u == c and v == b):
                    seen.add(v)
                    stack.append(v)
        mask = np.zeros(len(self.el), bool)
        mask[list(seen)] = True
        return mask

    @staticmethod
    def fmax_of(at) -> float:
        """Return the largest atomic force norm [eV/A] of at."""
        return float(np.max(np.linalg.norm(at.get_forces(), axis=1)))

    def rotatable_torsions(self) -> list[tuple[int, int, int, int]]:
        """Return one torsion per rotatable bond (single, not in a ring, both ends with other neighbours).

        Heavy atoms are preferred at the ends (then the lowest index).
        """
        el, nbr = self.el, self.nbr
        out = []
        ring = set()
        # ring bonds: bond whose removal keeps its ends connected
        for i, j in self.bonds:
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
        for k, (j, kk) in enumerate(self.bonds):
            if (j, kk) in ring or self.d["bond_orders"][k] != 1.0:
                continue
            a_ = [x for x in nbr[j] if x != kk]
            b_ = [x for x in nbr[kk] if x != j]
            if not a_ or not b_:
                continue

            def pick(c):
                """Return the end atom: heavy atoms first, then the lowest index."""
                return sorted(c, key=lambda x: (el[x] == "H", x))[0]

            out.append((pick(a_), j, kk, pick(b_)))
        return out

    def md(self, at, T: float, n_frames: int, every: int, equil: int, dt: float = 0.5):
        """Run Langevin MD and return frames X (n_frames, N, 3) [A] and energies E [eV].

        Parameters
        ----------
        at : ase.Atoms
            Start structure (velocities drawn at T, centre-of-mass motion and rotation removed).
        T : float
            Temperature [K].
        n_frames, every, equil : int
            Frames, steps between frames and equilibration steps.
        dt : float
            Time step [fs] (friction 0.01/fs).
        """
        from ase import units
        from ase.md.langevin import Langevin
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

        MaxwellBoltzmannDistribution(at, temperature_K=T, rng=self.rng)
        Stationary(at)
        ZeroRotation(at)
        dyn = Langevin(at, dt * units.fs, temperature_K=T, friction=0.01 / units.fs, rng=self.rng)
        dyn.run(equil)
        X, E = [], []
        for _ in range(n_frames):
            dyn.run(every)
            X.append(at.get_positions().copy())
            E.append(at.get_potential_energy())
        return np.array(X), np.array(E)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="molecule (data/bonded/molecules/<name>.json)")
    ap.add_argument("--device", default="cpu", help="torch device of MACE")
    ap.add_argument("--what", default="md,scan", help="comma list of md, scan, scan2d")
    ap.add_argument("--rows", default=None, help="scan2d: phi rows a:b of the 24-point grid")
    ap.add_argument("--n_train", type=int, default=200, help="500 K training frames (over all minima)")
    ap.add_argument("--n_test", type=int, default=400, help="298 K test frames (over all minima)")
    ap.add_argument("--threads", type=int, default=4, help="torch threads")
    ap.add_argument("--seed", type=int, default=0, help="seed of the MD random generator")
    return ap


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and write the requested frame files (see the module docstring)."""
    a = build_parser().parse_args(argv)
    import torch
    from ase.constraints import FixInternals
    from mace.calculators import mace_off

    torch.set_num_threads(a.threads)
    with open(repo_path("data", "bonded", "molecules", f"{a.name}.json")) as fh:
        d = json.load(fh)
    out_dir = repo_path("data", "bonded", "frames")
    os.makedirs(out_dir, exist_ok=True)
    calc = mace_off(
        model=repo_path("data", "bonded", "mace", "MACE-OFF23_medium.model"), device=a.device, default_dtype="float64"
    )
    S = Sampler(d, calc, a.seed)
    el = S.el

    t0 = time.time()
    minima = [S.relax(S.atoms_of(x)) for x in d["conformers"]]
    Emin = np.array([m.get_potential_energy() for m in minima])
    # drop duplicate minima (same energy within 1e-4 eV after relaxation)
    uniq = []
    for i in np.argsort(Emin):
        if all(abs(Emin[i] - Emin[j]) > 1e-4 for j in uniq):
            uniq.append(int(i))
    minima = [minima[i] for i in uniq]
    Xmin = np.array([m.get_positions() for m in minima])
    print(
        f"{a.name}: {len(minima)} minima, E rel (kcal/mol) {np.round((Emin[uniq] - Emin[uniq].min()) * EV_KCAL, 2)}",
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
                X, E = S.md(S.atoms_of(m.get_positions()), T, per[k], every, 2000)
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
        tors = S.rotatable_torsions()
        out = {"elements": np.array(el), "torsions": np.array(tors, int).reshape(-1, 4)}
        grid = np.arange(-180.0, 180.0, 15.0)
        for t, dih in enumerate(tors):
            X, E, FM = [], [], []
            at = S.atoms_of(Xmin[0])
            mask = S.side(dih[1], dih[2])
            for phi in grid:
                at.set_constraint()
                at.set_dihedral(*dih, phi, mask=mask)
                at.set_constraint(FixInternals(dihedrals_deg=[[phi, list(dih)]]))
                S.relax(at, fmax=0.005, steps=3000)
                X.append(at.get_positions().copy())
                E.append(at.get_potential_energy())
                FM.append(S.fmax_of(at))
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
        m_phi, m_psi = S.side(phi_idx[1], phi_idx[2]), S.side(psi_idx[1], psi_idx[2])
        for p in grid[lo:hi]:
            at = S.atoms_of(Xmin[0])
            for s in grid:
                at.set_constraint()
                at.set_dihedral(*phi_idx, p, mask=m_phi)
                at.set_dihedral(*psi_idx, s, mask=m_psi)
                at.set_constraint(FixInternals(dihedrals_deg=[[p, phi_idx], [s, psi_idx]]))
                S.relax(at, fmax=0.005, steps=3000)
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


if __name__ == "__main__":
    main()
