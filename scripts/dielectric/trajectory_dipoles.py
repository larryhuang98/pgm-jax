"""Cell-dipole series of an Amber NetCDF trajectory of a pGM system (`pgm-jax trajectory-dipoles`).

For a trajectory of pmemd.pgm / pmemd.pgm.cuda (or any Amber NetCDF trajectory of a pGM system),
the model's induced dipoles are solved by pgm_jax at every frame and the cell dipole is written as
a .dip file for scripts/dielectric/dielectric.py.  The dielectric constant of a pmemd-pgm run then
gets the same treatment as pgm_jax's own runs (docs/dielectric.md): M = M_q + M_perm + M_ind,
eps_inf from the cell polarizability.

The electrostatic settings of the solve (cutoff, Ewald coefficient, PME grid) should be those of
the run (converted from Amber's Angstrom to nm: ee_dsum_cut 9 A is --cutoff-nm 0.9, ew_coeff
0.4/A is --ewald-beta-per-nm 4); the induced dipoles are converged to --dipole-tol from zero at the
first frame and from the previous frame's afterwards.  Molecules must be whole in the frames (pmemd's default iwrap = 0;
otherwise unwrap first, e.g. cpptraj `unwrap`).  --point-charges also prints eps from the point
charges of the prmtop's CHARGE section (what a charge-only analysis with cpptraj or MDAnalysis of
the same topology computes; for a pGM water box built by tleap these are TIP3P's charges, not the
model's) and their mean molecular dipole.

Usage:

    python scripts/dielectric/trajectory_dipoles.py water.prmtop md.nc [md2.nc ...] -o md.dip --nfft 96 96 96
    python scripts/dielectric/dielectric.py md.dip --skip-ps 300
    python scripts/dielectric/trajectory_dipoles.py --help

Inputs: the pGM prmtop of the run and its NetCDF trajectories (with a periodic box).
Outputs: the .dip file (-o); progress and, with --point-charges, the point-charge summary printed.
Units: --cutoff-nm and --pme-spacing-nm nm, --ewald-beta-per-nm 1/nm, --temperature-K K,
--skip-ps ps; dipoles e nm, volumes nm^3.
Runtime: GPU or CPU; one induced-dipole solve per frame.  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import netcdf_file

from pgm_jax.analysis import dielectric as D
from pgm_jax.cli.args import add_dipole_tol_arg, add_precision_arg, add_temperature_arg
from pgm_jax.md.box import (
    box_from_cell,
    reduce_box,
)
from pgm_jax.md.dipoles import DIP_COLUMNS, CellDipole
from pgm_jax.md.forcefield import MDSettings, PGMForceField
from pgm_jax.md.neighbors import AtomNeighbors, _failed
from pgm_jax.param import AMBER_CHARGE, read_prmtop_molecules
from pgm_jax.prmtop import Prmtop
from pgm_jax.system import System
from pgm_jax.units import DEBYE_E_NM

jax.config.update("jax_enable_x64", True)


def frames(paths: list[str]) -> Iterator[tuple[float, np.ndarray, np.ndarray]]:
    """Yield (time [ps], coordinates (N, 3) [nm], reduced box (3, 3) [nm]) of every frame of the files, in order.

    Raises
    ------
    ValueError
        A trajectory without a periodic box.
    """
    for p in paths:
        f = netcdf_file(p, "r", mmap=False)
        v = f.variables
        if "cell_lengths" not in v:
            raise ValueError(f"{p}: no periodic box in the trajectory")
        X, L, A, T = v["coordinates"], v["cell_lengths"][:], v["cell_angles"][:], v["time"][:]
        for i in range(X.shape[0]):
            yield float(T[i]), np.asarray(X[i], float) * 0.1, np.asarray(reduce_box(box_from_cell(L[i], A[i]) * 0.1))
        f.close()


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prmtop", help="pGM prmtop of the run")
    ap.add_argument("traj", nargs="+", help="Amber NetCDF trajectories, in order")
    ap.add_argument("-o", "--out", required=True, help=".dip output")
    ap.add_argument("--cutoff-nm", type=float, default=0.9, help="direct-space cutoff [nm] (Amber ee_dsum_cut / 10)")
    ap.add_argument(
        "--ewald-beta-per-nm", type=float, default=4.0, help="Ewald coefficient [1/nm] (Amber ew_coeff x 10)"
    )
    ap.add_argument("--nfft", type=int, nargs=3, default=None, help="PME grid (default: from --pme-spacing-nm)")
    ap.add_argument("--pme-spacing-nm", type=float, default=0.08, help="PME grid spacing [nm] (without --nfft)")
    ap.add_argument("--order", type=int, default=6, help="PME order")
    add_dipole_tol_arg(ap, 1e-6)
    add_precision_arg(ap)
    add_temperature_arg(ap, 298.0, help="the run's thermostat target [K] (written to the header)")
    ap.add_argument("--stride", type=int, default=1, help="use every stride-th frame")
    ap.add_argument("--alpha-every", type=int, default=100, help="frames between cell-polarizability evaluations")
    ap.add_argument("--point-charges", action="store_true", help="also eps and dipole from the prmtop CHARGE section")
    ap.add_argument("--skip-ps", type=float, default=0.0, help="time skipped by the --point-charges summary [ps]")
    return ap


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, solve the dipoles frame by frame and write the .dip file (see the module docstring)."""
    a = build_parser().parse_args(argv)

    mols = read_prmtop_molecules(a.prmtop)
    S = System(mols)
    it = frames(a.traj)
    t, pos, H = next(it)
    st = MDSettings().replace(
        cutoff=a.cutoff_nm,
        skin=0.1,
        ewald_beta=a.ewald_beta_per_nm,
        pme_order=a.order,
        pme_grid=tuple(a.nfft) if a.nfft else None,
        pme_spacing=a.pme_spacing_nm,
        dipole_tol=a.dipole_tol,
        max_iter=500,
        precision=a.precision,
        predictor="none",
        vdw="none",
    )
    ff = PGMForceField(S, H, st)
    nb = AtomNeighbors(S.n, H, st.pair_cutoff, st.neighbors.skin)
    cd = CellDipole(ff)
    Qk = cd.molecular_charges()
    mol = np.asarray(S.mol)
    qpc = np.asarray(Prmtop.read(a.prmtop).get("CHARGE"), float) / AMBER_CHARGE if a.point_charges else None
    if qpc is not None and len(qpc) != S.n:
        raise ValueError("CHARGE section and pGM atoms differ in number")
    if qpc is not None and np.any(np.abs(np.bincount(np.asarray(S.mol), weights=qpc)) > 1e-4):
        raise ValueError("--point-charges: the CHARGE section has charged molecules (dipole origin-dependent)")

    @jax.jit
    def solve(pos, H, nbr, ind):
        """Update the list, solve the dipoles; return (M components, mean |mu_mol|, induction, list, overflow)."""
        nbr = nb.update(nbr, pos, None, H, True)
        res = ff.compute(pos, H, nbr.idx, ind)
        mu = res.induction.mu
        mmol = jnp.mean(jnp.linalg.norm(cd.molecular(pos, H, mu), axis=1))
        return cd.components(pos, H, mu), mmol, res.induction, nbr, res.overflow

    @jax.jit
    def alpha(pos, H, nbr):
        """Isotropic cell polarizability tr(alpha_cell) / 3 [nm^3]."""
        return jnp.trace(cd.polarizability(pos, H, nbr.idx)) / 3.0

    # molecules must be whole: largest distance of an atom from its molecule's first atom
    first = np.searchsorted(mol, mol)
    ext = np.max(np.linalg.norm(pos - pos[first], axis=1))
    half = 0.5 * np.min(np.abs(np.diag(H)))
    if ext > min(half, 1.0):
        sys.exit(f"an atom is {ext:.3f} nm from its molecule's first atom: molecules are split (unwrap first)")
    meta = {
        "temperature_K": a.temperature_K,
        "ensemble": "npt",
        "thermostat": "external_trajectory",
        "dt_ps": float("nan"),
        "interval": a.stride,
        "n_atoms": S.n,
        "n_molecules": S.nmol,
        "net_charge": round(float(Qk.sum()), 6),
        "charged_molecules": int(np.sum(np.abs(Qk) > 1e-6)),
        "elec": st.terms.elec,
        "alpha_every": a.alpha_every,
    }
    head = [
        "pgm_jax cell dipole series (scripts/dielectric/trajectory_dipoles.py; scripts/dielectric/dielectric.py)",
        f"from {a.prmtop} and {', '.join(a.traj)}; induced dipoles solved by pgm_jax at every frame",
        "cell dipole M: M_q + M_perm + M_ind in e nm; mol_dipole: mean |dipole| of the molecules (e nm);",
        "alpha_nm3: cell electronic polarizability (every alpha_every-th sample, nan otherwise); temp_K: the target",
    ]
    head += [f"{k} = {v}" for k, v in meta.items()] + ["columns = " + " ".join(DIP_COLUMNS)]
    nbr = nb.allocate(jnp.asarray(pos), None, jnp.asarray(H))
    ind = ff.init_induction()
    pc, V, T = [], [], []
    t0, k = time.time(), 0
    with open(a.out, "w") as fh:
        fh.write("".join(f"# {s}\n" for s in head))
        first = (t, pos, H)
        for j, (t, pos, H) in enumerate(_chain(first, it)):
            if j % a.stride:
                continue
            P, HH = jnp.asarray(pos), jnp.asarray(H)
            comps, mmol, ind, nbr, ovf = solve(P, HH, nbr, ind)
            if _failed(nbr) or bool(ovf):
                nbr = nb.allocate(P, None, HH)
                comps, mmol, ind, nbr, ovf = solve(P, HH, nbr, ind)
                if _failed(nbr) or bool(ovf):
                    raise RuntimeError(f"frame {j}: neighbour list or rows overflow after reallocation")
            al = float(alpha(P, HH, nbr)) if k % a.alpha_every == 0 else float("nan")
            vol = abs(float(np.linalg.det(H)))
            m = np.asarray(comps).reshape(-1)
            fh.write(
                f"{j:10d} {t:14.6f} {a.temperature_K:9.3f} {vol:14.8f} "
                + " ".join(f"{x:17.10e}" for x in m)
                + f" {float(mmol):14.8e} {al:14.8e}\n"
            )
            if qpc is not None:
                pc.append(np.sum(qpc[:, None] * pos, axis=0))
                V.append(vol)
                T.append(t)
            k += 1
            if k % 500 == 0:
                print(f"# {k} frames, {time.time() - t0:.0f} s", flush=True)
    print(f"# {a.out}: {k} frames in {time.time() - t0:.0f} s")
    if qpc is not None:
        T, pc, V = np.array(T), np.array(pc), np.array(V)
        sel = T >= T[0] + a.skip_ps
        one = np.nonzero(mol == 0)[0]
        mu1 = np.linalg.norm(np.sum(qpc[one, None] * pos[one], axis=0)) / DEBYE_E_NM
        print(
            f"# prmtop CHARGE point charges (molecule 0: {np.round(qpc[one], 4).tolist()} e): dipole of molecule 0 "
            f"in the last frame {mu1:.4f} D; eps = 1 + fluctuation = "
            f"{1 + D.fluctuation(pc[sel], V[sel], a.temperature_K):.2f} ({sel.sum()} frames after {a.skip_ps:g} ps)"
        )


def _chain(first: object, rest: Iterable) -> Iterator:
    """Yield `first`, then the items of `rest`."""
    yield first
    yield from rest


if __name__ == "__main__":
    main()
