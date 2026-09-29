"""Check fitted bonded force fields in flexible gas-phase MD (fitted bonded terms + pGM, Langevin BAOAB, JAX).

The fitted class II cross terms are unbounded below far from equilibrium; this checks whether the
fitted force fields stay bounded in MD and how their bond/angle/torsion fluctuations compare with
the MACE-OFF frames at the nearest temperature (298 K test / 500 K training geometries).  A
replica counts as stable when its energies and coordinates stay finite, no bond deviates by
0.05 A or more from the reference mean and its energy stays above 20 kcal/mol below the lowest
reference frame.

Usage:

    python scripts/bonded/md_check.py NAME --mols A1,A2,A3 --families paper [--elec 3]
    python scripts/bonded/md_check.py NAME --temperature-K 298 --time-ps 20 --dt-fs 0.5
    python scripts/bonded/md_check.py --help

Inputs: the bonded-study data (pgm_jax.bonded.study.data: data/bonded/...).
Outputs: runs/bonded/results/<name>.json (per molecule: stable replicas, fluctuation ratios,
torsion histogram distance, equipartition ratios), a printed line per molecule.
Units: --temperature-K K, --time-ps ps, --dt-fs fs; energies kJ/mol, lengths nm in the model.
Runtime: minutes per molecule (fit + MD) on CPU.  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.bonded import terms as T
from pgm_jax.bonded.fit import Fitter
from pgm_jax.bonded.model import BondedModel, BondedSettings
from pgm_jax.bonded.study.data import frames, load, mol_list
from pgm_jax.bonded.study.families import families_of
from pgm_jax.bonded.study.gas_md import langevin
from pgm_jax.paths import repo_path
from pgm_jax.system import MASSES
from pgm_jax.units import KB, KCAL

jax.config.update("jax_enable_x64", True)


def internals(top, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bond lengths, angles and torsions of the frames X (F, N, 3) of a bonded topology."""
    G = jax.vmap(lambda R: T.geometry(R, top))(jnp.asarray(X))
    return np.asarray(G["b"]), np.asarray(G["th"]), np.asarray(G["phi"])


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, fit and run MD for each molecule and write the result (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="result name (runs/bonded/results/<name>.json)")
    ap.add_argument("--mols", default="A1,A2,A3", help="molecule sets or names (bonded.study.data.mol_list)")
    ap.add_argument("--families", default="paper", help="term families (bonded.study.families)")
    ap.add_argument("--elec", type=int, default=0, help="electrostatics excluded up to this bond separation")
    ap.add_argument("--elec14", type=float, default=1.0, help="1-4 electrostatic scale")
    ap.add_argument("--lj14", type=float, default=0.0, help="1-4 Lennard-Jones scale")
    ap.add_argument("--temperature-K", type=float, default=298.0, help="MD temperature [K]")
    ap.add_argument("--time-ps", type=float, default=20.0, help="MD length per replica [ps]")
    ap.add_argument("--dt-fs", type=float, default=0.5, help="time step [fs]")
    ap.add_argument("--nrep", type=int, default=4, help="replicas per molecule")
    ap.add_argument("--maxiter", type=int, default=20000, help="fit iterations")
    ap.add_argument("--l1", type=float, default=0.0, help="L1 weight of the fit")
    a = ap.parse_args(argv)
    temp, dt = a.temperature_K, a.dt_fs / 1000  # K, ps
    names = [n for n in mol_list(a.mols) if n != "methanethiol"]
    specs, data = load(names)
    st = BondedSettings(families=families_of(a.families), elec_exclude=a.elec, elec14_scale=a.elec14, lj14_scale=a.lj14)
    out = {"name": a.name, "args": vars(a), "molecules": {}}
    for i, spec in enumerate(specs):
        t0 = time.time()
        model = BondedModel([spec], st)
        fit = Fitter(model, {0: {"train": data[i]["train"], "test": data[i]["test"]}})
        P = fit.fit(model.init_params(), maxiter=a.maxiter, l1=a.l1, verbose=False)

        def efun(X, model=model, P=P):
            """Return the fitted energy [kJ/mol] of the frame X [nm]."""
            return model.energy(0, X, P)[0]

        X0 = jnp.asarray(spec.ref_xyz)
        nsteps = int(round(a.time_ps / dt)) // 100 * 100
        Xs, Es = langevin(
            efun, X0, [MASSES[e] for e in spec.elements], temp, dt, nsteps, 100, a.nrep, jax.random.PRNGKey(i)
        )
        Xs, Es = np.asarray(Xs), np.asarray(Es)
        top = model.mols[0].top
        finite = np.isfinite(Es).all(1) & np.isfinite(Xs).reshape(a.nrep, -1).all(1)
        # reference: the DFT-labelled MACE frames at the nearest temperature (298 K test / 500 K training MD)
        te = data[i]["test"] if temp < 400 else frames(spec.name, "train500")
        e_te = np.asarray(jax.vmap(efun)(jnp.asarray(te.X)))
        e_floor = e_te.min() - 20 * KCAL  # 20 kcal/mol below the lowest test frame
        b_ref, th_ref, phi_ref = internals(top, te.X)
        rec = {"finite": finite.tolist()}
        stable = []
        for r in range(a.nrep):
            if not finite[r]:
                stable.append(False)
                continue
            b, th, phi = internals(top, Xs[r])
            ok = bool(np.abs(b - b_ref.mean(0)).max() < 0.05 and Es[r].min() > e_floor)
            stable.append(ok)
        rec["stable"] = stable
        good = [r for r in range(a.nrep) if stable[r]]
        if good:
            X = Xs[good].reshape(-1, *Xs.shape[2:])
            b, th, phi = internals(top, X)
            rec["bond_std_ratio"] = float(np.median(b.std(0) / b_ref.std(0)))
            rec["angle_std_ratio"] = float(np.median(th.std(0) / th_ref.std(0)))
            # torsion distributions: circular mean |cos| difference of the populated wells
            # pooled torsion histogram (symmetry-equivalent wells of one rotor pooled), L1 distance in [0, 2]
            if phi.shape[1]:

                def h(p):
                    """Return the 24-bin torsion histogram of p [rad] (normalized to 1)."""
                    return np.histogram(p.ravel(), bins=24, range=(-np.pi, np.pi))[0] / p.size

                rec["torsion_hist_L1"] = float(np.abs(h(phi) - h(phi_ref)).sum())
            # equipartition: <E_pot> - E(minimum) in units of (3N - 6) kT / 2 (1 for a harmonic system)
            e_min = float(efun(X0))
            half = (3 * len(spec.elements) - 6) * KB * temp / 2
            rec["epot_equipartition"] = float((Es[good][:, Es.shape[1] // 5 :].mean() - e_min) / half)
            rec["epot_equipartition_mace"] = float((e_te.mean() - e_min) / half)
        else:
            rec["min_E_kcal_below_test"] = (
                float((Es[np.isfinite(Es)].min() - e_te.min()) / KCAL) if np.isfinite(Es).any() else None
            )
        rec["time_s"] = time.time() - t0
        out["molecules"][spec.name] = rec
        print(
            f"  {spec.name:20s} stable {sum(stable)}/{a.nrep}  "
            + "  ".join(f"{k} {v:.2f}" for k, v in rec.items() if isinstance(v, float)),
            flush=True,
        )
    os.makedirs(repo_path("runs", "bonded", "results"), exist_ok=True)
    with open(repo_path("runs", "bonded", "results", f"{a.name}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    n = sum(sum(v["stable"]) for v in out["molecules"].values())
    print(a.name, "stable replicas", n, "of", a.nrep * len(out["molecules"]), flush=True)


if __name__ == "__main__":
    main()
