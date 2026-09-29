"""Flexible-molecule MD with fitted bonded terms + pGM (gas phase, Langevin BAOAB, jax).

The fitted class II cross terms are unbounded below far from equilibrium; this checks whether the
fitted force fields stay bounded in MD and how their 298 K bond/angle/torsion fluctuations compare
with the MACE-OFF frames at the same temperature (298 K test / 500 K training geometries).

    python scripts/bonded/md_check.py NAME --mols A1,A2,A3 --families paper [--elec 3] [--T 298 --ps 20]
"""

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
from pgm_jax.system import MASSES
from pgm_jax.units import KB, KCAL

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
jax.config.update("jax_enable_x64", True)


def internals(top, X):
    G = jax.vmap(lambda R: T.geometry(R, top))(jnp.asarray(X))
    return np.asarray(G["b"]), np.asarray(G["th"]), np.asarray(G["phi"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--mols", default="A1,A2,A3")
    ap.add_argument("--families", default="paper")
    ap.add_argument("--elec", type=int, default=0)
    ap.add_argument("--elec14", type=float, default=1.0)
    ap.add_argument("--lj14", type=float, default=0.0)
    ap.add_argument("--T", type=float, default=298.0)
    ap.add_argument("--ps", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.0005)
    ap.add_argument("--nrep", type=int, default=4)
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--l1", type=float, default=0.0)
    a = ap.parse_args()
    names = [n for n in mol_list(a.mols) if n != "methanethiol"]
    specs, data = load(names)
    st = BondedSettings(families=families_of(a.families), elec_exclude=a.elec, elec14_scale=a.elec14, lj14_scale=a.lj14)
    out = {"name": a.name, "args": vars(a), "molecules": {}}
    for i, spec in enumerate(specs):
        t0 = time.time()
        model = BondedModel([spec], st)
        fit = Fitter(model, {0: {"train": data[i]["train"], "test": data[i]["test"]}})
        P = fit.fit(model.init_params(), maxiter=a.maxiter, l1=a.l1, verbose=False)

        def efun(X):
            return model.energy(0, X, P)[0]

        X0 = jnp.asarray(spec.ref_xyz)
        nsteps = int(round(a.ps / a.dt)) // 100 * 100
        Xs, Es = langevin(
            efun, X0, [MASSES[e] for e in spec.elements], a.T, a.dt, nsteps, 100, a.nrep, jax.random.PRNGKey(i)
        )
        Xs, Es = np.asarray(Xs), np.asarray(Es)
        top = model.mols[0].top
        finite = np.isfinite(Es).all(1) & np.isfinite(Xs).reshape(a.nrep, -1).all(1)
        # reference: the DFT-labelled MACE frames at the nearest temperature (298 K test / 500 K training MD)
        te = data[i]["test"] if a.T < 400 else frames(spec.name, "train500")
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
                    return np.histogram(p.ravel(), bins=24, range=(-np.pi, np.pi))[0] / p.size

                rec["torsion_hist_L1"] = float(np.abs(h(phi) - h(phi_ref)).sum())
            # equipartition: <E_pot> - E(minimum) in units of (3N - 6) kT / 2 (1 for a harmonic system)
            e_min = float(efun(X0))
            half = (3 * len(spec.elements) - 6) * KB * a.T / 2
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
    os.makedirs(os.path.join(ROOT, "runs/bonded/results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "runs/bonded/results", f"{a.name}.json"), "w"), indent=1)
    n = sum(sum(v["stable"]) for v in out["molecules"].values())
    print(a.name, "stable replicas", n, "of", a.nrep * len(out["molecules"]), flush=True)


if __name__ == "__main__":
    main()
