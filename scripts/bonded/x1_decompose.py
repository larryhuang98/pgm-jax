"""X1: what the bonded terms must carry.  For each molecule, on the 298 K test frames:
spread of the residual E_DFT - E_nb (pGM all pairs vs classical 1-2/1-3/1-4 exclusion, LJ 1-5+),
the pGM force at the reference minimum, and the pGM dipole error (fixed charges and covalent
dipoles from the ESP fit).  -> runs/bonded/results/x1.json"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from pgm_jax.bonded.data import frames, mol_spec  # noqa: E402
from pgm_jax.bonded.fit import KCAL  # noqa: E402
from pgm_jax.bonded.model import BondedModel, BondedSettings  # noqa: E402
from pgm_jax.bonded.molecules import MOLECULES  # noqa: E402

out = {}
print(f"{'molecule':20s} {'sd resid pGM':>12s} {'sd resid cls':>12s} {'sd E_nb pGM':>11s} {'|F_nb| min pGM':>14s} {'cls':>6s} {'dip err D':>9s} {'|dip| D':>7s}")
for name in MOLECULES:
    te = frames(name, "test298")
    if te is None:
        continue
    spec = mol_spec(name)
    r = {}
    for tag, ex in (("pgm", 0), ("cls", 3)):
        model = BondedModel([spec], BondedSettings(families=("angle_cos",), elec_exclude=ex))
        e = jax.jit(jax.vmap(lambda X: model.nonbonded(0, X)[0]))(jnp.asarray(te.X))
        d = jax.jit(jax.vmap(lambda X: model.nonbonded(0, X)[1]))(jnp.asarray(te.X))
        g = jax.grad(lambda X: model.nonbonded(0, X)[0])(jnp.asarray(spec.ref_xyz))
        res = te.E - np.asarray(e)
        r[tag] = {"sd_resid": float(np.std(res)) / KCAL, "sd_Enb": float(np.std(e)) / KCAL, "sd_E": float(np.std(te.E)) / KCAL,
                  "F_min": float(np.mean(np.linalg.norm(np.asarray(g), axis=-1))) / (KCAL * 10),
                  "dip_rmse_D": float(np.sqrt(np.mean(np.sum((np.asarray(d) - te.mu) ** 2, -1)))) / 0.020819434,
                  "dip_mean_D": float(np.mean(np.linalg.norm(te.mu, axis=-1))) / 0.020819434}
    out[name] = r
    print(f"{name:20s} {r['pgm']['sd_resid']:12.2f} {r['cls']['sd_resid']:12.2f} {r['pgm']['sd_Enb']:11.2f} {r['pgm']['F_min']:14.2f} {r['cls']['F_min']:6.2f} {r['pgm']['dip_rmse_D']:9.3f} {r['pgm']['dip_mean_D']:7.2f}", flush=True)
os.makedirs(os.path.join(ROOT, "runs/bonded/results"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "runs/bonded/results/x1.json"), "w"), indent=1)
