"""Fit bonded terms for one molecule of the bonded data set and export a FlexibleTemplate for MD.

    python examples/fit_bonded_template.py methanol                      # class II set (T.PAPER)
    python examples/fit_bonded_template.py ethanol --families diag+ub --out runs/flex/ethanol_ub.flex
    python examples/fit_bonded_template.py methanol --flux 1 --wmu 1 --maxiter 4000   # + charge flux

The molecule needs data/bonded/molecules/<name>.json (topology), data/bonded/params/<name>.json
(pGM charges, covalent dipoles, polarizabilities and LJ) and DFT-labelled frames
(data/bonded/dft/<name>__train500__*.npz, ...__test298__*.npz); docs/howto_bonded.md explains how
to make them for a new molecule.  The fit uses pGM electrostatics with all pairs and LJ from 1-5
pairs on, i.e. exactly the model the MD engine runs, so the template is MD-ready.  With --flux the
charges and covalent dipoles depend on the bond lengths (BondedSettings.flux; md/flux.py runs it in
MD); --wmu weights the gas-phase dipoles in the loss, which the flux parameters mostly affect."""
import argparse, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts/bonded"))
import jax
jax.config.update("jax_enable_x64", True)
from experiments import FAMILY_SETS, load
from pgm_jax.bonded import terms as T
from pgm_jax.bonded.fit import Fitter
from pgm_jax.bonded.model import BondedModel, BondedSettings
from pgm_jax.md.flexible import FlexibleTemplate

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--families", default="paper", help="a key of FAMILY_SETS or families joined by '+'")
ap.add_argument("--lj14", type=float, default=0.0, help="scale of 1-4 LJ (0: LJ from 1-5 pairs on only)")
ap.add_argument("--maxiter", type=int, default=20000)
ap.add_argument("--flux", type=int, default=0, help="1: charge + covalent-dipole flux; 2: + quadratic dipole flux")
ap.add_argument("--wmu", type=float, default=0.0, help="weight of the molecular dipoles in the loss")
ap.add_argument("--out", default="")
a = ap.parse_args()

fams = FAMILY_SETS.get(a.families, tuple(a.families.split("+")))
unknown = [f for f in fams if f not in T.REGISTRY]
if unknown:
    raise SystemExit(f"unknown families {unknown}; available: {sorted(T.REGISTRY)}")
specs, data = load([a.name])
if not specs:
    raise SystemExit(f"no DFT frames for {a.name}")
model = BondedModel(specs, BondedSettings(families=fams, lj14_scale=a.lj14, flux=a.flux))
fit = Fitter(model, {0: {"train": data[0]["train"], "test": data[0]["test"]}}, w_mu=a.wmu)
t0 = time.time()
P = fit.fit(model.init_params(), maxiter=a.maxiter, verbose=False)
m = fit.metrics(P, "test")[0]
print(f"{a.name}: {len(fams)} families, fit {time.time() - t0:.0f} s; 298 K test frames: "
      + ", ".join(f"{k} {v:.3f}" for k, v in m.items() if isinstance(v, float)))
out = a.out or os.path.join(ROOT, f"runs/flex/{a.name}.flex")
os.makedirs(os.path.dirname(out), exist_ok=True)
FlexibleTemplate.from_fit(model, P).save(out)
print("template:", out)
