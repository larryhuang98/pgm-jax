"""MACE-OFF minimum of every molecule (lowest over the RDKit conformers): data/bonded/frames/<name>_min.npz."""
import json, os, sys
import numpy as np, torch
from ase import Atoms
from ase.optimize import BFGS
from mace.calculators import mace_off
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(2)
calc = mace_off(model=os.path.join(ROOT, "data/bonded/mace/MACE-OFF23_medium.model"), device="cpu", default_dtype="float64")
for name in sys.argv[1:]:
    d = json.load(open(os.path.join(ROOT, "data/bonded/molecules", f"{name}.json")))
    best = None
    for x in d["conformers"]:
        at = Atoms(d["elements"], positions=np.array(x)); at.calc = calc
        BFGS(at, logfile=None).run(fmax=0.005, steps=2000)
        e = at.get_potential_energy()
        if best is None or e < best[0]:
            best = (e, at.get_positions().copy())
    np.savez(os.path.join(ROOT, "data/bonded/frames", f"{name}_min.npz"), minima=best[1][None], minima_E=np.array([best[0]]))
    print(name, best[0], flush=True)
