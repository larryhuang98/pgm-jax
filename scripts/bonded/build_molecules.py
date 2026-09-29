"""Build the molecule set: data/bonded/molecules/<name>.json (elements, bonds, RDKit conformers)."""

import json
import os

from pgm_jax.bonded.study.molecules import MOLECULES, build

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out = os.path.join(ROOT, "data/bonded/molecules")
os.makedirs(out, exist_ok=True)
for name in MOLECULES:
    d = build(name)
    json.dump(d, open(os.path.join(out, f"{name}.json"), "w"), indent=1)
    print(
        f"{name:20s} {d['subset']} q={d['charge']:+d} atoms {len(d['elements']):2d} bonds {len(d['bonds']):2d} "
        f"conformers {len(d['conformers'])}"
    )
