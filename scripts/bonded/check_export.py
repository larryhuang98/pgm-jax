"""Check the Amber import / export of the protein term set against sander.

ACE-ALA-ALA-NME with ff19SB (tleap): (1) the model initialised from the prmtop (init_from_prmtop,
CMAP grids -> Fourier maps) against sander on that prmtop; (2) random per-instance parameters
exported into the prmtop (export_bonded) against sander on the exported file: BOND, ANGLE, DIHED
(up to the constant of the sign convention), CMAP (Fourier map vs Amber's bicubic interpolation
of its tabulation), and 1-4 VDW / 1-4 EEL unchanged (one dihedral per 1-4 pair carries them).
Needs AmberTools (tleap, sander).  Writes validation/check_export.json.

    python scripts/bonded/check_export.py
"""

import json
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from pgm_jax.bonded import terms as T  # noqa: E402
from pgm_jax.bonded.amber import export_bonded, init_from_prmtop, with_amber_impropers  # noqa: E402
from pgm_jax.bonded.model import BondedSettings, BondedTerms, MolSpec  # noqa: E402
from pgm_jax.prmtop import Prmtop  # noqa: E402
from pgm_jax.units import KCAL

WD = os.path.join(ROOT, "runs/check_export")
os.makedirs(WD, exist_ok=True)
EL = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}


def sh(cmd):
    subprocess.run(["bash", "-lc", cmd], cwd=WD, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def tleap(seq="ACE ALA ALA NME"):
    open(os.path.join(WD, "leap.in"), "w").write(
        f"source leaprc.protein.ff19SB\nm = sequence {{ {seq} }}\nsaveamberparm m pep.prmtop pep.inpcrd\nquit\n"
    )
    sh("tleap -f leap.in")


def read_inpcrd(path):
    lines = open(path).read().split("\n")
    n = int(lines[1].split()[0])
    vals = [float(l[i : i + 12]) for l in lines[2:] for i in range(0, len(l), 12) if l[i : i + 12].strip()]
    return np.array(vals[: 3 * n]).reshape(n, 3)


def write_inpcrd(path, X):
    with open(path, "w") as fh:
        fh.write(f"perturbed\n{len(X):6d}\n")
        flat = X.ravel()
        for i in range(0, len(flat), 6):
            fh.write("".join(f"{v:12.7f}" for v in flat[i : i + 6]) + "\n")


def sander(prmtop, crd):
    open(os.path.join(WD, "sp.in"), "w").write(
        "single point\n &cntrl\n  imin=1, maxcyc=0, ntb=0, igb=0, cut=999.0, ntpr=1,\n /\n"
    )
    sh(f"sander -O -i sp.in -p {prmtop} -c {crd} -o sp.out")
    txt = open(os.path.join(WD, "sp.out")).read()
    txt = txt[txt.index("NSTEP       ENERGY") :]

    def get(name):
        return float(re.search(re.escape(name) + r"\s*=\s*(-?[\d.]+)", txt).group(1))

    return {k: get(k) for k in ("BOND", "ANGLE", "DIHED", "CMAP", "1-4 VDW", "1-4 EEL")}


def spec_from_prmtop(prmtop, crd):
    pt = Prmtop.read(prmtop)
    el = [EL[int(z)] for z in pt.get("ATOMIC_NUMBER")]
    B = np.concatenate([pt.get("BONDS_INC_HYDROGEN"), pt.get("BONDS_WITHOUT_HYDROGEN")]).reshape(-1, 3)[:, :2] // 3
    bonds = [tuple(int(x) for x in b) for b in B]
    deg = np.bincount(np.asarray(bonds).ravel(), minlength=len(el))
    orders = [2.0 if {el[i], el[j]} == {"C", "O"} and deg[i if el[i] == "O" else j] == 1 else 1.0 for i, j in bonds]
    return MolSpec("pep", el, bonds, orders, 0, read_inpcrd(crd) * 0.1)


def ours(terms, P, R):
    """Per-term energies (kcal/mol) in Amber's convention (torsion constants included)."""
    P = jax.tree_util.tree_map(jnp.asarray, P)
    out = {}
    cf = [f for f in terms.fams if f.startswith("cmap")][0]
    for f, name in (("bond_harm", "BOND"), ("angle_harm", "ANGLE"), (cf, "CMAP")):
        Q = jax.tree_util.tree_map(jnp.zeros_like, P)
        Q["ref"] = P["ref"]
        Q[f] = P[f]
        out[name] = float(terms.bonded_energy(0, R, Q)) / KCAL
    Q = jax.tree_util.tree_map(jnp.zeros_like, P)
    Q["ref"] = P["ref"]
    Q["torsion_amber"] = P["torsion_amber"]
    Q["improper_amber"] = P["improper_amber"]
    Im = terms.I[0]
    const = 2 * np.sum(
        np.abs(np.minimum(np.asarray(P["torsion_amber"]["K"])[Im["torsion_amber"]["k"]], 0))
    ) + 2 * np.sum(np.abs(np.minimum(np.asarray(P["improper_amber"]["K"])[Im["improper_amber"]["k"]], 0)))
    out["DIHED"] = (float(terms.bonded_energy(0, R, Q)) + const) / KCAL
    return out


tleap()
prm, crd = os.path.join(WD, "pep.prmtop"), os.path.join(WD, "pep.inpcrd")
spec = with_amber_impropers(spec_from_prmtop(prm, crd), prm)
terms = BondedTerms([spec], BondedSettings(families=T.PROTEIN, lj14_scale=0.5))
P = init_from_prmtop(terms, terms.init_params(), {0: prm})
rng = np.random.default_rng(0)
X = spec.ref_xyz + 0.01 * rng.normal(size=spec.ref_xyz.shape)
write_inpcrd(os.path.join(WD, "pert.inpcrd"), X * 10.0)
R = jnp.asarray(X)
res = {"import": {"sander": sander("pep.prmtop", "pert.inpcrd"), "ours": ours(terms, P, R)}}
# ff19SB's maps are rougher than an order-3 series: the order-6 family gets closer
t6 = BondedTerms([spec], BondedSettings(families=T.AMBER + ("cmap6",), lj14_scale=0.5))
res["import_cmap6"] = {
    "sander": res["import"]["sander"],
    "ours": ours(t6, init_from_prmtop(t6, t6.init_params(), {0: prm}), R),
}
# random per-instance changes, exported
Q = jax.tree_util.tree_map(np.asarray, P)
Q["bond_harm"]["Kb"] = Q["bond_harm"]["Kb"] * (1 + 0.2 * rng.normal(size=Q["bond_harm"]["Kb"].shape))
Q["angle_harm"]["Ka"] = Q["angle_harm"]["Ka"] * (1 + 0.2 * rng.normal(size=Q["angle_harm"]["Ka"].shape))
Q["ref"]["b0"] = Q["ref"]["b0"] + 0.002 * rng.normal(size=Q["ref"]["b0"].shape)
Q["ref"]["th0"] = Q["ref"]["th0"] + 0.03 * rng.normal(size=Q["ref"]["th0"].shape)
Q["torsion_amber"]["K"] = Q["torsion_amber"]["K"] + 2.0 * rng.normal(size=Q["torsion_amber"]["K"].shape)
Q["improper_amber"]["K"] = Q["improper_amber"]["K"] + 5.0 * rng.normal(size=Q["improper_amber"]["K"].shape)
Q["cmap"]["cm"] = Q["cmap"]["cm"] + 1.0 * rng.normal(size=Q["cmap"]["cm"].shape)
counts = export_bonded(prm, os.path.join(WD, "exported.prmtop"), terms, Q, scnb=2.0)
res["export"] = {"sander": sander("exported.prmtop", "pert.inpcrd"), "ours": ours(terms, Q, R), "counts": counts}
# the Fourier maps have no constant term: the imported maps are the grids minus their means
from pgm_jax.bonded.amber import read_bonded  # noqa: E402

amb = read_bonded(prm)
const = float(sum(np.mean(amb["cmap_grids"][t]) for *_, t in amb["cmap"]))
for k in ("import", "import_cmap6"):
    res[k]["ours"]["CMAP"] += const
res["cmap_constant_kcal"] = const
for k in ("import", "import_cmap6", "export"):
    s_, o_ = res[k]["sander"], res[k]["ours"]
    res[k]["diff"] = {n: s_[n] - o_[n] for n in o_}
    print(k, {n: (round(s_[n], 4), round(o_[n], 4)) for n in o_})
print(
    "1-4 VDW / EEL import vs export:",
    res["import"]["sander"]["1-4 VDW"],
    res["export"]["sander"]["1-4 VDW"],
    res["import"]["sander"]["1-4 EEL"],
    res["export"]["sander"]["1-4 EEL"],
)
json.dump(res, open(os.path.join(ROOT, "validation/check_export.json"), "w"), indent=1)
