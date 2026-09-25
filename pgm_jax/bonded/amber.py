"""Amber / GAFF bonded parameters for the "amber" term set (terms.AMBER).

Amber's bonded energy with our families (bond_harm, angle_harm, torsion_amber, improper_amber):

    Amber                                      here
    K_b (r - r0)^2          kcal/mol/A^2       0.5 Kb (b - b0)^2,     Kb = 2 K_b * 418.4 kJ/mol/nm^2
    K_th (th - th0)^2       kcal/mol/rad^2     0.5 Ka (th - th0)^2,   Ka = 2 K_th * 4.184 kJ/mol/rad^2
    PK (1 + cos(n phi - g)) kcal/mol           K_n (1 + cos n phi),   K_n = +-PK * 4.184 (g = 0 / 180 deg)
    improper PK (1 + cos(2 w - 180))           K (1 - cos 2 w),       K = PK * 4.184

A prmtop made by tleap from GAFF types gives the initial values (`init_from_prmtop`); keys that
several instances share get the mean.  `with_amber_impropers` takes Amber's impropers with their
atom order (the dihedral of an improper depends on the order of the outer atoms away from
planarity); without it the impropers are those of our topology (planar 3-coordinated centres).  1-4 terms: Amber scales 1-4 LJ by 1/2 (lj14_scale = 0.5);
electrostatics stays pGM (every pair).  scripts/bonded/gaff_prmtop.py makes the prmtops.
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from ..param import _prmtop_sections

KCAL = 4.184


def read_bonded(path: str) -> dict:
    """Bonded terms of an Amber prmtop (0-based atoms, Amber units: kcal/mol, A, rad)."""
    s = _prmtop_sections(path)
    f = lambda k: np.array([float(x) for x in s.get(k, [])])
    i = lambda k: np.array([int(x) for x in s.get(k, [])], int)
    bk, br = f("BOND_FORCE_CONSTANT"), f("BOND_EQUIL_VALUE")
    ak, at = f("ANGLE_FORCE_CONSTANT"), f("ANGLE_EQUIL_VALUE")
    dk, dn, dp = f("DIHEDRAL_FORCE_CONSTANT"), f("DIHEDRAL_PERIODICITY"), f("DIHEDRAL_PHASE")
    B = np.concatenate([i("BONDS_INC_HYDROGEN"), i("BONDS_WITHOUT_HYDROGEN")]).reshape(-1, 3)
    A = np.concatenate([i("ANGLES_INC_HYDROGEN"), i("ANGLES_WITHOUT_HYDROGEN")]).reshape(-1, 4)
    D = np.concatenate([i("DIHEDRALS_INC_HYDROGEN"), i("DIHEDRALS_WITHOUT_HYDROGEN")]).reshape(-1, 5)
    out = {"bonds": [(a // 3, b // 3, bk[t - 1], br[t - 1]) for a, b, t in B],
           "angles": [(a // 3, b // 3, c // 3, ak[t - 1], at[t - 1]) for a, b, c, t in A],
           "dihedrals": [(a // 3, b // 3, abs(c) // 3, abs(d) // 3, dk[t - 1], int(round(dn[t - 1])), dp[t - 1], d < 0)
                         for a, b, c, d, t in D]}
    return out


def with_amber_impropers(spec, path: str):
    """Give a MolSpec Amber's impropers (atom order of the prmtop, centre third), so that
    improper_amber evaluates exactly Amber's dihedrals.  Call before building the BondedModel."""
    from .topology import build_topology
    if spec.top is None:
        spec.top = build_topology(spec.elements, spec.bonds, (spec.bonds, spec.bond_orders), spec.ref_xyz * 10.0)
    imp = sorted({(a, b, c, d) for a, b, c, d, PK, n, ph, im in read_bonded(path)["dihedrals"] if im})
    spec.top.amber_impropers = np.array(imp, int).reshape(-1, 4)
    return spec


def _assign(acc, key, value):
    acc.setdefault(key, []).append(value)


def init_from_prmtop(model, P: dict, prmtops: dict) -> dict:
    """Initial values of the amber families and reference values from prmtops {molecule index: path}
    (atom order = the molecule's).  Families of the model outside terms.AMBER keep their values."""
    P = {k: (dict(v) if isinstance(v, dict) else v) for k, v in P.items()}
    acc = {}
    for m, path in prmtops.items():
        top, Im = model.mols[m].top, model.I[m]
        amb = read_bonded(path)
        bond_of = {tuple(sorted((int(a), int(b)))): kk for kk, (a, b) in enumerate(top.bonds)}
        for a, b, K, r0 in amb["bonds"]:
            k = bond_of.get(tuple(sorted((a, b))))
            if k is None:
                raise ValueError(f"prmtop bond {a}-{b} is not a bond of {model.mols[m].name}")
            _assign(acc, ("ref", "b0", int(Im["bond"][k])), r0 * 0.1)
            if "bond_harm" in Im:
                _assign(acc, ("bond_harm", "Kb", int(Im["bond_harm"]["k"][k])), 2.0 * K * KCAL * 100.0)
        ang_of = {}
        for kk, (a, b, c) in enumerate(top.angles):
            ang_of[(int(a), int(b), int(c))] = ang_of[(int(c), int(b), int(a))] = kk
        for a, b, c, K, th in amb["angles"]:
            k = ang_of[(a, b, c)]
            _assign(acc, ("ref", "th0", int(Im["angle"][k])), th)
            if "angle_harm" in Im:
                _assign(acc, ("angle_harm", "Ka", int(Im["angle_harm"]["k"][k])), 2.0 * K * KCAL)
        if "torsion_amber" in Im:
            tor_of = {}
            for kk, (a, b, c, d) in enumerate(top.propers):
                tor_of[(int(a), int(b), int(c), int(d))] = tor_of[(int(d), int(c), int(b), int(a))] = kk
            Kn = np.zeros((len(top.propers), 4))
            for a, b, c, d, PK, n, ph, imp in amb["dihedrals"]:
                if imp:
                    continue
                if not 1 <= n <= 4:
                    raise ValueError(f"torsion periodicity {n} > 4 is not in torsion_amber")
                sign = 1.0 if abs(ph) < 1e-3 else (-1.0 if abs(ph - math.pi) < 1e-3 else None)
                if sign is None:
                    raise ValueError(f"torsion phase {ph} rad (only 0 and 180 deg)")
                Kn[tor_of[(a, b, c, d)], n - 1] += sign * PK * KCAL
            for kk in range(len(top.propers)):
                _assign(acc, ("torsion_amber", "K", int(Im["torsion_amber"]["k"][kk])), Kn[kk])
        if "improper_amber" in Im and len(Im["improper_amber"]["q"]):
            quads = [tuple(int(v) for v in q) for q in Im["improper_amber"]["q"]]
            by_quad = {q: kk for kk, q in enumerate(quads)}
            by_centre = {q[2]: kk for kk, q in enumerate(quads)}
            for a, b, c, d, PK, n, ph, imp in amb["dihedrals"]:
                if not imp:
                    continue
                kk = by_quad.get((a, b, c, d), by_centre.get(c))
                if kk is not None:
                    _assign(acc, ("improper_amber", "K", int(Im["improper_amber"]["k"][kk])), PK * KCAL)
    for (fam, name, idx), vals in acc.items():
        v = np.asarray(P[fam][name]).copy()
        v[idx] = np.mean(np.asarray(vals), axis=0)
        P[fam][name] = jnp.asarray(v)
    return P
