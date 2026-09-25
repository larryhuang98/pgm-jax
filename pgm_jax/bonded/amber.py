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
planarity); without it the impropers are those of our topology (planar 3-coordinated centres).
1-4 terms: Amber scales 1-4 LJ by 1/2 (lj14_scale = 0.5); electrostatics stays pGM (every pair).
scripts/bonded/gaff_prmtop.py makes the prmtops.

Backbone correction maps: Amber's CMAP is a 24 x 24 grid (kcal/mol) with bicubic interpolation;
ours is a Fourier series (terms/cmap.py).  `fit_cmap_fourier` fits the series to a grid (import,
e.g. ff19SB), `cmap_grid` tabulates it (export).

Export (`export_bonded`): the per-instance parameters of a fitted model (typed families or the
neural bonded terms, frozen or not) replace the bonded sections of an existing prmtop for the
molecule's atoms: bonds, angles, proper and improper dihedrals (with the 1-4 flags: one dihedral
per 1-4 pair computes the 1-4 interactions), and CMAP.  Terms of other atoms (water, ions) and
every other section (pGM's POL_GAUSS_*, LJ, exclusions) are kept, so the result runs in
sander / pmemd(-pgm) with the fitted bonded terms.
"""
from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from ..prmtop import Prmtop
from . import terms as T

KCAL = 4.184
_A2 = 100.0                                      # A^2 per nm^2


def read_bonded(path: str) -> dict:
    """Bonded terms of an Amber prmtop (0-based atoms, Amber units: kcal/mol, A, rad): bonds
    (i, j, K, r0), angles (i, j, k, K, th0), dihedrals (i, j, k, l, PK, n, phase, improper),
    cmap (i, j, k, l, m, type) and cmap_grids (per type, (res, res) kcal/mol, phi rows)."""
    top = Prmtop.read(path)
    f = lambda k: top.get(k).astype(float) if k in top else np.zeros(0)
    i = lambda k: top.get(k).astype(int) if k in top else np.zeros(0, int)
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
    out["cmap"], out["cmap_grids"] = _read_cmap(top)
    return out


def _read_cmap(top: Prmtop):
    if "CMAP_INDEX" not in top:
        return [], []
    res = top.get("CMAP_RESOLUTION").astype(int)
    grids = [np.asarray(top.get(f"CMAP_PARAMETER_{k + 1:02d}"), float).reshape(r, r) for k, r in enumerate(res)]
    idx = top.get("CMAP_INDEX").astype(int).reshape(-1, 6)
    return [tuple(int(a) - 1 for a in row[:5]) + (int(row[5]) - 1,) for row in idx], grids


def fit_cmap_fourier(grid_kcal, order: int = T.cmap.ORDER):
    """Coefficients (kJ/mol) of the Fourier map closest to an Amber CMAP grid (least squares on
    the grid points; phi rows, psi columns, from -180 deg)."""
    res = grid_kcal.shape[0]
    g = -np.pi + 2.0 * np.pi * np.arange(res) / res
    Pg, Sg = np.meshgrid(g, g, indexing="ij")
    B = np.asarray(T.cmap_basis(jnp.asarray(Pg.ravel()), jnp.asarray(Sg.ravel()), order))
    y = np.asarray(grid_kcal, float).ravel() * KCAL
    return np.linalg.lstsq(B, y - y.mean(), rcond=None)[0]


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
    """Initial values of the amber families, the backbone maps (Fourier fit of the prmtop's CMAP
    grids) and the reference values from prmtops {molecule index: path} (atom order = the
    molecule's).  Families of the model outside terms.PROTEIN keep their values."""
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
                    if n != 2:
                        raise ValueError(f"improper periodicity {n} (improper_amber has n = 2)")
                    # PK (1 + cos(2w - 180)) = K (1 - cos 2w) with K = PK; phase 0: K = -PK (+ constant)
                    sign = 1.0 if abs(abs(ph) - math.pi) < 1e-3 else (-1.0 if abs(ph) < 1e-3 else None)
                    if sign is None:
                        raise ValueError(f"improper phase {ph} rad (only 0 and 180 deg)")
                    _assign(acc, ("improper_amber", "K", int(Im["improper_amber"]["k"][kk])), sign * PK * KCAL)
        for cf in ("cmap", "cmap6"):
            if cf in Im and len(Im[cf]["q"]) and amb["cmap"]:
                by_q = {tuple(int(v) for v in q): kk for kk, q in enumerate(Im[cf]["q"])}
                for *q, t in amb["cmap"]:
                    kk = by_q.get(tuple(q))
                    if kk is not None:
                        _assign(acc, (cf, "cm", int(Im[cf]["k"][kk])),
                                fit_cmap_fourier(amb["cmap_grids"][t], T.REGISTRY[cf].order))
    for (fam, name, idx), vals in acc.items():
        v = np.asarray(P[fam][name]).copy()
        v[idx] = np.mean(np.asarray(vals), axis=0)
        P[fam][name] = jnp.asarray(v)
    return P


# ------------------------------------------------------------------ export
EXPORTABLE = set(T.PROTEIN) | {"cmap6"}


def instance_parameters(terms, P: dict, m: int = 0) -> dict:
    """Per-instance parameters of molecule m of a BondedTerms / BondedModel: {"b0", "th0",
    family: {name: (instances, ...)}}, from the typed families or from the neural bonded terms
    (live network or frozen {"coef": ...})."""
    out = {}
    if terms.fams:
        Im = terms.I[m]
        out["b0"] = np.asarray(P["ref"]["b0"])[Im["bond"]]
        out["th0"] = np.asarray(P["ref"]["th0"])[Im["angle"]]
        for f in terms.fams:
            out[f] = {k: np.asarray(v)[Im[f]["k"]] for k, v in P[f].items()}
    if terms.nnb is not None:
        C = P["nnb"]["coef"][m] if "coef" in P["nnb"] else terms.nnb.coefficients(P["nnb"], m)
        shared = [f for f in C if f in out and f not in ("b0", "th0")]
        if shared:
            raise ValueError(f"{shared} come from both typed and neural terms; export needs one source")
        if terms.fams and any(f in C for f in ("bond_harm", "angle_harm")):
            raise ValueError("bond / angle reference values come from both typed and neural terms")
        for k, v in C.items():
            out[k] = np.asarray(v) if k in ("b0", "th0") else {n: np.asarray(x) for n, x in v.items()}
    return out


def _dedupe(params, ndigits=8):
    """Type index per entry (1-based) and the list of unique parameter tuples."""
    types, table = [], {}
    for p in params:
        key = tuple(round(float(x), ndigits) for x in p)
        if key not in table:
            table[key] = len(table)
        types.append(table[key] + 1)
    return types, list(table)


def _common(values, default):
    v = [round(float(x), 6) for x in values]
    return max(set(v), key=v.count) if v else default


def export_bonded(prmtop_in: str, prmtop_out: str, terms, P: dict, m: int = 0, offset: int = 0,
                  scee: float | None = None, scnb: float | None = None, resolution: int = 24,
                  zero: float = 1e-10) -> dict:
    """Write prmtop_out = prmtop_in with the bonded terms of molecule m (atoms offset .. offset + n
    of the prmtop) replaced by the model's.  Families: bond_harm, angle_harm, torsion_amber,
    improper_amber, cmap (terms.PROTEIN).  scee / scnb: 1-4 scale factors of the new dihedral types
    (default: the most common values of the input).  Returns counts of what was written."""
    mol = terms.mols[m]
    top = mol.top
    n = top.n
    par = instance_parameters(terms, P, m)
    fams = [f for f in par if f not in ("b0", "th0")]
    bad = [f for f in fams if f not in EXPORTABLE]
    if bad:
        raise ValueError(f"families {bad} have no Amber form; export supports {sorted(EXPORTABLE)}")
    pt = Prmtop.read(prmtop_in)
    Z = pt.get("ATOMIC_NUMBER")
    el = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 15: "P"}
    got = [el.get(int(z), "?") for z in Z[offset:offset + n]]
    if len(got) != n or got != [e if e in el.values() else "?" for e in mol.elements]:
        raise ValueError("the prmtop atoms at the offset are not the molecule's (elements differ)")
    isH = Z == 1
    ours = lambda atoms: all(offset <= a < offset + n for a in atoms)
    old = read_bonded(prmtop_in)
    counts = {}

    # ---------------------------------------------------------------- bonds and angles
    def write_simple(kind, flag_k, flag_x, sec_h, sec_n, width, new):
        kept = [(tuple(e[:width]), (e[width], e[width + 1])) for e in old[kind] if not ours(e[:width])]
        entries = kept + new
        types, table = _dedupe([p for _, p in entries])
        pt.set(flag_k, [t[0] for t in table]); pt.set(flag_x, [t[1] for t in table])
        rows_h, rows_n = [], []
        for (atoms, _), t in zip(entries, types):
            row = [3 * a for a in atoms] + [t]
            (rows_h if any(isH[a] for a in atoms) else rows_n).extend(row)
        pt.set(sec_h, rows_h); pt.set(sec_n, rows_n)
        counts[kind] = (len(new), len(kept), len(table))
        return len(rows_h) // (width + 1), len(rows_n) // (width + 1), len(table)

    new_b = []
    if "bond_harm" in par:
        for (i, j), K, b0 in zip(top.bonds, par["bond_harm"]["Kb"], par["b0"]):
            new_b.append(((int(i) + offset, int(j) + offset), (float(K) / (2.0 * KCAL * _A2), float(b0) * 10.0)))
    elif any(ours(e[:2]) for e in old["bonds"]):
        raise ValueError("the model has no bond_harm terms for the molecule's bonds")
    nbh, nba, nbt = write_simple("bonds", "BOND_FORCE_CONSTANT", "BOND_EQUIL_VALUE",
                                 "BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN", 2, new_b)
    new_a = []
    if "angle_harm" in par:
        for (i, j, k), K, th in zip(top.angles, par["angle_harm"]["Ka"], par["th0"]):
            new_a.append(((int(i) + offset, int(j) + offset, int(k) + offset), (float(K) / (2.0 * KCAL), float(th))))
    elif any(ours(e[:3]) for e in old["angles"]):
        raise ValueError("the model has no angle_harm terms for the molecule's angles")
    nah, naa, nat = write_simple("angles", "ANGLE_FORCE_CONSTANT", "ANGLE_EQUIL_VALUE",
                                 "ANGLES_INC_HYDROGEN", "ANGLES_WITHOUT_HYDROGEN", 3, new_a)

    # ---------------------------------------------------------------- dihedrals
    old_types = list(zip(pt.get("DIHEDRAL_FORCE_CONSTANT"), pt.get("DIHEDRAL_PERIODICITY"), pt.get("DIHEDRAL_PHASE"),
                         pt.get("SCEE_SCALE_FACTOR") if "SCEE_SCALE_FACTOR" in pt else [1.2] * len(pt.get("DIHEDRAL_PHASE")),
                         pt.get("SCNB_SCALE_FACTOR") if "SCNB_SCALE_FACTOR" in pt else [2.0] * len(pt.get("DIHEDRAL_PHASE"))))
    scee = _common([t[3] for t in old_types], 1.2) if scee is None else scee
    scnb = _common([t[4] for t in old_types], 2.0) if scnb is None else scnb
    raw = np.concatenate([pt.get("DIHEDRALS_INC_HYDROGEN"), pt.get("DIHEDRALS_WITHOUT_HYDROGEN")]).reshape(-1, 5)
    kept = []                                          # (atoms signed as in the prmtop, type params)
    for a, b, c, d, t in raw:
        if not ours((a // 3, b // 3, abs(c) // 3, abs(d) // 3)):
            kept.append(((int(a), int(b), int(c), int(d)), tuple(float(x) for x in old_types[t - 1])))
    new_d = []
    if "torsion_amber" in par:
        seen14 = set()
        for (i, j, k, l), Kn in zip(top.propers, par["torsion_amber"]["K"]):
            i, j, k, l = (int(x) + offset for x in (i, j, k, l))
            pair = (min(i, l), max(i, l))
            do14 = top.graph_distance(i - offset, l - offset) == 3 and pair not in seen14
            terms_ = [(abs(float(K)) / KCAL, float(nn + 1), 0.0 if K >= 0 else math.pi)
                      for nn, K in enumerate(Kn) if abs(float(K)) > zero]
            if not terms_ and do14:
                terms_ = [(0.0, 1.0, 0.0)]           # carries the 1-4 interaction
            for q, (PK, per, ph) in enumerate(terms_):
                flag14 = do14 and q == 0
                if flag14:
                    seen14.add(pair)
                a, b, c, d = (i, j, k, l) if (flag14 or k != 0) else (l, k, j, i)   # -0 does not exist
                new_d.append(((3 * a, 3 * b, 3 * c if flag14 else -3 * c, 3 * d), (PK, per, ph, scee, scnb)))
    elif any(not e[7] and ours(e[:4]) for e in old["dihedrals"]):
        raise ValueError("the model has no torsion_amber terms for the molecule's proper dihedrals")
    if "improper_amber" in par:
        quads = np.asarray(T.REGISTRY["improper_amber"].index(top, lambda *x: "")[0]["q"]).reshape(-1, 4)
        for (a, b, c, d), K in zip(quads, par["improper_amber"]["K"]):
            a, b, c, d = (int(x) + offset for x in (a, b, c, d))
            if c == 0 or d == 0:
                raise ValueError("atom 0 as the third or fourth atom of an improper")
            K = float(K)
            new_d.append(((3 * a, 3 * b, -3 * c, -3 * d), (abs(K) / KCAL, 2.0, math.pi if K >= 0 else 0.0, scee, scnb)))
    entries = kept + new_d
    types, table = _dedupe([p for _, p in entries])
    for flag, col in (("DIHEDRAL_FORCE_CONSTANT", 0), ("DIHEDRAL_PERIODICITY", 1), ("DIHEDRAL_PHASE", 2),
                      ("SCEE_SCALE_FACTOR", 3), ("SCNB_SCALE_FACTOR", 4)):
        if flag in pt or col < 3:
            pt.set(flag, [t[col] for t in table])
    rows_h, rows_n = [], []
    for (atoms, _), t in zip(entries, types):
        idx = [abs(x) // 3 for x in atoms]
        (rows_h if any(isH[a] for a in idx) else rows_n).extend(list(atoms) + [t])
    pt.set("DIHEDRALS_INC_HYDROGEN", rows_h); pt.set("DIHEDRALS_WITHOUT_HYDROGEN", rows_n)
    n_dtypes = len(table)
    counts["dihedrals"] = (len(new_d), len(kept), n_dtypes)

    # ---------------------------------------------------------------- CMAP
    cm_f = [f for f in ("cmap", "cmap6") if f in par]
    if len(cm_f) > 1:
        raise ValueError("two backbone map families")
    if cm_f or old["cmap"]:
        qs = np.asarray(T.REGISTRY["cmap"].index(top, lambda *x: "")[0]["q"]).reshape(-1, 5)
        kept_c = [(tuple(e[:5]), old["cmap_grids"][e[5]]) for e in old["cmap"] if not ours(e[:5])]
        new_c = []
        if cm_f:
            order = T.REGISTRY[cm_f[0]].order
            for q, cm in zip(qs, par[cm_f[0]]["cm"]):
                new_c.append((tuple(int(x) + offset for x in q), T.cmap_grid(cm, order, resolution) / KCAL))
        elif any(ours(e[:5]) for e in old["cmap"]):
            raise ValueError("the model has no cmap terms for the molecule's CMAP entries")
        for k in [k for k in pt.sections if k.startswith("CMAP_PARAMETER_")]:
            pt.remove(k)
        entries = kept_c + new_c
        types, table = _dedupe([g.ravel() for _, g in entries], ndigits=5)
        after = "CMAP_RESOLUTION"
        pt.set("CMAP_COUNT", [len(entries), len(table)], fmt="2I8")
        pt.set("CMAP_RESOLUTION", [int(np.sqrt(len(t))) for t in table], fmt="20I4", after="CMAP_COUNT")
        for k, t in enumerate(table):
            name = f"CMAP_PARAMETER_{k + 1:02d}"
            pt.set(name, list(t), fmt="8F9.5", comments=[f"map {k + 1} (pgm_jax export)"], after=after)
            after = name
        pt.remove("CMAP_INDEX")
        pt.set("CMAP_INDEX", [x for (q, _), t in zip(entries, types) for x in [a + 1 for a in q] + [t]],
               fmt="6I8", after=after)
        counts["cmap"] = (len(new_c), len(kept_c), len(table))

    nph, npa = len(rows_h) // 5, len(rows_n) // 5
    pt.set_pointers(NBONH=nbh, MBONA=nba, NBONA=nba, NUMBND=nbt, NTHETH=nah, MTHETA=naa, NTHETA=naa, NUMANG=nat,
                    NPHIH=nph, MPHIA=npa, NPHIA=npa, NPTRA=n_dtypes)
    pt.write(prmtop_out)
    return counts
