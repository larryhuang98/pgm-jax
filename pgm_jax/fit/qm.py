"""Fitting pGM parameters directly to QM cluster data: interaction energies, SAPT components,
many-body (2-, 3-body) energies and rigid-body forces of molecular clusters.

Data (`QMSet`, JSON; `load_qm_set`): one record per cluster of rigid molecules, with coordinates
(Angstrom) and QM labels in kcal/mol (scripts/qmfit/collect_water_qm.py builds data/qm/water_qm.json):

  E["ref"]        reference interaction energy (counterpoise corrected)
  sapt[...]       SAPT components of dimers: elst, exch, ind, disp, total
  nb["nb2"|"nb3"] 2- and 3-body parts of the interaction energy of clusters (n >= 3)
  grad_int        gradient of the interaction energy (kcal/mol/A, (3n, 3)) -> net force and torque
                  on every rigid molecule (monomer terms carry neither, so they compare directly)

Model side (`ClusterModel`): the gas-phase pGM model (all pairs, no masking) of System([mol] * n),
with the SAPT-like split of `channels.elec_decomposition`:

  elst   interaction of the self-polarized monomers (Gaussian multipoles: charge penetration included)
         <-> SAPT electrostatics
  ind    relaxation of the induced dipoles in the cluster (<= 0)          <-> SAPT induction (incl. dHF)
  vdw    Lennard-Jones or GVDW                                           <-> SAPT exchange + dispersion
  total  elst + ind + vdw = E(cluster) - sum E(monomers)
The 3-body energy of the model is pure induction (everything else is pairwise), so the QM 3-body
energies test the polarization model.

Parameters (`parameter_space`, a `fit.params.ParameterSpace` of values): a chosen subset of the
ParamTable (quantity -> keys) as a flat vector theta, with bounds; free charges move in the null
space of the neutrality constraints (every molecule keeps its total charge). `QMFit` builds the
weighted residual vector r(theta) (energies, components, 3-body, forces, monomer dipole and
polarizability, a ridge prior towards the starting values) and minimizes |r|^2 with scipy's trust-
region least squares and the exact Jacobian (jax.jacfwd).

Units: model nm, kJ/mol, e; data and reports in Angstrom and kcal/mol.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ..channels import ElecChannel, elec_decomposition, molecular_polarizability
from ..lj import LJChannel
from ..system import Molecule, ParamTable, System
from ..units import ANG_NM, DEBYE_E_NM, KCAL
from ..vdw import GVDWChannel
from .params import ParameterSpace


# =================================================================== dataset ==
@dataclass
class QMSet:
    """Records (dicts) of clusters with QM labels; see the module docstring for the fields."""

    records: list
    monomer: dict = field(default_factory=dict)  # monomer properties (dipole_D, polarizability_A3, ...)
    about: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> QMSet:
        d = json.load(open(path))
        return cls(d["records"], d.get("monomer", {}), {k: v for k, v in d.items() if k not in ("records", "monomer")})

    def save(self, path: str):
        out = dict(self.about)
        out.update(monomer=self.monomer, records=self.records)
        json.dump(out, open(path, "w"), separators=(",", ":"))

    def __len__(self):
        return len(self.records)

    @property
    def ids(self):
        return [r["id"] for r in self.records]

    def select(self, pred) -> QMSet:
        return QMSet([r for r in self.records if pred(r)], self.monomer, self.about)

    def split(self, is_test) -> tuple[QMSet, QMSet]:
        return self.select(lambda r: not is_test(r)), self.select(is_test)

    def sets(self):
        return sorted({r["set"] for r in self.records})


load_qm_set = QMSet.load


def label(r, path: str):
    """r['E']['ref'] for path 'E.ref'; NaN if missing."""
    v = r
    for k in path.split("."):
        if not isinstance(v, dict) or k not in v or v[k] is None:
            return float("nan")
        v = v[k]
    return float(v)


# ================================================================== geometry ==
def rigid_water(r_oh: float, hoh_deg: float) -> np.ndarray:
    """O, H, H of a rigid water (any length unit), O at the origin, in the xy plane."""
    t = np.radians(hoh_deg) / 2
    return np.array([[0, 0, 0], [r_oh * np.sin(t), r_oh * np.cos(t), 0], [-r_oh * np.sin(t), r_oh * np.cos(t), 0]])


def superpose_monomers(X, template, masses) -> np.ndarray:
    """Replace every molecule of X ((n m, 3)) by `template` ((m, 3)) superposed on it (mass-weighted
    Kabsch, centre of mass kept).  Used to evaluate a model whose rigid monomer differs from the
    geometry of the data (e.g. TIP3P-geometry pGM water on pGM3P-25-geometry clusters)."""
    T = np.asarray(template, float)
    w = np.asarray(masses, float)
    m = len(T)
    out = []
    for Q in np.asarray(X, float).reshape(-1, m, 3):
        pc, qc = (w[:, None] * T).sum(0) / w.sum(), (w[:, None] * Q).sum(0) / w.sum()
        U, _, Vt = np.linalg.svd(((T - pc) * w[:, None]).T @ (Q - qc))
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1, 1, d]) @ U.T
        out.append((T - pc) @ R.T + qc)
    return np.concatenate(out)


def rigid_body_forces(g, X, masses, m: int):
    """Net force (n, 3) and torque about each molecule's centre of mass (n, 3) from an energy
    gradient g ((n m, 3)); works for numpy and jax arrays."""
    xp = jnp if isinstance(g, jax.Array) else np
    g = g.reshape(-1, m, 3)
    X = X.reshape(-1, m, 3)
    w = xp.asarray(masses)[None, :, None]
    com = (w * X).sum(1, keepdims=True) / w.sum(1, keepdims=True)
    F = -g
    return F.sum(1), xp.cross(X - com, F).sum(1)


# ============================================================== model side ==
class ClusterModel:
    """pGM (+ LJ or GVDW) predictions for clusters of copies of one rigid molecule."""

    def __init__(
        self, mol: Molecule, vdw: str = "lj", rep: str = "gauss", table: ParamTable | None = None, monomer_xyz_nm=None
    ):
        self.mol = mol
        self.m = mol.n
        self.table = ParamTable([mol]) if table is None else table
        self.vdw = LJChannel() if vdw == "lj" else GVDWChannel(rep=rep)
        self.vdw_kind = vdw
        self._sys, self._fn = {}, {}
        self.monomer_xyz = None if monomer_xyz_nm is None else jnp.asarray(monomer_xyz_nm)

    def system(self, n: int) -> System:
        if n not in self._sys:
            self._sys[n] = System([self.mol] * n, table=self.table)
        return self._sys[n]

    def components(self, X, P, n: int):
        """Intermolecular energy components (kJ/mol) of one cluster, X (n m, 3) nm."""
        s = self.system(n)
        d = elec_decomposition(X, s, P)
        v = self.vdw.energy(X, s, P)[0]["vdw"]
        return {"elst": d["elst"], "ind": d["ind"], "vdw": v, "total": d["elec"] + v}

    def batch(self, n: int):
        """f(X (B, n m, 3), P) -> dict of (B,) arrays."""
        if ("batch", n) not in self._fn:
            self._fn[("batch", n)] = jax.vmap(lambda x, p: self.components(x, p, n), in_axes=(0, None))
        return self._fn[("batch", n)]

    def batch_grad(self, n: int):
        """f(X (B, n m, 3), P) -> dE_total/dX (B, n m, 3), kJ/mol/nm."""
        if ("grad", n) not in self._fn:
            g = jax.grad(lambda x, p: self.components(x, p, n)["total"])
            self._fn[("grad", n)] = jax.vmap(g, in_axes=(0, None))
        return self._fn[("grad", n)]

    def monomer_dipole(self, P, X=None):
        """Dipole (e nm, (3,)) of the isolated molecule: charges + permanent + induced dipoles."""
        X = self.monomer_xyz if X is None else X
        s = self.system(1)
        _, aux = ElecChannel().energy(X, s, P)
        q = s.expand(P)["q"]
        return jnp.sum(q[:, None] * X, 0) + jnp.sum(aux["p"] + aux["mu"], 0)

    def monomer_polarizability(self, P, X=None):
        """Isotropic polarizability (nm^3) of the isolated molecule."""
        X = self.monomer_xyz if X is None else X
        return jnp.trace(molecular_polarizability(X, self.system(1), P)) / 3.0


class Prepared:
    """Static index arrays and coordinates (nm) of a QMSet for one ClusterModel."""

    def __init__(self, data: QMSet, cm: ClusterModel, xyz_key: str = "xyz_A"):
        self.data, self.cm = data, cm
        m = cm.m
        recs = data.records
        self.N = len(recs)
        X = [np.asarray(r[xyz_key], float) * ANG_NM for r in recs]
        self.n = np.array([len(x) // m for x in X])
        self.groups = {}
        for n in sorted(set(self.n)):
            pos = np.nonzero(self.n == n)[0]
            self.groups[int(n)] = (pos, jnp.asarray(np.stack([X[k] for k in pos])))
        # many-body: pairs and triples of the clusters with 3-body labels
        cl = [k for k in range(self.N) if self.n[k] >= 3 and np.isfinite(label(recs[k], "nb.nb3"))]
        pairs, triples, pair_owner, tri_owner, tri_pairs = [], [], [], [], []
        for c, k in enumerate(cl):
            pid = {}
            for i, j in itertools.combinations(range(self.n[k]), 2):
                pid[(i, j)] = len(pairs)
                pairs.append(np.concatenate([X[k][m * i : m * i + m], X[k][m * j : m * j + m]]))
                pair_owner.append(c)
            for i, j, l in itertools.combinations(range(self.n[k]), 3):
                triples.append(np.concatenate([X[k][m * a : m * a + m] for a in (i, j, l)]))
                tri_owner.append(c)
                tri_pairs.append([pid[(i, j)], pid[(i, l)], pid[(j, l)]])
        self.clusters = np.array(cl, int)
        self.pairs = jnp.asarray(np.stack(pairs)) if pairs else None
        self.triples = jnp.asarray(np.stack(triples)) if triples else None
        self.pair_owner, self.tri_owner = np.array(pair_owner, int), np.array(tri_owner, int)
        self.tri_pairs = np.array(tri_pairs, int).reshape(-1, 3)
        # rigid-body forces
        fk = [k for k in range(self.N) if recs[k].get("grad_int") is not None]
        self.force_recs = {}
        for n in sorted({int(self.n[k]) for k in fk}):
            ks = np.array([k for k in fk if self.n[k] == n], int)
            G = np.stack([np.asarray(recs[k]["grad_int"], float) for k in ks]) * (KCAL / ANG_NM)  # kJ/mol/nm
            Xn = np.stack([X[k] for k in ks])
            F, T = rigid_body_forces(G, Xn, cm.mol.masses, m)
            self.force_recs[n] = (ks, jnp.asarray(Xn), jnp.asarray(F), jnp.asarray(T))

    def predict(self, P):
        """Model energies (kJ/mol): per record 'total', 'elst', 'ind', 'vdw' (N,); per cluster with
        3-body labels 'nb2', 'nb3' (len(self.clusters),)."""
        out = {c: jnp.zeros(self.N) for c in ("total", "elst", "ind", "vdw")}
        for n, (pos, X) in self.groups.items():
            e = self.cm.batch(n)(X, P)
            for c in out:
                out[c] = out[c].at[pos].set(e[c])
        if self.pairs is not None:
            ep = self.cm.batch(2)(self.pairs, P)["total"]
            et = self.cm.batch(3)(self.triples, P)["total"]
            nc = len(self.clusters)
            out["nb2"] = jnp.zeros(nc).at[self.pair_owner].add(ep)
            out["nb3"] = jnp.zeros(nc).at[self.tri_owner].add(et - ep[self.tri_pairs].sum(1))
        return out

    def forces(self, P):
        """{n: (record indices, F (B, n, 3) kJ/mol/nm, torque (B, n, 3) kJ/mol)} of the model."""
        res = {}
        for n, (ks, X, _, _) in self.force_recs.items():
            g = self.cm.batch_grad(n)(X, P)
            res[n] = (ks,) + rigid_body_forces(g, X, self.cm.mol.masses, self.cm.m)
        return res


# ============================================================ parameters ==
# bounds and typical changes (table units) of the fitted values
BOUNDS = {
    "q": (-np.inf, np.inf),
    "cov": (-0.2, 0.2),
    "radius": (0.01, 0.3),
    "alpha": (1e-6, 1e-2),
    "lj_rmin_half": (0.0, 0.4),
    "lj_sqrt_eps": (0.0, 10.0),
    "gvdw_sqrt_a": (0.0, 1e4),
    "gvdw_sqrt_c6": (0.0, 10.0),
    "gvdw_b": (0.05, 20.0),
    "quad": (-1.0, 1.0),
}
STEPS = {
    "q": 0.05,
    "cov": 0.002,
    "radius": 0.005,
    "alpha": 1e-4,
    "lj_rmin_half": 0.01,
    "lj_sqrt_eps": 0.1,
    "gvdw_sqrt_a": 5.0,
    "gvdw_sqrt_c6": 0.005,
    "gvdw_b": 0.1,
    "quad": 0.002,
}


def parameter_space(
    table: ParamTable,
    molecules: list[Molecule],
    free: dict,
    p0=None,
    bounds: dict | None = None,
    steps: dict | None = None,
) -> ParameterSpace:
    """The parameters of a QM fit: the values of chosen table entries.

    Parameters
    ----------
    table : ParamTable
        The parameter table.
    molecules : list of Molecule
        Molecules whose total charge the free charges keep (the charges move in the null space
        of their neutrality constraints).
    free : dict
        {quantity: "all" | [keys]}.
    p0 : dict, optional
        Starting parameters (default: table.initial()).
    bounds, steps : dict, optional
        {quantity: (lower, upper)} and {quantity: typical change} overriding BOUNDS and STEPS
        (table units).

    Returns
    -------
    ParameterSpace
        theta0 = the starting values (charges: 0, offsets in the null space); lower / upper /
        step for the least-squares solver and the ridge prior.
    """
    return ParameterSpace.values(
        table,
        free,
        p0=p0,
        neutral=molecules,
        bounds=dict(BOUNDS, **(bounds or {})),
        steps=dict(STEPS, **(steps or {})),
    )


# ================================================================ the fit ==
@dataclass
class FitWeights:
    """Weights of the residual groups (0 switches a group off) and their scales.
    Energy residuals are divided by sigma_i = sigma_E sqrt(n_pairs_i) (1 + max(E_ref_i, 0) / e_soft)
    (larger clusters and repulsive geometries count less); SAPT components of a dimer use the same
    sigma_i as its total."""

    total: float = 1.0
    elst: float = 0.0
    ind: float = 0.0
    exch_disp: float = 0.0
    nb3: float = 0.0
    force: float = 0.0
    dipole: float = 0.0
    polarizability: float = 0.0
    prior: float = 0.01
    sigma_E: float = 1.0  # kcal/mol
    e_soft: float = 5.0  # kcal/mol
    sigma_nb3: float = 0.3  # kcal/mol (per cluster, times sqrt(n_triples))
    sigma_F: float = 1.0  # kcal/mol/A (forces), kcal/mol (torques)
    sigma_dip: float = 0.02  # D
    sigma_pol: float = 0.02  # A^3
    ref: str = "E.ref"  # label of the reference total


class QMFit:
    """Weighted least squares of pGM parameters against a QMSet."""

    def __init__(self, cm: ClusterModel, space: ParameterSpace, data: QMSet, weights: FitWeights = FitWeights()):
        """Set up the residuals.

        Parameters
        ----------
        cm : ClusterModel
            The model.
        space : ParameterSpace
            The fitted parameters (parameter_space).
        data : QMSet
            The QM records.
        weights : FitWeights
            Weights and widths of the residual groups.
        """
        w = weights
        self.cm, self.space, self.data, self.w = cm, space, data, w
        self.prep = Prepared(data, cm)
        recs = data.records
        npairs = np.array([n * (n - 1) / 2 for n in self.prep.n], float)
        Eref = np.array([label(r, w.ref) for r in recs])
        sig = w.sigma_E * np.sqrt(npairs) * (1 + np.maximum(np.nan_to_num(Eref), 0) / w.e_soft) * KCAL
        self.sig = sig
        self.targets = {}

        def add(name, weight, vals):
            vals = np.asarray(vals, float)
            ok = np.isfinite(vals)
            if weight > 0 and ok.any():
                self.targets[name] = (
                    np.nonzero(ok)[0],
                    jnp.asarray(vals[ok] * KCAL),
                    jnp.asarray(sig[ok] / np.sqrt(weight)),
                )

        add("total", w.total, Eref)
        add("elst", w.elst, [label(r, "sapt.elst") for r in recs])
        add("ind", w.ind, [label(r, "sapt.ind") for r in recs])
        add("exch_disp", w.exch_disp, [label(r, "sapt.exch") + label(r, "sapt.disp") for r in recs])
        cl = self.prep.clusters
        self.nb3 = None
        if w.nb3 > 0 and len(cl):
            ntri = np.array([self.prep.n[k] * (self.prep.n[k] - 1) * (self.prep.n[k] - 2) / 6 for k in cl])
            self.nb3 = (
                jnp.asarray([label(recs[k], "nb.nb3") * KCAL for k in cl]),
                jnp.asarray(w.sigma_nb3 * KCAL * np.sqrt(ntri) / np.sqrt(w.nb3)),
            )
        self.mono = data.monomer

    def residuals(self, theta):
        """Weighted residuals of all targets at theta.

        Parameters
        ----------
        theta : array_like (n,)
            Fitted parameters (see `space`).

        Returns
        -------
        jax.Array (m,)
            (model - reference) / sigma for the interaction-energy components, the three-body energies,
            the forces and torques, the monomer dipole and polarizability, and the prior
            sqrt(prior) (theta - theta0) / step, concatenated (dimensionless).
        """
        P = self.space(theta)
        w = self.w
        out = []
        pred = self.prep.predict(P)
        comp = {"total": pred["total"], "elst": pred["elst"], "ind": pred["ind"], "exch_disp": pred["vdw"]}
        for name, (idx, ref, sig) in self.targets.items():
            out.append((comp[name][idx] - ref) / sig)
        if self.nb3 is not None:
            out.append((pred["nb3"] - self.nb3[0]) / self.nb3[1])
        if w.force > 0 and self.prep.force_recs:
            s = w.sigma_F * KCAL / ANG_NM / np.sqrt(w.force)
            for n, (_ks, F, T) in self.prep.forces(P).items():
                _, _, Fq, Tq = self.prep.force_recs[n]
                out.append(((F - Fq) / s).ravel())
                out.append(((T - Tq) / (s * ANG_NM)).ravel())
        if w.dipole > 0 and "dipole_D" in self.mono:
            mu = jnp.linalg.norm(self.cm.monomer_dipole(P)) / DEBYE_E_NM
            out.append(jnp.atleast_1d((mu - self.mono["dipole_D"]) / w.sigma_dip * np.sqrt(w.dipole)))
        if w.polarizability > 0 and "polarizability_A3" in self.mono:
            a = self.cm.monomer_polarizability(P) / ANG_NM**3
            out.append(jnp.atleast_1d((a - self.mono["polarizability_A3"]) / w.sigma_pol * np.sqrt(w.polarizability)))
        if w.prior > 0:
            out.append(np.sqrt(w.prior) * (theta - self.space.theta0) / self.space.step)
        return jnp.concatenate(out)

    def loss(self, theta):
        """Half the sum of squared residuals at theta (dimensionless; differentiable with jax.grad)."""
        r = self.residuals(theta)
        return 0.5 * jnp.sum(r * r)

    def fit(self, theta0=None, max_nfev: int = 200, verbose: int = 0, **kw):
        """scipy.optimize.least_squares (trust-region reflective, bounds) with the exact Jacobian."""
        from scipy.optimize import least_squares

        rf = jax.jit(self.residuals)
        jf = jax.jit(jax.jacfwd(self.residuals))
        th0 = self.space.theta0 if theta0 is None else np.asarray(theta0, float)
        res = least_squares(
            lambda t: np.asarray(rf(jnp.asarray(t))),
            th0,
            jac=lambda t: np.asarray(jf(jnp.asarray(t))),
            bounds=(self.space.lower, self.space.upper),
            x_scale=self.space.step,
            method="trf",
            max_nfev=max_nfev,
            verbose=verbose,
            **kw,
        )
        self.result = res
        return res


# ================================================================ reports ==
def evaluate(cm: ClusterModel, data: QMSet, P, ref: str = "E.ref", prep: Prepared | None = None) -> dict:
    """Predictions and errors (kcal/mol) per record: {'ids', 'set', 'n', 'ref', 'total', 'elst', 'ind',
    'vdw', 'sapt_*', 'nb3_ref', 'nb3', 'nb3_ids'}."""
    prep = prep or Prepared(data, cm)
    pr = {k: np.asarray(v) / KCAL for k, v in prep.predict(P).items()}
    recs = data.records
    out = {
        "ids": data.ids,
        "set": [r["set"] for r in recs],
        "n": prep.n,
        "ref": np.array([label(r, ref) for r in recs]),
    }
    out.update({k: pr[k] for k in ("total", "elst", "ind", "vdw")})
    for c in ("elst", "exch", "ind", "disp", "total"):
        out[f"sapt_{c}"] = np.array([label(r, f"sapt.{c}") for r in recs])
    if "nb3" in pr:
        out["nb3"] = pr["nb3"]
        out["nb2"] = pr["nb2"]
        out["nb3_ref"] = np.array([label(recs[k], "nb.nb3") for k in prep.clusters])
        out["nb3_ids"] = [recs[k]["id"] for k in prep.clusters]
    return out


def error_table(ev: dict, groups: dict | None = None) -> list[dict]:
    """RMSE / MAE / max|err| / mean signed error (kcal/mol) of the total interaction energy per group of
    sets (default: each set), plus the SAPT components of dimers and the 3-body energies."""
    sets = np.array(ev["set"])
    groups = groups or {s: [s] for s in sorted(set(ev["set"]))}
    rows = []

    def stats(name, what, e):
        e = e[np.isfinite(e)]
        if len(e):
            rows.append(
                {
                    "group": name,
                    "quantity": what,
                    "N": int(len(e)),
                    "RMSE": float(np.sqrt(np.mean(e**2))),
                    "MAE": float(np.mean(np.abs(e))),
                    "MaxAE": float(np.max(np.abs(e))),
                    "MSE": float(np.mean(e)),
                }
            )

    for name, ss in groups.items():
        mk = np.isin(sets, ss)
        stats(name, "E_int", (ev["total"] - ev["ref"])[mk])
        stats(name, "elst", (ev["elst"] - ev["sapt_elst"])[mk])
        stats(name, "ind", (ev["ind"] - ev["sapt_ind"])[mk])
        stats(name, "exch+disp", (ev["vdw"] - ev["sapt_exch"] - ev["sapt_disp"])[mk])
        if "nb3" in ev:
            idx = {i: k for k, i in enumerate(ev["ids"])}
            m3 = np.array([sets[idx[i]] in ss for i in ev["nb3_ids"]], bool)
            if m3.any():
                stats(name, "3-body", (ev["nb3"] - ev["nb3_ref"])[m3])
    return rows


def format_table(rows: list[dict]) -> str:
    lines = [f"{'group':24s} {'quantity':10s} {'N':>5s} {'RMSE':>7s} {'MAE':>7s} {'MaxAE':>7s} {'MSE':>7s}"]
    for r in rows:
        lines.append(
            f"{r['group']:24s} {r['quantity']:10s} {r['N']:5d} {r['RMSE']:7.3f} {r['MAE']:7.3f} {r['MaxAE']:7.3f} "
            f"{r['MSE']:+7.3f}"
        )
    return "\n".join(lines)


# ======================================================= rigid-body minima ==
def _rodrigues(w):
    th2 = jnp.sum(w * w)
    th = jnp.sqrt(th2 + 1e-30)
    K = jnp.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    a = jnp.where(th2 < 1e-12, 1.0 - th2 / 6.0, jnp.sin(th) / th)
    b = jnp.where(th2 < 1e-12, 0.5 - th2 / 24.0, (1.0 - jnp.cos(th)) / (th2 + 1e-30))
    return jnp.eye(3) + a * K + b * K @ K


def rigid_minimize(cm: ClusterModel, X_A, P=None, gtol: float = 1e-6, maxiter: int = 2000):
    """Minimize the model's interaction energy over the rigid-body coordinates of every molecule
    (L-BFGS, exact gradients).  X_A: (n m, 3) Angstrom.  Returns (E_min kcal/mol, X_min Angstrom)."""
    from scipy.optimize import minimize

    m = cm.m
    X0 = jnp.asarray(np.asarray(X_A, float) * ANG_NM).reshape(-1, m, 3)
    n = X0.shape[0]
    w = jnp.asarray(cm.mol.masses)[None, :, None]
    com = (w * X0).sum(1, keepdims=True) / w.sum(1, keepdims=True)

    def coords(z):
        z = z.reshape(n, 6)
        R = jax.vmap(_rodrigues)(z[:, 3:])
        return (jnp.einsum("kab,kib->kia", R, X0 - com) + com + z[:, None, :3]).reshape(-1, 3)

    f = jax.jit(jax.value_and_grad(lambda z: cm.components(coords(z), P, n)["total"]))
    res = minimize(
        lambda z: tuple(np.asarray(v, float) for v in f(jnp.asarray(z))),
        np.zeros(6 * n),
        jac=True,
        method="L-BFGS-B",
        options={"gtol": gtol, "maxiter": maxiter},
    )
    return float(res.fun) / KCAL, np.asarray(coords(jnp.asarray(res.x))) / ANG_NM
