"""Fit pGM parameters directly to QM data of molecular clusters.

The targets are interaction energies, SAPT components, many-body (2-, 3-body) energies and
rigid-body forces of clusters, and monomer dipoles and polarizabilities.  Contents: QMSet /
load_qm_set / label (data), rigid_water, superpose_monomers, rigid_body_forces (geometry),
ClusterModel and Prepared (model side), parameter_space, BOUNDS, STEPS (parameters), FitWeights
and QMFit (the fit), evaluate, error_table, format_table (reports), rigid_minimize (rigid-body
minima of the model).

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

See also docs/qmfit.md.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..channels import ElecChannel, elec_decomposition, molecular_polarizability
from ..de import DEChannel
from ..lj import LJChannel
from ..system import Molecule, ParamTable, System
from ..units import ANG_NM, DEBYE_E_NM, KCAL
from ..vdw import GVDWChannel
from .params import ParameterSpace


# =================================================================== dataset ==
@dataclass
class QMSet:
    """Records (dicts) of clusters with QM labels (a mutable dataclass).

    Parameters
    ----------
    records : list of dict
        One record per cluster: "id", "set", coordinates "xyz_A" (n m, 3) [Angstrom] and labels
        (module docstring) in kcal/mol.
    monomer : dict
        Monomer properties ("dipole_D" [D], "polarizability_A3" [A^3], ...).
    about : dict
        Other top-level entries of the JSON file (kept on save).
    """

    records: list
    monomer: dict = field(default_factory=dict)  # monomer properties (dipole_D, polarizability_A3, ...)
    about: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> QMSet:
        """Read a QMSet from JSON ({"records": [...], "monomer": {...}, other keys -> about})."""
        d = json.load(open(path))
        return cls(d["records"], d.get("monomer", {}), {k: v for k, v in d.items() if k not in ("records", "monomer")})

    def save(self, path: str) -> None:
        """Write the set as compact JSON (about, monomer, records)."""
        out = dict(self.about)
        out.update(monomer=self.monomer, records=self.records)
        json.dump(out, open(path, "w"), separators=(",", ":"))

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.records)

    @property
    def ids(self) -> list:
        """Record ids, in record order."""
        return [r["id"] for r in self.records]

    def select(self, pred: Callable[[dict], bool]) -> QMSet:
        """Return the set of the records for which pred(record) is true (same monomer and about)."""
        return QMSet([r for r in self.records if pred(r)], self.monomer, self.about)

    def split(self, is_test: Callable[[dict], bool]) -> tuple[QMSet, QMSet]:
        """Return (training set, test set) by the predicate is_test(record)."""
        return self.select(lambda r: not is_test(r)), self.select(is_test)

    def sets(self) -> list:
        """Return the sorted names of the record sets (the "set" field)."""
        return sorted({r["set"] for r in self.records})


load_qm_set = QMSet.load


def label(r: dict, path: str) -> float:
    """Return the label at a dotted path of a record (r["E"]["ref"] for "E.ref") as a float; NaN if missing or None."""
    v = r
    for k in path.split("."):
        if not isinstance(v, dict) or k not in v or v[k] is None:
            return float("nan")
        v = v[k]
    return float(v)


# ================================================================== geometry ==
def rigid_water(r_oh: float, hoh_deg: float) -> np.ndarray:
    """Return the O, H, H coordinates (3, 3) of a rigid water, O at the origin, in the xy plane.

    `r_oh` is the O-H length (any length unit; the result has the same unit), `hoh_deg` the H-O-H
    angle [degrees]; the bisector is +y.
    """
    t = np.radians(hoh_deg) / 2
    return np.array([[0, 0, 0], [r_oh * np.sin(t), r_oh * np.cos(t), 0], [-r_oh * np.sin(t), r_oh * np.cos(t), 0]])


def superpose_monomers(X: ArrayLike, template: ArrayLike, masses: ArrayLike) -> np.ndarray:
    """Replace every molecule of a cluster by a template superposed on it.

    Mass-weighted Kabsch fit, centre of mass kept.  Used to evaluate a model whose rigid monomer
    differs from the geometry of the data (e.g. TIP3P-geometry pGM water on pGM3P-25-geometry
    clusters).

    Parameters
    ----------
    X : ArrayLike (n m, 3)
        Cluster of n molecules of m atoms (any length unit).
    template : ArrayLike (m, 3)
        Monomer geometry (same unit).
    masses : ArrayLike (m,)
        Atomic masses (weights) [amu].

    Returns
    -------
    np.ndarray (n m, 3)
        The superposed templates.
    """
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


def rigid_body_forces(g: ArrayLike, X: ArrayLike, masses: ArrayLike, m: int) -> tuple[Any, Any]:
    """Return the net force and the torque about each molecule's centre of mass from an energy gradient.

    Parameters
    ----------
    g : np.ndarray or jax.Array (..., n m, 3)
        Energy gradient (e.g. [kJ/mol/nm]); a leading batch axis is allowed.
    X : np.ndarray or jax.Array (..., n m, 3)
        Positions (e.g. [nm]).
    masses : ArrayLike (m,)
        Atomic masses [amu] (centre of mass).
    m : int
        Atoms per molecule.

    Returns
    -------
    F, T : array (..., n, 3)
        Net force -sum g [gradient units] and torque sum (r - R_com) x (-g) [gradient x length];
        jax arrays if g is a jax.Array, else numpy.
    """
    xp = jnp if isinstance(g, jax.Array) else np
    g = g.reshape(-1, m, 3)
    X = X.reshape(-1, m, 3)
    w = xp.asarray(masses)[None, :, None]
    com = (w * X).sum(1, keepdims=True) / w.sum(1, keepdims=True)
    F = -g
    return F.sum(1), xp.cross(X - com, F).sum(1)


# ============================================================== model side ==
class ClusterModel:
    """pGM (+ LJ or GVDW) predictions for clusters of copies of one rigid molecule.

    Systems System([mol] * n) and the vmapped energy functions are built per cluster size n and
    cached.  Not a pytree.

    Attributes
    ----------
    mol : Molecule
    m : int
        Atoms per molecule.
    table : ParamTable
    vdw : LJChannel or GVDWChannel
    vdw_kind : str
    monomer_xyz : jax.Array (m, 3) or None
        Monomer geometry [nm] for the monomer properties.
    """

    def __init__(
        self,
        mol: Molecule,
        vdw: str = "lj",
        rep: str = "gauss",
        table: ParamTable | None = None,
        monomer_xyz_nm: ArrayLike | None = None,
    ) -> None:
        """Build the model.

        Parameters
        ----------
        mol : Molecule
            The rigid monomer.
        vdw : {"lj", "de", "gvdw"}
            Van der Waals form (any value other than "lj" and "de" gives GVDW).
        rep : {"gauss", "slater"}
            GVDW repulsion.
        table : ParamTable, optional
            Parameter table; None: ParamTable([mol]).
        monomer_xyz_nm : ArrayLike (m, 3), optional
            Monomer geometry [nm] for monomer_dipole / monomer_polarizability.
        """
        self.mol = mol
        self.m = mol.n
        self.table = ParamTable([mol]) if table is None else table
        self.vdw = LJChannel() if vdw == "lj" else DEChannel() if vdw == "de" else GVDWChannel(rep=rep)
        self.vdw_kind = vdw
        self._sys, self._fn = {}, {}
        self.monomer_xyz = None if monomer_xyz_nm is None else jnp.asarray(monomer_xyz_nm)

    def system(self, n: int) -> System:
        """Return the (cached) System of n copies of the molecule."""
        if n not in self._sys:
            self._sys[n] = System([self.mol] * n, table=self.table)
        return self._sys[n]

    def components(self, X: jax.Array, P: dict | None, n: int) -> dict[str, jax.Array]:
        """Return the intermolecular energy components of one cluster [kJ/mol].

        `X` (n m, 3) positions [nm], `P` the parameter pytree (None: initial values).  Returns "elst",
        "ind" (channels.elec_decomposition), "vdw" and "total" = elec + vdw.
        """
        s = self.system(n)
        d = elec_decomposition(X, s, P)
        v = self.vdw.energy(X, s, P)[0]["vdw"]
        return {"elst": d["elst"], "ind": d["ind"], "vdw": v, "total": d["elec"] + v}

    def batch(self, n: int) -> Callable[[jax.Array, dict], dict[str, jax.Array]]:
        """Return f(X (B, n m, 3), P) -> components as (B,) arrays (vmapped over X, cached; not jitted)."""
        if ("batch", n) not in self._fn:
            self._fn[("batch", n)] = jax.vmap(lambda x, p: self.components(x, p, n), in_axes=(0, None))
        return self._fn[("batch", n)]

    def batch_grad(self, n: int) -> Callable[[jax.Array, dict], jax.Array]:
        """Return f(X (B, n m, 3), P) -> dE_total/dX (B, n m, 3) [kJ/mol/nm] (vmapped, cached)."""
        if ("grad", n) not in self._fn:
            g = jax.grad(lambda x, p: self.components(x, p, n)["total"])
            self._fn[("grad", n)] = jax.vmap(g, in_axes=(0, None))
        return self._fn[("grad", n)]

    def monomer_dipole(self, P: dict | None, X: jax.Array | None = None) -> jax.Array:
        """Return the dipole (3,) [e nm] of the isolated molecule: charges + permanent + induced dipoles.

        Uses full pGM (ElecChannel default level); `X` (m, 3) [nm], None: monomer_xyz.  The charge term
        is about the origin of X (origin-independent for a neutral molecule).
        """
        X = self.monomer_xyz if X is None else X
        s = self.system(1)
        _, aux = ElecChannel().energy(X, s, P)
        q = s.expand(P)["q"]
        return jnp.sum(q[:, None] * X, 0) + jnp.sum(aux["p"] + aux["mu"], 0)

    def monomer_polarizability(self, P: dict | None, X: jax.Array | None = None) -> jax.Array:
        """Return the isotropic polarizability [nm^3] of the isolated molecule (trace / 3; X as monomer_dipole)."""
        X = self.monomer_xyz if X is None else X
        return jnp.trace(molecular_polarizability(X, self.system(1), P)) / 3.0


class Prepared:
    """Static index arrays and coordinates (nm) of a QMSet for one ClusterModel.

    Records are grouped by cluster size (one vmapped call per size); the dimers and trimers of
    clusters with 3-body labels and the reference rigid-body forces are precomputed.

    Attributes
    ----------
    N : int
        Number of records.
    n : np.ndarray (N,) int
        Molecules per record.
    groups : dict of int to (np.ndarray, jax.Array)
        Per size: record indices and coordinates (B, n m, 3) [nm].
    clusters : np.ndarray int
        Records with 3-body labels (n >= 3).
    pairs, triples : jax.Array or None
        All dimers (P, 2 m, 3) and trimers (T, 3 m, 3) of those clusters [nm].
    pair_owner, tri_owner : np.ndarray int
        Cluster (position in `clusters`) of each dimer / trimer.
    tri_pairs : np.ndarray (T, 3) int
        The three dimers of each trimer.
    force_recs : dict of int to tuple
        Per size: record indices, coordinates [nm], reference net forces [kJ/mol/nm] and torques
        [kJ/mol] of the records with "grad_int".
    """

    def __init__(self, data: QMSet, cm: ClusterModel, xyz_key: str = "xyz_A") -> None:
        """Prepare the records.

        Parameters
        ----------
        data : QMSet
            The records.
        cm : ClusterModel
            The model (atoms per molecule, masses).
        xyz_key : str
            Record key of the coordinates [Angstrom].
        """
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

    def predict(self, P: dict | None) -> dict[str, jax.Array]:
        """Return the model energies [kJ/mol] of all records.

        Per record "total", "elst", "ind", "vdw" (N,); per cluster with 3-body labels "nb2" (sum of the
        dimer interaction energies) and "nb3" (sum over trimers of E_int(ijk) minus its three dimer
        energies) (len(self.clusters),).  Differentiable in P.
        """
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

    def forces(self, P: dict | None) -> dict[int, tuple]:
        """Return {n: (record indices, F (B, n, 3) [kJ/mol/nm], torque (B, n, 3) [kJ/mol])} of the model."""
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
    p0: dict | None = None,
    bounds: dict | None = None,
    steps: dict | None = None,
) -> ParameterSpace:
    """Return the parameters of a QM fit: the values of chosen table entries.

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
    """Weights of the residual groups (0 switches a group off) and their scales (a mutable dataclass).

    Energy residuals are divided by sigma_i = sigma_E sqrt(n_pairs_i) (1 + max(E_ref_i, 0) / e_soft)
    (larger clusters and repulsive geometries count less); SAPT components of a dimer use the same
    sigma_i as its total.  A group's sigma is divided by sqrt(weight).

    Parameters
    ----------
    total, elst, ind, exch_disp : float
        Weights of the interaction energy and of the SAPT components (exch_disp: the model's vdw
        against SAPT exchange + dispersion).
    nb3 : float
        Weight of the 3-body energies.
    force : float
        Weight of the rigid-body forces and torques.
    dipole, polarizability : float
        Weights of the monomer dipole and polarizability.
    prior : float
        Weight of the ridge prior sqrt(prior) (theta - theta0) / step.
    sigma_E : float
        Energy scale [kcal/mol].
    e_soft : float
        Energy above which repulsive geometries are down-weighted [kcal/mol].
    sigma_nb3 : float
        3-body scale per cluster [kcal/mol], times sqrt(number of triples).
    sigma_F : float
        Force scale [kcal/mol/A]; torques use the same number in kcal/mol.
    sigma_dip : float
        Dipole scale [D].
    sigma_pol : float
        Polarizability scale [A^3].
    ref : str
        Label path of the reference total (label()).
    """

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
    """Weighted least squares of pGM parameters against a QMSet.

    Attributes
    ----------
    cm : ClusterModel
    space : ParameterSpace
    data : QMSet
    w : FitWeights
    prep : Prepared
    sig : np.ndarray (N,)
        Energy scale per record [kJ/mol].
    targets : dict of str to tuple
        Per active energy group: record indices, reference values [kJ/mol], sigmas [kJ/mol].
    nb3 : tuple or None
        Reference 3-body energies and sigmas [kJ/mol].
    mono : dict
        Monomer reference properties.
    result : scipy.optimize.OptimizeResult
        Set by `fit`.
    """

    def __init__(
        self, cm: ClusterModel, space: ParameterSpace, data: QMSet, weights: FitWeights = FitWeights()
    ) -> None:
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

        def add(name: str, weight: float, vals: ArrayLike) -> None:
            """Register an energy group: its finite labels [kcal/mol -> kJ/mol] and sigmas, if weight > 0."""
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

    def residuals(self, theta: ArrayLike) -> jax.Array:
        """Return the weighted residuals of all targets at theta.

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

    def loss(self, theta: ArrayLike) -> jax.Array:
        """Return half the sum of squared residuals at theta (dimensionless; differentiable with jax.grad)."""
        r = self.residuals(theta)
        return 0.5 * jnp.sum(r * r)

    def fit(self, theta0: ArrayLike | None = None, max_nfev: int = 200, verbose: int = 0, **kw: Any) -> Any:
        """Minimise |r|^2 with scipy.optimize.least_squares (trust-region reflective, bounds, exact Jacobian).

        Parameters
        ----------
        theta0 : ArrayLike (n,), optional
            Starting point; None: space.theta0.
        max_nfev : int
            Largest number of residual evaluations.
        verbose : int
            scipy verbosity.
        **kw
            Further arguments of least_squares.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Also stored as self.result; x_scale is space.step, bounds space.lower / upper.
        """
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
def evaluate(cm: ClusterModel, data: QMSet, P: dict | None, ref: str = "E.ref", prep: Prepared | None = None) -> dict:
    """Return the model predictions and the QM labels of every record [kcal/mol].

    Parameters
    ----------
    cm : ClusterModel
        The model.
    data : QMSet
        The records.
    P : dict, optional
        Parameter pytree; None: initial values.
    ref : str
        Label path of the reference total.
    prep : Prepared, optional
        Reuse a Prepared of the same data and model.

    Returns
    -------
    dict
        "ids", "set", "n", "ref", model "total", "elst", "ind", "vdw", labels "sapt_<c>" (elst, exch,
        ind, disp, total), and with 3-body data "nb2", "nb3", "nb3_ref", "nb3_ids".
    """
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
    """Return error statistics of the model per group of sets [kcal/mol].

    Rows for the total interaction energy ("E_int"), the SAPT components ("elst", "ind",
    "exch+disp") and the 3-body energies ("3-body"), each with N, RMSE, MAE, MaxAE and the mean
    signed error MSE (model - reference); non-finite errors are skipped.

    Parameters
    ----------
    ev : dict
        Output of evaluate.
    groups : dict, optional
        {group name: [set names]}; None: one group per set.

    Returns
    -------
    list of dict
    """
    sets = np.array(ev["set"])
    groups = groups or {s: [s] for s in sorted(set(ev["set"]))}
    rows = []

    def stats(name: str, what: str, e: np.ndarray) -> None:
        """Append the statistics row of the finite errors `e` (none if all are NaN)."""
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
    """Return the rows of error_table as a fixed-width text table."""
    lines = [f"{'group':24s} {'quantity':10s} {'N':>5s} {'RMSE':>7s} {'MAE':>7s} {'MaxAE':>7s} {'MSE':>7s}"]
    for r in rows:
        lines.append(
            f"{r['group']:24s} {r['quantity']:10s} {r['N']:5d} {r['RMSE']:7.3f} {r['MAE']:7.3f} {r['MaxAE']:7.3f} "
            f"{r['MSE']:+7.3f}"
        )
    return "\n".join(lines)


# ======================================================= rigid-body minima ==
def _rodrigues(w: jax.Array) -> jax.Array:
    """Return the rotation matrix (3, 3) of a rotation vector w (3,) [rad] by Rodrigues' formula.

    R = 1 + (sin t / t) K + ((1 - cos t) / t^2) K^2, K the cross-product matrix of w, t = |w|;
    Taylor expansions for t^2 < 1e-12 keep it smooth and differentiable at w = 0.
    """
    th2 = jnp.sum(w * w)
    th = jnp.sqrt(th2 + 1e-30)
    K = jnp.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    a = jnp.where(th2 < 1e-12, 1.0 - th2 / 6.0, jnp.sin(th) / th)
    b = jnp.where(th2 < 1e-12, 0.5 - th2 / 24.0, (1.0 - jnp.cos(th)) / (th2 + 1e-30))
    return jnp.eye(3) + a * K + b * K @ K


def rigid_minimize(
    cm: ClusterModel, X_A: ArrayLike, P: dict | None = None, gtol: float = 1e-6, maxiter: int = 2000
) -> tuple[float, np.ndarray]:
    """Minimise the model's interaction energy over the rigid-body coordinates of every molecule.

    L-BFGS-B with exact gradients; each molecule has a translation and a rotation vector about its
    starting centre of mass.

    Parameters
    ----------
    cm : ClusterModel
        The model.
    X_A : ArrayLike (n m, 3)
        Starting cluster [Angstrom].
    P : dict, optional
        Parameter pytree; None: initial values.
    gtol : float
        Gradient tolerance of L-BFGS-B [kJ/mol per nm or rad].
    maxiter : int
        Largest number of iterations.

    Returns
    -------
    E_min : float
        Interaction energy at the minimum [kcal/mol].
    X_min : np.ndarray (n m, 3)
        Minimised cluster [Angstrom].
    """
    from scipy.optimize import minimize

    m = cm.m
    X0 = jnp.asarray(np.asarray(X_A, float) * ANG_NM).reshape(-1, m, 3)
    n = X0.shape[0]
    w = jnp.asarray(cm.mol.masses)[None, :, None]
    com = (w * X0).sum(1, keepdims=True) / w.sum(1, keepdims=True)

    def coords(z: jax.Array) -> jax.Array:
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
