"""Flexible-molecule model: bonded families + gas-phase pGM (all pairs, induced dipoles) + LJ.

    E(R) = sum_f E_f(internal coordinates; theta_f) + E_pGM(R; q(R), c(R)) + E_LJ(R)|dist >= lj_min_sep

Options (BondedSettings):
  families       bonded families from terms.REGISTRY (and "flux": geometry-dependent charges and
                 covalent-dipole strengths, F6)
  typing         "molecule": parameters tied by symmetry within each molecule (the paper);
                 "type": tied across molecules by the atom environment to `depth` bonds (transfer);
                 "amber": tied by Amber / GAFF atom types (the molecules' pGM types), as Amber
  Bonded term sets (terms.SETS): "amber" (Amber forms, with lj14_scale 0.5 and typing "amber";
  initial values from GAFF with bonded/amber.py), "explore" (the class II set of the bonded study
  and the families of terms.REGISTRY), "nn" (bonded/nn.py: fast neural bonded terms)
  elec, quadrupoles, vdw, gvdw_rep   nonbonded model options (options.py), as in the MD engine
  elec_exclude   0 = pGM (every pair); 3 = classical control (1-2, 1-3, 1-4 pairs removed from
                 permanent and induced electrostatics)
  lj_min_sep     LJ between atoms at least this many bonds apart (4 = 1-5 and beyond, the paper)
  lj14_scale     LJ on 1-4 pairs with this scale (0 = off)
  elec14_scale   scale of 1-4 electrostatics (with elec_exclude = 2: the Amber/OPLS-like control)
Units nm, kJ/mol, e; dipoles e nm.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ..channels import _dipole_tensor, _field_at_i, _pair_perm, perm_dipoles, quadrupole_field
from ..kernels import DENSITIES
from ..md.kernels import erf_kernels
from ..multipole import quadrupole_pair_terms
from ..multipole import quadrupoles as build_quadrupoles
from ..options import check_vdw, elec_flags
from ..solver import solve_linear_induction
from ..system import System
from ..units import HARTREE_KJMOL, KE
from ..vdw import gvdw_pair
from . import terms as T
from .topology import Topology, build_topology


@dataclass(frozen=True)
class BondedSettings:
    families: tuple = T.PAPER
    typing: str = "molecule"
    depth: int = 2
    elec_exclude: int = 0
    lj_min_sep: int = 4
    lj14_scale: float = 0.0
    elec14_scale: float = 1.0  # scale of 1-4 electrostatics when not excluded (Amber 1/1.2, OPLS 0.5)
    ind_exclude: int = -1  # exclusion for the induction (fields and dipole-dipole couplings);
    # -1: the same as elec_exclude (permanent pairs)
    flux: int = 0  # 1: charge + covalent-dipole flux (linear); 2: + quadratic dipole flux
    qfit: int = -1  # >= 0: pGM charges and covalent dipoles typed by atom environment
    # to this depth (shared across molecules) and fitted with the
    # bonded terms; total charge kept by a uniform shift per molecule
    qbci: int = -1  # >= 0: typed bond-charge increments and covalent-dipole corrections
    # (atom environments to this depth) added to each molecule's ESP
    # charges / dipoles and fitted; neutral by construction, zero = ESP
    escale: tuple = ()  # separations (1 = 1-2, 2 = 1-3, 3 = 1-4) whose permanent pGM
    # pair energies get a learned scale kappa per pair type (added to
    # the fixed weight): learned partial exclusion
    elec: str = "qpi"  # electrostatics level (options.py): "q" | "qp" | "qi" | "qpi"
    quadrupoles: bool = False  # permanent Gaussian quadrupoles (the molecules' quad terms)
    vdw: str = "lj"  # "lj" | "gvdw" | "none" (vdw.py)
    gvdw_rep: str = "gauss"  # GVDW repulsion "gauss" | "slater"
    nn_width: int = 32  # "nnb" (bonded/nn.py): embedding width, message-passing layers,
    nn_layers: int = 3  # reference values ("geometry": minimum geometry + learned
    nn_ref: str = "geometry"  # corrections; "predicted": covalent radii / hybridization)
    nn_basis: tuple = T.PAPER  # term families whose per-instance parameters the network predicts
    nn_th_span: float = 0.35  # max learned shift of reference angles (rad) / of bond lengths (nm)
    nn_b_span: float = 0.01
    nn_out_scale: float = 2.0  # head output -> parameter change, in units of fit.SCALES
    nn_pgm_features: bool = True  # atom features include the pGM q, alpha, radius, |covalent dipoles|
    nn_table_depth: int | None = None  # typed table (atom environments to this depth; 0 = elements) + residual
    nn_resid_l2: float = 0.0  # shrinkage of the network residual towards the typed table
    nn_context: bool = True  # sequence context (residues i-1, i, i+1) for the backbone map (cmap)


@dataclass
class MolSpec:
    name: str
    elements: list
    bonds: list
    bond_orders: list
    charge: int
    ref_xyz: np.ndarray  # (n, 3) nm, a minimum
    pgm: object = None  # pgm_jax.system.Molecule (same atom order) or None
    top: Topology = field(default=None)
    atom_names: list = None  # optional labels (Amber atom / residue names of proteins)
    residue_names: list = None


def _classes(elements, bonds, depth):
    """Atom environment strings: element and neighbours to `depth` bonds (canonical, comparable
    across molecules)."""
    n = len(elements)
    nbr = [[] for _ in range(n)]
    for i, j in bonds:
        nbr[i].append(j)
        nbr[j].append(i)
    col = [str(e) for e in elements]
    for _ in range(depth):
        col = [col[i] + "(" + ",".join(sorted(col[j] for j in nbr[i])) + ")" for i in range(n)]
    return [hashlib.md5(c.encode()).hexdigest()[:8] + ":" + c.split("(")[0] for c in col]


class BondedTerms:
    """The bonded half of the model: families, tying keys, index sets, parameters and the bonded
    energy (and the neural bonded terms).  No gas-phase nonbonded setup, so it serves MD templates
    of any size; BondedModel adds the pGM + van der Waals intramolecular model for fitting."""

    def __init__(self, mols: list[MolSpec], settings: BondedSettings = BondedSettings()):
        self.s = settings
        self.mols = mols
        self.fams = [f for f in settings.families if f not in ("flux", "nnb")]
        for m in mols:
            if m.top is None:
                m.top = build_topology(m.elements, m.bonds, (m.bonds, m.bond_orders), m.ref_xyz * 10.0)
        self.nnb = None
        if "nnb" in settings.families:
            from .nn import NNBConfig, NNBonded

            self.nnb = NNBonded.for_molecules(mols, NNBConfig.from_settings(settings))
        # tying keys
        self.keyf = []
        for m in mols:
            if settings.typing == "molecule":
                cl = [m.name + "/" + c for c in _classes(m.elements, [tuple(b) for b in m.top.bonds], 8)]
            elif settings.typing == "amber":  # Amber / GAFF atom types (shared across molecules)
                if m.pgm is None:
                    raise ValueError(f"{m.name}: typing='amber' needs the pGM molecule (its atom types)")
                cl = list(m.pgm.types)
            else:
                cl = _classes(m.elements, [tuple(b) for b in m.top.bonds], settings.depth)
            self.keyf.append(
                lambda atoms, kind, cl=cl, top=m.top: top.key(atoms, kind, cl) if kind != "atom" else cl[atoms[0]]
            )
        # reference keys (bonds, angles) and per-molecule indices
        self.ref_keys = {"b0": [], "th0": []}
        self.I = []
        for m, kf in zip(mols, self.keyf):
            Im = {
                "bond": [self._key("b0", kf(b, "bond")) for b in m.top.bonds],
                "angle": [self._key("th0", kf(a, "angle")) for a in m.top.angles],
            }
            for f in self.fams:
                fam = T.REGISTRY[f]
                idx, keys = fam.index(m.top, kf)
                Im[f] = {k: np.asarray(v) for k, v in idx.items()}
                Im[f]["_keys"] = keys
            Im = {k: (np.asarray(v, int) if k in ("bond", "angle") else v) for k, v in Im.items()}
            self.I.append(Im)
        self.keys = {}
        for f in self.fams:
            pos = {}
            for Im in self.I:
                for k in Im[f]["_keys"]:
                    pos.setdefault(k, len(pos))
            self.keys[f] = list(pos)
            for Im in self.I:
                Im[f]["k"] = np.array([pos[k] for k in Im[f]["_keys"]], int)
        self._static_extras()

    def _key(self, which, key):
        pos = self.__dict__.setdefault("_ref_pos", {"b0": {}, "th0": {}})[which]
        if key not in pos:
            pos[key] = len(self.ref_keys[which])
            self.ref_keys[which].append(key)
        return pos[key]

    def _static_extras(self):
        # pGM Gaussian pair exponents for the overlap families
        for f in self.fams:
            fam = T.REGISTRY[f]
            if getattr(fam, "needs_radius", False):
                from ..kernels import gauss_bij

                for m, Im in zip(self.mols, self.I):
                    rad = (
                        np.asarray(System([m.pgm]).expand()["radius"])
                        if m.pgm is not None
                        else np.full(len(m.elements), 0.08)
                    )
                    pr = np.asarray(getattr(m.top, "pairs" + fam.which)).reshape(-1, 2)
                    Im[f]["bij"] = np.asarray(gauss_bij(rad[pr[:, 0]], rad[pr[:, 1]])) if len(pr) else np.zeros(0)
        # Morse depths per bond key (mean over instances of the table value)
        if "bond_morse" in self.fams:
            acc = {}
            for m, Im in zip(self.mols, self.I):
                order = {tuple(sorted(b)): o for b, o in zip(m.bonds, m.bond_orders)}
                for (i, j), k in zip(m.top.bonds, Im["bond_morse"]["k"]):
                    acc.setdefault(int(k), []).append(
                        T.morse_depth(m.elements[i], m.elements[j], order[tuple(sorted((i, j)))])
                    )
            De = np.array([np.mean(acc[k]) for k in range(len(self.keys["bond_morse"]))])
            for Im in self.I:
                Im["bond_morse"]["De"] = De[Im["bond_morse"]["k"]]

    # ------------------------------------------------------------------ parameters
    def init_params(self, rng_scale: float = 0.0, hold=()) -> dict:
        """Reference values from the reference geometries (key means), force constants at the
        families' defaults, the neural bonded network's initial weights."""
        b0 = np.zeros(len(self.ref_keys["b0"]))
        nb = np.zeros_like(b0)
        th0 = np.zeros(len(self.ref_keys["th0"]))
        na = np.zeros_like(th0)
        pr = {}
        for m, Im in zip(self.mols, self.I):
            G = jax.tree_util.tree_map(np.asarray, T.geometry(jnp.asarray(m.ref_xyz), m.top))
            np.add.at(b0, Im["bond"], G["b"])
            np.add.at(nb, Im["bond"], 1)
            np.add.at(th0, Im["angle"], G["th"])
            np.add.at(na, Im["angle"], 1)
            for f in self.fams:
                fam = T.REGISTRY[f]
                if hasattr(fam, "init_from_geometry"):  # per-instance values -> key means
                    for pname, v in fam.init_from_geometry(G, Im[f], m.top).items():
                        acc = pr.setdefault((f, pname), [np.zeros(len(self.keys[f])), np.zeros(len(self.keys[f]))])
                        np.add.at(acc[0], Im[f]["k"], v)
                        np.add.at(acc[1], Im[f]["k"], 1)
                elif f.startswith("pair") and "r0" in fam.params:
                    r = G["r" + fam.which]
                    acc = pr.setdefault((f, "r0"), [np.zeros(len(self.keys[f])), np.zeros(len(self.keys[f]))])
                    np.add.at(acc[0], Im[f]["k"], r)
                    np.add.at(acc[1], Im[f]["k"], 1)
        P = {"ref": {"b0": jnp.asarray(b0 / np.maximum(nb, 1)), "th0": jnp.asarray(th0 / np.maximum(na, 1))}}
        for f in self.fams:
            fam = T.REGISTRY[f]
            nk = len(self.keys[f])
            P[f] = {}
            for pname, (shape, init) in fam.params.items():
                if init is None:  # from the reference geometries
                    acc = pr[(f, pname)]
                    P[f][pname] = jnp.asarray(acc[0] / np.maximum(acc[1], 1))
                else:
                    P[f][pname] = jnp.full((nk,) + shape, float(init))
        if self.nnb is not None:
            P["nnb"] = self.nnb.init_params()
        return P

    def linear_mask(self, P) -> dict:
        """Pytree of booleans: True for parameters entering the energy linearly."""
        out = jax.tree_util.tree_map(lambda x: False, P)
        for f in self.fams:
            for pname in T.REGISTRY[f].linear:
                out[f][pname] = True
        return out

    # ------------------------------------------------------------------ energies
    def bonded_energy(self, m: int, R, P):
        mol, Im = self.mols[m], self.I[m]
        G = T.geometry(R, mol.top)
        b0 = P["ref"]["b0"][Im["bond"]]
        th0 = P["ref"]["th0"][Im["angle"]]
        dev = {"db": G["b"] - b0, "dc": G["cos"] - jnp.cos(th0), "dth": G["th"] - th0}
        e = 0.0
        for f in self.fams:
            fam = T.REGISTRY[f]
            I = Im[f]
            if len(I["k"]) == 0:
                continue
            p = {k: v[I["k"]] for k, v in P[f].items()}
            e = e + fam.energy(G, dev, I, p)
        if self.nnb is not None:
            e = e + self.nnb.energy(P["nnb"], m, R)
        return e

    def n_params(self, P, only_linear: bool = False) -> int:
        leaves = jax.tree_util.tree_leaves(P)
        return int(sum(np.size(x) for x in leaves))


class BondedModel(BondedTerms):
    """Bonded terms + gas-phase pGM (all pairs, induced dipoles) + intramolecular van der Waals:
    the model the bonded terms are fitted with (Fitter)."""

    def __init__(self, mols: list[MolSpec], settings: BondedSettings = BondedSettings()):
        super().__init__(mols, settings)
        self.pd, self.ind = elec_flags(settings.elec)
        check_vdw(settings.vdw, settings.gvdw_rep)
        self._nonbonded_setup()

    def init_params(self, rng_scale: float = 0.0, hold=()) -> dict:
        """BondedTerms.init_params plus charge / dipole flux, learned pair scales and fitted
        charges or bond-charge increments when the settings ask for them (fitted pGM charges /
        covalent dipoles start at the type means of the ESP fits over the molecules not in `hold`)."""
        P = super().init_params(rng_scale, hold)
        if self.s.flux:
            P["flux"] = {"jb": jnp.zeros(len(self.ref_keys["b0"])), "jc": jnp.zeros(len(self.ref_keys["b0"]))}
            if int(self.s.flux) >= 2:  # quadratic covalent-dipole flux (field-responsive bonds)
                P["flux"]["jc2"] = jnp.zeros(len(self.ref_keys["b0"]))
        if self.s.escale:
            P["escale"] = {"kappa": jnp.zeros(len(self.es_pos))}
        if self.s.qbci >= 0:
            P["bci"] = {"t": jnp.zeros(len(self.t_pos)), "dc": jnp.zeros(len(self.dc_pos))}
        if self.s.qfit >= 0:  # start from the ESP-fitted values, averaged per type

            def mean(v):
                w = [x for i, x in v if i not in hold]
                return np.mean(w if w else [x for _, x in v])

            P["elec"] = {
                "q": jnp.asarray([mean(self.q_acc[k]) for k in self.q_keys]),
                "c": jnp.asarray([mean(self.c_acc[k]) for k in self.c_keys]),
            }
        return P

    def linear_mask(self, P) -> dict:
        out = super().linear_mask(P)
        if "escale" in P:
            out["escale"]["kappa"] = True
        return out

    def _nonbonded_setup(self):
        dens = DENSITIES["gaussian"]
        self._phi, self._bij = dens["coulomb"], dens["pair_exponent"]
        self.nb = []
        self.es_pos = {}
        self.q_acc, self.c_acc = {}, {}
        self.t_pos, self.dc_pos = {}, {}
        for mi, m in enumerate(self.mols):
            if m.pgm is None:
                self.nb.append(None)
                continue
            sys = System([m.pgm])
            D = m.top.dist
            ii, jj = sys.pair_i, sys.pair_j

            def w_of(d):
                return (d > self.s.elec_exclude) * np.where(d == 3, self.s.elec14_scale, 1.0)

            w_pair = w_of(D[ii, jj]).astype(float)
            oi, oj = np.nonzero(~np.eye(sys.n, dtype=bool))
            ie = self.s.elec_exclude if self.s.ind_exclude < 0 else self.s.ind_exclude

            def w_ind(d):
                return (d > ie) * np.where(d == 3, self.s.elec14_scale, 1.0)

            w_ord = w_ind(D[oi, oj]).astype(float)
            w_pind = w_ind(D[ii, jj]).astype(float)
            lj = (D[ii, jj] >= self.s.lj_min_sep).astype(float) + self.s.lj14_scale * (D[ii, jj] == 3)
            # covalent dipoles along bonds: their bond reference index (for dipole flux)
            bidx = {tuple(sorted(b)): k for k, b in enumerate(m.top.bonds)}
            cov_bond = np.array([bidx.get(tuple(sorted((i, j))), -1) for i, j in zip(sys.cov_i, sys.cov_j)], int)
            rec = {
                "sys": sys,
                "w_pair": w_pair,
                "w_pind": w_pind,
                "oi": oi,
                "oj": oj,
                "w_ord": w_ord,
                "lj": lj,
                "cov_bond": cov_bond,
            }
            if self.s.qbci >= 0:
                cl = _classes(m.elements, [tuple(b) for b in m.top.bonds], self.s.qbci)
                bi, bj = np.asarray(m.top.bonds).T
                bk = ["t|" + "-".join(sorted([cl[i], cl[j]])) for i, j in zip(bi, bj)]
                bsign = np.array([0.0 if cl[i] == cl[j] else (1.0 if cl[i] < cl[j] else -1.0) for i, j in zip(bi, bj)])
                ck = ["dc|" + cl[i] + ">" + cl[j] for i, j in zip(sys.cov_i, sys.cov_j)]
                for k in bk:
                    self.t_pos.setdefault(k, len(self.t_pos))
                for k in ck:
                    self.dc_pos.setdefault(k, len(self.dc_pos))
                rec.update(
                    t_idx=np.array([self.t_pos[k] for k in bk], int),
                    t_sign=bsign,
                    t_i=bi,
                    t_j=bj,
                    dc_idx=np.array([self.dc_pos[k] for k in ck], int),
                )
            if self.s.qfit >= 0:
                cl = _classes(m.elements, [tuple(b) for b in m.top.bonds], self.s.qfit)
                Q0 = sys.expand()
                qk = ["q|" + c for c in cl]
                ck = ["c|" + cl[i] + ">" + cl[j] for i, j in zip(sys.cov_i, sys.cov_j)]
                for k, v in zip(qk, np.asarray(Q0["q"])):
                    self.q_acc.setdefault(k, []).append((mi, float(v)))
                for k, v in zip(ck, np.asarray(Q0["cov"])):
                    self.c_acc.setdefault(k, []).append((mi, float(v)))
                rec.update(q_keys=qk, c_keys=ck, q_total=float(np.sum(np.asarray(Q0["q"]))))
            if self.s.escale:
                kf = self.keyf[mi]
                sel = np.nonzero(np.isin(D[ii, jj], self.s.escale))[0]
                keys = [
                    f"{D[ii[k], jj[k]]}|" + "-".join(sorted([kf([int(ii[k])], "atom"), kf([int(jj[k])], "atom")]))
                    for k in sel
                ]
                loc = {}
                for k in keys:
                    loc.setdefault(k, len(loc))
                for k in loc:
                    self.es_pos.setdefault(k, len(self.es_pos))
                rec.update(
                    es_sel=sel,
                    es_loc=np.array([loc[k] for k in keys], int),
                    es_n=len(loc),
                    es_glob=np.array([self.es_pos[k] for k in loc], int),
                )
            self.nb.append(rec)
        if self.s.qfit >= 0:
            self.q_keys, self.c_keys = list(self.q_acc), list(self.c_acc)
            qpos = {k: i for i, k in enumerate(self.q_keys)}
            cpos = {k: i for i, k in enumerate(self.c_keys)}
            for d in self.nb:
                if d is not None:
                    d["q_idx"] = np.array([qpos[k] for k in d["q_keys"]], int)
                    d["c_idx"] = np.array([cpos[k] for k in d["c_keys"]], int)

    @property
    def nb_dynamic(self) -> bool:
        """True when the nonbonded part depends on fitted parameters (no per-frame cache)."""
        return bool(self.s.flux or self.s.qfit >= 0 or self.s.qbci >= 0)

    def nonbonded(self, m: int, R, P=None, eparams=None, state=False):
        """pGM electrostatics + LJ: (energy kJ/mol, molecular dipole (3,) e nm about the origin)
        [, {q, p, mu, radius} with state=True]."""
        d = self.nb[m]
        if d is None:
            return 0.0, jnp.zeros(3)
        sys = d["sys"]
        Q = sys.expand(eparams)
        q, cov = Q["q"], Q["cov"]
        if self.s.qfit >= 0 and P is not None and "elec" in P:
            q = P["elec"]["q"][d["q_idx"]]
            q = q + (d["q_total"] - jnp.sum(q)) / sys.n
            cov = P["elec"]["c"][d["c_idx"]]
        if self.s.qbci >= 0 and P is not None and "bci" in P:
            t = d["t_sign"] * P["bci"]["t"][d["t_idx"]]  # charge moved from atom i to atom j of each bond
            q = q.at[d["t_i"]].add(-t).at[d["t_j"]].add(t)
            cov = cov + P["bci"]["dc"][d["dc_idx"]]
        if self.s.flux and P is not None and "flux" in P:
            q, cov = self._flux(m, R, P, q, cov)
        R_ = Q["radius"]
        p = perm_dipoles(R, sys, cov) if self.pd else jnp.zeros((sys.n, 3))
        ii, jj = sys.pair_i, sys.pair_j
        phi = self._phi
        b_pair = self._bij(R_[ii], R_[jj])
        e_perm = jnp.sum(
            d["w_pair"]
            * jax.vmap(lambda a, c, qa, pa, qc, pc, bb: _pair_perm(a, c, qa, pa, qc, pc, bb, phi))(
                R[ii], R[jj], q[ii], p[ii], q[jj], p[jj], b_pair
            )
        )
        Th = None
        if self.s.quadrupoles:
            Th = build_quadrupoles(R, sys, Q["quad"])
            x = R[ii] - R[jj]
            B = erf_kernels(b_pair, jnp.linalg.norm(x, axis=-1), 5)
            e_perm = e_perm + jnp.sum(
                d["w_pair"] * quadrupole_pair_terms(x, B, q[ii], p[ii], Th[ii], q[jj], p[jj], Th[jj])
            )
        if self.ind:
            oi, oj = d["oi"], d["oj"]
            b_ord = self._bij(R_[oi], R_[oj])
            F_ord = jax.vmap(lambda a, c, qc, pc, bb: _field_at_i(a, c, qc, pc, bb, phi))(
                R[oi], R[oj], q[oj], p[oj], b_ord
            )
            if Th is not None:
                F_ord = F_ord + quadrupole_field(R[oi] - R[oj], b_ord, Th[oj])
            F = jnp.zeros((sys.n, 3)).at[oi].add(d["w_ord"][:, None] * F_ord)
            Tp = (
                jax.vmap(lambda a, c, bb: _dipole_tensor(a, c, bb, phi))(R[ii], R[jj], b_pair)
                * d["w_pind"][:, None, None]
            )
            Tm = jnp.zeros((sys.n, sys.n, 3, 3)).at[ii, jj].set(Tp).at[jj, ii].set(jnp.swapaxes(Tp, 1, 2))
            mu = solve_linear_induction(Tm, Q["alpha"], F)
            e_ind = -0.5 * jnp.sum(mu * F)
        else:
            mu, e_ind = jnp.zeros((sys.n, 3)), 0.0
        e_lj = self._vdw(R, Q, d["lj"], ii, jj, b_pair)
        dip = jnp.sum(q[:, None] * R, 0) + jnp.sum(p, 0) + jnp.sum(mu, 0)
        if state:
            return KE * (e_perm + e_ind) + e_lj, dip, {"q": q, "p": p, "mu": mu, "radius": R_, "Theta": Th}
        return KE * (e_perm + e_ind) + e_lj, dip

    def _vdw(self, R, Q, w, ii, jj, b_pair):
        """Intramolecular van der Waals with pair weights w (lj_min_sep, lj14_scale), kJ/mol."""
        if self.s.vdw == "none":
            return 0.0
        r = jnp.linalg.norm(R[ii] - R[jj], axis=-1)
        if self.s.vdw == "lj":
            s6 = ((Q["lj_rmin_half"][ii] + Q["lj_rmin_half"][jj]) / r) ** 6
            return jnp.sum(w * Q["lj_sqrt_eps"][ii] * Q["lj_sqrt_eps"][jj] * (s6 * s6 - 2.0 * s6))
        A = Q["gvdw_sqrt_a"][ii] * Q["gvdw_sqrt_a"][jj]
        C6 = Q["gvdw_sqrt_c6"][ii] * Q["gvdw_sqrt_c6"][jj]
        B = 0.5 * (Q["gvdw_b"][ii] + Q["gvdw_b"][jj])
        return jnp.sum(w * gvdw_pair(r, b_pair, A, C6, B, self.s.gvdw_rep))

    def esp(self, m: int, R, grid, P=None, eparams=None):
        """Electrostatic potential (hartree/e) of the polarised pGM molecule at grid points (k, 3) nm:
        Gaussian charges and permanent + induced dipoles, as in the py_resp pGM-perm fit."""
        _, _, st = self.nonbonded(m, R, P, eparams, state=True)
        b = self._bij(st["radius"], 0.0)
        phi = self._phi

        def f(a, c, bb):
            return phi(jnp.linalg.norm(a - c), bb)

        pt = st["p"] + st["mu"]

        def at(g):
            def one(rj, qj, pj, bb):
                return qj * f(g, rj, bb) + pj @ jax.grad(f, 1)(g, rj, bb)

            v = jnp.sum(jax.vmap(one)(R, st["q"], pt, b))
            if st["Theta"] is not None:  # (1/3)(x Th x) B2, x = g - r_j
                x = g[None] - R
                B2 = erf_kernels(b, jnp.linalg.norm(x, axis=-1), 3)[2]
                v = v + jnp.sum(jnp.einsum("pa,pab,pb->p", x, st["Theta"], x) * B2) / 3.0
            return v

        return KE * jax.vmap(at)(grid) / HARTREE_KJMOL

    def _flux(self, m, R, P, q, cov):
        """Bond charge flux (charge moves along a bond as it stretches, from the first atom of
        the bond key's canonical order to the second) and covalent-dipole flux c = c0 + jc db."""
        mol, Im, d = self.mols[m], self.I[m], self.nb[m]
        G = T.geometry(R, mol.top)
        db = G["b"] - P["ref"]["b0"][Im["bond"]]
        cl = self.keyf[m]
        sign = np.array([1.0 if cl([i], "atom") <= cl([j], "atom") else -1.0 for i, j in mol.top.bonds])
        same = np.array([cl([i], "atom") == cl([j], "atom") for i, j in mol.top.bonds])
        t = jnp.where(same, 0.0, sign * P["flux"]["jb"][Im["bond"]] * db)
        q = q.at[mol.top.bonds[:, 0]].add(-t).at[mol.top.bonds[:, 1]].add(t)
        cb = d["cov_bond"]
        if len(cb):
            d_ = db[np.maximum(cb, 0)]
            kb = Im["bond"][np.maximum(cb, 0)]
            dc_ = P["flux"]["jc"][kb] * d_
            if "jc2" in P["flux"]:
                dc_ = dc_ + P["flux"]["jc2"][kb] * d_ * d_
            dcov = jnp.where(cb >= 0, dc_, 0.0)
            cov = cov + dcov
        return q, cov

    def escale_terms(self, m: int, R, eparams=None):
        """Permanent pGM pair energies (kJ/mol) summed per local pair class of the learned scales."""
        d = self.nb[m]
        sys = d["sys"]
        Q = sys.expand(eparams)
        p = perm_dipoles(R, sys, Q["cov"])
        sel = d["es_sel"]
        ii, jj = sys.pair_i[sel], sys.pair_j[sel]
        b = self._bij(Q["radius"][ii], Q["radius"][jj])
        phi = self._phi
        e = jax.vmap(lambda a, c, qa, pa, qc, pc, bb: _pair_perm(a, c, qa, pa, qc, pc, bb, phi))(
            R[ii], R[jj], Q["q"][ii], p[ii], Q["q"][jj], p[jj], b
        )
        return KE * jax.ops.segment_sum(e, d["es_loc"], num_segments=d["es_n"])

    def energy(self, m: int, R, P, eparams=None):
        e_nb, dip = self.nonbonded(m, R, P, eparams)
        e = self.bonded_energy(m, R, P) + e_nb
        if self.s.escale and "escale" in P and self.nb[m] is not None:
            e = e + jnp.sum(P["escale"]["kappa"][self.nb[m]["es_glob"]] * self.escale_terms(m, R, eparams))
        return e, dip
