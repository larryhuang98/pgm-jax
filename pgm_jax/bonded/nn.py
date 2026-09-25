"""NNB: fast neural bonded terms for pGM molecules ("nn" term set).

The network replaces *atom typing*, not the energy function: the bonded energy keeps the form of
a chosen set of physical term families (default: the class II set, terms.PAPER; any families of
terms.REGISTRY work), and a graph network predicts the parameters of every term instance.  At MD
time the parameters are fixed tables, so the speed is that of the classical family set.

  Stage 1, topology -> parameters (once per molecule or residue template).  Message passing over
  the bond graph gives atom embeddings h_i from the element, degree, bond orders, ring and
  aromatic flags and the atom's pGM electrostatic parameters (charge, polarizability, Gaussian
  width, covalent-dipole strength): the bonded model knows the all-pair electrostatic model it
  complements.  Every term instance is then described exactly as a typed force field would key
  it (the families' own `index(top, keyf)` is called with a recording keyf), but with the atom
  classes replaced by symmetric readouts of the embeddings:
      atom a              h_a
      bond / pair (i, j)  [h_i + h_j, h_i * h_j]
      angle (i, j, k)     [h_j, h_i + h_k, h_i * h_k]
      torsion (i,j,k,l)   [h_j + h_k, h_j * h_k, h_i + h_l, h_i * h_l, h_i * h_j + h_l * h_k]
      improper (c;a,b,d)  [h_c, sum h_x, sum_{x<y} h_x * h_y]
  concatenated in the order of the key's components (so oriented couplings, e.g. bond-angle with
  a given outer atom, stay oriented), plus the key's literal skeleton.  One small MLP head per
  family maps this to the family's parameters (init + scale * output, output layers start at
  zero: the untrained model is the family set at its default values); two more heads give
  corrections to the bond and angle reference values.

  Stage 2, coordinates -> energy: the families' own energy functions with per-instance
  parameters.  `freeze` evaluates stage 1 once (FlexibleTemplate does it for MD).

Reference values: `ref="geometry"` starts r0 and th0 from the molecule's minimum geometry (from
MACE-OFF or DFT, available for any new molecule); `ref="predicted"` from covalent radii and
hybridization angles.  Trained with pGM nonbonded in the loop (bonded/fit.py, Adam then L-BFGS).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from . import terms as T

from .fit import SCALES
ELEMENTS = ("H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I")
# Pyykko covalent radii (A): single, double, triple
_RCOV = {"H": (0.32, 0.32, 0.32), "C": (0.75, 0.67, 0.60), "N": (0.71, 0.60, 0.54), "O": (0.63, 0.57, 0.53),
         "F": (0.64, 0.59, 0.53), "P": (1.11, 1.02, 0.94), "S": (1.03, 0.94, 0.95), "Cl": (0.99, 0.95, 0.93),
         "Br": (1.14, 1.09, 1.10), "I": (1.33, 1.29, 1.25)}
K0_BOND, K0_ANGLE = 2.5e5, 400.0


def _rcov(e, order):
    r1, r2, r3 = _RCOV.get(e, (0.9, 0.85, 0.8))
    if order <= 1:
        return r1
    if order < 2:
        return r1 + (order - 1) * (r2 - r1)
    return r2 if order < 3 else r3


# ------------------------------------------------------------------ small MLPs (pure JAX)
def _init_mlp(key, n_in, n_hid, n_out, zero_out=False, scale_out=1.0):
    k1, k2 = jax.random.split(key)
    w1 = jax.random.normal(k1, (n_in, n_hid)) * math.sqrt(2.0 / (n_in + n_hid))
    w2 = jnp.zeros((n_hid, n_out)) if zero_out else jax.random.normal(k2, (n_hid, n_out)) * math.sqrt(2.0 / (n_hid + n_out)) * scale_out
    return {"w1": w1, "b1": jnp.zeros(n_hid), "w2": w2, "b2": jnp.zeros(n_out)}


def _mlp(p, x):
    return jax.nn.silu(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]


class NNBonded:
    """Stage 1 + stage 2 for a list of MolSpecs (their topologies must be built)."""

    def __init__(self, mols, width: int = 32, layers: int = 3, ref: str = "geometry", basis=T.PAPER,
                 b_span: float = 0.01, th_span: float = 0.35, out_scale: float = 2.0, pgm_features: bool = True):
        if ref not in ("geometry", "predicted"):
            raise ValueError("ref: geometry | predicted")
        for f in basis:
            fam = T.REGISTRY[f]
            if any(init is None for _, init in fam.params.values()):
                raise ValueError(f"family {f} initialises from the geometry; not supported as an NNB basis yet")
        self.W, self.L, self.ref, self.basis = width, layers, ref, tuple(basis)
        self.b_span, self.th_span, self.out_scale = b_span, th_span, out_scale
        self.pgm_features = pgm_features
        self.mols = mols
        self.data = [self._prepare(m) for m in mols]
        self.n_feat = self.data[0]["X"].shape[1]
        # heads: per family, skeleton vocabulary and slot count over all molecules
        self.skel, self.slots = {}, {}
        for d in self.data:
            for f, rec in d["fam"].items():
                voc = self.skel.setdefault(f, [])
                for sk in rec["skeletons"]:
                    if sk not in voc:
                        voc.append(sk)
                self.slots[f] = max(self.slots.get(f, 1), rec["n_slots"])
        for d in self.data:
            for f, rec in d["fam"].items():
                rec["skel_onehot"] = np.eye(len(self.skel[f]))[[self.skel[f].index(k) for k in rec["skel_of"]]] \
                    if len(rec["skel_of"]) else np.zeros((0, len(self.skel[f])))

    # ------------------------------------------------------------------ topology tables
    def _prepare(self, spec):
        top = spec.top
        n = len(spec.elements)
        order = {tuple(sorted(b)): float(o) for b, o in zip(spec.bonds, spec.bond_orders)}
        nbr = [[] for _ in range(n)]
        for i, j in top.bonds:
            nbr[i].append(j); nbr[j].append(i)
        bo_sum = np.zeros(n)
        for (i, j), o in order.items():
            bo_sum[i] += o; bo_sum[j] += o
        from .topology import _ring_bonds
        ring = np.zeros(n)
        for rb in _ring_bonds([tuple(x) for x in top.bonds], nbr):
            for x in rb:
                ring[x] = 1.0
        pg = spec.pgm
        cov_abs = np.zeros(n)
        if pg is not None:
            for i, j, c in pg.cov:
                cov_abs[i] += abs(c)
        X = []
        for i, e in enumerate(spec.elements):
            f = [1.0 if e == x else 0.0 for x in ELEMENTS] + [1.0 if len(nbr[i]) == d else 0.0 for d in (1, 2, 3, 4)]
            f += [bo_sum[i] / 4.0, ring[i],
                  1.0 if any(abs(order.get(tuple(sorted((i, j))), 1) - 1.5) < 1e-6 for j in nbr[i]) else 0.0]
            f += [float(pg.q[i]), float(pg.alpha[i]) * 1e3, float(pg.radius[i]) * 10.0, cov_abs[i] * 50.0] \
                if (pg is not None and self.pgm_features) else [0.0] * 4
            X.append(f)
        src = np.concatenate([top.bonds[:, 0], top.bonds[:, 1]])
        dst = np.concatenate([top.bonds[:, 1], top.bonds[:, 0]])
        bo = np.array([order.get(tuple(sorted(b)), 1.0) for b in top.bonds])
        ef = np.stack([bo == 1, np.abs(bo - 1.5) < 1e-6, bo == 2, bo == 3], -1).astype(float)
        ef = np.concatenate([ef, ef])
        if self.ref == "geometry":
            G = T.geometry(jnp.asarray(spec.ref_xyz), top)
            b_ref, th_ref = np.asarray(G["b"]), np.asarray(G["th"])
        else:
            b_ref = np.array([(_rcov(spec.elements[i], bo[k]) + _rcov(spec.elements[j], bo[k])) * 0.1
                              for k, (i, j) in enumerate(top.bonds)])
            tet = math.acos(-1 / 3)
            th_ref = np.array([math.pi if (len(nbr[j]) == 2 and bo_sum[j] >= 3.5) else
                               (2 * math.pi / 3 if (len(nbr[j]) == 3 and bo_sum[j] >= 3.5) else tet)
                               for i, j, k in top.angles])
        # every family's instances, described by the components of their tying keys
        fams = {}
        for f in self.basis:
            log = []

            def keyf(atoms, kind, log=log):
                log.append((kind, tuple(int(a) for a in atoms)))
                return f"\x00{len(log) - 1}\x00"
            idx, keys = T.REGISTRY[f].index(top, keyf)
            comps, skel_of = [], []
            for k in keys:
                parts = k.split("\x00")
                ids = [int(x) for x in parts[1::2]]
                comps.append([log[i] for i in ids])
                skel_of.append("X".join(parts[0::2]))
            n_slots = max([len(c) for c in comps] + [1])
            groups = {}                                        # (slot, kind, n_atoms) -> (instances, atoms)
            for inst, c in enumerate(comps):
                for slot, (kind, atoms) in enumerate(c):
                    g = groups.setdefault((slot, kind, len(atoms)), ([], []))
                    g[0].append(inst); g[1].append(atoms)
            fams[f] = {"I": {k: np.asarray(v) for k, v in idx.items()}, "n": len(keys), "n_slots": n_slots,
                       "skel_of": skel_of, "skeletons": sorted(set(skel_of)),
                       "groups": {k: (np.asarray(v[0], int), np.asarray(v[1], int)) for k, v in groups.items()}}
        return {"X": np.asarray(X, float), "src": src, "dst": dst, "ef": ef, "n": n, "top": top,
                "bonds": np.asarray(top.bonds), "angles": np.asarray(top.angles).reshape(-1, 3),
                "b_ref": b_ref, "th_ref": th_ref, "fam": fams}

    # ------------------------------------------------------------------ parameters
    def _n_out(self, f):
        return int(sum(int(np.prod(shape)) if shape else 1 for shape, _ in T.REGISTRY[f].params.values()))

    def init_params(self, seed: int = 0) -> dict:
        W = self.W
        ks = list(jax.random.split(jax.random.PRNGKey(seed), 4 + 2 * self.L + len(self.basis)))
        P = {"embed": _init_mlp(ks.pop(), self.n_feat, W, W)}
        for l in range(self.L):
            P[f"mp{l}"] = {"msg": _init_mlp(ks.pop(), 2 * W + 4, W, W, scale_out=0.3),
                           "upd": _init_mlp(ks.pop(), 2 * W, W, W, scale_out=0.3)}
        P["ref_bond"] = _init_mlp(ks.pop(), 2 * W + 1, W, 1, zero_out=True)
        P["ref_angle"] = _init_mlp(ks.pop(), 3 * W + 1, W, 1, zero_out=True)
        for f in self.basis:
            n_in = self.slots.get(f, 1) * 5 * W + len(self.skel.get(f, []))
            P["head_" + f] = _init_mlp(ks.pop(), n_in, W, self._n_out(f), zero_out=True)
        return P

    def embeddings(self, P, m):
        d = self.data[m]
        h = _mlp(P["embed"], jnp.asarray(d["X"]))
        src, dst, ef = d["src"], d["dst"], jnp.asarray(d["ef"])
        for l in range(self.L):
            msg = _mlp(P[f"mp{l}"]["msg"], jnp.concatenate([h[dst], h[src], ef], -1))
            agg = jax.ops.segment_sum(msg, dst, num_segments=d["n"])
            h = h + _mlp(P[f"mp{l}"]["upd"], jnp.concatenate([h, agg], -1))
        return h

    @staticmethod
    def readout(kind, h, atoms):
        """Symmetric description of a key component: atoms (k, n_atoms) -> (k, <= 5W)."""
        H = [h[atoms[:, a]] for a in range(atoms.shape[1])]
        if len(H) == 1:
            return H[0]
        if kind in ("bond", "pair") or len(H) == 2:
            return jnp.concatenate([H[0] + H[1], H[0] * H[1]], -1)
        if kind == "angle" and len(H) == 3:
            return jnp.concatenate([H[1], H[0] + H[2], H[0] * H[2]], -1)
        if kind == "torsion" and len(H) == 4:
            i, j, k, l = H
            return jnp.concatenate([j + k, j * k, i + l, i * l, i * j + l * k], -1)
        if kind == "improper" and len(H) == 4:
            c, a, b, d = H
            return jnp.concatenate([c, a + b + d, a * b + a * d + b * d], -1)
        S = sum(H)
        return jnp.concatenate([H[0], S, sum(x * y for p, x in enumerate(H) for y in H[p + 1:])], -1)

    def coefficients(self, P, m) -> dict:
        """Stage 1: reference values and per-instance family parameters of molecule m."""
        d = self.data[m]
        h = self.embeddings(P, m)
        W = self.W
        b, a = d["bonds"], d["angles"]
        # the corrections see the reference value itself (in the "geometry" mode it distinguishes
        # graph-equivalent bonds/angles that the minimum geometry does not treat alike)
        br, tr = jnp.asarray(d["b_ref"]), jnp.asarray(d["th_ref"])
        fb = jnp.concatenate([self.readout("bond", h, b), ((br - 0.12) / 0.03)[:, None]], -1)
        C = {"b0": br + self.b_span * jnp.tanh(_mlp(P["ref_bond"], fb)[:, 0])}
        if len(a):
            fa = jnp.concatenate([self.readout("angle", h, a), ((tr - 1.91) / 0.2)[:, None]], -1)
            C["th0"] = tr + self.th_span * jnp.tanh(_mlp(P["ref_angle"], fa)[:, 0])
        else:
            C["th0"] = jnp.zeros(0)
        for f in self.basis:
            rec = d["fam"][f]
            n = rec["n"]
            if n == 0:
                continue
            feat = jnp.zeros((n, self.slots[f] * 5 * W))
            for (slot, kind, na), (inst, atoms) in rec["groups"].items():
                r = self.readout(kind, h, atoms)
                feat = feat.at[inst, slot * 5 * W: slot * 5 * W + r.shape[1]].set(r)
            feat = jnp.concatenate([feat, jnp.asarray(rec["skel_onehot"])], -1)
            o = _mlp(P["head_" + f], feat)
            p, c0 = {}, 0
            for pname, (shape, init) in T.REGISTRY[f].params.items():
                size = int(np.prod(shape)) if shape else 1
                scale = self.out_scale * SCALES.get(pname, 1.0)
                p[pname] = float(init) + scale * o[:, c0:c0 + size].reshape((n,) + tuple(shape))
                c0 += size
            C[f] = p
        return C

    # ------------------------------------------------------------------ stage 2
    def energy_from(self, C, m, R):
        """Stage 2: bonded energy (kJ/mol) of coordinates R (n, 3) nm with stage-1 output C."""
        d = self.data[m]
        G = T.geometry(R, d["top"])
        dev = {"db": G["b"] - C["b0"], "dc": G["cos"] - jnp.cos(C["th0"]), "dth": G["th"] - C["th0"]}
        e = 0.0
        for f in self.basis:
            if f in C:
                I = dict(d["fam"][f]["I"])
                I.setdefault("De", self._de(m))
                e = e + T.REGISTRY[f].energy(G, dev, I, C[f])
        return e

    def _de(self, m):
        d = self.data[m]
        if "De" not in d:
            spec = self.mols[m]
            order = {tuple(sorted(b)): float(o) for b, o in zip(spec.bonds, spec.bond_orders)}
            d["De"] = np.asarray([T.morse_depth(spec.elements[i], spec.elements[j], order.get(tuple(sorted((i, j))), 1))
                                   for i, j in d["bonds"]])
        return d["De"]

    def energy(self, P, m, R):
        """Bonded energy with the network parameters P (or frozen tables {"coef": [...]})."""
        C = P["coef"][m] if "coef" in P else self.coefficients(P, m)
        return self.energy_from(C, m, R)

    def freeze(self, P) -> dict:
        """Stage 1 evaluated once: {"coef": per-molecule tables} (MD speed)."""
        return {"coef": [self.coefficients(P, m) for m in range(len(self.mols))]}
