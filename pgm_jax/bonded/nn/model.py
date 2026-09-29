"""Define the neural bonded model: configuration, vocabulary, stage 1 (parameters), stage 2 (energy).

Separation that makes a trained network reusable on new molecules (train on peptides, apply to a
protein): the *configuration* (widths, basis families, options) and the *vocabulary* (key
skeletons, component slots, typed-table keys: everything that fixes parameter shapes) belong to
the network and are saved with its weights; the *tables* of a molecule (graph inputs, instances)
are built for any molecule against a frozen vocabulary (`prepare`).

    net = NNBonded.for_molecules(train_specs, NNBConfig(basis=T.PROTEIN))   # vocabulary from the training set
    P = net.init_params()                     # ... fit through BondedModel / Fitter ...
    net.save("nnb.pkl", P)
    net, P = NNBonded.load("nnb.pkl")
    C = net.coefficients(P, net.prepare(protein_spec))    # frozen per-instance parameters

Units: reference values nm and rad, energies kJ/mol; family parameters in the units of their
families (terms/).
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from .. import terms as T
from ..fit import SCALES
from .features import N_EDGE, N_FEAT, graph_inputs
from .instances import CONTEXT_ATOMS, SLOT, decompose, readout, typed_keys
from .layers import embeddings, init_message_passing, init_mlp, mlp

if TYPE_CHECKING:
    from ..model import BondedSettings, MolSpec


@dataclass(frozen=True)
class NNBConfig:
    """Configuration of the neural bonded network (a frozen dataclass, saved with the weights).

    Parameters
    ----------
    width : int
        Embedding width W.
    layers : int
        Message-passing layers.
    ref : {"geometry", "predicted"}
        Reference values: minimum geometry + learned shift, or covalent radii / hybridisation.
    basis : tuple of str
        Families whose per-instance parameters the network predicts.
    b_span : float
        Largest learned shift of bond reference values [nm].
    th_span : float
        Largest learned shift of angle reference values [rad].
    out_scale : float
        Head output -> parameter change, in units of fit.SCALES.
    pgm_features : bool
        Atom features include the pGM q, alpha, radius, |covalent dipoles|.
    table_depth : int or None
        Typed table (atom environments to this depth; 0 = elements) + residual; None: network only.
    resid_l2 : float
        Shrinkage of the residual towards the typed table (training loss).
    context : bool
        Sequence context (residues i-1, i, i+1) for CONTEXT_ATOMS families.
    """

    width: int = 32  # embedding width W
    layers: int = 3  # message-passing layers
    ref: str = "geometry"  # reference values: "geometry" (minimum + learned shift) | "predicted"
    basis: tuple = T.PAPER  # families whose per-instance parameters the network predicts
    b_span: float = 0.01  # largest learned shift of bond reference values (nm)
    th_span: float = 0.35  # ... of angle reference values (rad)
    out_scale: float = 2.0  # head output -> parameter change, in units of fit.SCALES
    pgm_features: bool = True  # atom features include the pGM q, alpha, radius, |covalent dipoles|
    table_depth: int | None = None  # typed table (atom environments to this depth; 0 = elements) + residual
    resid_l2: float = 0.0  # shrinkage of the residual towards the typed table (training loss)
    context: bool = True  # sequence context (residues i-1, i, i+1) for CONTEXT_ATOMS families

    @classmethod
    def from_settings(cls, s: BondedSettings) -> NNBConfig:
        """Return the configuration from BondedSettings (its nn_* fields)."""
        return cls(
            width=s.nn_width,
            layers=s.nn_layers,
            ref=s.nn_ref,
            basis=tuple(s.nn_basis),
            b_span=s.nn_b_span,
            th_span=s.nn_th_span,
            out_scale=s.nn_out_scale,
            pgm_features=s.nn_pgm_features,
            table_depth=s.nn_table_depth,
            resid_l2=s.nn_resid_l2,
            context=getattr(s, "nn_context", True),
        )


@dataclass
class Vocabulary:
    """Everything that fixes the parameter shapes (a mutable dataclass, saved with the weights).

    It grows while training molecules are added and is frozen by `NNBonded.init_params` (later
    molecules must fit it).

    Parameters
    ----------
    skeletons : dict
        {family: [skeleton, ...]}: key skeletons seen (one-hot inputs of the heads).
    slots : dict
        {family: components per key}.
    table : dict
        {family / "b0" / "th0": {typed key: row}}: rows of the typed tables.
    frozen : bool
        No more growth.
    """

    skeletons: dict = field(default_factory=dict)  # family -> [skeleton, ...]
    slots: dict = field(default_factory=dict)  # family -> components per key
    table: dict = field(default_factory=dict)  # family / "b0" / "th0" -> {typed key: row}
    frozen: bool = False


def _n_out(f: str) -> int:
    """Return the number of scalar parameters of family f (output width of its head)."""
    return int(sum(int(np.prod(shape)) if shape else 1 for shape, _ in T.REGISTRY[f].params.values()))


class NNBonded:
    """The neural bonded model: topology -> per-instance parameters (stage 1) -> energy (stage 2).

    See bonded/nn/__init__.py for the design.  Parameters P are a dict pytree of MLP weights:
    "embed", "mp{l}" (message passing), "ref_bond", "ref_angle" (reference-value heads), "head_<f>"
    per basis family, "res" (sequence context) and "tab_<name>" (typed tables); or the frozen form
    {"coef": [stage-1 output per registered molecule]}.

    Attributes
    ----------
    config : NNBConfig
        Configuration.
    vocab : Vocabulary
        Vocabulary.
    mols : list of MolSpec
        Registered molecules.
    data : list of dict
        Their tables (`prepare`).
    """

    def __init__(self, config: NNBConfig = NNBConfig(), vocab: Vocabulary | None = None) -> None:
        """Set up an empty network (no molecules).

        Parameters
        ----------
        config : NNBConfig
            Configuration.
        vocab : Vocabulary, optional
            A saved vocabulary; None: an empty, growing one.

        Raises
        ------
        ValueError
            For an unknown basis family, a family that initialises from the geometry, or an unknown
            `ref`.
        """
        for f in config.basis:
            if f not in T.REGISTRY:
                raise ValueError(f"unknown family {f!r}")
            if any(init is None for _, init in T.REGISTRY[f].params.values()):
                raise ValueError(f"family {f} initialises from the geometry; not supported as an NNB basis yet")
        if config.ref not in ("geometry", "predicted"):
            raise ValueError("ref: geometry | predicted")
        self.config = config
        self.vocab = vocab or Vocabulary()
        self.mols, self.data = [], []  # registered molecules and their tables

    # ------------------------------------------------------------------ molecules
    @classmethod
    def for_molecules(cls, mols: Sequence[MolSpec], config: NNBConfig = NNBConfig()) -> NNBonded:
        """Return a network whose vocabulary is built from `mols` (all registered)."""
        net = cls(config)
        for m in mols:
            net.add(m)
        return net

    def add(self, spec: MolSpec) -> int:
        """Register a molecule and return its index; grows the vocabulary unless frozen.

        The index is the `m` of `coefficients` and `energy`.
        """
        self.mols.append(spec)
        self.data.append(self.prepare(spec, grow=not self.vocab.frozen))
        return len(self.data) - 1

    def prepare(self, spec: MolSpec, grow: bool = False) -> dict:
        """Return the tables of one molecule (its topology must be built).

        Graph inputs (`graph_inputs`), per basis family its instances (`decompose`) with skeleton
        indices and, for context families, the residues of the context atoms ("ctx_res"), and the
        typed-table rows "tid" (-1: key not in the table).

        Parameters
        ----------
        spec : MolSpec
            The molecule.
        grow : bool
            Add unseen skeletons, slots and table keys to the vocabulary.

        Returns
        -------
        dict
            The tables.

        Raises
        ------
        ValueError
            Without a topology, or (grow=False) for an instance kind or key width not in the training
            set.
        """
        c, v = self.config, self.vocab
        top = spec.top
        if top is None:
            raise ValueError(f"{spec.name}: build the topology first")
        d = graph_inputs(spec, top, c.ref, c.pgm_features)
        d["fam"] = {}
        for f in c.basis:
            rec = decompose(f, top)
            voc = v.skeletons.setdefault(f, []) if grow else v.skeletons.get(f, [])
            for sk in rec["skeletons"]:
                if sk not in voc:
                    if not grow:
                        raise ValueError(f"{spec.name}: {f} instance kind {sk!r} was not in the training set")
                    voc.append(sk)
            if grow:
                v.slots[f] = max(v.slots.get(f, 1), rec["n_slots"])
            elif rec["n_slots"] > v.slots.get(f, 1):
                raise ValueError(f"{spec.name}: {f} keys with more components than in the training set")
            rec["skel_idx"] = np.array([voc.index(k) for k in rec["skel_of"]], int)
            if f in CONTEXT_ATOMS and rec["n"]:
                rec["ctx_res"] = d["residue"][CONTEXT_ATOMS[f](rec["I"])]
            d["fam"][f] = rec
        d["tid"] = {}
        if c.table_depth is not None:
            for name, keys in typed_keys(spec, top, c.basis, c.table_depth).items():
                tab = v.table.setdefault(name, {}) if grow else v.table.get(name, {})
                if grow:
                    for k in keys:
                        tab.setdefault(k, len(tab))
                d["tid"][name] = np.array([tab.get(k, -1) for k in keys], int)
        return d

    def _tables(self, m: int | dict) -> dict:
        """Return the tables of registered molecule m, or m itself if it already is a table dict."""
        return self.data[m] if isinstance(m, (int, np.integer)) else m

    # ------------------------------------------------------------------ parameters
    def _uses_context(self, f: str) -> bool:
        """Return whether family f gets the sequence context."""
        return self.config.context and f in CONTEXT_ATOMS

    def init_params(self, seed: int = 0) -> dict:
        """Return initial network weights and freeze the vocabulary.

        Output layers of the heads start at zero, so the untrained model is the basis at its default
        parameters (and the reference values at the graph inputs' b_ref / th_ref).  `seed` seeds the
        PRNG.
        """
        c, v = self.config, self.vocab
        W, L = c.width, c.layers
        v.frozen = True
        ks = list(jax.random.split(jax.random.PRNGKey(seed), 4 + 2 * L + len(c.basis)))
        P = init_message_passing(ks, N_FEAT, W, L, N_EDGE)
        P["ref_bond"] = init_mlp(ks.pop(), 2 * W + 1, W, 1, zero_out=True)
        P["ref_angle"] = init_mlp(ks.pop(), 3 * W + 1, W, 1, zero_out=True)
        for f in c.basis:
            n_in = v.slots.get(f, 1) * SLOT * W + len(v.skeletons.get(f, [])) + (3 * W if self._uses_context(f) else 0)
            P["head_" + f] = init_mlp(ks.pop(), n_in, W, _n_out(f), zero_out=True)
        if any(self._uses_context(f) for f in c.basis):
            P["res"] = init_mlp(jax.random.fold_in(jax.random.PRNGKey(seed), 1), W, W, W)
        for name, tab in v.table.items():
            P["tab_" + name] = jnp.zeros((len(tab), 1 if name in ("b0", "th0") else _n_out(name)))
        return P

    # ------------------------------------------------------------------ stage 1
    def embeddings(self, P: dict, m: int | dict) -> jax.Array:
        """Return the atom embeddings (n, W) of molecule m."""
        d = self._tables(m)
        return embeddings(P, d["X"], d["src"], d["dst"], d["ef"], d["n"], self.config.layers)

    def _typed(self, P: dict, d: dict, name: str, out: jax.Array, res: list) -> jax.Array:
        """Return the typed-table row (zero for keys not in the table) + network residual `out`.

        Appends mean(out^2) to `res` (the residual penalty); without a table for `name` returns `out`.
        """
        res.append(jnp.mean(out**2) if out.size else 0.0)
        if "tab_" + name not in P:
            return out
        tid = d["tid"][name]
        row = P["tab_" + name][np.maximum(tid, 0)]
        return jnp.where((tid >= 0)[:, None], row, 0.0) + out

    def _coefficients(self, P: dict, d: dict) -> tuple[dict, list]:
        """Return stage 1 for tables d: the coefficients and the list of residual means.

        Returns
        -------
        C : dict
            "b0" (nb,) [nm] = b_ref + b_span tanh(head); "th0" (na,) [rad] = th_ref + th_span
            tanh(head); per basis family {name: (instances, *shape)} = init + out_scale * SCALES[name]
            * head output.
        res : list
            mean(residual^2) of every head (for `penalty`).
        """
        c, v = self.config, self.vocab
        W = c.width
        res = []
        h = embeddings(P, d["X"], d["src"], d["dst"], d["ef"], d["n"], c.layers)
        b, a = d["bonds"], d["angles"]
        # the reference heads also see the reference value (in the geometry mode it distinguishes
        # graph-equivalent bonds / angles that the minimum geometry does not treat alike)
        br, tr = jnp.asarray(d["b_ref"]), jnp.asarray(d["th_ref"])
        # reference value as an extra input, centred and scaled to O(1) (bonds ~0.12 +- 0.03 nm)
        fb = jnp.concatenate([readout("bond", h, b), ((br - 0.12) / 0.03)[:, None]], -1)
        C = {"b0": br + c.b_span * jnp.tanh(self._typed(P, d, "b0", mlp(P["ref_bond"], fb), res)[:, 0])}
        if len(a):
            # angles ~1.91 +- 0.2 rad (tetrahedral 109.5 deg)
            fa = jnp.concatenate([readout("angle", h, a), ((tr - 1.91) / 0.2)[:, None]], -1)
            C["th0"] = tr + c.th_span * jnp.tanh(self._typed(P, d, "th0", mlp(P["ref_angle"], fa), res)[:, 0])
        else:
            C["th0"] = jnp.zeros(0)
        r_emb = None
        if "res" in P:
            hm = (  # (n_res, W) mean atom embedding per residue
                jax.ops.segment_sum(h, d["residue"], d["n_res"])
                / jnp.maximum(jax.ops.segment_sum(jnp.ones(d["n"]), d["residue"], d["n_res"]), 1.0)[:, None]
            )
            r_emb = mlp(P["res"], hm)
        for f in c.basis:
            rec = d["fam"][f]
            n = rec["n"]
            if n == 0:
                continue
            feat = jnp.zeros((n, v.slots[f] * SLOT * W))
            for (slot, kind, _), (inst, atoms) in rec["groups"].items():
                r = readout(kind, h, atoms)
                feat = feat.at[inst, slot * SLOT * W : slot * SLOT * W + r.shape[1]].set(r)
            parts = [feat, jax.nn.one_hot(rec["skel_idx"], len(v.skeletons[f]))]
            if self._uses_context(f):
                parts.append(r_emb[rec["ctx_res"]].reshape(n, 3 * W))
            o = self._typed(P, d, f, mlp(P["head_" + f], jnp.concatenate(parts, -1)), res)
            p, c0 = {}, 0
            for pname, (shape, init) in T.REGISTRY[f].params.items():
                size = int(np.prod(shape)) if shape else 1
                p[pname] = float(init) + c.out_scale * SCALES.get(pname, 1.0) * o[:, c0 : c0 + size].reshape(
                    (n,) + tuple(shape)
                )
                c0 += size
            C[f] = p
        return C, res

    def coefficients(self, P: dict, m: int | dict) -> dict:
        """Return stage 1: reference values and per-instance family parameters of molecule m.

        `m` is the index of a registered molecule, or tables from `prepare`.
        """
        return self._coefficients(P, self._tables(m))[0]

    def penalty(self, P: dict, mols: Sequence[int | dict]) -> jax.Array | float:
        """Return resid_l2 x (mean over the molecules and heads of mean(residual^2)); 0.0 if off."""
        if not self.config.resid_l2 or not len(mols):
            return 0.0
        tot = 0.0
        for m in mols:
            res = self._coefficients(P, self._tables(m))[1]
            tot = tot + sum(res) / len(res)
        return self.config.resid_l2 * tot / len(mols)

    # ------------------------------------------------------------------ stage 2
    def energy_from(self, C: dict, m: int | dict, R: jax.Array) -> jax.Array:
        """Return stage 2: the bonded energy [kJ/mol] of coordinates R (n, 3) [nm] with stage-1 output C."""
        d = self._tables(m)
        G = T.geometry(R, d["top"])
        dev = {"db": G["b"] - C["b0"], "dc": G["cos"] - jnp.cos(C["th0"]), "dth": G["th"] - C["th0"]}
        e = 0.0
        for f in self.config.basis:
            if f in C:
                I = dict(d["fam"][f]["I"])
                I.setdefault("De", d["De"])
                e = e + T.REGISTRY[f].energy(G, dev, I, C[f])
        return e

    def energy(self, P: dict, m: int | dict, R: jax.Array) -> jax.Array:
        """Return the bonded energy [kJ/mol] with the network parameters P (or frozen tables {"coef": [...]})."""
        C = P["coef"][m] if "coef" in P else self.coefficients(P, m)
        return self.energy_from(C, m, R)

    def freeze(self, P: dict) -> dict:
        """Return stage 1 evaluated once for the registered molecules: {"coef": [...]} (MD speed)."""
        return {"coef": [self.coefficients(P, m) for m in range(len(self.data))]}

    # ------------------------------------------------------------------ persistence
    def save(self, path: str, P: dict) -> None:
        """Write the configuration, vocabulary and parameters P (as numpy arrays) to `path` (pickle)."""
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "config": asdict(self.config),
                    "vocab": asdict(self.vocab),
                    "params": jax.tree_util.tree_map(np.asarray, P),
                },
                fh,
            )

    @classmethod
    def load(cls, path: str) -> tuple[NNBonded, dict]:
        """Return (network with a frozen vocabulary, parameters) read from a file written by `save`.

        The file is a pickle: load only trusted files.
        """
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        cfg = dict(d["config"])
        cfg["basis"] = tuple(cfg["basis"])
        voc = Vocabulary(**d["vocab"])
        voc.frozen = True
        return cls(NNBConfig(**cfg), voc), jax.tree_util.tree_map(jnp.asarray, d["params"])

    # ------------------------------------------------------------------ compatibility
    @property
    def W(self) -> int:
        """Embedding width (compatibility alias of config.width)."""
        return self.config.width

    @property
    def basis(self) -> tuple:
        """Basis families (compatibility alias of config.basis)."""
        return self.config.basis

    @property
    def tvoc(self) -> dict:
        """Typed-table vocabulary (compatibility alias of vocab.table)."""
        return self.vocab.table

    @property
    def slots(self) -> dict:
        """Component slots per family (compatibility alias of vocab.slots)."""
        return self.vocab.slots

    @property
    def skel(self) -> dict:
        """Key skeletons per family (compatibility alias of vocab.skeletons)."""
        return self.vocab.skeletons
