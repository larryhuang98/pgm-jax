"""Charge flux in MD: Gaussian charges and covalent-dipole strengths that depend on bond lengths.

Contents: `ChargeFlux` (the flux tables of a system and the map R -> (q, c)), built from fitted
templates by `ChargeFlux.from_templates`; `molecule_at` (a rigid molecule frozen at one
geometry of its flux model); `template_flux_order`.

The functional form and conventions are those of the bonded fitting model (BondedModel._flux,
pgm_jax/bonded/model.py; BondedSettings.flux), so that a template fitted with flux runs in MD
unchanged.  For a bond b = (i, j) of length r_b = |r_j - r_i|, with reference length b0_b and
parameter index k = key_b:

    db_b = r_b - b0_b
    q_i  = q_i^0 - sum_{b = (i, .)} s_b jb_k db_b + sum_{b = (., i)} s_b jb_k db_b    bond charge flux
    c_m  = c_m^0 + jc_k db_b + jc2_k db_b^2        every covalent dipole m along bond b (dipole flux)

As the bond stretches, the charge s_b jb_k db_b moves from its first atom to its second.  In a
fitted template the sign s_b is +1 when the first atom has the lower tying key of the fit (the
canonical order of the bonded model's atom classes), -1 when it has the higher one, and 0 between
equivalent atoms: a bond between two atoms of one class carries no charge flux, while its covalent
dipoles still flux.  Both covalent dipoles of a bond (i -> j and j -> i) take the bond's jc and jc2;
dipoles along virtual bonds (pairs that are not bonds) have none.  jc2 is present for fits with
BondedSettings(flux=2).  Every molecule keeps its total charge.  b0 is the fit's reference bond
length (P["ref"]["b0"], the same b0 as the bond-stretch terms), not the geometry's.

Energy and forces (PGMForceField with `flux=`).  E(R, q(R), c(R), mu) with the induced dipoles mu
variational (dE/dmu = 0 at the solution), so

    F = -dE/dR|_{q,c,mu} - sum_i phi_i dq_i/dR - sum_m (dE/dc_m) dc_m/dR,

with phi_i = dE/dq_i, the electrostatic potential at atom i times KE (real-space rows, PME
reciprocal part, self and neutralising-background terms; kJ/mol/e), and dE/dc_m = (dE/dd_i) . u_m,
u_m the unit vector of covalent dipole m on atom i and dE/dd the dipole gradient the engine forms
anyway for its pull-back through the covalent frames.  The engine computes phi with one more row
sum (like the field; XLA fuses it into the force pass), the reciprocal part from the same autodiff
of the PME energy with the charges among the differentiated arguments, and pulls (phi, dE/dc)
back through the bond-local map R -> (q, c) with one vector-Jacobian product: no autodiff
through the solve or the rows.  The induction right-hand side uses q(R) and c(R) of the current
geometry, and every other use of the charges evaluates them at its own positions: Monte Carlo
barostat trial energies, the strain derivative (virial), the differentiable path (gradients with
respect to positions, box and all parameters, jb, jc, jc2 included) and the cell dipole (md/dipoles.py).

Virial.  The engine's molecular scaling (barostat, pressure) translates molecules rigidly, so no
bond length changes and the flux adds nothing to the molecular strain derivative; atomic scaling
(strain_derivative(molecular=False)) stretches bonds and the flux terms follow by autodiff.

Rigid molecules have a fixed geometry, so their flux is a constant shift of the charges and
covalent dipoles: `molecule_at(tpl, xyz)` gives the template's pGM molecule with the values of the
flux model at that geometry, for RigidTemplate or the rigid-body engine.  X-H bonds constrained at
their reference lengths (FlexibleSimulation(constraints="h-bonds")) have db = 0 and carry no flux;
the flux forces of a constrained bond lie along it and are removed with the constraint forces.

Parameters.  ChargeFlux.params = {"jb", "jc"[, "jc2"]}, one value per parameter key (bond keys of
the fits, one block per template); PGMForceField takes them from params["flux"] when the parameter
pytree has that entry (for gradients, e.g. {**sys.params0, "flux": ff.flux.params}) and from
ChargeFlux.params otherwise.  write_pgm_prmtop refuses models with flux (pmemd-pgm has none).

Units: nm, e; jb e/nm, jc e (e nm of dipole per nm), jc2 e/nm (e nm per nm^2).

See also docs/charge_flux.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .box import min_image

if TYPE_CHECKING:
    import jax

    from ..system import Molecule, System
    from .flexible import FlexibleTemplate, RigidTemplate

FLUX_PARAMS = ("jb", "jc", "jc2")


def template_flux_order(tpl: FlexibleTemplate | RigidTemplate | object) -> int:
    """Return BondedSettings.flux of a template's fit (0: no flux).

    Parameters
    ----------
    tpl : FlexibleTemplate, RigidTemplate or object
        A template; only a dict `settings` attribute (FlexibleTemplate) is read, anything else
        (RigidTemplate, other templates) gives 0.

    Returns
    -------
    int
        0 (no flux), 1 (linear dipole flux) or 2 (with jc2).
    """
    st = getattr(tpl, "settings", None)
    return int(st.get("flux", 0) or 0) if isinstance(st, dict) else 0


@dataclass
class ChargeFlux:
    """Charge and covalent-dipole flux of a system (global atom indices, System order).

    A mutable dataclass of host arrays; `__post_init__` checks the fields and builds the gather
    tables of `charges`.  Not a pytree: the parameters that are differentiated are passed to
    `charges` as `theta` (a dict pytree).

        flux = ChargeFlux.from_templates(system, templates)     # None without flux
        q, c = flux.charges(pos, H, q0, c0, flux.theta(params))

    Parameters
    ----------
    bonds : np.ndarray (nb, 2) int32
        Atoms of every bond with flux; charge flows from column 0 to column 1.
    b0 : np.ndarray (nb,)
        Reference lengths [nm].
    key : np.ndarray (nb,) int32
        Parameter index of each bond into the vectors of `params`.
    sign : np.ndarray (nb,)
        s_b in {-1, 0, 1}: the charge moved along bond b is s_b jb[key_b] db_b.
    cov_bond : np.ndarray (n_cov,) int32
        For each covalent dipole of the System (sys.cov_i order) the index of its bond in `bonds`,
        -1 for none (no dipole flux).
    params : dict of str to np.ndarray
        {"jb": (nk,) [e/nm], "jc": (nk,) [e][, "jc2": (nk,) [e/nm]]}.
    n_atoms : int
        Atoms of the System.
    names : tuple of str
        Parameter key names (nk,), for inspection (may be empty).
    """

    bonds: np.ndarray
    b0: np.ndarray
    key: np.ndarray
    sign: np.ndarray
    cov_bond: np.ndarray
    params: dict
    n_atoms: int
    names: tuple = field(default=())

    def __post_init__(self) -> None:
        """Normalize the dtypes and shapes, check the fields and build the gather tables.

        Raises
        ------
        ValueError
            Unknown or missing parameters (jb and jc are required), parameter vectors of different
            lengths, bonds outside the system or joining an atom to itself, keys out of range, signs
            other than -1, 0, 1, cov_bond entries out of range, or a wrong number of names.
        """
        self.bonds = np.asarray(self.bonds, np.int32).reshape(-1, 2)
        nb = len(self.bonds)
        self.b0 = np.asarray(self.b0, np.float64).reshape(nb)
        self.key = np.asarray(self.key, np.int32).reshape(nb)
        self.sign = np.asarray(self.sign, np.float64).reshape(nb)
        self.cov_bond = np.asarray(self.cov_bond, np.int32).reshape(-1)
        self.params = {k: np.asarray(v, np.float64) for k, v in self.params.items()}
        self.n_atoms = int(self.n_atoms)
        if set(self.params) - set(FLUX_PARAMS) or not {"jb", "jc"} <= set(self.params):
            raise ValueError(f"flux parameters are jb, jc and optionally jc2; got {sorted(self.params)}")
        nk = len(self.params["jb"])
        if any(v.shape != (nk,) for v in self.params.values()):
            raise ValueError(
                "flux parameters must be vectors of one length (one value per key): "
                + ", ".join(f"{k} {v.shape}" for k, v in self.params.items())
            )
        if nb and (
            self.bonds.min() < 0 or self.bonds.max() >= self.n_atoms or np.any(self.bonds[:, 0] == self.bonds[:, 1])
        ):
            raise ValueError("flux bonds must join two different atoms of the system")
        if nb and (self.key.min() < 0 or self.key.max() >= nk):
            raise ValueError(f"flux bond keys must index the {nk} parameter values")
        if not np.all(np.isin(self.sign, (-1.0, 0.0, 1.0))):
            raise ValueError("flux signs must be -1, 0 or 1")
        if len(self.cov_bond) and (self.cov_bond.min() < -1 or self.cov_bond.max() >= nb):
            raise ValueError("cov_bond entries must be -1 or a flux bond index")
        if self.names and len(self.names) != nk:
            raise ValueError("one name per parameter key")
        self.names = tuple(self.names)
        # gather tables: the map is written with gathers and its jax.vjp is the force pull-back (on a
        # GPU as fast as a scatter-add map at 10k atoms, faster at 1k, and faster than a hand-written
        # gather-only pull-back or a per-atom formulation)
        n = self.n_atoms
        deg = np.bincount(self.bonds.reshape(-1), minlength=n) if nb else np.zeros(n, int)
        D = max(int(deg.max()) if n else 0, 1)
        self._abond = np.full((n, D), nb, np.int32)  # flux bonds of each atom (nb: padding)
        self._asgn = np.zeros((n, D))  # -1 first atom, +1 second atom, 0 padding
        fill = np.zeros(n, int)
        for b, (i, j) in enumerate(self.bonds):
            for a, sg in ((i, -1.0), (j, 1.0)):
                self._abond[a, fill[a]], self._asgn[a, fill[a]] = b, sg
                fill[a] += 1
        has = self.cov_bond >= 0
        self._n_cov_flux = int(has.sum())
        self._cb = np.where(has, self.cov_bond, nb).astype(np.int32)  # bond of each dipole (nb: none)
        self._ck = np.where(has, self.key[np.maximum(self.cov_bond, 0)] if nb else 0, 0).astype(np.int32)
        self._ch = has.astype(np.float64)

    # ------------------------------------------------------------------ evaluation
    @property
    def n_bonds(self) -> int:
        """Number of bonds with flux."""
        return len(self.bonds)

    def theta(self, params: dict | None = None) -> dict[str, jax.Array]:
        """Return the flux parameters of a parameter pytree: params["flux"] if present, else self.params.

        Parameters
        ----------
        params : dict, optional
            Parameter pytree (None, or a dict without "flux": the model's own parameters).

        Returns
        -------
        dict of str to jax.Array
            {"jb", "jc"[, "jc2"]} as float64 arrays (nk,).

        Raises
        ------
        ValueError
            If the keys or shapes differ from the model's (a gather would clamp indices silently).
        """
        th = self.params if not isinstance(params, dict) or "flux" not in params else params["flux"]
        if set(th) != set(self.params):
            raise ValueError(f"flux parameters {sorted(th)}, the model has {sorted(self.params)}")
        bad = [k for k in th if jnp.shape(th[k]) != self.params[k].shape]
        if bad:  # (a gather would clamp indices silently)
            raise ValueError(
                f"flux parameters {bad}: shapes {[jnp.shape(th[k]) for k in bad]}, the model has "
                f"{[self.params[k].shape for k in bad]}"
            )
        return {k: jnp.asarray(v, jnp.float64) for k, v in th.items()}

    def deviations(self, pos: jax.Array, H: jax.Array) -> jax.Array:
        """Return db = bond lengths minus reference lengths (minimum image; molecules need not be whole).

        Parameters
        ----------
        pos : jax.Array (N, 3)
            Positions [nm].
        H : jax.Array (3, 3)
            Box, lattice vectors as rows [nm].

        Returns
        -------
        jax.Array (nb,)
            db [nm].
        """
        v = min_image(pos[self.bonds[:, 1]] - pos[self.bonds[:, 0]], H)
        return jnp.sqrt(jnp.sum(v * v, axis=-1)) - self.b0

    def charges(
        self, pos: jax.Array, H: jax.Array, q: jax.Array, cov: jax.Array, theta: dict[str, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """Return the charges and covalent-dipole strengths at the geometry pos.

        The base values plus the flux terms of the module docstring.  Differentiable in pos, H, q,
        cov and theta (its jax.vjp is the engine's force pull-back).

        Parameters
        ----------
        pos : jax.Array (N, 3)
            Positions [nm].
        H : jax.Array (3, 3)
            Box, lattice vectors as rows [nm].
        q : jax.Array (N,)
            Base charges q^0 [e].
        cov : jax.Array (n_cov,)
            Base covalent-dipole strengths c^0 [e nm].
        theta : dict of str to jax.Array
            Flux parameters, from `theta`.

        Returns
        -------
        q : jax.Array (N,)
            Charges [e] (the total of every molecule unchanged).
        cov : jax.Array (n_cov,)
            Covalent-dipole strengths [e nm].
        """
        db = self.deviations(pos, H)
        t = self.sign * theta["jb"][self.key] * db
        # t (nb,): charge moved along each bond; the appended 0 is the padding bond nb
        q = q + jnp.sum(self._asgn * jnp.concatenate([t, jnp.zeros(1)])[self._abond], axis=1)
        if self._n_cov_flux:
            d = jnp.concatenate([db, jnp.zeros(1)])[self._cb]
            dc = theta["jc"][self._ck] * d
            if "jc2" in theta:
                dc = dc + theta["jc2"][self._ck] * d * d
            cov = cov + self._ch * dc
        return q, cov

    # ------------------------------------------------------------------ from fitted templates
    @classmethod
    def from_templates(cls, sys: System, templates: Sequence[FlexibleTemplate | RigidTemplate]) -> ChargeFlux | None:
        """Return the flux of a system of templates, or None when no template was fitted with flux.

        Each template's parameters form one block of `params` (templates are identified by object,
        as the MD engine groups them); molecules without flux contribute nothing.  Covalent dipoles
        along a flux bond (either direction) get its jc, jc2; the others none.

        Parameters
        ----------
        sys : System
            The system.
        templates : Sequence[FlexibleTemplate or RigidTemplate] (nmol,)
            templates[k] belongs to sys.molecules[k], as in FlexibleSimulation.

        Returns
        -------
        ChargeFlux or None
            jc2 is present if any template has it (zeros for the others).

        Raises
        ------
        ValueError
            If len(templates) != sys.nmol, or from `_template_flux`.
        """
        if len(templates) != sys.nmol:
            raise ValueError("one template per molecule")
        blocks, bonds, b0, key, sign, cov_bond = {}, [], [], [], [], []
        nbond = 0
        for k, tpl in enumerate(templates):
            mol = sys.molecules[k]
            if not template_flux_order(tpl):
                cov_bond += [-1] * len(mol.cov)
                continue
            if id(tpl) not in blocks:
                blocks[id(tpl)] = (_template_flux(tpl), sum(len(b[0]["jb"]) for b in blocks.values()))
            (tb, off) = blocks[id(tpl)]
            loc = {tuple(sorted(map(int, b))): n for n, b in enumerate(tb["bonds"])}  # {(i, j) i < j: bond}
            a0 = int(sys.offsets[k])
            bonds.append(tb["bonds"] + a0)
            b0.append(tb["b0"])
            key.append(tb["key"] + off)
            sign.append(tb["sign"])
            cov_bond += [
                nbond + loc[tuple(sorted((i, j)))] if tuple(sorted((i, j))) in loc else -1 for i, j, _ in mol.cov
            ]
            nbond += len(tb["bonds"])
        if not blocks:
            return None
        quad = any("jc2" in b[0] for b in blocks.values())
        params = {}
        for p in ("jb", "jc", "jc2") if quad else ("jb", "jc"):
            params[p] = np.concatenate([b[0].get(p, np.zeros_like(b[0]["jb"])) for b in blocks.values()])
        names = sum((tuple(b[0]["names"]) for b in blocks.values()), ())
        return cls(
            np.concatenate(bonds),
            np.concatenate(b0),
            np.concatenate(key),
            np.concatenate(sign),
            np.asarray(cov_bond, int),
            params,
            sys.n,
            names,
        )

    def describe(self) -> str:
        """Return one line for the log header (bonds, charge-flux bonds, covalent dipoles, parameter keys)."""
        n_cov = self._n_cov_flux
        return (
            f"charge flux on {self.n_bonds} bonds ({int(np.sum(self.sign != 0))} with charge flux, {n_cov} covalent "
            f"dipoles{', quadratic' if 'jc2' in self.params else ''}), {len(self.params['jb'])} parameter keys"
        )


def _template_flux(tpl: FlexibleTemplate) -> dict:
    """Return the flux data of one fitted template (local atom indices).

    Parameters
    ----------
    tpl : FlexibleTemplate
        A template fitted with flux (its bonded terms, class keys and parameters are read).

    Returns
    -------
    dict
        "bonds" (nb, 2) int32 in the topology's order, "b0" (nb,) [nm] (the fit's reference
        lengths), "key" (nb,) int32 (the bond keys of the fit this molecule uses, renumbered from
        0), "sign" (nb,) (from the fit's canonical class order), "names" (list of key names), and
        the parameter vectors "jb" [e/nm], "jc" [e], "jc2" [e/nm] present in the fit, one value per
        renumbered key.

    Raises
    ------
    ValueError
        If the template's parameters have no "flux" or "ref" entry.
    """
    terms, m = tpl.terms, tpl.index
    top, Im, cl = terms.mols[m].top, terms.I[m], terms.keyf[m]
    P = tpl.P
    if "flux" not in P or "ref" not in P:
        raise ValueError(f"template {tpl.name}: fitted with flux but its parameters have no flux / reference values")
    bonds = np.asarray(top.bonds, np.int32).reshape(-1, 2)
    kb = np.asarray(Im["bond"], int)
    used, local = np.unique(kb, return_inverse=True)
    cls_ = [cl([int(a)], "atom") for a in range(len(tpl.spec.elements))]
    # s_b = +1 if the first atom has the lower class key, -1 if the higher, 0 between equivalent atoms
    sign = np.array([0.0 if cls_[i] == cls_[j] else (1.0 if cls_[i] < cls_[j] else -1.0) for i, j in bonds])
    out = {
        "bonds": bonds,
        "b0": np.asarray(P["ref"]["b0"], float)[kb],
        "key": local.astype(np.int32),
        "sign": sign,
        "names": [f"{tpl.name}:{terms.ref_keys['b0'][u]}" for u in used],
    }
    for p in FLUX_PARAMS:
        if p in P["flux"]:
            out[p] = np.asarray(P["flux"][p], float)[used]
    return out


def molecule_at(
    tpl: FlexibleTemplate, xyz: ArrayLike | None = None, params: dict | None = None, name: str | None = None
) -> Molecule:
    """Return the template's pGM molecule (a copy) with its flux values at one geometry.

    For molecules held rigid, whose flux is a constant shift.  Charges and covalent dipoles get
    per-atom / per-dipole tying keys (prefixed by `name`) so that the values are kept exactly; give
    molecules frozen at different geometries different names.

    Parameters
    ----------
    tpl : FlexibleTemplate
        A template fitted with flux.
    xyz : ArrayLike (n, 3), optional
        Geometry [nm] (None: the template's reference geometry).
    params : dict, optional
        Parameters of the template's System([tpl.pgm]) as for PGMForceField (None: initial values,
        flux from the fit).
    name : str, optional
        Name of the new molecule and prefix of its keys (None: the molecule's name + "@flux").

    Returns
    -------
    Molecule
        The molecule with charges q [e] and covalent-dipole strengths [e nm] at `xyz`.

    Raises
    ------
    ValueError
        If the template has no charge flux.
    """
    from dataclasses import replace

    from ..system import System

    mol = tpl.pgm
    sys = System([mol])
    fl = ChargeFlux.from_templates(sys, [tpl])
    if fl is None:
        raise ValueError(f"template {tpl.name} has no charge flux")
    x = jnp.asarray(tpl.spec.ref_xyz if xyz is None else xyz, jnp.float64).reshape(mol.n, 3)
    P = sys.expand(params)
    H = jnp.eye(3) * (4.0 * float(jnp.max(jnp.abs(x))) + 10.0)  # no image is nearer
    q, cov = fl.charges(x, H, jnp.asarray(P["q"], jnp.float64), jnp.asarray(P["cov"], jnp.float64), fl.theta(params))
    q, cov = np.asarray(q), np.asarray(cov)
    name = mol.name + "@flux" if name is None else str(name)
    keys = dict(mol.keys)
    keys["q"] = [f"{name}:q{a}" for a in range(mol.n)]
    keys["cov"] = [f"{name}:c{a}" for a in range(len(mol.cov))]
    return replace(mol, name=name, q=q, cov=[(i, j, float(c)) for (i, j, _), c in zip(mol.cov, cov)], keys=keys)
