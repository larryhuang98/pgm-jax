"""Periodic pGM + van der Waals reference model: energy, forces, strain derivative and pressure.

Contents: strain_derivative (dE/d eps of any energy function of positions and box),
pressure_bar (static pressure from a strain derivative) and PeriodicModel (Ewald pGM,
ewald.PeriodicPGM, plus LJ or GVDW, lj.py / vdw.py).  Everything is differentiable in positions,
parameters and the box; this is the exact, slow reference for the MD force field (md/).

Strain convention: a homogeneous deformation F = 1 + eps acts on the box as H -> H F^T (lattice
vectors as rows) and on the coordinates either atom by atom (atomic scaling) or through the
molecular centres of mass with rigid molecules (molecular scaling, the virial of rigid-molecule
MD and of Amber's printed VIRIAL).  The static pressure is P = -tr(dE/d eps) / (3 V).

Strain derivatives are validated against sander's molecular VIRIAL for 512 pGM3P-25 waters
(scripts/validate_amber.py virial).

    model = PeriodicModel(system, box, positions, cutoff=0.9, vdw="lj", lj_lrc=True)
    e = model.energy(positions)                     # {"perm", "ind", "elec", "vdw", "total"}
    p = model.pressure(positions)                   # bar, potential part only

Units: nm, kJ/mol, bar.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .ewald import PeriodicPGM, neighbor_list
from .lj import PeriodicLJ
from .md.box import centers_of_mass
from .system import System
from .units import BAR_PER_KJMOL_NM3


def strain_derivative(
    energy: Callable[[jax.Array, jax.Array], jax.Array], pos: ArrayLike, H: ArrayLike, sys: System | None = None
) -> jax.Array:
    """Return dE/d eps at eps = 0 for a homogeneous deformation F = 1 + eps of box and coordinates.

    Parameters
    ----------
    energy : callable
        energy(positions (N, 3) [nm], box (3, 3) [nm]) -> scalar [kJ/mol]; differentiable.
    pos : ArrayLike (N, 3)
        Positions [nm].
    H : ArrayLike (3, 3)
        Box [nm], lattice vectors as rows; deformed as H -> H F^T.
    sys : System, optional
        None: atomic scaling (every atom r -> F r).  Given: molecular scaling, each molecule is
        translated rigidly with its centre of mass (r -> r + eps R_com), the virial of
        rigid-molecule MD.

    Returns
    -------
    jax.Array (3, 3)
        dE/d eps [kJ/mol] (the negative of the potential virial tensor W = -dE/d eps).
    """
    pos, H = jnp.asarray(pos), jnp.asarray(H)
    if sys is not None:
        w = jnp.asarray(sys.masses)
        mol = jnp.asarray(sys.mol)
        com = centers_of_mass(pos, w, mol, sys.nmol)

    def e(eps: jax.Array) -> jax.Array:
        """Return the energy of the configuration deformed by F = 1 + eps."""
        F = jnp.eye(3) + eps
        x = pos @ F.T if sys is None else pos + (com @ eps.T)[mol]
        return energy(x, H @ F.T)

    return jax.grad(e)(jnp.zeros((3, 3)))


def pressure_bar(dE_deps: ArrayLike, H: ArrayLike) -> jax.Array:
    """Return the static (potential) pressure P = -tr(dE/d eps) / (3 V) [bar].

    Parameters
    ----------
    dE_deps : ArrayLike (3, 3)
        Strain derivative [kJ/mol] (strain_derivative, PeriodicModel.virial_derivative).
    H : ArrayLike (3, 3)
        Box [nm], lattice vectors as rows.

    Returns
    -------
    jax.Array ()
        Pressure [bar]; add the kinetic part in MD.

    Notes
    -----
    Amber's printed VIRIAL is tr(dE/d eps)/2 (kcal/mol) with molecular scaling.
    """
    V = jnp.abs(jnp.linalg.det(jnp.asarray(H)))
    return -jnp.trace(dE_deps) / (3.0 * V) * BAR_PER_KJMOL_NM3


class PeriodicModel:
    """pGM electrostatics (Ewald) + van der Waals in a periodic box: the differentiable reference model.

    Energies, forces, strain derivatives and pressure are exact derivatives of one energy function.
    One neighbour list (max(cutoff, vdw_cutoff) + skin) is built at (positions_ref, box) and shared
    by the electrostatics and the van der Waals; energies accept any positions, parameters and box.
    Not a pytree; methods are pure functions of their arguments.

    Attributes
    ----------
    sys : System
        The system.
    H : np.ndarray (3, 3)
        Reference box [nm], lattice vectors as rows.
    elec : ewald.PeriodicPGM
        Electrostatics.
    vdw : lj.PeriodicLJ or vdw.PeriodicGVDW or None
        Van der Waals (None for vdw="none").
    """

    def __init__(
        self,
        system: System,
        box: ArrayLike,
        positions_ref: ArrayLike,
        cutoff: float = 1.0,
        ewald_beta: float = 3.8,
        skin: float = 0.0,
        vdw_cutoff: float | None = None,
        lj_lrc: bool = False,
        k_tol: float = 1e-12,
        dipole_tol: float = 1e-12,
        elec: str = "qpi",
        vdw: str = "lj",
        gvdw_rep: str = "gauss",
    ) -> None:
        """Build the model.

        Parameters
        ----------
        system : System
            The molecules.
        box : ArrayLike (3, 3)
            Reference box [nm], lattice vectors as rows (fixes the k-vectors).
        positions_ref : ArrayLike (N, 3)
            Reference positions [nm] of the neighbour list.
        cutoff : float
            Real-space electrostatics cutoff [nm] (and the van der Waals cutoff by default).
        ewald_beta : float
            Ewald coefficient [1/nm].
        skin : float
            Neighbour-list skin [nm] (pairs beyond the cutoffs are masked, so small displacements
            from positions_ref stay exact).
        vdw_cutoff : float, optional
            Van der Waals cutoff [nm]; None: `cutoff`.  Applies to LJ and GVDW.
        lj_lrc : bool
            Long-range correction of the van der Waals tail (LJ or GVDW dispersion).
        k_tol : float
            Reciprocal-space truncation: exp(-k^2 / (4 beta^2)) below k_tol.
        dipole_tol : float
            Relative residual of the induced-dipole CG (jax.scipy.sparse.linalg.cg).
        elec : {"q", "qp", "qi", "qpi"}
            Electrostatics level (options.py).
        vdw : {"lj", "gvdw", "none"}
            Van der Waals form.
        gvdw_rep : {"gauss", "slater"}
            GVDW repulsion.

        Raises
        ------
        ValueError
            An unknown electrostatics level, van der Waals form or GVDW repulsion.
        """
        from .options import check_vdw
        from .vdw import PeriodicGVDW

        check_vdw(vdw, gvdw_rep)
        self.sys, self.H = system, np.asarray(box, float)
        vdw_cutoff = cutoff if vdw_cutoff is None else vdw_cutoff
        nl = neighbor_list(positions_ref, self.H, max(cutoff, vdw_cutoff) + skin)
        self.elec = PeriodicPGM(
            system,
            self.H,
            positions_ref,
            ewald_beta=ewald_beta,
            cutoff=cutoff,
            k_tol=k_tol,
            dipole_tol=dipole_tol,
            nlist=nl,
            elec=elec,
        )
        if vdw == "none":
            self.vdw = None
        elif vdw == "lj":
            self.vdw = PeriodicLJ(system, self.H, positions_ref, rc=vdw_cutoff, lrc=lj_lrc, nlist=nl)
        else:
            self.vdw = PeriodicGVDW(system, self.H, positions_ref, rc=vdw_cutoff, lrc=lj_lrc, nlist=nl, rep=gvdw_rep)

    def energy(
        self, pos: ArrayLike, params: Mapping[str, ArrayLike] | None = None, H: ArrayLike | None = None
    ) -> dict[str, jax.Array]:
        """Return the energy components of one configuration.

        Parameters
        ----------
        pos : ArrayLike (N, 3)
            Positions [nm] (molecules whole).
        params : Mapping of str to ArrayLike, optional
            Parameter pytree (system.py); None: the table's initial values.
        H : ArrayLike (3, 3), optional
            Box [nm], lattice vectors as rows; None: the reference box.

        Returns
        -------
        dict of str to jax.Array ()
            "perm", "ind", "elec" (= perm + ind), "vdw" (0 without van der Waals) and "total" [kJ/mol].
        """
        e, _ = self.elec.energy(pos, params, H)
        out = {"perm": e["perm"], "ind": e["ind"], "elec": e["total"]}
        out["vdw"] = self.vdw.energy(pos, params, H)[0]["vdw"] if self.vdw is not None else jnp.zeros(())
        out["total"] = out["elec"] + out["vdw"]
        return out

    def forces(
        self, pos: ArrayLike, params: Mapping[str, ArrayLike] | None = None, H: ArrayLike | None = None
    ) -> jax.Array:
        """Return the forces -dE_total/dpos (N, 3) [kJ/mol/nm]; arguments as in `energy`."""
        return -jax.grad(lambda x: self.energy(x, params, H)["total"])(jnp.asarray(pos))

    def strain_derivative(
        self,
        pos: ArrayLike,
        params: Mapping[str, ArrayLike] | None = None,
        H: ArrayLike | None = None,
        molecular: bool = True,
    ) -> jax.Array:
        """Return dE_total/d eps (3, 3) [kJ/mol] (exact derivative; see periodic.strain_derivative).

        Arguments as in `energy`; `molecular` selects molecular (rigid molecules, centre-of-mass)
        or atomic scaling.
        """
        H = self.H if H is None else H
        return strain_derivative(
            lambda x, h: self.energy(x, params, h)["total"], pos, H, self.sys if molecular else None
        )

    def virial_derivative(
        self,
        pos: ArrayLike,
        params: Mapping[str, ArrayLike] | None = None,
        H: ArrayLike | None = None,
        molecular: bool = True,
    ) -> jax.Array:
        """Return dE/d eps (3, 3) [kJ/mol] for the pressure.

        The strain derivative plus, with the van der Waals long-range correction, its cutoff-impulse
        term (PeriodicLJ.tail_virial, PeriodicGVDW.tail_virial).  Amber's printed VIRIAL is tr(.)/2 in
        kcal/mol (validated against sander, vdwmeth=0 and 1).  Arguments as in strain_derivative.
        """
        W = self.strain_derivative(pos, params, H, molecular)
        return W + self.vdw.tail_virial(params, H) if self.vdw is not None else W

    def pressure(
        self,
        pos: ArrayLike,
        params: Mapping[str, ArrayLike] | None = None,
        H: ArrayLike | None = None,
        molecular: bool = True,
    ) -> jax.Array:
        """Return the static (potential) pressure [bar] from virial_derivative; add the kinetic part in MD.

        Arguments as in strain_derivative.
        """
        H = self.H if H is None else H
        return pressure_bar(self.virial_derivative(pos, params, H, molecular), H)
