"""A periodic pGM + LJ model: energy, forces, strain derivative (virial) and pressure, all
differentiable in positions, parameters and the box.

Strain derivatives are validated against sander's molecular VIRIAL for 512 pGM3P-25 waters
(scripts/validate_amber.py virial)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .ewald import PeriodicPGM, neighbor_list
from .lj import PeriodicLJ
from .md.box import centers_of_mass
from .system import System
from .units import BAR_PER_KJMOL_NM3


def strain_derivative(energy, pos, H, sys: System | None = None):
    """dE/d eps (3, 3) at eps = 0 for the homogeneous deformation F = 1 + eps applied to the box
    (rows: H -> H F^T) and to the atoms (atomic scaling, sys=None) or to the molecular centres of
    mass with rigid molecules (molecular scaling, sys given; the virial of rigid-molecule MD).
    energy(pos, H) -> scalar."""
    pos, H = jnp.asarray(pos), jnp.asarray(H)
    if sys is not None:
        w = jnp.asarray(sys.masses)
        mol = jnp.asarray(sys.mol)
        com = centers_of_mass(pos, w, mol, sys.nmol)

    def e(eps):
        F = jnp.eye(3) + eps
        x = pos @ F.T if sys is None else pos + (com @ eps.T)[mol]
        return energy(x, H @ F.T)

    return jax.grad(e)(jnp.zeros((3, 3)))


def pressure_bar(dE_deps, H):
    """Static (potential) pressure P = -tr(dE/d eps) / (3 V), bar; add the kinetic part in MD.
    Amber's printed VIRIAL is tr(dE/d eps)/2 (kcal/mol) with molecular scaling."""
    V = jnp.abs(jnp.linalg.det(jnp.asarray(H)))
    return -jnp.trace(dE_deps) / (3.0 * V) * BAR_PER_KJMOL_NM3


class PeriodicModel:
    """pGM electrostatics (Ewald) + van der Waals in a periodic box: the differentiable reference
    model (energies, forces, strain derivatives and pressure are exact derivatives of one energy).

    One neighbour list (cutoff + skin, or the van der Waals cutoff if larger) is built at
    (positions_ref, box) and shared; energies accept any positions, parameters and box."""

    def __init__(
        self,
        system: System,
        box,
        positions_ref,
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
    ):
        """Build the model.

        Parameters
        ----------
        system : System
            The molecules.
        box : array (3, 3)
            Reference box [nm], lattice vectors as rows (fixes the k-vectors).
        positions_ref : array (N, 3)
            Reference positions [nm] of the neighbour list.
        cutoff : float
            Real-space electrostatics cutoff [nm] (and the van der Waals cutoff by default).
        ewald_beta : float
            Ewald coefficient [1/nm].
        skin : float
            Neighbour-list skin [nm] (pairs beyond the cutoffs are masked, so small displacements
            from positions_ref stay exact).
        vdw_cutoff : float or None
            Van der Waals cutoff [nm] (None: `cutoff`); applies to LJ and GVDW.
        lj_lrc : bool
            Long-range correction of the van der Waals tail (LJ or GVDW dispersion).
        k_tol : float
            Reciprocal-space truncation: exp(-k^2 / (4 beta^2)) below k_tol.
        dipole_tol : float
            Relative residual of the induced-dipole CG (jax.scipy.sparse.linalg.cg).
        elec : str
            "q" | "qp" | "qi" | "qpi" (options.py).
        vdw : str
            "lj" | "gvdw" | "none".
        gvdw_rep : str
            GVDW repulsion, "gauss" or "slater".

        Raises
        ------
        ValueError
            An unknown van der Waals form.
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

    def energy(self, pos, params=None, H=None):
        """-> {perm, ind, elec, vdw, total} kJ/mol."""
        e, _ = self.elec.energy(pos, params, H)
        out = {"perm": e["perm"], "ind": e["ind"], "elec": e["total"]}
        out["vdw"] = self.vdw.energy(pos, params, H)[0]["vdw"] if self.vdw is not None else jnp.zeros(())
        out["total"] = out["elec"] + out["vdw"]
        return out

    def forces(self, pos, params=None, H=None):
        return -jax.grad(lambda x: self.energy(x, params, H)["total"])(jnp.asarray(pos))

    def strain_derivative(self, pos, params=None, H=None, molecular: bool = True):
        """dE/d eps of the energy function (exact derivative; see periodic.strain_derivative)."""
        H = self.H if H is None else H
        return strain_derivative(
            lambda x, h: self.energy(x, params, h)["total"], pos, H, self.sys if molecular else None
        )

    def virial_derivative(self, pos, params=None, H=None, molecular: bool = True):
        """dE/d eps for the pressure: the strain derivative plus, with the LJ long-range
        correction, its cutoff-impulse term (PeriodicLJ.tail_virial).  Amber's printed VIRIAL is
        tr(.)/2 in kcal/mol (validated against sander, vdwmeth=0 and 1)."""
        W = self.strain_derivative(pos, params, H, molecular)
        return W + self.vdw.tail_virial(params, H) if self.vdw is not None else W

    def pressure(self, pos, params=None, H=None, molecular: bool = True):
        """Static (potential) pressure, bar; add the kinetic part in MD."""
        H = self.H if H is None else H
        return pressure_bar(self.virial_derivative(pos, params, H, molecular), H)
