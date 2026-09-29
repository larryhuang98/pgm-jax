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
from .system import System

KJMOL_NM3_BAR = 16.605390671738466  # 1 kJ/mol/nm^3 in bar


def strain_derivative(energy, pos, H, sys: System | None = None):
    """dE/d eps (3, 3) at eps = 0 for the homogeneous deformation F = 1 + eps applied to the box
    (rows: H -> H F^T) and to the atoms (atomic scaling, sys=None) or to the molecular centres of
    mass with rigid molecules (molecular scaling, sys given; the virial of rigid-molecule MD).
    energy(pos, H) -> scalar."""
    pos, H = jnp.asarray(pos), jnp.asarray(H)
    if sys is not None:
        w = jnp.asarray(sys.masses)
        mol = jnp.asarray(sys.mol)
        com = jax.ops.segment_sum(w[:, None] * pos, mol, sys.nmol) / jax.ops.segment_sum(w, mol, sys.nmol)[:, None]

    def e(eps):
        F = jnp.eye(3) + eps
        x = pos @ F.T if sys is None else pos + (com @ eps.T)[mol]
        return energy(x, H @ F.T)

    return jax.grad(e)(jnp.zeros((3, 3)))


def pressure_bar(dE_deps, H):
    """Static (potential) pressure P = -tr(dE/d eps) / (3 V), bar; add the kinetic part in MD.
    Amber's printed VIRIAL is tr(dE/d eps)/2 (kcal/mol) with molecular scaling."""
    V = jnp.abs(jnp.linalg.det(jnp.asarray(H)))
    return -jnp.trace(dE_deps) / (3.0 * V) * KJMOL_NM3_BAR


class PeriodicModel:
    """pGM electrostatics (Ewald) + LJ in a periodic box.

    One neighbour list (rc + skin, or the LJ cutoff if larger) is built at (pos_ref, H) and
    shared; energies accept any positions, parameters and box."""

    def __init__(
        self,
        sys: System,
        H,
        pos_ref,
        rc: float = 1.0,
        b0: float = 3.8,
        skin: float = 0.0,
        lj: bool = True,
        lj_rc: float | None = None,
        lj_lrc: bool = False,
        k_tol: float = 1e-12,
        cg_tol: float = 1e-12,
        elec: str = "qpi",
        vdw: str = "lj",
        gvdw_rep: str = "gauss",
    ):
        """elec: "q" | "qp" | "qi" | "qpi" (options.py); vdw: "lj" | "gvdw" | "none" (lj=False: none);
        lj_rc / lj_lrc apply to either van der Waals form."""
        from .options import check_vdw
        from .vdw import PeriodicGVDW

        check_vdw(vdw, gvdw_rep)
        self.sys, self.H = sys, np.asarray(H, float)
        lj_rc = rc if lj_rc is None else lj_rc
        nl = neighbor_list(pos_ref, self.H, max(rc, lj_rc) + skin)
        self.elec = PeriodicPGM(sys, self.H, pos_ref, b0=b0, rc=rc, k_tol=k_tol, cg_tol=cg_tol, nlist=nl, elec=elec)
        if not lj or vdw == "none":
            self.vdw = None
        elif vdw == "lj":
            self.vdw = PeriodicLJ(sys, self.H, pos_ref, rc=lj_rc, lrc=lj_lrc, nlist=nl)
        else:
            self.vdw = PeriodicGVDW(sys, self.H, pos_ref, rc=lj_rc, lrc=lj_lrc, nlist=nl, rep=gvdw_rep)

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
