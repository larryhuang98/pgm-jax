"""External electric fields: a uniform static field E0, a time-dependent field E0 cos(w t + phi),
and the finite-field static dielectric constant.

Physics (SI units in the docstrings, model units in the code).  A uniform field E acts on the
charge density of the model, Gaussian charges q_i at r_i, permanent (covalent) dipoles p_i and
induced dipoles mu_i (a Gaussian multipole density has the moments of the point multipole at its
centre, so a uniform field sees point multipoles):

    U(R, mu; E) = U_pGM(R, mu) - E . M,        M = sum_i q_i r_i + sum_i p_i + sum_i mu_i

  * forces: F_i += q_i E on every atom, plus the torque of E on the covalent dipoles, carried to the
    atoms through the covalent frames (the same vector-Jacobian product as the other dipole forces);
  * induction: U is minimised in mu, so E adds to the permanent field on the right-hand side of
    the induction equations, (alpha^-1 - T) mu = E_perm + E: the induced dipoles respond to the
    applied field with every dipole-dipole coupling, including the Ewald sums of the periodic cell;
  * energy: the term -E . M is reported as `field` (kJ/mol) and is part of the total.

Periodic boundary conditions.  Smooth PME drops the k = 0 term (tin-foil, conducting boundary
conditions), so the uniform applied field is the macroscopic (Maxwell) field inside the sample:
there is no depolarising field.  The dielectric constant is therefore

    eps = 1 + <M . e> / (eps0 V |E|)         (finite field, tin-foil; e the unit vector of E)

(SI: M in C m, V in m^3, E in V/m).  In the units of pgm_jax (M in e nm, V in nm^3, E in V/nm)
eps - 1 = EPS_FACTOR <M . e> / (V |E|) with EPS_FACTOR = e / (eps0 x 1 nm) = 18.0951.  This holds in
the linear regime, and running +E and -E cancels any bias of <M> at zero field.

M is evaluated with whole molecules (both MD engines keep molecules whole and only shift them by
lattice vectors).  For neutral molecules sum_i q_i r_i does not change when a molecule is shifted,
so the field energy is continuous across the driver's re-wrapping.  For charged molecules the driver
books the jump Q_k L of each re-wrapped molecule in MDState.fshift (the dipole of the unwrapped,
itinerant charges), so that E_tot stays conserved in NVE; the forces never depend on it.

Virial and NPT.  Under the molecular scaling of the barostat (and of the molecular virial) molecules
are translated rigidly, so for neutral molecules the field term does not change at fixed mu: its
contribution to the molecular virial is exactly zero, and the Monte Carlo trial energies see the
field only through the induced dipoles re-solved in the trial box.  That is the NPT ensemble of the
Hamiltonian U - E . M at constant applied (Maxwell) field.  For charged molecules the translational
term -E . sum_k Q_k R_k is not invariant under scaling (it depends on the image of each ion), so NPT
with a field and charged molecules is refused; run those NVT.  NVT at the zero-field density is the
recommended protocol for the dielectric constant (electrostriction at 0.1 V/nm is ~1e-4 in density).

Time-dependent field.  E(t) = E0 cos(w t + phi), with E0 a state variable (MDState.efield, V/nm)
and w, phi static (ExternalField).  The Hamiltonian depends on time, dH/dt = dH/dt|_x = -dE/dt . M
(Hellmann-Feynman: the induced dipoles minimise H at every t).  The driver evaluates the forces of
step n+1 at t_{n+1} = (n+1) dt.  That is velocity Verlet in the phase space extended by (t, p_t),
H_ext = H(x, p, t) + p_t, with t drifting like x and p_t kicked by -dH/dt|_x in the half kicks;
H_ext is its conserved (shadow) energy.  So the energy supplied by the field is booked in
MDState.heat by the trapezoid (dt/2) (dH/dt|_n + dH/dt|_{n+1}) with the dipoles of both steps, and
econs = E_tot + |aux|^2/2 - heat stays conserved to O(dt^2), as with a thermostat.  (The right-end
rule -(E_{n+1} - E_n) . M_{n+1} would be off by (dE . alpha_cell . dE)/2 per step, a systematic drift
at resonance.)

Constant electric displacement (kind="D"; Stengel, Spaldin & Vanderbilt, Nat. Phys. 5, 304 (2009);
Zhang & Sprik, PRB 93, 144201 (2016)).  Instead of E the displacement D is held fixed; with tin-foil
Ewald the Hamiltonian is

    U_D = U_pGM(R, mu) + (V eps0 / 2) |E(M)|^2,     E(M) = D/eps0 - M / (eps0 V),

so the macroscopic field E(M) acting on every charge and dipole follows the polarization (in model
units F = D - 4 pi M / V).  Forces are q_i E(M), torques as for a constant field, and the induction
equations gain the all-to-all term (4 pi / V) sum_j mu_j (the operator stays symmetric positive
definite).  D is given as D/eps0 in V/nm (the field that would act at zero polarization);
D = 0 is the open-circuit ("vacuum slab") electrostatics of a uniformly polarized sample.  The
dielectric constant is eps = D / (eps0 <E>) = (D/eps0) / (D/eps0 - <M.e> / (eps0 V)).  M must be
continuous in time: charged molecules use the itinerant (unwrapped) dipole, as above.  U_D depends
on V, and the molecular virial includes it (autodiff of the energy with respect to the strain).

Units: V/nm for fields (1 V/nm = 1e9 V/m; 1 V/nm acting on 1 e is 96.485 kJ/mol/nm), e nm for
dipoles, kJ/mol for energies.  In the force field's internal units a field is in e/nm^2 (energy
KE M . F): F = E [V/nm] x VNM_TO_INTERNAL."""
from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..units import KE

FARADAY_KJ = 96.48533212331002             # kJ/mol per (e V): e N_A
VNM_TO_INTERNAL = FARADAY_KJ / KE          # e/nm^2 per V/nm (0.694468)
E_CHARGE = 1.602176634e-19                 # C
EPS0 = 8.8541878128e-12                    # F/m
EPS_FACTOR = 4.0 * math.pi * KE / FARADAY_KJ   # eps - 1 = EPS_FACTOR <M.e> / (V |E|); M e nm, V nm^3, E V/nm:
                                               # e / (eps0 x 1 nm) = 18.0951 (the model's KE: equal to 1e-9)
C_CM_PS = 0.0299792458                     # speed of light, cm/ps


@dataclass(frozen=True)
class ExternalField:
    """A uniform external electric field E(t) = E0 cos(omega t + phase).

    E0: (3,) V/nm, the initial amplitude (MDState.efield holds it during a run, so that it can be
    changed with Simulation.set_field without recompiling); omega: angular frequency (rad/ps; 0 for a
    static field); phase: rad.  `ExternalField.from_wavenumber(E0, 500.0)` gives omega from a
    wavenumber in cm^-1."""
    E0: tuple = (0.0, 0.0, 0.0)
    omega: float = 0.0
    phase: float = 0.0
    kind: str = "E"                  # "E": constant field E0; "D": constant displacement D/eps0 = E0 (V/nm)

    def __post_init__(self):
        if self.kind not in ("E", "D"):
            raise ValueError(f"kind must be 'E' (constant field) or 'D' (constant displacement), got {self.kind!r}")
        e = np.asarray(self.E0, float).reshape(-1)
        if e.shape != (3,) or not np.all(np.isfinite(e)):
            raise ValueError(f"the field E0 must be three finite numbers (V/nm), got {self.E0!r}")
        object.__setattr__(self, "E0", tuple(float(x) for x in e))
        object.__setattr__(self, "omega", float(self.omega))
        object.__setattr__(self, "phase", float(self.phase))

    @classmethod
    def from_wavenumber(cls, E0, wavenumber_cm: float, phase: float = 0.0) -> "ExternalField":
        return cls(E0, 2.0 * math.pi * C_CM_PS * float(wavenumber_cm), phase)

    @property
    def time_dependent(self) -> bool:
        return self.omega != 0.0 or self.phase != 0.0

    def modulation(self, t):
        """cos(omega t + phase) (1 for a static field); t in ps (traced or not)."""
        if not self.time_dependent:
            return 1.0
        return jnp.cos(self.omega * t + self.phase)

    def value(self, E0, t):
        """The field (V/nm) at time t (ps) for the amplitude E0 (the state's)."""
        return jnp.asarray(E0, jnp.float64) * self.modulation(t)

    def rate(self, E0, t):
        """dE/dt (V/nm/ps) at time t."""
        if not self.time_dependent:
            return jnp.zeros(3)
        return -self.omega * jnp.asarray(E0, jnp.float64) * jnp.sin(self.omega * t + self.phase)

    def dHdt(self, E0, t, M, V):
        """Explicit time derivative of the Hamiltonian (kJ/mol/ps) at fixed nuclei: -dE/dt . M for a
        constant field; V eps0 E(M) . d(D/eps0)/dt for constant displacement."""
        r = self.rate(E0, t)
        if self.kind == "E":
            return field_energy(r, M)
        return KE * V / (4.0 * jnp.pi) * VNM_TO_INTERNAL ** 2 * jnp.dot(self.value(E0, t) - EPS_FACTOR * M / V, r)

    def energy(self, value, M, V):
        """Energy (kJ/mol) of the field term for the field (or D/eps0) `value` (V/nm), the dipole M
        (e nm) and the volume V (nm^3)."""
        if self.kind == "E":
            return field_energy(value, M)
        F = internal(value) - 4.0 * jnp.pi / V * jnp.asarray(M, jnp.float64)
        return KE * V / (8.0 * jnp.pi) * jnp.dot(F, F)

    def macroscopic(self, value, M, V):
        """The Maxwell field (V/nm) in the sample: value (E) or value - M / (eps0 V) (D)."""
        if self.kind == "E":
            return jnp.asarray(value, jnp.float64)
        return jnp.asarray(value, jnp.float64) - EPS_FACTOR * jnp.asarray(M, jnp.float64) / V

    def describe(self) -> str:
        e = np.asarray(self.E0)
        what = "external field E0" if self.kind == "E" else "constant displacement D/eps0"
        s = f"{what} = ({e[0]:g}, {e[1]:g}, {e[2]:g}) V/nm (|.| {np.linalg.norm(e):g})"
        if self.time_dependent:
            s += (f" x cos({self.omega:g} t + {self.phase:g}) (omega in rad/ps: "
                  f"{self.omega / (2 * math.pi * C_CM_PS):.6g} cm^-1, period {2 * math.pi / self.omega if self.omega else math.inf:.6g} ps)")
        return s


def displacement(D, omega: float = 0.0, phase: float = 0.0) -> ExternalField:
    """Constant electric displacement D/eps0 (three numbers, V/nm)."""
    return ExternalField(tuple(np.asarray(D, float).reshape(-1)), omega, phase, "D")


def finite_d_eps(M_par, volume_nm3, D_vnm):
    """eps = D / (eps0 <E>) = (D/eps0) / (D/eps0 - <M . e> / (eps0 V)) at constant displacement."""
    D = np.asarray(D_vnm)
    return D / (D - EPS_FACTOR * np.asarray(M_par) / np.asarray(volume_nm3))


def as_field(field) -> ExternalField | None:
    """None, an ExternalField, or three numbers (a static field in V/nm)."""
    if field is None or isinstance(field, ExternalField):
        return field
    return ExternalField(tuple(np.asarray(field, float).reshape(-1)))


def internal(E_vnm):
    """V/nm -> the force field's field unit (e/nm^2, energy KE M . F)."""
    return jnp.asarray(E_vnm, jnp.float64) * VNM_TO_INTERNAL


def field_energy(E_vnm, M):
    """-E . M in kJ/mol for E in V/nm and M in e nm."""
    return -FARADAY_KJ * jnp.dot(jnp.asarray(E_vnm, jnp.float64), jnp.asarray(M, jnp.float64))


def finite_field_eps(M_par, volume_nm3, E_vnm):
    """eps = 1 + <M . e> / (eps0 V |E|) (tin-foil boundary conditions): M_par = <M . e> (e nm), V in
    nm^3, |E| in V/nm."""
    return 1.0 + EPS_FACTOR * np.asarray(M_par) / (np.asarray(volume_nm3) * np.asarray(E_vnm))
