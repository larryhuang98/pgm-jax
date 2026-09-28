"""Path-integral molecular dynamics: PIMD (PILE-L / PILE-G), thermostatted RPMD and RPMD.

The quantum canonical partition function of the nuclei is sampled with a ring polymer of P beads
per atom (imaginary-time path integral, Trotter factorisation):

    H_P(q, p) = sum_k sum_i [ |p_i^k|^2 / (2 m_i) + m_i omega_P^2 |q_i^k - q_i^(k+1)|^2 / 2 ] + U(q),
    U(q)      = sum_k V(q^k),         omega_P = P kB T / hbar,         bead index k = 0..P-1 cyclic,

sampled classically at P T (beta_P = beta / P), with the physical masses on every bead.  Quantum
averages of position-dependent observables are bead averages; <V> = <U> / P.

Integrator (Ceriotti, Parrinello, Markland & Manolopoulos, JCP 133, 124104 (2010), in the BAOAB
order of Liu, Li & Liu, JCP 145, 024103 (2016)):

    B(dt/2)  p^k += dt/2 f^k                                (physical forces, one evaluation per step)
    A(dt/2)  free ring polymer, exactly, in normal modes    (harmonic mode frequencies omega_k)
    O(dt)    thermostat in normal modes
    A(dt/2), forces, B(dt/2).

NVE (RPMD) is B A(dt) B.  Normal modes are real and orthonormal, ordered by frequency,
q~_l = sum_j C_jl q_j with C = [1/sqrt(P), sqrt(2/P) cos(2 pi j l/P), sqrt(2/P) sin(2 pi j l/P), ...,
(-1)^j/sqrt(P) for even P], omega_l = 2 omega_P sin(pi l / P).  The free ring-polymer step is the
exact harmonic rotation ("exact") or its Cayley transform (Korol, Bou-Rabee & Miller, JCP 151,
124103 (2019); "cayley", the default): p' = [(1-a^2) p - m w^2 h q] / (1+a^2), q' = [h p/m + (1-a^2) q] / (1+a^2),
a = w h / 2.  Both conserve the free ring-polymer energy of every mode (Cayley = implicit midpoint on
a quadratic Hamiltonian), so both sample the free ring polymer exactly; Cayley is strongly stable
(no resonance of stiff modes with the physical forces as P grows).

Thermostats (the O step, at kB T_P = P kB T, heat booked so that econs = H_P - heat is conserved):
  "pimd"   PILE: Langevin on every internal mode with gamma_l = 2 lam omega_l (lam = 1: critical
           damping of the free modes), and on the centroid either Langevin with gamma_0 = 1/tau0
           (PILE-L, thermostat="pile-l") or Bussi's global stochastic rescaling with time constant
           tau0 (PILE-G, "pile-g": gentle on the centroid dynamics and the dipole predictor).
  "trpmd"  thermostatted RPMD (Rossi, Ceriotti & Manolopoulos, JCP 140, 234116 (2014)): the
           internal modes as above with lam = 1/2 by default, no thermostat on the centroid, whose
           dynamics estimates Kubo-transformed correlation functions (e.g. diffusion).
  "rpmd"   no thermostat (NVE ring polymer, Craig & Manolopoulos 2004).

Estimators (per atom, then summed or averaged by element):
  primitive        K_i = 3 P kB T / 2 - (1/P) sum_k m_i omega_P^2 |q_i^k - q_i^(k+1)|^2 / 2
  centroid virial  K_i = 3 kB T / 2 - (1 / 2P) sum_k (q_i^k - qbar_i) . f_i^k
Both have the same average (for a harmonic oscillator exactly <K> = <V> = omega^2/beta sum_l
1/(omega_l^2 + omega^2) / 2 at any P); the centroid virial one has a P-independent variance.

Engines.  `PotentialEngine` wraps any JAX potential V(x, box) (tests, model systems).  `PGMBeads`
runs the pGM force field of a `FlexibleSimulation` on every bead (vmapped: one program for all
beads), each bead with its own induced dipoles and predictor history (the step counter of the
predictor is shared, so its fused / unfused switch stays a branch).  One neighbour list, built from
the centroid, serves every bead: the list radius is enlarged by `bead_margin`, the largest
distance of a bead atom from its centroid (checked after every block).
Ring-polymer contraction (Markland & Manolopoulos, JCP 129, 024105 (2008); `contract=P'`): the
potential is split as V = V_mono + (V - V_mono), where V_mono is the sum of the gas-phase monomer
energies of the fitted flexible templates (bonded terms + intramolecular pGM + intramolecular van
der Waals: exactly the model the bonded terms were fitted with, cheap and stiff), evaluated on all
P beads, and the intermolecular remainder (PME, induction, van der Waals: expensive and smooth on
the scale of the ring polymer) on P' beads obtained by truncating the normal modes,
q' = T q with T = sqrt(P'/P) C'_{jl} C_{kl} over the P' lowest modes; the forces return by T^T
and U = sum_k V_mono(q^k) + (P/P') sum_k' [V - V_mono](q'^k').  P' = 1 puts the intermolecular
forces on the centroid.

NPT: isotropic Monte Carlo barostat, every bead of a molecule translated with the centroid's
molecular centre of mass (PIMDIntegrator).  Force beads are evaluated in lax.map chunks of 8
vmapped beads by default (bead_chunk; twice as fast as one vmap over 32 beads of 512 waters).

Rigid bodies and constraints are not supported: a rigid-rotor path integral is not a ring polymer
of atoms.  Quantum water needs flexible molecules (FlexibleTemplate; flexible_water below builds a
flexible pGM water with the q-TIP4P/F monomer surface; docs/pimd.md).  Units: nm, ps, amu, kJ/mol, K."""
from __future__ import annotations

import math
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from ._jaxmd import dataclasses
from .box import inv3, max_cutoff, volume
from .integrate import KB
from .io import NetCDFTrajectory, write_restart
from .neighbors import AtomNeighbors, MoleculeNeighbors
from .thermostats import Bussi
from ..units import DEBYE_E_NM

HBAR = 0.0635077993                  # kJ/mol ps  (1.054571817e-34 J s * N_A)
KJMOL_TO_MEV = 10.364269656262175    # 1 kJ/mol per particle in meV
BAR = 16.605390671738466             # bar per kJ/mol/nm^3
FORMAT = "pgm_jax pimd 1"


# ----------------------------------------------------------------------------- ring polymer
def normal_modes(P: int):
    """Real orthonormal normal-mode matrix C (P, P) (bead j, mode column l) ordered by frequency:
    centroid, cos 1, sin 1, cos 2, sin 2, ..., and (-1)^j / sqrt(P) for even P; and the frequency
    index of every column (omega = 2 omega_P sin(pi idx / P))."""
    P = int(P)
    j = np.arange(P)
    cols, idx = [np.full(P, 1.0 / math.sqrt(P))], [0]
    for l in range(1, (P - 1) // 2 + 1):
        cols.append(math.sqrt(2.0 / P) * np.cos(2.0 * math.pi * j * l / P)); idx.append(l)
        cols.append(math.sqrt(2.0 / P) * np.sin(2.0 * math.pi * j * l / P)); idx.append(l)
    if P % 2 == 0 and P > 1:
        cols.append((-1.0) ** j / math.sqrt(P)); idx.append(P // 2)
    return np.stack(cols, 1), np.array(idx, int)


def contraction_matrix(P: int, Pc: int) -> np.ndarray:
    """(Pc, P) ring-polymer contraction (Markland & Manolopoulos 2008): the P' = Pc lowest normal
    modes of the P-bead polymer, rescaled by sqrt(Pc / P), on Pc beads.  T T^T = (Pc/P) I, the
    centroid is kept, and Pc = P gives the identity."""
    if not 1 <= Pc <= P:
        raise ValueError("need 1 <= P' <= P")
    C, _ = normal_modes(P)
    Cc, _ = normal_modes(Pc)
    return math.sqrt(Pc / P) * Cc @ C[:, :Pc].T


class RingPolymer:
    """Normal modes and free ring-polymer propagation of P beads at temperature T (K)."""

    def __init__(self, nbeads: int, temperature: float):
        self.P = int(nbeads)
        if self.P < 1:
            raise ValueError("at least one bead")
        self.T = float(temperature)
        self.kT = KB * self.T
        self.kT_P = self.P * self.kT
        self.omega_P = self.P * self.kT / HBAR
        C, idx = normal_modes(self.P)
        self.C = C
        self.omega = 2.0 * self.omega_P * np.sin(np.pi * idx / self.P)     # (P,) rad/ps
        self.omega[0] = 0.0

    def to_nm(self, x):
        """Bead array (P, ...) -> normal-mode coordinates (P, ...)."""
        return jnp.tensordot(jnp.asarray(self.C.T), x, axes=1)

    def from_nm(self, y):
        return jnp.tensordot(jnp.asarray(self.C), y, axes=1)

    def spring(self, q, mass):
        """sum_k sum_i m_i omega_P^2 |q_i^k - q_i^(k+1)|^2 / 2 per atom, (N,) kJ/mol."""
        d = q - jnp.roll(q, -1, axis=0)
        return 0.5 * self.omega_P ** 2 * mass[:, 0] * jnp.sum(d * d, axis=(0, 2))

    def propagator(self, h: float, mass, kind: str = "cayley"):
        """Free ring-polymer step of length h for every mode and atom: p' = a p + b q, q' = c p + d q
        (arrays (P, N, 1)); mass (N, 1)."""
        w = self.omega[:, None, None]
        m = np.asarray(mass, float)[None]
        if kind == "exact":
            wh = w * h
            cs, sn = np.cos(wh), np.sin(wh)
            safe = np.where(w > 0, w, 1.0)
            a, d = cs + 0 * m, cs + 0 * m
            b = -m * w * sn
            c = np.where(w > 0, sn / (m * safe), h / m)
        elif kind == "cayley":
            x = (0.5 * w * h) ** 2
            den = 1.0 + x
            a = d = (1.0 - x) / den + 0 * m
            b = -m * w * w * h / den
            c = h / m / den + 0 * w
        else:
            raise ValueError("propagator: 'cayley' | 'exact'")
        return tuple(jnp.asarray(np.broadcast_to(v, (self.P,) + m.shape[1:]).copy()) for v in (a, b, c, d))

    def sample_free(self, key, centroid, mass):
        """Beads (P, N, 3) around a centroid (N, 3) drawn from the free ring-polymer distribution
        at T_P: internal mode l ~ N(0, kT_P / (m omega_l^2))."""
        if self.P == 1:
            return jnp.asarray(centroid)[None]
        m = jnp.asarray(mass)[None]
        w = jnp.asarray(self.omega[1:])[:, None, None]
        z = jax.random.normal(key, (self.P - 1,) + jnp.shape(centroid), jnp.float64)
        y = jnp.concatenate([math.sqrt(self.P) * jnp.asarray(centroid)[None], z * jnp.sqrt(self.kT_P / m) / w], 0)
        return self.from_nm(y)


class PILE:
    """Path-integral Langevin thermostat in normal modes (Ceriotti et al. 2010).

    mode "pimd": internal modes gamma_l = 2 lam omega_l, centroid Langevin 1/tau0 ("pile-l") or
    Bussi rescaling with time constant tau0 ("pile-g"); "trpmd": internal modes only (lam = 1/2
    by default); "rpmd": none."""

    def __init__(self, ring: RingPolymer, mode: str = "pimd", thermostat: str = "pile-l", tau0: float = 0.2,
                 lam: float | None = None):
        mode, thermostat = mode.lower(), thermostat.lower()
        if mode not in ("pimd", "trpmd", "rpmd"):
            raise ValueError("mode: 'pimd' | 'trpmd' | 'rpmd'")
        if thermostat not in ("pile-l", "pile-g"):
            raise ValueError("thermostat: 'pile-l' | 'pile-g'")
        self.ring, self.mode, self.kind, self.tau0 = ring, mode, thermostat, float(tau0)
        self.lam = (0.5 if mode == "trpmd" else 1.0) if lam is None else float(lam)
        g = 2.0 * self.lam * ring.omega
        g[0] = 1.0 / self.tau0 if (mode == "pimd" and thermostat == "pile-l") else 0.0
        if mode == "rpmd":
            g[:] = 0.0
        self.gamma = g
        self.centroid_bussi = mode == "pimd" and thermostat == "pile-g"
        self._bussi = Bussi(self.tau0)

    @property
    def active(self) -> bool:
        return self.mode != "rpmd"

    def describe(self) -> str:
        if self.mode == "rpmd":
            return "RPMD (no thermostat)"
        c = {"pile-l": f"centroid Langevin {1.0 / self.tau0:g}/ps", "pile-g": f"centroid Bussi {self.tau0:g} ps"}
        cen = c[self.kind] if self.mode == "pimd" else "centroid free"
        return f"{'PILE' if self.mode == 'pimd' else 'TRPMD'} (internal modes gamma = {2 * self.lam:g} omega_l, {cen})"

    def apply(self, pn, mass, key, h: float, dof: float):
        """O step of length h on normal-mode momenta pn (P, N, 3); returns pn and the kinetic
        energy change (heat taken up)."""
        k1, k2 = jax.random.split(key)
        m = mass[None]
        e0 = 0.5 * jnp.sum(pn * pn / m)
        c = jnp.asarray(np.exp(-self.gamma * h))[:, None, None]
        s = jnp.sqrt((1.0 - c * c) * self.ring.kT_P * m)
        out = c * pn + s * jax.random.normal(k1, pn.shape, pn.dtype)
        if self.centroid_bussi:
            v = pn[0] / jnp.sqrt(mass)
            v, _ = self._bussi.apply(v, None, k2, h, self.ring.kT_P, dof, lambda u: u, None)
            out = out.at[0].set(v * jnp.sqrt(mass))
        return out, 0.5 * jnp.sum(out * out / m) - e0


# ----------------------------------------------------------------------------- state and integrator
@dataclasses.dataclass
class PIMDState:
    q: jnp.ndarray            # (P, N, 3) bead positions, nm (ring polymers whole)
    p: jnp.ndarray            # (P, N, 3) bead momenta, amu nm / ps
    f: jnp.ndarray            # (P, N, 3) forces -dU/dq, kJ/mol/nm
    upot: jnp.ndarray         # () U = sum_k V(q^k) (with contraction: the contracted U), kJ/mol
    box: jnp.ndarray          # (3, 3) nm
    eng: object               # engine state (induced dipoles, neighbour list, ...)
    rng: jnp.ndarray
    heat: jnp.ndarray         # () heat taken up by the thermostat since the start, kJ/mol
    step: jnp.ndarray
    mc: jnp.ndarray = None    # barostat (tries, accepts, window tries, window accepts)
    mc_dv: jnp.ndarray = None  # current maximum volume change (nm^3)


class PIMDIntegrator:
    """BAOAB ring-polymer integrator for an engine with init(q, box) -> eng and
    compute(q, box, eng) -> (forces (P, N, 3), U, eng).

    ensemble "npt": isotropic Monte Carlo barostat every `barostat_interval` steps (engines with
    molecules: scale(q, s), energy(q, box, eng), nmol).  Every bead of a molecule is translated with
    the molecular centre of mass of the centroid, which leaves the springs and the intramolecular
    terms unchanged; acceptance on (U' - U) / P + p dV - N_mol kT ln(V'/V) (the ring polymer
    isomorphism at beta_P = beta / P), step size adapted to 25-75 % acceptance."""

    def __init__(self, engine, masses, nbeads: int, temperature: float, dt: float, mode: str = "pimd",
                 thermostat: str = "pile-l", tau0: float = 0.2, lam: float | None = None,
                 propagator: str = "cayley", ensemble: str = "nvt", pressure: float = 1.0,
                 barostat_interval: int = 100):
        if ensemble not in ("nvt", "npt"):
            raise ValueError("ensemble: 'nvt' | 'npt' (NVE: mode='rpmd')")
        if ensemble == "npt" and not hasattr(engine, "scale"):
            raise ValueError("the barostat needs an engine with molecules (PGMBeads)")
        self.ensemble = ensemble
        self.pressure = float(pressure) / BAR              # bar -> kJ/mol/nm^3
        self.interval = int(barostat_interval)
        self.engine = engine
        self.mass = jnp.asarray(np.asarray(masses, float).reshape(-1, 1))
        self.n = int(self.mass.shape[0])
        self.ring = RingPolymer(nbeads, temperature)
        self.P = self.ring.P
        self.dt = float(dt)
        self.propagator = propagator
        self.set_thermostat(mode, thermostat, tau0, lam)

    def set_thermostat(self, mode: str = "pimd", thermostat: str = "pile-l", tau0: float = 0.2,
                       lam: float | None = None):
        """(Re)configure the thermostat (e.g. PIMD equilibration, then TRPMD or RPMD) and re-jit."""
        self.thermo = PILE(self.ring, mode, thermostat, tau0, lam)
        self.mode = self.thermo.mode
        self.dof = 3.0 * self.n
        h = self.dt if self.mode == "rpmd" else 0.5 * self.dt
        self._A = self.ring.propagator(h, np.asarray(self.mass), self.propagator)
        self.run = jax.jit(self._run)
        self.forces = jax.jit(self._forces)

    # ------------------------------------------------------------------ pieces
    def _free(self, qn, pn):
        a, b, c, d = self._A
        return c * pn + d * qn, a * pn + b * qn

    def _forces(self, st: PIMDState) -> PIMDState:
        f, U, eng = self.engine.compute(st.q, st.box, st.eng)
        return st.set(f=f, upot=U, eng=eng)

    def init(self, q, box, key, momenta=None, spread: bool = True) -> PIMDState:
        """q: centroid positions (N, 3) (beads drawn from the free ring polymer if `spread`, else
        all on the centroid) or beads (P, N, 3); momenta (P, N, 3) or drawn at kT_P."""
        q = jnp.asarray(q, jnp.float64)
        box = jnp.asarray(box, jnp.float64)
        k1, k2, k3 = jax.random.split(key, 3)
        if q.ndim == 2:
            q = self.ring.sample_free(k1, q, self.mass) if spread else jnp.broadcast_to(q, (self.P,) + q.shape)
        if momenta is None:
            p = jnp.sqrt(self.ring.kT_P * self.mass)[None] * jax.random.normal(k2, q.shape, jnp.float64)
        else:
            p = jnp.asarray(momenta, jnp.float64)
        z = jnp.zeros((), jnp.float64)
        st = PIMDState(q=q, p=p, f=jnp.zeros_like(q), upot=z, box=box, eng=self.engine.init(q, box), rng=k3,
                       heat=z, step=jnp.zeros((), jnp.int32), mc=jnp.zeros(4, jnp.int32),
                       mc_dv=jnp.asarray(0.01 * float(volume(box)), jnp.float64))
        return self.forces(st)

    def _step(self, st: PIMDState) -> PIMDState:
        dt = self.dt
        p = st.p + 0.5 * dt * st.f
        qn, pn = self.ring.to_nm(st.q), self.ring.to_nm(p)
        rng, heat = st.rng, st.heat
        if self.mode == "rpmd":
            qn, pn = self._free(qn, pn)
        else:
            qn, pn = self._free(qn, pn)
            rng, k = jax.random.split(rng)
            pn, dh = self.thermo.apply(pn, self.mass, k, dt, self.dof)
            heat = heat + dh
            qn, pn = self._free(qn, pn)
        st = st.set(q=self.ring.from_nm(qn), p=self.ring.from_nm(pn), rng=rng, heat=heat)
        st = self._forces(st)
        st = st.set(p=st.p + 0.5 * dt * st.f, step=st.step + 1)
        if self.ensemble == "npt":
            st = jax.lax.cond(st.step % self.interval == 0, self._barostat, lambda s: s, st)
        return st

    def _barostat(self, st: PIMDState) -> PIMDState:
        eng = self.engine
        key, k1, k2 = jax.random.split(st.rng, 3)
        V = volume(st.box)
        dV = (2.0 * jax.random.uniform(k1, dtype=jnp.float64) - 1.0) * st.mc_dv
        Vn = V + dV
        s = jnp.cbrt(jnp.maximum(Vn, 1e-12) / V)
        qn, Hn = eng.scale(st.q, s), st.box * s
        Un, en = eng.energy(qn, Hn, st.eng)
        kT = self.ring.kT
        w = (Un - st.upot) / self.P + self.pressure * dV - eng.nmol * kT * jnp.log(jnp.maximum(Vn, 1e-12) / V)
        accept = (Vn > 0) & (jnp.log(jax.random.uniform(k2, dtype=jnp.float64)) < -w / kT)
        st = st.set(rng=key, eng=eng.flag(st.eng, en))
        st = jax.lax.cond(accept, lambda s: self._forces(s.set(q=qn, box=Hn, eng=en)), lambda s: s, st)
        mc = st.mc + jnp.array([1, 0, 1, 0], jnp.int32) + accept.astype(jnp.int32) * jnp.array([0, 1, 0, 1], jnp.int32)
        adapt = mc[2] >= 10
        rate = mc[3] / jnp.maximum(mc[2], 1)
        dv = jnp.where(adapt & (rate < 0.25), st.mc_dv / 1.1, jnp.where(adapt & (rate > 0.75), st.mc_dv * 1.1, st.mc_dv))
        dv = jnp.minimum(dv, 0.3 * volume(st.box))
        mc = jnp.where(adapt, mc.at[2].set(0).at[3].set(0), mc)
        return st.set(mc=mc, mc_dv=dv)

    def _run(self, st: PIMDState, n) -> PIMDState:
        if hasattr(self.engine, "reset_block"):
            st = st.set(eng=self.engine.reset_block(st.eng))
        return jax.lax.fori_loop(0, n, lambda _, s: self._step(s), st)

    # ------------------------------------------------------------------ estimators
    def estimators(self, st: PIMDState) -> dict:
        """Per-atom primitive and centroid-virial kinetic energies (N,), bead kinetic energy,
        spring energy, conserved quantity, centroid and mode temperatures (K)."""
        ring, m = self.ring, self.mass
        q, p, f = st.q, st.p, st.f
        qc = jnp.mean(q, 0)
        spring_i = ring.spring(q, m)
        prim = 1.5 * ring.P * ring.kT - spring_i / ring.P
        cv = 1.5 * ring.kT - 0.5 / ring.P * jnp.sum((q - qc[None]) * f, axis=(0, 2))
        ke = 0.5 * jnp.sum(p * p / m[None])
        pn = ring.to_nm(p)
        t_mode = jnp.sum(pn * pn / m[None], axis=(1, 2)) / (3.0 * self.n * KB * ring.P)
        return {"prim": prim, "cv": cv, "ke_beads": ke, "spring": jnp.sum(spring_i),
                "hamiltonian": ke + jnp.sum(spring_i) + st.upot, "econs": ke + jnp.sum(spring_i) + st.upot - st.heat,
                "t_beads": 2.0 * ke / (3.0 * self.n * ring.P * KB) / ring.P, "t_modes": t_mode,
                "t_centroid": t_mode[0], "epot": st.upot / ring.P}


# ----------------------------------------------------------------------------- engines
class PotentialEngine:
    """Any potential V(x (N, 3), box) -> kJ/mol on every bead (vmapped value_and_grad).  With `soft`
    and `contract` = P', the potential is V + soft, and soft is evaluated on P' contracted beads
    (U = sum_k V(q^k) + (P/P') sum_k' soft(q'^k'), as for the pGM engine)."""

    def __init__(self, energy_fn, soft=None, contract: int | None = None):
        self.energy_fn, self.soft, self.contract = energy_fn, soft, contract
        self._vg = jax.vmap(jax.value_and_grad(energy_fn), in_axes=(0, None))
        self._sg = None if soft is None else jax.vmap(jax.value_and_grad(soft), in_axes=(0, None))

    def init(self, q, box):
        return jnp.zeros(())

    def compute(self, q, box, eng):
        V, g = self._vg(q, box)
        U, f = jnp.sum(V), -g
        if self.soft is not None:
            P = q.shape[0]
            Pc = P if self.contract is None else min(int(self.contract), P)
            Tm = jnp.asarray(contraction_matrix(P, Pc))
            Vs, gs = self._sg(jnp.tensordot(Tm, q, axes=1), box)
            U = U + P / Pc * jnp.sum(Vs)
            f = f - P / Pc * jnp.tensordot(Tm.T, gs, axes=1)
        return f, U, eng


@dataclasses.dataclass
class PGMBeadState:
    induction: object          # InductionState batched over the force beads (count shared)
    nbr: object                # neighbour list of the centroid (shared by every bead)
    iters: jnp.ndarray         # CG iterations of the last solve (largest over the beads)
    max_iters: jnp.ndarray     # largest since the block started
    resid: jnp.ndarray
    overflow: jnp.ndarray
    cg_total: jnp.ndarray      # CG iterations (largest over the beads) summed over the steps
    elec: jnp.ndarray          # force-bead averages of the force field's parts (kJ/mol)
    vdw: jnp.ndarray


def _nocount(ind):
    return ind.set(count=None)


class PGMBeads:
    """The pGM force field of a FlexibleSimulation on every bead of a ring polymer.

    Every bead (or contracted bead) keeps its own induced dipoles and predictor history; one
    neighbour list of the centroid serves all beads (its radius is enlarged by `bead_margin` nm,
    the largest allowed distance of a bead atom from its centroid).  contract = P' < P: ring-polymer
    contraction of the intermolecular part (module docstring).  bead_chunk: force beads per vmapped
    chunk, the chunks run one after the other in a lax.map ("auto": 8 when that divides more than 8
    beads; None: all at once); results are identical."""

    def __init__(self, sim, nbeads: int, contract: int | None = None, bead_margin: float = 0.08,
                 bead_chunk: int | str | None = "auto"):
        integ = sim.integ
        if getattr(integ, "cons", None) is not None:
            raise ValueError("path integrals need flexible molecules without constraints (constraints='none', "
                             "no RigidTemplate)")
        if getattr(integ, "vsites", None) is not None:
            raise NotImplementedError("virtual sites are not supported with path integrals yet")
        if getattr(integ, "mts", None) is not None or integ.alchemy is not None or integ.restraints is not None:
            raise NotImplementedError("path integrals with mts / alchemy / restraints are not supported yet")
        self.sim, self.ff, self.flex, self.integ = sim, sim.ff, sim.flex, integ
        self.P = int(nbeads)
        self.Pc = None if (contract is None or int(contract) >= self.P) else int(contract)
        self.nf = self.P if self.Pc is None else self.Pc
        if bead_chunk == "auto":            # 512 waters, P = 32: 15.2 ms/step vmapped at once, 8.0 in chunks of 8
            bead_chunk = 8 if (self.nf > 8 and self.nf % 8 == 0) else None
        self.chunk = None if not bead_chunk else int(bead_chunk)
        self.bead_margin = float(bead_margin)
        self.params = integ.params
        if self.Pc is not None:
            self.T = jnp.asarray(contraction_matrix(self.P, self.Pc))
            self._mono = self._monomer_groups()
        self.make_neighbors(np.asarray(sim.state.box))

    # ------------------------------------------------------------------ monomer reference (contraction)
    def _monomer_groups(self):
        groups = []
        covered = 0
        for tpl, rows in self.flex.groups:
            model = tpl.model
            P = jax.tree_util.tree_map(jnp.asarray, tpl.P)
            fn = (lambda R, model=model, P=P, idx=tpl.index: model.nonbonded(idx, R, P)[0])
            groups.append((fn, rows))
            covered += int(np.size(rows))
        if covered != self.flex.n:
            raise ValueError("ring-polymer contraction needs every molecule to be a FlexibleTemplate "
                             "(its gas-phase model is the stiff reference)")
        return groups

    def monomer_nonbonded(self, x):
        """Sum over molecules of the gas-phase intramolecular pGM + van der Waals energy of each
        template's fitted model at positions x (N, 3), kJ/mol."""
        e = 0.0
        for fn, rows in self._mono:
            e = e + jnp.sum(jax.vmap(fn)(x[rows]))
        return e

    def reference(self, x):
        """Stiff reference on one bead: bonded + monomer nonbonded (contraction), or bonded only."""
        e = self.flex.energy(x)
        if self.Pc is not None:
            e = e + self.monomer_nonbonded(x)
        return e

    # ------------------------------------------------------------------ neighbour lists
    def make_neighbors(self, H):
        sim, s = self.sim, self.sim.settings
        self.r_list = float(sim.r_list) + self.bead_margin
        mode = sim._nb_mode
        if mode == "auto":
            mode = "molecule" if MoleculeNeighbors.fits(H, s.pair_cutoff, s.skin, self.r_list) else "atom"
        if mode == "molecule":
            self.nb = MoleculeNeighbors(sim.topology.group, sim.topology.n_group, self.r_list, H, s.pair_cutoff, s.skin)
        else:                    # pairs of bead atoms within the cutoff: centroid atoms within cutoff + 2 margins
            rc = s.pair_cutoff + 2.0 * self.bead_margin
            skin = min(s.skin, max_cutoff(H) - rc - 0.002)
            if skin < 0.02:
                raise ValueError(f"box too small for the bead margin: cutoff {s.pair_cutoff} + 2 x {self.bead_margin} "
                                 f"leaves a list skin of {skin:.3f} nm (half the box height {max_cutoff(H):.3f} nm)")
            self.nb = AtomNeighbors(sim.sys.n, H, rc, skin)
        self._nb_volume = float(volume(jnp.asarray(H)))

    def _centers(self, qc):
        return self.flex.list_centers(qc)

    def force_positions(self, q):
        """Positions at which the force field is evaluated: all beads, or the contracted beads."""
        return q if self.Pc is None else jnp.tensordot(self.T, q, axes=1)

    def extent(self, q):
        """Largest distance (nm) of a bead atom from what the list is built on: its centroid list
        group's centre (molecule list) or its centroid (atom list)."""
        qc = jnp.mean(q, 0)
        if self.nb.kind == "molecule":
            c = self._centers(qc)[self.flex.group]
            return jnp.max(jnp.linalg.norm(q - c[None], axis=-1))
        return jnp.max(jnp.linalg.norm(q - qc[None], axis=-1))

    def limit(self) -> float:
        return self.r_list if self.nb.kind == "molecule" else self.bead_margin

    def size(self, q, box, factor: float = 1.2, nbr=None):
        """Static sizes (molecule-list width, row capacities) for the largest of the force beads."""
        qc = jnp.mean(q, 0)
        c = self._centers(qc)
        nbr = self.nb.allocate(qc, c, box) if nbr is None else nbr
        x = self.force_positions(q)
        if self.nb.kind == "molecule":
            caps = [self.nb.size(nbr, c, box, x[k], factor) for k in range(x.shape[0])]
            self.nb.cap = max(caps)
        ff = self.ff
        counts = np.array([np.asarray(jax.jit(ff.pair_counts)(x[k], box, self.nb.candidates(nbr, c, box, x[k])[0]))
                           for k in range(x.shape[0])])
        caps = []
        for k in {int(np.argmax(counts[:, 0])), int(np.argmax(counts[:, 1]))}:
            ff.size_rows(x[k], box, self.nb.candidates(nbr, c, box, x[k])[0], factor)
            caps.append(ff.capacity)
        ff.fit_rows(caps)
        return nbr

    # ------------------------------------------------------------------ forces
    def init(self, q, box):
        ind0 = self.ff.init_induction()
        ind = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (self.nf,) + jnp.shape(x)), _nocount(ind0))
        ind = ind.set(count=ind0.count)
        qc = jnp.mean(q, 0)
        nbr = self.nb.allocate(qc, self._centers(qc), box)
        z, zi = jnp.zeros((), jnp.float64), jnp.zeros((), jnp.int32)
        return PGMBeadState(induction=ind, nbr=nbr, iters=zi, max_iters=zi, resid=z, overflow=jnp.zeros((), bool),
                            cg_total=z, elec=z, vdw=z)

    def reset_block(self, e: PGMBeadState) -> PGMBeadState:
        return e.set(max_iters=jnp.zeros((), jnp.int32), resid=jnp.zeros((), jnp.float64),
                     overflow=jnp.zeros((), bool))

    def _ind_axes(self, ind):
        return jax.tree_util.tree_map(lambda _: 0, _nocount(ind))

    def _over_beads(self, fn, x, ind, n_out: int, k_ind: int):
        """fn(x_k, ind_k) -> n_out outputs (output k_ind an InductionState) for every force bead:
        jax.vmap over all of them, or with `bead_chunk` a lax.map over chunks of vmapped beads (less
        memory traffic per kernel for many beads).  The predictor step counter stays unbatched."""
        ax = self._ind_axes(ind)
        out_ax = tuple(ax if i == k_ind else 0 for i in range(n_out))
        nf = x.shape[0]
        c = self.chunk
        if c is None or c >= nf or nf % c:
            return jax.vmap(fn, in_axes=(0, ax), out_axes=out_ax)(x, ind)
        count = ind.count
        split = lambda a: a.reshape((nf // c, c) + a.shape[1:])                     # noqa: E731
        xs = split(x)
        inds = jax.tree_util.tree_map(split, _nocount(ind))
        vf = jax.vmap(fn, in_axes=(0, ax), out_axes=out_ax)

        def body(args):
            xc, ic = args
            out = vf(xc, ic.set(count=count))
            return tuple(o.set(count=None) if i == k_ind else o for i, o in enumerate(out)), out[k_ind].count
        out, counts = jax.lax.map(body, (xs, inds))
        merge = lambda a: a.reshape((nf,) + a.shape[2:])                            # noqa: E731
        return tuple(jax.tree_util.tree_map(merge, o).set(count=counts[0]) if i == k_ind else merge(o)
                     for i, o in enumerate(out))

    def compute(self, q, box, e: PGMBeadState):
        qc = jnp.mean(q, 0)
        c = self._centers(qc)
        nbr = self.nb.update(e.nbr, qc, c, box)
        x = self.force_positions(q)
        ff, params, nb = self.ff, self.params, self.nb

        def one(xk, ind):
            cand, ovf = nb.candidates(nbr, c, box, xk)
            res = ff.compute(xk, box, cand, ind, params)
            return (res.energy["total"], res.energy["elec"], res.energy["vdw"], res.forces, res.induction,
                    res.iterations, res.residual, res.overflow | ovf)

        E, El, Ev, F, ind, it, err, ovf = self._over_beads(one, x, e.induction, 8, 4)
        ref_vg = jax.vmap(jax.value_and_grad(self.reference))
        if self.Pc is None:
            eb, gb = ref_vg(q)                                # bonded terms on every bead
            f = F - gb
            U = jnp.sum(E) + jnp.sum(eb)
        else:
            em, gm = jax.vmap(jax.value_and_grad(self.monomer_nonbonded))(x)
            dE, dF = E - em, F + gm                           # intermolecular remainder on the contracted beads
            er, gr = ref_vg(q)                                # stiff reference on every bead
            scale = self.P / self.Pc
            f = scale * jnp.tensordot(self.T.T, dF, axes=1) - gr
            U = scale * jnp.sum(dE) + jnp.sum(er)
        itm = jnp.max(it)
        e = e.set(induction=ind, nbr=nbr, iters=itm, max_iters=jnp.maximum(e.max_iters, itm),
                  resid=jnp.maximum(e.resid, jnp.max(err)), overflow=e.overflow | jnp.any(ovf),
                  cg_total=e.cg_total + itm, elec=jnp.mean(El), vdw=jnp.mean(Ev))
        return f, U, e

    # ------------------------------------------------------------------ barostat
    @property
    def nmol(self) -> int:
        return self.flex.nmol

    def scale(self, q, s):
        """Every bead of a molecule translated by (s - 1) times the centroid's molecular centre of mass."""
        com = self.flex.centers(jnp.mean(q, 0))
        return q + ((s - 1.0) * com)[self.flex.mol][None]

    def flag(self, e: PGMBeadState, trial: PGMBeadState) -> PGMBeadState:
        """Keep the overflow flag of a trial evaluation (the block is repeated if it overflowed)."""
        return e.set(overflow=e.overflow | trial.overflow)

    def energy(self, q, box, e: PGMBeadState):
        """U at (q, box) with the dipoles solved from the last converged ones (no predictor history
        update, Monte Carlo trials); the neighbour list is rebuilt.  Returns (U, engine state)."""
        qc = jnp.mean(q, 0)
        c = self._centers(qc)
        nbr = self.nb.update(e.nbr, qc, c, box, True)
        x = self.force_positions(q)
        ff, params, nb = self.ff, self.params, self.nb

        def one(xk, ind):
            cand, ovf = nb.candidates(nbr, c, box, xk)
            E, ind, it, ovf2 = ff.energy(xk, box, cand, ind, params)
            return E, ind, ovf | ovf2

        E, ind, ovf = self._over_beads(one, x, e.induction, 3, 1)
        ref = jax.vmap(self.reference)
        if self.Pc is None:
            U = jnp.sum(E) + jnp.sum(ref(q))
        else:
            U = self.P / self.Pc * jnp.sum(E - jax.vmap(self.monomer_nonbonded)(x)) + jnp.sum(ref(q))
        return U, e.set(nbr=nbr, induction=ind, overflow=e.overflow | jnp.any(ovf))

    # ------------------------------------------------------------------ pressure
    def strain_derivative(self, q, box, e: PGMBeadState):
        """(3, 3) (1/P) dU/d eps, with every bead of a molecule translated with the molecular centre
        of mass of the centroid (molecular centroid virial; the intramolecular reference does not
        change under such a strain)."""
        ff, flex = self.ff, self.flex
        qc = jnp.mean(q, 0)
        com = flex.centers(qc)
        c = self._centers(qc)
        x = self.force_positions(q)
        mu = e.induction.mu
        P = ff._atoms(self.params)

        def W(xk, muk):
            cand, _ = self.nb.candidates(e.nbr, c, box, xk)

            def en(eps):
                Fm = jnp.eye(3) + eps
                return ff.energy_fixed_mu(xk + (com @ eps.T)[flex.mol], box @ Fm.T, muk, cand, P)[0]
            return jax.grad(en)(jnp.zeros((3, 3))) - ff._vdw_tail(P, box) * jnp.eye(3)

        return jnp.mean(jax.vmap(W)(x, mu), 0)


# ----------------------------------------------------------------------------- driver
class PIMDSimulation:
    """Path-integral MD of a FlexibleSimulation's system (flexible molecules, NVT or NPT).

        sim = FlexibleSimulation(sys, [tpl] * n, pos, H, MDSettings(), dt=0.00025, ensemble="nvt", temperature=298)
        pi = PIMDSimulation(sim, beads=32, mode="pimd", thermostat="pile-g", tau0=0.5)
        pi.run(40000, report=400, traj=400, prefix="qwater")        # centroid trajectory qwater.nc
        pi.set_mode("trpmd"); pi.run(...)                           # dynamics from the PIMD ensemble

    dt, temperature and settings come from `sim` (its own integrator and thermostat are not used).
    contract: ring-polymer contraction of the intermolecular forces to P' beads (None: off).
    bead_margin: largest distance (nm) of a bead atom from its centroid (neighbour lists).
    ensemble "npt": Monte Carlo barostat at `pressure` (bar) every `barostat_interval` steps.
    bead_chunk: force beads per vmapped chunk ("auto": 8 for more than 8 beads)."""

    def __init__(self, sim, beads: int = 32, mode: str = "pimd", thermostat: str = "pile-l", tau0: float = 0.2,
                 lam: float | None = None, propagator: str = "cayley", contract: int | None = None,
                 bead_margin: float = 0.08, seed: int = 0, dt: float | None = None, spread: bool = True,
                 ensemble: str = "nvt", pressure: float = 1.0, barostat_interval: int = 100,
                 bead_chunk: int | str | None = "auto", log=sys.stdout):
        self.sim, self.log = sim, log
        self.engine = PGMBeads(sim, beads, contract, bead_margin, bead_chunk)
        self.P = int(beads)
        self.dt = float(sim.dt if dt is None else dt)
        self.T0 = float(sim.T0)
        self.integ = PIMDIntegrator(self.engine, np.asarray(sim.flex.masses), beads, self.T0, self.dt, mode,
                                    thermostat, tau0, lam, propagator, ensemble, pressure, barostat_interval)
        self.ensemble = ensemble
        self.elements = np.array(sim.sys.elements)
        H = jnp.asarray(sim.state.box)
        pos = sim.state.dyn.position
        key = jax.random.PRNGKey(int(seed))
        k1, k2 = jax.random.split(key)
        q = self.integ.ring.sample_free(k1, pos, self.integ.mass) if spread else jnp.broadcast_to(pos, (self.P,) + pos.shape)
        self._size(q, H)
        self.state = self.integ.init(q, H, k2)
        self.time_ps = 0.0
        e = self.engine
        self._print(f"# pgm_jax PIMD: {sim.sys.nmol} molecules, {sim.sys.n} atoms, {self.P} beads"
                    f"{'' if e.Pc is None else f' (intermolecular forces contracted to {e.Pc})'}, "
                    f"{self.integ.thermo.describe()}, {ensemble.upper()}"
                    f"{f' ({pressure:g} bar, Monte Carlo every {barostat_interval} steps)' if ensemble == 'npt' else ''}, "
                    f"T {self.T0:g} K, dt {self.dt * 1000:g} fs, "
                    f"{propagator} free ring-polymer step, {e.nb.kind} neighbour list of the centroid "
                    f"(bead margin {e.bead_margin:g} nm), {sim.settings.precision} precision, device {jax.devices()[0]}")

    def _print(self, s):
        if self.log is not None:
            print(s, file=self.log, flush=True)

    def set_mode(self, mode: str, thermostat: str | None = None, tau0: float | None = None, lam: float | None = None):
        """Switch between "pimd", "trpmd" and "rpmd" (same state; the conserved quantity restarts)."""
        th = self.integ.thermo
        self.integ.set_thermostat(mode, thermostat or th.kind, th.tau0 if tau0 is None else tau0, lam)
        self._print(f"# thermostat: {self.integ.thermo.describe()}")

    # ------------------------------------------------------------------ sizes and blocks
    def _size(self, q, H, factor=1.2, nbr=None):
        nbr = self.engine.size(q, H, factor, nbr)
        self._w_jit = None
        self.integ.run = jax.jit(self.integ._run)
        self.integ.forces = jax.jit(self.integ._forces)
        return nbr

    def _wrap(self, st: PIMDState) -> PIMDState:
        """Whole ring polymers of whole molecules shifted so that the centroid's molecular centres
        of mass lie in the primary cell."""
        flex = self.sim.flex
        H = st.box
        qc = jnp.mean(st.q, 0)
        fr = jnp.matmul(flex.centers(qc), inv3(H), precision=jax.lax.Precision.HIGHEST)
        shift = jnp.matmul(jnp.floor(fr), H, precision=jax.lax.Precision.HIGHEST)[flex.mol]
        return st.set(q=st.q - shift[None])

    def _rebuild_neighbors(self):
        H = np.asarray(self.state.box)
        self._print(f"# step {int(self.state.step)}: neighbour lists rebuilt for volume {float(volume(H)):.3f} nm^3")
        self.engine.make_neighbors(H)
        nbr = self._size(self.state.q, H)
        redo = self.integ.forces(self.state.set(eng=self.state.eng.set(nbr=nbr)))
        self.state = redo.set(eng=redo.eng.set(induction=self.state.eng.induction))

    def _advance(self, n: int):
        """n steps; under NPT the lists are rebuilt when the volume has drifted by 10 %, and a block that
        keeps overflowing (a box shrinking fast) is split in halves with rebuilds in between."""
        if abs(float(volume(self.state.box)) / self.engine._nb_volume - 1.0) > 0.10:
            self._rebuild_neighbors()
        try:
            self._advance_block(n)
        except RuntimeError as err:
            if "overflowing" not in str(err) or n < 2:
                raise
            self._rebuild_neighbors()
            self._advance(n // 2)
            self._advance(n - n // 2)

    def _advance_block(self, n: int):
        e = self.engine
        start = self.state
        for attempt in range(6):
            new = self.integ.run(start, n)
            jax.block_until_ready(new.upot)
            nb_bad, row_bad = e.nb.failed(new.eng.nbr), bool(new.eng.overflow)
            if not (nb_bad or row_bad):
                break
            old = (e.ff.capacity, getattr(e.nb, "cap", None))
            nbr = self._size(start.q, start.box, 1.3, None if nb_bad else start.eng.nbr)
            if row_bad:
                e.ff.grow_rows(old[0])
                if getattr(e.nb, "cap", None) is not None and old[1] is not None:
                    e.nb.cap = max(e.nb.cap, old[1] + 4)
                self.integ.run = jax.jit(self.integ._run)
                self.integ.forces = jax.jit(self.integ._forces)
            self._print(f"# {'neighbour list' if nb_bad else 'row capacity'} overflow in steps {int(start.step)}-"
                        f"{int(start.step) + n}: resized, repeating")
            redo = self.integ.forces(start.set(eng=start.eng.set(nbr=nbr)))
            start = redo.set(eng=redo.eng.set(induction=start.eng.induction))   # no second history entry
        else:
            raise RuntimeError("neighbour list keeps overflowing")
        new = self._wrap(new)
        ext = float(jax.jit(e.extent)(new.q))
        if ext > e.limit():
            raise RuntimeError(f"a bead atom is {ext:.3f} nm from its centroid reference, beyond the list margin "
                               f"{e.limit():.3f} nm; increase bead_margin")
        if not np.isfinite(float(new.upot)):
            raise FloatingPointError(f"energy is not finite at step {int(new.step)}")
        self.state = new
        self.time_ps += n * self.dt

    # ------------------------------------------------------------------ observables
    def _est(self):
        if getattr(self, "_est_jit", None) is None:
            self._est_jit = jax.jit(self.integ.estimators)
        return self._est_jit(self.state)

    def observables(self) -> dict:
        st, eng = self.state, self.state.eng
        est = {k: np.asarray(v) for k, v in self._est().items()}
        out = {"step": int(st.step), "time_ps": self.time_ps, "temp_K": float(est["t_beads"]),
               "temp_centroid": float(est["t_centroid"]), "epot": float(est["epot"]),
               "ekin_prim": float(est["prim"].sum()), "ekin_cv": float(est["cv"].sum()),
               "econs": float(est["econs"]), "elec": float(eng.elec), "vdw": float(eng.vdw)}
        V = float(volume(st.box))
        out["volume_nm3"] = V
        out["density_g_cm3"] = float(np.sum(self.sim.sys.masses)) / V * 1.66053906660e-3
        if self.ensemble == "npt":
            out["mc_accept"] = int(st.mc[1]) / max(int(st.mc[0]), 1)
        for el in sorted(set(self.elements.tolist())):
            sel = self.elements == el
            out[f"ke_{el}_cv_meV"] = float(est["cv"][sel].mean()) * KJMOL_TO_MEV
            out[f"ke_{el}_prim_meV"] = float(est["prim"][sel].mean()) * KJMOL_TO_MEV
        M, mb = self.molecular_dipoles()
        out["dipole_D"] = float(np.mean(np.linalg.norm(M, axis=-1))) / DEBYE_E_NM
        out["dipole_bead_D"] = mb / DEBYE_E_NM
        out.update({"cg_iter": int(eng.iters), "cg_iter_max": int(eng.max_iters), "cg_resid_max": float(eng.resid),
                    "cg_mean": float(eng.cg_total) / max(int(st.step), 1)})
        return out

    def pressure(self) -> float:
        """Molecular centroid-virial pressure (bar): N_mol kT / V - tr((1/P) dU/d eps) / (3 V)."""
        if getattr(self, "_w_jit", None) is None:
            self._w_jit = jax.jit(self.engine.strain_derivative)
        st = self.state
        W = self._w_jit(st.q, st.box, st.eng)
        V = float(volume(st.box))
        return (self.sim.sys.nmol * KB * self.T0 - float(jnp.trace(W)) / 3.0) / V * BAR

    def molecular_dipoles(self):
        """(bead-averaged molecular dipoles (nmol, 3), mean over beads and molecules of |mu_mol|), e nm:
        charges, covalent and induced dipoles of every force bead (the contracted beads with contraction)."""
        if getattr(self, "_dip_jit", None) is None:
            from .dipoles import CellDipole
            cd = CellDipole(self.engine.ff)
            params = self.engine.params

            def f(q, box, mu):
                x = self.engine.force_positions(q)
                M = jax.vmap(lambda xk, mk: cd.molecular(xk, box, mk, params))(x, mu)
                return jnp.mean(M, 0), jnp.mean(jnp.linalg.norm(M, axis=-1))
            self._dip_jit = jax.jit(f)
        st = self.state
        M, m = self._dip_jit(st.q, st.box, st.eng.induction.mu)
        return np.asarray(M), float(m)

    def centroid_nm(self) -> np.ndarray:
        return np.asarray(jnp.mean(self.state.q, 0))

    def beads_nm(self) -> np.ndarray:
        return np.asarray(self.state.q)

    # ------------------------------------------------------------------ running
    def run(self, nsteps: int, report: int = 100, traj: int = 0, beads_traj: int = 0, restart: int = 0,
            prefix: str = "pimd", append: bool = False, pressure: bool = False):
        """Every `report` steps a log line (prefix.log), `traj` a centroid frame (prefix.nc),
        `beads_traj` a frame of every bead (prefix_beads.nc, P x N atoms, bead-major), `restart` a
        checkpoint (prefix.pimd.chk) and an Amber restart of the centroid."""
        block = int(np.gcd.reduce([x for x in (report, traj, beads_traj, restart, nsteps) if x > 0]))
        n = self.sim.sys.n
        tfile = NetCDFTrajectory(prefix + ".nc", n, append=append) if traj else None
        bfile = NetCDFTrajectory(prefix + "_beads.nc", n * self.P, append=append) if beads_traj else None
        logf = open(prefix + ".log", "a" if append else "w")
        cols = None
        t0, s0 = time.time(), int(self.state.step)
        done = 0
        while done < nsteps:
            m = min(block, nsteps - done)
            self._advance(m)
            done += m
            step = int(self.state.step)
            if report and step % report == 0:
                obs = self.observables()
                if pressure:
                    obs["press_bar"] = self.pressure()
                el = time.time() - t0
                obs["ns_per_day"] = (step - s0) * self.dt / 1000.0 / max(el, 1e-9) * 86400.0
                if cols is None:
                    cols = list(obs)
                    header = "# " + " ".join(f"{c:>14s}" for c in cols)
                    if not append or logf.tell() == 0:
                        logf.write(header + "\n")
                    self._print(header)
                line = "  " + " ".join(f"{obs[c]:14.6f}" if isinstance(obs[c], float) else f"{obs[c]:14d}" for c in cols)
                logf.write(line + "\n")
                logf.flush()
                self._print(line)
            box_A = np.asarray(self.state.box) * 10.0
            if tfile is not None and step % traj == 0:
                tfile.write(self.time_ps, self.centroid_nm() * 10.0, box_A)
            if bfile is not None and step % beads_traj == 0:
                bfile.write(self.time_ps, self.beads_nm().reshape(-1, 3) * 10.0, box_A)
            if restart and step % restart == 0:
                self.save(prefix)
        logf.close()
        if restart:
            self.save(prefix)

    # ------------------------------------------------------------------ checkpoints
    def save(self, prefix: str):
        """prefix.pimd.chk (complete state: beads, momenta, forces, dipoles and predictor history of
        every bead, random state) and prefix.rst7 (Amber restart of the centroid)."""
        st = self.state
        vc = np.asarray(jnp.mean(st.p, 0) / self.integ.mass)
        write_restart(prefix + ".rst7", self.centroid_nm() * 10.0, vc * 10.0, np.asarray(st.box) * 10.0, self.time_ps,
                      title=f"pgm_jax PIMD centroid, {self.P} beads")
        host = jax.tree_util.tree_map(np.asarray, st.set(eng=st.eng.set(nbr=None)))
        with open(prefix + ".pimd.chk", "wb") as fh:
            pickle.dump({"format": FORMAT, "beads": self.P, "state": host, "time_ps": self.time_ps}, fh)

    def load(self, path: str):
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        if d.get("format") != FORMAT or int(d["beads"]) != self.P:
            raise ValueError(f"{path}: not a {self.P}-bead {FORMAT!r} checkpoint")
        st = jax.tree_util.tree_map(jnp.asarray, d["state"])
        if st.mc is None:
            st = st.set(mc=jnp.zeros(4, jnp.int32), mc_dv=jnp.asarray(0.01 * float(volume(st.box)), jnp.float64))
        self.engine.make_neighbors(np.asarray(st.box))
        nbr = self._size(st.q, st.box)
        self.state = st.set(eng=st.eng.set(nbr=nbr))
        self.time_ps = float(d["time_ps"])


# ----------------------------------------------------------------------------- flexible pGM water
KCAL = 4.184
QTIP4PF = {"D": 116.09 * KCAL, "alpha": 22.87, "r_eq": 0.09419, "k_theta": 87.85 * KCAL, "theta_eq": 107.4}
"""q-TIP4P/F intramolecular potential (Habershon, Markland & Manolopoulos, JCP 131, 024501 (2009)):
quartic Morse O-H bonds D [a^2 dr^2 - a^3 dr^3 + 7/12 a^4 dr^4] (D kJ/mol, alpha 1/nm, r_eq nm) and a
harmonic bend k_theta (theta - theta_eq)^2 / 2 (kJ/mol/rad^2, degrees)."""

WATER_FAMILIES = ("bond_quartic", "angle_harm", "angle_cubic", "bond_bond", "bond_angle")


def qtip4pf_intra(R, p: dict = QTIP4PF):
    """q-TIP4P/F intramolecular energy (kJ/mol) of one water R (3, 3) nm, atoms O, H, H."""
    u, v = R[1] - R[0], R[2] - R[0]
    r1, r2 = jnp.linalg.norm(u), jnp.linalg.norm(v)
    th = jnp.arccos(jnp.clip(jnp.dot(u, v) / (r1 * r2), -1.0, 1.0))
    a = p["alpha"]

    def morse4(r):
        x = a * (r - p["r_eq"])
        return p["D"] * (x * x - x ** 3 + 7.0 / 12.0 * x ** 4)
    return morse4(r1) + morse4(r2) + 0.5 * p["k_theta"] * (th - math.radians(p["theta_eq"])) ** 2


def water_geometry(r1, r2, theta_deg):
    """O, H, H (3, 3) nm with O at the origin, in the xy plane."""
    t = math.radians(theta_deg)
    return np.array([[0.0, 0.0, 0.0], [r1, 0.0, 0.0], [r2 * math.cos(t), r2 * math.sin(t), 0.0]])


def harmonic_frequencies(energy_fn, R, masses):
    """Harmonic vibrational wavenumbers (cm^-1) of energy_fn (kJ/mol) at R (n, 3) nm, masses amu:
    the 3n - 6 largest of the mass-weighted Hessian."""
    R = jnp.asarray(R, jnp.float64)
    n = R.shape[0]
    Hs = np.asarray(jax.hessian(lambda x: energy_fn(x.reshape(n, 3)))(R.reshape(-1)))
    m = np.repeat(np.asarray(masses, float), 3)
    w2 = np.linalg.eigvalsh(Hs / np.sqrt(np.outer(m, m)))
    w = np.sqrt(np.clip(np.sort(w2)[-(3 * n - 6):], 0.0, None))       # rad/ps
    return w / (2.0 * math.pi * 2.99792458e-2)                           # 1/ps -> cm^-1


def flexible_water(molecule, target=qtip4pf_intra, families=WATER_FAMILIES, n_samples: int = 2000,
                   sigma_r: float = 0.008, sigma_theta: float = 9.0, force_weight: float = 1e-4, seed: int = 0):
    """FlexibleTemplate of a pGM water whose gas-phase monomer potential (bonded terms + the
    all-pair intramolecular pGM electrostatics and induction of `molecule`) reproduces `target`
    (default: the q-TIP4P/F intramolecular potential).  The bonded families (quartic bonds, harmonic
    and cubic bend, bond-bond and bond-angle couplings; pgm_jax.bonded) are fitted
    to target energies and forces on geometries drawn around the target minimum (bonds +-sigma_r nm,
    angle +-sigma_theta deg, Gaussian): the force constants, on which the energy is linear, by linear
    least squares, inside a Nelder-Mead search over the reference values (b0, theta0; r0 of a
    Urey-Bradley term).  Returns (template, report) with the RMS errors and the harmonic
    frequencies of the fitted and target monomers."""
    import scipy.optimize
    from ..bonded import terms as TT
    from ..bonded.model import BondedModel, BondedSettings, MolSpec
    from .flexible import FlexibleTemplate

    if list(molecule.elements) != ["O", "H", "H"]:
        raise ValueError("atoms must be O, H, H")
    p = QTIP4PF
    x0 = water_geometry(p["r_eq"], p["r_eq"], p["theta_eq"])
    spec = MolSpec(molecule.name, ["O", "H", "H"], [(0, 1), (0, 2)], [1, 1], 0, x0, molecule)
    model = BondedModel([spec], BondedSettings(families=tuple(families)))
    P0 = jax.tree_util.tree_map(np.asarray, model.init_params())
    lin = [(f, k) for f in model.fams for k in TT.REGISTRY[f].linear]
    sizes = [int(np.size(P0[f][k])) for f, k in lin]
    nonlin = [("ref", "b0"), ("ref", "th0")] + [(f, "r0") for f in model.fams if "r0" in P0[f]]
    for f in model.fams:
        extra = [k for k in P0[f] if k not in TT.REGISTRY[f].linear and k != "r0"]
        if extra:
            raise ValueError(f"family {f}: nonlinear parameters {extra} are not fitted here")

    def build(theta, nl):
        P = {f: dict(v) for f, v in P0.items()}
        o = 0
        for (f, k), n in zip(lin, sizes):
            P[f][k] = jnp.reshape(theta[o:o + n], np.shape(P0[f][k])); o += n
        for (f, k), v in zip(nonlin, nl):
            P[f][k] = jnp.full(np.shape(P0[f][k]), v)
        return P

    rng = np.random.default_rng(seed)
    r = p["r_eq"] + sigma_r * rng.standard_normal((n_samples, 2))
    th = p["theta_eq"] + sigma_theta * rng.standard_normal(n_samples)
    X = jnp.asarray(np.stack([water_geometry(a, b, t) for (a, b), t in zip(r, th)]))
    enb = lambda R: model.nonbonded(0, R, None)[0]                                        # noqa: E731
    yE = np.asarray(jax.vmap(target)(X) - jax.vmap(enb)(X))
    yF = np.asarray(-jax.vmap(jax.grad(target))(X) + jax.vmap(jax.grad(enb))(X))
    L = sum(sizes)
    wF = math.sqrt(force_weight)

    @jax.jit
    def design(nl):
        def eb(theta):
            P = build(theta, nl)
            E = jax.vmap(lambda R: model.bonded_energy(0, R, P))(X)
            F = -jax.vmap(jax.grad(lambda R: model.bonded_energy(0, R, P)))(X)
            return E, F
        return jax.jacfwd(eb)(jnp.zeros(L))                  # energies are linear in theta: exact

    def solve(nl):
        JE, JF = (np.asarray(a) for a in design(jnp.asarray(nl)))
        A = np.concatenate([(JE - JE.mean(0)) / math.sqrt(len(yE)), wF * JF.reshape(-1, L) / math.sqrt(len(yE))])
        b = np.concatenate([(yE - yE.mean()) / math.sqrt(len(yE)), wF * yF.reshape(-1) / math.sqrt(len(yE))])
        scale = np.maximum(np.linalg.norm(A, axis=0), 1e-30)
        theta = np.linalg.lstsq(A / scale, b, rcond=None)[0] / scale
        return theta, float(np.sum((A @ theta - b) ** 2))

    t0 = [p["r_eq"], math.radians(p["theta_eq"])] + [float(np.mean(P0[f]["r0"])) for f, _ in nonlin[2:]]
    steps = np.array([0.002, 0.05] + [0.005] * (len(t0) - 2))
    opt = scipy.optimize.minimize(lambda nl: solve(nl)[1], np.array(t0), method="Nelder-Mead",
                                  options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 4000,
                                           "initial_simplex": np.vstack([t0, np.array(t0) + np.diag(steps)])})
    theta, loss = solve(opt.x)
    P = jax.tree_util.tree_map(jnp.asarray, build(jnp.asarray(theta), opt.x))
    tpl = FlexibleTemplate.from_fit(model, P)
    model_E = lambda R: model.energy(0, R, P)[0]                                          # noqa: E731
    E = jax.vmap(model_E)(X)
    Et = jax.vmap(target)(X)
    dE = (E - Et) - jnp.mean(E - Et)
    dF = -jax.vmap(jax.grad(model_E))(X) + jax.vmap(jax.grad(target))(X)
    m = np.asarray(molecule.masses, float)
    e_fit = jax.jit(model_E)
    g_fit = jax.jit(jax.grad(lambda x: model_E(x.reshape(3, 3))))
    mn = scipy.optimize.minimize(lambda x: float(e_fit(jnp.asarray(x).reshape(3, 3))), x0.reshape(-1),
                                 jac=lambda x: np.asarray(g_fit(jnp.asarray(x)), float), method="BFGS",
                                 options={"gtol": 1e-8})
    Rm = mn.x.reshape(3, 3)
    b = np.linalg.norm(Rm[1:] - Rm[0], axis=1)
    ang = math.degrees(math.acos(np.dot(Rm[1] - Rm[0], Rm[2] - Rm[0]) / (b[0] * b[1])))
    report = {"rms_energy_kJmol": float(jnp.sqrt(jnp.mean(dE ** 2))),
              "rms_force_kJmol_nm": float(jnp.sqrt(jnp.mean(jnp.sum(dF ** 2, -1)) / 3.0)),
              "rms_target_force": float(jnp.sqrt(jnp.mean(jnp.sum(jax.vmap(jax.grad(target))(X) ** 2, -1)) / 3.0)),
              "minimum_bonds_nm": b.tolist(), "minimum_angle_deg": ang,
              "freq_fit_cm": harmonic_frequencies(model_E, Rm, m).tolist(),
              "freq_target_cm": harmonic_frequencies(target, x0, m).tolist(),
              "params": {f: {k: np.asarray(v).tolist() for k, v in d.items()} for f, d in P.items()},
              "outer": str(opt.message)}
    return tpl, report
