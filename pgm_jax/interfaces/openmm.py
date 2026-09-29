"""OpenMM with pGM forces: an openmm.PythonForce (OpenMM >= 8.4) that calls the device-resident
pgm_jax engine, so OpenMM's integrators, constraints (SETTLE / SHAKE), barostats and reporters run
with pGM electrostatics.

    from pgm_jax.interfaces.openmm import PGMOpenMM
    om = PGMOpenMM(engine)                                   # PGMEngine (pgm_jax.interfaces.engine)
    system = om.system(rigid=True)                           # masses, box, constraints, the pGM force
    sim = openmm.app.Simulation(om.topology(), system, openmm.LangevinMiddleIntegrator(298*kelvin,
                                1/picosecond, 0.002*picoseconds), openmm.Platform.getPlatformByName("CPU"))
    sim.context.setPositions(pos_nm); sim.context.setPeriodicBoxVectors(*H_nm)
    sim.step(1000)

Why PythonForce: it is the only route that needs nothing beyond OpenMM itself.  Per energy / force
evaluation OpenMM hands the Python function a State (positions and box), the engine returns the
energy and forces (numpy); the jitted pGM call stays on the JAX device (CPU or GPU).  OpenMM's own
platform can be CPU or CUDA; with CUDA the positions and forces cross the host once per step in
each direction, as with any external force.  Alternatives considered (docs/interfaces.md): an
openmm-torch TorchForce around a jax2torch / DLPack bridge (keeps the arrays on the GPU but needs
openmm-torch and a TorchScript-compatible wrapper of a JAX function, which TorchScript cannot
trace), and a CustomExternalForce updated step by step (setParameters per atom per step: far slower).

Energies and forces are exact (the engine's); the MonteCarloBarostat works (it re-evaluates the
energy at the scaled box and molecular centres; the engine rebuilds its lists when the volume
changes by more than 10 %).  OpenMM's PythonForce gives no virial, so anisotropic or
pressure-reporting tools that need one are not available.  Units: nm, ps, kJ/mol (OpenMM's)."""

from __future__ import annotations

import time

import numpy as np

try:
    import openmm
    from openmm import app, unit
except ImportError as err:  # pragma: no cover
    raise ImportError("pgm_jax.interfaces.openmm needs OpenMM >= 8.4 (openmm.PythonForce)") from err


def _element(symbol: str):
    try:
        return app.Element.getBySymbol(symbol)
    except KeyError:
        return None


class PGMOpenMM:
    """OpenMM System / Topology / PythonForce for a PGMEngine (the engine's system and model)."""

    def __init__(self, engine):
        if not hasattr(openmm, "PythonForce"):
            raise ImportError(f"OpenMM {openmm.__version__} has no PythonForce (needs >= 8.4)")
        self.engine = engine
        self.sys = engine.sys
        self.stats = {"calls": 0, "t_call": 0.0, "t_convert": 0.0}

    # ------------------------------------------------------------------ the force
    def _compute(self, state):
        t0 = time.perf_counter()
        pos = state.getPositions(asNumpy=True)._value  # nm (OpenMM default units)
        box = state.getPeriodicBoxVectors(asNumpy=True)._value
        t1 = time.perf_counter()
        res = self.engine.compute(np.asarray(pos, float), np.asarray(box, float))
        t2 = time.perf_counter()
        self.stats["calls"] += 1
        self.stats["t_call"] += t2 - t0
        self.stats["t_convert"] += t1 - t0
        return res.energy, res.forces

    def force(self, group: int = 0):
        """The pGM force (energy and forces of the engine's model, periodic)."""
        f = openmm.PythonForce(self._compute)
        f.setUsesPeriodicBoundaryConditions(True)
        f.setForceGroup(int(group))
        return f

    # ------------------------------------------------------------------ system and topology
    def system(self, rigid: bool = True, constraints: str | None = None, hmr: float | None = None, cmm: bool = True):
        """openmm.System with the engine's masses, box and the pGM force.
        rigid: hold every molecule of up to three atoms rigid by distance constraints (the
        rigid-molecule model; OpenMM applies SETTLE to water); constraints="h-bonds": also X-H
        bonds of larger molecules at their current lengths.  cmm: remove centre-of-mass motion."""
        s = openmm.System()
        for m in np.asarray(self.sys.masses, float):
            s.addParticle(float(m))
        H = self.box()
        s.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*row) for row in H])
        x0 = self.positions()
        for i, j in self.constraint_pairs(rigid, constraints):
            s.addConstraint(int(i), int(j), float(np.linalg.norm(x0[i] - x0[j])))
        s.addForce(self.force())
        if cmm:
            s.addForce(openmm.CMMotionRemover())
        return s

    def constraint_pairs(self, rigid: bool = True, constraints: str | None = None):
        pairs = []
        for k, m in enumerate(self.sys.molecules):
            off = int(self.sys.offsets[k])
            if rigid and m.n <= 3:
                pairs += [(off + i, off + j) for i in range(m.n) for j in range(i + 1, m.n)]
            elif constraints == "h-bonds":
                el = list(m.elements)
                pairs += [(off + i, off + j) for i, j in m.bonds if "H" in (el[i], el[j])]
            elif rigid and m.n > 3 and self.engine.flex is None:
                raise ValueError(
                    f"molecule {k} ({m.name}, {m.n} atoms) of the rigid-molecule model cannot be held rigid "
                    "by distance constraints; use flexible templates"
                )
        return pairs

    def topology(self):
        """openmm.app.Topology (one residue per molecule), for reporters (PDB, DCD)."""
        top = app.Topology()
        chain = top.addChain()
        for m in self.sys.molecules:
            res = top.addResidue(m.name[:4] or "MOL", chain)
            atoms = [top.addAtom(t or e, _element(e), res) for e, t in zip(m.elements, getattr(m, "types", m.elements))]
            for i, j in getattr(m, "bonds", []) or []:
                top.addBond(atoms[int(i)], atoms[int(j)])
        top.setPeriodicBoxVectors([openmm.Vec3(*row) for row in self.box()] * unit.nanometer)
        return top

    def box(self) -> np.ndarray:
        """The engine's initial box (nm, reduced lower-triangular: OpenMM's form)."""
        return np.asarray(self.engine.initial_box, float)

    def positions(self) -> np.ndarray:
        """The engine's initial positions (nm, molecules whole, in the frame of box())."""
        return np.asarray(self.engine.initial_positions, float)
