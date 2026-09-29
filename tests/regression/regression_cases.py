"""Regression cases: single points and short trajectories whose outputs are recorded on master
(golden files) and compared after every refactoring step.

Every case returns a flat dict {name: numpy array}.  All cases run on the CPU in float64 (one
mixed-precision case), with fixed seeds, so the outputs are deterministic for a given JAX / XLA
build and thread count.

API policy: the only places that know the engine API are the helpers in the "API adapter" section
and the case bodies.  When a phase of the clean-up changes a public name, update the calls here in
the same commit; the recorded numbers must not change (bitwise, or to the tolerance stated in the
phase)."""

from __future__ import annotations

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import regression_systems as S

CASES: dict = {}


def case(name: str, needs: str | None = None, group: str = "a"):
    """Register a case.  needs: "pgm3p25" (the pGM3P-25 files of ~/pgm-gvdw-data); group: a label
    for splitting the cases over several jobs."""

    def deco(fn):
        CASES[name] = {"fn": fn, "needs": needs, "group": group, "doc": (fn.__doc__ or "").strip().split("\n")[0]}
        return fn

    return deco


def available(name: str) -> bool:
    needs = CASES[name]["needs"]
    return needs is None or (needs == "pgm3p25" and S.pgm3p25_available())


# ----------------------------------------------------------------------------- helpers
def _np(x):
    return np.asarray(x)


def _put(out: dict, prefix: str, tree):
    """Flatten a dict / pytree of arrays into out[prefix.key]."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            _put(out, f"{prefix}.{k}" if prefix else str(k), v)
    elif isinstance(tree, (list, tuple)):
        for k, v in enumerate(tree):
            _put(out, f"{prefix}.{k}", v)
    elif tree is None:
        return
    else:
        a = np.asarray(tree)
        if a.dtype == object:
            return
        out[prefix] = a


# ----------------------------------------------------------------------------- API adapter
def md_settings(**kw):
    from pgm_jax.md.forcefield import MDSettings

    return MDSettings(**kw)


def tight(**kw):
    """float64, tight induction, small cutoffs for the small boxes."""
    base = dict(precision="double", dipole_tol=1e-10, max_iter=300, cutoff=0.55, skin=0.05)
    base.update(kw)
    return md_settings(**base)


def rigid_sim(sys, pos, H, settings, **kw):
    from pgm_jax.md.simulation import Simulation

    kw.setdefault("log", None)
    return Simulation(sys, pos, H, settings, **kw)


def flexible_sim(sys, templates, pos, H, settings, **kw):
    from pgm_jax.md.flexible import FlexibleSimulation

    kw.setdefault("log", None)
    return FlexibleSimulation(sys, templates, pos, H, settings, **kw)


def rigid_template(mol, xyz):
    from pgm_jax.md.flexible import RigidTemplate

    return RigidTemplate(mol, xyz)


def advance(sim, n: int):
    """n steps without files (the driver's block loop: overflow checks, re-wrapping)."""
    sim._advance(n)


def snapshot(sim, out: dict, tag: str, mu: bool = True):
    """Observables, positions, velocities, box (and induced dipoles) of an engine's current state."""
    for k, v in sim.observables().items():
        if isinstance(v, (bool, int, float, np.integer, np.floating)):
            out[f"{tag}.obs.{k}"] = np.asarray(v)
    out[f"{tag}.pos"] = _np(sim.positions_nm())
    out[f"{tag}.vel"] = _np(sim.velocities_nm_ps())
    out[f"{tag}.box"] = _np(sim.state.box)
    if mu:
        out[f"{tag}.mu"] = _np(sim.state.induction.mu)


def trajectory(sim, out: dict, nsteps: int = 200, every: int = 50, tag: str = "t"):
    snapshot(sim, out, f"{tag}{0:04d}")
    for s in range(every, nsteps + 1, every):
        advance(sim, every)
        snapshot(sim, out, f"{tag}{s:04d}")
    return out


def water_system(n_side=4, spacing=0.31, seed=0, pgm3p25: bool = False):
    from pgm_jax.system import System

    if pgm3p25:
        w, geo = S.pgm3p25_water()
        pos, H, geo = S.water_lattice(n_side, spacing, seed, geometry=geo - geo.mean(0))
    else:
        w = S.water()
        pos, H, geo = S.water_lattice(n_side, spacing, seed)
    return System([w] * (len(pos) // 3)), pos, H, w, geo


def force_field_point(ff, pos, H, params=None, efield=None, out=None, tag="sp"):
    """Energy terms, forces, induced dipoles, CG iterations and strain derivative of one frame."""
    out = {} if out is None else out
    idx = ff.rows_for(pos, H)
    kw = {} if efield is None else {"efield": efield}
    res = jax.jit(lambda x, h, i, p: ff.compute(x, h, i, ff.init_induction(), p, **kw))(
        jnp.asarray(pos), jnp.asarray(H), idx, params
    )
    _put(out, f"{tag}.energy", res.energy)
    out[f"{tag}.forces"] = _np(res.forces)
    out[f"{tag}.mu"] = _np(res.induction.mu)
    out[f"{tag}.iterations"] = _np(res.iterations)
    out[f"{tag}.residual"] = _np(res.residual)
    if res.dipole is not None:
        out[f"{tag}.dipole"] = _np(res.dipole)
    W = ff.strain_derivative(jnp.asarray(pos), jnp.asarray(H), idx, res.induction.mu, params, **kw)
    out[f"{tag}.strain"] = _np(W)
    return out


# ============================================================================= single points
@case("gas_model")
def gas_model():
    """Gas phase Model: every electrostatics level, LJ and GVDW, forces, induced dipoles, parameter
    gradient, molecular polarizability, n-body decomposition, external field."""
    from pgm_jax import ElecChannel, LJChannel, Model, System, molecular_polarizability
    from pgm_jax.vdw import PGM3P_GVDW, GVDWChannel, set_gvdw

    sys, pos = S.cluster(0)
    x = jnp.asarray(pos)
    P = sys.table.initial()
    out = {}
    for lvl in ("q", "qp", "qi", "qpi"):
        _put(out, f"E_{lvl}", Model([ElecChannel.level(lvl), LJChannel()]).energy_fn(sys)(x, P))
    m = Model([ElecChannel(), LJChannel()])
    out["forces"] = _np(m.forces_fn(sys)(x, P))
    _, aux = ElecChannel().energy(x, sys, P)
    out["mu"], out["p"] = _np(aux["mu"]), _np(aux["p"])
    _put(out, "dE_dP", jax.grad(lambda p: m.energy_fn(sys)(x, p)["total"])(P))
    out["alpha_mol"] = _np(molecular_polarizability(x, sys, P))
    _put(out, "nbody", m.nbody(sys, pos[None], P))
    _put(out, "E_field", Model([ElecChannel(efield=(0.3, -0.2, 0.5)), LJChannel()]).energy_fn(sys)(x, P))
    g = PGM3P_GVDW["gauss"]
    sg = System([set_gvdw(mol, {"OW": g["OW"]}) for mol in sys.molecules])
    _put(out, "E_gvdw", Model([ElecChannel(), GVDWChannel(rep=g["rep"])]).energy_fn(sg)(x, sg.table.initial()))
    return out


@case("periodic_model")
def periodic_model():
    """PeriodicModel (Ewald) in a skewed triclinic box: energies, forces, strain derivative,
    pressure, induced dipoles; GVDW variant."""
    from pgm_jax import PeriodicModel

    sys, pos, H = S.small_box(1)
    P = sys.table.initial()
    out = {}
    box = PeriodicModel(sys, H, pos, rc=0.6, lj_lrc=True)
    x = jnp.asarray(pos)
    _put(out, "E", box.energy(x, P, H))
    out["forces"] = _np(box.forces(x, P, H))
    out["strain_mol"] = _np(box.strain_derivative(x, P, H))
    out["strain_atom"] = _np(box.strain_derivative(x, P, H, molecular=False))
    out["pressure"] = _np(box.pressure(x, P))
    out["mu"] = _np(box.elec.induced_dipoles(x, P))
    _put(out, "E_qp_none", PeriodicModel(sys, H, pos, rc=0.6, elec="qp", vdw="none").energy(x, P, H))
    return out


@case("ff_small_box")
def ff_small_box():
    """MD force field (PME, rows, CG) on the triclinic water/methanol box, float64 and mixed; the
    differentiable path (parameter gradient of a force/dipole loss)."""
    from pgm_jax.md.forcefield import PGMForceField

    sys, pos, H = S.small_box(0)
    out = {}
    base = dict(
        cutoff=0.6,
        skin=0.05,
        ewald_beta=6.0,
        pme_grid=(32, 32, 32),
        pme_order=6,
        lj_lrc=True,
        dipole_tol=1e-10,
        max_iter=500,
    )
    for prec in ("double", "mixed"):
        ff = PGMForceField(sys, H, md_settings(precision=prec, **base))
        force_field_point(ff, pos, H, out=out, tag=prec)
    ff = PGMForceField(sys, H, md_settings(precision="double", differentiable=True, **base))
    idx = ff.rows_for(pos, H)
    x, h = jnp.asarray(pos), jnp.asarray(H)

    def loss(theta):
        r = ff.compute(x, h, idx, ff.init_induction(), theta)
        return jnp.sum(r.forces**2) * 1e-6 + jnp.sum(r.induction.mu**2) * 1e2

    _put(out, "dloss", jax.jit(jax.grad(loss))(sys.params0))
    return out


@case("ff_pgm3p25_512", needs="pgm3p25", group="b")
def ff_pgm3p25_512():
    """pGM3P-25, 512 waters (the validation box): default MDSettings (mixed) and float64."""
    from pgm_jax.md.box import box_from_cell
    from pgm_jax.md.forcefield import PGMForceField
    from pgm_jax.md.io import read_coordinates
    from pgm_jax.param import read_prmtop_pgm
    from pgm_jax.system import System

    mols = read_prmtop_pgm(S.PGM3P25_TOP, first_residue_only=False)
    sys = System([mols[0]] * len(mols))
    xyz, _, box = read_coordinates(S.PGM3P25_RST)
    H, pos = box_from_cell(*box) * 0.1, xyz * 0.1
    out = {}
    for prec in ("mixed", "double"):
        ff = PGMForceField(sys, H, md_settings(precision=prec))
        force_field_point(ff, pos, H, out=out, tag=prec)
    return out


@case("efield_point")
def efield_point():
    """External fields in the MD force field: static field and constant displacement."""
    from pgm_jax.md.forcefield import PGMForceField

    sys, pos, H = S.small_box(13, nm=0)
    s = md_settings(
        cutoff=0.6,
        skin=0.05,
        ewald_beta=6.0,
        pme_grid=(32, 32, 32),
        pme_order=6,
        lj_lrc=False,
        dipole_tol=1e-10,
        max_iter=500,
        precision="double",
    )
    ff = PGMForceField(sys, H, s)
    out = {}
    force_field_point(ff, pos, H, efield=(jnp.asarray([0.1, -0.3, 0.8]), None), out=out, tag="E")
    return out


@case("iel_point")
def iel_point():
    """iEL/0-SCF shadow energy and forces (block preconditioner) at a frame."""
    from pgm_jax.md.forcefield import PGMForceField

    sys, pos, H = S.small_box(1)
    s = md_settings(
        cutoff=0.6,
        skin=0.05,
        pme_grid=(48, 48, 48),
        pme_order=8,
        ewald_beta=6.0,
        precision="double",
        dipole_tol=1e-10,
        max_iter=200,
        iel="0scf",
        iel_precond="block",
    )
    return force_field_point(PGMForceField(sys, H, s), pos, H)


# ============================================================================= rigid engine
def _rigid_water(ensemble, thermostat="langevin", dt=0.001, seed=3, settings=None, **kw):
    sys, pos, H, w, _ = water_system(pgm3p25=True)
    return rigid_sim(
        sys,
        pos,
        H,
        settings or tight(),
        dt=dt,
        ensemble=ensemble,
        thermostat=thermostat,
        temperature=298.0,
        seed=seed,
        **kw,
    )


@case("md_rigid_nve", needs="pgm3p25")
def md_rigid_nve():
    """Rigid pGM3P-25 water (64, rigid bodies), NVE 200 steps of 1 fs."""
    return trajectory(_rigid_water("nve"), {})


@case("md_rigid_langevin", needs="pgm3p25")
def md_rigid_langevin():
    """Rigid pGM3P-25 water, NVT Langevin (gamma 5/ps), 200 steps."""
    return trajectory(_rigid_water("nvt", "langevin", gamma=5.0), {})


@case("md_rigid_bussi", needs="pgm3p25")
def md_rigid_bussi():
    """Rigid pGM3P-25 water, NVT Bussi (tau 0.1 ps), 200 steps."""
    return trajectory(_rigid_water("nvt", "bussi", tau_t=0.1), {})


@case("md_rigid_gle", needs="pgm3p25", group="b")
def md_rigid_gle():
    """Rigid pGM3P-25 water, NVT smooth GLE, 200 steps."""
    return trajectory(_rigid_water("nvt", "gle"), {})


@case("md_rigid_npt", needs="pgm3p25", group="b")
def md_rigid_npt():
    """Rigid pGM3P-25 water, NPT (Bussi + Monte Carlo barostat every 10 steps), 200 steps; pressure."""
    sim = _rigid_water("npt", "bussi", tau_t=0.1, barostat_interval=10, pressure=1.0)
    out = trajectory(sim, {})
    out["pressure"] = np.asarray(sim.pressure())
    return out


@case("md_rigid_mixed", needs="pgm3p25", group="b")
def md_rigid_mixed():
    """Rigid pGM3P-25 water in mixed precision with production defaults (tol 1e-5, mu4 predictor)."""
    s = md_settings(cutoff=0.55, skin=0.05)
    return trajectory(_rigid_water("nvt", "bussi", settings=s, dt=0.002), {})


@case("md_rigid_run_files", needs="pgm3p25", group="c")
def md_rigid_run_files():
    """Simulation.run with every output (log, NetCDF trajectory, restart + checkpoint, cell dipoles,
    induced dipoles); a second simulation continued from the checkpoint."""
    from pgm_jax.md.dipoles import read_dipoles
    from pgm_jax.md.io import read_coordinates, read_trajectory

    out = {}
    with tempfile.TemporaryDirectory() as d:
        prefix = os.path.join(d, "md")
        sim = _rigid_water("nvt", "bussi", tau_t=0.1)
        sim.run(100, report=20, traj=20, restart=50, prefix=prefix, dipoles=10, induced=50, pressure_every_report=True)
        rows = [line.split() for line in open(prefix + ".log") if not line.startswith("#")]
        cols = [line.split()[1:] for line in open(prefix + ".log") if line.startswith("#")][0]
        tab = np.array(rows, float)
        for k, c in enumerate(cols):
            if c != "ns_per_day":
                out[f"log.{c}"] = tab[:, k]
        X, box, t = read_trajectory(prefix + ".nc")
        out["nc.xyz"], out["nc.box"], out["nc.time"] = _np(X), _np(box), _np(t)
        xyz, vel, cell = read_coordinates(prefix + ".rst7")
        out["rst7.xyz"], out["rst7.vel"], out["rst7.cell"] = _np(xyz), _np(vel), _np(cell)
        meta, dip = read_dipoles(prefix + ".dip")
        _put(out, "dip", {k: v for k, v in dip.items() if np.asarray(v).dtype != object})
        # continue: 50 steps from the 100-step checkpoint in a new simulation = 50 more steps here
        advance(sim, 50)
        snapshot(sim, out, "cont_a")
        sim2 = _rigid_water("nvt", "bussi", tau_t=0.1)
        sim2.load(prefix + ".chk")
        advance(sim2, 50)
        snapshot(sim2, out, "cont_b")
    return out


@case("md_rigid_mts", needs="pgm3p25", group="c")
def md_rigid_mts():
    """Rigid pGM3P-25 water, r-RESPA (short-range split, 2 inner steps), outer step 2 fs, Bussi."""
    from pgm_jax.md.mts import MTS

    sim = _rigid_water("nvt", "bussi", dt=0.002, tau_t=0.1, mts=MTS(inner=2, r_short=0.4, buffer=0.1))
    return trajectory(sim, {}, nsteps=100, every=25)


@case("md_iel", group="c")
def md_iel():
    """iEL/0-SCF dynamics of toy pGM water (rigid bodies), NVE 200 steps of 1 fs."""
    sys, pos, H, w, _ = water_system()
    sim = rigid_sim(sys, pos, H, tight(iel="0scf", pme_grid=(24, 24, 24)), dt=0.001, ensemble="nve", seed=3)
    return trajectory(sim, {})


@case("md_vsites", group="c")
def md_vsites():
    """TIP4P-Ew (Amber point charges, extra points as virtual sites) from a tleap prmtop, rigid
    engine (Simulation.from_amber) and constrained water with a placed site (flexible engine)."""
    from pgm_jax.md.forcefield import ewald_beta_for
    from pgm_jax.md.simulation import Simulation

    s = md_settings(
        elec="q", cutoff=0.65, skin=0.05, ewald_beta=ewald_beta_for(0.65), pme_spacing=0.06, precision="double"
    )
    prm, crd = os.path.join(S.DATA, "tip4pew_small.prmtop"), os.path.join(S.DATA, "tip4pew_small.inpcrd")
    rig = Simulation.from_amber(
        prm, crd, charges="amber", settings=s, dt=0.001, ensemble="nvt", thermostat="bussi", tau_t=0.1, seed=1, log=None
    )
    out = trajectory(rig, {}, nsteps=100, every=50, tag="rigid")
    pos = rig.positions_nm()
    tpl = rigid_template(rig.sys.molecules[0], pos[:4])
    flx = flexible_sim(
        rig.sys,
        [tpl] * rig.sys.nmol,
        pos,
        np.asarray(rig.state.box),
        s,
        dt=0.001,
        ensemble="nve",
        vel_nm_ps=rig.velocities_nm_ps(),
    )
    out["flex.force0"] = _np(flx.state.dyn.force)
    return trajectory(flx, out, nsteps=100, every=50, tag="flex")


@case("md_efield", group="c")
def md_efield():
    """Rigid engine with a static field (NVT Bussi) and at constant displacement D (NVE)."""
    from pgm_jax.md import efield as EF

    sys, pos, H = S.small_box(13, nm=0)
    s = tight(cutoff=0.6, dipole_tol=1e-8)
    out = {}
    sim = rigid_sim(sys, pos, H, s, dt=0.001, ensemble="nvt", thermostat="bussi", seed=2, efield=(0.0, 0.2, 0.6))
    trajectory(sim, out, nsteps=100, every=50, tag="E")
    sim = rigid_sim(sys, pos, H, s, dt=0.001, ensemble="nve", seed=5, efield=EF.displacement((0.0, 0.0, 2.0)))
    return trajectory(sim, out, nsteps=100, every=50, tag="D")


# ============================================================================= flexible engine
def _methanol_box(n=32, density=0.55, flux=0):
    from pgm_jax.md.flexible import liquid_box
    from pgm_jax.system import System

    tpl, _ = S.methanol_template(flux=flux)
    pos, H = liquid_box(tpl, n, density, seed=0, min_dist=0.18)
    return tpl, System([tpl.pgm] * n), pos, H


@case("flex_methanol_hbonds", group="d")
def flex_methanol_hbonds():
    """32 flexible methanols (class II bonded terms, scaled 1-4 LJ), X-H constraints, Bussi, 1 fs;
    minimize first."""
    tpl, sys, pos, H = _methanol_box()
    s = tight(cutoff=0.6, lj_lrc=False, dipole_tol=1e-10)
    sim = flexible_sim(
        sys,
        [tpl] * sys.nmol,
        pos,
        H,
        s,
        dt=0.001,
        ensemble="nvt",
        thermostat="bussi",
        tau_t=0.1,
        constraints="h-bonds",
        seed=4,
    )
    out = {}
    _put(out, "minimize", {k: v for k, v in sim.minimize(30).items()})
    return trajectory(sim, out)


@case("flex_methanol_npt", group="d")
def flex_methanol_npt():
    """32 flexible methanols, no constraints, NPT (Langevin + barostat every 10), 0.5 fs, 200 steps."""
    tpl, sys, pos, H = _methanol_box(density=0.6)
    s = tight(cutoff=0.6, lj_lrc=True, dipole_tol=1e-10)
    sim = flexible_sim(
        sys,
        [tpl] * sys.nmol,
        pos,
        H,
        s,
        dt=0.0005,
        ensemble="npt",
        thermostat="langevin",
        gamma=5.0,
        barostat_interval=10,
        seed=6,
    )
    out = trajectory(sim, {})
    out["pressure"] = np.asarray(sim.pressure())
    return out


@case("flex_water_constraints", group="d")
def flex_water_constraints():
    """Toy pGM water held rigid by SHAKE / RATTLE (flexible engine), 2 fs, GLE, 200 steps."""
    sys, pos, H, w, geo = water_system()
    sim = flexible_sim(
        sys, [rigid_template(w, geo)] * sys.nmol, pos, H, tight(), dt=0.002, ensemble="nvt", thermostat="gle", seed=1
    )
    return trajectory(sim, {})


@case("flex_flux", group="d")
def flex_flux():
    """16 flexible methanols with second-order charge flux: forces, dipoles and a 100-step NVE."""
    tpl, sys, pos, H = _methanol_box(n=16, flux=2)
    pos = pos + 0.004 * np.random.default_rng(1).normal(size=pos.shape)
    s = tight(
        cutoff=0.5,
        ewald_beta=6.0,
        pme_grid=(32, 32, 32),
        pme_order=8,
        lj_lrc=False,
        dipole_tol=1e-12,
        max_iter=500,
        peek=0.0,
    )
    sim = flexible_sim(sys, [tpl] * sys.nmol, pos, H, s, dt=0.0005, ensemble="nve", seed=2)
    out = {"force0": _np(sim.state.dyn.force)}
    return trajectory(sim, out, nsteps=100, every=50)


@case("flex_mts", group="d")
def flex_mts():
    """32 flexible methanols, MTS with the special-pair split (bonded + 1-2/1-3 pairs fast), Bussi."""
    from pgm_jax.md.mts import MTS

    tpl, sys, pos, H = _methanol_box()
    s = tight(cutoff=0.6, lj_lrc=False)
    sim = flexible_sim(
        sys,
        [tpl] * sys.nmol,
        pos,
        H,
        s,
        dt=0.001,
        ensemble="nvt",
        thermostat="bussi",
        tau_t=0.1,
        seed=3,
        mts=MTS(inner=2, split="special"),
    )
    return trajectory(sim, {}, nsteps=100, every=50)


# ============================================================================= multi-replica drivers
@case("pimd", group="e")
def pimd():
    """PIMD (PILE-L, 4 beads, Cayley) of 8 flexible waters, 100 steps of 0.2 fs; TRPMD 40 steps."""
    from pgm_jax.md.pimd import PIMDSimulation
    from pgm_jax.system import System

    tpl, pos, H = S.flexible_water_box()
    sys = System([tpl.pgm] * (len(pos) // 3))
    s = tight(dipole_tol=1e-10, cutoff=0.5, lj_lrc=False, max_iter=200)
    sim = flexible_sim(
        sys, [tpl] * sys.nmol, pos, H, s, dt=0.0002, ensemble="nvt", temperature=300.0, thermostat="bussi"
    )
    pi = PIMDSimulation(sim, beads=4, log=None, seed=1)
    out = {}
    for s_ in range(4):
        pi._advance(25)
        _put(out, f"t{s_}.obs", {k: v for k, v in pi.observables().items()})
        out[f"t{s_}.q"], out[f"t{s_}.p"] = _np(pi.state.q), _np(pi.state.p)
    out["pressure"] = np.asarray(pi.pressure())
    pi.set_mode("trpmd")
    pi._advance(40)
    _put(out, "trpmd.obs", pi.observables())
    out["trpmd.q"] = _np(pi.state.q)
    return out


@case("remd_batched", group="e")
def remd_batched():
    """Temperature REMD, 3 batched replicas of constrained toy water (GLE), 4 x (10 steps + exchange)."""
    from pgm_jax.md.remd import ReplicaExchange

    sys, pos, H, w, geo = water_system()
    sim = flexible_sim(
        sys,
        [rigid_template(w, geo)] * sys.nmol,
        pos,
        H,
        tight(dipole_tol=1e-9),
        dt=0.002,
        temperature=300.0,
        thermostat="gle",
        seed=1,
    )
    rex = ReplicaExchange(sim, np.array([300.0, 304.0, 308.0]), exchange_every=10, batched=True, seed=5, log=None)
    out = {}
    for s_ in range(4):
        rex.replicas.advance(10)
        pairs, acc = rex.exchange()
        out[f"x{s_}.replica"] = _np(rex.stats.replica).copy()
        out[f"x{s_}.acc"] = np.asarray(acc)
        out[f"x{s_}.U"] = _np(rex.replicas.potentials())
    for k in range(3):
        st = rex.replicas.state(k)
        out[f"state{k}.pos"], out[f"state{k}.mom"] = _np(st.dyn.position), _np(st.dyn.momentum)
        out[f"state{k}.heat"] = _np(st.heat)
    return out


def _cluster_bias(pace, height):
    from pgm_jax.bias import MetaD, cv

    d, phi = cv.Distance(0, 9), cv.Dihedral(1, 0, 9, 10)
    return MetaD([d, phi], sigma=[0.03, 0.4], height=height, pace=pace, biasfactor=5.0, temperature=300.0), d


@case("bias_metad", group="e")
def bias_metad():
    """Well-tempered metadynamics (distance + dihedral, hills every 10 steps) + harmonic restraint on
    8 rigid waters, NVE 0.5 fs, 200 steps; COLVAR rows."""
    from pgm_jax.bias import BiasSet, Harmonic
    from pgm_jax.system import System

    pos, H, w = S.water_cluster_box()
    sys = System([S.water()] * (len(pos) // 3))
    s = md_settings(precision="double", dipole_tol=1e-10, cutoff=1.2, skin=0.1, lj_lrc=False)
    m, d = _cluster_bias(10, 1.0)
    sim = rigid_sim(sys, pos, H, s, dt=0.0005, ensemble="nve", bias=BiasSet([m, Harmonic(d, 0.30, 2000.0)], colvar=5))
    out = trajectory(sim, {})
    out["hills.centers"] = (
        _np(sim.state.bias.parts[0].centers) if hasattr(sim.state.bias.parts[0], "centers") else np.zeros(0)
    )
    out["hills.heights"] = _np(sim.state.bias.parts[0].heights)
    out["colvar"] = _np(sim.bias_rows())
    out["bias_energies"] = _np(sim.bias_energies())
    return out


@case("bias_walkers", group="e")
def bias_walkers():
    """Three metadynamics walkers sharing one bias (vmapped), NVT Bussi, 100 steps."""
    from pgm_jax.bias import BiasSet
    from pgm_jax.bias.walkers import Walkers
    from pgm_jax.system import System

    pos, H, w = S.water_cluster_box()
    sys = System([S.water()] * (len(pos) // 3))
    s = md_settings(precision="double", dipole_tol=1e-10, cutoff=1.2, skin=0.1, lj_lrc=False)
    m, _ = _cluster_bias(10, 1.0)
    sim = rigid_sim(
        sys, pos, H, s, dt=0.001, ensemble="nvt", thermostat="bussi", temperature=300.0, bias=BiasSet([m], colvar=5)
    )
    wk = Walkers(sim, 3, shared=True, seed=4)
    wk.advance(100)
    out = {"heights": _np(wk.S.bias.parts[0].heights), "n": _np(wk.S.bias.parts[0].n)}
    for k in range(3):
        st = wk.state(k)
        out[f"w{k}.center"] = _np(st.dyn.position.center)
        out[f"w{k}.epot"] = _np(st.epot)
    return out


@case("field_replicas", group="e")
def field_replicas():
    """FieldReplicas (+E, -E, 0) of the small water box, vmapped, 40 steps, the .ffd series."""
    from pgm_jax.md.finite_field import FieldReplicas, read_series

    sys, pos, H = S.small_box(13, nm=0)
    sim = rigid_sim(
        sys,
        pos,
        H,
        tight(cutoff=0.6, dipole_tol=1e-6),
        dt=0.001,
        ensemble="nvt",
        thermostat="bussi",
        efield=(0.0, 0.0, 0.0),
    )
    rep = FieldReplicas(sim, [(0.0, 0.0, 0.5), (0.0, 0.0, -0.5), (0.0, 0.0, 0.0)], seed=1)
    out = {}
    with tempfile.TemporaryDirectory() as d:
        rep.run(40, every=10, prefix=os.path.join(d, "ff"), report=20, log=None)
        meta, data = read_series(os.path.join(d, "ff.ffd"))
    _put(out, "ffd", {k: v for k, v in data.items()})
    for k in range(3):
        out[f"r{k}.center"] = _np(rep.state(k).dyn.position.center)
    return out


# ============================================================================= free energies
def _alch_windows(batched=True, seed=1, dipole_tol=1e-9):
    from pgm_jax.md.alchemy import Alchemy, LambdaWindows, alchemical_system, standard_schedule

    sys0, pos, H, w, _ = water_system()
    sysA, P = alchemical_system(sys0, 0)
    alch = Alchemy(sysA, 0)
    s = tight(dipole_tol=dipole_tol, max_iter=400, ewald_beta=5.0, pme_grid=(32, 32, 32), peek=0.0)
    sim = rigid_sim(sysA, pos, H, s, dt=0.001, params=P, alchemy=alch, thermostat="bussi", ensemble="nvt")
    return LambdaWindows(sim, standard_schedule(3, [0.5, 0.0]), batched=batched, seed=seed), P, sim, alch


@case("alchemy_point", group="f")
def alchemy_point():
    """Alchemical Hamiltonian at lambda = (0.6, 0.8): energy, forces, pressure, dU/dlambda."""
    from pgm_jax.md.alchemy import Alchemy, alchemical_system

    sys0, pos, H, w, _ = water_system()
    sysA, P = alchemical_system(sys0, 0)
    alch = Alchemy(sysA, 0, lam=(0.6, 0.8))
    s = tight(dipole_tol=1e-11, max_iter=400, ewald_beta=5.0, pme_grid=(32, 32, 32), peek=0.0)
    sim = rigid_sim(sysA, pos, H, s, dt=0.001, params=P, alchemy=alch)
    out = {
        "epot": _np(sim.state.epot),
        "elec": _np(sim.state.elec),
        "vdw": _np(sim.state.vdw),
        "force_center": _np(sim.state.dyn.force.center),
        "force_orient": _np(sim.state.dyn.force.orientation.vec),
        "mu": _np(sim.state.induction.mu),
        "pressure": np.asarray(sim.pressure()),
    }
    return out


@case("fe_windows", group="f")
def fe_windows():
    """Batched lambda windows (5), 12 samples of reduced energies, dU/dlambda and dU/dtheta; TI / BAR /
    MBAR and the parameter-gradient estimators on the stored samples; FreeEnergyRun 40 steps."""
    from pgm_jax.analysis import free_energy as fe
    from pgm_jax.md import fe_grad as fg
    from pgm_jax.md.alchemy import FreeEnergyRun

    w, P, sim, alch = _alch_windows()
    pg = fg.ParamGradients(w)
    us, gs, Gs = [], [], []
    for _ in range(12):
        w.advance(5)
        u, g, _ = w.sample()
        us.append(u.T)
        gs.append(g)
        Gs.append(pg.sample())
    out = {"u": np.array(us), "dudl": np.array(gs), "dudp": np.array(Gs)}
    kT = float(w.integ.kT)
    S_ = {
        "u": np.array(us),
        "dudl": np.array(gs),
        "dudp": np.array(Gs),
        "lambdas": w.lambdas,
        "kT": kT,
        "time_ps": np.arange(1, 13, dtype=float) * 0.005,
        "meta": {"dudp_targets": [0, w.n - 1], "dudp_names": pg.space.names},
    }
    _put(
        out,
        "grad_est",
        {k: v for k, v in fg.gradient_estimate(S_, n_blocks=2).items() if not isinstance(v, (str, list))},
    )
    _put(out, "fe_est", {k: v for k, v in fe.estimate(S_, discard_ps=0.0).items() if not isinstance(v, (str, list))})
    w2, _, _, _ = _alch_windows(seed=2, dipole_tol=1e-8)
    run = FreeEnergyRun(w2, sample_every=5, exchange_every=10, log=None)
    run.run(40, prefix=None)
    _put(out, "run", {k: v for k, v in run.arrays().items() if k != "meta"})
    return out


# ============================================================================= fitting and interfaces
@case("fit_frames", group="f")
def fit_frames():
    """FrameAnalyzer (liquid fitting): U, M, alpha_cell, D, g(r) and their theta-derivatives."""
    from pgm_jax.fit import FrameAnalyzer, ParameterSpace, RDFSpec

    sys, pos, H = S.small_box(3)
    space = ParameterSpace.scales(sys.table, ["q", "cov", "alpha", "radius", "lj_r", "lj_eps"])
    st = md_settings(
        cutoff=0.6,
        skin=0.05,
        ewald_beta=6.0,
        pme_grid=(32, 32, 32),
        pme_order=6,
        lj_lrc=False,
        dipole_tol=1e-12,
        max_iter=500,
        peek=0.0,
        extrap_order=0,
        precision="double",
    )
    an = FrameAnalyzer(sys, H, st, space, rdf=RDFSpec.by_type(sys, "OW", rmax=0.8, nbins=40), tol=1e-12, chunk=2)
    th = np.array([0.02, -0.01, 0.03, 0.0, 0.01, -0.02])
    return {f"frame.{k}": np.asarray(v) for k, v in an.frame(th, pos, H).items()}


@case("interfaces_engine", group="f")
def interfaces_engine():
    """PGMEngine (external codes): energy, forces, dipoles, atomic virial; a second, displaced call
    through the predictor; a rotated general cell."""
    from pgm_jax.interfaces.engine import PGMEngine

    sys, pos, H = S.small_box(0)
    s = md_settings(
        cutoff=0.6,
        skin=0.05,
        ewald_beta=6.0,
        pme_grid=(32, 32, 32),
        pme_order=6,
        lj_lrc=False,
        dipole_tol=1e-10,
        max_iter=500,
        precision="double",
    )
    eng = PGMEngine(sys, pos, H, s, stress="atomic")
    out = {}
    rng = np.random.default_rng(3)
    for k, x in enumerate((pos, pos + 0.002 * rng.normal(size=pos.shape))):
        r = eng.compute(x, H, virial=True)
        out[f"c{k}.energy"], out[f"c{k}.forces"] = np.asarray(r.energy), _np(r.forces)
        out[f"c{k}.mu"], out[f"c{k}.virial"] = _np(r.induced_dipoles), _np(r.virial)
        out[f"c{k}.dipole"] = _np(r.dipole)
    return out


# ============================================================================= restraints, proteins, QM fitting,
# analysis
@case("md_restraints_npt", needs="pgm3p25", group="g")
def md_restraints_npt():
    """Rigid pGM3P-25 water, NPT with restraints of every kind (position: fixed / fractional / com,
    distance, angle, dihedral, com distance), 200 steps; restraint energies by kind."""
    from pgm_jax.md.restraints import (
        AngleRestraint,
        COMDistanceRestraint,
        DihedralRestraint,
        DistanceRestraint,
        PositionRestraint,
        Restraints,
        harmonic,
    )

    sys, pos, H, w, _ = water_system(pgm3p25=True)
    m = np.asarray(sys.masses, float)
    rs = Restraints(
        [
            PositionRestraint(
                [0, 3, 6, 9],
                pos[[0, 3, 6, 9]] + [[0.05, 0, 0], [0, 0.03, 0], [0.01, 0, 0], [0, 0, -0.09]],
                k=[400.0, 300.0, 500.0, 200.0],
                r0=[0.0, 0.01, 0.02, 0.0],
            ),
            PositionRestraint([12, 15], pos[[12, 15]] + 0.04, k=600.0, scaling="fractional", box=H),
            PositionRestraint(
                [18, 19, 20], pos[18:21] - 0.03, k=800.0, r0=0.01, scaling="com", box=H, weights=m[18:21]
            ),
            DistanceRestraint(
                [[0, 21], [3, 24]], ([0.0, 0.1], 0.2, [0.3, 0.3], [0.5, 0.5]), k2=[100.0, 50.0], k3=[200.0, 70.0]
            ),
            AngleRestraint([[0, 12, 27]], (0.3, 1.0, 1.5, 2.2), k2=40.0, k3=30.0),
            DihedralRestraint([[0, 9, 18, 27]], (-2.5, -1.0, 0.5, 2.0), k=25.0),
            DihedralRestraint([[3, 15, 24, 30]], harmonic(np.pi), k=15.0),
            COMDistanceRestraint([0, 1, 2], [33, 34, 35], (0.1, 0.25, 0.3, 0.4), k=300.0, masses=m),
        ]
    )
    sim = _rigid_water("npt", "bussi", tau_t=0.1, barostat_interval=10, restraints=rs)
    out = trajectory(sim, {})
    _put(out, "erestraint", sim.restraint_energies())
    out["pressure"] = np.asarray(sim.pressure())
    return out


@case("protein_peptide", group="g")
def protein_peptide():
    """Solvated ACE-ALA-SER-NME (ff19SB bonded terms, placeholder pGM, TIP3P as pGM, NaCl):
    load_amber + amber_template, flexible engine with X-H constraints and HMR, 2 fs, 60 steps;
    the pmemd-pgm prmtop and mdin written from it."""
    from pgm_jax.protein import ResidueLibrary, amber_template, load_amber, pmemd_grid, pmemd_mdin, write_pgm_prmtop

    prm, crd = os.path.join(S.DATA, "pep_wat.prmtop"), os.path.join(S.DATA, "pep_wat.inpcrd")
    lib = ResidueLibrary.placeholder(prm)
    lib.residues["ALA"]["cov"] = [
        ["N", "H", 0.0012],
        ["N", "CA", -0.004],
        ["C", "+N", 0.002],
        ["N", "-C", -0.003],
        ["C", "O", 0.005],
    ]
    lib.residues["SER"]["cov"] = [["OG", "HG", 0.003], ["N", "-C", -0.002]]
    asys = load_amber(prm, crd, electrostatics=lib)
    k = [i for i, mol in enumerate(asys.molecules) if mol.kind == "protein"][0]
    templates = asys.templates({k: amber_template(asys.molecules[k], prm)})
    s = tight(cutoff=0.8, skin=0.1, dipole_tol=1e-8, pme_grid=tuple(pmemd_grid(asys.box, 0.1)))
    sim = flexible_sim(
        asys.system(),
        templates,
        asys.system_positions(),
        asys.box,
        s,
        dt=0.002,
        ensemble="nvt",
        thermostat="bussi",
        tau_t=0.1,
        constraints="h-bonds",
        hmr=3.024,
        seed=7,
    )
    out = {"force0": _np(sim.state.dyn.force)}
    trajectory(sim, out, nsteps=60, every=30)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pep_pgm.prmtop")
        info = write_pgm_prmtop(asys, path, templates, hmr=3.024)
        out["prmtop_bytes"] = np.frombuffer(open(path, "rb").read(), np.uint8)
    _put(out, "prmtop_info", {kk: v for kk, v in info.items() if isinstance(v, (int, float, str))})
    out["mdin_bytes"] = np.frombuffer(pmemd_mdin(s, asys.box, nstlim=100).encode(), np.uint8)
    return out


@case("qmfit_synthetic", group="g")
def qmfit_synthetic():
    """QM cluster fitting: ClusterModel components of synthetic water clusters (dimers to a
    tetramer), the fit's residuals, loss and exact gradient at a displaced theta."""
    import pgm_jax.fit.qm as Q
    from pgm_jax.system import Molecule

    w = Molecule(
        "WAT",
        ["O", "H", "H"],
        ["OW", "HW", "HW"],
        np.array([-2.0405622, 1.0202811, 1.0202811]),
        np.array([0.0605150752, 0.0536231183, 0.0536231183]),
        np.array([1.11765163e-3, 3.2963077e-4, 3.2963077e-4]),
        cov=[(0, 1, -0.019120135), (0, 2, -0.019120135), (1, 0, 0.00858770326), (2, 0, 0.00858770326)],
        lj_rmin_half=np.array([0.178559032523734, 0, 0]),
        lj_sqrt_eps=np.array([0.7781620777141389, 0, 0]),
        bonds=[(0, 1), (0, 2), (1, 2)],
    )
    W0 = Q.rigid_water(0.9745, 103.64)
    cm = Q.ClusterModel(w, monomer_xyz_nm=W0 * 0.1)
    rng = np.random.default_rng(3)

    def rot():
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        return R if np.linalg.det(R) > 0 else -R

    def cluster(n, d):
        X = []
        for k in range(n):
            ang = 2 * np.pi * k / n
            c = (d / (2 * np.sin(np.pi / n)) if n > 2 else d / 2) * np.array(
                [np.cos(ang), np.sin(ang), 0.3 * rng.normal()]
            )
            X.append(W0 @ rot().T + c)
        return np.concatenate(X)

    recs = [
        {
            "id": f"c{k}",
            "set": "s",
            "n": n,
            "xyz_A": cluster(n, d).tolist(),
            "meta": {},
            "E": {"ref": -4.0 * n + 0.3 * k},
            "sapt": ({"elst": -6.0, "ind": -1.5, "exch": 4.0, "disp": -1.0} if n == 2 else {}),
            "nb": ({"nb3": -0.5, "nb2": -4.0 * n} if n > 2 else {}),
        }
        for k, (n, d) in enumerate([(2, 2.7), (2, 2.95), (2, 3.2), (3, 2.8), (3, 3.0), (4, 2.9)])
    ]
    data = Q.QMSet(recs, {"xyz_A": W0.tolist(), "dipole_D": 1.85, "polarizability_A3": 1.45})
    P = cm.table.initial()
    out = {}
    _put(out, "components3", cm.components(jnp.asarray(np.asarray(recs[3]["xyz_A"]) * 0.1), P, 3))
    prep = Q.Prepared(data, cm)
    _put(out, "predict", prep.predict(P))
    pm = Q.ParamMap(
        cm.table,
        [w],
        {"q": "all", "cov": "all", "radius": "all", "alpha": "all", "lj_rmin_half": ["OW"], "lj_sqrt_eps": ["OW"]},
    )
    fw = Q.FitWeights(
        total=1, elst=0.5, ind=0.5, exch_disp=0.5, nb3=1, force=0.0, dipole=1, polarizability=1, prior=0.1
    )
    fit = Q.QMFit(cm, pm, data, fw)
    th = jnp.asarray(pm.theta0 + 0.3 * rng.normal(size=len(pm)) * pm.scale)
    out["theta"], out["loss"] = _np(th), _np(fit.loss(th))
    out["grad"] = _np(jax.grad(fit.loss)(th))
    return out


@case("analysis_estimators", group="g")
def analysis_estimators():
    """Pure analysis code on synthetic data: dielectric fluctuation formula and jackknife, BAR,
    MBAR, TI, statistical inefficiency, WHAM, liquid-fit jackknife covariance, finite-field fits."""
    from pgm_jax.analysis import dielectric as D
    from pgm_jax.analysis import finite_field as FF
    from pgm_jax.analysis import free_energy as fe
    from pgm_jax.analysis import stats
    from pgm_jax.analysis.stats import jackknife_cov
    from pgm_jax.bias import analysis as A

    rng = np.random.default_rng(11)
    out = {}
    x = np.zeros(4000)
    for t in range(1, len(x)):  # AR(1): correlated series
        x[t] = 0.95 * x[t - 1] + rng.normal()
    M = np.stack([x, np.roll(x, 7) * 0.5, rng.normal(size=len(x))], 1) * 0.3
    out["eps_fluct"] = np.asarray(D.fluctuation(M, 15.6, 298.0))
    out["eps_jack"] = np.asarray(D.jackknife(M, 15.6, 298.0, nblocks=8))
    _put(out, "eps_static", D.static_dielectric(M, 15.6, 298.0, eps_inf_value=1.8, nblocks=8))
    out["g"] = np.asarray(stats.statistical_inefficiency(x))
    out["equil"] = np.asarray(stats.detect_equilibration(x[:1000], nskip=10))
    out["bar"] = np.asarray(fe.bar(rng.normal(1.0, 1.0, 500), rng.normal(-0.5, 1.0, 400)))
    K, N = 4, 300
    xs = [rng.normal(0.3 * k, 1.0, N) for k in range(K)]
    pooled = np.concatenate(xs)
    u_kn = np.array([0.5 * (pooled - 0.3 * k) ** 2 * (1.0 + 0.1 * k) for k in range(K)])
    f, Th = fe.mbar(u_kn, np.full(K, N))
    out["mbar_f"], out["mbar_theta"] = _np(f), _np(Th)
    out["ti"] = np.asarray(fe.ti(np.linspace(0, 1, 5), np.array([3.0, 2.1, 1.4, 0.9, 0.6]), np.full(5, 0.05)))
    axis = np.linspace(-1.0, 1.0, 41)
    cs = np.linspace(-0.8, 0.8, 5)
    samples = [rng.normal(c, 0.15, 800) for c in cs]
    Fw, fk = A.wham(samples, cs, np.full(5, 200.0), axis, 2.479)
    out["wham_F"], out["wham_f"] = _np(Fw), _np(fk)
    out["jk_cov"] = _np(jackknife_cov(rng.normal(size=(10, 3))))
    out["ff_block"] = np.asarray(stats.block_mean(x, 8))
    out["ff_tau"] = np.asarray(stats.integrated_correlation_time(x, 0.01))
    _put(out, "ff_pred", FF.predicted_errors(34.0, 1.8, 15.6, 298.0, 0.1, 5.0, 1000.0))
    return out
