"""Cell dipole, cell polarizability, dipole recording during MD and the dielectric analysis.

Modules: pgm_jax/md/dipoles.py (CellDipole, DipoleRecorder, read_dipoles) and
pgm_jax/analysis/dielectric.py.  What is checked, and against what: the decomposition of the cell
dipole into charge, permanent and induced parts (direct sums); one molecule in a large box against
the gas-phase molecule (the image field of tin-foil Ewald falls as 1 / V); invariance to wrapping
molecules and to the origin, also for a charged cell; the fluctuation formula, jackknife errors
and correlation time on Gaussian and AR(1) series with known answers; the IR line shape and sum
rule of an oscillating dipole; the .dip file format with continuations; dipoles recorded inside
the compiled MD blocks against the final state; scripts/trajectory_dipoles.py on an Amber
trajectory against the dipoles the run recorded.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _systems import PGM3P25_RST, PGM3P25_TOP, md_settings, methanol, requires_pgm3p25, small_box, water, water_geometry

from pgm_jax import ElecChannel, Molecule, System
from pgm_jax.analysis import dielectric as D
from pgm_jax.channels import molecular_polarizability, perm_dipoles
from pgm_jax.md.dipoles import DIP_COLUMNS, CellDipole, DipoleRecorder, cell_dipole, read_dipoles
from pgm_jax.md.forcefield import PGMForceField
from pgm_jax.md.simulation import Simulation
from pgm_jax.units import C_LIGHT_M_S, DEBYE_E_NM, E_NM_C_M, EPS0_SI, KB, KE


def _solve(sys, pos, H, **kw):
    """Return a PGMForceField (md_settings(**kw)), its neighbour rows and the solved result at pos."""
    ff = PGMForceField(sys, H, md_settings(**kw))
    idx = ff.rows_for(pos, H)
    return ff, idx, jax.jit(ff.compute)(pos, H, idx, ff.init_induction())


def test_decomposition_sums_to_charge_perm_and_induced_dipoles():
    """The cell dipole is the sum of the charge, permanent and induced dipoles (1e-12 e nm).

    For neutral molecules the charge part is sum q r; the permanent part equals perm_dipoles and the
    induced part the solved mu; the per-molecule dipoles add up to the total.
    """
    sys, pos, H = small_box(3)
    ff, idx, res = _solve(sys, pos, H)
    cd = CellDipole(ff)
    C = np.asarray(cd.components(pos, H, res.induction.mu))
    P = sys.expand(None)
    q = np.asarray(P["q"])
    assert np.allclose(C[0], (q[:, None] * pos).sum(0), atol=1e-12)  # neutral molecules: sum q r
    assert np.allclose(C[1], np.asarray(perm_dipoles(jnp.asarray(pos), sys, P["cov"])).sum(0), atol=1e-12)
    assert np.allclose(C[2], np.asarray(res.induction.mu).sum(0), atol=1e-12)
    assert np.allclose(np.asarray(cd.molecular(pos, H, res.induction.mu)).sum(0), C.sum(0), atol=1e-12)
    assert np.abs(C[2]).max() > 1e-3 and np.abs(C[1]).max() > 1e-3  # every part contributes


@pytest.mark.parametrize("mol", ["water", "methanol"])
def test_single_molecule_in_large_box_is_the_gas_phase_molecule(mol):
    """One molecule in a periodic box approaches the gas-phase molecule as 1 / V.

    Induced dipoles, total dipole and polarizability of one molecule in a periodic box approach
    the gas-phase model's (pgm_jax.Model / ElecChannel); the difference is the field of the
    periodic images, ~ 4 pi alpha p / (3 V) under tin-foil Ewald, so it falls as 1 / V.

    Tolerance: 3 times the image-field bound 4 pi alpha / (3 L^3); the error ratio between L = 3 and
    4.5 nm must lie in 1.5-6 around (4.5 / 3)^3 = 3.4.
    """
    if mol == "water":
        m = water()
        x = water_geometry()
    else:
        m, x = methanol()
    rng = np.random.default_rng(1)
    x = (x - x.mean(0)) @ np.linalg.qr(rng.normal(size=(3, 3)))[0].T
    sys = System([m])
    e, aux = ElecChannel().energy(jnp.asarray(x), sys, None)
    P = sys.expand(None)
    M_gas = (np.asarray(P["q"])[:, None] * x).sum(0) + np.asarray(aux["p"]).sum(0) + np.asarray(aux["mu"]).sum(0)
    A_gas = np.asarray(molecular_polarizability(jnp.asarray(x), sys))
    err = []
    for L in (3.0, 4.5):
        H = np.eye(3) * L
        pos = x + L / 2
        ff, idx, res = _solve(sys, pos, H, cutoff=0.9, skin=0.0, ewald_beta=4.0, pme_grid=(int(L * 16),) * 3)
        cd = CellDipole(ff)
        M = np.asarray(cd.components(pos, H, res.induction.mu)).sum(0)
        A = np.asarray(cd.polarizability(pos, H, idx, tol=1e-12))
        bound = 4 * np.pi / (3 * L**3) * np.abs(A_gas).max()  # image field / p
        assert np.abs(np.asarray(res.induction.mu) - np.asarray(aux["mu"])).max() < 3 * bound * np.linalg.norm(M_gas)
        assert np.abs(M - M_gas).max() < 3 * bound * np.linalg.norm(M_gas), (L, M, M_gas)
        assert np.abs(A - A_gas).max() < 3 * bound * np.abs(A_gas).max(), (L, A, A_gas)
        err.append(np.abs(A - A_gas).max())
    assert 1.5 < err[0] / err[1] < 6.0, err  # (4.5 / 3)^3 = 3.4


def test_cell_dipole_invariant_to_wrapping_and_origin():
    """The cell dipole is invariant to wrapping molecules and to translations, also when charged.

    Neutral molecules: M is unchanged when whole molecules are moved by lattice vectors or the
    system is translated.  A charged molecule contributes its dipole about its centre of mass, so
    M stays invariant for a charged (even net-charged) cell too.

    Charges and covalent dipoles are exact (1e-12); the induced part follows the PME grid (1e-6).
    """
    sys, pos, H = small_box(5)
    ff, idx, res = _solve(sys, pos, H)
    M0 = np.asarray(CellDipole(ff).components(pos, H, res.induction.mu))
    rng = np.random.default_rng(2)
    shift = rng.integers(-2, 3, size=(sys.nmol, 3)) @ H
    moved = pos + shift[sys.mol]
    ff1, idx1, res1 = _solve(sys, moved, H)
    M1 = np.asarray(CellDipole(ff1).components(moved, H, res1.induction.mu))
    assert np.abs(M1 - M0).max() < 1e-10 * np.abs(M0).max(), M1 - M0
    # translation: charges and covalent dipoles exact; induced dipoles to the PME grid accuracy
    moved = moved + np.array([0.37, -1.1, 0.23])
    ff1, idx1, res1 = _solve(sys, moved, H)
    M1 = np.asarray(CellDipole(ff1).components(moved, H, res1.induction.mu))
    assert np.abs(M1[:2] - M0[:2]).max() < 1e-12 and np.abs(M1[2] - M0[2]).max() < 1e-6 * np.abs(M0).max()
    # a charged methanol: net charge +0.3 e in the cell
    m, _ = methanol()
    ion = Molecule(
        "MeOH+",
        m.elements,
        m.types,
        m.q + 0.05,
        m.radius,
        m.alpha,
        cov=m.cov,
        lj_rmin_half=m.lj_rmin_half,
        lj_sqrt_eps=m.lj_sqrt_eps,
        bonds=m.bonds,
    )
    mols = [ion] + sys.molecules[1:]
    sysc = System(mols)
    assert abs(float(np.sum(sysc.expand(None)["q"])) - 0.3) < 1e-9
    fa, ia, ra = _solve(sysc, pos, H)
    fb, ib, rb = _solve(sysc, moved, H)
    cd = CellDipole(fa)
    assert np.allclose(cd.molecular_charges()[0], 0.3) and cd.molecular_charges()[1:].std() < 1e-12
    Ma, Mb = (
        np.asarray(cd.components(pos, H, ra.induction.mu)),
        np.asarray(CellDipole(fb).components(moved, H, rb.induction.mu)),
    )
    assert np.abs(Ma[:2] - Mb[:2]).max() < 1e-12 and np.abs(Ma[2] - Mb[2]).max() < 1e-6 * np.abs(Ma).max()


def test_dielectric_formula_on_gaussian_series():
    """The fluctuation formula, jackknife error and correlation time match analytic Gaussian results.

    Uncorrelated Gaussian M (sigma 1.3 e nm, 200,000 frames): eps - eps_inf = <dM^2> / (3 eps0 V kB T)
    within 5 standard errors (sqrt(2 / 3F)); the same in model units; eps_inf = 1 + 4 pi alpha / V;
    the jackknife error within 0.7-1.3 of theory.  AR(1) series (rho = 0.95): tau = -dt / ln(rho)
    within 10 %, the block jackknife error within 0.7-1.3 of sqrt(2 (1 + rho^2) / (3 F (1 - rho^2))).
    """
    rng = np.random.default_rng(0)
    V, T, sigma, F = 15.0, 298.0, 1.3, 200000
    M = rng.normal(size=(F, 3)) * sigma + np.array([0.4, -0.2, 0.1])
    expected = 3 * sigma**2 * E_NM_C_M**2 / (3 * EPS0_SI * V * 1e-27 * D.KB_SI * T)
    fl = D.fluctuation(M, V, T)
    assert abs(fl / expected - 1) < 5 * np.sqrt(2 / (3 * F))
    # the same in the model's units: 4 pi KE <dM^2> / (3 V kB T)
    dM2 = np.mean(np.sum(M * M, 1)) - np.sum(M.mean(0) ** 2)
    assert abs(4 * np.pi * KE * dM2 / (3 * V * KB * T) / fl - 1) < 1e-6
    r = D.static_dielectric(M, V, T, alpha=np.where(np.arange(F) % 100 == 0, 0.95, np.nan), nblocks=20)
    assert abs(r["eps_inf"] - (1 + 4 * np.pi * 0.95 / V)) < 1e-12
    assert abs(r["eps"] - r["eps_inf"] - fl) < 1e-9 * fl
    assert 0.7 < r["err"] / (expected * np.sqrt(2 / (3 * F))) < 1.3  # uncorrelated: jackknife = theory
    # correlated (AR(1)) series: tau, and the jackknife error with blocks >> tau
    rho, F = 0.95, 400000
    x = np.empty((F, 3))
    x[0] = rng.normal(size=3)
    xi = rng.normal(size=(F, 3)) * np.sqrt(1 - rho**2)
    for t in range(1, F):
        x[t] = rho * x[t - 1] + xi[t]
    tau = D.correlation_time(x, 0.01)
    assert abs(tau / (-0.01 / np.log(rho)) - 1) < 0.1, tau
    fl, err = D.jackknife(x, V, T, 50)
    theory = fl * np.sqrt(2 * (1 + rho**2) / (3 * F * (1 - rho**2)))
    assert 0.7 < err / theory < 1.3, (err, theory)
    run = D.running(x, V, T)
    assert run[-1][1] == pytest.approx(fl) and run[0][2] > run[-1][2]


def test_ir_spectrum_of_an_oscillating_dipole():
    """A rotating dipole gives an IR peak at its frequency and the analytic sum rule.

    Peak at the oscillation frequency and the sum rule int alpha n dw = pi beta <dM/dt^2> / (6 c eps0 V).

    Peak within 2 cm^-1 of 500 cm^-1; the integral within 2 % (finite segment length).
    """
    dt, nu = 0.002, 500.0  # ps, cm^-1
    w0 = 2 * np.pi * C_LIGHT_M_S * 100 * nu * 1e-12  # rad/ps
    t = np.arange(200000) * dt
    rng = np.random.default_rng(3)
    A = 0.2
    M = np.stack([A * np.cos(w0 * t + 0.3), A * np.sin(w0 * t + 0.3), 1e-4 * rng.normal(size=len(t))], 1)
    wn, an = D.ir_spectrum(M, dt, 15.0, 298.0, segment_ps=20.0)
    assert abs(wn[np.argmax(an)] - nu) < 2.0
    integral = np.sum(an * 100) * (wn[1] - wn[0]) * 100 * 2 * np.pi * C_LIGHT_M_S  # int alpha n dw, 1/m rad/s
    mdot2 = (A * w0 * 1e12 * E_NM_C_M) ** 2  # <|dM/dt|^2> of a rotating dipole
    expected = np.pi * mdot2 / (6 * C_LIGHT_M_S * EPS0_SI * 15.0e-27 * D.KB_SI * 298.0)
    assert abs(integral / expected - 1) < 0.02, integral / expected


def test_read_dipoles_drops_records_superseded_by_a_continuation(tmp_path):
    """read_dipoles keeps the header metadata and drops rows that a continuation overwrote."""
    p = tmp_path / "x.dip"
    rows = [
        (s, 0.001 * s, 300.0, 15.0) + tuple(np.full(9, v)) + (0.04, np.nan)
        for s, v in [(5, 1), (10, 2), (15, 3), (20, 4), (10, 20), (15, 30), (20, 40), (25, 50)]
    ]
    head = "# temperature_K = 298.0\n# n_atoms = 3\n# columns = " + " ".join(DIP_COLUMNS) + "\n"
    p.write_text(head + "".join(" ".join(str(v) for v in r) + "\n" for r in rows))
    meta, d = read_dipoles(str(p))
    assert meta["temperature_K"] == 298.0 and meta["n_atoms"] == 3
    assert list(d["step"]) == [5, 10, 15, 20, 25]
    assert np.allclose(d["M"][:, 0], [3, 60, 90, 120, 150])


def _run(tmp_path, name, report, engine="rigid"):
    """Run 40 steps of a small box (rigid or flexible engine) recording dipoles; return (sim, prefix).

    Parameters
    ----------
    tmp_path : pathlib.Path
        Output directory.
    name : str
        File prefix.
    report : int
        Block length [steps] (report_every); dipoles every 5 steps, induced dipoles every 20.
    engine : {"rigid", "flexible"}
        Rigid bodies (waters and methanols) or rigid waters by constraints (waters only).
    """
    sys, pos, H = small_box(0, nw=30, nm=0 if engine == "flexible" else 4)
    s = md_settings(cutoff=0.6, pme_grid=(32, 32, 32), pme_order=6, dipole_tol=1e-9, max_iter=200)
    if engine == "rigid":
        sim = Simulation(sys, pos, H, settings=s, dt=0.001, thermostat="bussi", log=None, seed=4)
    else:
        from pgm_jax.md.flexible import FlexibleSimulation, RigidTemplate

        tpl = RigidTemplate(sys.molecules[0], pos[sys.atom_slice(0)])
        sim = FlexibleSimulation(sys, [tpl] * sys.nmol, pos, H, s, dt=0.001, thermostat="bussi", seed=4)
    prefix = str(tmp_path / name)
    sim.run(40, report_every=report, prefix=prefix, dipoles_every=5, induced_every=20)
    return sim, prefix


@pytest.mark.parametrize("engine", ["rigid", "flexible"])
def test_recorded_series_match_the_state(tmp_path, monkeypatch, engine):
    """Dipoles recorded inside the compiled blocks equal direct evaluations of the state.

    Samples taken on the device inside the blocks equal those taken at block ends, the last one
    equals the final state's M, the polarizability rows equal a direct evaluation, and the per-atom
    induced dipoles are written (NetCDF).

    The in-block samples equal those taken at block ends to 1e-8 e nm (the same trajectory; the
    recording changes the summation order only), the last sample equals the final state's M (1e-9)
    and the polarizability rows a direct evaluation (1e-7 relative, the iterative solve).
    """
    monkeypatch.setattr(DipoleRecorder, "alpha_every", 2)
    sim, prefix = _run(tmp_path, "a", 20, engine)  # blocks of 20 steps, samples inside them
    meta, d = read_dipoles(prefix + ".dip")
    assert list(d["step"]) == [5, 10, 15, 20, 25, 30, 35, 40] and np.allclose(d["time_ps"], d["step"] * 0.001)
    assert meta["n_atoms"] == sim.sys.n and meta["charged_molecules"] == 0 and meta["interval"] == 5
    c = cell_dipole(sim)
    assert np.abs(d["M"][-1] - c["total"]).max() < 1e-9
    assert np.allclose(d["M_ind"][-1], c["ind"], atol=1e-9)
    assert np.allclose(c["debye"]["total"] * DEBYE_E_NM, c["total"])
    assert np.isnan(d["alpha_nm3"][::2]).all() and np.isfinite(d["alpha_nm3"][1::2]).all()
    st = sim.state
    pos = sim.positions()
    flex = getattr(sim, "flex", None)
    centers = st.dyn.position.center if flex is None else flex.list_centers(jnp.asarray(pos))
    idx = sim.nb.candidates(st.nbr, centers, st.box, jnp.asarray(pos))[0]
    a = float(jnp.trace(CellDipole(sim.ff).polarizability(pos, st.box, idx)) / 3)
    assert abs(d["alpha_nm3"][-1] - a) < 1e-7 * a
    from scipy.io import netcdf_file

    f = netcdf_file(prefix + ".mu.nc", "r", mmap=False)
    assert list(f.variables["step"][:]) == [20, 40] and f.variables["induced_dipoles"].units == b"e nm"
    assert np.allclose(np.array(f.variables["induced_dipoles"][-1]), np.asarray(st.induction.mu), atol=1e-7)
    f.close()
    if engine == "rigid":  # the same run sampled at block ends only
        _, prefix_b = _run(tmp_path, "b", 5, engine)
        _, db = read_dipoles(prefix_b + ".dip")
        assert np.abs(db["M"] - d["M"]).max() < 1e-8, np.abs(db["M"] - d["M"]).max()


@requires_pgm3p25
def test_trajectory_dipoles_of_an_amber_trajectory(tmp_path):
    """scripts/trajectory_dipoles.py reproduces the dipoles an MD run recorded (float32 frames).

    scripts/trajectory_dipoles.py on the Amber NetCDF trajectory of a run (the format pmemd
    writes) re-solves the induced dipoles and reproduces the cell dipoles the run recorded.

    Tolerance 1e-4 of the largest M: the NetCDF trajectory stores float32 coordinates.
    """
    import importlib.util
    import os

    from pgm_jax.md.forcefield import MDSettings

    spec = importlib.util.spec_from_file_location(
        "trajectory_dipoles",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "trajectory_dipoles.py"),
    )
    trajectory_dipoles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trajectory_dipoles)

    s = MDSettings().replace(
        cutoff=0.9,
        skin=0.1,
        ewald_beta=4.0,
        pme_grid=(48, 48, 48),
        pme_order=6,
        lj_lrc=True,
        dipole_tol=1e-10,
        max_iter=300,
        precision="double",
    )
    sim = Simulation.from_amber(PGM3P25_TOP, PGM3P25_RST, settings=s, dt=0.002, thermostat="bussi", log=None, seed=2)
    prefix = str(tmp_path / "w")
    sim.run(20, report_every=10, traj_every=10, prefix=prefix, dipoles_every=10)
    out = str(tmp_path / "t.dip")
    trajectory_dipoles.main(
        [
            PGM3P25_TOP,
            prefix + ".nc",
            "-o",
            out,
            "--nfft",
            "48",
            "48",
            "48",
            "--tol",
            "1e-10",
            "--precision",
            "double",
            "--alpha-every",
            "1",
        ]
    )
    _, a = read_dipoles(prefix + ".dip")
    meta, b = read_dipoles(out)
    assert len(a["M"]) == len(b["M"]) == 2 and meta["n_atoms"] == sim.sys.n
    scale = np.abs(a["M"]).max()
    for k in ("M_charge", "M_perm", "M_ind", "M"):  # coordinates are float32 in the trajectory
        assert np.abs(a[k] - b[k]).max() < 1e-4 * scale, (k, np.abs(a[k] - b[k]).max(), scale)
    assert np.allclose(a["volume_nm3"], b["volume_nm3"], rtol=1e-6) and np.isfinite(b["alpha_nm3"]).all()
