"""Fit Lennard-Jones parameters to liquid density and heat of vaporization (the `pgm-jax fit-liquid` command).

The fit uses exact ensemble gradients (fluctuation formulas + JAX derivatives of the pGM energy;
docs/liquid_fit.md).

For an NPT ensemble at parameters theta and any observable A(x, V),

    d<A>/dtheta = <dA/dtheta> - beta ( <A dU/dtheta> - <A><dU/dtheta> ),

where dU/dtheta is the derivative of the potential energy of each saved frame, taken by JAX at
the converged induced dipoles (the pGM energy is variational in the dipoles, so no derivative of
the dipole solve is needed).  With rho = M/V and Delta H_vap = <U_gas> - <U_liq>/N + RT this gives
the Jacobian of the targets; each iteration runs one NPT simulation, takes a damped Gauss-Newton
step, and the next simulation checks the predicted change.

The parameters are two scale factors on every atom's LJ parameters, theta = (ln s_R, ln s_eps):
R*_i -> s_R R*_i, eps_i -> s_eps eps_i; or (--params type) one pair of scales per atom type
(`lj_space`, a pgm_jax.fit.ParameterSpace).  Water: the rigid 512-water pGM3P-25 box (1 fs);
methanol: 216 flexible molecules from a FlexibleTemplate (--template, 0.5 fs).  Both NPT at
1 bar, Langevin 1/ps, Monte Carlo barostat every 25 steps.

Usage:

    python scripts/fitting/fit_liquid.py water    --start 0.0296,-0.357 --iters 6    # perturbed start
    python scripts/fitting/fit_liquid.py methanol --iters 6                         # from GAFF LJ
    python scripts/fitting/fit_liquid.py --help

Inputs: water: PGM_GVDW_DATA (pgm_jax.paths); methanol: the template (--template, default
runs/flex/methanol.flex, e.g. from examples/fit_bonded_template.py).
Outputs: --out (default runs/liquid/<system>.json): the targets, <U_gas>, and per iteration theta,
the estimates with block errors, the Jacobian, the step and the prediction of the next estimate;
one printed line per iteration.
Units: --temperature-K K; durations (--equil0-ps, --equil-ps, --prod-ps, --sample-ps) ps; densities
g/cm^3; heats of vaporization kcal/mol; energies kJ/mol.
Runtime: GPU; one NPT simulation per iteration.  Sets jax_enable_x64.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.bonded.study.gas_md import langevin
from pgm_jax.cli.args import add_temperature_arg, setup_logging
from pgm_jax.fit.params import Param, ParameterSpace
from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.box import box_from_cell, volume
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate, liquid_box
from pgm_jax.md.forcefield import MDSettings, PGMForceField
from pgm_jax.md.io import read_coordinates
from pgm_jax.md.rigid import _unwrap
from pgm_jax.md.simulation import Simulation
from pgm_jax.md.thermostats import Langevin
from pgm_jax.paths import pgm3p25_files, repo_path
from pgm_jax.system import MASSES, System
from pgm_jax.units import AMU_NM3_TO_G_CM3, KB, KCAL

jax.config.update("jax_enable_x64", True)
WATER_TOP, WATER_RST = pgm3p25_files()
METHANOL_TPL = repo_path("runs", "flex", "methanol.flex")
EXP = {  # experimental targets
    "water": {"rho": 0.997, "dhvap": 10.518},  # 298 K, 1 bar: g/cm^3, kcal/mol
    "methanol": {"rho": 0.7866, "dhvap": 37.43 / KCAL},
}


def lj_space(table: object, p0: dict, mode: str = "global") -> ParameterSpace:
    """Return the fitted Lennard-Jones parameters as a ParameterSpace of scales.

    Parameters
    ----------
    table : ParamTable
        The parameter table of the liquid.
    p0 : dict
        Starting parameters.
    mode : str
        "global": theta = (ln s_R, ln s_eps), every atom's R* times s_R and epsilon times s_eps;
        "type": one ln-scale of R* and one of epsilon per Lennard-Jones key (atom type) with
        epsilon > 0 (the Gauss-Newton step is then the minimum-norm step: two targets, more
        parameters, so add targets or a prior before trusting individual values).

    Returns
    -------
    ParameterSpace
    """
    if mode == "global":
        return ParameterSpace(table, [Param("lj_r"), Param("lj_eps")], p0)
    eps, rh = np.asarray(p0["lj_sqrt_eps"]), np.asarray(p0["lj_rmin_half"])
    kR, kE = table.keys["lj_rmin_half"], table.keys["lj_sqrt_eps"]
    params = [Param("lj_r", keys=[kR[i]]) for i in np.flatnonzero((rh > 0) & (eps > 0))]
    params += [Param("lj_eps", keys=[kE[i]]) for i in np.flatnonzero(eps > 0)]
    return ParameterSpace(table, params, p0)


def start_theta(space: ParameterSpace, text: str) -> np.ndarray:
    """Return theta from --start: one value per parameter, or (ln s_R, ln s_eps) for every type.

    Raises
    ------
    SystemExit
        Neither 2 nor `space.n` values.
    """
    v = np.array([float(x) for x in text.split(",")])
    if len(v) == space.n:
        return v
    if len(v) == 2:
        return v[lj_kind(space)]
    raise SystemExit(f"--start needs 2 or {space.n} values ({', '.join(space.names)})")


def lj_kind(space: ParameterSpace) -> np.ndarray:
    """Return 0 for the R* scales and 1 for the epsilon scales of the space, per parameter."""
    return np.array([0 if p.quantity == "lj_rmin_half" else 1 for p in space.params])


# ----------------------------------------------------------------------------- systems
def build(name: str, template: str = METHANOL_TPL, n_meth: int = 216) -> dict:
    """Return the liquid of the fit: system, positions and box [nm], template and time step [ps].

    Parameters
    ----------
    name : {"water", "methanol"}
        Water: the rigid pGM3P-25 box (dt 1 fs); methanol: n_meth flexible molecules of `template`
        in a random box at 0.55 of the liquid density (dt 0.5 fs).
    template : str
        FlexibleTemplate of methanol.
    n_meth : int
        Number of methanol molecules.

    Returns
    -------
    dict
        Keys sys (System), pos (N, 3) [nm], H (3, 3) [nm], tpl (FlexibleTemplate or None), dt [ps].
    """
    if name == "water":
        sys_ = System.from_prmtop(WATER_TOP)
        xyz, _, box = read_coordinates(WATER_RST)
        return {"sys": sys_, "pos": xyz * 0.1, "H": box_from_cell(*box) * 0.1, "tpl": None, "dt": 0.001}
    tpl = FlexibleTemplate.load(template)
    pos, H = liquid_box(tpl, n_meth, 0.55, seed=1, min_dist=0.18)
    return {"sys": System([tpl.pgm] * n_meth), "pos": pos, "H": H, "tpl": tpl, "dt": 0.0005}


def make_sim(
    sysd: dict,
    params: dict,
    T: float,
    settings: MDSettings,
    pos: np.ndarray,
    H: np.ndarray,
    vel: np.ndarray | None,
    seed: int,
) -> Simulation | FlexibleSimulation:
    """Return the NPT simulation of the liquid (Langevin 1/ps, MC barostat 1 bar every 25 steps).

    Parameters
    ----------
    sysd : dict
        The liquid (`build`).
    params : dict
        Force-field parameters (the fitted ones).
    T : float
        Temperature [K].
    settings : MDSettings
        Force-field settings.
    pos : np.ndarray (N, 3)
        Positions [nm].
    H : np.ndarray (3, 3)
        Box [nm].
    vel : np.ndarray (N, 3), optional
        Velocities [nm/ps] (None: Maxwell-Boltzmann).
    seed : int
        Random seed.

    Returns
    -------
    Simulation or FlexibleSimulation
        Rigid water, or the flexible engine with the template.
    """
    common = dict(
        dt=sysd["dt"],
        thermostat=Langevin(1.0),
        barostat=MonteCarloBarostat(1.0, 25),
        temperature=T,
        seed=seed,
        velocities=vel,
        params=params,
        log=None,
    )
    if sysd["tpl"] is None:
        return Simulation(sysd["sys"], pos, H, settings, **common)
    return FlexibleSimulation(sysd["sys"], [sysd["tpl"]] * sysd["sys"].nmol, pos, H, settings, **common)


def gas_energy(sysd: dict, p0: dict, T: float, settings: MDSettings) -> tuple[float, float]:
    """Return <U_gas> per molecule and its standard error [kJ/mol], in the MD engine's energy zero.

    Rigid water: the monomer energy (no error).  A flexible molecule: gas-phase Langevin MD with
    the fitted bonded model (16 replicas, 100 ps each at 0.5 fs, the first 10 ps dropped), shifted
    by the (MD engine - gas model) energy of one molecule at the reference geometry.  The MD
    engine's energy of one molecule is taken in a 5 nm box with a 2.2 nm cutoff, double precision.

    Parameters
    ----------
    sysd : dict
        The liquid (`build`).
    p0 : dict
        Starting parameters (not used: the gas phase does not depend on the LJ parameters).
    T : float
        Temperature [K].
    settings : MDSettings
        Settings of the liquid (not used; the single molecule has its own).

    Returns
    -------
    (float, float)
        <U_gas> and its standard error over the replicas [kJ/mol].

    Raises
    ------
    NotImplementedError
        A template with intramolecular LJ pairs.
    """
    mol = sysd["sys"].molecules[0]
    sl = sysd["sys"].atom_slice(0)
    x = _unwrap(np.asarray(sysd["pos"])[sl], np.asarray(sysd["H"]))
    Hb = np.eye(3) * 5.0
    big = MDSettings().replace(precision="double", dipole_tol=1e-9, cutoff=2.2, skin=0.05, lj_lrc=False)
    ff = PGMForceField(System([mol]), Hb, big)

    def e_md(y):
        """MD-engine energy [kJ/mol] of one molecule at y (A x 3) [nm], centred in the 5 nm box."""
        y = jnp.asarray(y - y.mean(0) + 2.5)
        idx = ff.rows_for(y, jnp.asarray(Hb))
        return float(ff.compute(y, jnp.asarray(Hb), idx, ff.init_induction(), None).energy["total"])

    if sysd["tpl"] is None:
        return e_md(x), 0.0
    tpl = sysd["tpl"]
    if len(tpl.lj_pairs()[0]):
        raise NotImplementedError("intramolecular LJ pairs: <U_gas> would depend on the LJ parameters")
    P = jax.tree_util.tree_map(jnp.asarray, tpl.P)

    def efun(R):
        """Gas-model energy [kJ/mol] of the template at R [nm]."""
        return tpl.model.energy(tpl.index, R, P)[0]

    x0 = jnp.asarray(tpl.spec.ref_xyz)
    offset = (e_md(np.asarray(x0)) + tpl.bonded_energy(x0)) - float(efun(x0))
    Xs, Es = langevin(
        efun, x0, [MASSES[e] for e in tpl.spec.elements], T, 0.0005, 200000, 100, 16, jax.random.PRNGKey(7)
    )
    Es = np.asarray(Es)[:, 200:]  # drop 10 ps per replica
    return float(Es.mean()) + float(offset), float(Es.mean(1).std() / np.sqrt(len(Es)))


def advance(sim: Simulation | FlexibleSimulation, nsteps: int, chunk: int = 2000) -> None:
    """Advance `nsteps` steps in chunks of `chunk` (the driver checks the neighbour list and box between chunks)."""
    while nsteps > 0:
        k = min(chunk, nsteps)
        sim.advance(k)
        nsteps -= k


# ----------------------------------------------------------------------------- one iteration
def sample(
    sim: Simulation | FlexibleSimulation,
    sysd: dict,
    theta: np.ndarray,
    T: float,
    n_prod: int,
    every: int,
    space: ParameterSpace,
) -> dict[str, np.ndarray]:
    """Run the production and return per frame U, rho and dU/dtheta.

    dU/dtheta is taken by JAX at the frame's converged induced dipoles (energy_fixed_mu; the pGM
    energy is variational in the dipoles); U_check is the same energy recomputed, compared with
    the engine's U as a consistency check.

    Parameters
    ----------
    sim : Simulation or FlexibleSimulation
        The equilibrated simulation.
    sysd : dict
        The liquid (`build`).
    theta : np.ndarray (n,)
        Current parameters.
    T : float
        Temperature [K] (not used).
    n_prod : int
        Production steps.
    every : int
        Steps between frames.
    space : ParameterSpace
        The fitted parameters.

    Returns
    -------
    dict
        U (F,) and U_check (F,) [kJ/mol], rho (F,) [g/cm^3], dU (F, n) [kJ/mol].
    """
    ff, flexible = sim.ff, sysd["tpl"] is not None
    M = float(np.sum(sysd["sys"].masses))

    def U(th, pos, H, mu, idx):
        """Potential energy [kJ/mol] at theta with the dipoles mu fixed (plus the flexible bonded terms)."""
        P = ff._atoms(space(th))
        e = ff.energy_fixed_mu(pos, H, mu, idx, P)[0]
        return e + (sim.flex.energy(pos, P) if flexible else 0.0)

    dU = jax.jit(jax.value_and_grad(U))
    out = {"U": [], "rho": [], "dU": [], "U_check": []}
    for _ in range(n_prod // every):
        sim.advance(every)
        st = sim.state
        if flexible:
            pos = st.dyn.position
            c = sim.flex.centers(pos)
        else:
            pos = sim.rigid.positions(st.dyn.position)
            c = st.dyn.position.center
        idx = sim.nb.candidates(st.nbr, c, st.box, pos)[0]
        u, g = dU(jnp.asarray(theta), pos, st.box, st.induction.mu, idx)
        out["U"].append(float(st.epot))
        out["U_check"].append(float(u))
        out["rho"].append(M / float(volume(st.box)) * AMU_NM3_TO_G_CM3)
        out["dU"].append(np.asarray(g))
    return {k: np.asarray(v) for k, v in out.items()}


def estimates(fr: dict[str, np.ndarray], T: float, N: int, u_gas: float) -> dict:
    """Return the density, heat of vaporization, their block errors and the Jacobian of both.

    d<A>/dtheta = <dA/dtheta> - beta (<A dU/dtheta> - <A><dU/dtheta>), with rho independent of
    theta at fixed configuration and Delta H_vap = <U_gas> - U / N + kT.

    Parameters
    ----------
    fr : dict
        Frames from `sample`.
    T : float
        Temperature [K].
    N : int
        Number of molecules.
    u_gas : float
        <U_gas> per molecule [kJ/mol].

    Returns
    -------
    dict
        rho, rho_se [g/cm^3]; dhvap, dhvap_se [kcal/mol] (5-block errors); J (2, n): rows
        d rho/dtheta [g/cm^3] and d dHvap/dtheta [kcal/mol] (theta-dependence of U_gas ignored);
        U_consistency_kJ (largest |U - U_check|).
    """
    beta = 1.0 / (KB * T)
    U, rho, dU = fr["U"], fr["rho"], fr["dU"]

    def cov(a):
        """Covariance of a (F,) with every column of dU (F, n)."""
        return (a[:, None] * dU).mean(0) - a.mean() * dU.mean(0)

    d_rho = -beta * cov(rho)
    d_U = dU.mean(0) - beta * cov(U)
    dh = (u_gas - U / N + KB * T) / KCAL
    J = np.stack([d_rho, -d_U / N / KCAL])

    def blocks(a):
        """Return the standard error of the mean of a from 5 blocks."""
        return np.array([b.mean() for b in np.array_split(a, 5)]).std(ddof=1) / np.sqrt(5)

    return {
        "rho": rho.mean(),
        "rho_se": blocks(rho),
        "dhvap": dh.mean(),
        "dhvap_se": blocks(dh),
        "J": J,
        "U_consistency_kJ": float(np.abs(fr["U"] - fr["U_check"]).max()),
    }


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system", choices=["water", "methanol"], help="the liquid")
    ap.add_argument("--template", default=METHANOL_TPL, help="methanol: FlexibleTemplate file")
    ap.add_argument(
        "--params",
        choices=["global", "type"],
        default="global",
        help="global: two scales for all atoms; type: R* and eps scale per LJ atom type",
    )
    ap.add_argument("--start", default="0,0", help="initial ln scales: 2 values, or one per parameter (--params type)")
    ap.add_argument("--iters", type=int, default=6, help="Gauss-Newton iterations")
    add_temperature_arg(ap, 298.0)
    ap.add_argument("--equil0-ps", type=float, default=100.0, help="equilibration before the first iteration [ps]")
    ap.add_argument("--equil-ps", type=float, default=20.0, help="equilibration before each later iteration [ps]")
    ap.add_argument("--prod-ps", type=float, default=200.0, help="production per iteration [ps]")
    ap.add_argument("--sample-ps", type=float, default=0.2, help="time between frames [ps]")
    ap.add_argument("--sigma-rho-g-cm3", type=float, default=0.005, help="residual scale of the density [g/cm^3]")
    ap.add_argument("--sigma-dhvap-kcal", type=float, default=0.05, help="residual scale of dHvap [kcal/mol]")
    ap.add_argument(
        "--targets",
        default="",
        help="rho,dHvap (g/cm^3, kcal/mol) instead of experiment, "
        "e.g. the model's own values for a parameter-recovery test",
    )
    ap.add_argument("-o", "--out", default="", help="JSON log (default runs/liquid/<system>.json)")
    return ap


def main(argv: list[str] | None = None) -> None:
    """Parse the command line, set up the liquid and gas phase, and run the fit (see the module docstring)."""
    a = build_parser().parse_args(argv)
    setup_logging()
    T = a.temperature_K
    out = a.out or repo_path("runs", "liquid", f"{a.system}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sysd = build(a.system, a.template)
    settings = MDSettings()
    p0 = jax.tree_util.tree_map(jnp.asarray, sysd["sys"].table.initial())
    N = sysd["sys"].nmol
    t0 = time.time()
    u_gas, u_gas_se = gas_energy(sysd, p0, T, settings)
    print(
        f"# {a.system}: {N} molecules, <U_gas> = {u_gas:.3f} +- {u_gas_se:.3f} kJ/mol ({time.time() - t0:.0f} s)",
        flush=True,
    )
    space = lj_space(sysd["sys"].table, p0, a.params)
    theta = start_theta(space, a.start)
    clip = np.where(lj_kind(space) == 0, 0.02, 0.25)  # largest step of ln s_R and ln s_eps
    exp_ = dict(EXP[a.system])
    if a.targets:
        exp_["rho"], exp_["dhvap"] = (float(v) for v in a.targets.split(","))
    y_exp = np.array([exp_["rho"], exp_["dhvap"]])
    sig = np.array([a.sigma_rho_g_cm3, a.sigma_dhvap_kcal])
    log = {"system": a.system, "T": T, "exp": exp_, "u_gas": u_gas, "u_gas_se": u_gas_se, "iters": []}
    pos, H, vel = sysd["pos"], sysd["H"], None
    pred = None
    dt = sysd["dt"]
    for it in range(a.iters):
        t1 = time.time()
        params = space(jnp.asarray(theta))
        sim = make_sim(sysd, params, T, settings, pos, H, vel, seed=100 + it)
        advance(sim, int(round((a.equil0_ps if it == 0 else a.equil_ps) / dt)))
        fr = sample(sim, sysd, theta, T, int(round(a.prod_ps / dt)), int(round(a.sample_ps / dt)), space)
        est = estimates(fr, T, N, u_gas)
        y = np.array([est["rho"], est["dhvap"]])
        r = (y - y_exp) / sig
        Js = est["J"] / sig[:, None]
        step = -np.linalg.solve(Js.T @ Js + 1e-3 * np.eye(space.n), Js.T @ r)  # damped Gauss-Newton
        step = np.clip(step, -clip, clip)
        rec = {
            "iter": it,
            "theta": theta.tolist(),
            "names": space.names,
            "s_R": float(np.exp(theta[0])),
            "s_eps": float(np.exp(theta[-1])),
            "rho": est["rho"],
            "rho_se": est["rho_se"],
            "dhvap": est["dhvap"],
            "dhvap_se": est["dhvap_se"],
            "J": est["J"].tolist(),
            "predicted_from_previous": pred,
            "step": step.tolist(),
            "U_consistency_kJ": est["U_consistency_kJ"],
            "frames": len(fr["U"]),
            "wall_s": time.time() - t1,
        }
        log["iters"].append(rec)
        par = (
            f"s_R {rec['s_R']:.4f} s_eps {rec['s_eps']:.4f}"
            if a.params == "global"
            else "scales " + " ".join(f"{v:.4f}" for v in np.exp(theta))
        )
        print(
            f"iter {it}: {par}  rho {est['rho']:.4f} +- {est['rho_se']:.4f}  "
            f"dHvap {est['dhvap']:.3f} +- {est['dhvap_se']:.3f} kcal/mol  (target {y_exp[0]:.4f}, {y_exp[1]:.3f}); "
            f"predicted {pred}; J {np.round(est['J'], 4).tolist()}; {time.time() - t1:.0f} s",
            flush=True,
        )
        json.dump(log, open(out, "w"), indent=1)
        pred = (y + est["J"] @ step).tolist()
        theta = theta + step
        pos, H, vel = sim.positions(), np.asarray(sim.state.box), sim.velocities()
    print(f"# done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
