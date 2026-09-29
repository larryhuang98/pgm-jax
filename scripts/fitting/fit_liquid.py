"""Fit Lennard-Jones parameters to liquid density and heat of vaporization with exact ensemble
gradients (fluctuation formulas + JAX derivatives of the pGM energy).

For an NPT ensemble at parameters theta and any observable A(x, V),

    d<A>/dtheta = <dA/dtheta> - beta ( <A dU/dtheta> - <A><dU/dtheta> ),

where dU/dtheta is the derivative of the potential energy of each saved frame, taken by JAX at
the converged induced dipoles (the pGM energy is variational in the dipoles, so no derivative of
the dipole solve is needed).  With rho = M/V and Delta H_vap = <U_gas> - <U_liq>/N + RT this gives
the Jacobian of the targets; each iteration runs one NPT simulation, takes a damped Gauss-Newton
step, and the next simulation checks the predicted change.

The parameters are two scale factors on every atom's LJ parameters, theta = (ln s_R, ln s_eps):
R*_i -> s_R R*_i, eps_i -> s_eps eps_i; or (--params type) one pair of scales per atom type
(`lj_space`, a pgm_jax.fit.ParameterSpace).

    python scripts/fit_liquid.py water    --start 0.0296,-0.357 --iters 6    # perturbed start
    python scripts/fit_liquid.py methanol --iters 6                         # from GAFF LJ
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.cli.args import setup_logging
from pgm_jax.fit.params import Param, ParameterSpace
from pgm_jax.md.barostats import MonteCarloBarostat
from pgm_jax.md.box import box_from_cell, volume
from pgm_jax.md.forcefield import MDSettings, PGMForceField
from pgm_jax.md.io import read_coordinates
from pgm_jax.md.simulation import Simulation
from pgm_jax.md.thermostats import Langevin
from pgm_jax.paths import resource
from pgm_jax.system import System
from pgm_jax.units import AMU_NM3_TO_G_CM3, KB, KCAL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
jax.config.update("jax_enable_x64", True)
WATER_TOP = resource("gvdw_data", "topology/rayl_512_v2.prmtop")
WATER_RST = resource("gvdw_data", "inputs/lj/inpcrd.restrt")
EXP = {
    "water": {"rho": 0.997, "dhvap": 10.518},  # 298 K, 1 bar: g/cm^3, kcal/mol
    "methanol": {"rho": 0.7866, "dhvap": 37.43 / KCAL},
}


def lj_space(table, p0, mode: str = "global") -> ParameterSpace:
    """The fitted Lennard-Jones parameters as a ParameterSpace of scales.

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
    """theta from --start: one value per parameter, or (ln s_R, ln s_eps) for every type."""
    v = np.array([float(x) for x in text.split(",")])
    if len(v) == space.n:
        return v
    if len(v) == 2:
        return v[lj_kind(space)]
    raise SystemExit(f"--start needs 2 or {space.n} values ({', '.join(space.names)})")


def lj_kind(space: ParameterSpace) -> np.ndarray:
    """0 for the R* scales, 1 for the epsilon scales of the space."""
    return np.array([0 if p.quantity == "lj_rmin_half" else 1 for p in space.params])


# ----------------------------------------------------------------------------- systems
def build(name, n_meth=216):
    if name == "water":
        sys_ = System.from_prmtop(WATER_TOP)
        xyz, _, box = read_coordinates(WATER_RST)
        return {"sys": sys_, "pos": xyz * 0.1, "H": box_from_cell(*box) * 0.1, "tpl": None, "dt": 0.001}
    from pgm_jax.md.flexible import FlexibleTemplate, liquid_box

    tpl = FlexibleTemplate.load(os.path.join(ROOT, "runs/flex/methanol.flex"))
    pos, H = liquid_box(tpl, n_meth, 0.55, seed=1, min_dist=0.18)
    return {"sys": System([tpl.pgm] * n_meth), "pos": pos, "H": H, "tpl": tpl, "dt": 0.0005}


def make_sim(sysd, params, T, settings, pos, H, vel, seed):
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
    from pgm_jax.md.flexible import FlexibleSimulation

    return FlexibleSimulation(sysd["sys"], [sysd["tpl"]] * sysd["sys"].nmol, pos, H, settings, **common)


def gas_energy(sysd, p0, T, settings):
    """<U_gas> per molecule (kJ/mol) and its standard error, in the MD engine's own units: rigid
    water = the monomer energy; a flexible molecule = gas-phase Langevin MD with the fitted
    bonded model, shifted by (MD engine - gas model) energy of one molecule in a large box."""
    mol = sysd["sys"].molecules[0]
    sl = sysd["sys"].atom_slice(0)
    from pgm_jax.md.rigid import _unwrap

    x = _unwrap(np.asarray(sysd["pos"])[sl], np.asarray(sysd["H"]))
    Hb = np.eye(3) * 5.0
    big = MDSettings().replace(precision="double", dipole_tol=1e-9, cutoff=2.2, skin=0.05, lj_lrc=False)
    ff = PGMForceField(System([mol]), Hb, big)

    def e_md(y):
        y = jnp.asarray(y - y.mean(0) + 2.5)
        idx = ff.rows_for(y, jnp.asarray(Hb))
        return float(ff.compute(y, jnp.asarray(Hb), idx, ff.init_induction(), None).energy["total"])

    if sysd["tpl"] is None:
        return e_md(x), 0.0
    tpl = sysd["tpl"]
    if len(tpl.lj_pairs()[0]):
        raise NotImplementedError("intramolecular LJ pairs: <U_gas> would depend on the LJ parameters")
    from pgm_jax.bonded.study.gas_md import langevin
    from pgm_jax.system import MASSES

    P = jax.tree_util.tree_map(jnp.asarray, tpl.P)

    def efun(R):
        return tpl.model.energy(tpl.index, R, P)[0]

    x0 = jnp.asarray(tpl.spec.ref_xyz)
    offset = (e_md(np.asarray(x0)) + tpl.bonded_energy(x0)) - float(efun(x0))
    Xs, Es = langevin(
        efun, x0, [MASSES[e] for e in tpl.spec.elements], T, 0.0005, 200000, 100, 16, jax.random.PRNGKey(7)
    )
    Es = np.asarray(Es)[:, 200:]  # drop 10 ps per replica
    return float(Es.mean()) + float(offset), float(Es.mean(1).std() / np.sqrt(len(Es)))


def advance(sim, nsteps, chunk=2000):
    """Run in chunks (the driver checks the neighbour list and box between chunks)."""
    while nsteps > 0:
        k = min(chunk, nsteps)
        sim.advance(k)
        nsteps -= k


# ----------------------------------------------------------------------------- one iteration
def sample(sim, sysd, theta, T, n_prod, every, space):
    """Production run; per frame: U (kJ/mol), rho (g/cm^3), dU/dtheta."""
    ff, flexible = sim.ff, sysd["tpl"] is not None
    M = float(np.sum(sysd["sys"].masses))

    def U(th, pos, H, mu, idx):
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


def estimates(fr, T, N, u_gas):
    beta = 1.0 / (KB * T)
    U, rho, dU = fr["U"], fr["rho"], fr["dU"]

    def cov(a):
        return (a[:, None] * dU).mean(0) - a.mean() * dU.mean(0)

    d_rho = -beta * cov(rho)
    d_U = dU.mean(0) - beta * cov(U)
    dh = (u_gas - U / N + KB * T) / KCAL
    J = np.stack([d_rho, -d_U / N / KCAL])

    def blocks(a):
        return np.array([b.mean() for b in np.array_split(a, 5)]).std(ddof=1) / np.sqrt(5)

    return {
        "rho": rho.mean(),
        "rho_se": blocks(rho),
        "dhvap": dh.mean(),
        "dhvap_se": blocks(dh),
        "J": J,
        "U_consistency_kJ": float(np.abs(fr["U"] - fr["U_check"]).max()),
    }


def main():
    """Command line: parse the options, set up the liquid and gas phase, and run the fit."""
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=["water", "methanol"])
    ap.add_argument(
        "--params",
        choices=["global", "type"],
        default="global",
        help="global: two scales for all atoms; type: R* and eps scale per LJ atom type",
    )
    ap.add_argument("--start", default="0,0", help="initial ln scales: 2 values, or one per parameter (--params type)")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--T", type=float, default=298.0)
    ap.add_argument("--equil0", type=float, default=100.0, help="ps before the first iteration")
    ap.add_argument("--equil", type=float, default=20.0, help="ps before each later iteration")
    ap.add_argument("--prod", type=float, default=200.0, help="ps of production per iteration")
    ap.add_argument("--every", type=float, default=0.2, help="ps between frames")
    ap.add_argument("--sig_rho", type=float, default=0.005)
    ap.add_argument("--sig_dh", type=float, default=0.05)
    ap.add_argument(
        "--targets",
        default="",
        help="rho,dHvap (g/cm^3, kcal/mol) instead of experiment, "
        "e.g. the model's own values for a parameter-recovery test",
    )
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    setup_logging()
    out = a.out or os.path.join(ROOT, f"runs/liquid/{a.system}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sysd = build(a.system)
    settings = MDSettings()
    p0 = jax.tree_util.tree_map(jnp.asarray, sysd["sys"].table.initial())
    N = sysd["sys"].nmol
    t0 = time.time()
    u_gas, u_gas_se = gas_energy(sysd, p0, a.T, settings)
    print(
        f"# {a.system}: {N} molecules, <U_gas> = {u_gas:.3f} +- {u_gas_se:.3f} kJ/mol ({time.time() - t0:.0f} s)",
        flush=True,
    )
    space = lj_space(sysd["sys"].table, p0, a.params)
    theta = start_theta(space, a.start)
    clip = np.where(lj_kind(space) == 0, 0.02, 0.25)
    exp_ = dict(EXP[a.system])
    if a.targets:
        exp_["rho"], exp_["dhvap"] = (float(v) for v in a.targets.split(","))
    y_exp = np.array([exp_["rho"], exp_["dhvap"]])
    sig = np.array([a.sig_rho, a.sig_dh])
    log = {"system": a.system, "T": a.T, "exp": exp_, "u_gas": u_gas, "u_gas_se": u_gas_se, "iters": []}
    pos, H, vel = sysd["pos"], sysd["H"], None
    pred = None
    dt = sysd["dt"]
    for it in range(a.iters):
        t1 = time.time()
        params = space(jnp.asarray(theta))
        sim = make_sim(sysd, params, a.T, settings, pos, H, vel, seed=100 + it)
        advance(sim, int(round((a.equil0 if it == 0 else a.equil) / dt)))
        fr = sample(sim, sysd, theta, a.T, int(round(a.prod / dt)), int(round(a.every / dt)), space)
        est = estimates(fr, a.T, N, u_gas)
        y = np.array([est["rho"], est["dhvap"]])
        r = (y - y_exp) / sig
        Js = est["J"] / sig[:, None]
        step = -np.linalg.solve(Js.T @ Js + 1e-3 * np.eye(space.n), Js.T @ r)
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
