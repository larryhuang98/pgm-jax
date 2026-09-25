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
R*_i -> s_R R*_i, eps_i -> s_eps eps_i (per-type parameters work the same way; see `to_params`).

    python scripts/fit_liquid.py water    --start 0.0296,-0.357 --iters 6    # perturbed start
    python scripts/fit_liquid.py methanol --iters 6                         # from GAFF LJ
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from pgm_jax.md.box import volume
from pgm_jax.md.forcefield import MDSettings, PGMForceField
from pgm_jax.md.io import box_from_cell, read_coordinates
from pgm_jax.md.simulation import Simulation, _dedupe
from pgm_jax.param import read_prmtop_pgm
from pgm_jax.system import System

KB = 0.0083144626181532            # kJ/mol/K
KCAL = 4.184
G_CM3 = 1.66053906660e-3           # amu/nm^3 -> g/cm^3
WATER_TOP = os.path.expanduser("~/pgm-gvdw-data/topology/rayl_512_v2.prmtop")
WATER_RST = os.path.expanduser("~/pgm-gvdw-data/inputs/lj/inpcrd.restrt")
EXP = {"water": {"rho": 0.997, "dhvap": 10.518},      # 298 K, 1 bar: g/cm^3, kcal/mol
       "methanol": {"rho": 0.7866, "dhvap": 37.43 / KCAL}}


def to_params(theta, p0):
    """theta = (ln s_R, ln s_eps) -> the parameter table (R* scaled by s_R, eps by s_eps)."""
    p = dict(p0)
    p["lj_rmin_half"] = p0["lj_rmin_half"] * jnp.exp(theta[0])
    p["lj_sqrt_eps"] = p0["lj_sqrt_eps"] * jnp.exp(0.5 * theta[1])
    return p


# ----------------------------------------------------------------------------- systems
def build(name, n_meth=216):
    if name == "water":
        sys_ = System(_dedupe(read_prmtop_pgm(WATER_TOP, first_residue_only=False)))
        xyz, _, box = read_coordinates(WATER_RST)
        return {"sys": sys_, "pos": xyz * 0.1, "H": box_from_cell(*box) * 0.1, "tpl": None, "dt": 0.001}
    from pgm_jax.md.flexible import FlexibleTemplate, liquid_box
    tpl = FlexibleTemplate.load(os.path.join(ROOT, "runs/flex/methanol.flex"))
    pos, H = liquid_box(tpl, n_meth, 0.55, seed=1, min_dist=0.18)
    return {"sys": System([tpl.pgm] * n_meth), "pos": pos, "H": H, "tpl": tpl, "dt": 0.0005}


def make_sim(sysd, params, T, settings, pos, H, vel, seed):
    common = dict(dt=sysd["dt"], ensemble="npt", temperature=T, gamma=1.0, pressure=1.0, barostat_interval=25,
                  seed=seed, vel_nm_ps=vel, params=params, log=None)
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
    big = MDSettings(precision="double", dipole_tol=1e-9, cutoff=2.2, skin=0.05, lj_lrc=False)
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
    sys.path.insert(0, os.path.join(ROOT, "scripts/bonded"))
    from md_check import MASS, langevin
    P = jax.tree_util.tree_map(jnp.asarray, tpl.P)
    efun = lambda R: tpl.model.energy(tpl.index, R, P)[0]
    x0 = jnp.asarray(tpl.spec.ref_xyz)
    offset = (e_md(np.asarray(x0)) + tpl.bonded_energy(x0)) - float(efun(x0))
    Xs, Es = langevin(efun, x0, [MASS[e] for e in tpl.spec.elements], T, 0.0005, 200000, 100, 16, jax.random.PRNGKey(7))
    Es = np.asarray(Es)[:, 200:]                                # drop 10 ps per replica
    return float(Es.mean()) + float(offset), float(Es.mean(1).std() / np.sqrt(len(Es)))


def advance(sim, nsteps, chunk=2000):
    """Run in chunks (the driver checks the neighbour list and box between chunks)."""
    while nsteps > 0:
        k = min(chunk, nsteps)
        sim._advance(k)
        nsteps -= k


# ----------------------------------------------------------------------------- one iteration
def sample(sim, sysd, theta, p0, T, n_prod, every):
    """Production run; per frame: U (kJ/mol), rho (g/cm^3), dU/dtheta."""
    ff, flexible = sim.ff, sysd["tpl"] is not None
    M = float(np.sum(sysd["sys"].masses))

    def U(th, pos, H, mu, idx):
        P = ff._atoms(to_params(th, p0))
        e = ff.energy_fixed_mu(pos, H, mu, idx, P)[0]
        return e + (sim.flex.energy(pos, P) if flexible else 0.0)

    dU = jax.jit(jax.value_and_grad(U))
    out = {"U": [], "rho": [], "dU": [], "U_check": []}
    for _ in range(n_prod // every):
        sim._advance(every)
        st = sim.state
        if flexible:
            pos = st.dyn.position
            c = sim.flex.centers(pos)
        else:
            pos = sim.rigid.positions(st.dyn.position)
            c = st.dyn.position.center
        idx = sim.nb.candidates(st.nbr, c, st.box, pos)[0]
        u, g = dU(jnp.asarray(theta), pos, st.box, st.induction.mu, idx)
        out["U"].append(float(st.epot)); out["U_check"].append(float(u))
        out["rho"].append(M / float(volume(st.box)) * G_CM3)
        out["dU"].append(np.asarray(g))
    return {k: np.asarray(v) for k, v in out.items()}


def estimates(fr, T, N, u_gas):
    beta = 1.0 / (KB * T)
    U, rho, dU = fr["U"], fr["rho"], fr["dU"]
    cov = lambda a: (a[:, None] * dU).mean(0) - a.mean() * dU.mean(0)
    d_rho = -beta * cov(rho)
    d_U = dU.mean(0) - beta * cov(U)
    dh = (u_gas - U / N + KB * T) / KCAL
    J = np.stack([d_rho, -d_U / N / KCAL])
    blocks = lambda a: np.array([b.mean() for b in np.array_split(a, 5)]).std(ddof=1) / np.sqrt(5)
    return {"rho": rho.mean(), "rho_se": blocks(rho), "dhvap": dh.mean(), "dhvap_se": blocks(dh), "J": J,
            "U_consistency_kJ": float(np.abs(fr["U"] - fr["U_check"]).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=["water", "methanol"])
    ap.add_argument("--start", default="0,0", help="initial ln s_R, ln s_eps")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--T", type=float, default=298.0)
    ap.add_argument("--equil0", type=float, default=100.0, help="ps before the first iteration")
    ap.add_argument("--equil", type=float, default=20.0, help="ps before each later iteration")
    ap.add_argument("--prod", type=float, default=200.0, help="ps of production per iteration")
    ap.add_argument("--every", type=float, default=0.2, help="ps between frames")
    ap.add_argument("--sig_rho", type=float, default=0.005)
    ap.add_argument("--sig_dh", type=float, default=0.05)
    ap.add_argument("--targets", default="", help="rho,dHvap (g/cm^3, kcal/mol) instead of experiment, "
                    "e.g. the model's own values for a parameter-recovery test")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, f"runs/liquid/{a.system}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sysd = build(a.system)
    settings = MDSettings()
    p0 = jax.tree_util.tree_map(jnp.asarray, sysd["sys"].table.initial())
    N = sysd["sys"].nmol
    t0 = time.time()
    u_gas, u_gas_se = gas_energy(sysd, p0, a.T, settings)
    print(f"# {a.system}: {N} molecules, <U_gas> = {u_gas:.3f} +- {u_gas_se:.3f} kJ/mol ({time.time() - t0:.0f} s)", flush=True)
    theta = np.array([float(v) for v in a.start.split(",")])
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
        params = to_params(jnp.asarray(theta), p0)
        sim = make_sim(sysd, params, a.T, settings, pos, H, vel, seed=100 + it)
        advance(sim, int(round((a.equil0 if it == 0 else a.equil) / dt)))
        fr = sample(sim, sysd, theta, p0, a.T, int(round(a.prod / dt)), int(round(a.every / dt)))
        est = estimates(fr, a.T, N, u_gas)
        y = np.array([est["rho"], est["dhvap"]])
        r = (y - y_exp) / sig
        Js = est["J"] / sig[:, None]
        step = -np.linalg.solve(Js.T @ Js + 1e-3 * np.eye(2), Js.T @ r)
        step = np.clip(step, [-0.02, -0.25], [0.02, 0.25])
        rec = {"iter": it, "theta": theta.tolist(), "s_R": float(np.exp(theta[0])), "s_eps": float(np.exp(theta[1])),
               "rho": est["rho"], "rho_se": est["rho_se"], "dhvap": est["dhvap"], "dhvap_se": est["dhvap_se"],
               "J": est["J"].tolist(), "predicted_from_previous": pred, "step": step.tolist(),
               "U_consistency_kJ": est["U_consistency_kJ"], "frames": len(fr["U"]), "wall_s": time.time() - t1}
        log["iters"].append(rec)
        print(f"iter {it}: s_R {rec['s_R']:.4f} s_eps {rec['s_eps']:.4f}  rho {est['rho']:.4f} +- {est['rho_se']:.4f}  "
              f"dHvap {est['dhvap']:.3f} +- {est['dhvap_se']:.3f} kcal/mol  (target {y_exp[0]:.4f}, {y_exp[1]:.3f}); "
              f"predicted {pred}; J {np.round(est['J'], 4).tolist()}; {time.time() - t1:.0f} s", flush=True)
        json.dump(log, open(out, "w"), indent=1)
        pred = (y + est["J"] @ step).tolist()
        theta = theta + step
        pos, H, vel = sim.positions_nm(), np.asarray(sim.state.box), sim.velocities_nm_ps()
    print(f"# done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
