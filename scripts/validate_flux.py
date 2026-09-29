"""Charge flux in MD (pgm_jax/md/flux.py): single points against the bonded model, forces, virial
and gradients against finite differences, in float64 with the dipoles solved to 1e-12.

    python examples/fit_bonded_template.py methanol --flux 2 --wmu 1 --maxiter 4000 --out runs/flux/methanol_flux2.flex
    python scripts/validate_flux.py runs/flux/methanol_flux2.flex     # -> the tables of docs/charge_flux.md

1. One molecule: flux charges and covalent dipoles vs BondedModel._flux; the bonded model's gas-phase
   pGM energy vs the gas-phase Model (ElecChannel) of the molecule with those charges; the MD engine
   with flux vs the MD engine without flux on the same charges (large box); MD forces vs the
   gradient of the gas-phase model the template was fitted with.
2. 32 flexible molecules in a periodic box (bonds perturbed off their references): forces vs
   autodiff of the energy at fixed dipoles and vs central differences with the dipoles re-solved;
   molecular and atomic strain derivatives vs differences of the energy under box scaling;
   differentiable path (forces and dipoles differentiated w.r.t. parameters incl. jb/jc/jc2,
   positions, box) and dE/d(flux parameters) vs central differences."""

import os
import sys
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from pgm_jax.channels import ElecChannel
from pgm_jax.md.flexible import FlexibleSimulation, FlexibleTemplate, liquid_box
from pgm_jax.md.flux import ChargeFlux, molecule_at
from pgm_jax.md.forcefield import MDSettings, PGMForceField
from pgm_jax.system import System

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
jax.config.update("jax_enable_x64", True)
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs/flux/methanol_flux2.flex")
tpl = FlexibleTemplate.load(path)
rng = np.random.default_rng(0)


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(np.max(np.abs(np.asarray(b))), 1e-300))


print(f"template {path}: {tpl.name}, flux {tpl.settings['flux']}")

# ------------------------------------------------------------------ 1. single molecule
x = np.asarray(tpl.spec.ref_xyz) + 0.004 * rng.normal(size=(tpl.n, 3))
sys1 = System([tpl.pgm])
fl = ChargeFlux.from_templates(sys1, [tpl])
print(fl.describe())
Q = sys1.expand()
model, P = tpl.model, jax.tree_util.tree_map(jnp.asarray, tpl.P)
q_md, c_md = fl.charges(jnp.asarray(x), jnp.eye(3) * 10.0, Q["q"], Q["cov"], fl.theta())
q_b, c_b = model._flux(tpl.index, jnp.asarray(x), P, Q["q"], Q["cov"])
print(
    f"1a charges vs BondedModel._flux: max |dq| {float(jnp.abs(q_md - q_b).max()):.1e} e, max |dc| "
    f"{float(jnp.abs(c_md - c_b).max()):.1e} e nm  (flux shifts: q {float(jnp.abs(q_md - Q['q']).max()):.3g} e, "
    f"c {float(jnp.abs(c_md - Q['cov']).max()):.3g} e nm)"
)
e_nb, dip_b, st = model.nonbonded(tpl.index, jnp.asarray(x), P, state=True)
sysx = System([molecule_at(tpl, x)])
d = model.nb[tpl.index]
Rr = Q["radius"]
e_lj = model._vdw(jnp.asarray(x), Q, d["lj"], sys1.pair_i, sys1.pair_j, model._bij(Rr[sys1.pair_i], Rr[sys1.pair_j]))
out, aux = ElecChannel().energy(jnp.asarray(x), sysx)
e_gas = out["perm"] + out["ind"]
print(
    f"1b gas phase: BondedModel pGM {float(e_nb - e_lj):.10f} kJ/mol, Model(ElecChannel) at the flux charges "
    f"{float(e_gas):.10f}: rel. diff {abs(float(e_nb - e_lj - e_gas)) / abs(float(e_gas)):.1e}; induced dipoles "
    f"max diff {float(jnp.abs(aux['mu'] - st['mu']).max()):.1e} e nm"
)
s1 = MDSettings(precision="double", dipole_tol=1e-12, max_iter=500, peek=0.0, cutoff=2.4, skin=0.05, lj_lrc=False)
L = 5.0
sim_f = FlexibleSimulation(sys1, [tpl], x + L / 2, np.eye(3) * L, s1, thermostat=None, log=None)
Hb = jnp.eye(3) * L
xb = sim_f.flex.pos0
idx = sim_f.ff.rows_for(xb, Hb)
ff_f = PGMForceField(sys1, Hb, s1, topology=sim_f.topology, flux=sim_f.ff.flux)
ff_0 = PGMForceField(sysx, Hb, s1, topology=sim_f.topology)
r_f = jax.jit(ff_f.compute)(xb, Hb, idx, ff_f.init_induction())
r_0 = jax.jit(ff_0.compute)(xb, Hb, idx, ff_0.init_induction())
print(
    f"1c MD engine (one molecule, {L} nm box): flux {float(r_f.energy['elec']):.10f} kJ/mol vs no flux at the "
    f"same charges {float(r_0.energy['elec']):.10f}: rel. diff "
    f"{abs(float(r_f.energy['elec'] - r_0.energy['elec'])) / abs(float(r_0.energy['elec'])):.1e}; induced "
    f"dipoles {rel(r_f.induction.mu, r_0.induction.mu):.1e} rel.; vs gas phase "
    f"{float(r_f.energy['elec'] - e_gas):+.2e} "
    f"kJ/mol (periodic images); flux forces max {float(jnp.abs(r_f.forces - r_0.forces).max()):.1f} kJ/mol/nm"
)
F = np.asarray(sim_f.state.dyn.force)
g = np.asarray(jax.grad(lambda R: model.energy(tpl.index, R, P)[0])(jnp.asarray(x)))
P0 = dict(P, flux={k: 0.0 * v for k, v in P["flux"].items()})
g0 = np.asarray(jax.grad(lambda R: model.energy(tpl.index, R, P0)[0])(jnp.asarray(x)))
tpl0 = FlexibleTemplate(
    tpl.specs, tpl.settings, dict(tpl.P, flux={k: 0.0 * v for k, v in tpl.P["flux"].items()}), tpl.index
)
F0 = np.asarray(
    FlexibleSimulation(sys1, [tpl0], x + L / 2, np.eye(3) * L, s1, thermostat=None, log=None).state.dyn.force
)
print(
    f"1d MD forces vs gradient of the gas-phase model: max |F + g| {np.abs(F + g).max():.2e} kJ/mol/nm "
    f"(RMS force {np.sqrt(np.mean(g**2)):.0f}; flux part of the force up to {np.abs(g - g0).max():.0f}); "
    f"the same template with the flux parameters zeroed: {np.abs(F0 + g0).max():.2e} (periodic images, PME)"
)

# ------------------------------------------------------------------ 2. periodic box
n = 32
pos, H = liquid_box(tpl, n, 0.55, seed=0, min_dist=0.18)
pos = pos + 0.004 * rng.normal(size=pos.shape)
sysn = System([tpl.pgm] * n)
s = MDSettings(
    cutoff=0.6,
    skin=0.05,
    ewald_beta=6.0,
    pme_grid=(32, 32, 32),
    pme_order=8,
    lj_lrc=False,
    dipole_tol=1e-12,
    max_iter=500,
    peek=0.0,
    precision="double",
)
sim = FlexibleSimulation(sysn, [tpl] * n, pos, H, s, thermostat=None, log=None)
pos = sim.flex.pos0
H = jnp.asarray(H)
ff = PGMForceField(sysn, H, s, topology=sim.topology, flux=sim.ff.flux)
idx = ff.rows_for(pos, H)
res = jax.jit(ff.compute)(pos, H, idx, ff.init_induction())
mu = res.induction.mu
db = np.asarray(ff.flux.deviations(pos, H))
print(
    f"\n2. {n} flexible {tpl.name} in a {float(H[0, 0]):.3f} nm box, bonds off reference by up to "
    f"{np.abs(db).max():.4f} nm"
)
Pb = ff._atoms(None)
F_ad = -jax.grad(lambda y: ff.energy_fixed_mu(y, H, mu, idx, Pb)[0])(pos)
print(f"2a forces vs autodiff of E(R, q(R), c(R), mu fixed): max rel. diff {rel(res.forces, F_ad):.1e}")
ej = jax.jit(lambda y, h: ff.energy(y, h, idx, res.induction)[0])
h = 1e-5
errs = []
for a in rng.choice(sysn.n, 8, replace=False):
    for k in range(3):
        dd = np.zeros(pos.shape)
        dd[a, k] = h
        fd = -(float(ej(pos + dd, H)) - float(ej(pos - dd, H))) / (2 * h)
        errs.append(abs(fd - float(res.forces[a, k])) / max(1.0, abs(fd)))
Fflux = res.forces - PGMForceField(sysn, H, s, topology=sim.topology).compute(pos, H, idx, ff.init_induction()).forces
print(
    f"2b forces vs central differences (24 components, dipoles re-solved, h {h:g} nm): max rel. error "
    f"{max(errs):.1e}  (RMS force {float(jnp.sqrt(jnp.mean(res.forces**2))):.0f} kJ/mol/nm)"
)
m = np.asarray(sysn.masses)
mol = np.asarray(sysn.mol)
com = np.array([np.average(np.asarray(pos)[mol == k], 0, weights=m[mol == k]) for k in range(n)])
W = ff.strain_derivative(pos, H, idx, mu)
e_s = jax.jit(lambda t: ff.energy(pos + (t * com)[mol], H * (1 + t), idx, res.induction)[0])
fd = (float(e_s(1e-6)) - float(e_s(-1e-6))) / 2e-6
print(
    f"2c molecular strain derivative: tr W {float(jnp.trace(W)):.8f}, FD of E(box, centres scaled) {fd:.8f} kJ/mol: "
    f"rel. {abs(fd - float(jnp.trace(W))) / abs(fd):.1e}"
)
Wa = ff.strain_derivative(pos, H, idx, mu, molecular=False)
ea = jax.jit(lambda e: ff.energy(pos @ (jnp.eye(3) + e).T, H @ (jnp.eye(3) + e).T, idx, res.induction)[0])
ws = []
for a, b in ((0, 0), (1, 1), (2, 2), (1, 0), (2, 1), (0, 2)):
    E = np.zeros((3, 3))
    E[a, b] = 1e-6
    fd = (float(ea(E)) - float(ea(-E))) / 2e-6
    ws.append(abs(fd - float(Wa[a, b])) / max(1.0, abs(fd)))
print(
    f"   atomic strain derivative (bonds stretched, flux active): 6 components vs FD, max rel. {max(ws):.1e}; "
    f"tr W_atomic - tr W_molecular {float(jnp.trace(Wa) - jnp.trace(W)):.3f} kJ/mol"
)

sd = replace(s, differentiable=True, adjoint_tol=1e-12)
ffd = PGMForceField(sysn, H, sd, topology=sim.topology, flux=sim.ff.flux)
wF, wmu = rng.normal(size=pos.shape), rng.normal(size=pos.shape)


def loss(theta, y, Hx):
    r = ffd.compute(y, Hx, idx, ffd.init_induction(), theta)
    return jnp.sum(wF * r.forces) + 1e3 * jnp.sum(wmu * r.induction.mu)


theta0 = {**sysn.params0, "flux": {k: jnp.asarray(v) for k, v in ffd.flux.params.items()}}
Lj = jax.jit(loss)
t0 = time.time()
g_th, g_x, g_H = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(theta0, pos, H)
print(f"2d differentiable path (gradient of sum w.F + 1e3 sum w'.mu; {time.time() - t0:.0f} s incl. compile):")
leaves, tree = jax.tree_util.tree_flatten(theta0)
for label, only_flux in (("all parameters", False), ("jb, jc, jc2 only", True)):
    for trial in range(2):
        v = [rng.normal(size=np.shape(l)) * np.maximum(np.abs(np.asarray(l)), 1e-3) for l in leaves]
        vt = jax.tree_util.tree_unflatten(tree, [jnp.asarray(a) for a in v])
        if only_flux:
            vt = {k: (vv if k == "flux" else jax.tree_util.tree_map(jnp.zeros_like, vv)) for k, vv in vt.items()}
        hh = 1e-6
        fp = float(Lj(jax.tree_util.tree_map(lambda a, b: a + hh * b, theta0, vt), pos, H))
        fm = float(Lj(jax.tree_util.tree_map(lambda a, b: a - hh * b, theta0, vt), pos, H))
        fd = (fp - fm) / (2 * hh)
        ad = sum(float(jnp.sum(a * b)) for a, b in zip(jax.tree_util.tree_leaves(g_th), jax.tree_util.tree_leaves(vt)))
        print(
            f"    {label:18s} direction {trial}: AD {ad:+.10e}  FD {fd:+.10e}  rel. "
            f"{abs(fd - ad) / max(1.0, abs(fd)):.1e}"
        )
for trial in range(2):
    dx = rng.normal(size=pos.shape) * 1e-3
    hh = 1e-4
    fd = (float(Lj(theta0, pos + hh * dx, H)) - float(Lj(theta0, pos - hh * dx, H))) / (2 * hh)
    ad = float(jnp.sum(g_x * dx))
    print(
        f"    positions          direction {trial}: AD {ad:+.10e}  FD {fd:+.10e}  rel. "
        f"{abs(fd - ad) / max(1.0, abs(fd)):.1e}"
    )
dH = np.tril(rng.normal(size=(3, 3))) * 1e-3
hh = 1e-4
fd = (float(Lj(theta0, pos, H + hh * dH)) - float(Lj(theta0, pos, H - hh * dH))) / (2 * hh)
ad = float(jnp.sum(g_H * dH))
print(f"    box                direction 0: AD {ad:+.10e}  FD {fd:+.10e}  rel. {abs(fd - ad) / max(1.0, abs(fd)):.1e}")
Ej = jax.jit(lambda th: ffd.compute(pos, H, idx, ffd.init_induction(), th).energy["total"])
gE = jax.grad(Ej)(theta0)["flux"]
worst = 0.0
for k, v in theta0["flux"].items():
    for j in range(len(v)):
        hh = 1e-4  # E ~ 3e4 kJ/mol: rounding would dominate a smaller step
        up = {**theta0, "flux": {**theta0["flux"], k: v.at[j].add(hh)}}
        dn = {**theta0, "flux": {**theta0["flux"], k: v.at[j].add(-hh)}}
        fd = (float(Ej(up)) - float(Ej(dn))) / (2 * hh)
        worst = max(worst, abs(fd - float(gE[k][j])) / max(1.0, abs(fd)))
print(
    f"2e dE/d(jb, jc, jc2) ({sum(len(v) for v in theta0['flux'].values())} parameters) vs central differences: max "
    f"rel. {worst:.1e}; "
    f"dE/djb = {np.round(np.asarray(gE['jb']), 2).tolist()} kJ/mol per e/nm"
)
