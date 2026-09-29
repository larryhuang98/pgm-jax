"""Iterative multi-target fitting of a rigid-molecule pGM liquid (NPT simulations + ensemble
gradients + Levenberg-Marquardt), with uncertainty quantification.

Each iteration:
  1. NPT MD at theta (md.Simulation), equilibration then production; every `every_ps` a frame
     (positions, box, induced dipoles) is kept on the device and analysed in vmapped chunks by
     FrameAnalyzer (U, dU/dtheta, M, dM/dtheta, alpha_cell, D, V, g(r)).
  2. Observables, their Jacobians (fluctuation formulas) and jackknife errors (Objective.estimate).
  3. Check of the previous iteration's predictions against the new measurement; trust radius update.
  4. LM step in the trust region; predictions of the next iteration: linear, linear-exponential
     reweighting of this run's frames (with n_eff), optionally exact reweighting (frames re-evaluated
     at the new parameters); parameter covariance (sampling) and its propagation; block bootstrap.
  5. JSON record (prefix.json) and the MD state (prefix_state.npz) for resuming.

The frames need no neighbour-list state (FrameAnalyzer builds its own rows), so the analysis costs
the same whatever the MD engine does between frames."""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time

import jax.numpy as jnp
import numpy as np

from ..md.forcefield import MDSettings
from ..md.simulation import Simulation
from .estimators import LiquidSamples
from .frames import FrameAnalyzer


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        return _jsonable(np.asarray(x).tolist())
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


class LiquidFit:
    def __init__(self, sys_, pos, H, space, objective, T: float = 298.0, pressure: float = 1.0,
                 settings: MDSettings = MDSettings(), dt: float = 0.002, thermostat: str = "bussi",
                 tau_t: float = 1.0, gamma: float = 1.0, barostat_interval: int = 100,
                 equil_ps: float = 20.0, prod_ps: float = 200.0, every_ps: float = 0.5, rdf=None,
                 chunk: int = 8, tol: float = 1e-6, nblocks: int = 10, radius: float = 1.0,
                 radius_max: float = 4.0, prefix: str = "fit", exact_every: int = 0, bootstrap: int = 200,
                 log=sys.stdout, seed: int = 0, fixed: bool = False, save_frames: bool = True,
                 ensemble: str = "npt", replicas: int = 1, equil_rep_ps: float = 20.0):
        if settings.pme_grid is None:
            from ..md.pme import grid_size
            settings = dataclasses.replace(settings, pme_grid=tuple(int(k) for k in grid_size(H, settings.pme_spacing)))
        self.sys, self.space, self.obj = sys_, space, objective
        self.pos, self.H, self.vel = np.asarray(pos, float), np.asarray(H, float), None
        self.T, self.p, self.settings, self.dt = float(T), float(pressure), settings, float(dt)
        self.md_kw = dict(thermostat=thermostat, tau_t=tau_t, gamma=gamma, barostat_interval=barostat_interval)
        self.equil_ps, self.prod_ps, self.every_ps = float(equil_ps), float(prod_ps), float(every_ps)
        self.nblocks, self.radius, self.radius_max = int(nblocks), float(radius), float(radius_max)
        self.prefix, self.exact_every, self.nboot, self.seed = prefix, int(exact_every), int(bootstrap), int(seed)
        self.log = log
        self.fixed, self.save_frames = bool(fixed), bool(save_frames)
        if ensemble not in ("npt", "nvt"):
            raise ValueError("ensemble: npt or nvt")
        if replicas > 1 and ensemble != "nvt":
            raise ValueError("batched replicas run NVT only (md/remd.MDReplicas)")
        self.ensemble, self.replicas, self.equil_rep_ps = ensemble, int(replicas), float(equil_rep_ps)
        if any(t.name == "rdf" for t in objective.targets) and rdf is None:
            raise ValueError("an rdf target needs the RDFSpec (rdf=)")
        self.analyzer = FrameAnalyzer(sys_, self.H, settings, space, rdf=rdf, tol=tol, chunk=chunk)
        self.records, self.pending = [], None

    def _print(self, s):
        if self.log is not None:
            print(s, file=self.log, flush=True)

    # ------------------------------------------------------------------ sampling
    def simulate(self, theta, seed, equil_ps=None, prod_ps=None, keep_frames=False):
        """One NPT run at theta; returns (frames dict from the analyser, stored frames or None,
        run info).  Updates the stored coordinates, box and velocities."""
        equil_ps = self.equil_ps if equil_ps is None else equil_ps
        prod_ps = self.prod_ps if prod_ps is None else prod_ps
        params = self.space(jnp.asarray(theta, float))
        sim = Simulation(self.sys, self.pos, self.H, self.settings, dt=self.dt, ensemble=self.ensemble, temperature=self.T,
                         pressure=self.p, seed=seed, vel_nm_ps=self.vel, params=params, log=None, **self.md_kw)
        t0 = time.time()
        n_eq = int(round(equil_ps / self.dt))
        every = max(1, int(round(self.every_ps / self.dt)))
        blk = every * max(1, int(round(10.0 / (every * self.dt))))
        while n_eq > 0:
            k = min(blk, n_eq)
            sim._advance(k)
            n_eq -= k
        if self.replicas > 1:
            return self._simulate_replicas(sim, theta, seed, every, prod_ps, keep_frames, t0)
        t1 = time.time()
        nframes = int(round(prod_ps / self.dt)) // every
        pending, results, stored = [], [], [] if keep_frames else None
        dens, ta = [], 0.0
        th = jnp.asarray(theta, float)
        for i in range(nframes):
            sim._advance(every)
            st = sim.state
            fr = (sim.rigid.positions(st.dyn.position), st.box, st.induction.mu)
            pending.append(fr)
            if keep_frames:
                stored.append(tuple(np.asarray(x) for x in fr))
            if len(pending) == self.analyzer.chunk or i == nframes - 1:
                ta0 = time.time()
                results.append(self.analyzer.analyze(th, pending))
                ta += time.time() - ta0
                pending = []
        frames = {k: np.concatenate([r[k] for r in results]) for k in results[0]}
        obs = sim.observables()
        self.pos, self.H, self.vel = sim.positions_nm(), np.asarray(sim.state.box), sim.velocities_nm_ps()
        info = {"equil_s": t1 - t0, "prod_s": time.time() - t1, "analysis_s": ta, "frames": nframes,
                "steps": int(sim.state.step), "cg_mean": obs["cg_mean"], "mc_accept": obs.get("mc_accept"),
                "temp_K": obs["temp_K"], "unconverged_frames": int(np.sum(~frames["converged"])),
                "analysis_ms_per_frame": 1000.0 * ta / max(nframes, 1)}
        return frames, stored, info

    def _simulate_replicas(self, sim, theta, seed, every, prod_ps, keep_frames, t0):
        """NVT: `replicas` copies advanced together (jax.vmap, md/remd.MDReplicas at one temperature),
        each prod_ps long; frames ordered by replica, then time (contiguous blocks never mix replicas
        when nblocks is a multiple of the number of replicas)."""
        from ..md.remd import MDReplicas
        R = self.replicas
        rep = MDReplicas(sim, self.T + 1e-6 * np.arange(R), batched=True, seed=seed + 7)
        rep.advance(int(round(self.equil_rep_ps / self.dt)))
        t1 = time.time()
        n = int(round(prod_ps / self.dt)) // every
        th = jnp.asarray(theta, float)
        out, stored, ta = [], [] if keep_frames else None, 0.0
        for i in range(n):
            rep.advance(every)
            P = rep._positions(rep.S.dyn.position)
            frames = [(P[k], rep.S.box[k], rep.S.induction.mu[k]) for k in range(R)]
            if keep_frames:
                stored.append([tuple(np.asarray(x) for x in f) for f in frames])
            ta0 = time.time()
            out.append(self.analyzer.analyze(th, frames))
            ta += time.time() - ta0
        fr = {k: np.stack([o[k] for o in out], axis=1) for k in out[0]}           # (replica, time, ...)
        fr = {k: v.reshape((-1,) + v.shape[2:]) for k, v in fr.items()}
        if keep_frames:
            stored = [stored[i][k] for k in range(R) for i in range(n)]
        st0 = rep.state(0)
        self.pos = np.asarray(rep.sim.rigid.positions(st0.dyn.position))
        self.H, self.vel = np.asarray(st0.box), None
        info = {"equil_s": t1 - t0, "prod_s": time.time() - t1, "analysis_s": ta, "frames": n * R, "replicas": R,
                "unconverged_frames": int(np.sum(~fr["converged"])), "analysis_ms_per_frame": 1000.0 * ta / max(n * R, 1)}
        return fr, stored, info

    # ------------------------------------------------------------------ one iteration
    def iterate(self, theta, it: int = 0) -> dict:
        theta = np.asarray(theta, float)
        keep = self.exact_every > 0
        equil = 0.0 if (self.fixed and self.records) else None      # fixed theta: segments continue
        frames, stored, info = self.simulate(theta, seed=self.seed + 1000 * it + 1, keep_frames=keep, equil_ps=equil)
        if self.save_frames:
            np.savez(f"{self.prefix}_frames{it:02d}.npz", theta=theta, **{k: v for k, v in frames.items()})
        samples = LiquidSamples(frames, self.T, self.sys.nmol, self.analyzer.mass, self.nblocks)
        est = self.obj.estimate(samples, theta)
        chi2, prior = self.obj.chi2(est.y, est, theta)
        rec = {"iter": it, "theta": theta, "names": self.space.names, "scales": np.exp(theta),
               "estimate": est.as_dict(), "chi2": chi2, "chi2_prior": prior, "info": info}
        # check of the previous prediction, trust region
        if self.pending is not None:
            pv = self.pending
            s = np.sqrt(est.tol ** 2 + np.diag(est.cov_y) + np.asarray(pv["y_pred_err"]) ** 2)
            rec["check"] = {"y_pred": pv["y_pred"], "y_rw": pv.get("y_rw"), "y": est.y,
                            "z_linear": (est.y - np.asarray(pv["y_pred"])) / np.sqrt(np.diag(est.cov_y) + np.asarray(pv["y_pred_err"]) ** 2 + 1e-300)}
            achieved = pv["chi2_prev"] - (chi2 + prior)
            predicted = pv["chi2_prev"] - pv["chi2_pred"]
            ratio = achieved / predicted if predicted > 0 else np.nan
            rec["check"]["ratio"] = ratio
            if not np.isfinite(ratio) or ratio < 0.25:
                self.radius = max(0.25 * self.radius, 0.05)
            elif ratio > 0.75 and pv["at_boundary"]:
                self.radius = min(2.0 * self.radius, self.radius_max)
        st = self.obj.step(est, self.radius)
        if self.fixed:                                             # measurement only: theta stays
            st.update(delta=np.zeros(self.space.n), y_pred=est.y, chi2_pred=(chi2, prior), at_boundary=False)
        d = st["delta"]
        cov = self.obj.covariance(est)
        yrw = np.asarray(self.obj.model(samples, theta, jnp.asarray(d)))
        rec["step"] = {"delta": d, "lambda": st["lambda"], "radius": self.radius, "size": st["size"],
                       "at_boundary": st["at_boundary"], "y_pred": st["y_pred"], "chi2_pred": sum(st["chi2_pred"]),
                       "y_rw": yrw, "n_eff_rw": samples.n_eff(d), "n_frames": samples.F}
        if keep and stored:
            sub = stored[::self.exact_every]
            new = self.analyzer.analyze(theta + d, sub, grad=False)
            base = self.analyzer.analyze(theta, sub, grad=False)
            avg, neff = samples.exact_average(new, base)
            gas = self.obj.gas(jnp.asarray(theta + d)) if self.obj.gas is not None else None
            rec["step"]["y_exact"] = np.asarray(self.obj._assemble(samples, avg, gas))
            rec["step"]["n_eff_exact"] = neff
            rec["step"]["n_exact"] = len(sub)
        rec["uq"] = {"theta_err": cov["theta_err"], "theta_err_posterior": cov["theta_err_posterior"],
                     "C_theta": cov["C_theta"], "G": cov["G"],
                     "y_err_propagated": self.obj.propagate(est, cov["C_theta"]),
                     "y_err_propagated_posterior": self.obj.propagate(est, cov["G"])}
        if self.nboot:
            b = self.obj.bootstrap(samples, est, self.radius, self.nboot, seed=it)
            rec["uq"]["bootstrap_theta_sd"] = b["theta_sd"]
            rec["uq"]["bootstrap_y_sd"] = b["y_sd"]
        # propagated error of the predicted observables: from the sampling error of theta + d
        y_pred_err = np.sqrt(np.clip(np.einsum("mi,ij,mj->m", est.J, cov["C_theta"], est.J), 0, None)
                             + (est.J_err ** 2) @ (d ** 2))              # + the Jacobian's own noise times the step
        self.pending = {"y_pred": st["y_pred"], "y_pred_err": y_pred_err, "y_rw": yrw, "chi2_prev": chi2 + prior,
                        "chi2_pred": sum(st["chi2_pred"]), "at_boundary": st["at_boundary"]}
        rec["next_theta"] = theta + d
        rec["radius_next"] = self.radius
        self.records.append(rec)
        self._report(rec, est)
        return rec

    def _report(self, rec, est):
        it = rec["iter"]
        self._print(f"== iter {it}: {self.space.describe(rec['theta'])}; chi2 {rec['chi2']:.3f} + prior {rec['chi2_prior']:.3f}; "
                    f"{rec['info']['frames']} frames, MD {rec['info']['prod_s'] + rec['info']['equil_s']:.0f} s, "
                    f"analysis {rec['info']['analysis_s']:.0f} s ({rec['info']['analysis_ms_per_frame']:.0f} ms/frame)")
        chk = rec.get("check")
        for i, nm in enumerate(est.names):
            if nm.startswith("rdf(") and i % 10:
                continue
            t = est.target[i]
            line = f"   {nm:20s} {est.y[i]:12.5f} +- {est.err[i]:9.5f}"
            line += f"   target {t:10.5f}" if np.isfinite(t) else " " * 20
            if chk is not None:
                line += f"   predicted {chk['y_pred'][i]:10.5f} (z {chk['z_linear'][i]:+.2f})"
            line += f"   next {rec['step']['y_pred'][i]:10.5f}"
            self._print(line)
        s = rec["step"]
        self._print(f"   step {np.round(s['delta'], 5).tolist()} (radius {s['radius']:.3g}, lambda {s['lambda']:.3g}); "
                    f"n_eff(linear reweighting) {s['n_eff_rw']:.1f}/{s['n_frames']}"
                    + (f", exact {s['n_eff_exact']:.1f}/{s['n_exact']}" if "n_eff_exact" in s else "")
                    + (f"; trust ratio {chk['ratio']:.2f}" if chk is not None else ""))
        self._print(f"   theta error (sampling) {np.round(rec['uq']['theta_err'], 5).tolist()}"
                    + (f", bootstrap {np.round(rec['uq']['bootstrap_theta_sd'], 5).tolist()}" if "bootstrap_theta_sd" in rec["uq"] else ""))

    # ------------------------------------------------------------------ driver
    def save(self):
        out = {"names": self.space.names, "T": self.T, "targets": [dataclasses.asdict(t) for t in self.obj.targets],
               "records": self.records, "pending": self.pending, "radius": self.radius}
        tmp = self.prefix + ".json.tmp"
        with open(tmp, "w") as fh:
            json.dump(_jsonable(out), fh, indent=1)
        os.replace(tmp, self.prefix + ".json")
        np.savez(self.prefix + "_state.npz", pos=self.pos, H=self.H, vel=self.vel if self.vel is not None else np.zeros(0))

    def resume(self) -> np.ndarray | None:
        """Continue from prefix.json / prefix_state.npz: returns the next theta (None: fresh)."""
        if not os.path.exists(self.prefix + ".json"):
            return None
        d = json.load(open(self.prefix + ".json"))
        self.records = d["records"]
        self.pending = d["pending"]
        if self.pending is not None:
            self.pending = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in self.pending.items()}
        self.radius = d["radius"]
        s = np.load(self.prefix + "_state.npz")
        self.pos, self.H = s["pos"], s["H"]
        self.vel = s["vel"] if s["vel"].size else None
        return np.asarray(self.records[-1]["next_theta"], float) if self.records else None

    def run(self, theta0, iters: int, resume: bool = True, max_seconds: float | None = None):
        theta = np.asarray(theta0, float)
        start = 0
        if resume:
            t = self.resume()
            if t is not None:
                theta, start = t, len(self.records)
                self._print(f"# resumed at iteration {start}: {self.space.describe(theta)}")
        t0 = time.time()
        for it in range(start, iters):
            rec = self.iterate(theta, it)
            self.save()
            theta = np.asarray(rec["next_theta"], float)
            if max_seconds is not None and it + 1 < iters:
                per = (time.time() - t0) / (it + 1 - start)
                if time.time() - t0 + per > max_seconds:
                    self._print(f"# stopping after iteration {it} (time budget); resume with the same prefix")
                    break
        return theta
