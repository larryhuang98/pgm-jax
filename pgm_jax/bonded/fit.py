"""Fitting bonded parameters to QM energies, forces (and dipoles) of sampled frames.

Loss per molecule (the paper's objective, both terms in kcal units):
    L = w_E mean_k (dE_k - <dE>)^2 / s_E^2 + w_F mean_{k,atoms,xyz} (F_QM - F)^2 / s_F^2
        [+ w_mu mean_k |mu_QM - mu|^2 / s_mu^2] + l2 |z|^2 + l1 |theta_linear|_1
with dE = E_QM - E_model and the per-molecule mean removed (the free energy offset);
s_E = 1 kcal/mol, s_F = 1 kcal/mol/A, s_mu = 0.1 D.  Parameters are optimised in scaled
units z = (theta - theta_0) / scale by L-BFGS (scipy) with JAX gradients; the pGM and LJ part
is fixed (precomputed per frame) unless charge flux or fitted charges are on; learned pair scales (escale) enter
linearly through precomputed per-class pair energies.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

KCAL = 4.184                  # kJ/mol
EH = 2625.4996394799          # kJ/mol
BOHR_NM = 0.052917721067
S_E, S_F, S_MU = KCAL, KCAL * 10.0, 0.1 * 0.020819434   # kJ/mol, kJ/mol/nm, e nm (0.1 D)
S_ESP = 0.002                                              # hartree/e

SCALES = {"q": 0.05, "c": 0.002, "t": 0.05, "dc": 0.002, "b0": 0.005, "th0": 0.03, "Kb": 2e4, "Ka": 50.0, "Ka3": 50.0, "K": 20.0, "r0": 0.01,
          "A": 10.0, "B": 5.0, "jb": 0.5, "jc": 0.05,
          "l_mid": 5.0, "l_end": 5.0, "l_ang": 1.0,
          "C": 20.0, "lm": 0.3, "k": 100.0, "lkap": 1.0, "D": 1.0, "jc2": 0.5, "cm": 2.0}


@dataclass
class FrameSet:
    X: np.ndarray           # (k, n, 3) nm
    E: np.ndarray           # (k,) kJ/mol
    F: np.ndarray           # (k, n, 3) kJ/mol/nm
    mu: np.ndarray | None = None   # (k, 3) e nm
    extra: dict | None = None

    def __len__(self):
        return len(self.E)

    def subset(self, idx):
        idx = np.asarray(idx)
        return FrameSet(self.X[idx], self.E[idx], self.F[idx], None if self.mu is None else self.mu[idx],
                        {k: np.asarray(v)[idx] for k, v in (self.extra or {}).items()})


class Fitter:
    def __init__(self, model, data: dict, w_E=1.0, w_F=1.0, w_mu=0.0, l2=1e-4, l2_elec=None, esp=None, w_esp=0.0):
        """data: {mol index: {split: FrameSet}}.  l2_elec: ridge weight on the scaled changes of
        fitted pGM charges / covalent dipoles (prior = the ESP-fitted values; default l2)."""
        self.model, self.data = model, data
        self.w = (w_E, w_F, w_mu)
        self.l2 = l2
        self.l2_elec = l2 if l2_elec is None else l2_elec
        self.esp = esp or {}                     # {mol: (R nm, grid nm, V hartree/e)}: ESP restraint
        self.w_esp = w_esp
        self._nb_cache = {}

    # ------------------------------------------------------------------ nonbonded (fixed)
    def nonbonded(self, m, split):
        key = (m, split)
        if key not in self._nb_cache:
            fs = self.data[m][split]
            f = jax.jit(jax.vmap(lambda X: jax.value_and_grad(lambda X: self.model.nonbonded(m, X)[0])(X)))
            d = jax.jit(jax.vmap(lambda X: self.model.nonbonded(m, X)[1]))
            e, g = f(jnp.asarray(fs.X))
            out = (np.asarray(e), -np.asarray(g), np.asarray(d(jnp.asarray(fs.X))))
            if self.model.s.escale:                  # per-class permanent pair energies and their gradients
                t = jax.jit(jax.vmap(lambda X: self.model.escale_terms(m, X)))
                dt = jax.jit(jax.vmap(jax.jacrev(lambda X: self.model.escale_terms(m, X))))
                out = out + (np.asarray(t(jnp.asarray(fs.X))), np.asarray(dt(jnp.asarray(fs.X))))
            self._nb_cache[key] = out
        return self._nb_cache[key]

    def prepare(self, split):
        """Precompute the fixed pGM + LJ part of every frame (outside any trace)."""
        if not self.model.nb_dynamic:
            for m in self.data:
                if split in self.data[m]:
                    self.nonbonded(m, split)

    def _predict(self, m, P, X, nb=None):
        """Energies, forces (and dipoles) of frames X for molecule m."""
        model = self.model
        if model.nb_dynamic:
            def one(X):
                (e, dip), g = jax.value_and_grad(lambda X: model.energy(m, X, P), has_aux=True)(X)
                return e, -g, dip
            return jax.vmap(one)(X)
        eb, gb = jax.vmap(jax.value_and_grad(lambda X: model.bonded_energy(m, X, P)))(X)
        e_nb, f_nb, d_nb = nb[:3]
        if len(nb) > 3 and "escale" in P:
            kap = P["escale"]["kappa"][model.nb[m]["es_glob"]]
            e_nb = e_nb + nb[3] @ kap
            f_nb = f_nb - jnp.einsum("fcna,c->fna", nb[4], kap)
        return eb + e_nb, f_nb - gb, d_nb

    def _terms(self, P, split="train"):
        out = {}
        for m in self.data:
            if split not in self.data[m]:
                continue
            fs = self.data[m][split]
            nb = None if self.model.nb_dynamic else self.nonbonded(m, split)
            E, F, D = self._predict(m, P, jnp.asarray(fs.X), nb)
            dE = jnp.asarray(fs.E) - E
            lE = jnp.mean((dE - jnp.mean(dE)) ** 2) / S_E ** 2
            lF = jnp.mean((jnp.asarray(fs.F) - F) ** 2) / S_F ** 2
            lmu = jnp.mean(jnp.sum((jnp.asarray(fs.mu) - D) ** 2, -1)) / S_MU ** 2 if fs.mu is not None else 0.0
            out[m] = (lE, lF, lmu)
        return out

    def esp_rmse(self, m, P):
        """(RMSE hartree/e, relative RMSE) of the model ESP against the QM ESP of molecule m."""
        R, G, V = self.esp[m]
        d = self.model.esp(m, jnp.asarray(R), jnp.asarray(G), P) - V
        msd = jnp.mean(d ** 2)
        return jnp.sqrt(msd), jnp.sqrt(msd / jnp.mean(jnp.asarray(V) ** 2))

    def loss(self, P, split="train"):
        t = self._terms(P, split)
        w_E, w_F, w_mu = self.w
        L = sum(w_E * a + w_F * b + w_mu * c for a, b, c in t.values()) / max(len(t), 1)
        if self.w_esp and split == "train":
            ms = [m for m in t if m in self.esp]
            if ms:          # (ESP RMSE / 2 mhartree/e)^2; the py_resp pGM fits reach 0.4-2 mhartree/e
                L = L + self.w_esp * sum(self.esp_rmse(m, P)[0] ** 2 for m in ms) / S_ESP ** 2 / len(ms)
        nnb = getattr(self.model, "nnb", None)
        if nnb is not None and split == "train" and "coef" not in P["nnb"]:
            L = L + nnb.penalty(P["nnb"], list(t))        # shrinks the network residual to the typed table
        return L

    # ------------------------------------------------------------------ optimisation
    def fit(self, P0, maxiter=2000, frozen=(), l1=0.0, mask=None, verbose=True, tol=1e-10,
            adam_steps: int = 0, lr: float = 3e-3):
        """L-BFGS on all parameters except the families/names in `frozen` ("ref", "Kb", ...).
        `mask`: pytree of 0/1 (1 = free), e.g. to switch off terms for a Lasso path.
        adam_steps > 0: first that many Adam steps (learning rate lr, cosine decay) in the same
        scaled variables, then L-BFGS (neural bonded terms: the network is not well conditioned
        for L-BFGS from its initial point)."""
        from scipy.optimize import minimize
        self.prepare("train")
        scale = jax.tree_util.tree_map_with_path(
            lambda path, v: jnp.full(jnp.shape(v), SCALES.get(path[-1].key, 1.0)), P0)
        free = jax.tree_util.tree_map_with_path(
            lambda path, v: jnp.full(jnp.shape(v), 0.0 if (path[0].key in frozen or path[-1].key in frozen) else 1.0), P0)
        if mask is not None:
            free = jax.tree_util.tree_map(lambda a, b: a * b, free, mask)
        lin = self.model.linear_mask(P0)
        z0, unravel = ravel_pytree(jax.tree_util.tree_map(jnp.zeros_like, P0))
        sc, _ = ravel_pytree(scale)
        fr, _ = ravel_pytree(free)
        ln, _ = ravel_pytree(jax.tree_util.tree_map(lambda v, l: jnp.full(jnp.shape(v), 1.0 if l else 0.0), P0, lin))
        p0, _ = ravel_pytree(P0)
        l2v, _ = ravel_pytree(jax.tree_util.tree_map_with_path(
            lambda path, v: jnp.full(jnp.shape(v), self.l2_elec if path[0].key in ("elec", "bci") else self.l2), P0))

        def theta(z):
            return unravel(p0 + fr * sc * z)

        def obj(z):
            th = p0 + fr * sc * z
            L = self.loss(unravel(th)) + jnp.sum(l2v * (fr * z) ** 2)
            if l1:
                L = L + l1 * jnp.sum(jnp.sqrt((ln * fr * th / sc) ** 2 + 1e-8))
            return L

        vg = jax.jit(jax.value_and_grad(obj))
        t0 = time.time()
        z_start = jnp.zeros_like(z0)
        if adam_steps > 0:
            @jax.jit
            def step(carry, k):
                z, m, v = carry
                L, g = jax.value_and_grad(obj)(z)
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * g * g
                kk = k + 1.0
                eta = lr * 0.5 * (1.0 + jnp.cos(jnp.pi * k / adam_steps))
                z = z - eta * (m / (1 - 0.9 ** kk)) / (jnp.sqrt(v / (1 - 0.999 ** kk)) + 1e-8)
                return (z, m, v), L
            carry = (z_start, jnp.zeros_like(z0), jnp.zeros_like(z0))
            chunk = 100
            for c0 in range(0, adam_steps, chunk):
                ks = jnp.arange(c0, min(c0 + chunk, adam_steps), dtype=float)
                carry, Ls = jax.lax.scan(step, carry, ks)
            z_start = carry[0]
            if verbose:
                print(f"    adam: {adam_steps} steps, loss {float(Ls[-1]):.4g} ({time.time() - t0:.1f} s)", flush=True)
        f = lambda z: tuple(np.asarray(v, float) for v in vg(jnp.asarray(z)))
        res = minimize(f, np.asarray(z_start), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxfun": maxiter * 2, "ftol": tol, "gtol": 1e-8})
        if verbose:
            print(f"    fit: loss {float(obj(jnp.zeros_like(z0))):.4g} -> {res.fun:.4g} in {res.nit} it, {time.time() - t0:.1f} s ({res.message})", flush=True)
        return theta(jnp.asarray(res.x))

    # ------------------------------------------------------------------ metrics
    def metrics(self, P, split="test"):
        """Per molecule: energy MAE (offset removed), mean force-error norm per atom (kcal/mol/A),
        dipole RMSE (D), in the paper's units."""
        self.prepare(split)
        out = {}
        for m in self.data:
            if split not in self.data[m]:
                continue
            fs = self.data[m][split]
            nb = None if self.model.nb_dynamic else self.nonbonded(m, split)
            E, F, D = (np.asarray(v) for v in self._predict(m, P, jnp.asarray(fs.X), nb))
            dE = fs.E - E
            dF = np.linalg.norm(fs.F - F, axis=-1)
            r = {"E_MAE": float(np.mean(np.abs(dE - dE.mean()))) / KCAL,
                 "F_MAE": float(np.mean(dF)) / (KCAL * 10.0), "n": len(fs)}
            if fs.mu is not None:
                r["mu_RMSE_D"] = float(np.sqrt(np.mean(np.sum((fs.mu - D) ** 2, -1)))) / 0.020819434
            if m in self.esp:
                a, b = self.esp_rmse(m, P)
                r["esp_RMSE_mEh"], r["esp_RRMSE"] = 1e3 * float(a), float(b)
            out[m] = r
        return out
