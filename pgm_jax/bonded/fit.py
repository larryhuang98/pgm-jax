"""Fit bonded parameters to QM energies, forces (and dipoles) of sampled frames.

Contents: `FrameSet` (labelled frames of one molecule), `Fitter` (loss, L-BFGS / Adam fit,
metrics), the loss scales S_E, S_F, S_MU, S_ESP and the parameter scales `SCALES`.

Loss per molecule (the paper's objective, both terms in kcal units):

    L = w_E mean_k (dE_k - <dE>)^2 / s_E^2 + w_F mean_{k,atoms,xyz} (F_QM - F)^2 / s_F^2
        [+ w_mu mean_k |mu_QM - mu|^2 / s_mu^2] + l2 |z|^2 + l1 |theta_linear|_1

with dE = E_QM - E_model and the per-molecule mean removed (the free energy offset); s_E = 1
kcal/mol, s_F = 1 kcal/mol/A, s_mu = 0.1 D.  The molecule losses are averaged; an optional ESP
restraint adds w_esp (ESP RMSE / 2 mhartree/e)^2.  Parameters are optimised in scaled units
z = (theta - theta_0) / scale by L-BFGS (scipy) with JAX gradients; the pGM and LJ part is fixed
(precomputed per frame) unless charge flux or fitted charges are on; learned pair scales
(escale) enter linearly through precomputed per-class pair energies.

    fit = Fitter(model, {0: {"train": frames("methanol", "train500"), "test": frames("methanol", "test298")}})
    P = fit.fit(model.init_params())
    fit.metrics(P, "test")

Units: positions nm, energies kJ/mol, forces kJ/mol/nm, dipoles e nm (library units); metrics
in kcal/mol, kcal/mol/A and D (the paper's).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from ..units import DEBYE_E_NM, KCAL

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .model import BondedModel

S_E, S_F, S_MU = KCAL, KCAL * 10.0, 0.1 * DEBYE_E_NM  # kJ/mol, kJ/mol/nm, e nm (0.1 D)
S_ESP = 0.002  # hartree/e

# typical size of each parameter (by leaf name): the optimiser works in z = (theta - theta_0) / scale
SCALES = {
    "q": 0.05,
    "c": 0.002,
    "t": 0.05,
    "dc": 0.002,
    "b0": 0.005,
    "th0": 0.03,
    "Kb": 2e4,
    "Ka": 50.0,
    "Ka3": 50.0,
    "K": 20.0,
    "r0": 0.01,
    "A": 10.0,
    "B": 5.0,
    "jb": 0.5,
    "jc": 0.05,
    "l_mid": 5.0,
    "l_end": 5.0,
    "l_ang": 1.0,
    "C": 20.0,
    "lm": 0.3,
    "k": 100.0,
    "lkap": 1.0,
    "D": 1.0,
    "jc2": 0.5,
    "cm": 2.0,
}


@dataclass
class FrameSet:
    """Labelled frames of one molecule (a dataclass of host arrays).

    Parameters
    ----------
    X : np.ndarray (k, n, 3)
        Positions [nm].
    E : np.ndarray (k,)
        Energies [kJ/mol] (any offset).
    F : np.ndarray (k, n, 3)
        Forces [kJ/mol/nm].
    mu : np.ndarray (k, 3) or None
        Dipoles [e nm]; None: no dipole labels.
    extra : dict or None
        Per-frame metadata (arrays with a leading axis k: "index", "src", scan "angle", ...).
    """

    X: np.ndarray  # (k, n, 3) nm
    E: np.ndarray  # (k,) kJ/mol
    F: np.ndarray  # (k, n, 3) kJ/mol/nm
    mu: np.ndarray | None = None  # (k, 3) e nm
    extra: dict | None = None

    def __len__(self) -> int:
        """Return the number of frames."""
        return len(self.E)

    def subset(self, idx: ArrayLike) -> FrameSet:
        """Return the frames `idx` (index array or mask), extras included."""
        idx = np.asarray(idx)
        return FrameSet(
            self.X[idx],
            self.E[idx],
            self.F[idx],
            None if self.mu is None else self.mu[idx],
            {k: np.asarray(v)[idx] for k, v in (self.extra or {}).items()},
        )


class Fitter:
    """Loss, optimisation and metrics of a BondedModel on labelled frames (see the module docstring).

    Attributes
    ----------
    model : BondedModel
        The model.
    data : dict
        {molecule index: {split name: FrameSet}}.
    w : tuple of float
        (w_E, w_F, w_mu).
    l2, l2_elec : float
        Ridge weights on the scaled changes z (l2_elec for fitted charges / bond-charge increments).
    esp : dict
        {molecule index: (R [nm], grid [nm], V [hartree/e])}: ESP restraint data.
    w_esp : float
        Weight of the ESP restraint.
    """

    def __init__(
        self,
        model: BondedModel,
        data: dict,
        w_E: float = 1.0,
        w_F: float = 1.0,
        w_mu: float = 0.0,
        l2: float = 1e-4,
        l2_elec: float | None = None,
        esp: dict | None = None,
        w_esp: float = 0.0,
    ) -> None:
        """Set up the fitter.

        Parameters
        ----------
        model : BondedModel
            The model.
        data : dict
            {mol index: {split: FrameSet}}; the fit uses split "train".
        w_E, w_F, w_mu : float
            Weights of the energy, force and dipole terms.
        l2 : float
            Ridge weight on the scaled parameter changes z.
        l2_elec : float, optional
            Ridge weight on the scaled changes of fitted pGM charges / covalent dipoles (prior = the
            ESP-fitted values); None: l2.
        esp : dict, optional
            {mol index: (R nm, grid nm, V hartree/e)}: QM ESP for the restraint; None: none.
        w_esp : float
            Weight of the ESP restraint (0: off).
        """
        self.model, self.data = model, data
        self.w = (w_E, w_F, w_mu)
        self.l2 = l2
        self.l2_elec = l2 if l2_elec is None else l2_elec
        self.esp = esp or {}  # {mol: (R nm, grid nm, V hartree/e)}: ESP restraint
        self.w_esp = w_esp
        self._nb_cache = {}

    # ------------------------------------------------------------------ nonbonded (fixed)
    def nonbonded(self, m: int, split: str) -> tuple[np.ndarray, ...]:
        """Return the fixed nonbonded part of every frame of (molecule m, split), cached.

        Returns
        -------
        tuple of np.ndarray
            (energies (k,) [kJ/mol], forces (k, n, 3) [kJ/mol/nm], dipoles (k, 3) [e nm]) and, with
            learned pair scales, the per-class pair energies (k, C) [kJ/mol] and their gradients
            (k, C, n, 3) [kJ/mol/nm].
        """
        key = (m, split)
        if key not in self._nb_cache:
            fs = self.data[m][split]
            f = jax.jit(jax.vmap(lambda X: jax.value_and_grad(lambda X: self.model.nonbonded(m, X)[0])(X)))
            d = jax.jit(jax.vmap(lambda X: self.model.nonbonded(m, X)[1]))
            e, g = f(jnp.asarray(fs.X))
            out = (np.asarray(e), -np.asarray(g), np.asarray(d(jnp.asarray(fs.X))))
            if self.model.s.escale:  # per-class permanent pair energies and their gradients
                t = jax.jit(jax.vmap(lambda X: self.model.escale_terms(m, X)))
                dt = jax.jit(jax.vmap(jax.jacrev(lambda X: self.model.escale_terms(m, X))))
                out = out + (np.asarray(t(jnp.asarray(fs.X))), np.asarray(dt(jnp.asarray(fs.X))))
            self._nb_cache[key] = out
        return self._nb_cache[key]

    def prepare(self, split: str) -> None:
        """Precompute the fixed pGM + LJ part of every frame of `split` (outside any trace; no-op if dynamic)."""
        if not self.model.nb_dynamic:
            for m in self.data:
                if split in self.data[m]:
                    self.nonbonded(m, split)

    def _predict(
        self, m: int, P: dict, X: jax.Array, nb: tuple | None = None
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return the model's energies, forces and dipoles of frames X of molecule m.

        With a dynamic nonbonded part the full model is evaluated (vmapped over frames); otherwise the
        bonded part is added to the cached nonbonded values `nb` (with learned pair scales applied).

        Returns
        -------
        E : jax.Array (k,)
            Energies [kJ/mol].
        F : jax.Array (k, n, 3)
            Forces [kJ/mol/nm].
        mu : jax.Array (k, 3)
            Dipoles [e nm].
        """
        model = self.model
        if model.nb_dynamic:

            def one(X: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
                """Return energy, forces and dipole of one frame."""
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

    def _terms(self, P: dict, split: str = "train") -> dict:
        """Return {molecule: (energy term, force term, dipole term)} of the loss on `split` (dimensionless)."""
        out = {}
        for m in self.data:
            if split not in self.data[m]:
                continue
            fs = self.data[m][split]
            nb = None if self.model.nb_dynamic else self.nonbonded(m, split)
            E, F, D = self._predict(m, P, jnp.asarray(fs.X), nb)
            dE = jnp.asarray(fs.E) - E
            lE = jnp.mean((dE - jnp.mean(dE)) ** 2) / S_E**2
            lF = jnp.mean((jnp.asarray(fs.F) - F) ** 2) / S_F**2
            lmu = jnp.mean(jnp.sum((jnp.asarray(fs.mu) - D) ** 2, -1)) / S_MU**2 if fs.mu is not None else 0.0
            out[m] = (lE, lF, lmu)
        return out

    def esp_rmse(self, m: int, P: dict) -> tuple[jax.Array, jax.Array]:
        """Return the RMSE [hartree/e] and the relative RMSE of the model ESP against the QM ESP of molecule m."""
        R, G, V = self.esp[m]
        d = self.model.esp(m, jnp.asarray(R), jnp.asarray(G), P) - V
        msd = jnp.mean(d**2)
        return jnp.sqrt(msd), jnp.sqrt(msd / jnp.mean(jnp.asarray(V) ** 2))

    def loss(self, P: dict, split: str = "train") -> jax.Array:
        """Return the loss L on `split` (without the l2 / l1 penalties, which `fit` adds).

        On "train" it includes the ESP restraint (w_esp > 0) and the neural bonded residual penalty
        (`NNBonded.penalty`, unless the network is frozen).
        """
        t = self._terms(P, split)
        w_E, w_F, w_mu = self.w
        L = sum(w_E * a + w_F * b + w_mu * c for a, b, c in t.values()) / max(len(t), 1)
        if self.w_esp and split == "train":
            ms = [m for m in t if m in self.esp]
            if ms:  # (ESP RMSE / 2 mhartree/e)^2; the py_resp pGM fits reach 0.4-2 mhartree/e
                L = L + self.w_esp * sum(self.esp_rmse(m, P)[0] ** 2 for m in ms) / S_ESP**2 / len(ms)
        nnb = getattr(self.model, "nnb", None)
        if nnb is not None and split == "train" and "coef" not in P["nnb"]:
            L = L + nnb.penalty(P["nnb"], list(t))  # shrinks the network residual to the typed table
        return L

    # ------------------------------------------------------------------ optimisation
    def fit(
        self,
        P0: dict,
        maxiter: int = 2000,
        frozen: Sequence[str] = (),
        l1: float = 0.0,
        mask: dict | None = None,
        verbose: bool = True,
        tol: float = 1e-10,
        adam_steps: int = 0,
        lr: float = 3e-3,
    ) -> dict:
        """Fit the parameters by L-BFGS (optionally after Adam) in scaled variables; return the fitted P.

        Parameters
        ----------
        P0 : dict
            Initial parameters (theta_0; the ridge prior).
        maxiter : int
            L-BFGS iterations (function evaluations: 2 maxiter).
        frozen : sequence of str
            Families or parameter names kept fixed ("ref", "Kb", ...).
        l1 : float
            Weight of the smooth L1 penalty on the free linear parameters (in units of their scale).
        mask : dict, optional
            Pytree of 0/1 like P0 (1 = free), e.g. to switch off terms for a Lasso path; None: all free.
        verbose : bool
            Print the loss and timing.
        tol : float
            L-BFGS-B ftol.
        adam_steps : int
            > 0: first that many Adam steps (learning rate lr, cosine decay) in the same scaled
            variables, then L-BFGS (neural bonded terms: the network is not well conditioned for
            L-BFGS from its initial point).
        lr : float
            Adam learning rate (in scaled units).

        Returns
        -------
        dict
            The fitted parameters.

        Notes
        -----
        theta = theta_0 + free * scale * z with scale from SCALES (by leaf name, else 1); the objective
        is loss + sum l2 (free z)^2 + l1 sum sqrt((linear free theta / scale)^2 + 1e-8), jitted with
        its gradient.  Adam runs as jitted lax.scan chunks of 100 steps.
        """
        from scipy.optimize import minimize

        self.prepare("train")
        scale = jax.tree_util.tree_map_with_path(
            lambda path, v: jnp.full(jnp.shape(v), SCALES.get(path[-1].key, 1.0)), P0
        )
        free = jax.tree_util.tree_map_with_path(
            lambda path, v: jnp.full(jnp.shape(v), 0.0 if (path[0].key in frozen or path[-1].key in frozen) else 1.0),
            P0,
        )
        if mask is not None:
            free = jax.tree_util.tree_map(lambda a, b: a * b, free, mask)
        lin = self.model.linear_mask(P0)
        z0, unravel = ravel_pytree(jax.tree_util.tree_map(jnp.zeros_like, P0))
        sc, _ = ravel_pytree(scale)
        fr, _ = ravel_pytree(free)
        ln, _ = ravel_pytree(jax.tree_util.tree_map(lambda v, l: jnp.full(jnp.shape(v), 1.0 if l else 0.0), P0, lin))
        p0, _ = ravel_pytree(P0)
        l2v, _ = ravel_pytree(
            jax.tree_util.tree_map_with_path(
                lambda path, v: jnp.full(jnp.shape(v), self.l2_elec if path[0].key in ("elec", "bci") else self.l2), P0
            )
        )

        def theta(z: jax.Array) -> dict:
            """Return the parameter pytree at scaled variables z."""
            return unravel(p0 + fr * sc * z)

        def obj(z: jax.Array) -> jax.Array:
            """Return the objective (loss + penalties) at scaled variables z."""
            th = p0 + fr * sc * z
            L = self.loss(unravel(th)) + jnp.sum(l2v * (fr * z) ** 2)
            if l1:  # smooth |theta / scale| of the free linear parameters
                L = L + l1 * jnp.sum(jnp.sqrt((ln * fr * th / sc) ** 2 + 1e-8))
            return L

        vg = jax.jit(jax.value_and_grad(obj))
        t0 = time.time()
        z_start = jnp.zeros_like(z0)
        if adam_steps > 0:

            @jax.jit
            def step(carry: tuple, k: jax.Array) -> tuple[tuple, jax.Array]:
                """Take one Adam step with cosine learning-rate decay (scan body; carry (z, m, v), step index k)."""
                z, m, v = carry
                L, g = jax.value_and_grad(obj)(z)
                m = 0.9 * m + 0.1 * g  # Adam moments (beta1 0.9, beta2 0.999), bias-corrected below
                v = 0.999 * v + 0.001 * g * g
                kk = k + 1.0
                eta = lr * 0.5 * (1.0 + jnp.cos(jnp.pi * k / adam_steps))
                z = z - eta * (m / (1 - 0.9**kk)) / (jnp.sqrt(v / (1 - 0.999**kk)) + 1e-8)
                return (z, m, v), L

            carry = (z_start, jnp.zeros_like(z0), jnp.zeros_like(z0))
            chunk = 100
            for c0 in range(0, adam_steps, chunk):
                ks = jnp.arange(c0, min(c0 + chunk, adam_steps), dtype=float)
                carry, Ls = jax.lax.scan(step, carry, ks)
            z_start = carry[0]
            if verbose:
                print(f"    adam: {adam_steps} steps, loss {float(Ls[-1]):.4g} ({time.time() - t0:.1f} s)", flush=True)

        def f(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Return the objective and its gradient as float64 numpy arrays (for scipy)."""
            return tuple(np.asarray(v, float) for v in vg(jnp.asarray(z)))

        res = minimize(
            f,
            np.asarray(z_start),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "maxfun": maxiter * 2, "ftol": tol, "gtol": 1e-8},
        )
        if verbose:
            print(
                f"    fit: loss {float(obj(jnp.zeros_like(z0))):.4g} -> {res.fun:.4g} in {res.nit} it, "
                f"{time.time() - t0:.1f} s ({res.message})",
                flush=True,
            )
        return theta(jnp.asarray(res.x))

    # ------------------------------------------------------------------ metrics
    def metrics(self, P: dict, split: str = "test") -> dict:
        """Return per-molecule error metrics on `split`, in the paper's units.

        Returns
        -------
        dict
            {molecule: {"E_MAE" energy MAE with the mean offset removed [kcal/mol], "F_MAE" mean
            per-atom force-error norm [kcal/mol/A], "n" frames, "mu_RMSE_D" dipole RMSE [D] (with
            dipole labels), "esp_RMSE_mEh" [mhartree/e] and "esp_RRMSE" (with ESP data)}}.
        """
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
            r = {
                "E_MAE": float(np.mean(np.abs(dE - dE.mean()))) / KCAL,
                "F_MAE": float(np.mean(dF)) / (KCAL * 10.0),
                "n": len(fs),
            }
            if fs.mu is not None:
                r["mu_RMSE_D"] = float(np.sqrt(np.mean(np.sum((fs.mu - D) ** 2, -1)))) / DEBYE_E_NM
            if m in self.esp:
                a, b = self.esp_rmse(m, P)
                r["esp_RMSE_mEh"], r["esp_RRMSE"] = 1e3 * float(a), float(b)
            out[m] = r
        return out
