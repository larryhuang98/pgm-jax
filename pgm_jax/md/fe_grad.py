"""Parameter gradients of alchemical free energies, sampling side.

Contents: `ParameterGradients` (dU/dP of the end states at the configurations of every lambda
window, sampled by alchemy.FreeEnergyRun), `gas_leg_gradient` (the exact gas-phase leg of a
rigid solute), and the parameter maps `scaled_params` and `alchemical_map`.  The estimators,
errors and fitting targets are in pgm_jax/fit/free_energy.py.

Thermodynamic identity.  For a Hamiltonian U_k(x; theta) sampled at fixed volume and temperature,
f_k(theta) = -kT ln Z_k(theta) has

    df_k / dtheta = < dU_k / dtheta >_k ,

so the free energy of the solution leg (window 0 = full coupling -> window K-1 = decoupled) has

    d DeltaG_solv / dtheta = < dU_{K-1}/dtheta >_{K-1} - < dU_0/dtheta >_0

and the hydration free energy DeltaG_hyd = DeltaG_gas - DeltaG_solv (alchemy.py) has
d DeltaG_hyd / dtheta = d DeltaG_gas / dtheta - d DeltaG_solv / dtheta.  The gas-phase leg of a
rigid solute is exact (E_gas(0) - E_gas(1), gradient by autodiff: gas_leg_gradient); with
intramolecular="keep" it is part of the Hamiltonian (the decoupled end state then still depends on
the solute's electrostatic parameters through the gas-phase correction, and the identity takes
care of it).  Only the two end states enter; the intermediate windows help only through MBAR.

dU_k/dtheta at a configuration is the partial derivative at the induced dipoles converged in
Hamiltonian k: the pGM energy is stationary in the dipoles (Hellmann-Feynman), so no derivative
of the solve is needed (ParameterGradients: one dipole solve and one reverse-mode pass per target
Hamiltonian and configuration, batched over the windows with jax.vmap, on the device).  theta is
the parameter table (every entry of sys.table, flattened: ParameterSpace.values); gradients with respect to
any other parameters theta' follow by the chain rule dG/dtheta' = (dP/dtheta')^T dG/dP
(pgm_jax/fit/free_energy.py: FEGradient.chain, FreeEnergyTarget.value_and_grad).

Cost: per sample, 2 x K dipole re-solves and 2 x K reverse passes of the energy (K windows, two end
states); see docs/fe_gradients.md.

Units: kJ/mol, nm, e, e nm; gradients in kJ/mol per unit of each table entry.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from ..fit.params import SCALE_GROUPS, ParameterSpace
from ..system import QUANTITIES
from .alchemy import PREFIX

if TYPE_CHECKING:
    from ..system import System
    from .alchemy import GasPhaseLeg, LambdaWindows
    from .integrate import MDState


# ----------------------------------------------------------------------------- parameter space
def scaled_params(
    space: ParameterSpace, P: dict, scales: dict[str, float | jax.Array], solute: bool | None = True
) -> dict:
    """Return the table P with the parameters of each scale group multiplied by its scale.

    charge (q and cov), rmin, alpha and radius are multiplied by s; eps by s, i.e. sqrt(eps) by
    s^1/2 (fit/params.py SCALE_GROUPS).  JAX-differentiable in the scales.

    Parameters
    ----------
    space : ParameterSpace
        A "values" space over the table (which entries exist, `select`).
    P : dict
        Parameter table {quantity: array}.
    scales : dict of str to float
        Scale per group, e.g. {"charge": 1.05, "eps": 0.9}.
    solute : bool, optional
        Which entries are scaled, as in ParameterSpace.select (True: the solute's, False: the
        others', None: all).

    Returns
    -------
    dict of str to jax.Array
        The scaled table (float64).

    Raises
    ------
    ValueError
        An unknown scale group.
    """
    out = {q: jnp.asarray(P[q], jnp.float64) for q in P}
    for g, s in scales.items():
        if g not in SCALE_GROUPS:
            raise ValueError(f"unknown scale group {g!r} (one of {sorted(SCALE_GROUPS)})")
        for q, e in SCALE_GROUPS[g].items():
            if q not in space.slices:
                continue
            m = np.zeros(out[q].shape, bool)
            m[space.select((q,), solute) - space.slices[q].start] = True
            out[q] = out[q] * jnp.where(m, jnp.asarray(s, jnp.float64) ** e, 1.0)
    return out


def alchemical_map(sys0: System, sysA: System) -> Callable[[dict], dict[str, jax.Array]]:
    """Return the map P0 -> PA from the original parameter table to the alchemical system's.

    The table of sys0 is mapped onto the table of alchemical_system(sys0, k): the solute's "alch:"
    keys take the values of the keys they were copied from.  A JAX gather, so gradients of
    PA-functions with respect to P0 sum the solute's copy and the original key (e.g. water in
    water: the solute and the solvent share the model).

    Parameters
    ----------
    sys0 : System
        The original system.
    sysA : System
        alchemical_system(sys0, k)[0].

    Returns
    -------
    Callable[[dict], dict of str to jax.Array]
        f(P0) = PA, for every quantity of the table.
    """
    idx = {}
    for q in QUANTITIES:
        pos = {k: i for i, k in enumerate(sys0.table.keys[q])}
        idx[q] = np.array([pos[k[len(PREFIX) :] if k.startswith(PREFIX) else k] for k in sysA.table.keys[q]], int)

    def f(P0: dict) -> dict[str, jax.Array]:
        return {q: jnp.asarray(P0[q])[idx[q]] for q in QUANTITIES}

    return f


# ----------------------------------------------------------------------------- sampling
def _frame(windows: LambdaWindows, st: MDState) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return (atom positions (N, 3) [nm], candidate rows (N, C), overflow) of one window state.

    Works for the rigid engine (positions from the rigid bodies, molecule centres) and the
    flexible one (atoms, neighbour-list group centres).
    """
    integ = windows.integ
    if hasattr(integ, "flex"):
        pos = st.dyn.position
        centers = integ.flex.list_centers(pos)
    else:
        pos = windows.sim.rigid.positions(st.dyn.position)
        centers = st.dyn.position.center
    cand, ovf = integ.nb.candidates(st.nbr, centers, st.box, pos)
    return pos, cand, ovf


class ParameterGradients:
    """Sampler of dU_k/dP of target Hamiltonians k at the configuration of every lambda window.

    The targets are window indices (default the two end states 0 and K-1); the samples are those
    of alchemy.FreeEnergyRun(param_grad=...).

        run = FreeEnergyRun(windows, sample_every=500, exchange_every=500,
                            param_grad=ParameterGradients(windows))

    sample() -> (T, K, M): [t, n] = dU_{targets[t]}/dP at the configuration of window n (kJ/mol per
    unit of each of the M entries of `space`), at the induced dipoles re-solved in Hamiltonian
    targets[t] from the configuration's own (Hellmann-Feynman).  Batched windows: one vmapped
    program over the windows (T target Hamiltonians by lax.map inside).

    Attributes
    ----------
    windows : LambdaWindows
        The windows.
    targets : np.ndarray (T,) int
        Target window indices (non-negative).
    space : ParameterSpace
        The "values" space whose M entries are differentiated (names "quantity:key").
    """

    def __init__(
        self, windows: LambdaWindows, targets: Sequence[int] = (0, -1), quantities: Sequence[str] | None = None
    ) -> None:
        """Set up the sampler.

        Parameters
        ----------
        windows : LambdaWindows
            The lambda windows of the free-energy run.
        targets : sequence of int
            Window indices of the target Hamiltonians (negative indices count from the end).
        quantities : sequence of str, optional
            Parameter-table quantities to differentiate (default: all of the table's quantities);
            one ParameterSpace.values over them defines the M entries.

        Raises
        ------
        ValueError
            Repeated or no target indices.
        """
        K = windows.n
        t = [int(x) % K for x in targets]
        if len(set(t)) != len(t) or not t:
            raise ValueError("targets: distinct window indices")
        self.windows, self.targets = windows, np.array(t, int)
        self.space = ParameterSpace.values(windows.alchemy.sys.table, quantities)
        self._fns = {}

    @property
    def params(self) -> dict:
        """Parameter table sampled: the integrator's parameters, or the alchemical region's params0."""
        integ = self.windows.integ
        return integ.params if integ.params is not None else self.windows.alchemy.params0

    def _one(self, st: MDState, lam_t: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return dU/dP (T, M) of one configuration in the T target Hamiltonians, max CG, overflow.

        For each target lambda (lax.map): the dipoles re-solved from the state's own, then
        jax.grad of the fixed-dipole energy with respect to the parameter table, flattened by
        `space`.
        """
        w = self.windows
        ff, alch = w.sim.ff, w.alchemy
        params = self.params
        pos, cand, ovf0 = _frame(w, st)

        def one(lam: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
            """Return (dU/dP (M,), CG iterations, overflow) in the Hamiltonian at lam."""
            _, ind, it, ovf = alch.energy(ff, pos, st.box, cand, st.induction, params, lam)
            g = jax.grad(lambda P: alch.energy_fixed_mu(ff, pos, st.box, cand, ind.mu, P, lam))(params)
            return self.space.flatten(g), it, ovf

        G, it, ovf = jax.lax.map(one, lam_t)
        return G, jnp.max(it), jnp.any(ovf) | ovf0

    def _fn(self) -> Callable:
        """Return the jitted (and, batched, vmapped) `_one`, cached by static sizes and layout."""
        w = self.windows
        key = (w._sizes(), jax.tree_util.tree_structure(w.S) if w.batched else None)
        if key not in self._fns:
            if w.batched:
                from .remd import _axes

                f = jax.vmap(self._one, in_axes=(_axes(w.S), None))
            else:
                f = self._one
            self._fns = {key: jax.jit(f)}
        return self._fns[key]

    def sample(self) -> np.ndarray:
        """Return dU_{targets[t]}/dP at the configuration of every window.

        Returns
        -------
        np.ndarray (T, K, M)
            [t, n] = dU_{targets[t]}/dP at the configuration of window n [kJ/mol per unit of each
            table entry].

        Raises
        ------
        RuntimeError
            If the row capacity is exceeded.
        """
        w = self.windows
        f = self._fn()
        lam_t = jnp.asarray(w.lambdas[self.targets], jnp.float64)
        if w.batched:
            G, it, ovf = f(w.S, lam_t)
        else:
            outs = [f(s, lam_t) for s in w.states]
            G, it, ovf = (jnp.stack([o[j] for o in outs]) for j in range(3))
        if bool(np.any(np.asarray(ovf))):
            raise RuntimeError("row capacity exceeded while sampling parameter gradients")
        return np.transpose(np.asarray(G, float), (1, 0, 2))

    def meta(self) -> dict[str, Any]:
        """Return what FreeEnergyRun stores with the samples: targets, entry names, the parameters."""
        p = np.asarray(self.space.flatten(self.params), float)
        return {"dudp_targets": self.targets.tolist(), "dudp_names": self.space.names, "params_flat": p.tolist()}


def gas_leg_gradient(gas: GasPhaseLeg, params: dict, space: ParameterSpace) -> tuple[float, np.ndarray]:
    """Return Delta G_gas(1 -> 0) and its parameter gradient for a rigid solute's gas-phase leg.

    alchemy.GasPhaseLeg: Delta G_gas = E_gas(0) - E_gas(1), exact; the gradient by autodiff.

    Parameters
    ----------
    gas : GasPhaseLeg
        The gas-phase leg.
    params : dict
        Parameter table.
    space : ParameterSpace
        Space that flattens the gradient (M entries).

    Returns
    -------
    delta_g : float
        Delta G_gas(1 -> 0) [kJ/mol].
    grad : np.ndarray (M,)
        d Delta G_gas / dP [kJ/mol per unit of each entry].
    """

    def f(P: dict) -> jax.Array:
        return gas._e(jnp.asarray(0.0), P) - gas._e(jnp.asarray(1.0), P)

    v, g = jax.value_and_grad(f)(params)
    return float(v), np.asarray(space.flatten(g), float)
