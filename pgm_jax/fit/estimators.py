"""Ensemble estimates of liquid and gas-phase observables, their parameter Jacobians, statistical
errors (block jackknife, block bootstrap) and reweighting predictions.

Liquid observables are smooth functions f(<a_1>, ..., <a_k>) of ensemble averages of per-frame
quantities a_j(x; theta) (frames.py gives their values and explicit derivatives).  For frames sampled
at theta0 (NVT or NPT; the pV term does not depend on theta),

    d<a>/dtheta = <da/dtheta> - beta (<a dU/dtheta> - <a><dU/dtheta>),

which is what one gets by differentiating, at delta = 0, the reweighted average

    <a>(delta) = sum_k w_k(delta) [a_k + da_k/dtheta . delta],   w_k ~ exp(-beta dU_k/dtheta . delta).

LiquidSamples implements <a>(delta) in JAX, so every observable's Jacobian is a jax.jacfwd of its
estimator at delta = 0, and the same function at delta != 0 is the first-order (linear-exponential)
reweighting prediction, with its Kish effective sample size n_eff.  exact_average() reweights with
energies and observables re-evaluated at the new parameters on the stored frames.

Observables (name: estimator, unit):
  density         <rho> = <m / V>                                        g/cm^3
  hvap            u_gas(theta) - <U>/N + R T                              kcal/mol
  eps             1 + 4 pi <alpha_cell/V> + 4 pi KE (<M.M> - <M>.<M>) / (3 kB T <V>)
                  (tin-foil Ewald, adiabatic induced dipoles; analysis/dielectric.py)
  liquid_dipole   <mean |molecular dipole|>                               D
  rdf             <g(r)> per bin (frames.RDFSpec)
  volume, energy  <V> (nm^3), <U>/N (kJ/mol)
  alpha_p         (<V H> - <V><H>) / (kB T^2 <V>), H = U + pV              1/K     (NPT; temperature derivative)
  kappa_t         (<V^2> - <V>^2) / (kB T <V>)                            1/bar   (NPT)
                  (their theta-gradients include the third cumulants, through the reweighted averages)
  gas_dipole, gas_polarizability, gas_energy: GasPhase (one rigid molecule, exact gradients)   D, A^3, kJ/mol
Errors: the frames are cut into contiguous blocks; the jackknife over blocks (leave one out,
full estimator) gives the covariance of all observables of a run, and of their Jacobians."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from ..units import AMU_NM3_TO_G_CM3, BAR_PER_KJMOL_NM3, DEBYE_E_NM, KB, KCAL, KE

LIQUID = (
    "density",
    "hvap",
    "eps",
    "liquid_dipole",
    "rdf",
    "volume",
    "energy",
    "eps_fluct",
    "eps_inf",
    "alpha_p",
    "kappa_t",
)
GAS = ("gas_dipole", "gas_polarizability", "gas_energy")


class LiquidSamples:
    """Per-frame data of one run at theta0: `frames` from FrameAnalyzer.analyze (grad=True)."""

    def __init__(
        self, frames: dict, temperature: float, n_mol: int, mass: float, nblocks: int = 10, pressure: float = 1.0
    ):
        """Reweighting-ready averages of one run.

        Parameters
        ----------
        frames : dict
            Per-frame arrays of FrameAnalyzer.analyze(grad=True): U [kJ/mol], V [nm^3], M [e nm],
            alpha [nm^3], D, and their theta-derivatives.
        temperature : float
            Temperature of the run [K].
        n_mol : int
            Number of molecules.
        mass : float
            Total mass [amu].
        nblocks : int
            Contiguous blocks of the jackknife errors.
        pressure : float
            Pressure of the run [bar] (enthalpy H = U + p V).
        """
        self.temperature, self.beta, self.N, self.mass = (
            float(temperature),
            1.0 / (KB * float(temperature)),
            int(n_mol),
            float(mass),
        )
        self.p = float(pressure) / BAR_PER_KJMOL_NM3
        f = {k: np.asarray(v) for k, v in frames.items()}
        self.F = len(f["U"])
        V = f["V"]
        n = f["dU"].shape[1]
        v, g = {}, {}
        v["U"], g["U"] = f["U"], f["dU"]
        v["V"], g["V"] = V, np.zeros((self.F, n))
        v["rho"], g["rho"] = mass / V * AMU_NM3_TO_G_CM3, np.zeros((self.F, n))
        v["M"], g["M"] = f["M"], f["dM"]
        v["M2"] = np.sum(f["M"] ** 2, axis=1)
        g["M2"] = 2.0 * np.einsum("fc,fcn->fn", f["M"], f["dM"])
        v["aV"], g["aV"] = f["alpha"] / V, f["dalpha"] / V[:, None]
        v["D"], g["D"] = f["D"], f["dD"]
        Hh = f["U"] + self.p * V  # enthalpy (configurational part)
        v["H"], g["H"] = Hh, f["dU"]
        v["VH"], g["VH"] = V * Hh, V[:, None] * f["dU"]
        v["V2"], g["V2"] = V * V, np.zeros((self.F, n))
        if "rdf" in f:
            v["rdf"], g["rdf"] = f["rdf"], np.zeros(f["rdf"].shape + (n,))
        self.v = {k: jnp.asarray(x) for k, x in v.items()}
        self.g = {k: jnp.asarray(x) for k, x in g.items()}
        dU = f["dU"] - f["dU"].mean(0)  # centred: only fluctuations enter the weights
        self.dU = jnp.asarray(dU)
        self.n = n
        self.nblocks = int(nblocks)
        self.block = np.minimum((np.arange(self.F) * self.nblocks) // self.F, self.nblocks - 1)

    # ------------------------------------------------------------------ averages
    def log_weights(self, delta, frame_weights=None):
        lw = -self.beta * (self.dU @ delta)
        if frame_weights is not None:
            lw = lw + jnp.log(jnp.asarray(frame_weights, float))
        return lw - logsumexp(lw)

    def averages(self, delta, frame_weights=None) -> dict:
        """<a>(delta) for every per-frame quantity (linear-exponential reweighting)."""
        w = jnp.exp(self.log_weights(delta, frame_weights))
        return {k: jnp.tensordot(w, self.v[k] + self.g[k] @ delta, axes=(0, 0)) for k in self.v}

    def n_eff(self, delta) -> float:
        w = jnp.exp(self.log_weights(jnp.asarray(delta, float)))
        return float(1.0 / jnp.sum(w * w))

    def exact_average(self, new: dict, base: dict | None = None) -> tuple[dict, float]:
        """Averages at new parameters from values re-evaluated on the same frames (FrameAnalyzer
        .analyze(theta_new, frames, grad=False)); base: the same frames evaluated at theta0 (default:
        this run's own values).  Returns (averages, n_eff)."""
        U0 = np.asarray(base["U"]) if base is not None else np.asarray(self.v["U"])
        dUe = np.asarray(new["U"]) - U0
        lw = -self.beta * (dUe - dUe.mean())
        lw -= logsumexp(lw)
        w = np.exp(lw)
        V = np.asarray(new["V"])
        vals = {
            "U": new["U"],
            "V": V,
            "rho": self.mass / V * AMU_NM3_TO_G_CM3,
            "M": new["M"],
            "M2": np.sum(new["M"] ** 2, 1),
            "aV": new["alpha"] / V,
            "D": new["D"],
            "H": new["U"] + self.p * V,
            "VH": V * (new["U"] + self.p * V),
            "V2": V * V,
        }
        if "rdf" in new:
            vals["rdf"] = new["rdf"]
        return {k: jnp.asarray(np.tensordot(w, np.asarray(x), axes=(0, 0))) for k, x in vals.items()}, float(
            1.0 / np.sum(w * w)
        )

    # ------------------------------------------------------------------ observables
    def observable(self, name: str, avg: dict, gas: dict | None = None):
        """Value of one liquid observable from (reweighted) ensemble averages.

        Parameters
        ----------
        name : str
            "density" [g/cm^3], "volume" [nm^3], "energy" (<U>/N, kJ/mol), "hvap" [kcal/mol],
            "eps_fluct", "eps_inf", "eps" (dimensionless; tin-foil boundary), "liquid_dipole" [D],
            "rdf", "alpha_p" [1/K] or "kappa_t" [1/bar] (the last two need an NPT run).
        avg : dict
            Ensemble averages of the per-frame quantities (keys "rho", "V", "U", "M", "M2", "aV", "D",
            "rdf", "VH", "V2", "H"), as returned by averages(); jax arrays, so that the result can be
            differentiated with respect to the reweighting step.
        gas : dict, optional
            Gas-phase values of GasPhase ("gas_energy" [kJ/mol]); needed for "hvap".

        Returns
        -------
        jax.Array
            The observable (a scalar, or the g(r) bins for "rdf").

        Raises
        ------
        ValueError
            "hvap" without `gas`.
        KeyError
            Unknown `name`.

        Notes
        -----
        eps = 1 + 4 pi <alpha/V> + 4 pi KE (<M.M> - <M>.<M>) / (3 kB T <V>) (Neumann, Mol. Phys. 50, 841
        (1983)); hvap = u_gas - <U>/N + kB T; alpha_p and kappa_t are the enthalpy-volume and volume
        fluctuation formulas (Allen and Tildesley, Computer Simulation of Liquids, 2nd ed., sec. 2.5).
        """
        kT = KB * self.temperature
        if name == "density":
            return avg["rho"]
        if name == "volume":
            return avg["V"]
        if name == "energy":
            return avg["U"] / self.N
        if name == "hvap":
            if gas is None:
                raise ValueError("hvap needs the gas-phase energy (GasPhase)")
            return (gas["gas_energy"] - avg["U"] / self.N + kT) / KCAL
        fl = 4.0 * np.pi * KE * (avg["M2"] - jnp.sum(avg["M"] ** 2)) / (3.0 * kT * avg["V"])
        if name == "eps_fluct":
            return fl
        if name == "eps_inf":
            return 1.0 + 4.0 * np.pi * avg["aV"]
        if name == "eps":
            return 1.0 + 4.0 * np.pi * avg["aV"] + fl
        if name == "liquid_dipole":
            return avg["D"] / DEBYE_E_NM
        if name == "rdf":
            return avg["rdf"]
        if name == "alpha_p":  # thermal expansion (1/K), NPT
            return (avg["VH"] - avg["V"] * avg["H"]) / (KB * self.temperature**2 * avg["V"])
        if name == "kappa_t":  # isothermal compressibility (1/bar), NPT
            return (avg["V2"] - avg["V"] ** 2) / (kT * avg["V"]) / BAR_PER_KJMOL_NM3
        raise KeyError(f"unknown liquid observable {name!r}")

    def blocks_weights(self, nblocks: int | None = None):
        """Leave-one-block-out frame weights (nblocks, F) for nblocks contiguous blocks (default: self.nblocks)."""
        B = self.nblocks if nblocks is None else int(nblocks)
        block = self.block if nblocks is None else np.minimum((np.arange(self.F) * B) // self.F, B - 1)
        return np.array([(block != b).astype(float) for b in range(B)])

    def bootstrap_weights(self, rng, n: int):
        """Block-bootstrap frame weights (n, F): each block's multiplicity in a resample."""
        c = rng.multinomial(self.nblocks, np.full(self.nblocks, 1.0 / self.nblocks), size=n)
        return c[:, self.block].astype(float)


class GasPhase:
    """One rigid molecule in the gas phase (the monomer of the liquid, same ParamTable): energy
    (kJ/mol, the MD engine's intramolecular energy of the isolated molecule: pGM with every pair,
    no LJ inside a rigid molecule), dipole (D) and isotropic polarizability (A^3), as JAX functions
    of theta."""

    def __init__(self, molecule, positions, table, space, elec: str = "qpi"):
        """The monomer model.

        Parameters
        ----------
        molecule : Molecule
            The rigid molecule.
        positions : array (n, 3)
            Its geometry [nm] (centred).
        table : ParamTable
            The parameter table of the liquid (so theta means the same).
        space : ParameterSpace
            theta -> parameters.
        elec : str
            Electrostatics level ("q", "qp", "qi", "qpi").
        """
        from ..channels import ElecChannel, molecular_polarizability
        from ..system import System

        self.sys = System([molecule], table=table)
        self.pos = jnp.asarray(np.asarray(positions, float) - np.mean(positions, axis=0))
        self.space = space
        self.chan = ElecChannel.level(elec)
        self._mp = molecular_polarizability
        self._fn = jax.jit(self._props)
        self._jac = jax.jit(jax.jacfwd(lambda th: jnp.stack(list(self._props(th).values()))))

    def _props(self, theta):
        P = self.space(theta)
        e, aux = self.chan.energy(self.pos, self.sys, P)
        Pa = self.sys.expand(P)
        m = jnp.sum(Pa["q"][:, None] * self.pos, axis=0) + jnp.sum(aux["p"], axis=0)
        if "mu" in aux:
            m = m + jnp.sum(aux["mu"], axis=0)
            a = jnp.trace(self._mp(self.pos, self.sys, P)) / 3.0 * 1000.0
        else:
            a = jnp.zeros(())
        return {"gas_energy": sum(e.values()), "gas_dipole": jnp.linalg.norm(m) / DEBYE_E_NM, "gas_polarizability": a}

    def __call__(self, theta) -> dict:
        return self._props(jnp.asarray(theta, float))

    def values(self, theta) -> dict:
        return {k: float(v) for k, v in self._fn(jnp.asarray(theta, float)).items()}

    def jacobian(self, theta) -> dict:
        J = np.asarray(self._jac(jnp.asarray(theta, float)))
        return dict(zip(("gas_energy", "gas_dipole", "gas_polarizability"), J))
