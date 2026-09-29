"""Fit pGM parameters to liquid and gas-phase properties with ensemble gradients.

Contents:

  params.py      ParameterSpace: theta (scale factors, shifts or values on the ParamTable, optional
                 charge neutrality) -> parameter pytree; Param, SCALE_GROUPS
  frames.py      FrameAnalyzer: per-frame U, M, cell polarizability, molecular dipoles, g(r) and their
                 explicit theta-derivatives (adjoint of the induction solve), batched on the device
  estimators.py  LiquidSamples (fluctuation-formula Jacobians via reweighted averages, jackknife,
                 bootstrap, reweighting predictions), GasPhase (monomer energy, dipole, polarizability)
  optimize.py    Target, Objective (residuals, LM trust-region step, parameter covariance, propagation)
  liquid.py      LiquidFit: NPT simulation -> analysis -> step, iterated, with JSON records
  free_energy.py alchemical free energies with parameter gradients as fitting targets (gradient_estimate,
                 FEGradient, FreeEnergyTarget, combine; samples from md/fe_grad.py)
  qm.py          fits to QM cluster energies, forces and monomer properties (QMFit)
  reweighting.py Reweighting: ensemble averages and their gradients by reweighting stored frames

The package namespace re-exports GasPhase, LiquidSamples, FrameAnalyzer, RDFSpec, Estimate,
Objective, Target, Param and ParameterSpace.

Units: library units (nm, ps, K, bar, kJ/mol, e); observables in their conventional units
(g/cm^3, kcal/mol, D, A^3), stated per observable in estimators.py.

See docs/liquid_fit.md, docs/fe_gradients.md, docs/qmfit.md.
"""

from .estimators import GasPhase, LiquidSamples
from .frames import FrameAnalyzer, RDFSpec
from .optimize import Estimate, Objective, Target
from .params import Param, ParameterSpace

__all__ = [
    "GasPhase",
    "LiquidSamples",
    "FrameAnalyzer",
    "RDFSpec",
    "Estimate",
    "Objective",
    "Target",
    "Param",
    "ParameterSpace",
]
