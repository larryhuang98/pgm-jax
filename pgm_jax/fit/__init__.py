"""Fitting pGM parameters to liquid and gas-phase properties with ensemble gradients.

  params.py      ParameterSpace: theta (scale factors / shifts on the ParamTable) -> parameter pytree
  frames.py      FrameAnalyzer: per-frame U, M, cell polarizability, molecular dipoles, g(r) and their
                 explicit theta-derivatives (adjoint of the induction solve), batched on the device
  estimators.py  LiquidSamples (fluctuation-formula Jacobians via reweighted averages, jackknife,
                 bootstrap, reweighting predictions), GasPhase (monomer energy, dipole, polarizability)
  optimize.py    Target, Objective (residuals, LM trust-region step, parameter covariance, propagation)
  liquid.py      LiquidFit: NPT simulation -> analysis -> step, iterated, with JSON records
See docs/liquid_fit.md."""

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
