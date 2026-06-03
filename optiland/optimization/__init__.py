# flake8: noqa
"""Optiland Optimization Package

Public surface:
  - ``minimize`` — canonical entry point (all methods, all backends).
  - ``OptimizationResult``, ``OptimizationState``, ``Capability``, ``EvalCapability``
  - ``ConfigurationError`` — raised for impossible method/backend combinations.
  - Native solvers: ``LevenbergMarquardt``, ``GaussNewton``.
  - Controllers: ``LevenbergController``, ``LineSearchController``.
  - Observers: ``CheckpointObserver``.
  - Constraints: ``NullSpaceStrategy``.
  - Variable / Operand types (unchanged legacy surface).
  - Deprecated shims: ``OptimizerGeneric``, ``LeastSquares``, ``DualAnnealing``,
    ``DifferentialEvolution``, ``SHGO``, ``BasinHopping``, ``TorchAdamOptimizer``,
    ``TorchSGDOptimizer``.  These emit ``DeprecationWarning`` on construction;
    removal planned for v0.7.0.
  - Non-deprecated standalone: ``GlassExpert``, ``OrthogonalDescent``,
    ``ParticleSwarm`` — use directly (not accessible via minimize()).
"""

from .variable import (
    VariableBehavior,
    RadiusVariable,
    ConicVariable,
    ThicknessVariable,
    IndexVariable,
    AsphereCoeffVariable,
    PolynomialCoeffVariable,
    ChebyshevCoeffVariable,
    Variable,
)
from .operand import ParaxialOperand, AberrationOperand, RayOperand, Operand
from .problem import OptimizationProblem

# --- Deprecated shims (emit DeprecationWarning on construction) ---
from .optimizer.scipy import (
    OptimizerGeneric,
    LeastSquares,
    DualAnnealing,
    DifferentialEvolution,
    SHGO,
    BasinHopping,
    GlassExpert,  # not deprecated (standalone)
    OrthogonalDescent,  # not deprecated (standalone)
)

try:
    from .optimizer.torch.adam import TorchAdamOptimizer
    from .optimizer.torch.sgd import TorchSGDOptimizer
except (ImportError, ModuleNotFoundError, OSError):
    pass

from .optimizer.scipy import glass_expert

from .optimizer.custom import (
    ParticleSwarm,  # not deprecated (standalone)
)

# --- Canonical public surface ---
from .api import minimize
from .errors import ConfigurationError
from .state import (
    Capability,
    EvalCapability,
    OptimizationResult,
    OptimizationState,
)
from .native.least_squares import GaussNewton, LevenbergMarquardt
from .control.levenberg import LevenbergController
from .control.line_search import LineSearchController
from .observers.checkpoint import CheckpointObserver
from .constraints.null_space import NullSpaceStrategy

import sys

optimization = sys.modules[__name__]
