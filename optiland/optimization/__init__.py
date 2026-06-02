# flake8: noqa

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
from .optimizer.scipy import (
    OptimizerGeneric,
    LeastSquares,
    DualAnnealing,
    DifferentialEvolution,
    SHGO,
    BasinHopping,
    GlassExpert,
    OrthogonalDescent,
)

try:
    from .optimizer.torch.adam import TorchAdamOptimizer
    from .optimizer.torch.sgd import TorchSGDOptimizer
except (ImportError, ModuleNotFoundError, OSError):
    pass

from .optimizer.scipy import glass_expert

from .optimizer.custom import (
    ParticleSwarm,
)

from .api import minimize  # noqa: E402
from .errors import ConfigurationError  # noqa: E402
from .state import (  # noqa: E402
    Capability,
    EvalCapability,
    OptimizationResult,
    OptimizationState,
)
from .native.least_squares import GaussNewton, LevenbergMarquardt  # noqa: E402
from .control.levenberg import LevenbergController  # noqa: E402
from .control.line_search import LineSearchController  # noqa: E402
from .observers.checkpoint import CheckpointObserver  # noqa: E402
from .constraints.null_space import NullSpaceStrategy  # noqa: E402

import sys

optimization = sys.modules[__name__]
