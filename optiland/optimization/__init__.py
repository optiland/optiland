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
from .constraint import Constraint, ConstraintManager
from .problem import OptimizationProblem
from .api import minimize
from .state import OptimizationResult
from .failure import ConfigurationError
from .observers import (
    CancelToken,
    CheckpointObserver,
    ConsoleObserver,
    HistoryObserver,
    Observer,
)
from .stopping import (
    CostTolerance,
    GradNormTolerance,
    MaxEvals,
    MaxIter,
    StepStall,
    StoppingCriterion,
)
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
    CMAES,
    ParticleSwarm,
)

import sys

optimization = sys.modules[__name__]
