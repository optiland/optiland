"""Analytic least-squares benchmarks for the native DLS validation suite.

Moré, Garbow and Hillstrom (MGH, 1981) test functions and NIST StRD nonlinear
regression datasets with certified values. Each problem is plain numpy: a
residual function, starting points and the known solution.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class LSProblem:
    """A least-squares benchmark.

    Attributes:
        name: Identifier.
        residuals: ``residuals(x) -> r``.
        starts: Starting points.
        solution: Known minimizer (None when only the merit is certified).
        merit: Certified (or known) minimum of ``sum(r ** 2)``.
        certified: Whether ``solution`` / ``merit`` are NIST-certified values.
    """

    name: str
    residuals: Callable[[np.ndarray], np.ndarray]
    starts: tuple[tuple[float, ...], ...]
    solution: tuple[float, ...] | None
    merit: float
    certified: bool = False
    tags: tuple[str, ...] = field(default=())


# ---------------------------------------------------------------------------
# MGH test functions
# ---------------------------------------------------------------------------


def _rosenbrock(x: np.ndarray) -> np.ndarray:
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


def _powell_singular(x: np.ndarray) -> np.ndarray:
    return np.array(
        [
            x[0] + 10.0 * x[1],
            np.sqrt(5.0) * (x[2] - x[3]),
            (x[1] - 2.0 * x[2]) ** 2,
            np.sqrt(10.0) * (x[0] - x[3]) ** 2,
        ]
    )


def _brown_badly_scaled(x: np.ndarray) -> np.ndarray:
    return np.array([x[0] - 1e6, x[1] - 2e-6, x[0] * x[1] - 2.0])


# ---------------------------------------------------------------------------
# NIST StRD nonlinear regression (https://www.itl.nist.gov/div898/strd/nls/)
# ---------------------------------------------------------------------------

MISRA1A_X = np.array(
    [77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4, 434.8, 477.3,
     536.8, 593.1, 689.1, 760.0]
)  # fmt: skip
MISRA1A_Y = np.array(
    [10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 55.05,
     61.01, 66.40, 75.47, 81.78]
)  # fmt: skip


def _misra1a(b: np.ndarray) -> np.ndarray:
    return b[0] * (1.0 - np.exp(-b[1] * MISRA1A_X)) - MISRA1A_Y


THURBER_X = np.array(
    [-3.067, -2.981, -2.921, -2.912, -2.840, -2.797, -2.702, -2.699, -2.633,
     -2.481, -2.363, -2.322, -1.501, -1.460, -1.274, -1.212, -1.100, -1.046,
     -0.915, -0.714, -0.566, -0.545, -0.400, -0.309, -0.109, -0.103, 0.010,
     0.119, 0.377, 0.790, 0.963, 1.006, 1.115, 1.572, 1.841, 2.047, 2.200]
)  # fmt: skip
THURBER_Y = np.array(
    [80.574, 84.248, 87.264, 87.195, 89.076, 89.608, 89.868, 90.101, 92.405,
     95.854, 100.696, 101.060, 401.672, 390.724, 567.534, 635.316, 733.054,
     759.087, 894.206, 990.785, 1090.109, 1080.914, 1122.643, 1178.351,
     1260.531, 1273.514, 1288.339, 1327.543, 1353.863, 1414.509, 1425.208,
     1421.384, 1442.962, 1464.350, 1468.705, 1447.894, 1457.628]
)  # fmt: skip


def _thurber(b: np.ndarray) -> np.ndarray:
    x = THURBER_X
    num = b[0] + b[1] * x + b[2] * x**2 + b[3] * x**3
    den = 1.0 + b[4] * x + b[5] * x**2 + b[6] * x**3
    return num / den - THURBER_Y


MGH09_X = np.array(
    [4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625]
)
MGH09_Y = np.array(
    [1.957e-01, 1.947e-01, 1.735e-01, 1.600e-01, 8.440e-02, 6.270e-02,
     4.560e-02, 3.420e-02, 3.230e-02, 2.350e-02, 2.460e-02]
)  # fmt: skip


def _mgh09(b: np.ndarray) -> np.ndarray:
    x = MGH09_X
    return b[0] * (x**2 + x * b[1]) / (x**2 + x * b[2] + b[3]) - MGH09_Y


PROBLEMS: dict[str, LSProblem] = {
    "rosenbrock": LSProblem("rosenbrock", _rosenbrock, ((-1.2, 1.0),), (1.0, 1.0), 0.0),
    "powell_singular": LSProblem(
        "powell_singular",
        _powell_singular,
        ((3.0, -1.0, 0.0, 1.0),),
        (0.0, 0.0, 0.0, 0.0),
        0.0,
        tags=("singular",),
    ),
    "brown_badly_scaled": LSProblem(
        "brown_badly_scaled", _brown_badly_scaled, ((1.0, 1.0),), (1e6, 2e-6), 0.0
    ),
    "misra1a": LSProblem(
        "misra1a",
        _misra1a,
        ((500.0, 1e-4), (250.0, 5e-4)),
        (2.3894212918e02, 5.5015643181e-04),
        1.2455138894e-01,
        certified=True,
    ),
    "thurber": LSProblem(
        "thurber",
        _thurber,
        (
            (1000.0, 1000.0, 400.0, 40.0, 0.7, 0.3, 0.03),
            (1300.0, 1500.0, 500.0, 75.0, 1.0, 0.4, 0.05),
        ),
        (
            1.2881396800e03,
            1.4910792535e03,
            5.8323836877e02,
            7.5416644291e01,
            9.6629502864e-01,
            3.9797285797e-01,
            4.9727297349e-02,
        ),
        5.6427082397e03,
        certified=True,
    ),
    "mgh09": LSProblem(
        "mgh09",
        _mgh09,
        ((25.0, 39.0, 41.5, 39.0), (0.25, 0.39, 0.415, 0.39)),
        (1.9280693458e-01, 1.9128232873e-01, 1.2305650693e-01, 1.3606233068e-01),
        3.0750560385e-04,
        certified=True,
    ),
}


class CallableEvaluator:
    """Minimal evaluator over a residual function, with forward differences.

    Mirrors the counting of
    :class:`~optiland.optimization.evaluators.FiniteDiffEvaluator`: ``nfev``
    includes one evaluation per Jacobian column.
    """

    def __init__(self, fun: Callable[[np.ndarray], np.ndarray], x0: np.ndarray):
        self.fun = fun
        self.x = np.asarray(x0, dtype=float).copy()
        self.n_vars = self.x.size
        self.backend = "numpy"
        self.nfev = 0
        self.njev = 0

    def read_x(self) -> np.ndarray:
        return self.x.copy()

    def write_x(self, x: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=float).copy()

    def residuals(self, x: np.ndarray) -> np.ndarray:
        self.nfev += 1
        with np.errstate(all="ignore"):
            return np.asarray(self.fun(np.asarray(x, dtype=float)), dtype=float)

    def value(self, x: np.ndarray) -> float:
        r = self.residuals(x)
        return float(r @ r)

    def jacobian(self, x: np.ndarray, r0: np.ndarray | None = None) -> np.ndarray:
        self.njev += 1
        x = np.asarray(x, dtype=float)
        r0 = self.residuals(x) if r0 is None else r0
        J = np.empty((r0.size, x.size))
        eps = np.finfo(float).eps ** 0.5
        for j in range(x.size):
            h = eps * max(abs(x[j]), 1.0)
            xp = x.copy()
            xp[j] += h
            h = xp[j] - x[j]
            J[:, j] = (self.residuals(xp) - r0) / h
        return J
