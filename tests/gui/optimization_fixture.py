"""Importable deterministic optimizers for process ownership/race tests."""

from __future__ import annotations

import time
from types import SimpleNamespace

from optiland.optimization.optimizer.scipy import OptimizerGeneric


class DelayedOptimizer(OptimizerGeneric):
    def optimize(self, delay=0.25, **kwargs):
        time.sleep(delay)
        self._fun([self.problem.variables[0].variable.scale(8.0)])
        return SimpleNamespace(success=True, message="fixture converged", nit=1, nfev=1)


class FailedOptimizer(OptimizerGeneric):
    def optimize(self, **kwargs):
        raise RuntimeError("intentional optimization failure")


class NonFiniteOptimizer(OptimizerGeneric):
    """An importable optimizer returning a malformed candidate despite success."""

    def optimize(self, **kwargs):
        self.problem.variables[0].update(float("nan"))
        return SimpleNamespace(success=True, message="incorrect success")
