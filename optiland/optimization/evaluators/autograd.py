"""Autograd Evaluator

Residuals and Jacobians on the torch backend through PyTorch autograd, in the
verified *stateful* mode only: the parameters are pushed into the surfaces via
:class:`~optiland.optimization.evaluators._param_sync.TorchParameterSync`,
then ``problem.residual_vector()`` is evaluated through the regular trace
pipeline inside one evaluation epoch. Residuals are ``effective_weight *
delta`` (decision V7). Hard-constraint rows follow the residuals in the same
forward pass.

The Jacobian is built row by row with ``torch.autograd.grad``: ``m`` backward
passes over one forward pass. Functional and compiled modes are left to a
future spec.

Failures follow :mod:`optiland.optimization.failure`: failed rows are ``NaN``,
or :class:`OptimizationFailure` is raised with ``on_failure="raise"``.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import optiland.backend as be
from optiland import _epoch
from optiland.optimization.evaluators._param_sync import TorchParameterSync
from optiland.optimization.failure import (
    ConfigurationError,
    OptimizationFailure,
    describe_rows,
    failing_rows,
    normalize_failure_mode,
)

if TYPE_CHECKING:
    from optiland.optimization.problem import OptimizationProblem

try:
    import torch
except (ImportError, ModuleNotFoundError):  # pragma: no cover - torch optional
    torch = None  # type: ignore[assignment]


class AutogradEvaluator:
    """Evaluate residuals and autograd Jacobians on the torch backend.

    Args:
        problem: The optimization problem.
        on_failure: ``"reject"`` (default: failed rows become ``NaN``) or
            ``"raise"``.

    Attributes:
        nfev: Forward evaluations so far.
        njev: Jacobians computed so far.

    Raises:
        ImportError: If torch is not installed.
        ConfigurationError: If the torch backend is not active, or a variable
            is a glass/material choice.
    """

    def __init__(self, problem: OptimizationProblem, on_failure: str = "reject"):
        if torch is None:
            raise ImportError("torch is required for AutogradEvaluator")
        if be.get_backend() != "torch":
            raise ConfigurationError(
                "AutogradEvaluator requires the torch backend; call "
                "be.set_backend('torch') first."
            )
        if any(isinstance(var.value, str) for var in problem.variables):
            raise ConfigurationError(
                "Glass/material variables are not supported by autograd; use "
                "GlassExpert."
            )
        self.problem = problem
        self.backend = "torch"
        self.n_vars = len(problem.variables)
        self.on_failure = normalize_failure_mode(on_failure)
        self.nfev = 0
        self.njev = 0
        self._sync = TorchParameterSync(problem)
        self._optics = tuple({id(v.optic): v.optic for v in problem.variables}.values())
        self._constraint_cache: dict[str, tuple[bytes, Any]] = {}

    @property
    def n_constraints(self) -> int:
        """Number of hard-constraint rows."""
        return self.problem.constraints.n_rows

    def constraint_rows(self) -> tuple[Any, Any, list[str]]:
        """Equality mask, tolerances and labels of the constraint rows."""
        constraints = self.problem.constraints
        return constraints.equality_mask, constraints.tols, constraints.labels

    @property
    def params(self) -> list[Any]:
        """The ``nn.Parameter`` objects, one per variable."""
        return self._sync.params

    def read_x(self) -> Any:
        """Return the parameter values as a 1-D detached tensor."""
        return self._sync.read_x()

    def write_x(self, x: Any) -> None:
        """Load ``x`` into the parameters, push them into the variables, update."""
        self._sync.load_x(x)
        self._sync.write_params()

    def bounds(self) -> tuple[Any, Any]:
        """Lower and upper bounds of the scaled variables as numpy arrays."""
        lower = np.full(self.n_vars, -np.inf)
        upper = np.full(self.n_vars, np.inf)
        for i, var in enumerate(self.problem.variables):
            lo, hi = var.bounds
            if lo is not None:
                lower[i] = float(be.to_numpy(lo))
            if hi is not None:
                upper[i] = float(be.to_numpy(hi))
        return lower, upper

    def _forward(self, x: Any) -> Any:
        """Residuals, then constraint rows, with the graph attached."""
        self.nfev += 1
        try:
            with be.grad_mode.temporary_enable():
                self.write_x(x)
                with _epoch.evaluation_epoch(self._optics):
                    r = be.ravel(
                        self.problem.residuals_and_constraints()
                        if self.n_constraints
                        else self.problem.residual_vector()
                    )
        except _epoch.EpochViolationError:
            raise
        except Exception as exc:
            if self.on_failure == "raise":
                raise OptimizationFailure(f"Design evaluation failed: {exc}") from exc
            m = max(len(self.problem.operands), 1) + self.n_constraints
            return torch.full((m,), float("nan"), dtype=self.params[0].dtype)
        if self.on_failure == "raise":
            rows = failing_rows(be.to_numpy(r.detach()))
            if rows.size:
                raise OptimizationFailure(
                    "Non-finite residuals for operand(s) "
                    f"{describe_rows(self.problem, rows)}."
                )
        return r

    @staticmethod
    def _key(x: Any) -> bytes:
        return np.asarray(be.to_numpy(x), dtype=float).tobytes()

    def _merit_rows(self, values: Any) -> Any:
        """Drop the constraint rows of one forward pass."""
        return values[: values.shape[0] - self.n_constraints]

    def residuals(self, x: Any) -> Any:
        """Return the residuals at ``x`` as a detached tensor (``NaN`` on failure)."""
        values = self._forward(x).detach()
        if self.n_constraints:
            rows = values[values.shape[0] - self.n_constraints :]
            self._constraint_cache["values"] = (self._key(x), rows)
        return self._merit_rows(values)

    def constraints(self, x: Any) -> Any:
        """Return the signed constraint rows at ``x`` as a detached tensor.

        Reads the forward pass of :meth:`residuals` at ``x`` when it was the
        last one, else runs one.
        """
        cached = self._constraint_cache.get("values")
        if cached is None or cached[0] != self._key(x):
            self.residuals(x)
            cached = self._constraint_cache["values"]
        return cached[1]

    def constraint_jacobian(self, x: Any) -> Any:
        """Jacobian of the constraint rows at ``x``, shape ``(c, n)``.

        Reads the rows of the last :meth:`jacobian` at ``x`` when it is the
        same point, else computes them.
        """
        cached = self._constraint_cache.get("jacobian")
        if cached is None or cached[0] != self._key(x):
            self.jacobian(x)
            cached = self._constraint_cache["jacobian"]
        return cached[1]

    def value(self, x: Any) -> Any:
        """Return the merit ``sum(r ** 2)`` at ``x`` as a detached 0-d tensor."""
        r = self.residuals(x)
        return torch.sum(r * r)

    def jacobian(self, x: Any, r0: Any = None) -> Any:  # noqa: ARG002
        """Autograd Jacobian of the residuals at ``x``, shape ``(m, n)``.

        ``r0`` is accepted for protocol parity and ignored: the forward pass
        that builds the graph also computes the residuals.
        """
        self.njev += 1
        r = self._forward(x)
        params = self.params
        rows = []
        for i in range(r.shape[0]):
            if not r.requires_grad or not torch.isfinite(r[i]):
                rows.append(torch.zeros(len(params), dtype=params[0].dtype))
                continue
            grads = torch.autograd.grad(
                r[i], params, retain_graph=i < r.shape[0] - 1, allow_unused=True
            )
            rows.append(
                torch.stack(
                    [
                        g.reshape(()) if g is not None else torch.zeros_like(p)
                        for g, p in zip(grads, params, strict=True)
                    ]
                )
            )
        J = torch.stack(rows) if rows else torch.zeros((0, len(params)))
        J = torch.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.n_constraints:
            return J
        constraint_rows = J[J.shape[0] - self.n_constraints :]
        self._constraint_cache["jacobian"] = (self._key(x), constraint_rows)
        return self._merit_rows(J)

    def gradient(self, x: Any) -> Any:
        """Merit gradient ``d(sum r^2)/dx`` by one backward pass."""
        for param in self.params:
            param.grad = None
        r = self._merit_rows(self._forward(x))
        loss = torch.sum(r * r)
        loss.backward()
        return torch.stack(
            [
                p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                for p in self.params
            ]
        )
