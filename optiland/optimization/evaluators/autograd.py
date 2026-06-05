"""AutogradEvaluator — torch autograd path

Computes value, gradient, and Jacobian using PyTorch's autograd engine.
Uses the stateful forward pass: parameters are pushed into surfaces via
``var.update(param)`` each call, then ``sum_squared()`` / ``weighted_residuals()``
are evaluated through the existing trace pipeline.

**Functional Jacobian fast path** (experimental):
The ``"functional"`` and ``"compiled"`` Jacobian modes expose a closure
``f(x_tensor) -> r`` that is differentiable via ``torch.autograd``.

*  ``"functional"`` — uses ``torch.func.jacrev`` when the closure is
   vmap-compatible (all-tensor-op trace); otherwise falls back to
   ``torch.autograd.functional.jacobian`` which handles the stateful
   attribute-write path correctly at the cost of m sequential backward passes.
*  ``"compiled"`` — applies ``torch.compile`` to the forward closure before
   computing the Jacobian.  The JIT-compiled graph reduces per-pass overhead
   and provides speedup on long traces.

**Experimental:** ``"functional"`` and ``"compiled"`` modes are opt-in and
not covered by the all-backend correctness guarantee.  ``"stateful"`` is the
verified default.

The ``jacobian_mode`` constructor argument selects the path.

Kramer Harrison, 2026
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Literal

import optiland.backend as be
from optiland.optimization.evaluators._param_sync import TorchParameterSync
from optiland.optimization.failure import (
    OptimizationFailure,
    is_nonfinite,
    normalize_failure_mode,
    penalty_merit,
)
from optiland.optimization.state import EvalCapability

if TYPE_CHECKING:
    from collections.abc import Callable

    from optiland.optimization.problem import OptimizationProblem

try:
    import torch
    import torch.nn as nn
except (ImportError, ModuleNotFoundError):
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

_PROVIDES = frozenset(
    {
        EvalCapability.VALUE,
        EvalCapability.RESIDUALS,
        EvalCapability.GRADIENT,
        EvalCapability.JACOBIAN,
    }
)

JacobianMode = Literal["stateful", "functional", "compiled"]


class AutogradEvaluator:
    """Evaluator for the torch backend using PyTorch autograd.

    Delegates parameter ↔ surface synchronisation to :class:`TorchParameterSync`
    The evaluator owns only computation logic.

    Args:
        problem: The optimization problem to wrap.
        jacobian_mode: Selects the Jacobian computation strategy.

            * ``"stateful"`` (default, verified) — row-by-row
              ``torch.autograd.grad`` with ``retain_graph``.  Always correct;
              no extra dependencies.
            * ``"functional"`` (**experimental**) — builds a differentiable
              closure ``f(x) -> r`` and dispatches to ``torch.func.jacrev``
              when the trace is vmap-compatible; otherwise falls back to
              ``torch.autograd.functional.jacobian``.
            * ``"compiled"`` (**experimental**) — same as ``"functional"`` but
              first applies ``torch.compile`` to the forward closure for
              JIT-compilation of the entire forward pass.
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        jacobian_mode: JacobianMode = "stateful",
        on_failure: str = "reject",
    ):
        if torch is None:
            raise ImportError("torch is required for AutogradEvaluator")
        assert be.get_backend() == "torch", (
            "AutogradEvaluator requires the 'torch' backend. "
            "Call be.set_backend('torch') first."
        )
        assert not any(isinstance(var.value, str) for var in problem.variables), (
            "Glass/material variables are not supported by AutogradEvaluator. "
            "Use GlassExpert directly."
        )
        self.problem = problem
        self.backend: str = "torch"
        self.n_vars: int = len(list(problem.variables))
        self.provides: frozenset[EvalCapability] = _PROVIDES
        self.jacobian_mode: JacobianMode = jacobian_mode
        self._on_failure = normalize_failure_mode(on_failure)
        self._ref_merit: float = 1.0

        # Shared param↔surface sync core
        self._sync = TorchParameterSync(problem)

        # Cached forward closure (built lazily)
        self._forward_fn: Callable | None = None
        self._compiled_fn: Callable | None = None

    # ------------------------------------------------------------------
    # Convenience proxy — exposing sync.params for legacy callers
    # ------------------------------------------------------------------

    @property
    def _params(self) -> list[Any]:
        """Proxy for tests/internal code that accessed ``_params`` directly."""
        return self._sync.params

    # ------------------------------------------------------------------
    # read / write
    # ------------------------------------------------------------------

    def read_x(self) -> Any:
        """Return current param values stacked as a 1-D detached tensor."""
        return self._sync.read_x()

    def write_x(self, x: Any) -> None:
        """Sync params from ``x``, then push into variables and update optics.

        Values are loaded via a single vectorised copy per param (no
        per-element ``.item()`` loop), then the autograd graph is
        updated via :meth:`TorchParameterSync.write_params`.
        """
        self._sync.load_x(x)
        for i, param in enumerate(self._sync.params):
            self.problem.variables[i].update(param)
        self.problem.update_optics()

    # ------------------------------------------------------------------
    # Evaluation primitives
    # ------------------------------------------------------------------

    def value(self, x: Any) -> Any:
        """Evaluate merit at ``x``; returns a backend tensor (no ``.item()``).

        Honors the configured ``on_failure`` policy: a ray-trace exception or a
        non-finite merit becomes ``NaN`` (``"reject"``), an
        :class:`OptimizationFailure` (``"raise"``), or a finite penalty
        (``"penalty"``).
        """
        self.write_x(x)
        try:
            v = self.problem.sum_squared()
        except Exception as exc:  # noqa: BLE001
            return self._handle_value_failure(exc)
        if is_nonfinite(v):
            return self._handle_value_failure(None)
        with contextlib.suppress(Exception):
            detached = v.detach() if hasattr(v, "detach") else v
            self._ref_merit = float(be.to_numpy(detached))
        return v

    def residuals(self, x: Any) -> Any:
        """Return ``weighted_residuals()`` at ``x`` as a backend tensor.

        Honors the configured ``on_failure`` policy (see :meth:`value`).
        """
        self.write_x(x)
        try:
            r = self.problem.weighted_residuals()
        except Exception as exc:  # noqa: BLE001
            return self._handle_residual_failure(exc, r=None)
        if bool(torch.any(~torch.isfinite(r))):
            return self._handle_residual_failure(None, r=r)
        return r

    # ------------------------------------------------------------------
    # Failure policy
    # ------------------------------------------------------------------

    def _handle_value_failure(self, exc: Exception | None) -> Any:
        if self._on_failure == "raise":
            raise OptimizationFailure(
                "Merit evaluation failed (non-finite or ray-trace error)."
            ) from exc
        fill = (
            penalty_merit(self._ref_merit)
            if self._on_failure == "penalty"
            else float("nan")
        )
        return torch.tensor(fill, dtype=torch.float64)

    def _handle_residual_failure(self, exc: Exception | None, r: Any) -> Any:
        if self._on_failure == "raise":
            raise OptimizationFailure(
                "Residual evaluation failed (non-finite or ray-trace error)."
            ) from exc
        # When the whole vector is unavailable (exception), return a NaN vector
        # sized from the operand count; otherwise replace only the non-finite
        # entries so the autograd graph is preserved where it is valid.
        if r is None:
            m = max(len(list(self.problem.operands)), 1)
            return torch.full((m,), float("nan"), dtype=torch.float64)
        return torch.where(torch.isfinite(r), r, torch.full_like(r, float("nan")))

    # ------------------------------------------------------------------
    # Autograd gradient
    # ------------------------------------------------------------------

    def gradient(self, x: Any) -> Any:
        """Compute d(value)/d(x) via ``backward()``.

        Clears existing gradients before the forward pass and returns a
        detached tensor of shape ``(n,)``.  Does **not** call ``.item()``
        (no device sync in the hot loop).
        """
        self._sync.load_x(x)
        for param in self._sync.params:
            if param.grad is not None:
                param.grad.zero_()

        with be.grad_mode.temporary_enable():
            self._sync.write_params()
            loss = self.problem.sum_squared()
            loss.backward()

        return torch.stack([p.grad.detach().clone() for p in self._sync.params])

    # ------------------------------------------------------------------
    # Functional forward closure (experimental)
    # ------------------------------------------------------------------

    def build_forward_fn(self) -> Callable:
        """Build a differentiable closure ``f(x_tensor) -> r``.

        **Experimental** — see class docstring.

        Returns:
            A callable ``f(x_tensor: Tensor) -> Tensor``.
        """
        vars_list = list(self.problem.variables)
        problem = self.problem

        def _fn(x_tensor: Any) -> Any:
            xs = x_tensor.unbind(0)
            for i, var in enumerate(vars_list):
                var.update(xs[i])
            problem.update_optics()
            return problem.weighted_residuals()

        return _fn

    def _get_or_build_forward_fn(self) -> Callable:
        """Return the (possibly compiled) forward closure, building lazily."""
        if self._forward_fn is None:
            self._forward_fn = self.build_forward_fn()
        return self._forward_fn

    def _get_or_build_compiled_fn(self) -> Callable:
        """Return a ``torch.compile``'d forward closure, building lazily."""
        if self._compiled_fn is None:
            fn = self._get_or_build_forward_fn()
            try:
                self._compiled_fn = torch.compile(fn)
            except Exception:
                self._compiled_fn = fn
        return self._compiled_fn

    def _jacobian_functional(self, x: Any) -> Any:
        """Jacobian via the functional closure (experimental).

        Dispatches in preference order:

        1. ``torch.func.jacrev`` — uses vmap-based batched backward passes.
        2. ``torch.autograd.functional.jacobian`` — loops m backward passes
           through the functional closure.

        After computing the Jacobian, the problem state is restored via
        ``write_x(x)`` so that subsequent evaluations see a clean state.

        Args:
            x: Parameter vector (1-D tensor or compatible).

        Returns:
            Jacobian tensor of shape ``(m, n)``.
        """
        if self.jacobian_mode == "compiled":
            fn = self._get_or_build_compiled_fn()
        else:
            fn = self._get_or_build_forward_fn()

        params = self._sync.params
        dtype = params[0].dtype if params else torch.float64
        device = params[0].device if params else torch.device("cpu")
        x_t = torch.as_tensor(
            x if not isinstance(x, torch.Tensor) else x.detach(),
            dtype=dtype,
            device=device,
        ).requires_grad_(True)

        try:
            if hasattr(torch, "func") and hasattr(torch.func, "jacrev"):
                try:
                    J = torch.func.jacrev(fn)(x_t)
                    if isinstance(J, torch.Tensor) and J.requires_grad:
                        J = J.detach()
                    return J
                except Exception:
                    pass

            from torch.autograd.functional import jacobian as _jac

            J = _jac(fn, x_t, create_graph=False, vectorize=False)
            return J.detach() if isinstance(J, torch.Tensor) and J.requires_grad else J
        finally:
            self.write_x(x)

    # ------------------------------------------------------------------
    # Autograd Jacobian — dispatcher
    # ------------------------------------------------------------------

    def jacobian(self, x: Any, r0: Any = None) -> Any:  # noqa: ARG002
        """Compute d(residuals)/d(x), shape ``(m, n)``.

        The ``r0`` argument (a pre-computed baseline residual) is accepted for
        interface parity with :class:`FiniteDiffEvaluator` but ignored — the
        autograd path computes residuals and their gradients in a single pass,
        so there is no redundant baseline to skip.

        Dispatches to the appropriate path based on ``jacobian_mode``:

        * ``"stateful"`` (default, verified) — row-by-row
          ``torch.autograd.grad`` with ``retain_graph=True`` for all but the
          last row.  Always correct.
        * ``"functional"`` (**experimental**) — functional closure +
          ``torch.func.jacrev`` (with fallback).
        * ``"compiled"`` (**experimental**) — same as ``"functional"`` but
          with the forward closure pre-compiled via ``torch.compile``.

        Args:
            x: Parameter vector (1-D tensor or compatible).

        Returns:
            Jacobian tensor of shape ``(m, n)``.
        """
        if self.jacobian_mode in ("functional", "compiled"):
            return self._jacobian_functional(x)
        return self._jacobian_stateful(x)

    def _jacobian_stateful(self, x: Any) -> Any:
        """Jacobian via row-by-row ``torch.autograd.grad`` (default stateful path).

        Returns a tensor of shape ``(m, n)``.  Each row requires one backward
        pass with ``retain_graph=True`` for all but the last row.

        Device-correct: zero-fill for ``allow_unused`` grads uses
        ``torch.zeros_like(param)`` so the result stays on the param's device.
        """
        self._sync.load_x(x)

        with be.grad_mode.temporary_enable():
            self._sync.write_params()
            r = self.problem.weighted_residuals()

        m = r.shape[0]
        params = self._sync.params
        rows = []
        for i in range(m):
            grads = torch.autograd.grad(
                r[i], params, retain_graph=(i < m - 1), allow_unused=True
            )
            row = torch.cat(
                [
                    g.reshape(-1)
                    if g is not None
                    else torch.zeros_like(param).reshape(-1)
                    for g, param in zip(grads, params, strict=True)
                ]
            )
            rows.append(row)
        return torch.stack(rows)  # (m, n)
