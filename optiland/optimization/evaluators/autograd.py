"""AutogradEvaluator — torch autograd path

Computes value, gradient, and Jacobian using PyTorch's autograd engine.
Uses the stateful forward pass (D4): parameters are pushed into surfaces via
``var.update(param)`` each call, then ``sum_squared()`` / ``weighted_residuals()``
are evaluated through the existing trace pipeline.

**Phase 4 — functional Jacobian fast path** (D4):
The ``"functional"`` and ``"compiled"`` Jacobian modes expose a closure
``f(x_tensor) -> r`` that is differentiable via ``torch.autograd``.

*  ``"functional"`` — uses ``torch.func.jacrev`` when the closure is
   vmap-compatible (all-tensor-op trace); otherwise falls back to
   ``torch.autograd.functional.jacobian`` which handles the stateful
   attribute-write path correctly at the cost of m sequential backward passes.
*  ``"compiled"`` — applies ``torch.compile`` to the forward closure before
   computing the Jacobian.  The JIT-compiled graph reduces per-pass overhead
   and provides speedup on long traces.

The ``jacobian_mode`` constructor argument selects the path.  The default
remains ``"stateful"`` (row-by-row autograd.grad) for backward compatibility;
switch to ``"functional"`` or ``"compiled"`` for the Phase 4 fast path.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import optiland.backend as be
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

    Maintains a list of ``nn.Parameter`` objects (one per variable) that
    serve as the leaf tensors for gradient computation.  Values are propagated
    into the problem's variables via ``var.update(param)`` before each trace,
    mirroring ``TorchBaseOptimizer``'s approach.

    Args:
        problem: The optimization problem to wrap.
        jacobian_mode: Selects the Jacobian computation strategy.

            * ``"stateful"`` (default) — row-by-row ``torch.autograd.grad``
              with ``retain_graph``.  Always correct; no extra dependencies.
            * ``"functional"`` — builds a differentiable closure
              ``f(x) -> r`` and dispatches to ``torch.func.jacrev`` when
              the trace is vmap-compatible; otherwise falls back to
              ``torch.autograd.functional.jacobian`` (same m backward passes
              but cleaner and ready for future vmap support).
            * ``"compiled"`` — same as ``"functional"`` but first applies
              ``torch.compile`` to the forward closure for JIT-compilation of
              the entire forward pass (variable writes + ray trace + residuals),
              reducing per-evaluation overhead on long traces.
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        jacobian_mode: JacobianMode = "stateful",
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

        # Leaf tensors — the canonical "x" for this evaluator
        if not be.grad_mode.requires_grad:
            be.grad_mode.enable()
        self._params: list[nn.Parameter] = [
            nn.Parameter(be.array(float(var.value))) for var in problem.variables
        ]

        # Phase 4: cached forward closure (built lazily)
        self._forward_fn: Callable | None = None
        self._compiled_fn: Callable | None = None

    # ------------------------------------------------------------------
    # read / write
    # ------------------------------------------------------------------

    def read_x(self) -> Any:
        """Return current param values stacked as a 1-D detached tensor."""
        return torch.stack([p.detach().clone() for p in self._params])

    def write_x(self, x: Any) -> None:
        """Sync params from ``x``, then push into variables and update optics.

        ``x`` may be a 1-D tensor or a numpy array; values are written as
        scalars so the params remain the leaf nodes.
        """
        with torch.no_grad():
            for i, param in enumerate(self._params):
                v = x[i]
                val = float(v.item()) if hasattr(v, "item") else float(v)
                param.data.fill_(val)
        for i, param in enumerate(self._params):
            self.problem.variables[i].update(param)
        self.problem.update_optics()

    # ------------------------------------------------------------------
    # Evaluation primitives
    # ------------------------------------------------------------------

    def value(self, x: Any) -> Any:
        """Evaluate merit at ``x``; returns a backend tensor (no ``.item()``)."""
        self.write_x(x)
        return self.problem.sum_squared()

    def residuals(self, x: Any) -> Any:
        """Return ``weighted_residuals()`` at ``x`` as a backend tensor."""
        self.write_x(x)
        return self.problem.weighted_residuals()

    # ------------------------------------------------------------------
    # Autograd gradient
    # ------------------------------------------------------------------

    def gradient(self, x: Any) -> Any:
        """Compute d(value)/d(x) via ``backward()``.

        Clears existing gradients before the forward pass and returns a
        detached tensor of shape ``(n,)``.  Does **not** call ``.item()``
        (no device sync in the hot loop — D9).
        """
        with torch.no_grad():
            for i, param in enumerate(self._params):
                v = x[i]
                val = float(v.item()) if hasattr(v, "item") else float(v)
                param.data.fill_(val)
        for param in self._params:
            if param.grad is not None:
                param.grad.zero_()

        with be.grad_mode.temporary_enable():
            for i, param in enumerate(self._params):
                self.problem.variables[i].update(param)
            self.problem.update_optics()
            loss = self.problem.sum_squared()
            loss.backward()

        return torch.stack([p.grad.detach().clone() for p in self._params])

    # ------------------------------------------------------------------
    # Phase 4 — functional forward closure
    # ------------------------------------------------------------------

    def build_forward_fn(self) -> Callable:
        """Build a differentiable closure ``f(x_tensor) -> r`` (Phase 4).

        The returned function takes a 1-D tensor ``x_tensor`` of shape
        ``(n,)`` and returns the weighted residual vector ``r`` of shape
        ``(m,)``.  When ``x_tensor`` has ``requires_grad=True``, ``r``
        depends on ``x_tensor`` through the autograd graph:

        .. code-block:: text

            x_tensor[i]  →  inverse_scale(x_tensor[i])
                         →  surface.attr = unscaled        (Python attr write)
                         →  ray trace reads surface.attr   (tensor op)
                         →  weighted_residuals()  →  r

        The closure is differentiable via ``torch.autograd`` (both
        ``autograd.grad`` and ``autograd.functional.jacobian``).  It is
        *not* vmap-compatible in the general case because surface attribute
        writes are Python-level side effects that ``torch.func`` transforms
        cannot functionalize.  ``torch.func.jacrev`` with a vmap-capable
        trace is therefore attempted opportunistically; a fallback to
        ``torch.autograd.functional.jacobian`` handles the stateful case.

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
                # torch.compile not supported in this environment; fall back
                self._compiled_fn = fn
        return self._compiled_fn

    def _jacobian_functional(self, x: Any) -> Any:
        """Jacobian via the functional closure (Phase 4 fast path).

        Dispatches in preference order:

        1. ``torch.func.jacrev`` — uses vmap-based batched backward passes.
           Fastest when the trace is vmap-compatible (all tensor ops, no
           Python-level surface attribute mutations).
        2. ``torch.autograd.functional.jacobian`` — loops m backward passes
           through the functional closure.  Correct for the stateful case;
           same compute as the stateful default but through the unified
           closure (enabling future ``torch.compile`` on the forward graph).

        After computing the Jacobian, the problem state is restored via
        ``write_x(x)`` so that subsequent evaluations see a clean state
        (surface attributes are set to ``nn.Parameter``-derived tensors,
        not the temporary grad-enabled leaf used during differentiation).

        Args:
            x: Parameter vector (1-D tensor or compatible).

        Returns:
            Jacobian tensor of shape ``(m, n)``.
        """
        if self.jacobian_mode == "compiled":
            fn = self._get_or_build_compiled_fn()
        else:
            fn = self._get_or_build_forward_fn()

        dtype = self._params[0].dtype if self._params else torch.float64
        device = self._params[0].device if self._params else torch.device("cpu")
        x_t = torch.as_tensor(
            x if not isinstance(x, torch.Tensor) else x.detach(),
            dtype=dtype,
            device=device,
        ).requires_grad_(True)

        try:
            # --- attempt torch.func.jacrev (eliminates Python loop when vmap works) ---
            if hasattr(torch, "func") and hasattr(torch.func, "jacrev"):
                try:
                    J = torch.func.jacrev(fn)(x_t)
                    if isinstance(J, torch.Tensor) and J.requires_grad:
                        J = J.detach()
                    return J
                except Exception:
                    pass  # vmap incompatible; fall through to autograd.functional

            # --- fallback: autograd.functional.jacobian (stateful-safe) ---
            from torch.autograd.functional import jacobian as _jac

            J = _jac(fn, x_t, create_graph=False, vectorize=False)
            # _jac returns a single tensor when fn has one Tensor input
            return J.detach() if isinstance(J, torch.Tensor) and J.requires_grad else J
        finally:
            # Restore surface state: write params (nn.Parameters) back into
            # surfaces, replacing any grad-tracked leaf tensors left by fn(x_t).
            self.write_x(x)

    # ------------------------------------------------------------------
    # Autograd Jacobian — dispatcher (Phase 4 gate)
    # ------------------------------------------------------------------

    def jacobian(self, x: Any) -> Any:
        """Compute d(residuals)/d(x), shape ``(m, n)``.

        Dispatches to the appropriate path based on ``jacobian_mode``:

        * ``"stateful"`` (default) — row-by-row ``torch.autograd.grad`` with
          ``retain_graph=True`` for all but the last row.  Always correct.
        * ``"functional"`` — functional closure + ``torch.func.jacrev``
          (with fallback to ``torch.autograd.functional.jacobian``).
        * ``"compiled"`` — same as ``"functional"`` but with the forward
          closure pre-compiled via ``torch.compile``.

        Args:
            x: Parameter vector (1-D tensor or compatible).

        Returns:
            Jacobian tensor of shape ``(m, n)``.
        """
        if self.jacobian_mode in ("functional", "compiled"):
            return self._jacobian_functional(x)
        return self._jacobian_stateful(x)

    def _jacobian_stateful(self, x: Any) -> Any:
        """Jacobian via row-by-row ``torch.autograd.grad`` (D4 default).

        Returns a tensor of shape ``(m, n)``.  Each row requires one backward
        pass with ``retain_graph=True`` for all but the last row.

        This is the default stateful path (D4), guaranteed-correct for any
        trace regardless of vmap compatibility.
        """
        with torch.no_grad():
            for i, param in enumerate(self._params):
                v = x[i]
                val = float(v.item()) if hasattr(v, "item") else float(v)
                param.data.fill_(val)

        with be.grad_mode.temporary_enable():
            for i, param in enumerate(self._params):
                self.problem.variables[i].update(param)
            self.problem.update_optics()
            r = self.problem.weighted_residuals()

        m = r.shape[0]
        rows = []
        for i in range(m):
            grads = torch.autograd.grad(
                r[i], self._params, retain_graph=(i < m - 1), allow_unused=True
            )
            row = torch.cat(
                [g.reshape(-1) if g is not None else torch.zeros(1) for g in grads]
            )
            rows.append(row)
        return torch.stack(rows)  # (m, n)
