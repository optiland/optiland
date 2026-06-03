"""NullSpaceStrategy — exact CODE V-style constraint projection.

Enforces equality constraints ``h(x) = 0`` and active inequality constraints
via null-space (tangent-space) projection.  Each LM step is:

1. **Projection**: project the proposed step onto the null space of the active
   constraint Jacobian ``A = ∂h/∂x`` via SVD, so the move stays on (or near)
   the constraint manifold.
2. **Restoration**: Newton correction to drive residual ``h(x_new)`` to zero.

**Inequality constraints** ``g(x) ≤ 0`` are included in the active set
whenever ``g(x) ≥ -active_tol`` (violated or nearly-active); inactive
constraints are ignored each step.

The constraint Jacobian is user-supplied or computed by forward finite
differences.  Rank deficiency is handled via SVD truncation.

Kramer Harrison, 2026
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from optiland.optimization.state import OptimizationState

# Type alias for a constraint function x -> scalar or array
ConstraintFn = Callable[[np.ndarray], np.ndarray | float]


class NullSpaceStrategy:
    """Exact constraint enforcement via null-space projection + Newton restoration.

    Integrates with ``LevenbergMarquardt`` (and ``GaussNewton``) via the
    ``PER_STEP_CONSTRAINTS`` capability: at each accepted step, the proposed
    ``x_new`` is projected onto the feasible manifold before it is committed.

    Args:
        equality_fn: Callable ``h(x) -> 1-D array`` of length ``p``.  Must
            return **zero** on the feasible manifold.  Pass ``None`` if only
            inequality constraints are used.
        jacobian_fn: Callable ``dh_dx(x) -> 2-D array (p, n)`` for the
            constraint Jacobian.  If ``None``, the Jacobian is estimated by
            forward finite differences with step size ``fd_step``.
        inequality_fns: Optional list of callables ``g_i(x) -> scalar`` where
            ``g_i(x) ≤ 0`` defines the constraint.  Each active inequality
            (``g_i(x) ≥ -active_tol``) is included in the projection as an
            equality constraint.
        active_tol: An inequality is active when ``g(x) ≥ -active_tol``.
            Default ``1e-6``.
        restore_max_iter: Maximum Newton restoration iterations.  Default 10.
        restore_tol: Feasibility restoration convergence tolerance (L2 norm of
            constraint residuals).  Default ``1e-8``.
        rank_tol: Relative singular-value threshold for rank determination.
            Singular values ``s_i < rank_tol * s_max`` are treated as zero.
            Default ``1e-10``.
        fd_step: Forward finite-difference step size for the constraint
            Jacobian when ``jacobian_fn`` is ``None``.  Default ``1e-7``.
    """

    def __init__(
        self,
        equality_fn: ConstraintFn | None = None,
        jacobian_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        inequality_fns: list[ConstraintFn] | None = None,
        active_tol: float = 1e-6,
        restore_max_iter: int = 10,
        restore_tol: float = 1e-8,
        rank_tol: float = 1e-10,
        fd_step: float = 1e-7,
    ) -> None:
        if equality_fn is None and not inequality_fns:
            raise ValueError(
                "NullSpaceStrategy requires at least one equality constraint "
                "(equality_fn) or inequality constraint (inequality_fns)."
            )
        self._equality_fn = equality_fn
        self._jacobian_fn = jacobian_fn
        self._inequality_fns: list[ConstraintFn] = list(inequality_fns or [])
        self._active_tol = active_tol
        self._restore_max_iter = restore_max_iter
        self._restore_tol = restore_tol
        self._rank_tol = rank_tol
        self._fd_step = fd_step

        # Set by prepare(); not required for standalone use
        self._n_vars: int | None = None

    # ------------------------------------------------------------------
    # ConstraintStrategy protocol
    # ------------------------------------------------------------------

    def prepare(self, evaluator: Any, variables: Any) -> None:
        """Record variable count for size validation at run time.

        Args:
            evaluator: The active evaluator (unused here).
            variables: Variable list from the optimization problem.
        """
        self._n_vars = len(list(variables))

    def apply_to_step(self, x_proposed: Any, state: OptimizationState) -> np.ndarray:
        """Project ``x_proposed`` onto the feasible manifold.

        Computes the active constraint Jacobian at ``state.x``, projects the
        proposed step onto its null space, then applies Newton restoration to
        re-establish feasibility.

        Args:
            x_proposed: Proposed next iterate (after the LM direction).
            state: Current optimization state (provides ``state.x``).

        Returns:
            np.ndarray: Feasibility-restored parameter vector.
        """
        x_curr = np.asarray(state.x, dtype=float)
        x_prop = np.asarray(x_proposed, dtype=float)

        # Collect active constraints at the current point
        h_curr = self._active_residuals(x_curr)
        if h_curr.size == 0:
            return x_prop  # no active constraints — pass through

        # Constraint Jacobian A (p × n) at current point
        A = self._active_jacobian(x_curr, h_curr)
        p, n = A.shape

        # ---- Null-space projection via SVD --------------------------------
        Z = _null_space_basis(A, self._rank_tol)

        dx = x_prop - x_curr
        if Z.shape[1] == 0:
            # No null space (fully constrained); keep x_curr
            x_new = x_curr.copy()
        else:
            dx_null = Z @ (Z.T @ dx)
            x_new = x_curr + dx_null

        # ---- Newton restoration -------------------------------------------
        x_new = self._restore(x_new)

        return x_new

    def to_scipy(self) -> list | None:
        """Return ``None`` — null-space projection is not a SciPy constraint.

        Returns:
            None: This strategy has no SciPy representation.
        """
        return None

    def is_feasible(self, x: Any, tol: float = 1e-8) -> bool:
        """Return True if all equality constraints are satisfied within ``tol``.

        Args:
            x: Parameter vector to check.
            tol: L2 tolerance on the equality constraint residuals.

        Returns:
            bool: Whether ``||h(x)||₂ ≤ tol``.
        """
        x = np.asarray(x, dtype=float)
        h = self._equality_residuals(x)
        if h.size == 0:
            return True
        return bool(np.linalg.norm(h) <= tol)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _equality_residuals(self, x: np.ndarray) -> np.ndarray:
        """Evaluate equality constraints h(x)."""
        if self._equality_fn is None:
            return np.empty(0)
        return np.atleast_1d(np.asarray(self._equality_fn(x), dtype=float))

    def _active_residuals(self, x: np.ndarray) -> np.ndarray:
        """Return residuals for equality + active inequality constraints."""
        parts: list[np.ndarray] = []
        eq = self._equality_residuals(x)
        if eq.size:
            parts.append(eq)
        for g_fn in self._inequality_fns:
            g = float(g_fn(x))
            if g >= -self._active_tol:
                parts.append(np.array([g]))
        return np.concatenate(parts) if parts else np.empty(0)

    def _active_jacobian(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """Return the constraint Jacobian (p × n) for the active set."""
        if self._jacobian_fn is not None and len(h) == len(self._equality_residuals(x)):
            # User-supplied Jacobian covers equality only; use it when no
            # active inequalities add extra rows
            eq_len = len(self._equality_residuals(x))
            if len(h) == eq_len:
                return np.atleast_2d(np.asarray(self._jacobian_fn(x), dtype=float))
        # Fall back to FD for the full active constraint vector
        return self._fd_jacobian(self._active_residuals, x)

    def _fd_jacobian(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
    ) -> np.ndarray:
        """Forward finite-difference Jacobian of fn at x."""
        h0 = fn(x)
        p = len(h0)
        n = len(x)
        J = np.zeros((p, n), dtype=float)
        for i in range(n):
            x_p = x.copy()
            x_p[i] += self._fd_step
            J[:, i] = (fn(x_p) - h0) / self._fd_step
        return J

    def _restore(self, x: np.ndarray) -> np.ndarray:
        """Newton restoration: drive active constraint residuals to zero.

        Iterates: ``x ← x − A.T (A A.T)^{-1} h(x)``
        """
        for _ in range(self._restore_max_iter):
            h = self._active_residuals(x)
            if h.size == 0 or np.linalg.norm(h) <= self._restore_tol:
                break
            A = self._fd_jacobian(self._active_residuals, x)
            AAt = A @ A.T
            try:
                alpha = np.linalg.solve(AAt, h)
            except np.linalg.LinAlgError:
                alpha, _, _, _ = np.linalg.lstsq(AAt, h, rcond=None)
            x = x - A.T @ alpha
        return x


# ---------------------------------------------------------------------------
# Pure helper — exported so tests can verify null-space properties
# ---------------------------------------------------------------------------


def _null_space_basis(A: np.ndarray, rank_tol: float = 1e-10) -> np.ndarray:
    """Return an orthonormal basis for the null space of matrix A (p × n).

    Uses SVD: the right singular vectors corresponding to singular values below
    ``rank_tol * s_max`` span the null space of A.

    Args:
        A: Matrix of shape (p, n).
        rank_tol: Relative threshold for singular-value truncation.

    Returns:
        np.ndarray: Basis matrix of shape (n, n-rank) whose columns are the
            null-space vectors.  Returns an (n, n) identity if p == 0.
    """
    p, n = A.shape
    if p == 0:
        return np.eye(n)

    # Full SVD: A = U S V^T,  V is (n, n)
    _, s, Vt = np.linalg.svd(A, full_matrices=True)

    # Determine rank with relative threshold
    s_max = s[0] if len(s) > 0 and s[0] > 0 else 0.0
    if s_max == 0.0:
        return np.eye(n)  # A is zero matrix; null space is all of R^n

    rank = int(np.sum(s > rank_tol * s_max))
    # Null space: rows rank..n of Vt (columns of V)
    Z = Vt[rank:, :].T  # (n, n - rank)
    return Z
