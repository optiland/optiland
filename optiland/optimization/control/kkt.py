"""KKTController — KKT active-set constrained Levenberg-Marquardt step (§4.3).

When an :class:`OptimizationProblem` declares hard constraints
(``add_constraint``), the native LM / Gauss-Newton optimizers use this
controller instead of :class:`LevenbergController`.  Each accepted step solves a
**damped KKT system**

    ┌ JᵀJ + λD   Aᵀ ┐ ┌ dx ┐   ┌ −Jᵀr ┐
    │               │ │    │ = │       │
    └ A           0 ┘ └ μ  ┘   └  b    ┘

where ``A = [A_eq; A_W]`` stacks the equality-constraint Jacobian and the
Jacobian rows of the *working set* ``W`` of active inequalities/bounds, and
``b = −[c; g_W]`` drives the linearized constraints toward feasibility.

Highlights (per the spec):

* **Active set** (D5): inequalities and bounds are unified as ``g(x) ≤ 0`` rows
  with a multiplier-sign **release test** (drop rows whose multiplier is the
  wrong sign) and an **add test** (activate rows the step would violate).
* **Graceful degrade** (D18): a singular / rank-deficient KKT matrix falls back
  to a least-squares (SVD) solve so inconsistent active sets do not crash.
* **Globalization**: the step is accepted on an L1 **penalty merit**
  ``φ(x) = Σrᵢ² + σ·‖[c; g_W₊]‖₁`` (``σ`` from ``max|μ|``) with Levenberg
  damping; non-finite trials and penalty increases reject and raise λ.
* **Diagnostics** (D12): Lagrange multipliers, active set, and constraint
  feasibility are written onto the state each step.

The linear algebra runs in NumPy, matching :class:`LevenbergController` (the
native stepped path converts ``J`` / ``r`` to NumPy via ``be.to_numpy``).
``be.linalg.qr`` / ``be.linalg.cholesky`` are available for an on-device torch
extension (future work).

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import StepOutcome

if TYPE_CHECKING:
    from optiland.optimization.control.base import StepInfo
    from optiland.optimization.state import OptimizationState


class KKTController:
    """KKT active-set constrained Levenberg-Marquardt step controller.

    Args:
        lam_init: Initial damping λ₀.
        lam_max: λ ceiling above which the step is declared failed.
        min_diag: Floor for Moré diagonal damping entries.
        max_trials: Maximum inner λ trials per accepted step.
        active_tol: An inequality with ``g >= −active_tol`` is considered
            near-active and seeded into the working set.
        max_working_set_changes: Cap on release/add iterations per step (guards
            against active-set chatter).
        linear_solver: ``"normal"`` (default, symmetric-indefinite KKT solve) or
            ``"qr"`` (range-space QR; falls back to the normal solve here).
    """

    def __init__(
        self,
        lam_init: float = 1e-3,
        lam_max: float = 1e16,
        min_diag: float = 1e-8,
        max_trials: int = 20,
        active_tol: float = 1e-8,
        max_working_set_changes: int = 20,
        linear_solver: str = "normal",
    ) -> None:
        self._lam_init = lam_init
        self._lam_max = lam_max
        self._min_diag = min_diag
        self._max_trials = max_trials
        self._active_tol = active_tol
        self._max_ws = max_working_set_changes
        self._linear_solver = linear_solver
        # Bound by the optimizer at initialize() time.
        self._evaluator: Any = None
        self._constraints: Any = None

    # ------------------------------------------------------------------
    # Binding + lifecycle
    # ------------------------------------------------------------------

    def bind(self, evaluator: Any, constraints_manager: Any) -> None:
        """Receive the evaluator and constraint manager from the optimizer."""
        self._evaluator = evaluator
        self._constraints = constraints_manager

    def reset(self, state: OptimizationState) -> None:
        state.scratch["lambda"] = self._lam_init
        state.scratch["nu"] = 2.0

    # ------------------------------------------------------------------
    # Constraint helpers
    # ------------------------------------------------------------------

    def _c_vector(self) -> np.ndarray:
        import optiland.backend as be

        return np.asarray(be.to_numpy(self._constraints.c_vector()), dtype=float)

    def _g_vector(self) -> np.ndarray:
        import optiland.backend as be

        return np.asarray(be.to_numpy(self._constraints.g_vector()), dtype=float)

    def _penalty_merit(self, x_trial: Any, value_fn, sigma: float) -> float:
        """φ(x) = F(x) + σ·(Σ|c| + Σ max(0, g)) evaluated at ``x_trial``."""
        f = value_fn(x_trial)  # also writes x_trial into the optic
        if not math.isfinite(float(f)):
            return float("inf")
        c = self._c_vector()
        g = self._g_vector()
        l1 = float(np.sum(np.abs(c))) + float(np.sum(np.maximum(0.0, g)))
        return float(f) + sigma * l1

    # ------------------------------------------------------------------
    # KKT solve
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_kkt(
        H: np.ndarray, A: np.ndarray, rhs_top: np.ndarray, rhs_bot: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve the symmetric KKT system; SVD-lstsq fallback on singularity."""
        n = H.shape[0]
        k = A.shape[0]
        if k == 0:
            try:
                dx = np.linalg.solve(H, rhs_top)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(H, rhs_top, rcond=None)[0]
            return dx, np.zeros(0)
        K = np.zeros((n + k, n + k))
        K[:n, :n] = H
        K[:n, n:] = A.T
        K[n:, :n] = A
        rhs = np.concatenate([rhs_top, rhs_bot])
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        return sol[:n], sol[n:]

    def _active_set_solve(
        self,
        H: np.ndarray,
        neg_grad: np.ndarray,
        A_eq: np.ndarray,
        c: np.ndarray,
        A_ineq: np.ndarray,
        g: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
        """Run the active-set loop, returning ``(dx, mu_eq, mu_W, working_set)``."""
        q = A_ineq.shape[0]
        # Seed the working set with violated / near-active inequalities.
        working = [j for j in range(q) if g[j] >= -self._active_tol]

        dx = np.zeros(H.shape[0])
        mu_eq = np.zeros(A_eq.shape[0])
        mu_W = np.zeros(0)

        for _ in range(self._max_ws + 1):
            if working:
                A_W = A_ineq[working, :]
                b = -np.concatenate([c, g[working]])
                A = np.vstack([A_eq, A_W]) if A_eq.shape[0] else A_W
            else:
                A = A_eq
                b = -c if A_eq.shape[0] else np.zeros(0)

            dx, mu = self._solve_kkt(H, A, neg_grad, b)
            p = A_eq.shape[0]
            mu_eq = mu[:p]
            mu_W = mu[p:]

            # Release test: drop the most wrong-signed active multiplier (μ_W<0
            # means the inequality is not truly binding).
            released = False
            if working:
                min_idx = int(np.argmin(mu_W))
                if mu_W[min_idx] < -self._active_tol:
                    working.pop(min_idx)
                    released = True
            if released:
                continue

            # Add test: activate the most-violated inactive inequality the step
            # would push past feasibility (linearized g + A·dx > tol).
            if q:
                g_pred = g + A_ineq @ dx
                inactive = [j for j in range(q) if j not in working]
                if inactive:
                    viol = [(g_pred[j], j) for j in inactive]
                    worst_val, worst_j = max(viol, key=lambda t: t[0])
                    if worst_val > self._active_tol:
                        working.append(worst_j)
                        working.sort()
                        continue
            break

        return dx, mu_eq, mu_W, working

    # ------------------------------------------------------------------
    # StepController interface
    # ------------------------------------------------------------------

    def transform(
        self, direction: Any, info: StepInfo, state: OptimizationState
    ) -> StepOutcome:
        """Compute one accepted constrained LM step via the active-set KKT solve."""
        import optiland.backend as be

        J = np.asarray(be.to_numpy(info.jacobian), dtype=float)
        r = np.asarray(be.to_numpy(info.residuals), dtype=float)
        x = np.asarray(state.x, dtype=float)

        JTJ = J.T @ J
        JTr = J.T @ r
        d = np.maximum(np.abs(np.diag(JTJ)), self._min_diag)

        # Constraint Jacobians at the current point (system is at x).
        self._evaluator.write_x(x)
        A_eq = np.asarray(self._constraints.equality_jacobian(self._evaluator), float)
        A_ineq = np.asarray(
            self._constraints.inequality_jacobian(self._evaluator), float
        )
        self._evaluator.write_x(x)
        c = self._c_vector()
        g = self._g_vector()

        # Diagnostics: gradient + conditioning.
        state.grad_norm = float(np.linalg.norm(2.0 * JTr))

        lam = state.scratch.get("lambda", self._lam_init)
        nu = state.scratch.get("nu", 2.0)
        current_val = _to_float(state.value)

        n_trials = 0
        while n_trials < self._max_trials and lam <= self._lam_max:
            H = JTJ + lam * np.diag(d)
            dx, mu_eq, mu_W, working = self._active_set_solve(
                H, -JTr, A_eq, c, A_ineq, g
            )
            n_trials += 1

            # Penalty weight σ from the largest multiplier magnitude.
            mu_all = (
                np.concatenate([mu_eq, mu_W])
                if (mu_eq.size or mu_W.size)
                else (np.zeros(0))
            )
            sigma = max(1.0, float(np.max(np.abs(mu_all))) if mu_all.size else 1.0)

            x_trial = x + dx
            # Current penalty merit (constraints already at x from write_x above).
            phi0 = current_val + sigma * (
                float(np.sum(np.abs(c))) + float(np.sum(np.maximum(0.0, g)))
            )
            phi_trial = self._penalty_merit(x_trial, info.value_fn, sigma)

            if (not math.isfinite(phi_trial)) or phi_trial >= phi0:
                lam *= nu
                nu *= 2.0
                continue

            # Accept.
            lam = max(lam * (1.0 / 3.0), 1e-15)
            state.scratch["lambda"] = lam
            state.scratch["nu"] = 2.0

            # Diagnostics on the accepted state (optic currently at x_trial).
            self._write_diagnostics(state, mu_eq, mu_W, working)
            # value_fn left the optic at x_trial; record the true merit there.
            trial_value = float(be.to_numpy(self._evaluator.value(x_trial)))
            return StepOutcome(
                delta_x=dx,
                accepted=True,
                info={"n_trials": n_trials, "trial_value": trial_value},
            )

        # All trials rejected. Refresh feasibility but keep the multipliers /
        # active set from the last accepted step (do not wipe diagnostics).
        state.scratch["lambda"] = lam
        state.scratch["nu"] = nu
        self._evaluator.write_x(x)
        state.scratch["constraint_report"] = self._constraints.report()
        state.scratch["max_constraint_violation"] = self._constraints.max_violation()
        return StepOutcome(delta_x=None, accepted=False, info={"n_trials": n_trials})

    def _write_diagnostics(
        self,
        state: OptimizationState,
        mu_eq: np.ndarray,
        mu_W: np.ndarray,
        working: list[int],
    ) -> None:
        eq_labels = self._constraints.labels("equality")
        ineq_labels = self._constraints.labels("inequality")
        multipliers: dict[str, float] = {}
        for lbl, m in zip(eq_labels, mu_eq.tolist(), strict=False):
            multipliers[lbl] = float(m)
        active_labels: list[str] = list(eq_labels)
        for w, m in zip(working, mu_W.tolist(), strict=False):
            if 0 <= w < len(ineq_labels):
                multipliers[ineq_labels[w]] = float(m)
                active_labels.append(ineq_labels[w])
        state.multipliers = multipliers
        state.active_set = active_labels
        state.scratch["constraint_report"] = self._constraints.report()
        state.scratch["max_constraint_violation"] = self._constraints.max_violation()


def _to_float(v: Any) -> float:
    if hasattr(v, "item"):
        try:
            return float(v.item())
        except Exception:  # noqa: BLE001
            pass
    import optiland.backend as be

    return float(be.to_numpy(v))
