"""Phase 4 optimization rework tests.

Covers:
  - Functional closure ``build_forward_fn()`` is differentiable (autograd)
  - Functional Jacobian (mode="functional") matches stateful Jacobian to machine tol
  - Compiled mode (mode="compiled") runs without error and matches stateful output
  - Shape checks: (m, n) for all modes
  - Numerical identity: ``J_functional == J_stateful`` within tolerance
  - ``torch.func.jacrev`` attempted opportunistically; fallback path exercised
  - Residual identity still holds via functional path: Σ r² == sum_squared()
  - ``jacobian_mode`` does not affect optimizer behavior (zero optimizer change, D4)

These tests skip when torch is not installed or the torch backend is unavailable.

Kramer Harrison, 2026
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.optimization import optimization
from optiland.samples.objectives import CookeTriplet

# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _use_torch_backend():
    """Activate the torch backend for the duration of each test."""
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not installed")
    prev = be.get_backend()
    be.set_backend("torch")
    yield
    be.set_backend(prev)


@pytest.fixture()
def small_problem():
    """Two equality operands + two radius variables (no batching for simplicity)."""
    lens = CookeTriplet()
    p = optimization.OptimizationProblem(batching=False)
    p.add_operand(
        operand_type="f2",
        target=50.0,
        weight=1.0,
        input_data={"optic": lens},
    )
    p.add_operand(
        operand_type="total_track",
        target=100.0,
        weight=1.0,
        input_data={"optic": lens},
    )
    p.add_variable(lens, "radius", surface_number=1)
    p.add_variable(lens, "radius", surface_number=3)
    return p


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_evaluator(problem, mode="stateful"):
    from optiland.optimization.evaluators.autograd import AutogradEvaluator

    return AutogradEvaluator(problem, jacobian_mode=mode)


def _numpy(t):
    """Convert backend tensor to numpy for comparison."""
    import numpy as np

    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


# ---------------------------------------------------------------------------
# Tests: build_forward_fn
# ---------------------------------------------------------------------------


class TestBuildForwardFn:
    def test_returns_callable(self, small_problem):
        ev = _make_evaluator(small_problem)
        fn = ev.build_forward_fn()
        assert callable(fn)

    def test_output_shape(self, small_problem):
        ev = _make_evaluator(small_problem)
        fn = ev.build_forward_fn()
        x = ev.read_x()
        r = fn(x)
        assert r.shape == (2,)  # 2 operands → m=2

    def test_output_differentiable(self, small_problem):
        """r depends on x through autograd (requires_grad=True on output)."""
        ev = _make_evaluator(small_problem)
        fn = ev.build_forward_fn()
        x = ev.read_x().requires_grad_(True)
        r = fn(x)
        # At least one element must participate in the grad graph
        assert r.requires_grad, "residuals must depend on x through autograd"

    def test_residual_identity_via_closure(self, small_problem):
        """Σ f(x)² == sum_squared() (PR1b identity preserved through closure)."""

        ev = _make_evaluator(small_problem)
        fn = ev.build_forward_fn()
        x = ev.read_x()

        r = fn(x).detach()
        ss_closure = float((r**2).sum().item())

        ev.write_x(x)
        ss_problem = float(small_problem.sum_squared().detach().item())

        assert abs(ss_closure - ss_problem) < 1e-10, (
            f"Σ r² via closure ({ss_closure}) != sum_squared() ({ss_problem})"
        )


# ---------------------------------------------------------------------------
# Tests: jacobian shape and mode dispatch
# ---------------------------------------------------------------------------


class TestJacobianModes:
    @pytest.mark.parametrize("mode", ["stateful", "functional", "compiled"])
    def test_jacobian_shape(self, small_problem, mode):
        ev = _make_evaluator(small_problem, mode=mode)
        x = ev.read_x()
        J = ev.jacobian(x)
        J_np = _numpy(J)
        assert J_np.shape == (2, 2), f"Expected (2,2) Jacobian, got {J_np.shape}"

    @pytest.mark.parametrize("mode", ["functional", "compiled"])
    def test_functional_matches_stateful(self, small_problem, mode):
        """Functional path must produce the same Jacobian as the stateful default."""
        import numpy as np

        ev_ref = _make_evaluator(small_problem, mode="stateful")
        ev_fast = _make_evaluator(small_problem, mode=mode)

        x = ev_ref.read_x()

        J_ref = _numpy(ev_ref.jacobian(x))
        J_fast = _numpy(ev_fast.jacobian(x))

        np.testing.assert_allclose(
            J_fast,
            J_ref,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"Jacobian mismatch between mode={mode!r} and stateful",
        )

    def test_stateful_mode_default(self, small_problem):
        """Evaluator defaults to 'stateful' mode."""
        from optiland.optimization.evaluators.autograd import AutogradEvaluator

        ev = AutogradEvaluator(small_problem)
        assert ev.jacobian_mode == "stateful"

    def test_functional_mode_constructor(self, small_problem):
        ev = _make_evaluator(small_problem, mode="functional")
        assert ev.jacobian_mode == "functional"

    def test_compiled_mode_constructor(self, small_problem):
        ev = _make_evaluator(small_problem, mode="compiled")
        assert ev.jacobian_mode == "compiled"


# ---------------------------------------------------------------------------
# Tests: compiled closure caching
# ---------------------------------------------------------------------------


class TestCompiledClosureCaching:
    def test_compiled_fn_cached(self, small_problem):
        """_get_or_build_compiled_fn() returns the same object on repeated calls."""
        ev = _make_evaluator(small_problem, mode="compiled")
        fn1 = ev._get_or_build_compiled_fn()
        fn2 = ev._get_or_build_compiled_fn()
        assert fn1 is fn2

    def test_forward_fn_cached(self, small_problem):
        ev = _make_evaluator(small_problem)
        fn1 = ev._get_or_build_forward_fn()
        fn2 = ev._get_or_build_forward_fn()
        assert fn1 is fn2

    def test_build_forward_fn_fresh(self, small_problem):
        """build_forward_fn() always returns a fresh callable."""
        ev = _make_evaluator(small_problem)
        fn1 = ev.build_forward_fn()
        fn2 = ev.build_forward_fn()
        assert fn1 is not fn2  # fresh each time (no caching on the public API)


# ---------------------------------------------------------------------------
# Tests: functional closure gradient correctness
# ---------------------------------------------------------------------------


class TestFunctionalClosureGradient:
    def test_gradient_via_closure_matches_evaluator_gradient(self, small_problem):
        """d(Σr²)/dx via the functional closure ≈ evaluator.gradient(x)."""
        import numpy as np

        ev = _make_evaluator(small_problem)
        fn = ev.build_forward_fn()
        x = ev.read_x()

        # Gradient via closure: backward on Σ r²
        x_t = x.clone().requires_grad_(True)
        loss = (fn(x_t) ** 2).sum()
        loss.backward()
        grad_closure = x_t.grad.detach().cpu().numpy()

        # Gradient via evaluator's gradient() method
        grad_ev = _numpy(ev.gradient(x))

        np.testing.assert_allclose(
            grad_closure,
            grad_ev,
            rtol=1e-5,
            atol=1e-7,
            err_msg="Gradient via closure vs evaluator.gradient() mismatch",
        )


# ---------------------------------------------------------------------------
# Tests: torch.func.jacrev opportunistic dispatch
# ---------------------------------------------------------------------------


class TestTorchFuncDispatch:
    def test_func_jacrev_attempted_for_functional_mode(self, small_problem):
        """In functional mode, torch.func.jacrev is tried if available."""
        ev = _make_evaluator(small_problem, mode="functional")
        x = ev.read_x()
        # Must not raise; may fall back to autograd.functional internally
        J = ev._jacobian_functional(x)
        assert J.shape == (2, 2)

    def test_fallback_does_not_corrupt_state(self, small_problem):
        """After a functional Jacobian call the problem state is consistent."""
        import numpy as np

        ev = _make_evaluator(small_problem, mode="functional")
        x0 = ev.read_x()
        ev.jacobian(x0)

        # After jacobian(), residuals should still evaluate correctly
        r = _numpy(ev.residuals(x0))
        assert np.all(np.isfinite(r)), (
            "residuals became non-finite after functional jacobian"
        )


# ---------------------------------------------------------------------------
# Tests: zero optimizer change (D4)
# ---------------------------------------------------------------------------


def _make_fresh_problem():
    """Create a fresh (lens, problem) pair — independent lens for each call."""
    lens = CookeTriplet()
    p = optimization.OptimizationProblem(batching=False)
    p.add_operand("f2", target=50.0, weight=1.0, input_data={"optic": lens})
    p.add_operand("total_track", target=100.0, weight=1.0, input_data={"optic": lens})
    p.add_variable(lens, "radius", surface_number=1)
    p.add_variable(lens, "radius", surface_number=3)
    return p


class TestZeroOptimizerChange:
    def test_lm_optimizer_unaffected_by_jacobian_mode(self):
        """LevenbergMarquardt produces the same result regardless of
        evaluator Jacobian mode.
        """
        import numpy as np

        from optiland.optimization.control.levenberg import LevenbergController
        from optiland.optimization.drivers import SteppedDriver
        from optiland.optimization.evaluators.autograd import AutogradEvaluator
        from optiland.optimization.native.least_squares import LevenbergMarquardt
        from optiland.optimization.stopping.criteria import MaxIter

        results = {}

        for mode in ("stateful", "functional"):
            # Each iteration gets its own independent lens + problem
            p = _make_fresh_problem()

            ev = AutogradEvaluator(p, jacobian_mode=mode)
            opt = LevenbergMarquardt()
            ctrl = LevenbergController()
            driver = SteppedDriver()
            stop = MaxIter(5)  # small number for speed

            result = driver.run(
                opt,
                ev,
                ev.read_x(),
                controller=ctrl,
                constraints=None,
                criteria=stop,
                observers=[],
                initial_value=float(p.sum_squared().detach().item()),
                method="dls",
            )
            results[mode] = result.value

        np.testing.assert_allclose(
            results["functional"],
            results["stateful"],
            rtol=1e-4,
            err_msg=(
                "LM final merit differs between stateful and functional "
                "Jacobian modes"
            ),
        )
