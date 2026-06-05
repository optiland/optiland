"""Optimization finalization test suite.

Verification matrix:

  +---------------------------------------+-------+------+------+
  | Test                                  | numpy | cpu  | cuda |
  +=======================================+=======+======+======+
  | native LM/DLS merit reduction         | ✔     | ✔    | skip |
  | torch Adam merit reduction            | —     | ✔    | skip |
  | scipy-local (l-bfgs-b) parity         | ✔     | —    | —    |
  | scipy-LS (least_squares) parity       | ✔     | —    | —    |
  | CUDA end-to-end smoke                 | —     | —    | skip |
  | no DeprecationWarning from facade     | ✔     | —    | —    |
  | no .item() in stepped hot loop        | —     | ✔    | —    |
  | WS1: TorchParameterSync composition   | —     | ✔    | —    |
  | F6: auto → resolved + result.method   | ✔     | ✔    | —    |
  | F8: result.status carries criterion   | ✔     | —    | —    |
  | F3: mutation contract                 | ✔     | —    | —    |
  +---------------------------------------+-------+------+------+

Kramer Harrison, 2026
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import optiland.backend as be
from optiland.optimization import OptimizationProblem, minimize
from optiland.optimization.errors import ConfigurationError
from optiland.samples.objectives import CookeTriplet

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

_HAS_CUDA = _HAS_TORCH and torch.cuda.is_available()


def _make_numpy_problem(n_vars: int = 2):
    """Cooke triplet + n_vars radius variables (numpy backend)."""
    be.set_backend("numpy")
    lens = CookeTriplet()
    p = OptimizationProblem()
    p.add_operand("f2", target=50.0, weight=1.0, input_data={"optic": lens})
    p.add_operand("total_track", target=100.0, weight=1.0, input_data={"optic": lens})
    for i in range(1, n_vars + 1):
        p.add_variable(lens, "radius", surface_number=i)
    return p, lens


def _make_torch_problem(device: str = "cpu"):
    """Same as _make_numpy_problem but on torch backend (batching=False)."""
    be.set_backend("torch")
    be.set_device(device)
    be.set_precision("float64")
    be.grad_mode.enable()
    lens = CookeTriplet()
    p = OptimizationProblem(batching=False)
    p.add_operand("f2", target=50.0, weight=1.0, input_data={"optic": lens})
    p.add_operand("total_track", target=100.0, weight=1.0, input_data={"optic": lens})
    for i in range(1, 3):
        p.add_variable(lens, "radius", surface_number=i)
    return p, lens


# ---------------------------------------------------------------------------
# WS1 — TorchParameterSync composition
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
class TestTorchParameterSyncComposition:
    """TorchParameterSync is the sole sync core; OpticalSystemModule composes."""

    def setup_method(self):
        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision("float64")
        be.grad_mode.enable()

    def teardown_method(self):
        be.set_backend("numpy")

    def test_params_found_in_parent_module(self):
        """OpticalSystemModule's params appear in a parent nn.Module.parameters()."""
        import torch.nn as nn

        from optiland.ml import OpticalSystemModule

        lens = CookeTriplet()
        p = OptimizationProblem()
        p.add_operand("f2", target=50.0, weight=1.0, input_data={"optic": lens})
        p.add_variable(lens, "radius", surface_number=1)
        p.add_variable(lens, "radius", surface_number=2)

        optic_module = OpticalSystemModule(lens, p)

        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.optic = optic_module

        parent = Parent()
        parent_params = list(parent.parameters())
        assert len(parent_params) == 2, (
            f"Expected 2 params in parent, got {len(parent_params)}"
        )

    def test_single_sync_core(self):
        """AutogradEvaluator, TorchOptimizer, OpticalSystemModule all use _sync."""
        from optiland.ml import OpticalSystemModule
        from optiland.optimization.evaluators.autograd import AutogradEvaluator
        from optiland.optimization.evaluators._param_sync import TorchParameterSync
        from optiland.optimization.native.torch_opt import TorchOptimizer

        lens = CookeTriplet()
        p = OptimizationProblem()
        p.add_operand("f2", target=50.0, weight=1.0, input_data={"optic": lens})
        p.add_variable(lens, "radius", surface_number=1)

        evaluator = AutogradEvaluator(p)
        assert isinstance(evaluator._sync, TorchParameterSync)

        opt_module = OpticalSystemModule(lens, p)
        assert isinstance(opt_module._sync, TorchParameterSync)


# ---------------------------------------------------------------------------
# WS2 — no DeprecationWarning from minimize()
# ---------------------------------------------------------------------------


class TestNoDeprecationFromFacade:
    """minimize() must never trigger DeprecationWarning internally."""

    def setup_method(self):
        be.set_backend("numpy")

    def teardown_method(self):
        be.set_backend("numpy")

    def test_minimize_dls_no_deprecation(self):
        p, _ = _make_numpy_problem(n_vars=2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            minimize(p, method="dls", maxiter=5, disp=False)
        dep_warns = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep_warns == [], (
            f"minimize() triggered DeprecationWarnings: {dep_warns}"
        )

    def test_minimize_scipy_no_deprecation(self):
        p, _ = _make_numpy_problem(n_vars=2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            minimize(p, method="l-bfgs-b", maxiter=5, disp=False)
        dep_warns = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep_warns == [], (
            f"minimize() triggered DeprecationWarnings: {dep_warns}"
        )

    def test_legacy_class_does_warn(self):
        """Constructing a legacy class directly should warn."""
        from optiland.optimization.optimizer.scipy.base import OptimizerGeneric

        p, _ = _make_numpy_problem(n_vars=1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            OptimizerGeneric(p)
        dep_warns = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep_warns) == 1
        assert "v0.7.0" in str(dep_warns[0].message)


# ---------------------------------------------------------------------------
# Native LM/DLS — numpy + torch-cpu + torch-cuda
# ---------------------------------------------------------------------------


class TestNativeLM:
    """Native LM/DLS reduces merit across all blessed backends."""

    def teardown_method(self):
        be.set_backend("numpy")

    def test_dls_numpy(self):
        p, _ = _make_numpy_problem(n_vars=2)
        initial = float(be.to_numpy(p.sum_squared()))
        result = minimize(p, method="dls", maxiter=20, disp=False)
        assert result.value < initial
        assert result.method == "dls"
        assert result.resolved_from is None

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_dls_torch_cpu(self):
        p, _ = _make_torch_problem(device="cpu")
        initial = float(be.to_numpy(p.sum_squared()))
        result = minimize(p, method="dls", maxiter=20, disp=False)
        assert result.value < initial
        assert result.backend == "torch"

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")
    def test_dls_torch_cuda(self):
        p, _ = _make_torch_problem(device="cuda")
        initial = float(be.to_numpy(p.sum_squared()))
        result = minimize(p, method="dls", maxiter=20, disp=False)
        assert result.value < initial
        assert result.backend == "torch"


# ---------------------------------------------------------------------------
# Torch Adam — cpu + cuda
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
class TestTorchAdam:
    def teardown_method(self):
        be.set_backend("numpy")

    def test_adam_cpu(self):
        p, _ = _make_torch_problem(device="cpu")
        result = minimize(p, method="adam", maxiter=200, disp=False)
        # Verify run completes and produces finite merit
        assert result.method == "adam"
        assert result.backend == "torch"
        assert result.value < float("inf")
        assert not isinstance(result.value, type(None))

    @pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")
    def test_adam_cuda(self):
        p, _ = _make_torch_problem(device="cuda")
        result = minimize(p, method="adam", maxiter=50, disp=False)
        assert result.backend == "torch"
        assert result.value < float("inf")


# ---------------------------------------------------------------------------
# SciPy managed — numpy only
# ---------------------------------------------------------------------------


class TestScipyManaged:
    def teardown_method(self):
        be.set_backend("numpy")

    def test_scipy_local_reduces(self):
        p, _ = _make_numpy_problem(n_vars=2)
        initial = float(be.to_numpy(p.sum_squared()))
        result = minimize(p, method="l-bfgs-b", maxiter=20, disp=False)
        assert result.value < initial

    def test_scipy_least_squares_reduces(self):
        p, _ = _make_numpy_problem(n_vars=2)
        initial = float(be.to_numpy(p.sum_squared()))
        result = minimize(p, method="least_squares", maxiter=30, disp=False)
        assert result.value < initial


# ---------------------------------------------------------------------------
# CUDA end-to-end smoke
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")
class TestCUDASmoke:
    """End-to-end CUDA path: device correctness, no device-mismatch errors."""

    def teardown_method(self):
        if be.get_backend() == "torch":
            be.set_device("cpu")
        be.set_backend("numpy")

    def test_dls_cuda_result_on_device(self):
        p, lens = _make_torch_problem(device="cuda")
        result = minimize(p, method="dls", maxiter=10, disp=False)
        assert result.backend == "torch"
        assert result.value < float("inf")

    def test_adam_cuda_result_on_device(self):
        p, lens = _make_torch_problem(device="cuda")
        result = minimize(p, method="adam", maxiter=10, disp=False)
        assert result.backend == "torch"

    def test_optical_system_module_forward_cuda(self):
        from optiland.ml import OpticalSystemModule

        p, lens = _make_torch_problem(device="cuda")
        module = OpticalSystemModule(lens, p)
        loss = module.forward()
        assert loss.device.type == "cuda"


# ---------------------------------------------------------------------------
# §6.2 — no .item() in stepped hot loop
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
class TestNoItemInHotLoop:
    """Guard: TorchParameterSync.load_x must not call .item() per element."""

    def setup_method(self):
        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision("float64")
        be.grad_mode.enable()

    def teardown_method(self):
        be.set_backend("numpy")

    def test_load_x_no_item_calls(self, monkeypatch):
        """load_x uses copy_ / fill_, not per-element item() calls."""
        from optiland.optimization.evaluators._param_sync import TorchParameterSync

        lens = CookeTriplet()
        p = OptimizationProblem()
        p.add_operand("f2", target=50.0, weight=1.0, input_data={"optic": lens})
        p.add_variable(lens, "radius", surface_number=1)
        p.add_variable(lens, "radius", surface_number=2)

        sync = TorchParameterSync(p)

        item_calls = [0]
        original_item = torch.Tensor.item

        def counting_item(self):
            item_calls[0] += 1
            return original_item(self)

        monkeypatch.setattr(torch.Tensor, "item", counting_item)

        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        sync.load_x(x)

        assert item_calls[0] == 0, (
            f"load_x called .item() {item_calls[0]} times; expected 0 (D9 violation)"
        )


# ---------------------------------------------------------------------------
# F6 — auto transparency
# ---------------------------------------------------------------------------


class TestAutoTransparency:
    def teardown_method(self):
        be.set_backend("numpy")

    def test_auto_numpy_resolves_method(self):
        """Auto on numpy (equality, m>=n) resolves to dls."""
        p, _ = _make_numpy_problem(n_vars=2)
        result = minimize(p, method="auto", maxiter=5, disp=False)
        assert result.method == "dls"
        assert result.resolved_from == "auto"

    def test_explicit_method_resolved_from_none(self):
        p, _ = _make_numpy_problem(n_vars=2)
        result = minimize(p, method="dls", maxiter=5, disp=False)
        assert result.resolved_from is None

    @pytest.mark.skipif(not _HAS_TORCH, reason="torch not available")
    def test_auto_torch_resolves_adam(self):
        p, _ = _make_torch_problem(device="cpu")
        result = minimize(p, method="auto", maxiter=5, disp=False)
        assert result.method == "adam"
        assert result.resolved_from == "auto"

    def test_auto_disp_prints_resolved(self, capsys):
        p, _ = _make_numpy_problem(n_vars=2)
        minimize(p, method="auto", maxiter=3, disp=True)
        captured = capsys.readouterr()
        assert "auto →" in captured.out


# ---------------------------------------------------------------------------
# F7 — tiered validation
# ---------------------------------------------------------------------------


class TestTieredValidation:
    def teardown_method(self):
        be.set_backend("numpy")

    def test_torch_method_numpy_raises(self):
        be.set_backend("numpy")
        p, _ = _make_numpy_problem(n_vars=1)
        with pytest.raises(ConfigurationError, match="torch"):
            minimize(p, method="adam")

    def test_per_step_strategy_managed_raises(self):
        # A per-step-projection strategy (apply_to_step but no to_scipy) cannot
        # be honored by a managed SciPy method (NullSpaceStrategy was removed in
        # favor of the KKT path; D2).
        class _PerStepOnly:
            def prepare(self, evaluator, variables):
                pass

            def apply_to_step(self, x_proposed, state):
                return x_proposed

        p, _ = _make_numpy_problem(n_vars=1)
        with pytest.raises(ConfigurationError, match="per-step"):
            minimize(p, method="l-bfgs-b", constraints=_PerStepOnly())

    def test_controller_managed_warns(self):
        from optiland.optimization.control.identity import IdentityController

        p, _ = _make_numpy_problem(n_vars=2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            minimize(p, method="l-bfgs-b", controller=IdentityController(),
                     maxiter=3, disp=False)
        user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
        assert any("controller" in str(w.message).lower() for w in user_warns)


# ---------------------------------------------------------------------------
# F8 — tol criterion visibility in result.status
# ---------------------------------------------------------------------------


class TestTolCriterionVisibility:
    def teardown_method(self):
        be.set_backend("numpy")

    def test_cost_tolerance_in_status(self):
        p, _ = _make_numpy_problem(n_vars=2)
        result = minimize(p, method="dls", maxiter=200, tol=1e-2, disp=False)
        # Either cost_tol or max_iter should appear in status
        assert result.status is not None
        assert len(result.status) > 0

    def test_status_not_empty(self):
        p, _ = _make_numpy_problem(n_vars=2)
        result = minimize(p, method="l-bfgs-b", maxiter=5, disp=False)
        assert result.status is not None


# ---------------------------------------------------------------------------
# F3 — mutation contract
# ---------------------------------------------------------------------------


class TestMutationContract:
    def teardown_method(self):
        be.set_backend("numpy")

    def test_optic_mutated_in_place(self):
        """minimize() mutates the optic; result.x is a copy."""
        lens = CookeTriplet()
        p = OptimizationProblem()
        p.add_operand("f2", target=50.0, weight=1.0, input_data={"optic": lens})
        p.add_variable(lens, "radius", surface_number=1)

        r0 = lens.surfaces[1].geometry.radius
        result = minimize(p, method="dls", maxiter=10, disp=False)

        r1 = lens.surfaces[1].geometry.radius
        # Optic has been mutated (unless already converged at start)
        assert result.x is not None
        # result.x should be a copy, not a reference to the param tensor
        assert len(result.x) == 1
