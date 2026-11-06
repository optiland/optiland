from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.rays.ray_aiming.base import BaseRayAimer
from optiland.rays.ray_aiming.cached import CachedRayAimer
from optiland.rays.ray_aiming.robust import RobustRayAimer

from .utils import assert_allclose

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def numerical_backend(set_test_backend: None) -> Iterator[None]:
    """Disable both Torch gradient controls locally for numerical cache tests."""
    backend = be.get_backend()
    precision = be.get_precision()
    with ExitStack() as stack:
        if backend == "torch":
            import torch

            grad_mode = be.grad_mode
            requires_grad = grad_mode.requires_grad
            stack.callback(setattr, grad_mode, "requires_grad", requires_grad)
            grad_mode.disable()
            stack.enter_context(torch.no_grad())
        try:
            be.set_precision("float64")
            yield
        finally:
            be.set_backend(backend)
            be.set_precision(f"float{precision}")


@pytest.fixture
def mock_dependencies():
    optic = MagicMock(spec=Optic)
    surface_group_mock = MagicMock()
    surface_group_mock.to_dict.return_value = {"surfaces": []}
    surface_group_mock.stop_index = 1
    optic.surfaces = surface_group_mock
    optic.fields = MagicMock()
    optic.fields.to_dict.return_value = {}
    optic.wavelengths = MagicMock()
    optic.wavelengths.to_dict.return_value = {}
    optic.aperture = MagicMock()
    optic.aperture.to_dict.return_value = {}
    optic.object_surface = MagicMock()

    ray_tracer_mock = MagicMock()
    ray_tracer_mock.ray_aiming_config = {
        "mode": "paraxial",
        "max_iter": 10,
        "tol": 1e-6,
    }
    optic.ray_tracer = ray_tracer_mock

    wrapped_aimer = MagicMock(spec=BaseRayAimer)
    return optic, wrapped_aimer


def test_cache_miss_calls_wrapped(numerical_backend, mock_dependencies):
    """Test that the wrapped aimer is called on a cache miss."""
    optic, wrapped_aimer = mock_dependencies
    cached_aimer = CachedRayAimer(optic, wrapped_aimer)

    inputs = ((0.0,), 0.55, (0.0, 0.0))
    expected_result = (1, 2, 3, 4, 5, 6)
    wrapped_aimer.aim_rays.return_value = expected_result

    result = cached_aimer.aim_rays(*inputs)

    assert result == expected_result
    wrapped_aimer.aim_rays.assert_called_once_with(*inputs)


@pytest.mark.parametrize("array_inputs", [False, True])
def test_cache_hit_skips_wrapped(
    numerical_backend, mock_dependencies, array_inputs: bool
) -> None:
    """Test that the wrapped aimer is NOT called on a cache hit."""
    optic, wrapped_aimer = mock_dependencies
    cached_aimer = CachedRayAimer(optic, wrapped_aimer)

    inputs = ((0.0,), 0.55, (0.0, 0.0))
    if array_inputs:
        inputs = (
            (be.array([0.0]), be.array([1.0])),
            be.array([0.55]),
            (be.array([0.0]), be.array([0.0])),
        )
    expected_result = (1, 2, 3, 4, 5, 6)
    wrapped_aimer.aim_rays.return_value = expected_result

    # First call (miss)
    cached_aimer.aim_rays(*inputs)
    wrapped_aimer.aim_rays.reset_mock()

    # Second call (hit)
    result = cached_aimer.aim_rays(*inputs)

    assert result == expected_result
    wrapped_aimer.aim_rays.assert_not_called()


def test_system_change_delegates_fresh(numerical_backend, mock_dependencies):
    """A system change must not reuse fixed components of an old launch."""
    optic, wrapped_aimer = mock_dependencies
    cached_aimer = CachedRayAimer(optic, wrapped_aimer)

    inputs = ((0.0,), 0.55, (0.0, 0.0))
    result1 = (1, 2, 3, 4, 5, 6)
    wrapped_aimer.aim_rays.return_value = result1

    # First call (miss)
    cached_aimer.aim_rays(*inputs)

    # Perturb system (change dict representation)
    optic.surfaces.to_dict.return_value = {"surfaces": ["changed"]}

    # Second call
    result2 = (7, 8, 9, 10, 11, 12)
    wrapped_aimer.aim_rays.return_value = result2

    final_result = cached_aimer.aim_rays(*inputs)

    assert final_result == result2
    assert wrapped_aimer.aim_rays.call_count == 2
    wrapped_aimer.aim_rays.assert_called_with(*inputs)
    wrapped_aimer.aim_rays.reset_mock()
    assert cached_aimer.aim_rays(*inputs) is result2
    wrapped_aimer.aim_rays.assert_not_called()


def test_robust_aimer_integration_with_cache(numerical_backend, request):
    """System changes recalibrate; only caller guesses use the direct solve."""
    from optiland.aperture import FloatByStopAperture
    from optiland.rays.ray_aiming.pupil_map import PupilMapCache

    # Create valid dummy field/pupil data
    fields = (0.0, 0.0)
    pupil = (0.0, 0.0)
    wl = 0.55

    # Mock Optic
    optic = MagicMock(spec=Optic)
    optic.paraxial = MagicMock()
    surface_group_mock = MagicMock()
    surface_group_mock.to_dict.return_value = {"surfaces": []}
    surface_group_mock.stop_index = 1

    stop_surface_mock = MagicMock()
    stop_surface_mock.aperture.r_max = 1.0
    surface_group_mock.__getitem__.return_value = stop_surface_mock

    optic.surfaces = surface_group_mock
    optic.fields = MagicMock()
    optic.fields.to_dict.return_value = {}
    optic.fields.get_vig_factor.return_value = (0.0, 0.0)
    optic.wavelengths = MagicMock()
    optic.wavelengths.to_dict.return_value = {}
    optic.aperture = MagicMock(spec=FloatByStopAperture)
    optic.aperture.to_dict.return_value = {}
    optic.object_surface = MagicMock()

    ray_tracer_mock = MagicMock()
    ray_tracer_mock.ray_aiming_config = {
        "mode": "paraxial",
        "max_iter": 10,
        "tol": 1e-6,
    }
    optic.ray_tracer = ray_tracer_mock

    # Mock Paraxial Aimer (needed by Robust)
    paraxial_mock = MagicMock(spec=BaseRayAimer)
    paraxial_mock.aim_rays.return_value = (0, 0, 0, 0, 0, 1)  # Dummy sol

    # Mock Iterative Aimer (needed by Robust)
    from optiland.rays.ray_aiming.iterative import IterativeRayAimer
    iterative_mock = MagicMock(spec=IterativeRayAimer)
    iterative_mock.aim_rays.return_value = (1, 1, 1, 0, 0, 1)  # Dummy converged sol

    # Mock _solve_core to avoid AttributeError and ValueError during
    # unpacking; the report slot must be a genuine SolveReport because the
    # robust aimer reads it to build its aggregate last_report.
    from optiland.rays.ray_aiming.parameterization import SolveReport

    # Sized to the batch: the edge probes are solved four at a time.
    def solve_core(*args, **kwargs):
        n = len(be.as_array_1d(args[0]))
        return (
            be.ones(n),
            be.ones(n),
            be.ones(n),
            be.zeros(n),
            be.zeros(n),
            be.ones(n),
            be.ones(n) > 0.0,
            None,
            SolveReport(
                seed_residual=0.0,
                final_residual=0.0,
                converged=True,
                iterations=1,
                num_rays=n,
                num_converged=n,
            ),
        )

    iterative_mock._solve_core.side_effect = solve_core
    iterative_mock.last_report = SolveReport(
        seed_residual=0.0,
        final_residual=0.0,
        converged=True,
        iterations=1,
        num_rays=1,
        num_converged=1,
    )
    iterative_mock.tol = 1e-6
    iterative_mock._paraxial_aimer = paraxial_mock

    # Instantiate RobustRayAimer with mocked internals
    # We patch __init__ or just inject deps manually to avoid full Optic setup
    robust_aimer = RobustRayAimer.__new__(RobustRayAimer)
    robust_aimer.optic = optic
    robust_aimer.scale_fields = True
    robust_aimer._paraxial = paraxial_mock
    # Inject our mock iterative aimer
    robust_aimer._iterative = iterative_mock
    robust_aimer._cache = PupilMapCache()

    # Create CachedRayAimer wrapping RobustRayAimer
    cached_aimer = CachedRayAimer(optic, robust_aimer)

    # The launch parameterization is built from the optic's entry frame,
    # which the MagicMock optic cannot provide; supply the canonical +z one.
    from optiland.rays.ray_aiming.parameterization import LaunchParameterization

    legacy_param = LaunchParameterization(
        is_infinite=True, u=(1.0, 0.0, 0.0), v=(0.0, 1.0, 0.0)
    )
    patcher = patch.object(
        LaunchParameterization, "for_optic", return_value=legacy_param
    )
    # Exception-safe: a failure inside this test must not leak the patched
    # class method into later tests (request.addfinalizer over bare stop()).
    patcher.start()
    request.addfinalizer(patcher.stop)

    # A cache miss runs robust calibration and the iterative core polish.
    res1 = cached_aimer.aim_rays(fields, wl, pupil)

    # Expectation: robust aimer uses iterative aimer result
    expected_first_result = (1.0, 1.0, 1.0, 0.0, 0.0, 1.0)

    def to_floats(tup):
        return tuple(float(be.to_numpy(x).reshape(-1)[0]) for x in tup)

    assert to_floats(res1) == expected_first_result

    # 2. Perturb System
    optic.surfaces.to_dict.return_value = {"surfaces": ["changed"]}

    # A stale full launch must not select robust's direct initial-guess path.
    iterative_mock._solve_core.reset_mock()
    res2 = cached_aimer.aim_rays(fields, wl, pupil)
    assert to_floats(res2) == expected_first_result
    assert iterative_mock._solve_core.called
    iterative_mock.aim_rays.assert_not_called()

    iterative_mock.aim_rays.return_value = (9, 9, 9, 8, 8, 8)
    res3 = cached_aimer.aim_rays(fields, wl, pupil, initial_guess=res1)
    assert res3 == (9, 9, 9, 8, 8, 8)
    iterative_mock.aim_rays.assert_called_once_with(
        fields, wl, pupil, initial_guess=res1
    )


def test_explicit_guess_bypasses_hashing_and_preserves_cache(
    numerical_backend: None, mock_dependencies: tuple[Optic, MagicMock]
) -> None:
    optic, wrapped = mock_dependencies
    aimer = CachedRayAimer(optic, wrapped)
    inputs = ((0.0, 1.0), 0.55, (0.0, 0.0))
    wrapped.aim_rays.return_value = (1, 2, 3, 4, 5, 6)
    numerical = aimer.aim_rays(*inputs)
    guess = (6, 5, 4, 3, 2, 1)
    with (
        patch.object(aimer, "_get_input_hash", side_effect=AssertionError),
        patch.object(aimer, "_get_system_hash", side_effect=AssertionError),
    ):
        aimer.aim_rays(*inputs, initial_guess=guess)
    wrapped.aim_rays.assert_called_with(*inputs, initial_guess=guess)
    wrapped.aim_rays.reset_mock()
    assert aimer.aim_rays(*inputs) is numerical
    wrapped.aim_rays.assert_not_called()


class _NumericalAimer(BaseRayAimer):
    """A three-argument public API with no initial-guess extension."""

    def aim_rays(
        self, fields: tuple, wavelengths: Any, pupil_coords: tuple
    ) -> tuple:
        return tuple(be.array([float(i)]) for i in range(6))


def test_numerical_context_invalidates_all_entries(
    numerical_backend: None, mock_dependencies: tuple[Optic, MagicMock]
) -> None:
    optic, _ = mock_dependencies
    aimer = CachedRayAimer(optic, _NumericalAimer(optic))
    inputs = ((0.0, 1.0), 0.55, (0.0, 0.0))
    other = ((0.0, 0.5), 0.55, (0.0, 0.0))
    be.set_precision("float64")
    original = aimer.aim_rays(*inputs)
    aimer.aim_rays(*other)
    assert len(aimer._cache) == 2
    assert aimer.aim_rays(*inputs) is original

    be.set_precision("float32")
    changed = aimer.aim_rays(*inputs)
    assert len(aimer._cache) == 1
    assert changed is not original
    assert changed[0].dtype == be.array([0.0]).dtype
    assert changed[0].dtype != original[0].dtype
    assert aimer.aim_rays(*inputs) is changed

    be.set_precision("float64")
    restored = aimer.aim_rays(*inputs)
    assert restored is not original
    assert restored is not changed
    assert restored[0].dtype == original[0].dtype


def test_backend_context_invalidates_before_explicit_guess(
    numerical_backend: None, mock_dependencies: tuple[Optic, MagicMock]
) -> None:
    if be.get_backend() != "torch":
        pytest.skip("Requires Torch and NumPy in one cache instance")
    optic, wrapped = mock_dependencies
    wrapped.aim_rays.side_effect = _NumericalAimer(optic).aim_rays
    aimer = CachedRayAimer(optic, wrapped)
    inputs = ((0.0, 1.0), 0.55, (0.0, 0.0))
    original = aimer.aim_rays(*inputs)

    be.set_backend("numpy")
    wrapped.aim_rays.side_effect = None
    wrapped.aim_rays.return_value = (0, 0, 0, 0, 0, 1)
    aimer.aim_rays(*inputs, initial_guess=original)
    assert not aimer._cache
    wrapped.aim_rays.assert_called_with(*inputs, initial_guess=original)
    wrapped.aim_rays.side_effect = _NumericalAimer(optic).aim_rays
    changed = aimer.aim_rays(*inputs)
    assert type(changed[0]) is not type(original[0])
    assert aimer.aim_rays(*inputs) is changed

    be.set_backend("torch")
    restored = aimer.aim_rays(*inputs)
    assert type(restored[0]) is type(original[0])
    assert restored is not original
    assert len(aimer._cache) == 1


def test_device_context_invalidates_cache(
    numerical_backend: None, mock_dependencies: tuple[Optic, MagicMock]
) -> None:
    if be.get_backend() != "torch":
        pytest.skip("Device context applies only to Torch")
    optic, wrapped = mock_dependencies
    aimer = CachedRayAimer(optic, wrapped)
    inputs = ((0.0, 1.0), 0.55, (0.0, 0.0))
    aimer.aim_rays(*inputs)
    wrapped.aim_rays.reset_mock()
    # Exercise the device token on CPU-only machines without allocating on CUDA.
    with patch.object(be, "get_device", return_value="cuda"):
        aimer.aim_rays(*inputs)
        wrapped.aim_rays.assert_called_once_with(*inputs)
        aimer.aim_rays(*inputs)
        wrapped.aim_rays.assert_called_once()
    aimer.aim_rays(*inputs)
    assert wrapped.aim_rays.call_count == 2


@pytest.mark.parametrize("inference_first", [False, True])
def test_inference_context_invalidates_cache(
    numerical_backend: None,
    mock_dependencies: tuple[Optic, MagicMock],
    inference_first: bool,
) -> None:
    if be.get_backend() != "torch":
        pytest.skip("Inference mode applies only to Torch")
    import torch

    optic, _ = mock_dependencies
    aimer = CachedRayAimer(optic, _NumericalAimer(optic))
    inputs = ((0.0, 1.0), 0.55, (0.0, 0.0))
    other = ((0.0, 0.5), 0.55, (0.0, 0.0))
    with torch.inference_mode(inference_first), torch.no_grad():
        original = aimer.aim_rays(*inputs)
        aimer.aim_rays(*other)
        assert len(aimer._cache) == 2
        assert original[0].is_inference() is inference_first
        assert aimer.aim_rays(*inputs) is original

    with torch.inference_mode(not inference_first), torch.no_grad():
        changed = aimer.aim_rays(*inputs)
        assert changed is not original
        assert len(aimer._cache) == 1
        assert changed[0].is_inference() is not inference_first
        assert aimer.aim_rays(*inputs) is changed

    with torch.inference_mode(inference_first), torch.no_grad():
        restored = aimer.aim_rays(*inputs)
        assert restored is not original
        assert restored is not changed
        assert restored[0].is_inference() is inference_first
        assert aimer.aim_rays(*inputs) is restored

    # Unlike inference tensors, no-grad launches remain mutable outside inference.
    numerical = changed if inference_first else restored
    numerical[0].add_(1.0)
    assert_allclose(numerical[0], [1.0])


@pytest.mark.parametrize(
    "backend_grad,ambient_grad", [(True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("explicit_guess", [False, True])
def test_grad_bypass_precedes_all_cache_operations(
    numerical_backend: None,
    mock_dependencies: tuple[Optic, MagicMock],
    backend_grad: bool,
    ambient_grad: bool,
    explicit_guess: bool,
) -> None:
    if be.get_backend() != "torch":
        pytest.skip("Autograd is Torch-only")
    import torch

    optic, wrapped = mock_dependencies
    aimer = CachedRayAimer(optic, wrapped)
    inputs = ((0.0, 1.0), 0.55, (0.0, 0.0))
    wrapped.aim_rays.return_value = (1, 2, 3, 4, 5, 6)
    numerical = aimer.aim_rays(*inputs)
    entries = aimer._cache
    guarded_cache = MagicMock(wraps=entries)
    aimer._cache = guarded_cache
    wrapped.aim_rays.reset_mock()
    guess = (6, 5, 4, 3, 2, 1)
    kwargs = {"initial_guess": guess} if explicit_guess else {}

    def delegate(*args: Any, **kw: Any) -> tuple:
        assert not entries
        assert be.grad_mode.requires_grad is backend_grad
        assert torch.is_grad_enabled() is ambient_grad
        return (7, 8, 9, 10, 11, 12)

    wrapped.aim_rays.side_effect = delegate
    with (
        patch.object(be.grad_mode, "requires_grad", backend_grad),
        torch.set_grad_enabled(ambient_grad),
        patch.object(aimer, "_get_input_hash", side_effect=AssertionError),
        patch.object(aimer, "_get_system_hash", side_effect=AssertionError),
        patch.object(be, "get_precision", side_effect=AssertionError),
        patch.object(be, "get_device", side_effect=AssertionError),
    ):
        aimer.aim_rays(*inputs, **kwargs)
        aimer.aim_rays(*inputs, **kwargs)
    assert guarded_cache.clear.call_count == 2
    guarded_cache.get.assert_not_called()
    guarded_cache.__setitem__.assert_not_called()
    assert wrapped.aim_rays.call_count == 2
    wrapped.aim_rays.assert_called_with(*inputs, **kwargs)

    aimer._cache = entries
    wrapped.aim_rays.side_effect = None
    wrapped.aim_rays.return_value = (7, 8, 9, 10, 11, 12)
    resumed = aimer.aim_rays(*inputs)
    assert resumed is not numerical
    assert wrapped.aim_rays.call_count == 3
    assert aimer.aim_rays(*inputs) is resumed
    assert wrapped.aim_rays.call_count == 3


class _NonlinearAimer(BaseRayAimer):
    """Isolate wrapper graph lifetime from the physical solver's derivatives."""

    def aim_rays(
        self, fields: tuple, wavelengths: Any, pupil_coords: tuple
    ) -> tuple:
        radius = self.optic.surfaces[1].geometry.radius
        return (radius**2,) * 6


@pytest.mark.parametrize("replace_leaf", [False, True])
@pytest.mark.parametrize("backend_grad", [False, True])
def test_each_forward_builds_a_fresh_graph(
    numerical_backend: None, replace_leaf: bool, backend_grad: bool
) -> None:
    if be.get_backend() != "torch":
        pytest.skip("Autograd is Torch-only")
    import torch

    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf)
    optic.surfaces.add(index=1, radius=2.0, is_stop=True)
    radius = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    optic.surfaces[1].geometry.radius = radius
    aimer = CachedRayAimer(optic, _NonlinearAimer(optic))
    inputs = ((0.0, 1.0), 0.55, (0.0, 0.0))
    numerical = aimer.aim_rays(*inputs)
    assert not numerical[0].requires_grad
    assert aimer.aim_rays(*inputs) is numerical

    with (
        patch.object(be.grad_mode, "requires_grad", backend_grad),
        torch.enable_grad(),
        patch.object(aimer, "_get_input_hash", side_effect=AssertionError),
        patch.object(aimer, "_get_system_hash", side_effect=AssertionError),
    ):
        first = aimer.aim_rays(*inputs)
        first[0].sum().backward()
        assert_allclose(radius.grad, [4.0])
        radius.grad = None

        if replace_leaf:
            new_radius = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
            optic.surfaces[1].geometry.radius = new_radius
        else:
            new_radius = radius
        second = aimer.aim_rays(*inputs)
        assert second[0] is not first[0]
        second[0].sum().backward()
        assert_allclose(new_radius.grad, [4.0])
        if replace_leaf:
            assert radius.grad is None
        assert not aimer._cache

    resumed = aimer.aim_rays(*inputs)
    assert resumed is not numerical
    assert not resumed[0].requires_grad
    assert aimer.aim_rays(*inputs) is resumed


if __name__ == "__main__":
    pytest.main([__file__])
