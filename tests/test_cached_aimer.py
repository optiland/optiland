from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from optiland.optic import Optic
from optiland.rays.ray_aiming.base import BaseRayAimer
from optiland.rays.ray_aiming.cached import CachedRayAimer
from optiland.rays.ray_aiming.robust import RobustRayAimer


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


def test_cache_miss_calls_wrapped(set_test_backend, mock_dependencies):
    """Test that the wrapped aimer is called on a cache miss."""
    optic, wrapped_aimer = mock_dependencies
    cached_aimer = CachedRayAimer(optic, wrapped_aimer)

    inputs = ((0.0,), 0.55, (0.0, 0.0))
    expected_result = (1, 2, 3, 4, 5, 6)
    wrapped_aimer.aim_rays.return_value = expected_result

    result = cached_aimer.aim_rays(*inputs)

    assert result == expected_result
    wrapped_aimer.aim_rays.assert_called_once()


def test_cache_hit_skips_wrapped(set_test_backend, mock_dependencies):
    """Test that the wrapped aimer is NOT called on a cache hit."""
    optic, wrapped_aimer = mock_dependencies
    cached_aimer = CachedRayAimer(optic, wrapped_aimer)

    inputs = ((0.0,), 0.55, (0.0, 0.0))
    expected_result = (1, 2, 3, 4, 5, 6)
    wrapped_aimer.aim_rays.return_value = expected_result

    # First call (miss)
    cached_aimer.aim_rays(*inputs)
    wrapped_aimer.aim_rays.reset_mock()

    # Second call (hit)
    result = cached_aimer.aim_rays(*inputs)

    assert result == expected_result
    wrapped_aimer.aim_rays.assert_not_called()


def test_system_change_uses_cached_as_guess(set_test_backend, mock_dependencies):
    """Test that a system change uses the cached result as a starting guess."""
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
    # Check that wrapped aimer was called with initial_guess=result1
    wrapped_aimer.aim_rays.assert_called_with(
        inputs[0], inputs[1], inputs[2], initial_guess=result1
    )


def test_robust_aimer_integration_with_cache(set_test_backend):
    """Regression test for RobustRayAimer integration with caching.

    Ensures that RobustRayAimer accepts initial_guess passed by CachedRayAimer.
    """
    import optiland.backend as be
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

    # Mock _solve_core to avoid AttributeError and ValueError during unpacking
    iterative_mock._solve_core.side_effect = lambda *args, **kwargs: (
        be.array([1.0]),
        be.array([1.0]),
        be.array([1.0]),
        be.array([0.0]),
        be.array([0.0]),
        be.array([1.0]),
        be.array([1.0]) > 0.0,
        None
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

    # 1. First call (Cache Miss)
    # This will call robust_aimer -> _solve -> iterative.aim_rays
    res1 = cached_aimer.aim_rays(fields, wl, pupil)

    # Expectation: robust aimer uses iterative aimer result
    expected_first_result = (1.0, 1.0, 1.0, 0.0, 0.0, 1.0)

    def to_floats(tup):
        return tuple(float(be.to_numpy(x).reshape(-1)[0]) for x in tup)

    assert to_floats(res1) == expected_first_result

    # 2. Perturb System
    optic.surfaces.to_dict.return_value = {"surfaces": ["changed"]}

    # 3. Second call (System Change -> Pass cached result as initial_guess)
    # This calls robust_aimer.aim_rays(..., initial_guess=expected_first_result)
    # robust_aimer should call _iterative.aim_rays(...,
    # initial_guess=expected_first_result)

    # We update iterative mock return value to distinguish second call result
    iterative_mock.aim_rays.return_value = (9, 9, 9, 8, 8, 8)

    res2 = cached_aimer.aim_rays(fields, wl, pupil)

    # Confirm result matches new iterative return value
    assert res2 == (9, 9, 9, 8, 8, 8)

    # Verify _iterative.aim_rays was called with initial_guess
    # It might be called multiple times (first run, second run).
    # We want to check the *last* call or ensure one of them had the kwarg.

    # Get all calls
    calls = iterative_mock.aim_rays.call_args_list
    # Search for the call with initial_guess
    found_guess = False
    for c in calls:
        if "initial_guess" in c.kwargs:
            guess = c.kwargs["initial_guess"]
            if to_floats(guess) == expected_first_result:
                found_guess = True
                break
    assert found_guess, "Initial guess was not passed to iterative aimer on second call"


if __name__ == "__main__":
    pytest.main([__file__])
