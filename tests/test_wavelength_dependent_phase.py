from __future__ import annotations

import json
from unittest.mock import Mock

import numpy as np
import pytest

from optiland import backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.plane import Plane
from optiland.interactions.base import BaseInteractionModel
from optiland.interactions.phase_interaction_model import PhaseInteractionModel
from optiland.materials.ideal import IdealMaterial
from optiland.optic import Optic
from optiland.phase import (
    BasePhaseProfile,
    HeightProfile,
    LinearGratingPhaseProfile,
    RadialPhaseProfile,
    WavelengthDependentPhaseProfile,
)
from optiland.rays.paraxial_rays import ParaxialRays
from optiland.rays.real_rays import RealRays
from optiland.surfaces.standard_surface import Surface

from .utils import assert_allclose

W1 = 0.48
W2 = 0.55
W_UNDEFINED = 0.65


@pytest.fixture
def mock_surface(set_test_backend):
    surface = Mock(spec=Surface)
    surface.geometry = Plane(coordinate_system=CoordinateSystem())
    surface.material_pre = IdealMaterial(n=1.0)
    surface.material_post = IdealMaterial(n=1.5)
    surface.geometry.surface_normal = Mock(
        return_value=(be.zeros(1), be.zeros(1), be.ones(1))
    )
    return surface


def _children():
    return {
        W1: RadialPhaseProfile(coefficients=[-2.0, 0.3]),
        W2: RadialPhaseProfile(coefficients=[-5.0, 0.7]),
    }


def _points():
    return be.array([0.1, 0.4, -0.3, 0.2]), be.array([0.2, -0.1, 0.5, 0.0])


def _select(mask, a, b):
    """Reference element-wise selection, done in NumPy."""
    return np.where(mask, be.to_numpy(a), be.to_numpy(b))


# --- dispatch ---------------------------------------------------------------


@pytest.mark.parametrize("wavelength", [W1, W2])
def test_single_wavelength_matches_child(set_test_backend, wavelength):
    children = _children()
    child = children[wavelength]
    profile = WavelengthDependentPhaseProfile(children)
    x, y = _points()
    w = be.full_like(x, wavelength)

    assert_allclose(profile.get_phase(x, y, w), child.get_phase(x, y, w))
    for got, expected in zip(
        profile.get_gradient(x, y, w), child.get_gradient(x, y, w), strict=True
    ):
        assert_allclose(got, expected)
    assert_allclose(
        profile.get_paraxial_gradient(y, w), child.get_paraxial_gradient(y, w)
    )


def test_scalar_wavelength(set_test_backend):
    children = _children()
    profile = WavelengthDependentPhaseProfile(children)
    x, y = _points()
    assert_allclose(profile.get_phase(x, y, W2), children[W2].get_phase(x, y, W2))


def test_mixed_wavelengths_select_per_element(set_test_backend):
    children = _children()
    profile = WavelengthDependentPhaseProfile(children)
    x, y = _points()
    w = be.array([W1, W2, W2, W1])
    is_w1 = np.array([True, False, False, True])
    c1, c2 = children[W1], children[W2]

    assert_allclose(
        profile.get_phase(x, y, w),
        _select(is_w1, c1.get_phase(x, y, w), c2.get_phase(x, y, w)),
    )
    for got, g1, g2 in zip(
        profile.get_gradient(x, y, w),
        c1.get_gradient(x, y, w),
        c2.get_gradient(x, y, w),
        strict=True,
    ):
        assert_allclose(got, _select(is_w1, g1, g2))
    assert_allclose(
        profile.get_paraxial_gradient(y, w),
        _select(
            is_w1,
            c1.get_paraxial_gradient(y, w),
            c2.get_paraxial_gradient(y, w),
        ),
    )


def test_float_representation_error_still_matches(set_test_backend):
    profile = WavelengthDependentPhaseProfile(_children())
    x, y = _points()
    w = be.full_like(x, W2 * (1 + 1e-9))
    assert_allclose(profile.get_phase(x, y, w), _children()[W2].get_phase(x, y, w))


# --- missing wavelengths: exact lookup, never nearest-neighbour -------------


def test_undefined_wavelength_raises(set_test_backend):
    profile = WavelengthDependentPhaseProfile(_children())
    x, y = _points()
    w = be.full_like(x, W_UNDEFINED)
    with pytest.raises(ValueError, match=r"wavelength\(s\) \[0\.65\]") as err:
        profile.get_phase(x, y, w)
    assert "Defined wavelengths: [0.48, 0.55]" in str(err.value)


def test_partly_undefined_wavelengths_raise_and_name_only_the_missing(
    set_test_backend,
):
    profile = WavelengthDependentPhaseProfile(_children())
    x, y = _points()
    w = be.array([W1, W_UNDEFINED, W2, W_UNDEFINED])
    with pytest.raises(ValueError, match=r"wavelength\(s\) \[0\.65\] um"):
        profile.get_gradient(x, y, w)


def test_near_miss_is_not_matched_to_nearest(set_test_backend):
    profile = WavelengthDependentPhaseProfile(_children())
    _, y = _points()
    w = be.full_like(y, W2 * (1 + 1e-4))
    with pytest.raises(ValueError, match="never falls back to the nearest"):
        profile.get_paraxial_gradient(y, w)


def test_none_wavelength_raises(set_test_backend):
    profile = WavelengthDependentPhaseProfile(_children())
    x, y = _points()
    with pytest.raises(ValueError, match="wavelength=None"):
        profile.get_phase(x, y, None)


# --- construction -----------------------------------------------------------


@pytest.mark.parametrize(
    ("profiles", "error", "match"),
    [
        ({}, ValueError, "at least one"),
        ({-0.5: RadialPhaseProfile([1.0])}, ValueError, "positive and finite"),
        ({0.0: RadialPhaseProfile([1.0])}, ValueError, "positive and finite"),
        ({float("nan"): RadialPhaseProfile([1.0])}, ValueError, "positive"),
        ({float("inf"): RadialPhaseProfile([1.0])}, ValueError, "positive"),
        ({0.55: "not a profile"}, TypeError, "must be a BasePhaseProfile"),
        (
            {
                0.55: RadialPhaseProfile([1.0]),
                0.55 * (1 + 1e-8): RadialPhaseProfile([2.0]),
            },
            ValueError,
            "ambiguous",
        ),
        (
            {
                0.48: LinearGratingPhaseProfile(period=1.0, efficiency=0.8),
                0.55: LinearGratingPhaseProfile(period=1.0, efficiency=0.9),
            },
            ValueError,
            "same efficiency",
        ),
    ],
)
def test_invalid_construction(set_test_backend, profiles, error, match):
    with pytest.raises(error, match=match):
        WavelengthDependentPhaseProfile(profiles)


def test_wavelengths_are_sorted_and_profiles_is_a_copy(set_test_backend):
    children = _children()
    profile = WavelengthDependentPhaseProfile({W2: children[W2], W1: children[W1]})
    assert profile.wavelengths == [W1, W2]

    snapshot = profile.profiles
    snapshot.clear()
    assert profile.wavelengths == [W1, W2]


def test_shared_efficiency_is_applied(mock_surface):
    efficiency = 0.7
    profile = WavelengthDependentPhaseProfile(
        {
            W1: LinearGratingPhaseProfile(period=1.0, efficiency=efficiency),
            W2: LinearGratingPhaseProfile(period=2.0, efficiency=efficiency),
        }
    )
    assert profile.efficiency == efficiency

    model = PhaseInteractionModel(mock_surface, profile, is_reflective=False)
    rays = RealRays(
        x=be.array([0.0]),
        y=be.array([0.0]),
        z=be.array([0.0]),
        L=be.array([0.0]),
        M=be.array([0.0]),
        N=be.array([1.0]),
        wavelength=W2,
        intensity=be.array([1.0]),
    )
    rays = model.interact_real_rays(rays)
    assert_allclose(rays.i, be.array([efficiency]))


def test_parent_surface_propagates_to_children(mock_surface):
    x_grid = be.linspace(-1.0, 1.0, 5)
    y_grid = be.linspace(-1.0, 1.0, 5)
    height = be.ones((5, 5)) * 1e-3
    children = {
        W1: HeightProfile(x_grid, y_grid, height),
        W2: HeightProfile(x_grid, y_grid, 2 * height),
    }
    profile = WavelengthDependentPhaseProfile(children)
    PhaseInteractionModel(mock_surface, profile, is_reflective=False)

    assert profile.parent_surface is mock_surface
    assert all(c.parent_surface is mock_surface for c in children.values())

    # HeightProfile reads the surface materials, so this only works if the
    # surface reached the child.
    x, y = be.array([0.0]), be.array([0.0])
    phase = profile.get_phase(x, y, be.array([W2]))
    dn = 1.5 - 1.0
    expected = 2 * be.pi / (W2 * 1e-3) * dn * 2e-3
    assert_allclose(phase, be.array([expected]))


# --- serialization ----------------------------------------------------------


def test_to_dict_from_dict_round_trip(set_test_backend):
    profile = WavelengthDependentPhaseProfile(
        {
            W2: LinearGratingPhaseProfile(period=0.5, angle=0.3, order=2),
            W1: RadialPhaseProfile(coefficients=[-2.0, 0.3]),
        }
    )
    data = profile.to_dict()

    # Plain JSON, with wavelengths stored as values, not as dict keys.
    assert json.loads(json.dumps(data)) == data
    assert data["phase_type"] == "wavelength_dependent"
    assert [entry["wavelength"] for entry in data["profiles"]] == [W1, W2]

    restored = BasePhaseProfile.from_dict(data)
    assert isinstance(restored, WavelengthDependentPhaseProfile)
    assert restored.to_dict() == data

    x, y = _points()
    w = be.array([W1, W2, W1, W2])
    assert_allclose(restored.get_phase(x, y, w), profile.get_phase(x, y, w))
    for got, expected in zip(
        restored.get_gradient(x, y, w), profile.get_gradient(x, y, w), strict=True
    ):
        assert_allclose(got, expected)


def test_interaction_model_round_trip(mock_surface):
    profile = WavelengthDependentPhaseProfile(_children())
    model = PhaseInteractionModel(mock_surface, profile, is_reflective=False)

    restored = BaseInteractionModel.from_dict(model.to_dict(), mock_surface)
    assert isinstance(restored.phase_profile, WavelengthDependentPhaseProfile)
    assert restored.phase_profile.to_dict() == profile.to_dict()
    assert restored.phase_profile.parent_surface is mock_surface


# --- PhaseInteractionModel ----------------------------------------------------


def test_real_rays_use_the_grating_of_their_own_wavelength(mock_surface):
    periods = {W1: 1.0, W2: 0.5}
    profile = WavelengthDependentPhaseProfile(
        {w: LinearGratingPhaseProfile(period=p) for w, p in periods.items()}
    )
    model = PhaseInteractionModel(mock_surface, profile, is_reflective=False)
    w = be.array([W1, W2])
    rays = RealRays(
        x=be.zeros(2),
        y=be.zeros(2),
        z=be.zeros(2),
        L=be.zeros(2),
        M=be.zeros(2),
        N=be.ones(2),
        wavelength=w,
        intensity=be.ones(2),
    )
    rays = model.interact_real_rays(rays)

    # Grating equation at normal incidence: n2 * L = m * lambda / d
    n2 = 1.5
    expected_L = [wl * 1e-3 / (periods[wl] * n2) for wl in (W1, W2)]
    assert_allclose(rays.L, be.array(expected_L), atol=1e-9)


def test_paraxial_rays_use_the_grating_of_their_wavelength(mock_surface):
    periods = {W1: 1.0, W2: 0.5}
    profile = WavelengthDependentPhaseProfile(
        {
            w: LinearGratingPhaseProfile(period=p, angle=be.pi / 2)
            for w, p in periods.items()
        }
    )
    model = PhaseInteractionModel(mock_surface, profile, is_reflective=False)
    rays = ParaxialRays(
        y=be.array([0.0]), u=be.array([0.0]), z=be.array([0.0]), wavelength=W2
    )
    rays = model.interact_paraxial_rays(rays)

    expected_u = -(W2 * 1e-3) / (periods[W2] * 1.5)
    assert_allclose(rays.u, be.array([expected_u]), atol=1e-9)


# --- full optic ---------------------------------------------------------------


def _focusing_coefficients(wavelength, f):
    k = 2 * np.pi / (wavelength * 1e-3)
    return [-k / (2 * f), k / (8 * f**3), -k / (16 * f**5)]


def _wavelength_corrected_lens(f):
    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        radius=be.inf,
        thickness=f,
        is_stop=True,
        phase_profile=WavelengthDependentPhaseProfile(
            {
                w: RadialPhaseProfile(coefficients=_focusing_coefficients(w, f))
                for w in (W1, W2)
            }
        ),
    )
    lens.surfaces.add(index=2)
    lens.set_aperture("EPD", 20.0)
    lens.fields.set_type("angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=W1)
    lens.wavelengths.add(value=W2, is_primary=True)
    return lens


def _marginal_ray_heights(lens, wavelength):
    py = be.linspace(0.0, 1.0, 6)
    zero = be.zeros_like(py)
    rays = lens.trace_generic(zero, zero, zero, py, wavelength)
    return be.to_numpy(rays.y)


def test_optic_focuses_every_defined_wavelength(set_test_backend):
    # A single radial profile would focus W1 and W2 at different distances;
    # one profile per wavelength focuses both at f.
    f = 100.0
    lens = _wavelength_corrected_lens(f)
    for wavelength in (W1, W2):
        assert np.max(np.abs(_marginal_ray_heights(lens, wavelength))) < 4e-6

    with pytest.raises(ValueError, match="no phase profile"):
        lens.trace_generic(
            be.zeros(1), be.zeros(1), be.zeros(1), be.ones(1), W_UNDEFINED
        )


def test_optic_round_trip(set_test_backend):
    f = 100.0
    lens = _wavelength_corrected_lens(f)
    data = json.loads(json.dumps(lens.to_dict()))
    restored = Optic.from_dict(data)

    profile = restored.surfaces[1].interaction_model.phase_profile
    assert isinstance(profile, WavelengthDependentPhaseProfile)
    for wavelength in (W1, W2):
        np.testing.assert_allclose(
            _marginal_ray_heights(restored, wavelength),
            _marginal_ray_heights(lens, wavelength),
            atol=1e-12,
        )


# --- differentiability --------------------------------------------------------


def test_torch_gradients_reach_only_the_matching_child(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("gradient flow is a PyTorch-only property")
    import torch

    a1 = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
    a2 = torch.tensor(-5.0, dtype=torch.float64, requires_grad=True)
    profile = WavelengthDependentPhaseProfile(
        {
            W1: RadialPhaseProfile(coefficients=[a1]),
            W2: RadialPhaseProfile(coefficients=[a2]),
        }
    )
    x, y = _points()
    w = be.array([W1, W2, W2, W1])
    profile.get_phase(x, y, w).sum().backward()

    # phi = a * r^2, so d(sum phi)/da is the sum of r^2 over that child's rays.
    r2 = be.to_numpy(x) ** 2 + be.to_numpy(y) ** 2
    is_w1 = np.array([True, False, False, True])
    assert a1.grad is not None
    assert a2.grad is not None
    np.testing.assert_allclose(a1.grad.item(), r2[is_w1].sum())
    np.testing.assert_allclose(a2.grad.item(), r2[~is_w1].sum())
