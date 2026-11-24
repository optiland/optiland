"""Tests for element moves: the element_decenter, element_tilt and
element_roll optimization variables and tolerance types."""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials.ideal import IdealMaterial
from optiland.optic import Optic
from optiland.optimization.operand.ray import RayOperand
from optiland.optimization.variable import Variable
from optiland.optimization.variable.element import RigidElement
from optiland.samples.objectives import CookeTriplet
from optiland.tolerancing import SensitivityAnalysis, Tolerancing, tolerance_registry
from optiland.tolerancing.monte_carlo import MonteCarlo
from tests.utils import assert_allclose

N_PLATE = 1.5
T_PLATE = 10.0


def _f(value) -> float:
    return float(np.ravel(be.to_numpy(value))[0])


def _poses(optic) -> np.ndarray:
    """Return x, y, z, rx, ry, rz of every surface."""
    return np.array(
        [
            [_f(getattr(s.geometry.cs, a)) for a in ("x", "y", "z", "rx", "ry", "rz")]
            for s in optic.surfaces
        ]
    )


def _plate() -> Optic:
    """A stop in air, then a plane-parallel plate (surfaces 2-3)."""
    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf)
    optic.surfaces.add(index=1, thickness=5.0, is_stop=True)
    optic.surfaces.add(index=2, thickness=T_PLATE, material=IdealMaterial(N_PLATE))
    optic.surfaces.add(index=3, thickness=20.0)
    optic.surfaces.add(index=4)
    optic.set_aperture("EPD", 4.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def _chief_y(optic):
    return RayOperand.y_intercept(optic, -1, 0.0, 0.0, 0.0, 0.0, 0.55)


def _plate_shift(theta: float) -> float:
    """Exact lateral shift of a ray by a plate tilted by ``theta``."""
    s, c = math.sin(theta), math.cos(theta)
    return T_PLATE * s * (1.0 - c / math.sqrt(N_PLATE**2 - s**2))


# --- RigidElement ---


def test_tilt_about_center_moves_vertices(set_test_backend):
    optic = CookeTriplet()
    before = _poses(optic)
    element = RigidElement(optic, 3, 4, pivot="center")
    theta = 0.01
    element.rotate("x", theta)
    after = _poses(optic)

    zc = (before[3, 2] + before[4, 2]) / 2
    for k in (3, 4):
        dz = before[k, 2] - zc
        assert_allclose(after[k, 1], -dz * math.sin(theta), atol=1e-12)
        assert_allclose(after[k, 2], zc + dz * math.cos(theta), atol=1e-12)
        assert_allclose(after[k, 3], theta, atol=1e-15)
    # surfaces outside the element do not move
    others = [k for k in range(len(before)) if k not in (3, 4)]
    assert_allclose(after[others], before[others], atol=0)


@pytest.mark.parametrize("pivot", ["first", "last", 2.5])
def test_pivot_point_stays_fixed(set_test_backend, pivot):
    optic = CookeTriplet()
    element = RigidElement(optic, 3, 4, pivot=pivot)
    p = [_f(v) for v in element.pivot_point()]
    element.rotate("y", 0.02)
    # the pivot is a fixed point of the rotation
    assert_allclose([_f(v) for v in element.pivot_point()], p, atol=1e-12)


def test_single_surface_tilt_about_vertex_equals_surface_tilt(set_test_backend):
    optic = CookeTriplet()
    reference = CookeTriplet()
    RigidElement(optic, 3, 3, pivot="first").rotate("x", 0.03)
    reference.surfaces[3].geometry.cs.rx = 0.03
    assert_allclose(_poses(optic), _poses(reference), atol=1e-15)


def test_roll_composes_with_tilt(set_test_backend):
    optic = CookeTriplet()
    element = RigidElement(optic, 3, 4)
    element.rotate("x", 0.02)
    element.rotate("z", 0.1)
    # the roll is about the tilted element axis, so the axis is unchanged
    axis = [_f(v) for v in element.axis("z")]
    assert_allclose(axis, [0.0, -math.sin(0.02), math.cos(0.02)], atol=1e-14)


def test_invalid_element(set_test_backend):
    optic = CookeTriplet()
    last = optic.surfaces.num_surfaces - 1
    with pytest.raises(ValueError, match="first"):
        RigidElement(optic, 0, 2)
    with pytest.raises(ValueError, match="first"):
        RigidElement(optic, 3, last)
    with pytest.raises(ValueError, match="first"):
        RigidElement(optic, 4, 3)
    with pytest.raises(ValueError, match="pivot"):
        RigidElement(optic, 3, 4, pivot="middle")


# --- trace: plate lateral shift ---


@pytest.mark.parametrize("pivot", ["center", "first", 3.0])
def test_tilted_plate_shifts_ray_exactly(set_test_backend, pivot):
    optic = _plate()
    assert_allclose(_chief_y(optic), 0.0, atol=1e-12)
    theta = math.radians(5.0)
    RigidElement(optic, 2, 3, pivot=pivot).rotate("x", theta)
    assert_allclose(abs(_f(_chief_y(optic))), _plate_shift(theta), rtol=1e-9)


def test_decentered_plate_does_not_shift_ray(set_test_backend):
    optic = _plate()
    RigidElement(optic, 2, 3).decenter("y", 0.5)
    assert_allclose(_chief_y(optic), 0.0, atol=1e-12)


# --- optimization variables ---


@pytest.mark.parametrize(
    "type_name, kwargs",
    [
        ("element_decenter", {"axis": "y"}),
        ("element_tilt", {"axis": "x"}),
        ("element_roll", {}),
    ],
)
def test_variable_round_trip(set_test_backend, type_name, kwargs):
    optic = CookeTriplet()
    before = _poses(optic)
    variable = Variable(optic, type_name, first=3, last=4, **kwargs)
    assert _f(variable.value) == 0.0
    variable.update(0.01)
    assert_allclose(_f(variable.value), 0.01)
    assert not np.allclose(_poses(optic), before)
    variable.update(0.0)
    assert_allclose(_poses(optic), before, atol=1e-12)


def test_variable_keeps_thickness_changes():
    optic = _plate()
    tilt = Variable(optic, "element_tilt", first=2, last=3, axis="x")
    thickness = Variable(optic, "thickness", surface_number=2)
    # a thickness change made after the variable was created is kept
    thickness.update(thickness.value + 0.1)
    changed = _poses(optic)
    assert not np.allclose(changed, _poses(_plate()))
    tilt.update(0.05)
    tilt.update(0.0)
    assert_allclose(_poses(optic), changed, atol=1e-12)


def test_variable_str():
    optic = CookeTriplet()
    variable = Variable(optic, "element_tilt", first=3, last=4, axis="y")
    assert str(variable) == "Element Tilt Y, Surfaces 3-4"


def test_variable_gradient_matches_finite_difference(set_test_backend):
    if be.get_backend() != "torch":
        pytest.skip("autograd check runs on torch only")
    optic = _plate()
    variable = Variable(optic, "element_tilt", first=2, last=3, axis="x")
    theta0, h = 0.05, 1e-6

    theta = be.array(theta0)
    theta.requires_grad_(True)
    variable.update(theta)
    _chief_y(optic).sum().backward()
    gradient = float(theta.grad)

    variable.variable.release_graph()
    variable.update(theta0 + h)
    plus = _f(_chief_y(optic))
    variable.update(theta0 - h)
    minus = _f(_chief_y(optic))
    assert_allclose(gradient, (plus - minus) / (2 * h), rtol=1e-5)


# --- tolerance types ---


def test_types_registered():
    for name in ("element_decenter", "element_tilt", "element_roll"):
        assert name in tolerance_registry


def test_tolerance_labels(set_test_backend):
    tol = Tolerancing(CookeTriplet())
    decenter = tol.add_tolerance("element_decenter", first=3, last=4, axis="x", limit=1)
    tilt = tol.add_tolerance("element_tilt", first=3, last=4, axis="y", limit=1)
    roll = tol.add_tolerance("element_roll", first=3, last=4, limit=1)
    assert decenter.label == "Element Decenter X S3-4"
    assert tilt.label == "Element Tilt Y S3-4"
    assert roll.label == "Element Roll S3-4"


def test_tolerance_tilt_in_degrees(set_test_backend):
    optic = _plate()
    tol = Tolerancing(optic)
    tilt = tol.add_tolerance("element_tilt", first=2, last=3, axis="x", limit=5.0)
    tilt.apply_value(5.0)
    shift = abs(_f(_chief_y(optic)))
    assert_allclose(shift, _plate_shift(math.radians(5.0)), rtol=1e-9)
    tilt.reset()
    assert_allclose(_chief_y(optic), 0.0, atol=1e-12)


@pytest.mark.parametrize("thickness_first", [True, False])
def test_reset_restores_nominal_with_thickness_inside(
    set_test_backend, thickness_first
):
    optic = CookeTriplet()
    before = _poses(optic)
    tol = Tolerancing(optic)
    tol.add_operand(
        "rms_spot_size",
        {
            "optic": optic,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "num_rays": 3,
            "wavelength": 0.55,
        },
    )
    adds = [
        lambda: tol.add_tolerance("thickness", surface_number=3, limit=0.05),
        lambda: tol.add_tolerance("element_tilt", first=3, last=4, axis="x", limit=0.1),
        lambda: tol.add_tolerance(
            "element_decenter", first=3, last=4, axis="y", limit=0.02
        ),
    ]
    if not thickness_first:
        adds.reverse()
    for add in adds:
        add()

    SensitivityAnalysis(tol).run()
    assert_allclose(_poses(optic), before, atol=0)
    MonteCarlo(tol).run(4)
    assert_allclose(_poses(optic), before, atol=0)


def test_tilt_and_thickness_combine_in_trial(set_test_backend):
    """The tilt applies to the thickness-perturbed element (phase order)."""
    optic = CookeTriplet()
    tol = Tolerancing(optic)
    tilt = tol.add_tolerance("element_tilt", first=3, last=4, axis="x", limit=1.0)
    thickness = tol.add_tolerance("thickness", surface_number=3, limit=0.5)
    for tolerance in tol.ordered_perturbations():
        tolerance.apply_value(tolerance.plus)
    # with pivot "center", the vertices are 0.5 * thickness from the pivot
    cs3, cs4 = optic.surfaces[3].geometry.cs, optic.surfaces[4].geometry.cs
    distance = math.dist(
        [_f(cs3.x), _f(cs3.y), _f(cs3.z)], [_f(cs4.x), _f(cs4.y), _f(cs4.z)]
    )
    nominal = CookeTriplet().surfaces.get_thickness(3)
    assert_allclose(distance, _f(nominal) + 0.5, atol=1e-12)
    assert_allclose(_f(cs3.rx), math.radians(1.0), atol=1e-15)
    tol.reset()
    assert thickness.value == 0.0
    assert tilt.value == 0.0
