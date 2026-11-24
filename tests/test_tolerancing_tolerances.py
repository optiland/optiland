"""Tests for tolerances: add_tolerance, tolerance types, distributions, and
their use in SensitivityAnalysis and MonteCarlo."""

from __future__ import annotations

import math

import numpy as np
import pytest

import optiland.backend as be
from optiland.samples.objectives import CookeTriplet
from optiland.tolerancing import (
    EndPoint,
    Parabolic,
    Perturbation,
    RangeSampler,
    SensitivityAnalysis,
    Tolerance,
    ToleranceType,
    Tolerancing,
    TruncatedNormal,
    Uniform,
    tolerance_registry,
)
from optiland.tolerancing.base import BasePerturbation, Phase
from optiland.tolerancing.distributions import get_distribution
from optiland.tolerancing.monte_carlo import MonteCarlo
from optiland.tolerancing.registry import ToleranceRegistry
from optiland.tolerancing.types import FringesTolerance
from tests.utils import assert_allclose


def _f(value) -> float:
    return float(np.ravel(be.to_numpy(value))[0])


def _state(optic) -> list[tuple[float, ...]]:
    """Return radius, conic, thickness and pose of every surface."""
    rows = []
    for i, surface in enumerate(optic.surfaces):
        cs = surface.geometry.cs
        rows.append(
            (
                _f(surface.geometry.radius),
                _f(getattr(surface.geometry, "k", 0.0)),
                _f(optic.surfaces.get_thickness(i))
                if i < len(optic.surfaces) - 1
                else 0.0,
                _f(cs.x),
                _f(cs.y),
                _f(cs.z),
                _f(cs.rx),
                _f(cs.ry),
            )
        )
    return rows


@pytest.fixture
def tolerancing(set_test_backend):
    """Tolerancing on a Cooke triplet, built after the backend is set."""
    optic = CookeTriplet()
    tol = Tolerancing(optic)
    tol.add_operand(
        "rms_spot_size",
        {
            "optic": optic,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 0.0,
            "num_rays": 5,
            "wavelength": 0.55,
            "distribution": "hexapolar",
        },
    )
    return tol


def test_registry_builtin_types_registered():
    for name in (
        "radius",
        "fringes",
        "conic",
        "thickness",
        "surface_decenter",
        "surface_tilt",
    ):
        assert name in tolerance_registry


def test_registry_unknown_type():
    with pytest.raises(ValueError, match="Unknown tolerance type"):
        tolerance_registry.get("not_a_type")


def test_registry_register_rejects_non_subclass():
    registry = ToleranceRegistry()
    with pytest.raises(TypeError):
        registry.register("bad", object)


def test_registry_register_duplicate():
    registry = ToleranceRegistry()

    class Dummy(ToleranceType):
        label = "Dummy"

        def apply(self, delta):
            pass

        def reset(self):
            pass

    registry.register("dummy", Dummy)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("dummy", Dummy)
    registry.register("dummy", Dummy, overwrite=True)
    assert registry.names() == ["dummy"]


def test_registry_custom_type_through_add_tolerance(tolerancing):
    applied = []

    class Recorder(ToleranceType):
        label = "Recorder"

        def apply(self, delta):
            applied.append(delta)

        def reset(self):
            applied.append("reset")

    tolerance_registry.register("recorder", Recorder)
    try:
        tol = tolerancing.add_tolerance("recorder", limit=1.0)
        tol.apply_value(0.5)
        tol.reset()
        assert applied == [0.5, "reset"]
    finally:
        tolerance_registry._types.pop("recorder")


def test_add_tolerance_symmetric_limit(tolerancing, set_test_backend):
    tol = tolerancing.add_tolerance("radius", surface_number=1, limit=0.1)
    assert isinstance(tol, Tolerance)
    assert tol.limits == (-0.1, 0.1)
    assert tolerancing.perturbations == [tol]


def test_add_tolerance_asymmetric_limits(tolerancing, set_test_backend):
    tol = tolerancing.add_tolerance(
        "thickness", surface_number=2, minus=-0.02, plus=0.05
    )
    assert tol.limits == (-0.02, 0.05)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"minus": -0.1},
        {"limit": 0.1, "plus": 0.1},
        {"limit": -0.1},
        {"minus": 0.1, "plus": -0.1},
    ],
)
def test_add_tolerance_invalid_limits(tolerancing, kwargs):
    with pytest.raises(ValueError):
        tolerancing.add_tolerance("radius", surface_number=1, **kwargs)


def test_add_tolerance_default_distribution(tolerancing):
    tol = tolerancing.add_tolerance("radius", surface_number=1, limit=0.1)
    assert isinstance(tol.distribution, TruncatedNormal)
    assert tol.distribution.num_sigma == 2.0


def test_add_tolerance_str(tolerancing):
    tol = tolerancing.add_tolerance("radius", surface_number=1, limit=0.1)
    assert str(tol) == "Radius S1: [-0.1, +0.1] lens units"


@pytest.mark.parametrize(
    "tolerance_type, kwargs, getter, scale",
    [
        ("radius", {"surface_number": 1}, lambda s: s.geometry.radius, 1.0),
        ("conic", {"surface_number": 1}, lambda s: s.geometry.k, 1.0),
        (
            "surface_decenter",
            {"surface_number": 3, "axis": "x"},
            lambda s: s.geometry.cs.x,
            1.0,
        ),
        (
            "surface_decenter",
            {"surface_number": 3, "axis": "y"},
            lambda s: s.geometry.cs.y,
            1.0,
        ),
        (
            "surface_tilt",
            {"surface_number": 3, "axis": "x"},
            lambda s: s.geometry.cs.rx,
            math.pi / 180,
        ),
        (
            "surface_tilt",
            {"surface_number": 3, "axis": "y"},
            lambda s: s.geometry.cs.ry,
            math.pi / 180,
        ),
    ],
)
def test_surface_apply_and_reset(
    tolerancing, set_test_backend, tolerance_type, kwargs, getter, scale
):
    optic = tolerancing.optic
    nominal_state = _state(optic)
    surface = optic.surfaces[kwargs["surface_number"]]
    nominal = _f(getter(surface))

    tol = tolerancing.add_tolerance(tolerance_type, limit=0.2, **kwargs)
    tol.apply_value(0.1)
    assert_allclose(_f(getter(surface)), nominal + 0.1 * scale)
    assert tol.value == 0.1

    # only the targeted surface changed
    changed = [
        i
        for i, (a, b) in enumerate(zip(nominal_state, _state(optic), strict=True))
        if not np.allclose(a, b)
    ]
    assert changed == [kwargs["surface_number"]]

    tol.reset()
    assert tol.value == 0.0
    assert_allclose(np.array(_state(optic)), np.array(nominal_state))


def test_surface_thickness_moves_downstream(tolerancing, set_test_backend):
    optic = tolerancing.optic
    z_nominal = [_f(s.geometry.cs.z) for s in optic.surfaces]
    tol = tolerancing.add_tolerance("thickness", surface_number=2, limit=0.1)
    tol.apply_value(0.05)
    assert_allclose(
        _f(optic.surfaces.get_thickness(2)), _f(z_nominal[3]) - z_nominal[2] + 0.05
    )
    for i in range(3, len(optic.surfaces)):
        assert_allclose(_f(optic.surfaces[i].geometry.cs.z), z_nominal[i] + 0.05)
    tol.reset()
    z_after = [_f(s.geometry.cs.z) for s in optic.surfaces]
    assert_allclose(np.array(z_after), np.array(z_nominal))


def test_surface_thickness_adjust_keeps_downstream_fixed(tolerancing, set_test_backend):
    optic = tolerancing.optic
    z_nominal = [_f(s.geometry.cs.z) for s in optic.surfaces]
    tol = tolerancing.add_tolerance("thickness", surface_number=2, adjust=3, limit=0.1)
    tol.apply_value(0.05)
    assert_allclose(_f(optic.surfaces[3].geometry.cs.z), z_nominal[3] + 0.05)
    for i in range(4, len(optic.surfaces)):
        assert_allclose(_f(optic.surfaces[i].geometry.cs.z), z_nominal[i])
    tol.reset()
    z_after = [_f(s.geometry.cs.z) for s in optic.surfaces]
    assert_allclose(np.array(z_after), np.array(z_nominal))


def test_surface_thickness_adjust_same_surface(tolerancing):
    with pytest.raises(ValueError, match="adjust"):
        tolerancing.add_tolerance("thickness", surface_number=2, adjust=2, limit=0.1)


def test_surface_radius_on_flat_surface_raises(tolerancing):
    # the stop of the Cooke triplet is not flat; make surface 1 flat
    tolerancing.optic.surfaces[1].geometry.radius = be.array(np.inf)
    with pytest.raises(ValueError, match="fringes"):
        tolerancing.add_tolerance("radius", surface_number=1, limit=0.1)


@pytest.mark.parametrize("axis", ["z", "a"])
@pytest.mark.parametrize("tolerance_type", ["surface_decenter", "surface_tilt"])
def test_surface_invalid_axis(tolerancing, tolerance_type, axis):
    with pytest.raises(ValueError, match="axis"):
        tolerancing.add_tolerance(
            tolerance_type, surface_number=1, axis=axis, limit=0.1
        )


def test_surface_phases(tolerancing):
    tilt = tolerancing.add_tolerance(
        "surface_tilt", surface_number=1, axis="x", limit=0.1
    )
    radius = tolerancing.add_tolerance("radius", surface_number=1, limit=0.1)
    legacy = Perturbation(
        tolerancing.optic, "radius", RangeSampler(10, 20, 3), surface_number=2
    )
    tolerancing.perturbations.append(legacy)
    assert tilt.phase is Phase.RIGID_BODY
    assert radius.phase is Phase.STRUCTURE
    assert tolerancing.ordered_perturbations() == [radius, legacy, tilt]


def _sag(radius: float, k: float, h: float) -> float:
    c = 0.0 if math.isinf(radius) else 1.0 / radius
    return c * h**2 / (1 + math.sqrt(1 - (1 + k) * c**2 * h**2))


def test_fringes_zero_fringes_is_nominal(tolerancing, set_test_backend):
    tol = tolerancing.add_tolerance("fringes", surface_number=1, limit=2)
    assert_allclose(tol.type.radius_for(0.0), tol.type.nominal)


@pytest.mark.parametrize("surface_number", [1, 2, 4])
@pytest.mark.parametrize("fringes", [-3.0, 1.5])
def test_fringes_sag_change_is_half_wave_per_fringe(
    tolerancing, set_test_backend, surface_number, fringes
):
    tol = tolerancing.add_tolerance("fringes", surface_number=surface_number, limit=5)
    t = tol.type
    h = t.semi_aperture
    s0 = _sag(t.nominal, t.conic, h)
    s1 = _sag(t.radius_for(fringes), t.conic, h)
    # positive fringes increase |sag|
    assert_allclose(abs(s1) - abs(s0), fringes * 0.6328e-3 / 2, atol=1e-12)


def test_fringes_conic_surface_exact(tolerancing, set_test_backend):
    tolerancing.optic.surfaces[1].geometry.k = be.array(-0.7)
    tol = tolerancing.add_tolerance(
        "fringes", surface_number=1, limit=5, semi_aperture=5.0
    )
    t = tol.type
    s0 = _sag(t.nominal, -0.7, 5.0)
    s1 = _sag(t.radius_for(2.0), -0.7, 5.0)
    assert_allclose(abs(s1) - abs(s0), 2.0 * 0.6328e-3 / 2, atol=1e-12)


def test_fringes_flat_surface(tolerancing):
    tolerancing.optic.surfaces[1].geometry.radius = be.array(np.inf)
    tol = tolerancing.add_tolerance(
        "fringes", surface_number=1, limit=5, semi_aperture=10.0
    )
    assert math.isinf(tol.type.radius_for(0.0))
    radius = tol.type.radius_for(1.0)
    assert radius > 0
    assert_allclose(_sag(radius, 0.0, 10.0), 0.6328e-3 / 2, atol=1e-12)


def test_fringes_custom_test_wavelength(tolerancing):
    tol = tolerancing.add_tolerance(
        "fringes", surface_number=1, limit=5, test_wavelength=0.5461
    )
    t = tol.type
    h = t.semi_aperture
    ds = abs(_sag(t.radius_for(1.0), t.conic, h)) - abs(_sag(t.nominal, t.conic, h))
    assert_allclose(ds, 0.5461e-3 / 2, atol=1e-12)


def test_fringes_default_semi_aperture_uses_surface_value(tolerancing):
    tolerancing.optic.surfaces[1].set_semi_aperture(6.5)
    tol = FringesTolerance(tolerancing.optic, surface_number=1)
    assert_allclose(tol.semi_aperture, 6.5)


def test_fringes_invalid_semi_aperture(tolerancing):
    with pytest.raises(ValueError, match="semi_aperture"):
        tolerancing.add_tolerance(
            "fringes", surface_number=1, limit=1, semi_aperture=0.0
        )


def test_fringes_apply_and_reset(tolerancing, set_test_backend):
    optic = tolerancing.optic
    nominal = _f(optic.surfaces[2].geometry.radius)
    tol = tolerancing.add_tolerance("fringes", surface_number=2, limit=5)
    tol.apply_value(3.0)
    assert_allclose(_f(optic.surfaces[2].geometry.radius), tol.type.radius_for(3))
    tol.reset()
    assert_allclose(_f(optic.surfaces[2].geometry.radius), nominal)


@pytest.mark.parametrize(
    "distribution", [TruncatedNormal(), Uniform(), Parabolic(), EndPoint()]
)
def test_distribution_within_limits(distribution):
    rng = np.random.default_rng(0)
    values = np.array([distribution.sample(rng, -0.2, 0.6) for _ in range(2000)])
    assert values.min() >= -0.2
    assert values.max() <= 0.6
    assert_allclose(values.mean(), 0.2, atol=0.03)


def test_distribution_end_point_only_limits():
    rng = np.random.default_rng(1)
    values = {EndPoint().sample(rng, -1.0, 2.0) for _ in range(50)}
    assert values == {-1.0, 2.0}


def test_distribution_truncated_normal_spread():
    rng = np.random.default_rng(2)
    values = np.array([TruncatedNormal().sample(rng, -1, 1) for _ in range(20000)])
    # std of a standard normal truncated at ±2, scaled by 1/2
    assert_allclose(values.std(), 0.8796 / 2, atol=0.01)


def test_distribution_parabolic_favours_limits():
    rng = np.random.default_rng(3)
    values = np.array([Parabolic().sample(rng, -1, 1) for _ in range(20000)])
    assert np.mean(np.abs(values) > 0.5) > 0.8


def test_distribution_zero_width():
    rng = np.random.default_rng(4)
    assert TruncatedNormal().sample(rng, 0.3, 0.3) == 0.3


def test_distribution_lookup():
    assert isinstance(get_distribution("uniform"), Uniform)
    instance = Parabolic()
    assert get_distribution(instance) is instance
    with pytest.raises(ValueError, match="Unknown distribution"):
        get_distribution("gaussian")


def test_distribution_invalid_num_sigma():
    with pytest.raises(ValueError):
        TruncatedNormal(num_sigma=0)


def _add_tolerances(tolerancing):
    tolerancing.add_tolerance("radius", surface_number=1, limit=0.05)
    tolerancing.add_tolerance("thickness", surface_number=2, minus=-0.02, plus=0.04)
    tolerancing.add_tolerance("surface_tilt", surface_number=3, axis="x", limit=0.05)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_analysis_sensitivity_evaluates_limits(tolerancing, set_test_backend):
    _add_tolerances(tolerancing)
    nominal_state = _state(tolerancing.optic)
    sa = SensitivityAnalysis(tolerancing)
    sa.run()
    df = sa.get_results()
    assert list(df["perturbation_type"]) == [
        "Radius S1",
        "Radius S1",
        "Thickness S2",
        "Thickness S2",
        "Tilt X S3",
        "Tilt X S3",
    ]
    assert_allclose(
        df["perturbation_value"].to_numpy(),
        np.array([-0.05, 0.05, -0.02, 0.04, -0.05, 0.05]),
    )
    assert np.all(np.isfinite(df["0: rms spot size"].to_numpy()))
    assert_allclose(np.array(_state(tolerancing.optic)), np.array(nominal_state))


def test_analysis_sensitivity_mixes_legacy_perturbations(tolerancing, set_test_backend):
    optic = tolerancing.optic
    nominal_radius = _f(optic.surfaces[2].geometry.radius)
    tolerancing.add_tolerance("conic", surface_number=1, limit=0.1)
    tolerancing.add_perturbation(
        "radius",
        RangeSampler(nominal_radius - 1, nominal_radius + 1, 3),
        surface_number=2,
    )
    sa = SensitivityAnalysis(tolerancing)
    sa.run()
    df = sa.get_results()
    assert len(df) == 2 + 3
    assert list(df["perturbation_type"])[2:] == ["Radius of Curvature, Surface 2"] * 3


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_analysis_monte_carlo(tolerancing, set_test_backend):
    _add_tolerances(tolerancing)
    nominal_state = _state(tolerancing.optic)
    mc = MonteCarlo(tolerancing)
    mc.run(8)
    df = mc.get_results()
    assert len(df) == 8
    for tol in tolerancing.perturbations:
        column = df[tol.label].to_numpy()
        assert column.min() >= tol.minus
        assert column.max() <= tol.plus
    assert np.all(np.isfinite(df["0: rms spot size"].to_numpy()))
    assert_allclose(np.array(_state(tolerancing.optic)), np.array(nominal_state))


def test_legacy_perturbation_is_base_perturbation():
    optic = CookeTriplet()
    perturbation = Perturbation(
        optic, "radius", RangeSampler(10, 20, 3), surface_number=1
    )
    assert isinstance(perturbation, BasePerturbation)
    assert perturbation.label == str(perturbation.variable)
    assert perturbation.phase is Phase.STRUCTURE
