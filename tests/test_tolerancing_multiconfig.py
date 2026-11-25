"""Tests for MultiConfigTolerancing: tolerancing a multi-configuration system."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import optiland.backend as be
from optiland.iso10110 import DrawingSpec
from optiland.materials.offset import OffsetMaterial
from optiland.multiconfig import MultiConfiguration
from optiland.optimization.operand import operand_registry
from optiland.samples.objectives import CookeTriplet, DoubleGauss
from optiland.tolerancing import (
    ConfigurationGroupTolerance,
    InverseSensitivity,
    MultiConfigTolerancing,
    SensitivityAnalysis,
    Tolerancing,
)
from optiland.tolerancing.monte_carlo import MonteCarlo
from tests.utils import assert_allclose

ZOOM_GAP = 4  # air gap that differs between the configurations
IMAGE_GAP = 6  # gap in front of the image, refocused per configuration

SPOT = {
    "surface_number": -1,
    "Hx": 0.0,
    "Hy": 1.0,
    "num_rays": 3,
    "wavelength": 0.55,
}


def _f(value) -> float:
    return float(np.ravel(be.to_numpy(value))[0])


def _zoom(independent_focus: bool = False) -> MultiConfiguration:
    """A two-configuration Cooke triplet that differs in one air gap."""
    lens = CookeTriplet()
    mc = MultiConfiguration(lens)
    config = mc.add_configuration()
    mc.set_thickness(ZOOM_GAP, 6.0, [1])
    if independent_focus:
        image_gap = _f(lens.surfaces.surfaces[IMAGE_GAP].thickness)
        mc.set_thickness(IMAGE_GAP, image_gap - 1.0, [1])
    config.updater.update()
    return mc


def _state(optic) -> list[float]:
    """Radii, thicknesses, vertex positions and materials of an optic."""
    values = []
    for k, surface in enumerate(optic.surfaces):
        cs = surface.geometry.cs
        values += [_f(getattr(surface.geometry, "radius", np.inf))]
        values += [_f(getattr(cs, a)) for a in ("x", "y", "z", "rx", "ry")]
        if k < optic.surfaces.num_surfaces - 1:
            values.append(_f(optic.surfaces.get_thickness(k)) if k else 0.0)
    return values


def _materials(optic) -> list[int]:
    return [id(surface.material_post) for surface in optic.surfaces]


@pytest.fixture
def zoom(set_test_backend):
    return _zoom()


def test_plain_tolerancing_does_not_reach_other_configuration(zoom):
    """Motivating regression (D3): with a plain Tolerancing, a tolerance on
    configuration 0 leaves configuration 1 at nominal."""
    c0, c1 = zoom.configurations
    nominal = [_f(c0.paraxial.f2()), _f(c1.paraxial.f2())]

    plain = Tolerancing(c0)
    plain.add_operand("f2", input_data={"optic": c0})
    plain.add_operand("f2", input_data={"optic": c1})
    plain.add_tolerance("radius", 0.5, surface_number=1).apply_value(0.5)
    plain_values = plain.evaluate()
    plain.reset()
    assert plain_values[0] != pytest.approx(nominal[0])
    assert_allclose(plain_values[1], nominal[1])

    tolerancing = MultiConfigTolerancing(zoom)
    tolerancing.add_operand("f2", configuration="all")
    tolerancing.add_tolerance("radius", 0.5, surface_number=1).apply_value(0.5)
    values = tolerancing.evaluate()
    tolerancing.reset()
    assert_allclose(values[0], plain_values[0])
    assert values[1] != pytest.approx(nominal[1])


def test_radius_tolerance_moves_f2_of_both_configurations(zoom):
    c0, c1 = zoom.configurations
    radius = _f(c0.surfaces[1].geometry.radius)
    tolerancing = MultiConfigTolerancing(zoom)
    tolerancing.add_operand("f2", configuration="all")
    nominal = tolerancing.evaluate()
    tolerance = tolerancing.add_tolerance("radius", 0.5, surface_number=1)

    # config 1's radius follows config 0 through its pickup
    assert isinstance(tolerance.type, ConfigurationGroupTolerance)
    assert tolerance.type.direct == [True, False]
    assert tolerance.label == "Radius S1"

    tolerance.apply_value(0.5)
    assert_allclose(c0.surfaces[1].geometry.radius, radius + 0.5)
    assert_allclose(c1.surfaces[1].geometry.radius, radius + 0.5)
    perturbed = tolerancing.evaluate()
    assert perturbed[0] != pytest.approx(nominal[0])
    assert perturbed[1] != pytest.approx(nominal[1])

    # the same change made by hand on a fresh zoom gives the same f2
    reference = _zoom()
    reference.set_radius(1, radius + 0.5)
    reference.configurations[1].updater.update()
    assert_allclose(perturbed[1], reference.configurations[1].paraxial.f2())


def test_decenter_tolerance_reaches_configuration_1(zoom):
    c0, c1 = zoom.configurations
    tolerancing = MultiConfigTolerancing(zoom)
    tolerancing.add_operand("rms_spot_size", SPOT, configuration=1)
    nominal = tolerancing.evaluate()[0]
    tolerance = tolerancing.add_tolerance(
        "surface_decenter", 0.2, surface_number=3, axis="y"
    )
    assert tolerance.type.direct == [True, True]

    tolerance.apply_value(0.2)
    assert_allclose(c1.surfaces[3].geometry.cs.y, 0.2)
    assert tolerancing.evaluate()[0] != pytest.approx(nominal)
    tolerancing.reset()
    assert_allclose(c1.surfaces[3].geometry.cs.y, 0.0)
    assert_allclose(tolerancing.evaluate()[0], nominal)


def test_scaled_pickup_member_follows_its_pickup(zoom):
    c0, c1 = zoom.configurations
    pickup = next(
        p
        for p in c1.pickups.pickups
        if p.attr_type == "radius" and p.target_surface_idx == 1
    )
    pickup.scale, pickup.offset = 1.1, 0.5
    radius = _f(c0.surfaces[1].geometry.radius)

    tolerancing = MultiConfigTolerancing(zoom)
    assert_allclose(c1.surfaces[1].geometry.radius, 1.1 * radius + 0.5)
    tolerance = tolerancing.add_tolerance("radius", 0.3, surface_number=1)
    tolerance.apply_value(0.3)
    assert_allclose(c0.surfaces[1].geometry.radius, radius + 0.3)
    assert_allclose(c1.surfaces[1].geometry.radius, 1.1 * (radius + 0.3) + 0.5)
    tolerancing.reset()
    assert_allclose(c1.surfaces[1].geometry.radius, 1.1 * radius + 0.5)


def test_unlinked_thickness_gets_delta_in_every_configuration(zoom):
    c0, c1 = zoom.configurations
    gaps = [_f(c.surfaces.get_thickness(ZOOM_GAP)) for c in (c0, c1)]
    tolerancing = MultiConfigTolerancing(zoom)
    tolerance = tolerancing.add_tolerance("thickness", 0.1, surface_number=ZOOM_GAP)
    assert tolerance.type.direct == [True, True]
    tolerance.apply_value(0.1)
    assert_allclose(c0.surfaces.get_thickness(ZOOM_GAP), gaps[0] + 0.1)
    assert_allclose(c1.surfaces.get_thickness(ZOOM_GAP), gaps[1] + 0.1)

    tolerancing.reset()
    only_1 = tolerancing.add_tolerance(
        "thickness", 0.1, surface_number=ZOOM_GAP, configurations=[1]
    )
    only_1.apply_value(-0.1)
    assert_allclose(c0.surfaces.get_thickness(ZOOM_GAP), gaps[0])
    assert_allclose(c1.surfaces.get_thickness(ZOOM_GAP), gaps[1] - 0.1)
    tolerancing.reset()


def test_material_tolerance_follows_material_pickup(zoom):
    c0, c1 = zoom.configurations
    zoom.set_material(3, c0.surfaces[3].material_post, "all")
    tolerancing = MultiConfigTolerancing(zoom)
    tolerance = tolerancing.add_tolerance("index", 0.001, surface_number=3)
    assert tolerance.type.direct == [True, False]
    tolerance.apply_value(0.001)
    assert isinstance(c0.surfaces[3].material_post, OffsetMaterial)
    assert c1.surfaces[3].material_post is c0.surfaces[3].material_post
    tolerancing.reset()
    assert not isinstance(c1.surfaces[3].material_post, OffsetMaterial)


def test_pickup_from_unselected_configuration_raises(zoom):
    tolerancing = MultiConfigTolerancing(zoom)
    with pytest.raises(ValueError, match="configuration 0, which is not selected"):
        tolerancing.add_tolerance("radius", 0.1, surface_number=1, configurations=[1])


@pytest.mark.parametrize("configurations", [[], [0, 0], [2]])
def test_invalid_selection_raises(zoom, configurations):
    tolerancing = MultiConfigTolerancing(zoom)
    with pytest.raises(ValueError):
        tolerancing.add_tolerance(
            "radius", 0.1, surface_number=1, configurations=configurations
        )


def test_different_surface_counts_raise(set_test_backend):
    mc = MultiConfiguration(CookeTriplet())
    mc.configurations.append(DoubleGauss())
    with pytest.raises(ValueError, match="same number of surfaces"):
        MultiConfigTolerancing(mc)


def test_operand_names(zoom):
    c0, c1 = zoom.configurations
    tolerancing = MultiConfigTolerancing(zoom)
    tolerancing.add_operand("f2")
    tolerancing.add_operand("f2", configuration="all")
    assert [str(op) for op in tolerancing.operands] == [
        "f2",
        "f2 (config 0)",
        "f2 (config 1)",
    ]
    assert [op.input_data["optic"] for op in tolerancing.operands] == [c0, c0, c1]
    with pytest.raises(ValueError, match="input_data"):
        tolerancing.add_operand("f2", {"optic": c1}, configuration=0)


def test_per_configuration_thickness_compensator(set_test_backend):
    mc = _zoom(independent_focus=True)
    c0, c1 = mc.configurations
    gaps = [_f(c.surfaces.get_thickness(IMAGE_GAP)) for c in (c0, c1)]
    tolerancing = MultiConfigTolerancing(mc)
    tolerancing.add_operand("rms_spot_size", SPOT, target=0.0, configuration="all")
    tolerancing.add_tolerance("radius", 0.5, surface_number=1)
    tolerancing.add_compensator(
        "thickness", surface_number=IMAGE_GAP, configuration="all"
    )
    assert [v.optic for v in tolerancing.compensator.variables] == [c0, c1]

    tolerancing.perturbations[0].apply_value(0.5)
    before = tolerancing.evaluate()
    result = tolerancing.apply_compensators()
    after = tolerancing.evaluate()
    assert len(result) == 2
    assert after[0] < before[0]
    assert after[1] < before[1]
    for optic, gap in zip((c0, c1), gaps, strict=True):
        assert _f(optic.surfaces.get_thickness(IMAGE_GAP)) != pytest.approx(gap)

    tolerancing.reset()
    for optic, gap in zip((c0, c1), gaps, strict=True):
        assert_allclose(optic.surfaces.get_thickness(IMAGE_GAP), gap)


def test_per_configuration_quick_focus(zoom):
    c0, c1 = zoom.configurations
    image_z = [_f(c.surfaces[-1].geometry.cs.z) for c in (c0, c1)]
    tolerancing = MultiConfigTolerancing(zoom)
    tolerancing.add_compensator("quick_focus", configuration="all")
    assert sorted(tolerancing.quick_focus_compensators) == [0, 1]

    tolerancing.add_tolerance("radius", 0.5, surface_number=1).apply_value(0.5)
    result = tolerancing.apply_compensators()
    assert list(result) == [
        "quick_focus: image distance",
        "quick_focus: image distance (config 1)",
    ]
    for optic, z in zip((c0, c1), image_z, strict=True):
        assert _f(optic.surfaces[-1].geometry.cs.z) != pytest.approx(z)

    tolerancing.reset()
    for optic, z in zip((c0, c1), image_z, strict=True):
        assert_allclose(optic.surfaces[-1].geometry.cs.z, z)


def test_reset_restores_every_configuration(set_test_backend):
    mc = _zoom(independent_focus=True)
    mc.set_material(3, mc.configurations[0].surfaces[3].material_post, "all")
    configs = mc.configurations
    tolerancing = MultiConfigTolerancing(mc)
    nominal = [_state(c) for c in configs]
    materials = [_materials(c) for c in configs]

    tolerancing.add_tolerance("radius", 0.2, surface_number=1)
    tolerancing.add_tolerance("thickness", 0.05, surface_number=2, adjust=ZOOM_GAP)
    tolerancing.add_tolerance("index", 0.001, surface_number=3)
    tolerancing.add_tolerance("irregularity", 0.5, surface_number=5)
    tolerancing.add_tolerance("surface_tilt", 0.05, surface_number=3, axis="x")
    tolerancing.add_tolerance("element_tilt", 0.05, first=5, last=6, axis="y")
    tolerancing.add_compensator("quick_focus", configuration="all")
    tolerancing.add_operand("rms_spot_size", SPOT, configuration="all")

    for tolerance in tolerancing.ordered_perturbations():
        tolerance.apply_value(tolerance.plus)
    tolerancing.apply_compensators()
    assert _state(configs[1]) != pytest.approx(nominal[1])
    tolerancing.reset()

    for config, state, material in zip(configs, nominal, materials, strict=True):
        assert_allclose(np.array(_state(config)), np.array(state))
        assert _materials(config) == material


def test_default_tolerances_and_drawing_reach_all_configurations(zoom):
    c0, c1 = zoom.configurations
    tolerancing = MultiConfigTolerancing(zoom)
    added = tolerancing.add_default_tolerances({"surface_decenter": 0.01})
    assert added
    for tolerance in added:
        assert [m.optic for m in tolerance.type.members] == [c0, c1]

    drawing = DrawingSpec(c0)
    drawing.set_surface(1, r_tolerance=0.05, sag_tolerance=2.0, test_wavelength=632.8)
    from_drawing = tolerancing.add_tolerances_from_drawing(drawing)
    assert {t.label for t in from_drawing} == {"Radius S1", "Fringes S1"}
    for tolerance in from_drawing:
        assert [m.optic for m in tolerance.type.members] == [c0, c1]
    fringes = next(t for t in from_drawing if t.label == "Fringes S1")
    # one physical surface: config 1 uses config 0's semi-aperture
    members = fringes.type.members
    assert members[1].semi_aperture == members[0].semi_aperture


def _worst_rms(optic, configurations):
    """Largest RMS spot size over the configurations (``optic`` is unused)."""
    rms = operand_registry.get("rms_spot_size")
    return max(float(be.to_numpy(rms(optic=c, **SPOT))) for c in configurations)


@pytest.fixture
def worst_rms_operand():
    operand_registry.register("mc_test_worst_rms", _worst_rms)
    yield "mc_test_worst_rms"
    operand_registry._registry.pop("mc_test_worst_rms")


def test_sensitivity_and_inverse_on_worst_configuration(zoom, worst_rms_operand):
    tolerancing = MultiConfigTolerancing(zoom)
    tolerancing.add_operand(worst_rms_operand, {"configurations": zoom.configurations})
    tolerancing.add_tolerance("radius", 0.5, surface_number=1)
    tolerancing.add_tolerance("surface_decenter", 0.05, surface_number=3, axis="y")
    nominal = [_state(c) for c in zoom.configurations]

    sensitivity = SensitivityAnalysis(tolerancing)
    sensitivity.run()
    df = sensitivity.get_results()
    assert len(df) == 4
    worst = df[sensitivity.operand_names[0]].to_numpy()
    assert np.all(worst != pytest.approx(sensitivity._nominal[0]))

    change = float(worst[1] - sensitivity._nominal[0])
    assert change > 0
    inverse = InverseSensitivity(tolerancing, {0: change})
    inverse.run()
    row = inverse.get_results().iloc[0]
    assert_allclose(row["inverse_plus"], 0.5, rtol=1e-3)
    for config, state in zip(zoom.configurations, nominal, strict=True):
        assert_allclose(np.array(_state(config)), np.array(state))


def test_monte_carlo_parallel_equals_serial():
    be.set_backend("numpy")
    mc = _zoom()
    tolerancing = MultiConfigTolerancing(mc)
    tolerancing.add_operand("rms_spot_size", SPOT, configuration="all")
    tolerancing.add_tolerance("radius", 0.2, surface_number=1)
    tolerancing.add_tolerance("thickness", 0.05, surface_number=ZOOM_GAP)
    tolerancing.add_tolerance("surface_decenter", 0.02, surface_number=3, axis="y")
    tolerancing.add_compensator("quick_focus", configuration="all")

    analysis = MonteCarlo(tolerancing)
    analysis.run(4, seed=3)
    serial = analysis.get_results().copy()
    assert "1: rms spot size (config 1)" in serial.columns
    assert "quick_focus: image distance (config 1)" in serial.columns

    analysis.run(4, seed=3, workers=2)
    pd.testing.assert_frame_equal(analysis.get_results(), serial)
    assert analysis.yield_fraction({0: ("<", 1.0), 1: ("<", 1.0)}) == 1.0
