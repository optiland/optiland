"""Tests for the bridge between ISO 10110 drawings and tolerances."""

from __future__ import annotations

import math

import pytest

from optiland.iso10110 import DrawingSpec
from optiland.samples.objectives import CookeTriplet
from optiland.samples.simple import AsphericSinglet, CementedAchromat
from optiland.tolerancing import (
    SensitivityAnalysis,
    Tolerancing,
    drawing_values_from_tolerancing,
    tolerance_registry,
    tolerances_from_drawing,
)
from optiland.tolerancing.iso10110_bridge import _limits
from optiland.tolerancing.types import (
    FringesTolerance,
    IrregularityTolerance,
    ThicknessTolerance,
)
from tests.utils import assert_allclose

SURFACE_1 = {
    "r_tolerance": "±0.05",
    "sag_tolerance": 2.0,
    "irregularity": 0.5,
    "rotationally_symmetric": 0.25,
    "test_wavelength": 632.8,
    "centration": 2.0,
    "ca_diameter": 20.0,
}
SURFACE_2 = {
    "r_tolerance": "+0.1/-0.05",
    "irregularity": 1.0,
    "centration": 1.0,
    "centration_decentration": 0.01,
}


@pytest.fixture
def triplet(set_test_backend):
    """A Cooke triplet, its drawing with ISO 10110 entries, and a problem."""
    optic = CookeTriplet()
    drawing = DrawingSpec(optic)
    drawing.set_surface(1, **SURFACE_1)
    drawing.set_surface(2, **SURFACE_2)
    drawing.set_element(0, ct_tolerance=0.05)
    drawing.set_element(1, ct_tolerance="+0.1/-0.02")
    tolerancing = Tolerancing(optic)
    tolerancing.add_operand(
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
    return optic, drawing, tolerancing


def _by_label(tolerances):
    return {t.label: t for t in tolerances}


@pytest.mark.parametrize(
    "tol, expected",
    [
        ("±0.05", (-0.05, 0.05)),
        (0.05, (-0.05, 0.05)),
        ("0.05", (-0.05, 0.05)),
        ("+0.1/-0", (0.0, 0.1)),
        ("+0.1/−0.05", (-0.05, 0.1)),
    ],
)
def test_limits(tol, expected):
    assert _limits(tol) == pytest.approx(expected)


def test_forward_mapping(triplet):
    _, drawing, tolerancing = triplet
    added = tolerances_from_drawing(tolerancing, drawing)
    assert added == tolerancing.perturbations
    tols = _by_label(added)
    assert sorted(tols) == sorted(
        [
            "Radius S1",
            "Fringes S1",
            "Irregularity S1",
            "Rot. irregularity S1",
            "Tilt X S1",
            "Tilt Y S1",
            "Radius S2",
            "Irregularity S2",
            "Tilt X S2",
            "Tilt Y S2",
            "Decenter X S2",
            "Decenter Y S2",
            "Thickness S1",
            "Thickness S3",
        ]
    )
    expected_limits = {
        "Radius S1": (-0.05, 0.05),
        "Fringes S1": (-2.0, 2.0),
        "Irregularity S1": (-0.5, 0.5),
        "Rot. irregularity S1": (-0.25, 0.25),
        "Tilt X S1": (-2.0 / 60.0, 2.0 / 60.0),
        "Tilt Y S2": (-1.0 / 60.0, 1.0 / 60.0),
        "Radius S2": (-0.05, 0.1),
        "Decenter X S2": (-0.01, 0.01),
        "Thickness S1": (-0.05, 0.05),
        "Thickness S3": (-0.02, 0.1),
    }
    for label, limits in expected_limits.items():
        assert_allclose(tols[label].limits, limits)
    assert tols["Tilt X S1"].type.units == "degrees"
    assert tols["Fringes S1"].type.units == "fringes"


def test_form_arguments(triplet):
    _, drawing, tolerancing = triplet
    tols = _by_label(tolerances_from_drawing(tolerancing, drawing))
    fringes = tols["Fringes S1"].type
    assert isinstance(fringes, FringesTolerance)
    assert_allclose(fringes.test_wavelength, 0.6328)
    assert_allclose(fringes.semi_aperture, 10.0)
    b, c = tols["Irregularity S1"].type, tols["Rot. irregularity S1"].type
    assert isinstance(b, IrregularityTolerance)
    assert (b.kind, c.kind) == ("B", "C")
    assert_allclose(c.semi_aperture, 10.0)
    # no test wavelength or clear aperture on surface 2: ISO default, surface
    default = tols["Irregularity S2"].type
    assert_allclose(default.test_wavelength, 0.54607)
    assert default.semi_aperture != 10.0


def test_form_in_nanometres(set_test_backend):
    optic = CookeTriplet()
    drawing = DrawingSpec(optic)
    drawing.set_surface(1, irregularity=100.0, form_unit="nm", test_wavelength=500.0)
    tol = tolerances_from_drawing(Tolerancing(optic), drawing)[0]
    # 100 nm of sag at 500 nm is 100 / 250 fringes
    assert_allclose(tol.limits, (-0.4, 0.4))
    assert_allclose(tol.type.test_wavelength, 0.5)


def test_centering_rss(triplet):
    _, drawing, tolerancing = triplet
    tols = _by_label(tolerances_from_drawing(tolerancing, drawing, centering="rss"))
    v = 2.0 / 60.0 / math.sqrt(2.0)
    assert_allclose(tols["Tilt X S1"].limits, (-v, v))
    assert_allclose(tols["Tilt Y S1"].limits, (-v, v))
    with pytest.raises(ValueError, match="centering"):
        tolerances_from_drawing(tolerancing, drawing, centering="radial")


def test_thickness_adjust(triplet):
    _, drawing, tolerancing = triplet
    tols = _by_label(tolerances_from_drawing(tolerancing, drawing))
    assert isinstance(tols["Thickness S1"].type, ThicknessTolerance)
    assert tols["Thickness S1"].type.adjust == 2
    assert tols["Thickness S3"].type.adjust == 4
    other = Tolerancing(tolerancing.optic)
    tols = _by_label(tolerances_from_drawing(other, drawing, adjust_thickness=False))
    assert tols["Thickness S1"].type.adjust is None


def test_distribution_passed(triplet):
    _, drawing, tolerancing = triplet
    added = tolerances_from_drawing(tolerancing, drawing, distribution="uniform")
    assert all(type(t.distribution).__name__ == "Uniform" for t in added)


def test_method_delegates(triplet):
    _, drawing, tolerancing = triplet
    added = tolerancing.add_tolerances_from_drawing(drawing, centering="rss")
    other = tolerances_from_drawing(Tolerancing(tolerancing.optic), drawing)
    assert [t.label for t in added] == [t.label for t in other]


def test_flat_surface_radius_skipped(set_test_backend):
    optic = AsphericSinglet()
    drawing = DrawingSpec(optic)
    drawing.set_surface(2, r_tolerance=0.1, sag_tolerance=1.0)
    with pytest.warns(UserWarning, match="flat"):
        added = tolerances_from_drawing(Tolerancing(optic), drawing)
    assert [t.label for t in added] == ["Fringes S2"]
    with pytest.raises(ValueError, match="flat"):
        tolerances_from_drawing(Tolerancing(optic), drawing, strict=True)


def test_cemented_thickness_skipped(set_test_backend):
    optic = CementedAchromat()
    drawing = DrawingSpec(optic)
    drawing.set_element(0, ct_tolerance=0.05)
    with pytest.warns(UserWarning, match="cemented"):
        assert tolerances_from_drawing(Tolerancing(optic), drawing) == []
    with pytest.raises(ValueError, match="cemented"):
        tolerances_from_drawing(Tolerancing(optic), drawing, strict=True)


def test_unregistered_type(triplet, monkeypatch):
    _, drawing, tolerancing = triplet
    monkeypatch.delitem(tolerance_registry._types, "surface_tilt")
    with pytest.warns(UserWarning, match="surface_tilt"):
        added = tolerances_from_drawing(tolerancing, drawing)
    assert not any(t.label.startswith("Tilt") for t in added)
    assert "Radius S1" in _by_label(added)
    strict = Tolerancing(tolerancing.optic)
    with pytest.raises(ValueError, match="surface_tilt"):
        tolerances_from_drawing(strict, drawing, strict=True)
    # nothing is added on an error
    assert strict.perturbations == []


def test_mismatched_optic(triplet):
    _, drawing, _ = triplet
    with pytest.raises(ValueError, match="differ"):
        tolerances_from_drawing(Tolerancing(CementedAchromat()), drawing)


def test_round_trip(triplet):
    _, drawing, tolerancing = triplet
    tolerances_from_drawing(tolerancing, drawing)
    values = drawing_values_from_tolerancing(tolerancing)
    s1, s2 = values["surfaces"][1], values["surfaces"][2]
    assert set(s1) == set(SURFACE_1) - {"ca_diameter"}
    assert_allclose(s1["r_tolerance"], 0.05)
    for key in ("sag_tolerance", "irregularity", "rotationally_symmetric"):
        assert_allclose(s1[key], SURFACE_1[key])
    assert_allclose(s1["test_wavelength"], 632.8)
    assert_allclose(s1["centration"], 2.0)
    assert s2["r_tolerance"] == "+0.1/-0.05"
    assert_allclose(s2["irregularity"], 1.0)
    assert_allclose(s2["test_wavelength"], 546.07)
    assert_allclose(s2["centration"], 1.0)
    assert_allclose(s2["centration_decentration"], 0.01)
    assert values["elements"][0] == {"ct_tolerance": pytest.approx(0.05)}
    assert values["elements"][1] == {"ct_tolerance": "+0.1/-0.02"}

    # the reported values build an equivalent drawing
    rebuilt = DrawingSpec(tolerancing.optic)
    for s, fields in values["surfaces"].items():
        rebuilt.set_surface(s, **fields)
    for e, fields in values["elements"].items():
        rebuilt.set_element(e, **fields)
    again = drawing_values_from_tolerancing(
        _with(tolerances_from_drawing, rebuilt, tolerancing.optic)
    )
    assert again["elements"] == values["elements"]
    assert again["surfaces"].keys() == values["surfaces"].keys()
    for s, fields in values["surfaces"].items():
        for key, value in fields.items():
            if isinstance(value, str):
                assert again["surfaces"][s][key] == value
            else:
                assert_allclose(again["surfaces"][s][key], value)


def _with(function, drawing, optic):
    tolerancing = Tolerancing(optic)
    function(tolerancing, drawing)
    return tolerancing


def test_reverse_common_wavelength(set_test_backend):
    optic = CookeTriplet()
    tolerancing = Tolerancing(optic)
    tolerancing.add_tolerance("fringes", 2.0, surface_number=1, test_wavelength=0.5)
    tolerancing.add_tolerance(
        "irregularity", 1.0, surface_number=1, test_wavelength=1.0
    )
    tolerancing.add_tolerance(
        "surface_tilt", minus=-0.01, plus=0.02, surface_number=1, axis="x"
    )
    tolerancing.add_tolerance("conic", 0.1, surface_number=1)
    tolerancing.add_tolerance("thickness", 0.1, surface_number=2)
    values = drawing_values_from_tolerancing(tolerancing)
    s1 = values["surfaces"][1]
    assert_allclose(s1["test_wavelength"], 500.0)
    assert_allclose(s1["sag_tolerance"], 2.0)
    # 1 fringe at 1 um is 2 fringes at 0.5 um
    assert_allclose(s1["irregularity"], 2.0)
    assert_allclose(s1["centration"], 0.02 * 60.0)
    assert "conic" not in s1
    # an air-space thickness has no ISO 10110 element field
    assert values["elements"] == {}


def test_sensitivity_analysis_runs(triplet):
    optic, _, tolerancing = triplet
    # sensitivity analysis is limited to 6 perturbations
    drawing = DrawingSpec(optic)
    drawing.set_surface(3, r_tolerance=0.05, sag_tolerance=2.0, irregularity=0.5)
    drawing.set_surface(4, centration=2.0)
    drawing.set_element(1, ct_tolerance=0.05)
    nominal = tolerancing.evaluate()
    added = tolerances_from_drawing(tolerancing, drawing)
    assert len(added) == 6
    analysis = SensitivityAnalysis(tolerancing)
    analysis.run()
    results = analysis.get_results()
    assert len(results) == 2 * len(added)
    spots = results[analysis.operand_names[0]].to_numpy(dtype=float)
    assert all(math.isfinite(v) and v > 0 for v in spots)
    assert_allclose(tolerancing.evaluate(), nominal)
