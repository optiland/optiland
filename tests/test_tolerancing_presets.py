"""Tests for add_default_tolerances: tolerances from a user table."""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.materials.ideal import IdealMaterial
from optiland.optic import Optic
from optiland.samples.objectives import CookeTriplet
from optiland.tolerancing import SensitivityAnalysis, Tolerancing
from optiland.tolerancing.presets import SCOPES, _elements, _targets


def _labels(tolerances) -> list[str]:
    return [t.label for t in tolerances]


def _with_dummy_plane() -> Optic:
    """Singlet (surfaces 2-3) after a flat stop plane in air (surface 1)."""
    optic = Optic()
    optic.surfaces.add(index=0, thickness=be.inf)
    optic.surfaces.add(index=1, thickness=5.0, is_stop=True)
    optic.surfaces.add(index=2, radius=50.0, thickness=5.0, material=IdealMaterial(1.5))
    optic.surfaces.add(index=3, radius=-50.0, thickness=40.0)
    optic.surfaces.add(index=4)
    optic.set_aperture("EPD", 10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(0.55, is_primary=True)
    return optic


def test_cooke_triplet_placement(set_test_backend):
    tol = Tolerancing(CookeTriplet())
    added = tol.add_default_tolerances(
        {"fringes": 2, "thickness": 0.05, "surface_decenter": 0.01}
    )
    labels = _labels(added)
    assert labels[:6] == [f"Fringes S{k}" for k in range(1, 7)]
    # every gap except the one in front of the image
    assert labels[6:11] == [f"Thickness S{k}" for k in range(1, 6)]
    assert labels[11:13] == ["Decenter X S1", "Decenter Y S1"]
    assert len(labels) == 6 + 5 + 12
    assert tol.perturbations == added


def test_limits_from_dict(set_test_backend):
    tol = Tolerancing(CookeTriplet())
    added = tol.add_default_tolerances(
        {"thickness": {"minus": -0.02, "plus": 0.05, "distribution": "uniform"}},
        surfaces=[2],
    )
    assert len(added) == 1
    assert added[0].limits == (-0.02, 0.05)
    assert type(added[0].distribution).__name__ == "Uniform"


def test_surfaces_filter(set_test_backend):
    tol = Tolerancing(CookeTriplet())
    added = tol.add_default_tolerances({"radius": 0.1}, surfaces=[1, 5])
    assert _labels(added) == ["Radius S1", "Radius S5"]


def test_dummy_plane_is_skipped(set_test_backend):
    tol = Tolerancing(_with_dummy_plane())
    added = tol.add_default_tolerances(
        {"fringes": 1, "surface_tilt": 0.01, "thickness": 0.1}
    )
    labels = _labels(added)
    assert "Fringes S1" not in labels
    assert "Tilt X S1" not in labels
    assert labels[:2] == ["Fringes S2", "Fringes S3"]
    # the gap after the stop plane reaches a lens surface, so it is kept
    assert [label for label in labels if label.startswith("Thickness")] == [
        "Thickness S1",
        "Thickness S2",
    ]


def test_radius_skips_flat_surfaces(set_test_backend):
    optic = _with_dummy_plane()
    optic.surfaces[3].geometry.radius = be.inf
    tol = Tolerancing(optic)
    assert _labels(tol.add_default_tolerances({"radius": 0.1})) == ["Radius S2"]


def test_element_and_glass_targets(set_test_backend):
    tol = Tolerancing(CookeTriplet())
    assert _elements(tol, 0.55) == [(1, 2), (3, 4), (5, 6)]
    assert _targets(tol, "element", "element_tilt", 0.55)[0] == {
        "first": 1,
        "last": 2,
    }
    glass = _targets(tol, "glass", "index", 0.55)
    assert [t["surface_number"] for t in glass] == [1, 3, 5]


def test_unknown_type_and_bad_entries(set_test_backend):
    tol = Tolerancing(CookeTriplet())
    with pytest.raises(ValueError, match="No default placement"):
        tol.add_default_tolerances({"wedge": 0.1})
    with pytest.raises(ValueError, match="Unknown keys"):
        tol.add_default_tolerances({"radius": {"limt": 0.1}})
    with pytest.raises(TypeError, match="number or a dict"):
        tol.add_default_tolerances({"radius": "0.1"})


def test_every_builtin_type_has_a_scope():
    assert set(SCOPES.values()) == {"surface", "gap", "element", "glass"}


def test_defaults_run_in_sensitivity_analysis(set_test_backend):
    optic = CookeTriplet()
    tol = Tolerancing(optic)
    tol.add_operand(
        "rms_spot_size",
        {
            "optic": optic,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 0.0,
            "num_rays": 3,
            "wavelength": 0.55,
        },
    )
    tol.add_default_tolerances({"fringes": 2, "thickness": 0.05}, surfaces=[1, 2, 3])
    sa = SensitivityAnalysis(tol)
    sa.run()
    assert len(sa.get_results()) == 2 * 6
