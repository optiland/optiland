"""Structural unit tests for sequential_to_nonsequential converter.

Kramer Harrison, 2026
"""

from __future__ import annotations

import pytest

from optiland.nonsequential.convert import (
    ConversionError,
    ConversionReport,
    sequential_to_nonsequential,
)

# ---------------------------------------------------------------------------
# Helpers — build sequential optics
# ---------------------------------------------------------------------------


def _singlet_optic():
    """Build a simple BK7 singlet with one angle field."""
    from optiland.optic import Optic

    optic = Optic()
    optic.add_surface(index=0, thickness=float("inf"))
    optic.add_surface(
        index=1, radius=50.0, thickness=5.0, material="N-BK7", is_stop=True
    )
    optic.add_surface(index=2, radius=-50.0, thickness=50.0)
    optic.add_surface(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.set_field_type(field_type="angle")
    optic.add_field(y=0.0)
    optic.add_wavelength(value=0.55, is_primary=True)
    return optic


def _doublet_optic():
    """Build a simple cemented doublet."""
    from optiland.optic import Optic

    optic = Optic()
    optic.add_surface(index=0, thickness=float("inf"))
    optic.add_surface(
        index=1, radius=60.0, thickness=6.0, material="N-BK7", is_stop=True
    )
    optic.add_surface(index=2, radius=-30.0, thickness=2.0, material="N-F2")
    optic.add_surface(index=3, radius=-80.0, thickness=50.0)
    optic.add_surface(index=4)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.set_field_type(field_type="angle")
    optic.add_field(y=0.0)
    optic.add_wavelength(value=0.55, is_primary=True)
    return optic


def _multi_field_optic():
    """Build a singlet with three angle fields."""
    from optiland.optic import Optic

    optic = Optic()
    optic.add_surface(index=0, thickness=float("inf"))
    optic.add_surface(
        index=1, radius=50.0, thickness=5.0, material="N-BK7", is_stop=True
    )
    optic.add_surface(index=2, radius=-50.0, thickness=50.0)
    optic.add_surface(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.set_field_type(field_type="angle")
    optic.add_field(y=0.0)
    optic.add_field(y=5.0)
    optic.add_field(y=10.0)
    optic.add_wavelength(value=0.55, is_primary=True)
    return optic


# ---------------------------------------------------------------------------
# Converter import / type tests
# ---------------------------------------------------------------------------


def test_singlet_scene_structure():
    """Converting a singlet should yield: 1 compound, 1 source, 1 detector."""
    from optiland.nonsequential.components.lens import Lens
    from optiland.nonsequential.detectors.irradiance import IrradianceDetector
    from optiland.nonsequential.scene import NSQScene

    optic = _singlet_optic()
    with pytest.warns(UserWarning, match="Fresnel"):
        scene = sequential_to_nonsequential(optic)

    assert isinstance(scene, NSQScene)
    compounds = scene.component_registry.compounds
    assert len(compounds) == 1
    assert isinstance(compounds[0], Lens)
    assert len(scene.sources) == 1
    assert len(scene.detectors) == 1
    assert isinstance(scene.detectors[0], IrradianceDetector)


def test_doublet_scene_structure():
    """Converting a doublet should yield a Doublet compound component."""
    from optiland.nonsequential.components.doublet import Doublet

    optic = _doublet_optic()
    with pytest.warns(UserWarning, match="Fresnel"):
        scene = sequential_to_nonsequential(optic)

    compounds = scene.component_registry.compounds
    assert len(compounds) == 1
    assert isinstance(compounds[0], Doublet)


def test_multi_field_sources():
    """Three angle fields should produce three CollimatedSource objects."""
    from optiland.nonsequential.sources.collimated import CollimatedSource

    optic = _multi_field_optic()
    with pytest.warns(UserWarning, match="Fresnel"):
        scene = sequential_to_nonsequential(optic)

    assert len(scene.sources) == 3
    for src in scene.sources:
        assert isinstance(src, CollimatedSource)


def test_angle_field_creates_collimated_source():
    """An angle-field optic must produce CollimatedSource objects."""
    from optiland.nonsequential.sources.collimated import CollimatedSource

    optic = _singlet_optic()
    with pytest.warns(UserWarning, match="Fresnel"):
        scene = sequential_to_nonsequential(optic)

    assert isinstance(scene.sources[0], CollimatedSource)


def test_height_field_creates_point_source():
    """An object-height-field optic must produce PointSource objects."""
    from optiland.nonsequential.sources.point import PointSource
    from optiland.optic import Optic

    optic = Optic()
    optic.add_surface(index=0, thickness=100.0)
    optic.add_surface(
        index=1, radius=50.0, thickness=5.0, material="N-BK7", is_stop=True
    )
    optic.add_surface(index=2, radius=-50.0, thickness=50.0)
    optic.add_surface(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.set_field_type(field_type="object_height")
    optic.add_field(y=0.0)
    optic.add_wavelength(value=0.55, is_primary=True)

    with pytest.warns(UserWarning, match="Fresnel"):
        scene = sequential_to_nonsequential(optic)

    assert isinstance(scene.sources[0], PointSource)


def test_beam_diameter_override():
    """Passing beam_diameter=5.0 must set aperture_radius = 2.5 on CollimatedSource."""
    optic = _singlet_optic()
    with pytest.warns(UserWarning, match="Fresnel"):
        scene = sequential_to_nonsequential(optic, beam_diameter=5.0)

    src = scene.sources[0]
    assert abs(src.aperture_radius - 2.5) < 1e-9


def test_detector_at_image_plane():
    """Detector CS z-coordinate must match the image surface z-position."""
    from optiland.nonsequential.convert import _surface_z

    optic = _singlet_optic()
    with pytest.warns(UserWarning, match="Fresnel"):
        scene = sequential_to_nonsequential(optic)

    img_surf = optic.image_surface
    expected_z = _surface_z(img_surf)

    det = scene.detectors[0]
    det_z = float(det.cs.z)
    assert abs(det_z - expected_z) < 1e-6


def test_material_name_preserved():
    """LensConfig material must equal 'N-BK7' after converting an N-BK7 singlet."""
    optic = _singlet_optic()
    with pytest.warns(UserWarning, match="Fresnel"):
        scene = sequential_to_nonsequential(optic)

    compound = scene.component_registry.compounds[0]
    # LensConfig is stored as _config on the compound
    assert compound._config.material == "N-BK7"


def test_unsupported_geometry_raises():
    """A surface with an unsupported geometry must raise ConversionError."""
    from optiland.optic import Optic

    optic = Optic()
    optic.add_surface(index=0, thickness=float("inf"))
    optic.add_surface(
        index=1,
        radius=50.0,
        thickness=5.0,
        material="N-BK7",
        is_stop=True,
        surface_type="even_asphere",
    )
    optic.add_surface(index=2, radius=-50.0, thickness=50.0)
    optic.add_surface(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.set_field_type(field_type="angle")
    optic.add_field(y=0.0)
    optic.add_wavelength(value=0.55, is_primary=True)

    # EvenAsphere is in the unsupported set — it may or may not be supported
    # depending on implementation; test that *if* unsupported it raises.
    # We test with a known-unsupported type by patching geometry name.
    from unittest.mock import MagicMock

    from optiland.nonsequential.convert import _check_geometry

    surf = MagicMock()
    surf.geometry = MagicMock()
    type(surf.geometry).__name__ = "CoordinateBreak"

    with pytest.raises(ConversionError, match="CoordinateBreak"):
        _check_geometry(surf, index=1)


def test_fresnel_warning_issued():
    """sequential_to_nonsequential must emit a UserWarning about Fresnel."""
    optic = _singlet_optic()
    with pytest.warns(UserWarning, match="Fresnel"):
        sequential_to_nonsequential(optic)


def test_has_polarization_no_deprecation_warning():
    """_has_polarization_surfaces must not trigger a DeprecationWarning.

    Previously the function used the deprecated ``surf.coating`` property.
    It must now use ``surf.interaction_model.coating`` exclusively.
    """
    import warnings

    from optiland.nonsequential.convert import _has_polarization_surfaces

    optic = _singlet_optic()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # Must not raise DeprecationWarning
        result = _has_polarization_surfaces(optic)
    assert result is False  # plain singlet has no polarization coatings


# ---------------------------------------------------------------------------
# ConversionReport
# ---------------------------------------------------------------------------


def test_conversion_report_attached_to_scene():
    optic = _singlet_optic()
    with pytest.warns(UserWarning, match="Fresnel"):
        scene = sequential_to_nonsequential(optic)

    assert isinstance(scene.conversion_report, ConversionReport)


def test_uncoated_surfaces_reported_for_plain_optic():
    """A plain (uncoated) singlet's surfaces must show up as uncoated."""
    optic = _singlet_optic()
    with pytest.warns(UserWarning, match="Fresnel"):
        scene = sequential_to_nonsequential(optic)

    report = scene.conversion_report
    assert "L1.front" in report.uncoated_surfaces
    assert "L1.back" in report.uncoated_surfaces
    assert report.coated_surfaces == []


def test_coated_surface_carried_over():
    """An AR-coated sequential surface's coating must reach SurfaceConfig."""
    from optiland.coatings import SimpleCoating
    from optiland.optic import Optic

    optic = Optic()
    optic.add_surface(index=0, thickness=float("inf"))
    optic.add_surface(
        index=1,
        radius=50.0,
        thickness=5.0,
        material="N-BK7",
        is_stop=True,
        coating=SimpleCoating(reflectance=0.005, transmittance=0.995),
    )
    optic.add_surface(index=2, radius=-50.0, thickness=50.0)
    optic.add_surface(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.set_field_type(field_type="angle")
    optic.add_field(y=0.0)
    optic.add_wavelength(value=0.55, is_primary=True)

    with pytest.warns(UserWarning, match="Fresnel"):
        scene = sequential_to_nonsequential(optic)

    report = scene.conversion_report
    assert report.coated_surfaces == ["L1.front"]
    assert "L1.back" in report.uncoated_surfaces

    compound = scene.component_registry.compounds[0]
    front_cfg = compound._config.front
    assert front_cfg is not None
    assert front_cfg.coating.reflectance == pytest.approx(0.005)
    assert compound._config.back is None


def test_estimated_apertures_reported_when_not_set_explicitly():
    """Neither aperture nor semi_aperture is set on this optic's surfaces,
    so both lens faces must be flagged as estimated.
    """
    optic = _singlet_optic()
    with pytest.warns(UserWarning, match="Fresnel"):
        scene = sequential_to_nonsequential(optic)

    report = scene.conversion_report
    assert "L1.front" in report.estimated_apertures
    assert "L1.back" in report.estimated_apertures


def test_mirror_reflectance_defaulted_reported():
    from optiland.optic import Optic

    optic = Optic()
    optic.add_surface(index=0, thickness=float("inf"))
    optic.add_surface(
        index=1,
        radius=-100.0,
        thickness=-50.0,
        is_stop=True,
        material="mirror",
    )
    optic.add_surface(index=2)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.set_field_type(field_type="angle")
    optic.add_field(y=0.0)
    optic.add_wavelength(value=0.55, is_primary=True)

    with pytest.warns(UserWarning, match="perfect reflector"):
        scene = sequential_to_nonsequential(optic)

    report = scene.conversion_report
    assert "M1" in report.mirror_reflectance_defaulted


def test_report_summary_lists_everything():
    optic = _singlet_optic()
    with pytest.warns(UserWarning, match="Fresnel"):
        scene = sequential_to_nonsequential(optic)

    summary = scene.conversion_report.summary()
    assert "L1.front" in summary
    assert "L1.back" in summary


def test_report_summary_clean_when_nothing_dropped():
    report = ConversionReport()
    assert "fully faithful" in report.summary()


def test_fresnel_warning_mentions_uncoated_count():
    optic = _singlet_optic()
    with pytest.warns(UserWarning, match=r"2 refractive surface\(s\)"):
        sequential_to_nonsequential(optic)


def test_image_height_field_raises():
    """Fields of type 'paraxial_image_height' must raise ConversionError."""
    from optiland.optic import Optic

    optic = Optic()
    optic.add_surface(index=0, thickness=float("inf"))
    optic.add_surface(
        index=1, radius=50.0, thickness=5.0, material="N-BK7", is_stop=True
    )
    optic.add_surface(index=2, radius=-50.0, thickness=50.0)
    optic.add_surface(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.set_field_type(field_type="paraxial_image_height")
    optic.add_field(y=0.0)
    optic.add_wavelength(value=0.55, is_primary=True)

    with pytest.raises(ConversionError, match="paraxial_image_height"):
        sequential_to_nonsequential(optic)
