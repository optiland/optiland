"""OSLO boundary cases preserve destinations and reject undefined ray launches."""

from __future__ import annotations

import math

import pytest

import optiland.backend as be
from optiland.aperture import EPDAperture
from optiland.fileio import load_oslo_file, save_oslo_file
from optiland.fileio.oslo.reader.converter import OsloToOpticConverter
from optiland.fileio.oslo.reader.parser import OsloDataParser
from optiland.geometries import EvenAsphere, OddAsphere, StandardGeometry
from optiland.interactions import RefractiveReflectiveModel, ThinLensInteractionModel
from optiland.physical_apertures import RadialAperture, UnclippedAperture
from optiland.rays import RealRays
from tests.utils import assert_allclose


def test_export_without_system_aperture_omits_aperture_command(lens_file, tmp_path):
    optic = load_oslo_file(lens_file(), strict=True)
    optic.aperture = None
    output = tmp_path / "no-aperture.len"
    save_oslo_file(optic, output)
    assert OsloDataParser(output).parse().aperture == {}


def test_export_rejects_custom_system_aperture_before_overwriting(lens_file, tmp_path):
    class CustomAperture(EPDAperture):
        @property
        def ap_type(self):
            return "custom"

    optic = load_oslo_file(lens_file(), strict=True)
    optic.aperture = CustomAperture(4)
    output = tmp_path / "custom-aperture.len"
    output.write_text("saved design", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="system aperture"):
        save_oslo_file(optic, output)
    assert output.read_text() == "saved design"


@pytest.mark.parametrize("diameter", [0, -1, math.inf, math.nan])
def test_export_rejects_invalid_entrance_beam_without_overwriting(
    lens_file, tmp_path, set_test_backend, diameter
):
    optic = load_oslo_file(lens_file(), strict=True)
    optic.set_aperture("EPD", diameter)
    output = tmp_path / "invalid-beam.len"
    output.write_text("saved design", encoding="utf-8")
    with pytest.raises(ValueError, match="finite positive entrance beam"):
        save_oslo_file(optic, output)
    assert output.read_text() == "saved design"


@pytest.mark.parametrize(
    "aperture_type,distance",
    [("EPD", "1e20"), ("float_by_stop_size", "1e20"), ("imageFNO", "100")],
)
def test_export_preserves_aperture_calculation_error_and_destination(
    lens_file, tmp_path, set_test_backend, aperture_type, distance
):
    optic = load_oslo_file(lens_file(distance=distance), strict=True)
    optic.set_aperture(aperture_type, 4)
    while optic.wavelengths:
        optic.wavelengths.remove(0)
    output = tmp_path / "missing-spectrum.len"
    output.write_text("saved design", encoding="utf-8")
    with pytest.raises(ValueError, match="No primary wavelength"):
        save_oslo_file(optic, output)
    assert output.read_text() == "saved design"


def test_model_conversion_rejects_empty_modeled_glass(lens_file):
    # OsloToOpticConverter also accepts models constructed outside the parser.
    model = OsloDataParser(lens_file()).parse()
    model.surfaces[1]["material"] = "GLA MOD"
    with pytest.raises(ValueError, match="GLA MOD requires refractive-index data"):
        OsloToOpticConverter(model, strict=True).convert()


def test_strict_conversion_rejects_diagnostics_from_permissive_parser(lens_file):
    with pytest.warns(UserWarning, match="GRIN"):
        model = OsloDataParser(lens_file(surface="GRIN 1")).parse()
    diagnostic = model.diagnostics[0]
    with pytest.raises(ValueError, match=rf"GRIN.*line {diagnostic.line}.*surface 1"):
        OsloToOpticConverter(model, strict=True).convert()
    assert OsloToOpticConverter(model).convert().surfaces.num_surfaces == 4


def test_finite_object_on_first_surface_cannot_define_nonzero_beam(
    lens_file, set_test_backend
):
    # An axial point at surface 1 has zero height there, even with a later stop.
    # No finite cone from this point can have the requested EBR=2 at surface 1.
    with pytest.raises(ValueError, match="finite nonzero beam at surface 1"):
        load_oslo_file(
            lens_file(distance="0", surface="AIR\nRD 0", second="AST"), strict=True
        )


def test_export_cannot_discard_custom_ray_interactions(
    lens_file, tmp_path, set_test_backend
):
    class AttenuatingInteraction(RefractiveReflectiveModel):
        def interact_real_rays(self, rays):
            rays = super().interact_real_rays(rays)
            rays.i *= 0.25
            return rays

    optic = load_oslo_file(lens_file(), strict=True)
    surface = optic.surfaces[1]
    surface.interaction_model = AttenuatingInteraction(surface, is_reflective=False)
    zeros = be.zeros(1)
    rays = RealRays(zeros, zeros, zeros, zeros, zeros, be.ones(1), 1, 0.55)
    surface.interaction_model.interact_real_rays(rays)
    # This model inherits the standard interaction_type string, but changes
    # transmitted power. Exporting only its GLA record would lose that physics.
    assert_allclose(rays.i, [0.25])
    output = tmp_path / "custom-interaction.len"
    output.write_text("saved design", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="interaction model"):
        save_oslo_file(optic, output)
    assert output.read_text() == "saved design"


@pytest.mark.parametrize("radius", [0, -1, math.inf, math.nan])
@pytest.mark.parametrize("checked", [True, False])
def test_export_rejects_unrepresentable_surface_radius_without_overwriting(
    lens_file, tmp_path, radius, checked
):
    optic = load_oslo_file(lens_file(), strict=True)
    aperture = RadialAperture(radius)
    optic.surfaces[1].aperture = aperture if checked else UnclippedAperture(aperture)
    output = tmp_path / "invalid-surface-radius.len"
    output.write_text("saved design", encoding="utf-8")
    with pytest.raises(ValueError, match="finite positive surface aperture radius"):
        save_oslo_file(optic, output)
    assert output.read_text() == "saved design"


def test_export_cannot_turn_a_zero_radius_clip_into_an_open_aperture(
    lens_file, tmp_path, set_test_backend
):
    optic = load_oslo_file(lens_file(), strict=True)
    optic.surfaces[1].aperture = RadialAperture(0)
    zeros = be.zeros(2)
    rays = RealRays(zeros, [0, 1], zeros, zeros, zeros, be.ones(2), [1, 1], 0.55)
    optic.surfaces[1].aperture.clip(rays)
    assert_allclose(rays.i, [1, 0])
    # AP CHK 0 has no explicit radius in the importer. A native zero-radius
    # aperture really blocks off-axis rays, so the writer must reject the loss.
    with pytest.raises(ValueError, match="finite positive surface aperture radius"):
        save_oslo_file(optic, tmp_path / "zero-radius.len")


@pytest.mark.parametrize(
    "geometry_type,surface_type,sag_excess",
    [(EvenAsphere, "standard", 0.04), (OddAsphere, "even_asphere", 0.02)],
)
def test_export_checks_geometry_instead_of_trusting_the_surface_label(
    lens_file, tmp_path, set_test_backend, geometry_type, surface_type, sag_excess
):
    optic = load_oslo_file(lens_file(), strict=True)
    surface = optic.surfaces[1]
    base = StandardGeometry(surface.geometry.cs, 20)
    surface.geometry = geometry_type(surface.geometry.cs, 20, coefficients=[0.01])
    surface.surface_type = surface_type
    x, y = be.zeros(1), be.array([2.0])
    assert_allclose(surface.geometry.sag(x, y) - base.sag(x, y), [sag_excess])
    # A mutable surface label cannot establish the sag equation. The standard
    # handler would drop this term; the even handler would change its power.
    output = tmp_path / "mismatched-geometry.len"
    output.write_text("saved design", encoding="utf-8")
    with pytest.raises(NotImplementedError, match="geometry"):
        save_oslo_file(optic, output)
    assert output.read_text() == "saved design"


@pytest.mark.parametrize("surface_type", ["standard", "paraxial"])
def test_thin_lens_export_does_not_silently_claim_perfect_imaging(
    lens_file, tmp_path, set_test_backend, surface_type
):
    optic = load_oslo_file(lens_file(), strict=True)
    surface = optic.surfaces[1]
    surface.interaction_model = ThinLensInteractionModel(
        surface, is_reflective=False, focal_length=20
    )
    surface.surface_type = surface_type
    output = tmp_path / "thin-lens.len"
    output.write_text("saved design", encoding="utf-8")
    # OSLO PFL is an exact perfect-imaging model, not the same off-axis
    # interaction as the native thin-lens phase. Import already diagnoses this
    # approximation; export must not silently describe it as equivalent either.
    with pytest.raises(NotImplementedError, match="thin-lens.*perfect-imaging"):
        save_oslo_file(optic, output)
    assert output.read_text() == "saved design"
