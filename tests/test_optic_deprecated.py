"""Tests for deprecated methods on the Optic class.

Confirms that every deprecated API introduced in the legacy->hierarchical
refactor is still accessible, emits a DeprecationWarning, and produces
exactly the effect described in the original method contract.

NOTE: This file will be removed with the v0.7.0 release when the deprecated
methods are deleted from ``Optic``.
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.apodization import GaussianApodization
from optiland.fields import AngleField
from optiland.optic import Optic
from optiland.samples.objectives import HeliarLens
from optiland.surfaces import SurfaceGroup
from optiland.surfaces.factories.material_factory import MaterialFactory
from tests.utils import assert_allclose

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _simple_optic() -> Optic:
    """Minimal, fully-configured singlet lens."""
    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        radius=50.0,
        thickness=5.0,
        is_stop=True,
        material="N-BK7",
    )
    lens.surfaces.add(index=2, radius=-50.0, thickness=45.0)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=10)
    lens.fields.set_type("angle")
    lens.fields.add(y=0)
    lens.wavelengths.add(value=0.55, is_primary=True)
    return lens


def _three_surface_optic() -> Optic:
    """Bare three-surface optic (no aperture/field/wavelength)."""
    optic = Optic()
    optic.surfaces.add(index=0, material="air", thickness=10, radius=be.inf)
    optic.surfaces.add(index=1, material="air", thickness=5, radius=50.0)
    optic.surfaces.add(index=2, material="air", thickness=10, radius=be.inf)
    return optic


def _zernike_optic() -> Optic:
    """Singlet with a Zernike surface that exposes norm_radius."""
    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        surface_type="zernike",
        radius=50.0,
        thickness=5.0,
        is_stop=True,
        material="N-BK7",
        norm_radius=10.0,
        coefficients=[0.0] * 10,
    )
    lens.surfaces.add(index=2, radius=-50.0, thickness=45.0)
    lens.surfaces.add(index=3)
    lens.set_aperture("EPD", 10)
    lens.fields.set_type("angle")
    lens.fields.add(y=0)
    lens.wavelengths.add(0.55, is_primary=True)
    return lens


# ===========================================================================
# surface_group property
# ===========================================================================


class TestDeprecatedSurfaceGroupGetter:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning, match="surface_group"):
            _ = optic.surface_group

    def test_returns_surfaces(self, set_test_backend):
        optic = _simple_optic()
        with pytest.warns(DeprecationWarning):
            sg = optic.surface_group
        assert sg is optic.surfaces

    def test_return_type_is_surface_group(self, set_test_backend):
        optic = _simple_optic()
        with pytest.warns(DeprecationWarning):
            sg = optic.surface_group
        assert isinstance(sg, SurfaceGroup)


class TestDeprecatedSurfaceGroupSetter:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = Optic()
        new_sg = SurfaceGroup()
        with pytest.warns(DeprecationWarning, match="surface_group"):
            optic.surface_group = new_sg

    def test_sets_surfaces(self, set_test_backend):
        optic = Optic()
        new_sg = SurfaceGroup()
        with pytest.warns(DeprecationWarning):
            optic.surface_group = new_sg
        assert optic.surfaces is new_sg


# ===========================================================================
# add_surface
# ===========================================================================


class TestDeprecatedAddSurface:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning, match="add_surface"):
            optic.add_surface(
                index=0, surface_type="standard", material="air", thickness=5
            )

    def test_surface_is_added(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning):
            optic.add_surface(
                index=0, surface_type="standard", material="air", thickness=5
            )
        assert len(optic.surfaces) == 1

    def test_radius_is_preserved(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning):
            optic.add_surface(
                index=0,
                material="N-BK7",
                thickness=5,
                radius=50.0,
                is_stop=True,
            )
        assert_allclose(optic.surfaces[0].geometry.radius, 50.0)

    def test_matches_new_api(self, set_test_backend):
        optic_old = Optic()
        with pytest.warns(DeprecationWarning):
            optic_old.add_surface(index=0, material="N-BK7", thickness=5, radius=50.0)

        optic_new = Optic()
        optic_new.surfaces.add(index=0, material="N-BK7", thickness=5, radius=50.0)

        assert len(optic_old.surfaces) == len(optic_new.surfaces)
        assert (
            optic_old.surfaces[0].geometry.radius
            == optic_new.surfaces[0].geometry.radius
        )


# ===========================================================================
# remove_surface
# ===========================================================================


class TestDeprecatedRemoveSurface:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = _three_surface_optic()
        with pytest.warns(DeprecationWarning, match="remove_surface"):
            optic.remove_surface(1)

    def test_surface_is_removed(self, set_test_backend):
        optic = _three_surface_optic()
        n_before = len(optic.surfaces)
        with pytest.warns(DeprecationWarning):
            optic.remove_surface(1)
        assert len(optic.surfaces) == n_before - 1

    def test_correct_surface_is_removed(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, material="air", thickness=5, radius=100.0)
        optic.surfaces.add(index=1, material="air", thickness=5, radius=200.0)
        optic.surfaces.add(index=2, material="air", thickness=5, radius=300.0)
        with pytest.warns(DeprecationWarning):
            optic.remove_surface(1)
        # Surface at index 1 (radius=200) should be gone; old index 2 is now 1
        assert_allclose(optic.surfaces[1].geometry.radius, 300.0)


# ===========================================================================
# add_field
# ===========================================================================


class TestDeprecatedAddField:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning, match="add_field"):
            optic.add_field(y=10.0)

    def test_field_is_added(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning):
            optic.add_field(y=10.0)
        assert len(optic.fields.fields) == 1

    def test_field_y_value(self, set_test_backend):
        """Intended: add_field(y=10) must produce a field with y=10."""
        optic = Optic()
        with pytest.warns(DeprecationWarning):
            optic.add_field(y=10.0)
        assert optic.fields.fields[0].y == pytest.approx(10.0)

    def test_field_x_y_values(self, set_test_backend):
        """Intended: add_field(y=10, x=5) must produce a field with y=10, x=5."""
        optic = Optic()
        with pytest.warns(DeprecationWarning):
            optic.add_field(y=10.0, x=5.0)
        assert optic.fields.fields[0].y == pytest.approx(10.0)
        assert optic.fields.fields[0].x == pytest.approx(5.0)

    def test_vignetting_factors(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning):
            optic.add_field(y=0.0, x=0.0, vx=0.1, vy=0.2)
        field = optic.fields.fields[0]
        assert field.vx == pytest.approx(0.1)
        assert field.vy == pytest.approx(0.2)


# ===========================================================================
# set_field_type
# ===========================================================================


class TestDeprecatedSetFieldType:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning, match="set_field_type"):
            optic.set_field_type("angle")

    def test_field_type_is_angle(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning):
            optic.set_field_type("angle")
        assert isinstance(optic.fields.field_definition, AngleField)

    def test_matches_new_api(self, set_test_backend):
        optic_old = Optic()
        with pytest.warns(DeprecationWarning):
            optic_old.set_field_type("angle")

        optic_new = Optic()
        optic_new.fields.set_type("angle")

        assert type(optic_old.fields.field_definition) is type(
            optic_new.fields.field_definition
        )


# ===========================================================================
# add_wavelength
# ===========================================================================


class TestDeprecatedAddWavelength:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning, match="add_wavelength"):
            optic.add_wavelength(0.55, is_primary=True)

    def test_wavelength_is_added(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning):
            optic.add_wavelength(0.55, is_primary=True)
        assert len(optic.wavelengths.wavelengths) == 1

    def test_wavelength_value(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning):
            optic.add_wavelength(0.55, is_primary=True)
        assert optic.wavelengths.wavelengths[0].value == pytest.approx(0.55)

    def test_is_primary(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning):
            optic.add_wavelength(0.55, is_primary=True)
        assert optic.wavelengths.wavelengths[0].is_primary

    def test_unit_nm_conversion(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning):
            optic.add_wavelength(550.0, is_primary=True, unit="nm")
        assert optic.wavelengths.wavelengths[0].value == pytest.approx(0.55, abs=1e-6)

    def test_weight(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning):
            optic.add_wavelength(0.55, is_primary=True, weight=2.0)
        assert optic.wavelengths.wavelengths[0].weight == pytest.approx(2.0)


# ===========================================================================
# set_radius
# ===========================================================================


class TestDeprecatedSetRadius:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = _three_surface_optic()
        with pytest.warns(DeprecationWarning, match="set_radius"):
            optic.set_radius(100.0, 1)

    def test_radius_is_updated(self, set_test_backend):
        optic = _three_surface_optic()
        with pytest.warns(DeprecationWarning):
            optic.set_radius(100.0, 1)
        assert_allclose(optic.surfaces[1].geometry.radius, 100.0)

    def test_matches_new_api(self, set_test_backend):
        optic_old = _three_surface_optic()
        optic_new = _three_surface_optic()
        with pytest.warns(DeprecationWarning):
            optic_old.set_radius(100.0, 1)
        optic_new.updater.set_radius(100.0, 1)
        assert_allclose(optic_old.surfaces[1].geometry.radius, optic_new.surfaces[1].geometry.radius)


# ===========================================================================
# set_conic
# ===========================================================================


class TestDeprecatedSetConic:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = _three_surface_optic()
        with pytest.warns(DeprecationWarning, match="set_conic"):
            optic.set_conic(-1.0, 1)

    def test_conic_is_updated(self, set_test_backend):
        optic = _three_surface_optic()
        with pytest.warns(DeprecationWarning):
            optic.set_conic(-1.0, 1)
        assert optic.surfaces[1].geometry.k == pytest.approx(-1.0)

    def test_matches_new_api(self, set_test_backend):
        optic_old = _three_surface_optic()
        optic_new = _three_surface_optic()
        with pytest.warns(DeprecationWarning):
            optic_old.set_conic(-0.5, 1)
        optic_new.updater.set_conic(-0.5, 1)
        assert optic_old.surfaces[1].geometry.k == pytest.approx(
            optic_new.surfaces[1].geometry.k
        )


# ===========================================================================
# set_thickness
# ===========================================================================


class TestDeprecatedSetThickness:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = _three_surface_optic()
        with pytest.warns(DeprecationWarning, match="set_thickness"):
            optic.set_thickness(20.0, 1)

    def test_thickness_is_updated(self, set_test_backend):
        optic = _three_surface_optic()
        with pytest.warns(DeprecationWarning):
            optic.set_thickness(20.0, 1)
        assert_allclose(optic.surfaces.get_thickness(1), 20.0)

    def test_matches_new_api(self, set_test_backend):
        optic_old = _three_surface_optic()
        optic_new = _three_surface_optic()
        with pytest.warns(DeprecationWarning):
            optic_old.set_thickness(20.0, 1)
        optic_new.updater.set_thickness(20.0, 1)
        assert_allclose(
            optic_old.surfaces.get_thickness(1),
            optic_new.surfaces.get_thickness(1),
        )


# ===========================================================================
# set_index
# ===========================================================================


class TestDeprecatedSetIndex:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = _three_surface_optic()
        with pytest.warns(DeprecationWarning, match="set_index"):
            optic.set_index(1.5, 1)

    def test_index_is_updated(self, set_test_backend):
        optic = _three_surface_optic()
        with pytest.warns(DeprecationWarning):
            optic.set_index(1.5, 1)
        assert_allclose(optic.surfaces[1].material_post.n(1), 1.5)

    def test_matches_new_api(self, set_test_backend):
        optic_old = _three_surface_optic()
        optic_new = _three_surface_optic()
        with pytest.warns(DeprecationWarning):
            optic_old.set_index(1.5, 1)
        optic_new.updater.set_index(1.5, 1)
        assert_allclose(
            optic_old.surfaces[1].material_post.n(1),
            optic_new.surfaces[1].material_post.n(1),
        )


# ===========================================================================
# set_material
# ===========================================================================


class TestDeprecatedSetMaterial:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = _three_surface_optic()
        material = MaterialFactory._configure_post_material("N-BK7")
        with pytest.warns(DeprecationWarning, match="set_material"):
            optic.set_material(material, 1)

    def test_material_is_updated(self, set_test_backend):
        optic = _three_surface_optic()
        material = MaterialFactory._configure_post_material("N-BK7")
        with pytest.warns(DeprecationWarning):
            optic.set_material(material, 1)
        assert optic.surfaces[1].material_post == material

    def test_matches_new_api(self, set_test_backend):
        optic_old = _three_surface_optic()
        optic_new = _three_surface_optic()
        material = MaterialFactory._configure_post_material("N-BK7")
        with pytest.warns(DeprecationWarning):
            optic_old.set_material(material, 1)
        optic_new.updater.set_material(material, 1)
        assert (
            optic_old.surfaces[1].material_post == optic_new.surfaces[1].material_post
        )


# ===========================================================================
# set_norm_radius
# ===========================================================================


class TestDeprecatedSetNormRadius:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = _zernike_optic()
        with pytest.warns(DeprecationWarning, match="set_norm_radius"):
            optic.set_norm_radius(15.0, 1)

    def test_norm_radius_is_updated(self, set_test_backend):
        optic = _zernike_optic()
        with pytest.warns(DeprecationWarning):
            optic.set_norm_radius(15.0, 1)
        assert optic.surfaces[1].geometry.norm_radius == pytest.approx(15.0)

    def test_is_fixed_sets_manual_mode(self, set_test_backend):
        optic = _zernike_optic()
        with pytest.warns(DeprecationWarning):
            optic.set_norm_radius(15.0, 1, is_fixed=True)
        assert optic.surfaces[1].geometry.normalization_mode == "manual"

    def test_not_fixed_sets_auto_mode(self, set_test_backend):
        optic = _zernike_optic()
        with pytest.warns(DeprecationWarning):
            optic.set_norm_radius(15.0, 1, is_fixed=False)
        assert optic.surfaces[1].geometry.normalization_mode == "auto"

    def test_raises_attribute_error_on_standard_surface(self, set_test_backend):
        optic = _three_surface_optic()
        with pytest.warns(DeprecationWarning), pytest.raises(AttributeError):
            optic.set_norm_radius(10.0, 1)

    def test_matches_new_api(self, set_test_backend):
        optic_old = _zernike_optic()
        optic_new = _zernike_optic()
        with pytest.warns(DeprecationWarning):
            optic_old.set_norm_radius(15.0, 1)
        optic_new.updater.set_norm_radius(15.0, 1)
        assert optic_old.surfaces[1].geometry.norm_radius == pytest.approx(
            optic_new.surfaces[1].geometry.norm_radius
        )


# ===========================================================================
# set_asphere_coeff
# ===========================================================================


class TestDeprecatedSetAsphereCoeff:
    @pytest.fixture(autouse=True)
    def setup(self, set_test_backend):
        self.optic = Optic()
        self.optic.surfaces.add(
            index=0,
            surface_type="even_asphere",
            material="air",
            thickness=5,
            coefficients=[0.0, 0.0, 0.0],
        )

    def test_emits_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match="set_asphere_coeff"):
            self.optic.set_asphere_coeff(0.1, 0, 2)

    def test_coefficient_is_updated(self):
        with pytest.warns(DeprecationWarning):
            self.optic.set_asphere_coeff(0.1, 0, 2)
        assert self.optic.surfaces[0].geometry.coefficients[2] == pytest.approx(0.1)

    def test_matches_new_api(self):
        optic_new = Optic()
        optic_new.surfaces.add(
            index=0,
            surface_type="even_asphere",
            material="air",
            thickness=5,
            coefficients=[0.0, 0.0, 0.0],
        )
        with pytest.warns(DeprecationWarning):
            self.optic.set_asphere_coeff(0.1, 0, 0)
        optic_new.updater.set_asphere_coeff(0.1, 0, 0)
        assert self.optic.surfaces[0].geometry.coefficients[0] == pytest.approx(
            optic_new.surfaces[0].geometry.coefficients[0]
        )


# ===========================================================================
# set_polarization
# ===========================================================================


class TestDeprecatedSetPolarization:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning, match="set_polarization"):
            optic.set_polarization("ignore")

    def test_polarization_set_to_ignore(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning):
            optic.set_polarization("ignore")
        assert optic.polarization == "ignore"

    def test_invalid_string_raises_value_error(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning), pytest.raises(ValueError):
            optic.set_polarization("invalid")

    def test_matches_new_api(self, set_test_backend):
        optic_old = Optic()
        optic_new = Optic()
        with pytest.warns(DeprecationWarning):
            optic_old.set_polarization("ignore")
        optic_new.updater.set_polarization("ignore")
        assert optic_old.polarization == optic_new.polarization


# ===========================================================================
# set_apodization
# ===========================================================================


class TestDeprecatedSetApodization:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = Optic()
        apod = GaussianApodization(sigma=0.5)
        with pytest.warns(DeprecationWarning, match="set_apodization"):
            optic.set_apodization(apod)

    def test_apodization_object_is_set(self, set_test_backend):
        optic = Optic()
        apod = GaussianApodization(sigma=0.5)
        with pytest.warns(DeprecationWarning):
            optic.set_apodization(apod)
        assert optic.apodization == apod

    def test_none_clears_apodization(self, set_test_backend):
        optic = Optic()
        optic.updater.set_apodization(GaussianApodization(sigma=0.5))
        with pytest.warns(DeprecationWarning):
            optic.set_apodization(None)
        assert optic.apodization is None

    def test_invalid_string_raises_value_error(self, set_test_backend):
        optic = Optic()
        with pytest.warns(DeprecationWarning), pytest.raises(ValueError):
            optic.set_apodization("not_an_apodization_object")

    def test_matches_new_api(self, set_test_backend):
        optic_old = Optic()
        optic_new = Optic()
        apod = GaussianApodization(sigma=0.5)
        with pytest.warns(DeprecationWarning):
            optic_old.set_apodization(apod)
        optic_new.updater.set_apodization(apod)
        assert optic_old.apodization == optic_new.apodization


# ===========================================================================
# scale_system
# ===========================================================================


class TestDeprecatedScaleSystem:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = _three_surface_optic()
        optic.set_aperture("EPD", 5.0)
        with pytest.warns(DeprecationWarning, match="scale_system"):
            optic.scale_system(2.0)

    def test_radius_is_scaled(self, set_test_backend):
        optic = _three_surface_optic()
        optic.set_aperture("EPD", 5.0)
        original_radius = float(be.to_numpy(optic.surfaces[1].geometry.radius))
        with pytest.warns(DeprecationWarning):
            optic.scale_system(2.0)
        assert_allclose(optic.surfaces[1].geometry.radius, 2.0 * original_radius)

    def test_aperture_is_scaled(self, set_test_backend):
        optic = _three_surface_optic()
        optic.set_aperture("EPD", 5.0)
        with pytest.warns(DeprecationWarning):
            optic.scale_system(3.0)
        assert optic.aperture.value == pytest.approx(15.0)

    def test_matches_new_api(self, set_test_backend):
        optic_old = _three_surface_optic()
        optic_new = _three_surface_optic()
        optic_old.set_aperture("EPD", 5.0)
        optic_new.set_aperture("EPD", 5.0)
        with pytest.warns(DeprecationWarning):
            optic_old.scale_system(2.0)
        optic_new.updater.scale_system(2.0)
        assert optic_old.aperture.value == pytest.approx(optic_new.aperture.value)


# ===========================================================================
# update_paraxial
# ===========================================================================


class TestDeprecatedUpdateParaxial:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = _simple_optic()
        with pytest.warns(DeprecationWarning, match="update_paraxial"):
            optic.update_paraxial()

    def test_does_not_raise(self, set_test_backend):
        optic = _simple_optic()
        with pytest.warns(DeprecationWarning):
            optic.update_paraxial()

    def test_same_semi_apertures_as_new_api(self, set_test_backend):
        optic_old = _simple_optic()
        optic_new = _simple_optic()
        with pytest.warns(DeprecationWarning):
            optic_old.update_paraxial()
        optic_new.updater.update_paraxial()
        for s_old, s_new in zip(optic_old.surfaces, optic_new.surfaces, strict=False):
            assert_allclose(s_old.semi_aperture, s_new.semi_aperture)


# ===========================================================================
# update_normalization
# ===========================================================================


class TestDeprecatedUpdateNormalization:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = _simple_optic()
        surface = optic.surfaces[1]
        with pytest.warns(DeprecationWarning, match="update_normalization"):
            optic.update_normalization(surface)

    def test_does_not_raise_on_standard_surface(self, set_test_backend):
        optic = _simple_optic()
        surface = optic.surfaces[1]
        with pytest.warns(DeprecationWarning):
            optic.update_normalization(surface)

    def test_same_result_as_new_api(self, set_test_backend):
        optic_old = _zernike_optic()
        optic_new = _zernike_optic()
        surface_old = optic_old.surfaces[1]
        surface_new = optic_new.surfaces[1]
        with pytest.warns(DeprecationWarning):
            optic_old.update_normalization(surface_old)
        optic_new.updater.update_normalization(surface_new)
        assert_allclose(
            optic_old.surfaces[1].geometry.norm_radius,
            optic_new.surfaces[1].geometry.norm_radius,
        )


# ===========================================================================
# update
# ===========================================================================


class TestDeprecatedUpdate:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = _simple_optic()
        with pytest.warns(DeprecationWarning, match=r"Optic\.update"):
            optic.update()

    def test_does_not_raise(self, set_test_backend):
        optic = _simple_optic()
        with pytest.warns(DeprecationWarning):
            optic.update()

    def test_pickups_are_applied(self, set_test_backend):
        """update() must apply pickups — same effect as optic.updater.update()."""
        optic_old = _three_surface_optic()
        optic_new = _three_surface_optic()
        optic_old.pickups.add(0, "radius", 1, scale=1.0, offset=0.0)
        optic_new.pickups.add(0, "radius", 1, scale=1.0, offset=0.0)
        with pytest.warns(DeprecationWarning):
            optic_old.update()
        optic_new.updater.update()
        assert_allclose(optic_old.surfaces[1].geometry.radius, optic_new.surfaces[1].geometry.radius)


# ===========================================================================
# image_solve
# ===========================================================================


class TestDeprecatedImageSolve:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = _simple_optic()
        with pytest.warns(DeprecationWarning, match="image_solve"):
            optic.image_solve()

    def test_does_not_raise(self, set_test_backend):
        optic = _simple_optic()
        with pytest.warns(DeprecationWarning):
            optic.image_solve()

    def test_same_image_position_as_new_api(self, set_test_backend):
        optic_old = _simple_optic()
        optic_new = _simple_optic()
        with pytest.warns(DeprecationWarning):
            optic_old.image_solve()
        optic_new.updater.image_solve()
        assert_allclose(
            optic_old.surfaces[-1].geometry.cs.z,
            optic_new.surfaces[-1].geometry.cs.z,
        )


# ===========================================================================
# flip
# ===========================================================================


class TestDeprecatedFlip:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = HeliarLens()
        with pytest.warns(DeprecationWarning, match="flip"):
            optic.flip()

    def test_surface_count_preserved(self, set_test_backend):
        optic = HeliarLens()
        n_before = len(optic.surfaces)
        with pytest.warns(DeprecationWarning):
            optic.flip()
        assert len(optic.surfaces) == n_before

    def test_same_result_as_new_api(self, set_test_backend):
        optic_old = HeliarLens()
        optic_new = HeliarLens()
        with pytest.warns(DeprecationWarning):
            optic_old.flip()
        optic_new.updater.flip()
        for s_old, s_new in zip(optic_old.surfaces, optic_new.surfaces, strict=False):
            assert_allclose(s_old.geometry.cs.z, s_new.geometry.cs.z)


# ===========================================================================
# set_ray_aiming
# ===========================================================================


class TestDeprecatedSetRayAiming:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = _simple_optic()
        with pytest.warns(DeprecationWarning, match="set_ray_aiming"):
            optic.set_ray_aiming("paraxial")

    def test_mode_is_set(self, set_test_backend):
        optic = _simple_optic()
        with pytest.warns(DeprecationWarning):
            optic.set_ray_aiming("iterative", max_iter=20, tol=1e-8)
        cfg = optic.ray_tracer.ray_aiming_config
        assert cfg["mode"] == "iterative"
        assert cfg["max_iter"] == 20
        assert cfg["tol"] == pytest.approx(1e-8)

    def test_matches_new_api(self, set_test_backend):
        optic_old = _simple_optic()
        optic_new = _simple_optic()
        with pytest.warns(DeprecationWarning):
            optic_old.set_ray_aiming("iterative")
        optic_new.ray_tracer.set_aiming("iterative")
        assert (
            optic_old.ray_tracer.ray_aiming_config
            == optic_new.ray_tracer.ray_aiming_config
        )


# ===========================================================================
# n() — refractive index
# ===========================================================================


class TestDeprecatedN:
    def test_emits_deprecation_warning(self, set_test_backend):
        optic = _simple_optic()
        with pytest.warns(DeprecationWarning, match=r"Optic\.n"):
            optic.n(0.55)

    def test_primary_keyword_resolves(self, set_test_backend):
        optic = _simple_optic()
        with pytest.warns(DeprecationWarning):
            result = optic.n("primary")
        expected = optic.surfaces.n(optic.primary_wavelength)
        assert_allclose(result, expected)

    def test_float_wavelength(self, set_test_backend):
        optic = _simple_optic()
        with pytest.warns(DeprecationWarning):
            result = optic.n(0.55)
        expected = optic.surfaces.n(0.55)
        assert_allclose(result, expected)

    def test_matches_new_api_heliar(self, set_test_backend):
        optic = HeliarLens()
        with pytest.warns(DeprecationWarning):
            old_result = optic.n(0.55)
        new_result = optic.surfaces.n(0.55)
        assert_allclose(old_result, new_result)

    def test_length_matches_surface_count(self, set_test_backend):
        optic = _simple_optic()
        with pytest.warns(DeprecationWarning):
            result = optic.n(0.55)
        # one refractive index per inter-surface space
        assert len(result) == len(optic.surfaces)
