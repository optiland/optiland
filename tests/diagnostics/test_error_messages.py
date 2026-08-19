"""Regression tests for the actionable-error-message pass (WS1.3).

Each test asserts that a failure a newcomer can plausibly hit names the
offending value *and* what to do about it. Assertions are deliberately
substring-based so that wording can be improved without breaking tests.
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland._suggest import did_you_mean, options_hint
from optiland.aperture import make_system_aperture
from optiland.distribution import create_distribution
from optiland.fields.field_types.base import BaseFieldDefinition
from optiland.materials import Material
from optiland.materials.base import BaseMaterial
from optiland.optic import Optic
from optiland.optimization.variable.variable import Variable
from optiland.surfaces.factories.geometry_factory import GeometryFactory
from optiland.wavelength import WavelengthGroup


class TestDidYouMean:
    def test_suggests_close_match(self):
        assert did_you_mean("NSF11", ["N-SF11", "N-SF10", "N-BK7"]) == (
            " Did you mean: N-SF11, N-SF10?"
        )

    def test_silent_when_nothing_is_close(self):
        assert did_you_mean("zzzzzz", ["N-SF11", "N-BK7"]) == ""

    def test_case_insensitive_fallback(self):
        assert "EPD" in did_you_mean("epd", ["EPD", "imageFNO"])

    def test_non_string_value_is_ignored(self):
        assert did_you_mean(None, ["EPD"]) == ""

    def test_respects_n(self):
        out = did_you_mean("N-SF1", ["N-SF11", "N-SF10", "N-SF2", "N-SF5"], n=2)
        assert out.count(",") == 1

    def test_options_hint_lists_candidates(self):
        out = options_hint("epd", ["EPD", "imageFNO"])
        assert "Valid options are: EPD, imageFNO." in out
        assert "Did you mean: EPD?" in out

    def test_options_hint_omits_long_lists(self):
        candidates = [f"opt{i}" for i in range(50)]
        out = options_hint("opt3", candidates)
        assert "Valid options are" not in out

    def test_options_hint_with_no_candidates(self):
        assert options_hint("anything", []) == ""


class TestStringKeyedLookups:
    """Every string-keyed lookup suggests what the user probably meant."""

    def test_material_name(self, set_test_backend):
        with pytest.raises(ValueError, match="N-SF11"):
            Material("NSF11")

    def test_material_catalog(self, set_test_backend):
        with pytest.raises(ValueError, match="list_catalogs"):
            Material("N-BK7", catalog="schoot")

    def test_material_type(self, set_test_backend):
        with pytest.raises(ValueError, match="Did you mean: IdealMaterial"):
            BaseMaterial.from_dict({"type": "IdealMateril"})

    def test_aperture_type(self, set_test_backend):
        with pytest.raises(ValueError, match="Did you mean: EPD"):
            make_system_aperture("epd", 10.0)

    def test_field_type(self, set_test_backend):
        with pytest.raises(ValueError, match="Did you mean: angle"):
            BaseFieldDefinition.create("angel")

    def test_distribution_type(self, set_test_backend):
        with pytest.raises(ValueError, match="Did you mean: hexapolar"):
            create_distribution("hexpolar")

    def test_backend_name(self, set_test_backend):
        current = be.get_backend()
        try:
            with pytest.raises(ValueError, match="Did you mean: torch"):
                be.set_backend("torhc")
        finally:
            be.set_backend(current)

    def test_surface_type(self, set_test_backend):
        with pytest.raises(ValueError, match="Did you mean: standard"):
            GeometryFactory.create("standrd", cs=None)

    def test_variable_type(self, set_test_backend):
        optic = Optic()
        with pytest.raises(ValueError, match="Did you mean: radius"):
            Variable(optic, type_name="radiu", surface_number=1)

    def test_operand_type(self, set_test_backend):
        from optiland.optimization.operand.operand import Operand
        from optiland.samples.objectives import CookeTriplet

        operand = Operand(operand_type="f2", input_data={"optic": CookeTriplet()})
        operand.operand_type = "f22"
        with pytest.raises(ValueError, match="Did you mean: f2"):
            _ = operand.value

    def test_wavelength_unit(self, set_test_backend):
        with pytest.raises(ValueError, match="Did you mean: nm"):
            WavelengthGroup().add(500, unit="nmm")

    def test_apodization_type(self, set_test_backend):
        optic = Optic()
        with pytest.raises(ValueError, match="Unknown apodization type"):
            optic.set_apodization("gausian")


class TestActionableFixes:
    """Errors from ordinary user input state what to do next."""

    def test_no_stop_surface(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=be.inf)
        optic.surfaces.add(index=1, radius=20, thickness=5)
        optic.surfaces.add(index=2)
        with pytest.raises(ValueError, match="is_stop=True"):
            _ = optic.surfaces.stop_index

    def test_no_aperture(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=be.inf)
        optic.surfaces.add(index=1, radius=20, thickness=5, is_stop=True)
        optic.surfaces.add(index=2)
        optic.fields.set_type("angle")
        optic.fields.add(y=0)
        optic.wavelengths.add(0.55)
        with pytest.raises(ValueError, match="set_aperture"):
            optic.paraxial.EPD()

    def test_no_primary_wavelength(self, set_test_backend):
        group = WavelengthGroup()
        with pytest.raises(ValueError, match="is_primary=True"):
            _ = group.primary_index

    def test_primary_index_out_of_range(self, set_test_backend):
        group = WavelengthGroup()
        group.add(0.55)
        with pytest.raises(ValueError, match="has 1 wavelength"):
            group.primary_index = 4

    def test_surface_added_out_of_order(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=be.inf)
        with pytest.raises(IndexError, match="must be added in order"):
            optic.surfaces.add(index=3, radius=20, thickness=5)

    def test_surface_without_index(self, set_test_backend):
        optic = Optic()
        with pytest.raises(ValueError, match=r"lens.add_surface\(index="):
            optic.surfaces.add(radius=20, thickness=5)

    def test_duplicate_object_surface(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=be.inf)
        with pytest.raises(ValueError, match="index 0 is the object surface"):
            optic.surfaces.add(index=0, thickness=be.inf)

    def test_stop_index_out_of_range(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=be.inf)
        optic.surfaces.add(index=1, radius=20, thickness=5)
        optic.surfaces.add(index=2)
        with pytest.raises(ValueError, match="object surface"):
            optic.surfaces.stop_index = 0

    def test_total_track_too_few_surfaces(self, set_test_backend):
        optic = Optic()
        with pytest.raises(ValueError, match="at least 2 are"):
            _ = optic.surfaces.total_track

    def test_flip_too_few_surfaces(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=be.inf)
        with pytest.raises(ValueError, match="at least 3"):
            optic.flip()

    def test_invalid_polarization_state(self, set_test_backend):
        optic = Optic()
        with pytest.raises(ValueError, match="PolarizationState instance"):
            optic.set_polarization("circular")

    def test_normalized_coordinates_out_of_range(self, set_test_backend):
        from optiland.samples.objectives import CookeTriplet

        lens = CookeTriplet()
        with pytest.raises(ValueError, match="normalized"):
            lens.trace(Hx=0, Hy=3, wavelength=0.55, num_rays=4)

    def test_odd_wavelength_count(self, set_test_backend):
        from optiland.wavelength import add_wavelengths

        with pytest.raises(ValueError, match="odd positive integer"):
            add_wavelengths(WavelengthGroup(), 0.4, 0.7, num_wavelengths=4)
