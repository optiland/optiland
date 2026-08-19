from __future__ import annotations

from unittest.mock import patch

import pytest

import optiland.backend as be
from optiland.aperture import BaseSystemAperture
from optiland.apodization import GaussianApodization
from optiland.fields import FieldGroup, ParaxialImageHeightField
from optiland.optic import Optic
from optiland.rays import create_polarization
from optiland.samples.objectives import HeliarLens
from optiland.surfaces import SurfaceGroup
from optiland.surfaces.factories.material_factory import MaterialFactory
from optiland.wavelength import WavelengthGroup
from tests.utils import assert_allclose


def singlet_infinite_object():
    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        thickness=7,
        radius=43.7354,
        is_stop=True,
        material="N-SF11",
    )
    lens.surfaces.add(index=2, radius=-46.2795, thickness=50)
    lens.surfaces.add(index=3)

    lens.set_aperture(aperture_type="EPD", value=25)

    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)

    lens.wavelengths.add(value=0.5, is_primary=True)

    return lens


def singlet_finite_object():
    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=50)
    lens.surfaces.add(
        index=1,
        thickness=7,
        radius=43.7354,
        is_stop=True,
        material="N-SF11",
    )
    lens.surfaces.add(index=2, radius=-46.2795, thickness=50)
    lens.surfaces.add(index=3)

    lens.set_aperture(aperture_type="EPD", value=25)

    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0)

    lens.wavelengths.add(value=0.5, is_primary=True)

    return lens


class TestOptic:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Optic()

    def test_initialization(self, set_test_backend):
        assert self.optic.aperture is None
        assert self.optic.fields.field_definition is None
        assert isinstance(self.optic.surfaces, SurfaceGroup)
        assert isinstance(self.optic.fields, FieldGroup)
        assert isinstance(self.optic.wavelengths, WavelengthGroup)
        assert self.optic.polarization == "ignore"
        assert not self.optic.obj_space_telecentric

    def test_add_surface(self, set_test_backend):
        self.optic.surfaces.add(
            index=0,
            surface_type="standard",
            material="SF11",
            thickness=5,
        )
        assert len(self.optic.surfaces) == 1

    def test_add_field(self, set_test_backend):
        self.optic.fields.add(10.0, 5.0)
        assert len(self.optic.fields.fields) == 1
        assert self.optic.fields.fields[0].y == 10.0
        assert self.optic.fields.fields[0].x == 5.0

    def test_add_wavelength(self, set_test_backend):
        self.optic.wavelengths.add(0.55, is_primary=True)
        assert len(self.optic.wavelengths.wavelengths) == 1
        assert self.optic.wavelengths.wavelengths[0].value == 0.55
        assert self.optic.wavelengths.wavelengths[0].is_primary

    def test_set_aperture(self, set_test_backend):
        self.optic.set_aperture("EPD", 5.0)
        assert isinstance(self.optic.aperture, BaseSystemAperture)
        assert self.optic.aperture.ap_type == "EPD"
        assert self.optic.aperture.value == 5.0

    def test_set_field_type(self, set_test_backend):
        self.optic.fields.set_type("paraxial_image_height")
        assert isinstance(self.optic.fields.field_definition, ParaxialImageHeightField)

    def test_set_comment(self, set_test_backend):
        self.optic.surfaces.add(
            index=0,
            surface_type="standard",
            material="SF11",
            thickness=5,
            comment="Object surface",
        )
        self.optic.surfaces.add(
            index=1,
            surface_type="standard",
            material="SF11",
            thickness=5,
            comment="First surface",
        )

        assert self.optic.surfaces[0].comment == "Object surface"
        assert self.optic.surfaces[1].comment == "First surface"

    def test_set_radius(self, set_test_backend):
        self.optic.surfaces.add(
            index=0,
            surface_type="standard",
            material="SF11",
            thickness=5,
        )
        self.optic.updater.set_radius(10.0, 0)
        assert self.optic.surfaces[0].geometry.radius == 10.0

    def test_set_conic(self, set_test_backend):
        self.optic.surfaces.add(
            index=0,
            surface_type="standard",
            material="SF11",
            thickness=5,
        )
        self.optic.updater.set_conic(-1.0, 0)
        assert self.optic.surfaces[0].geometry.k == -1.0

    def test_set_thickness(self, set_test_backend):
        self.optic.surfaces.add(
            index=0,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.surfaces.add(
            index=1,
            surface_type="standard",
            material="SF11",
            thickness=5,
        )
        self.optic.surfaces.add(
            index=2,
            surface_type="standard",
            material="air",
            thickness=10,
        )
        self.optic.updater.set_thickness(10.0, 1)
        assert self.optic.surfaces.get_thickness(1) == 10.0

    def test_set_index(self, set_test_backend):
        self.optic.surfaces.add(
            index=0,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.surfaces.add(
            index=1,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.surfaces.add(
            index=2,
            surface_type="standard",
            material="air",
            thickness=10,
        )
        self.optic.updater.set_index(1.5, 1)
        assert self.optic.surfaces[1].material_post.n(1) == 1.5

    def test_set_material(self, set_test_backend):
        self.optic.surfaces.add(
            index=0,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.surfaces.add(
            index=1,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.surfaces.add(
            index=2,
            surface_type="standard",
            material="air",
            thickness=10,
        )
        surface_number = 1
        material_post = MaterialFactory._configure_post_material("N-BK7")
        self.optic.updater.set_material(material_post, surface_number)
        surface = self.optic.surfaces[surface_number]
        assert surface.material_post == material_post

    def test_set_asphere_coeff(self, set_test_backend):
        self.optic.surfaces.add(
            index=0,
            surface_type="even_asphere",
            material="air",
            thickness=5,
            coefficients=[0.0, 0.0, 0.0],
        )
        self.optic.updater.set_asphere_coeff(0.1, 0, 2)
        assert self.optic.surfaces[0].geometry.coefficients[2] == 0.1

    def test_set_polarization(self, set_test_backend):
        self.optic.updater.set_polarization("ignore")
        assert self.optic.polarization == "ignore"

    def test_optic_default_apodization(self, set_test_backend):
        assert self.optic.apodization is None

    def test_optic_set_apodization(self, set_test_backend):
        gaussian_apod = GaussianApodization(sigma=0.5)
        self.optic.updater.set_apodization(gaussian_apod)
        assert self.optic.apodization == gaussian_apod, "Apodization not set correctly"

        # Test setting with a non-Apodization type
        with pytest.raises(ValueError):
            self.optic.updater.set_apodization("not_an_apodization_object")

    def test_set_invalid_polarization(self, set_test_backend):
        with pytest.raises(ValueError):
            self.optic.updater.set_polarization("invalid")

    def test_set_pickup(self, set_test_backend):
        self.optic.surfaces.add(
            index=0,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.surfaces.add(
            index=1,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.pickups.add(0, "radius", 1, scale=2, offset=1)
        assert len(self.optic.pickups) == 1

    def test_clear_pickups(self, set_test_backend):
        self.optic.surfaces.add(
            index=0,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.surfaces.add(
            index=1,
            surface_type="standard",
            material="air",
            thickness=5,
        )
        self.optic.pickups.add(0, "radius", 1, scale=2, offset=1)
        self.optic.pickups.clear()
        assert len(self.optic.pickups) == 0

    def test_set_solve(self, set_test_backend):
        optic = HeliarLens()
        optic.solves.add("marginal_ray_height", 6, height=10)
        assert len(optic.solves) == 1

    def test_clear_solves(self, set_test_backend):
        optic = HeliarLens()
        optic.solves.add("marginal_ray_height", 6, height=10)
        optic.solves.clear()
        assert len(optic.solves) == 0

    def test_scale_system(self, set_test_backend):
        self.optic.surfaces.add(
            index=0,
            surface_type="standard",
            material="air",
            radius=10,
            thickness=5,
        )
        self.optic.set_aperture("EPD", 5.0)
        self.optic.updater.scale_system(2)
        assert self.optic.surfaces[0].geometry.radius == 2 * 10.0
        assert self.optic.aperture.value == 2 * 5.0

    def test_reset(self, set_test_backend):
        self.optic.surfaces.add(
            index=0,
            surface_type="standard",
            material="air",
            radius=10,
            thickness=5,
        )
        self.optic.reset()
        assert self.optic.aperture is None
        assert self.optic.fields.field_definition is None
        assert len(self.optic.surfaces) == 0
        assert len(self.optic.fields.fields) == 0
        assert len(self.optic.wavelengths.wavelengths) == 0
        assert len(self.optic.pickups) == 0
        assert len(self.optic.solves) == 0

    def test_n(self, set_test_backend):
        self.optic.surfaces.add(
            index=0,
            surface_type="standard",
            material="air",
            radius=10,
            thickness=5,
        )
        self.optic.wavelengths.add(0.55, is_primary=True)
        n_values = self.optic.surfaces.n(self.optic.primary_wavelength)
        assert len(n_values) == 1

    def test_update_paraxial(self, set_test_backend):
        lens = HeliarLens()
        lens.updater.update_paraxial()
        # This test is just to ensure the method runs without error

    def test_update(self, set_test_backend):
        lens = HeliarLens()
        lens.updater.update()
        # This test is just to ensure the method runs without error

    def test_image_solve(self, set_test_backend):
        lens = HeliarLens()
        lens.updater.image_solve()
        # This test is just to ensure the method runs without error

    def test_trace(self, set_test_backend):
        lens = HeliarLens()
        rays = lens.trace(0.0, 0.0, 0.55)
        assert rays is not None

    def test_trace_generic(self, set_test_backend):
        lens = HeliarLens()
        rays = lens.trace_generic(0.0, 0.0, 0.0, 0.0, 0.55)
        assert rays is not None

    def test_trace_invalid_field(self, set_test_backend):
        lens = HeliarLens()
        with pytest.raises(ValueError):
            lens.trace(0.0, 2.0, 0.55)

    def test_trace_generic_invalid_field(self, set_test_backend):
        lens = HeliarLens()
        with pytest.raises(ValueError):
            lens.trace_generic(0.0, 0.0, 0.0, 2.0, 0.55)

    def test_trace_generic_invalid_pupil(self, set_test_backend):
        lens = HeliarLens()
        with pytest.raises(ValueError):
            lens.trace_generic(0.0, 5.0, 0.0, 0.0, 0.55)

    def test_trace_polarized(self, set_test_backend):
        lens = HeliarLens()
        state = create_polarization("unpolarized")
        lens.updater.set_polarization(state)
        rays = lens.trace(0.0, 0.0, 0.55)
        assert rays is not None

    def test_object_property(self, set_test_backend):
        assert self.optic.object_surface is None

    def test_image_surface_property(self, set_test_backend):
        self.optic.surfaces.add(
            index=0,
            surface_type="standard",
            material="air",
            radius=10,
            thickness=5,
        )
        assert self.optic.image_surface is self.optic.surfaces[0]

    def test_total_track_property(self, set_test_backend):
        lens = HeliarLens()
        assert lens.total_track == 12.1357

    def test_total_track_error(self, set_test_backend):
        lens = HeliarLens()
        # manually remove all but first surface
        for k in range(len(lens.surfaces) - 1, 0, -1):
            lens.surfaces.remove(k)
        with pytest.raises(ValueError):
            _ = lens.total_track

    def test_polarization_state_property(self, set_test_backend):
        lens = HeliarLens()
        assert lens.polarization_state is None

        state = create_polarization("unpolarized")
        lens.updater.set_polarization(state)
        assert lens.polarization_state == state

    def test_polarization_state_error(self, set_test_backend):
        lens = HeliarLens()
        lens.polarization = "invalid"
        with pytest.raises(ValueError):
            _ = lens.polarization_state

    def test_to_dict(self, set_test_backend):
        lens = HeliarLens()
        lens.updater.set_apodization(GaussianApodization(sigma=0.5))
        lens_dict = lens.to_dict()
        assert isinstance(lens_dict, dict)
        assert "apodization" in lens_dict
        assert lens_dict["apodization"]["type"] == "GaussianApodization"
        assert lens_dict["apodization"]["sigma"] == 0.5

    def test_from_dict(self, set_test_backend):
        lens = HeliarLens()
        lens.updater.set_apodization(GaussianApodization(sigma=0.5))
        basic_dict = lens.to_dict()

        new_optic = Optic.from_dict(basic_dict)
        assert isinstance(new_optic, Optic)
        assert isinstance(new_optic.apodization, GaussianApodization)

        basic_dict.pop("apodization", None)
        new_optic = Optic.from_dict(basic_dict)
        assert new_optic.apodization is None

    def test_invalid_field_type(self, set_test_backend):
        with pytest.raises(ValueError):
            self.optic.fields.set_type("invalid")

    def test_no_stop(self, set_test_backend):
        for surface in self.optic.surfaces:
            surface.is_stop = False
        with pytest.raises(ValueError):
            _ = self.optic.surfaces.stop_index

    def test_add_infinite_object(self):  # do not test for torch backend
        lens1 = singlet_infinite_object()
        lens2 = singlet_infinite_object()
        lens_combined = lens1 + lens2
        assert lens_combined.surfaces.num_surfaces == 6

        # test that a ray trace through the combined lens works
        rays = lens_combined.trace(
            Hx=0, Hy=0, distribution="random", num_rays=42, wavelength=0.5
        )
        assert rays is not None

    def test_add_finite_object(self):  # do not test for torch backend
        lens1 = singlet_finite_object()
        lens2 = singlet_finite_object()
        lens_combined = lens1 + lens2
        assert lens_combined.surfaces.num_surfaces == 6

        # test that a ray trace through the combined lens works
        rays = lens_combined.trace(
            Hx=0, Hy=0, distribution="random", num_rays=42, wavelength=0.5
        )
        assert rays is not None

    def test_invalid_coordinate_system(self, set_test_backend):
        with pytest.raises(ValueError):
            self.optic.surfaces.add(index=0, radius=be.inf, z=-100)
            self.optic.surfaces.add(
                index=1, radius=be.inf, z=0, dx=15
            )  # cannot use dx or dy with abs. z

    def test_flip_optic(self, set_test_backend):
        lens = HeliarLens()
        lens.surfaces.set_fresnel_coatings()
        radii_orig = be.copy(lens.surfaces.radii)
        radii_orig = radii_orig[~be.isinf(radii_orig)]  # ignore inf
        n_orig = be.copy(lens.surfaces.n(0.55))
        lens.updater.flip()
        radii_flipped = be.copy(lens.surfaces.radii)
        radii_flipped = radii_flipped[~be.isinf(radii_flipped)]  # ignore inf
        n_flipped = be.copy(lens.surfaces.n(0.55))
        assert_allclose(radii_orig, -be.flip(radii_flipped))
        assert_allclose(n_orig[:-1], be.flip(n_flipped[:-1]))

    def test_invalid_flip(self, set_test_backend):
        lens = Optic()
        with pytest.raises(ValueError):
            lens.updater.flip()

    def test_flip_solves_pickups(self, set_test_backend):
        lens = Optic()

        lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        lens.surfaces.add(
            index=1, radius=100, thickness=4, material="SK16", is_stop=True
        )
        lens.surfaces.add(index=2, radius=-1000, thickness=20)
        lens.surfaces.add(index=3)
        lens.set_aperture(aperture_type="EPD", value=10.0)
        lens.fields.set_type(field_type="angle")
        lens.fields.add(y=0)
        lens.wavelengths.add(value=0.5876, is_primary=True)

        lens.solves.add("quick_focus")
        lens.pickups.add(
            source_surface_idx=1,
            attr_type="radius",
            target_surface_idx=2,
            scale=-1,
            offset=0,
        )
        lens.updater.update()
        lens.updater.flip()
        assert lens.pickups.pickups[0].source_surface_idx == 2
        assert lens.pickups.pickups[0].target_surface_idx == 1

    def test_remove_surface(self, set_test_backend):
        lens = singlet_infinite_object()

        num_surfaces_before = len(lens.surfaces)
        lens.surfaces.remove(index=2)
        assert len(lens.surfaces) == num_surfaces_before - 1

    @patch("optiland.visualization.SurfaceSagViewer")
    def test_plot_surface_sag(self, mock_viewer, set_test_backend):
        lens = singlet_infinite_object()

        lens.plot_surface_sag(
            surface_index=1, y_cross_section=2.0, x_cross_section=-2.0
        )
        mock_viewer.assert_called_once_with(lens)
        viewer_instance = mock_viewer.return_value
        viewer_instance.view.assert_called_once_with(
            surface_index=1,
            y_cross_section=2.0,
            x_cross_section=-2.0,
            fig_to_plot_on=None,
            max_extent=None,
            num_points_grid=50,
            buffer_factor=1.1,
        )


def test_flip_updates_thickness_attribute(set_test_backend):
    """
    Verify that Optic.flip() correctly updates the thickness attribute
    of each surface to maintain consistency with z-positions.

    From bug reported in issue #362
    """
    lens = Optic()
    lens.surfaces.add(index=0, thickness=be.inf, material="Air")
    lens.surfaces.add(index=1, material="N-BK7", thickness=7.0)
    lens.surfaces.add(index=2, material="SF5", thickness=2.5)
    lens.surfaces.add(index=3, material="Air", thickness=70.0)
    lens.surfaces.add(index=4, material="Air")

    lens.updater.flip()

    flipped_surfaces = lens.surfaces

    # Expected thicknesses after flip (reversed order)
    expected_thicknesses = [be.inf, 2.5, 7.0, 70.0, 0.0]

    # Assertions
    assert flipped_surfaces[0].thickness == expected_thicknesses[0]
    assert flipped_surfaces[1].thickness == expected_thicknesses[1]
    assert flipped_surfaces[2].thickness == expected_thicknesses[2]
    assert flipped_surfaces[3].thickness == expected_thicknesses[3]
    assert flipped_surfaces[4].thickness == expected_thicknesses[4]

    # Also verify relationship between thickness and z-position
    thicknesses = be.diff(
        be.array([surf.geometry.cs.z for surf in flipped_surfaces]),
        append=be.array([79.5]),
    )
    assert_allclose(thicknesses, expected_thicknesses)
