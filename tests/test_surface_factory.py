from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.coatings import FresnelCoating, SimpleCoating
from optiland.interactions.diffractive_model import DiffractiveInteractionModel
from optiland.interactions.thin_lens_interaction_model import ThinLensInteractionModel
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.phase.radial import RadialPhaseProfile
from optiland.samples.objectives import TessarLens
from optiland.surfaces import SurfaceFactory
from optiland.surfaces.object_surface import ObjectSurface
from optiland.surfaces.standard_surface import Surface

from .utils import assert_allclose


class TestSurfaceFactory:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.lens = TessarLens()
        self.surfaces = self.lens.surfaces
        self.factory = SurfaceFactory(self.surfaces)

    def test_create_surface_standard(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="standard",
            comment="Standard",
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
        )
        assert isinstance(surface, Surface)
        assert surface.geometry.radius == 10
        assert surface.geometry.k == 0
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_even_asphere(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="even_asphere",
            comment="Even Asphere",
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
            coefficients=[1, 2, 3],
        )
        assert isinstance(surface, Surface)
        assert surface.geometry.radius == 10
        assert surface.geometry.k == 0
        assert surface.geometry.coefficients == [1, 2, 3]
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_odd_asphere(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="odd_asphere",
            comment="Odd Asphere",
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
            coefficients=[1, 2, 3],
        )
        assert isinstance(surface, Surface)
        assert surface.geometry.radius == 10
        assert surface.geometry.k == 0
        assert surface.geometry.coefficients == [1, 2, 3]
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_polynomial(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="polynomial",
            comment="Polynomial",
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
            coefficients=[1, 2, 3],
            tol=1e-6,
            max_iter=100,
        )
        assert isinstance(surface, Surface)
        assert surface.geometry.radius == 10
        assert surface.geometry.k == 0
        assert be.array_equal(surface.geometry.coefficients, be.array([[1, 2, 3]]))
        assert surface.geometry.tol == 1e-6
        assert surface.geometry.max_iter == 100
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_chebyshev(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="chebyshev",
            comment="Chebyshev",
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
            coefficients=be.arange(9).reshape(3, 3),
            tol=1e-6,
            max_iter=100,
            norm_x=1,
            norm_y=1,
        )
        assert isinstance(surface, Surface)
        assert surface.geometry.radius == 10
        assert surface.geometry.k == 0
        assert be.all(surface.geometry.coefficients == be.arange(9).reshape(3, 3))
        assert surface.geometry.tol == 1e-6
        assert surface.geometry.max_iter == 100
        assert surface.geometry.norm_x == 1
        assert surface.geometry.norm_y == 1
        assert isinstance(surface.material_pre, IdealMaterial)
        assert isinstance(surface.material_post, IdealMaterial)

    def test_create_surface_object(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="standard",
            comment="Object",
            index=0,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
        )
        assert isinstance(surface, ObjectSurface)
        assert surface.geometry.radius == 10
        assert surface.geometry.k == 0
        assert isinstance(surface.material_post, IdealMaterial)

    def test_invalid_surface_type(self, set_test_backend):
        with pytest.raises(ValueError):
            self.factory.create_surface(
                surface_type="invalid",
                comment="Invalid",
                index=1,
                is_stop=False,
                material="air",
                thickness=5,
            )

    def test_metalens_integration_focus(self, set_test_backend):
        """Test creating and tracing a metalens as an integration test."""
        if be.get_backend() == "torch":
            pytest.skip("This test requires functionality not yet in torch backend")

        optic = Optic()
        focal_length = 100.0
        wavelength = 0.55
        k0 = 2 * be.pi / wavelength

        # The Optic starts with an object surface. We need to set its material.
        optic.surfaces.add(index=0, radius=be.inf, thickness=be.inf)

        # Define the phase profile for a lens: phi = -k0/(2f) * r^2
        # k0 should be calculated in mm^-1 for the phase profile coefficient
        k0_mm = 2 * be.pi / (wavelength * 1e-3)
        lens_coeff = -k0_mm / (2 * focal_length)
        phase_profile = RadialPhaseProfile(coefficients=[lens_coeff])

        # Add the metalens surface
        optic.surfaces.add(
            index=1,
            surface_type="plane",
            interaction_type="phase",
            phase_profile=phase_profile,
            is_stop=True,
            material="air",
            thickness=focal_length,  # Propagate to the focal plane
        )
        optic.surfaces.add(index=2)

        # Configure optic for tracing
        optic.wavelengths.add(wavelength)
        optic.fields.set_type("angle")
        optic.fields.add(0)  # On-axis field
        optic.set_aperture("EPD", 10.0)

        # Trace rays
        rays = optic.trace(Hx=0, Hy=0, wavelength=wavelength, num_rays=5)

        # Verification: at the focal plane, all rays should be at the focus
        assert_allclose(rays.z, be.full_like(rays.z, focal_length))
        assert be.all(be.abs(rays.x) < 1e-2)
        assert be.all(be.abs(rays.y) < 1e-2)

    def test_invalid_surface_index(self, set_test_backend):
        with pytest.raises(IndexError):
            self.factory.create_surface(
                surface_type="standard",
                comment="Invalid",
                index=42,
                is_stop=False,
                material="air",
                thickness=5,
            )

    def test_create_surface_with_coating(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="standard",
            comment="Coating",
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
            coating=SimpleCoating(0.5, 0.5),
        )
        assert isinstance(surface, Surface)
        assert isinstance(surface.interaction_model.coating, SimpleCoating)

    def test_create_surface_with_fresnel(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="standard",
            comment="Fresnel",
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
            coating="fresnel",
        )
        assert isinstance(surface, Surface)
        assert isinstance(surface.interaction_model.coating, FresnelCoating)

    def test_invalid_z_with_thickness(self, set_test_backend):
        with pytest.raises(ValueError):
            self.factory.create_surface(
                surface_type="standard",
                comment="Invalid",
                index=1,
                is_stop=False,
                material="air",
                thickness=5,
                z=1,
            )

    def test_absolute_coordinates(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="standard",
            comment="Absolute",
            index=1,
            is_stop=False,
            material="air",
            radius=10,
            conic=0,
            x=1,
            y=2,
            z=3,
        )
        assert isinstance(surface, Surface)
        assert surface.geometry.cs.x == 1
        assert surface.geometry.cs.y == 2
        assert surface.geometry.cs.z == 3
        assert self.factory.use_absolute_cs

    def test_invalid_thickness_for_abs_cs(self, set_test_backend):
        self.factory.use_absolute_cs = True
        with pytest.raises(ValueError):
            self.factory.create_surface(
                surface_type="standard",
                comment="Invalid",
                index=1,
                is_stop=False,
                material="air",
                radius=10,
                conic=0,
                thickness=5,
            )

    def test_create_paraxial_standard(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="paraxial",
            comment="Paraxial",
            f=100,
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
        )
        assert isinstance(surface, Surface)
        assert isinstance(surface.interaction_model, ThinLensInteractionModel)
        assert surface.interaction_model.f == 100

    def test_create_grating_surface(self, set_test_backend):
        surface = self.factory.create_surface(
            surface_type="grating",
            comment="Grating",
            index=1,
            is_stop=False,
            material="air",
            thickness=5,
            radius=10,
            conic=0,
            grating_period=1,
            grating_order=1,
        )
        assert isinstance(surface, Surface)
        assert isinstance(surface.interaction_model, DiffractiveInteractionModel)

    def test_invalid_paraxial_surface(self, set_test_backend):
        with pytest.raises(ValueError):
            self.factory.create_surface(
                surface_type="paraxial",
                comment="Paraxial",
                f=100,
                index=0,
                is_stop=False,
                material="air",
                thickness=5,
            )
