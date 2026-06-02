from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.coatings import SimpleCoating
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import Plane
from optiland.interactions import ThinLensInteractionModel
from optiland.materials import IdealMaterial
from optiland.optic import Optic
from optiland.rays import ParaxialRays, RealRays
from optiland.surfaces import Surface

from .utils import assert_allclose


@pytest.fixture
def surface(set_test_backend):
    cs = CoordinateSystem()
    focal_length = 42
    geometry = Plane(cs)
    material_pre = IdealMaterial(1, 0)
    material_post = IdealMaterial(1.5, 0)
    aperture = None
    coating = SimpleCoating(0.5, 0.5)
    bsdf = None
    interaction_model = ThinLensInteractionModel(
        parent_surface=None,
        focal_length=focal_length,
        is_reflective=True,
        coating=coating,
        bsdf=bsdf,
    )
    surf = Surface(
        previous_surface=None,
        geometry=geometry,
        material_post=material_post,
        is_stop=True,
        aperture=aperture,
        interaction_model=interaction_model,
        surface_type="paraxial",
    )
    interaction_model.parent_surface = surf
    return surf


class TestThinLensInteractionModel:
    def test_init(self, surface):
        assert surface.interaction_model.f == be.array(42)
        assert surface.geometry is not None
        assert surface.material_pre is not None
        assert surface.material_post is not None
        assert surface.aperture is None
        assert surface.interaction_model.coating is not None
        assert surface.interaction_model.bsdf is None
        assert surface.interaction_model.is_reflective is True
        assert surface.surface_type == "paraxial"

    def test_trace_paraxial_rays(self, surface):
        y = be.array([1])
        u = be.array([0])
        z = be.array([-10])
        w = be.array([1])
        rays = ParaxialRays(y, u, z, w)
        traced_rays = surface.trace(rays)
        assert isinstance(traced_rays, ParaxialRays)

    def test_trace_real_rays(self, surface):
        x = be.random_uniform(size=10)
        rays = RealRays(x, x, x, x, x, x, x, x)
        traced_rays = surface.trace(rays)
        assert isinstance(traced_rays, RealRays)

    @pytest.mark.parametrize(
        "focal_length, n1, n2",
        [
            (50.0, 1.0, 1.0),
            (50.0, 1.5, 2.0),
            (50.0, 2.0, 1.5),
            (-50.0, 1.0, 1.0),
            (-50.0, 1.5, 2.0),
            (-50.0, 2.0, 1.5),
        ],
    )
    def test_paraxial_surface_perfect_imaging_refraction(
        self, focal_length, n1, n2, set_test_backend
    ):
        lens = Optic()

        # add surfaces
        lens.surfaces.add(index=0, thickness=be.inf, material=IdealMaterial(n1))
        lens.surfaces.add(
            index=1,
            surface_type="paraxial",
            material=IdealMaterial(n2),
            thickness=focal_length * n2,
            f=focal_length,
            is_stop=True,
        )
        lens.surfaces.add(index=2)

        # add aperture
        lens.set_aperture(aperture_type="EPD", value=20)

        # add field
        lens.fields.set_type(field_type="angle")
        lens.fields.add(y=0)

        # add wavelength
        lens.wavelengths.add(value=0.55, is_primary=True)

        rays = lens.trace(
            Hx=0,
            Hy=0,
            wavelength=0.55,
            distribution="uniform",
            num_rays=32,
        )

        # confirm all points exactly on axis
        assert_allclose(rays.y, 0)

        # confirm all points at exact same z
        assert_allclose(rays.z, focal_length * n2)

    @pytest.mark.parametrize(
        "focal_length, n1",
        [
            (50.0, 1.0),
            (50.0, 1.5),
            (50.0, 2.0),
            (-50.0, 1.0),
            (-50.0, 1.5),
            (-50.0, 2.0),
        ],
    )
    def test_paraxial_surface_perfect_imaging_reflection(
        self, focal_length, n1, set_test_backend
    ):
        lens = Optic()

        # add surfaces
        lens.surfaces.add(index=0, thickness=be.inf, material=IdealMaterial(n1))
        lens.surfaces.add(
            index=1,
            surface_type="paraxial",
            material="mirror",
            thickness=focal_length,
            f=focal_length,
            is_stop=True,
        )
        lens.surfaces.add(index=2)

        # add aperture
        lens.set_aperture(aperture_type="EPD", value=20)

        # add field
        lens.fields.set_type(field_type="angle")
        lens.fields.add(y=0)

        # add wavelength
        lens.wavelengths.add(value=0.55, is_primary=True)

        rays = lens.trace(
            Hx=0,
            Hy=0,
            wavelength=0.55,
            distribution="uniform",
            num_rays=32,
        )

        # confirm all points exactly on axis
        assert_allclose(rays.y, 0)

        # confirm all points at exact same z
        assert_allclose(rays.z, focal_length)

    @pytest.mark.parametrize("n1, n2", [(1.0, 1.0), (2.0, 1.5), (1.5, 2.0)])
    def test_plane_paraxial_surface_refraction(self, n1, n2, set_test_backend):
        lens = Optic()

        # add surfaces
        lens.add_surface(index=0, thickness=be.inf, material=IdealMaterial(n1))
        lens.add_surface(
            index=1,
            surface_type="paraxial",
            material=IdealMaterial(n2),
            thickness=0.0,
            f=be.inf,
            is_stop=True,
        )
        lens.add_surface(index=2, material=IdealMaterial(n2))

        model = lens.surface_group.surfaces[1].interaction_model
        assert isinstance(model, ThinLensInteractionModel)

        ray_in = RealRays(0.0, 0.0, -1.0, 0.1, 0.2, 1.0, intensity=1.0, wavelength=0.55)
        ray_out = RealRays(
            0.0, 0.0, -1.0, 0.1, 0.2, 1.0 * n2 / n1, intensity=1.0, wavelength=0.55
        )
        ray_out.normalize()
        model.interact_real_rays(ray_in)

        assert_allclose(
            [ray_in.x, ray_in.y, ray_in.z, ray_in.L, ray_in.M, ray_in.N],
            [ray_out.x, ray_out.y, ray_out.z, ray_out.L, ray_out.M, ray_out.N],
        )

    def test_flip(self, surface):
        f_initial = be.copy(surface.interaction_model.f)
        surface.interaction_model.flip()
        assert_allclose(surface.interaction_model.f, f_initial)

    def test_to_dict(self, surface):
        data = surface.interaction_model.to_dict()
        assert "focal_length" in data
        assert data["focal_length"] == 42

    def test_from_dict(self, surface):
        data = surface.interaction_model.to_dict()
        data["material_pre"] = None
        assert surface.interaction_model.from_dict(data, None)
