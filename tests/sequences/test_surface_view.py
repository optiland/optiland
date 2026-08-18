from __future__ import annotations

import optiland.backend as be
from optiland.coatings import FresnelCoating
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.plane import Plane
from optiland.materials.ideal import IdealMaterial
from optiland.rays.real_rays import RealRays
from optiland.sequences.resolver import resolve_sequence
from optiland.surfaces.standard_surface import Surface

from ..utils import assert_allclose


def _plane_surface(previous, material_post, z=0.0):
    geometry = Plane(CoordinateSystem(z=z))
    return Surface(
        previous_surface=previous,
        material_post=material_post,
        geometry=geometry,
    )


def _build_chain(materials):
    surfaces = []
    previous = None
    for i, material in enumerate(materials):
        surfaces.append(_plane_surface(previous, material, z=float(i)))
        previous = surfaces[-1]
    return surfaces


def _make_rays():
    return RealRays(
        x=be.array([0.0, 0.5]),
        y=be.array([0.0, -0.3]),
        z=be.array([0.0, 0.0]),
        L=be.array([0.0, 0.05]),
        M=be.array([0.0, -0.02]),
        N=be.array([1.0, 0.998]),
        intensity=be.array([1.0, 1.0]),
        wavelength=0.55,
    )


class TestSurfaceViewTracesLikeSurface:
    """A pure-forward view sequence must reproduce the nominal Surface trace."""

    def test_real_ray_trace_matches_nominal(self, set_test_backend):
        materials = [
            IdealMaterial(n=1.0),
            IdealMaterial(n=1.5),
            IdealMaterial(n=1.0),
        ]
        surfaces = _build_chain(materials)

        nominal_rays = _make_rays()
        for surface in surfaces[1:]:
            nominal_rays = surface.trace(nominal_rays)

        views = resolve_sequence(surfaces, [1, 2])
        view_rays = _make_rays()
        for view in views:
            view_rays = view.trace(view_rays)

        assert_allclose(view_rays.x, nominal_rays.x)
        assert_allclose(view_rays.y, nominal_rays.y)
        assert_allclose(view_rays.z, nominal_rays.z)
        assert_allclose(view_rays.L, nominal_rays.L)
        assert_allclose(view_rays.M, nominal_rays.M)
        assert_allclose(view_rays.N, nominal_rays.N)
        assert_allclose(view_rays.opd, nominal_rays.opd)

        for view, surface in zip(views, surfaces[1:], strict=True):
            assert_allclose(view.x, surface.x)
            assert_allclose(view.y, surface.y)
            assert_allclose(view.opd, surface.opd)


class TestSurfaceViewRevisit:
    """Revisiting a base surface must not corrupt either view's buffers."""

    def test_revisited_surface_has_independent_buffers(self, set_test_backend):
        materials = [
            IdealMaterial(n=1.0),
            IdealMaterial(n=1.5),
            IdealMaterial(n=1.0),
        ]
        surfaces = _build_chain(materials)

        views = resolve_sequence(surfaces, [1, (2, "reflect"), (1, "reflect"), 2])
        first_hit_of_1, _, second_hit_of_1, _ = views

        rays = _make_rays()
        for view in views:
            rays = view.trace(rays)

        assert first_hit_of_1 is not second_hit_of_1
        # Each visit recorded its own ray state; the second visit's record
        # did not overwrite the first's.
        assert not be.all(first_hit_of_1.x == second_hit_of_1.x) or not be.all(
            first_hit_of_1.L == second_hit_of_1.L
        )

    def test_base_surface_own_record_untouched_by_views(self, set_test_backend):
        materials = [IdealMaterial(n=1.0), IdealMaterial(n=1.5)]
        surfaces = _build_chain(materials)
        base = surfaces[1]

        assert be.size(base.x) == 0

        views = resolve_sequence(surfaces, [1])
        rays = _make_rays()
        for view in views:
            rays = view.trace(rays)

        # The view recorded data; the shared base Surface object was never
        # traced directly and keeps its own empty buffers.
        assert be.size(views[0].x) == 2
        assert be.size(base.x) == 0


class TestSurfaceViewCoatings:
    def test_coating_rebinds_media_in_reverse_view(self):
        air = IdealMaterial(n=1.0)
        glass = IdealMaterial(n=1.5)
        surfaces = _build_chain([air, glass, air])
        surfaces[1].interaction_model.coating = FresnelCoating(air, glass)

        views = resolve_sequence(surfaces, [1, (2, "reflect"), (1, "reflect"), 2])
        reverse_view = views[2]
        assert reverse_view.reverse is True
        assert reverse_view.material_pre == glass
        assert reverse_view.material_post == glass
        assert reverse_view.interaction_model.coating.material_pre == glass
        assert reverse_view.interaction_model.coating.material_post == glass
