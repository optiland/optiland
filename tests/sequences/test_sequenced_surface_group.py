from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.plane import Plane
from optiland.materials.ideal import IdealMaterial
from optiland.rays.real_rays import RealRays
from optiland.sequences.sequenced_surface_group import SequencedSurfaceGroup
from optiland.surfaces.standard_surface import Surface

from ..utils import assert_allclose


def _plane_surface(previous, material_post, z=0.0, is_stop=False):
    geometry = Plane(CoordinateSystem(z=z))
    return Surface(
        previous_surface=previous,
        material_post=material_post,
        geometry=geometry,
        is_stop=is_stop,
    )


def _build_chain(materials, stop_index=None):
    surfaces = []
    previous = None
    for i, material in enumerate(materials):
        surfaces.append(
            _plane_surface(previous, material, z=float(i), is_stop=(i == stop_index))
        )
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


AIR = IdealMaterial(n=1.0)
GLASS_A = IdealMaterial(n=1.5)
GLASS_B = IdealMaterial(n=1.6)


class TestSequencedSurfaceGroupBasics:
    def test_len_iter_getitem(self, set_test_backend):
        surfaces = _build_chain([AIR, GLASS_A, GLASS_B])
        group = SequencedSurfaceGroup(surfaces, [0, 1, 2])

        assert len(group) == 3
        assert group.num_surfaces == 3
        assert [v.base_surface for v in group] == surfaces
        assert group[1].base_surface is surfaces[1]

    def test_stop_index_found_via_base_surface(self, set_test_backend):
        surfaces = _build_chain([AIR, GLASS_A, GLASS_B], stop_index=1)
        group = SequencedSurfaceGroup(surfaces, [0, 1, 2])
        assert group.stop_index == 1

    def test_stop_index_raises_when_absent(self, set_test_backend):
        surfaces = _build_chain([AIR, GLASS_A, GLASS_B])
        group = SequencedSurfaceGroup(surfaces, [0, 1, 2])
        with pytest.raises(ValueError, match="No stop surface"):
            _ = group.stop_index

    def test_invalid_sequence_raises_at_construction(self, set_test_backend):
        from optiland.sequences.resolver import SequenceValidationError

        surfaces = _build_chain([AIR, GLASS_A, GLASS_B])
        with pytest.raises(SequenceValidationError):
            SequencedSurfaceGroup(surfaces, [0, 2])


class TestSequencedSurfaceGroupTrace:
    def test_forward_trace_matches_nominal(self, set_test_backend):
        materials = [AIR, GLASS_A, AIR]
        surfaces = _build_chain(materials)

        nominal_rays = _make_rays()
        for surface in surfaces[1:]:
            nominal_rays = surface.trace(nominal_rays)

        group = SequencedSurfaceGroup(surfaces, [1, 2])
        view_rays = group.trace(_make_rays())

        assert_allclose(view_rays.x, nominal_rays.x)
        assert_allclose(view_rays.y, nominal_rays.y)
        assert_allclose(view_rays.L, nominal_rays.L)
        assert_allclose(view_rays.opd, nominal_rays.opd)

        assert_allclose(group.x[-1], surfaces[-1].x)
        assert_allclose(group.opd[-1], surfaces[-1].opd)

    def test_stacked_records_have_one_row_per_step_not_per_surface(
        self, set_test_backend
    ):
        # A revisit sequence: 4 steps over only 2 distinct base surfaces.
        materials = [AIR, GLASS_A, AIR]
        surfaces = _build_chain(materials)
        group = SequencedSurfaceGroup(surfaces, [1, (2, "reflect"), (1, "reflect"), 2])
        group.trace(_make_rays())

        assert group.x.shape[0] == 4
        assert group.opd.shape[0] == 4

    def test_reset_clears_all_views(self, set_test_backend):
        materials = [AIR, GLASS_A, AIR]
        surfaces = _build_chain(materials)
        group = SequencedSurfaceGroup(surfaces, [1, 2])
        group.trace(_make_rays())
        assert be.size(group[0].x) > 0

        group.reset()
        assert be.size(group[0].x) == 0
        assert be.size(group[1].x) == 0

    def test_skip_argument(self, set_test_backend):
        materials = [AIR, GLASS_A, GLASS_B, AIR]
        surfaces = _build_chain(materials)
        group = SequencedSurfaceGroup(surfaces, [1, 2, 3])

        group.trace(_make_rays(), skip=1)
        assert be.size(group[0].x) == 0
        assert be.size(group[1].x) > 0
        assert be.size(group[2].x) > 0


class TestSequencedSurfaceGroupGeometryAccessors:
    def test_positions_and_thickness(self, set_test_backend):
        materials = [AIR, GLASS_A, GLASS_B]
        surfaces = _build_chain(materials)
        group = SequencedSurfaceGroup(surfaces, [0, 1, 2])

        assert_allclose(group.positions.ravel(), be.array([0.0, 1.0, 2.0]))
        assert float(group.get_thickness(0).ravel()[0]) == pytest.approx(1.0)

    def test_n_returns_exit_medium_per_step(self, set_test_backend):
        materials = [AIR, GLASS_A, GLASS_B]
        surfaces = _build_chain(materials)
        group = SequencedSurfaceGroup(surfaces, [0, 1, 2])

        n = group.n(0.55)
        assert_allclose(n, be.array([1.0, 1.5, 1.6]))
