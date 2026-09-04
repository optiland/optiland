from __future__ import annotations

import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.plane import Plane
from optiland.interactions.refractive_reflective_model import RefractiveReflectiveModel
from optiland.materials.ideal import IdealMaterial
from optiland.sequences.resolver import SequenceValidationError, resolve_sequence
from optiland.sequences.surface_view import resolve_view_materials
from optiland.surfaces.standard_surface import Surface


def _plane_surface(previous, material_post, z=0.0, is_reflective=False):
    geometry = Plane(CoordinateSystem(z=z))
    surf = Surface(
        previous_surface=previous, material_post=material_post, geometry=geometry
    )
    if is_reflective:
        surf.interaction_model = RefractiveReflectiveModel(
            parent_surface=surf, is_reflective=True
        )
    return surf


def _build_chain(materials, reflective_indices=None):
    """Build a forward chain of surfaces whose material_post is ``materials[i]``."""
    surfaces = []
    previous = None
    reflective_indices = reflective_indices or set()
    for i, material in enumerate(materials):
        is_refl = i in reflective_indices
        surface = _plane_surface(previous, material, z=float(i), is_reflective=is_refl)
        surfaces.append(surface)
        previous = surface
    return surfaces


AIR = IdealMaterial(n=1.0)
GLASS_A = IdealMaterial(n=1.5)
GLASS_B = IdealMaterial(n=1.6)


class TestResolveViewMaterials:
    def test_forward_nominal(self):
        surfaces = _build_chain([AIR, GLASS_A, GLASS_B])
        pre, post = resolve_view_materials(
            surfaces[1], reverse=False, interaction_override=None
        )
        assert pre == AIR
        assert post == GLASS_A

    def test_reverse_swaps_pre_and_post(self):
        surfaces = _build_chain([AIR, GLASS_A, GLASS_B])
        pre, post = resolve_view_materials(
            surfaces[1], reverse=True, interaction_override=None
        )
        assert pre == GLASS_A
        assert post == AIR

    def test_reflect_override_collapses_post_to_pre(self):
        surfaces = _build_chain([AIR, GLASS_A, GLASS_B])
        pre, post = resolve_view_materials(
            surfaces[1], reverse=False, interaction_override="reflect"
        )
        assert pre == AIR
        assert post == AIR

    def test_reflect_override_in_reverse(self):
        surfaces = _build_chain([AIR, GLASS_A, GLASS_B])
        pre, post = resolve_view_materials(
            surfaces[1], reverse=True, interaction_override="reflect"
        )
        assert pre == GLASS_A
        assert post == GLASS_A


class TestResolveSequence:
    def test_simple_forward_sequence_is_valid(self):
        surfaces = _build_chain([AIR, GLASS_A, GLASS_B])
        views = resolve_sequence(surfaces, [0, 1, 2])
        assert [v.base_surface for v in views] == surfaces
        assert [v.reverse for v in views] == [False, False, False]

    def test_ghost_example_from_spec_is_valid(self):
        # SPEC_multi_sequence_20260731.md section 3.3's motivating example:
        # a two-bounce ghost between surfaces 2 and 3.
        surfaces = _build_chain([AIR, GLASS_A, GLASS_B, AIR, AIR])
        views = resolve_sequence(
            surfaces, [0, 1, 2, (3, "reflect"), (2, "reflect"), 3, 4]
        )
        assert len(views) == 7
        assert [v.reverse for v in views] == [
            False,
            False,
            False,
            False,
            True,
            False,
            False,
        ]
        assert [v.interaction_override for v in views] == [
            None,
            None,
            None,
            "reflect",
            "reflect",
            None,
            None,
        ]

    def test_list_steps_syntax_is_valid(self):
        surfaces = _build_chain([AIR, GLASS_A, GLASS_B, AIR, AIR])
        views = resolve_sequence(
            surfaces, [0, 1, 2, [3, "reflect"], [2, "reflect"], 3, 4]
        )
        assert len(views) == 7
        assert [v.reverse for v in views] == [
            False,
            False,
            False,
            False,
            True,
            False,
            False,
        ]

    def test_nominal_mirror_infers_reverse_direction(self):
        # Surface 1 is a nominal mirror (e.g. in a Cassegrain telescope)
        surfaces = _build_chain([AIR, AIR, AIR], reflective_indices={1})
        views = resolve_sequence(surfaces, [0, 1, 2])
        assert [v.reverse for v in views] == [False, False, True]

    def test_skipped_surface_breaks_medium_chain(self):
        surfaces = _build_chain([AIR, GLASS_A, GLASS_B])
        with pytest.raises(SequenceValidationError, match="step 1"):
            resolve_sequence(surfaces, [0, 2])

    def test_out_of_range_index_raises(self):
        surfaces = _build_chain([AIR, GLASS_A])
        with pytest.raises(ValueError, match="index 5"):
            resolve_sequence(surfaces, [0, 5])

    def test_empty_steps_raises(self):
        surfaces = _build_chain([AIR, GLASS_A])
        with pytest.raises(ValueError, match="at least one step"):
            resolve_sequence(surfaces, [])

    def test_material_edit_is_visible_through_view(self):
        """Editing the base surface's material must be visible in the view.

        This is the linkage requirement from the SPEC: geometry/materials are
        shared by reference, so an optimizer touching the base surface is
        immediately reflected in every sequence.
        """
        surfaces = _build_chain([AIR, GLASS_A, GLASS_B])
        views = resolve_sequence(surfaces, [0, 1, 2])

        new_glass = IdealMaterial(n=1.9)
        surfaces[1].material_post = new_glass

        assert views[1].material_post == new_glass
        assert views[2].material_pre == new_glass
