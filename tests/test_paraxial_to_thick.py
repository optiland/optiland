from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import optiland.backend as be
from optiland.interactions import ThinLensInteractionModel
from optiland.optic import Optic
from optiland.samples.objectives import HeliarLens
from optiland.surfaces import (
    ParaxialToThickLensConverter,
    convert_to_thick_lens,
)

from .utils import assert_allclose


class TestParaxialToThickLensConverter:
    def test_init_with_invalid_surface_type(self, set_test_backend):
        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="standard", f=50)
        lens.wavelengths.add(0.55, is_primary=True)
        with pytest.raises(TypeError):
            ParaxialToThickLensConverter(lens.surfaces[1], lens)

    def test_missing_paraxial_surface(self, set_test_backend):
        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="paraxial", f=50)
        lens.wavelengths.add(0.55, is_primary=True)
        conv = ParaxialToThickLensConverter(lens.surfaces[1], lens, "N-BK7")
        lens.surfaces.remove(index=1)
        with pytest.raises(RuntimeError):
            conv.convert()

    def test_resolve_material_string_success(self, set_test_backend):
        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="paraxial", f=50)
        lens.wavelengths.add(0.55, is_primary=True)
        conv = ParaxialToThickLensConverter(lens.surfaces[1], lens, "N-BK7")
        assert hasattr(conv._material_instance, "n")

    def test_resolve_material_string_failure(self, set_test_backend):
        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="paraxial", f=50)
        lens.wavelengths.add(0.55, is_primary=True)
        with pytest.raises(ValueError):
            ParaxialToThickLensConverter(lens.surfaces[1], lens, "NOT_A_MATERIAL")

    def test_resolve_material_float(self, set_test_backend):
        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="paraxial", f=50)
        lens.wavelengths.add(0.55, is_primary=True)
        conv = ParaxialToThickLensConverter(lens.surfaces[1], lens, 1.5)
        assert_allclose(conv._material_instance.n(0.55), 1.5)

    def test_resolve_material_basematerial(self, set_test_backend):
        from optiland.materials.ideal import IdealMaterial

        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="paraxial", f=50)
        lens.wavelengths.add(0.55, is_primary=True)
        mat = IdealMaterial(1.7)
        conv = ParaxialToThickLensConverter(lens.surfaces[1], lens, mat)
        assert conv._material_instance is mat

    def test_resolve_material_invalid_type(self, set_test_backend):
        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="paraxial", f=50)
        lens.wavelengths.add(0.55, is_primary=True)
        with pytest.raises(TypeError):
            ParaxialToThickLensConverter(lens.surfaces[1], lens, [1, 2, 3])

    def test_get_paraxial_surface_index_found(self, set_test_backend):
        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="paraxial", f=50)
        lens.surfaces.add(index=2)
        lens.wavelengths.add(0.55, is_primary=True)
        paraxial = lens.surfaces[1]
        conv = ParaxialToThickLensConverter(paraxial, lens)
        assert conv._get_paraxial_surface_index() == 1

    def test_get_paraxial_surface_index_not_found(self, set_test_backend):
        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="paraxial", f=50)
        lens.wavelengths.add(0.55, is_primary=True)
        paraxial = lens.surfaces[1]
        conv = ParaxialToThickLensConverter(paraxial, lens)
        conv.paraxial_surface = MagicMock()  # force mismatch
        assert conv._get_paraxial_surface_index() is None

    def test_calculate_radii_zero_focal_length(self, set_test_backend):
        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="paraxial", f=0.0)
        lens.wavelengths.add(0.55, is_primary=True)
        conv = ParaxialToThickLensConverter(lens.surfaces[1], lens, 1.5)
        r1, r2 = conv._calculate_radii()
        assert np.isinf(r1) and np.isinf(r2)

    def test_calculate_radii_positive_and_negative(self, set_test_backend):
        lens = Optic()
        lens.surfaces.add(index=0, thickness=100)
        lens.surfaces.add(index=1, surface_type="paraxial", f=100.0)
        lens.wavelengths.add(0.55, is_primary=True)
        conv = ParaxialToThickLensConverter(lens.surfaces[1], lens, 1.5)
        r1, r2 = conv._calculate_radii()
        assert r1 > 0 and r2 < 0

        lens2 = Optic()
        lens2.surfaces.add(index=0, thickness=be.inf)
        lens2.surfaces.add(index=1, surface_type="paraxial", f=-100.0)
        lens2.wavelengths.add(0.55, is_primary=True)
        conv2 = ParaxialToThickLensConverter(lens2.surfaces[1], lens2, 1.5)
        r1n, r2n = conv2._calculate_radii()
        assert r1n < 0 and r2n > 0

    def test_remove_paraxial_surface_valid_and_invalid(self, set_test_backend):
        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="paraxial", f=50)
        lens.surfaces.add(index=2)
        lens.wavelengths.add(0.55, is_primary=True)
        paraxial = lens.surfaces[1]
        conv = ParaxialToThickLensConverter(paraxial, lens)

        # valid removal
        conv._remove_paraxial_surface(1)
        assert all(
            not isinstance(s.interaction_model, ThinLensInteractionModel)
            for s in lens.surfaces
        )

        # invalid removal
        with pytest.raises(IndexError):
            conv._remove_paraxial_surface(0)

    def test_convert_replaces_surface(self, set_test_backend):
        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="paraxial", thickness=10, f=50)
        lens.surfaces.add(index=2)
        lens.wavelengths.add(0.55, is_primary=True)
        paraxial = lens.surfaces[1]
        conv = ParaxialToThickLensConverter(paraxial, lens, 1.5)
        conv.convert()
        assert all(
            not isinstance(s.interaction_model, ThinLensInteractionModel)
            for s in lens.surfaces
        )

    def test_convert_to_thick_lens_function(self, set_test_backend):
        lens = HeliarLens()
        # ensure there is at least one paraxial surface to convert
        lens.surfaces.add(index=1, surface_type="paraxial", f=50)
        lens.wavelengths.add(0.55, is_primary=True)
        new_lens = convert_to_thick_lens(lens)
        assert isinstance(new_lens, Optic)
        assert all(
            not isinstance(s.interaction_model, ThinLensInteractionModel)
            for s in new_lens.surfaces
        )

    def test_biconvex_unsolvable_linear_case(self, set_test_backend):
        lens = Optic()
        # n = 1, f_target > 0 --> (n-1)=0 --> a_quad=0, b_quad=0
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="paraxial", f=100)
        lens.wavelengths.add(0.55, is_primary=True)
        conv = ParaxialToThickLensConverter(lens.surfaces[1], lens, 1.0)
        with pytest.raises(ValueError):
            conv._calculate_radii()

    def test_biconvex_discriminant_negative(self, set_test_backend):
        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="paraxial", f=1.0)
        lens.wavelengths.add(0.55, is_primary=True)
        conv = ParaxialToThickLensConverter(lens.surfaces[1], lens, 10.0)
        conv.center_thickness = 1e6  # Force discriminant < 0
        with pytest.raises(ValueError):
            conv._calculate_radii()

    def test_biconvex_no_positive_solution(self, set_test_backend):
        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="paraxial", f=1.0)
        lens.wavelengths.add(0.55, is_primary=True)
        conv = ParaxialToThickLensConverter(lens.surfaces[1], lens, 0.5)
        conv.center_thickness = 0.1
        with pytest.raises(ValueError):
            conv._calculate_radii()

    def test_biconcave_unsolvable_linear_case(self, set_test_backend):
        lens = Optic()
        # n = 1, f_target < 0 --> (n-1)=0 --> a_quad=0, b_quad=0
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="paraxial", f=-100)
        lens.wavelengths.add(0.55, is_primary=True)
        conv = ParaxialToThickLensConverter(lens.surfaces[1], lens, 1.0)
        with pytest.raises(ValueError):
            conv._calculate_radii()

    def test_biconcave_discriminant_negative(self, set_test_backend):
        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="paraxial", f=-1.0)  # diverging
        lens.wavelengths.add(0.55, is_primary=True)
        conv = ParaxialToThickLensConverter(lens.surfaces[1], lens, 1.5)
        # Force discriminant < 0 with large negative d
        conv.center_thickness = -1e6
        with pytest.raises(ValueError, match="discriminant < 0"):
            conv._calculate_radii()

    def test_biconcave_no_negative_solution(self, set_test_backend):
        lens = Optic()
        lens.surfaces.add(index=0, thickness=be.inf)
        lens.surfaces.add(index=1, surface_type="paraxial", f=-1.0)
        lens.wavelengths.add(0.55, is_primary=True)
        conv = ParaxialToThickLensConverter(lens.surfaces[1], lens, 1.01)
        # Adjust thickness to force both roots ≥ 0
        conv.center_thickness = -1e3
        with pytest.raises(ValueError):
            conv._calculate_radii()
