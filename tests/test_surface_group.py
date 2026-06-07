from __future__ import annotations

import pytest

import optiland.backend as be
from optiland import optic
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import StandardGeometry
from optiland.materials import IdealMaterial
from optiland.surfaces.standard_surface import Surface
from optiland.surfaces.surface_group import SurfaceGroup
from tests.utils import assert_allclose


# Helper to create a real Surface instance for tests
def create_real_surface(
    name="s",
    thickness_val=10.0,
    initial_z=0.0,
    is_stop=False,
    comment="",
    radius=be.inf,
    material_n=1.0,
):
    """
    Creates a real Surface object.
    The 'thickness' attribute is crucial for SurfaceGroup._update_coordinate_systems.
    We will set it directly on the surface instance.
    """
    cs = CoordinateSystem(z=initial_z)
    # StandardGeometry needs a radius; be.inf can represent a plane.
    geom = StandardGeometry(coordinate_system=cs, radius=radius)

    # Surfaces require material_pre and material_post
    mat_post = IdealMaterial(n=material_n)  # Keep it simple for now, or vary if needed

    surface = Surface(
        previous_surface=None,
        geometry=geom,
        material_post=mat_post,
        is_stop=is_stop,
        comment=comment or name,
        surface_type="Standard",  # A default type
    )
    surface.thickness = thickness_val
    surface.name = name  # add for testing

    return surface


class TestSurfaceGroupUpdatesRealObjects:
    def _setup_surface_group(
        self,
        num_initial_surfaces=0,
        use_absolute_cs=False,
        thicknesses=None,
        initial_zs=None,
    ):
        """Helper to create a SurfaceGroup with real surfaces."""
        sg = SurfaceGroup([])  # Start with empty
        # The SurfaceGroup initializes its own SurfaceFactory.
        # We can configure its 'use_absolute_cs' property.
        sg.surface_factory.use_absolute_cs = use_absolute_cs

        initial_surfaces = []
        if thicknesses is None:
            thicknesses = [10.0] * num_initial_surfaces
        if initial_zs is None:
            # Default initial Z: obj at -100, others at 0 before potential update
            initial_zs = [
                -100.0 if i == 0 else 0.0 for i in range(num_initial_surfaces)
            ]
            if num_initial_surfaces == 0:
                initial_zs = []

        for i in range(num_initial_surfaces):
            surface = create_real_surface(
                name=f"s{i}", thickness_val=thicknesses[i], initial_z=initial_zs[i]
            )
            initial_surfaces.append(surface)

        sg._surfaces = initial_surfaces
        sg._update_surface_links()

        if not use_absolute_cs:
            sg._update_coordinate_systems(start_index=0)

        return sg

    @pytest.mark.parametrize("use_absolute_cs", [True, False])
    def test_add_surface_new_object_append_empty(
        self, set_test_backend, use_absolute_cs
    ):
        sg = self._setup_surface_group(use_absolute_cs=use_absolute_cs)
        new_surf = create_real_surface(name="new", thickness_val=5.0, initial_z=10.0)
        sg.add(new_surface=new_surf)

        assert len(sg.surfaces) == 1
        assert sg.surfaces[0] is new_surf
        assert_allclose(
            sg.surfaces[0].geometry.cs.z, be.array(10.0)
        )  # Keeps its initial Z

    def test_add_surface_new_object_append_non_empty_abs_cs(self, set_test_backend):
        sg = self._setup_surface_group(
            num_initial_surfaces=1,
            use_absolute_cs=True,
            thicknesses=[10],
            initial_zs=[5.0],
        )
        s0_z = be.copy(sg.surfaces[0].geometry.cs.z)

        new_surf = create_real_surface(name="new", thickness_val=5.0, initial_z=20.0)
        sg.add(new_surface=new_surf)  # Appends

        assert len(sg.surfaces) == 2
        assert sg.surfaces[1] is new_surf
        assert_allclose(sg.surfaces[0].geometry.cs.z, s0_z)  # Unchanged
        assert_allclose(sg.surfaces[1].geometry.cs.z, be.array(20.0))  # Keeps its Z

    def test_add_surface_new_object_append_non_empty_rel_cs_no_update(
        self, set_test_backend
    ):
        sg = self._setup_surface_group(
            num_initial_surfaces=1,
            use_absolute_cs=False,
            thicknesses=[10],
            initial_zs=[-100.0],
        )
        # S0(z=-100, t=10)
        s0 = sg.surfaces[0]

        new_surf = create_real_surface(
            name="S1_appended", thickness_val=5.0, initial_z=123.0
        )
        # Appending: index is not specified or index == len(sg.surfaces)
        sg.add(
            new_surface=new_surf
        )  # Appends, update_start_index points to new_surf, is_last_surface=True

        assert len(sg.surfaces) == 2
        assert sg.surfaces[1] is new_surf
        assert_allclose(s0.geometry.cs.z, be.array(-100.0))
        # _update_coordinate_systems is NOT called when appending to make it the last surface
        assert_allclose(sg.surfaces[1].geometry.cs.z, be.array(123.0))

    def test_add_surface_new_object_insert_middle_rel_cs_triggers_update(
        self, set_test_backend
    ):
        sg = self._setup_surface_group(
            num_initial_surfaces=2,
            use_absolute_cs=False,
            thicknesses=[10, 20],
            initial_zs=[-100.0, 0.0],
        )
        # After setup: S0(z=-100, t=10), S1_orig(z=0, t=20)
        s0_orig = sg.surfaces[0]
        s1_orig = sg.surfaces[1]

        s_new = create_real_surface(name="S_new", thickness_val=5.0, initial_z=123.0)
        sg.add(new_surface=s_new, index=1)  # Insert s_new at index 1

        # Expected: [s0_orig, s_new, s1_orig]
        assert len(sg.surfaces) == 3
        assert sg.surfaces[0] is s0_orig
        assert sg.surfaces[1] is s_new
        assert sg.surfaces[2] is s1_orig
        assert sg.surfaces[0].name == "s0"
        assert sg.surfaces[1].name == "S_new"
        assert sg.surfaces[2].name == "s1"

        # Coordinate checks for relative CS update
        assert_allclose(
            sg.surfaces[0].geometry.cs.z, be.array(-100.0)
        )  # S0 z unchanged
        # s_new (at index 1) z should be updated to 0.0 by _update_coordinate_systems
        assert_allclose(sg.surfaces[1].geometry.cs.z, be.array(0.0))
        # s1_orig (now at index 2) z should be s_new.thickness
        assert_allclose(
            sg.surfaces[2].geometry.cs.z,
            be.array(s_new.thickness),  # 5.0
        )

    def test_add_surface_new_object_is_stop_propagation(self, set_test_backend):
        sg = self._setup_surface_group(num_initial_surfaces=2, use_absolute_cs=False)
        sg.surfaces[0].is_stop = True  # Set S0 as stop
        new_surf = create_real_surface(name="new_stop", is_stop=True)
        # Append by providing index = len(sg.surfaces)
        sg.add(new_surface=new_surf, index=len(sg.surfaces))

        assert len(sg.surfaces) == 3
        assert not sg.surfaces[0].is_stop  # Old stop is cleared
        assert not sg.surfaces[1].is_stop
        assert sg.surfaces[2].is_stop

    # --- Tests for add_surface by creation (new_surface is None) ---
    def test_add_surface_by_creation_rel_cs(self, set_test_backend):
        sg = self._setup_surface_group(
            num_initial_surfaces=1,
            use_absolute_cs=False,
            thicknesses=[10],
            initial_zs=[-100.0],
        )

        # Add S1_created at index 1. sg: [s0 (z=-100, t=10), S1_created (z=0, t=5.0)]
        # The 'thickness' kwarg should be handled by the actual SurfaceFactory
        sg.add(
            surface_type="standard",
            comment="S1_created",
            index=1,  # Insert at index 1
            material="air",
            thickness=5.0,
        )

        assert len(sg.surfaces) == 2
        s0 = sg.surfaces[0]
        s1_created = sg.surfaces[1]
        assert s1_created.comment == "S1_created"
        assert s1_created.thickness == 5.0
        assert_allclose(s0.geometry.cs.z, be.array(-100.0))
        assert_allclose(
            s1_created.geometry.cs.z, be.array(0.0)
        )  # Updated by _update_coordinate_systems

        # Add S2_inserted at index 1.
        # sg: [s0 (z=-100, t=10), S2_inserted (z=0, t=12.0), S1_created (z=12.0, t=5.0)]
        sg.add(
            surface_type="standard",
            comment="S2_inserted",
            index=1,  # Insert S2_inserted at index 1, shifting S1_created
            material="glass",
            thickness=12.0,
        )

        assert len(sg.surfaces) == 3
        s0_after_second_add = sg.surfaces[0]
        s2_inserted = sg.surfaces[1]
        s1_created_shifted = sg.surfaces[2]

        assert s0_after_second_add is s0  # s0 object should be the same
        assert s2_inserted.comment == "S2_inserted"
        assert s2_inserted.thickness == 12.0
        assert (
            s1_created_shifted is s1_created
        )  # s1_created object should be the same, just shifted
        assert s1_created_shifted.comment == "S1_created"  # Verify it's the original S1
        assert s1_created_shifted.thickness == 5.0

        # Coordinate checks
        assert_allclose(s0_after_second_add.geometry.cs.z, be.array(-100.0))
        # S2_inserted is at index 1, so its z is updated to 0.0
        assert_allclose(s2_inserted.geometry.cs.z, be.array(0.0))
        # S1_created_shifted is at index 2, its z is S2_inserted.thickness
        assert_allclose(
            s1_created_shifted.geometry.cs.z, be.array(s2_inserted.thickness)
        )  # 12.0

    def test_add_surface_by_creation_error_no_index(self, set_test_backend):
        sg = self._setup_surface_group(use_absolute_cs=False)
        with pytest.raises(
            ValueError, match="Must define index when defining surface."
        ):
            sg.add(surface_type="standard", comment="no_index_surf")

    # --- Tests for remove_surface ---
    def test_remove_surface_middle_rel_cs_triggers_update(self, set_test_backend):
        sg = self._setup_surface_group(
            num_initial_surfaces=3,
            use_absolute_cs=False,
            thicknesses=[10, 20, 30],
            initial_zs=[-100, 0, 20],
        )
        # Initial: S0(z=-100,t=10), S1(z=0,t=20), S2(z=20,t=30)
        s2_original_comment = sg.surfaces[2].comment  # Was s2

        sg.remove(index=1)  # Remove S1

        assert len(sg.surfaces) == 2
        assert sg.surfaces[0].comment == "s0"
        assert sg.surfaces[1].comment == s2_original_comment  # Old S2 is now S1

        assert_allclose(sg.surfaces[0].geometry.cs.z, be.array(-100.0))  # S0 unchanged
        # Old S2 (now new S1) z should be 0.0 because it's now the surface at index 1 and update was called
        assert_allclose(sg.surfaces[1].geometry.cs.z, be.array(0.0))

    def test_remove_surface_last_rel_cs_no_update_triggered(self, set_test_backend):
        sg = self._setup_surface_group(
            num_initial_surfaces=3,
            use_absolute_cs=False,
            thicknesses=[10, 20, 30],
            initial_zs=[-100, 0, 20],
        )
        # S0(z=-100,t=10), S1(z=0,t=20), S2(z=20,t=30)
        s1_z_before_removal = be.copy(sg.surfaces[1].geometry.cs.z)

        sg.remove(index=2)  # Remove S2 (last optical surface)
        # _update_coordinate_systems not called because was_not_last_surface is false

        assert len(sg.surfaces) == 2
        assert_allclose(
            sg.surfaces[1].geometry.cs.z, s1_z_before_removal
        )  # S1 z unchanged

    def test_remove_surface_abs_cs_no_update(self, set_test_backend):
        sg = self._setup_surface_group(num_initial_surfaces=3, use_absolute_cs=True)
        # Set specific Zs that wouldn't occur with relative CS updates
        sg.surfaces[0].geometry.cs.z = be.array(0.0)
        sg.surfaces[1].geometry.cs.z = be.array(10.0)
        sg.surfaces[2].geometry.cs.z = be.array(30.0)

        s2_z_original = be.copy(sg.surfaces[2].geometry.cs.z)

        sg.remove(index=1)  # Remove S1

        assert len(sg.surfaces) == 2
        # With absolute_cs=True, _update_coordinate_systems is not called
        assert_allclose(
            sg.surfaces[1].geometry.cs.z, s2_z_original
        )  # Old S2 (now S1) keeps its original Z

    def test_remove_surface_error_remove_object(self, set_test_backend):
        sg = self._setup_surface_group(num_initial_surfaces=1)
        with pytest.raises(ValueError, match="Cannot remove object surface"):
            sg.remove(index=0)

    # --- Error condition tests from SurfaceGroup code directly ---
    def test_add_surface_new_object_error_negative_index(self, set_test_backend):
        sg = self._setup_surface_group(num_initial_surfaces=1)
        with pytest.raises(IndexError, match="Index -1 cannot be negative."):
            sg.add(new_surface=create_real_surface(), index=-1)

    def test_add_surface_new_object_error_index_out_of_bounds(self, set_test_backend):
        sg = self._setup_surface_group(num_initial_surfaces=1)  # surfaces[0] exists
        # Valid indices for insertion: 0, 1 (append). len(sg.surfaces) = 1.
        # Max index for insertion is len(sg.surfaces) = 1.
        # index=2 is out of bounds.
        with pytest.raises(
            IndexError,
            match=r"Index 2 is out of bounds for insertion. Max index for insertion is 1 \(to append\)\.",
        ):
            sg.add(new_surface=create_real_surface(), index=2)

    def test_add_surface_by_creation_error_negative_index(self, set_test_backend):
        sg = self._setup_surface_group()
        with pytest.raises(IndexError):
            sg.add(
                surface_type="standard", index=-1, thickness=1
            )  # Assuming factory handles thickness

    def test_add_surface_by_creation_error_index_out_of_bounds(self, set_test_backend):
        sg = self._setup_surface_group(num_initial_surfaces=1)  # len(sg.surfaces) = 1
        # Max index for insertion is 1. index=2 is out of bounds.
        with pytest.raises(IndexError):
            sg.add(surface_type="standard", index=2, thickness=1)

    def test_remove_surface_error_index_out_of_bounds_negative(self, set_test_backend):
        sg = self._setup_surface_group(num_initial_surfaces=2)
        with pytest.raises(IndexError, match="Index -1 is out of bounds"):
            sg.remove(index=-1)

    def test_remove_surface_error_index_out_of_bounds_too_large(self, set_test_backend):
        sg = self._setup_surface_group(num_initial_surfaces=2)  # Valid remove index: 1
        with pytest.raises(IndexError, match="Index 2 is out of bounds"):
            sg.remove(index=2)
        with pytest.raises(
            IndexError, match="Index 3 is out of bounds"
        ):  # Also too large
            sg.remove(index=3)

    def test_remove_surface_updates_material_link(self, set_test_backend):
        """
        Verify that removing a surface correctly updates the `material_pre`
        attribute of the subsequent surface. Identified in issue #363.
        """
        # 1. Setup lens with multiple materials
        lens = optic.Optic()
        mat1 = IdealMaterial(n=1.5)
        mat2 = IdealMaterial(n=2.0)
        mat3 = IdealMaterial(n=2.5)

        lens.surfaces.add(index=0, material="Air")
        lens.surfaces.add(index=1, material=mat1, thickness=5)  # Surface 1
        lens.surfaces.add(
            index=2, material=mat2, thickness=5
        )  # Surface 2 (to be removed)
        lens.surfaces.add(index=3, material=mat3, thickness=5)  # Surface 3
        lens.surfaces.add(index=4, material="Air")

        surface_before_removal = lens.surfaces.surfaces[1]
        surface_to_remove = lens.surfaces.surfaces[2]
        surface_after_removal = lens.surfaces.surfaces[3]

        # 2. Check initial state
        # The material before surface 3 should be mat2 (from surface 2)
        assert surface_after_removal.material_pre is surface_to_remove.material_post

        # 3. Remove surface at index 2
        lens.surfaces.remove(2)

        # 4. Get the new surface at index 2 (which was old surface 3)
        new_surface_at_index_2 = lens.surfaces.surfaces[2]

        # 5. Assert that the material link is updated
        # The material before the new surface at index 2 should now be mat1
        assert (
            new_surface_at_index_2.material_pre is surface_before_removal.material_post
        )

    def test_remove_surface_and_compare_raytrace(self, set_test_backend):
        """
        Tests that removing a surface and raytracing yields the same result
        as creating a new system in the final configuration.
        """
        # 1. Create the initial lens with two glass slabs
        lens1 = optic.Optic(name="Two Slabs")
        lens1.surfaces.add(index=0, thickness=be.inf, material="Air")
        lens1.surfaces.add(
            index=1,
            surface_type="standard",
            material=IdealMaterial(n=2.5),
            thickness=10,
            radius=be.inf,
        )
        lens1.surfaces.add(
            index=2,
            surface_type="standard",
            material=IdealMaterial(n=1.0001),
            thickness=10,
            radius=be.inf,
            is_stop=True,
        )
        lens1.surfaces.add(
            index=3,
            surface_type="standard",
            material="Air",
            thickness=20,
            radius=be.inf,
        )
        lens1.surfaces.add(index=4, material="Air")
        lens1.fields.set_type("angle")
        lens1.fields.add(y=10)
        lens1.wavelengths.add(500, unit="nm")
        lens1.set_aperture("float_by_stop_size", 25)

        # 2. Remove the first slab (the one with n=2.5) from lens1
        lens1.surfaces.remove(1)

        # 3. Trace rays through the modified lens1 and get final y-coordinates
        traced_rays1 = lens1.trace(Hx=0, Hy=1, wavelength=0.5, num_rays=3)
        y_coords_modified = traced_rays1.y

        # 4. Create a second lens from scratch with the expected final configuration
        lens2 = optic.Optic(name="One Slab")
        lens2.surfaces.add(index=0, thickness=be.inf, material="Air")
        # This surface corresponds to the one that remained in lens1
        lens2.surfaces.add(
            index=1,
            surface_type="standard",
            material=IdealMaterial(n=1.0001),
            thickness=10,
            radius=be.inf,
            is_stop=True,
        )
        lens2.surfaces.add(
            index=2,
            surface_type="standard",
            material="Air",
            thickness=20,
            radius=be.inf,
        )
        lens2.surfaces.add(index=3, material="Air")
        lens2.fields.set_type("angle")
        lens2.fields.add(y=10)
        lens2.wavelengths.add(500, unit="nm")
        lens2.set_aperture("float_by_stop_size", 25)

        # 5. Trace rays through the new lens2 and get final y-coordinates
        traced_rays2 = lens2.trace(Hx=0, Hy=1, wavelength=0.5, num_rays=3)
        y_coords_new = traced_rays2.y

        # 6. Assert that the ray trace results are identical
        assert_allclose(y_coords_modified, y_coords_new)

    # --- Tests for _update_coordinate_systems specific cases ---
    def test_update_coordinate_systems_infinite_thickness_error(self, set_test_backend):
        sg = self._setup_surface_group(
            num_initial_surfaces=2,
            use_absolute_cs=False,
            thicknesses=[10, be.inf],
            initial_zs=[-100, 0],
        )
        # S0(z=-100, t=10), S1(z=0, t=inf)
        # Add a third surface; its position calculation will depend on S1's infinite thickness
        s2 = create_real_surface(name="s2_after_inf", thickness_val=5.0)
        sg._surfaces.append(s2)  # Now [S0, S1, s2]
        sg._update_surface_links()

        # Update starting from index 2 (s2_after_inf), which looks at s1's thickness
        with pytest.raises(
            ValueError,
            match="Coordinate system update failed due to infinite thickness at surface 1",
        ):
            sg._update_coordinate_systems(start_index=2)

    def test_update_coordinate_systems_thickness_is_be_array(self, set_test_backend):
        sg = self._setup_surface_group(
            num_initial_surfaces=3,
            use_absolute_cs=False,
            thicknesses=[10, 20, 30],
            initial_zs=[-100, 0, 20],
        )
        # S0(z=-100,t=10), S1(z=0,t=20), S2(z=20,t=30)
        sg.surfaces[1].thickness = be.array(25.0)  # S1 thickness is a 0-d array

        sg._update_coordinate_systems(start_index=2)  # Update S2 based on S1

        # S2.z = S1.z + S1.thickness = 0.0 + 25.0
        assert_allclose(sg.surfaces[2].geometry.cs.z, be.array(25.0))

    def test_insert_all_at_index_1(self, set_test_backend):
        from optiland.samples import CookeTriplet

        cooke = CookeTriplet()

        lens = optic.Optic()
        lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        lens.surfaces.add(index=1)

        for surf in cooke.surfaces.surfaces[-2:0:-1]:
            lens.surfaces.add(
                radius=surf.geometry.radius,
                index=1,
                material=surf.material_post,
                is_stop=surf.is_stop,
                thickness=surf.thickness,
            )

        lens.set_aperture(aperture_type="EPD", value=10)
        lens.fields.set_type(field_type="angle")
        lens.fields.fields = cooke.fields.fields
        lens.wavelengths.wavelengths = cooke.wavelengths.wavelengths

        rays_lens = lens.trace(
            Hx=0, Hy=1, distribution="hexapolar", num_rays=3, wavelength=0.55
        )
        rays_cooke = cooke.trace(
            Hx=0, Hy=1, distribution="hexapolar", num_rays=3, wavelength=0.55
        )
        assert_allclose(
            be.mean(rays_cooke.y),
            be.tan(be.radians(be.max([field.y for field in cooke.fields.fields])))
            * cooke.paraxial.f2(),
            atol=0.1,
        )

        assert_allclose(
            rays_cooke.y, rays_lens.y
        )  # mean y position for Cooke triplet defined above

    def test_set_stop_index(self):
        lens = optic.Optic()

        lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
        lens.surfaces.add(index=1, radius=be.inf, thickness=5, is_stop=True)
        lens.surfaces.add(index=2, radius=be.inf, thickness=5)
        lens.surfaces.add(index=3, radius=be.inf, thickness=5)
        lens.surfaces.stop_index = 2
        assert lens.surfaces.surfaces[2].is_stop == True
        with pytest.raises(ValueError, match="Index out of range"):
            lens.surfaces.stop_index = 0
        with pytest.raises(ValueError, match="Index out of range"):
            lens.surfaces.stop_index = 3

    @pytest.mark.skipif(
        be.get_backend() == "torch",
        reason="Independent of backend: does not need to run twice",
    )
    def test_second_object_surface_raises(self):
        lens1 = optic.Optic()
        lens1.surfaces.add(index=0, thickness=be.inf, material="Air")
        with pytest.raises(
            ValueError,
            match=("Surface index cannot be zero after first surface is created."),
        ):
            lens1.surfaces.add(index=0, thickness=be.inf, material="Air")


class TestSurfaceGroupAdd:
    """Tests for SurfaceGroup.__add__ (concatenation operator). - Issue #477"""

    def _make_relay(self):
        """Two-element relay with an explicit image-plane marker."""
        o = optic.Optic()
        o.set_aperture(aperture_type="EPD", value=10.0)
        o.fields.set_type(field_type="angle")
        o.fields.add(y=0)
        o.fields.add(y=10)
        o.wavelengths.add(value=0.55, is_primary=True)
        o.surfaces.add(index=0, thickness=be.inf)
        o.surfaces.add(index=1, radius=100, thickness=5, material="N-BK7", is_stop=True)
        o.surfaces.add(index=2, radius=-100, thickness=40)
        o.surfaces.add(index=3, radius=be.inf, thickness=0)  # image plane
        return o

    def _make_eye(self):
        """Simple eye model with object plane and retina image surface."""
        o = optic.Optic()
        o.surfaces.add(index=0, thickness=0)  # object at image plane
        o.surfaces.add(index=1, radius=7.8, thickness=3.6, material="N-BK7")
        o.surfaces.add(index=2, radius=-6.0, thickness=16.6)
        o.surfaces.add(index=3, radius=be.inf)  # retina
        return o

    def _make_monolithic(self):
        """Monolithic equivalent of relay + eye."""
        o = optic.Optic()
        o.set_aperture(aperture_type="EPD", value=10.0)
        o.fields.set_type(field_type="angle")
        o.fields.add(y=0)
        o.fields.add(y=10)
        o.wavelengths.add(value=0.55, is_primary=True)
        o.surfaces.add(index=0, thickness=be.inf)
        o.surfaces.add(index=1, radius=100, thickness=5, material="N-BK7", is_stop=True)
        o.surfaces.add(index=2, radius=-100, thickness=40)
        o.surfaces.add(index=3, radius=7.8, thickness=3.6, material="N-BK7")
        o.surfaces.add(index=4, radius=-6.0, thickness=16.6)
        o.surfaces.add(index=5, radius=be.inf)
        return o

    def test_add_surface_count(self, set_test_backend):
        """Combined system has the correct number of surfaces."""
        relay = self._make_relay()
        eye = self._make_eye()
        combined = relay.surfaces + eye.surfaces
        # relay has 4 surfaces (drop last=1), eye has 4 surfaces (drop first=1)
        assert len(combined) == 6

    def test_add_surface_positions_match_monolithic(self, set_test_backend):
        """z-positions of the combined group match the monolithic system."""
        relay = self._make_relay()
        eye = self._make_eye()
        mono = self._make_monolithic()
        combined = relay.surfaces + eye.surfaces

        for i, (cs, ms) in enumerate(zip(combined.surfaces, mono.surfaces.surfaces)):
            assert_allclose(
                cs.geometry.cs.z,
                ms.geometry.cs.z,
            ), f"z mismatch at surface {i}"

    def test_add_surface_radii_match_monolithic(self, set_test_backend):
        """Radii of the combined group match the monolithic system."""
        relay = self._make_relay()
        eye = self._make_eye()
        mono = self._make_monolithic()
        combined = relay.surfaces + eye.surfaces

        assert_allclose(combined.radii, mono.surfaces.radii)

    def test_add_stop_preserved_from_self(self, set_test_backend):
        """Stop surface comes from self (relay), not from other (eye)."""
        relay = self._make_relay()
        eye = self._make_eye()
        # Give the eye a stop on one of its surfaces before composition.
        eye.surfaces[1].is_stop = True
        combined = relay.surfaces + eye.surfaces
        assert combined.stop_index == 1  # relay's stop at index 1

    def test_add_does_not_mutate_other_z_positions(self, set_test_backend):
        """other's surface z-positions are unchanged after __add__."""
        relay = self._make_relay()
        eye = self._make_eye()
        z_before = [float(s.geometry.cs.z) for s in eye.surfaces]
        _ = relay.surfaces + eye.surfaces
        z_after = [float(s.geometry.cs.z) for s in eye.surfaces]
        assert z_before == z_after, "eye surfaces were mutated by __add__"

    def test_add_does_not_mutate_other_stop_flags(self, set_test_backend):
        """other's is_stop flags are unchanged after __add__."""
        relay = self._make_relay()
        eye = self._make_eye()
        eye.surfaces[1].is_stop = True
        stops_before = [s.is_stop for s in eye.surfaces]
        _ = relay.surfaces + eye.surfaces
        stops_after = [s.is_stop for s in eye.surfaces]
        assert stops_before == stops_after, "eye stop flags were mutated by __add__"

    def test_add_combined_surfaces_independent_of_other(self, set_test_backend):
        """Modifying other after __add__ does not affect the combined group."""
        relay = self._make_relay()
        eye = self._make_eye()
        combined = relay.surfaces + eye.surfaces

        # Change a z-position in eye post-composition
        original_z = float(combined.surfaces[3].geometry.cs.z)
        eye.surfaces[1].geometry.cs.z = be.array(999.0)

        assert_allclose(
            combined.surfaces[3].geometry.cs.z,
            be.array(original_z),
        ), "combined group shares surface objects with other"

    def test_add_junction_uses_last_surface_thickness(self, set_test_backend):
        """Junction z = last_relay_z + last_relay_thickness, not just last_relay_z."""
        relay = optic.Optic()
        relay.set_aperture(aperture_type="EPD", value=10.0)
        relay.fields.set_type(field_type="angle")
        relay.fields.add(y=0)
        relay.wavelengths.add(value=0.55, is_primary=True)
        relay.surfaces.add(index=0, thickness=be.inf)
        relay.surfaces.add(index=1, radius=100, thickness=5, material="N-BK7", is_stop=True)
        # Last surface has thickness=40 and NO explicit image plane.
        relay.surfaces.add(index=2, radius=-100, thickness=40)

        eye = optic.Optic()
        eye.surfaces.add(index=0, thickness=0)  # object at junction
        eye.surfaces.add(index=1, radius=7.8, thickness=3.6, material="N-BK7")
        eye.surfaces.add(index=2, radius=be.inf)

        combined = relay.surfaces + eye.surfaces

        # relay[2] (dropped, z=5, thickness=40) -> junction_z = 5+40 = 45
        # eye[1] should land at z=45
        expected_cornea_z = 5.0 + 40.0
        assert_allclose(
            combined.surfaces[2].geometry.cs.z,
            be.array(expected_cornea_z),
        )

    def test_optic_add_spot_matches_monolithic(self, set_test_backend):
        """Optic.__add__ produces the same on-axis spot RMS as the monolithic system."""
        from optiland.analysis import SpotDiagram

        relay = self._make_relay()
        eye = self._make_eye()
        combined = relay + eye
        mono = self._make_monolithic()

        sc = SpotDiagram(combined, num_rings=3)
        sm = SpotDiagram(mono, num_rings=3)

        rms_c = sc.rms_spot_radius()
        rms_m = sm.rms_spot_radius()

        # On-axis field (index 0) must match
        assert_allclose(rms_c[0][0], rms_m[0][0], rtol=1e-6)

    def test_optic_add_fields_preserved(self, set_test_backend):
        """Optic.__add__ preserves field angles from the left-hand operand."""
        relay = self._make_relay()
        eye = self._make_eye()
        combined = relay + eye

        relay_coords = relay.fields.get_field_coords()
        combined_coords = combined.fields.get_field_coords()
        assert combined_coords == relay_coords

    def test_optic_add_aperture_preserved(self, set_test_backend):
        """Optic.__add__ preserves the aperture from the left-hand operand."""
        relay = self._make_relay()
        eye = self._make_eye()
        combined = relay + eye
        assert_allclose(
            be.array(combined.paraxial.EPD()),
            be.array(relay.paraxial.EPD()),
        )

    def test_optic_add_does_not_mutate_other(self, set_test_backend):
        """Optic.__add__ leaves the right-hand Optic completely unchanged."""
        relay = self._make_relay()
        eye = self._make_eye()
        eye.surfaces[1].is_stop = True
        z_before = [float(s.geometry.cs.z) for s in eye.surfaces]
        stops_before = [s.is_stop for s in eye.surfaces]

        _ = relay + eye

        z_after = [float(s.geometry.cs.z) for s in eye.surfaces]
        stops_after = [s.is_stop for s in eye.surfaces]
        assert z_before == z_after
        assert stops_before == stops_after

    def test_optic_add_combined_surfaces_independent_of_other(self, set_test_backend):
        """Mutating other after Optic.__add__ does not affect the combined system."""
        relay = self._make_relay()
        eye = self._make_eye()
        combined = relay + eye

        original_z = float(combined.surfaces[3].geometry.cs.z)
        eye.surfaces[1].geometry.cs.z = be.array(9999.0)

        assert_allclose(
            combined.surfaces[3].geometry.cs.z,
            be.array(original_z),
        )
