from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest

from optiland.multiconfig.multi_configuration import MultiConfiguration
from optiland.optic import Optic

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture
def base_optic():
    optic = Optic()
    # Create a simple triplet-like structure
    # Surfaces: 0 (Obj), 1 (Lens1), 2 (Air), 3 (Lens2), 4 (Img)
    optic.surfaces.add(radius=100, thickness=10, material="air", index=0)
    optic.surfaces.add(radius=-100, thickness=5, material="air", index=1)
    optic.surfaces.add(radius=50, thickness=10, material="air", index=2)
    optic.surfaces.add(radius=-50, thickness=0, material="air", index=3)  # Image
    return optic


def test_set_global_radius(base_optic):
    """Test setting a global radius updates all linked configs."""
    mc = MultiConfiguration(base_optic)
    mc.add_configuration()

    mc.set_radius(1, 200, configurations="all")
    mc.current_config(0).updater.update()
    mc.current_config(1).updater.update()

    assert mc.current_config(0).surfaces[1].geometry.radius == 200
    assert mc.current_config(1).surfaces[1].geometry.radius == 200


def test_set_unique_radius(base_optic):
    """Test setting a unique radius breaks the link."""
    mc = MultiConfiguration(base_optic)
    mc.add_configuration()

    # Set unique on config 1
    mc.set_radius(1, 300, configurations=[1])
    mc.current_config(0).updater.update()
    mc.current_config(1).updater.update()

    assert mc.current_config(0).surfaces[1].geometry.radius == -100  # Original
    assert mc.current_config(1).surfaces[1].geometry.radius == 300

    # Modify global, config 1 should NOT change
    mc.set_radius(1, 400, configurations=[0])
    mc.current_config(0).updater.update()
    mc.current_config(1).updater.update()

    assert mc.current_config(0).surfaces[1].geometry.radius == 400
    assert mc.current_config(1).surfaces[1].geometry.radius == 300


def test_relink_radius(base_optic):
    """Test that setting configurations='all' relinks unique configs."""
    mc = MultiConfiguration(base_optic)
    mc.add_configuration()

    # Make unique
    mc.set_radius(1, 300, configurations=[1])
    assert mc.current_config(1).surfaces[1].geometry.radius == 300

    # Set global 'all', should overwrite unique
    mc.set_radius(1, 500, configurations="all")
    mc.current_config(0).updater.update()
    mc.current_config(1).updater.update()

    assert mc.current_config(0).surfaces[1].geometry.radius == 500
    assert mc.current_config(1).surfaces[1].geometry.radius == 500

    # Update global again, check link persists
    mc.set_radius(1, 600, configurations=[0])  # Only set 0, but 1 is linked now
    mc.current_config(0).updater.update()
    mc.current_config(1).updater.update()

    assert mc.current_config(0).surfaces[1].geometry.radius == 600
    assert mc.current_config(1).surfaces[1].geometry.radius == 600


def test_set_generic_property(base_optic):
    """Test setting a generic property (e.g. asphere coefficient)."""
    # Use asphere coefficient on surface 1 (need to change type first?)
    # Or just use an existing property like 'material_post.n_p'? No, read-only mostly.
    # Let's use 'thickness' via generic path for test, or assign a dummy var?
    # Better: set 'geometry.radius' via generic path to prove generic logic works.

    mc = MultiConfiguration(base_optic)
    mc.add_configuration()

    # Set global generic
    mc.set_surface_property(1, "geometry.radius", 250, configurations="all")
    mc.current_config(0).updater.update()
    mc.current_config(1).updater.update()

    assert mc.current_config(0).surfaces[1].geometry.radius == 250
    assert mc.current_config(1).surfaces[1].geometry.radius == 250

    # Confirm pickup was created (since radius has standard pickup, but we used generic path)
    # Generic path 'surface_group.surfaces[1].geometry.radius'
    # Check Pickups
    # Note: `set_property` maps "radius" alias, but here we passed "geometry.radius" explicitly.
    # So it used _ensure_generic_pickup.

    has_generic_pickup = False
    target_path = "surfaces.surfaces[1].geometry.radius"
    for p in mc.current_config(1).pickups.pickups:
        if p.attr_type == target_path:
            has_generic_pickup = True
            break
    assert has_generic_pickup


def test_set_optic_property(base_optic):
    """Test setting an optic-level property."""
    mc = MultiConfiguration(base_optic)
    mc.add_configuration()

    # Set global field type? Or maybe just 'name'
    mc.set_optic_property("name", "TestOptic", configurations="all")

    assert mc.current_config(0).name == "TestOptic"
    # Config 1 initially copied name?
    # Generic pickup should handle future updates.

    mc.set_optic_property("name", "NewName", configurations="all")
    mc.current_config(0).updater.update()
    mc.current_config(1).updater.update()

    # Pickups apply on update().
    # Generic pickup sets attr on config 1 from config 0.
    assert mc.current_config(0).name == "NewName"
    assert mc.current_config(1).name == "NewName"


def test_multiconfig_draw(base_optic):
    """Test that the draw method runs and returns a figure."""
    mc = MultiConfiguration(base_optic)

    # Setup optic for drawing (needs aperture/fields)
    base_optic.set_aperture("EPD", 20)
    base_optic.fields.set_type("angle")
    base_optic.fields.add(0)
    base_optic.wavelengths.add(0.55, is_primary=True)
    base_optic.surfaces[1].is_stop = True

    mc.add_configuration()

    fig, axes = mc.draw(figsize=(5, 2))

    assert fig is not None
    assert len(axes) == 2

    plt.close(fig)
