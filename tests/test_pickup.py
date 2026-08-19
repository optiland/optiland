from __future__ import annotations

import pytest

from optiland.pickup import Pickup, PickupManager
from optiland.samples.objectives import CookeTriplet


class TestPickup:
    def test_apply_radius(self, set_test_backend):
        lens = CookeTriplet()
        pickup = Pickup(lens, 1, "radius", 2, scale=2, offset=3)
        pickup.apply()
        r0 = lens.surfaces.radii[1]
        r1 = lens.surfaces.radii[2]
        assert r1 == 2 * r0 + 3

    def test_apply_conic(self, set_test_backend):
        lens = CookeTriplet()
        pickup = Pickup(lens, 1, "conic", 2, scale=2, offset=3)
        pickup.apply()
        k0 = lens.surfaces.conic[1]
        k1 = lens.surfaces.conic[2]
        assert k1 == 2 * k0 + 3

    def test_apply_thickness(self, set_test_backend):
        lens = CookeTriplet()
        pickup = Pickup(lens, 1, "thickness", 2, scale=1, offset=0)
        pickup.apply()
        t1 = lens.surfaces.get_thickness(1)
        t2 = lens.surfaces.get_thickness(2)
        assert t1 == t2

    def test_invalid_attr_type(self, set_test_backend):
        lens = CookeTriplet()
        pickup = Pickup(lens, 1, "invalid", 2, scale=1, offset=0)
        with pytest.raises(ValueError):
            pickup.apply()

    def test_apply_generic_coefficients(self, set_test_backend):
        lens = CookeTriplet()
        # Ensure it has aspheric coefficients
        lens.surfaces.add(
            index=1,
            radius=50,
            thickness=3,
            surface_type="even_asphere",
            coefficients=[-2e-4, -4e-6],
        )
        lens.surfaces.add(
            index=2,
            radius=-50,
            thickness=30,
            surface_type="even_asphere",
            coefficients=[0, 0],
        )
        pickup = Pickup(
            lens,
            1,
            "surfaces.surfaces[i].geometry.coefficients",
            2,
            scale=1,
            offset=0,
        )
        pickup.apply()
        assert lens.surfaces[2].geometry.coefficients == [-2e-4, -4e-6]

    def test_invalid_set_attr(self, set_test_backend):
        lens = CookeTriplet()
        pickup = Pickup(lens, 1, "invalid", 2, scale=1, offset=0)
        with pytest.raises(ValueError):
            pickup._set_value(5.0)

    def test_to_dict(self, set_test_backend):
        lens = CookeTriplet()
        pickup = Pickup(lens, 1, "radius", 2, scale=2, offset=3)
        d = pickup.to_dict()
        assert d["source_surface_idx"] == 1
        assert d["attr_type"] == "radius"
        assert d["target_surface_idx"] == 2
        assert d["scale"] == 2
        assert d["offset"] == 3

    def test_from_dict(self, set_test_backend):
        lens = CookeTriplet()
        d = {
            "source_surface_idx": 1,
            "attr_type": "radius",
            "target_surface_idx": 2,
            "scale": 2,
            "offset": 3,
        }
        pickup = Pickup.from_dict(lens, d)
        assert pickup.source_surface_idx == 1
        assert pickup.attr_type == "radius"
        assert pickup.target_surface_idx == 2
        assert pickup.scale == 2
        assert pickup.offset == 3


class TestPickupManager:
    def test_add_pickup(self, set_test_backend):
        lens = CookeTriplet()
        manager = PickupManager(lens)
        manager.add(1, "radius", 2, scale=2, offset=3)
        assert len(manager) == 1

    def test_apply_pickups(self, set_test_backend):
        lens = CookeTriplet()
        manager = PickupManager(lens)
        manager.add(1, "radius", 2, scale=2, offset=3)
        manager.apply()
        r0 = lens.surfaces[1].geometry.radius
        r1 = lens.surfaces[2].geometry.radius
        assert r1 == 2 * r0 + 3

    def test_clear_pickups(self, set_test_backend):
        lens = CookeTriplet()
        manager = PickupManager(lens)
        manager.add(1, "radius", 2, scale=2, offset=3)
        manager.clear()
        assert len(manager) == 0

    def test_to_dict(self, set_test_backend):
        lens = CookeTriplet()
        manager = PickupManager(lens)
        manager.add(1, "radius", 2, scale=2, offset=3)
        d = manager.to_dict()
        assert len(d) == 1
        assert d[0]["source_surface_idx"] == 1
        assert d[0]["attr_type"] == "radius"
        assert d[0]["target_surface_idx"] == 2
        assert d[0]["scale"] == 2
        assert d[0]["offset"] == 3

    def test_from_dict(self, set_test_backend):
        lens = CookeTriplet()
        d = [
            {
                "source_surface_idx": 1,
                "attr_type": "radius",
                "target_surface_idx": 2,
                "scale": 2,
                "offset": 3,
            },
        ]
        manager = PickupManager.from_dict(lens, d)
        assert len(manager) == 1
        assert manager.pickups[0].source_surface_idx == 1
        assert manager.pickups[0].attr_type == "radius"
        assert manager.pickups[0].target_surface_idx == 2
        assert manager.pickups[0].scale == 2
        assert manager.pickups[0].offset == 3
