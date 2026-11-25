"""Tests for configurations derived from another configuration."""

from __future__ import annotations

from optiland.materials import IdealMaterial
from optiland.multiconfig import MultiConfiguration
from optiland.samples import CookeTriplet
from tests.multiconfig.utils import deepcopy_safe

MATERIAL_PATH = "surfaces.surfaces[3].material_post"
STANDARD_ATTRS = ("radius", "conic", "thickness")


def _driver_key(pickup):
    """Return the attribute a pickup drives on its target optic."""
    if pickup.attr_type in STANDARD_ATTRS:
        return (pickup.target_surface_idx, pickup.attr_type)
    return pickup.attr_type


def _chained_with_material():
    """Config 1 from config 0 with a material link, config 2 from config 1."""
    with deepcopy_safe():
        mc = MultiConfiguration(CookeTriplet())
        mc.add_configuration()
        mc.set_material(3, IdealMaterial(n=1.62), configurations="all")
        mc.add_configuration(1)
    return mc


def test_chained_configuration_has_no_phantom_source(set_test_backend):
    mc = _chained_with_material()
    configs = {id(optic) for optic in mc.configurations}
    for optic in mc.configurations:
        for pickup in optic.pickups.pickups:
            assert id(pickup.source_optic) in configs


def test_chained_configuration_links_to_its_source(set_test_backend):
    mc = _chained_with_material()
    config_1, config_2 = mc.configurations[1], mc.configurations[2]
    for pickup in config_2.pickups.pickups:
        if pickup.attr_type in STANDARD_ATTRS:
            assert pickup.source_optic is config_1
        else:
            assert pickup.attr_type == MATERIAL_PATH
            assert pickup.source_optic is mc.configurations[0]


def test_chained_configuration_has_one_driver_per_attribute(set_test_backend):
    mc = _chained_with_material()
    mc.set_material(3, IdealMaterial(n=1.7), configurations="all")
    for optic in mc.configurations:
        keys = [_driver_key(p) for p in optic.pickups.pickups]
        assert len(keys) == len(set(keys))


def test_material_on_config_0_reaches_chained_config(set_test_backend):
    mc = _chained_with_material()
    # A direct change on config 0 reaches config 2 through its inherited link.
    material = IdealMaterial(n=1.7)
    mc.configurations[0].updater.set_material(material, 3)
    mc.configurations[2].updater.update()
    assert mc.configurations[2].surfaces[3].material_post is material

    other = IdealMaterial(n=1.75)
    mc.set_material(3, other, configurations="all")
    mc.configurations[2].updater.update()
    assert mc.configurations[2].surfaces[3].material_post is other
