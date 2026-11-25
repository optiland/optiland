"""Tests for saving and loading a MultiConfiguration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import optiland.backend as be
from optiland.fileio import load_obj_from_json, save_obj_to_json
from optiland.materials import IdealMaterial
from optiland.multiconfig import MultiConfiguration
from optiland.samples import CookeTriplet
from tests.multiconfig.utils import deepcopy_safe
from tests.utils import assert_allclose

if TYPE_CHECKING:
    from optiland.optic import Optic


def _topology(mc: MultiConfiguration) -> list[tuple]:
    """Describe every pickup with its source configuration index."""
    index_of = {id(optic): i for i, optic in enumerate(mc.configurations)}
    topology = []
    for config_idx, optic in enumerate(mc.configurations):
        for pickup in optic.pickups.pickups:
            assert pickup.optic is optic
            topology.append(
                (
                    config_idx,
                    index_of[id(pickup.source_optic)],
                    pickup.attr_type,
                    pickup.source_surface_idx,
                    pickup.target_surface_idx,
                    float(be.to_numpy(pickup.scale)),
                    float(be.to_numpy(pickup.offset)),
                )
            )
    return topology


def _zoom():
    """A 3-configuration zoom with shared and independent parameters."""
    with deepcopy_safe():
        mc = MultiConfiguration(CookeTriplet())
        mc.add_configuration()
        mc.add_configuration(1)
    mc.set_radius(1, 22.5, configurations="all")
    mc.set_material(3, IdealMaterial(n=1.62), configurations="all")
    mc.set_optic_property("name", "zoom", configurations="all")
    for config_idx, (gap, back) in enumerate(
        [(6.00755, 42.20778), (5.0, 43.0), (4.0, 44.5)]
    ):
        mc.set_thickness(2, gap, configurations=[config_idx])
        mc.set_thickness(6, back, configurations=[config_idx])
    for optic in mc.configurations:
        optic.updater.update()
    return mc


def _trace(optic: Optic):
    rays = optic.trace_generic(
        Hx=0.0, Hy=0.7, Px=0.0, Py=0.5, wavelength=optic.primary_wavelength
    )
    return rays.x, rays.y, rays.z, rays.L, rays.M, rays.N


def _assert_same_systems(mc_a: MultiConfiguration, mc_b: MultiConfiguration):
    assert len(mc_a.configurations) == len(mc_b.configurations)
    assert _topology(mc_a) == _topology(mc_b)
    for optic_a, optic_b in zip(mc_a.configurations, mc_b.configurations, strict=True):
        assert_allclose(optic_b.paraxial.f2(), optic_a.paraxial.f2())
        for value_a, value_b in zip(_trace(optic_a), _trace(optic_b), strict=True):
            assert_allclose(value_b, value_a)


def test_to_dict_lists_cross_configuration_links(set_test_backend):
    mc = _zoom()
    data = mc.to_dict()
    assert data["version"] == 1
    assert len(data["configurations"]) == 3
    expected = [
        [config_idx, pickup_idx, mc.configurations.index(pickup.source_optic)]
        for config_idx, optic in enumerate(mc.configurations)
        for pickup_idx, pickup in enumerate(optic.pickups.pickups)
        if pickup.source_optic is not optic
    ]
    assert data["links"] == expected
    assert {link[2] for link in data["links"] if link[0] == 2} == {0, 1}


def test_to_dict_rejects_link_outside_multiconfiguration(set_test_backend):
    mc = MultiConfiguration(CookeTriplet())
    mc.add_configuration()
    mc.configurations[1].pickups.add(
        source_surface_idx=1,
        attr_type="radius",
        target_surface_idx=1,
        source_optic=CookeTriplet(),
    )
    with pytest.raises(ValueError, match="not part of this MultiConfiguration"):
        mc.to_dict()


def test_round_trip_dict(set_test_backend):
    mc = _zoom()
    loaded = MultiConfiguration.from_dict(mc.to_dict())
    _assert_same_systems(mc, loaded)
    # Independent values per configuration survive.
    for config_idx, gap in enumerate([6.00755, 5.0, 4.0]):
        assert_allclose(
            loaded.configurations[config_idx].surfaces.get_thickness(2), gap
        )
    assert loaded.configurations[2].name == "zoom"


def test_round_trip_keeps_self_pickups(set_test_backend):
    optic = CookeTriplet()
    optic.pickups.add(
        source_surface_idx=1, attr_type="radius", target_surface_idx=5, scale=1.0
    )
    mc = MultiConfiguration(optic)
    mc.add_configuration()
    loaded = MultiConfiguration.from_dict(mc.to_dict())
    _assert_same_systems(mc, loaded)
    first = loaded.configurations[0].pickups.pickups[0]
    assert first.source_optic is loaded.configurations[0]


def test_mutation_after_load_propagates(set_test_backend):
    loaded = MultiConfiguration.from_dict(_zoom().to_dict())
    base, config_1, config_2 = loaded.configurations

    base.updater.set_radius(25.0, 1)
    material = IdealMaterial(n=1.7)
    base.updater.set_material(material, 3)
    base.name = "renamed"
    for optic in loaded.configurations:
        optic.updater.update()

    for optic in (config_1, config_2):
        assert_allclose(optic.surfaces[1].geometry.radius, 25.0)
        assert optic.surfaces[3].material_post is material
        assert optic.name == "renamed"
    # Independent thicknesses stay independent.
    assert_allclose(config_1.surfaces.get_thickness(2), 5.0)
    assert_allclose(config_2.surfaces.get_thickness(2), 4.0)


def test_json_file_round_trip(set_test_backend, tmp_path):
    mc = _zoom()
    path = tmp_path / "zoom.json"
    save_obj_to_json(mc, path)
    loaded = load_obj_from_json(MultiConfiguration, path)
    assert isinstance(loaded, MultiConfiguration)
    _assert_same_systems(mc, loaded)


def test_from_dict_rejects_bad_data():
    data = MultiConfiguration(CookeTriplet()).to_dict()
    with pytest.raises(ValueError, match="version"):
        MultiConfiguration.from_dict({**data, "version": 2})
    with pytest.raises(ValueError, match="no configurations"):
        MultiConfiguration.from_dict({**data, "configurations": []})
    with pytest.raises(ValueError, match="configuration outside"):
        MultiConfiguration.from_dict({**data, "links": [[0, 0, 3]]})
    with pytest.raises(ValueError, match="pickup outside"):
        MultiConfiguration.from_dict({**data, "links": [[0, 0, 0]]})
