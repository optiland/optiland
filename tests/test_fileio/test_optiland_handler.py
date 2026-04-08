"""Tests for the Optiland JSON file handler.

Migrated from tests/test_fileio.py.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

import optiland.backend as be
from optiland.fileio import load_optiland_file, save_optiland_file
from optiland.fileio.optiland_handler import load_obj_from_json, save_obj_to_json
from optiland.materials import Material
from optiland.optic import Optic
from optiland.samples.objectives import HeliarLens
from tests.utils import assert_allclose


def test_save_load_json_obj():
    mat = Material("SF11")
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as tmp:
        save_obj_to_json(mat, tmp.name)
    assert os.path.exists(tmp.name)
    mat2 = load_obj_from_json(Material, tmp.name)
    assert mat.to_dict() == mat2.to_dict()


def test_load_invalid_json():
    with pytest.raises(FileNotFoundError):
        load_obj_from_json(Material, "non_existent_file.json")


def test_save_load_optiland_file():
    lens = HeliarLens()
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as tmp:
        save_optiland_file(lens, tmp.name)
        lens2 = load_optiland_file(tmp.name)
    assert lens.to_dict() == lens2.to_dict()


def test_load_legacy_optiland_file_with_field_type():
    from optiland.fields import AngleField

    lens = HeliarLens()
    lens.fields.set_type("angle")
    modern_dict = lens.to_dict()

    legacy_dict = lens.to_dict()
    legacy_dict["fields"]["field_type"] = "angle"
    del legacy_dict["fields"]["field_definition"]

    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as tmp:
        json.dump(legacy_dict, tmp)
        filepath = tmp.name

    loaded = load_optiland_file(filepath)
    assert isinstance(loaded.fields.field_definition, AngleField)
    assert modern_dict == loaded.to_dict()
    os.remove(filepath)


def test_save_load_optiland_file_with_tensor(set_test_backend):
    try:
        import torch

        tensor_val = torch.tensor(1.23)
        has_torch = True
    except ImportError:
        has_torch = False
        tensor_val = 1.23

    lens = HeliarLens()
    lens.surfaces.add(
        index=2,
        surface_type="even_asphere",
        radius=10.0,
        conic=-1.0,
        coefficients=[1.23, 1.23],
    )

    if has_torch:
        lens.surfaces[1].thickness = torch.tensor(1.23)
        lens.surfaces[1].geometry.radius = torch.tensor(1.23)
        lens.surfaces[2].geometry.coefficients = [
            torch.tensor(1.23),
            torch.tensor(1.23),
        ]
    else:

        class MockTensor:
            def __init__(self, val):
                self.val = val

            def tolist(self):
                return self.val if isinstance(self.val, list) else [self.val]

            def item(self):
                return self.val

        lens.surfaces[1].thickness = MockTensor(1.23)
        lens.surfaces[1].geometry.radius = MockTensor(1.23)
        lens.surfaces[2].geometry.coefficients = [MockTensor(1.23), MockTensor(1.23)]

    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json") as tmp:
        save_optiland_file(lens, tmp.name)
        lens2 = load_optiland_file(tmp.name)

    assert abs(lens2.surfaces[1].thickness - 1.23) < 1e-6
    assert abs(lens2.surfaces[1].geometry.radius - 1.23) < 1e-6
    assert abs(lens2.surfaces[2].geometry.coefficients[0] - 1.23) < 1e-6
    os.remove(tmp.name)


def test_remove_surface_after_load(set_test_backend, tmp_path):
    lens = Optic(name="TestLens")
    lens.surfaces.add(index=0, thickness=be.inf, material="Air")
    lens.surfaces.add(
        index=1, surface_type="standard", material="Air", thickness=10, radius=150
    )
    lens.surfaces.add(
        index=2,
        surface_type="standard",
        material="N-BK7",
        thickness=10,
        radius=150,
        is_stop=True,
    )
    lens.surfaces.add(
        index=3, surface_type="standard", material="Air", thickness=20, radius=be.inf
    )
    lens.surfaces.add(index=4)
    lens.set_aperture("float_by_stop_size", 25)

    filepath = tmp_path / "lens.json"
    save_optiland_file(lens, filepath)
    loaded = load_optiland_file(filepath)
    loaded.surfaces.remove(1)

    positions = loaded.surfaces.positions.flatten()
    expected = be.array([0.0, 10.0, 30.0])
    assert_allclose(positions[1:], expected)


def test_zemax_to_optic_converter_shim():
    from optiland.fileio import ZemaxToOpticConverter, _make_deprecated_ztoc

    with pytest.warns(
        DeprecationWarning,
        match="Import ZemaxToOpticConverter from optiland.fileio.zemax",
    ):
        conv = _make_deprecated_ztoc({})
        assert isinstance(conv, ZemaxToOpticConverter)
