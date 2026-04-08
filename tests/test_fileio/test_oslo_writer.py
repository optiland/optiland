"""Tests for the OSLO writer path."""

from __future__ import annotations

import os

import pytest

from optiland.fileio import load_oslo_file, save_oslo_file
from optiland.optic import Optic
from tests.utils import assert_allclose


@pytest.fixture
def oslo_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "oslo", "cox3_07.len")


class TestOsloWriter:
    def test_round_trip_cox3(self, oslo_file, tmp_path):
        # 1. Load original
        optic_orig = load_oslo_file(oslo_file)

        # 2. Save to temp
        out_path = os.path.join(tmp_path, "test_out.len")
        save_oslo_file(optic_orig, out_path)

        # 3. Load back
        optic_new = load_oslo_file(out_path)

        # 4. Verify
        assert optic_new.name == optic_orig.name
        assert len(optic_new.surfaces) == len(optic_orig.surfaces)
        assert_allclose(optic_new.aperture.value, optic_orig.aperture.value)

        for s1, s2 in zip(optic_orig.surfaces, optic_new.surfaces, strict=False):
            assert_allclose(s1.geometry.radius, s2.geometry.radius)
            assert_allclose(s1.thickness, s2.thickness)

    def test_write_standard_optic(self, tmp_path):
        optic = Optic("Simple Lens")
        optic.set_aperture("EPD", 10.0)
        optic.wavelengths.add(0.55)
        optic.fields.add(y=0)

        optic.surfaces.add(index=0, radius=0, thickness=1e10)
        optic.surfaces.add(index=1, radius=100.0, thickness=5.0, material="N-BK7", is_stop=True)
        optic.surfaces.add(index=2, radius=-100.0, thickness=50.0)
        optic.surfaces.add(index=3)

        out_path = os.path.join(tmp_path, "simple.len")
        save_oslo_file(optic, out_path)

        assert os.path.exists(out_path)

        # Load back
        optic2 = load_oslo_file(out_path)
        assert len(optic2.surfaces) == 4  # obj + 2 surfaces + image
        assert_allclose(optic2.surfaces[1].geometry.radius, 100.0)
        assert optic2.surfaces[1].is_stop is True
