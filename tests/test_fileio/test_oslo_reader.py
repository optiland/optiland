"""Tests for the OSLO reader path."""

from __future__ import annotations

import os

import pytest

import optiland.backend as be
from optiland.fileio import load_oslo_file
from optiland.fileio.oslo.reader.parser import OsloDataParser
from optiland.optic import Optic
from tests.utils import assert_allclose


@pytest.fixture
def oslo_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "oslo", "cox3_07.len")


class TestOsloDataParser:
    def test_read_len(self):
        parser = OsloDataParser("dummy")
        parser._read_len(["LEN", "NEW", '"TEST"', "1.0", "5"])
        assert parser.data_model.name == "TEST"
        assert parser.data_model.scaling == 1.0
        assert parser.data_model.num_surfaces == 5

    def test_read_ebr(self):
        parser = OsloDataParser("dummy")
        parser._read_ebr(["EBR", "5.0"])
        assert parser.data_model.aperture["EPD"] == 10.0

    def test_read_rd(self):
        parser = OsloDataParser("dummy")
        parser._read_rd(["RD", "50.0"])
        assert parser._current_surf_data["RD"] == 50.0


class TestOsloReader:
    def test_load_oslo_file(self, oslo_file):
        optic = load_oslo_file(oslo_file)
        assert isinstance(optic, Optic)
        assert optic.name == "COX PROBLEM 3-07"
        assert len(optic.surfaces) == 13  # 12 + object

        # Check some surface data
        # Surface 0 (Object) is infinite
        assert be.isinf(optic.surfaces[0].thickness)

        # Surface 2 has glass SF5
        assert optic.surfaces[2].material_post.name.upper() == "SF5"

    def test_load_oslo_file_aperture(self, oslo_file):
        optic = load_oslo_file(oslo_file)
        # EBR 0.33 -> EPD 0.66
        assert_allclose(optic.aperture.value, 0.66)


if __name__ == "__main__":
    t = TestOsloDataParser()
    t.test_read_len()
    print("Parser test passed.")

    tr = TestOsloReader()
    # Mocking oslo_file fixture path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    oslo_path = os.path.join(current_dir, "oslo", "cox3_07.len")
    tr.test_load_oslo_file(oslo_path)
    print("Reader test passed.")
