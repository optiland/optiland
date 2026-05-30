"""Tests for the OSLO reader path."""

from __future__ import annotations

import os

import pytest

import optiland.backend as be
from optiland.fileio import load_oslo_file
from optiland.fileio.oslo.reader.converter import OsloToOpticConverter
from optiland.fileio.oslo.reader.parser import OsloDataParser
from optiland.materials import AbbeMaterial, Material
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

    # ISSUE-P1: DES command stored in notes (quotes are stripped by implementation)
    def test_issue_p1_des_stored_in_notes(self):
        parser = OsloDataParser("dummy")
        parser._read_des(["DES", '"OpticDesigner"'])
        assert parser.data_model.notes["DES"] == "OpticDesigner"

    def test_issue_p1_des_in_dispatch_table(self):
        parser = OsloDataParser("dummy")
        assert "DES" in parser._dispatch_table


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

    # BUG-R1: PY solve places image at paraxial focus (F2 ≈ 0)
    def test_bug_r1_py_solve_places_image_at_focus(self, oslo_file):
        optic = load_oslo_file(oslo_file)
        f2 = float(optic.paraxial.F2())
        assert abs(f2) < 1e-6, f"F2 should be ~0 after PY solve, got {f2}"


@pytest.fixture
def fallback_glass_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "oslo", "test_glass_fallback.len")


class TestGlassFallback:
    """Glass name resolution: built-in catalog fallback and prefix stripping."""

    def test_lakn6_resolves_to_abbe_material(self, fallback_glass_file):
        optic = load_oslo_file(fallback_glass_file)
        mat = optic.surfaces[1].material_post
        assert isinstance(mat, AbbeMaterial), (
            f"LAKN6 should resolve to AbbeMaterial; got {type(mat).__name__}"
        )
        assert_allclose(float(mat.n(0.58756).item()), 1.6400, atol=1e-3)

    def test_sf18_resolves_to_abbe_material(self, fallback_glass_file):
        optic = load_oslo_file(fallback_glass_file)
        mat = optic.surfaces[2].material_post
        assert isinstance(mat, AbbeMaterial), (
            f"SF18 should resolve to AbbeMaterial; got {type(mat).__name__}"
        )
        assert_allclose(float(mat.n(0.58756).item()), 1.7215, atol=1e-3)

    def test_h_laf2_resolves_via_prefix_strip(self, fallback_glass_file):
        optic = load_oslo_file(fallback_glass_file)
        mat = optic.surfaces[4].material_post
        # H_LAF2 → strip H_ → LAF2 → found as Material or AbbeMaterial
        assert isinstance(mat, (Material, AbbeMaterial)), (
            f"H_LAF2 should resolve to a real glass; got {type(mat).__name__}"
        )
        nd = float(mat.n(0.58756).item())
        assert nd > 1.5, f"H_LAF2 nd should be >1.5 (lanthanum flint); got {nd:.4f}"

    def test_no_glass_silently_becomes_air(self, fallback_glass_file):
        optic = load_oslo_file(fallback_glass_file)
        for i, surf in enumerate(optic.surfaces):
            mat = surf.material_post
            nd = float(mat.n(0.58756).item())
            # None of the glass surfaces (1, 2, 4) should resolve to n=1.0 (air)
            if i in (1, 2, 4):
                assert nd > 1.1, (
                    f"Surface {i} glass silently became air (n={nd:.4f})"
                )


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
