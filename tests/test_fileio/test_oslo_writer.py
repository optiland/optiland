"""Tests for the OSLO writer path."""

from __future__ import annotations

import os

import pytest

import optiland.backend as be
from optiland.fileio import load_oslo_file, save_oslo_file
from optiland.fileio.oslo.writer.encoder import OpticToOsloEncoder
from optiland.fileio.oslo.writer.formatter import OsloDataFormatter
from optiland.optic import Optic
from tests.utils import assert_allclose


@pytest.fixture
def oslo_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "oslo", "cox3_07.len")


def _make_simple_optic(name: str = "Test", fields: list[float] | None = None) -> Optic:
    """Return a minimal valid optic for writer tests."""
    optic = Optic(name)
    optic.set_aperture("EPD", 10.0)
    optic.wavelengths.add(0.55)
    for y in (fields if fields is not None else [0.0, 14.0, 20.0]):
        optic.fields.add(y=y, x=0.0)
    optic.surfaces.add(index=0, radius=0, thickness=1e10)
    optic.surfaces.add(index=1, radius=100.0, thickness=5.0, material="N-BK7", is_stop=True)
    optic.surfaces.add(index=2, radius=-100.0, thickness=50.0)
    optic.surfaces.add(index=3)
    return optic


class TestBugFixes:
    """Unit tests for each bug fix in the OSLO writer."""

    # BUG-W1: only max field value written
    def test_w1_only_max_field_written(self):
        optic = _make_simple_optic(fields=[0.0, 14.0, 20.0])
        model = OpticToOsloEncoder(optic).encode()
        assert model.fields["y"] == [20.0]

    def test_w1_single_ang_line_in_output(self, tmp_path):
        optic = _make_simple_optic(fields=[0.0, 14.0, 20.0])
        out = tmp_path / "w1.len"
        save_oslo_file(optic, str(out))
        text = out.read_text()
        ang_lines = [ln for ln in text.splitlines() if ln.startswith("ANG ")]
        assert len(ang_lines) == 1
        assert "20" in ang_lines[0]

    # BUG-W2: compact float format (no 12-digit scientific notation)
    def test_w2_compact_float_format(self):
        formatter = OsloDataFormatter.__new__(OsloDataFormatter)
        import math

        formatter.model = None  # not needed for _fmt
        # Use the module-level helper
        from optiland.fileio.oslo.writer.formatter import OsloDataFormatter as F

        fmt = F._fmt
        # Should produce compact notation, not 12-digit E format
        result = fmt(None, 22.01359)
        assert "E" not in result
        assert result == "22.01359"

    def test_w2_output_has_no_12digit_scientific(self, tmp_path):
        optic = _make_simple_optic()
        out = tmp_path / "w2.len"
        save_oslo_file(optic, str(out))
        text = out.read_text()
        import re

        assert not re.search(r"\d\.\d{12}E[+-]\d{2}", text)

    # BUG-W3: LEN NEW uses actual EFL, not 1.0
    def test_w3_len_new_has_real_efl(self, tmp_path):
        from optiland.samples.simple import CementedAchromat

        optic = CementedAchromat()
        out = tmp_path / "w3.len"
        save_oslo_file(optic, str(out))
        first_lines = out.read_text().splitlines()
        len_line = next(ln for ln in first_lines if ln.startswith("LEN NEW"))
        # Extract scaling token
        tokens = len_line.split()
        scaling = float(tokens[3])
        assert scaling != pytest.approx(1.0)  # should be real EFL, not fallback

    # BUG-W4: DES and UNI always emitted
    def test_w4_des_and_uni_in_output(self, tmp_path):
        optic = _make_simple_optic()
        out = tmp_path / "w4.len"
        save_oslo_file(optic, str(out))
        text = out.read_text()
        assert 'DES "Optiland"' in text
        assert "UNI 1" in text

    # BUG-W5: object surface AP sentinel for infinite-conjugate system
    def test_w5_object_surface_ap_sentinel(self, tmp_path):
        optic = _make_simple_optic(fields=[0.0, 14.0, 20.0])
        out = tmp_path / "w5.len"
        save_oslo_file(optic, str(out))
        optic2 = load_oslo_file(str(out))
        # The AP sentinel should be large (>=1e9) and absorbed as infinite AP by reader
        model = OpticToOsloEncoder(optic).encode()
        assert "AP" in model.surfaces[0]
        assert model.surfaces[0]["AP"] > 1e8

    def test_w5_object_ap_sentinel_not_applied_as_physical_aperture(self, tmp_path):
        optic = _make_simple_optic(fields=[0.0, 14.0, 20.0])
        out = tmp_path / "w5.len"
        save_oslo_file(optic, str(out))
        optic2 = load_oslo_file(str(out))
        # Reader must NOT apply sentinel as a RadialAperture (AP < 1e6 check)
        assert optic2.surfaces[0].aperture is None

    # Stop AP derived from paraxial EPD when no explicit RadialAperture is set
    def test_stop_ap_written_when_no_physical_aperture(self, tmp_path):
        from optiland.samples.simple import CementedAchromat

        optic = CementedAchromat()
        # Stop is surf 1, no explicit RadialAperture
        assert optic.surfaces[1].aperture is None
        model = OpticToOsloEncoder(optic).encode()
        stop_data = model.surfaces[1]
        assert "AP" in stop_data
        expected_r = float(optic.paraxial.EPD()) / 2.0
        assert_allclose(stop_data["AP"], expected_r)

    # W6: NXT lines must be plain "NXT" with no inline comment so that
    # OSLO EDU can parse the file without errors.
    def test_w6_nxt_lines_are_clean(self, tmp_path):
        optic = _make_simple_optic()
        out = tmp_path / "w6.len"
        save_oslo_file(optic, str(out))
        nxt_lines = [ln for ln in out.read_text().splitlines() if "NXT" in ln]
        assert nxt_lines, "expected at least one NXT line"
        for ln in nxt_lines:
            assert ln.strip() == "NXT", f"NXT line must be bare; got: {ln!r}"


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
