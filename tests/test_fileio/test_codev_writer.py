"""Tests for the CODE V writer path.

Covers OpticToCodeVConverter, CodeVFileEncoder, CodeVWriter,
and the save_codev_file public entry point.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from optiland.fileio import load_codev_file, save_codev_file
from optiland.fileio.codev.model import CodeVDataModel
from optiland.fileio.codev.writer.encoder import CodeVFileEncoder
from optiland.fileio.codev.writer.exporter import CodeVWriter
from optiland.fileio.codev.writer.formatter import OpticToCodeVConverter
from optiland.optic import Optic
from tests.utils import assert_allclose

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codev_files")


def _seq(name: str) -> str:
    return os.path.join(_DIR, name)


def _make_singlet() -> Optic:
    """Build a minimal singlet optic for writer tests."""
    from optiland.materials import Material

    optic = Optic()
    optic.surfaces.add(index=0, thickness=1e10)
    optic.surfaces.add(
        index=1,
        radius=50.0,
        thickness=5.0,
        material=Material("N-BK7", "schott"),
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=-50.0, thickness=45.0)
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.fields.add(y=5.0)
    optic.add_wavelength(value=0.5876, is_primary=True)
    optic.add_wavelength(value=0.4861)
    optic.add_wavelength(value=0.6563)
    return optic


# ---------------------------------------------------------------------------
# OpticToCodeVConverter
# ---------------------------------------------------------------------------


class TestOpticToCodeVConverter:
    def test_convert_returns_model(self):
        optic = _make_singlet()
        model = OpticToCodeVConverter(optic).convert()
        assert isinstance(model, CodeVDataModel)

    def test_aperture_epd(self):
        optic = _make_singlet()
        model = OpticToCodeVConverter(optic).convert()
        assert "EPD" in model.aperture
        assert_allclose(model.aperture["EPD"], 10.0)

    def test_wavelengths_in_microns(self):
        optic = _make_singlet()
        model = OpticToCodeVConverter(optic).convert()
        data = model.wavelengths["data"]
        assert len(data) == 3
        assert_allclose(data[0], 0.5876, atol=1e-4)

    def test_primary_wavelength_index(self):
        optic = _make_singlet()
        model = OpticToCodeVConverter(optic).convert()
        assert model.wavelengths["primary_index"] == 0

    def test_fields(self):
        optic = _make_singlet()
        model = OpticToCodeVConverter(optic).convert()
        assert model.fields["type"] == "angle"
        assert model.fields["y"] == [0.0, 5.0]

    def test_surfaces_contain_object_and_image(self):
        optic = _make_singlet()
        model = OpticToCodeVConverter(optic).convert()
        surf_types = [s["type"] for s in model.surfaces.values()]
        assert "object" in surf_types
        assert "image" in surf_types

    def test_catalog_glass_format(self):
        optic = _make_singlet()
        model = OpticToCodeVConverter(optic).convert()
        # Find the surface with glass
        glass_surfs = [s for s in model.surfaces.values() if s.get("glass") is not None]
        assert len(glass_surfs) >= 1
        glass = glass_surfs[0]["glass"]
        assert "name" in glass

    def test_pickups_warning(self):
        """Solves should issue a UserWarning."""
        optic = _make_singlet()
        optic.solves.add("marginal_ray_height", surface_idx=2, height=0.0)
        with pytest.warns(UserWarning, match="solve"):
            OpticToCodeVConverter(optic).convert()


# ---------------------------------------------------------------------------
# CodeVFileEncoder
# ---------------------------------------------------------------------------


class TestCodeVFileEncoder:
    def _make_model(self) -> CodeVDataModel:
        optic = _make_singlet()
        return OpticToCodeVConverter(optic).convert()

    def test_encode_returns_lines(self):
        model = self._make_model()
        lines = CodeVFileEncoder(model).encode()
        assert isinstance(lines, list)
        assert len(lines) > 0
        assert all(isinstance(line, str) for line in lines)

    def test_header_has_rdm(self):
        model = self._make_model()
        lines = CodeVFileEncoder(model).encode()
        assert any(line.strip().startswith("RDM") for line in lines)

    def test_header_has_wl(self):
        model = self._make_model()
        lines = CodeVFileEncoder(model).encode()
        assert any(line.strip().startswith("WL") for line in lines)

    def test_wavelengths_in_nm(self):
        model = self._make_model()
        lines = CodeVFileEncoder(model).encode()
        wl_line = next(l for l in lines if l.strip().startswith("WL"))
        tokens = wl_line.split()
        nm_vals = [float(t) for t in tokens[1:]]
        # Primary wavelength ~587.6 nm
        assert any(abs(v - 587.6) < 1.0 for v in nm_vals)

    def test_has_ref_line(self):
        model = self._make_model()
        lines = CodeVFileEncoder(model).encode()
        assert any(line.strip().startswith("REF") for line in lines)

    def test_has_yan_line(self):
        model = self._make_model()
        lines = CodeVFileEncoder(model).encode()
        assert any(line.strip().startswith("YAN") for line in lines)

    def test_has_epd_line(self):
        model = self._make_model()
        lines = CodeVFileEncoder(model).encode()
        assert any(line.strip().startswith("EPD") for line in lines)

    def test_has_so_and_si(self):
        model = self._make_model()
        lines = CodeVFileEncoder(model).encode()
        assert any(line.strip().startswith("SO") for line in lines)
        assert any(line.strip().startswith("SI") for line in lines)

    def test_title_quoted(self):
        model = self._make_model()
        model.name = "My Optic"
        lines = CodeVFileEncoder(model).encode()
        assert any("TITLE" in line and "My Optic" in line for line in lines)

    def test_aspheric_coeff_A(self):
        """Aspheric coefficient A should appear for even-asphere surfaces."""
        optic = load_codev_file(_seq("asphere.seq"))
        model = OpticToCodeVConverter(optic).convert()
        lines = CodeVFileEncoder(model).encode()
        assert any(line.strip().startswith("A ") for line in lines)

    def test_conic_K(self):
        optic = load_codev_file(_seq("asphere.seq"))
        model = OpticToCodeVConverter(optic).convert()
        lines = CodeVFileEncoder(model).encode()
        assert any(line.strip().startswith("K ") for line in lines)

    def test_stop_sto(self):
        optic = load_codev_file(_seq("cooke_triplet.seq"))
        model = OpticToCodeVConverter(optic).convert()
        lines = CodeVFileEncoder(model).encode()
        assert any(line.strip() == "STO" for line in lines)

    def test_mirror_refl(self):
        optic = load_codev_file(_seq("mirror.seq"))
        model = OpticToCodeVConverter(optic).convert()
        lines = CodeVFileEncoder(model).encode()
        assert any("REFL" in line for line in lines)

    def test_fictitious_glass_nd_vd(self):
        optic = load_codev_file(_seq("asphere.seq"))
        model = OpticToCodeVConverter(optic).convert()
        lines = CodeVFileEncoder(model).encode()
        # Fictitious glass is written as Nd:Vd on the surface line
        s_lines = [
            l for l in lines if l.strip().startswith("S ") or l.startswith("S  ")
        ]
        assert any(":" in line for line in s_lines)


# ---------------------------------------------------------------------------
# save_codev_file / CodeVWriter
# ---------------------------------------------------------------------------


class TestSaveCodeVFile:
    def test_save_creates_file(self):
        optic = _make_singlet()
        with tempfile.NamedTemporaryFile(suffix=".seq", delete=False) as tmp:
            path = tmp.name
        try:
            save_codev_file(optic, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_save_utf8_encoding(self):
        optic = _make_singlet()
        with tempfile.NamedTemporaryFile(
            suffix=".seq", delete=False, mode="w", encoding="utf-8"
        ) as tmp:
            path = tmp.name
        try:
            save_codev_file(optic, path)
            # Should open without error with utf-8
            with open(path, encoding="utf-8") as fh:
                content = fh.read()
            assert len(content) > 0
        finally:
            os.unlink(path)

    def test_codev_writer_interface(self):
        optic = _make_singlet()
        writer = CodeVWriter()
        with tempfile.NamedTemporaryFile(suffix=".seq", delete=False) as tmp:
            path = tmp.name
        try:
            warnings_list = writer.write(optic, path)
            assert isinstance(warnings_list, list)
            assert os.path.exists(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def _round_trip(self, seq_name: str) -> tuple[Optic, Optic]:
        optic1 = load_codev_file(_seq(seq_name))
        with tempfile.NamedTemporaryFile(suffix=".seq", delete=False) as tmp:
            path = tmp.name
        try:
            save_codev_file(optic1, path)
            optic2 = load_codev_file(path)
        finally:
            os.unlink(path)
        return optic1, optic2

    def test_roundtrip_wavelength_count(self):
        o1, o2 = self._round_trip("cooke_triplet.seq")
        assert o1.wavelengths.num_wavelengths == o2.wavelengths.num_wavelengths

    def test_roundtrip_primary_wavelength(self):
        o1, o2 = self._round_trip("cooke_triplet.seq")
        assert_allclose(
            float(o1.primary_wavelength), float(o2.primary_wavelength), rtol=1e-5
        )

    def test_roundtrip_fields_count(self):
        o1, o2 = self._round_trip("cooke_triplet.seq")
        assert o1.fields.num_fields == o2.fields.num_fields

    def test_roundtrip_aperture_value(self):
        o1, o2 = self._round_trip("cooke_triplet.seq")
        assert_allclose(float(o1.aperture.value), float(o2.aperture.value), rtol=1e-5)

    def test_roundtrip_surface_count(self):
        o1, o2 = self._round_trip("cooke_triplet.seq")
        assert o1.surface_group.num_surfaces == o2.surface_group.num_surfaces

    def test_roundtrip_asphere(self):
        o1, o2 = self._round_trip("asphere.seq")
        assert o1.surface_group.num_surfaces == o2.surface_group.num_surfaces

    def test_roundtrip_fno(self):
        o1, o2 = self._round_trip("fno_fields.seq")
        assert_allclose(float(o1.aperture.value), float(o2.aperture.value), rtol=1e-4)


# ---------------------------------------------------------------------------
# OpticToCodeVConverter Extended
# ---------------------------------------------------------------------------


class TestOpticToCodeVConverterExtended:
    def test_convert_aperture_codev(self):
        from optiland.aperture import ImageFNOAperture

        optic = Optic()
        optic.aperture = ImageFNOAperture(4.0)
        conv = OpticToCodeVConverter(optic)
        model = conv.convert()
        assert model.aperture["FNO"] == 4.0

    def test_warn_unknown_aperture_codev(self):
        optic = Optic()

        class UnknownAp:
            ap_type = "unknown"
            value = 0.0

        optic.aperture = UnknownAp()
        optic.surfaces.add(index=0, thickness=0.0)
        conv = OpticToCodeVConverter(optic)
        with pytest.warns(UserWarning, match="Unknown aperture type"):
            conv.convert()

    def test_no_surfaces_codev(self):
        optic = Optic()
        conv = OpticToCodeVConverter(optic)
        model = conv.convert()
        assert model.surfaces == {}

    def test_glass_fictitious_codev(self):
        from optiland.materials import IdealMaterial

        optic = Optic()
        # Add 3 surfaces: Obj, Real, Img
        optic.surfaces.add(index=0, thickness=0.0)
        optic.surfaces.add(
            index=1, radius=50.0, thickness=5.0, material=IdealMaterial(1.7)
        )
        optic.surfaces.add(index=2, thickness=0.0)
        optic.add_wavelength(0.5876, is_primary=True)
        conv = OpticToCodeVConverter(optic)
        with pytest.warns(UserWarning, match="writing as fictitious glass"):
            model = conv.convert()
        # Obj(0) + Surf(1) + Img(2)
        assert "glass" in model.surfaces[1]
        assert float(model.surfaces[1]["glass"]["nd"]) == pytest.approx(1.7)

    def test_reflective_surface_codev(self):
        optic = Optic()
        # Add object and reflective surface
        optic.surfaces.add(index=0, thickness=0.0)
        optic.surfaces.add(index=1, radius=100.0, thickness=-50.0, material="mirror")
        optic.surfaces.add(index=2, thickness=0.0)
        conv = OpticToCodeVConverter(optic)
        model = conv.convert()
        assert model.surfaces[1]["glass"]["name"] == "REFL"
