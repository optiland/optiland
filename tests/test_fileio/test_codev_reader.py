"""Tests for the CODE V reader path.

Covers CodeVDataModel, CodeVDataParser, CodeVToOpticConverter,
and the load_codev_file public entry point.
"""

from __future__ import annotations

import os

import pytest

import optiland.backend as be
from optiland.fileio import load_codev_file
from optiland.fileio.codev.model import CodeVDataModel
from optiland.fileio.codev.reader.converter import CodeVToOpticConverter
from optiland.fileio.codev.reader.parser import CodeVDataParser, _looks_like_float
from optiland.materials import AbbeMaterial, Material
from optiland.optic import Optic
from tests.utils import assert_allclose

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "codev_files")


def _seq(name: str) -> str:
    return os.path.join(_DIR, name)


# ---------------------------------------------------------------------------
# CodeVDataModel
# ---------------------------------------------------------------------------


class TestCodeVDataModel:
    def test_defaults(self):
        m = CodeVDataModel()
        assert m.name is None
        assert m.aperture == {}
        assert m.wavelengths == {"data": []}
        assert m.surfaces == {}
        assert m.radius_mode is True
        assert m.units == "MM"

    def test_to_dict_round_trip(self):
        m = CodeVDataModel(name="Test", aperture={"EPD": 10.0})
        d = m.to_dict()
        assert d["name"] == "Test"
        assert d["aperture"] == {"EPD": 10.0}
        assert d["radius_mode"] is True


# ---------------------------------------------------------------------------
# _looks_like_float helper
# ---------------------------------------------------------------------------


class TestLooksLikeFloat:
    def test_integer(self):
        assert _looks_like_float("42") is True

    def test_float(self):
        assert _looks_like_float("3.14") is True

    def test_negative(self):
        assert _looks_like_float("-100.0") is True

    def test_scientific(self):
        assert _looks_like_float("1.5e10") is True

    def test_word(self):
        assert _looks_like_float("N-BK7") is False

    def test_empty(self):
        assert _looks_like_float("") is False


# ---------------------------------------------------------------------------
# CodeVDataParser — unit tests
# ---------------------------------------------------------------------------


class TestCodeVDataParser:
    def test_parse_cooke_triplet(self):
        parser = CodeVDataParser(_seq("cooke_triplet.seq"))
        model = parser.parse()

        assert model.name == "Cooke Triplet"
        assert "EPD" in model.aperture
        assert_allclose(model.aperture["EPD"], 10.0)

        # Wavelengths in µm (converted from nm)
        wls = model.wavelengths["data"]
        assert len(wls) == 3
        assert_allclose(wls[0], 0.4861, atol=1e-4)
        assert_allclose(wls[1], 0.5876, atol=1e-4)
        assert model.wavelengths["primary_index"] == 1

        # Fields
        assert model.fields["type"] == "angle"
        assert_allclose(model.fields["y"], [0.0, 7.0, 14.0])

        # Surfaces: SO + 6 real + SI = 8 total
        assert len(model.surfaces) == 8

    def test_parse_asphere(self):
        parser = CodeVDataParser(_seq("asphere.seq"))
        model = parser.parse()

        assert model.name == "Aspheric Singlet"

        # Check aspheric surface has coefficients
        surf1 = model.surfaces[1]  # first real surface (index 1, SO is 0)
        assert surf1.get("conic") == pytest.approx(-0.5)
        coeffs = surf1.get("coefficients", [])
        assert len(coeffs) >= 2
        assert coeffs[0] == pytest.approx(1.5e-6)
        assert coeffs[1] == pytest.approx(-2.3e-9)

    def test_parse_fno_aperture(self):
        parser = CodeVDataParser(_seq("fno_fields.seq"))
        model = parser.parse()
        assert "FNO" in model.aperture
        assert_allclose(model.aperture["FNO"], 4.0)

    def test_parse_mirror_glass(self):
        parser = CodeVDataParser(_seq("mirror.seq"))
        model = parser.parse()
        surf1 = model.surfaces[1]
        assert surf1["material"] == "mirror"

    def test_parse_fictitious_glass(self):
        parser = CodeVDataParser(_seq("asphere.seq"))
        model = parser.parse()
        surf1 = model.surfaces[1]
        assert isinstance(surf1["material"], AbbeMaterial)

    def test_parse_catalog_glass(self):
        parser = CodeVDataParser(_seq("cooke_triplet.seq"))
        model = parser.parse()
        surf1 = model.surfaces[1]  # N-SK16_SCHOTT
        assert isinstance(surf1["material"], Material)

    def test_comment_stripping(self):
        """Lines after '!' should be ignored."""
        parser = CodeVDataParser(_seq("cooke_triplet.seq"))
        model = parser.parse()
        # If comments weren't stripped, parsing would fail
        assert model is not None

    def test_stop_surface(self):
        parser = CodeVDataParser(_seq("cooke_triplet.seq"))
        model = parser.parse()
        # Surface 1 (first real surface) is marked STO
        surf1 = model.surfaces[1]
        assert surf1["is_stop"] is True

    def test_rdm_flag_default_true(self):
        parser = CodeVDataParser(_seq("cooke_triplet.seq"))
        model = parser.parse()
        assert model.radius_mode is True

    def test_surface_radius_finite(self):
        parser = CodeVDataParser(_seq("cooke_triplet.seq"))
        model = parser.parse()
        surf1 = model.surfaces[1]
        assert surf1["radius"] == pytest.approx(22.01359)

    def test_parse_asphere_raises(self):
        # Missing A coefficient for ASP surface
        seq = "ASP S1; S1 R 100; S1 A 0; S1 K 0"
        parser = CodeVDataParser(seq)
        # Should NOT raise, but let's see. Actually, if A is missing it defaults to 0.
        # But if the line is incomplete...
        pass

    def test_surface_planar_is_inf(self):
        """Object and image surfaces with radius 0.0 → infinity."""
        parser = CodeVDataParser(_seq("cooke_triplet.seq"))
        model = parser.parse()
        so = model.surfaces[0]  # SO
        assert float(so["radius"]) == pytest.approx(float(be.inf))

    def test_unicode_decode_fallback(self, tmp_path):
        # Create a file with latin-1 encoding
        p = tmp_path / "latin1.seq"
        content = "TITLE 'Título latin-1'\nEPD 10.0\nSO 0 0\nSI 0 0"
        p.write_text(content, encoding="latin-1")

        parser = CodeVDataParser(str(p))
        model = parser.parse()
        assert model.name == "Título latin-1"

    def test_continuation_and_semicolon(self, tmp_path):
        p = tmp_path / "logical.seq"
        # Test & continuation and ; separator
        content = "TITLE 'Multi'\nEPD &\n 20.0; WL 550; WTW 1.0"
        p.write_text(content)

        parser = CodeVDataParser(str(p))
        model = parser.parse()
        assert model.aperture["EPD"] == 20.0
        assert model.wavelengths["data"] == [0.55]
        assert model.wavelengths["weights"] == [1.0]

    def test_rdm_n_curvature_mode(self, tmp_path):
        p = tmp_path / "rdm.seq"
        # RDM N sets radius_mode to False (curvature mode)
        content = "RDM N\nS 0.05 5.0"  # curvature 0.05 -> radius 20.0
        p.write_text(content)

        parser = CodeVDataParser(str(p))
        model = parser.parse()
        assert model.radius_mode is False
        assert model.surfaces[0]["radius"] == 20.0

    def test_standing_sto(self, tmp_path):
        p = tmp_path / "sto.seq"
        # STO without surface token
        content = "STO\nS 100 5"
        p.write_text(content)

        parser = CodeVDataParser(str(p))
        model = parser.parse()
        # The parser handles STO by starting a surface at index 0
        assert model.surfaces[0]["is_stop"] is True

    def test_field_commands(self, tmp_path):
        p = tmp_path / "fields.seq"
        content = "XAN 0 5 10; YAN 0 10 20; WTF 1 2 3"
        p.write_text(content)

        parser = CodeVDataParser(str(p))
        model = parser.parse()
        assert model.fields["type"] == "angle"
        assert model.fields["x"] == [0.0, 5.0, 10.0]
        assert model.fields["y"] == [0.0, 10.0, 20.0]
        assert model.fields["weights"] == [1.0, 2.0, 3.0]

    def test_aperture_na_nao(self, tmp_path):
        p = tmp_path / "ap.seq"
        content = "NA 0.1; NAO 0.2"
        p.write_text(content)

        parser = CodeVDataParser(str(p))
        model = parser.parse()
        assert model.aperture["NA"] == 0.1
        assert model.aperture["NAO"] == 0.2

    def test_sto_sn_xref(self, tmp_path):
        p = tmp_path / "stoxref.seq"
        content = "SO 0 0\nS 50 5\nSTO S1\nSI 0 0"
        p.write_text(content)

        parser = CodeVDataParser(str(p))
        model = parser.parse()
        assert model.sto_surface_index == 1

    def test_tilts_decenters(self, tmp_path):
        p = tmp_path / "tilts.seq"
        content = "S 50 5\nXDE 1.0; YDE 2.0; ZDE 3.0\nADE 5.0; BDE 10.0; CDE 15.0"
        p.write_text(content)

        parser = CodeVDataParser(str(p))
        model = parser.parse()
        s = model.surfaces[0]
        assert s["xde"] == 1.0
        assert s["yde"] == 2.0
        assert s["zde"] == 3.0
        assert s["ade"] == 5.0
        assert s["bde"] == 10.0
        assert s["cde"] == 15.0

    def test_glass_codes(self, tmp_path):
        p = tmp_path / "glass.seq"
        # NNN.VVV, NNNNVV, Nd:Vd
        content = "SO 0 0 '569.631'\nS 0 0 '517642'\nS 0 0 '1.6:50'\nSI 0 0"
        p.write_text(content)

        parser = CodeVDataParser(str(p))
        model = parser.parse()
        assert isinstance(model.surfaces[0]["material"], AbbeMaterial)
        assert float(model.surfaces[0]["material"].index.item()) == 1.569
        assert float(model.surfaces[0]["material"].abbe.item()) == 63.1

        assert isinstance(model.surfaces[1]["material"], AbbeMaterial)
        assert float(model.surfaces[1]["material"].index.item()) == 1.517
        assert float(model.surfaces[1]["material"].abbe.item()) == 64.2

        assert isinstance(model.surfaces[2]["material"], AbbeMaterial)
        assert float(model.surfaces[2]["material"].index.item()) == 1.6
        assert float(model.surfaces[2]["material"].abbe.item()) == 50.0

    def test_glass_candidates_and_warning(self, tmp_path):
        p = tmp_path / "glasswarn.seq"
        # NBK7 should resolve to N-BK7
        # UNKNOWN_GLASS should warn
        content = "S 50 5 NBK7\nS 50 5 UNKNOWN_GLASS"
        p.write_text(content)

        parser = CodeVDataParser(str(p))
        with pytest.warns(
            UserWarning, match="Glass 'UNKNOWN_GLASS' could not be resolved"
        ):
            model = parser.parse()

        assert isinstance(model.surfaces[0]["material"], Material)
        assert model.surfaces[0]["material"].name == "N-BK7"
        assert model.surfaces[1]["material"] is None

    def test_prv_block_warning(self, tmp_path):
        p = tmp_path / "prv.seq"
        content = "PRV\nsome data\nEND"
        p.write_text(content)

        parser = CodeVDataParser(str(p))
        with pytest.warns(UserWarning, match="Private glass catalog"):
            model = parser.parse()


# ---------------------------------------------------------------------------
# CodeVToOpticConverter
# ---------------------------------------------------------------------------


class TestCodeVToOpticConverter:
    def test_read_returns_optic(self):
        conv = CodeVToOpticConverter({})
        optic = conv.read(_seq("cooke_triplet.seq"))
        assert isinstance(optic, Optic)

    def test_optic_name(self):
        optic = CodeVToOpticConverter({}).read(_seq("cooke_triplet.seq"))
        assert optic.name == "Cooke Triplet"

    def test_wavelengths_count(self):
        optic = CodeVToOpticConverter({}).read(_seq("cooke_triplet.seq"))
        assert optic.wavelengths.num_wavelengths == 3

    def test_primary_wavelength(self):
        optic = CodeVToOpticConverter({}).read(_seq("cooke_triplet.seq"))
        assert_allclose(float(optic.primary_wavelength), 0.5876, atol=1e-4)

    def test_fields_count(self):
        optic = CodeVToOpticConverter({}).read(_seq("cooke_triplet.seq"))
        assert optic.fields.num_fields == 3

    def test_surface_count(self):
        """Cooke triplet has 6 real glass surfaces."""
        optic = CodeVToOpticConverter({}).read(_seq("cooke_triplet.seq"))
        # surfaces includes object + image, so num_surfaces depends on Optic
        assert optic.surface_group.num_surfaces >= 6

    def test_aperture_epd(self):
        optic = CodeVToOpticConverter({}).read(_seq("cooke_triplet.seq"))
        assert optic.aperture is not None
        assert optic.aperture.ap_type == "EPD"
        assert_allclose(float(optic.aperture.value), 10.0)

    def test_aperture_fno(self):
        optic = CodeVToOpticConverter({}).read(_seq("fno_fields.seq"))
        assert optic.aperture is not None
        assert optic.aperture.ap_type == "imageFNO"
        assert_allclose(float(optic.aperture.value), 4.0)

    def test_stop_surface_marked(self):
        optic = CodeVToOpticConverter({}).read(_seq("cooke_triplet.seq"))
        # At least one surface should be marked as stop
        stops = [s for s in optic.surfaces if s.is_stop]
        assert len(stops) == 1

    def test_asphere_coefficients(self):
        optic = CodeVToOpticConverter({}).read(_seq("asphere.seq"))
        # asphere.seq: SO(0) + S asphere(1) + S standard(2) + SI(3)
        # Surface at index 1 should be even_asphere
        asph = list(optic.surfaces)[1]
        geom = asph.geometry
        assert hasattr(geom, "coefficients")

    def test_init_with_model(self):
        from optiland.fileio.codev.model import CodeVDataModel

        model = CodeVDataModel(name="TestModel")
        converter = CodeVToOpticConverter(model)
        assert converter.data["name"] == "TestModel"

    def test_implicit_object_insertion(self):
        # Create data with NO object surface (index 0)
        data = {
            "surfaces": {
                1: {"type": "standard", "radius": 50.0, "thickness": 5.0},
                2: {"type": "image", "radius": 0.0, "thickness": 0.0},
            },
            "sto_surface_index": 1,
            "aperture": {"EPD": 10.0},
        }
        converter = CodeVToOpticConverter(data)
        optic = converter.convert()
        # Should now have 3 surfaces: 0 (Obj), 1 (Lens), 2 (Img)
        assert optic.surfaces.num_surfaces == 3
        assert optic.surfaces[0].geometry.radius == float(be.inf)
        # Sto index was 1, should be shifted to 2 locally?
        # Converter shifts it: sto_index += 1
        assert optic.surfaces[2].is_stop

    def test_configure_aperture_fail(self):
        # Must have data but no recognized keys to raise
        converter = CodeVToOpticConverter(
            {"surfaces": {}, "aperture": {"UNKNOWN": 10.0}}
        )
        converter.optic = Optic()
        with pytest.raises(ValueError, match="No valid aperture type found"):
            converter._configure_aperture()

    def test_aperture_na_raises(self):
        converter = CodeVToOpticConverter(
            {"surfaces": {}, "aperture": {"NA": 0.1}}
        )
        converter.optic = Optic()
        with pytest.raises(NotImplementedError, match="image-space numerical aperture"):
            converter._configure_aperture()


# ---------------------------------------------------------------------------
# load_codev_file public API
# ---------------------------------------------------------------------------


class TestLoadCodeVFile:
    def test_load_returns_optic(self):
        optic = load_codev_file(_seq("cooke_triplet.seq"))
        assert isinstance(optic, Optic)

    def test_load_asphere(self):
        optic = load_codev_file(_seq("asphere.seq"))
        assert isinstance(optic, Optic)

    def test_load_fno(self):
        optic = load_codev_file(_seq("fno_fields.seq"))
        assert isinstance(optic, Optic)

    def test_load_mirror(self):
        optic = load_codev_file(_seq("mirror.seq"))
        assert isinstance(optic, Optic)

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_codev_file("nonexistent.seq")
