"""Tests for the Zemax reader path.

Migrated from tests/test_fileio.py and updated to use new module paths.
"""

from __future__ import annotations

import os
from unittest.mock import mock_open, patch

import pytest

import optiland.backend as be
from optiland.fileio import load_zemax_file
from optiland.fileio.zemax.reader.converter import ZemaxToOpticConverter
from optiland.fileio.zemax.reader.parser import ZemaxDataParser
from optiland.fileio.zemax.reader.source import ZemaxFileSourceHandler
from optiland.geometries import ToroidalGeometry
from optiland.materials import Material
from optiland.optic import Optic
from tests.utils import assert_allclose

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zemax_file():
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(current_dir, "zemax_files", "lens1.zmx")


@pytest.fixture
def zemax_dir():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "zemax_files"
    )


# ---------------------------------------------------------------------------
# ZemaxFileSourceHandler
# ---------------------------------------------------------------------------


class TestZemaxFileSourceHandler:
    def test_is_url(self):
        handler = ZemaxFileSourceHandler("http://example.com/test.zmx")
        assert handler._is_url()

    def test_is_not_url(self):
        handler = ZemaxFileSourceHandler("not_a_url")
        assert not handler._is_url()

    @patch("requests.get")
    @patch("builtins.open", new_callable=mock_open)
    @patch("tempfile.NamedTemporaryFile")
    def test_get_local_file_url(self, mock_tempfile, mock_open_f, mock_get, zemax_file):
        mock_resp = mock_get.return_value
        mock_resp.status_code = 200
        mock_resp.content = b"Test content"

        tmp = mock_tempfile.return_value.__enter__.return_value
        tmp.name = "temp.zmx"

        handler = ZemaxFileSourceHandler("http://example.com/test.zmx")
        local = handler.get_local_file()

        mock_get.assert_called_once_with("http://example.com/test.zmx", timeout=10)
        tmp.write.assert_called_once_with(b"Test content")
        assert local == "temp.zmx"

    @patch("requests.get")
    def test_get_local_file_url_fail(self, mock_get):
        mock_get.return_value.status_code = 404
        handler = ZemaxFileSourceHandler("http://example.com/test.zmx")
        with pytest.raises(ValueError, match="Failed to download Zemax file."):
            handler.get_local_file()

    def test_get_local_file_local_path(self, zemax_file):
        handler = ZemaxFileSourceHandler(zemax_file)
        local = handler.get_local_file()
        assert local == zemax_file


# ---------------------------------------------------------------------------
# ZemaxDataParser
# ---------------------------------------------------------------------------


class TestZemaxDataParser:
    def setup_method(self):
        self.parser = ZemaxDataParser("dummy")

    def test_read_fno(self):
        self.parser._read_fno(["FNO", "1.5", "0"])
        assert self.parser.data_model.aperture["imageFNO"] == 1.5

    def test_read_epd(self):
        self.parser._read_epd(["ENPD", "2.5"])
        assert self.parser.data_model.aperture["EPD"] == 2.5

    def test_read_object_na(self):
        self.parser._read_object_na(["OBNA", "0.1", "0"])
        assert self.parser.data_model.aperture["objectNA"] == 0.1

    def test_read_conic(self):
        self.parser._read_conic(["CONI", "0"])
        assert self.parser._current_surf_data["conic"] == 0.0

    def test_read_glass(self):
        self.parser._read_glass(["GLAS", "N-BK7", "0", "0", "1.5", "50"])
        mat = self.parser._current_surf_data["material"]
        assert isinstance(mat, Material)

    def test_read_stop(self):
        self.parser._read_stop([])
        assert self.parser._current_surf_data["is_stop"]

    def test_read_mode_valid(self):
        self.parser._read_mode(["MODE", "SEQ"])

    def test_read_mode_invalid(self):
        with pytest.raises(ValueError):
            self.parser._read_mode(["MODE", "NONSEQ"])

    def test_read_surface_type(self):
        self.parser._read_surf_type(["TYPE", "STANDARD"])
        assert self.parser._current_surf_data["type"] == "standard"

    def test_read_floating_stop(self):
        self.parser._read_floating_stop(["FLOA"])
        assert self.parser.data_model.aperture["floating_stop"] is True

    def test_read_diameter(self):
        self.parser._read_diameter(["DIAM", "8.5", "1", "0", "0", "1", '""'])
        assert self.parser._current_surf_data["diameter"] == 8.5

    def test_read_config_data_legacy_short_ftyp(self):
        # Legacy ZEMAX (e.g. VERS 6133) writes FTYP with only 1-2 tokens
        # while modern files emit 8. Verify the parser falls back to
        # sensible defaults instead of raising IndexError.
        self.parser._read_config_data(["FTYP", "0"])
        fields = self.parser.data_model.fields
        # angle-type field (default for legacy on-axis layout)
        assert fields["type"] == "angle"
        # one on-axis field seeded so downstream KeyError 'x'/'y' is avoided
        assert fields["num_fields"] == 1
        assert fields["x"] == [0.0]
        assert fields["y"] == [0.0]

    def test_read_config_data_empty_tokens(self):
        # FTYP positions present but empty (some malformed files do this) —
        # _safe_int should treat empty as missing and apply defaults.
        self.parser._read_config_data(
            ["FTYP", "", "", "", "", "", "", "", ""]
        )
        fields = self.parser.data_model.fields
        assert fields["type"] == "angle"
        assert fields["num_fields"] == 1
        assert fields["object_space_telecentric"] is False
        assert fields["afocal_image_space"] is False

    def test_read_config_data_non_integer_tokens(self):
        # FTYP positions present but non-integer text — _safe_int should
        # swallow the ValueError and apply defaults.
        self.parser._read_config_data(
            ["FTYP", "x", "x", "x", "x", "x", "x", "x", "x"]
        )
        fields = self.parser.data_model.fields
        assert fields["type"] == "angle"
        assert fields["num_fields"] == 1
        assert fields["afocal_image_space"] is False


# ---------------------------------------------------------------------------
# End-to-end reader tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_load_zemax_file(self, zemax_file):
        optic = load_zemax_file(zemax_file)
        assert isinstance(optic, Optic)

    def test_load_and_convert_asphere(self, zemax_dir):
        filename = os.path.join(zemax_dir, "lens2.zmx")
        optic = load_zemax_file(filename)
        assert isinstance(optic, Optic)

    def test_load_floa_aperture(self, zemax_dir):
        filename = os.path.join(zemax_dir, "lens_floa.zmx")
        optic = load_zemax_file(filename)
        assert isinstance(optic, Optic)
        assert optic.aperture.ap_type == "float_by_stop_size"
        assert optic.aperture.value == 8.5

    def test_load_toroidal_surface(self, zemax_dir):
        filename = os.path.join(zemax_dir, "thorlabs_lj1598l1.zmx")
        optic = load_zemax_file(filename)
        assert isinstance(optic, Optic)
        surf1 = optic.surfaces[1]
        surf2 = optic.surfaces[2]
        assert_allclose(surf1.geometry.R_yz, 1 / 0.4950495049504951)
        assert_allclose(surf1.geometry.R_rot, be.inf)
        assert_allclose(surf2.geometry.R_yz, be.inf)
        assert_allclose(surf2.geometry.R_rot, be.inf)


# ---------------------------------------------------------------------------
# ZemaxToOpticConverter extended tests
# ---------------------------------------------------------------------------


class TestZemaxToOpticConverterExtended:
    def test_configure_aperture_floating_stop_no_diameter(self):
        zemax_data = {
            "surfaces": {
                0: {
                    "type": "standard",
                    "is_stop": True,
                    "radius": 0.0,
                    "conic": 0.0,
                    "thickness": 0.0,
                    "material": "Air",
                },
            },
            "aperture": {"floating_stop": True},
            "fields": {"type": "angle", "x": [0], "y": [0]},
            "wavelengths": {"primary_index": 0, "data": [0.55]},
        }
        converter = ZemaxToOpticConverter(zemax_data)
        converter.optic = Optic()
        converter._configure_surfaces()
        with pytest.raises(
            ValueError,
            match="Floating stop aperture specified but no stop diameter found",
        ):
            converter._configure_aperture()

    def test_configure_aperture_no_valid_type(self):
        zemax_data = {
            "surfaces": {},
            "aperture": {"floating_stop": False},
            "fields": {"type": "angle", "x": [0], "y": [0]},
            "wavelengths": {"primary_index": 0, "data": [0.55]},
        }
        converter = ZemaxToOpticConverter(zemax_data)
        converter.optic = Optic()
        with pytest.raises(ValueError, match="No valid aperture type found"):
            converter._configure_aperture()

    def test_configure_surface_coefficients_unsupported_type(self):
        converter = ZemaxToOpticConverter(
            {
                "surfaces": {},
                "aperture": {"EPD": 10},
                "fields": {"type": "angle", "x": [0], "y": [0]},
                "wavelengths": {"primary_index": 0, "data": [0.55]},
            }
        )
        with pytest.raises(ValueError, match="Unsupported Zemax surface type"):
            converter._configure_surface_coefficients(
                {"type": "unsupported_surface_type"}
            )

    def test_configure_fields_vignette_warning(self, capsys):
        zemax_data = {
            "surfaces": {},
            "aperture": {"EPD": 10},
            "fields": {
                "type": "angle",
                "x": [0],
                "y": [0],
                "vignette_decenter_x": [0.1],
                "vignette_decenter_y": [0.0],
            },
            "wavelengths": {"primary_index": 0, "data": [0.55]},
        }
        converter = ZemaxToOpticConverter(zemax_data)
        converter.optic = Optic()
        converter._configure_fields()
        captured = capsys.readouterr()
        assert "Warning: Vignette decentering is not supported." in captured.out

    def test_configure_surfaces_coordinate_break(self):
        zemax_data = {
            "surfaces": {
                0: {
                    "type": "coordinate_break",
                    "param_0": 1.0,
                    "param_1": 2.0,
                    "thickness": 5.0,
                    "param_2": 10.0,
                    "param_3": 20.0,
                    "param_4": 30.0,
                    "conic": 0.0,
                },
                1: {
                    "type": "standard",
                    "radius": 100.0,
                    "thickness": 10.0,
                    "conic": 0.0,
                    "material": "N-BK7",
                },
            },
            "aperture": {"EPD": 10},
            "fields": {"type": "angle", "x": [0], "y": [0]},
            "wavelengths": {"primary_index": 0, "data": [0.55]},
        }
        converter = ZemaxToOpticConverter(zemax_data)
        optic = converter.convert()
        surf = optic.surfaces[0]
        assert surf.geometry.radius == 100.0
        cs = surf.geometry.cs
        assert (
            cs.x != 0
            or cs.y != 0
            or cs.z != 0
            or cs.rx != 0
            or cs.ry != 0
            or cs.rz != 0
        )

    def test_configure_surfaces_toroidal(self):
        zemax_data = {
            "surfaces": {
                0: {
                    "type": "toroidal",
                    "radius": 50.0,
                    "param_1": 60.0,
                    "param_2": 0.1,
                    "thickness": 5.0,
                    "conic": 0.0,
                    "material": "Air",
                },
            },
            "aperture": {"EPD": 10},
            "fields": {"type": "angle", "x": [0], "y": [0]},
            "wavelengths": {"primary_index": 0, "data": [0.55]},
        }
        optic = ZemaxToOpticConverter(zemax_data).convert()
        surf = optic.surfaces[0]
        assert isinstance(surf.geometry, ToroidalGeometry)
        assert surf.geometry.R_yz == 50.0
        assert surf.geometry.R_rot == 60.0

    def test_configure_surfaces_paraxial(self):
        # PARAXIAL surface in zemax → 'paraxial' surface_type in Optiland.
        # PARM 1 carries the focal length (Zemax convention).
        zemax_data = {
            "surfaces": {
                0: {
                    "type": "standard",
                    "radius": be.inf,
                    "thickness": be.inf,
                    "conic": 0.0,
                    "material": "Air",
                },
                1: {
                    "type": "paraxial",
                    "radius": be.inf,
                    "thickness": 100.0,
                    "conic": 0.0,
                    "param_0": 100.0,  # focal length
                    "is_stop": True,
                    "material": "Air",
                },
                2: {
                    "type": "standard",
                    "radius": be.inf,
                    "thickness": 0.0,
                    "conic": 0.0,
                    "material": "Air",
                },
            },
            "aperture": {"EPD": 10},
            "fields": {"type": "angle", "x": [0], "y": [0]},
            "wavelengths": {"primary_index": 0, "data": [0.55]},
        }
        optic = ZemaxToOpticConverter(zemax_data).convert()
        # Surface 1 should be paraxial with f=100
        paraxial_surf = optic.surfaces[1]
        assert paraxial_surf.surface_type == "paraxial"
        assert paraxial_surf.interaction_model.f == 100.0
        # Paraxial EFL must trace to 100 mm
        efl = float(optic.paraxial.f2())
        assert_allclose(efl, 100.0, rtol=1e-6)

    def test_configure_surfaces_paraxial_with_coordinate_break(self):
        # A coordinate_break anywhere in the surface list forces the CB code
        # path inside _configure_surfaces, which has its own paraxial f-injection
        # block. Exercise it explicitly.
        zemax_data = {
            "surfaces": {
                0: {
                    "type": "standard",
                    "radius": be.inf,
                    "thickness": be.inf,
                    "conic": 0.0,
                    "material": "Air",
                },
                1: {
                    "type": "coordinate_break",
                    "param_0": 0.0,
                    "param_1": 0.0,
                    "thickness": 0.0,
                    "param_2": 0.0,
                    "param_3": 0.0,
                    "param_4": 0.0,
                    "conic": 0.0,
                },
                2: {
                    "type": "paraxial",
                    "radius": be.inf,
                    "thickness": 50.0,
                    "conic": 0.0,
                    "param_0": 50.0,
                    "is_stop": True,
                    "material": "Air",
                },
                3: {
                    "type": "standard",
                    "radius": be.inf,
                    "thickness": 0.0,
                    "conic": 0.0,
                    "material": "Air",
                },
            },
            "aperture": {"EPD": 10},
            "fields": {"type": "angle", "x": [0], "y": [0]},
            "wavelengths": {"primary_index": 0, "data": [0.55]},
        }
        optic = ZemaxToOpticConverter(zemax_data).convert()
        # Surface 1 in the resulting Optic corresponds to the paraxial element
        # (surface 0 is the object plane, the coordinate_break is consumed).
        paraxial_surf = optic.surfaces[1]
        assert paraxial_surf.surface_type == "paraxial"
        assert paraxial_surf.interaction_model.f == 50.0

    def test_configure_surface_coefficients_paraxial_returns_none(self):
        converter = ZemaxToOpticConverter({
            "surfaces": {},
            "aperture": {"EPD": 10},
            "fields": {"type": "angle", "x": [0], "y": [0]},
            "wavelengths": {"primary_index": 0, "data": [0.55]},
        })
        assert converter._configure_surface_coefficients({"type": "paraxial"}) is None

    def test_configure_surfaces_infinity_thickness(self):
        zemax_data = {
            "surfaces": {
                0: {
                    "type": "standard",
                    "radius": be.inf,
                    "thickness": be.inf,
                    "conic": 0.0,
                    "material": "Air",
                },
            },
            "aperture": {"EPD": 10},
            "fields": {"type": "angle", "x": [0], "y": [0]},
            "wavelengths": {"primary_index": 0, "data": [0.55]},
        }
        optic = ZemaxToOpticConverter(zemax_data).convert()
        assert be.isinf(optic.surfaces[0].thickness)


# ---------------------------------------------------------------------------
# Zemax Surfaces
# ---------------------------------------------------------------------------


class TestZemaxSurfaces:
    def test_get_handler_error(self):
        from optiland.fileio.zemax.surfaces import get_handler

        with pytest.raises(
            NotImplementedError,
            match="Zemax surface type 'NON_EXISTENT_SURFACE' is not supported",
        ):
            get_handler("NON_EXISTENT_SURFACE")

    def test_base_surface_handler_radius(self):
        from optiland.fileio.zemax.surfaces import _curvature, _radius

        # Test _radius helper (Zemax CURV → Optiland RAD)
        assert _radius(0.0) == float(be.inf)
        assert _radius(0.02) == 50.0

        # Test _curvature helper (Optiland RAD → Zemax CURV)
        assert _curvature(float(be.inf)) == 0.0
        assert _curvature(50.0) == 0.02

    def test_standard_surface_handler_defaults(self):
        from optiland.fileio.zemax.surfaces import StandardSurfaceHandler

        handler = StandardSurfaceHandler()
        data = {"radius": 100.0, "conic": 0.0}
        params = handler.parse(data)
        assert params["radius"] == 100.0
        assert params["conic"] == 0.0
