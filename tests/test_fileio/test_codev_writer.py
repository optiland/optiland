"""Tests for the CODE V writer path.

Covers OpticToCodeVConverter, CodeVFileEncoder, CodeVWriter,
and the save_codev_file public entry point.
"""

from __future__ import annotations

import os
import re
import tempfile
import warnings

import pytest

import optiland.backend as be
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


def _complete(optic: Optic) -> Optic:
    """Give a hand-built optic the aperture, field and wavelength CODE V needs."""
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    if not len(optic.wavelengths):
        optic.wavelengths.add(value=0.5876, is_primary=True)
    return optic


def _make_conic_sphere(conic: float = 0.0, material=None) -> Optic:
    """Singlet built in code (no reader) with a conic and any surface-1 material."""
    from optiland.materials import IdealMaterial

    optic = Optic()
    optic.surfaces.add(index=0, thickness=1e10)
    optic.surfaces.add(
        index=1,
        radius=50.0,
        thickness=5.0,
        conic=conic,
        material=material if material is not None else IdealMaterial(1.5),
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=-50.0, thickness=45.0)
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=0.5876, is_primary=True)
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

    def test_header_starts_with_len_new_dim_m(self):
        lines = CodeVFileEncoder(self._make_model()).encode()
        assert lines[0] == "LEN NEW"
        assert lines[1] == "DIM M"
        assert lines.index("RDM Y") > 1

    def test_ends_with_go(self):
        lines = CodeVFileEncoder(self._make_model()).encode()
        assert lines[-1] == "GO"
        assert lines[-2].startswith("SI ")

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
        """Letters A and B are written after ASP and K, and no others."""
        optic = load_codev_file(_seq("asphere.seq"))
        model = OpticToCodeVConverter(optic).convert()
        lines = CodeVFileEncoder(model).encode()
        asp = lines.index("  ASP")
        k = lines.index("  K -0.5")
        a = next(i for i, line in enumerate(lines) if line.startswith("  A "))
        assert asp < k < a
        assert float(lines[a].split()[1]) == 1.5e-6
        b = next(line for line in lines if line.startswith("  B "))
        assert float(b.split()[1]) == -2.3e-9
        assert not any(line.split()[:1] == [c] for line in lines for c in "CDEFGHJ")

    def test_conic_K(self):
        optic = _make_conic_sphere(conic=-0.5)
        model = OpticToCodeVConverter(optic).convert()
        lines = CodeVFileEncoder(model).encode()
        k_idx = next(i for i, line in enumerate(lines) if line.strip() == "K -0.5")
        assert lines[k_idx - 1].strip() == "CON"

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
        from optiland.materials import AbbeMaterial

        optic = _make_conic_sphere(material=AbbeMaterial(1.5168, 64.17))
        model = OpticToCodeVConverter(optic).convert()
        lines = CodeVFileEncoder(model).encode()
        # An unnamed glass is a private glass, never an Nd:Vd token
        s_lines = [
            l for l in lines if l.strip().startswith("S ") or l.startswith("S  ")
        ]
        assert any(line.endswith("'OPT1'") for line in s_lines)
        assert not any(":" in line for line in s_lines)


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

    def test_saved_file_ends_with_go_newline(self, tmp_path):
        path = tmp_path / "out.seq"
        save_codev_file(_make_singlet(), str(path))
        assert path.read_text(encoding="utf-8").endswith("GO\n")

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

    def test_roundtrip_asphere(self, set_test_backend):
        optic = load_codev_file(_seq("asphere.seq"))
        with tempfile.NamedTemporaryFile(suffix=".seq", delete=False) as tmp:
            path = tmp.name
        try:
            save_codev_file(optic, path)
            optic2 = load_codev_file(path)
            with open(path, encoding="utf-8") as fh:
                text = fh.read().splitlines()
        finally:
            os.unlink(path)
        geom, geom2 = optic.surfaces[1].geometry, optic2.surfaces[1].geometry
        coeffs = [float(c) for c in geom.coefficients]
        assert coeffs[0] == 0.0
        assert [float(c) for c in geom2.coefficients] == coeffs
        # The sag of the reloaded surface equals the sag the file text describes.
        emitted = _emitted_asphere(text)[0]
        letters = [emitted["letters"].get(c, 0.0) for c in _LETTERS]
        for r in (1.0, 2.5, 4.0):
            expected = _codev_sag(emitted["radius"], emitted["k"], letters, r)
            assert_allclose(geom2.sag(0.0, r), expected, rtol=1e-12, atol=0.0)

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
        optic.surfaces.add(index=0, thickness=10.0)
        optic.surfaces.add(
            index=1,
            radius=50.0,
            thickness=5.0,
            material=IdealMaterial(1.7),
            is_stop=True,
        )
        optic.surfaces.add(index=2, thickness=0.0)
        optic.add_wavelength(0.5876, is_primary=True)
        conv = OpticToCodeVConverter(_complete(optic))
        with pytest.warns(UserWarning, match="private glass"):
            model = conv.convert()
        # Obj(0) + Surf(1) + Img(2)
        assert model.surfaces[1]["glass"] == {"private": "OPT1"}
        assert model.private_glasses["OPT1"] == [pytest.approx(1.7)]

    def test_reflective_surface_codev(self):
        optic = Optic()
        # Add object and reflective surface
        optic.surfaces.add(index=0, thickness=10.0)
        optic.surfaces.add(
            index=1,
            radius=100.0,
            thickness=-50.0,
            material="mirror",
            is_stop=True,
        )
        optic.surfaces.add(index=2, thickness=0.0)
        _complete(optic)
        conv = OpticToCodeVConverter(optic)
        model = conv.convert()
        assert model.surfaces[1]["glass"]["name"] == "REFL"


# ---------------------------------------------------------------------------
# Non-finite values (CODE V has no inf / nan)
# ---------------------------------------------------------------------------


def _make_lens(
    object_thickness: float = float("inf"),
    conic: float = 0.0,
    field_weight: float = 1.0,
    radius: float = 50.0,
) -> Optic:
    """Singlet with an object at ``object_thickness``, built via the public API."""
    from optiland.materials import Material

    optic = Optic()
    optic.surfaces.add(index=0, thickness=object_thickness)
    optic.surfaces.add(
        index=1,
        radius=radius,
        thickness=5.0,
        conic=conic,
        material=Material("N-BK7", "schott"),
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=-50.0, thickness=45.0)
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0, weight=field_weight)
    optic.wavelengths.add(value=0.5876, is_primary=True)
    return optic


def _lens_with_aperture(aperture) -> Optic:
    """Singlet whose first surface carries ``aperture``."""
    optic = _make_lens()
    optic.surfaces[1].aperture = aperture
    return optic


def _encode(optic: Optic) -> list[str]:
    return CodeVFileEncoder(OpticToCodeVConverter(optic).convert()).encode()


def _make_asphere(coefficients, conic: float = 0.0) -> Optic:
    """Singlet whose first surface is an even asphere."""
    from optiland.materials import Material

    optic = Optic()
    optic.surfaces.add(index=0, thickness=float("inf"))
    optic.surfaces.add(
        index=1,
        surface_type="even_asphere",
        radius=50.0,
        thickness=5.0,
        conic=conic,
        coefficients=coefficients,
        material=Material("N-BK7", "schott"),
        is_stop=True,
    )
    optic.surfaces.add(index=2, radius=-50.0, thickness=45.0)
    optic.surfaces.add(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=0.5876, is_primary=True)
    return optic


_LETTERS = ("A", "B", "C", "D", "E", "F", "G", "H", "J")

# Distinct, alternating-sign values for the letters A..H, J.
_ALL_LETTERS = [1.5e-6, -2.3e-9, 3.1e-12, -4.7e-15, 5.3e-18]
_ALL_LETTERS += [-6.1e-21, 7.9e-24, -8.3e-27, 9.7e-30]


def _codev_sag(radius: float, conic: float, letters: list[float], r: float) -> float:
    """CODE V even-asphere sag on Python floats: letter j multiplies r**(2j + 4)."""
    c = 1.0 / radius
    z = c * r**2 / (1.0 + (1.0 - (1.0 + conic) * c**2 * r**2) ** 0.5)
    return z + sum(a * r ** (2 * j + 4) for j, a in enumerate(letters))


def _emitted_asphere(lines: list[str]) -> list[dict]:
    """Parse the S, ASP, K and letter lines of every ``S`` surface in the text.

    Returns one dict per surface with its radius, whether an ``ASP`` line was
    written, the conic (0.0 if no ``K`` line), the letters as floats and the
    line index of ``ASP``, ``K`` and each letter.
    """
    surfaces: list[dict] = []
    for i, line in enumerate(lines):
        if line.startswith("S "):
            surfaces.append(
                {
                    "radius": float(line.split()[1]),
                    "asp": False,
                    "k": 0.0,
                    "letters": {},
                    "at": {},
                }
            )
        elif surfaces and line == "  ASP":
            surfaces[-1]["asp"] = True
            surfaces[-1]["at"]["ASP"] = i
        elif surfaces and line.startswith("  K "):
            surfaces[-1]["k"] = float(line.split()[1])
            surfaces[-1]["at"]["K"] = i
        elif surfaces and (m := re.fullmatch(r"  ([A-HJ]) (\S+)", line)):
            surfaces[-1]["letters"][m.group(1)] = float(m.group(2))
            surfaces[-1]["at"][m.group(1)] = i
    return surfaces


def _letters_of(optic: Optic) -> dict[str, float]:
    """Letters written for the first surface of ``optic``."""
    return _emitted_asphere(_encode(optic))[0]["letters"]


class TestEvenAsphereExport:
    @pytest.mark.parametrize("conic", [-0.5, -1.0])
    @pytest.mark.parametrize(
        "coefficients",
        [[1e-6], [1e-4, 1.5e-6, -2.3e-9], [-1e-12], [1e-6, 0.0, 0.0]],
    )
    def test_r2_term_refused(self, coefficients, conic, tmp_path, set_test_backend):
        optic = _make_asphere(coefficients, conic=conic)
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(NotImplementedError, match=r"Surface 1: r\^2"):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    def test_tiny_r2_term_refused(self, set_test_backend):
        optic = _make_asphere([1e-31, 1e-6], conic=-0.5)
        with pytest.raises(NotImplementedError, match=r"Surface 1: r\^2"):
            OpticToCodeVConverter(optic).convert()

    @pytest.mark.parametrize(
        "coefficients",
        [
            [0.0, 1e-6] + [0.0] * 8 + [1e-26],
            [0.0] * 11 + [1e-26],
            [0.0] * 11 + [1e-31],
            [0.0, 1e-6] + [0.0] * 8 + [1e-43],
        ],
    )
    def test_terms_above_r20_refused(self, coefficients, set_test_backend):
        optic = _make_asphere(coefficients, conic=-0.5)
        with pytest.raises(NotImplementedError, match=r"Surface 1: .*above r\^20"):
            OpticToCodeVConverter(optic).convert()

    def test_trailing_zeros_beyond_r20_ok(self, set_test_backend):
        letters = _letters_of(_make_asphere([0.0, 1e-6] + [0.0] * 12, conic=-0.5))
        assert letters == {"A": 1e-6}
        letters = _letters_of(_make_asphere([0.0] * 9 + [9.7e-30], conic=-0.5))
        assert letters == {"J": 9.7e-30}

    @pytest.mark.parametrize(
        "coefficients, bad",
        [
            ([0.0, float("nan")], 1),
            ([0.0, float("inf")], 1),
            ([0.0, float("-inf")], 1),
            ([float("nan")], 0),
            ([0.0] * 10 + [float("nan")], 10),
        ],
    )
    def test_nan_coefficient_refused(self, coefficients, bad, set_test_backend):
        optic = _make_asphere(coefficients)
        message = (
            r"Surface 1: non-finite even-asphere coefficient "
            rf"coefficients\[{bad}\]"
        )
        with pytest.raises(ValueError, match=message):
            OpticToCodeVConverter(optic).convert()

    @pytest.mark.parametrize("coefficients", [[], [0.0], [0.0, 0.0, 0.0], [-0.0, 0.0]])
    def test_conic_only_asphere_written_as_con(self, coefficients, set_test_backend):
        lines = _encode(_make_asphere(coefficients, conic=-0.5))
        k = lines.index("  K -0.5")
        assert lines[k - 1] == "  CON"
        assert "  ASP" not in lines
        letters = ("A", "B", "C", "D", "E", "F", "G", "H", "J")
        assert not any(line.split()[:1] == [c] for line in lines for c in letters)

    def test_tiny_letters_written(self, set_test_backend):
        lines = _encode(_make_asphere([0.0, 1e-31], conic=-0.5))
        k = lines.index("  K -0.5")
        assert lines[k - 1] == "  ASP"
        assert lines[k + 1] == "  A 1e-31"
        lines = _encode(_make_asphere([0.0] * 9 + [1e-43], conic=-0.5))
        k = lines.index("  K -0.5")
        assert lines[k - 1] == "  ASP"
        assert lines[k + 1] == "  J 1e-43"

    def test_zero_coefficients_and_zero_conic_write_nothing_extra(
        self, set_test_backend
    ):
        lines = _encode(_make_asphere([0.0, 0.0]))
        assert "  CON" not in lines
        assert "  ASP" not in lines
        assert not any(line.strip().startswith("K ") for line in lines)

    def test_aspheric_singlet_sample_refused(self, set_test_backend):
        from optiland.samples.simple import AsphericSinglet

        with pytest.raises(NotImplementedError, match=r"Surface \d+: r\^2"):
            OpticToCodeVConverter(AsphericSinglet()).convert()

    def test_aspheric_singlet_without_r2_exports(self, set_test_backend):
        from optiland.samples.simple import AsphericSinglet

        optic = AsphericSinglet()
        optic.surfaces[1].geometry.coefficients[0] = 0.0
        letters = _letters_of(optic)
        assert letters == {"A": -4.690412e-6, "B": -6.404376e-8}

    def test_writer_letters_follow_codev_powers(self, set_test_backend):
        optic = _make_asphere([0.0, *_ALL_LETTERS], conic=-0.5)
        lines = _encode(optic)
        emitted = _emitted_asphere(lines)[0]
        assert list(emitted["letters"]) == list(_LETTERS)
        assert list(emitted["letters"].values()) == _ALL_LETTERS
        assert not any(line.startswith("  I ") for line in lines)
        at = emitted["at"]
        assert at["ASP"] < at["K"] < at["A"]
        geom = optic.surfaces[1].geometry
        for r in (2.5, 5.0, 7.5, 10.0):
            expected = _codev_sag(
                emitted["radius"], emitted["k"], list(emitted["letters"].values()), r
            )
            assert_allclose(geom.sag(0.0, r), expected, rtol=1e-12, atol=0.0)

    @pytest.mark.parametrize("conic", [0.0, -0.5])
    @pytest.mark.parametrize("j", range(9))
    def test_writer_each_letter_isolated(self, j, conic, set_test_backend):
        optic = _make_asphere([0.0] * (j + 1) + [1.25e-7], conic=conic)
        emitted = _emitted_asphere(_encode(optic))[0]
        assert emitted["asp"]
        assert emitted["letters"] == {_LETTERS[j]: 1.25e-7}

    def test_writer_scale_consistency(self, set_test_backend):
        letters = [*_ALL_LETTERS[:-1], 1e-29]
        optic = _make_asphere([0.0, *letters], conic=-0.5)
        before = _letters_of(optic)
        optic.updater.scale_system(2.0)
        after = _letters_of(optic)
        assert list(after) == list(_LETTERS)
        for j, key in enumerate(_LETTERS):
            assert_allclose(
                after[key], before[key] * 2.0 ** (-2 * j - 3), rtol=1e-12, atol=0.0
            )

    def test_two_asphere_surfaces_and_sphere(self, set_test_backend):
        from optiland.materials import Material

        optic = _make_asphere([0.0, 1e-6], conic=-0.5)
        optic.surfaces.add(
            index=2,
            surface_type="even_asphere",
            radius=80.0,
            thickness=2.0,
            coefficients=[0.0, 0.0, 3e-9, -4e-12],
            material=Material("N-BK7", "schott"),
        )
        optic.surfaces.add(
            index=3, radius=-60.0, thickness=40.0, material=Material("N-BK7", "schott")
        )
        lines = _encode(optic)
        first, second, sphere, _ = _emitted_asphere(lines)
        assert first["letters"] == {"A": 1e-6}
        assert first["k"] == -0.5
        assert second["asp"]
        assert second["letters"] == {"B": 3e-9, "C": -4e-12}
        assert second["k"] == 0.0
        assert not sphere["asp"]
        assert sphere["letters"] == {}
        assert lines.count("  ASP") == 2

    def test_write_read_write_fixed_point(self, set_test_backend, tmp_path):
        optic = _make_asphere([0.0, *_ALL_LETTERS], conic=-0.5)
        first = _encode(optic)
        path = tmp_path / "again.seq"
        save_codev_file(optic, str(path))
        assert _encode(load_codev_file(str(path))) == first

    def test_ndarray_coefficients_export(self, set_test_backend):
        # After optimization the coefficients can be an array, not a list.
        optic = _make_asphere([0.0], conic=-0.5)
        optic.surfaces[1].geometry.coefficients = be.array([0.0, 1e-6, 2e-9, 0.0])
        emitted = _emitted_asphere(_encode(optic))[0]
        assert emitted["asp"]
        assert emitted["letters"] == {"A": 1e-6, "B": 2e-9}

    def test_padded_coefficients_from_variable_export(self, set_test_backend):
        from optiland.optimization.variable.asphere_coeff import (
            AsphereCoeffVariable,
        )

        # get_value() past the end of the list replaces it with a padded array.
        optic = _make_asphere([0.0, 1e-6], conic=-0.5)
        AsphereCoeffVariable(optic, 1, 5).get_value()
        geom = optic.surfaces[1].geometry
        assert not isinstance(geom.coefficients, list)
        geom.coefficients[5] = 3e-9
        emitted = _emitted_asphere(_encode(optic))[0]
        assert emitted["asp"]
        assert emitted["letters"] == {"A": 1e-6, "E": 3e-9}

    def test_torch_grad_coefficients_export(self, set_test_backend):
        if be.get_backend() != "torch":
            pytest.skip("requires the torch backend")
        import torch

        optic = _make_asphere([0.0], conic=-0.5)
        grads = [
            torch.tensor(v, dtype=torch.float64, requires_grad=True)
            for v in (0.0, 1e-6, 2e-9)
        ]
        optic.surfaces[1].geometry.coefficients = grads
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # float() on a grad tensor
            letters = _letters_of(optic)
        assert letters == {"A": 1e-6, "B": 2e-9}
        assert all(g.requires_grad for g in optic.surfaces[1].geometry.coefficients)


class TestProfileKeyword:
    def test_conic_writes_con_before_k(self, set_test_backend):
        lines = _encode(_make_lens(conic=-1.0))
        k = lines.index("  K -1")
        assert lines[k - 1] == "  CON"
        assert lines.count("  CON") == 1

    def test_sphere_no_profile_keyword(self, set_test_backend):
        lines = _encode(_make_lens())
        assert "  CON" not in lines
        assert "  ASP" not in lines
        assert not any(line.strip().startswith("K ") for line in lines)

    def test_profile_keyword_per_surface(self, set_test_backend):
        from optiland.materials import Material

        optic = _make_lens()
        optic.surfaces.add(
            index=2,
            radius=80.0,
            thickness=2.0,
            conic=-0.5,
            material=Material("N-BK7", "schott"),
        )
        lines = _encode(optic)
        s_idx = [i for i, line in enumerate(lines) if line.startswith("S ")]
        assert len(s_idx) == 3
        # CON follows only the conic surface (second S line), directly before K
        con = lines.index("  CON")
        assert s_idx[1] < con < s_idx[2]
        assert lines[con + 1] == "  K -0.5"
        assert lines.count("  CON") == 1

    def test_tiny_conic_below_threshold_writes_nothing(self, set_test_backend):
        lines = _encode(_make_lens(conic=1e-18))
        assert "  CON" not in lines
        assert not any(line.strip().startswith("K ") for line in lines)


class TestNonFinite:
    def test_infinite_object_written_as_1e10(self, set_test_backend):
        lines = _encode(_make_lens())
        so = next(line for line in lines if line.startswith("SO "))
        assert float(so.split()[2]) == 1e10
        text = "\n".join(lines).lower()
        assert "inf" not in text
        assert "nan" not in text

    def test_finite_object_distance_unchanged(self, set_test_backend):
        lines = _encode(_make_lens(object_thickness=250.0))
        so = next(line for line in lines if line.startswith("SO "))
        assert float(so.split()[2]) == 250.0

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"object_thickness": float("-inf")}, "Object"),
            ({"object_thickness": float("nan")}, "Object"),
            ({"conic": float("nan")}, "Surface 1"),
            ({"radius": float("nan")}, "Surface 1"),
            ({"field_weight": float("nan")}, "header"),
        ],
    )
    def test_nonfinite_values_raise_and_leave_file(
        self, kwargs, match, tmp_path, set_test_backend
    ):
        optic = _make_lens(**kwargs)
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(ValueError, match=match):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    def test_planar_radius_not_flagged(self, set_test_backend):
        lines = _encode(_make_lens(radius=float("inf")))
        text = "\n".join(lines).lower()
        assert "inf" not in text
        s1 = next(line for line in lines if line.startswith("S "))
        assert s1.split()[1] == "0"

    def test_infinite_aperture_writes_no_cir(self, set_test_backend):
        from optiland.physical_apertures import RadialAperture

        lines = _encode(_lens_with_aperture(RadialAperture(r_max=float("inf"))))
        text = "\n".join(lines)
        assert "CIR" not in text
        assert "inf" not in text.lower()

    def test_finite_aperture_written(self, set_test_backend):
        from optiland.physical_apertures import RadialAperture

        lines = _encode(_lens_with_aperture(RadialAperture(5.0)))
        assert lines.count("  CIR CLR 5") == 1

    def test_nan_aperture_raises(self, set_test_backend):
        from optiland.physical_apertures import RadialAperture

        with pytest.raises(ValueError, match="non-finite"):
            _encode(_lens_with_aperture(RadialAperture(r_max=float("nan"))))

    @pytest.mark.parametrize("case", ["ideal_nan", "hikari_30um"])
    def test_nonfinite_glass_index_raises_and_leaves_file(
        self, case, tmp_path, set_test_backend
    ):
        import warnings

        from optiland.materials import IdealMaterial, Material

        if case == "ideal_nan":
            optic = _make_conic_sphere(material=IdealMaterial(float("nan")))
        else:
            optic = _make_conic_sphere(material=Material("SK16", "hikari"))
            optic.wavelengths.wavelengths.clear()
            optic.wavelengths.add(value=30.0, is_primary=True)
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(
                ValueError, match=r"Surface 1: non-finite index .* private glass"
            ):
                save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    @pytest.mark.parametrize("bad_n", [float("nan"), float("inf")])
    @pytest.mark.parametrize("bad_wl", [0.6563, 0.4861], ids=["first", "last"])
    def test_nonfinite_index_at_any_wavelength_raises_and_leaves_file(
        self, bad_n, bad_wl, tmp_path, set_test_backend
    ):
        import warnings

        import numpy as np

        from optiland.materials import IdealMaterial

        class _OddDispersion(IdealMaterial):
            """Finite index everywhere except at ``bad_wl``."""

            def n(self, wavelength, **kwargs):
                wl = np.asarray(wavelength, dtype=float)
                return np.where(np.isclose(wl, bad_wl), bad_n, 1.5)

        # written order: 0.6563, 0.55, 0.4861
        optic = _make_conic_sphere(material=_OddDispersion(1.5))
        optic.wavelengths.wavelengths.clear()
        optic.wavelengths.add(value=0.55, is_primary=True)
        optic.wavelengths.add(value=0.4861)
        optic.wavelengths.add(value=0.6563)
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(
                ValueError,
                match=rf"Surface 1: non-finite index .* private glass 'OPT1' "
                rf"at {bad_wl * 1000:g} nm",
            ):
                save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    def test_private_glass_without_row_raises(self):
        model = OpticToCodeVConverter(_make_conic_sphere()).convert()
        model.surfaces[1]["glass"] = {"private": "OPT9"}
        with pytest.raises(ValueError, match=r"Surface 1: private glass 'OPT9'"):
            CodeVFileEncoder(model).encode()

    @pytest.mark.parametrize("r_max", [float("nan"), float("-inf")])
    def test_nonfinite_aperture_names_surface_and_leaves_file(
        self, r_max, tmp_path, set_test_backend
    ):
        from optiland.physical_apertures import RadialAperture

        optic = _lens_with_aperture(RadialAperture(r_max=r_max))
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(ValueError, match=r"Surface 1: non-finite aperture r_max"):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    def test_fmt_rejects_nonfinite(self):
        from optiland.fileio.codev.writer.encoder import _fmt, _fmt_list

        with pytest.raises(ValueError, match="non-finite"):
            _fmt(float("inf"))
        with pytest.raises(ValueError, match="non-finite"):
            _fmt_list([1.0, float("nan")])


# ---------------------------------------------------------------------------
# CODE V glass names
# ---------------------------------------------------------------------------


def _surface_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith("S ")]


def _glass_token(line: str) -> str | None:
    """Fourth field of an ``S R T GLASS`` line, or None for air."""
    fields = line.split()
    return fields[3] if len(fields) > 3 else None


class TestGlassNames:
    @staticmethod
    def _lines_with(material) -> list[str]:
        return _encode(_make_conic_sphere(material=material))

    @pytest.mark.parametrize(
        "args, expected",
        [
            (("N-BK7", "schott"), "NBK7_SCHOTT"),
            (("N-SF11",), "NSF11_SCHOTT"),
            (("BK7",), "NBK7_SCHOTT"),
            (("SF5", "schott"), "NSF5_SCHOTT"),
            (("LAC9", "hoya"), "LAC9_HOYA"),
            (("S-BSL7", "ohara"), "SBSL7_OHARA"),
            (("H-K9L", "cdgm"), "HK9L_CDGM"),
            (("K-BK7", "sumita"), "KBK7_SUMITA"),
        ],
    )
    def test_name_from_resolved_file(self, args, expected, set_test_backend):
        from optiland.materials import Material

        lines = self._lines_with(Material(*args))
        assert _glass_token(_surface_lines(lines)[0]) == expected

    @pytest.mark.parametrize(
        "args",
        [("SK16", "hikari"), ("LITHOTEC-CAF2", "schott"), ("S-APL", "ohara")],
    )
    def test_unnamed_materials_use_private_glass_and_warn(self, args, set_test_backend):
        from optiland.fileio.codev.writer.glass_names import codev_glass_name
        from optiland.materials import Material

        mat = Material(*args)
        assert codev_glass_name(mat) is None
        with pytest.warns(UserWarning, match="private glass"):
            lines = self._lines_with(mat)
        token = _glass_token(_surface_lines(lines)[0])
        assert token == "'OPT1'"
        assert "HIKARI" not in "\n".join(lines).upper()

    def test_catalog_air_not_named(self, set_test_backend):
        from optiland.materials import Material

        with pytest.warns(UserWarning, match="private glass"):
            lines = self._lines_with(Material("air"))
        text = "\n".join(lines).upper()
        assert "AIR_" not in text
        assert "CIDDOR" not in text

    def test_filename_mapping_rules(self):
        from optiland.fileio.codev.writer.glass_names import (
            NOT_IN_CODEV,
            codev_name_for_filename,
        )

        assert codev_name_for_filename("glass/schott/N-BK7.yml") == ("NBK7", "SCHOTT")
        assert codev_name_for_filename(r"glass\schott\N-BK7.yml") == (
            "NBK7",
            "SCHOTT",
        )
        # sub-folders, other catalogs and non-glass folders are never named
        assert codev_name_for_filename("glass/schott/misc/ZERODUR.yml") is None
        assert codev_name_for_filename("glass/hikari/SK16.yml") is None
        assert codev_name_for_filename("other/mixed gases/air/Ciddor.yml") is None
        assert codev_name_for_filename("main/SiO2/Malitson.yml") is None
        # path-shape checks: each must go unnamed on its own
        assert codev_name_for_filename("glass/schott/N-BK7.txt") is None
        assert codev_name_for_filename("glass/schott/N-BK7") is None
        assert codev_name_for_filename("other/schott/N-BK7.yml") is None
        assert codev_name_for_filename("schott/misc/N-BK7.yml") is None
        assert codev_name_for_filename("glass/schott/misc/N-BK7.yml") is None
        assert codev_name_for_filename("glass/schott.yml") is None
        # the denylist is matched on the upper-cased stem
        assert codev_name_for_filename("glass/schott/sf5.yml") is None
        assert len(NOT_IN_CODEV) == 10
        for catalog, stem in NOT_IN_CODEV:
            assert codev_name_for_filename(f"glass/{catalog}/{stem}.yml") is None

    def test_non_material_classes_unnamed(self):
        from optiland.fileio.codev.writer.glass_names import codev_glass_name
        from optiland.materials import AbbeMaterial, IdealMaterial

        assert codev_glass_name(IdealMaterial(1.5)) is None
        assert codev_glass_name(AbbeMaterial(1.5, 60.0, model="buchdahl")) is None
        assert codev_glass_name("N-BK7") is None
        assert codev_glass_name(None) is None

    def test_material_without_catalog_file_unnamed(self):
        from optiland.fileio.codev.writer.glass_names import codev_glass_name
        from optiland.materials import Material

        mat = Material("N-BK7", "schott")
        mat.material_data = {}
        assert codev_glass_name(mat) is None

    def test_material_without_data_has_no_range(self):
        from optiland.fileio.codev.writer.glass_names import (
            wavelength_outside_range,
        )
        from optiland.materials import IdealMaterial

        assert wavelength_outside_range(IdealMaterial(1.5), [0.1, 10.0]) is None

    def test_name_charset_whole_db(self):
        import re

        from optiland.fileio.codev.writer.glass_names import (
            CODEV_CATALOGS,
            codev_name_for_filename,
        )
        from optiland.materials import Material

        named = 0
        for filename in Material._load_dataframe()["filename"]:
            result = codev_name_for_filename(filename)
            if result is None:
                continue
            named += 1
            assert re.fullmatch(r"[A-Z0-9]+", result[0]), filename
            assert result[1] in CODEV_CATALOGS.values()
        assert named > 100

    @pytest.mark.parametrize(
        "sample",
        [
            "CookeTriplet",
            "DoubleGauss",
            "EyepieceErfle",
            "ObjectiveUS008879901",
            "HubbleTelescope",
        ],
    )
    def test_samples_have_no_unsupported_tokens(
        self, sample, tmp_path, set_test_backend
    ):
        import re
        import warnings

        from optiland import samples

        cls = next(
            getattr(module, sample)
            for module in (
                samples.objectives,
                samples.eyepieces,
                samples.telescopes,
            )
            if hasattr(module, sample)
        )
        path = tmp_path / f"{sample}.seq"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            save_codev_file(cls(), str(path))
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        assert "inf" not in lowered
        assert "nan" not in lowered
        assert "mfl" not in lowered
        for line in _surface_lines(text.splitlines()):
            token = _glass_token(line)
            if token is None or token == "REFL":
                continue
            assert re.fullmatch(
                r"[A-Z0-9]+_(SCHOTT|OHARA|HOYA|CDGM|SUMITA)|'OPT\d+'", token
            ), line
            assert "-" not in token
            assert ":" not in token


# ---------------------------------------------------------------------------
# Non-refractive interaction models are refused
# ---------------------------------------------------------------------------


class TestInteractionRefusal:
    @staticmethod
    def _thin_lens_optic() -> Optic:
        optic = Optic()
        optic.surfaces.add(index=0, thickness=float("inf"))
        optic.surfaces.add(
            index=1, surface_type="paraxial", f=100.0, thickness=50.0, is_stop=True
        )
        optic.surfaces.add(index=2)
        optic.set_aperture(aperture_type="EPD", value=10.0)
        optic.fields.set_type("angle")
        optic.fields.add(y=0.0)
        optic.wavelengths.add(value=0.5876, is_primary=True)
        return optic

    def test_thin_lens_refused_and_file_untouched(self, tmp_path, set_test_backend):
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(
            NotImplementedError,
            match=r"Surface 1: thin-lens surfaces cannot be exported to CODE V yet",
        ):
            save_codev_file(self._thin_lens_optic(), str(path))
        assert path.read_bytes() == b"previous design"

    @pytest.mark.parametrize("interaction_type", ["diffractive", "phase"])
    def test_diffractive_and_phase_refused(
        self, interaction_type, monkeypatch, set_test_backend
    ):
        optic = _make_lens()
        monkeypatch.setattr(
            optic.surfaces[1].interaction_model, "interaction_type", interaction_type
        )
        with pytest.raises(
            NotImplementedError,
            match=rf"Surface 1: {interaction_type} surfaces cannot be exported",
        ):
            OpticToCodeVConverter(optic).convert()

    def test_refractive_and_mirror_still_exported(self, set_test_backend):
        lines = _encode(_make_lens())
        assert not any(line.strip().startswith("MFL") for line in lines)
        optic = Optic()
        optic.surfaces.add(index=0, thickness=10.0)
        optic.surfaces.add(
            index=1,
            radius=100.0,
            thickness=-50.0,
            material="mirror",
            is_stop=True,
        )
        optic.surfaces.add(index=2, thickness=0.0)
        _complete(optic)
        assert any("REFL" in line for line in _encode(optic))


# ---------------------------------------------------------------------------
# Wavelength weights reach the model
# ---------------------------------------------------------------------------


class TestWavelengthWeightsInModel:
    @staticmethod
    def _optic(weights, primary=0) -> Optic:
        optic = _make_lens()
        optic.wavelengths.wavelengths.clear()
        values = (0.5876, 0.4861, 0.6563)
        for i, (value, weight) in enumerate(zip(values, weights, strict=True)):
            optic.wavelengths.add(value=value, is_primary=i == primary, weight=weight)
        return optic

    def test_wavelength_weights_in_model(self, set_test_backend):
        model = OpticToCodeVConverter(self._optic((1.0, 2.0, 0.5))).convert()
        assert model.wavelengths["weights"] == [1.0, 2.0, 0.5]
        assert model.wavelengths["data"] == [0.5876, 0.4861, 0.6563]
        assert model.wavelengths["primary_index"] == 0

    def test_zero_weight_in_model(self, set_test_backend):
        model = OpticToCodeVConverter(self._optic((1.0, 0.0, 1.0))).convert()
        assert model.wavelengths["weights"] == [1.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# Wavelength order: CODE V sorts WL descending, then applies REF/WTW by position
# ---------------------------------------------------------------------------


def _cmd(lines: list[str], name: str) -> list[float]:
    """Numbers after the single ``name`` command in the emitted lines."""
    found = [line.split()[1:] for line in lines if line.split()[:1] == [name]]
    assert len(found) == 1, name
    return [float(tok) for tok in found[0]]


class TestWavelengthOrder:
    @staticmethod
    def _optic(values, primary, weights=None) -> Optic:
        optic = _make_lens()
        optic.wavelengths.wavelengths.clear()
        for i, value in enumerate(values):
            weight = 1.0 if weights is None else weights[i]
            optic.wavelengths.add(value=value, is_primary=i == primary, weight=weight)
        return optic

    def test_wl_written_descending(self, set_test_backend):
        lines = _encode(_make_singlet())
        assert lines.count("WL 656.3 587.6 486.1") == 1
        assert "REF 2" in lines

    @pytest.mark.parametrize(
        "primary", [0, 1, 2], ids=["shortest", "middle", "longest"]
    )
    def test_ref_points_at_primary(self, primary, set_test_backend):
        values = [0.4861, 0.5876, 0.6563]
        lines = _encode(self._optic(values, primary))
        ref = int(_cmd(lines, "REF")[0])
        assert_allclose(_cmd(lines, "WL")[ref - 1], values[primary] * 1000.0)
        # unsorted input must give the same answer
        shuffled = [values[2], values[0], values[1]]
        lines = _encode(self._optic(shuffled, shuffled.index(values[primary])))
        ref = int(_cmd(lines, "REF")[0])
        assert_allclose(_cmd(lines, "WL")[ref - 1], values[primary] * 1000.0)

    def test_wtw_permuted(self, set_test_backend):
        # weights: 587.6 -> 0.5, 486.1 -> 2, 656.3 -> 1
        optic = self._optic([0.5876, 0.4861, 0.6563], 0, weights=[0.5, 2.0, 1.0])
        lines = _encode(optic)
        assert _cmd(lines, "WL") == [656.3, 587.6, 486.1]
        assert _cmd(lines, "WTW") == [1.0, 0.5, 2.0]
        assert _cmd(lines, "REF") == [2.0]

    def test_equal_wavelengths_ref_by_index(self, set_test_backend):
        optic = self._optic([0.55, 0.55, 0.65], 1, weights=[1.0, 2.0, 3.0])
        lines = _encode(optic)
        assert _cmd(lines, "WL") == [650.0, 550.0, 550.0]
        assert _cmd(lines, "WTW") == [3.0, 1.0, 2.0]
        assert _cmd(lines, "REF") == [3.0]

    def test_weight_length_mismatch_raises(self, set_test_backend):
        model = OpticToCodeVConverter(_make_singlet()).convert()
        model.wavelengths["weights"] = [1.0, 1.0]
        with pytest.raises(ValueError, match="wavelength weights"):
            CodeVFileEncoder(model).encode()

    def test_model_not_mutated_by_encode(self, set_test_backend):
        import copy

        model = OpticToCodeVConverter(_make_singlet()).convert()
        before = copy.deepcopy(model.wavelengths)
        CodeVFileEncoder(model).encode()
        assert model.wavelengths == before
        assert model.wavelengths["data"][0] == 0.5876
        assert model.wavelengths["primary_index"] == 0

    def test_write_order_helper(self):
        from optiland.fileio.codev.writer.encoder import wavelength_write_order

        assert wavelength_write_order({"data": [0.5, 0.7, 0.5, 0.6]}) == [1, 3, 0, 2]
        assert wavelength_write_order({"data": []}) == []
        assert wavelength_write_order({}) == []

    def test_cooke_triplet_regression(self, set_test_backend):
        import warnings

        from optiland.samples.objectives import CookeTriplet

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lines = _encode(CookeTriplet())
        assert "WL 650 550 480" in lines
        assert "REF 2" in lines


# ---------------------------------------------------------------------------
# Private glass (PRV) for materials without a verified CODE V name
# ---------------------------------------------------------------------------


def _three_wavelengths(optic: Optic) -> Optic:
    """Primary 587.6 first, then 486.1 and 656.3 (not in descending order)."""
    optic.wavelengths.wavelengths.clear()
    optic.wavelengths.add(value=0.5876, is_primary=True)
    optic.wavelengths.add(value=0.4861)
    optic.wavelengths.add(value=0.6563)
    return optic


def _prv_block(lines: list[str]) -> list[str]:
    start = lines.index("PRV")
    return lines[start : lines.index("END", start) + 1]


def _row(line: str) -> tuple[str, list[float]]:
    name, *values = line.split()
    return name, [float(v) for v in values]


def _lens_with_materials(*materials) -> Optic:
    """Lens with one surface per material (the last gap is air) at 3 wavelengths."""
    optic = Optic()
    optic.surfaces.add(index=0, thickness=float("inf"))
    for i, material in enumerate(materials, start=1):
        optic.surfaces.add(
            index=i,
            radius=50.0 * (-1) ** i,
            thickness=5.0,
            material=material,
            is_stop=i == 1,
        )
    optic.surfaces.add(index=len(materials) + 1)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    return _three_wavelengths(optic)


def _n_at_written_wavelengths(mat) -> list[float]:
    import optiland.backend as be

    return [float(be.to_numpy(mat.n(nm / 1000.0))) for nm in (656.3, 587.6, 486.1)]


class TestPrivateGlass:
    @staticmethod
    def _encode_quiet(optic: Optic) -> list[str]:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _encode(optic)

    def test_ideal_prv_block(self, set_test_backend):
        from optiland.materials import IdealMaterial

        with pytest.warns(UserWarning, match="private glass"):
            lines = _encode(_lens_with_materials(IdealMaterial(1.7)))
        start = lines.index("PRV")
        assert lines[start : start + 4] == [
            "PRV",
            "  PWL 656.3 587.6 486.1",
            "  'OPT1' 1.7 1.7 1.7",
            "END",
        ]
        # after the header lines, before the object surface
        assert lines[start - 1].startswith("YAN")
        assert start < next(i for i, line in enumerate(lines) if line.startswith("SO "))
        assert _glass_token(_surface_lines(lines)[0]) == "'OPT1'"
        assert not any(":" in line for line in _surface_lines(lines))

    def test_abbe_prv_matches_optiland_n(self, set_test_backend):
        from optiland.materials import AbbeMaterial

        mat = AbbeMaterial(1.6, 50, "buchdahl")
        lines = self._encode_quiet(_lens_with_materials(mat))
        name, row = _row(_prv_block(lines)[2].strip())
        assert name == "'OPT1'"
        assert_allclose(row, _n_at_written_wavelengths(mat), rtol=1e-9)
        # not the constant nd the old Nd:Vd path would imply
        assert abs(row[0] - 1.6) > 1e-4
        assert abs(row[2] - 1.6) > 1e-4

    def test_abbe_polynomial_out_of_range(self, tmp_path, set_test_backend):
        from optiland.materials import AbbeMaterial

        optic = _lens_with_materials(AbbeMaterial(1.6, 50, "polynomial"))
        optic.wavelengths.add(value=1.0)
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(ValueError, match=r"Surface 1: material .* no index at 1.0"):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    def test_dedup_and_names(self, set_test_backend):
        from optiland.materials import IdealMaterial

        same = self._encode_quiet(
            _lens_with_materials(IdealMaterial(1.5), IdealMaterial(1.5))
        )
        assert _prv_block(same)[2:-1] == ["  'OPT1' 1.5 1.5 1.5"]
        assert [_glass_token(line) for line in _surface_lines(same)[:2]] == [
            "'OPT1'",
            "'OPT1'",
        ]
        distinct = self._encode_quiet(
            _lens_with_materials(
                IdealMaterial(1.5), IdealMaterial(1.6), IdealMaterial(1.7)
            )
        )
        names = [_row(line.strip())[0] for line in _prv_block(distinct)[2:-1]]
        assert names == ["'OPT1'", "'OPT2'", "'OPT3'"]
        assert [_glass_token(line) for line in _surface_lines(distinct)[:3]] == names

    def test_pwl_equals_wl(self, set_test_backend):
        from optiland.materials import IdealMaterial

        lines = self._encode_quiet(_lens_with_materials(IdealMaterial(1.5)))
        assert _cmd(_prv_block(lines)[1:2], "PWL") == _cmd(lines, "WL")
        assert _cmd(lines, "WL") == [656.3, 587.6, 486.1]

    def test_catalog_air_is_private(self, set_test_backend):
        import warnings

        from optiland.materials import Material

        mat = Material("air")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lines = _encode(_lens_with_materials(mat))
        assert not any("catalog air" in str(w.message) for w in caught)
        _, row = _row(_prv_block(lines)[2].strip())
        assert_allclose(row, _n_at_written_wavelengths(mat), rtol=1e-9)
        assert row[1] == pytest.approx(1.000277, abs=1e-6)
        assert _glass_token(_surface_lines(lines)[0]) == "'OPT1'"

    def test_hikari_is_private(self, set_test_backend):
        import warnings

        from optiland.materials import Material

        mat = Material("SK16", "hikari")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lines = _encode(_lens_with_materials(mat))
        messages = [str(w.message) for w in caught]
        assert any("private glass 'OPT1'" in m for m in messages)
        assert any("extinction k" in m and "not exported" in m for m in messages)
        _, row = _row(_prv_block(lines)[2].strip())
        assert_allclose(row, _n_at_written_wavelengths(mat), rtol=1e-9)

    def test_no_k_warning_without_extinction(self, set_test_backend):
        import warnings

        from optiland.materials import IdealMaterial

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _encode(_lens_with_materials(IdealMaterial(1.5)))
        assert not any("extinction" in str(w.message) for w in caught)

    def test_named_and_private_mixed(self, set_test_backend):
        from optiland.materials import IdealMaterial, Material

        lines = self._encode_quiet(
            _lens_with_materials(Material("N-BK7", "schott"), IdealMaterial(1.7))
        )
        tokens = [_glass_token(line) for line in _surface_lines(lines)[:2]]
        assert tokens == ["NBK7_SCHOTT", "'OPT1'"]
        assert len(_prv_block(lines)) == 4

    def test_no_prv_block_for_named_glass(self, set_test_backend):
        lines = _encode(_make_singlet())
        assert "PRV" not in lines
        assert "END" not in lines

    def test_ideal_absorption_still_refused(self, tmp_path, set_test_backend):
        from optiland.materials import IdealMaterial

        optic = _lens_with_materials(IdealMaterial(1.5, 0.1))
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(NotImplementedError, match="absorption"):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    def test_empty_and_object_only_optics_have_no_prv(self, set_test_backend):
        empty = OpticToCodeVConverter(Optic()).convert()
        assert empty.private_glasses == {}
        assert "PRV" not in CodeVFileEncoder(empty).encode()
        optic = Optic()
        optic.surfaces.add(index=0, thickness=1.0)
        model = OpticToCodeVConverter(optic).convert()
        assert model.private_glasses == {}
        assert "PRV" not in CodeVFileEncoder(model).encode()

    def test_no_wavelengths_with_private_glass_raises(self, tmp_path, set_test_backend):
        from optiland.materials import IdealMaterial

        optic = _lens_with_materials(IdealMaterial(1.5))
        optic.wavelengths.wavelengths.clear()
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(
            ValueError,
            match="The optic has no wavelengths",
        ):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    def test_private_glass_without_wavelengths_raises(self, set_test_backend):
        from optiland.materials import IdealMaterial

        converter = OpticToCodeVConverter(_lens_with_materials(IdealMaterial(1.5)))
        converter._private = {}
        converter._private_model = CodeVDataModel()
        with pytest.raises(ValueError, match="needs a private glass but the system"):
            converter._register_private_glass(IdealMaterial(1.5), 1)

    def test_converter_reuse_does_not_carry_private_glasses(self, set_test_backend):
        import warnings

        from optiland.materials import IdealMaterial

        converter = OpticToCodeVConverter(_lens_with_materials(IdealMaterial(1.5)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = converter.convert()
            converter._optic = _lens_with_materials(IdealMaterial(1.5))
            second = converter.convert()
        assert first.private_glasses == {"OPT1": [1.5, 1.5, 1.5]}
        assert second.private_glasses == {"OPT1": [1.5, 1.5, 1.5]}
        assert second.private_glasses is not first.private_glasses

    @pytest.mark.parametrize("bad_n", [0.0, -1.5])
    @pytest.mark.parametrize("bad_wl", [0.6563, 0.4861], ids=["first", "last"])
    def test_nonpositive_index_raises_and_leaves_file(
        self, bad_n, bad_wl, tmp_path, set_test_backend
    ):
        import warnings

        import numpy as np

        from optiland.materials import IdealMaterial

        class _OddDispersion(IdealMaterial):
            """Index 1.5 everywhere except at ``bad_wl``."""

            def n(self, wavelength, **kwargs):
                wl = np.asarray(wavelength, dtype=float)
                return np.where(np.isclose(wl, bad_wl), bad_n, 1.5)

        # written order: 0.6563, 0.55, 0.4861; the first surface takes OPT1
        optic = _lens_with_materials(IdealMaterial(1.4), _OddDispersion(1.5))
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(
                ValueError,
                match=rf"Surface 2: non-physical index \({bad_n!r}\) of private "
                rf"glass 'OPT2' at {bad_wl * 1000:g} nm",
            ):
                save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"


# ---------------------------------------------------------------------------
# A non-air object-space medium is refused
# ---------------------------------------------------------------------------


class TestObjectMediumRefusal:
    @staticmethod
    def _optic(object_material) -> Optic:
        from optiland.materials import IdealMaterial

        optic = Optic()
        optic.surfaces.add(index=0, thickness=10.0, material=object_material)
        optic.surfaces.add(
            index=1,
            radius=50.0,
            thickness=5.0,
            material=IdealMaterial(1.7),
            is_stop=True,
        )
        optic.surfaces.add(index=2)
        optic.set_aperture(aperture_type="EPD", value=10.0)
        optic.fields.set_type("angle")
        optic.fields.add(y=0.0)
        optic.wavelengths.add(value=0.5876, is_primary=True)
        return optic

    @pytest.mark.parametrize("kind", ["ideal", "catalog_air", "glass"])
    def test_object_medium_refused_and_file_untouched(
        self, kind, tmp_path, set_test_backend
    ):
        from optiland.materials import IdealMaterial, Material

        material = {
            "ideal": lambda: IdealMaterial(1.5151),
            "catalog_air": lambda: Material("air"),
            "glass": lambda: Material("N-BK7", "schott"),
        }[kind]()
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(
            NotImplementedError,
            match=r"^Surface 0: an object-space medium cannot be exported to CODE V "
            r"yet$",
        ):
            save_codev_file(self._optic(material), str(path))
        assert path.read_bytes() == b"previous design"

    def test_air_object_has_no_glass_token(self, set_test_backend):
        import warnings

        from optiland.materials import IdealMaterial

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lines = _encode(self._optic(IdealMaterial(1.0)))
        assert next(line for line in lines if line.startswith("SO ")) == "SO 0 10"

    def test_default_object_still_exported(self, set_test_backend):
        lines = _encode(_make_lens())
        assert any(line.startswith("SO ") for line in lines)

    def test_empty_and_object_only_optics_not_refused(self, set_test_backend):
        assert OpticToCodeVConverter(Optic()).convert().surfaces == {}
        optic = Optic()
        optic.surfaces.add(index=0, thickness=1.0)
        assert 0 in OpticToCodeVConverter(optic).convert().surfaces


# ---------------------------------------------------------------------------
# Decenters, tilts and chained coordinate systems are refused
# ---------------------------------------------------------------------------


def _decenter_lines(lines: list[str]) -> list[str]:
    """Lines that set a decenter or tilt (``XDE``, ``ADE``, ...)."""
    return [
        line for line in lines if line.split() and line.split()[0] in _DECENTER_KEYS
    ]


_DECENTER_KEYS = ("XDE", "YDE", "ADE", "BDE", "CDE", "ZDE")


class TestOffsetRefusal:
    @staticmethod
    def _converter(optic: Optic) -> OpticToCodeVConverter:
        return OpticToCodeVConverter(optic)

    @pytest.mark.parametrize("key", ["x", "y", "rx", "ry", "rz"])
    def test_lens_surface_offset_refused_and_file_untouched(
        self, key, tmp_path, set_test_backend
    ):
        optic = _make_lens()
        setattr(optic.surfaces[1].geometry.cs, key, 0.01)
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(
            NotImplementedError,
            match=rf"^Surface 1: decenters and tilts cannot be exported to CODE V "
            rf"yet \(non-zero {key}\)$",
        ):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    def test_decenter_through_the_public_api_refused(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=float("inf"))
        optic.surfaces.add(index=1, radius=50.0, thickness=5.0, dy=1.0, is_stop=True)
        optic.surfaces.add(index=2)
        with pytest.raises(NotImplementedError, match=r"Surface 1: .*non-zero y"):
            self._converter(optic).convert()

    def test_all_offending_keys_named(self, set_test_backend):
        optic = _make_lens()
        cs = optic.surfaces[2].geometry.cs
        cs.x = 0.5
        cs.rz = 0.1
        with pytest.raises(NotImplementedError, match=r"Surface 2: .*non-zero x, rz"):
            self._converter(optic).convert()

    def test_object_offset_refused(self, set_test_backend):
        optic = _make_lens()
        optic.surfaces[0].geometry.cs.y = 1.0
        with pytest.raises(NotImplementedError, match=r"Surface 0: decenters"):
            self._converter(optic).convert()

    def test_image_offset_refused(self, set_test_backend):
        optic = _make_lens()
        optic.surfaces[3].geometry.cs.rx = 0.1
        with pytest.raises(NotImplementedError, match=r"Surface 3: decenters"):
            self._converter(optic).convert()

    def test_chained_cs_refused(self, set_test_backend):
        optic = _make_lens()
        optic.surfaces[2].geometry.cs.reference_cs = optic.surfaces[1].geometry.cs
        with pytest.raises(
            NotImplementedError, match=r"Surface 2: chained coordinate systems"
        ):
            self._converter(optic).convert()

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
    def test_nonfinite_offset_value_error(self, value, tmp_path, set_test_backend):
        optic = _make_lens()
        optic.surfaces[1].geometry.cs.ry = value
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(ValueError, match=r"Surface 1: non-finite ry"):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    def test_offset_below_threshold_not_refused(self, set_test_backend):
        optic = _make_lens()
        optic.surfaces[1].geometry.cs.x = 1e-13
        self._converter(optic).convert()

    @pytest.mark.parametrize("key", ["x", "y", "rx", "ry", "rz"])
    def test_offset_at_the_threshold_writes_no_decenter_line(
        self, key, set_test_backend
    ):
        optic = _make_lens()
        setattr(optic.surfaces[1].geometry.cs, key, 1e-12)
        assert not _decenter_lines(_encode(optic))

    @pytest.mark.parametrize("value", [1.0000001e-12, 1e-11, 1e-10])
    def test_offset_just_above_the_threshold_refused(self, value, set_test_backend):
        optic = _make_lens()
        optic.surfaces[1].geometry.cs.x = value
        with pytest.raises(NotImplementedError, match=r"non-zero x"):
            self._converter(optic).convert()

    @pytest.mark.parametrize("value", [1e-12, -1e-12, 0.0])
    def test_offset_up_to_the_threshold_not_refused(self, value, set_test_backend):
        optic = _make_lens()
        optic.surfaces[1].geometry.cs.y = value
        self._converter(optic).convert()

    def test_propagation_check_precedes_the_offset_check(self, set_test_backend):
        from optiland.materials import IdealMaterial
        from optiland.propagation.homogeneous import HomogeneousPropagation

        class CustomPropagation(HomogeneousPropagation):
            pass

        optic = _make_lens()
        material = IdealMaterial(1.5)
        material.propagation_model = CustomPropagation(material)
        optic.surfaces[1].material_post = material
        optic.surfaces[2].geometry.cs.x = 1.0  # also violates the offset check
        with pytest.raises(NotImplementedError, match="custom propagation"):
            self._converter(optic).convert()

    def test_object_medium_check_precedes_the_offset_check(self, set_test_backend):
        from optiland.materials import IdealMaterial

        optic = _make_lens()
        optic.surfaces[0].material_post = IdealMaterial(1.5)
        optic.surfaces[2].geometry.cs.x = 1.0
        with pytest.raises(NotImplementedError, match="object-space medium"):
            self._converter(optic).convert()

    def test_propagation_check_precedes_the_object_medium_check(self, set_test_backend):
        from optiland.materials import IdealMaterial
        from optiland.propagation.homogeneous import HomogeneousPropagation

        class CustomPropagation(HomogeneousPropagation):
            pass

        optic = _make_lens()
        optic.surfaces[0].material_post = IdealMaterial(1.5)  # object medium
        material = IdealMaterial(1.5)
        material.propagation_model = CustomPropagation(material)
        optic.surfaces[1].material_post = material  # custom propagation
        with pytest.raises(NotImplementedError, match="custom propagation"):
            self._converter(optic).convert()

    def test_axial_position_is_not_an_offset(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=float("inf"))
        optic.surfaces.add(index=1, radius=50.0, z=0.0, is_stop=True)
        optic.surfaces.add(index=2, z=30.0)
        self._converter(_complete(optic)).convert()

    def test_empty_and_object_only_optics_not_refused(self, set_test_backend):
        assert self._converter(Optic()).convert().surfaces == {}
        optic = Optic()
        optic.surfaces.add(index=0, thickness=1.0)
        assert 0 in self._converter(optic).convert().surfaces

    def test_all_samples_pass_offset_check(self, set_test_backend):
        import inspect
        import warnings

        from optiland import samples

        checked = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for module in vars(samples).values():
                if not inspect.ismodule(module):
                    continue
                for cls in vars(module).values():
                    if not (
                        inspect.isclass(cls)
                        and issubclass(cls, Optic)
                        and cls.__module__ == module.__name__
                    ):
                        continue
                    try:
                        lines = CodeVFileEncoder(
                            self._converter(cls()).convert()
                        ).encode()
                    except NotImplementedError as exc:
                        # other refusals (asphere, ...) are not the offset guard
                        assert "decenters and tilts" not in str(exc), cls.__name__
                        assert "chained" not in str(exc), cls.__name__
                    else:
                        assert not _decenter_lines(lines), cls.__name__
                    checked += 1
        assert checked >= 25


# ---------------------------------------------------------------------------
# Thickness is the global z spacing of the vertices
# ---------------------------------------------------------------------------


def _thi_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith(("SO ", "S ", "SI "))]


def _thi(lines: list[str]) -> list[float]:
    """The thickness token of each SO / S / SI line, in file order."""
    return [float(line.split()[2]) for line in _thi_lines(lines)]


def _gcs_z(optic: Optic) -> list[float]:
    return [float(s.geometry.cs.position_in_gcs[2]) for s in optic.surfaces]


class TestThicknessFromGlobalZ:
    @staticmethod
    def _finish(optic: Optic) -> Optic:
        optic.set_aperture(aperture_type="EPD", value=10.0)
        optic.fields.set_type("angle")
        optic.fields.add(y=0.0)
        optic.wavelengths.add(value=0.5876, is_primary=True)
        return optic

    def test_absolute_z_lens(self, set_test_backend):
        from optiland.materials import IdealMaterial

        optic = Optic()
        optic.surfaces.add(index=0, z=-100.0)
        optic.surfaces.add(
            index=1,
            radius=50.0,
            z=0.0,
            material=IdealMaterial(1.5),
            is_stop=True,
        )
        optic.surfaces.add(index=2, radius=-50.0, z=5.0)
        optic.surfaces.add(index=3, z=45.0)
        self._finish(optic)
        # nothing is stored as a thickness
        assert all(float(s.thickness) == 0.0 for s in optic.surfaces)
        assert _thi(_encode(optic)) == [100.0, 5.0, 40.0, 0.0]

    def test_absolute_z_with_object_at_infinity(self, set_test_backend):
        from optiland.materials import IdealMaterial

        optic = Optic()
        optic.surfaces.add(index=0, thickness=float("inf"))
        optic.surfaces.add(
            index=1,
            radius=50.0,
            z=0.0,
            material=IdealMaterial(1.5),
            is_stop=True,
        )
        optic.surfaces.add(index=2, radius=-50.0, z=5.0)
        optic.surfaces.add(index=3, z=45.0)
        self._finish(optic)
        assert _thi(_encode(optic)) == [1e10, 5.0, 40.0, 0.0]

    def test_thickness_solve_exported(self, set_test_backend):
        optic = _make_lens()
        optic.solves.add("marginal_ray_height", 2, height=0.0)
        optic.solves.apply()
        z = _gcs_z(optic)
        with pytest.warns(UserWarning, match="solve"):
            lines = _encode(optic)
        written = _thi(lines)
        assert_allclose(written[1], z[2] - z[1], rtol=1e-7)
        assert_allclose(written[2], z[3] - z[2], rtol=1e-7)
        # the solved value, not the thickness the surface was created with
        assert abs(written[1] - 5.0) > 1.0

    def test_float32_relative_lens_writes_exact_thicknesses(self):
        torch = pytest.importorskip("torch")
        import optiland.backend as be

        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision("float32")
        try:
            optic = Optic()
            optic.surfaces.add(index=0, thickness=float("inf"))
            for i, t in enumerate([0.3, 10.1, 0.7, 123.456, 0.1], start=1):
                optic.surfaces.add(
                    index=i, radius=50.0 * (-1) ** i, thickness=t, is_stop=(i == 1)
                )
            optic.surfaces.add(index=6)
            self._finish(optic)
            assert optic.surfaces.global_z_positions.dtype == torch.float32
            written = _thi(_encode(optic))
        finally:
            be.set_precision("float64")
            be.set_backend("numpy")
        # the attributes, not the float32 cumulative-sum differences
        assert written == [1e10, 0.3, 10.1, 0.7, 123.456, 0.1, 0.0]

    def test_thickness_attribute_preferred_only_within_tolerance(
        self, set_test_backend
    ):
        optic = _make_lens()
        # the vertex moved by 1e-3 while the attribute stayed at 5.0
        optic.surfaces[2].geometry.cs.z = optic.surfaces[2].geometry.cs.z + 1e-3
        assert _thi(_encode(optic))[1] == 5.001

    @staticmethod
    def _far_lens() -> Optic:
        """Relative lens whose third vertex sits at z ~ 200 mm."""
        optic = Optic()
        optic.surfaces.add(index=0, thickness=float("inf"))
        for i, t in enumerate([200.0, 0.1, 3.0], start=1):
            optic.surfaces.add(index=i, radius=50.0 * (-1) ** i, thickness=t)
        optic.surfaces.add(index=4)
        return optic

    @pytest.mark.parametrize("dz", [1e-3, 2e-4, 1e-5, 1e-6])
    def test_small_vertex_shift_at_z200_is_written(self, dz, set_test_backend):
        optic = self._far_lens()
        optic.surfaces[3].geometry.cs.z = optic.surfaces[3].geometry.cs.z + dz
        spacings = OpticToCodeVConverter(optic)._vertex_spacings()
        assert spacings[2] == pytest.approx(0.1 + dz, rel=0, abs=1e-10)
        assert spacings[3] == pytest.approx(3.0 - dz, rel=0, abs=1e-10)

    @pytest.mark.parametrize("gap", [1e-3, 2e-4, 1e-4, 1e-5])
    def test_small_absolute_z_gap_at_z200_is_not_zeroed(self, gap, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=float("inf"))
        optic.surfaces.add(index=1, radius=50.0, z=200.0, is_stop=True)
        optic.surfaces.add(index=2, radius=-50.0, z=200.0 + gap)
        optic.surfaces.add(index=3, z=210.0)
        assert all(float(s.thickness) == 0.0 for s in optic.surfaces[1:])
        spacings = OpticToCodeVConverter(optic)._vertex_spacings()
        assert spacings[1] == pytest.approx(gap, rel=0, abs=1e-10)

    def test_relative_lens_unchanged(self, set_test_backend):
        assert _thi(_encode(_make_lens())) == [1e10, 5.0, 45.0, 0.0]

    def test_axial_mirror_negative_thickness(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=float("inf"))
        optic.surfaces.add(
            index=1, radius=-100.0, thickness=-50.0, material="mirror", is_stop=True
        )
        optic.surfaces.add(index=2)
        self._finish(optic)
        z = _gcs_z(optic)
        written = _thi(_encode(optic))
        assert written[1] < 0
        assert_allclose(written[1], z[2] - z[1], rtol=1e-7)

    def test_mirror_then_lens_follows_the_fold_back(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=float("inf"))
        optic.surfaces.add(
            index=1, radius=-100.0, thickness=-20.0, material="mirror", is_stop=True
        )
        optic.surfaces.add(index=2, thickness=-30.0)
        optic.surfaces.add(index=3)
        self._finish(optic)
        z = _gcs_z(optic)
        written = _thi(_encode(optic))
        assert_allclose(written[1:3], [z[2] - z[1], z[3] - z[2]], rtol=1e-7)
        assert written[1:3] == [-20.0, -30.0]

    def test_no_negative_zero(self, set_test_backend):
        optic = _make_lens()
        optic.surfaces[2].geometry.cs.z = optic.surfaces[1].geometry.cs.z - 1e-14
        lines = _thi_lines(_encode(optic))
        assert all("-0 " not in line + " " for line in lines)
        assert lines[1].split()[2] == "0"

    @pytest.mark.parametrize(
        "gap, expected",
        [(1e-7, 1e-7), (1e-9, 1e-9), (2e-12, 2e-12), (5e-13, 0.0), (-5e-13, 0.0)],
    )
    def test_spacing_zeroed_only_below_1e_12(self, gap, expected, set_test_backend):
        optic = _make_lens()
        optic.surfaces[2].geometry.cs.z = optic.surfaces[1].geometry.cs.z + gap
        token = _thi_lines(_encode(optic))[1].split()[2]
        assert float(token) == pytest.approx(expected, rel=1e-6, abs=0.0)
        if expected == 0.0:
            assert token == "0"

    def test_zeroing_boundary_is_inclusive_at_1e_12(self, set_test_backend):
        optic = _make_lens()
        base = optic.surfaces[1].geometry.cs.z
        optic.surfaces[2].geometry.cs.z = base + 1e-12
        # strictly below 1e-12 is zeroed; a spacing of exactly 1e-12 is written
        assert OpticToCodeVConverter(optic)._vertex_spacings()[1] == 1e-12
        optic.surfaces[2].geometry.cs.z = base + 1.0000001e-12
        assert OpticToCodeVConverter(optic)._vertex_spacings()[1] > 1e-12

    def test_empty_and_object_only_optics(self, set_test_backend):
        assert OpticToCodeVConverter(Optic()).convert().surfaces == {}
        optic = Optic()
        optic.surfaces.add(index=0, thickness=7.0)
        assert OpticToCodeVConverter(optic).convert().surfaces[0]["thickness"] == 7.0

    def test_object_and_image_only(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=12.0)
        optic.surfaces.add(index=1)
        assert _thi(_encode(optic)) == [12.0, 0.0]

    def test_samples_unchanged(self, set_test_backend):
        import inspect
        import math
        import warnings

        from optiland import samples

        checked = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for module in vars(samples).values():
                if not inspect.ismodule(module):
                    continue
                for cls in vars(module).values():
                    if not (
                        inspect.isclass(cls)
                        and issubclass(cls, Optic)
                        and cls.__module__ == module.__name__
                    ):
                        continue
                    optic = cls()
                    try:
                        written = _thi(_encode(optic))
                    except NotImplementedError:
                        continue  # refused for another reason (asphere, ...)
                    attributes = [float(s.thickness) for s in optic.surfaces]
                    expected = [
                        1e10 if math.isinf(t) else t for t in attributes[:-1]
                    ] + [0.0]
                    assert_allclose(written, expected, rtol=1e-6, atol=1e-9)
                    checked += 1
        assert checked >= 20


# ---------------------------------------------------------------------------
# The image surface keeps its radius; a conic or asphere there is refused
# ---------------------------------------------------------------------------


def _lens_with_image(**image) -> Optic:
    """Object at infinity, one stop surface, and an image surface from ``image``."""
    optic = Optic()
    optic.surfaces.add(index=0, thickness=float("inf"))
    optic.surfaces.add(index=1, radius=50.0, thickness=45.0, is_stop=True)
    optic.surfaces.add(index=2, **image)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.fields.set_type("angle")
    optic.fields.add(y=0.0)
    optic.wavelengths.add(value=0.5876, is_primary=True)
    return optic


class TestImageSurface:
    def test_planar_image_writes_zero_radius(self, set_test_backend):
        lines = _encode(_lens_with_image())
        assert [line for line in lines if line.startswith("SI")] == ["SI 0 0"]

    @pytest.mark.parametrize("radius", [-70.0, 35.5])
    def test_image_radius_written(self, radius, set_test_backend):
        lines = _encode(_lens_with_image(radius=radius))
        assert [line for line in lines if line.startswith("SI")] == [f"SI {radius:g} 0"]

    def test_image_radius_does_not_change_other_lines(self, set_test_backend):
        flat = _encode(_lens_with_image())
        curved = _encode(_lens_with_image(radius=-70.0))
        assert [line for line in flat if not line.startswith("SI")] == [
            line for line in curved if not line.startswith("SI")
        ]

    @pytest.mark.parametrize(
        "image, message",
        [
            ({"radius": -70.0, "conic": -0.5}, "an image surface conic"),
            (
                {"radius": -70.0, "surface_type": "even_asphere"},
                "an aspheric image surface",
            ),
            (
                {
                    "radius": -70.0,
                    "surface_type": "even_asphere",
                    "coefficients": [1e-5],
                },
                "an aspheric image surface",
            ),
            (
                {"surface_type": "biconic", "radius_x": -70.0, "radius_y": -60.0},
                "image surface geometry 'Biconic'",
            ),
        ],
    )
    def test_image_geometry_refused_and_file_untouched(
        self, image, message, tmp_path, set_test_backend
    ):
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(
            NotImplementedError,
            match=rf"^Surface 2: {message} (cannot be exported to CODE V yet)$",
        ):
            save_codev_file(_lens_with_image(**image), str(path))
        assert path.read_bytes() == b"previous design"

    def test_nan_image_conic_refused(self, set_test_backend):
        with pytest.raises(NotImplementedError, match="image surface conic"):
            _encode(_lens_with_image(radius=-70.0, conic=float("nan")))

    @pytest.mark.parametrize("conic", [0.0, 1e-17, -1e-16, 1e-16])
    def test_image_conic_up_to_the_tolerance_not_refused(self, conic, set_test_backend):
        lines = _encode(_lens_with_image(radius=-70.0, conic=conic))
        assert [line for line in lines if line.startswith("SI")] == ["SI -70 0"]

    @pytest.mark.parametrize("conic", [1e-15, -1e-15, 1e-9, 1e-3])
    def test_image_conic_above_the_tolerance_refused(self, conic, set_test_backend):
        with pytest.raises(NotImplementedError, match="image surface conic"):
            _encode(_lens_with_image(radius=-70.0, conic=conic))

    def test_nonfinite_image_radius_raises(self, set_test_backend):
        with pytest.raises(ValueError, match="Surface 2: non-finite radius"):
            _encode(_lens_with_image(radius=float("nan")))

    def test_object_only_optic_keeps_a_plane_image(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=5.0)
        model = OpticToCodeVConverter(optic).convert()
        assert model.surfaces[1]["radius"] == 0.0

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("HubbleTelescope", "SI -635.38227 0"),
            ("NavarroWideAngleEye", "SI -12 0"),
        ],
    )
    def test_sample_image_radii(self, name, expected, set_test_backend):
        import warnings

        from optiland import samples

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lines = _encode(getattr(samples, name)())
        assert [line for line in lines if line.startswith("SI")] == [expected]


# ---------------------------------------------------------------------------
# Surface apertures
# ---------------------------------------------------------------------------


def _ap_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if "CIR" in line]


class TestSurfaceApertures:
    @pytest.mark.parametrize(
        ("r_max", "r_min", "expected"),
        [
            (5.0, 0.0, ["  CIR CLR 5"]),
            (5.0, 2.0, ["  CIR CLR 5", "  CIR OBS 2"]),
            (float("inf"), 2.0, ["  CIR OBS 2"]),
            (float("inf"), 0.0, []),
        ],
    )
    def test_radial_aperture_lines(self, r_max, r_min, expected, set_test_backend):
        from optiland.physical_apertures import RadialAperture

        lines = _encode(_lens_with_aperture(RadialAperture(r_max, r_min)))
        assert _ap_lines(lines) == expected

    @pytest.mark.parametrize(
        ("r_max", "r_min"),
        [
            (2.0, 2.0),
            (2.0, 3.0),
            (-1.0, 0.0),
            (5.0, -1.0),
            (5.0, float("nan")),
            (5.0, float("inf")),
        ],
    )
    def test_invalid_radial_aperture_raises_and_leaves_file(
        self, r_max, r_min, tmp_path, set_test_backend
    ):
        from optiland.physical_apertures import RadialAperture

        optic = _lens_with_aperture(RadialAperture(r_max, r_min))
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(ValueError, match=r"Surface 1: invalid aperture"):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    @pytest.mark.parametrize(
        "make",
        [
            lambda m: m.OffsetRadialAperture(5.0, 0.0, 1.0, -2.0),
            lambda m: m.RectangularAperture(-3, 3, -2, 2),
            lambda m: m.EllipticalAperture(4, 2),
            lambda m: m.PolygonAperture([0, 1, 1], [0, 0, 1]),
            lambda m: m.UnionAperture(m.RadialAperture(5.0), m.RadialAperture(3.0)),
            lambda m: m.RotatedAperture(m.RadialAperture(5.0), 10.0),
        ],
    )
    def test_other_apertures_warn_and_write_nothing(self, make, set_test_backend):
        import optiland.physical_apertures as m

        aperture = make(m)
        name = type(aperture).__name__
        with pytest.warns(UserWarning, match=f"aperture {name} has no verified CODE V"):
            lines = _encode(_lens_with_aperture(aperture))
        assert _ap_lines(lines) == []

    def test_unclipped_aperture_is_silent(self, set_test_backend):
        import warnings

        from optiland.physical_apertures import RadialAperture, UnclippedAperture

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lines = _encode(_lens_with_aperture(UnclippedAperture(RadialAperture(5.0))))
        assert _ap_lines(lines) == []

    @pytest.mark.parametrize("index", [0, 3])
    def test_end_surface_aperture_warns_and_writes_nothing(
        self, index, set_test_backend
    ):
        from optiland.physical_apertures import RadialAperture

        optic = _make_lens()
        optic.surfaces[index].aperture = RadialAperture(5.0, 1.0)
        kind = "object" if index == 0 else "image"
        with pytest.warns(
            UserWarning,
            match=rf"Surface {index}: an aperture on the {kind} surface is not "
            "exported to CODE V",
        ) as record:
            lines = _encode(optic)
        assert not any("no verified" in str(w.message) for w in record)
        assert _ap_lines(lines) == []

    def test_empty_and_one_surface_optics_do_not_raise(self, set_test_backend):
        assert OpticToCodeVConverter(Optic()).convert().surfaces == {}
        optic = Optic()
        optic.surfaces.add(index=0, thickness=float("inf"))
        OpticToCodeVConverter(optic).convert()


# ---------------------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------------------


def _lens_with_field_type(field_type: str, ys=(0.0, 5.0)) -> Optic:
    optic = _make_lens()
    optic.fields.set_type(field_type)
    optic.fields.fields.clear()
    for y in ys:
        optic.fields.add(y=y)
    return optic


class TestFieldDefinitions:
    @pytest.mark.parametrize(
        ("field_type", "command", "absent"),
        [
            ("real_image_height", "YRI", ("YIM", "YAN", "YOB")),
            ("paraxial_image_height", "YIM", ("YRI", "YAN", "YOB")),
            ("object_height", "YOB", ("YRI", "YIM", "YAN")),
            ("angle", "YAN", ("YRI", "YIM", "YOB")),
        ],
    )
    def test_field_type_command(self, field_type, command, absent, set_test_backend):
        lines = _encode(_lens_with_field_type(field_type))
        assert f"{command} 0 5" in lines
        for other in absent:
            assert not any(line.startswith(other) for line in lines)

    def test_real_image_height_x_field_writes_xri(self, set_test_backend):
        optic = _lens_with_field_type("real_image_height")
        optic.fields.add(x=2.0, y=1.0)
        lines = _encode(optic)
        assert "XRI 0 0 2" in lines
        assert "YRI 0 5 1" in lines

    def test_unknown_field_definition_refused(self, tmp_path, set_test_backend):
        class MysteryField:
            pass

        optic = _make_lens()
        optic.fields.field_definition = MysteryField()
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(NotImplementedError, match="MysteryField"):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    def test_field_definition_none_defaults_to_angle(self, set_test_backend):
        optic = _make_lens()
        optic.fields.field_definition = None
        assert "YAN 0" in _encode(optic)

    def test_encoder_refuses_unknown_field_type(self):
        model = CodeVDataModel()
        model.fields = {"type": "mystery", "x": [0.0], "y": [1.0]}
        with pytest.raises(ValueError, match="unknown field type 'mystery'"):
            CodeVFileEncoder(model).encode()


# ---------------------------------------------------------------------------
# Field vignetting
# ---------------------------------------------------------------------------


class TestVignettingWarning:
    @pytest.mark.parametrize("axis", ["vx", "vy"])
    def test_nonzero_vignetting_warns_and_writes_no_vu_lines(
        self, axis, set_test_backend
    ):
        optic = _make_lens()
        optic.fields.add(y=5.0, **{axis: 0.2})
        with pytest.warns(UserWarning, match="vignetting factors"):
            lines = _encode(optic)
        assert not any(line.startswith(("VU", "VL")) for line in lines)

    @pytest.mark.parametrize("axis", ["vx", "vy"])
    def test_nan_vignetting_warns(self, axis, set_test_backend):
        optic = _make_lens()
        optic.fields.add(y=5.0, **{axis: float("nan")})
        with pytest.warns(UserWarning, match="vignetting factors"):
            _encode(optic)

    def test_no_vignetting_no_warning(self, set_test_backend):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _encode(_make_lens())


# ---------------------------------------------------------------------------
# float_by_stop_size aperture
# ---------------------------------------------------------------------------


class TestFloatByStopAperture:
    def test_written_as_fixed_epd_with_warning(self, set_test_backend):
        import warnings

        import optiland.backend as be
        from optiland.samples.objectives import CookeTriplet

        optic = CookeTriplet()
        optic.set_aperture(aperture_type="float_by_stop_size", value=8.0)
        expected = float(be.to_numpy(be.atleast_1d(optic.paraxial.EPD())).ravel()[0])
        with pytest.warns(
            UserWarning, match=r"Aperture float_by_stop_size \(stop diameter"
        ) as record:
            model = OpticToCodeVConverter(optic).convert()
        assert any("fixed EPD" in str(w.message) for w in record)
        assert isinstance(model.aperture["EPD"], float)
        assert_allclose(model.aperture["EPD"], expected)
        assert abs(model.aperture["EPD"] - 8.0) > 0.1
        assert list(model.aperture) == ["EPD"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lines = CodeVFileEncoder(model).encode()
        epd_lines = [line for line in lines if line.startswith("EPD ")]
        assert len(epd_lines) == 1
        assert float(epd_lines[0].split()[1]) == model.aperture["EPD"]

    def test_epd_aperture_no_warning(self, set_test_backend):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = OpticToCodeVConverter(_make_lens()).convert()
        assert_allclose(model.aperture["EPD"], 10.0)


# ---------------------------------------------------------------------------
# Full-precision numbers
# ---------------------------------------------------------------------------


class TestFullPrecisionNumbers:
    def test_hubble_lines_keep_every_digit(self, set_test_backend):
        import warnings

        from optiland import samples

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lines = _encode(samples.HubbleTelescope())
        s_lines = [line for line in lines if line.startswith("S ")]
        for token in ("-11040.02286", "4910.01016", "6365.20955", "-1349.31166"):
            assert any(token in line.split() for line in s_lines), token

    def test_long_float_round_trips_exactly(self):
        from optiland.fileio.codev.writer.encoder import _fmt

        value = 50.123456789012345
        assert float(_fmt(value)) == value

    @pytest.mark.parametrize("value", [0.0, -0.0])
    def test_zero_is_written_without_sign(self, value):
        from optiland.fileio.codev.writer.encoder import _fmt

        assert _fmt(value) == "0"

    @pytest.mark.parametrize(
        "value, text",
        [(5.0, "5"), (0.1, "0.1"), (1e10, "1e+10"), (-12.0, "-12")],
    )
    def test_short_values_stay_short(self, value, text):
        from optiland.fileio.codev.writer.encoder import _fmt

        assert _fmt(value) == text

    @pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
    def test_non_finite_still_refused(self, value):
        from optiland.fileio.codev.writer.encoder import _fmt

        with pytest.raises(ValueError, match="non-finite"):
            _fmt(value)

    def test_wavelengths_carry_no_binary_noise(self, set_test_backend):
        import warnings

        from optiland.materials import IdealMaterial

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lines = _encode(_lens_with_materials(IdealMaterial(1.5)))
        wl_lines = [line for line in lines if line.startswith("WL ")]
        assert wl_lines == ["WL 656.3 587.6 486.1"]


# ---------------------------------------------------------------------------
# CODE V count limits (21 wavelengths, 25 fields; CODE V 2025.03)
# ---------------------------------------------------------------------------


def _lens_with_counts(n_wavelengths: int, n_fields: int) -> Optic:
    optic = _make_lens()
    optic.wavelengths.wavelengths.clear()
    for i in range(n_wavelengths):
        optic.wavelengths.add(value=0.5 + 0.01 * i, is_primary=i == 0)
    optic.fields.fields.clear()
    for i in range(n_fields):
        optic.fields.add(y=float(i))
    return optic


class TestCountLimits:
    def test_limits_export(self, set_test_backend):
        lines = _encode(_lens_with_counts(21, 25))
        wl = next(line for line in lines if line.startswith("WL "))
        yan = next(line for line in lines if line.startswith("YAN "))
        assert len(wl.split()) - 1 == 21
        assert len(yan.split()) - 1 == 25

    @pytest.mark.parametrize(
        ("counts", "message"),
        [
            ((22, 1), "CODE V supports at most 21 wavelengths; this optic has 22"),
            ((1, 26), "CODE V supports at most 25 fields; this optic has 26"),
        ],
    )
    def test_over_limit_refused(self, counts, message, tmp_path, set_test_backend):
        optic = _lens_with_counts(*counts)
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(ValueError, match=message):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"


# ---------------------------------------------------------------------------
# The aperture stop: one STO, on the surface Optiland resolves as the stop
# ---------------------------------------------------------------------------


def _sto_surfaces(lines: list[str]) -> list[int]:
    """Return the output surface numbers (0 = object) followed by an ``STO``."""
    numbers = []
    surface = -1
    for line in lines:
        if line.startswith(("SO ", "S ", "SI ")):
            surface += 1
        elif line.strip() == "STO":
            numbers.append(surface)
    return numbers


class TestStopSurface:
    def test_single_stop_is_written_once(self, set_test_backend):
        assert _sto_surfaces(_encode(_make_lens())) == [1]

    def test_several_flags_write_sto_only_at_stop_index(self, set_test_backend):
        optic = _make_lens()
        optic.surfaces[2].is_stop = True  # a second flagged surface
        assert [s.is_stop for s in optic.surfaces] == [False, True, True, False]
        stop_index = optic.surfaces.stop_index
        assert stop_index == 1
        assert _sto_surfaces(_encode(optic)) == [stop_index]

    def test_stop_follows_stop_index_setter(self, set_test_backend):
        optic = _make_lens()
        optic.surfaces.stop_index = 2
        assert _sto_surfaces(_encode(optic)) == [2]

    @pytest.mark.parametrize("which", ["no stop", "object", "image"])
    def test_refused_stop_leaves_existing_file_untouched(
        self, which, tmp_path, set_test_backend
    ):
        optic = _make_lens()
        optic.surfaces[1].is_stop = False
        if which == "object":
            optic.surfaces[0].is_stop = True
            match = r"Surface 0: the aperture stop on the object surface cannot be"
            error = NotImplementedError
        elif which == "image":
            optic.surfaces[-1].is_stop = True
            match = r"Surface 3: the aperture stop on the image surface cannot be"
            error = NotImplementedError
        else:
            match = "no aperture stop surface"
            error = ValueError
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(error, match=match):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    def test_optic_without_real_surface_needs_no_stop(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=1.0)
        optic.surfaces.add(index=1)
        assert _sto_surfaces(_encode(optic)) == []


# ---------------------------------------------------------------------------
# imageFNO at a finite object is written as a fixed EPD (CODE V reads FNO as
# the working F-number there)
# ---------------------------------------------------------------------------


class TestImageFnoFiniteObject:
    @staticmethod
    def _fno_lens(object_thickness: float) -> Optic:
        optic = _make_lens(object_thickness=object_thickness)
        optic.set_aperture(aperture_type="imageFNO", value=4.0)
        return optic

    def test_finite_object_writes_epd_with_warning(self, set_test_backend):
        import optiland.backend as be

        optic = self._fno_lens(300.0)
        with pytest.warns(UserWarning, match=r"imageFNO .* finite distance .* EPD"):
            lines = _encode(optic)
        epd = float(be.to_numpy(optic.paraxial.EPD()))
        epd_lines = [line for line in lines if line.startswith("EPD ")]
        assert len(epd_lines) == 1
        assert_allclose(float(epd_lines[0].split()[1]), epd, rtol=1e-12)
        assert not any(line.startswith("FNO") for line in lines)

    def test_finite_object_epd_has_full_precision(self, set_test_backend):
        import optiland.backend as be

        optic = self._fno_lens(300.0)
        with pytest.warns(UserWarning, match="imageFNO"):
            model = OpticToCodeVConverter(optic).convert()
        assert model.aperture == {"EPD": float(be.to_numpy(optic.paraxial.EPD()))}

    def test_infinite_object_keeps_fno(self, set_test_backend, recwarn):
        lines = _encode(self._fno_lens(float("inf")))
        assert "FNO 4" in lines
        assert not any(line.startswith("EPD") for line in lines)
        assert not [w for w in recwarn if "imageFNO" in str(w.message)]

    def test_object_at_codev_infinity_keeps_fno(self, set_test_backend):
        assert "FNO 4" in _encode(self._fno_lens(1e10))

    def test_epd_aperture_at_finite_object_is_silent(self, set_test_backend, recwarn):
        lines = _encode(_make_lens(object_thickness=300.0))
        assert "EPD 10" in lines
        assert not [w for w in recwarn if "imageFNO" in str(w.message)]


# ---------------------------------------------------------------------------
# A catalog glass is named only if every system wavelength is in its range
# ---------------------------------------------------------------------------


class TestGlassWavelengthRange:
    @staticmethod
    def _lens(*wavelengths: float) -> Optic:
        optic = _make_lens()
        optic.wavelengths.wavelengths.clear()
        for i, wavelength in enumerate(wavelengths):
            optic.wavelengths.add(value=wavelength, is_primary=i == 0)
        return optic

    @staticmethod
    def _s1(lines: list[str]) -> str:
        return next(line for line in lines if line.startswith("S ") and "BK7" in line)

    def test_range_comes_from_the_material(self, set_test_backend):
        from optiland.materials import Material

        data = Material("N-BK7", "schott").material_data
        assert (data["min_wavelength"], data["max_wavelength"]) == (0.3, 2.5)

    @pytest.mark.parametrize("wavelength", [0.3, 0.5876, 2.5])
    def test_inside_the_range_is_named(self, wavelength, set_test_backend, recwarn):
        lines = _encode(self._lens(wavelength))
        assert self._s1(lines).endswith("NBK7_SCHOTT")
        assert "PRV" not in lines
        assert not [w for w in recwarn if "private glass" in str(w.message)]

    @pytest.mark.parametrize("wavelength", [10.6, 0.25, 2.5000001, 0.2999999])
    def test_outside_the_range_is_written_private(self, wavelength, set_test_backend):
        import optiland.backend as be
        from optiland.materials import Material

        optic = self._lens(0.5876, wavelength)
        with pytest.warns(UserWarning) as record:
            lines = _encode(optic)
        messages = [str(w.message) for w in record]
        assert any(
            "N-BK7" in m and f"{wavelength:.8g} um" in m and "abort" in m
            for m in messages
        )
        assert any("private glass 'OPT1'" in m for m in messages)
        assert "BK7" not in " ".join(line for line in lines if line.startswith("S "))
        assert _glass_token(_surface_lines(lines)[0]) == "'OPT1'"
        n = [
            float(be.to_numpy(Material("N-BK7", "schott").n(w)))
            for w in sorted((0.5876, wavelength), reverse=True)
        ]
        row = next(line for line in lines if line.startswith("  'OPT1'"))
        assert_allclose([float(v) for v in row.split()[1:]], n, rtol=1e-9)

    def test_out_of_range_glass_is_not_called_unnamed(self, set_test_backend):
        with pytest.warns(UserWarning) as record:
            _encode(self._lens(0.5876, 10.6))
        messages = [str(w.message) for w in record]
        assert not [m for m in messages if "no verified" in m]
        (private,) = [m for m in messages if "private glass 'OPT1'" in m]
        assert "catalog range does not cover them" in private
        assert len([m for m in messages if "N-BK7" in m and "abort" in m]) == 1

    def test_unnamed_glass_still_says_it_has_no_catalog_entry(self, set_test_backend):
        from optiland.materials import IdealMaterial

        optic = _make_lens()
        optic.surfaces[1].material_post = IdealMaterial(1.5)
        with pytest.warns(UserWarning, match="no verified CODE V catalog entry"):
            _encode(optic)

    def test_only_the_glass_out_of_range_becomes_private(self, set_test_backend):
        from optiland.materials import Material

        # 0.31 um: inside N-BK7 (0.3 to 2.5) and outside N-LAK9 (0.32 to 2.5)
        optic = _make_lens()
        optic.wavelengths.add(value=0.31)
        optic.surfaces[2].material_post = Material("N-LAK9", "schott")
        with pytest.warns(UserWarning, match="N-LAK9") as record:
            lines = _encode(optic)
        assert not [w for w in record if "'N-BK7'" in str(w.message)]
        tokens = [_glass_token(line) for line in _surface_lines(lines)]
        assert tokens[0] == "NBK7_SCHOTT"
        assert tokens[1] == "'OPT1'"


# ---------------------------------------------------------------------------
# Object-space telecentric optics are written with TEL Y
# ---------------------------------------------------------------------------


class TestTelecentric:
    def test_tel_y_written_before_the_aperture(self, set_test_backend):
        optic = _make_lens()
        optic.obj_space_telecentric = True
        lines = _encode(optic)
        assert lines.count("TEL Y") == 1
        assert lines.index("TEL Y") + 1 == lines.index("EPD 10")

    def test_tel_y_written_before_a_na_aperture(self, set_test_backend):
        optic = _make_lens(object_thickness=100.0)
        optic.set_aperture(aperture_type="objectNA", value=0.1)
        optic.obj_space_telecentric = True
        lines = _encode(optic)
        assert lines.index("TEL Y") + 1 == lines.index("NAO 0.1")

    def test_non_telecentric_has_no_tel_line(self, set_test_backend):
        optic = _make_lens()
        assert optic.obj_space_telecentric is False
        lines = _encode(optic)
        assert any(line.startswith("SO ") for line in lines)
        assert not any(line.startswith("TEL") for line in lines)

    def test_uv_projection_lens_exports(self, tmp_path, set_test_backend):
        import warnings

        from optiland.samples.lithography import UVProjectionLens

        path = tmp_path / "uv.seq"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            save_codev_file(UVProjectionLens(), str(path))
        assert "TEL Y" in path.read_text().splitlines()


# ---------------------------------------------------------------------------
# A vertex spacing must agree with the light direction set by the mirrors
# ---------------------------------------------------------------------------


class TestSpacingDirection:
    @staticmethod
    def _finish(optic: Optic) -> Optic:
        optic.set_aperture(aperture_type="EPD", value=10.0)
        optic.fields.set_type("angle")
        optic.fields.add(y=0.0)
        optic.wavelengths.add(value=0.5876, is_primary=True)
        return optic

    def _backwards_lens(self) -> Optic:
        """Absolute-z lens whose vertices descend with no mirror."""
        from optiland.materials import IdealMaterial

        optic = Optic()
        optic.surfaces.add(index=0, z=10.0)
        optic.surfaces.add(
            index=1, radius=50.0, z=0.0, material=IdealMaterial(1.5), is_stop=True
        )
        optic.surfaces.add(index=2, radius=-50.0, z=-5.0)
        optic.surfaces.add(index=3, z=-60.0)
        return self._finish(optic)

    def test_negative_spacing_without_a_mirror_refused(
        self, tmp_path, set_test_backend
    ):
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(
            NotImplementedError,
            match=r"Surface 1: Optiland starts the light out travelling in -z",
        ):
            save_codev_file(self._backwards_lens(), str(path))
        assert path.read_bytes() == b"previous design"

    def test_negative_first_gap_after_an_infinite_object_exports(
        self, set_test_backend
    ):
        """Optiland enters +z from an infinite object whatever the gaps are."""
        from optiland.materials import IdealMaterial

        optic = Optic()
        optic.surfaces.add(index=0, thickness=float("inf"))
        optic.surfaces.add(
            index=1,
            radius=50.0,
            thickness=-5.0,
            material=IdealMaterial(1.5),
            is_stop=True,
        )
        optic.surfaces.add(index=2, radius=-50.0, thickness=-45.0)
        optic.surfaces.add(index=3)
        assert _thi(_encode(self._finish(optic))) == [1e10, -5.0, -45.0, 0.0]

    def test_negative_gap_after_the_first_spacing_exports(self, set_test_backend):
        """A negative gap inside a lens (Microscope20x cover glass) is written."""
        from optiland.materials import IdealMaterial

        optic = Optic()
        optic.surfaces.add(index=0, thickness=float("inf"))
        optic.surfaces.add(
            index=1,
            radius=50.0,
            thickness=5.0,
            material=IdealMaterial(1.5),
            is_stop=True,
        )
        optic.surfaces.add(index=2, radius=-50.0, thickness=-1.0)
        optic.surfaces.add(index=3, thickness=46.0)
        optic.surfaces.add(index=4)
        assert _thi(_encode(self._finish(optic))) == [1e10, 5.0, -1.0, 46.0, 0.0]

    def test_microscope20x_exports_with_global_z_spacings(
        self, tmp_path, set_test_backend
    ):
        import warnings

        from optiland.samples.microscopes import Microscope20x

        optic = Microscope20x()
        path = tmp_path / "m.seq"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            save_codev_file(optic, str(path))
        thi = _thi(path.read_text().splitlines())
        z = _gcs_z(optic)
        assert any(t < 0 for t in thi[1:])
        for t, z0, z1 in zip(thi[1:], z[1:], z[2:], strict=False):
            assert t == pytest.approx(z1 - z0, abs=1e-9)

    def test_hubble_telescope_still_exports(self, tmp_path, set_test_backend):
        import warnings

        from optiland.samples.telescopes import HubbleTelescope

        path = tmp_path / "h.seq"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            save_codev_file(HubbleTelescope(), str(path))
        assert any(t < 0 for t in _thi(path.read_text().splitlines()))

    def test_mirror_systems_and_zero_gaps_still_export(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=float("inf"))
        optic.surfaces.add(
            index=1, radius=-100.0, thickness=-50.0, material="mirror", is_stop=True
        )
        optic.surfaces.add(index=2, radius=100.0, thickness=50.0, material="mirror")
        optic.surfaces.add(index=3, thickness=0.0)
        optic.surfaces.add(index=4)
        assert _thi(_encode(self._finish(optic))) == [1e10, -50.0, 50.0, 0.0, 0.0]

    def test_virtual_object_exports(self, set_test_backend):
        """Object past surface 1, light still travelling +z (CODE V: SO 0 -10)."""
        from optiland.materials import IdealMaterial

        optic = Optic()
        optic.surfaces.add(index=0, z=10.0)
        optic.surfaces.add(
            index=1, radius=50.0, z=0.0, material=IdealMaterial(1.5), is_stop=True
        )
        optic.surfaces.add(index=2, radius=-50.0, z=5.0)
        optic.surfaces.add(index=3, z=50.0)
        lines = _encode(self._finish(optic))
        assert "SO 0 -10" in lines
        assert _thi(lines) == [-10.0, 5.0, 45.0, 0.0]

    def test_virtual_object_then_backwards_spacing_still_refused(
        self, set_test_backend
    ):
        optic = self._backwards_lens()
        with pytest.raises(NotImplementedError, match="Surface 1: Optiland starts"):
            _encode(optic)

    def _stop_dummy_lens(self, object_thickness: float, dummy_gap: float) -> Optic:
        """Object, a plano stop dummy, then R50 / 5 / n1.5 and R-50 / 45."""
        from optiland.materials import IdealMaterial

        optic = Optic()
        optic.surfaces.add(index=0, thickness=object_thickness)
        optic.surfaces.add(index=1, thickness=dummy_gap, is_stop=True)
        optic.surfaces.add(
            index=2, radius=50.0, thickness=5.0, material=IdealMaterial(1.5)
        )
        optic.surfaces.add(index=3, radius=-50.0, thickness=45.0)
        optic.surfaces.add(index=4)
        return self._finish(optic)

    @pytest.mark.parametrize("object_thickness", [float("inf"), 100.0])
    def test_stop_dummy_behind_the_lens_exports(
        self, object_thickness, set_test_backend
    ):
        """Optiland travels +z, so it is written (CODE V gives the same EFL)."""
        optic = self._stop_dummy_lens(object_thickness, -10.0)
        thi = _thi(_encode(optic))
        assert thi == [1e10 if object_thickness == float("inf") else 100.0] + [
            -10.0,
            5.0,
            45.0,
            0.0,
        ]

    def test_virtual_object_then_a_zero_dummy_is_refused(self, set_test_backend):
        """The virtual-object override does not fire, so Optiland travels -z."""
        optic = self._stop_dummy_lens(-10.0, 0.0)
        with pytest.raises(NotImplementedError, match="Surface 1: Optiland starts"):
            _encode(optic)

    def test_mirror_first_with_a_negative_gap_before_it_exports(self, set_test_backend):
        """Infinite object: Optiland enters +z, the -20 before a mirror is written."""
        optic = Optic()
        optic.surfaces.add(index=0, thickness=float("inf"))
        optic.surfaces.add(index=1, thickness=-20.0, is_stop=True)
        optic.surfaces.add(index=2, radius=-100.0, thickness=-50.0, material="mirror")
        optic.surfaces.add(index=3)
        assert _thi(_encode(self._finish(optic))) == [1e10, -20.0, -50.0, 0.0]

    def _after_a_dummy_surface(self, first_gap: float) -> Optic:
        """Infinite object, a zero-thickness dummy surface, then a real lens."""
        from optiland.materials import IdealMaterial

        optic = Optic()
        optic.surfaces.add(index=0, thickness=float("inf"))
        optic.surfaces.add(index=1, thickness=0.0)
        optic.surfaces.add(
            index=2,
            radius=50.0,
            thickness=first_gap,
            material=IdealMaterial(1.5),
            is_stop=True,
        )
        optic.surfaces.add(index=3, radius=-50.0, thickness=45.0)
        optic.surfaces.add(index=4)
        return self._finish(optic)

    def test_zero_spacing_is_skipped_before_a_positive_lens(self, set_test_backend):
        optic = self._after_a_dummy_surface(5.0)
        assert _thi(_encode(optic)) == [1e10, 0.0, 5.0, 45.0, 0.0]

    def test_zero_spacing_dummy_then_a_negative_first_gap_exports(
        self, set_test_backend
    ):
        """Infinite object: Optiland enters +z, the -3 after the dummy is written."""
        optic = self._after_a_dummy_surface(-3.0)
        assert _thi(_encode(optic)) == [1e10, 0.0, -3.0, 45.0, 0.0]

    def test_infinite_object_distance_is_not_checked(self, set_test_backend):
        assert _thi(_encode(_make_lens()))[0] == 1e10


# ---------------------------------------------------------------------------
# A curved object surface is refused
# ---------------------------------------------------------------------------


class TestObjectSurfaceShape:
    @staticmethod
    def _lens_with_object(**kwargs) -> Optic:
        from optiland.materials import IdealMaterial

        optic = Optic()
        optic.surfaces.add(index=0, thickness=100.0, **kwargs)
        optic.surfaces.add(
            index=1,
            radius=50.0,
            thickness=5.0,
            material=IdealMaterial(1.5),
            is_stop=True,
        )
        optic.surfaces.add(index=2, radius=-50.0, thickness=45.0)
        optic.surfaces.add(index=3)
        optic.set_aperture(aperture_type="EPD", value=10.0)
        optic.fields.set_type("angle")
        optic.fields.add(y=0.0)
        optic.wavelengths.add(value=0.5876, is_primary=True)
        return optic

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"radius": 100.0},
            {"radius": -100.0},
            {"surface_type": "even_asphere", "radius": 100.0, "coefficients": [0.0]},
        ],
    )
    def test_curved_object_refused_and_file_untouched(
        self, kwargs, tmp_path, set_test_backend
    ):
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(
            NotImplementedError,
            match="Surface 0: a curved object surface cannot be exported to CODE V",
        ):
            save_codev_file(self._lens_with_object(**kwargs), str(path))
        assert path.read_bytes() == b"previous design"

    @pytest.mark.parametrize("kwargs", [{}, {"radius": float("inf")}])
    def test_plane_object_exported(self, kwargs, set_test_backend):
        lines = _encode(self._lens_with_object(**kwargs))
        assert next(line for line in lines if line.startswith("SO ")) == "SO 0 100"


# ---------------------------------------------------------------------------
# Wavelength validity: positive values and Optiland's own primary
# ---------------------------------------------------------------------------


class TestWavelengthValidity:
    @pytest.mark.parametrize("value", [0.0, -0.55])
    def test_non_positive_wavelength_refused(self, value, tmp_path, set_test_backend):
        optic = _make_lens()
        optic.wavelengths.add(value=value)
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(ValueError, match="wavelength 2 is not positive"):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    def test_no_primary_wavelength_refused(self, tmp_path, set_test_backend):
        optic = _make_lens()
        optic.wavelengths[0].is_primary = False
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(ValueError, match="no primary wavelength"):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    def test_several_flagged_primaries_use_optilands_primary(self, set_test_backend):
        optic = _make_lens()
        optic.wavelengths.add(value=0.4861)
        optic.wavelengths[1].is_primary = True  # both flagged
        assert optic.wavelengths.primary_index == 0
        lines = _encode(optic)
        wl = next(line for line in lines if line.startswith("WL ")).split()[1:]
        ref = int(next(line for line in lines if line.startswith("REF ")).split()[1])
        assert_allclose(
            float(wl[ref - 1]), optic.wavelengths.primary_wavelength.value * 1000.0
        )
        assert OpticToCodeVConverter(optic).convert().wavelengths["primary_index"] == 0

    def test_single_primary_unchanged(self, set_test_backend):
        optic = _make_lens()
        optic.wavelengths.add(value=0.4861)
        optic.wavelengths.add(value=0.6563, is_primary=True)
        lines = _encode(optic)
        assert "WL 656.3 587.6 486.1" in lines
        ref = int(next(line for line in lines if line.startswith("REF ")).split()[1])
        assert ref == 1


# ---------------------------------------------------------------------------
# System sanity: aperture, fields, wavelengths, objectNA and the object distance
# ---------------------------------------------------------------------------


class TestSystemSanity:
    @staticmethod
    def _refused(optic: Optic, match: str, tmp_path) -> None:
        path = tmp_path / "existing.seq"
        path.write_bytes(b"previous design")
        with pytest.raises(ValueError, match=match):
            save_codev_file(optic, str(path))
        assert path.read_bytes() == b"previous design"

    def test_no_aperture_refused(self, tmp_path, set_test_backend):
        optic = _make_lens()
        optic.aperture = None
        self._refused(optic, "no aperture set", tmp_path)

    def test_no_fields_refused(self, tmp_path, set_test_backend):
        optic = _make_lens()
        optic.fields.fields.clear()
        self._refused(optic, "no fields", tmp_path)

    def test_no_wavelengths_refused(self, tmp_path, set_test_backend):
        optic = _make_lens()
        optic.wavelengths.wavelengths.clear()
        self._refused(optic, "no wavelengths", tmp_path)

    @pytest.mark.parametrize("value", [1.0, 1.5])
    def test_object_na_of_one_or_more_refused(self, value, tmp_path, set_test_backend):
        optic = _make_lens(object_thickness=100.0)
        optic.set_aperture(aperture_type="objectNA", value=value)
        self._refused(optic, r"objectNA .* not below 1 in air", tmp_path)

    def test_object_na_below_one_exported(self, set_test_backend):
        optic = _make_lens(object_thickness=100.0)
        optic.set_aperture(aperture_type="objectNA", value=0.99)
        assert "NAO 0.99" in _encode(optic)

    @pytest.mark.parametrize("value", [1.0, 1.5])
    def test_object_na_refused_without_a_real_surface(
        self, value, tmp_path, set_test_backend
    ):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=12.0)
        optic.surfaces.add(index=1)
        optic.set_aperture(aperture_type="objectNA", value=value)
        self._refused(optic, r"objectNA .* not below 1 in air", tmp_path)

    def test_object_distance_zero_refused(self, tmp_path, set_test_backend):
        self._refused(
            _make_lens(object_thickness=0.0),
            "object distance of 0 cannot be exported",
            tmp_path,
        )

    def test_small_object_distance_is_exported(self, set_test_backend):
        lines = _encode(_make_lens(object_thickness=1e-3))
        assert next(line for line in lines if line.startswith("SO ")) == "SO 0 0.001"

    def test_optics_without_a_real_surface_are_not_refused(self, set_test_backend):
        assert OpticToCodeVConverter(Optic()).convert().surfaces == {}
        optic = Optic()
        optic.surfaces.add(index=0, thickness=12.0)
        optic.surfaces.add(index=1)
        assert _thi(_encode(optic)) == [12.0, 0.0]


# ---------------------------------------------------------------------------
# TITLE: a quote or a line break must not cut the title
# ---------------------------------------------------------------------------


class TestTitleQuoting:
    @pytest.mark.parametrize(
        ("name", "title"),
        [
            ("Cooke triplet", "Cooke triplet"),
            ("Smith's doublet", 'Smith"s doublet'),
            ("'quoted'", '"quoted"'),
            ("line one\nline two", "line one line two"),
            ("a\r\nb\rc", "a b c"),
        ],
    )
    def test_title_survives(self, name, title, set_test_backend):
        optic = _make_lens()
        optic.name = name
        lines = _encode(optic)
        assert f"TITLE '{title}'" in lines
        title_line = next(line for line in lines if line.startswith("TITLE"))
        assert title_line.count("'") == 2

    @pytest.mark.parametrize("name", [None, "", "\n"])
    def test_no_title_line_without_a_name(self, name, set_test_backend):
        optic = _make_lens()
        optic.name = name
        assert not any(line.startswith("TITLE") for line in _encode(optic))
