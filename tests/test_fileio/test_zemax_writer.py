"""Tests for the Zemax writer path (.zmx export).

Covers:
- Round-trip tests (load → save → reload → compare)
- Warning assertions for MODEL glass and pickups/solves
- NotImplementedError for unsupported surface types
- Backend parametrization via set_test_backend fixture
"""

from __future__ import annotations

import math
import os

import pytest

import optiland.backend as be
from optiland.fileio import load_zemax_file, save_zemax_file
from optiland.fileio.zemax.writer.formatter import OpticToZemaxConverter
from optiland.optic import Optic
from tests.utils import assert_allclose

_ZEMAX_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "zemax_files"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zemax(filename: str) -> str:
    return os.path.join(_ZEMAX_DIR, filename)


def _round_trip(filename: str, tmp_path) -> tuple[Optic, Optic]:
    """Load a zmx, save it, reload it, return (original, reloaded)."""
    original = load_zemax_file(_zemax(filename))
    out = tmp_path / "out.zmx"
    save_zemax_file(original, str(out))
    reloaded = load_zemax_file(str(out))
    return original, reloaded


def _surfaces_match(orig: Optic, reloaded: Optic, rtol: float = 1e-5) -> None:
    """Assert that surface radii and thicknesses match within rtol."""
    assert orig.surfaces.num_surfaces == reloaded.surfaces.num_surfaces
    for i in range(orig.surfaces.num_surfaces):
        s1 = orig.surfaces[i]
        s2 = reloaded.surfaces[i]
        r1 = float(s1.geometry.radius)
        r2 = float(s2.geometry.radius)
        if math.isinf(r1):
            assert math.isinf(r2), f"Surface {i}: radius {r1} vs {r2}"
        else:
            assert_allclose(r1, r2, rtol=rtol)

        t1 = float(s1.thickness)
        t2 = float(s2.thickness)
        if math.isinf(t1):
            assert math.isinf(t2), f"Surface {i}: thickness {t1} vs {t2}"
        else:
            assert_allclose(t1, t2, rtol=rtol)


# ---------------------------------------------------------------------------
# Round-trip: standard spherical lens (Cooke-style triplet)
# ---------------------------------------------------------------------------


class TestRoundTripStandard:
    def test_surfaces(self, tmp_path, set_test_backend):
        orig, reloaded = _round_trip("lens1.zmx", tmp_path)
        _surfaces_match(orig, reloaded)

    def test_aperture(self, tmp_path, set_test_backend):
        orig, reloaded = _round_trip("lens1.zmx", tmp_path)
        assert orig.aperture.ap_type == reloaded.aperture.ap_type
        assert_allclose(orig.aperture.value, reloaded.aperture.value, rtol=1e-5)

    def test_fields(self, tmp_path, set_test_backend):
        orig, reloaded = _round_trip("lens1.zmx", tmp_path)
        assert orig.fields.num_fields == reloaded.fields.num_fields

    def test_wavelengths(self, tmp_path, set_test_backend):
        orig, reloaded = _round_trip("lens1.zmx", tmp_path)
        assert orig.wavelengths.num_wavelengths == reloaded.wavelengths.num_wavelengths
        for w1, w2 in zip(orig.wavelengths, reloaded.wavelengths, strict=False):
            assert_allclose(w1.value, w2.value, rtol=1e-6)

    def test_stop_surface(self, tmp_path, set_test_backend):
        orig, reloaded = _round_trip("lens1.zmx", tmp_path)
        orig_stops = [i for i, s in enumerate(orig.surfaces) if s.is_stop]
        new_stops = [i for i, s in enumerate(reloaded.surfaces) if s.is_stop]
        assert orig_stops == new_stops


# ---------------------------------------------------------------------------
# Round-trip: even asphere
# ---------------------------------------------------------------------------


class TestRoundTripAsphere:
    def test_roundtrip(self, tmp_path, set_test_backend):
        orig, reloaded = _round_trip("lens2.zmx", tmp_path)
        _surfaces_match(orig, reloaded)

    def test_aspheric_coefficients(self, tmp_path, set_test_backend):
        orig, reloaded = _round_trip("lens2.zmx", tmp_path)
        # Find the first even_asphere surface and compare coefficients
        from optiland.geometries import EvenAsphere

        for i in range(orig.surfaces.num_surfaces):
            s1 = orig.surfaces[i]
            s2 = reloaded.surfaces[i]
            if isinstance(s1.geometry, EvenAsphere):
                c1 = list(s1.geometry.coefficients)
                c2 = list(s2.geometry.coefficients)
                for a, b in zip(c1, c2, strict=False):
                    assert_allclose(float(a), float(b), rtol=1e-5)
                break


# ---------------------------------------------------------------------------
# Round-trip: toroidal surface
# ---------------------------------------------------------------------------


class TestRoundTripToroidal:
    def test_roundtrip(self, tmp_path, set_test_backend):
        orig, reloaded = _round_trip("thorlabs_lj1598l1.zmx", tmp_path)
        _surfaces_match(orig, reloaded)

    def test_toroidal_radii(self, tmp_path, set_test_backend):
        orig, reloaded = _round_trip("thorlabs_lj1598l1.zmx", tmp_path)
        from optiland.geometries import ToroidalGeometry

        for i in range(orig.surfaces.num_surfaces):
            s1 = orig.surfaces[i]
            s2 = reloaded.surfaces[i]
            if isinstance(s1.geometry, ToroidalGeometry):
                assert_allclose(s1.geometry.R_yz, s2.geometry.R_yz, rtol=1e-5)


# ---------------------------------------------------------------------------
# Round-trip: floating stop aperture
# ---------------------------------------------------------------------------


class TestRoundTripFloaAperture:
    def test_aperture_type(self, tmp_path, set_test_backend):
        orig, reloaded = _round_trip("lens_floa.zmx", tmp_path)
        assert reloaded.aperture.ap_type == "float_by_stop_size"

    def test_aperture_value(self, tmp_path, set_test_backend):
        orig, reloaded = _round_trip("lens_floa.zmx", tmp_path)
        assert_allclose(orig.aperture.value, reloaded.aperture.value, rtol=1e-5)


# ---------------------------------------------------------------------------
# Warning: MODEL glass
# ---------------------------------------------------------------------------


class TestModelGlassWarning:
    def test_abbe_material_warns(self, tmp_path, set_test_backend):
        """An AbbeMaterial has no catalog; it must trigger a MODEL glass warning."""
        from optiland.materials import AbbeMaterial

        optic = Optic(name="test")
        optic.surfaces.add(index=0, thickness=be.inf, material="Air")
        optic.surfaces.add(
            index=1,
            surface_type="standard",
            radius=50.0,
            thickness=5.0,
            material=AbbeMaterial(n=1.5, abbe=50.0),
        )
        optic.surfaces.add(
            index=2,
            surface_type="standard",
            radius=be.inf,
            thickness=0.0,
            is_stop=True,
        )
        optic.surfaces.add(index=3)
        optic.set_aperture("EPD", 10.0)
        optic.fields.set_type("angle")
        optic.fields.add(x=0, y=0)
        optic.wavelengths.add(value=0.5876, is_primary=True)

        out = tmp_path / "model_glass.zmx"
        with pytest.warns(UserWarning, match="MODEL glass"):
            save_zemax_file(optic, str(out))


# ---------------------------------------------------------------------------
# Warning: pickups
# ---------------------------------------------------------------------------


class TestPickupWarning:
    def test_pickup_warns(self, tmp_path, set_test_backend):
        """A pickup on the optic must trigger a warning during export."""
        orig = load_zemax_file(_zemax("lens1.zmx"))

        # Add a pickup: link surface 1 thickness to surface 3
        orig.pickups.add(
            source_surface_idx=1,
            attr_type="thickness",
            target_surface_idx=3,
            scale=1.0,
            offset=0.0,
        )

        out = tmp_path / "pickup.zmx"
        with pytest.warns(UserWarning, match="pickup"):
            save_zemax_file(orig, str(out))


# ---------------------------------------------------------------------------
# NotImplementedError: unsupported surface type
# ---------------------------------------------------------------------------


class TestUnsupportedSurface:
    def test_unsupported_geometry_raises(self, tmp_path, set_test_backend):
        """A surface with a Zernike or NURBS geometry must raise NotImplementedError."""

        optic = Optic(name="test_unsupported")
        optic.surfaces.add(index=0, thickness=be.inf, material="Air")
        optic.surfaces.add(
            index=1,
            surface_type="zernike",
            radius=50.0,
            thickness=5.0,
        )
        optic.surfaces.add(index=2)
        optic.set_aperture("EPD", 10.0)
        optic.fields.set_type("angle")
        optic.fields.add(x=0, y=0)
        optic.wavelengths.add(value=0.5876, is_primary=True)

        with pytest.raises(NotImplementedError):
            save_zemax_file(optic, str(tmp_path / "unsupported.zmx"))


# ---------------------------------------------------------------------------
# OpticToZemaxConverter Extended
# ---------------------------------------------------------------------------


class TestOpticToZemaxConverterExtended:
    def test_field_type_string_none(self):
        from optiland.fileio.zemax.writer.formatter import _field_type_string

        optic = Optic()
        # No fields added
        assert _field_type_string(optic) == "angle"

    def test_warn_unknown_aperture(self):
        optic = Optic()

        # Mock an unknown aperture type
        class UnknownAp:
            ap_type = "unknown"
            value = 0.0

        optic.aperture = UnknownAp()
        conv = OpticToZemaxConverter(optic)
        with pytest.warns(UserWarning, match="Unknown aperture type"):
            conv.convert()

    def test_coordinate_break_insertion(self):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=0.0)
        optic.surfaces.add(index=1, radius=50.0, thickness=5.0, dx=1.0)
        optic.surfaces.add(index=2, thickness=0.0)
        conv = OpticToZemaxConverter(optic)
        model = conv.convert()
        # Obj(0) + CB(1) + Surf(2) + Img(3) = 4 surfaces?
        # Actually OpticToZemaxConverter adds a CB AFTER to return to original CS if needed?
        # Let's check the length and that at least one CB exists.
        assert any(s.get("TYPE") == "COORDBRK" for s in model.surfaces.values())
        # Find the CB
        cb = next(s for s in model.surfaces.values() if s.get("TYPE") == "COORDBRK")
        assert cb["PARM_1"] == 1.0  # dx

    def test_format_glass_model(self):
        from optiland.materials import IdealMaterial

        optic = Optic()
        optic.surfaces.add(index=0, thickness=0.0)
        # Use an IdealMaterial to trigger the MODEL glass branch
        optic.surfaces.add(
            index=1, radius=50.0, thickness=5.0, material=IdealMaterial(1.6)
        )
        optic.surfaces.add(index=2, thickness=0.0)
        optic.add_wavelength(0.5876, is_primary=True)
        conv = OpticToZemaxConverter(optic)
        with pytest.warns(UserWarning, match="writing as MODEL glass"):
            model = conv.convert()
        assert model.surfaces[1]["GLAS"]["name"] == "MODEL"

    def test_is_air_ideal(self):
        from optiland.fileio.zemax.writer.formatter import _is_air
        from optiland.materials import IdealMaterial

        assert _is_air(IdealMaterial(1.0)) is True
        assert _is_air(IdealMaterial(1.5)) is False
