from __future__ import annotations

import pytest

from optiland.diagnostics import check_system
from optiland.materials import Material
from optiland.materials.ideal import IdealMaterial
from optiland.optic import Optic


def _build_valid_system() -> Optic:
    """A minimal, fully-specified system that every check should pass."""
    optic = Optic()
    optic.surfaces.add(index=0, thickness=100)
    optic.surfaces.add(
        index=1, radius=20, thickness=10, material=IdealMaterial(n=1.5)
    )
    optic.surfaces.add(index=2, is_stop=True, aperture=10.0, thickness=10)
    optic.surfaces.add(index=3)
    optic.fields.set_type("angle")
    optic.fields.add(y=0)
    optic.set_aperture("EPD", 10.0)
    optic.wavelengths.add(0.55)
    return optic


def _codes(report) -> set[str]:
    return {d.code for d in report}


class TestValidSystem:
    def test_no_findings(self, set_test_backend):
        report = check_system(_build_valid_system())
        assert report.ok
        assert len(report) == 0


class TestOPT001NoWavelengths:
    def test_fires_when_missing(self, set_test_backend):
        optic = _build_valid_system()
        optic.wavelengths.wavelengths.clear()
        report = check_system(optic)
        assert "OPT001" in _codes(report)
        assert not report.ok

    def test_silent_when_present(self, set_test_backend):
        report = check_system(_build_valid_system())
        assert "OPT001" not in _codes(report)


class TestOPT002NoPrimaryWavelength:
    def test_fires_when_none_primary(self, set_test_backend):
        optic = _build_valid_system()
        for w in optic.wavelengths:
            w.is_primary = False
        report = check_system(optic)
        assert "OPT002" in _codes(report)

    def test_silent_when_primary_set(self, set_test_backend):
        report = check_system(_build_valid_system())
        assert "OPT002" not in _codes(report)


class TestOPT003NoAperture:
    def test_fires_when_missing(self, set_test_backend):
        optic = _build_valid_system()
        optic.aperture = None
        report = check_system(optic)
        assert "OPT003" in _codes(report)

    def test_silent_when_present(self, set_test_backend):
        report = check_system(_build_valid_system())
        assert "OPT003" not in _codes(report)


class TestOPT004NoStopSurface:
    def test_fires_when_missing(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=100)
        optic.surfaces.add(
            index=1, radius=20, thickness=10, material=IdealMaterial(n=1.5)
        )
        optic.surfaces.add(index=2, thickness=10)
        optic.surfaces.add(index=3)
        optic.fields.set_type("angle")
        optic.fields.add(y=0)
        optic.set_aperture("EPD", 10.0)
        optic.wavelengths.add(0.55)
        report = check_system(optic)
        assert "OPT004" in _codes(report)

    def test_silent_when_present(self, set_test_backend):
        report = check_system(_build_valid_system())
        assert "OPT004" not in _codes(report)


class TestOPT005TooFewSurfaces:
    def test_fires_when_missing(self, set_test_backend):
        report = check_system(Optic())
        assert "OPT005" in _codes(report)

    def test_silent_when_present(self, set_test_backend):
        report = check_system(_build_valid_system())
        assert "OPT005" not in _codes(report)


class TestOPT006NoFields:
    def test_fires_when_missing(self, set_test_backend):
        optic = _build_valid_system()
        optic.fields.fields.clear()
        report = check_system(optic)
        assert "OPT006" in _codes(report)

    def test_silent_when_present(self, set_test_backend):
        report = check_system(_build_valid_system())
        assert "OPT006" not in _codes(report)


class TestOPT007ObjectSurfaceNotFirst:
    def test_silent_on_valid_system(self, set_test_backend):
        # Object surfaces are always constructed at index 0 by the surface
        # factory, so this check is a defensive guard; confirm it stays
        # silent on every normally-built system.
        report = check_system(_build_valid_system())
        assert "OPT007" not in _codes(report)


class TestOPT008NonFiniteInteriorGeometry:
    def test_fires_on_nonfinite_interior_thickness(self, set_test_backend):
        optic = _build_valid_system()
        optic.surfaces.surfaces[1].thickness = float("inf")
        report = check_system(optic)
        assert "OPT008" in _codes(report)

    def test_silent_when_finite(self, set_test_backend):
        report = check_system(_build_valid_system())
        assert "OPT008" not in _codes(report)


class TestOPT009WavelengthOutsideMaterialRange:
    def test_fires_when_outside_range(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=100)
        optic.surfaces.add(
            index=1, radius=20, thickness=10, material=Material("N-BK7")
        )
        optic.surfaces.add(index=2, is_stop=True, aperture=10.0, thickness=10)
        optic.surfaces.add(index=3)
        optic.fields.set_type("angle")
        optic.fields.add(y=0)
        optic.set_aperture("EPD", 10.0)
        optic.wavelengths.add(20.0)
        report = check_system(optic)
        assert "OPT009" in _codes(report)

    def test_silent_when_inside_range(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=100)
        optic.surfaces.add(
            index=1, radius=20, thickness=10, material=Material("N-BK7")
        )
        optic.surfaces.add(index=2, is_stop=True, aperture=10.0, thickness=10)
        optic.surfaces.add(index=3)
        optic.fields.set_type("angle")
        optic.fields.add(y=0)
        optic.set_aperture("EPD", 10.0)
        optic.wavelengths.add(0.55)
        report = check_system(optic)
        assert "OPT009" not in _codes(report)


class TestOPT010NonPositiveThickness:
    def test_fires_on_negative_thickness(self, set_test_backend):
        optic = _build_valid_system()
        optic.surfaces.surfaces[1].thickness = -5
        report = check_system(optic)
        assert "OPT010" in _codes(report)

    def test_silent_when_positive(self, set_test_backend):
        report = check_system(_build_valid_system())
        assert "OPT010" not in _codes(report)

    def test_silent_after_single_mirror(self, set_test_backend):
        # A reflective surface reverses the propagation direction, so the
        # thickness that follows it is expected to be negative, not
        # positive. This must not be flagged.
        optic = Optic()
        optic.surfaces.add(index=0, thickness=100)
        optic.surfaces.add(
            index=1, radius=50, thickness=-50, material="mirror", is_stop=True
        )
        optic.surfaces.add(index=2, thickness=30)
        optic.fields.set_type("angle")
        optic.fields.add(y=0)
        optic.set_aperture("EPD", 10.0)
        optic.wavelengths.add(0.55)
        report = check_system(optic)
        assert "OPT010" not in _codes(report)

    def test_fires_on_wrong_sign_after_single_mirror(self, set_test_backend):
        # After one mirror, a *positive* thickness is the anomaly.
        optic = Optic()
        optic.surfaces.add(index=0, thickness=100)
        optic.surfaces.add(
            index=1, radius=50, thickness=50, material="mirror", is_stop=True
        )
        optic.surfaces.add(index=2, thickness=30)
        optic.fields.set_type("angle")
        optic.fields.add(y=0)
        optic.set_aperture("EPD", 10.0)
        optic.wavelengths.add(0.55)
        report = check_system(optic)
        assert "OPT010" in _codes(report)

    def test_silent_on_hubble_telescope_two_mirror_system(self, set_test_backend):
        # Real two-mirror system: negative thickness after the primary,
        # positive again after the secondary. Regression test for a false
        # positive found against this exact sample.
        from optiland.samples.telescopes import HubbleTelescope

        report = check_system(HubbleTelescope())
        assert "OPT010" not in _codes(report)


class TestOPT011StopAtObjectOrImage:
    def test_fires_when_stop_is_image_surface(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=100)
        optic.surfaces.add(
            index=1, radius=20, thickness=10, material=IdealMaterial(n=1.5)
        )
        optic.surfaces.add(index=2, thickness=10)
        optic.surfaces.add(index=3, is_stop=True)
        optic.fields.set_type("angle")
        optic.fields.add(y=0)
        optic.set_aperture("EPD", 10.0)
        optic.wavelengths.add(0.55)
        report = check_system(optic)
        assert "OPT011" in _codes(report)

    def test_silent_when_interior(self, set_test_backend):
        report = check_system(_build_valid_system())
        assert "OPT011" not in _codes(report)


class TestOPT012NoRaysReachImage:
    def test_fires_when_rays_blocked(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=100)
        optic.surfaces.add(
            index=1, radius=20, thickness=10, material=IdealMaterial(n=1.5)
        )
        optic.surfaces.add(index=2, is_stop=True, aperture=0.001, thickness=10)
        optic.surfaces.add(index=3)
        optic.fields.set_type("angle")
        optic.fields.add(y=89)
        optic.set_aperture("EPD", 500.0)
        optic.wavelengths.add(0.55)
        report = check_system(optic)
        assert "OPT012" in _codes(report)

    def test_silent_on_valid_system(self, set_test_backend):
        report = check_system(_build_valid_system())
        assert "OPT012" not in _codes(report)


class TestDiagnosticMessages:
    def test_primary_wavelength_error_is_actionable(self, set_test_backend):
        from optiland.wavelength import WavelengthGroup

        wg = WavelengthGroup()
        wg.add(0.55)
        wg.wavelengths[0].is_primary = False
        with pytest.raises(ValueError, match="wavelengths.add"):
            _ = wg.primary_index

    def test_surface_index_error_names_cause_and_fix(self, set_test_backend):
        optic = Optic()
        optic.surfaces.add(index=0, thickness=100)
        with pytest.raises(IndexError, match="highest valid index is 1"):
            optic.surfaces.add(index=5, thickness=10)
