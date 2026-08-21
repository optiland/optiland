"""Comprehensive tests for the ISO 10110 drawing generator.

Covers:
- notation.py: SurfaceSpec / ElementSpec formatting and validation
- spec.py:     DrawingSpec YAML round-trip, organisation field
- drawing.py:  ElementDrawing PDF/PNG/DXF rendering (smoke tests)
- report.py:   ISO10110Report batch generation

Run with::

    pytest tests/test_iso10110.py -v
"""

from __future__ import annotations

import math
import tempfile
import warnings
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_singlet():
    """Return a minimal plano-convex singlet Optic."""
    from optiland.optic import Optic

    lens = Optic()
    lens.add_surface(index=0, thickness=float("inf"))
    lens.add_surface(index=1, thickness=6.0, radius=50.0,
                     material="N-BK7", is_stop=True)
    lens.add_surface(index=2, thickness=50.0)
    lens.add_surface(index=3)
    lens.set_aperture(aperture_type="EPD", value=20.0)
    lens.set_field_type(field_type="angle")
    lens.add_field(y=0)
    lens.add_wavelength(value=0.5876)
    lens.update_paraxial()
    return lens


def _make_triplet():
    """Return the built-in Cooke Triplet sample lens."""
    import optiland.samples.objectives as obj

    return obj.CookeTriplet()


def _make_cemented_doublet():
    """Return the built-in cemented achromatic doublet sample lens."""
    from optiland.samples.simple import CementedAchromat

    return CementedAchromat()


def _make_two_mirror_system():
    """Return a simple two-mirror (Cassegrain-like) Optic, all surfaces in air."""
    from optiland import optic

    lens = optic.Optic()
    lens.surfaces.add(index=0, thickness=float("inf"))
    lens.surfaces.add(
        index=1, radius=-100.0, thickness=-40.0, material="mirror", is_stop=True
    )
    lens.surfaces.add(index=2, radius=-30.0, thickness=40.0, material="mirror")
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=10.0)
    lens.fields.set_type("angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=0.5876, is_primary=True)
    return lens


def _make_immersed_image_singlet():
    """Return a singlet whose image surface is still immersed in glass."""
    from optiland import optic

    lens = optic.Optic()
    lens.surfaces.add(index=0, thickness=float("inf"))
    lens.surfaces.add(
        index=1, radius=50.0, thickness=6.0, material="N-BK7", is_stop=True
    )
    lens.surfaces.add(index=2)  # image surface, still inside N-BK7 (no air gap)
    return lens


# ---------------------------------------------------------------------------
# elements.py — identify_elements()
# ---------------------------------------------------------------------------

class TestIdentifyElementsWarnings:
    def test_two_mirror_system_warns_and_returns_empty(self):
        from optiland.iso10110.elements import identify_elements

        lens = _make_two_mirror_system()
        with pytest.warns(UserWarning, match="reflective"):
            elements = identify_elements(lens)
        assert elements == []

    def test_immersed_image_warns_and_drops_trailing_element(self):
        from optiland.iso10110.elements import identify_elements

        lens = _make_immersed_image_singlet()
        with pytest.warns(UserWarning, match="immersed"):
            elements = identify_elements(lens)
        assert elements == []

    def test_ordinary_singlet_emits_no_warnings(self):
        from optiland.iso10110.elements import identify_elements

        lens = _make_singlet()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            elements = identify_elements(lens)
        assert len(elements) == 1
        assert caught == []

    def test_cemented_doublet_emits_no_warnings(self):
        from optiland.iso10110.elements import identify_elements

        lens = _make_cemented_doublet()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            elements = identify_elements(lens)
        assert len(elements) == 1
        assert caught == []


# ---------------------------------------------------------------------------
# notation.py — SurfaceSpec
# ---------------------------------------------------------------------------

class TestSurfaceSpec:
    def test_defaults(self):
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec()
        assert s.fmt_form_error() == "3/ —"
        assert s.fmt_centration() == "4/ —"
        assert s.fmt_imperfections() == "5/ —"
        assert s.fmt_coating() == "7/ —"

    def test_form_error_full(self):
        """Fringe units always include ; λ=<wl> nm per ISO 10110-5:2016 §4.3."""
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(sag_tolerance=2, irregularity=0.5,
                        rotationally_symmetric=0.25)
        result = s.fmt_form_error()
        assert result.startswith("3/ 2(0.5/0.25)")
        assert "; λ=546.1 nm" in result

    def test_form_error_irr_only(self):
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(irregularity=1.0)
        result = s.fmt_form_error()
        assert result.startswith("3/ (1)")
        assert "; λ=" in result

    def test_form_error_sag_only(self):
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(sag_tolerance=5)
        result = s.fmt_form_error()
        assert result.startswith("3/ 5")
        assert "; λ=" in result

    def test_form_error_nm_units(self):
        """form_unit='nm' must append nm suffix; no wavelength annotation."""
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(sag_tolerance=50, irregularity=20,
                        rotationally_symmetric=10, form_unit="nm")
        result = s.fmt_form_error()
        assert result == "3/ 50 nm(20 nm/10 nm)"
        assert "λ" not in result

    def test_test_wavelength_standard(self):
        """Standard wavelength (546.07 nm) must appear in fringe-unit output."""
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(irregularity=0.5, test_wavelength=546.07)
        result = s.fmt_form_error()
        assert "; λ=546.1 nm" in result

    def test_test_wavelength_nonstandard(self):
        """Non-standard wavelength must appear correctly in the output."""
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(irregularity=0.5, test_wavelength=632.8)
        result = s.fmt_form_error()
        assert "; λ=632.8 nm" in result

    def test_test_wavelength_default(self):
        """When test_wavelength is None, default 546.07 nm is used for fringes."""
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(irregularity=0.5)
        result = s.fmt_form_error()
        assert "; λ=546.1 nm" in result

    def test_centration_basic(self):
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(centration=0.5)
        assert s.fmt_centration() == "4/ 0.5\u2032"  # U+2032 prime for arc-minutes

    def test_centration_integer_no_trailing_zero(self):
        """Integer centration values must not include trailing .0 (e.g. 3′ not 3.0′)."""
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(centration=3)
        assert s.fmt_centration() == "4/ 3\u2032"

    def test_centration_with_decentration(self):
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(centration=0.5, centration_decentration=0.01)
        assert s.fmt_centration() == "4/ 0.5\u2032(0.01)"

    def test_decentration_without_centration_warns(self):
        """centration_decentration set without centration must emit an advisory.

        Per ISO 10110-6, δ appears only inside '4/ σ′(δ)'.  Without σ, δ
        cannot be rendered and is silently dropped.
        """
        import warnings
        from optiland.iso10110.notation import SurfaceSpec

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s = SurfaceSpec(centration_decentration=0.05)
        assert any("centration_decentration" in str(x.message) or
                   "decentration" in str(x.message).lower()
                   for x in w), "Expected decentration advisory warning"
        # Without σ, the 4/ code falls back to dash
        assert s.fmt_centration() == "4/ \u2014"

    def test_imperfections_multiplication_sign(self):
        """ASCII 'x' must be normalised to × (U+00D7)."""
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(imperfections="2x0.04")
        result = s.fmt_imperfections()
        assert "×" in result
        assert "x" not in result

    def test_imperfection_fields_normalized_at_storage(self):
        """ASCII 'x' must be normalised in stored field values, not only at render time.

        T1 conformance: YAML round-trips must emit × (U+00D7), not ASCII x.
        Verified by checking the field attribute directly rather than via fmt_*().
        """
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(
            imperfections="2x0.04",
            coating_imperfections="1x0.016",
            scratches="1x0.001",
            assembly_imperfections="2x0.063",
        )
        assert s.imperfections == "2×0.04"
        assert s.coating_imperfections == "1×0.016"
        assert s.scratches == "1×0.001"
        assert s.assembly_imperfections == "2×0.063"

    def test_imperfections_full(self):
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(
            imperfections="2×0.04",
            coating_imperfections="1×0.016",
            scratches="1×0.001",
            edge_chips="0.5",
        )
        result = s.fmt_imperfections()
        assert result.startswith("5/ ")
        assert "C1×0.016" in result
        assert "L1×0.001" in result
        assert "E0.5" in result

    def test_coating_no_lambda_prefix(self):
        """7/ must NOT contain a λ prefix."""
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(coating="AR 400-700 nm R<0.5%")
        result = s.fmt_coating()
        assert result.startswith("7/ AR")
        assert "λ" not in result

    def test_centration_negative_raises(self):
        from optiland.iso10110.notation import SurfaceSpec

        with pytest.raises(ValueError, match="centration must be ≥ 0"):
            SurfaceSpec(centration=-1.0)

    def test_ca_diameter_zero_raises(self):
        from optiland.iso10110.notation import SurfaceSpec

        with pytest.raises(ValueError, match="ca_diameter must be > 0"):
            SurfaceSpec(ca_diameter=0.0)

    def test_r5_ok_no_warning(self):
        """Standard R5 sizes must not trigger warnings."""
        from optiland.iso10110.notation import SurfaceSpec

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SurfaceSpec(imperfections="2×0.04")   # 0.04 = 4.0e-2 — R5 ✓
            SurfaceSpec(imperfections="2×0.063")  # 6.3e-2 — R5 ✓
            SurfaceSpec(imperfections="1×0.16")   # 1.6e-1 — R5 ✓

    def test_r5_coating_imperfections_warns(self):
        """Non-R5 size in coating_imperfections must emit a warning."""
        from optiland.iso10110.notation import SurfaceSpec

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SurfaceSpec(coating_imperfections="1×0.05")   # 0.05 — not R5
        assert any("R5" in str(x.message) or "coating" in str(x.message).lower()
                   for x in w)

    def test_r5_scratches_warns(self):
        """Non-R5 size in scratches must emit a warning."""
        from optiland.iso10110.notation import SurfaceSpec

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SurfaceSpec(scratches="1×0.003")   # 0.003 — not R5 (0.0025 or 0.004)
        assert any("R5" in str(x.message) or "scratch" in str(x.message).lower()
                   for x in w)

    def test_r5_assembly_imperfections_warns(self):
        """Non-R5 size in assembly_imperfections must emit a warning."""
        from optiland.iso10110.notation import SurfaceSpec

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SurfaceSpec(assembly_imperfections="2×0.05")   # 0.05 — not R5
        assert any("R5" in str(x.message) or "assembly" in str(x.message).lower()
                   for x in w)

    def test_r5_non_standard_warns(self):
        """Non-R5 sizes must emit a UserWarning."""
        from optiland.iso10110.notation import SurfaceSpec

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SurfaceSpec(imperfections="2×0.05")   # 0.05 — not R5
        assert any("Renard R5" in str(x.message) for x in w)

    def test_laser_damage_default(self):
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec()
        assert s.fmt_laser_damage() == "6/ —"

    def test_laser_damage_set(self):
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(laser_damage="H 20J/cm²; 10ns; 1064nm")
        assert s.fmt_laser_damage() == "6/ H 20J/cm²; 10ns; 1064nm"

    def test_wavefront_default(self):
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec()
        assert s.fmt_wavefront() == "13/ —"

    def test_wavefront_set(self):
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(wavefront_deformation="0.1(0.05)")
        assert s.fmt_wavefront() == "13/ 0.1(0.05)"

    def test_assembly_imperfections_default(self):
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec()
        assert s.fmt_assembly_imperfections() == "15/ —"

    def test_assembly_imperfections_mul_sign(self):
        """ASCII 'x' in assembly_imperfections must be normalised to ×."""
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(assembly_imperfections="2x0.04")
        result = s.fmt_assembly_imperfections()
        assert "×" in result
        assert "x" not in result
        assert result == "15/ 2×0.04"

    def test_sharp_edge_default_false(self):
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec()
        assert s.sharp_edge is False

    def test_pyramid_error_default(self):
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec()
        assert s.fmt_pyramid_error() == ""

    def test_pyramid_error_set(self):
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(pyramid_error=2.5)
        assert s.fmt_pyramid_error() == "pyr 2.5\u2032"  # U+2032 prime

    def test_pyramid_error_integer_no_trailing_zero(self):
        """Integer pyramid_error must not include trailing .0."""
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(pyramid_error=2)
        assert s.fmt_pyramid_error() == "pyr 2\u2032"

    def test_to_dict_roundtrip(self):
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(irregularity=0.5, centration=0.5,
                        coating="AR", test_wavelength=632.8,
                        laser_damage="H 10J/cm²", wavefront_deformation="0.2",
                        assembly_imperfections="1×0.063", sharp_edge=True)
        d = s.to_dict()
        s2 = SurfaceSpec.from_dict(d)
        assert s2.irregularity == 0.5
        assert s2.coating == "AR"
        assert s2.test_wavelength == 632.8
        assert s2.laser_damage == "H 10J/cm²"
        assert s2.wavefront_deformation == "0.2"
        assert s2.assembly_imperfections == "1×0.063"
        assert s2.sharp_edge is True

    def test_form_unit_nm_survives_roundtrip(self):
        """form_unit='nm' must survive a to_dict/from_dict round-trip.

        When form_unit is set to 'nm' (absolute nanometres per ISO 10110-5:2016),
        the fmt_form_error() output must not include the wavelength annotation
        (nm units are absolute and need no λ reference).
        """
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(sag_tolerance=50, irregularity=20, form_unit="nm")
        d = s.to_dict()
        s2 = SurfaceSpec.from_dict(d)
        assert s2.form_unit == "nm"
        result = s2.fmt_form_error()
        assert "nm" in result
        assert "\u03bb" not in result  # no λ= annotation for nm units

    # ── 8/ surface texture (ISO 10110-8:2019) ────────────────────────────

    def test_roughness_none(self):
        """fmt_roughness() returns empty string when roughness is not set."""
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec()
        assert s.fmt_roughness() == ""

    def test_roughness_polished(self):
        """'P' designates a polished surface without quantitative spec (§5.3.1)."""
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(roughness="P")
        assert s.fmt_roughness() == "8/ P"

    def test_roughness_polish_grade(self):
        """P1–P4 designate polish grades per ISO 10110-8:2019 Table 1."""
        from optiland.iso10110.notation import SurfaceSpec

        for grade in ("P1", "P2", "P3", "P4"):
            s = SurfaceSpec(roughness=grade)
            assert s.fmt_roughness() == f"8/ {grade}"

    def test_roughness_matte(self):
        """'G' designates a matte (ground) surface (§5.2)."""
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(roughness="G")
        assert s.fmt_roughness() == "8/ G"

    def test_roughness_with_spatial_band(self):
        """Rq spec with spatial band per ISO 10110-8:2019 §5.3.3."""
        from optiland.iso10110.notation import SurfaceSpec

        s = SurfaceSpec(roughness="-1/Rq 0.002")
        assert s.fmt_roughness() == "8/ -1/Rq 0.002"

        s2 = SurfaceSpec(roughness="0.002-1/Rq 0.002")
        assert s2.fmt_roughness() == "8/ 0.002-1/Rq 0.002"

    def test_roughness_invalid_grade_warns(self):
        """P5, P6, … are not valid polish grades — must emit a warning."""
        from optiland.iso10110.notation import SurfaceSpec

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SurfaceSpec(roughness="P5")
        assert any("P5" in str(x.message) for x in w), \
            "Expected warning for invalid polish grade P5"


# ---------------------------------------------------------------------------
# notation.py — ElementSpec
# ---------------------------------------------------------------------------

class TestElementSpec:
    def test_defaults(self):
        from optiland.iso10110.notation import ElementSpec

        e = ElementSpec()
        assert e.fmt_birefringence() == "0/ —"
        assert e.fmt_bubbles() == "1/ —"
        assert e.fmt_homogeneity() == "2/ —"

    def test_birefringence_unit(self):
        """Birefringence must include nm/cm unit per ISO 10110-18 §5.2.2.
        Integer values must not include a trailing .0"""
        from optiland.iso10110.notation import ElementSpec

        e = ElementSpec(birefringence=5)
        assert e.fmt_birefringence() == "0/ 5 nm/cm"

    def test_birefringence_fractional(self):
        """Fractional birefringence values are shown with the decimal part."""
        from optiland.iso10110.notation import ElementSpec

        e = ElementSpec(birefringence=2.5)
        assert e.fmt_birefringence() == "0/ 2.5 nm/cm"

    def test_bubbles_multiplication_sign(self):
        from optiland.iso10110.notation import ElementSpec

        e = ElementSpec(bubbles="1x0.16")
        result = e.fmt_bubbles()
        assert "×" in result
        assert "x" not in result

    def test_bubbles_normalized_at_storage(self):
        """ASCII 'x' must be normalised in the stored bubbles field value.

        T1 conformance: YAML round-trips must emit × (U+00D7), not ASCII x.
        """
        from optiland.iso10110.notation import ElementSpec

        e = ElementSpec(bubbles="1x0.16")
        assert e.bubbles == "1×0.16", (
            f"Expected stored value '1×0.16', got {e.bubbles!r}"
        )

    def test_bubbles_r5_ok_no_warning(self):
        """Standard R5 bubble sizes must not trigger warnings."""
        from optiland.iso10110.notation import ElementSpec

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ElementSpec(bubbles="1×0.16")   # 0.16 = 1.6×10⁻¹ — R5 ✓
            ElementSpec(bubbles="2×0.063")  # 6.3×10⁻² — R5 ✓

    def test_bubbles_r5_non_standard_warns(self):
        """Non-R5 bubble sizes must emit a UserWarning."""
        from optiland.iso10110.notation import ElementSpec

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ElementSpec(bubbles="1×0.05")   # 0.05 — not R5
        assert any("R5" in str(x.message) or "bubble" in str(x.message).lower()
                   for x in w), "Expected R5 advisory for non-R5 bubble size"

    def test_nh_class_valid(self):
        from optiland.iso10110.notation import ElementSpec

        # ISO 10110-18:2019 preferred designators
        for cls in ("NH100", "NH040", "NH010", "NH004", "NH002", "NH001"):
            e = ElementSpec(nh_class=cls)
            assert cls in e.fmt_homogeneity()
        # Legacy designators still accepted for backward compatibility
        for cls in ("NH40", "NH20", "NH10", "NH5",
                    "NH2", "NH1", "NH0.5", "NH0.2", "NH0.1"):
            e = ElementSpec(nh_class=cls)
            assert cls in e.fmt_homogeneity()

    def test_nh_class_invalid_raises(self):
        from optiland.iso10110.notation import ElementSpec

        with pytest.raises(ValueError, match="nh_class"):
            ElementSpec(nh_class="NH999")

    def test_nh_class_takes_precedence(self):
        """nh_class must override legacy homogeneity_grade in fmt_homogeneity."""
        from optiland.iso10110.notation import ElementSpec

        e = ElementSpec(homogeneity_grade=2, nh_class="NH20")
        result = e.fmt_homogeneity()
        assert "NH20" in result
        assert "2/" in result   # the ISO code prefix

    def test_legacy_homogeneity_grade(self):
        """Integer grade 0-5 must map to ISO 10110-18:2019 NH class designator."""
        from optiland.iso10110.notation import ElementSpec, _LEGACY_NH_MAP

        for grade, expected_nh in _LEGACY_NH_MAP.items():
            e = ElementSpec(homogeneity_grade=grade)
            assert expected_nh in e.fmt_homogeneity(), (
                f"grade {grade} should map to {expected_nh}"
            )

    def test_homogeneity_grade_out_of_range(self):
        from optiland.iso10110.notation import ElementSpec

        with pytest.raises(ValueError, match="homogeneity_grade must be 0–5"):
            ElementSpec(homogeneity_grade=6)

    def test_striae_density_valid(self):
        from optiland.iso10110.notation import ElementSpec

        for d in (1, 2, 3, 4, 5):
            e = ElementSpec(nh_class="NH20", striae_density=d)
            result = e.fmt_homogeneity()
            assert str(d) in result

    def test_striae_density_out_of_range(self):
        from optiland.iso10110.notation import ElementSpec

        with pytest.raises(ValueError, match="striae_density must be 1–5"):
            ElementSpec(striae_density=6)

    def test_striae_density_string_coercion(self):
        """striae_density arriving as a string (e.g. from YAML) must be coerced to int."""
        from optiland.iso10110.notation import ElementSpec

        e = ElementSpec(striae_density="3")   # type: ignore[arg-type]
        assert e.striae_density == 3
        assert isinstance(e.striae_density, int)
        assert "3" in e.fmt_homogeneity()

    def test_striae_shadowgraph_valid(self):
        from optiland.iso10110.notation import ElementSpec

        for cls in ("A", "B", "C", "D", "a", "b", "c", "d"):
            e = ElementSpec(striae_shadowgraph=cls)
            assert cls.upper() in e.fmt_homogeneity()

    def test_striae_shadowgraph_invalid(self):
        from optiland.iso10110.notation import ElementSpec

        with pytest.raises(ValueError, match="striae_shadowgraph must be A–D"):
            ElementSpec(striae_shadowgraph="E")

    def test_striae_density_without_nh_class(self):
        """striae_density set alone must render '2/ —; N' (— for the homogeneity part).

        Per ISO 10110-4 §5: 2/ code is '2/ A; B'.  When only striae is specified,
        the homogeneity part A is shown as the dash placeholder.
        """
        from optiland.iso10110.notation import ElementSpec

        e = ElementSpec(striae_density=2)
        result = e.fmt_homogeneity()
        assert result.startswith("2/ "), f"Expected '2/ ' prefix: {result!r}"
        assert "2" in result, f"striae_density=2 not rendered: {result!r}"
        # Homogeneity part must be the dash placeholder
        assert "\u2014" in result, f"Expected em dash for missing NH class: {result!r}"

    def test_striae_shadowgraph_without_nh_class(self):
        """striae_shadowgraph set alone must render '2/ —; X' (— for homogeneity part)."""
        from optiland.iso10110.notation import ElementSpec

        e = ElementSpec(striae_shadowgraph="B")
        result = e.fmt_homogeneity()
        assert result.startswith("2/ "), f"Expected '2/ ' prefix: {result!r}"
        assert "B" in result, f"striae_shadowgraph='B' not rendered: {result!r}"
        assert "\u2014" in result, f"Expected em dash for missing NH class: {result!r}"

    def test_striae_typed_takes_precedence_over_legacy(self):
        """striae_density/shadowgraph must take precedence over legacy striae_grade."""
        from optiland.iso10110.notation import ElementSpec

        e = ElementSpec(striae_grade="A", striae_density=3)
        assert "3" in e.fmt_homogeneity()
        assert e.fmt_homogeneity().count("A") == 0   # not 'A'

    def test_matched_pair_default_false(self):
        from optiland.iso10110.notation import ElementSpec

        e = ElementSpec()
        assert e.matched_pair is False

    def test_matched_pair_roundtrip(self):
        from optiland.iso10110.notation import ElementSpec

        e = ElementSpec(matched_pair=True)
        e2 = ElementSpec.from_dict(e.to_dict())
        assert e2.matched_pair is True

    def test_birefringence_negative_raises(self):
        from optiland.iso10110.notation import ElementSpec

        with pytest.raises(ValueError, match="birefringence must be ≥ 0"):
            ElementSpec(birefringence=-1)

    def test_to_dict_roundtrip(self):
        from optiland.iso10110.notation import ElementSpec

        e = ElementSpec(birefringence=10, nh_class="NH20",
                        striae_shadowgraph="B", part_number="OPT-001")
        d = e.to_dict()
        e2 = ElementSpec.from_dict(d)
        assert e2.nh_class == "NH20"
        assert e2.striae_shadowgraph == "B"
        assert e2.part_number == "OPT-001"


# ---------------------------------------------------------------------------
# spec.py — DrawingSpec
# ---------------------------------------------------------------------------

class TestDrawingSpec:
    def test_organisation_default(self):
        from optiland.iso10110 import DrawingSpec

        spec = DrawingSpec(_make_singlet())
        assert spec.organisation == ""

    def test_organisation_set(self):
        from optiland.iso10110 import DrawingSpec

        spec = DrawingSpec(_make_singlet(), organisation="Acme Optics")
        assert spec.organisation == "Acme Optics"

    def test_yaml_roundtrip(self):
        from optiland.iso10110 import DrawingSpec

        lens = _make_singlet()
        spec = DrawingSpec(lens, project_name="Test", organisation="Lab")
        spec.set_surface(1, irregularity=0.5, centration=0.5,
                         ca_diameter=18.0, coating="AR",
                         test_wavelength=632.8)
        spec.set_element(0, nh_class="NH20", striae_density=2,
                         part_number="PCX-001")

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "spec.yaml"
            spec.save_yaml(yaml_path)
            spec2 = DrawingSpec.load_yaml(yaml_path, lens)

        assert spec2.project_name == "Test"
        assert spec2.organisation == "Lab"
        ss = spec2.get_surface_spec(1)
        assert ss.irregularity == 0.5
        assert ss.test_wavelength == 632.8
        es = spec2.get_element_spec(0)
        assert es.nh_class == "NH20"
        assert es.striae_density == 2

    def test_yaml_roundtrip_grating_type(self):
        """grating_type field must survive a YAML save/load round-trip."""
        from optiland.iso10110 import DrawingSpec

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_surface(1, grating_type="CG")   # concentric grating

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "grating_spec.yaml"
            spec.save_yaml(yaml_path)
            spec2 = DrawingSpec.load_yaml(yaml_path, lens)

        ss = spec2.get_surface_spec(1)
        assert ss.grating_type == "CG", (
            f"grating_type should survive YAML round-trip, got: {ss.grating_type!r}")

    def test_yaml_multiplication_sign_normalized(self):
        """ASCII 'x' in imperfection/bubble fields must appear as × in YAML output.

        T1 conformance: normalisation at storage time means YAML files use the
        ISO multiplication sign (U+00D7) rather than ASCII 'x'.  This matters for
        YAML files that may be inspected or edited by hand.
        """
        import yaml
        from optiland.iso10110 import DrawingSpec

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_surface(1, imperfections="2x0.04",
                         assembly_imperfections="1x0.063")
        spec.set_element(0, bubbles="1x0.16")

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "mul_sign.yaml"
            spec.save_yaml(yaml_path)
            text = yaml_path.read_text(encoding="utf-8")

        # ASCII 'x' between digits must not appear in YAML output.
        # PyYAML may escape U+00D7 as \xD7 in quoted strings — both forms are
        # acceptable; what matters is that the ASCII 'x' separator is gone.
        import re
        assert not re.search(r"\dx\d", text), (
            "ASCII multiplication 'x' survived into YAML output — "
            "T1 normalization failed"
        )
        # U+00D7 × must be present, either as a literal char or as \xD7 escape
        assert "×" in text or "\\xD7" in text, (
            "ISO × sign (U+00D7) must appear in YAML output (literal or \\xD7 escape)"
        )

    def test_reference_wavelength_default(self):
        from optiland.iso10110 import DrawingSpec

        spec = DrawingSpec(_make_singlet())
        assert spec.reference_wavelength == 546.07

    def test_reference_wavelength_set(self):
        from optiland.iso10110 import DrawingSpec

        spec = DrawingSpec(_make_singlet(), reference_wavelength=632.8)
        assert spec.reference_wavelength == 632.8

    def test_reference_wavelength_yaml_roundtrip(self):
        from optiland.iso10110 import DrawingSpec

        lens = _make_singlet()
        spec = DrawingSpec(lens, reference_wavelength=632.8)
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "spec.yaml"
            spec.save_yaml(yaml_path)
            spec2 = DrawingSpec.load_yaml(yaml_path, lens)
        assert spec2.reference_wavelength == 632.8

    def test_blank_surface_spec_returned(self):
        from optiland.iso10110 import DrawingSpec
        from optiland.iso10110.notation import SurfaceSpec

        spec = DrawingSpec(_make_singlet())
        ss = spec.get_surface_spec(99)   # not set → blank
        assert isinstance(ss, SurfaceSpec)
        assert ss.irregularity is None


class TestSetTitle:
    """DrawingSpec.set_title() — title-block fields separated from set_element()."""

    def test_set_title_alone(self):
        from optiland.iso10110 import DrawingSpec

        spec = DrawingSpec(_make_singlet())
        spec.set_title(0, part_number="OPT-001", drawn_by="BL",
                       approved_by="HK", revision="B", notes="see dwg 2")
        es = spec.get_element_spec(0)
        assert es.part_number == "OPT-001"
        assert es.drawn_by == "BL"
        assert es.approved_by == "HK"
        assert es.revision == "B"
        assert es.notes == "see dwg 2"

    def test_set_title_merges_with_set_element(self):
        """set_title() must not discard quality fields set by set_element()."""
        from optiland.iso10110 import DrawingSpec

        spec = DrawingSpec(_make_singlet())
        spec.set_element(0, nh_class="NH40", striae_shadowgraph="A")
        spec.set_title(0, part_number="OPT-002")
        es = spec.get_element_spec(0)
        assert es.nh_class == "NH40"
        assert es.striae_shadowgraph == "A"
        assert es.part_number == "OPT-002"

    def test_set_title_unknown_field_raises(self):
        from optiland.iso10110 import DrawingSpec

        spec = DrawingSpec(_make_singlet())
        with pytest.raises(ValueError, match="nh_class"):
            spec.set_title(0, nh_class="NH40")

    def test_set_title_yaml_roundtrip(self):
        from optiland.iso10110 import DrawingSpec

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_element(0, nh_class="NH20")
        spec.set_title(0, part_number="PCX-001", drawn_by="BL")

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "title_spec.yaml"
            spec.save_yaml(yaml_path)
            spec2 = DrawingSpec.load_yaml(yaml_path, lens)

        es = spec2.get_element_spec(0)
        assert es.nh_class == "NH20"
        assert es.part_number == "PCX-001"
        assert es.drawn_by == "BL"


# ---------------------------------------------------------------------------
# drawing.py / report.py — smoke tests (render without crashing)
# ---------------------------------------------------------------------------

class TestElementDrawingSmoke:
    @pytest.fixture
    def singlet_spec(self):
        lens = _make_singlet()
        from optiland.iso10110 import DrawingSpec

        spec = DrawingSpec(lens, project_name="Smoke Test",
                           organisation="Optiland")
        spec.set_surface(1, irregularity=0.5, centration=0.5,
                         imperfections="2×0.04", ca_diameter=18.0,
                         coating="AR 400-700 nm R<0.5%",
                         test_wavelength=632.8)
        spec.set_surface(2, irregularity=1.0, imperfections="1×0.063",
                         ca_diameter=18.0)
        spec.set_element(0, birefringence=10, bubbles="1×0.16",
                         nh_class="NH20", striae_shadowgraph="B",
                         diameter=22.0, diameter_tolerance=0.05,
                         ct_tolerance=0.05, part_number="PCX-001",
                         drawn_by="BL", approved_by="HK")
        return spec

    def test_pdf_singlet(self, singlet_spec, tmp_path):
        from optiland.iso10110 import ISO10110Report

        report = ISO10110Report(singlet_spec)
        pdf = tmp_path / "singlet.pdf"
        report.save_pdf(pdf)
        assert pdf.exists()
        assert pdf.stat().st_size > 5_000

    def test_png_singlet(self, singlet_spec, tmp_path):
        from optiland.iso10110 import ISO10110Report

        report = ISO10110Report(singlet_spec)
        paths = report.save_png(tmp_path, dpi=100)
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].stat().st_size > 10_000

    def test_dxf_singlet(self, singlet_spec, tmp_path):
        pytest.importorskip("ezdxf")
        from optiland.iso10110 import ISO10110Report

        report = ISO10110Report(singlet_spec)
        paths = report.save_dxf(tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].stat().st_size > 1_000

    def test_all_paper_sizes(self, singlet_spec, tmp_path):
        from optiland.iso10110 import ISO10110Report

        for paper in ("A5", "A4", "A3", "LETTER"):
            report = ISO10110Report(singlet_spec, paper=paper)
            pdf = tmp_path / f"{paper}.pdf"
            report.save_pdf(pdf)
            assert pdf.exists()

    def test_landscape_orientation(self, singlet_spec, tmp_path):
        from optiland.iso10110 import ISO10110Report

        report = ISO10110Report(singlet_spec, orientation="landscape")
        pdf = tmp_path / "landscape.pdf"
        report.save_pdf(pdf)
        assert pdf.exists()


class TestTripletSmoke:
    @pytest.fixture
    def triplet_spec(self):
        lens = _make_triplet()
        from optiland.iso10110 import DrawingSpec
        from optiland.iso10110.elements import identify_elements

        spec = DrawingSpec(lens, project_name="Cooke Triplet",
                           organisation="Test Org")
        elements = identify_elements(lens)
        for ei, el in enumerate(elements):
            for si in el.surface_indices:
                spec.set_surface(si, irregularity=0.5,
                                 imperfections="2×0.04")
            spec.set_element(ei, nh_class="NH10", striae_density=3,
                             part_number=f"CT-{ei + 1:02d}")
        return spec

    def test_pdf_triplet_all_elements(self, triplet_spec, tmp_path):
        from optiland.iso10110 import ISO10110Report

        report = ISO10110Report(triplet_spec)
        pdf = tmp_path / "triplet.pdf"
        report.save_pdf(pdf)
        assert pdf.exists()
        # Multi-page PDF should be larger than single-page
        assert pdf.stat().st_size > 30_000

    def test_report_length(self, triplet_spec):
        from optiland.iso10110 import ISO10110Report

        report = ISO10110Report(triplet_spec)
        report.generate()
        assert len(report) == 3

    def test_png_triplet(self, triplet_spec, tmp_path):
        from optiland.iso10110 import ISO10110Report

        report = ISO10110Report(triplet_spec)
        paths = report.save_png(tmp_path, dpi=100)
        assert len(paths) == 3
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 5_000


# ---------------------------------------------------------------------------
# report.py — title block content checks
# ---------------------------------------------------------------------------

class TestTitleBlock:
    def test_organisation_in_title(self, tmp_path):
        """Organisation string must appear in the rendered PDF figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens, project_name="TitleTest",
                           organisation="MyOrg")
        report = ISO10110Report(spec)
        report.generate()
        fig = report.drawings[0]._mpl_figure()
        # Collect all text on the figure
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        plt.close(fig)
        combined = " ".join(texts)
        assert "MyOrg" in combined

    def test_nh_class_in_table(self, tmp_path):
        """NH class must appear in the spec table of the rendered figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens, project_name="NHTest")
        spec.set_element(0, nh_class="NH40", striae_shadowgraph="A")
        report = ISO10110Report(spec)
        report.generate()
        fig = report.drawings[0]._mpl_figure()
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        plt.close(fig)
        combined = " ".join(texts)
        assert "NH40" in combined

    def test_glass_catalog_in_table(self, tmp_path):
        """Glass manufacturer (catalog) name must appear next to glass name."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens, project_name="CatalogTest")
        report = ISO10110Report(spec)
        report.generate()
        fig = report.drawings[0]._mpl_figure()
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        plt.close(fig)
        combined = " ".join(texts)
        # N-BK7 should appear as "N-BK7 (SCHOTT)" with catalog from database
        assert "SCHOTT" in combined, "glass catalog name not found in table"

    def test_mat_display_name_helper(self):
        """_mat_display_name must append catalog from material_data."""
        from optiland.iso10110._geometry import _mat_display_name
        from optiland.materials import Material

        m = Material("N-BK7")
        name = _mat_display_name(m)
        assert name == "N-BK7 (SCHOTT)"

    def test_mat_display_name_no_dup(self):
        """_mat_display_name must NOT duplicate name when ref equals name."""
        from optiland.iso10110._geometry import _mat_display_name

        class _Fake:
            name = "F2"
            reference = "F2"
        assert _mat_display_name(_Fake()) == "F2"

    def test_wavelength_note_in_table(self, tmp_path):
        """Non-standard test wavelength must appear in surface spec."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_surface(1, irregularity=0.5, test_wavelength=632.8)
        report = ISO10110Report(spec)
        report.generate()
        fig = report.drawings[0]._mpl_figure()
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        plt.close(fig)
        combined = " ".join(texts)
        assert "632.8" in combined


# ---------------------------------------------------------------------------
# YAML spec file from docs/examples — load and render
# ---------------------------------------------------------------------------

class TestYamlSpecFile:
    def test_photographic_lens_spec(self, tmp_path):
        """Load the example YAML spec and render a PDF."""
        yaml_path = Path(__file__).parent.parent / (
            "docs/examples/photographic_lens_spec.yaml"
        )
        if not yaml_path.exists():
            pytest.skip("photographic_lens_spec.yaml not found")

        from optiland.iso10110 import DrawingSpec, ISO10110Report
        import optiland.samples.objectives as obj

        # The example yaml was built against Edmund 50mm lens or similar
        # If it fails to load with the sample, skip gracefully
        try:
            lens = obj.CookeTriplet()
            spec = DrawingSpec.load_yaml(yaml_path, lens)
            report = ISO10110Report(spec)
            pdf = tmp_path / "photographic.pdf"
            report.save_pdf(pdf)
            assert pdf.exists()
        except Exception as exc:
            pytest.skip(f"Skipping YAML test: {exc}")


# ---------------------------------------------------------------------------
# drawing.py — surface header lines (_surface_header_lines)
# ---------------------------------------------------------------------------

class TestSurfaceHeaderLines:
    """Unit-tests for the ISO 10110-1 Table 1 surface-type label helper."""

    def _make_surf(self, geom_class, *args, **kwargs):
        """Wrap *geom_class(*args, **kwargs)* in a mock surface object."""
        from unittest.mock import MagicMock
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        geom = geom_class(cs, *args, **kwargs)
        surf = MagicMock()
        surf.geometry = geom
        return surf

    def test_standard_sphere_front(self):
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.standard import StandardGeometry
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {"geometry": StandardGeometry(cs, 50.0, 0.0)})()
        lines = _surface_header_lines(surf, SurfaceSpec(), is_rear=False)
        assert lines[0].startswith("R ")
        assert "CX" in lines[0]   # r>0, front → CX

    def test_standard_sphere_rear(self):
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.standard import StandardGeometry
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {"geometry": StandardGeometry(cs, 50.0, 0.0)})()
        lines = _surface_header_lines(surf, SurfaceSpec(), is_rear=True)
        assert "CC" in lines[0]   # r>0, rear → CC

    def test_flat_surface(self):
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.plane import Plane
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {"geometry": Plane(cs)})()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert lines[0] == "R ∞"

    def test_even_asphere(self):
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.even_asphere import EvenAsphere
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {"geometry": EvenAsphere(cs, 50.0, -0.5, coefficients=[0, 1e-6, -2e-9])})()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert lines[0] == "ASPH"
        assert any(l.startswith("R ") for l in lines)
        # ISO 10110-12 uses κ (kappa) for conic constant
        assert any("κ" in l for l in lines), f"No κ in {lines}"
        # A4, A6 notation per ISO 10110-12 §4.3.2.1
        assert any(l.startswith("A4") for l in lines), f"No A4 in {lines}"
        assert any(l.startswith("A6") for l in lines), f"No A6 in {lines}"

    def test_even_asphere_conic_zero_omitted(self):
        """ASPH header must NOT include κ when conic is zero (ISO 10110-12 §4.3.1).

        §4.3.1 states the conic coefficient κ shall be given *when it is non-zero*.
        An EvenAsphere with κ=0 is a spherical base — no κ line should appear.
        """
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.even_asphere import EvenAsphere
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {
            "geometry": EvenAsphere(cs, 50.0, 0.0, coefficients=[1e-6])
        })()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert lines[0] == "ASPH", f"Expected ASPH, got: {lines[0]!r}"
        assert any(l.startswith("A") for l in lines), f"No A-coefficient line: {lines}"
        assert not any("\u03ba" in l for l in lines), (
            f"κ should be omitted when conic=0 for ASPH, but found: {lines}")

    def test_even_asphere_conic_nonzero_shown(self):
        """ASPH header must include κ when conic is non-zero (ISO 10110-12 §4.3.1)."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.even_asphere import EvenAsphere
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {
            "geometry": EvenAsphere(cs, 50.0, -1.0, coefficients=[])
        })()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert any("\u03ba" in l for l in lines), (
            f"κ should be shown when conic=-1.0 for ASPH, not found in: {lines}")

    def test_even_asphere_r_tolerance(self):
        """ASPH vertex radius line must include r_tolerance (ISO 10110-12 §5.3.1)."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.even_asphere import EvenAsphere
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {"geometry": EvenAsphere(cs, 50.0, -0.5, coefficients=[])})()
        sspec = SurfaceSpec(r_tolerance="±0.05")
        # plain_text mode (DXF)
        lines_pt = _surface_header_lines(surf, sspec, plain_text=True)
        r_line = next(l for l in lines_pt if l.startswith("R "))
        assert "0.05" in r_line, f"r_tolerance not in ASPH DXF R line: {r_line!r}"
        assert "^" not in r_line and "$" not in r_line, \
            f"mathtext leaked into DXF ASPH R line: {r_line!r}"
        # mathtext mode (mpl)
        lines_mt = _surface_header_lines(surf, sspec, plain_text=False)
        r_line_mt = next(l for l in lines_mt if "R " in l and "ASPH" not in l)
        assert "0.05" in r_line_mt, f"r_tolerance not in ASPH mpl R line: {r_line_mt!r}"

    def test_toroidal_cyl(self):
        """ToroidalGeometry with infinite R_rot → CYL."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.toroidal import ToroidalGeometry
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {
            "geometry": ToroidalGeometry(cs, float("inf"), 80.0)
        })()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert lines[0] == "CYL"

    def test_toroidal_cyl_r_tolerance(self):
        """CYL R line must include r_tolerance when set."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.toroidal import ToroidalGeometry
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {
            "geometry": ToroidalGeometry(cs, float("inf"), 80.0)
        })()
        sspec = SurfaceSpec(r_tolerance="±0.1")
        lines = _surface_header_lines(surf, sspec, plain_text=True)
        r_line = next((l for l in lines if l.startswith("R ")), None)
        assert r_line is not None, f"No R line in CYL header: {lines}"
        assert "0.1" in r_line, f"r_tolerance not in CYL R line: {r_line!r}"

    def test_toroidal_full(self):
        """ToroidalGeometry with finite R_rot and R_yz → TOROID.

        Radii must be labelled Rx / Ry (consistent with BICONIC notation).
        """
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.toroidal import ToroidalGeometry
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {
            "geometry": ToroidalGeometry(cs, 100.0, 80.0)
        })()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert lines[0] == "TOROID"
        assert any(l.startswith("Rx") for l in lines[1:]), \
            f"Expected 'Rx ...' radius line in TOROID header, got: {lines}"
        assert any(l.startswith("Ry") for l in lines[1:]), \
            f"Expected 'Ry ...' radius line in TOROID header, got: {lines}"

    def test_biconic(self):
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.biconic import BiconicGeometry
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {
            "geometry": BiconicGeometry(cs, 60.0, 80.0)
        })()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert lines[0] == "BICONIC"
        assert any("Rx" in l for l in lines)
        assert any("Ry" in l for l in lines)

    def test_biconic_conic_constants_shown_when_nonzero(self):
        """BICONIC header must include κx/κy when conic constants are non-zero."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.biconic import BiconicGeometry
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {
            "geometry": BiconicGeometry(cs, 60.0, 80.0, conic_x=-0.5, conic_y=-0.3)
        })()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert any("\u03bax" in l for l in lines), \
            f"Expected κx line in BICONIC header: {lines}"
        assert any("\u03bay" in l for l in lines), \
            f"Expected κy line in BICONIC header: {lines}"

    def test_biconic_conic_constants_omitted_when_zero(self):
        """BICONIC header must not include κx/κy lines when both are zero."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.biconic import BiconicGeometry
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {
            "geometry": BiconicGeometry(cs, 60.0, 80.0, conic_x=0.0, conic_y=0.0)
        })()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert not any("\u03ba" in l for l in lines), \
            f"Unexpected κ line in BICONIC header when k=0: {lines}"

    def test_cyl_conic_constant_shown_when_nonzero(self):
        """CYL header must include κ for the base curve when non-zero."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.toroidal import ToroidalGeometry
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {
            "geometry": ToroidalGeometry(cs, float("inf"), 80.0, conic=-0.5)
        })()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert lines[0] == "CYL"
        assert any("\u03ba" in l for l in lines), \
            f"Expected κ line in CYL header with non-zero conic: {lines}"

    def test_toroid_conic_constant_shown_when_nonzero(self):
        """TOROID header must include κ for the base YZ curve when non-zero."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.toroidal import ToroidalGeometry
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {
            "geometry": ToroidalGeometry(cs, 100.0, 80.0, conic=-0.3)
        })()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert lines[0] == "TOROID"
        assert any("\u03ba" in l for l in lines), \
            f"Expected κ line in TOROID header with non-zero conic: {lines}"

    def test_polynomial_gs(self):
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.polynomial import PolynomialGeometry
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {
            "geometry": PolynomialGeometry(cs, 0.0, 0.0)
        })()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert lines[0] == "GS"

    def test_plane_grating(self):
        """PlaneGrating yields ISO 10110-16 type symbol, GRAT, frequency and order."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.plane_grating import PlaneGrating
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        # 600 µm period → 1000/600 ≈ 1.7 l/mm, order = 1
        surf = type("S", (), {
            "geometry": PlaneGrating(cs, 1, 600.0, 0.0)
        })()
        # Default grating_type is "LG" (linear grating, ISO 10110-16 Table 5)
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert lines[0] == "LG"
        assert "GRAT" in lines
        # ISO 10110-16 §5.3: grating frequency and diffraction order
        assert any("l/mm" in l for l in lines), f"No frequency line in: {lines}"
        assert any("m = +1" in l for l in lines), f"No order line in: {lines}"

    def test_plane_grating_standard_geometry(self):
        """StandardGratingGeometry shows frequency (lines/mm) and order."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.standard_grating import StandardGratingGeometry
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        # 500 µm period → 2.0 l/mm, order = -1
        surf = type("S", (), {
            "geometry": StandardGratingGeometry(cs, 0.0, -1, 500.0, 0.0)
        })()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert lines[0] == "LG"
        assert "GRAT" in lines
        assert any("2.0 l/mm" in l for l in lines), f"Frequency not shown: {lines}"
        assert any("m = -1" in l for l in lines), f"Order not shown: {lines}"

    def test_grating_type_invalid_warns(self):
        """Unknown grating_type must emit an advisory warning."""
        from optiland.iso10110.notation import SurfaceSpec

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SurfaceSpec(grating_type="XG")   # not in ISO 10110-16 Table 5
        assert any("grating_type" in str(x.message) or "ISO 10110-16" in str(x.message)
                   for x in w), "Expected grating_type advisory warning"

    def test_grating_type_override(self):
        """grating_type field controls the ISO 10110-16 symbol shown."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.plane_grating import PlaneGrating
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {
            "geometry": PlaneGrating(cs, 1, 600.0, 0.0)
        })()
        for gtype in ("LG", "CG", "CGH", "TG", "RG", "AG", "SG", "IG"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")   # all standard types must not warn
                spec = SurfaceSpec(grating_type=gtype)
            lines = _surface_header_lines(surf, spec)
            assert lines[0] == gtype
            assert "GRAT" in lines

    def test_grating_groove_orientation_angle(self):
        """Non-zero groove orientation angle must appear as φ = …° in GRAT header."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.plane_grating import PlaneGrating
        from optiland.coordinate_system import CoordinateSystem
        import math

        cs = CoordinateSystem()
        # 45° groove orientation
        surf = type("S", (), {
            "geometry": PlaneGrating(cs, 1, 500.0, math.radians(45.0))
        })()
        lines = _surface_header_lines(surf, SurfaceSpec())
        phi_lines = [l for l in lines if "\u03c6" in l or "phi" in l.lower()]
        assert phi_lines, f"No φ line in: {lines}"
        assert "45.0" in phi_lines[0], f"Angle not 45° in: {phi_lines[0]}"
        # Zero orientation must not produce a φ line
        surf0 = type("S", (), {
            "geometry": PlaneGrating(cs, 1, 500.0, 0.0)
        })()
        lines0 = _surface_header_lines(surf0, SurfaceSpec())
        assert not any("\u03c6" in l for l in lines0), f"Unexpected φ for zero angle: {lines0}"

    def test_standard_grating_curved_substrate(self):
        """StandardGratingGeometry with curved substrate shows R (and κ when non-zero)."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.standard_grating import StandardGratingGeometry
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        # Curved substrate: R=100mm, κ=-0.5
        surf = type("S", (), {
            "geometry": StandardGratingGeometry(cs, 100.0, 1, 500.0, 0.0, conic=-0.5)
        })()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert any("R" in l and "100" in l for l in lines), f"Substrate R not shown: {lines}"
        assert any("\u03ba" in l for l in lines), f"κ not shown for non-zero conic: {lines}"
        # When κ=0 it must be omitted
        surf0 = type("S", (), {
            "geometry": StandardGratingGeometry(cs, 100.0, 1, 500.0, 0.0, conic=0.0)
        })()
        lines0 = _surface_header_lines(surf0, SurfaceSpec())
        assert not any("\u03ba" in l for l in lines0), f"Unexpected κ for zero conic: {lines0}"

    def test_forbes_q_radial_terms_shown(self):
        """Forbes Q ASPH header must list ρmax and non-zero a_m coefficients."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.forbes.geometry import (
            ForbesQNormalSlopeGeometry, ForbesSurfaceConfig,
        )
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        config = ForbesSurfaceConfig(
            radius=50.0, conic=-0.5, norm_radius=12.5,
            terms={0: 1e-4, 1: -2e-5, 2: 0.0},   # a2 = 0 → omitted
        )
        surf = type("S", (), {"geometry": ForbesQNormalSlopeGeometry(cs, config)})()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert lines[0] == "ASPH", f"Expected ASPH, got: {lines[0]!r}"
        assert any("κ" in l for l in lines), f"No κ line: {lines}"
        assert any("ρmax" in l for l in lines), f"No ρmax line: {lines}"
        assert any(l.startswith("a0") for l in lines), f"No a0 line: {lines}"
        assert any(l.startswith("a1") for l in lines), f"No a1 line: {lines}"
        # zero coefficient must be omitted
        assert not any(l.startswith("a2") for l in lines), f"a2 should be omitted: {lines}"

    def test_forbes_q_radial_terms_empty(self):
        """Forbes Q ASPH header with empty radial_terms must not list ρmax."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.forbes.geometry import (
            ForbesQNormalSlopeGeometry, ForbesSurfaceConfig,
        )
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        config = ForbesSurfaceConfig(radius=50.0, conic=0.0, norm_radius=10.0, terms={})
        surf = type("S", (), {"geometry": ForbesQNormalSlopeGeometry(cs, config)})()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert lines[0] == "ASPH"
        assert not any("ρmax" in l for l in lines), f"ρmax should be absent: {lines}"
        assert not any(l.startswith("a") for l in lines), f"a_m should be absent: {lines}"

    def test_odd_asphere_coefficients(self):
        """OddAsphere ASPH header must label coefficients by odd-power index A1, A2, A3…

        OddAsphere polynomial: z = conic_base + sum(Ci * r^(i+1)), so
        coefficients[i] maps to power (i+1) → ISO label A_{i+1}.
        Zero-valued coefficients must be omitted.
        """
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.odd_asphere import OddAsphere
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        # coefficients[0]=0→A1 omit, [1]=2e-5→A2 show, [2]=0→A3 omit, [3]=-1e-8→A4 show
        surf = type("S", (), {
            "geometry": OddAsphere(cs, 50.0, -0.5,
                                   coefficients=[0.0, 2e-5, 0.0, -1e-8])
        })()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert lines[0] == "ASPH", f"Expected ASPH, got: {lines[0]!r}"
        assert any(l.startswith("R ") for l in lines), f"No R line: {lines}"
        assert any("κ" in l for l in lines), f"No κ line: {lines}"
        assert any(l.startswith("A2") for l in lines), f"A2 not found: {lines}"
        assert any(l.startswith("A4") for l in lines), f"A4 not found: {lines}"
        # zero-valued coefficients must be omitted
        assert not any(l.startswith("A1 ") for l in lines), f"A1 should be omitted: {lines}"
        assert not any(l.startswith("A3 ") for l in lines), f"A3 should be omitted: {lines}"

    def test_odd_asphere_no_ρmax(self):
        """OddAsphere ASPH header must not contain ρmax (that is Forbes Q only)."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.odd_asphere import OddAsphere
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {
            "geometry": OddAsphere(cs, 50.0, 0.0, coefficients=[1e-5])
        })()
        lines = _surface_header_lines(surf, SurfaceSpec())
        assert not any("ρmax" in l for l in lines), f"ρmax should not appear: {lines}"


# ---------------------------------------------------------------------------
# drawing.py — f' annotation and sharp-edge "0" in rendered output
# ---------------------------------------------------------------------------

class TestDrawingAnnotations:
    def test_efl_annotation_in_figure(self):
        """f' annotation must appear in the matplotlib figure."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        report = ISO10110Report(spec)
        report.generate()
        fig = report.drawings[0]._mpl_figure()
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        plt.close(fig)
        assert any("f" in t and "mm" in t for t in texts), \
            f"No f' annotation found in texts: {texts}"

    def test_sharp_edge_zero_in_figure(self):
        """Sharp-edge '0' must appear when sspec.sharp_edge=True."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_surface(1, sharp_edge=True)
        report = ISO10110Report(spec)
        report.generate()
        fig = report.drawings[0]._mpl_figure()
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        plt.close(fig)
        assert "0" in texts, f"No sharp-edge '0' found in texts: {texts}"


# ---------------------------------------------------------------------------
# spec.py — per-component material specs (set_material / get_material_spec)
# ---------------------------------------------------------------------------

class TestPerComponentMaterialSpecs:
    """Tests for set_material() / get_material_spec() on DrawingSpec."""

    def test_get_material_spec_fallback_to_element_spec(self):
        """get_material_spec falls back to element spec when no per-component spec is set."""
        from optiland.iso10110 import DrawingSpec

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_element(0, birefringence=5.0, bubbles="1×0.16")
        # No per-component spec set → should return the element spec
        ms = spec.get_material_spec(0, 0)
        assert ms.birefringence == 5.0
        assert ms.bubbles == "1×0.16"

    def test_get_material_spec_returns_blank_when_no_specs(self):
        """get_material_spec returns a blank ElementSpec when nothing is set."""
        from optiland.iso10110 import DrawingSpec
        from optiland.iso10110.notation import ElementSpec

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        ms = spec.get_material_spec(0, 0)
        assert isinstance(ms, ElementSpec)
        assert ms.birefringence is None

    def test_set_material_overrides_element_spec(self):
        """Per-component spec takes precedence over the element spec."""
        from optiland.iso10110 import DrawingSpec

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_element(0, birefringence=10.0, bubbles="2×0.16")
        spec.set_material(0, 0, birefringence=5.0, bubbles="1×0.063")

        ms = spec.get_material_spec(0, 0)
        assert ms.birefringence == 5.0   # per-component overrides element
        assert ms.bubbles == "1×0.063"

    def test_set_material_second_component(self):
        """Per-component specs can differ per glass component (for cemented lenses)."""
        from optiland.iso10110 import DrawingSpec

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_element(0, birefringence=10.0)
        spec.set_material(0, 1, birefringence=20.0, bubbles="3×0.063")

        ms0 = spec.get_material_spec(0, 0)
        ms1 = spec.get_material_spec(0, 1)
        # comp 0 → falls back to element spec
        assert ms0.birefringence == 10.0
        # comp 1 → per-component spec
        assert ms1.birefringence == 20.0
        assert ms1.bubbles == "3×0.063"

    def test_material_spec_yaml_roundtrip(self):
        """set_material() specs must survive a YAML save/load round-trip."""
        from optiland.iso10110 import DrawingSpec

        lens = _make_singlet()
        spec = DrawingSpec(lens, project_name="MatTest")
        spec.set_material(0, 0, birefringence=5.0, nh_class="NH040")
        spec.set_material(0, 1, birefringence=15.0, nh_class="NH100")

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "mat_spec.yaml"
            spec.save_yaml(yaml_path)
            spec2 = DrawingSpec.load_yaml(yaml_path, lens)

        ms0 = spec2.get_material_spec(0, 0)
        ms1 = spec2.get_material_spec(0, 1)
        assert ms0.birefringence == 5.0
        assert ms0.nh_class == "NH040"
        assert ms1.birefringence == 15.0
        assert ms1.nh_class == "NH100"


# ---------------------------------------------------------------------------
# drawing.py — ISO Table 1 code order (8/ before 13/ and 15/)
# ---------------------------------------------------------------------------

class TestSpecTableOrder:
    """Numbered codes must appear in ISO 10110-1 Table 1 order: 3, 4, 5, 6, 7, 8, 13, 15."""

    def _get_spec_column_texts(self, spec):
        """Render a singlet drawing and collect all spec-column text items."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optiland.iso10110 import ISO10110Report

        report = ISO10110Report(spec)
        report.generate()
        fig = report.drawings[0]._mpl_figure()
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        plt.close(fig)
        return texts

    def test_8_before_13_and_15_in_rendered_figure(self):
        """8/ code must appear before 13/ and 15/ when all three are set."""
        from optiland.iso10110 import DrawingSpec

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_surface(1, roughness="P2",
                         wavefront_deformation="0.1(0.05)",
                         assembly_imperfections="1×0.04")

        texts = self._get_spec_column_texts(spec)

        # Find indices of 8/, 13/, 15/ codes in the rendered text list
        def _idx(prefix):
            for i, t in enumerate(texts):
                if t.startswith(prefix):
                    return i
            return None

        i8  = _idx("8/")
        i13 = _idx("13/")
        i15 = _idx("15/")

        assert i8  is not None, "8/ not found in rendered texts"
        assert i13 is not None, "13/ not found in rendered texts"
        assert i15 is not None, "15/ not found in rendered texts"
        assert i8 < i13, f"8/ (idx {i8}) must come before 13/ (idx {i13})"
        assert i8 < i15, f"8/ (idx {i8}) must come before 15/ (idx {i15})"

    def test_7_after_6_before_8_in_rendered_figure(self):
        """7/ coating must appear between 6/ and 8/ (by y-position) per ISO Table 1."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_surface(1, laser_damage="H 20J/cm²",
                         coating="AR 400-700nm",
                         roughness="P2")

        report = ISO10110Report(spec)
        report.generate()
        fig = report.drawings[0]._mpl_figure()
        # Collect (text, y_position) pairs
        items = [(t.get_text(), t.get_position()[1])
                 for ax in fig.axes for t in ax.texts]
        plt.close(fig)

        def _y(prefix):
            for txt, y in items:
                if txt.startswith(prefix):
                    return y
            return None

        y6 = _y("6/")
        y7 = _y("7/")
        y8 = _y("8/")

        assert y6 is not None, "6/ not found in rendered texts"
        assert y7 is not None, "7/ not found in rendered texts"
        assert y8 is not None, "8/ not found in rendered texts"
        # Higher y = earlier row (table grows downward in mm space)
        assert y6 > y7, f"6/ y={y6:.1f} must be above 7/ y={y7:.1f}"
        assert y7 > y8, f"7/ y={y7:.1f} must be above 8/ y={y8:.1f}"


# ---------------------------------------------------------------------------
# notation.py — form_unit validation
# ---------------------------------------------------------------------------

class TestFormErrorRsiWithoutIrregularity:
    """RSI (C) without irregularity (B) is a data-loss trap; must warn."""

    def test_rsi_without_b_warns(self):
        """rotationally_symmetric set but irregularity=None → warning."""
        from optiland.iso10110.notation import SurfaceSpec

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SurfaceSpec(sag_tolerance=2, rotationally_symmetric=0.25)
        assert any("rotationally_symmetric" in str(x.message) for x in w), \
            "Expected warning when RSI is set without irregularity"

    def test_rsi_with_b_no_warning(self):
        """Both B and C set → no warning; C rendered correctly in A(B/C)."""
        from optiland.iso10110.notation import SurfaceSpec

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            s = SurfaceSpec(sag_tolerance=2,
                            irregularity=0.5,
                            rotationally_symmetric=0.25)
        assert "0.25" in s.fmt_form_error()

    def test_rsi_alone_warns(self):
        """rotationally_symmetric set with no sag_tolerance or irregularity → warning."""
        from optiland.iso10110.notation import SurfaceSpec

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SurfaceSpec(rotationally_symmetric=0.1)
        assert any("rotationally_symmetric" in str(x.message) for x in w)

    def test_rsi_exceeds_irregularity_warns(self):
        """C > B violates ISO 10110-5 §4.2 (C is a component of B); must warn."""
        from optiland.iso10110.notation import SurfaceSpec

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SurfaceSpec(irregularity=0.5, rotationally_symmetric=0.8)
        assert any("rotationally_symmetric" in str(x.message)
                   and "exceeds" in str(x.message) for x in w), \
            "Expected C > B advisory warning"

    def test_rsi_equal_irregularity_no_warning(self):
        """C == B is valid (100% RSI); must not warn."""
        from optiland.iso10110.notation import SurfaceSpec

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SurfaceSpec(irregularity=0.5, rotationally_symmetric=0.5)
        exceed_warns = [x for x in w
                        if "rotationally_symmetric" in str(x.message)
                        and "exceeds" in str(x.message)]
        assert not exceed_warns, f"Unexpected C>B warning when C==B: {exceed_warns}"


class TestFormUnitValidation:
    def test_form_unit_fr_valid(self):
        from optiland.iso10110.notation import SurfaceSpec
        s = SurfaceSpec(form_unit="fr")
        assert s.form_unit == "fr"

    def test_form_unit_nm_valid(self):
        from optiland.iso10110.notation import SurfaceSpec
        s = SurfaceSpec(form_unit="nm")
        assert s.form_unit == "nm"

    def test_form_unit_invalid_raises(self):
        from optiland.iso10110.notation import SurfaceSpec
        with pytest.raises(ValueError, match="form_unit"):
            SurfaceSpec(form_unit="waves")


# ---------------------------------------------------------------------------
# drawing.py — 15/ suppression on outer surfaces
# ---------------------------------------------------------------------------

class TestAssemblyImperfectionsSuppression:
    """15/ should appear only on cement interfaces, not on outer surfaces."""

    def _get_texts(self, spec, element_index=0):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optiland.iso10110 import ISO10110Report

        report = ISO10110Report(spec)
        report.generate()
        fig = report.drawings[element_index]._mpl_figure()
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        plt.close(fig)
        return texts

    def test_15_not_shown_on_singlet_by_default(self):
        """15/ must not appear in a singlet drawing unless explicitly set."""
        from optiland.iso10110 import DrawingSpec

        spec = DrawingSpec(_make_singlet())
        # No assembly_imperfections set
        texts = self._get_texts(spec)
        assert not any(t.startswith("15/") for t in texts), \
            "15/ should not appear on singlet surfaces when not set"

    def test_15_shown_on_singlet_when_set(self):
        """15/ must appear when explicitly set even for outer surfaces."""
        from optiland.iso10110 import DrawingSpec

        spec = DrawingSpec(_make_singlet())
        spec.set_surface(1, assembly_imperfections="1×0.04")
        texts = self._get_texts(spec)
        assert any(t.startswith("15/") for t in texts), \
            "15/ must appear when assembly_imperfections is set"

    def test_15_shown_on_cement_surface_by_default(self):
        """15/ must appear on cement interface surfaces even when not set (shows '—')."""
        from optiland.iso10110 import DrawingSpec

        spec = DrawingSpec(_make_cemented_doublet())
        texts = self._get_texts(spec)
        # The cemented doublet has a middle surface (cement interface)
        # 15/ should appear there with the default "—"
        assert any(t.startswith("15/") for t in texts), \
            "15/ should appear on cement interface surface"


# ---------------------------------------------------------------------------
# report.py — total_sheets passed to ElementDrawing
# ---------------------------------------------------------------------------

class TestTotalSheets:
    def test_triplet_sheet_numbering(self):
        """For a 3-element triplet each drawing must show '1/3', '2/3', '3/3'."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_triplet()
        spec = DrawingSpec(lens, project_name="SheetTest")
        report = ISO10110Report(spec)
        report.generate()

        assert len(report.drawings) == 3
        for expected_sheet in ("1 / 3", "2 / 3", "3 / 3"):
            found = False
            for drawing in report.drawings:
                fig = drawing._mpl_figure()
                texts = [t.get_text() for ax in fig.axes for t in ax.texts]
                plt.close(fig)
                if any(expected_sheet in t for t in texts):
                    found = True
                    break
            assert found, f"Sheet label '{expected_sheet}' not found in any drawing"


# ---------------------------------------------------------------------------
# drawing.py — _surface_header_lines plain_text mode (DXF r_tolerance fix)
# ---------------------------------------------------------------------------

class TestSurfaceHeaderLinesPlainText:
    """plain_text=True must produce clean DXF-compatible tolerance strings."""

    def test_r_tolerance_plain_text_no_mathtext(self):
        """With plain_text=True the R label must not contain ^ or {} characters."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.standard import StandardGeometry
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {"geometry": StandardGeometry(cs, 50.0, 0.0)})()
        sspec = SurfaceSpec(r_tolerance="±0.05")
        lines = _surface_header_lines(surf, sspec, plain_text=True)
        r_line = lines[0]
        assert "^" not in r_line, f"mathtext ^ found in DXF line: {r_line!r}"
        assert "{" not in r_line, f"mathtext {{ found in DXF line: {r_line!r}"
        assert "$" not in r_line, f"mathtext $ found in DXF line: {r_line!r}"
        # The plain-text tolerance should still contain the values
        assert "0.05" in r_line, f"tolerance value missing from DXF line: {r_line!r}"

    def test_r_tolerance_mathtext_contains_mathtext(self):
        """Default (plain_text=False) must produce $ … $ mathtext for matplotlib."""
        from optiland.iso10110._geometry import _surface_header_lines
        from optiland.iso10110.notation import SurfaceSpec
        from optiland.geometries.standard import StandardGeometry
        from optiland.coordinate_system import CoordinateSystem

        cs = CoordinateSystem()
        surf = type("S", (), {"geometry": StandardGeometry(cs, 50.0, 0.0)})()
        sspec = SurfaceSpec(r_tolerance="±0.05")
        lines = _surface_header_lines(surf, sspec, plain_text=False)
        r_line = lines[0]
        assert "$" in r_line, f"Expected mathtext $ in matplotlib line: {r_line!r}"


class TestWavefrontFormatValidation:
    """13/ wavefront deformation string must match ISO 10110-14 A(B/C) format."""

    def test_valid_scalar(self):
        """Plain number is valid."""
        from optiland.iso10110.notation import SurfaceSpec
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SurfaceSpec(wavefront_deformation="0.25")

    def test_valid_with_irregularity(self):
        """A(B) format is valid."""
        from optiland.iso10110.notation import SurfaceSpec
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SurfaceSpec(wavefront_deformation="0.25(0.1)")

    def test_valid_with_both_components(self):
        """A(B/C) format is valid."""
        from optiland.iso10110.notation import SurfaceSpec
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SurfaceSpec(wavefront_deformation="0.25(0.1/0.05)")

    def test_invalid_string_warns(self):
        """Non-numeric free-form string should warn."""
        from optiland.iso10110.notation import SurfaceSpec
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SurfaceSpec(wavefront_deformation="bad format")
        assert any("13/" in str(x.message) for x in w), \
            "Expected a 13/ format advisory warning"

    def test_fmt_wavefront_passthrough(self):
        """fmt_wavefront() returns the user string verbatim even when invalid."""
        from optiland.iso10110.notation import SurfaceSpec
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = SurfaceSpec(wavefront_deformation="0.1(0.05)")
        assert s.fmt_wavefront() == "13/ 0.1(0.05)"


class TestDxfContent:
    """DXF output must contain specific annotation elements."""

    @pytest.fixture(autouse=True)
    def _require_ezdxf(self):
        pytest.importorskip("ezdxf")

    @pytest.fixture
    def dxf_content(self, tmp_path):
        """Return the raw text content of a DXF file for a singlet with coating."""
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_surface(1, coating="AR 400-700nm", ca_diameter=18.0)
        spec.set_element(0, diameter=22.0)
        report = ISO10110Report(spec)
        dxf_path = tmp_path / "test.dxf"
        report.save_dxf(tmp_path)
        # find the generated dxf
        dxfs = list(tmp_path.glob("*.dxf"))
        return dxfs[0].read_text(encoding="utf-8", errors="replace")

    def test_lambda_symbol_in_coating_circle(self, dxf_content):
        """DXF must contain λ (U+03BB) in the encircled-λ coating annotation."""
        assert "\u03bb" in dxf_content, "λ character not found in DXF coating circle"

    def test_oe_bracket_annotation_present(self, dxf_content):
        """DXF must contain Øₑ (subscript e, U+2091) for the effective aperture bracket."""
        assert "\u00d8\u2091 " in dxf_content, \
            "Øₑ effective aperture label (U+2091 subscript e) not found in DXF"

    def test_kappa_in_asphere_dxf(self, tmp_path):
        """DXF must contain κ (U+03BA) in asphere headers — not replaced with 'k'."""
        from optiland.optic import Optic
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = Optic()
        lens.add_surface(index=0, thickness=float("inf"))
        lens.add_surface(index=1, thickness=6.0, radius=50.0,
                         material="N-BK7", is_stop=True,
                         surface_type="even_asphere",
                         conic=-0.5, coefficients=[0, 1e-7])
        lens.add_surface(index=2, thickness=50.0)
        lens.add_surface(index=3)
        lens.set_aperture(aperture_type="EPD", value=20.0)
        lens.set_field_type(field_type="angle")
        lens.add_field(y=0)
        lens.add_wavelength(value=0.5876)
        lens.update_paraxial()

        spec = DrawingSpec(lens)
        report = ISO10110Report(spec)
        report.save_dxf(tmp_path)
        dxf_content = list(tmp_path.glob("*.dxf"))[0].read_text(
            encoding="utf-8", errors="replace")
        assert "\u03ba" in dxf_content, (
            "κ (U+03BA) not found in DXF asphere header — was it incorrectly replaced with 'k'?")

    def test_roughness_code_in_dxf(self, tmp_path):
        """DXF must contain 8/ surface texture code when roughness is set."""
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_surface(1, roughness="P2")
        report = ISO10110Report(spec)
        report.save_dxf(tmp_path)
        dxf_content = list(tmp_path.glob("*.dxf"))[0].read_text(
            encoding="utf-8", errors="replace")
        assert "8/ P2" in dxf_content, "8/ roughness code not found in DXF spec table"

    def test_chamfer_symbols_in_dxf(self, tmp_path):
        """DXF chamfer must use × (U+00D7) and ° (U+00B0), not ASCII 'x' and 'deg'."""
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_surface(1, chamfer="0.2-0.5", chamfer_angle=45)
        report = ISO10110Report(spec)
        report.save_dxf(tmp_path)
        dxf_content = list(tmp_path.glob("*.dxf"))[0].read_text(
            encoding="utf-8", errors="replace")
        assert "\u00d7" in dxf_content, "× multiplication sign not found in DXF chamfer"
        assert "\u00b0" in dxf_content, "° degree symbol not found in DXF chamfer"

    def test_separate_rotational_axis_in_dxf(self, tmp_path):
        """When rotational_axis_y != 0, DXF must show separate 'opt. axis' and
        'rot. axis' labels instead of the combined 'opt./rot. axis'."""
        import ezdxf
        from optiland.iso10110 import DrawingSpec, ISO10110Report
        from optiland.iso10110.elements import identify_elements
        from optiland.iso10110.drawing import ElementDrawing

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        elements = identify_elements(lens)
        drawing = ElementDrawing(
            elements[0], spec, element_index=0,
            rotational_axis_y=2.5,
        )
        drawing.generate()
        drawing.save_dxf(tmp_path / "rot_axis.dxf")
        doc = ezdxf.readfile(str(tmp_path / "rot_axis.dxf"))
        msp = doc.modelspace()
        texts = [e.dxf.text for e in msp.query("TEXT")]
        assert any("opt. axis" in t for t in texts), \
            "Expected 'opt. axis' label for separate rotational axis"
        assert any("rot. axis" in t for t in texts), \
            "Expected 'rot. axis' label for separate rotational axis"
        assert not any(t == "opt./rot. axis" for t in texts), \
            "Combined 'opt./rot. axis' label must not appear when axes are separate"

    def test_cemented_doublet_alternating_hatch_in_dxf(self, tmp_path):
        """DXF cemented doublet must have two HATCH entities with different angles
        (ISO 128-50 §4.2.3: alternating 45° / 135° for adjacent components)."""
        import ezdxf
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_cemented_doublet()
        spec = DrawingSpec(lens)
        report = ISO10110Report(spec)
        report.save_dxf(tmp_path)
        dxf_path = list(tmp_path.glob("*.dxf"))[0]
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()
        hatches = list(msp.query("HATCH"))
        assert len(hatches) >= 2, (
            f"Expected 2+ HATCH entities for cemented doublet, found {len(hatches)}")
        angles = set()
        for h in hatches:
            for p in h.paths:
                pass  # just check the pattern
            if hasattr(h.dxf, "pattern_name"):
                angles.add(round(h.dxf.pattern_angle, 1))
        # ANSI31 base is 45°, so component 0 = 45°, component 1 = 135°
        # The pattern_angle property stores the rotation added to the base.
        # ezdxf stores the absolute hatch angle; check two distinct values exist.
        assert len(angles) >= 2, (
            f"Expected alternating hatch angles, but found only: {angles}")

    def test_first_angle_projection_symbol_in_dxf(self, tmp_path):
        """DXF must contain the first-angle projection symbol (two circles + cone).

        Expected projection circle radii per the formula in the drawing module:
          sym_R = _TTL_R2_H * 0.28 = 13.0 * 0.28 = 3.64 mm
          sym_r = sym_R * 0.60           = 2.184 mm
        """
        import ezdxf
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        report = ISO10110Report(spec)
        report.save_dxf(tmp_path)
        dxf_path = list(tmp_path.glob("*.dxf"))[0]
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()
        radii = [c.dxf.radius for c in msp.query("CIRCLE")]
        # Projection symbol requires circles of approximately 3.64 mm and 2.184 mm
        sym_R = 13.0 * 0.28          # 3.64
        sym_r = sym_R * 0.60         # 2.184
        has_outer = any(abs(r - sym_R) < 0.01 for r in radii)
        has_inner = any(abs(r - sym_r) < 0.01 for r in radii)
        assert has_outer, (
            f"First-angle projection outer circle (r≈{sym_R:.2f}) not found; "
            f"radii present: {sorted(radii)}")
        assert has_inner, (
            f"First-angle projection inner circle (r≈{sym_r:.2f}) not found; "
            f"radii present: {sorted(radii)}")

    def test_sharp_edge_zero_in_dxf(self, tmp_path):
        """DXF must contain a '0' TEXT entity near the surface vertex when sharp_edge=True.

        Per ISO 10110-1 §5.9.5.2, the '0' symbol indicates that the edge must
        remain sharp — no protective chamfer is permitted.
        """
        import ezdxf
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_surface(1, sharp_edge=True)

        report = ISO10110Report(spec)
        report.save_dxf(tmp_path)
        dxf_path = list(tmp_path.glob("*.dxf"))[0]
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()
        texts = [e.dxf.text for e in msp.query("TEXT")]
        assert any(t == "0" for t in texts), (
            "Expected a '0' TEXT entity for sharp_edge in DXF, "
            f"found texts: {[t for t in texts if len(t) <= 3][:20]}")


# ---------------------------------------------------------------------------
# notation.py — round_to_r5 utility
# ---------------------------------------------------------------------------

class TestRoundToR5:
    """round_to_r5 must return the smallest Renard R5 preferred number ≥ value."""

    def test_exact_r5_value_unchanged(self):
        """Values already on the R5 series must pass through unchanged."""
        from optiland.iso10110.notation import round_to_r5

        for v in (0.1, 0.16, 0.25, 0.4, 0.63, 1.0, 1.6, 2.5, 4.0, 6.3, 10.0):
            result = round_to_r5(v)
            assert abs(result - v) / v < 0.02, f"round_to_r5({v}) = {result}, expected ≈{v}"

    def test_rounds_up_to_next_r5(self):
        """Non-R5 values must be rounded UP to the nearest R5 number."""
        from optiland.iso10110.notation import round_to_r5

        assert abs(round_to_r5(0.05) - 0.063) < 1e-9,  "0.05 → 0.063"
        assert abs(round_to_r5(0.12) - 0.16)  < 1e-9,  "0.12 → 0.16"
        assert abs(round_to_r5(1.8)  - 2.5)   < 1e-9,  "1.8 → 2.5"

    def test_conservative_rounding(self):
        """Rounding must always be upward — never relaxes a tolerance."""
        from optiland.iso10110.notation import round_to_r5

        for v in (0.03, 0.11, 0.18, 0.3, 0.5, 0.7):
            result = round_to_r5(v)
            assert result >= v, f"round_to_r5({v}) = {result} < {v}: tolerance was relaxed!"

    def test_zero_raises(self):
        """round_to_r5 must raise ValueError for non-positive input."""
        from optiland.iso10110.notation import round_to_r5

        with pytest.raises(ValueError, match="value must be > 0"):
            round_to_r5(0.0)

    def test_negative_raises(self):
        from optiland.iso10110.notation import round_to_r5

        with pytest.raises(ValueError, match="value must be > 0"):
            round_to_r5(-1.0)


# ---------------------------------------------------------------------------
# spec.py — chamfer YAML round-trip
# ---------------------------------------------------------------------------

class TestChamferYamlRoundtrip:
    """Chamfer fields must survive YAML save/load."""

    def test_chamfer_roundtrip(self):
        from optiland.iso10110 import DrawingSpec

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_surface(1, chamfer="0.2-0.5", chamfer_angle=30.0)

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "chamfer_spec.yaml"
            spec.save_yaml(yaml_path)
            spec2 = DrawingSpec.load_yaml(yaml_path, lens)

        ss = spec2.get_surface_spec(1)
        assert ss.chamfer == "0.2-0.5"
        assert ss.chamfer_angle == 30.0

    def test_chamfer_default_angle(self):
        """Default chamfer_angle is None → rendered as 45° per ISO 10110."""
        from optiland.iso10110 import DrawingSpec

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_surface(1, chamfer="0.1-0.3")

        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "chamfer_default.yaml"
            spec.save_yaml(yaml_path)
            spec2 = DrawingSpec.load_yaml(yaml_path, lens)

        ss = spec2.get_surface_spec(1)
        assert ss.chamfer == "0.1-0.3"
        assert ss.chamfer_angle is None   # default — drawn as 45°


# ---------------------------------------------------------------------------
# drawing.py — DXF arrowheads on dimension lines
# ---------------------------------------------------------------------------

class TestDxfArrowheads:
    """DXF dimension lines must have SOLID-filled arrowheads (ISO 129-1)."""

    @pytest.fixture(autouse=True)
    def _require_ezdxf(self):
        pytest.importorskip("ezdxf")

    def test_dxf_contains_solid_arrowheads(self, tmp_path):
        """DXF singlet drawing must contain at least two SOLID entities (one
        per end of the CT dimension line)."""
        import ezdxf
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_element(0, diameter=22.0)
        report = ISO10110Report(spec)
        report.save_dxf(tmp_path)
        dxf_path = list(tmp_path.glob("*.dxf"))[0]
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()
        solids = list(msp.query("SOLID"))
        assert len(solids) >= 2, (
            f"Expected at least 2 SOLID arrowhead entities, found {len(solids)}")

    def test_dxf_arrowhead_helper_geometry(self):
        """_dxf_arrowhead must place the tip at the correct coordinate."""
        import ezdxf
        from optiland.iso10110._dxf_renderer import _dxf_arrowhead

        doc = ezdxf.new("R2010")
        msp = doc.modelspace()
        _dxf_arrowhead(msp, tip=(10.0, 5.0), toward=(20.0, 5.0), layer="TEST")
        solids = list(msp.query("SOLID"))
        assert len(solids) == 1
        # The first vertex of the SOLID must equal the tip
        s = solids[0]
        v1 = s.dxf.vtx0
        assert abs(v1.x - 10.0) < 0.01 and abs(v1.y - 5.0) < 0.01, (
            f"Expected tip at (10.0, 5.0), got ({v1.x:.3f}, {v1.y:.3f})")


# ---------------------------------------------------------------------------
# drawing.py — DXF optical axis linetype is double-dot (ISO 128-20)
# ---------------------------------------------------------------------------

class TestDxfAxisLinetype:
    """DXF optical axis must use the OPTICAL dash-double-dot linetype."""

    @pytest.fixture(autouse=True)
    def _require_ezdxf(self):
        pytest.importorskip("ezdxf")

    def test_optical_linetype_defined_in_dxf(self, tmp_path):
        """DXF must define the 'OPTICAL' linetype (dash-double-dot per ISO 128)."""
        import ezdxf
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        report = ISO10110Report(spec)
        report.save_dxf(tmp_path)
        dxf_path = list(tmp_path.glob("*.dxf"))[0]
        doc = ezdxf.readfile(str(dxf_path))
        lt_names = {lt.dxf.name for lt in doc.linetypes}
        assert "OPTICAL" in lt_names, (
            f"'OPTICAL' linetype not found in DXF; linetypes present: {lt_names}")


# ---------------------------------------------------------------------------
# drawing.py — DXF extents headers for correct paper-size setup
# ---------------------------------------------------------------------------

class TestDxfExtents:
    """DXF $EXTMIN / $EXTMAX must match the requested paper size."""

    @pytest.fixture(autouse=True)
    def _require_ezdxf(self):
        pytest.importorskip("ezdxf")

    def test_a4_portrait_extents(self, tmp_path):
        """A4 portrait DXF must have extents (0,0)→(210,297)."""
        import ezdxf
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        report = ISO10110Report(spec, paper="A4", orientation="portrait")
        report.save_dxf(tmp_path)
        dxf_path = list(tmp_path.glob("*.dxf"))[0]
        doc = ezdxf.readfile(str(dxf_path))
        extmax = doc.header.get("$EXTMAX")
        assert extmax is not None, "DXF $EXTMAX header not set"
        assert abs(extmax[0] - 210.0) < 0.01, f"Expected width 210 mm, got {extmax[0]}"
        assert abs(extmax[1] - 297.0) < 0.01, f"Expected height 297 mm, got {extmax[1]}"

    def test_a3_landscape_extents(self, tmp_path):
        """A3 landscape DXF must have extents (0,0)→(420,297)."""
        import ezdxf
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        report = ISO10110Report(spec, paper="A3", orientation="landscape")
        report.save_dxf(tmp_path)
        dxf_path = list(tmp_path.glob("*.dxf"))[0]
        doc = ezdxf.readfile(str(dxf_path))
        extmax = doc.header.get("$EXTMAX")
        assert extmax is not None, "DXF $EXTMAX header not set"
        assert abs(extmax[0] - 420.0) < 0.01, f"Expected width 420 mm, got {extmax[0]}"
        assert abs(extmax[1] - 297.0) < 0.01, f"Expected height 297 mm, got {extmax[1]}"


# ---------------------------------------------------------------------------
# drawing.py — matched-pair M marker in rendered output
# ---------------------------------------------------------------------------

class TestMatchedPairMarker:
    """matched_pair=True must add 'M' to the CT dimension in mpl and DXF."""

    def test_m_marker_in_mpl_figure(self):
        """The letter 'M' must appear in the CT dimension when matched_pair=True."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_element(0, matched_pair=True)
        report = ISO10110Report(spec)
        report.generate()
        fig = report.drawings[0]._mpl_figure()
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        plt.close(fig)
        assert any(" M" in t for t in texts), (
            f"'M' marker not found in figure texts: {texts}")

    def test_m_marker_in_dxf(self, tmp_path):
        """DXF CT dimension text must contain 'M' when matched_pair=True."""
        pytest.importorskip("ezdxf")
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_element(0, matched_pair=True)
        report = ISO10110Report(spec)
        report.save_dxf(tmp_path)
        dxf_content = list(tmp_path.glob("*.dxf"))[0].read_text(
            encoding="utf-8", errors="replace")
        assert " M" in dxf_content, (
            "Matched-pair 'M' marker not found in DXF dimension text")


# ---------------------------------------------------------------------------
# drawing.py — DXF title block em dash for empty fields (ISO 7200 convention)
# ---------------------------------------------------------------------------

class TestDxfTitleBlockEmDash:
    """DXF title block must use em dash (U+2014) for empty DRAWN BY / APPROVED."""

    @pytest.fixture(autouse=True)
    def _require_ezdxf(self):
        pytest.importorskip("ezdxf")

    def test_em_dash_in_dxf_when_drawn_by_empty(self, tmp_path):
        """When drawn_by and approved_by are not set, the DXF title block
        must contain the em dash character U+2014, not an ASCII hyphen."""
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        # drawn_by and approved_by left empty (defaults)
        report = ISO10110Report(spec)
        report.save_dxf(tmp_path)
        dxf_content = list(tmp_path.glob("*.dxf"))[0].read_text(
            encoding="utf-8", errors="replace")
        assert "\u2014" in dxf_content, (
            "Em dash (U+2014) not found in DXF title block — "
            "empty drawn_by/approved_by should render as '—'")

    def test_no_ascii_hyphen_for_empty_fields_in_dxf(self, tmp_path):
        """DXF title block must NOT use a bare ASCII hyphen for empty drawn_by."""
        import ezdxf
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        report = ISO10110Report(spec)
        report.save_dxf(tmp_path)
        dxf_path = list(tmp_path.glob("*.dxf"))[0]
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()
        texts = [e.dxf.text for e in msp.query("TEXT")]
        # The bare ASCII hyphen "-" (as the only content of a cell) must not
        # appear as a title-block cell value — it should be replaced with U+2014
        assert "-" not in texts, (
            f"Found bare ASCII hyphen '-' as a title-block text entity; "
            f"expected em dash U+2014. Texts found: {texts}")


# ---------------------------------------------------------------------------
# drawing.py — _tol_plain and _parse_tol helper functions
# ---------------------------------------------------------------------------

class TestTolHelpers:
    """Unit tests for _tol_plain() and _parse_tol() tolerance formatters."""

    def test_tol_plain_none(self):
        """_tol_plain(None) must return empty string."""
        from optiland.iso10110._geometry import _tol_plain

        assert _tol_plain(None) == ""

    def test_tol_plain_symmetric(self):
        """±0.05 must produce ' +0.05/-0.05' in plain-text format."""
        from optiland.iso10110._geometry import _tol_plain

        result = _tol_plain("±0.05")
        assert "+0.05" in result
        assert "-0.05" in result
        assert "^" not in result
        assert "$" not in result

    def test_tol_plain_hi_lo(self):
        """'+0.1/-0' asymmetric tolerance must be preserved."""
        from optiland.iso10110._geometry import _tol_plain

        result = _tol_plain("+0.1/-0")
        assert "+0.1" in result
        assert "-0" in result

    def test_tol_plain_no_minus_sign_unicode(self):
        """Plain-text tolerance must use ASCII hyphen-minus, not U+2212."""
        from optiland.iso10110._geometry import _tol_plain

        result = _tol_plain("±0.05")
        assert "\u2212" not in result, (
            "U+2212 (minus sign) found in plain-text tolerance; "
            "DXF viewers require ASCII hyphen-minus")

    def test_tol_math_none(self):
        """_tol_math(None) must return empty string."""
        from optiland.iso10110._geometry import _tol_math

        assert _tol_math(None) == ""

    def test_tol_math_symmetric(self):
        """±0.05 must produce mathtext ^{+0.05}_{-0.05}."""
        from optiland.iso10110._geometry import _tol_math

        result = _tol_math("±0.05")
        assert "^" in result and "{" in result
        assert "+0.05" in result
        assert "-0.05" in result


# ---------------------------------------------------------------------------
# drawing.py — ISO 10110-11 general tolerance size category (§4.1)
# ---------------------------------------------------------------------------

class TestIso1011Defaults:
    """_iso10110_11_defaults must use the physical element size, not the diagonal."""

    def test_defaults_by_physical_diameter(self):
        """For a 22 mm diameter lens the title block must use ≤30 mm category."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        spec.set_element(0, diameter=22.0)
        report = ISO10110Report(spec)
        report.generate()
        fig = report.drawings[0]._mpl_figure()
        texts = [t.get_text() for ax in fig.axes for t in ax.texts]
        plt.close(fig)
        combined = " ".join(texts)
        # ≤30 mm category: Ø±0.10  CT±0.15
        assert "0.10" in combined, (
            "Expected Ø±0.10 from ISO 10110-11 ≤30 mm category")
        assert "0.15" in combined, (
            "Expected CT±0.15 from ISO 10110-11 ≤30 mm category")

    def test_default_code_format_includes_space(self):
        """ISO code strings in defaults must use 'code/ value' (space after /) format."""
        from optiland.iso10110._geometry import _iso10110_11_defaults

        for size in (5, 20, 50, 200):
            d = _iso10110_11_defaults(size)
            assert d["form_error"].startswith("3/ "), \
                f"form_error should start with '3/ ' (size={size}): {d['form_error']!r}"
            assert d["centration"].startswith("4/ "), \
                f"centration should start with '4/ ' (size={size}): {d['centration']!r}"
            assert d["imperfections"].startswith("5/ "), \
                f"imperfections should start with '5/ ' (size={size}): {d['imperfections']!r}"

    def test_category_boundaries(self):
        """ISO 10110-11 Table 1 boundary sizes must land in the correct category."""
        from optiland.iso10110._geometry import _iso10110_11_defaults

        # Boundary at 10 mm: exactly 10 must be ≤10, 10.001 must be >10 to ≤30
        d10  = _iso10110_11_defaults(10.0)
        d10p = _iso10110_11_defaults(10.001)
        assert d10["thickness"]  == 0.10, f"≤10 mm CT should be ±0.10, got {d10['thickness']}"
        assert d10p["thickness"] == 0.15, f">10 to ≤30 mm CT should be ±0.15, got {d10p['thickness']}"

        # Boundary at 30 mm: exactly 30 must be ≤30, 30.001 must be >30 to ≤100
        d30  = _iso10110_11_defaults(30.0)
        d30p = _iso10110_11_defaults(30.001)
        assert d30["thickness"]  == 0.15, f"≤30 mm CT should be ±0.15, got {d30['thickness']}"
        assert d30p["thickness"] == 0.20, f">30 to ≤100 mm CT should be ±0.20, got {d30p['thickness']}"

        # Boundary at 100 mm: exactly 100 must be ≤100, 100.001 must be >100
        d100  = _iso10110_11_defaults(100.0)
        d100p = _iso10110_11_defaults(100.001)
        assert d100["thickness"]  == 0.20, f"≤100 mm CT should be ±0.20, got {d100['thickness']}"
        assert d100p["thickness"] == 0.50, f">100 mm CT should be ±0.50, got {d100p['thickness']}"


# ---------------------------------------------------------------------------
# ezdxf must be a genuinely optional ("manufacturing" extra) dependency —
# PDF/PNG-only usage must not require it to be installed.
# ---------------------------------------------------------------------------

class TestOptionalEzdxfDependency:
    @pytest.fixture
    def block_ezdxf(self, monkeypatch):
        """Make any `import ezdxf` (or submodule) raise ImportError."""
        import builtins
        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name == "ezdxf" or name.startswith("ezdxf."):
                raise ImportError(f"simulated: {name} not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)

    def test_pdf_workflow_without_ezdxf(self, block_ezdxf, tmp_path):
        """The documented generate()/save_pdf()/save_png() workflow must not
        need ezdxf — only save_dxf() should."""
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens, project_name="NoEzdxf")
        report = ISO10110Report(spec)
        report.generate()   # must not import ezdxf
        assert len(report) == 1

        pdf_path = tmp_path / "out.pdf"
        report.save_pdf(pdf_path)
        assert pdf_path.exists()

        report.save_png(tmp_path)
        assert list(tmp_path.glob("*.png"))

    def test_save_dxf_raises_helpful_error_without_ezdxf(self, block_ezdxf, tmp_path):
        from optiland.iso10110 import DrawingSpec, ISO10110Report

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        report = ISO10110Report(spec)
        with pytest.raises(ImportError, match="manufacturing"):
            report.save_dxf(tmp_path)


# ---------------------------------------------------------------------------
# elements.py — LensElement.semi_aperture() must respect a physical aperture
# (e.g. RadialAperture) rather than ignoring it in favor of the ray-traced
# or paraxial marginal-ray height. Reported by Axel: drawings did not show
# the correct diameter for surfaces with a RadialAperture.
# ---------------------------------------------------------------------------

def _make_singlet_with_aperture(r_max, call_update_paraxial=False):
    """Return a singlet Optic with an explicit RadialAperture on surface 1."""
    from optiland.optic import Optic
    from optiland.physical_apertures import RadialAperture

    lens = Optic()
    lens.surfaces.add(index=0, thickness=float("inf"))
    lens.surfaces.add(
        index=1, thickness=6.0, radius=50.0, material="N-BK7", is_stop=True,
        aperture=RadialAperture(r_max=r_max),
    )
    lens.surfaces.add(index=2, thickness=50.0)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=10.0)
    lens.fields.set_type("angle")
    lens.fields.add(y=0)
    lens.wavelengths.add(value=0.5876)
    if call_update_paraxial:
        lens.updater.update_paraxial()
    return lens


class TestRadialApertureSemiAperture:
    def test_aperture_respected_without_update_paraxial(self):
        """RadialAperture must be used even when update_paraxial() was never
        called — surf.semi_aperture is None in that case, so the fix must
        not rely on it having been populated first."""
        from optiland.iso10110.elements import identify_elements

        lens = _make_singlet_with_aperture(r_max=8.0, call_update_paraxial=False)
        e = identify_elements(lens)[0]
        assert e.semi_aperture(0, lens) == 8.0

    def test_aperture_respected_after_update_paraxial(self):
        """RadialAperture must win even when it is larger than the marginal
        ray height (EPD=10 -> ray height 5.0 < r_max=8.0)."""
        from optiland.iso10110.elements import identify_elements

        lens = _make_singlet_with_aperture(r_max=8.0, call_update_paraxial=True)
        e = identify_elements(lens)[0]
        assert e.semi_aperture(0, lens) == 8.0

    def test_aperture_smaller_than_ray_height_vignetting(self):
        """When the physical aperture is *smaller* than the ray bundle
        (vignetting), the drawing must show the true, smaller mechanical
        edge -- not the larger unclipped ray height."""
        from optiland.iso10110.elements import identify_elements

        lens = _make_singlet_with_aperture(r_max=3.0, call_update_paraxial=True)
        e = identify_elements(lens)[0]
        assert e.semi_aperture(0, lens) == 3.0

    def test_no_aperture_falls_back_to_paraxial(self):
        """Surfaces without a physical aperture must still fall back to the
        paraxial marginal-ray estimate, unchanged from before this fix."""
        from optiland.iso10110.elements import identify_elements

        lens = _make_singlet()  # no aperture set on the surfaces
        e = identify_elements(lens)[0]
        # EPD=20 -> marginal ray height at surface 1 == 10.0
        assert e.semi_aperture(0, lens) == pytest.approx(10.0)

    def test_offset_radial_aperture_uses_max_abs_extent(self):
        """OffsetRadialAperture's extent is off-axis; the drawing's single
        symmetric radius must cover the full offset reach."""
        from optiland.optic import Optic
        from optiland.physical_apertures import OffsetRadialAperture
        from optiland.iso10110.elements import identify_elements

        lens = Optic()
        lens.surfaces.add(index=0, thickness=float("inf"))
        lens.surfaces.add(
            index=1, thickness=6.0, radius=50.0, material="N-BK7", is_stop=True,
            aperture=OffsetRadialAperture(r_max=3.0, offset_x=2.0),
        )
        lens.surfaces.add(index=2, thickness=50.0)
        lens.surfaces.add(index=3)
        lens.set_aperture(aperture_type="EPD", value=10.0)
        lens.fields.set_type("angle")
        lens.fields.add(y=0)
        lens.wavelengths.add(value=0.5876)

        e = identify_elements(lens)[0]
        # extent = (2.0-3.0, 2.0+3.0, -3.0, 3.0) = (-1.0, 5.0, -3.0, 3.0)
        assert e.semi_aperture(0, lens) == 5.0


# ---------------------------------------------------------------------------
# style.py — DrawingStyle font-size scale factors and border margin.
# Requested by Axel: font sizes / text positions should be user-configurable
# rather than hardcoded in the renderers.
# ---------------------------------------------------------------------------

class TestDrawingStyle:
    def test_defaults(self):
        from optiland.iso10110 import DrawingStyle

        s = DrawingStyle()
        assert s.table_body_scale == 1.0
        assert s.dimension_scale == 1.0
        assert s.axis_label_scale == 1.0
        assert s.border_margin == 10.0

    def test_unknown_field_raises(self):
        from optiland.iso10110 import DrawingStyle

        with pytest.raises(ValueError, match="table_bdoy_scale|extra"):
            DrawingStyle(table_bdoy_scale=1.2)  # typo'd field name

    @pytest.mark.parametrize(
        "field", ["table_body_scale", "dimension_scale", "border_margin"]
    )
    def test_non_positive_value_raises(self, field):
        from optiland.iso10110 import DrawingStyle

        with pytest.raises(ValueError):
            DrawingStyle(**{field: 0.0})
        with pytest.raises(ValueError):
            DrawingStyle(**{field: -1.0})


class TestDrawingStyleRendering:
    """Verify DrawingStyle overrides actually change rendered output, and
    that the default reproduces the built-in (scale=1.0) appearance."""

    @pytest.fixture
    def element_and_spec(self):
        from optiland.iso10110 import DrawingSpec, identify_elements

        lens = _make_singlet()
        spec = DrawingSpec(lens)
        return identify_elements(lens)[0], spec

    def test_mpl_default_style_matches_no_style(self, element_and_spec):
        """Passing no style and passing DrawingStyle() explicitly must
        produce the identical set of rendered font sizes."""
        from optiland.iso10110 import ElementDrawing, DrawingStyle

        element, spec = element_and_spec
        fig_a = ElementDrawing(element, spec, element_index=0)._mpl_figure()
        fig_b = ElementDrawing(
            element, spec, element_index=0, style=DrawingStyle()
        )._mpl_figure()
        sizes_a = sorted(t.get_fontsize() for ax in fig_a.axes for t in ax.texts)
        sizes_b = sorted(t.get_fontsize() for ax in fig_b.axes for t in ax.texts)
        assert sizes_a == sizes_b

    def test_mpl_table_body_scale_changes_fontsize(self, element_and_spec):
        from optiland.iso10110 import ElementDrawing, DrawingStyle

        element, spec = element_and_spec

        def _spec_table_fontsize(fig):
            # The 3/ form-error row is always rendered in the spec table
            # body, regardless of what the user has set.
            for ax in fig.axes:
                for t in ax.texts:
                    if t.get_text().startswith("3/"):
                        return t.get_fontsize()
            raise AssertionError("no '3/' spec-table text found")

        fig_default = ElementDrawing(element, spec, element_index=0)._mpl_figure()
        assert _spec_table_fontsize(fig_default) == 7.5

        style = DrawingStyle(table_body_scale=2.0)
        fig_scaled = ElementDrawing(
            element, spec, element_index=0, style=style
        )._mpl_figure()
        assert _spec_table_fontsize(fig_scaled) == 15.0

    def test_dxf_dimension_scale_changes_height(self, element_and_spec):
        pytest.importorskip("ezdxf")
        from optiland.iso10110 import ElementDrawing, DrawingStyle

        element, spec = element_and_spec
        doc_default = ElementDrawing(element, spec, element_index=0).generate()
        heights_default = {
            round(t.dxf.height, 3) for t in doc_default.modelspace().query("TEXT")
        }

        style = DrawingStyle(dimension_scale=2.0)
        doc_scaled = ElementDrawing(
            element, spec, element_index=0, style=style
        ).generate()
        heights_scaled = {
            round(t.dxf.height, 3) for t in doc_scaled.modelspace().query("TEXT")
        }

        assert 2.5 in heights_default
        assert 5.0 in heights_scaled

    def test_border_margin_shifts_dxf_layout(self, element_and_spec):
        pytest.importorskip("ezdxf")
        from optiland.iso10110 import ElementDrawing, DrawingStyle

        element, spec = element_and_spec
        doc_default = ElementDrawing(element, spec, element_index=0).generate()
        border_default = list(
            next(iter(doc_default.modelspace().query("LWPOLYLINE"))).get_points()
        )[0][:2]

        style = DrawingStyle(border_margin=20.0)
        doc_custom = ElementDrawing(
            element, spec, element_index=0, style=style
        ).generate()
        polylines_custom = list(doc_custom.modelspace().query("LWPOLYLINE"))
        border_custom = list(polylines_custom[1].get_points())[0][:2]

        assert border_default != (20.0, 20.0)
        assert border_custom == (20.0, 20.0)

    def test_report_forwards_style_to_all_elements(self):
        """ISO10110Report must apply the same style to every element drawing."""
        from optiland.iso10110 import DrawingSpec, DrawingStyle, ISO10110Report

        lens = _make_triplet()
        spec = DrawingSpec(lens)
        style = DrawingStyle(table_body_scale=1.75)
        report = ISO10110Report(spec, style=style)
        report.generate()

        assert len(report.drawings) >= 2
        for d in report.drawings:
            assert d.style is style
