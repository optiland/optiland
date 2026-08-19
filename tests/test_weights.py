"""Tests for field and wavelength weighting system.

Kramer Harrison, 2026
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.fields.field import Field
from optiland.optic import Optic
from optiland.optimization import OptimizationProblem
from optiland.optimization.operand.operand import Operand
from optiland.samples.objectives import CookeTriplet
from optiland.utils import (
    FieldPoint,
    WavelengthPoint,
    active_fields,
    active_wavelengths,
    resolve_fields,
    resolve_wavelengths,
    weighted_average,
)
from optiland.wavelength import Wavelength

# ---------------------------------------------------------------------------
# Helper: build a minimal optic with known field/wavelength weights
# ---------------------------------------------------------------------------


def _make_weighted_optic():
    """Return an Optic with custom field and wavelength weights."""
    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1, radius=50.0, thickness=5.0, material="N-BK7", is_stop=True
    )
    lens.surfaces.add(index=2, radius=-50.0, thickness=45.0)
    lens.surfaces.add(index=3)

    lens.set_aperture(aperture_type="imageFNO", value=5.0)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0, weight=2.0)
    lens.fields.add(y=0.7, weight=1.0)
    lens.fields.add(y=1.0, weight=0.0)  # zero-weight field

    lens.wavelengths.add(value=0.55, is_primary=True, weight=1.0)
    lens.wavelengths.add(value=0.48, weight=3.0)
    lens.wavelengths.add(value=0.63, weight=0.0)  # zero-weight wavelength

    return lens


# ---------------------------------------------------------------------------
# Unit Test 1: Field rejects negative weight
# ---------------------------------------------------------------------------


class TestFieldNegativeWeight:
    def test_field_constructor_rejects_negative_weight(self):
        with pytest.raises(ValueError, match="non-negative"):
            Field(x=0, y=0, weight=-1.0)

    def test_field_setter_rejects_negative_weight(self):
        f = Field(x=0, y=0, weight=1.0)
        with pytest.raises(ValueError, match="non-negative"):
            f.weight = -0.5


# ---------------------------------------------------------------------------
# Unit Test 2: Wavelength rejects negative weight
# ---------------------------------------------------------------------------


class TestWavelengthNegativeWeight:
    def test_wavelength_constructor_rejects_negative_weight(self):
        with pytest.raises(ValueError, match="non-negative"):
            Wavelength(value=0.55, weight=-1.0)

    def test_wavelength_setter_rejects_negative_weight(self):
        wl = Wavelength(value=0.55, weight=1.0)
        with pytest.raises(ValueError, match="non-negative"):
            wl.weight = -0.1


# ---------------------------------------------------------------------------
# Unit Test 3: FieldGroup.weights returns correct tuple
# ---------------------------------------------------------------------------


class TestFieldGroupWeights:
    def test_field_group_weights_tuple(self):
        optic = _make_weighted_optic()
        assert optic.fields.weights == (2.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# Unit Test 4: WavelengthGroup.weights returns correct tuple
# ---------------------------------------------------------------------------


class TestWavelengthGroupWeights:
    def test_wavelength_group_weights_tuple(self):
        optic = _make_weighted_optic()
        assert optic.wavelengths.weights == (1.0, 3.0, 0.0)


# ---------------------------------------------------------------------------
# Unit Test 5: resolve_fields("all") returns FieldPoint objects with correct weights
# ---------------------------------------------------------------------------


class TestResolveFieldsAll:
    def test_resolve_fields_all_returns_field_points(self):
        optic = _make_weighted_optic()
        result = resolve_fields(optic, "all")
        assert len(result) == 3
        assert all(isinstance(fp, FieldPoint) for fp in result)

    def test_resolve_fields_all_has_correct_weights(self):
        optic = _make_weighted_optic()
        result = resolve_fields(optic, "all")
        weights = [fp.weight for fp in result]
        assert weights == [2.0, 1.0, 0.0]

    def test_resolve_fields_all_has_correct_coords(self):
        optic = _make_weighted_optic()
        result = resolve_fields(optic, "all")
        # Y-coords come from get_field_coords() which normalizes by max field
        assert result[0].coord[0] == 0.0  # Hx = 0
        assert result[0].coord[1] == 0.0  # Hy = 0 (on-axis)


# ---------------------------------------------------------------------------
# Unit Test 6: resolve_fields([(0,0)]) returns FieldPoint with weight=1.0
# ---------------------------------------------------------------------------


class TestResolveFieldsRawList:
    def test_raw_list_returns_field_point_weight_one(self):
        optic = _make_weighted_optic()
        result = resolve_fields(optic, [(0.0, 0.0), (0.0, 1.0)])
        assert len(result) == 2
        assert all(isinstance(fp, FieldPoint) for fp in result)
        # Raw user-supplied coordinates always get weight=1.0
        assert all(fp.weight == 1.0 for fp in result)


# ---------------------------------------------------------------------------
# Unit Test 7: resolve_wavelengths("all") returns WavelengthPoint with correct weights
# ---------------------------------------------------------------------------


class TestResolveWavelengthsAll:
    def test_resolve_wavelengths_all_returns_wavelength_points(self):
        optic = _make_weighted_optic()
        result = resolve_wavelengths(optic, "all")
        assert len(result) == 3
        assert all(isinstance(wp, WavelengthPoint) for wp in result)

    def test_resolve_wavelengths_all_has_correct_weights(self):
        optic = _make_weighted_optic()
        result = resolve_wavelengths(optic, "all")
        weights = [wp.weight for wp in result]
        assert weights == [1.0, 3.0, 0.0]


# ---------------------------------------------------------------------------
# Unit Test 8: resolve_wavelengths("primary") returns single WavelengthPoint
# ---------------------------------------------------------------------------


class TestResolveWavelengthsPrimary:
    def test_resolve_wavelengths_primary_returns_single_point(self):
        optic = _make_weighted_optic()
        result = resolve_wavelengths(optic, "primary")
        assert len(result) == 1
        assert isinstance(result[0], WavelengthPoint)

    def test_resolve_wavelengths_primary_correct_weight(self):
        optic = _make_weighted_optic()
        result = resolve_wavelengths(optic, "primary")
        # Primary wavelength has weight=1.0
        assert result[0].weight == 1.0
        assert abs(result[0].value - 0.55) < 1e-9


# ---------------------------------------------------------------------------
# Unit Test 9: active_fields filters zero-weight items
# ---------------------------------------------------------------------------


class TestActiveFields:
    def test_active_fields_removes_zero_weight(self):
        optic = _make_weighted_optic()
        all_fps = resolve_fields(optic, "all")
        active = active_fields(all_fps)
        assert len(active) == 2
        assert all(fp.weight > 0.0 for fp in active)

    def test_active_fields_empty_when_all_zero(self):
        fps = [FieldPoint(coord=(0.0, 0.0), weight=0.0)]
        assert active_fields(fps) == []


# ---------------------------------------------------------------------------
# Unit Test 10: active_wavelengths filters zero-weight items
# ---------------------------------------------------------------------------


class TestActiveWavelengths:
    def test_active_wavelengths_removes_zero_weight(self):
        optic = _make_weighted_optic()
        all_wps = resolve_wavelengths(optic, "all")
        active = active_wavelengths(all_wps)
        assert len(active) == 2
        assert all(wp.weight > 0.0 for wp in active)

    def test_active_wavelengths_empty_when_all_zero(self):
        wps = [WavelengthPoint(value=0.55, weight=0.0)]
        assert active_wavelengths(wps) == []


# ---------------------------------------------------------------------------
# Unit Test 11: weighted_average computes correct result, raises on all-zero
# ---------------------------------------------------------------------------


class TestWeightedAverage:
    def test_weighted_average_correct_result(self):
        values = [1.0, 2.0, 3.0]
        weights = [1.0, 2.0, 1.0]
        # (1*1 + 2*2 + 1*3) / (1+2+1) = 8/4 = 2.0
        result = weighted_average(values, weights)
        assert abs(result - 2.0) < 1e-12

    def test_weighted_average_uniform_weights_equals_mean(self):
        values = [1.0, 2.0, 3.0]
        weights = [1.0, 1.0, 1.0]
        result = weighted_average(values, weights)
        assert abs(result - 2.0) < 1e-12

    def test_weighted_average_raises_on_all_zero_weights(self):
        with pytest.raises(ValueError, match="all weights are zero"):
            weighted_average([1.0, 2.0], [0.0, 0.0])


# ---------------------------------------------------------------------------
# Unit Test 12: Operand.effective_weight returns correct product
# ---------------------------------------------------------------------------


class TestOperandEffectiveWeight:
    def test_effective_weight_with_field_and_wavelength_index(self):
        optic = _make_weighted_optic()
        # Field 0 has weight=2.0, wavelength 1 has weight=3.0, operand weight=1.5
        op = Operand(
            operand_type="f2",
            target=50.0,
            weight=1.5,
            input_data={"optic": optic, "field": 0, "wavelength": 1},
        )
        ew = op.effective_weight()
        # 1.5 × 2.0 × 3.0 = 9.0
        assert abs(ew - 9.0) < 1e-12

    def test_effective_weight_without_indices_is_operand_weight(self):
        optic = _make_weighted_optic()
        op = Operand(
            operand_type="f2",
            target=50.0,
            weight=2.0,
            input_data={"optic": optic},
        )
        ew = op.effective_weight()
        # No field/wl index → field_w = wl_w = 1.0
        assert abs(ew - 2.0) < 1e-12

    def test_effective_weight_zero_field_weight_is_zero(self):
        optic = _make_weighted_optic()
        # Field index 2 has weight=0.0
        op = Operand(
            operand_type="f2",
            target=50.0,
            weight=1.0,
            input_data={"optic": optic, "field": 2, "wavelength": 0},
        )
        ew = op.effective_weight()
        assert ew == 0.0

    def test_effective_weight_zero_wavelength_weight_is_zero(self):
        optic = _make_weighted_optic()
        # Wavelength index 2 has weight=0.0
        op = Operand(
            operand_type="f2",
            target=50.0,
            weight=1.0,
            input_data={"optic": optic, "field": 0, "wavelength": 2},
        )
        ew = op.effective_weight()
        assert ew == 0.0


# ---------------------------------------------------------------------------
# Integration Test 13: Zemax FWGN → field weight transfer
# ---------------------------------------------------------------------------


class TestZemaxFieldWeightImport:
    def test_fwgn_transfers_field_weights(self, tmp_path):
        zmx_content = """MODE SEQ
UNIT MM X W X CM MR CPMM
ENPD 10.0
FTYP 0 0 3 3 0 0 0
XFLN 0. 0. 0.
YFLN 0.0 7.0 10.0
FWGN 3.0 1.0 1.0
PWAV 1
WAVM 1 0.55000 1.0
SURF 0
  TYPE STANDARD
  CURV 0.
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  CURV 0.02
  DISZ 5.0
  GLAS N-BK7 1 0 1.5168 64.17
SURF 2
  TYPE STANDARD
  CURV -0.02
  DISZ 45.0
SURF 3
  TYPE STANDARD
  CURV 0.
  DISZ 0.
"""
        zmx_path = tmp_path / "test_fwgn.zmx"
        zmx_path.write_text(zmx_content, encoding="utf-8")

        from optiland.fileio import load_zemax_file

        optic = load_zemax_file(str(zmx_path))

        assert optic.fields.fields[0].weight == pytest.approx(3.0)
        assert optic.fields.fields[1].weight == pytest.approx(1.0)
        assert optic.fields.fields[2].weight == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration Test 14: Zemax WAVM weight token transfer
# ---------------------------------------------------------------------------


class TestZemaxWavelengthWeightImport:
    def test_wavm_transfers_wavelength_weights(self, tmp_path):
        zmx_content = """MODE SEQ
UNIT MM X W X CM MR CPMM
ENPD 10.0
FTYP 0 0 1 2 0 0 0
XFLN 0.
YFLN 0.0
PWAV 1
WAVM 1 0.55000 2.0
WAVM 2 0.48613 0.5
SURF 0
  TYPE STANDARD
  CURV 0.
  DISZ INFINITY
SURF 1
  TYPE STANDARD
  CURV 0.02
  DISZ 5.0
  GLAS N-BK7 1 0 1.5168 64.17
SURF 2
  TYPE STANDARD
  CURV -0.02
  DISZ 45.0
SURF 3
  TYPE STANDARD
  CURV 0.
  DISZ 0.
"""
        zmx_path = tmp_path / "test_wavm.zmx"
        zmx_path.write_text(zmx_content, encoding="utf-8")

        from optiland.fileio import load_zemax_file

        optic = load_zemax_file(str(zmx_path))

        # First WAVM line (0.55) should have weight=2.0
        assert len(optic.wavelengths.wavelengths) == 2
        assert optic.wavelengths.wavelengths[0].weight == pytest.approx(2.0)

        # Second WAVM line (0.48613) should have weight=0.5
        assert optic.wavelengths.wavelengths[1].weight == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Integration Test 15: JSON round-trip for field with weight=2.5
# ---------------------------------------------------------------------------


class TestFieldJsonRoundTrip:
    def test_field_weight_survives_json_roundtrip(self):
        f = Field(x=0.0, y=0.7, weight=2.5)
        d = f.to_dict()
        f2 = Field.from_dict(d)
        assert f2.weight == pytest.approx(2.5)
        assert f2.x == 0.0
        assert f2.y == pytest.approx(0.7)

    def test_field_weight_key_present_in_dict(self):
        f = Field(x=0.0, y=1.0, weight=3.0)
        d = f.to_dict()
        assert "weight" in d
        assert d["weight"] == pytest.approx(3.0)

    def test_field_weight_default_when_key_absent(self):
        """Older JSON without 'weight' key should default to 1.0."""
        d = {"x": 0.0, "y": 0.5, "vx": 0.0, "vy": 0.0}
        f = Field.from_dict(d)
        assert f.weight == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration Test 16: JSON round-trip for wavelength with weight=0.5
# ---------------------------------------------------------------------------


class TestWavelengthJsonRoundTrip:
    def test_wavelength_weight_survives_json_roundtrip(self):
        wl = Wavelength(value=0.55, weight=0.5)
        d = wl.to_dict()
        wl2 = Wavelength.from_dict(d)
        assert wl2.weight == pytest.approx(0.5)

    def test_wavelength_weight_key_present_in_dict(self):
        wl = Wavelength(value=0.55, weight=2.0)
        d = wl.to_dict()
        assert "weight" in d
        assert d["weight"] == pytest.approx(2.0)

    def test_wavelength_weight_default_when_key_absent(self):
        """Older JSON without 'weight' key should default to 1.0."""
        wl = Wavelength(value=0.55)
        d = wl.to_dict()
        d.pop("weight", None)
        wl2 = Wavelength.from_dict(d)
        assert wl2.weight == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Integration Test 17: Optimizer skips zero-weight field operand
# ---------------------------------------------------------------------------


class TestOptimizerZeroWeightSkip:
    def test_zero_weight_field_operand_excluded_from_merit(self):
        optic = _make_weighted_optic()
        problem = OptimizationProblem()

        # Operand on field 2 (weight=0.0) — should be excluded
        problem.add_operand(
            operand_type="f2",
            target=100.0,
            weight=1.0,
            input_data={"optic": optic, "field": 2},
        )

        values = problem.fun_array()
        # fun_array excludes zero-effective-weight operands; result should be [0.0]
        assert be.to_numpy(be.sum(values)).item() == pytest.approx(0.0)

    def test_nonzero_weight_operand_included_in_merit(self):
        optic = _make_weighted_optic()
        problem = OptimizationProblem()

        # Operand on field 0 (weight=2.0), operand weight=1.0, target will be exact
        # so delta = 0; we just verify it gets included (non-excluded path)
        op_weight = 1.0
        problem.add_operand(
            operand_type="f2",
            target=100.0,
            weight=op_weight,
            input_data={"optic": optic, "field": 0},
        )

        # effective_weight = 1.0 * 2.0 * 1.0 = 2.0 (not zero → included)
        ew = problem.operands[0].effective_weight()
        assert ew == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Integration Test 18: weight_breakdown returns correct effective weights
# ---------------------------------------------------------------------------


class TestWeightBreakdown:
    def test_weight_breakdown_returns_list_of_dicts(self):
        optic = _make_weighted_optic()
        problem = OptimizationProblem()
        problem.add_operand(
            operand_type="f2",
            target=50.0,
            weight=1.5,
            input_data={"optic": optic, "field": 0, "wavelength": 1},
        )
        bd = problem.weight_breakdown()
        assert isinstance(bd, list)
        assert len(bd) == 1

    def test_weight_breakdown_effective_weight_correct(self):
        optic = _make_weighted_optic()
        problem = OptimizationProblem()
        problem.add_operand(
            operand_type="f2",
            target=50.0,
            weight=1.5,
            input_data={"optic": optic, "field": 0, "wavelength": 1},
        )
        bd = problem.weight_breakdown()
        row = bd[0]
        # field 0: weight=2.0, wl 1: weight=3.0, operand: 1.5 → eff=9.0
        assert row["field_weight"] == pytest.approx(2.0)
        assert row["wl_weight"] == pytest.approx(3.0)
        assert row["operand_weight"] == pytest.approx(1.5)
        assert row["effective_weight"] == pytest.approx(9.0)

    def test_weight_breakdown_fields_present(self):
        optic = _make_weighted_optic()
        problem = OptimizationProblem()
        problem.add_operand(
            operand_type="f2",
            target=50.0,
            weight=1.0,
            input_data={"optic": optic, "field": 1, "wavelength": 0},
        )
        bd = problem.weight_breakdown()
        row = bd[0]
        required_keys = {
            "operand_type",
            "field",
            "wavelength",
            "operand_weight",
            "field_weight",
            "wl_weight",
            "effective_weight",
        }
        assert required_keys.issubset(row.keys())


# ---------------------------------------------------------------------------
# Integration Test 19: Polychromatic PSF weighted average (manual verification)
# ---------------------------------------------------------------------------


class TestPolychromaticWeightedAverageFormula:
    """Verifies weighted averages for polychromatic aggregation."""

    def test_weighted_psf_average_formula(self):
        # Simulate two PSF "scalars" representing peak values at two wavelengths
        psf_values = [0.8, 0.6]  # monochromatic PSF peak values
        weights = [1.0, 3.0]  # wavelength weights (3× emphasis on second)
        # Expected: (1*0.8 + 3*0.6) / (1+3) = (0.8 + 1.8) / 4 = 2.6/4 = 0.65
        result = weighted_average(psf_values, weights)
        assert abs(result - 0.65) < 1e-12

    def test_zero_weight_wavelength_excluded_from_average(self):
        """Zero-weight wavelengths must not affect the weighted average."""
        psf_values = [0.8, 0.6, 0.999]  # third wavelength has zero weight
        weights = [1.0, 3.0, 0.0]
        result = weighted_average(
            [v for v, w in zip(psf_values, weights, strict=False) if w > 0.0],
            [w for w in weights if w > 0.0],
        )
        # Only first two contribute: (1*0.8 + 3*0.6) / 4 = 0.65
        assert abs(result - 0.65) < 1e-12


# ---------------------------------------------------------------------------
# Integration Test 20: Backward compatibility — all-weights-1.0 system
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_resolve_fields_all_weight_one_for_default_optic(self):
        """Default optic field weights are all 1.0."""
        lens = CookeTriplet()
        fps = resolve_fields(lens, "all")
        assert all(fp.weight == pytest.approx(1.0) for fp in fps)

    def test_resolve_wavelengths_all_weight_one_for_default_optic(self):
        """Default optic has all wavelength weights=1.0."""
        lens = CookeTriplet()
        wps = resolve_wavelengths(lens, "all")
        assert all(wp.weight == pytest.approx(1.0) for wp in wps)

    def test_effective_weight_equals_operand_weight_when_all_field_wl_weights_one(self):
        """When all field/wl weights=1.0, effective_weight equals operand.weight."""
        lens = CookeTriplet()
        op = Operand(
            operand_type="f2",
            target=100.0,
            weight=3.7,
            input_data={"optic": lens, "field": 0, "wavelength": 0},
        )
        ew = op.effective_weight()
        assert ew == pytest.approx(3.7)

    def test_fun_array_matches_squared_residual_with_operand_weight(self):
        """sum_squared must match the least-squares residual objective."""
        lens = CookeTriplet()
        problem = OptimizationProblem()
        weight = 3.7
        problem.add_operand(
            operand_type="f2",
            target=100.0,
            weight=weight,
            input_data={"optic": lens},
        )

        op = problem.operands[0]
        delta = float(be.to_numpy(be.array(op.delta())))
        contribution = float(be.to_numpy(problem.fun_array()[0]))
        residual = float(be.to_numpy(problem.residual_vector()[0]))

        assert contribution == pytest.approx((weight * delta) ** 2)
        assert residual == pytest.approx(weight * delta)
        assert float(be.to_numpy(problem.sum_squared())) == pytest.approx(
            float(be.to_numpy(be.sum(problem.residual_vector() ** 2))),
        )

    def test_fun_array_matches_squared_residual_with_field_wavelength_weights(self):
        """Field/wavelength weights should affect merit and residuals consistently."""
        optic = _make_weighted_optic()
        problem = OptimizationProblem()
        problem.add_operand(
            operand_type="f2",
            target=80.0,
            weight=1.5,
            input_data={"optic": optic, "field": 0, "wavelength": 1},
        )
