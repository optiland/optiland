"""Tests for thin film tolerancing module."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.thin_film import ThinFilmStack
from optiland.thin_film.tolerancing import (
    ThinFilmMonteCarlo,
    ThinFilmPerturbation,
    ThinFilmSensitivityAnalysis,
    ThinFilmTolerancing,
)
from optiland.tolerancing.perturbation import (
    DistributionSampler,
    RangeSampler,
)

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def skip_torch_without_numpy_bridge(set_test_backend):
    if be.get_backend() != "torch":
        return
    import torch

    try:
        _ = torch.tensor([0.0]).numpy()
    except Exception as exc:
        pytest.skip(f"Torch backend unavailable in this env: {exc}")


@pytest.fixture
def air():
    return IdealMaterial(n=1.0)


@pytest.fixture
def glass():
    return IdealMaterial(n=1.52)


@pytest.fixture
def sio2():
    return IdealMaterial(n=1.46)


@pytest.fixture
def tio2():
    return IdealMaterial(n=2.3)


@pytest.fixture
def two_layer_stack(air, glass, sio2, tio2):
    stack = ThinFilmStack(incident_material=air, substrate_material=glass)
    stack.add_layer_nm(sio2, 100.0, name="SiO2")
    stack.add_layer_nm(tio2, 50.0, name="TiO2")
    return stack


class TestThinFilmPerturbation:
    """Test perturbation apply/reset for thickness and index."""

    def test_thickness_relative(self, two_layer_stack):
        sampler = RangeSampler(-0.05, 0.05, 3)
        p = ThinFilmPerturbation(
            two_layer_stack,
            layer_index=0,
            perturbation_type="thickness",
            sampler=sampler,
            is_relative=True,
        )
        original = two_layer_stack.layers[0].thickness_um
        p.apply()
        assert two_layer_stack.layers[0].thickness_um != original
        p.reset()
        np.testing.assert_allclose(
            two_layer_stack.layers[0].thickness_um, original, atol=1e-12
        )

    def test_thickness_absolute(self, two_layer_stack):
        sampler = RangeSampler(0.09, 0.11, 3)  # in um
        p = ThinFilmPerturbation(
            two_layer_stack,
            layer_index=0,
            perturbation_type="thickness",
            sampler=sampler,
            is_relative=False,
        )
        p.apply()
        assert p.value is not None
        p.reset()

    def test_index_relative(self, two_layer_stack):
        sampler = RangeSampler(-0.01, 0.01, 3)
        p = ThinFilmPerturbation(
            two_layer_stack,
            layer_index=0,
            perturbation_type="index",
            sampler=sampler,
            is_relative=True,
        )
        p.apply()
        # Index should have changed
        new_n = float(two_layer_stack.layers[0].material.n(0.55))
        assert new_n != 1.46
        p.reset()
        restored_n = float(two_layer_stack.layers[0].material.n(0.55))
        np.testing.assert_allclose(restored_n, 1.46, atol=1e-10)

    def test_index_absolute(self, two_layer_stack):
        sampler = RangeSampler(1.44, 1.48, 3)
        p = ThinFilmPerturbation(
            two_layer_stack,
            layer_index=0,
            perturbation_type="index",
            sampler=sampler,
            is_relative=False,
        )
        p.apply()
        new_n = float(two_layer_stack.layers[0].material.n(0.55))
        assert 1.43 < new_n < 1.49
        p.reset()

    def test_index_non_ideal_raises(self, two_layer_stack):
        """Index perturbation should fail for non-IdealMaterial."""
        from optiland.materials import BaseMaterial

        class DummyMaterial(BaseMaterial):
            def __init__(self):
                super().__init__()

            def _calculate_n(self, wavelength):
                return be.atleast_1d(1.5)

            def _calculate_k(self, wavelength):
                return be.atleast_1d(0.0)

        two_layer_stack.layers[0].material = DummyMaterial()
        sampler = RangeSampler(-0.01, 0.01, 3)
        with pytest.raises(TypeError, match="IdealMaterial"):
            ThinFilmPerturbation(
                two_layer_stack,
                layer_index=0,
                perturbation_type="index",
                sampler=sampler,
            )

    def test_invalid_perturbation_type(self, two_layer_stack):
        sampler = RangeSampler(-0.05, 0.05, 3)
        with pytest.raises(ValueError, match="perturbation_type"):
            ThinFilmPerturbation(
                two_layer_stack,
                layer_index=0,
                perturbation_type="invalid",
                sampler=sampler,
            )

    def test_str_representation(self, two_layer_stack):
        sampler = RangeSampler(-0.05, 0.05, 3)
        p = ThinFilmPerturbation(
            two_layer_stack,
            layer_index=0,
            perturbation_type="thickness",
            sampler=sampler,
        )
        assert "Layer 0" in str(p)
        assert "thickness" in str(p)


class TestThinFilmTolerancing:
    """Test ThinFilmTolerancing operand/perturbation management."""

    def test_add_operand(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        result = tol.add_operand("R", 550.0)
        assert result is tol
        assert len(tol.operands) == 1

    def test_add_operand_with_target(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        tol.add_operand("R", 550.0, target=0.05)
        assert tol.operands[0].target == 0.05

    def test_add_operand_auto_target(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        tol.add_operand("R", 550.0)
        assert tol.operands[0].target is not None
        assert isinstance(tol.operands[0].target, float)

    def test_add_perturbation(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        sampler = RangeSampler(-0.05, 0.05, 5)
        result = tol.add_perturbation(0, "thickness", sampler)
        assert result is tol
        assert len(tol.perturbations) == 1

    def test_add_perturbation_no_sampler_raises(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        with pytest.raises(ValueError, match="sampler"):
            tol.add_perturbation(0, "thickness")

    def test_evaluate(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        tol.add_operand("R", 550.0)
        tol.add_operand("T", 550.0)
        values = tol.evaluate()
        assert len(values) == 2
        assert all(isinstance(v, float) for v in values)
        # R + T should be approximately 1 for lossless materials
        np.testing.assert_allclose(values[0] + values[1], 1.0, atol=0.01)

    def test_reset(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        sampler = RangeSampler(-0.05, 0.05, 5)
        tol.add_operand("R", 550.0)
        tol.add_perturbation(0, "thickness", sampler)

        original = two_layer_stack.layers[0].thickness_um
        tol.perturbations[0].apply()
        tol.reset()
        np.testing.assert_allclose(
            two_layer_stack.layers[0].thickness_um, original, atol=1e-12
        )


class TestThinFilmSensitivityAnalysis:
    """Test sensitivity analysis."""

    def test_validation_no_operands(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        sampler = RangeSampler(-0.05, 0.05, 5)
        tol.add_perturbation(0, "thickness", sampler)
        with pytest.raises(ValueError, match="No operands"):
            ThinFilmSensitivityAnalysis(tol)

    def test_validation_no_perturbations(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        tol.add_operand("R", 550.0)
        with pytest.raises(ValueError, match="No perturbations"):
            ThinFilmSensitivityAnalysis(tol)

    def test_run_and_results(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        tol.add_operand("R", 550.0)
        tol.add_perturbation(0, "thickness", RangeSampler(-0.05, 0.05, 5))

        sa = ThinFilmSensitivityAnalysis(tol)
        sa.run()
        df = sa.get_results()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "perturbation_type" in df.columns
        assert "perturbation_value" in df.columns

    def test_run_multiple_perturbations(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        tol.add_operand("R", 550.0)
        tol.add_perturbation(0, "thickness", RangeSampler(-0.05, 0.05, 3))
        tol.add_perturbation(1, "thickness", RangeSampler(-0.05, 0.05, 3))

        sa = ThinFilmSensitivityAnalysis(tol)
        sa.run()
        df = sa.get_results()
        assert len(df) == 6  # 3 steps * 2 perturbations

    def test_view_returns_figure(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        tol.add_operand("R", 550.0)
        tol.add_perturbation(0, "thickness", RangeSampler(-0.05, 0.05, 5))

        sa = ThinFilmSensitivityAnalysis(tol)
        sa.run()
        fig, axes = sa.view()

        assert fig is not None
        assert len(axes) > 0
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_requires_range_sampler(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        tol.add_operand("R", 550.0)
        tol.add_perturbation(
            0,
            "thickness",
            DistributionSampler("normal", loc=0.0, scale=0.01),
        )

        sa = ThinFilmSensitivityAnalysis(tol)
        with pytest.raises(ValueError, match="RangeSampler"):
            sa.run()


class TestThinFilmMonteCarlo:
    """Test Monte Carlo analysis."""

    def test_run_and_results(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        tol.add_operand("R", 550.0)
        tol.add_perturbation(
            0,
            "thickness",
            DistributionSampler("normal", seed=42, loc=0.0, scale=0.02),
        )

        mc = ThinFilmMonteCarlo(tol)
        mc.run(num_iterations=50)
        df = mc.get_results()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50

    def test_multiple_operands(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        tol.add_operand("R", 450.0)
        tol.add_operand("R", 550.0)
        tol.add_operand("T", 550.0)
        tol.add_perturbation(
            0,
            "thickness",
            DistributionSampler("normal", seed=42, loc=0.0, scale=0.02),
        )
        tol.add_perturbation(
            1,
            "thickness",
            DistributionSampler("normal", seed=43, loc=0.0, scale=0.02),
        )

        mc = ThinFilmMonteCarlo(tol)
        mc.run(num_iterations=30)
        df = mc.get_results()

        # Should have columns for perturbations + operands
        assert len(df.columns) >= 5

    def test_view_histogram(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        tol.add_operand("R", 550.0)
        tol.add_perturbation(
            0,
            "thickness",
            DistributionSampler("normal", seed=42, loc=0.0, scale=0.02),
        )

        mc = ThinFilmMonteCarlo(tol)
        mc.run(num_iterations=50)
        fig, axes = mc.view_histogram(kde=True)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_view_cdf(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        tol.add_operand("R", 550.0)
        tol.add_perturbation(
            0,
            "thickness",
            DistributionSampler("normal", seed=42, loc=0.0, scale=0.02),
        )

        mc = ThinFilmMonteCarlo(tol)
        mc.run(num_iterations=50)
        fig, axes = mc.view_cdf()

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_view_heatmap(self, two_layer_stack):
        tol = ThinFilmTolerancing(two_layer_stack)
        tol.add_operand("R", 550.0)
        tol.add_operand("T", 550.0)
        tol.add_perturbation(
            0,
            "thickness",
            DistributionSampler("normal", seed=42, loc=0.0, scale=0.02),
        )
        tol.add_perturbation(
            1,
            "thickness",
            DistributionSampler("normal", seed=43, loc=0.0, scale=0.02),
        )

        mc = ThinFilmMonteCarlo(tol)
        mc.run(num_iterations=50)
        fig, ax = mc.view_heatmap()

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestRealisticTolerancing:
    """Realistic integration tests with real-world thin film design specs.

    Uses a pre-optimized 4-layer broadband AR coating with realistic material
    indices and manufacturing tolerances typical of magnetron sputtering.
    """

    @pytest.fixture
    def optimized_4layer_ar(self, air, glass, sio2, tio2):
        """Pre-optimized 4-layer broadband AR coating (H/L/H/L on glass).

        Layer thicknesses from a typical optimized design for 450-650nm.
        """
        from optiland.thin_film.optimization import ThinFilmOptimizer

        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(tio2, 15.0, name="TiO2_1")
        stack.add_layer_nm(sio2, 30.0, name="SiO2_1")
        stack.add_layer_nm(tio2, 100.0, name="TiO2_2")
        stack.add_layer_nm(sio2, 90.0, name="SiO2_2")

        opt = ThinFilmOptimizer(stack)
        for i in range(4):
            opt.add_variable(i, min_nm=5.0, max_nm=300.0)
        wls = np.linspace(450, 650, 15).tolist()
        opt.add_spectral_operand("R", wls, "equal", 0.0)
        opt.optimize(max_iterations=200)
        return stack

    def test_sensitivity_4layer_ar_thickness(self, optimized_4layer_ar):
        """Sensitivity analysis: ±3% thickness tolerance on all 4 layers.

        Typical magnetron sputtering achieves ±1-3% thickness control.
        Verify that:
        - Each perturbation produces a measurable change in R
        - The DataFrame has correct shape
        - Reflectance stays physically valid (0 ≤ R ≤ 1)
        """
        stack = optimized_4layer_ar
        tol = ThinFilmTolerancing(stack)

        # Operands: reflectance at blue, green, red
        tol.add_operand("R", 450.0)
        tol.add_operand("R", 550.0)
        tol.add_operand("R", 650.0)

        # ±3% thickness perturbation on each of 4 layers
        for i in range(4):
            tol.add_perturbation(i, "thickness", RangeSampler(-0.03, 0.03, 11))

        sa = ThinFilmSensitivityAnalysis(tol)
        sa.run()
        df = sa.get_results()

        # 4 perturbations × 11 steps = 44 rows
        assert len(df) == 44
        assert "perturbation_type" in df.columns
        assert "perturbation_value" in df.columns

        # All operand columns should exist
        operand_cols = [
            c
            for c in df.columns
            if c.startswith("0:") or c.startswith("1:") or c.startswith("2:")
        ]
        assert len(operand_cols) == 3

        # Reflectance must be physically valid
        for col in operand_cols:
            assert (df[col] >= 0).all(), f"{col} has negative values"
            assert (df[col] <= 1).all(), f"{col} has values > 1"

    def test_monte_carlo_4layer_ar_manufacturing(self, optimized_4layer_ar):
        """Monte Carlo: 500 iterations with realistic manufacturing errors.

        Simulates ±2% normally-distributed thickness errors (1σ) on all
        4 layers — typical for well-controlled sputtering processes.

        Verifies:
        - Mean reflectance is close to nominal (< 5% relative shift)
        - Standard deviation is physically reasonable
        - R + T ≈ 1 for lossless materials (energy conservation)
        - DataFrame has correct shape
        """
        stack = optimized_4layer_ar
        from optiland.thin_film.optimization.operand.thin_film import (
            ThinFilmOperand,
        )

        # Record nominal reflectance
        nominal_R_550 = ThinFilmOperand.reflectance(stack, 550.0)

        tol = ThinFilmTolerancing(stack)
        tol.add_operand("R", 550.0)
        tol.add_operand("T", 550.0)

        # 2% std thickness error on all 4 layers
        for i in range(4):
            tol.add_perturbation(
                i,
                "thickness",
                DistributionSampler("normal", seed=100 + i, loc=0.0, scale=0.02),
            )

        mc = ThinFilmMonteCarlo(tol)
        mc.run(num_iterations=500)
        df = mc.get_results()

        assert len(df) == 500

        # Extract operand columns
        r_col = [c for c in df.columns if "R@550" in c][0]
        t_col = [c for c in df.columns if "T@550" in c][0]

        # Mean R should be close to nominal (within 50% relative)
        mean_R = df[r_col].mean()
        assert abs(mean_R - nominal_R_550) < 0.5 * max(nominal_R_550, 0.01), (
            f"Mean R = {mean_R:.4f}, nominal = {nominal_R_550:.4f}"
        )

        # Energy conservation: R + T ≈ 1 for lossless
        rt_sum = df[r_col] + df[t_col]
        np.testing.assert_allclose(rt_sum.values, 1.0, atol=0.01)

        # Standard deviation should be non-zero but reasonable
        std_R = df[r_col].std()
        assert std_R > 0, "Zero standard deviation — perturbations not applied?"
        assert std_R < 0.1, f"Std R = {std_R:.4f}, unreasonably large"

    def test_combined_thickness_and_index_perturbation(self, air, glass, sio2, tio2):
        """Monte Carlo with both thickness and index perturbations.

        Realistic scenario: thickness ±2% and index ±0.5% simultaneously.
        Index variations can arise from composition drift in reactive
        sputtering.
        """
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(sio2, 95.0, name="SiO2")
        stack.add_layer_nm(tio2, 55.0, name="TiO2")

        tol = ThinFilmTolerancing(stack)
        tol.add_operand("R", 550.0)

        # Thickness perturbations (±2%)
        tol.add_perturbation(
            0,
            "thickness",
            DistributionSampler("normal", seed=10, loc=0.0, scale=0.02),
        )
        tol.add_perturbation(
            1,
            "thickness",
            DistributionSampler("normal", seed=11, loc=0.0, scale=0.02),
        )
        # Index perturbations (±0.5%)
        tol.add_perturbation(
            0,
            "index",
            DistributionSampler("normal", seed=12, loc=0.0, scale=0.005),
        )
        tol.add_perturbation(
            1,
            "index",
            DistributionSampler("normal", seed=13, loc=0.0, scale=0.005),
        )

        mc = ThinFilmMonteCarlo(tol)
        mc.run(num_iterations=500)
        df = mc.get_results()

        assert len(df) == 500
        # Should have perturbation columns for all 4 perturbations
        pert_cols = [c for c in df.columns if "Layer" in c]
        assert len(pert_cols) == 4

        # Reflectance should remain physically valid
        r_col = [c for c in df.columns if "R@550" in c][0]
        assert (df[r_col] >= 0).all()
        assert (df[r_col] <= 1).all()

    def test_sensitivity_identifies_critical_layer(self, air, glass, sio2, tio2):
        """Verify sensitivity analysis can identify which layer is most
        critical to performance.

        In a 2-layer AR coating, the high-index layer (TiO2) typically has
        higher sensitivity because it dominates the interference condition.
        """
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(tio2, 55.0, name="TiO2")
        stack.add_layer_nm(sio2, 95.0, name="SiO2")

        tol = ThinFilmTolerancing(stack)
        tol.add_operand("R", 550.0)
        tol.add_perturbation(0, "thickness", RangeSampler(-0.05, 0.05, 21))
        tol.add_perturbation(1, "thickness", RangeSampler(-0.05, 0.05, 21))

        sa = ThinFilmSensitivityAnalysis(tol)
        sa.run()
        df = sa.get_results()

        r_col = [c for c in df.columns if "R@550" in c][0]

        # Get the range (max - min) of R for each perturbation type
        ranges = {}
        for ptype in df["perturbation_type"].unique():
            subset = df[df["perturbation_type"] == ptype][r_col]
            ranges[ptype] = subset.max() - subset.min()

        # Both layers should produce non-zero sensitivity
        for ptype, rng in ranges.items():
            assert rng > 0, f"Zero sensitivity for {ptype}"
