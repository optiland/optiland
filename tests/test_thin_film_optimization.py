"""Tests for thin film optimization module.

This module contains tests for the thin film optimization functionality,
including variables, operands, and the optimizer.

Corentin Nannini, 2025
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from types import SimpleNamespace
import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.thin_film import ThinFilmStack
import optiland.thin_film.optimization.optimizer as optimizer_module
from optiland.thin_film.optimization import (
    ThinFilmOptimizer,
    LayerThicknessVariable,
    ThinFilmOperand,
)
from optiland.thin_film.optimization.operand import thin_film_operand_registry


@pytest.fixture(autouse=True)
def skip_torch_without_numpy_bridge(set_test_backend):
    if be.get_backend() != "torch":
        return

    import torch

    try:
        _ = torch.tensor([0.0]).numpy()
    except Exception as exc:
        pytest.skip(f"Torch backend unavailable in this env: {exc}")


@pytest.fixture(autouse=True)
def thin_film_opt_backend_compat():
    """No-op compatibility fixture retained to minimize test churn."""
    yield


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
    return IdealMaterial(n=2.4)


@pytest.fixture
def simple_stack(air, glass, sio2):
    """Simple stack with one SiO2 layer."""
    stack = ThinFilmStack(incident_material=air, substrate_material=glass)
    stack.add_layer_nm(sio2, 100.0, name="SiO2")
    return stack


@pytest.fixture
def multilayer_stack(air, glass, sio2, tio2):
    """Multilayer stack with two layers."""
    stack = ThinFilmStack(incident_material=air, substrate_material=glass)
    stack.add_layer_nm(sio2, 100.0, name="SiO2")
    stack.add_layer_nm(tio2, 80.0, name="TiO2")
    return stack


class TestLayerThicknessVariable:
    """Test LayerThicknessVariable functionality."""

    def test_variable_creation(self, simple_stack):
        """Test creating a layer thickness variable."""
        var = LayerThicknessVariable(simple_stack, layer_index=0)
        assert var.stack is simple_stack
        assert var.layer_index == 0
        assert var.apply_scaling is True

    def test_variable_creation_invalid_index(self, simple_stack):
        """Test creating variable with invalid layer index."""
        with pytest.raises(ValueError, match="layer_index.*out of range"):
            LayerThicknessVariable(simple_stack, layer_index=5)

    def test_get_value(self, simple_stack):
        """Test getting the current thickness value."""
        var = LayerThicknessVariable(simple_stack, layer_index=0, apply_scaling=False)
        # Initial thickness is 0.1 μm (100 nm)
        assert var.get_value() == 0.1

    def test_get_value_with_scaling(self, simple_stack):
        """Test getting value with scaling applied."""
        var = LayerThicknessVariable(simple_stack, layer_index=0, apply_scaling=True)
        expected = var.scale(0.1)
        assert var.get_value() == expected

    def test_update_value(self, simple_stack):
        """Test updating the thickness value."""
        var = LayerThicknessVariable(simple_stack, layer_index=0, apply_scaling=False)
        var.update_value(0.15)  # 150 nm
        assert simple_stack.layers[0].thickness_um == 0.15

    def test_update_value_with_scaling(self, simple_stack):
        """Test updating value with scaling applied."""
        var = LayerThicknessVariable(simple_stack, layer_index=0, apply_scaling=True)
        scaled_value = var.scale(0.15)
        var.update_value(scaled_value)
        assert abs(simple_stack.layers[0].thickness_um - 0.15) < 1e-10

    def test_update_value_negative_corrected(self, simple_stack):
        """Test that negative thickness is corrected to minimum value."""
        var = LayerThicknessVariable(simple_stack, layer_index=0, apply_scaling=False)
        # Try to set negative thickness - should be corrected to minimum
        var.update_value(-0.05)
        # Should be corrected to minimum thickness (0.01 nm = 0.000001 μm)
        assert simple_stack.layers[0].thickness_um == 0.000001

    def test_scaling_functions(self, simple_stack):
        """Test scaling and inverse scaling."""
        var = LayerThicknessVariable(simple_stack, layer_index=0)
        original = 0.1
        scaled = var.scale(original)
        unscaled = var.inverse_scale(scaled)
        assert abs(unscaled - original) < 1e-10

    def test_thickness_nm_property(self, simple_stack):
        """Test thickness_nm property."""
        var = LayerThicknessVariable(simple_stack, layer_index=0)
        assert var.thickness_nm == 100.0


class TestThinFilmOperand:
    """Test ThinFilmOperand functionality."""

    def test_reflectance_single_wavelength(self, simple_stack):
        """Test reflectance computation for single wavelength."""
        R = ThinFilmOperand.reflectance(
            simple_stack, 550.0, aoi_deg=0.0, polarization="u"
        )
        assert isinstance(R, float)
        assert 0 <= R <= 1

    def test_transmittance_single_wavelength(self, simple_stack):
        """Test transmittance computation for single wavelength."""
        T = ThinFilmOperand.transmittance(
            simple_stack, 550.0, aoi_deg=0.0, polarization="u"
        )
        assert isinstance(T, float)
        assert 0 <= T <= 1

    def test_absorptance_single_wavelength(self, simple_stack):
        """Test absorptance computation for single wavelength."""
        A = ThinFilmOperand.absorptance(
            simple_stack, 550.0, aoi_deg=0.0, polarization="u"
        )
        assert isinstance(A, float)
        assert 0 <= A <= 1

    def test_energy_conservation(self, simple_stack):
        """Test that R + T + A = 1."""
        wavelength = 550.0
        R = ThinFilmOperand.reflectance(simple_stack, wavelength)
        T = ThinFilmOperand.transmittance(simple_stack, wavelength)
        A = ThinFilmOperand.absorptance(simple_stack, wavelength)
        assert abs(R + T + A - 1.0) < 1e-10

    def test_multiple_wavelengths(self, simple_stack):
        """Test operands with multiple wavelengths."""
        wavelengths = [500.0, 550.0, 600.0]
        R = ThinFilmOperand.reflectance(simple_stack, wavelengths)
        assert isinstance(R, float)
        assert 0 <= R <= 1

    def test_weighted_operands(self, simple_stack):
        """Test weighted operand functions."""
        wavelengths = [500.0, 550.0, 600.0]
        weights = [1.0, 2.0, 1.0]

        R_weighted = ThinFilmOperand.reflectance_weighted(
            simple_stack, wavelengths, weights
        )
        assert isinstance(R_weighted, float)
        assert 0 <= R_weighted <= 1

    def test_spectral_phase_and_delay_helpers(self, simple_stack):
        """Test phase, GD and GDD helper operands from the complex coefficients."""
        wavelength_nm = 550.0
        phase_r = ThinFilmOperand.reflection_phase(simple_stack, wavelength_nm)
        phase_t = ThinFilmOperand.transmission_phase(simple_stack, wavelength_nm)
        gd_r = ThinFilmOperand.reflection_gd(simple_stack, wavelength_nm)
        gd_t = ThinFilmOperand.transmission_gd(simple_stack, wavelength_nm)
        gdd_r = ThinFilmOperand.reflection_gdd(simple_stack, wavelength_nm)
        gdd_t = ThinFilmOperand.transmission_gdd(simple_stack, wavelength_nm)

        assert isinstance(phase_r, float)
        assert isinstance(phase_t, float)
        assert isinstance(gd_r, float)
        assert isinstance(gd_t, float)
        assert isinstance(gdd_r, float)
        assert isinstance(gdd_t, float)

    def test_internal_field_helpers(self, simple_stack):
        """Test field phase and amplitude helpers for a selected layer."""
        phase = ThinFilmOperand.field_phase(
            simple_stack,
            layer_index=0,
            wavelength_nm=550.0,
            aoi_deg=0.0,
            polarization="s",
            position_fraction=0.5,
        )
        amplitude = ThinFilmOperand.field_amplitude(
            simple_stack,
            layer_index=0,
            wavelength_nm=550.0,
            aoi_deg=0.0,
            polarization="s",
            position_fraction=0.5,
        )

        assert isinstance(phase, float)
        assert isinstance(amplitude, float)
        assert amplitude >= 0


class TestThinFilmOptimizer:
    """Test ThinFilmOptimizer functionality."""

    def test_optimizer_creation(self, simple_stack):
        """Test creating a thin film optimizer."""
        optimizer = ThinFilmOptimizer(simple_stack)
        assert optimizer.stack is simple_stack
        assert len(optimizer.variables) == 0
        assert len(optimizer.targets) == 0

    def test_add_variable(self, simple_stack):
        """Test adding a variable."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_variable(layer_index=0, min_nm=50, max_nm=200)

        assert len(optimizer.variables) == 1
        assert optimizer.variables[0].layer_index == 0

    def test_add_variable_invalid_index(self, simple_stack):
        """Test adding variable with invalid layer index."""
        optimizer = ThinFilmOptimizer(simple_stack)
        with pytest.raises(ValueError, match="out of range"):
            optimizer.add_variable(layer_index=5)

    def test_add_operand(self, simple_stack):
        """Test adding an optimization operand."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_operand(
            property="R",
            wavelength_nm=550.0,
            target_type="below",
            value=0.05,
            weight=1.0,
        )

        assert len(optimizer.targets) == 1
        assert optimizer.targets[0].property == "R"

    def test_add_spectral_operand(self, simple_stack):
        """Test adding a spectral operand."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_spectral_operand(
            property="T",
            wavelengths_nm=[500.0, 550.0, 600.0],
            target_type="over",
            value=0.9,
        )

        assert len(optimizer.targets) == 1

    def test_add_phase_and_delay_operands(self, simple_stack):
        """Test that the optimizer accepts newly registered thin-film metrics."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_operand(
            property="reflection_phase",
            target=0.0,
            input_data={
                "wavelength_nm": 550.0,
                "aoi_deg": 0.0,
                "polarization": "s",
            },
        )
        optimizer.add_operand(
            property="reflection_gd",
            target=0.0,
            input_data={
                "wavelength_nm": 550.0,
                "aoi_deg": 0.0,
                "polarization": "s",
            },
        )
        optimizer.add_operand(
            property="field_phase",
            target=0.0,
            input_data={
                "layer_index": 0,
                "wavelength_nm": 550.0,
                "aoi_deg": 0.0,
                "polarization": "s",
                "position_fraction": 0.5,
            },
        )

        assert len(optimizer.targets) == 3
        performance = optimizer.get_current_performance()
        assert "target_0" in performance
        assert "target_1" in performance
        assert "target_2" in performance
        assert performance["target_0"]["property"] == "reflection_phase"
        assert performance["target_1"]["property"] == "reflection_gd"
        assert performance["target_2"]["property"] == "field_phase"

    def test_registered_operand_names(self):
        """Test that the new optimization operand names are registered."""
        for name in [
            "reflection_phase",
            "transmission_phase",
            "reflection_gd",
            "transmission_gd",
            "reflection_gdd",
            "transmission_gdd",
            "field_phase",
            "field_amplitude",
        ]:
            assert name in thin_film_operand_registry

    def test_method_chaining(self, multilayer_stack):
        """Test that methods can be chained."""
        optimizer = ThinFilmOptimizer(multilayer_stack)
        result = (
            optimizer.add_variable(layer_index=0, min_nm=50, max_nm=150)
            .add_variable(layer_index=1, min_nm=60, max_nm=120)
            .add_operand("R", wavelength_nm=550, target_type="below", value=0.05)
        )

        assert result is optimizer
        assert len(optimizer.variables) == 2
        assert len(optimizer.targets) == 1

    def test_optimize_no_variables_fails(self, simple_stack):
        """Test that optimization fails without variables."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_operand("R", wavelength_nm=550, target_type="below", value=0.05)

        with pytest.raises(ValueError, match="No variables"):
            optimizer.optimize(maxiter=5)

    def test_optimize_no_operands_fails(self, simple_stack):
        """Test that optimization fails without operands."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_variable(layer_index=0)

        with pytest.raises(ValueError, match="No operands"):
            optimizer.optimize(maxiter=5)

    def test_basic_optimization(self, simple_stack):
        """Test a basic optimization run."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_variable(layer_index=0, min_nm=50, max_nm=200)
        optimizer.add_operand("R", wavelength_nm=550, target_type="below", value=0.01)

        # Run a short optimization
        result = optimizer.optimize(max_iterations=5, verbose=False)

        # Check that optimization ran
        assert isinstance(result, dict)
        assert "success" in result
        assert "final_merit" in result

    def test_reset(self, simple_stack):
        """Test resetting to initial state."""
        optimizer = ThinFilmOptimizer(simple_stack)
        initial_thickness = simple_stack.layers[0].thickness_um

        # Modify thickness
        simple_stack.layers[0].update_thickness(0.2)
        assert simple_stack.layers[0].thickness_um != initial_thickness

        # Reset
        optimizer.reset()
        assert simple_stack.layers[0].thickness_um == initial_thickness

    def test_info_display(self, multilayer_stack, capsys):
        """Test info display method."""
        optimizer = ThinFilmOptimizer(multilayer_stack)
        optimizer.add_variable(layer_index=0)
        optimizer.add_operand("R", wavelength_nm=550, target_type="below", value=0.05)

        optimizer.info()
        captured = capsys.readouterr()

        assert (
            "ThinFilm Optimizer Information" in captured.out
            or "ThinFilm Optimizer" in captured.out
        )
        assert "2" in captured.out or "variables" in captured.out

    def test_repr(self, multilayer_stack):
        """Test string representation."""
        optimizer = ThinFilmOptimizer(multilayer_stack)
        optimizer.add_variable(layer_index=0)
        optimizer.add_operand("R", wavelength_nm=550, target_type="below", value=0.05)

        repr_str = repr(optimizer)
        assert "ThinFilmOptimizer" in repr_str or "thin_film" in repr_str.lower()


class TestOptimizationIntegration:
    """Integration tests for the complete optimization workflow."""

    def test_anti_reflection_coating_optimization(self, air, glass, sio2):
        """Test optimization of a simple AR coating."""
        # Create a simple AR coating (single layer)
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(sio2, 100.0, name="AR coating")

        # Set up optimization
        optimizer = ThinFilmOptimizer(stack)
        optimizer.add_variable(layer_index=0, min_nm=50, max_nm=200)
        optimizer.add_operand("R", wavelength_nm=550, target_type="below", value=0.01)

        # Run optimization
        result = optimizer.optimize(max_iterations=10, verbose=False)

        # Check that optimization ran
        assert isinstance(result, dict)
        assert "success" in result
        assert "final_merit" in result

        # Reset should work
        optimizer.reset()

    def test_bandpass_filter_optimization(self, air, glass, sio2, tio2):
        """Test optimization of a simple bandpass filter."""
        # Create alternating layer structure
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(sio2, 100.0)
        stack.add_layer_nm(tio2, 80.0)
        stack.add_layer_nm(sio2, 100.0)

        # Set up optimization for transmission at 550 nm, reflection elsewhere
        optimizer = ThinFilmOptimizer(stack)
        for i in range(3):
            optimizer.add_variable(layer_index=i, min_nm=50, max_nm=200)

        # Target high transmission at 550 nm
        optimizer.add_operand("T", wavelength_nm=550, target_type="over", value=0.8)

        # Target low transmission at other wavelengths
        optimizer.add_spectral_operand(
            "T", wavelengths_nm=[450, 650], target_type="below", value=0.1, weight=0.5
        )

        # Run optimization
        result = optimizer.optimize(max_iterations=5, verbose=False)
        assert isinstance(result, dict)
        assert "success" in result

    def test_optimization_with_report(self, simple_stack):
        """Test optimization returns expected result structure."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_variable(layer_index=0, min_nm=50, max_nm=200)
        optimizer.add_operand("R", wavelength_nm=550, target_type="below", value=0.01)

        # Run optimization
        result = optimizer.optimize(max_iterations=5, verbose=False)

        # Check that result contains expected fields
        assert isinstance(result, dict)
        assert "success" in result
        assert "initial_merit" in result
        assert "final_merit" in result
        assert "improvement" in result
        assert "iterations" in result
        assert "function_evaluations" in result
        assert "thickness_changes" in result


class TestAdditionalOperands:
    """Test additional operand functionality and weighted methods."""

    def test_reflectance_weighted(self, simple_stack):
        """Test weighted reflectance computation."""
        wavelengths = [500.0, 550.0, 600.0]
        weights = [1.0, 2.0, 1.0]

        R_weighted = ThinFilmOperand.reflectance_weighted(
            simple_stack, wavelengths, weights
        )
        assert isinstance(R_weighted, float)
        assert 0 <= R_weighted <= 1

    def test_transmittance_weighted(self, simple_stack):
        """Test weighted transmittance computation."""
        wavelengths = [500.0, 550.0, 600.0]
        weights = [1.0, 2.0, 1.0]

        T_weighted = ThinFilmOperand.transmittance_weighted(
            simple_stack, wavelengths, weights
        )
        assert isinstance(T_weighted, float)
        assert 0 <= T_weighted <= 1

    def test_absorptance_weighted(self, simple_stack):
        """Test weighted absorptance computation."""
        wavelengths = [500.0, 550.0, 600.0]
        weights = [1.0, 2.0, 1.0]

        A_weighted = ThinFilmOperand.absorptance_weighted(
            simple_stack, wavelengths, weights
        )
        assert isinstance(A_weighted, float)
        # Allow for small numerical errors
        assert -1e-10 <= A_weighted <= 1

    def test_weighted_energy_conservation(self, simple_stack):
        """Test that weighted R + T + A ≈ 1."""
        wavelengths = [500.0, 550.0, 600.0]
        weights = [1.0, 2.0, 1.0]

        R_w = ThinFilmOperand.reflectance_weighted(simple_stack, wavelengths, weights)
        T_w = ThinFilmOperand.transmittance_weighted(simple_stack, wavelengths, weights)
        A_w = ThinFilmOperand.absorptance_weighted(simple_stack, wavelengths, weights)

        # Should sum to approximately 1
        assert abs(R_w + T_w + A_w - 1.0) < 0.1

    def test_reflectance_with_different_polarizations(self, simple_stack):
        """Test reflectance with different polarizations."""
        R_s = ThinFilmOperand.reflectance(simple_stack, 550.0, polarization="s")
        R_p = ThinFilmOperand.reflectance(simple_stack, 550.0, polarization="p")
        R_u = ThinFilmOperand.reflectance(simple_stack, 550.0, polarization="u")

        assert isinstance(R_s, float)
        assert isinstance(R_p, float)
        assert isinstance(R_u, float)
        assert all(0 <= v <= 1 for v in [R_s, R_p, R_u])

    def test_operands_with_array_angles(self, simple_stack):
        """Test operands with array of angles."""
        angles = [0.0, 30.0, 60.0]

        R = ThinFilmOperand.reflectance(simple_stack, 550.0, aoi_deg=angles)
        T = ThinFilmOperand.transmittance(simple_stack, 550.0, aoi_deg=angles)
        A = ThinFilmOperand.absorptance(simple_stack, 550.0, aoi_deg=angles)

        assert isinstance(R, float)
        assert isinstance(T, float)
        assert isinstance(A, float)


class TestTargetTypes:
    """Test different target types and configurations."""

    def test_target_below(self, simple_stack):
        """Test 'below' target type."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_variable(layer_index=0, min_nm=50, max_nm=200)
        optimizer.add_operand("R", wavelength_nm=550, target_type="below", value=0.2)

        assert len(optimizer.targets) == 1
        assert optimizer.targets[0].target_type == "below"

    def test_target_over(self, simple_stack):
        """Test 'over' target type."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_variable(layer_index=0, min_nm=50, max_nm=200)
        optimizer.add_operand("T", wavelength_nm=550, target_type="over", value=0.5)

        assert len(optimizer.targets) == 1
        assert optimizer.targets[0].target_type == "over"

    def test_target_equal(self, simple_stack):
        """Test 'equal' target type."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_variable(layer_index=0)
        optimizer.add_operand(
            "R", wavelength_nm=550, target_type="equal", value=0.1, tolerance=1e-3
        )

        assert optimizer.targets[0].target_type == "equal"
        assert optimizer.targets[0].tolerance == 1e-3

    def test_invalid_target_type(self, simple_stack):
        """Test invalid target type raises error."""
        optimizer = ThinFilmOptimizer(simple_stack)

        with pytest.raises(ValueError, match="Invalid target_type"):
            optimizer.add_operand("R", 550, target_type="invalid", value=0.1)

    def test_invalid_property(self, simple_stack):
        """Test invalid property raises error."""
        optimizer = ThinFilmOptimizer(simple_stack)

        with pytest.raises(ValueError, match="Invalid property"):
            optimizer.add_operand("Z", 550, target_type="below", value=0.1)

    def test_spectral_operand_convenience(self, simple_stack):
        """Test add_spectral_operand convenience method."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_variable(layer_index=0)
        optimizer.add_spectral_operand(
            property="R",
            wavelengths_nm=[500.0, 550.0, 600.0],
            target_type="below",
            value=[0.1, 0.2, 0.1],
        )

        assert len(optimizer.targets) == 1
        assert isinstance(optimizer.targets[0].wavelength_nm, (list, np.ndarray))

    def test_angular_operand_convenience(self, simple_stack):
        """Test add_angular_operand convenience method."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_variable(layer_index=0)
        optimizer.add_angular_operand(
            property="R",
            wavelength_nm=550,
            aoi_deg_range=[0.0, 30.0, 60.0],
            target_type="below",
            value=0.2,
        )

        assert len(optimizer.targets) == 1
        assert isinstance(optimizer.targets[0].aoi_deg, (list, np.ndarray))

    def test_interpolated_operand_convenience(self, simple_stack):
        """Test add_interpolated_operand convenience method."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_variable(layer_index=0)
        optimizer.add_interpolated_operand(
            property="T",
            wavelength_nm=[500.0, 550.0, 600.0],
            target_type="over",
            value=[0.3, 0.5, 0.3],
        )

        assert len(optimizer.targets) == 1


class TestOptimizerMethods:
    """Test various optimizer methods."""

    def test_get_current_performance(self, simple_stack):
        """Test get_current_performance method."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_operand("R", wavelength_nm=550, target_type="below", value=0.2)
        optimizer.add_operand("T", wavelength_nm=550, target_type="over", value=0.5)

        performance = optimizer.get_current_performance()

        assert isinstance(performance, dict)
        assert len(performance) >= 2  # At least 2 targets

    def test_optimizer_with_multiple_variables(self, multilayer_stack):
        """Test optimizer with multiple variables."""
        optimizer = ThinFilmOptimizer(multilayer_stack)

        for i in range(len(multilayer_stack.layers)):
            optimizer.add_variable(layer_index=i, min_nm=50, max_nm=200)

        assert len(optimizer.variables) == len(multilayer_stack.layers)

    def test_method_chaining_fluent_interface(self, multilayer_stack):
        """Test fluent interface with method chaining."""
        optimizer = (
            ThinFilmOptimizer(multilayer_stack)
            .add_variable(layer_index=0, min_nm=50, max_nm=150)
            .add_variable(layer_index=1, min_nm=60, max_nm=120)
            .add_operand("R", wavelength_nm=500, target_type="below", value=0.1)
            .add_operand("R", wavelength_nm=600, target_type="below", value=0.1)
        )

        assert len(optimizer.variables) == 2
        assert len(optimizer.targets) == 2

    def test_bounds_validation(self, simple_stack):
        """Test bounds validation and correction."""
        optimizer = ThinFilmOptimizer(simple_stack)
        # Test with min > max (should be corrected internally)
        optimizer.add_variable(layer_index=0, min_nm=200, max_nm=50)

        assert len(optimizer.variables) == 1
        # The optimizer should handle this gracefully

    def test_custom_operand_registration_and_optimization(self, simple_stack):
        def layer_thickness_nm(stack, layer_index=0):
            return stack.layers[layer_index].thickness_um * 1000.0

        ThinFilmOptimizer.register_operand(
            "layer_thickness_nm_test", layer_thickness_nm, overwrite=True
        )

        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_variable(layer_index=0, min_nm=50, max_nm=200)
        optimizer.add_operand(
            "layer_thickness_nm_test",
            max_val=80.0,
            input_data={"layer_index": 0},
            label="Layer thickness",
        )

        result = optimizer.optimize(max_iterations=5, verbose=False)
        assert isinstance(result, dict)
        assert optimizer.targets[0].display_name == "Layer thickness"
        assert optimizer.get_current_performance()["target_0"]["current_value"] <= 80.1


class TestOptimizationWithArrayTargets:
    """Test optimization with array-based targets."""

    def test_optimization_with_wavelength_array(self, simple_stack):
        """Test optimization with wavelength array target."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_variable(layer_index=0, min_nm=50, max_nm=200)
        optimizer.add_operand(
            "R",
            wavelength_nm=[500.0, 550.0, 600.0],
            target_type="below",
            value=[0.1, 0.05, 0.1],
        )

        result = optimizer.optimize(max_iterations=3, verbose=False)
        assert isinstance(result, dict)
        assert "final_merit" in result

    def test_optimization_with_angular_array(self, simple_stack):
        """Test optimization with angular array target."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_variable(layer_index=0)
        optimizer.add_operand(
            "R",
            wavelength_nm=550.0,
            aoi_deg=[0.0, 30.0, 60.0],
            target_type="below",
            value=0.3,
        )

        result = optimizer.optimize(max_iterations=3, verbose=False)
        assert isinstance(result, dict)

    def test_target_value_array_validation(self, simple_stack):
        """Test that target value arrays are validated."""
        optimizer = ThinFilmOptimizer(simple_stack)

        # Should fail if value array length doesn't match wavelength array
        with pytest.raises(ValueError):
            optimizer.add_operand(
                "R",
                wavelength_nm=[500.0, 550.0, 600.0],
                target_type="below",
                value=[0.1, 0.05],  # Wrong length
            )

    def test_aoi_wavelength_array_conflict(self, simple_stack):
        """Test that both wavelength and AOI arrays raise error."""
        optimizer = ThinFilmOptimizer(simple_stack)

        with pytest.raises(ValueError, match="Cannot specify both"):
            optimizer.add_operand(
                "R",
                wavelength_nm=[500.0, 550.0],
                aoi_deg=[0.0, 30.0],
                target_type="below",
                value=0.1,
            )


class TestReportFunctionality:
    """Test report generation and analysis."""

    def test_report_summary_table(self, simple_stack):
        """Test report summary table generation."""
        from optiland.thin_film.optimization.report import ThinFilmReport

        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_variable(layer_index=0)
        optimizer.add_operand("R", wavelength_nm=550, target_type="below", value=0.01)

        # Mock result
        result = {"success": True, "fun": 0.001}

        report = ThinFilmReport(optimizer, result)
        summary = report.summary_table()

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 1
        assert "Layer 0 thickness" in summary["Variable"].values[0]

    def test_error_handling_invalid_layer_index(self, simple_stack):
        """Test error handling for invalid layer index."""
        optimizer = ThinFilmOptimizer(simple_stack)

        with pytest.raises(ValueError, match="out of range"):
            optimizer.add_variable(layer_index=10)

    def test_optimization_options(self, simple_stack):
        """Test optimization with different methods."""
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_variable(layer_index=0)
        optimizer.add_operand("R", wavelength_nm=550, target_type="below", value=0.1)

        # Test L-BFGS-B method (default)
        result = optimizer.optimize(method="L-BFGS-B", max_iterations=3, verbose=False)
        assert isinstance(result, dict)

    def test_zero_initial_thickness_handling(self, air, glass, sio2):
        """Test handling of edge case with very small initial thicknesses."""
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        # Create with very small thickness
        stack.add_layer_nm(sio2, 1.0, name="Thin SiO2")

        optimizer = ThinFilmOptimizer(stack)
        optimizer.add_variable(layer_index=0, min_nm=0.5, max_nm=10.0)
        optimizer.add_operand("R", wavelength_nm=550, target_type="below", value=0.2)

        result = optimizer.optimize(max_iterations=3, verbose=False)
        assert isinstance(result, dict)

    def test_large_thickness_values(self, air, glass, sio2):
        """Test handling of large thickness values."""
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(sio2, 500.0, name="Thick SiO2")

        optimizer = ThinFilmOptimizer(stack)
        optimizer.add_variable(layer_index=0, min_nm=100, max_nm=1000)
        optimizer.add_operand("R", wavelength_nm=550, target_type="below", value=0.2)

        result = optimizer.optimize(max_iterations=3, verbose=False)
        assert isinstance(result, dict)


class TestThinFilmOptimizerCoverageBranches:
    def test_interpolate_target_value_additional_branches(self, simple_stack):
        optimizer = ThinFilmOptimizer(simple_stack)

        target_aoi = optimizer_module.OptimizationTarget(
            property="R",
            wavelength_nm=550.0,
            target_type="equal",
            value=[0.1, 0.2, 0.3],
            weight=1.0,
            aoi_deg=[0.0, 30.0, 60.0],
            polarization="s",
            tolerance=1e-6,
        )
        interp_value = optimizer._interpolate_target_value(target_aoi, current_aoi=15.0)
        assert isinstance(interp_value, float)

        with pytest.raises(ValueError, match="current_aoi"):
            optimizer._interpolate_target_value(target_aoi)

        mismatch_target = optimizer_module.OptimizationTarget(
            property="R",
            wavelength_nm=550.0,
            target_type="equal",
            value=[0.1, 0.2],
            weight=1.0,
            aoi_deg=[0.0, 30.0, 60.0],
            polarization="s",
            tolerance=1e-6,
        )
        with pytest.raises(ValueError, match="same length"):
            optimizer._interpolate_target_value(mismatch_target, current_aoi=10.0)

        fallback_target = optimizer_module.OptimizationTarget(
            property="R",
            wavelength_nm=550.0,
            target_type="equal",
            value=[0.7, 0.8],
            weight=1.0,
            aoi_deg=0.0,
            polarization="u",
            tolerance=1e-6,
        )
        assert optimizer._interpolate_target_value(fallback_target) == 0.7

    def test_merit_function_and_optimize_kwargs_branches(
        self, simple_stack, monkeypatch
    ):
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_variable(layer_index=0, min_nm=50, max_nm=180)
        optimizer.add_operand(
            property="A",
            wavelength_nm=[500.0, 550.0],
            target_type="equal",
            value=[0.0, 0.0],
            polarization="s",
        )
        optimizer.add_operand(
            property="T",
            wavelength_nm=[500.0, 550.0],
            target_type="over",
            value=[0.6, 0.6],
            polarization="p",
        )

        # Force numpy-scalar merit so the ".item()" path is exercised.
        monkeypatch.setattr(
            ThinFilmOperand,
            "absorptance",
            staticmethod(lambda *a, **k: np.float64(0.05)),
        )
        monkeypatch.setattr(
            ThinFilmOperand,
            "transmittance",
            staticmethod(lambda *a, **k: np.float64(0.7)),
        )

        x0 = np.array([optimizer.variables[0].variable.get_value()])
        merit = optimizer._merit_function(x0)
        assert isinstance(merit, float)

        result = optimizer.optimize(max_iterations=2, disp=True, iprint=1, gtol=1e-9)
        assert isinstance(result, dict)
        assert "final_merit" in result

    def test_get_current_performance_array_cases(self, simple_stack):
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_operand(
            property="T",
            wavelength_nm=[500.0, 550.0, 600.0],
            target_type="over",
            value=[0.5, 0.6, 0.5],
            aoi_deg=10.0,
        )
        optimizer.add_operand(
            property="A",
            wavelength_nm=550.0,
            target_type="below",
            value=[0.2, 0.2, 0.2],
            aoi_deg=[0.0, 30.0, 60.0],
        )

        perf = optimizer.get_current_performance()
        assert "target_0" in perf
        assert "target_1" in perf
        assert len(perf["target_0"]["target_values"]) == 3
        assert len(perf["target_1"]["target_values"]) == 3

    def test_add_spectral_operand_with_weights_average(self, simple_stack):
        optimizer = ThinFilmOptimizer(simple_stack)
        optimizer.add_spectral_operand(
            property="R",
            wavelengths_nm=[450.0, 550.0, 650.0],
            target_type="below",
            value=0.2,
            weights=[1.0, 2.0, 3.0],
        )
        assert len(optimizer.targets) == 1
        assert optimizer.targets[0].weight == pytest.approx(2.0)

    def test_plot_operands_branches(self, simple_stack, monkeypatch):
        optimizer = ThinFilmOptimizer(simple_stack)

        # No target branch should return cleanly.
        fig, ax = plt.subplots()
        optimizer.plot_operands(ax)
        plt.close(fig)

        optimizer.add_operand(
            property="R",
            wavelength_nm=[500.0, 550.0, 600.0],
            target_type="below",
            value=[0.3, 0.2, 0.3],
            aoi_deg=10.0,
        )
        optimizer.add_operand(
            property="T",
            wavelength_nm=550.0,
            target_type="over",
            value=0.7,
            aoi_deg=20.0,
        )
        optimizer.add_operand(
            property="A",
            wavelength_nm=550.0,
            target_type="below",
            value=[0.1, 0.15, 0.2],
            aoi_deg=[0.0, 30.0, 60.0],
        )

        fig, ax = plt.subplots()
        optimizer.plot_operands(ax, plot_type="wavelength")
        plt.close(fig)

        fig, ax = plt.subplots()
        optimizer.plot_operands(ax, plot_type="angle")
        plt.close(fig)

        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="Invalid plot_type"):
            optimizer.plot_operands(ax, plot_type="invalid")
        plt.close(fig)

        monkeypatch.setattr(optimizer_module, "plt", None)
        fig, ax = plt.subplots()
        with pytest.raises(ImportError, match="matplotlib is required"):
            optimizer.plot_operands(ax)
        plt.close(fig)

    def test_info_without_tabulate_and_with_result(
        self, multilayer_stack, capsys, monkeypatch
    ):
        optimizer = ThinFilmOptimizer(multilayer_stack)
        optimizer.add_variable(layer_index=0, min_nm=50, max_nm=150, apply_scaling=True)
        optimizer.add_variable(
            layer_index=1, min_nm=60, max_nm=120, apply_scaling=False
        )
        optimizer.add_operand(
            property="R",
            wavelength_nm=[500.0, 600.0],
            target_type="below",
            value=[0.2, 0.2],
            aoi_deg=10.0,
        )

        optimizer.result = SimpleNamespace(
            success=True, fun=0.123, nit=4, method="L-BFGS-B"
        )

        import builtins

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "tabulate":
                raise ImportError("forced for coverage")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        optimizer.info()
        out = capsys.readouterr().out
        assert "ThinFilm Optimizer Information" in out
        assert "Last Optimization Result" in out
