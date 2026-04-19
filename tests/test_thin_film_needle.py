"""Tests for needle synthesis module."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.materials import IdealMaterial
from optiland.thin_film import ThinFilmStack
from optiland.thin_film.optimization import ThinFilmOperand
from optiland.thin_film.optimization.needle import (
    NeedleResult,
    NeedleSynthesis,
    NeedleSynthesisResult,
)


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
def simple_stack(air, glass, sio2):
    stack = ThinFilmStack(incident_material=air, substrate_material=glass)
    stack.add_layer_nm(sio2, 100.0, name="SiO2")
    return stack


class TestStackInfrastructure:
    """Test the new stack methods needed for needle synthesis."""

    def test_insert_layer(self, air, glass, sio2, tio2):
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(sio2, 100.0)
        stack.insert_layer_nm(0, tio2, 50.0)
        assert len(stack) == 2
        assert stack.layers[0].material is tio2
        assert stack.layers[1].material is sio2

    def test_insert_layer_end(self, air, glass, sio2, tio2):
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(sio2, 100.0)
        stack.insert_layer_nm(1, tio2, 50.0)
        assert len(stack) == 2
        assert stack.layers[1].material is tio2

    def test_remove_layer(self, air, glass, sio2, tio2):
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(sio2, 100.0)
        stack.add_layer_nm(tio2, 50.0)
        removed = stack.remove_layer(0)
        assert removed.material is sio2
        assert len(stack) == 1
        assert stack.layers[0].material is tio2

    def test_split_layer(self, air, glass, sio2):
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(sio2, 100.0)
        stack.split_layer(0, 0.3)
        assert len(stack) == 2
        assert stack.layers[0].material is sio2
        assert stack.layers[1].material is sio2
        np.testing.assert_allclose(
            stack.layers[0].thickness_um * 1000.0, 30.0, atol=1e-10
        )
        np.testing.assert_allclose(
            stack.layers[1].thickness_um * 1000.0, 70.0, atol=1e-10
        )

    def test_split_layer_boundary_fractions(self, air, glass, sio2):
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(sio2, 100.0)
        with pytest.raises(ValueError):
            stack.split_layer(0, 0.0)
        with pytest.raises(ValueError):
            stack.split_layer(0, 1.0)

    def test_deep_copy(self, air, glass, sio2, tio2):
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(sio2, 100.0)
        stack.add_layer_nm(tio2, 50.0)
        copy = stack.deep_copy()

        assert len(copy) == len(stack)
        assert copy.layers[0].material is stack.layers[0].material
        assert copy.layers[0] is not stack.layers[0]

        # Modifying copy shouldn't affect original
        copy.layers[0].thickness_um = 0.5
        assert stack.layers[0].thickness_um != 0.5


class TestNeedleSynthesisSetup:
    """Test NeedleSynthesis initialization and target management."""

    def test_init(self, simple_stack, sio2, tio2):
        ns = NeedleSynthesis(
            stack=simple_stack,
            candidate_materials=[sio2, tio2],
        )
        assert ns.max_iterations == 50
        assert ns.needle_thickness_nm == 1.0

    def test_add_target_chaining(self, simple_stack, sio2, tio2):
        ns = NeedleSynthesis(
            stack=simple_stack,
            candidate_materials=[sio2, tio2],
        )
        result = ns.add_target("R", 550.0, "equal", 0.0)
        assert result is ns

    def test_add_spectral_target(self, simple_stack, sio2, tio2):
        ns = NeedleSynthesis(
            stack=simple_stack,
            candidate_materials=[sio2, tio2],
        )
        ns.add_spectral_target("R", [400.0, 500.0, 600.0], "below", 0.01)
        assert len(ns._targets) == 1

    def test_add_custom_target(self, simple_stack, sio2, tio2):
        ns = NeedleSynthesis(
            stack=simple_stack,
            candidate_materials=[sio2, tio2],
        )
        ns.add_custom_target(
            "reflection_phase",
            target=0.0,
            input_data={
                "wavelength_nm": 550.0,
                "aoi_deg": 0.0,
                "polarization": "s",
            },
        )
        assert len(ns._targets) == 1
        opt = ns._build_optimizer(simple_stack)
        perf = opt.get_current_performance()
        assert perf["target_0"]["property"] == "reflection_phase"
        np.testing.assert_allclose(
            perf["target_0"]["current_value"],
            ThinFilmOperand.reflection_phase(simple_stack, 550.0, 0.0, "s"),
            atol=1e-12,
        )

    def test_run_without_targets_raises(self, simple_stack, sio2, tio2):
        ns = NeedleSynthesis(
            stack=simple_stack,
            candidate_materials=[sio2, tio2],
        )
        with pytest.raises(ValueError, match="No targets"):
            ns.run()


class TestNeedleSynthesisInternal:
    """Test internal needle synthesis methods."""

    def test_generate_trial_positions(self, simple_stack, sio2, tio2):
        ns = NeedleSynthesis(
            stack=simple_stack,
            candidate_materials=[sio2, tio2],
            num_positions_per_layer=5,
        )
        positions = ns._generate_trial_positions(simple_stack)
        # 1 layer * 5 internal + 2 boundary positions (before layer 0, after layer 0)
        assert len(positions) == 5 + 2

    def test_insert_needle_at_boundary(self, simple_stack, sio2, tio2):
        stack = simple_stack.deep_copy()
        ns = NeedleSynthesis(stack=stack, candidate_materials=[sio2, tio2])
        ns._insert_needle_at(stack, 0, 0.0, tio2, 10.0)
        assert len(stack) == 2
        assert stack.layers[0].material is tio2

    def test_insert_needle_internal(self, simple_stack, sio2, tio2):
        stack = simple_stack.deep_copy()
        ns = NeedleSynthesis(stack=stack, candidate_materials=[sio2, tio2])
        ns._insert_needle_at(stack, 0, 0.5, tio2, 10.0)
        # Original layer split into 2, plus needle = 3 layers
        assert len(stack) == 3
        assert stack.layers[0].material is sio2
        assert stack.layers[1].material is tio2
        assert stack.layers[2].material is sio2

    def test_cleanup_removes_thin_layers(self, air, glass, sio2, tio2):
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(sio2, 100.0)
        stack.add_layer_nm(tio2, 0.5)  # below min_thickness_nm
        stack.add_layer_nm(sio2, 80.0)

        ns = NeedleSynthesis(
            stack=stack, candidate_materials=[sio2, tio2], min_thickness_nm=1.0
        )
        ns._cleanup_stack(stack)
        assert len(stack) == 1  # Two SiO2 layers merged after TiO2 removed
        np.testing.assert_allclose(
            stack.layers[0].thickness_um * 1000.0, 180.0, atol=1e-10
        )

    def test_cleanup_merges_adjacent(self, air, glass, sio2):
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(sio2, 50.0)
        stack.add_layer_nm(sio2, 30.0)

        ns = NeedleSynthesis(stack=stack, candidate_materials=[sio2])
        ns._cleanup_stack(stack)
        assert len(stack) == 1
        np.testing.assert_allclose(
            stack.layers[0].thickness_um * 1000.0, 80.0, atol=1e-10
        )


class TestNeedleSynthesisEndToEnd:
    """End-to-end tests for needle synthesis."""

    def test_simple_ar_merit_decreases(self, air, glass, sio2, tio2):
        """Start from a single layer and verify merit decreases."""
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(sio2, 90.0)

        ns = NeedleSynthesis(
            stack=stack,
            candidate_materials=[sio2, tio2],
            max_iterations=3,
            num_positions_per_layer=3,
            optimizer_max_iter=50,
        )
        wls = np.linspace(450, 650, 5).tolist()
        ns.add_spectral_target("R", wls, "equal", 0.0)

        result = ns.run()

        assert isinstance(result, NeedleSynthesisResult)
        assert result.final_merit <= result.initial_merit

    def test_convergence_no_improvement(self, air, glass, sio2):
        """With only one candidate material same as existing layer, should
        converge quickly when already optimized."""
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(sio2, 94.0)

        ns = NeedleSynthesis(
            stack=stack,
            candidate_materials=[sio2],
            max_iterations=2,
            num_positions_per_layer=2,
            optimizer_max_iter=50,
        )
        ns.add_target("R", 550.0, "equal", 0.0)
        result = ns.run()

        assert isinstance(result, NeedleSynthesisResult)
        assert result.num_iterations <= 2

    def test_result_has_history(self, air, glass, sio2, tio2):
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(sio2, 100.0)

        ns = NeedleSynthesis(
            stack=stack,
            candidate_materials=[sio2, tio2],
            max_iterations=2,
            num_positions_per_layer=3,
            optimizer_max_iter=30,
        )
        ns.add_target("R", 550.0, "equal", 0.0)
        result = ns.run()

        assert isinstance(result.history, list)
        for entry in result.history:
            assert isinstance(entry, NeedleResult)
            assert entry.improvement >= 0


class TestNeedleSynthesisRealistic:
    """Realistic integration tests with real-world thin film design specs.

    These tests use realistic materials, wavelength ranges, and performance
    targets typical of commercial thin film coatings.

    Run numpy-only: the physics correctness (avg_R < 3%, monotonic merit) is
    backend-independent. Torch coverage is maintained by TestNeedleSynthesisEndToEnd.
    """

    @pytest.fixture(autouse=True)
    def numpy_only(self, set_test_backend):
        """Skip torch backend for these expensive integration tests."""
        if be.get_backend() != "numpy":
            pytest.skip("Realistic integration tests run numpy-only for speed")

    def test_broadband_ar_from_v_coat(self, air, glass, sio2, tio2):
        """Broadband visible AR coating (430-670nm) starting from a
        deliberately suboptimal 2-layer H/L design.

        Realistic spec: average R < 3% over 430-670 nm after synthesis.
        Typical commercial broadband AR coatings achieve < 1% average R with
        4-6 layers.
        """
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(tio2, 110.0)  # suboptimal quarter-wave
        stack.add_layer_nm(sio2, 160.0)  # suboptimal quarter-wave

        ns = NeedleSynthesis(
            stack=stack,
            candidate_materials=[sio2, tio2],
            max_iterations=5,
            num_positions_per_layer=3,
            optimizer_max_iter=80,
        )
        wls = np.linspace(430, 670, 12).tolist()
        ns.add_spectral_target("R", wls, "equal", 0.0)

        result = ns.run()

        # Merit must decrease
        assert result.final_merit < result.initial_merit
        # Must have added at least one layer
        assert len(result.stack.layers) > 2
        # Monotonic merit in history (each accepted needle improves things)
        for entry in result.history:
            assert entry.improvement > 0
        # Average reflectance over visible band should be significantly
        # reduced from starting design (~10% → < 3%)
        from optiland.thin_film.optimization.operand.thin_film import (
            ThinFilmOperand,
        )

        R_vals = [
            ThinFilmOperand.reflectance(result.stack, wl)
            for wl in np.linspace(430, 670, 30)
        ]
        avg_R = np.mean(R_vals)
        assert avg_R < 0.03, f"Average R = {avg_R:.4f}, expected < 0.03"

    def test_needle_merit_monotonically_decreases(self, air, glass, sio2, tio2):
        """Verify the rollback mechanism ensures merit never increases
        between accepted iterations."""
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(tio2, 100.0)
        stack.add_layer_nm(sio2, 150.0)

        ns = NeedleSynthesis(
            stack=stack,
            candidate_materials=[sio2, tio2],
            max_iterations=4,
            num_positions_per_layer=3,
            optimizer_max_iter=60,
        )
        wls = np.linspace(450, 650, 9).tolist()
        ns.add_spectral_target("R", wls, "equal", 0.0)

        result = ns.run()

        # Build merit trajectory: initial → each accepted iteration
        merits = [result.initial_merit]
        for entry in result.history:
            merits.append(entry.merit_after)

        # Each step must be <= previous (monotonic decrease)
        for i in range(1, len(merits)):
            assert (
                merits[i] <= merits[i - 1]
            ), f"Merit increased at step {i}: {merits[i - 1]:.6e} → {merits[i]:.6e}"

    def test_single_wavelength_ar_at_550nm(self, air, glass, sio2, tio2):
        """Single-wavelength V-coat design at 550 nm.

        Starting from a single SiO2 layer, needle synthesis should discover
        that a thin TiO2 layer improves AR at the target wavelength.
        """
        stack = ThinFilmStack(incident_material=air, substrate_material=glass)
        stack.add_layer_nm(sio2, 94.0)  # near quarter-wave for SiO2 at 550nm

        ns = NeedleSynthesis(
            stack=stack,
            candidate_materials=[sio2, tio2],
            max_iterations=5,
            num_positions_per_layer=5,
            optimizer_max_iter=150,
        )
        ns.add_target("R", 550.0, "equal", 0.0)

        result = ns.run()

        assert result.final_merit <= result.initial_merit
        # Check that R at 550nm is reasonably low
        from optiland.thin_film.optimization.operand.thin_film import (
            ThinFilmOperand,
        )

        R_550 = ThinFilmOperand.reflectance(result.stack, 550.0)
        # Single layer SiO2 on glass gives ~4.2% at 550nm; needle synthesis
        # should improve this noticeably
        assert R_550 < 0.04, f"R(550nm) = {R_550:.4f}, expected < 0.04"
