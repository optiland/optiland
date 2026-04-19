"""Needle Synthesis for thin film stack design.

Iterative layer insertion algorithm based on:
Tikhonravov & Trubetskov, "Development of the needle optimization technique
and new features of OptiLayer design software", SPIE Vol. 2253, 1994.

This module uses a numerical approach: trial needles are inserted at sampled
positions and the full merit function is evaluated, rather than computing
analytical variational derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import minimize_scalar

from .optimizer import ThinFilmOptimizer

if TYPE_CHECKING:
    from optiland.materials import BaseMaterial
    from optiland.thin_film import ThinFilmStack

OpticalProperty = Literal["R", "T", "A"]
TargetType = Literal["equal", "below", "over"]
OptimizerMethod = Literal["L-BFGS-B", "TNC", "SLSQP"]


def _material_label(material: BaseMaterial) -> str:
    """Human-readable label for a material."""
    name = getattr(material, "name", None)
    if name and name != material.__class__.__name__:
        return name
    # For IdealMaterial, show the refractive index
    try:
        n_val = float(material.n(0.55))
        return f"n={n_val:.2f}"
    except Exception:
        return material.__class__.__name__


@dataclass
class NeedleResult:
    """Result of a single needle insertion."""

    iteration: int
    layer_index: int
    position_fraction: float
    material_name: str
    thickness_nm: float
    merit_before: float
    merit_after: float
    improvement: float


@dataclass
class NeedleSynthesisResult:
    """Overall result of needle synthesis."""

    success: bool
    num_iterations: int
    num_layers_added: int
    initial_merit: float
    final_merit: float
    history: list[NeedleResult]
    stack: ThinFilmStack


@dataclass
class _NeedleCandidate:
    """Internal: best needle found during search (stores material ref)."""

    layer_index: int
    position_fraction: float
    material: BaseMaterial
    material_name: str
    improvement: float


class NeedleSynthesis:
    """Needle synthesis algorithm for discovering optimal thin film stack
    configurations by iteratively inserting thin layers at optimal positions.

    Args:
        stack: Starting thin film stack design.
        candidate_materials: Materials to try inserting as needles.
        needle_thickness_nm: Trial needle thickness in nm used for position
            screening.  Smaller values (0.5–2 nm) give a better
            finite-difference approximation of the variational gradient.
        min_thickness_nm: Layers thinner than this are removed during cleanup.
        max_iterations: Maximum number of needle insertion iterations.
        target_merit: Stop early if merit falls below this value.
        num_positions_per_layer: Number of internal sampling points per layer.
        optimizer_method: Scipy method for thickness re-optimization.
        optimizer_max_iter: Max iterations for thickness re-optimization.
    """

    def __init__(
        self,
        stack: ThinFilmStack,
        candidate_materials: list[BaseMaterial],
        needle_thickness_nm: float = 1.0,
        min_thickness_nm: float = 1.0,
        max_iterations: int = 50,
        target_merit: float | None = None,
        num_positions_per_layer: int = 10,
        optimizer_method: OptimizerMethod = "L-BFGS-B",
        optimizer_max_iter: int = 200,
    ):
        self.stack = stack
        self.candidate_materials = candidate_materials
        self.needle_thickness_nm = needle_thickness_nm
        self.min_thickness_nm = min_thickness_nm
        self.max_iterations = max_iterations
        self.target_merit = target_merit
        self.num_positions_per_layer = num_positions_per_layer
        self.optimizer_method = optimizer_method
        self.optimizer_max_iter = optimizer_max_iter
        self._targets: list[dict] = []

    def add_target(
        self,
        property: OpticalProperty,
        wavelength_nm: float | list[float],
        target_type: TargetType,
        value: float | list[float],
        weight: float = 1.0,
        aoi_deg: float | list[float] = 0.0,
        polarization: str = "u",
        tolerance: float = 1e-6,
    ) -> NeedleSynthesis:
        """Add an optimization target.

        Args:
            property: Optical property to target ('R', 'T', or 'A').
            wavelength_nm: Wavelength(s) in nanometers.
            target_type: Type of target ('equal', 'below', 'over').
            value: Target value(s).
            weight: Weight for this target.
            aoi_deg: Angle(s) of incidence in degrees.
            polarization: Polarization state ('s', 'p', 'u').
            tolerance: Tolerance for 'equal' targets.

        Returns:
            self for method chaining.
        """
        self._targets.append(
            dict(
                property=property,
                wavelength_nm=wavelength_nm,
                target_type=target_type,
                value=value,
                weight=weight,
                aoi_deg=aoi_deg,
                polarization=polarization,
                tolerance=tolerance,
            )
        )
        return self

    def add_spectral_target(
        self,
        property: OpticalProperty,
        wavelengths_nm: list[float],
        target_type: TargetType,
        value: float,
        weight: float = 1.0,
        aoi_deg: float = 0.0,
        polarization: str = "u",
        tolerance: float = 1e-6,
    ) -> NeedleSynthesis:
        """Add a spectral target over multiple wavelengths.

        Args:
            property: Optical property ('R', 'T', or 'A').
            wavelengths_nm: List of wavelengths in nanometers.
            target_type: Type of target ('equal', 'below', 'over').
            value: Target value (single scalar applied at all wavelengths).
            weight: Weight for this target.
            aoi_deg: Angle of incidence in degrees.
            polarization: Polarization state ('s', 'p', 'u').
            tolerance: Tolerance for 'equal' targets.

        Returns:
            self for method chaining.
        """
        return self.add_target(
            property=property,
            wavelength_nm=wavelengths_nm,
            target_type=target_type,
            value=value,
            weight=weight,
            aoi_deg=aoi_deg,
            polarization=polarization,
            tolerance=tolerance,
        )

    def add_custom_target(
        self,
        operand_type: str,
        target: float | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
        weight: float = 1.0,
        input_data: dict[str, Any] | None = None,
        label: str | None = None,
    ) -> NeedleSynthesis:
        """Add a custom thin-film target backed by a registered operand.

        This mirrors the custom operand path of ``ThinFilmOptimizer`` and makes
        thin-film metrics such as phase, GD, GDD, and internal field values
        available to needle synthesis.
        """
        self._targets.append(
            dict(
                operand_type=operand_type,
                target=target,
                min_val=min_val,
                max_val=max_val,
                weight=weight,
                input_data=input_data,
                label=label,
            )
        )
        return self

    def _apply_targets(self, optimizer: ThinFilmOptimizer) -> None:
        """Copy stored targets onto a ThinFilmOptimizer instance."""
        for t in self._targets:
            optimizer.add_operand(**t)

    def _build_optimizer(self, stack: ThinFilmStack) -> ThinFilmOptimizer:
        """Build a fresh ThinFilmOptimizer for *stack* with all variables and
        targets configured."""
        opt = ThinFilmOptimizer(stack)
        for i in range(len(stack.layers)):
            opt.add_variable(i, min_nm=0.0)
        self._apply_targets(opt)
        return opt

    def _compute_merit(self, stack: ThinFilmStack) -> float:
        """Compute the merit function for *stack*."""
        opt = self._build_optimizer(stack)
        x = np.array([v.variable.get_value() for v in opt.variables])
        return opt._merit_function(x)

    def _reoptimize(self, stack: ThinFilmStack) -> float:
        """Re-optimize all layer thicknesses and return the new merit."""
        opt = self._build_optimizer(stack)
        opt.optimize(
            method=self.optimizer_method,
            max_iterations=self.optimizer_max_iter,
        )
        return self._compute_merit(stack)

    def _generate_trial_positions(
        self, stack: ThinFilmStack
    ) -> list[tuple[int, float]]:
        """Generate (layer_index, fraction) positions for trial needles.

        Includes positions at layer boundaries (fraction 0 = before layer)
        and internal points within each layer.
        """
        positions: list[tuple[int, float]] = []
        n_layers = len(stack.layers)

        for i in range(n_layers):
            # Internal positions within the layer
            for j in range(1, self.num_positions_per_layer + 1):
                frac = j / (self.num_positions_per_layer + 1)
                positions.append((i, frac))

        # Also include insertion at layer boundaries (as new layer before index)
        for i in range(n_layers + 1):
            positions.append((i, 0.0))  # 0.0 signals "insert at boundary"

        return positions

    def _insert_needle_at(
        self,
        stack: ThinFilmStack,
        layer_index: int,
        fraction: float,
        material: BaseMaterial,
        thickness_nm: float,
    ) -> None:
        """Insert a needle into *stack* (mutates in place).

        If fraction == 0.0, insert a new layer before layer_index.
        Otherwise split the layer at the given fraction and insert the needle
        between the two halves.
        """
        if fraction == 0.0:
            # Insert at boundary (before layer_index)
            stack.insert_layer_nm(layer_index, material, thickness_nm)
        else:
            # Split the host layer, then insert needle between the halves
            stack.split_layer(layer_index, fraction)
            # After split, layer_index is the first half, layer_index+1 is the
            # second half.  Insert needle between them.
            stack.insert_layer_nm(layer_index + 1, material, thickness_nm)

    def _find_best_needle(
        self,
        stack: ThinFilmStack,
        current_merit: float,
        rejected: set[tuple[int, float, int]] | None = None,
    ) -> _NeedleCandidate | None:
        """Find the best needle insertion (material + position).

        Args:
            stack: Current stack.
            current_merit: Current merit value.
            rejected: Set of ``(layer_index, fraction, material_idx)`` tuples
                to skip (previously rejected candidates).

        Returns None if no insertion improves the merit function.
        """
        if rejected is None:
            rejected = set()

        positions = self._generate_trial_positions(stack)
        best: _NeedleCandidate | None = None

        for mat_idx, material in enumerate(self.candidate_materials):
            for layer_index, fraction in positions:
                if (layer_index, fraction, mat_idx) in rejected:
                    continue

                trial = stack.deep_copy()
                self._insert_needle_at(
                    trial, layer_index, fraction, material, self.needle_thickness_nm
                )
                trial_merit = self._compute_merit(trial)
                improvement = current_merit - trial_merit

                if improvement > 0 and (best is None or improvement > best.improvement):
                    best = _NeedleCandidate(
                        layer_index=layer_index,
                        position_fraction=fraction,
                        material=material,
                        material_name=_material_label(material),
                        improvement=improvement,
                    )

        return best

    def _optimize_needle_thickness(
        self,
        stack: ThinFilmStack,
        layer_index: int,
        fraction: float,
        material: BaseMaterial,
    ) -> float:
        """Find optimal needle thickness via scalar minimization."""

        def _merit_for_delta(delta_nm: float) -> float:
            trial = stack.deep_copy()
            self._insert_needle_at(trial, layer_index, fraction, material, delta_nm)
            return self._compute_merit(trial)

        result = minimize_scalar(
            _merit_for_delta,
            bounds=(0.5, 500.0),
            method="bounded",
        )
        return float(result.x)

    def _cleanup_stack(self, stack: ThinFilmStack) -> None:
        """Remove very thin layers and merge adjacent same-material layers."""
        # Remove layers below min_thickness_nm
        i = 0
        while i < len(stack.layers):
            if stack.layers[i].thickness_um * 1000.0 < self.min_thickness_nm:
                stack.layers.pop(i)
            else:
                i += 1

        # Merge adjacent layers of the same material
        i = 0
        while i < len(stack.layers) - 1:
            if stack.layers[i].material is stack.layers[i + 1].material:
                stack.layers[i].thickness_um += stack.layers[i + 1].thickness_um
                stack.layers.pop(i + 1)
            else:
                i += 1

    def run(self, verbose: bool = False) -> NeedleSynthesisResult:
        """Execute the needle synthesis algorithm.

        Args:
            verbose: If True, print progress information.

        Returns:
            NeedleSynthesisResult with the optimized stack and history.
        """
        if not self._targets:
            raise ValueError("No targets defined. Use add_target() first.")

        history: list[NeedleResult] = []

        # Step 1: refine the starting design
        current_merit = self._reoptimize(self.stack)
        initial_merit = current_merit

        if verbose:
            print(f"Initial merit after refinement: {current_merit:.6e}")

        # Track rejected (layer_index, fraction, material_idx) combos to
        # avoid retrying the same positions.  Reset after each accepted needle
        # because the stack geometry has changed.
        rejected: set[tuple[int, float, int]] = set()

        for iteration in range(self.max_iterations):
            # Step 3a: find best needle
            candidate = self._find_best_needle(self.stack, current_merit, rejected)

            if candidate is None:
                if verbose:
                    print(
                        f"Iteration {iteration}: no improving needle found — converged."
                    )
                break

            # Step 3c: optimize needle thickness
            optimal_delta = self._optimize_needle_thickness(
                self.stack,
                candidate.layer_index,
                candidate.position_fraction,
                candidate.material,
            )

            if optimal_delta < self.min_thickness_nm:
                if verbose:
                    print(
                        f"Iteration {iteration}: optimal needle thickness "
                        f"{optimal_delta:.3f} nm < min — converged."
                    )
                break

            # Save a snapshot so we can roll back if merit worsens
            snapshot = self.stack.deep_copy()
            merit_before_insert = current_merit

            # Step 3d: insert needle at optimal position with optimal thickness
            self._insert_needle_at(
                self.stack,
                candidate.layer_index,
                candidate.position_fraction,
                candidate.material,
                optimal_delta,
            )

            # Step 3e: re-optimize all thicknesses
            new_merit = self._reoptimize(self.stack)

            # Step 3f: cleanup then re-optimize
            self._cleanup_stack(self.stack)
            if len(self.stack.layers) > 0:
                new_merit = self._reoptimize(self.stack)
            else:
                new_merit = self._compute_merit(self.stack)

            # Reject insertion if merit worsened after full re-optimization
            if new_merit >= merit_before_insert:
                # Roll back
                self.stack.layers[:] = snapshot.layers
                # Record this candidate so we don't retry it
                mat_idx = self.candidate_materials.index(candidate.material)
                rejected.add(
                    (candidate.layer_index, candidate.position_fraction, mat_idx)
                )
                if verbose:
                    print(
                        f"Iteration {iteration}: {candidate.material_name} "
                        f"rejected (merit {new_merit:.6e} "
                        f">= {merit_before_insert:.6e})"
                    )
                continue

            # Accepted — reset rejected set since stack geometry changed
            rejected.clear()
            current_merit = new_merit

            entry = NeedleResult(
                iteration=iteration,
                layer_index=candidate.layer_index,
                position_fraction=candidate.position_fraction,
                material_name=candidate.material_name,
                thickness_nm=optimal_delta,
                merit_before=merit_before_insert,
                merit_after=current_merit,
                improvement=merit_before_insert - current_merit,
            )
            history.append(entry)

            if verbose:
                print(
                    f"Iteration {iteration}: inserted {candidate.material_name} "
                    f"({optimal_delta:.1f} nm), "
                    f"layers={len(self.stack.layers)}, "
                    f"merit = {current_merit:.6e}"
                )

            # Step 3g: early stop
            if self.target_merit is not None and current_merit <= self.target_merit:
                if verbose:
                    print(f"Target merit {self.target_merit:.6e} reached.")
                break

        return NeedleSynthesisResult(
            success=current_merit < initial_merit,
            num_iterations=len(history),
            num_layers_added=len(history),
            initial_merit=initial_merit,
            final_merit=current_merit,
            history=history,
            stack=self.stack,
        )
