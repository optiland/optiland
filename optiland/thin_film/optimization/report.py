"""Thin Film Optimization Report Module

This module contains classes for generating detailed reports of thin film
optimization results, including before/after comparisons and performance
analysis.

Corentin Nannini, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .optimizer import ThinFilmOptimizer


@dataclass
class OptimizationResult:
    """Enhanced optimization result with reporting capabilities."""

    original_result: Any
    report: ThinFilmReport
    optimizer: ThinFilmOptimizer

    def __getattr__(self, name):
        """Delegate attribute access to original result."""
        return getattr(self.original_result, name)


class ThinFilmReport:
    """Generates detailed reports for thin film optimization results.

    This class provides methods to analyze and visualize the results of
    thin film optimization, including before/after comparisons and
    performance metrics.

    Args:
        optimizer: The ThinFilmOptimizer that was used.
        result: The optimization result object.
    """

    def __init__(self, optimizer: ThinFilmOptimizer, result: Any):
        self.optimizer = optimizer
        self.result = result
        self.stack = optimizer.stack

        # Store optimization data
        self._initial_thicknesses = optimizer._initial_thicknesses.copy()
        self._final_thicknesses = [layer.thickness_um for layer in self.stack.layers]

    def summary_table(self) -> pd.DataFrame:
        """Generate a summary table of optimization variables.

        Returns:
            DataFrame with columns: Variable, Initial, Final, Change, Unit
        """
        data = []

        for _i, var in enumerate(self.optimizer.variables):
            layer_idx = var.layer_index
            initial_nm = self._initial_thicknesses[layer_idx] * 1000
            final_nm = self._final_thicknesses[layer_idx] * 1000
            change_nm = final_nm - initial_nm
            change_pct = (change_nm / initial_nm) * 100 if initial_nm != 0 else 0

            data.append(
                {
                    "Variable": f"Layer {layer_idx} thickness",
                    "Initial": f"{initial_nm:.1f}",
                    "Final": f"{final_nm:.1f}",
                    "Change": f"{change_nm:+.1f} ({change_pct:+.1f}%)",
                    "Unit": "nm",
                }
            )

        return pd.DataFrame(data)
