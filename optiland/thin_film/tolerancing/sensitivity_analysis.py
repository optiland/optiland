"""Sensitivity analysis for thin film stacks.

Sweeps each perturbation individually (requires ``RangeSampler``) while
evaluating all operands, following the pattern of
``optiland.tolerancing.sensitivity_analysis``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import optiland.backend as be
from optiland.tolerancing.perturbation import RangeSampler

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .core import ThinFilmTolerancing


class ThinFilmSensitivityAnalysis:
    """Sensitivity analysis for a thin-film tolerancing setup.

    Args:
        tolerancing: A configured ``ThinFilmTolerancing`` instance with at
            least one operand and one perturbation.
    """

    def __init__(self, tolerancing: ThinFilmTolerancing):
        self.tolerancing = tolerancing
        self.operand_names = [
            f"{i}: {op.property}@{op.wavelength_nm}nm"
            for i, op in enumerate(tolerancing.operands)
        ]
        self._results = pd.DataFrame()
        self._validate()

    def run(self) -> None:
        """Run the sensitivity analysis.

        Each perturbation is swept individually while the others remain at
        their nominal values.
        """
        results: list[dict] = []

        for perturbation in self.tolerancing.perturbations:
            if not isinstance(perturbation.sampler, RangeSampler):
                raise ValueError(
                    "Sensitivity analysis requires RangeSampler for all perturbations."
                )

            num_steps = perturbation.sampler.size
            for _ in range(num_steps):
                self.tolerancing.reset()
                perturbation.apply()

                operand_values = self.tolerancing.evaluate()

                result: dict = {
                    "perturbation_type": str(perturbation),
                    "perturbation_value": perturbation.value,
                }
                result.update(
                    {
                        name: value
                        for name, value in zip(
                            self.operand_names, operand_values, strict=False
                        )
                    }
                )
                results.append(result)

        self._results = pd.DataFrame(results)
        self.tolerancing.reset()

    def get_results(self) -> pd.DataFrame:
        """Return the results DataFrame."""
        return self._results

    def view(
        self,
        figsize: tuple[float, float] = (2.5, 3.3),
        sharex: Literal["none", "all", "row", "col"] | bool = "col",
        sharey: Literal["none", "all", "row", "col"] | bool = "row",
    ) -> tuple[Figure, list[Axes]]:
        """Visualize sensitivity results as a grid of subplots.

        Returns:
            Tuple of (figure, flat list of axes).
        """
        df = self._results
        unique_types = df["perturbation_type"].unique()

        m = len(self.operand_names)
        n = len(unique_types)

        size_x = m * figsize[0]
        size_y = n * figsize[1]

        fig, axes = plt.subplots(
            m, n, figsize=(size_y, size_x), sharex=sharex, sharey=sharey
        )
        axes = np.array(axes).reshape(m, n)

        for i, name in enumerate(self.operand_names):
            for j, pert_type in enumerate(unique_types):
                x = df.loc[
                    df.perturbation_type == pert_type, "perturbation_value"
                ].values
                y = df.loc[df.perturbation_type == pert_type, name].values

                axes[i, j].plot(
                    be.to_numpy(x), be.to_numpy(y), color=f"C{i}", linewidth=2
                )
                axes[i, j].grid()

                if j == 0:
                    axes[i, j].set_ylabel(name)
                if i == m - 1:
                    axes[i, j].set_xlabel(pert_type)

        fig.tight_layout()
        return fig, fig.get_axes()

    def _validate(self) -> None:
        if not self.tolerancing.operands:
            raise ValueError("No operands found in the tolerancing system.")
        if not self.tolerancing.perturbations:
            raise ValueError("No perturbations found in the tolerancing system.")
