"""Monte Carlo analysis for thin film stacks.

Applies all perturbations simultaneously per iteration and collects
statistics, following the pattern of ``optiland.tolerancing.monte_carlo``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .sensitivity_analysis import ThinFilmSensitivityAnalysis

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from .core import ThinFilmTolerancing


class ThinFilmMonteCarlo(ThinFilmSensitivityAnalysis):
    """Monte Carlo simulation for a thin-film tolerancing setup.

    Args:
        tolerancing: A configured ``ThinFilmTolerancing`` instance.
    """

    def __init__(self, tolerancing: ThinFilmTolerancing):
        super().__init__(tolerancing)

    def run(self, num_iterations: int) -> None:  # type: ignore[override]
        """Run the Monte Carlo simulation.

        All perturbations are applied simultaneously in each iteration.

        Args:
            num_iterations: Number of Monte Carlo trials.
        """
        results: list[dict] = []

        for _ in range(num_iterations):
            self.tolerancing.reset()

            for perturbation in self.tolerancing.perturbations:
                perturbation.apply()

            operand_values = self.tolerancing.evaluate()

            result: dict = {}
            for perturbation in self.tolerancing.perturbations:
                key = str(perturbation)
                result[key] = float(perturbation.value)

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

    def view_histogram(self, kde: bool = True) -> tuple[Figure, NDArray[np.object_]]:
        """Display histograms of operand distributions."""
        return self._plot(plot_type="histogram", kde=kde)

    def view_cdf(self) -> tuple[Figure, NDArray[np.object_]]:
        """Display CDF plots of operand distributions."""
        return self._plot(plot_type="cdf")

    def view_heatmap(
        self,
        figsize: tuple[float, float] = (8, 6),
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> tuple[Figure, Axes]:
        """Display a correlation heatmap of the results."""
        df = self._results
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=figsize)
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={"shrink": 0.5},
            ax=ax,
        )
        fig.tight_layout()
        return fig, ax

    def _plot(
        self, plot_type: str, kde: bool = True
    ) -> tuple[Figure, NDArray[np.object_]]:
        num = len(self.operand_names)
        cols = 3
        rows = (num + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = np.atleast_1d(axes).flatten()

        colors = sns.color_palette("viridis", num)

        df = self._results
        for i in range(num):
            key = self.operand_names[i]
            if plot_type == "histogram":
                if kde:
                    sns.kdeplot(
                        df[key], ax=axes[i], color=colors[i], fill=True, alpha=0.3
                    )
                else:
                    sns.histplot(
                        df[key], kde=False, ax=axes[i], color=colors[i], alpha=0.5
                    )
            elif plot_type == "cdf":
                data = df[key]
                data_sorted = np.sort(data)
                cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
                axes[i].plot(data_sorted, cdf, color=colors[i])
                axes[i].fill_between(data_sorted, 0, cdf, color=colors[i], alpha=0.3)
                axes[i].grid()
                axes[i].set_xlabel(key)
                axes[i].set_title(key)
            else:
                raise ValueError(f"Invalid plot type: {plot_type}")

        for j in range(num, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        return fig, axes

    def _validate(self) -> None:
        if not self.tolerancing.operands:
            raise ValueError("No operands found in the tolerancing system.")
        if not self.tolerancing.perturbations:
            raise ValueError("No perturbations found in the tolerancing system.")
