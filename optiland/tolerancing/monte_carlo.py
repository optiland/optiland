"""Monte Carlo Module

This module contains the Monte Carlo class for performing Monte Carlo analysis
on a tolerancing system. The Monte Carlo class is a subclass of the
SensitivityAnalysis class and provides methods for running Monte Carlo
simulations, visualizing the results, and analyzing the correlation between
operands.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from optiland.tolerancing.core import Tolerancing


class MonteCarlo:
    """Standalone Monte Carlo analysis for a tolerancing system.

    Unlike SensitivityAnalysis (which sweeps each perturbation in sequence),
    MonteCarlo applies all perturbations simultaneously from random samples
    on each iteration.

    Args:
        tolerancing: The tolerancing system to analyse.

    Attributes:
        tolerancing: The tolerancing system.
        operand_names: List of operand names in the tolerancing system.
        _results: DataFrame storing the Monte Carlo results.

    """

    def __init__(self, tolerancing: Tolerancing) -> None:
        self.tolerancing = tolerancing
        self.operand_names = [
            f"{i}: {operand}" for i, operand in enumerate(tolerancing.operands)
        ]
        self._results = pd.DataFrame()
        self._validate()

    def run(self, num_iterations: int):
        """Executes the Monte Carlo simulation for a specified number of
        iterations.

        Args:
            num_iterations (int): The number of iterations to run the
                simulation.

        Returns:
            None: The results are stored in the instance variable `_results`
                as a pandas DataFrame.

        The method performs the following steps for each iteration:
            1. Resets the tolerancing system.
            2. Applies perturbations to the system.
            3. Applies compensators to the system and stores the results.
            4. Evaluates the operands and stores their values.
            5. Saves the perturbation types and values, operand values, and
                compensator values in a dictionary.
            6. Appends the dictionary to the results list.
        The final results are converted to a pandas DataFrame and stored in
            the `_results` attribute.

        """
        results = []

        for _ in range(num_iterations):
            # reset the tolerancing system
            self.tolerancing.reset()

            # apply perturbations
            for perturbation in self.tolerancing.perturbations:
                perturbation.apply()

            # apply compensators
            compensator_result = self.tolerancing.apply_compensators()

            # evaluate operands
            operand_values = self.tolerancing.evaluate()

            # save results
            result = {}

            # save results - perturbation type & value
            for perturbation in self.tolerancing.perturbations:
                key = str(perturbation.variable)
                result[key] = float(perturbation.value)

            # save results - operand values
            result.update(
                {
                    f"{name}": value
                    for name, value in zip(
                        self.operand_names, operand_values, strict=False
                    )
                },
            )

            # save results - compensator values
            result.update(compensator_result)

            results.append(result)

        self._results = pd.DataFrame(results)

    def get_results(self) -> pd.DataFrame:
        """Return the Monte Carlo analysis results.

        Returns:
            pd.DataFrame: The results of the Monte Carlo simulation.

        """
        return self._results

    def view_histogram(self, kde: bool = True) -> tuple[Figure, NDArray[np.object_]]:
        """Displays a histogram of the data.

        Args:
            kde (bool): If True, a Kernel Density Estimate (KDE) is plotted.
                Otherwise, a histogram is plotted.

        """
        return self._plot(plot_type="histogram", kde=kde)

    def view_cdf(self) -> tuple[Figure, NDArray[np.object_]]:
        """Generates and displays a cumulative distribution function (CDF) plot
        of the data.
        """
        return self._plot(plot_type="cdf")

    def view_heatmap(
        self,
        figsize: tuple[float, float] = (8, 6),
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> tuple[Figure, Axes]:
        """Generates and displays a heatmap of the correlation matrix of the
        results.

        Args:
            figsize (tuple, optional): A tuple representing the size of the
                figure (width, height). Default is (8, 6).
            vmin (float, optional): Minimum value for the color scale.
                Default is None, which uses the minimum value in the data.
            vmax (float, optional): Maximum value for the color scale.
                Default is None, which uses the maximum value in the data.

        Returns:
            tuple: A tuple containing the figure and axes of the heatmap.

        """
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
        """Plot the Monte Carlo analysis results.

        Args:
            plot_type (str): The type of plot to generate. Can be 'histogram'
                or 'cdf'.
            kde (bool, optional): If True, plot a Kernel Density Estimate
                (KDE) for histograms. Default is True.
        Returns:
            tuple: A tuple containing the figure and axes of the plot.
        Raises:
            ValueError: If an invalid plot type is provided.

        """
        num = len(self.operand_names)
        cols = 3
        rows = (num + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
        axes = axes.flatten()

        colors = sns.color_palette("viridis", len(self.operand_names))

        df = self._results
        for i in range(num):
            key = self.operand_names[i]
            if plot_type == "histogram":
                if kde:
                    sns.kdeplot(
                        df[key],
                        ax=axes[i],
                        color=colors[i],
                        fill=True,
                        alpha=0.3,
                    )
                else:
                    sns.histplot(
                        df[key],
                        kde=False,
                        ax=axes[i],
                        color=colors[i],
                        alpha=0.5,
                    )

            elif plot_type == "cdf":
                data = df[key]
                data_sorted = np.sort(data)
                cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
                axes[i].plot(data_sorted, cdf, color=colors[i])
                axes[i].fill_between(data_sorted, 0, cdf, color=colors[i], alpha=0.3)
                axes[i].grid()
                axes[i].set_xlim([None, data_sorted[-1]])
                axes[i].set_ylim([0, None])
                axes[i].set_xlabel(key)
                axes[i].set_title(key)

            else:
                raise ValueError(f"Invalid plot type: {plot_type}")

        for j in range(num, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        return fig, axes

    def _validate(self):
        """Validates the tolerancing system before performing Monte Carlo
        analysis.

        Raises:
            ValueError: If no operands are found in the tolerancing system.
            ValueError: If no perturbations are found in the tolerancing
                system.

        """
        if not self.tolerancing.operands:
            raise ValueError("No operands found in the tolerancing system.")

        if not self.tolerancing.perturbations:
            raise ValueError("No perturbations found in the tolerancing system.")
