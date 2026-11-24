"""Sensitivity Analysis Module

This module contains the SensitivityAnalysis class for performing sensitivity
analysis on a tolerancing system. The SensitivityAnalysis class allows users to
run a sensitivity analysis on a Tolerancing object and visualize the results.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import optiland.backend as be
from optiland.tolerancing.perturbation import BaseSampler, RangeSampler
from optiland.tolerancing.tolerance import Tolerance

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland.tolerancing.core import Tolerancing


class SensitivityAnalysis:
    """Class for performing sensitivity analysis on a tolerancing system.

    Args:
        tolerancing: The tolerancing system to perform sensitivity analysis on.
        sampler: Optional sampler strategy. Defaults to RangeSampler, which
            produces a linear sweep for each perturbation. Provide a different
            BaseSampler subclass to change the sampling behaviour.

    Attributes:
        tolerancing: The tolerancing system.
        operand_names: List of operand names in the tolerancing system.
        _results: DataFrame storing the sensitivity analysis results.

    """

    def __init__(
        self,
        tolerancing: Tolerancing,
        sampler: BaseSampler | None = None,
    ) -> None:
        self.tolerancing = tolerancing
        self._sampler = sampler
        self.operand_names = [
            f"{i}: {operand}" for i, operand in enumerate(tolerancing.operands)
        ]
        self._results = pd.DataFrame()
        self._nominal: list[float] = []
        self._row_owner: list[int] = []
        self._validate()

    def run(self):
        """Run the sensitivity analysis.

        This method performs a sensitivity analysis by iterating over the
        perturbations defined in the tolerancing object. For each
        perturbation, it applies the perturbation, applies compensators,
        evaluates operands, and saves the results. The results are stored in a
        pandas DataFrame in the _results attribute.

        Registered perturbations and compensators are reset to their nominal
        values on exit, including when a trial raises an exception or is
        interrupted. Results retain the trial values. If a run fails before
        publishing its DataFrame, results from the last completed run remain
        available and the exception propagates. Samplers are not rewound.

        Tolerances (added with ``Tolerancing.add_tolerance``) are evaluated at
        their lower and upper limits, one row each.

        After the trials, the operands are evaluated once at nominal, with
        compensators applied. ``rss_estimate`` and ``view_tornado`` report
        changes relative to these values.

        Raises:
            ValueError: If a perturbation sampler other than RangeSampler is
                used.

        """
        results = []
        row_owner = []

        try:
            for index, perturbation in enumerate(self.tolerancing.perturbations):
                for apply in self._trial_steps(perturbation):
                    # reset system
                    self.tolerancing.reset()

                    # apply perturbation
                    apply()

                    # apply compensators & save results
                    compensator_result = self.tolerancing.apply_compensators()

                    # evaluate operands
                    operand_values = self.tolerancing.evaluate()

                    # save results - perturbation type & value
                    result = {
                        "perturbation_type": perturbation.label,
                        "perturbation_value": perturbation.value,
                    }

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
                    row_owner.append(index)

            # nominal operand values, compensated like every trial
            self.tolerancing.reset()
            self.tolerancing.apply_compensators()
            nominal = self.tolerancing.evaluate()

            self._results = pd.DataFrame(results)
            self._nominal = nominal
            self._row_owner = row_owner
        finally:
            self.tolerancing.reset()

    @staticmethod
    def _trial_steps(perturbation):
        """Return one callable per trial that applies the perturbation.

        Raises:
            ValueError: If a legacy perturbation does not use a RangeSampler.
        """
        if isinstance(perturbation, Tolerance):
            return [
                lambda value=value: perturbation.apply_value(value)
                for value in perturbation.limits
            ]
        if not isinstance(perturbation.sampler, RangeSampler):
            raise ValueError("Only range samplers are supported.")
        return [perturbation.apply] * perturbation.sampler.size

    def get_results(self):
        """Returns the results of the sensitivity analysis.

        Returns:
            pd.DataFrame: The results of the sensitivity analysis.

        """
        return self._results

    def rss_estimate(self) -> pd.DataFrame:
        """Estimate the combined effect of all perturbations by RSS.

        For each perturbation, the change of an operand is the largest
        absolute change from nominal over its trials (for a tolerance, the
        worse of its two limits). The estimated change is the root sum of
        squares of these changes, which assumes small, independent
        perturbations. A NaN trial makes the estimate NaN.

        Returns:
            One row per operand, with columns ``nominal`` (the compensated
            nominal value), ``rss_change`` and ``estimate``
            (``nominal + rss_change``).

        Raises:
            RuntimeError: If ``run`` has not been called.
        """
        changes = self._worst_changes()
        rss = np.sqrt((changes**2).sum(axis=0, skipna=False))
        nominal = pd.Series(self._nominal, index=self.operand_names)
        return pd.DataFrame(
            {"nominal": nominal, "rss_change": rss, "estimate": nominal + rss}
        )

    def view_tornado(
        self,
        operand_index: int = 0,
        top: int | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot a tornado chart of the operand changes per perturbation.

        Each perturbation gets two bars: the change from nominal at its lower
        and at its upper end (the limits of a tolerance, or the first and last
        value of a sweep). Perturbations are sorted by the larger change, with
        the largest on top.

        Args:
            operand_index: Index of the operand to plot. Defaults to 0.
            top: Plot only this many perturbations, the largest first.
                Defaults to all.
            figsize: Figure size in inches. Defaults to a height that grows
                with the number of perturbations.

        Returns:
            The figure and axes of the plot.

        Raises:
            RuntimeError: If ``run`` has not been called.
            IndexError: If ``operand_index`` is out of range.
        """
        self._require_results()
        name = self.operand_names[operand_index]
        delta = self._results[name].to_numpy(dtype=float) - self._nominal[operand_index]
        owner = np.asarray(self._row_owner)

        rows = []
        for index, perturbation in enumerate(self.tolerancing.perturbations):
            d = delta[owner == index]
            rows.append((perturbation.label, d[0], d[-1]))
        rows.sort(key=lambda r: np.nan_to_num(max(abs(r[1]), abs(r[2]))))
        if top is not None:
            rows = rows[-top:]

        if figsize is None:
            figsize = (6.0, 1.0 + 0.45 * len(rows))
        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(len(rows))
        # side by side, so a bar is not hidden when both ends have one sign
        ax.barh(y - 0.2, [r[1] for r in rows], 0.4, color="C0", label="lower end")
        ax.barh(y + 0.2, [r[2] for r in rows], 0.4, color="C1", label="upper end")
        ax.axvline(0.0, color="k", linewidth=0.8)
        ax.set_yticks(y, [r[0] for r in rows])
        ax.set_xlabel(f"Change in {name}")
        ax.grid(axis="x")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def _require_results(self) -> None:
        """Raise if there are no results to summarize."""
        if self._results.empty:
            raise RuntimeError("No results. Call run() first.")

    def _worst_changes(self) -> pd.DataFrame:
        """Return the largest absolute change per perturbation and operand.

        Returns:
            One row per perturbation and one column per operand.
        """
        self._require_results()
        values = self._results[self.operand_names].to_numpy(dtype=float)
        delta = np.abs(values - np.asarray(self._nominal, dtype=float))
        owner = np.asarray(self._row_owner)
        perturbations = self.tolerancing.perturbations
        # np.max (not nanmax) so a NaN trial propagates
        worst = [delta[owner == i].max(axis=0) for i in range(len(perturbations))]
        return pd.DataFrame(
            worst,
            index=[p.label for p in perturbations],
            columns=self.operand_names,
        )

    def view(
        self,
        figsize: tuple[float, float] = (2.5, 3.3),
        sharex: Literal["none", "all", "row", "col"] | bool = "col",
        sharey: Literal["none", "all", "row", "col"] | bool = "row",
    ) -> tuple[Figure, list[Axes]]:
        """Visualizes the sensitivity analysis results.

        Args:
            figsize (tuple, optional): The size of the figure in inches for
                each subplot. Default is (2.2, 3).
            sharex (str, optional): Specifies how the x-axis is shared among
                subplots. Default is 'col'.
            sharey (str, optional): Specifies how the y-axis is shared among
                subplots. Default is 'row'.

        Returns:
            tuple: A tuple containing the figure and axes of the plot.

        """
        df = self._results
        unique_types = df["perturbation_type"].unique()

        m = len(self.operand_names)
        n = len(unique_types)

        size_x = m * figsize[0]
        size_y = n * figsize[1]

        fig, axes = plt.subplots(
            m,
            n,
            figsize=(size_y, size_x),
            sharex=sharex,
            sharey=sharey,
        )

        # handle single row and/or column
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

    def _validate(self):
        """Validates the tolerancing system before performing sensitivity
        analysis.

        Raises:
            ValueError: If no operands are found in the tolerancing system.
            ValueError: If no perturbations are found in the tolerancing
                system.
            ValueError: If the number of operands exceeds 6.
            ValueError: If the number of perturbations exceeds 6.

        """
        if not self.tolerancing.operands:
            raise ValueError("No operands found in the tolerancing system.")

        if not self.tolerancing.perturbations:
            raise ValueError("No perturbations found in the tolerancing system.")

        if len(self.tolerancing.operands) > 6:
            raise ValueError("Sensitivity analysis is limited to 6 operands.")

        if len(self.tolerancing.perturbations) > 6:
            raise ValueError("Sensitivity analysis is limited to 6 perturbations.")
