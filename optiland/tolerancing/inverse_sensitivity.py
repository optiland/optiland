"""Inverse Sensitivity Module

This module contains the InverseSensitivity class, which computes, for each
tolerance, the largest deviation from nominal on each side that keeps chosen
operands within an allowed change from their compensated nominal values. It is
the inverse of ``SensitivityAnalysis``: instead of reporting the operand change
at given limits, it reports the limits for a given operand change.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from optiland.tolerancing.tolerance import Tolerance

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland.tolerancing.core import Tolerancing


class InverseSensitivity:
    """Find the largest tolerance limits that keep operands within bounds.

    For each ``Tolerance`` and each side (negative deviations for ``minus``,
    positive deviations for ``plus``), the search applies the deviation alone,
    applies compensators, evaluates the operands and finds the largest
    deviation for which every operand in ``max_change`` changes by at most its
    allowed amount from the compensated nominal.

    Search, per side, with ``x`` the deviation magnitude and ``x0`` the
    magnitude of the current limit on that side (or the larger of the two
    limits when that side is zero):

    1. Define ``g(x) = max_k |op_k(x) - op_k(0)| / max_change[k] - 1``, so
       ``g(0) = -1`` and ``g(x) <= 0`` means every operand is within bounds.
    2. Bracket: if ``g(x0) > 0`` the bracket is ``[0, x0]``. Otherwise ``x``
       doubles from ``x0`` until ``g(x) > 0``. If ``g`` is still within bounds
       at ``max_factor * x0``, the search stops there and the side is reported
       as capped.
    3. Refine the crossing inside the bracket with Brent's method
       (``scipy.optimize.brentq``), to within ``xtol = rtol * x0``, with at most
       ``max_iter`` iterations.

    The response is not assumed monotonic beyond the bracket: the result is the
    first crossing found as described, so a response that leaves and re-enters
    the allowed band between bracket points can yield a smaller or larger limit
    than the true largest one. Only one tolerance is applied at a time; other
    tolerances and legacy perturbations stay at nominal.

    Legacy perturbations (added with ``Tolerancing.add_perturbation``) have no
    deviation limits and are skipped.

    Args:
        tolerancing: The tolerancing system.
        max_change: Maps an operand index (in ``tolerancing.operands``) to the
            allowed absolute change from the compensated nominal value.
            Operands not listed are ignored.
        max_factor: Largest searched deviation, as a multiple of ``x0``.
            Defaults to 100.
        rtol: Search tolerance on the limit, relative to ``x0``. Defaults to
            1e-4.
        max_iter: Iteration cap for the Brent refinement. Defaults to 50.

    Attributes:
        tolerancing: The tolerancing system.
        max_change: The allowed operand changes.
        operand_names: List of operand names in the tolerancing system.
        nominal: Operand values at nominal, after compensation, from the last
            completed run.

    Raises:
        ValueError: If there are no operands or tolerances, if ``max_change``
            is empty, names an unknown operand or holds a non-positive value,
            if ``max_factor`` is not greater than 1, or if a tolerance has two
            zero limits.
    """

    def __init__(
        self,
        tolerancing: Tolerancing,
        max_change: dict[int, float],
        max_factor: float = 100.0,
        rtol: float = 1e-4,
        max_iter: int = 50,
    ) -> None:
        self.tolerancing = tolerancing
        self.max_change = {int(k): float(v) for k, v in max_change.items()}
        self.max_factor = float(max_factor)
        self.rtol = float(rtol)
        self.max_iter = int(max_iter)
        self.operand_names = [
            f"{i}: {operand}" for i, operand in enumerate(tolerancing.operands)
        ]
        self.nominal: list[float] = []
        self._tolerances = [
            p for p in tolerancing.perturbations if isinstance(p, Tolerance)
        ]
        self._results = pd.DataFrame()
        self._validate()

    def run(self) -> None:
        """Run the inverse sensitivity search for every tolerance.

        The tolerancing system is reset to nominal on exit, including when a
        trial raises an exception or is interrupted. If a run fails, results
        and ``nominal`` from the last completed run remain available and the
        exception propagates.
        """
        rows = []
        try:
            nominal = self._trial(None, 0.0)
            for tolerance in self._tolerances:
                row = {
                    "tolerance": tolerance.label,
                    "minus": tolerance.minus,
                    "plus": tolerance.plus,
                }
                scale = max(abs(tolerance.minus), abs(tolerance.plus))
                for side, sign in (("minus", -1.0), ("plus", 1.0)):
                    x0 = abs(getattr(tolerance, side)) or scale
                    x, limiting, capped = self._search(tolerance, sign, x0, nominal)
                    row[f"inverse_{side}"] = sign * x
                    row[f"limiting_{side}"] = limiting
                    row[f"capped_{side}"] = capped
                rows.append(row)
            self.nominal = nominal
            self._results = pd.DataFrame(rows)
        finally:
            self.tolerancing.reset()

    def _trial(self, tolerance: Tolerance | None, delta: float) -> list[float]:
        """Apply one deviation from nominal, compensate and evaluate."""
        self.tolerancing.reset()
        if tolerance is not None:
            tolerance.apply_value(delta)
        self.tolerancing.apply_compensators()
        return self.tolerancing.evaluate()

    def _search(
        self, tolerance: Tolerance, sign: float, x0: float, nominal: list[float]
    ) -> tuple[float, str | None, bool]:
        """Return the deviation magnitude, limiting operand and capped flag."""
        ratios: dict[float, list[float]] = {}

        def g(x: float) -> float:
            if x not in ratios:
                values = self._trial(tolerance, sign * x)
                ratios[x] = [
                    abs(values[k] - nominal[k]) / limit
                    for k, limit in self.max_change.items()
                ]
            return max(ratios[x]) - 1.0

        cap = self.max_factor * x0
        lo, hi = 0.0, x0
        if g(x0) <= 0.0:
            lo = x0
            while True:
                hi = min(2.0 * lo, cap)
                if g(hi) > 0.0:
                    break
                if hi >= cap:
                    return cap, None, True
                lo = hi

        x = brentq(g, lo, hi, xtol=self.rtol * x0, maxiter=self.max_iter)
        g(x)
        # the operand closest to (or furthest past) its bound at the result
        operand = list(self.max_change)[int(np.argmax(ratios[x]))]
        return float(x), self.operand_names[operand], False

    def get_results(self) -> pd.DataFrame:
        """Return the results of the last completed run.

        Returns:
            pd.DataFrame: One row per tolerance with columns ``tolerance``,
            ``minus``, ``plus`` (current limits), ``inverse_minus``,
            ``inverse_plus`` (limits found), ``limiting_minus``,
            ``limiting_plus`` (name of the operand that sets each limit, or
            None when capped) and ``capped_minus``, ``capped_plus`` (True when
            the search stopped at ``max_factor`` times the start).
        """
        return self._results

    def apply_limits(self) -> None:
        """Write the inverse limits from the last run into the tolerances.

        Raises:
            RuntimeError: If no run has completed.
        """
        if self._results.empty:
            raise RuntimeError("Run the inverse sensitivity analysis first.")
        for tolerance, (_, row) in zip(
            self._tolerances, self._results.iterrows(), strict=True
        ):
            tolerance.minus = float(row["inverse_minus"])
            tolerance.plus = float(row["inverse_plus"])

    def view(self, figsize: tuple[float, float] | None = None) -> tuple[Figure, Axes]:
        """Plot current and inverse limits as horizontal bars per tolerance.

        Args:
            figsize: Figure size in inches. Defaults to a height that grows
                with the number of tolerances.

        Returns:
            tuple: The figure and axes of the plot.
        """
        df = self._results
        n = len(df)
        figsize = figsize or (7.0, 1.0 + 0.6 * n)
        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(n)
        height = 0.38
        for offset, (prefix, name, color) in enumerate(
            (("", "Current", "C7"), ("inverse_", "Inverse", "C0"))
        ):
            pos = y + (offset - 0.5) * height
            for j, side in enumerate(("minus", "plus")):
                ax.barh(
                    pos,
                    df[f"{prefix}{side}"].to_numpy(dtype=float),
                    height=height,
                    color=color,
                    label=name if j == 0 else None,
                )
        ax.axvline(0.0, color="k", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{i}: {label}" for i, label in enumerate(df.tolerance)])
        ax.invert_yaxis()
        ax.set_xlabel("Deviation from nominal")
        ax.grid(axis="x")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def _validate(self) -> None:
        """Check the tolerancing system and search settings."""
        n = len(self.tolerancing.operands)
        if n == 0:
            raise ValueError("No operands found in the tolerancing system.")
        if not self._tolerances:
            raise ValueError(
                "No tolerances found in the tolerancing system; add them with "
                "Tolerancing.add_tolerance (legacy perturbations are skipped)."
            )
        if not self.max_change:
            raise ValueError("max_change must name at least one operand.")
        for k, limit in self.max_change.items():
            if not 0 <= k < n:
                raise ValueError(f"Operand index {k} out of range (0 to {n - 1}).")
            if not limit > 0:
                raise ValueError(f"max_change[{k}] must be positive, got {limit}.")
        if not self.max_factor > 1:
            raise ValueError(f"max_factor must exceed 1, got {self.max_factor}.")
        for tolerance in self._tolerances:
            if tolerance.minus == 0 and tolerance.plus == 0:
                raise ValueError(f"Tolerance {tolerance.label} has zero limits.")
