"""Tolerance Module

This module contains the Tolerance class: a signed deviation from nominal,
bounded by a lower and an upper limit and drawn from a distribution. Unlike a
``Perturbation``, which sets absolute values of an optimization variable, a
tolerance is defined the way a drawing or a shop states it, e.g. ``±0.05``.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optiland.tolerancing.base import BasePerturbation, Phase
from optiland.tolerancing.distributions import Distribution, get_distribution

if TYPE_CHECKING:
    from optiland.tolerancing.registry import ToleranceType


class Tolerance(BasePerturbation):
    """A signed deviation from nominal between two limits.

    Args:
        tolerance_type: The bound tolerance type that applies the deviation.
        minus: The lower limit (signed, usually negative).
        plus: The upper limit (signed, usually positive).
        distribution: The distribution used by Monte Carlo analyses, as a
            name or a ``Distribution`` instance. Defaults to ``"normal"``,
            a normal distribution truncated at the limits, with the limits at
            ±2σ.

    Attributes:
        type: The bound tolerance type.
        minus: The lower limit.
        plus: The upper limit.
        distribution: The distribution instance.
        value: The deviation applied last, or 0.0 at nominal.

    Raises:
        ValueError: If ``minus`` is greater than ``plus``.
    """

    def __init__(
        self,
        tolerance_type: ToleranceType,
        minus: float,
        plus: float,
        distribution: str | Distribution = "normal",
    ) -> None:
        if minus > plus:
            raise ValueError(f"minus ({minus}) must not exceed plus ({plus}).")
        self.type = tolerance_type
        self.minus = float(minus)
        self.plus = float(plus)
        self.distribution = get_distribution(distribution)
        self.rng = np.random.default_rng()
        self.value = 0.0

    @property
    def phase(self) -> Phase:
        """Return the phase of the underlying tolerance type."""
        return self.type.phase

    @property
    def label(self) -> str:
        """Return a short, human-readable name."""
        return self.type.label

    @property
    def limits(self) -> tuple[float, float]:
        """Return ``(minus, plus)``."""
        return self.minus, self.plus

    def apply(self, rng: np.random.Generator | None = None) -> None:
        """Draw a deviation from the distribution and apply it.

        Args:
            rng: Generator to draw from. Defaults to the tolerance's own.
        """
        rng = self.rng if rng is None else rng
        self.apply_value(self.distribution.sample(rng, self.minus, self.plus))

    def apply_value(self, delta: float) -> None:
        """Apply a given deviation from nominal.

        Args:
            delta: The signed deviation, in the units of the tolerance type.
        """
        self.type.apply(delta)
        self.value = float(delta)

    def reset(self) -> None:
        """Restore the nominal state."""
        self.type.reset()
        self.value = 0.0

    def __str__(self) -> str:
        units = f" {self.type.units}" if self.type.units else ""
        return f"{self.label}: [{self.minus:+g}, {self.plus:+g}]{units}"
