"""Tolerance Distributions Module

This module contains the statistical distributions used to draw tolerance
deviations between a lower and an upper limit in Monte Carlo analyses.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Distribution(ABC):
    """Base class for distributions of a deviation between two limits.

    Sampling always runs on a NumPy generator, independent of the active
    backend, so that results are reproducible across backends.
    """

    @abstractmethod
    def sample(self, rng: np.random.Generator, minus: float, plus: float) -> float:
        """Draw one deviation.

        Args:
            rng: The random number generator.
            minus: The lower limit (signed).
            plus: The upper limit (signed).

        Returns:
            A deviation in ``[minus, plus]``.
        """


class TruncatedNormal(Distribution):
    """Normal distribution truncated at the limits.

    The mean is the midpoint of the limits and the standard deviation is chosen
    so that each limit lies ``num_sigma`` standard deviations from the mean.

    Args:
        num_sigma: Number of standard deviations between the mean and each
            limit. Defaults to 2.
    """

    def __init__(self, num_sigma: float = 2.0) -> None:
        if num_sigma <= 0:
            raise ValueError("num_sigma must be positive.")
        self.num_sigma = num_sigma

    def sample(self, rng: np.random.Generator, minus: float, plus: float) -> float:
        """Draw one deviation by rejection sampling."""
        center = 0.5 * (minus + plus)
        half_width = 0.5 * (plus - minus)
        if half_width == 0:
            return float(center)
        while True:
            z = rng.standard_normal()
            if abs(z) <= self.num_sigma:
                return float(center + z * half_width / self.num_sigma)


class Uniform(Distribution):
    """Uniform distribution between the limits."""

    def sample(self, rng: np.random.Generator, minus: float, plus: float) -> float:
        """Draw one deviation."""
        return float(rng.uniform(minus, plus))


class Parabolic(Distribution):
    """Distribution with density proportional to the squared distance from the
    midpoint, so that values near the limits are the most likely."""

    def sample(self, rng: np.random.Generator, minus: float, plus: float) -> float:
        """Draw one deviation by inverting the cumulative distribution."""
        u = float(np.cbrt(rng.uniform(-1.0, 1.0)))
        return float(0.5 * (minus + plus) + u * 0.5 * (plus - minus))


class EndPoint(Distribution):
    """Distribution that returns either limit with equal probability."""

    def sample(self, rng: np.random.Generator, minus: float, plus: float) -> float:
        """Draw one deviation."""
        return float(plus if rng.random() < 0.5 else minus)


_DISTRIBUTIONS: dict[str, type[Distribution]] = {
    "normal": TruncatedNormal,
    "uniform": Uniform,
    "parabolic": Parabolic,
    "end_point": EndPoint,
}


def get_distribution(distribution: str | Distribution) -> Distribution:
    """Return a distribution instance from a name or an instance.

    Args:
        distribution: One of ``"normal"`` (truncated at ±2σ), ``"uniform"``,
            ``"parabolic"`` or ``"end_point"``, or a ``Distribution`` instance.

    Returns:
        The distribution instance.

    Raises:
        ValueError: If the name is unknown.
    """
    if isinstance(distribution, Distribution):
        return distribution
    try:
        return _DISTRIBUTIONS[distribution]()
    except KeyError:
        valid = ", ".join(sorted(_DISTRIBUTIONS))
        raise ValueError(
            f"Unknown distribution {distribution!r}. Valid options: {valid}."
        ) from None
