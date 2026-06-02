"""Distribution Module

This module provides various classes representing 2D pupil distributions.

Kramer Harrison, 2024
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland._types import BEArray, DistributionType


class BaseDistribution(ABC):
    """Base class for distributions.

    This class provides a base implementation for generating points and
        visualizing the distribution.

    Attributes:
        x: The x-coordinates of the generated points.
        y: The y-coordinates of the generated points.

    """

    def __init__(self):
        self.x: BEArray = be.empty(0)
        self.y: BEArray = be.empty(0)

    @abstractmethod
    def generate_points(self, num_points: int):
        """Generate points based on the distribution.

        Args:
            num_points (int): The number of points to generate.

        """
        # pragma: no cover

    def view(self) -> tuple[Figure, Axes]:
        """Visualize the distribution.

        This method plots the distribution points and a unit circle for
        reference.
        Returns:
            A tuple containing the figure and axes of the plot.
        """
        fig, ax = plt.subplots()
        ax.plot(be.to_numpy(self.x), be.to_numpy(self.y), "k*")
        t = np.linspace(0, 2 * be.pi, 256)
        x, y = np.cos(t), np.sin(t)
        ax.plot(x, y, "r")
        ax.set_xlabel("Normalized Pupil Coordinate X")
        ax.set_ylabel("Normalized Pupil Coordinate Y")
        ax.axis("equal")
        return fig, ax


class LineXDistribution(BaseDistribution):
    """A class representing a line distribution along the x-axis.

    Generates `num_points` along the x-axis.

    Attributes:
        positive_only (bool): Flag indicating whether the distribution should
            be limited to positive values only.
        x: The x-coordinates of the generated points.
        y: The y-coordinates of the generated points.

    """

    def __init__(self, positive_only: bool = False):
        self.positive_only = positive_only

    def generate_points(self, num_points: int):
        """Generates points along the x-axis based on the specified parameters.

        Args:
            num_points (int): The number of points to generate.

        """
        if self.positive_only:
            self.x = be.linspace(0, 1, num_points)
        else:
            self.x = be.linspace(-1, 1, num_points)
        self.y = be.zeros([num_points])


class LineYDistribution(BaseDistribution):
    """A class representing a line distribution along the y-axis.

    Generates `num_points` along the y-axis.

    Attributes:
        positive_only: Flag indicating whether the distribution should
            be positive-only.
        x: The x-coordinates of the generated points.
        y: The y-coordinates of the generated points.

    """

    def __init__(self, positive_only: bool = False):
        self.positive_only = positive_only

    def generate_points(self, num_points: int):
        """Generates points along the line distribution.

        Args:
            num_points (int): The number of points to generate.

        """
        self.x = be.zeros([num_points])
        if self.positive_only:
            self.y = be.linspace(0, 1, num_points)
        else:
            self.y = be.linspace(-1, 1, num_points)


class RandomDistribution(BaseDistribution):
    """A class representing a random distribution.

    Generates `num_points` random points within the unit disk.

    Attributes:
        rng (be.Generator): The random number generator from the backend.
        x: The x-coordinates of the generated points.
        y: The y-coordinates of the generated points.

    """

    def __init__(self, seed=None):
        self.rng = be.default_rng(seed)

    def generate_points(self, num_points: int):
        """Generates random points.

        Args:
            num_points (int): The number of points to generate.

        """
        r = be.random_uniform(size=num_points, generator=self.rng)
        theta = be.random_uniform(0, 2 * be.pi, size=num_points, generator=self.rng)

        self.x = be.sqrt(r) * be.cos(theta)
        self.y = be.sqrt(r) * be.sin(theta)


class UniformDistribution(BaseDistribution):
    """Represents a uniform distribution of points within a square, which is
        masked to the unit disk.

    Generates points on a square grid of `num_points` x `num_points` and then
    masks them to the unit disk. The resulting number of points is
    approximately `num_points^2 * pi / 4`.

    Attributes:
        x: The x-coordinates of the generated points.
        y: The y-coordinates of the generated points.

    """

    def generate_points(self, num_points: int):
        """Generates a grid of points within the unit disk.

        Args:
            num_points (int): The number of points along each axis to generate.

        """
        x = be.linspace(-1, 1, num_points)
        x, y = be.meshgrid(x, x)
        r2 = x**2 + y**2
        self.x = x[r2 <= 1]
        self.y = y[r2 <= 1]


class HexagonalDistribution(BaseDistribution):
    """A class representing a hexagonal distribution.

    Generates points in a hexagonal pattern. The total number of points is
    `1 + 3 * num_rings * (num_rings + 1)`, including the center point.

    Attributes:
        x: Array of x-coordinates of the generated points.
        y: Array of y-coordinates of the generated points.

    """

    def generate_points(self, num_rings: int = 6):
        """Generate points in a hexagonal distribution.

        Args:
            num_rings: Number of rings in the hexagonal distribution.
                Defaults to 6.

        """
        x = be.zeros([1])
        y = be.zeros([1])
        r = be.linspace(0, 1, num_rings + 1)

        for i in range(num_rings):
            num_theta = 6 * (i + 1)
            theta = be.linspace(0, 2 * be.pi, num_theta + 1)[:-1]
            x = be.concatenate([x, r[i + 1] * be.cos(theta)])
            y = be.concatenate([y, r[i + 1] * be.sin(theta)])

        self.x = x
        self.y = y


class CrossDistribution(BaseDistribution):
    """A class representing a cross-shaped distribution.

    This distribution generates points in the shape of a cross,
        with the x-axis and y-axis as the arms of the cross.
    If `num_points` is odd, it generates `2 * num_points - 1` points.
    If `num_points` is even and positive, it generates `2 * num_points` points.
    `num_points` represents the number of points along the full extent of each
    axis before potential origin merging. If `num_points` is 0, 0 points are
    generated.

    Attributes:
        x: Array of x-coordinates of the generated points.
        y: Array of y-coordinates of the generated points.

    """

    def generate_points(self, num_points: int):
        """Generate points in the shape of a cross.

        Args:
            num_points: The number of points to generate in each axis.

        """
        # Generate points for the y-axis (vertical line)
        y_line_x = be.zeros([num_points])
        y_line_y = be.linspace(-1, 1, num_points)

        # Generate points for the x-axis (horizontal line)
        x_line_x = be.linspace(-1, 1, num_points)
        x_line_y = be.zeros([num_points])

        # If num_points is odd, linspace(-1, 1, num_points) includes 0 at the midpoint.
        # This means (0,0) is part of y_line (x=0, y=0) and x_line (x=0, y=0).
        # To avoid duplication, we remove the (0,0) point from the x_line set.
        if num_points % 2 == 1:
            mid_idx = num_points // 2
            # Remove the middle element which corresponds to (0,0) for the x_line
            x_line_x = be.concatenate((x_line_x[:mid_idx], x_line_x[mid_idx + 1 :]))
            x_line_y = be.concatenate((x_line_y[:mid_idx], x_line_y[mid_idx + 1 :]))

        self.x = be.concatenate((y_line_x, x_line_x))
        self.y = be.concatenate((y_line_y, x_line_y))


class GaussianQuadrature(BaseDistribution):
    """A class for Gaussian quadrature on circular domains, based on _[1].

    Generates points in a circular pattern, with optimal placement for Gaussian
    quadrature over the unit disk. The total number of points is `num_rings *
    num_spokes`.

    Attributes:
        x: Array of x-coordinates of the generated points.
        y: Array of y-coordinates of the generated points.
        weights: Array of weights, normalized to 1.0.

    .. [1] William H. Peirce, "Numerical Integration Over the Planar Annulus,",  Journal
            of the Society for Industrial and Applied Mathematics, Vol. 5, No. 2 (Jun.,
            1957), pp. 66-73
    """

    def generate_points(self, num_rings: int, num_spokes: int | None = None):
        """Generate radially symmetric points.
        Args:
            num_rings (int): Number of rings.
            num_angles (int | None) : Number of spokes, by default None. If None, the
                number of spokes is `4 * (num_rings + 1)`. In that case, the integration
                over the unit disk is exact for polynomials of degree `num_rings` in x
                and y.
        """
        from scipy.special import roots_legendre

        if num_rings < 1 or num_spokes is not None and num_spokes < 1:
            raise ValueError("The number of ring or spokes has to be ≥ 1")

        k = 4 * num_rings + 3 if num_spokes is None else num_spokes - 1
        theta_i = 2 * be.pi / (k + 1) * be.arange(1, k + 2)

        xi, wi = roots_legendre(num_rings)
        xi = be.array(xi)
        wi = be.array(wi)

        ri = (0.5 + 0.5 * xi) ** 0.5
        wi = 0.5 * wi / (k + 1)

        self.weights = be.tile(wi, k + 1)

        ri, theta_i = be.meshgrid(ri, theta_i)
        self.x = (ri * be.cos(theta_i)).flatten()
        self.y = (ri * be.sin(theta_i)).flatten()


class RingDistribution(BaseDistribution):
    """RingDistribution class for generating points along a single ring.

    Generates `num_points` along a single ring at the maximum aperture value
    (radius 1).
    """

    def generate_points(self, num_points: int):
        """Generate points along a ring at the maximum aperture value.

        Args:
            num_points (int): The number of points to generate in each ring.

        """
        theta = be.linspace(0, 2 * be.pi, num_points + 1)[:-1]

        self.x = be.cos(theta)
        self.y = be.sin(theta)


class SobolDistribution(BaseDistribution):
    """A class representing a Sobol distribution.

    Generates `num_points` points using a Sobol low-discrepancy sequence
    within the unit disk.

    Attributes:
        seed (int | None): Seed for the Sobol sequence generator.
        x: The x-coordinates of the generated points.
        y: The y-coordinates of the generated points.
    """

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.seed = seed

    def generate_points(self, num_points: int):
        """Generates Sobol points.

        Args:
            num_points (int): The number of points to generate.

        """
        sample = be.sobol_sampler(
            dim=2, num_samples=num_points, scramble=True, seed=self.seed
        )

        u1 = sample[:, 0]
        u2 = sample[:, 1]

        r = be.sqrt(u1)
        theta = 2 * be.pi * u2

        self.x = r * be.cos(theta)
        self.y = r * be.sin(theta)


def create_distribution(distribution_type: DistributionType) -> BaseDistribution:
    """Create a distribution based on the given distribution type.

    Args:
        distribution_type: The type of distribution to create.

    Returns:
        An instance of the specified distribution type.

    Raises:
        ValueError: If an invalid distribution type is provided.

    """
    distribution_classes: dict[
        DistributionType, type[BaseDistribution] | Callable[[], BaseDistribution]
    ] = {
        "line_x": LineXDistribution,
        "line_y": LineYDistribution,
        "positive_line_x": lambda: LineXDistribution(positive_only=True),
        "positive_line_y": lambda: LineYDistribution(positive_only=True),
        "random": RandomDistribution,
        "uniform": UniformDistribution,
        "hexapolar": HexagonalDistribution,
        "cross": CrossDistribution,
        "ring": RingDistribution,
        "sobol": SobolDistribution,
    }

    if distribution_type not in distribution_classes:
        raise ValueError("Invalid distribution type.")

    return distribution_classes[distribution_type]()
