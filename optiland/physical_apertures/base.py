"""Physical Apertures Base Module

This module contains the base classes for physical apertures. The BaseAperture
class is an abstract base class that defines the interface for physical
apertures. The BaseBooleanAperture class is an abstract base class for boolean
operations on apertures. The UnionAperture, IntersectionAperture, and
DifferenceAperture classes are concrete classes that implement the union,
intersection, and difference of two apertures, respectively.

Kramer Harrison, 2024
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

import optiland.backend as be

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from optiland._types import BEArray
    from optiland.rays import RealRays


class BaseAperture(ABC):
    """Base class for physical apertures.

    Methods:
        clip(RealRays): Clips the given rays based on the aperture's shape.

    """

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses."""
        super().__init_subclass__(**kwargs)
        BaseAperture._registry[cls.__name__] = cls

    @property
    @abstractmethod
    def extent(self) -> tuple[float, float, float, float]:
        """Returns the extent of the aperture.

        Returns:
            tuple: The extent of the aperture in the x and y directions.

        """
        # pragma: no cover

    @abstractmethod
    def contains(self, x: BEArray, y: BEArray) -> BEArray:
        """Checks if the given point is inside the aperture.

        Args:
            x (be.ndarray): The x-coordinate of the point.
            y (be.ndarray): The y-coordinate of the point.

        Returns:
            be.ndarray: Boolean array indicating if the point is inside the
                aperture

        """
        # pragma: no cover

    def clip(self, rays: RealRays):
        """Clips the given rays based on the aperture's shape.

        Args:
            rays (RealRays): List of rays to be clipped.

        Returns:
            list: List of clipped rays.

        """
        inside = self.contains(rays.x, rays.y)
        rays.clip(~inside)

    @abstractmethod
    def scale(self, scale_factor: float):
        """Scales the aperture by the given factor.

        Args:
            scale_factor (float): The factor by which to scale the aperture.

        """
        # pragma: no cover

    def to_dict(self) -> dict:
        """Convert the aperture to a dictionary.

        Returns:
            dict: The dictionary representation of the aperture.

        """
        return {"type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, data: dict) -> BaseAperture:
        """Create an aperture from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the aperture.

        Returns:
            BaseAperture: The aperture object.

        """
        aperture_type = data["type"]
        return cls._registry[aperture_type].from_dict(data)

    def view(
        self,
        nx: int = 256,
        ny: int = 256,
        ax: Axes | None = None,
        buffer: float = 1.1,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """Visualize the aperture.

        Args:
            nx (int): The number of points in the x-direction.
            ny (int): The number of points in the y-direction.
            ax (Axes): The axes to plot on.
            buffer (float): The buffer around the aperture.
            **kwargs: Additional keyword arguments to pass to the plot
                function.

        Returns:
            tuple: A tuple containing the figure and axes objects.

        """
        x_min, x_max, y_min, y_max = self.extent
        x_min = x_min * buffer
        x_max = x_max * buffer
        y_min = y_min * buffer
        y_max = y_max * buffer
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        x = be.linspace(x_min, x_max, nx)
        y = be.linspace(y_min, y_max, ny)
        X, Y = be.meshgrid(x, y)
        Z = self.contains(X, Y)

        ax.contourf(be.to_numpy(X), be.to_numpy(Y), be.to_numpy(Z), **kwargs)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_aspect("equal")

        return fig, ax

    def __or__(self, other) -> UnionAperture:
        """Union: a point is inside if it is in either region."""
        return UnionAperture(self, other)

    def __add__(self, other) -> UnionAperture:
        """Alternative operator for union."""
        return self.__or__(other)

    def __and__(self, other) -> IntersectionAperture:
        """Intersection: a point is inside if it is in both regions."""
        return IntersectionAperture(self, other)

    def __sub__(self, other) -> DifferenceAperture:
        """Difference: a point is allowed if it is in self but not in other."""
        return DifferenceAperture(self, other)


class BaseBooleanAperture(BaseAperture):
    """Base class for boolean operations on apertures.

    Args:
        a (BaseAperture): The first aperture.
        b (BaseAperture): The second aperture.

    """

    def __init__(self, a: BaseAperture, b: BaseAperture):
        self.a = a
        self.b = b

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Returns the extent of the aperture.

        Returns:
            tuple: The extent of the aperture in the x and y directions.

        """
        a_extent = self.a.extent
        b_extent = self.b.extent
        x_min = min(a_extent[0], b_extent[0])
        x_max = max(a_extent[1], b_extent[1])
        y_min = min(a_extent[2], b_extent[2])
        y_max = max(a_extent[3], b_extent[3])
        return x_min, x_max, y_min, y_max

    @abstractmethod
    def contains(self, x, y) -> BEArray:
        """Checks if the given point is inside the aperture.

        Args:
            x (be.ndarray): The x-coordinate of the point.
            y (be.ndarray): The y-coordinate of the point.

        Returns:
            be.ndarray: Boolean array indicating if the point is inside the
                aperture

        """
        # pragma: no cover

    def scale(self, scale_factor: float):
        """Scales the aperture by the given factor.

        Args:
            scale_factor (float): The factor by which to scale the aperture.

        """
        self.a.scale(scale_factor)
        self.b.scale(scale_factor)

    def to_dict(self) -> dict:
        """Convert the aperture to a dictionary.

        Returns:
            dict: The dictionary representation of the aperture.

        """
        data = super().to_dict()
        data.update({"a": self.a.to_dict(), "b": self.b.to_dict()})
        return data

    @classmethod
    def from_dict(cls, data: dict):
        """Create an aperture from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the aperture.

        Returns:
            BaseBooleanAperture: The aperture object.

        """
        a = BaseAperture.from_dict(data["a"])
        b = BaseAperture.from_dict(data["b"])
        return cls(a, b)


class UnionAperture(BaseBooleanAperture):
    """Class for union of two apertures.

    Args:
        a (BaseAperture): The first aperture.
        b (BaseAperture): The second aperture.

    """

    def __init__(self, a: BaseAperture, b: BaseAperture):
        super().__init__(a, b)

    def contains(self, x: BEArray, y: BEArray) -> BEArray:
        """Checks if the given point is inside either aperture.

        Args:
            x (be.ndarray): The x-coordinate of the point.
            y (be.ndarray): The y-coordinate of the point.

        Returns:
            be.ndarray: Boolean array indicating if the point is inside the
                aperture

        """
        return be.logical_or(self.a.contains(x, y), self.b.contains(x, y))


class IntersectionAperture(BaseBooleanAperture):
    """Class for intersection of two apertures.

    Args:
        a (BaseAperture): The first aperture.
        b (BaseAperture): The second aperture.

    """

    def __init__(self, a: BaseAperture, b: BaseAperture):
        super().__init__(a, b)

    def contains(self, x: BEArray, y: BEArray) -> BEArray:
        """Checks if the given point is inside the aperture.

        Args:
            x (be.ndarray): The x-coordinate of the point.
            y (be.ndarray): The y-coordinate of the point.

        Returns:
            be.ndarray: Boolean array indicating if the point is inside the
                aperture

        """
        return be.logical_and(self.a.contains(x, y), self.b.contains(x, y))


class DifferenceAperture(BaseBooleanAperture):
    """Class for difference of two apertures.

    Args:
        a (BaseAperture): The first aperture.
        b (BaseAperture): The second aperture.

    """

    def __init__(self, a: BaseAperture, b: BaseAperture):
        super().__init__(a, b)

    def contains(self, x: BEArray, y: BEArray) -> BEArray:
        """Checks if the given point is inside the aperture.

        Args:
            x (be.ndarray): The x-coordinate of the point.
            y (be.ndarray): The y-coordinate of the point.

        Returns:
            be.ndarray: Boolean array indicating if the point is inside the
                aperture

        """
        return be.logical_and(
            self.a.contains(x, y),
            be.logical_not(self.b.contains(x, y)),
        )
