"""Rotation of an existing physical aperture around a specified pivot."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from optiland.physical_apertures.base import BaseAperture

if TYPE_CHECKING:
    from optiland._types import BEArray


class RotatedAperture(BaseAperture):
    """Rotate an aperture counterclockwise about +z.

    Args:
        aperture: The aperture to rotate.
        angle: Rotation angle in radians.
        center_x: Pivot x-coordinate, defaulting to the surface origin.
        center_y: Pivot y-coordinate, defaulting to the surface origin.

    Raises:
        ValueError: If the angle or pivot coordinates are not finite.
    """

    def __init__(
        self,
        aperture: BaseAperture,
        angle: float,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> None:
        if not all(math.isfinite(value) for value in (angle, center_x, center_y)):
            raise ValueError("Aperture angle and pivot coordinates must be finite")
        self.aperture = aperture
        self.angle = angle
        self.center_x = center_x
        self.center_y = center_y

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Return an enclosing rotated bounding box, including unbounded shapes."""
        x0, x1, y0, y1 = self.aperture.extent
        x0, x1 = x0 - self.center_x, x1 - self.center_x
        y0, y1 = y0 - self.center_y, y1 - self.center_y
        c, s = math.cos(self.angle), math.sin(self.angle)

        def project(a: float, b: float, origin: float) -> tuple[float, float]:
            # A zero coefficient contributes nothing, even for infinite bounds.
            # Project intervals instead of corners to avoid inf - inf and 0*inf.
            xs = (a * x0, a * x1) if a else (0.0, 0.0)
            ys = (b * y0, b * y1) if b else (0.0, 0.0)
            return origin + min(xs) + min(ys), origin + max(xs) + max(ys)

        return (*project(c, -s, self.center_x), *project(s, c, self.center_y))

    def contains(self, x: BEArray, y: BEArray) -> BEArray:
        """Test points after inverse rotation into the child aperture's frame."""
        c, s = math.cos(self.angle), math.sin(self.angle)
        x, y = x - self.center_x, y - self.center_y
        return self.aperture.contains(
            self.center_x + c * x + s * y, self.center_y - s * x + c * y
        )

    def scale(self, scale_factor: float) -> None:
        """Scale the child aperture and pivot together, retaining the angle."""
        self.aperture.scale(scale_factor)
        self.center_x *= scale_factor
        self.center_y *= scale_factor

    def to_dict(self) -> dict[str, Any]:
        """Persist the child definition, angle and pivot in native aperture data."""
        return {
            **super().to_dict(),
            "aperture": self.aperture.to_dict(),
            "angle": self.angle,
            "center_x": self.center_x,
            "center_y": self.center_y,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RotatedAperture:
        """Restore a rotation using the shared native aperture dispatch."""
        return cls(
            BaseAperture.from_dict(data["aperture"]),
            data["angle"],
            data["center_x"],
            data["center_y"],
        )
