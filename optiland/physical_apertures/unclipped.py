"""A drawing boundary whose shape does not clip traced rays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optiland.physical_apertures.base import BaseAperture

if TYPE_CHECKING:
    from optiland._types import BEArray
    from optiland.rays import RealRays


class UnclippedAperture(BaseAperture):
    """Retain an aperture's display shape while disabling ray clipping.

    Args:
        aperture: Boundary used for display and membership queries. Its shape
            does not limit transmitted rays through this wrapper's ``clip``.
    """

    def __init__(self, aperture: BaseAperture) -> None:
        self.aperture = aperture

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Return the child boundary's drawing extent."""
        return self.aperture.extent

    def contains(self, x: BEArray, y: BEArray) -> BEArray:
        """Retain the child membership mask for drawing, without clipping rays."""
        return self.aperture.contains(x, y)

    def clip(self, rays: RealRays) -> None:
        """Leave ray intensity unchanged; this is a drawing boundary."""

    def scale(self, scale_factor: float) -> None:
        """Scale the drawing boundary in place."""
        self.aperture.scale(scale_factor)

    def to_dict(self) -> dict[str, Any]:
        """Persist the drawing-only wrapper and its child aperture."""
        return {**super().to_dict(), "aperture": self.aperture.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnclippedAperture:
        """Restore the drawing-only wrapper through native aperture dispatch."""
        return cls(BaseAperture.from_dict(data["aperture"]))
