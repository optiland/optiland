"""Homogeneous, straight-line propagation model.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland import backend as be
from optiland.propagation.base import BasePropagationModel

if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial
    from optiland.rays.real_rays import RealRays


class HomogeneousPropagation(BasePropagationModel):
    """Propagates rays in a straight line through a homogeneous medium."""

    def __init__(self, material: BaseMaterial):
        """Initializes the HomogeneousPropagation model.

        Args:
            material: A reference to the parent material instance, used to
                query for properties like the extinction coefficient k.
        """
        self.material = material

    def propagate(self, rays: RealRays, t: float) -> None:
        """Propagate the rays a distance t.

        This implements straight-line propagation and accounts for material
        absorption (attenuation). The ray's state is modified in-place.

        Args:
            rays: The rays object to be propagated.
            t: The distance to propagate.
        """
        rays.x = rays.x + t * rays.L
        rays.y = rays.y + t * rays.M
        rays.z = rays.z + t * rays.N

        # Handle absorption based on the material's extinction coefficient k
        k = self.material.k(rays.w)

        # Only apply attenuation if there is a non-zero extinction coefficient
        if be.any(k > 0):
            # The absorption coefficient alpha is given by 4 * pi * k / lambda
            alpha = 4 * be.pi * k / rays.w
            # Intensity loss is I = I_0 * exp(-alpha * z)
            # Distance t is in mm; wavelength w is in um. Convert t to um.
            rays.i = rays.i * be.exp(-alpha * t * 1e3)

        # normalize, if required
        if not rays.is_normalized:
            rays.normalize()

    @classmethod
    def from_dict(cls, d: dict, material: BaseMaterial) -> HomogeneousPropagation:
        """Creates a HomogeneousPropagation model from a dictionary.

        This method is called by the parent material during its own
        deserialization process to resolve dependencies.

        Args:
            d: The dictionary representation of the model.
            material: The parent material instance.

        Returns:
            An instance of the HomogeneousPropagation model.
        """
        return cls(material=material)
