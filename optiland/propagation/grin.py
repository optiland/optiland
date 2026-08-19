"""Placeholder for Graded-Index (GRIN) propagation model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.propagation.base import BasePropagationModel

if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial
    from optiland.rays.real_rays import RealRays


class GRINPropagation(BasePropagationModel):
    """Placeholder for GRIN propagation.

    This model is not yet implemented and will raise an error if used.
    """

    def propagate(self, rays: RealRays, t: float) -> None:
        """Raises NotImplementedError.

        Args:
            rays: The rays object to be propagated.
            t: The distance to propagate.
        """
        raise NotImplementedError("GRIN propagation is not yet implemented.")

    @classmethod
    def from_dict(cls, d: dict, material: BaseMaterial = None) -> GRINPropagation:
        """Creates a GRINPropagation model from a dictionary.

        This method is called by the parent material during its own
        deserialization process.

        Args:
            d: The dictionary representation of the model.
            material: The parent material instance. This is accepted for
                API consistency but is not used by this model.

        Returns:
            An instance of the GRINPropagation model.
        """
        return cls()
