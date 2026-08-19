"""Material Variable Module

This module contains the MaterialVariable class, which represents a variable for
the material at a specific surface such as a glass.
The variable can be used in optimization problems to optimize the material
at a specific surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.materials.abbe import AbbeMaterial
from optiland.materials.material_utils import find_closest_glass, get_nd_vd
from optiland.optimization.scaling.identity import IdentityScaler
from optiland.optimization.variable.base import VariableBehavior
from optiland.surfaces.factories.material_factory import MaterialFactory

if TYPE_CHECKING:
    from optiland.materials.base import BaseMaterial
    from optiland.optimization.scaling.base import Scaler


class MaterialVariable(VariableBehavior):
    """Represents a variable for the material at a specific surface.

    Args:
        optic (Optic): The optic object associated with the variable.
        surface_number (int): The surface number where the variable is applied.
            calculated.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        optic,
        surface_number: int,
        glass_selection: list[str],
        scaler: Scaler = None,
        **kwargs,
    ):
        if scaler is None:
            scaler = IdentityScaler()
        super().__init__(optic, surface_number, scaler=scaler, **kwargs)
        self.glass_selection = glass_selection
        self.surface = self.optic.surfaces[self.surface_number]

        if isinstance(self.surface.material_post, AbbeMaterial):
            nd_vd = (
                float(self.surface.material_post.index[0].item()),
                float(self.surface.material_post.abbe[0].item()),
            )
            glass = find_closest_glass(nd_vd=nd_vd, catalog=glass_selection)
            new_nd, new_vd = get_nd_vd(glass)
            self.update_value(new_value=glass)
            print(
                f"The material of surface {surface_number:<2} is defined "
                f"by its AbbeMaterial {nd_vd}. GlassExpert converted it "
                f"to real material {glass:<7} ({new_nd:.4f}, {new_vd:.2f}) "
                f"to proceed with optimization."
            )

    def get_value(self) -> str:
        """Returns the name of the material at the specified surface.

        Returns:
            str: The material name.

        """
        return self.surface.material_post.name

    def update_value(self, new_value: str) -> None:
        """Updates the material at the specified surface.

        Args:
            new_value (float): The new material name.

        """
        material_post: BaseMaterial = MaterialFactory._configure_post_material(
            new_value
        )
        self.optic.updater.set_material(material_post, self.surface_number)

    def __str__(self) -> str:
        """Return a string representation of the variable.

        Returns:
            str: A string representation of the variable.

        """
        return f"Material, Surface {self.surface_number}"
