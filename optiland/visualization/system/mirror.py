"""Mirror Visualization Module

This module contains the Mirror3D class, which is used to visualize a 3D mirror
surface.

Kramer Harrison, 2024
"""

from __future__ import annotations

from optiland.visualization.system.surface import Surface3D


class Mirror3D(Surface3D):
    """A class used to represent a 3D Mirror surface.
    Inherits from Surface3D.

    Args:
        surface (Surface): The mirror surface to be plotted.
        extent (tuple): The extent of the mirror surface in the x and y
            directions.

    Methods:
        _configure_material(actor):
            Configures the material properties of the mirror surface.

    """

    def __init__(self, surface, extent):
        super().__init__(surface, extent)

    def _configure_material(self, actor, theme=None):
        """Configures the material properties of the mirror surface.

        Args:
            actor (vtkActor): The actor representing the mirror surface.

        """
        actor.GetProperty().SetColor(1, 1, 1)
        actor.GetProperty().SetAmbient(0.3)
        actor.GetProperty().SetDiffuse(0.1)
        actor.GetProperty().SetSpecular(1.0)
        actor.GetProperty().SetSpecularPower(100)

        return actor
