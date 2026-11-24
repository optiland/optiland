"""Material Tolerance Types Module

This module contains the tolerance types that act on the material after a
surface: deviations of the d-line refractive index and of the Abbe number.
Both wrap the nominal material in an
:class:`~optiland.materials.offset.OffsetMaterial`, which keeps the shape of
the dispersion curve.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.materials.offset import OffsetMaterial
from optiland.tolerancing.base import Phase
from optiland.tolerancing.registry import ToleranceType

if TYPE_CHECKING:
    from optiland.optic import Optic


class MaterialTolerance(ToleranceType):
    """Tolerance that offsets the material after a surface.

    The nominal material object is never modified: ``apply`` replaces the
    surface's ``material_post`` with an ``OffsetMaterial`` around it, and
    ``reset`` restores the original object.

    Args:
        optic: The optic the tolerance applies to.
        surface_number: The surface whose following material is toleranced.

    Attributes:
        offset: The ``OffsetMaterial`` argument that receives the deviation.
        name: The name used in ``label``.
    """

    phase = Phase.MATERIAL
    offset: str = ""
    name: str = ""

    def __init__(self, optic: Optic, surface_number: int) -> None:
        super().__init__(optic)
        self.surface_number = surface_number
        self.surface = optic.surfaces[surface_number]
        self.nominal = self.surface.material_post

    @property
    def label(self) -> str:
        """Return a short, human-readable name."""
        return f"{self.name} S{self.surface_number}"

    def apply(self, delta: float) -> None:
        """Replace the material with the nominal one offset by ``delta``."""
        if delta == 0:
            self.reset()
            return
        self.optic.updater.set_material(
            OffsetMaterial(self.nominal, **{self.offset: delta}), self.surface_number
        )

    def reset(self) -> None:
        """Restore the nominal material object."""
        self.optic.updater.set_material(self.nominal, self.surface_number)


class IndexTolerance(MaterialTolerance):
    """Deviation of the d-line refractive index of the material after a
    surface. The Abbe number and the relative partial dispersions are kept."""

    offset = "delta_nd"
    units = "Δn_d"
    name = "Index"


class AbbeTolerance(MaterialTolerance):
    """Deviation of the Abbe number of the material after a surface. The
    d-line index and the relative partial dispersions are kept.

    Raises:
        ValueError: If the material is not dispersive.
    """

    offset = "delta_vd"
    units = "Δν_d"
    name = "Abbe"

    def __init__(self, optic: Optic, surface_number: int) -> None:
        super().__init__(optic, surface_number)
        # fails early for a non-dispersive material
        OffsetMaterial(self.nominal, delta_vd=1.0)
