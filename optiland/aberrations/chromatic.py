"""Chromatic Aberrations

Computes first-order chromatic aberration quantities.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.aberrations.third_order import ThirdOrderAberrations

if TYPE_CHECKING:
    from optiland._types import BEArray
    from optiland.optic.context import OpticDataContext


class ChromaticAberrations:
    """First-order chromatic aberration quantities.

    Args:
        optic: Any object satisfying the OpticDataContext Protocol.
    """

    def __init__(self, optic: OpticDataContext) -> None:
        self._third_order = ThirdOrderAberrations(optic)

    def TAchC(self) -> BEArray:
        """Compute first-order transverse axial color.

        Returns:
            First-order transverse axial color per surface.
        """
        self._third_order._precalculations()
        return self._third_order._compute_over_surfaces(
            self._third_order._TAchC_term
        ).flatten()

    def LchC(self) -> BEArray:
        """Compute first-order longitudinal axial color.

        Returns:
            First-order longitudinal axial color per surface.
        """
        self._third_order._precalculations()
        TAchC = self._third_order._compute_over_surfaces(self._third_order._TAchC_term)
        return (-TAchC / self._third_order._ua[-1]).flatten()

    def TchC(self) -> BEArray:
        """Compute first-order lateral color.

        Returns:
            First-order lateral color per surface.
        """
        self._third_order._precalculations()
        return self._third_order._compute_over_surfaces(
            self._third_order._TchC_term
        ).flatten()
