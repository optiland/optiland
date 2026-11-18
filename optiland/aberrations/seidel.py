"""Seidel Aberrations

Computes Seidel aberration sums.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.aberrations.third_order import ThirdOrderAberrations

if TYPE_CHECKING:
    from optiland._types import BEArray
    from optiland.optic.context import OpticDataContext


class SeidelAberrations:
    """Seidel aberration coefficient sums (SI through SV).

    Args:
        optic: Any object satisfying the OpticDataContext Protocol.
    """

    def __init__(self, optic: OpticDataContext) -> None:
        self._third_order = ThirdOrderAberrations(optic)

    def seidels(self) -> BEArray:
        """Compute the Seidel aberration coefficients.

        Returns:
            Array of Seidel aberration coefficients [SI, SII, SIII, SIV, SV].
        """
        S = self._third_order._sum_seidels(*self._surface_terms())
        return S.squeeze()

    def seidels_per_surface(self) -> BEArray:
        """Compute the Seidel aberration coefficients of each surface.

        Summing over the surfaces gives ``seidels()``.

        Returns:
            Array of shape (5, num_surfaces - 2). Rows are SI through SV;
            columns are surfaces 1 through num_surfaces - 2, i.e. every
            surface except the object and image surfaces.
        """
        return self._third_order._seidels_per_surface(*self._surface_terms())

    def _surface_terms(self) -> tuple[BEArray, ...]:
        third_order = self._third_order
        third_order._precalculations()
        return tuple(
            third_order._compute_over_surfaces(term)
            for term in (
                third_order._TSC_term,
                third_order._CC_term,
                third_order._TAC_term,
                third_order._TPC_term,
                third_order._DC_term,
            )
        )
