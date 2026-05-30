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
        self._third_order._precalculations()
        TSC = self._third_order._compute_over_surfaces(self._third_order._TSC_term)
        CC = self._third_order._compute_over_surfaces(self._third_order._CC_term)
        TAC = self._third_order._compute_over_surfaces(self._third_order._TAC_term)
        TPC = self._third_order._compute_over_surfaces(self._third_order._TPC_term)
        DC = self._third_order._compute_over_surfaces(self._third_order._DC_term)
        S = self._third_order._sum_seidels(TSC, CC, TAC, TPC, DC)
        return S.squeeze()
