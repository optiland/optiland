"""Aberrations Package

Provides the Aberrations facade and focused sub-classes for computing
first- and third-order aberrations.

The public Aberrations class delegates to three focused classes:
  - ThirdOrderAberrations  (TSC, SC, CC, TCC, TAC, AC, TPC, PC, DC)
  - SeidelAberrations      (seidels)
  - ChromaticAberrations   (TAchC, LchC, TchC)

All public methods on Aberrations continue to work unchanged.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.aberrations.chromatic import ChromaticAberrations
from optiland.aberrations.seidel import SeidelAberrations
from optiland.aberrations.third_order import ThirdOrderAberrations

if TYPE_CHECKING:
    from optiland._types import BEArray
    from optiland.optic.context import OpticDataContext


class Aberrations:
    """Facade over ThirdOrderAberrations, SeidelAberrations, ChromaticAberrations.

    All public methods delegate to the appropriate focused class.
    ``optic.aberrations.seidels()`` continues to work exactly as before.

    Args:
        optic: The optical system (or any object satisfying OpticDataContext).
    """

    def __init__(self, optic: OpticDataContext) -> None:
        self.optic = optic
        self._third_order = ThirdOrderAberrations(optic)
        self._seidel = SeidelAberrations(optic)
        self._chromatic = ChromaticAberrations(optic)

    # ── Third-order delegates ──────────────────────────────────────────────

    def third_order(self) -> tuple[BEArray, ...]:
        """Compute all third-order aberrations and first-order color terms."""
        return self._third_order.third_order()

    def TSC(self) -> BEArray:
        """Compute third-order transverse spherical aberration."""
        return self._third_order.TSC()

    def SC(self) -> BEArray:
        """Compute third-order longitudinal spherical aberration."""
        return self._third_order.SC()

    def CC(self) -> BEArray:
        """Compute third-order sagittal coma."""
        return self._third_order.CC()

    def TCC(self) -> BEArray:
        """Compute third-order tangential coma."""
        return self._third_order.TCC()

    def TAC(self) -> BEArray:
        """Compute third-order transverse astigmatism."""
        return self._third_order.TAC()

    def AC(self) -> BEArray:
        """Compute third-order longitudinal astigmatism."""
        return self._third_order.AC()

    def TPC(self) -> BEArray:
        """Compute third-order transverse Petzval sum."""
        return self._third_order.TPC()

    def PC(self) -> BEArray:
        """Compute third-order longitudinal Petzval sum."""
        return self._third_order.PC()

    def DC(self) -> BEArray:
        """Compute third-order distortion."""
        return self._third_order.DC()

    # ── Seidel delegate ───────────────────────────────────────────────────

    def seidels(self) -> BEArray:
        """Compute Seidel aberration coefficients."""
        return self._seidel.seidels()

    # ── Chromatic delegates ───────────────────────────────────────────────

    def TAchC(self) -> BEArray:
        """Compute first-order transverse axial color."""
        return self._chromatic.TAchC()

    def LchC(self) -> BEArray:
        """Compute first-order longitudinal axial color."""
        return self._chromatic.LchC()

    def TchC(self) -> BEArray:
        """Compute first-order lateral color."""
        return self._chromatic.TchC()


__all__ = [
    "Aberrations",
    "ThirdOrderAberrations",
    "SeidelAberrations",
    "ChromaticAberrations",
]
