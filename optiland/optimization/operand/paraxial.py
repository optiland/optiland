"""Paraxial Operands Module

This module provides a class that calculates various paraxial values for an
optical system. It is used in conjunction with the optimization module to
optimize optical systems.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optiland._types import ScalarOrArray
    from optiland.optic import Optic


class ParaxialOperand:
    """A class representing paraxial operands.

    This class provides static methods to calculate various paraxial
        properties of an optic.

    Attributes:
        None

    Methods:
        f1(optic): Returns the first focal length of the optic.
        f2(optic): Returns the second focal length of the optic.
        f2_range(optic, start, end, wavelength): Returns the effective focal
            length of a range of surfaces (a lens group).
        F1(optic): Returns the first principal plane distance of the optic.
        F2(optic): Returns the second principal plane distance of the optic.
        P1(optic): Returns the first principal point distance of the optic.
        P2(optic): Returns the second principal point distance of the optic.
        N1(optic): Returns the first nodal point distance of the optic.
        N2(optic): Returns the second nodal point distance of the optic.
        EPL(optic): Returns the entrance pupil distance of the optic.
        EPD(optic): Returns the entrance pupil diameter of the optic.
        XPL(optic): Returns the exit pupil distance of the optic.
        XPD(optic): Returns the exit pupil diameter of the optic.
        magnification(optic): Returns the magnification of the optic.
        total_track(optic): Returns the total track of the optic.

    """

    @staticmethod
    def f1(optic):
        return optic.paraxial.f1()

    @staticmethod
    def f2(optic):
        return optic.paraxial.f2()

    @staticmethod
    def f2_range(
        optic: Optic,
        start: int,
        end: int,
        wavelength: float | None = None,
    ) -> ScalarOrArray:
        """Calculate the effective focal length of a range of surfaces.

        The range is treated as a lens group in isolation, as described in
        :meth:`optiland.paraxial.Paraxial.f2_range`. Use it to target the
        focal length of one element or lens group while the surrounding
        system is optimized.

        Args:
            optic: The optic object.
            start: Index of the first surface of the range, inclusive.
            end: Index of the last surface of the range, inclusive.
            wavelength: Wavelength in micrometers. Defaults to the system's
                primary wavelength.

        Returns:
            Effective focal length of the surface range. An afocal range
                returns infinity.
        """
        return optic.paraxial.f2_range(start, end, wavelength)

    @staticmethod
    def F1(optic):
        return optic.paraxial.F1()

    @staticmethod
    def F2(optic):
        return optic.paraxial.F2()

    @staticmethod
    def P1(optic):
        return optic.paraxial.P1()

    @staticmethod
    def P2(optic):
        return optic.paraxial.P2()

    @staticmethod
    def N1(optic):
        return optic.paraxial.N1()

    @staticmethod
    def N2(optic):
        return optic.paraxial.N2()

    @staticmethod
    def EPL(optic):
        return optic.paraxial.EPL()

    @staticmethod
    def EPD(optic):
        return optic.paraxial.EPD()

    @staticmethod
    def XPL(optic):
        return optic.paraxial.XPL()

    @staticmethod
    def XPD(optic):
        return optic.paraxial.XPD()

    @staticmethod
    def magnification(optic):
        return optic.paraxial.magnification()

    @staticmethod
    def total_track(optic):
        return optic.total_track
