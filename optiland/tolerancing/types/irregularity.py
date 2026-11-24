"""Irregularity Tolerance Type Module

This module contains the surface irregularity tolerance, the B and C terms of
the ISO 10110-5 form tolerance ``3/A(B/C)``. The power term A is the
``"fringes"`` tolerance type.

The irregularity is a Zernike sag error added to the nominal surface with a
``SagErrorGeometry``. One fringe is half a test wavelength of sag, and the
peak-to-valley sag of the error over the clear aperture equals the deviation
in fringes times half the test wavelength. The Zernike polynomials (Fringe
order, no normalization constant) are normalized to the clear semi-aperture:

- ``kind="B"`` (irregularity): astigmatism, ``a rho^2 cos(2 (phi - angle))``
  with ``a = s / 2``, where ``s`` is the peak-to-valley sag. Astigmatism is
  the lowest-order irregularity, has no power and is the common worst case.
  OpticStudio's TIRR operand instead models the irregularity as half
  spherical aberration and half astigmatism.
- ``kind="C"`` (rotationally symmetric irregularity): primary spherical
  aberration with power removed and no vertex offset,
  ``c (6 rho^4 - 6 rho^2)`` with ``c = s / 1.5``: Fringe Z9 with its piston
  removed.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland import _epoch
from optiland.geometries.sag_error import SagErrorGeometry
from optiland.tolerancing.base import Phase
from optiland.tolerancing.registry import ToleranceType
from optiland.tolerancing.types.surface import default_semi_aperture

if TYPE_CHECKING:
    from optiland.geometries.base import BaseGeometry
    from optiland.optic import Optic

_KINDS = {"B": "Irregularity", "C": "Rot. irregularity"}


class IrregularityTolerance(ToleranceType):
    """Surface irregularity in fringes, as a Zernike sag error.

    Applying a deviation installs a ``SagErrorGeometry`` around the current
    geometry of the surface, which is not changed; resetting restores that
    geometry object.

    Args:
        optic: The optic the tolerance applies to.
        surface_number: The surface index.
        test_wavelength: The test wavelength in µm. Defaults to 0.6328.
        semi_aperture: The clear semi-aperture in lens units, which is also
            the normalization radius. Defaults to the surface's
            semi-aperture, or the paraxial marginal plus chief ray heights
            when none is set.
        kind: ``"B"`` for irregularity (astigmatism, default) or ``"C"`` for
            rotationally symmetric irregularity (spherical aberration), as in
            ISO 10110-5 ``3/A(B/C)``.
        angle: Orientation of the astigmatism for kind ``"B"``, in degrees.
            Defaults to 0.
    """

    units = "fringes"
    phase = Phase.FORM

    def __init__(
        self,
        optic: Optic,
        surface_number: int,
        test_wavelength: float = 0.6328,
        semi_aperture: float | None = None,
        kind: str = "B",
        angle: float = 0.0,
    ) -> None:
        super().__init__(optic)
        if kind not in _KINDS:
            raise ValueError(f'kind must be "B" or "C", got {kind!r}.')
        if kind == "C" and angle != 0.0:
            raise ValueError('angle applies to kind "B" only.')
        self.surface_number = surface_number
        self.test_wavelength = test_wavelength
        self.kind = kind
        self.angle = angle
        self.semi_aperture = (
            default_semi_aperture(optic, surface_number)
            if semi_aperture is None
            else semi_aperture
        )
        if not self.semi_aperture > 0:
            raise ValueError("semi_aperture must be positive.")
        self._original: BaseGeometry | None = None
        self._installed: SagErrorGeometry | None = None

    @property
    def label(self) -> str:
        """Return a short, human-readable name."""
        return f"{_KINDS[self.kind]} S{self.surface_number}"

    def coefficients(self, fringes: float) -> list[float]:
        """Return the Fringe Zernike coefficients, from Z1, of a deviation.

        Args:
            fringes: The signed deviation in fringes.

        Returns:
            The coefficients in lens units, normalized to ``semi_aperture``.
        """
        # one fringe = lambda / 2 of sag; wavelength converted from um to mm
        sag_pv = fringes * self.test_wavelength * 1e-3 / 2.0
        coefficients = [0.0] * 9
        if self.kind == "B":
            # rho^2 cos(2 phi) spans [-1, 1]: a peak-to-valley of 2
            two_angle = 2.0 * math.radians(self.angle)
            coefficients[4] = sag_pv / 2.0 * math.cos(two_angle)
            coefficients[5] = sag_pv / 2.0 * math.sin(two_angle)
        else:
            # 6 rho^4 - 6 rho^2 spans [-1.5, 0]: a peak-to-valley of 1.5
            coefficients[8] = sag_pv / 1.5
            coefficients[0] = -coefficients[8]
        return coefficients

    def apply(self, delta: float) -> None:
        """Install the sag error of a deviation of ``delta`` fringes."""
        _epoch.assert_writable()
        surface = self.optic.surfaces[self.surface_number]
        if self._installed is None or surface.geometry is not self._installed:
            self._original = surface.geometry
        self._installed = SagErrorGeometry(
            self._original, be.array(self.coefficients(delta)), self.semi_aperture
        )
        surface.geometry = self._installed

    def reset(self) -> None:
        """Restore the geometry object found by ``apply``."""
        if self._installed is not None:
            _epoch.assert_writable()
            self.optic.surfaces[self.surface_number].geometry = self._original
        self._original = None
        self._installed = None
