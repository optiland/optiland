"""Surface Tolerance Types Module

This module contains the tolerance types that act on a single surface: radius,
power in fringes, conic constant, thickness, and surface decenter and tilt.

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be
from optiland.optimization.scaling.identity import IdentityScaler
from optiland.optimization.variable import Variable
from optiland.tolerancing.base import Phase
from optiland.tolerancing.registry import ToleranceType

if TYPE_CHECKING:
    from optiland.optic import Optic


def _scalar(value) -> float:
    """Convert a backend scalar (or one-element array) to a Python float."""
    return float(np.ravel(be.to_numpy(value))[0])


def default_semi_aperture(optic: Optic, surface_number: int) -> float:
    """Return the clear semi-aperture of a surface for a form tolerance.

    Args:
        optic: The optic.
        surface_number: The surface index.

    Returns:
        The surface's semi-aperture, or the paraxial marginal plus chief ray
        heights when none is set.
    """
    surface = optic.surfaces[surface_number]
    if surface.semi_aperture is not None:
        return _scalar(surface.semi_aperture)
    ya, _ = optic.paraxial.marginal_ray()
    yb, _ = optic.paraxial.chief_ray()
    k = surface_number
    return abs(_scalar(be.ravel(ya)[k])) + abs(_scalar(be.ravel(yb)[k]))


class VariableTolerance(ToleranceType):
    """Tolerance that adds a scaled deviation to an optimization variable.

    Args:
        optic: The optic the tolerance applies to.
        surface_number: The surface index.
        **kwargs: Extra arguments for the variable, such as ``axis``.

    Attributes:
        variable_type: The optimization variable type that is perturbed.
        scale: Factor from deviation units to variable units.
        pickup_attr: The pickup ``attr_type`` that can drive the variable, or
            None if no pickup can.
    """

    variable_type: str = ""
    scale: float = 1.0
    name: str = ""
    pickup_attr: str | None = None

    def __init__(self, optic: Optic, surface_number: int, **kwargs) -> None:
        super().__init__(optic)
        self.surface_number = surface_number
        self.variable = Variable(
            optic,
            self.variable_type,
            scaler=IdentityScaler(),
            surface_number=surface_number,
            **kwargs,
        )
        self.nominal = _scalar(self.variable.value)

    @property
    def label(self) -> str:
        """Return a short, human-readable name."""
        return f"{self.name} S{self.surface_number}"

    def pickup_targets(self) -> list[tuple[int, str]]:
        """Return the pickup-linkable attribute, if any."""
        if self.pickup_attr is None:
            return []
        return [(self.surface_number, self.pickup_attr)]

    def apply(self, delta: float) -> None:
        """Set the variable to nominal plus the scaled deviation."""
        self.variable.update(self.nominal + self.scale * delta)

    def reset(self) -> None:
        """Restore the nominal value."""
        self.variable.update(self.nominal)


class RadiusTolerance(VariableTolerance):
    """Deviation of the radius of curvature, in lens units.

    Raises:
        ValueError: If the surface is flat. Use ``"fringes"`` instead.
    """

    variable_type = "radius"
    units = "lens units"
    name = "Radius"
    pickup_attr = "radius"

    def __init__(self, optic: Optic, surface_number: int) -> None:
        super().__init__(optic, surface_number)
        if not math.isfinite(self.nominal):
            raise ValueError(
                f"Surface {surface_number} is flat; a radius tolerance is "
                'undefined. Use the "fringes" tolerance type instead.'
            )


class ConicTolerance(VariableTolerance):
    """Deviation of the conic constant."""

    variable_type = "conic"
    name = "Conic"
    pickup_attr = "conic"


class SurfaceDecenterTolerance(VariableTolerance):
    """Decenter of one surface along ``axis`` ("x" or "y"), in lens units."""

    variable_type = "decenter"
    units = "lens units"
    phase = Phase.RIGID_BODY

    def __init__(self, optic: Optic, surface_number: int, axis: str) -> None:
        if axis not in ("x", "y"):
            raise ValueError(f'axis must be "x" or "y", got {axis!r}.')
        self.name = f"Decenter {axis.upper()}"
        super().__init__(optic, surface_number, axis=axis)


class SurfaceTiltTolerance(VariableTolerance):
    """Tilt of one surface about its own vertex, around ``axis`` ("x" or
    "y"), in degrees."""

    variable_type = "tilt"
    units = "degrees"
    scale = math.pi / 180.0
    phase = Phase.RIGID_BODY

    def __init__(self, optic: Optic, surface_number: int, axis: str) -> None:
        if axis not in ("x", "y"):
            raise ValueError(f'axis must be "x" or "y", got {axis!r}.')
        self.name = f"Tilt {axis.upper()}"
        super().__init__(optic, surface_number, axis=axis)


class ThicknessTolerance(ToleranceType):
    """Deviation of the thickness after a surface, in lens units.

    Args:
        optic: The optic the tolerance applies to.
        surface_number: The surface whose following thickness changes.
        adjust: Optional surface whose following thickness absorbs the
            opposite deviation, so that surfaces after it do not move.
    """

    units = "lens units"

    def __init__(
        self, optic: Optic, surface_number: int, adjust: int | None = None
    ) -> None:
        super().__init__(optic)
        if adjust is not None and adjust == surface_number:
            raise ValueError("adjust must differ from surface_number.")
        self.surface_number = surface_number
        self.adjust = adjust
        self._thickness = self._variable(surface_number)
        self._adjust = None if adjust is None else self._variable(adjust)

    def _variable(self, surface_number: int) -> tuple[Variable, float]:
        variable = Variable(
            self.optic,
            "thickness",
            scaler=IdentityScaler(),
            surface_number=surface_number,
        )
        return variable, _scalar(variable.value)

    @property
    def label(self) -> str:
        """Return a short, human-readable name."""
        return f"Thickness S{self.surface_number}"

    def pickup_targets(self) -> list[tuple[int, str]]:
        """Return the thickness and, if set, the adjust thickness."""
        targets = [(self.surface_number, "thickness")]
        if self.adjust is not None:
            targets.append((self.adjust, "thickness"))
        return targets

    def apply(self, delta: float) -> None:
        """Change the thickness, and the adjust thickness by the opposite
        amount."""
        variable, nominal = self._thickness
        variable.update(nominal + delta)
        if self._adjust is not None:
            variable, nominal = self._adjust
            variable.update(nominal - delta)

    def reset(self) -> None:
        """Restore both thicknesses."""
        self.apply(0.0)


class FringesTolerance(ToleranceType):
    """Power deviation in fringes, applied as a change of radius.

    One fringe is half a test wavelength of sag difference at the edge of the
    clear aperture. A positive deviation increases the magnitude of the sag (a
    more strongly curved surface); on a flat surface it gives a positive
    radius. The conversion is exact for conic surfaces:
    ``c = 2 s / (h**2 + (1 + k) s**2)`` for sag ``s`` at height ``h``.

    Args:
        optic: The optic the tolerance applies to.
        surface_number: The surface index.
        test_wavelength: The test wavelength in µm. Defaults to 0.6328.
        semi_aperture: The clear semi-aperture in lens units. Defaults to the
            surface's semi-aperture, or the paraxial marginal plus chief ray
            heights when none is set.
    """

    units = "fringes"

    def __init__(
        self,
        optic: Optic,
        surface_number: int,
        test_wavelength: float = 0.6328,
        semi_aperture: float | None = None,
    ) -> None:
        super().__init__(optic)
        self.surface_number = surface_number
        self.test_wavelength = test_wavelength
        self.variable = Variable(
            optic, "radius", scaler=IdentityScaler(), surface_number=surface_number
        )
        self.nominal = _scalar(self.variable.value)
        self.semi_aperture = (
            self._default_semi_aperture() if semi_aperture is None else semi_aperture
        )
        if not self.semi_aperture > 0:
            raise ValueError("semi_aperture must be positive.")
        geometry = optic.surfaces[surface_number].geometry
        self.conic = _scalar(getattr(geometry, "k", 0.0))

    def _default_semi_aperture(self) -> float:
        return default_semi_aperture(self.optic, self.surface_number)

    def radius_for(self, fringes: float) -> float:
        """Return the radius that corresponds to a power deviation.

        Args:
            fringes: The signed deviation in fringes.

        Returns:
            The perturbed radius, ``inf`` for a flat surface.
        """
        h = self.semi_aperture
        k = self.conic
        c = 0.0 if math.isinf(self.nominal) else 1.0 / self.nominal
        arg = 1.0 - (1.0 + k) * c**2 * h**2
        if arg < 0:
            raise ValueError(
                f"Semi-aperture {h} exceeds the sag domain of surface "
                f"{self.surface_number}."
            )
        sag = c * h**2 / (1.0 + math.sqrt(arg))
        sign = 1.0 if sag >= 0 else -1.0
        # one fringe = lambda / 2 of sag; wavelength converted from um to mm
        new_sag = sag + sign * fringes * self.test_wavelength * 1e-3 / 2.0
        new_c = 2.0 * new_sag / (h**2 + (1.0 + k) * new_sag**2)
        return math.inf if new_c == 0 else 1.0 / new_c

    @property
    def label(self) -> str:
        """Return a short, human-readable name."""
        return f"Fringes S{self.surface_number}"

    def pickup_targets(self) -> list[tuple[int, str]]:
        """Return the radius, which carries the power deviation."""
        return [(self.surface_number, "radius")]

    def apply(self, delta: float) -> None:
        """Set the radius for a power deviation of ``delta`` fringes."""
        self.variable.update(self.radius_for(delta))

    def reset(self) -> None:
        """Restore the nominal radius."""
        self.variable.update(self.nominal)
