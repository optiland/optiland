"""Float-By-Stop-Size Aperture

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.aperture.base import BaseSystemAperture, aperture_memo

if TYPE_CHECKING:
    from optiland._types import ScalarOrArray
    from optiland.paraxial import Paraxial


class FloatByStopAperture(BaseSystemAperture):
    """Aperture specified as the diameter of the stop surface.

    The entrance pupil diameter is derived via paraxial ray tracing through
    the stop surface.  For infinite-conjugate systems the marginal ray is
    traced forward; for finite-conjugate systems the object-space angle is
    scaled from the paraxial stop height.

    Args:
        value: Stop surface diameter in lens units.

    """

    _ap_type_key = "float_by_stop_size"

    def __init__(self, value: float) -> None:
        self._value = value

    @property
    def ap_type(self) -> str:
        return "float_by_stop_size"

    @property
    def value(self) -> float:
        return self._value

    @property
    def supports_telecentric(self) -> bool:
        return True

    @property
    def is_scalable(self) -> bool:
        return True

    @aperture_memo
    def compute_epd(self, paraxial: Paraxial, wavelength: float | None = None) -> float:
        """Compute EPD by paraxial ray tracing through the stop surface.

        Args:
            paraxial: Paraxial engine providing ray tracing and system access.
            wavelength: Primary wavelength in micrometers.  When ``None``,
                falls back to ``paraxial.optic.primary_wavelength``.

        Returns:
            Entrance pupil diameter.

        Raises:
            ValueError: If the object surface is not defined.

        """
        if paraxial.optic.object_surface is None:
            raise ValueError(
                "float_by_stop_size aperture requires a defined object surface."
            )

        if wavelength is None:
            wavelength = paraxial.optic.primary_wavelength

        if paraxial.optic.object_surface.is_infinite:
            stop_index = paraxial.surfaces.stop_index
            y, _ = paraxial.trace_generic(1.0, 0.0, -1, wavelength)
            return self._projected_stop_diameter(paraxial) / y[stop_index]
        else:
            obj_z = paraxial.optic.object_surface.geometry.cs.z
            epl_global = paraxial.entrance_pupil_axial_position()
            u0 = 2 * self._finite_object_slope(paraxial, wavelength)
            return u0 * (epl_global - obj_z)

    def object_space_marginal_slope(
        self, paraxial: Paraxial, wavelength: float | None = None
    ) -> ScalarOrArray | None:
        """Return the paraxial marginal slope set by the stop diameter.

        For a finite object, this is the slope of the axial ray that reaches
        the stop edge, the slope :meth:`compute_epd` already uses. It stays
        finite when the entrance pupil is at infinity.

        Args:
            paraxial: Paraxial engine providing ray tracing and system access.
            wavelength: Primary wavelength in micrometers.  When ``None``,
                falls back to ``paraxial.optic.primary_wavelength``.

        Returns:
            Marginal-ray slope in object space, or ``None`` for an object at
            infinity, where the marginal ray has no slope to fix.

        Raises:
            ValueError: If the object surface is not defined.

        """
        if paraxial.optic.object_surface is None:
            raise ValueError(
                "float_by_stop_size aperture requires a defined object surface."
            )
        if paraxial.optic.object_surface.is_infinite:
            return None
        if wavelength is None:
            wavelength = paraxial.optic.primary_wavelength
        return self._finite_object_slope(paraxial, wavelength)

    def _projected_stop_diameter(self, paraxial: Paraxial) -> ScalarOrArray:
        """Return the stop diameter projected perpendicular to the axis."""
        # The stop diameter is a clear aperture, measured in the stop's own plane,
        # but the paraxial heights are measured perpendicular to the axis. On a
        # tilted stop the beam footprint is stretched by 1 / cos(tilt), so only
        # this projection of the clear aperture gets through.
        stop_index = paraxial.surfaces.stop_index
        return self._value * paraxial.surfaces.obliquity[stop_index]

    def _finite_object_slope(
        self, paraxial: Paraxial, wavelength: float
    ) -> ScalarOrArray:
        """Return the slope of the axial ray from the object to the stop edge."""
        stop_index = paraxial.surfaces.stop_index
        obj_z = paraxial.optic.object_surface.geometry.cs.z
        y, _ = paraxial.trace_generic(0.0, 0.1, obj_z, wavelength)
        return 0.05 * self._projected_stop_diameter(paraxial) / y[stop_index]

    def scale(self, factor: float) -> FloatByStopAperture:
        """Return a new :class:`FloatByStopAperture` with value scaled by *factor*.

        Args:
            factor: Multiplicative scale factor.

        Returns:
            A new FloatByStopAperture instance.

        """
        return FloatByStopAperture(self._value * factor)

    def to_dict(self) -> dict:
        return {"type": "float_by_stop_size", "value": self._value}

    @classmethod
    def _from_dict(cls, data: dict) -> FloatByStopAperture:
        return cls(data["value"])
