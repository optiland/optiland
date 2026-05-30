"""Float-By-Stop-Size Aperture

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.aperture.base import BaseSystemAperture

if TYPE_CHECKING:
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

        stop_index = paraxial.surfaces.stop_index

        if paraxial.optic.object_surface.is_infinite:
            y, _ = paraxial.trace_generic(1.0, 0.0, -1, wavelength)
            return self._value / y[stop_index]
        else:
            obj_z = paraxial.optic.object_surface.geometry.cs.z
            epl_global = paraxial.entrance_pupil_z()
            y, _ = paraxial.trace_generic(0.0, 0.1, obj_z, wavelength)
            u0 = 0.1 * self._value / y[stop_index]
            return u0 * (epl_global - obj_z)

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
