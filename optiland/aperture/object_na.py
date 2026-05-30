"""Object-Space Numerical Aperture (objectNA) Aperture

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.aperture.base import BaseSystemAperture

if TYPE_CHECKING:
    from optiland.paraxial import Paraxial


class ObjectNAAperture(BaseSystemAperture):
    """Aperture specified as an object-space numerical aperture.

    The entrance pupil diameter is derived from the object-space NA using the
    object distance, primary wavelength, and the refractive index of the medium
    at the object surface.

    Args:
        value: Object-space numerical aperture (NA = n * sin(θ)).

    """

    _ap_type_key = "objectNA"

    def __init__(self, value: float) -> None:
        self._value = value

    @property
    def ap_type(self) -> str:
        return "objectNA"

    @property
    def value(self) -> float:
        return self._value

    @property
    def supports_telecentric(self) -> bool:
        return True

    @property
    def is_scalable(self) -> bool:
        return False

    def compute_epd(self, paraxial: Paraxial, wavelength: float | None = None) -> float:
        """Compute EPD from object-space NA.

        Args:
            paraxial: Paraxial engine providing access to system geometry and
                material data.
            wavelength: Primary wavelength in micrometers.  When ``None``,
                falls back to ``paraxial.optic.primary_wavelength``.

        Returns:
            Entrance pupil diameter.

        Raises:
            ValueError: If the object surface is not defined.

        """
        if paraxial.optic.object_surface is None:
            raise ValueError("objectNA aperture requires a defined object surface.")

        if wavelength is None:
            wavelength = paraxial.optic.primary_wavelength

        obj_z = paraxial.optic.object_surface.geometry.cs.z
        n0 = paraxial.optic.object_surface.material_post.n(wavelength)
        u0 = be.arcsin(self._value / n0)
        z = paraxial.entrance_pupil_z() - obj_z
        return 2 * z * be.tan(u0)

    def scale(self, factor: float) -> ObjectNAAperture:
        """Return ``self`` — NA is dimensionless and does not scale.

        Args:
            factor: Ignored.

        Returns:
            This same instance (immutable, so returning self is safe).

        """
        return self

    def to_dict(self) -> dict:
        return {"type": "objectNA", "value": self._value}

    @classmethod
    def _from_dict(cls, data: dict) -> ObjectNAAperture:
        return cls(data["value"])
