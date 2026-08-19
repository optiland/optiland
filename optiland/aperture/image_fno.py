"""Image-Space F-Number (imageFNO) Aperture

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.aperture.base import BaseSystemAperture

if TYPE_CHECKING:
    from optiland.paraxial import Paraxial


class ImageFNOAperture(BaseSystemAperture):
    """Aperture specified as an image-space F-number.

    The entrance pupil diameter is derived as ``EPD = f2 / FNO``.

    Args:
        value: Image-space F-number.

    """

    _ap_type_key = "imageFNO"

    def __init__(self, value: float) -> None:
        self._value = value

    @property
    def ap_type(self) -> str:
        return "imageFNO"

    @property
    def value(self) -> float:
        return self._value

    @property
    def supports_telecentric(self) -> bool:
        return False

    @property
    def is_scalable(self) -> bool:
        return False

    def compute_epd(self, paraxial: Paraxial, wavelength: float | None = None) -> float:
        """Compute EPD as ``f2 / FNO``.

        Args:
            paraxial: Paraxial engine used to obtain the back focal length.
            wavelength: Unused for this aperture type.

        Returns:
            Entrance pupil diameter.

        """
        return paraxial.f2() / self._value

    def direct_fno(self) -> float:
        """Return the stored F-number directly, bypassing EPD computation.

        Returns:
            The image-space F-number.

        """
        return self._value

    def scale(self, factor: float) -> ImageFNOAperture:
        """Return ``self`` — F-number is dimensionless and does not scale.

        Args:
            factor: Ignored.

        Returns:
            This same instance (immutable, so returning self is safe).

        """
        return self

    def to_dict(self) -> dict:
        return {"type": "imageFNO", "value": self._value}

    @classmethod
    def _from_dict(cls, data: dict) -> ImageFNOAperture:
        return cls(data["value"])
