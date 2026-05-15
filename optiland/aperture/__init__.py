"""Aperture Package

Provides the system aperture type hierarchy for Optiland.  Each aperture type
encapsulates how to compute the entrance pupil diameter (EPD), whether the
type supports telecentric object space, and whether its value scales with
system scaling.

Public API
----------
- :class:`BaseSystemAperture` — abstract base class
- :class:`EPDAperture` — entrance pupil diameter
- :class:`ImageFNOAperture` — image-space F-number
- :class:`ObjectNAAperture` — object-space numerical aperture
- :class:`FloatByStopAperture` — float by stop surface diameter
- :func:`make_system_aperture` — factory from legacy type string

"""

from __future__ import annotations

from optiland.aperture.base import BaseSystemAperture
from optiland.aperture.epd import EPDAperture
from optiland.aperture.float_by_stop import FloatByStopAperture
from optiland.aperture.image_fno import ImageFNOAperture
from optiland.aperture.object_na import ObjectNAAperture


def make_system_aperture(aperture_type: str, value: float) -> BaseSystemAperture:
    """Create a system aperture from a legacy type string and value.

    This is the preferred factory for constructing aperture objects when using
    the string-based API (e.g. via :meth:`~optiland.optic.Optic.set_aperture`).

    Args:
        aperture_type: One of ``'EPD'``, ``'imageFNO'``, ``'objectNA'``,
            ``'float_by_stop_size'``.
        value: The aperture value in lens units (or dimensionless for FNO/NA).

    Returns:
        A concrete :class:`BaseSystemAperture` instance.

    Raises:
        ValueError: If *aperture_type* is not a registered type.

    """
    # Aliases for legacy Zemax aperture identifiers. ``paraxialImageFNO`` is
    # the paraxial image-space F/#; in the paraxial regime it matches
    # Optiland's ``imageFNO`` (differences only appear as second-order
    # corrections in fast systems), which is good enough as a starting point.
    _ALIASES = {
        "paraxialImageFNO": "imageFNO",
    }
    aperture_type = _ALIASES.get(aperture_type, aperture_type)
    if aperture_type not in BaseSystemAperture._registry:
        raise ValueError(
            f"Aperture type must be one of "
            f"{list(BaseSystemAperture._registry)}; got '{aperture_type}'."
        )
    return BaseSystemAperture._registry[aperture_type](value)


__all__ = [
    "BaseSystemAperture",
    "EPDAperture",
    "ImageFNOAperture",
    "ObjectNAAperture",
    "FloatByStopAperture",
    "make_system_aperture",
]
