"""Object-Space Numerical Aperture (objectNA) Aperture

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.aperture.base import BaseSystemAperture

if TYPE_CHECKING:
    from optiland._types import BEArray, ScalarOrArray
    from optiland.optic import Optic
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

    def __init__(self, value: ScalarOrArray) -> None:
        self._value = value

    @property
    def ap_type(self) -> str:
        return "objectNA"

    @property
    def value(self) -> ScalarOrArray:
        return self._value

    @property
    def supports_telecentric(self) -> bool:
        return True

    @property
    def is_scalable(self) -> bool:
        return False

    def object_space_sine(
        self, optic: Optic, wavelength: float | None = None
    ) -> BEArray:
        """Convert NA to a launch sine in the actual object medium.

        Args:
            optic: Optical system providing the object material.
            wavelength: Reference wavelength; defaults to the primary wavelength,
                independently of the wavelengths of the rays being aimed.

        Returns:
            Scalar backend-valued sine, retaining NA and refractive-index
            gradients. Single-element arrays of any shape are normalized.
            Floating input precision is preserved; untyped values use a floating
            counterpart's dtype, or the configured precision if neither has one.

        Raises:
            ValueError: If the object is missing, NA or index is not a single
                value, the index is not finite and positive, or NA lies outside
                ``0 <= NA < n``. Zero defines a chief-only cone; a grazing cone
                has no finite forward slope.
        """
        if optic.object_surface is None:
            raise ValueError("objectNA aperture requires a defined object surface.")
        if wavelength is None:
            wavelength = optic.primary_wavelength
        # Preserve input precision: rounding to the backend default can turn
        # a valid NA < n into NA == n before validation.
        values = [optic.object_surface.material_post.n(wavelength), self._value]
        for i, value in enumerate(values):
            if hasattr(value, "dtype") and not be.is_torch_tensor(value):
                # Keep floating precision; other host dtypes retain the usual
                # backend conversion (some Torch integer types lack comparisons).
                values[i] = (
                    be.asarray(value, dtype=None)
                    if getattr(value.dtype, "kind", None) == "f"
                    else be.array(value)
                )
        for i, value in enumerate(values):
            if hasattr(value, "dtype"):
                continue
            dtype = getattr(values[1 - i], "dtype", None)
            values[i] = (
                be.asarray(value, dtype=dtype)
                if getattr(dtype, "kind", None) == "f"
                or getattr(dtype, "is_floating_point", False)
                else be.array(value)
            )
        index, na = values
        if be.size(index) != 1:
            raise ValueError("Object refractive index must contain exactly one value.")
        index = index.reshape(())
        if not bool(be.all(be.isfinite(index) & (index > 0))):
            raise ValueError("Object refractive index must be finite and positive.")
        if be.size(na) != 1:
            raise ValueError("Object NA must contain exactly one value.")
        na = na.reshape(())
        if not bool(be.all(be.isfinite(na) & (na >= 0) & (na < index))):
            raise ValueError("Object NA must be finite and satisfy 0 <= NA < n.")
        sine = na / index
        return sine if be.is_torch_tensor(sine) else be.asarray(sine, dtype=sine.dtype)

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
            ValueError: If the object surface is missing, its refractive index
                is invalid, or NA lies outside ``0 <= NA < n``.

        """
        sine = self.object_space_sine(paraxial.optic, wavelength)
        obj_z = paraxial.optic.object_surface.geometry.cs.z
        u0 = be.arcsin(sine)
        z = paraxial.entrance_pupil_axial_position() - obj_z
        # A diameter: a virtual entrance pupil behind the object (z < 0) is as wide
        # as one in front of it at the same distance, not negative.
        return 2 * be.abs(z) * be.tan(u0)

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
