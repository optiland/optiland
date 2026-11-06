"""Gradient-index (GRIN) material.

This module defines :class:`GradientMaterial`, a material whose refractive index
varies with position inside the medium.

Jie Cai (zhazhajust), 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.materials.base import BaseMaterial
from optiland.propagation.grin import GRINPropagation

if TYPE_CHECKING:
    from optiland._types import BEArray, ScalarOrArray

_COEFFICIENTS = ("n0", "nr2", "nr4", "nr6", "nz1", "nz2", "nz3")


class GradientMaterial(BaseMaterial):
    """Material with a radial and axial gradient index.

    The index follows the Gradient 3 law

    .. math::

        n(r, z) = n_0 + n_{r2} r^2 + n_{r4} r^4 + n_{r6} r^6
                  + n_{z1} z + n_{z2} z^2 + n_{z3} z^3,

    with :math:`r^2 = x^2 + y^2`. Coordinates are in mm, in the local frame of
    the surface where the medium starts. The index does not depend on
    wavelength, and the medium does not absorb.

    Rays are traced through the medium by :class:`GRINPropagation`.

    Args:
        n0: Index at the medium origin.
        nr2: Coefficient of :math:`r^2`, in mm⁻².
        nr4: Coefficient of :math:`r^4`, in mm⁻⁴.
        nr6: Coefficient of :math:`r^6`, in mm⁻⁶.
        nz1: Coefficient of :math:`z`, in mm⁻¹.
        nz2: Coefficient of :math:`z^2`, in mm⁻².
        nz3: Coefficient of :math:`z^3`, in mm⁻³.
        propagation_model: The integrator settings. Defaults to
            ``GRINPropagation()``.
    """

    def __init__(
        self,
        n0: ScalarOrArray = 1.5,
        nr2: ScalarOrArray = 0.0,
        nr4: ScalarOrArray = 0.0,
        nr6: ScalarOrArray = 0.0,
        nz1: ScalarOrArray = 0.0,
        nz2: ScalarOrArray = 0.0,
        nz3: ScalarOrArray = 0.0,
        propagation_model: GRINPropagation | None = None,
    ):
        super().__init__(propagation_model or GRINPropagation())
        self.n0 = n0
        self.nr2 = nr2
        self.nr4 = nr4
        self.nr6 = nr6
        self.nz1 = nz1
        self.nz2 = nz2
        self.nz3 = nz3

    def index_and_gradient(
        self,
        x: BEArray,
        y: BEArray,
        z: BEArray,
        wavelength: ScalarOrArray,
    ) -> tuple[BEArray, BEArray, BEArray, BEArray]:
        """Returns the index and its gradient at points in the medium.

        Args:
            x: x-coordinates in the medium frame, in mm.
            y: y-coordinates in the medium frame, in mm.
            z: z-coordinates in the medium frame, in mm.
            wavelength: Wavelength in µm. Not used by this law.

        Returns:
            The index ``n`` and the partial derivatives ``dn/dx``, ``dn/dy``
            and ``dn/dz``.
        """
        r2 = x**2 + y**2
        n = (
            self.n0
            + r2 * (self.nr2 + r2 * (self.nr4 + r2 * self.nr6))
            + z * (self.nz1 + z * (self.nz2 + z * self.nz3))
        )
        radial = 2 * self.nr2 + r2 * (4 * self.nr4 + 6 * self.nr6 * r2)
        dn_dz = self.nz1 + z * (2 * self.nz2 + 3 * self.nz3 * z)
        return n, x * radial, y * radial, dn_dz + be.zeros_like(x)

    def _calculate_n(self, wavelength: ScalarOrArray, **kwargs) -> BEArray:
        """Returns the index at ``x``, ``y``, ``z``, or ``n0`` without a position.

        Args:
            wavelength: Wavelength in µm. Not used by this law.
            **kwargs: Optional ``x``, ``y`` and ``z`` in the medium frame.

        Returns:
            The refractive index.
        """
        if not {"x", "y", "z"} & kwargs.keys():
            return self._broadcast_wavelength(self.n0, wavelength)
        x, y, z = (be.asarray(kwargs.get(key, 0.0)) for key in ("x", "y", "z"))
        return self.index_and_gradient(x, y, z, wavelength)[0]

    def _calculate_k(self, wavelength: ScalarOrArray, **kwargs) -> BEArray:
        """Returns zero: the medium does not absorb.

        Args:
            wavelength: Wavelength in µm.
            **kwargs: Ignored.

        Returns:
            Zero, shaped like ``wavelength`` when it is an array.
        """
        return self._broadcast_wavelength(0.0, wavelength)

    @staticmethod
    def _broadcast_wavelength(
        value: ScalarOrArray, wavelength: ScalarOrArray
    ) -> BEArray:
        """Returns ``value`` as a backend array, shaped like an array wavelength."""
        value = be.array(value)
        if be.is_array_like(wavelength):
            return value + be.zeros_like(be.array(wavelength))
        return value

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the material.

        Returns:
            The dictionary representation.
        """
        data = super().to_dict()
        data.update(
            {key: float(be.to_numpy(getattr(self, key))) for key in _COEFFICIENTS}
        )
        return data

    @classmethod
    def from_dict(cls, data: dict) -> GradientMaterial:
        """Creates a material from a dictionary representation.

        Args:
            data: The dictionary representation.

        Returns:
            The material.
        """
        return cls(**{key: data[key] for key in _COEFFICIENTS if key in data})
