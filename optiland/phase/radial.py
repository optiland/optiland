"""
Provides a radially symmetric phase profile.
"""

from __future__ import annotations

from optiland import backend as be
from optiland.phase.base import BasePhaseProfile


class RadialPhaseProfile(BasePhaseProfile):
    """A radially symmetric phase profile defined by a polynomial in r.

    The phase is defined as a sum of even powers of the radial coordinate r:
    φ(r) = a_2 * r^2 + a_4 * r^4 + ...

    Args:
        coefficients (List[float]): A list of coefficients [a_2, a_4, ...].
    """

    phase_type = "radial"

    def __init__(self, coefficients: list[float]):
        super().__init__()
        self.coefficients = coefficients

    def get_phase(
        self, x: be.Array, y: be.Array, wavelength: be.Array = None
    ) -> be.Array:
        """Calculates the phase added by the profile at coordinates (x, y).

        Args:
            x: The x-coordinates of the points of interest.
            y: The y-coordinates of the points of interest.

        Returns:
            The phase at each (x, y) coordinate.
        """
        r_squared = x**2 + y**2
        phase = be.zeros_like(x)
        for i, coeff in enumerate(self.coefficients):
            power = i + 1
            phase = phase + coeff * (r_squared**power)
        return phase

    def get_gradient(
        self, x: be.Array, y: be.Array, wavelength: be.Array = None
    ) -> tuple[be.Array, be.Array, be.Array]:
        """Calculates the gradient of the phase at coordinates (x, y).

        Args:
            x: The x-coordinates of the points of interest.
            y: The y-coordinates of the points of interest.

        Returns:
            A tuple containing the x, y, and z components of the phase
            gradient (d_phi/dx, d_phi/dy, 0).
        """
        r_squared = x**2 + y**2
        r = be.sqrt(r_squared)

        # Calculate d_phi/dr
        d_phi_dr = be.zeros_like(r)
        for i, coeff in enumerate(self.coefficients):
            power = i + 1
            d_phi_dr = d_phi_dr + coeff * 2 * power * (r ** (2 * power - 1))

        # Handle r=0 case to avoid division by zero
        # At r=0, the gradient is (0, 0)
        safe_r = be.where(r == 0, 1.0, r)
        d_phi_dx = (d_phi_dr / safe_r) * x
        d_phi_dy = (d_phi_dr / safe_r) * y

        d_phi_dx = be.where(r == 0, 0.0, d_phi_dx)
        d_phi_dy = be.where(r == 0, 0.0, d_phi_dy)
        d_phi_dz = be.zeros_like(x)

        return d_phi_dx, d_phi_dy, d_phi_dz

    def get_paraxial_gradient(
        self, y: be.Array, wavelength: be.Array = None
    ) -> be.Array:
        """Calculates the paraxial phase gradient at y-coordinate.

        This is the gradient d_phi/dy evaluated at x=0.

        Args:
            y: The y-coordinates of the points of interest.

        Returns:
            The paraxial phase gradient at each y-coordinate.
        """
        # At x=0, r = |y| and d_phi/dy = (d_phi/dr) * sign(y)
        r = be.abs(y)

        # Calculate d_phi/dr
        d_phi_dr = be.zeros_like(r)
        for i, coeff in enumerate(self.coefficients):
            power = i + 1
            d_phi_dr = d_phi_dr + coeff * 2 * power * (r ** (2 * power - 1))

        return d_phi_dr * be.sign(y)

    def to_dict(self) -> dict:
        """Serializes the phase profile to a dictionary.

        Returns:
            A dictionary representation of the phase profile.
        """
        data = super().to_dict()
        data["coefficients"] = self.coefficients
        return data

    @classmethod
    def from_dict(cls, data: dict) -> RadialPhaseProfile:
        """Deserializes a phase profile from a dictionary.

        Args:
            data: A dictionary representation of a phase profile.

        Returns:
            An instance of a `RadialPhaseProfile`.
        """
        return cls(coefficients=data["coefficients"])
