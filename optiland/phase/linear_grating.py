"""
Provides a linear grating phase profile.
"""

from __future__ import annotations

from optiland import backend as be
from optiland.phase.base import BasePhaseProfile


class LinearGratingPhaseProfile(BasePhaseProfile):
    """A linear grating phase profile.

    This profile defines a constant phase gradient across the surface,
    representing a simple transmission or reflection grating.

    The phase is defined as:
    φ(x, y) = K_x * x + K_y * y
    where (K_x, K_y) is the grating wavevector.

    The grating vector is defined by its magnitude |K| = 2π / period and its
    angle (theta) relative to the positive x-axis.
    K_x = (2π / period) * cos(angle)
    K_y = (2π / period) * sin(angle)

    Args:
        period (float): The spatial period of the grating in mm. Must be positive.
        angle (float, optional): The angle of the grating's wavevector
            (direction of phase gradient) in radians, measured
            counter-clockwise from the positive x-axis. Defaults to 0,
            which creates grooves parallel to the y-axis.
        order (int, optional): The diffraction order. Defaults to 1.
        efficiency (float, optional): The diffraction efficiency for the
            specified order, between 0 and 1. Defaults to 1.0.
    """

    phase_type = "linear_grating"

    def __init__(
        self, period: float, angle: float = 0.0, order: int = 1, efficiency: float = 1.0
    ):
        super().__init__()
        if period <= 0:
            raise ValueError("Grating period must be positive.")
        if not (0.0 <= efficiency <= 1.0):
            raise ValueError("Efficiency must be between 0 and 1.")
        self.period = period
        self.angle = angle
        self.order = order
        self._efficiency = efficiency

        # Pre-calculate the constant gradient components
        K = self.order * 2 * be.pi / self.period
        self._K_x = K * be.cos(self.angle)
        self._K_y = K * be.sin(self.angle)

    @property
    def efficiency(self) -> float:
        return self._efficiency

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
        return self._K_x * x + self._K_y * y

    def get_gradient(
        self, x: be.Array, y: be.Array, wavelength: be.Array = None
    ) -> tuple[be.Array, be.Array, be.Array]:
        """Calculates the gradient of the phase at coordinates (x, y).

        For a linear grating, the gradient (d_phi/dx, d_phi/dy) is constant,
        and d_phi/dz is zero.

        Args:
            x: The x-coordinates of the points of interest. Used for shape.
            y: The y-coordinates of the points of interest. Used for shape.

        Returns:
            A tuple containing the x, y, and z components of the phase
            gradient (K_x, K_y, 0), broadcast to the shape of the input
            coordinates.
        """
        phi_x = be.full_like(x, self._K_x)
        phi_y = be.full_like(y, self._K_y)
        phi_z = be.zeros_like(x)
        return phi_x, phi_y, phi_z

    def get_paraxial_gradient(self, y: be.Array) -> be.Array:
        """Calculates the paraxial phase gradient at y-coordinate.

        This is the gradient d_phi/dy evaluated at x=0. For a linear
        grating, this is constant and equal to K_y.

        Args:
            y: The y-coordinates of the points of interest. Used for shape.

        Returns:
            The paraxial phase gradient (K_y) at each y-coordinate.
        """
        return be.full_like(y, self._K_y)

    def to_dict(self) -> dict:
        """Serializes the phase profile to a dictionary.

        Returns:
            A dictionary representation of the phase profile.
        """
        data = super().to_dict()
        data.update(
            {
                "period": self.period,
                "angle": self.angle,
                "order": self.order,
                "efficiency": self.efficiency,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict) -> LinearGratingPhaseProfile:
        """Deserializes a phase profile from a dictionary.

        Args:
            data: A dictionary representation of a phase profile.

        Returns:
            An instance of a `LinearGratingPhaseProfile`.
        """
        return cls(
            period=data["period"],
            angle=data.get("angle", 0.0),
            order=data.get("order", 1),
            efficiency=data.get("efficiency", 1.0),
        )
