"""Curvature Solve Module

This module defines `CurvatureSolve`, an abstract base class for solves
that adjust surface curvature (radius) to satisfy a condition.

Kramer Harrison, 2026
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from optiland.solves.base import BaseSolve


class CurvatureSolve(BaseSolve, ABC):
    """Abstract base class for curvature solves.

    This class provides the common structure for solves that aim to satisfy
    a condition by adjusting the curvature of a specific surface.

    Attributes:
        optic (Optic): The optic object.
        surface_idx (int): The index of the surface where the curvature
            is to be controlled.
    """

    def __init__(self, optic, surface_idx: int):
        """Initializes a CurvatureSolve object.

        Args:
            optic (Optic): The optic object.
            surface_idx (int): The index of the surface.
        """
        if surface_idx is None:
            raise ValueError("'surface_idx' argument must be provided.")
        super().__init__()
        self.optic = optic
        self.surface_idx = surface_idx

    @abstractmethod
    def apply(self):
        """Applies the curvature solve to the optic."""
        pass  # pragma: no cover

    def _set_curvature(self, c):
        """Set surface curvature through the optic updater."""
        radius = 1.0 / c if c != 0 else float("inf")
        self.optic.updater.set_radius(radius, self.surface_idx)

    def to_dict(self):
        """Returns a dictionary representation of the solve."""
        solve_dict = super().to_dict()
        solve_dict.update(
            {
                "surface_idx": self.surface_idx,
            }
        )
        return solve_dict

    @classmethod
    def from_dict(cls, optic, data):
        """Creates a solve instance from a dictionary representation."""
        if cls is CurvatureSolve:
            raise TypeError(
                "CurvatureSolve is an abstract class and cannot be "
                "instantiated directly."
            )

        if cls is MarginalRayAngleCurvatureSolve or cls is ChiefRayAngleCurvatureSolve:
            return cls(optic, data["surface_idx"], data["angle"])

        raise NotImplementedError(f"from_dict not implemented for {cls.__name__}")


class MarginalRayAngleCurvatureSolve(CurvatureSolve):
    """Adjusts surface curvature to achieve a target marginal ray exit angle.

    This solve uses the paraxial refraction equation:
        n'u' - nu = -y(n' - n)c

    to solve for curvature c:
        c = (nu - n'u') / (y(n' - n))

    Attributes:
        optic (Optic): The optic object.
        surface_idx (int): The index of the surface.
        angle (float): The target exit angle (u') of the marginal ray.
    """

    def __init__(self, optic, surface_idx: int, angle: float):
        """Initializes a MarginalRayAngleSolve object.

        Args:
            optic (Optic): The optic object.
            surface_idx (int): The index of the surface.
            angle (float): The target exit angle.
        """
        super().__init__(optic, surface_idx)
        self.angle = angle

    def apply(self):
        """Applies the marginal ray angle solve."""
        y, u = self.optic.paraxial.marginal_ray()

        u_in = u[0] if self.surface_idx == 0 else u[self.surface_idx - 1]

        y_surf = y[self.surface_idx]
        u_out_target = self.angle

        # Indices of refraction
        if self.surface_idx == 0:
            n_pre = self.optic.surfaces[self.surface_idx - 1].material_post.n(
                self.optic.primary_wavelength
            )
        else:
            n_pre = self.optic.surfaces[self.surface_idx - 1].material_post.n(
                self.optic.primary_wavelength
            )

        n_post = self.optic.surfaces[self.surface_idx].material_post.n(
            self.optic.primary_wavelength
        )

        # Delta n
        delta_n = n_post - n_pre

        if delta_n == 0:
            return

        if y_surf == 0:
            return

        # Solve for c
        # n'u' - nu = -y * delta_n * c
        # c = (nu - n'u') / (y * delta_n)

        num = (n_pre * u_in) - (n_post * u_out_target)
        den = y_surf * delta_n
        c = (num / den).item()

        self._set_curvature(c)

    def to_dict(self):
        """Returns a dictionary representation of the solve."""
        solve_dict = super().to_dict()
        solve_dict.update(
            {
                "angle": self.angle,
            }
        )
        return solve_dict


class ChiefRayAngleCurvatureSolve(CurvatureSolve):
    """Adjusts surface curvature to achieve a target chief ray exit angle.

    This solve uses the paraxial refraction equation:
        n'u' - nu = -y(n' - n)c

    to solve for curvature c:
        c = (nu - n'u') / (y(n' - n))

    Attributes:
        optic (Optic): The optic object.
        surface_idx (int): The index of the surface.
        angle (float): The target exit angle (u') of the chief ray.
    """

    def __init__(self, optic, surface_idx: int, angle: float):
        """Initializes a ChiefRayAngleSolve object.

        Args:
            optic (Optic): The optic object.
            surface_idx (int): The index of the surface.
            angle (float): The target exit angle.
        """
        super().__init__(optic, surface_idx)
        self.angle = angle

    def apply(self):
        """Applies the chief ray angle solve.

        Since changing the system affects the chief ray path (it must pass
        through the stop), this solve is iterative.
        """
        for _ in range(50):
            y, u = self.optic.paraxial.chief_ray()

            # u[i] is the slope AFTER surface i.
            # Therefore, the slope incident on surface i is u[i-1].
            u_in = u[0] if self.surface_idx == 0 else u[self.surface_idx - 1]

            y_surf = y[self.surface_idx]
            u_out_target = self.angle

            # Check if we are already close enough
            if (
                self.surface_idx < len(u)
                and abs(u[self.surface_idx] - u_out_target) < 1e-5
            ):
                return

            # Indices of refraction
            if self.surface_idx == 0:
                n_pre = self.optic.surfaces[self.surface_idx - 1].material_post.n(
                    self.optic.primary_wavelength
                )
            else:
                n_pre = self.optic.surfaces[self.surface_idx - 1].material_post.n(
                    self.optic.primary_wavelength
                )

            n_post = self.optic.surfaces[self.surface_idx].material_post.n(
                self.optic.primary_wavelength
            )

            # Delta n
            delta_n = n_post - n_pre

            if delta_n == 0:
                return

            if y_surf == 0:
                return

            # Solve for new c target
            # n'u' - nu = -y * delta_n * c
            # c = (nu - n'u') / (y * delta_n)
            num = (n_pre * u_in) - (n_post * u_out_target)
            den = y_surf * delta_n
            c_target = (num / den).item()

            # Get current curvature
            if hasattr(self.optic.surfaces[self.surface_idx].geometry, "radius"):
                r = self.optic.surfaces[self.surface_idx].geometry.radius
                c_current = 1.0 / r if r != 0 else 0.0
            else:
                return

            # Damping
            damping = 0.5
            c = (1 - damping) * c_current + damping * c_target

            self._set_curvature(c)

    def to_dict(self):
        """Returns a dictionary representation of the solve."""
        solve_dict = super().to_dict()
        solve_dict.update(
            {
                "angle": self.angle,
            }
        )
        return solve_dict
