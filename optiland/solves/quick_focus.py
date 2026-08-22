"""Quick Focus Solve Module

Defines the quick focus solve.

Seçkin Berkay Öztürk, 2025
"""

from __future__ import annotations

import optiland.backend as be
from optiland.solves.base import BaseSolve


class QuickFocusSolve(BaseSolve):
    """Quick Focus
    Args:
        optic (Optic): The optic object.

    Raises:
            ValueError: If the optical system is not defined.

    """

    def __init__(self, optic, *args):
        self.optic = optic
        self.num_surfaces = self.optic.surfaces.num_surfaces
        if self.num_surfaces <= 2:
            raise ValueError("Can not optimize for an empty optical system")

    def optimal_focus_distance(
        self,
        Hx=0,
        Hy=0,
        wavelength=0.55,
        num_rays=5,
        distribution="hexapolar",
    ):
        """Compute the optimal location of the image plane where the RMS spot
        size is minimized. This is based on solving the quadratic equation
        that describes the RMS spot size as a function of the propagation
        distance.

        Args:
            Hx (float): The normalized x field.
            Hy (float): The normalized y field.
            wavelength (float): The wavelength of the light.
            num_rays (int): The number of rays to trace.
            distribution (str): The distribution of rays to trace.

        Returns:
            float: The optimal axial position (z-coordinate) of the image plane
                that minimizes the RMS spot size.

        Raises:
            UnsupportedParaxialGeometryError: If the beam path is folded off
                global +z (or entered along another direction), before any
                ray is traced. The returned value is a Cartesian global-z
                focus coordinate, which is not frame-correct for a folded
                output arm -- a caller writing it into ``cs.z`` would move
                the image plane off its physical leg.
        """
        from optiland.paraxial_path import require_global_z_geometry

        require_global_z_geometry(
            self.optic.surfaces, "QuickFocusSolve.optimal_focus_distance"
        )

        rays = self.optic.trace(
            Hx=Hx,
            Hy=Hy,
            wavelength=wavelength,
            num_rays=num_rays,
            distribution=distribution,
        )

        A = rays.L**2 + rays.M**2
        B = rays.L * rays.x + rays.M * rays.y
        with be.errstate(divide="ignore", invalid="ignore"):
            t_opt = be.where(A != 0, -B / A, be.nan)
        z_focus = be.nanmean(rays.z + t_opt * rays.N)

        return z_focus

    def apply(self):
        """Applies the QuickFocusSolve to the optic.

        This method calculates the optimal focus distance and sets the
        z-position of the last surface (image plane) accordingly.

        Raises:
            UnsupportedParaxialGeometryError: If the beam path is folded off
                global +z, before any mutation -- moving the image plane
                along z only would take it off its physical leg.
        """
        from optiland.paraxial_path import require_global_z_geometry

        require_global_z_geometry(self.optic.surfaces, "QuickFocusSolve")

        z_focus = self.optimal_focus_distance(
            wavelength=self.optic.wavelengths.primary_wavelength.value,
        )

        self.optic.surfaces[-1].geometry.cs.z = z_focus
