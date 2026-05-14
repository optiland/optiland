"""Paraxial Ray Tracer Module

This module contains the ParaxialRayTracer class, which is responsible for tracing
paraxial rays through an optical system.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.rays.paraxial_rays import ParaxialRays
from optiland.surfaces import ObjectSurface

if TYPE_CHECKING:
    from optiland._types import BEArray, ScalarOrArray
    from optiland.optic import Optic


class ParaxialRayTracer:
    """Class to trace paraxial rays through an optical system"""

    def __init__(self, optic: Optic):
        """Initializes a ParaxialRayTracer instance.

        Args:
            optic: The optical system to be traced.
        """
        self.optic = optic

    def trace(self, Hy: ScalarOrArray, Py: ScalarOrArray, wavelength: ScalarOrArray):
        """Trace paraxial ray through the optical system based on specified field
        and pupil coordinates.

        Args:
            Hy: Normalized field coordinate.
            Py: Normalized pupil coordinate.
            wavelength: Wavelength of the light.

        """
        EPL = self.optic.paraxial.EPL()
        EPD = self.optic.paraxial.EPD()

        y1 = Py * EPD / 2

        if self.optic.fields.field_definition is None:
            raise ValueError()
        y0, z0 = self.optic.fields.field_definition.get_paraxial_object_position(
            self.optic, Hy, y1, EPL
        )
        u0 = (y1 - y0) / (EPL - z0)
        rays = ParaxialRays(y0, u0, z0, wavelength)

        self.optic.surfaces.trace(rays)

    def trace_generic(
        self,
        y: BEArray | float,
        u: BEArray | float,
        z: BEArray | float,
        wavelength: float,
        reverse: bool = False,
        skip: int = 0,
    ) -> tuple[BEArray, BEArray]:
        """
        Trace generically-defined paraxial rays through the optical system.

        Args:
            y: The initial height(s) of the rays.
            u: The initial slope(s) of the rays.
            z: The initial axial position(s) of the rays.
            wavelength: The wavelength of the rays.
            reverse: If True, trace the rays in reverse
                direction. Defaults to False.
            skip: The number of surfaces to skip during
                tracing. Defaults to 0.

        Returns:
            tuple: A tuple containing the final height(s) and slope(s) of the
                rays after tracing.
        """
        y_ = self._process_input(y)
        u_ = self._process_input(u)
        z_ = self._process_input(z)

        R = self.optic.surfaces.radii
        n = self.optic.surfaces.n(wavelength)
        pos = be.ravel(self.optic.surfaces.positions)
        surfs = self.optic.surfaces

        if reverse:
            R = -be.flip(R)
            n = be.roll(n, shift=1)
            n = be.flip(n)
            pos = pos[-1] - be.flip(pos)
            surfs = surfs[::-1]

        power = be.diff(n, prepend=be.array([n[0]])) / R

        heights = []
        slopes = []

        for k in range(skip, len(R)):
            if isinstance(surfs[k], ObjectSurface):
                heights.append(be.copy(y_))
                slopes.append(be.copy(u_))
                continue

            # propagate to surface
            t = pos[k] - z_
            z_ = pos[k]
            y_ = y_ + t * u_

            # reflect or refract
            if surfs[k].interaction_model.is_reflective:
                if surfs[k].surface_type == "paraxial":
                    f = (
                        -surfs[k].interaction_model.f
                        if reverse
                        else surfs[k].interaction_model.f
                    )
                    u_ = -u_ - y_ / f
                else:
                    u_ = -u_ - 2 * y_ / R[k]
            else:
                if surfs[k].surface_type == "paraxial":
                    f = surfs[k].interaction_model.f
                    u_ = (n[k - 1] * u_ - y_ / f) / n[k]
                else:
                    u_ = (n[k - 1] * u_ - y_ * power[k]) / n[k]

            heights.append(be.copy(y_))
            slopes.append(be.copy(u_))

        heights = be.array(heights).reshape(-1, 1)
        slopes = be.array(slopes).reshape(-1, 1)

        return heights, slopes

    def _process_input(self, x: BEArray | float) -> BEArray:
        """
        Process input to ensure it is a numpy array.

        Args:
            x (float or array-like): The input to process.

        Returns:
            np.ndarray: The processed input.
        """
        if isinstance(x, int | float):
            return be.array([x])
        else:
            return be.array(x)
