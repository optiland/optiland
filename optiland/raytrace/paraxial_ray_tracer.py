"""Paraxial Ray Tracer Module

This module contains the ParaxialRayTracer class, which is responsible for tracing
paraxial rays through an optical system.

Kramer Harrison, 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.rays.paraxial_rays import ParaxialRays
from optiland.raytrace.base import BaseRayTracer
from optiland.surfaces import ObjectSurface

if TYPE_CHECKING:
    from optiland._types import BEArray, ScalarOrArray
    from optiland.optic import Optic


class ParaxialRayTracer(BaseRayTracer):
    """Class to trace paraxial rays through an optical system"""

    def __init__(self, optic: Optic):
        """Initializes a ParaxialRayTracer instance.

        Args:
            optic: The optical system to be traced.
        """
        super().__init__(optic)

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

        y0, z0 = self.optic.fields.require_definition().get_paraxial_object_position(
            self.optic, Hy, y1, EPL
        )
        # z0 is a global z (object frame); use the global entrance-pupil z so
        # both terms share a frame. EPL above stays relative — that is what
        # get_paraxial_object_position expects.
        epl_global = self.optic.paraxial.entrance_pupil_z()
        u0 = (y1 - y0) / (epl_global - z0)
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

        path = self.optic.surfaces.build_paraxial_path()

        R = self.optic.surfaces.radii
        n = self.optic.surfaces.n(wavelength)
        pos = be.ravel(path.axial_positions)
        surfs = self.optic.surfaces

        if path.is_folded_or_off_axis:
            # The scalar folded model is only defined on its supported
            # domain; reject anything outside it rather than returning
            # plausible numbers.
            path.require_scalar_paraxial("paraxial ray tracing")
            # A powered surface on an odd-parity leg, or one authored with
            # its local +z against the beam, needs its paraxial power sign
            # corrected: R_eff = parity * sgn(z_axis . d_in) * R_authored.
            # Authored radii are never mutated; infinities are preserved to
            # keep +inf/-inf output unchanged. Real-ray geometry is
            # untouched.
            sign = path.orientation_sign_array
            R = be.where(be.isfinite(R), sign * R, R)
            f_signs = [float(s) for s in path.orientation_sign]
        else:
            # Canonical +/-z chains always satisfy
            # parity * sgn(z_axis . d_in) = +1, so the authored values are
            # already the effective ones -- kept bit-for-bit.
            f_signs = [1.0] * len(self.optic.surfaces.surfaces)

        if reverse:
            # The reverse transform (flip order, negate radii, mirror
            # positions, roll indices) is a pure 1-D map of the forward
            # paraxial system, so it applies to the orientation-corrected
            # effective values exactly as it did to the authored ones --
            # the orientation sign must not be applied a second time.
            R = -be.flip(R)
            n = be.roll(n, shift=1)
            n = be.flip(n)
            pos = pos[-1] - be.flip(pos)
            surfs = surfs[::-1]
            f_signs = f_signs[::-1]

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
                    f = f_signs[k] * surfs[k].interaction_model.f
                    f = -f if reverse else f
                    u_ = -u_ - y_ / f
                else:
                    u_ = -u_ - 2 * y_ / R[k]
            else:
                if surfs[k].surface_type == "paraxial":
                    f = f_signs[k] * surfs[k].interaction_model.f
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
