"""Paraxial Ray Tracer Module

This module contains the ParaxialRayTracer class, which is responsible for tracing
paraxial rays through an optical system.

Kramer Harrison, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.rays.paraxial_rays import ParaxialRays
from optiland.raytrace.base import BaseRayTracer
from optiland.surfaces import ObjectSurface

if TYPE_CHECKING:
    from optiland._types import BEArray, ScalarOrArray
    from optiland.optic import Optic
    from optiland.paraxial_path import ParaxialPath


@dataclass(frozen=True)
class _ScalarParaxialSequence:
    """Validated, orientation-corrected inputs of one scalar first-order op.

    This is the single source of truth shared by the explicit paraxial
    tracer (:meth:`ParaxialRayTracer.trace_generic`) and the ray-transfer
    matrix assembly (:meth:`optiland.paraxial.Paraxial.ray_transfer_matrix`),
    so the two public first-order APIs can never disagree on the same
    prescription because they selected different power conventions.

    Attributes:
        path: The :class:`~optiland.paraxial_path.ParaxialPath` snapshot the
            sequence was built from.
        positions: Signed unfolded axial surface positions (1-D).
        radii: Paraxial-effective radii of curvature: authored radii with
            the collinear orientation sign applied (infinities preserved
            exactly). Authored geometry is never mutated.
        focal_signs: Per-surface orientation sign to apply to an explicit
            paraxial surface's focal length.
        refractive_indices: Refractive index following each surface.
        surfaces: The surface objects, in trace order.
    """

    path: ParaxialPath
    positions: BEArray
    radii: BEArray
    focal_signs: tuple
    refractive_indices: BEArray
    surfaces: tuple


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
        path = self.optic.surfaces.build_paraxial_path()
        EPL = self.optic.paraxial.EPL(path=path)
        EPD = self.optic.paraxial.EPD()

        y1 = Py * EPD / 2

        y0, z0 = self.optic.fields.require_definition().get_paraxial_object_position(
            self.optic, Hy, y1, EPL
        )
        # z0 is a global z (object frame); use the global entrance-pupil z so
        # both terms share a frame. EPL above stays relative — that is what
        # get_paraxial_object_position expects.
        epl_global = self.optic.paraxial.entrance_pupil_axial_position(path=path)
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
        path: ParaxialPath | None = None,
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
            path: Optional prebuilt :class:`ParaxialPath` for the current
                geometry, so a high-level operation making several traces
                pays the path construction once. Must be a fresh snapshot
                of the surfaces being traced -- never one built before a
                geometry mutation.

        Returns:
            tuple: A tuple containing the final height(s) and slope(s) of the
                rays after tracing.
        """
        y_ = self._process_input(y)
        u_ = self._process_input(u)
        z_ = self._process_input(z)

        sequence = self.prepare_scalar_sequence(
            wavelength,
            path=path,
            reverse=reverse,
            operation="paraxial ray tracing",
        )
        R = sequence.radii
        n = sequence.refractive_indices
        pos = sequence.positions
        surfs = sequence.surfaces
        f_signs = sequence.focal_signs

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

    def prepare_scalar_sequence(
        self,
        wavelength: float,
        *,
        path: ParaxialPath | None = None,
        reverse: bool = False,
        operation: str = "scalar paraxial analysis",
    ) -> _ScalarParaxialSequence:
        """Build the validated scalar sequence shared by all first-order APIs.

        Exactly one of these preparation steps runs per scalar first-order
        operation (explicit trace or matrix assembly):

        1. a fresh :class:`~optiland.paraxial_path.ParaxialPath` is built
           when none is supplied;
        2. folded/off-axis paths are validated against the supported scalar
           domain (``path.require_scalar_paraxial``);
        3. straight paths surface their scalar-approximation advisories
           (``path.warn_scalar_approximations``);
        4. authored radii and explicit focal lengths of centered/collinear
           powered surfaces are mapped to paraxial-effective values via the
           collinear orientation policy
           (``path.effective_orientation_signs``), preserving radius
           infinities exactly -- genuinely oblique surfaces are never
           re-signed by a heuristic;
        5. the reverse transformation is applied exactly once when
           requested.

        Args:
            wavelength: Wavelength in micrometers for the refractive indices.
            path: Optional prebuilt path snapshot to reuse. Must be fresh
                for the surfaces being traced.
            reverse: Whether to produce the reversed (image-to-object)
                sequence.
            operation: Operation name used in diagnostics and warnings.

        Returns:
            The assembled :class:`_ScalarParaxialSequence`.

        Raises:
            UnsupportedParaxialGeometryError: If the geometry lies outside
                the supported scalar folded domain (and no seed scope is
                active).
        """
        surface_group = self.optic.surfaces
        if path is None:
            path = surface_group.build_paraxial_path()

        R = surface_group.radii
        n = surface_group.n(wavelength)
        pos = be.ravel(path.axial_positions)
        surfaces = tuple(surface_group.surfaces)

        if path.is_folded_or_off_axis:
            # The scalar folded model is only defined on its supported
            # domain; reject anything outside it rather than returning
            # plausible numbers.
            path.require_scalar_paraxial(operation)
        else:
            # Tilted or decentered surfaces on a straight chain are still
            # ignored by the scalar model (historical behavior); surface
            # that as a warning rather than staying silent.
            path.warn_scalar_approximations(operation)

        # A powered surface on an odd-parity leg, or one authored with its
        # local +z against the beam, needs its paraxial power sign
        # corrected: R_eff = parity * sgn(z_axis . d_in) * R_authored --
        # applied to every centered/collinear surface, on straight and
        # folded paths alike (see effective_orientation_signs for the
        # oblique-surface policy). Authored radii are never mutated;
        # infinities are preserved to keep +inf/-inf output unchanged.
        # Real-ray geometry is untouched. The canonical default authoring
        # has every sign equal to +1, so the sign application is skipped
        # entirely and the authored values pass through bit-for-bit.
        f_signs = tuple(float(s) for s in path.effective_orientation_signs())
        if not f_signs:
            f_signs = (1.0,) * len(surfaces)
        if any(s != 1.0 for s in f_signs):
            sign = be.array(list(f_signs))
            R = be.where(be.isfinite(R), sign * R, R)

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
            surfaces = surfaces[::-1]
            f_signs = f_signs[::-1]

        return _ScalarParaxialSequence(
            path=path,
            positions=pos,
            radii=R,
            focal_signs=f_signs,
            refractive_indices=n,
            surfaces=surfaces,
        )

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
