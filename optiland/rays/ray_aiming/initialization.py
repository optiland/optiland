"""Ray Aiming Initialization Module

This module implements the initialization logic for determining the physical
aperture stop size (Stop Radius) before the main Ray Aiming iteration begins.
It uses a Strategy Pattern to handle different ways of calculating this radius.

Kramer Harrison, 2025
"""

from __future__ import annotations

import abc
import warnings
from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.aperture import FloatByStopAperture
from optiland.rays import RealRays
from optiland.rays.ray_aiming.pupil_map import to_float

if TYPE_CHECKING:
    from optiland.optic import Optic


class StopSizeStrategy(abc.ABC):
    """Abstract base class for stop size determination strategies."""

    def __init__(self, optic: Optic):
        self.optic = optic

    @abc.abstractmethod
    def calculate_stop_radius(self) -> float:
        """Calculate the radius of the stop surface."""
        pass


class FloatByStopStrategy(StopSizeStrategy):
    """Strategy for 'Float By Stop Size' aperture type.

    Simply returns the user-defined semi-diameter of the Stop Surface.
    """

    def calculate_stop_radius(self) -> float:
        stop_index = self.optic.surfaces.stop_index
        surface = self.optic.surfaces[stop_index]

        # Check for explicit aperture object first
        if surface.aperture and hasattr(surface.aperture, "r_max"):
            return surface.aperture.r_max
        elif surface.aperture and hasattr(surface.aperture, "x_max"):
            return (surface.aperture.x_max + surface.aperture.y_max) / 2.0

        # Fallback to semi_aperture attribute
        return float(surface.semi_aperture)


class ParaxialReferenceStrategy(StopSizeStrategy):
    """Strategy using paraxial ray trace to determine stop radius.

    Traces a Paraxial Marginal Ray from the center of the object to the
    Stop Surface.
    """

    def calculate_stop_radius(self) -> float:
        stop_index = self.optic.surfaces.stop_index
        para = self.optic.paraxial

        # Determine marginal ray height at the stop surface
        y_marginal, _ = para.marginal_ray()
        return float(be.abs(y_marginal[stop_index].item()))


class RealReferenceStrategy(StopSizeStrategy):
    """Strategy using real ray trace to determine stop radius.

    Traces a Real Ray from the center of the object toward the edge of the
    Entrance Pupil. Fallbacks to ParaxialReferenceStrategy on failure.
    """

    def calculate_stop_radius(self) -> float:
        try:
            return self._trace_real_marginal_ray()
        except Exception as e:
            warnings.warn(
                f"RealReferenceStrategy failed: {e}. "
                "Falling back to ParaxialReferenceStrategy.",
                stacklevel=2,
            )
            fallback = ParaxialReferenceStrategy(self.optic)
            return fallback.calculate_stop_radius()

    def _trace_real_marginal_ray(self) -> float:
        wavelength = self.optic.primary_wavelength
        EPL = float(self.optic.paraxial.entrance_pupil_axial_position())
        EPD = float(self.optic.paraxial.EPD())

        stop_index = self.optic.surfaces.stop_index
        obj_surf = self.optic.object_surface
        is_inf = bool(obj_surf and obj_surf.is_infinite)

        frame = self._entry_frame_floats()

        # The entrance pupil is the image of the stop, so a decentered stop has
        # a decentered entrance pupil. Offsetting by EPD/2 from the axis would
        # then launch a ray that never goes near the pupil, and the height it
        # lands at is a distance from the stop center that has nothing to do
        # with the stop radius (issue #654). Locate the pupil center first and
        # offset from there.
        cx, cy, cz, cL, cM, cN = self._solve_pupil_center(stop_index, is_inf, frame)

        if frame is None:
            if is_inf:
                # Object-space rays run parallel to the axis, so the launch
                # height is the entrance-pupil coordinate and can be offset
                # directly.
                origin = (cx, cy + EPD / 2.0, cz)
                direction = (cL, cM, cN)
            else:
                # Offset the chief ray's pupil crossing, then re-aim at it
                # from the same object point.
                t = (EPL - cz) / cN
                target_x = cx + cL * t
                target_y = cy + cM * t + EPD / 2.0

                dx = target_x - cx
                dy = target_y - cy
                dz = EPL - cz
                mag = float(be.sqrt(be.array(dx**2 + dy**2 + dz**2)))
                origin = (cx, cy, cz)
                direction = (dx / mag, dy / mag, dz / mag)
        else:
            # Frame-aware branch: the marginal probe is displaced from the
            # pupil-center reference along the entry frame's meridional
            # transverse axis v -- the generalization of the historical +y
            # offset -- so no global +z assumption remains.
            _anchor, axial, d0, _u0, v0 = frame
            if is_inf:
                origin = (
                    cx + EPD / 2.0 * v0[0],
                    cy + EPD / 2.0 * v0[1],
                    cz + EPD / 2.0 * v0[2],
                )
                direction = (cL, cM, cN)
            else:
                # Push the chief launch to the pupil plane (the plane through
                # the apparent pupil point, perpendicular to the entry
                # direction), offset there by EPD/2 along v, and re-aim.
                r_ep = tuple(_anchor[i] + (EPL - axial) * d0[i] for i in range(3))
                c_pos = (cx, cy, cz)
                c_dir = (cL, cM, cN)
                denom = sum(c_dir[i] * d0[i] for i in range(3))
                t = sum((r_ep[i] - c_pos[i]) * d0[i] for i in range(3)) / denom
                target = tuple(
                    c_pos[i] + t * c_dir[i] + EPD / 2.0 * v0[i] for i in range(3)
                )
                delta = tuple(target[i] - c_pos[i] for i in range(3))
                mag = float(be.sqrt(be.array(sum(d**2 for d in delta))))
                origin = c_pos
                direction = tuple(d / mag for d in delta)

        rays = RealRays(
            x=be.array([origin[0]]),
            y=be.array([origin[1]]),
            z=be.array([origin[2]]),
            L=be.array([direction[0]]),
            M=be.array([direction[1]]),
            N=be.array([direction[2]]),
            wavelength=be.array([wavelength]),
            intensity=be.array([1.0]),
        )

        # Trace from the first surface up to the stop surface
        for i in range(1, stop_index + 1):
            self.optic.surfaces[i].trace(rays)
            if be.any(be.isnan(rays.x)):
                raise ValueError("Ray trace resulted in NaNs (TIR or missed surface).")

        # Localize rays to the stop surface's local frame so the radial height
        # is measured from the stop center, not the global origin. The pupil
        # center solve puts the chief ray on that origin, so the marginal ray's
        # local radius is its separation from the chief ray.
        stop_cs = self.optic.surfaces[stop_index].geometry.cs
        local_rays = RealRays(
            be.copy(rays.x),
            be.copy(rays.y),
            be.copy(rays.z),
            be.copy(rays.L),
            be.copy(rays.M),
            be.copy(rays.N),
            intensity=be.copy(rays.i),
            wavelength=rays.w,
        )
        stop_cs.localize(local_rays)

        # Return intersection radial height at Stop in local coords
        return float(be.sqrt(local_rays.x[0] ** 2 + local_rays.y[0] ** 2))

    def _entry_frame_floats(self) -> tuple | None:
        """The optic's entry frame with components as plain floats.

        ``None`` on the canonical global-z path, keeping the historical
        code branch (and its exact arithmetic) for legacy systems.
        """
        frame = self.optic.surfaces._entry_frame()
        if frame is None:
            return None
        anchor, axial, d0, u0, v0 = frame
        as_floats = tuple(
            tuple(to_float(component) for component in vector)
            for vector in (anchor, d0, u0, v0)
        )
        return as_floats[0], to_float(axial), as_floats[1], as_floats[2], as_floats[3]

    def _solve_pupil_center(
        self, stop_index: int, is_inf: bool, frame: tuple | None = None
    ) -> tuple:
        """Solve the launch state of the axial ray landing on the stop center.

        For a centered system this is the on-axis launch and the solve is a
        no-op, but for a decentered or tilted stop it is what locates the
        entrance pupil, which is the reference the marginal probe is offset
        from.

        Args:
            stop_index: Index of the stop surface.
            is_inf: Whether the object is at infinity.
            frame: Entry frame as plain floats (see
                :meth:`_entry_frame_floats`), or ``None`` on the canonical
                global-z path.

        Returns:
            tuple: ``(x, y, z, L, M, N)`` launch state of the axial ray that
            lands on the stop surface's local origin.

        Raises:
            ValueError: If the solve does not converge.
        """
        # Imported here: iterative.py pulls this module in at call time, so a
        # module-level import would be circular.
        from optiland.rays.ray_aiming.iterative import IterativeRayAimer

        wavelength = self.optic.primary_wavelength
        EPL = float(self.optic.paraxial.entrance_pupil_axial_position())

        if frame is None:
            # First-order seed: the pupil is the image of the stop, so the
            # stop's decenter maps to the pupil scaled by the pupil
            # magnification. Exact for a centered system ahead of the stop;
            # elsewhere the Newton polish below carries it the rest of the
            # way.
            m = self._pupil_magnification()
            stop_cs = self.optic.surfaces[stop_index].geometry.cs
            gx, gy, _gz = stop_cs.position_in_gcs
            ex = to_float(gx) * m
            ey = to_float(gy) * m

            if is_inf:
                # Launch well ahead of surface 1 for robust intersection.
                z_start = to_float(self.optic.surfaces[1].geometry.cs.z) - 100.0
                x0, y0, z0 = be.array([ex]), be.array([ey]), be.array([z_start])
                L0, M0, N0 = be.array([0.0]), be.array([0.0]), be.array([1.0])
            else:
                obj_z = to_float(self.optic.object_surface.geometry.cs.z)
                dz = EPL - obj_z
                mag = float(be.sqrt(be.array(ex**2 + ey**2 + dz**2)))
                x0, y0, z0 = be.array([0.0]), be.array([0.0]), be.array([obj_z])
                L0 = be.array([ex / mag])
                M0 = be.array([ey / mag])
                N0 = be.array([dz / mag])
        else:
            # Frame-aware seeds: probe origins sit on the entry line, a
            # chosen axial distance before the first surface along -d0, and
            # launch along d0 (infinite) or from the object vertex toward
            # the apparent entrance-pupil point (finite). The supported
            # folded domain is piecewise-centered, so no decenter seed is
            # needed; the Newton polish absorbs any residual.
            anchor, axial, d0, _u0, _v0 = frame
            r_ep = tuple(anchor[i] + (EPL - axial) * d0[i] for i in range(3))
            if is_inf:
                origin = tuple(r_ep[i] - 100.0 * d0[i] for i in range(3))
                x0 = be.array([origin[0]])
                y0 = be.array([origin[1]])
                z0 = be.array([origin[2]])
                L0 = be.array([d0[0]])
                M0 = be.array([d0[1]])
                N0 = be.array([d0[2]])
            else:
                obj_cs = self.optic.object_surface.geometry.cs
                obj_pos = tuple(to_float(c) for c in obj_cs.position_in_gcs)
                delta = tuple(r_ep[i] - obj_pos[i] for i in range(3))
                mag = float(be.sqrt(be.array(sum(d**2 for d in delta))))
                x0 = be.array([obj_pos[0]])
                y0 = be.array([obj_pos[1]])
                z0 = be.array([obj_pos[2]])
                L0 = be.array([delta[0] / mag])
                M0 = be.array([delta[1] / mag])
                N0 = be.array([delta[2] / mag])

        zero = be.array([0.0])
        aimer = IterativeRayAimer(self.optic)
        x, y, z, L, M, N, converged, _, _report = aimer._solve_core(
            x0,
            y0,
            z0,
            L0,
            M0,
            N0,
            be.array([wavelength]),
            stop_index,
            is_inf,
            zero,
            zero,
        )
        if not bool(be.all(converged)):
            raise ValueError("Could not solve for the entrance pupil center.")

        return (
            to_float(x),
            to_float(y),
            to_float(z),
            to_float(L),
            to_float(M),
            to_float(N),
        )

    def _pupil_magnification(self) -> float:
        """Signed paraxial magnification from the stop plane to the pupil plane.

        Returns:
            float: ``(EPD / 2) / y_stop``, where ``y_stop`` is the paraxial
            marginal ray height at the stop.

        Raises:
            ValueError: If the paraxial marginal ray height at the stop is zero.
        """
        para = self.optic.paraxial
        y_marginal, _ = para.marginal_ray()
        y_stop = to_float(y_marginal[self.optic.surfaces.stop_index])
        if abs(y_stop) < 1e-12:
            raise ValueError("Paraxial marginal ray height at the stop is zero.")
        return float(para.EPD()) / 2.0 / y_stop


def get_stop_radius_strategy(optic: Optic, aiming_mode: str) -> StopSizeStrategy:
    """Factory function to select the appropriate stop size strategy.

    Args:
        optic: The optical system.
        aiming_mode: The ray aiming mode ('paraxial', 'iterative', 'robust').

    Returns:
        The instantiated strategy instance.
    """
    if optic.aperture and isinstance(optic.aperture, FloatByStopAperture):
        return FloatByStopStrategy(optic)

    if aiming_mode in ["iterative", "robust"]:
        return RealReferenceStrategy(optic)

    return ParaxialReferenceStrategy(optic)
