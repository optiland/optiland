"""Paraxial Module

This module provides various functionalities for the computation of paraxial
properties of lens systems.

Note that object-space coordinates are defined relative to the first surface
(at index 1), while image-space coordinates are defined relative to the image surface.
This is relevant for the focal points (F1 & F2), principal planes (P1 & P2),
anti-principal planes (P1anti & P2anti), nodal planes (N1 & N2), and anti-nodal
planes (N1anti & N2anti). In the Optiland convention, the 1 denotes object space and
the 2 denotes image space. For example, P1 is the object space principle plane and F2
is the back focal point.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.fields import ParaxialImageHeightField
from optiland.raytrace.paraxial_ray_tracer import ParaxialRayTracer

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from optiland._types import BEArray, ScalarOrArray
    from optiland.optic import Optic
    from optiland.surfaces import SurfaceGroup


class Paraxial:
    """A class representing a paraxial optical system.

    This class provides methods to calculate various properties of the optical
    system, such as focal lengths, entrance pupil location, exit pupil
    location, entrance pupil diameter, exit pupil diameter, image-space
    F-number, magnification, and more.


    Attributes:
        optic (Optic): The optical system being analyzed.
        surfaces (SurfaceGroup): The surface group of the optical system.

    """

    def __init__(self, optic: Optic):
        """Initializes a Paraxial instance

        Args:
            optic (Optic): The optical system to analyze.
        """
        self.optic = optic
        self._ray_tracer = ParaxialRayTracer(self.optic)

    @property
    def surfaces(self) -> SurfaceGroup:
        """SurfaceGroup: the surface group of the optical system."""
        return self.optic.surfaces

    def f1(self) -> ScalarOrArray:
        """Calculate the front focal length (f1).

        Returns:
            Front focal length.

        """
        z_start = -1
        wavelength = self.optic.primary_wavelength
        y, u = self.trace_generic(1.0, 0.0, z_start, wavelength, reverse=True, skip=1)
        f1 = y[0] / u[-1]
        return f1[0]

    def f2(self) -> ScalarOrArray:
        """Calculate the back focal length (f2), also known as effective focal length.

        Returns:
            Back focal length.

        """
        # start tracing 1 lens unit before first surface
        z_start = self.surfaces.positions[1] - 1
        wavelength = self.optic.primary_wavelength
        y, u = self.trace_generic(1.0, 0.0, z_start, wavelength)
        f2 = -y[0] / u[-1]
        return f2[0]

    def F1(self) -> ScalarOrArray:
        """Calculate the front focal point (F1) location.

        Note that this is defined relative to the first surface (at index 1).

        Returns:
            Front focal point location.

        """
        # start tracing 1 lens unit before first surface
        z_start = -1
        wavelength = self.optic.primary_wavelength
        y, u = self.trace_generic(1.0, 0.0, z_start, wavelength, reverse=True, skip=1)
        F1 = y[-1] / u[-1]
        return F1[0]

    def F2(self) -> ScalarOrArray:
        """Calculate the back focal point (F2) location.

        Note that this is defined relative to the image surface location.

        Returns:
            Back focal point location.

        """
        # start tracing 1 lens unit before first surface
        z_start = self.surfaces.positions[1] - 1
        wavelength = self.optic.primary_wavelength
        y, u = self.trace_generic(1.0, 0.0, z_start, wavelength)
        F2 = -y[-1] / u[-1]
        return F2[0]

    def P1(self) -> ScalarOrArray:
        """Calculate the front principal plane (P1) location.

        Note that this is defined relative to the first surface (at index 1).

        Returns:
            Front principal plane location.

        """
        return self.F1() - self.f1()

    def P2(self) -> ScalarOrArray:
        """Calculate the back principal plane (P2) location.

        Note that this is defined relative to the image surface location.

        Returns:
            Back principal plane location.

        """
        return self.F2() - self.f2()

    def P1anti(self) -> ScalarOrArray:
        """Calculate the front anti-principal plane (P1anti) location.

        Note that this is defined relative to the first surface (at index 1).

        Returns:
            Front anti-principal plane location.
        """
        return self.F1() + self.f1()

    def P2anti(self) -> ScalarOrArray:
        """Calculate the back anti-principal plane (P2anti) location.

        Note that this is defined relative to the image surface location.

        Returns:
            Back anti-principal plane location.
        """
        return self.F2() + self.f2()

    def N1(self) -> ScalarOrArray:
        """Calculate the front nodal plane (N1) location.

        Note that this is defined relative to the first surface (at index 1).

        Returns:
            Front nodal plane location.

        """
        return self.F1() + self.f2()

    def N2(self) -> ScalarOrArray:
        """Calculate the back nodal plane (N2) location.

        Note that this is defined relative to the image surface location.

        Returns:
            Back nodal plane location.

        """
        return self.F2() + self.f1()

    def N1anti(self) -> ScalarOrArray:
        """Calculate the front anti-nodal plane (N1anti) location.

        Note that this is defined relative to the first surface (at index 1).

        Returns:
            Front anti-nodal plane location.

        """
        return self.F1() - self.f2()

    def N2anti(self) -> ScalarOrArray:
        """Calculate the back anti-nodal plane (N2anti) location.

        Note that this is defined relative to the image surface location.

        Returns:
            Back anti-nodal plane location.

        """
        return self.F2() - self.f1()

    def EPL(self) -> ScalarOrArray:
        """Calculate the entrance pupil location (EPL) in global coordinates.

        Returns:
            Entrance pupil position relative to the first surface
                (which lies at z=0 by definition in its local coordinate system).

        """
        stop_index = self.surfaces.stop_index
        if stop_index == 1:
            return self.surfaces.positions[1, 0]

        y0 = 0
        u0 = 0.1
        pos = self.surfaces.positions
        z0 = pos[-1] - pos[stop_index]
        wavelength = self.optic.primary_wavelength

        # trace from center of stop on axis
        skip = self.surfaces.num_surfaces - stop_index
        y, u = self.trace_generic(y0, u0, z0[0], wavelength, reverse=True, skip=skip)

        loc_relative = y[-1] / u[-1]
        return loc_relative[0]

    def EPD(self) -> ScalarOrArray:
        """Calculate the entrance pupil diameter (EPD).

        Returns:
            Entrance pupil diameter.

        """
        if self.optic.aperture is None:
            raise ValueError("No aperture is defined on the optical system.")

        wavelength = self.optic.primary_wavelength
        return self.optic.aperture.compute_epd(self, wavelength)

    def XPL(self) -> ScalarOrArray:
        """Calculate the exit pupil location (XPL).

        Returns:
            Exit pupil location relative to the image surface.

        """
        stop_index = self.surfaces.stop_index
        z_start = self.surfaces.positions[stop_index]
        wavelength = self.optic.primary_wavelength
        y, u = self.trace_generic(0.0, 0.1, z_start, wavelength, skip=stop_index + 1)
        loc_relative = -y[-1] / u[-1]
        return loc_relative[0]

    def XPD(self) -> ScalarOrArray:
        """Calculate the exit pupil diameter (XPD).

        Returns:
            Exit pupil diameter.

        """
        # find marginal ray height at image surface
        ya, ua = self.marginal_ray()
        yi = ya[-1]
        ui = ua[-1]

        # find distance from image surface to exit pupil location
        xpl = self.XPL()

        # propagate marginal ray to this location
        yxp = yi + ui * xpl
        return 2 * yxp[0]

    def FNO(self) -> ScalarOrArray:
        """Calculate the image-space F-number (FNO).

        Returns:
            float: Image-space F-number.

        """
        if self.optic.aperture is None:
            raise ValueError("No aperture is defined on the optical system.")
        fno = self.optic.aperture.direct_fno()
        if fno is not None:
            return fno
        return self.f2() / self.EPD()

    def magnification(self) -> ScalarOrArray:
        """Calculate the transverse magnification.

        Returns:
            The system's transverse magnification.

        """
        _, ua = self.marginal_ray()
        n = self.optic.surfaces.n(self.optic.primary_wavelength)
        mag = n[0] * ua[0] / (n[-1] * ua[-1])
        return mag[0]

    def invariant(self) -> ScalarOrArray:
        """Calculate the Lagrange invariant.

        Returns:
            The Lagrange invariant of the system.

        """
        ya, ua = self.marginal_ray()
        yb, ub = self.chief_ray()
        n = self.optic.surfaces.n(self.optic.primary_wavelength)
        inv = yb[1] * n[1] * ua[1] - ya[1] * n[1] * ub[1]
        return inv[0]

    def marginal_ray(self) -> tuple[BEArray, BEArray]:
        """Calculates the marginal ray heights and angles at each surface.

        The marginal ray originates from the center of the object and passes
        through the edge of the aperture stop.

        Returns:
            A tuple containing two arrays:
                - y_marginal: Heights of the marginal ray at each surface.
                - u_marginal: Slopes of the marginal ray after each surface.

        """
        EPD = self.EPD()
        obj_z = self.surfaces.positions[1] - 10  # 10 mm before first surface

        if self.optic.object_surface is None:
            # TODO: make some nice error message
            raise ValueError()

        if self.optic.object_surface.is_infinite:
            ya = EPD / 2
            ua = 0
        else:
            obj_z = self.optic.object_surface.geometry.cs.z
            z = self.EPL() - obj_z
            ya = 0
            ua = EPD / (2 * z)

        wavelength = self.optic.primary_wavelength
        return self.trace_generic(ya, ua, obj_z, wavelength)

    def chief_ray(self) -> tuple[BEArray, BEArray]:
        """Calculates the chief ray heights and angles at each surface.

        The chief ray originates from the edge of the field of view and passes
        through the center of the aperture stop.

        Returns:
            A tuple containing two arrays:
                - y_chief: Heights of the chief ray at each surface.
                - u_chief: Slopes of the chief ray after each surface.

        """
        stop_index = self.optic.surfaces.stop_index
        pos = self.optic.surfaces.positions
        wavelength = self.optic.primary_wavelength
        num_surf = self.surfaces.num_surfaces
        y0 = 0.0
        u0 = 0.1  # Arbitrary small angle for unit trace

        # Trace a unit ray forward from stop to image
        z_fwd = pos[stop_index]
        skip_fwd = stop_index
        y_fwd_unit, _ = self.trace_generic(y0, u0, z_fwd, wavelength, skip=skip_fwd)
        y_img_unit = y_fwd_unit[-1]

        # Trace the same unit ray backward from stop to object
        z_rev = pos[-1] - pos[stop_index]
        skip_rev = num_surf - stop_index
        y_rev_unit, u_rev_unit = self.trace_generic(
            y0, u0, z_rev, wavelength, reverse=True, skip=skip_rev
        )
        y_obj_unit = y_rev_unit[-1]
        u_obj_unit = u_rev_unit[-1]

        # Scale based on field definition
        if self.optic.fields.field_definition is None:
            # TODO: make some nice error message
            raise ValueError()

        scaling_factor = self.optic.fields.field_definition.scale_chief_ray_for_field(
            self.optic, y_obj_unit, u_obj_unit, y_img_unit
        )

        # Determine initial ray parameters for final forward trace
        if isinstance(self.optic.fields.field_definition, ParaxialImageHeightField):
            y_obj_start = y_obj_unit * scaling_factor
        else:
            y_obj_start = -(y_obj_unit * scaling_factor)
        u_obj_start = u_obj_unit * scaling_factor

        if self.optic.object_surface.is_infinite:
            # For infinite conjugates, chief ray is defined by angle in object space.
            # We find its height at the first surface by propagating from the EPL,
            # where its height is zero.
            EPL = self.EPL()
            z_surf1 = self.surfaces.positions[1, 0]
            y1_start = u_obj_start * (z_surf1 - EPL)
            u1_start = u_obj_start
            z1_start = z_surf1
            return self.trace_generic(y1_start, u1_start, z1_start, wavelength)
        else:  # Finite conjugate
            # For finite conjugates, ray starts at y_obj_start on the object plane.
            z_start = self.optic.object_surface.geometry.cs.z
            return self.trace_generic(y_obj_start, u_obj_start, z_start, wavelength)

    def trace(self, Hy: ArrayLike, Py: ArrayLike, wavelength: float):
        """Trace paraxial ray through the optical system based on specified field
        and pupil coordinates.

        Args:
            Hy (float): Normalized field coordinate (typically in y).
            Py (float): Normalized pupil coordinate (typically in y).
            wavelength (float): Wavelength of the light in micrometers.

        Returns:
            tuple[be.ndarray, be.ndarray]: A tuple containing two arrays:
                - y_ray: Heights of the traced ray at each surface.
                - u_ray: Slopes of the traced ray after each surface.
        """
        return self._ray_tracer.trace(Hy, Py, wavelength)

    def trace_generic(
        self,
        y: BEArray | float,
        u: BEArray | float,
        z: BEArray | float,
        wavelength: float,
        reverse: bool = False,
        skip: int = 0,
    ) -> tuple[BEArray, BEArray]:
        """Trace generically-defined paraxial rays through the optical system.

        Args:
            y: The initial height(s) of the rays.
            u: The initial slope(s) of the rays.
            z: The initial axial position(s) of the rays,
                relative to the first surface if tracing forward, or relative
                to the last surface if tracing in reverse (before internal reversal).
            wavelength: The wavelength of the rays in micrometers.
            reverse: If True, trace the rays in reverse
                direction (from image to object space). Defaults to False.
            skip: The number of surfaces to skip from the
                beginning of the trace (or end if reverse). Defaults to 0.

        Returns:
            A tuple containing the height(s)
                and slope(s) of the rays at each surface interface after tracing.

        """
        return self._ray_tracer.trace_generic(y, u, z, wavelength, reverse, skip)
