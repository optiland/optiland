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

import optiland.backend as be
from optiland._deprecation import deprecated
from optiland.fields import ObjectHeightField, ParaxialImageHeightField
from optiland.raytrace.paraxial_ray_tracer import ParaxialRayTracer

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from optiland._types import BEArray, ScalarOrArray
    from optiland.optic import Optic
    from optiland.paraxial_path import ParaxialPath
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
        path = self.surfaces.build_paraxial_path()
        z_start = path.axial_positions.reshape(-1, 1)[1] - 1
        wavelength = self.optic.primary_wavelength
        y, u = self.trace_generic(1.0, 0.0, z_start, wavelength, path=path)
        f2 = -y[0] / u[-1]
        return f2[0]

    def f2_range(
        self,
        start: int,
        end: int,
        wavelength: float | None = None,
        *,
        path: ParaxialPath | None = None,
    ) -> ScalarOrArray:
        """Calculate the effective focal length of a range of surfaces.

        The range is treated as a lens group in isolation: the returned value
        is the paraxial EFL that surfaces ``start`` through ``end`` have with
        their own conjugates, evaluated in the media that bound the range in
        the parent system.

        This is deliberately not a decomposition of the full system's power.
        The focal lengths of a system's groups do not, in general, recombine
        into :meth:`f2`, since that would additionally require the separations
        between the groups' principal planes.

        Delegates to :meth:`ray_transfer_matrix`, so it shares the same
        validated scalar sequence (and supported scalar domain) as the
        explicit paraxial trace.

        Args:
            start: Index of the first surface of the range, inclusive.
            end: Index of the last surface of the range, inclusive.
            wavelength: Wavelength in micrometers at which the refractive
                indices are evaluated. Defaults to the system's primary
                wavelength.
            path: Optional prebuilt :class:`~optiland.paraxial_path.ParaxialPath`
                to reuse across several first-order calls.

        Returns:
            Effective focal length of the surface range. A range with no net
                power (an afocal group) returns infinity.

        Raises:
            ValueError: If the surface range is invalid.
            UnsupportedParaxialGeometryError: If the geometry lies outside
                the supported scalar folded paraxial domain.

        """
        matrix = self.ray_transfer_matrix(start, end, wavelength, path=path)

        # A ray entering parallel to the axis (y=1, u=0) leaves the range with
        # slope u_out = C, so the EFL is -y_in / u_out = -1 / C. This is the
        # same ratio f2() forms from an explicit paraxial trace.
        C = matrix[1, 0]
        if C == 0:
            return be.array(float("inf"))
        return -1.0 / C

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
        path = self.surfaces.build_paraxial_path()
        z_start = path.axial_positions.reshape(-1, 1)[1] - 1
        wavelength = self.optic.primary_wavelength
        y, u = self.trace_generic(1.0, 0.0, z_start, wavelength, path=path)
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

    def EPL(self, path: ParaxialPath | None = None) -> ScalarOrArray:
        """Calculate the entrance pupil location (EPL).

        This value is relative to the first physical surface (index 1),
        matching the convention of other first-order quantities on this
        class (``XPL`` is relative to the image surface, ``N1anti`` /
        ``N2anti`` relative to their respective reference surfaces). The
        prescription report and the ``EPL`` optimization operand both
        surface this value as-is. Call sites that need a global z (object
        space, surface positions, ray launch points) should call
        :meth:`entrance_pupil_z` instead.

        Args:
            path: Optional prebuilt :class:`ParaxialPath` to reuse.

        Returns:
            Entrance pupil position relative to the first surface
                (which lies at z=0 by definition in its local coordinate system).

        """
        stop_index = self.surfaces.stop_index
        if stop_index == 1:
            # Entrance pupil coincides with surface 1, so its location in
            # surface 1's local frame is zero. (The earlier ``positions[1, 0]``
            # return here mixed conventions — it agreed with the relative
            # convention only when surface 1 was at the origin.)
            return be.array(0.0)

        y0 = 0
        u0 = 0.1
        if path is None:
            path = self.surfaces.build_paraxial_path()
        pos = path.axial_positions.reshape(-1, 1)
        z0 = pos[-1] - pos[stop_index]
        wavelength = self.optic.primary_wavelength

        # trace from center of stop on axis
        skip = self.surfaces.num_surfaces - stop_index
        y, u = self.trace_generic(
            y0, u0, z0[0], wavelength, reverse=True, skip=skip, path=path
        )

        loc_relative = y[-1] / u[-1]
        return loc_relative[0]

    def entrance_pupil_axial_position(
        self, path: ParaxialPath | None = None
    ) -> ScalarOrArray:
        """Entrance pupil location on the unfolded signed axial coordinate.

        This is ``EPL()`` re-anchored to the same axial coordinate that
        ``SurfaceGroup.positions`` uses, so it can be differenced directly
        against surface positions. It is a 1-D unfolded axial scalar, never
        a Cartesian coordinate; use :meth:`entrance_pupil_point_gcs` for the
        pupil's real-space location.

        Args:
            path: Optional prebuilt :class:`ParaxialPath` to reuse.
        """
        if path is None:
            path = self.surfaces.build_paraxial_path()
        return self.EPL(path=path) + path.axial_positions.reshape(-1, 1)[1, 0]

    @deprecated("paraxial.entrance_pupil_axial_position()")
    def entrance_pupil_z(self, path: ParaxialPath | None = None) -> ScalarOrArray:
        """Entrance pupil location as an axial scalar (legacy name).

        ``EPL()`` returns a value relative to the first physical surface (per
        the documented convention). Call sites that mix the pupil location
        with other axial coordinates (object position, surface positions)
        should use this helper so the conversion lives in one place. Issue
        #613 was caused by call sites silently assuming EPL was global;
        routing them through this helper makes the convention explicit at
        the boundary.

        Despite the historical name, this is NOT a Cartesian global z: it is
        the pupil's coordinate along the signed unfolded axis of
        ``SurfaceGroup.positions``. The two coincide only while every leg of
        the beam path runs along +z. For a folded or off-axis-entered system
        the pupil's position in space is a point on the entry line -- use
        :meth:`entrance_pupil_point_gcs` for that point, and prefer
        :meth:`entrance_pupil_axial_position` (same value, honest name) in
        new code.

        Args:
            path: Optional prebuilt :class:`ParaxialPath` to reuse.
        """
        return self.entrance_pupil_axial_position(path=path)

    def entrance_pupil_point_gcs(self, path: ParaxialPath | None = None) -> tuple:
        """Entrance pupil position as a 3-D point in global coordinates.

        The entrance pupil is the stop imaged into object space; in the
        scalar folded model its apparent point lies on the unfolded entry
        line: ``r_EP = r_1 + EPL * d_0``, with ``r_1`` the first physical
        surface's vertex and ``d_0`` the unit entry direction.

        Args:
            path: Optional prebuilt :class:`ParaxialPath` to reuse.

        Returns:
            The pupil point as an ``(x, y, z)`` tuple of backend scalars.
        """
        if path is None:
            path = self.surfaces.build_paraxial_path()
        epl = self.EPL(path=path)
        anchor = path.vertices_gcs[1]
        d0 = path.entry_direction
        return tuple(anchor[i] + epl * d0[i] for i in range(3))

    def exit_pupil_point_gcs(self, path: ParaxialPath | None = None) -> tuple:
        """Exit pupil position as a 3-D point in global coordinates.

        Maps the axial ``XPL()`` scalar onto the physical image-space leg:
        ``r_XP = r_I + p_I * XPL * d_I``, with ``r_I`` the image vertex and
        ``p_I``, ``d_I`` the reflection parity and physical beam direction
        arriving at the image plane.

        Args:
            path: Optional prebuilt :class:`ParaxialPath` to reuse.

        Returns:
            The pupil point as an ``(x, y, z)`` tuple of backend scalars.
        """
        if path is None:
            path = self.surfaces.build_paraxial_path()
        xpl = self.XPL(path=path)
        return path.point_from_axial_offset(-1, xpl, side="incoming")

    def EPD(self) -> ScalarOrArray:
        """Calculate the entrance pupil diameter (EPD).

        Returns:
            Entrance pupil diameter.

        """
        if self.optic.aperture is None:
            raise ValueError(
                "No aperture is defined on the optical system, so the pupil "
                "size is unknown. Set one with "
                'lens.set_aperture(aperture_type="EPD", value=25).'
            )

        wavelength = self.optic.primary_wavelength
        return self.optic.aperture.compute_epd(self, wavelength)

    def XPL(self, path: ParaxialPath | None = None) -> ScalarOrArray:
        """Calculate the exit pupil location (XPL).

        Args:
            path: Optional prebuilt :class:`ParaxialPath` to reuse.

        Returns:
            Exit pupil location relative to the image surface.

        """
        stop_index = self.surfaces.stop_index
        if path is None:
            path = self.surfaces.build_paraxial_path()
        z_start = path.axial_positions.reshape(-1, 1)[stop_index]
        wavelength = self.optic.primary_wavelength
        y, u = self.trace_generic(
            0.0, 0.1, z_start, wavelength, skip=stop_index + 1, path=path
        )
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
            raise ValueError(
                "No aperture is defined on the optical system, so the pupil "
                "size is unknown. Set one with "
                'lens.set_aperture(aperture_type="EPD", value=25).'
            )
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
        path = self.surfaces.build_paraxial_path()
        pos = path.axial_positions.reshape(-1, 1)
        obj_z = pos[1] - 10  # 10 mm before first surface

        if self.optic.object_surface is None:
            raise ValueError(
                "No object surface is defined on the optical system. The marginal "
                "ray starts at the object, so an object surface is required. Add "
                "one with `optic.add_surface(index=0, ...)`."
            )

        if self.optic.object_surface.is_infinite:
            ya = EPD / 2
            ua = 0
        else:
            obj_z = self.optic.object_surface.geometry.cs.z
            z = self.entrance_pupil_axial_position(path=path) - obj_z
            ya = 0
            ua = EPD / (2 * z)

        wavelength = self.optic.primary_wavelength
        return self.trace_generic(ya, ua, obj_z, wavelength, path=path)

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
        path = self.surfaces.build_paraxial_path()
        pos = path.axial_positions.reshape(-1, 1)
        wavelength = self.optic.primary_wavelength
        num_surf = self.surfaces.num_surfaces
        y0 = 0.0
        u0 = 0.1  # Arbitrary small angle for unit trace

        # Trace a unit ray forward from stop to image
        z_fwd = pos[stop_index]
        skip_fwd = stop_index
        y_fwd_unit, _ = self.trace_generic(
            y0, u0, z_fwd, wavelength, skip=skip_fwd, path=path
        )
        y_img_unit = y_fwd_unit[-1]

        # Trace the same unit ray backward from stop to object
        z_rev = pos[-1] - pos[stop_index]
        skip_rev = num_surf - stop_index
        y_rev_unit, u_rev_unit = self.trace_generic(
            y0, u0, z_rev, wavelength, reverse=True, skip=skip_rev, path=path
        )
        y_obj_unit = y_rev_unit[-1]
        u_obj_unit = u_rev_unit[-1]

        field_definition = self.optic.fields.field_definition
        if not self.optic.object_surface.is_infinite and isinstance(
            field_definition, ObjectHeightField
        ):
            first_surface_z = pos[1, 0]
            object_z = self.optic.object_surface.geometry.cs.z
            y_obj_unit = y_obj_unit + (first_surface_z - object_z) * u_obj_unit

        # Scale based on field definition
        if field_definition is None:
            raise ValueError(
                "No field definition is set on the optical system. The chief ray "
                "is scaled by the field, so a field type is required. Set one with "
                '`optic.fields.set_type(...)`, e.g. "angle" or "object_height".'
            )

        scaling_factor = field_definition.scale_chief_ray_for_field(
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
            z_surf1 = pos[1, 0]
            y1_start = u_obj_start * (
                z_surf1 - self.entrance_pupil_axial_position(path=path)
            )
            u1_start = u_obj_start
            z1_start = z_surf1
            return self.trace_generic(
                y1_start, u1_start, z1_start, wavelength, path=path
            )
        else:  # Finite conjugate
            # For finite conjugates, ray starts at y_obj_start on the object plane.
            z_start = self.optic.object_surface.geometry.cs.z
            return self.trace_generic(
                y_obj_start, u_obj_start, z_start, wavelength, path=path
            )

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

    def ray_transfer_matrix(
        self,
        start: int,
        end: int,
        wavelength: float | None = None,
        *,
        path: ParaxialPath | None = None,
    ) -> BEArray:
        """Build the paraxial ray-transfer (ABCD) matrix of a surface range.

        The matrix maps a paraxial ray incident on surface ``start`` to the
        ray leaving surface ``end``::

            [y_out]   [A  B] [y_in]
            [u_out] = [C  D] [u_in]

        Both indices are inclusive. The input height and slope are those
        immediately before surface ``start``, and the output height and slope
        those immediately after surface ``end``. Only surfaces inside the
        range contribute: no propagation is included before ``start`` or after
        ``end``.

        The matrix is assembled from the same validated scalar sequence as
        :meth:`trace_generic` (shared path metadata, scalar-domain
        validation, straight-system advisories, and orientation-aware
        effective radii and focal lengths), so the matrix and an explicit
        paraxial trace of the same range always agree. The supported scalar
        domain is that of :class:`~optiland.paraxial_path.ParaxialPath`:
        piecewise-centered legs joined by plane fold mirrors, powered
        surfaces normal to their local beam segment; geometry outside it
        raises :class:`~optiland.paraxial_path.UnsupportedParaxialGeometryError`.

        Args:
            start: Index of the first surface of the range, inclusive. Must be
                at least 1, since surface 0 is the object surface and carries
                no power.
            end: Index of the last surface of the range, inclusive.
            wavelength: Wavelength in micrometers at which the refractive
                indices are evaluated. Defaults to the system's primary
                wavelength.
            path: Optional prebuilt :class:`~optiland.paraxial_path.ParaxialPath`
                for the current geometry, so a high-level operation making
                several first-order calls pays the path construction once.
                Must be a fresh snapshot of the surfaces being analyzed.

        Returns:
            The 2x2 ray-transfer matrix of the surface range.

        Raises:
            ValueError: If the surface range is invalid.
            UnsupportedParaxialGeometryError: If the geometry lies outside
                the supported scalar folded paraxial domain.

        """
        num_surfaces = self.surfaces.num_surfaces
        if start < 1:
            raise ValueError(
                f"Invalid start surface, got {start}. The range must begin at "
                "an optical surface, so its index must be at least 1 (index 0 "
                "is the object surface)."
            )
        if end > num_surfaces - 1:
            raise ValueError(
                f"Invalid end surface, got {end}. The system has "
                f"{num_surfaces} surfaces, so the largest valid index is "
                f"{num_surfaces - 1}."
            )
        if start > end:
            raise ValueError(
                f"Invalid surface range, got start={start} and end={end}. The "
                "range is inclusive and ordered, so start must not exceed end."
            )

        if wavelength is None:
            wavelength = self.optic.primary_wavelength

        sequence = self._ray_tracer.prepare_scalar_sequence(
            wavelength, path=path, operation="ray transfer matrix assembly"
        )
        R = sequence.radii
        n = sequence.refractive_indices
        pos = sequence.positions

        matrix = self._interaction_matrix(
            start, R, n, sequence.surfaces[start], sequence.focal_signs[start]
        )
        for k in range(start + 1, end + 1):
            transfer = self._transfer_matrix(pos[k] - pos[k - 1], n)
            matrix = be.matmul(
                self._interaction_matrix(
                    k, R, n, sequence.surfaces[k], sequence.focal_signs[k]
                ),
                be.matmul(transfer, matrix),
            )
        return matrix

    def _interaction_matrix(
        self,
        k: int,
        R: BEArray,
        n: BEArray,
        surface,
        focal_sign: float = 1.0,
    ) -> BEArray:
        """Build the ray-transfer matrix of a single surface interaction.

        Receives already-effective values from the shared scalar sequence:
        ``R`` carries the orientation-corrected radii, and ``focal_sign`` is
        the per-surface orientation sign to apply to an explicit paraxial
        focal length. No orientation logic is duplicated here.

        Args:
            k: Index of the surface.
            R: Paraxial-effective radii of curvature of all surfaces.
            n: Refractive indices following each surface.
            surface: The surface object at index ``k``.
            focal_sign: Orientation sign for an explicit paraxial surface's
                focal length.

        Returns:
            The 2x2 ray-transfer matrix of the surface interaction.

        """
        # Derive the constants from n so they carry the backend's dtype and
        # device. R is unsuitable, as it is infinite for a plane surface.
        zero = n[k] * 0.0
        one = zero + 1.0

        if surface.interaction_model.is_reflective:
            D = -one
            if surface.surface_type == "paraxial":
                C = -one / (focal_sign * surface.interaction_model.f)
            else:
                C = -2.0 / R[k]
        else:
            D = n[k - 1] / n[k]
            if surface.surface_type == "paraxial":
                C = -one / (focal_sign * surface.interaction_model.f * n[k])
            else:
                C = -((n[k] - n[k - 1]) / R[k]) / n[k]

        return be.stack([be.stack([one, zero]), be.stack([C, D])])

    @staticmethod
    def _transfer_matrix(t: BEArray, n: BEArray) -> BEArray:
        """Build the ray-transfer matrix for propagation over a distance.

        Args:
            t: The axial distance to propagate.
            n: Refractive indices following each surface, used only as a
                template for the backend's dtype and device.

        Returns:
            The 2x2 ray-transfer matrix of the propagation.

        """
        zero = n[0] * 0.0
        one = zero + 1.0
        return be.stack([be.stack([one, t + zero]), be.stack([zero, one])])

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
            path: Optional prebuilt :class:`ParaxialPath` for the current
                geometry (built once per high-level operation). Must be a
                fresh snapshot of the surfaces being traced.

        Returns:
            A tuple containing the height(s)
                and slope(s) of the rays at each surface interface after tracing.

        """
        return self._ray_tracer.trace_generic(y, u, z, wavelength, reverse, skip, path)
