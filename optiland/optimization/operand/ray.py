"""Ray Operands Module

This module provides a class that calculates various ray tracing values for an
optical system. It is used in conjunction with the optimization module to
optimize optical systems.

Kramer Harrison, 2024
"""

from __future__ import annotations

import optiland.backend as be
from optiland import wavefront
from optiland.distribution import GaussianQuadrature


class RayOperand:
    """A class that provides static methods for performing ray tracing
        calculations on an optic.

    Methods:
        x_intercept: Calculates the x-coordinate of the intercept point on a
            specific surface.
        y_intercept: Calculates the y-coordinate of the intercept point on a
            specific surface.
        z_intercept: Calculates the z-coordinate of the intercept point on a
            specific surface.
        L: Calculates the direction cosine L of the ray on a specific surface.
        M: Calculates the direction cosine M of the ray on a specific surface.
        N: Calculates the direction cosine N of the ray on a specific surface.
        rms_spot_size: Calculates the root mean square (RMS) spot size on a
            specific surface.
        OPD_difference: Calculates the optical path difference (OPD)
            difference for a given ray distribution.

    """

    @staticmethod
    def x_intercept(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """Calculates the x-coordinate of the intercept point on a specific
            surface.

        Args:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The normalized x field coordinate.
            Hy: The normalized y field coordinate.
            Px: The normalized x pupil coordinate.
            Py: The normalized y pupil coordinate.
            wavelength: The wavelength of the ray.

        Returns:
            The x-coordinate of the intercept point.

        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surfaces.x[surface_number, 0]

    @staticmethod
    def y_intercept(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """Calculates the y-coordinate of the intercept point on a specific
            surface.

        Args:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The normalized x field coordinate.
            Hy: The normalized y field coordinate.
            Px: The normalized x pupil coordinate.
            Py: The normalized y pupil coordinate.
            wavelength: The wavelength of the ray.

        Returns:
            The y-coordinate of the intercept point.

        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surfaces.y[surface_number, 0]

    @staticmethod
    def z_intercept(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """Calculates the z-coordinate of the intercept point on a specific
            surface.

        Args:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The normalized x field coordinate.
            Hy: The normalized y field coordinate.
            Px: The normalized x pupil coordinate.
            Py: The normalized y pupil coordinate.
            wavelength: The wavelength of the ray.

        Returns:
            The z-coordinate of the intercept point.

        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surfaces.z[surface_number, 0]

    @staticmethod
    def x_intercept_lcs(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """Calculates the x-coordinate of the intercept point on a specific
            surface in its lcs, ie wrt to its vertex.

        Args:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The normalized x field coordinate.
            Hy: The normalized y field coordinate.
            Px: The normalized x pupil coordinate.
            Py: The normalized y pupil coordinate.
            wavelength: The wavelength of the ray.

        Returns:
            The x-coordinate of the intercept point.

        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        intercept = optic.surfaces.x[surface_number, 0]
        decenter = optic.surfaces[surface_number].geometry.cs.x
        return intercept - decenter

    @staticmethod
    def y_intercept_lcs(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """Calculates the y-coordinate of the intercept point on a specific
            surface in its lcs, ie wrt to its vertex.

        Args:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The normalized x field coordinate.
            Hy: The normalized y field coordinate.
            Px: The normalized x pupil coordinate.
            Py: The normalized y pupil coordinate.
            wavelength: The wavelength of the ray.

        Returns:
            The y-coordinate of the intercept point.

        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        intercept = optic.surfaces.y[surface_number, 0]
        decenter = optic.surfaces[surface_number].geometry.cs.y
        return intercept - decenter

    @staticmethod
    def z_intercept_lcs(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """Calculates the z-coordinate of the intercept point on a specific
            surface in its lcs, ie wrt to its vertex.

        Args:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The normalized x field coordinate.
            Hy: The normalized y field coordinate.
            Px: The normalized x pupil coordinate.
            Py: The normalized y pupil coordinate.
            wavelength: The wavelength of the ray.

        Returns:
            The z-coordinate of the intercept point.

        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        intercept = optic.surfaces.z[surface_number, 0]
        decenter = optic.surfaces[surface_number].geometry.cs.z

        # For some reason decenter can sometimes be a single-element array.
        # In that case, retreive the float inside.
        # This is a workaround until a solution is found.
        if be.is_array_like(decenter):
            decenter = decenter.item()

        return intercept - decenter

    @staticmethod
    def L(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """Calculates the direction cosine L of the ray on a specific surface.

        Args:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The normalized x field coordinate.
            Hy: The normalized y field coordinate.
            Px: The normalized x pupil coordinate.
            Py: The normalized y pupil coordinate.
            wavelength: The wavelength of the ray.

        Returns:
            The direction cosine L of the ray.

        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surfaces.L[surface_number, 0]

    @staticmethod
    def M(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """Calculates the direction cosine M of the ray on a specific surface.

        Args:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The normalized x field coordinate.
            Hy: The normalized y field coordinate.
            Px: The normalized x pupil coordinate.
            Py: The normalized y pupil coordinate.
            wavelength: The wavelength of the ray.

        Returns:
            The direction cosine M of the ray.

        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surfaces.M[surface_number, 0]

    @staticmethod
    def N(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """Calculates the direction cosine N of the ray on a specific surface.

        Args:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The normalized x field coordinate.
            Hy: The normalized y field coordinate.
            Px: The normalized x pupil coordinate.
            Py: The normalized y pupil coordinate.
            wavelength: The wavelength of the ray.

        Returns:
            The direction cosine N of the ray.

        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surfaces.N[surface_number, 0]

    @staticmethod
    def AOI(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """
        Calculates the real ray angle of incidence in degrees at a specific surface.
        This angle is always positive, and it is the angle between the incident ray and
        the surface normal.

        Args:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The normalized x field coordinate.
            Hy: The normalized y field coordinate.
            Px: The normalized x pupil coordinate.
            Py: The normalized y pupil coordinate.
            wavelength: The wavelength of the ray.

        Returns:
            The angle of incidence in degrees (always positive as in zemax).
        """

        optic.trace_generic(Hx, Hy, Px, Py, wavelength)

        surface = optic.surfaces[surface_number]
        geometry = surface.geometry

        L_inc = optic.surfaces.L[surface_number - 1, 0]
        M_inc = optic.surfaces.M[surface_number - 1, 0]
        N_inc = optic.surfaces.N[surface_number - 1, 0]

        from optiland.rays import RealRays

        rays_at_surface = RealRays(
            x=optic.surfaces.x[surface_number, 0],
            y=optic.surfaces.y[surface_number, 0],
            z=optic.surfaces.z[surface_number, 0],
            L=L_inc,
            M=M_inc,
            N=N_inc,
            intensity=1.0,  # irrelevant for AOI
            wavelength=wavelength,
        )

        # public surface_normal method
        nx, ny, nz = geometry.surface_normal(rays=rays_at_surface)

        # dot product between incident ray and surface normal
        dot_product = be.abs(L_inc * nx + M_inc * ny + N_inc * nz)

        # handle potential floating point errors where dot_product > 1.0
        dot_product_clip = be.minimum(dot_product, be.array(1.0))

        angle_rad = be.arccos(dot_product_clip)
        angle_deg = be.rad2deg(angle_rad)

        # For some reason angle_deg can sometimes be a single-element array.
        # In that case, retreive the float inside.
        # This is a workaround until a solution is found.
        if be.is_array_like(angle_deg):
            angle_deg = angle_deg.item()

        return angle_deg

    @staticmethod
    def rms_spot_size(
        optic,
        surface_number,
        Hx,
        Hy,
        num_rays,
        wavelength,
        distribution="hexapolar",
    ):
        """Calculates the root mean square (RMS) spot size on a specific surface.

        Args:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The normalized x field coordinate.
            Hy: The normalized y field coordinate.
            num_rays: The number of rays to trace.
            wavelength: The wavelength of the rays.
            distribution: The distribution of the rays. Default is 'hexapolar'.

        Returns:
            The RMS spot size on the specified surface.

        """
        if wavelength == "all":
            x = []
            y = []
            for wave in optic.wavelengths.get_wavelengths():
                optic.trace(Hx, Hy, wave, num_rays, distribution)
                x.append(optic.surfaces.x[surface_number, :].flatten())
                y.append(optic.surfaces.y[surface_number, :].flatten())
            wave_idx = optic.wavelengths.primary_index
            mean_x = be.mean(x[wave_idx])
            mean_y = be.mean(y[wave_idx])
            r2 = [(x[i] - mean_x) ** 2 + (y[i] - mean_y) ** 2 for i in range(len(x))]
            return be.sqrt(be.mean(be.concatenate(r2)))
        optic.trace(Hx, Hy, wavelength, num_rays, distribution)
        x = optic.surfaces.x[surface_number, :].flatten()
        y = optic.surfaces.y[surface_number, :].flatten()
        r2 = (x - be.mean(x)) ** 2 + (y - be.mean(y)) ** 2
        return be.sqrt(be.mean(r2))

    @staticmethod
    def OPD_difference(
        optic,
        Hx,
        Hy,
        num_rays,
        wavelength,
        distribution="gaussian_quad",
    ):
        """Calculates the mean optical path difference (OPD) difference for a
            given ray distribution.

        Args:
            optic: The optic object.
            Hx: The normalized x field coordinate.
            Hy: The normalized y field coordinate.
            num_rays: The number of rays to trace.
            wavelength: The wavelength of the rays.
            distribution: The distribution of the rays.
                Default is 'gaussian_quad'.

        Returns:
            The OPD difference for the given ray distribution.

        """
        weights = None

        if distribution == "gaussian_quad":
            distribution = GaussianQuadrature()

            distribution.generate_points(num_rays)
            weights = distribution.weights

        wf = wavefront.Wavefront(
            optic,
            [(Hx, Hy)],
            [wavelength],
            num_rays,
            distribution,
        )
        wavefront_data = wf.get_data((Hx, Hy), wavelength)
        opd = wavefront_data.opd
        if weights is None:
            weights = 1.0 / len(wf.distribution.x)
        delta = be.abs(opd - be.mean(opd)) * weights
        return be.sum(delta)

    @staticmethod
    def clearance(
        optic,
        line_ray_surface_idx,
        line_ray_field_coords,
        line_ray_pupil_coords,
        point_ray_surface_idx,
        point_ray_field_coords,
        point_ray_pupil_coords,
        wavelength,
    ):
        """Computes the signed perpendicular distance in the YZ plane from a
        reference line (Line A) to a reference point (Point B).

        Line A is defined by a ray (RA) traced at field FA, after it leaves
        surface SA. Point B is the intersection of a ray (RB) traced at
        field FB with surface SB.

        This operand is useful for creating clearance or interference constraints,
        particularly in off-axis reflective systems.

        The sign convention is such that for Line A propagating generally in the
        +Z direction (N direction cosine > 0), the signed distance is positive
        if Point B is on the +Y side of Line A. If Line A propagates generally
        in the -Z direction (N direction cosine < 0), this sign is flipped.

        Args:
            optic: The optical system model.
            line_ray_surface_idx: The index of the surface (SA) from which
                Line A originates (i.e., ray data is taken *after* this
                surface).
            line_ray_field_coords: A tuple (Hx, Hy) representing the
                normalized field coordinates for the ray defining Line A (FA).
            line_ray_pupil_coords: A tuple (Px, Py) representing the
                normalized pupil coordinates for the ray defining Line A (FA).
            point_ray_surface_idx: The index of the surface (SB) with which
                the ray defining Point B intersects.
            point_ray_field_coords: A tuple (Hx, Hy) representing the
                normalized field coordinates for the ray defining Point B (FB).
            point_ray_pupil_coords: A tuple (Px, Py) representing the
                normalized pupil coordinates for the ray defining Point B (FB).
            wavelength: The wavelength at which to trace the rays.

        Returns:
            float: The signed perpendicular distance in the YZ plane from
                   Line A to Point B. Returns 0.0 if Line A has zero length
                   in the YZ plane (i.e., mA and nA are both zero).
        """
        FA_Hx, FA_Hy = line_ray_field_coords
        FA_Px, FA_Py = line_ray_pupil_coords
        optic.trace_generic(FA_Hx, FA_Hy, FA_Px, FA_Py, wavelength)
        yA = optic.surfaces.y[line_ray_surface_idx, 0]
        zA = optic.surfaces.z[line_ray_surface_idx, 0]
        mA = optic.surfaces.M[line_ray_surface_idx, 0]
        nA = optic.surfaces.N[line_ray_surface_idx, 0]

        FB_Hx, FB_Hy = point_ray_field_coords
        FB_Px, FB_Py = point_ray_pupil_coords
        optic.trace_generic(FB_Hx, FB_Hy, FB_Px, FB_Py, wavelength)
        yB = optic.surfaces.y[point_ray_surface_idx, 0]
        zB = optic.surfaces.z[point_ray_surface_idx, 0]

        denominator = be.sqrt(mA**2 + nA**2)
        epsilon = 1e-9

        if be.abs(denominator) < epsilon:
            d = 0.0
        else:
            numerator = nA * (yB - yA) - mA * (zB - zA)
            d = numerator / denominator
            if nA < 0:
                d = -d
        return d
