"""Real Rays

This module contains the RealRays class, which represents a collection of real
rays in 3D space.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import optiland.backend as be
from optiland.rays.base import BaseRays

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from optiland._types import BEArray, ScalarOrArray
    from optiland.surfaces.standard_surface import Surface


class RealRays(BaseRays):
    """Represents a collection of real rays in 3D space.

    This class stores ray positions, directions, and properties as arrays,
    supporting both NumPy and PyTorch backends for efficient computation.

    Attributes:
        x: x-coordinates of ray positions.
        y: y-coordinates of ray positions.
        z: z-coordinates of ray positions.
        L: x-components of ray direction cosines.
        M: y-components of ray direction cosines.
        N: z-components of ray direction cosines.
        i: Intensity values of the rays.
        w: Wavelength values of the rays.
        opd: Optical path difference values.
        L0: Pre-surface x-direction cosines (None until surface interaction).
        M0: Pre-surface y-direction cosines (None until surface interaction).
        N0: Pre-surface z-direction cosines (None until surface interaction).

    Note:
        Direction cosines should be normalized: L² + M² + N² = 1.
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        L: ArrayLike,
        M: ArrayLike,
        N: ArrayLike,
        intensity: ArrayLike,
        wavelength: ArrayLike,
    ):
        """Initialize a collection of real rays in 3D space.

        Args:
            x: The x-coordinates of the ray starting positions.
            y: The y-coordinates of the ray starting positions.
            z: The z-coordinates of the ray starting positions.
            L: The x-components of the ray direction cosines.
            M: The y-components of the ray direction cosines.
            N: The z-components of the ray direction cosines.
            intensity: The intensity values of the rays.
            wavelength: The wavelength values of the rays.

        Note:
            All input arrays are converted to 1D arrays. Direction cosines
            (L, M, N) should be normalized such that L² + M² + N² = 1.
        """
        self.x = be.atleast_1d(x)
        self.y = be.atleast_1d(y)
        self.z = be.atleast_1d(z)
        self.L = be.atleast_1d(L)
        self.M = be.atleast_1d(M)
        self.N = be.atleast_1d(N)
        self.i = be.atleast_1d(intensity)
        self.w = be.atleast_1d(wavelength)
        self.opd = be.zeros_like(self.x)

        # variables to hold pre-surface direction cosines
        self.L0: BEArray | None = None
        self.M0: BEArray | None = None
        self.N0: BEArray | None = None

        self.is_normalized = True

    def trace_on_surface(self, surface: Surface) -> RealRays:
        """Dispatch to the surface's real ray trace kernel.

        Args:
            surface (Surface): The surface to trace through.

        Returns:
            RealRays: The traced real rays.

        """
        return surface._trace_real(self)

    def record_on_surface(self, surface: Surface) -> None:
        """Dispatch to the surface's real ray record method.

        Args:
            surface (Surface): The surface to record onto.

        """
        surface._record_real(self)

    def rotate_x(self, rx: ScalarOrArray):
        """Rotate the rays about the x-axis.

        Args:
            rx: Rotation angle around x-axis in radians.
        """
        rx = be.array(rx)
        self.y, self.z, self.M, self.N = (
            self.y * be.cos(rx) - self.z * be.sin(rx),
            self.y * be.sin(rx) + self.z * be.cos(rx),
            self.M * be.cos(rx) - self.N * be.sin(rx),
            self.M * be.sin(rx) + self.N * be.cos(rx),
        )

    def rotate_y(self, ry: ScalarOrArray):
        """Rotate the rays about the y-axis.

        Args:
            ry: Rotation angle around y-axis in radians.
        """
        ry = be.array(ry)
        self.x, self.z, self.L, self.N = (
            self.x * be.cos(ry) + self.z * be.sin(ry),
            -self.x * be.sin(ry) + self.z * be.cos(ry),
            self.L * be.cos(ry) + self.N * be.sin(ry),
            -self.L * be.sin(ry) + self.N * be.cos(ry),
        )

    def rotate_z(self, rz: ScalarOrArray):
        """Rotate the rays about the z-axis.

        Args:
            rz: Rotation angle around z-axis in radians.
        """
        rz = be.array(rz)
        self.x, self.y, self.L, self.M = (
            self.x * be.cos(rz) - self.y * be.sin(rz),
            self.x * be.sin(rz) + self.y * be.cos(rz),
            self.L * be.cos(rz) - self.M * be.sin(rz),
            self.L * be.sin(rz) + self.M * be.cos(rz),
        )

    def clip(self, condition: BEArray):
        """Clip the rays based on a condition."""
        cond = be.array(condition)
        try:
            cond = cond.astype(bool)
        except AttributeError:
            cond = cond.bool()
        self.i = be.where(cond, be.zeros_like(self.i), self.i)

    def refract(self, nx: float, ny: float, nz: float, n1: float, n2: float):
        """Refract rays on the surface.

        Args:
            nx: The x-component of the surface normals.
            ny: The y-component of the surface normals.
            nz: The z-component of the surface normals.
        """
        self.L0 = be.copy(self.L)
        self.M0 = be.copy(self.M)
        self.N0 = be.copy(self.N)

        u = n1 / n2
        nx, ny, nz, dot = self._align_surface_normal(nx, ny, nz)

        # ignore runtime warnings for total internal reflection
        with be.errstate(invalid="ignore"):
            root = be.sqrt(1 - u**2 * (1 - dot**2))
        tx = u * self.L0 + nx * root - u * nx * dot
        ty = u * self.M0 + ny * root - u * ny * dot
        tz = u * self.N0 + nz * root - u * nz * dot

        self.L = tx
        self.M = ty
        self.N = tz

    def reflect(self, nx: float, ny: float, nz: float):
        """Reflects the rays on the surface.

        Args:
            nx: The x-component of the surface normal.
            ny: The y-component of the surface normal.
            nz: The z-component of the surface normal.
        """
        self.L0 = be.copy(self.L)
        self.M0 = be.copy(self.M)
        self.N0 = be.copy(self.N)

        nx, ny, nz, dot = self._align_surface_normal(nx, ny, nz)

        self.L = self.L - 2 * dot * nx
        self.M = self.M - 2 * dot * ny
        self.N = self.N - 2 * dot * nz

    def gratingdiffract(
        self,
        nx: float,
        ny: float,
        nz: float,
        fx: float,
        fy: float,
        fz: float,
        m: int,
        d: float,
        n1: float,
        n2: float,
        is_reflective: bool,
    ):
        """Diffract the rays on a surface with a grating.

        Args:
            nx: The x-component of the surface normal.
            ny: The y-component of the surface normal.
            nz: The z-component of the surface normal.
            fx: The x-component of the grating vector.
            fy: The y-component of the grating vector.
            fz: The z-component of the grating vector.
            d:  The grating spacing
            m:  The grating diffraction order
            n1:  IOR of the pre surface material
            n2:  IOR of the post surface material
            is_reflective: Wether the surface is reflective or not
        """
        self.L0 = be.copy(self.L)
        self.M0 = be.copy(self.M)
        self.N0 = be.copy(self.N)

        nx, ny, nz, dot = self._align_surface_normal(nx, ny, nz)

        if is_reflective:
            sgn = -1
            n2c = n2 * sgn
            self.L = (
                self.L0 * d * n1 * ny**2
                + self.L0 * d * n1 * nz**2
                - self.M0 * d * n1 * nx * ny
                - self.N0 * d * n1 * nx * nz
                + fx * m * ny**2 * self.w
                + fx * m * nz**2 * self.w
                - fy * m * nx * ny * self.w
                - fz * m * nx * nz * self.w
                - nx
                * be.sqrt(
                    -(self.L0**2) * d**2 * n1**2 * ny**2
                    - self.L0**2 * d**2 * n1**2 * nz**2
                    + 2 * self.L0 * self.M0 * d**2 * n1**2 * nx * ny
                    + 2 * self.L0 * self.N0 * d**2 * n1**2 * nx * nz
                    - 2 * self.L0 * d * fx * m * n1 * ny**2 * self.w
                    - 2 * self.L0 * d * fx * m * n1 * nz**2 * self.w
                    + 2 * self.L0 * d * fy * m * n1 * nx * ny * self.w
                    + 2 * self.L0 * d * fz * m * n1 * nx * nz * self.w
                    - self.M0**2 * d**2 * n1**2 * nx**2
                    - self.M0**2 * d**2 * n1**2 * nz**2
                    + 2 * self.M0 * self.N0 * d**2 * n1**2 * ny * nz
                    + 2 * self.M0 * d * fx * m * n1 * nx * ny * self.w
                    - 2 * self.M0 * d * fy * m * n1 * nx**2 * self.w
                    - 2 * self.M0 * d * fy * m * n1 * nz**2 * self.w
                    + 2 * self.M0 * d * fz * m * n1 * ny * nz * self.w
                    - self.N0**2 * d**2 * n1**2 * nx**2
                    - self.N0**2 * d**2 * n1**2 * ny**2
                    + 2 * self.N0 * d * fx * m * n1 * nx * nz * self.w
                    + 2 * self.N0 * d * fy * m * n1 * ny * nz * self.w
                    - 2 * self.N0 * d * fz * m * n1 * nx**2 * self.w
                    - 2 * self.N0 * d * fz * m * n1 * ny**2 * self.w
                    + d**2 * n2c**2 * nx**2
                    + d**2 * n2c**2 * ny**2
                    + d**2 * n2c**2 * nz**2
                    - fx**2 * m**2 * ny**2 * self.w**2
                    - fx**2 * m**2 * nz**2 * self.w**2
                    + 2 * fx * fy * m**2 * nx * ny * self.w**2
                    + 2 * fx * fz * m**2 * nx * nz * self.w**2
                    - fy**2 * m**2 * nx**2 * self.w**2
                    - fy**2 * m**2 * nz**2 * self.w**2
                    + 2 * fy * fz * m**2 * ny * nz * self.w**2
                    - fz**2 * m**2 * nx**2 * self.w**2
                    - fz**2 * m**2 * ny**2 * self.w**2
                )
            ) / (d * n2c)
            self.M = (
                -self.L0 * d * n1 * nx * ny
                + self.M0 * d * n1 * nx**2
                + self.M0 * d * n1 * nz**2
                - self.N0 * d * n1 * ny * nz
                - fx * m * nx * ny * self.w
                + fy * m * nx**2 * self.w
                + fy * m * nz**2 * self.w
                - fz * m * ny * nz * self.w
                - ny
                * be.sqrt(
                    -(self.L0**2) * d**2 * n1**2 * ny**2
                    - self.L0**2 * d**2 * n1**2 * nz**2
                    + 2 * self.L0 * self.M0 * d**2 * n1**2 * nx * ny
                    + 2 * self.L0 * self.N0 * d**2 * n1**2 * nx * nz
                    - 2 * self.L0 * d * fx * m * n1 * ny**2 * self.w
                    - 2 * self.L0 * d * fx * m * n1 * nz**2 * self.w
                    + 2 * self.L0 * d * fy * m * n1 * nx * ny * self.w
                    + 2 * self.L0 * d * fz * m * n1 * nx * nz * self.w
                    - self.M0**2 * d**2 * n1**2 * nx**2
                    - self.M0**2 * d**2 * n1**2 * nz**2
                    + 2 * self.M0 * self.N0 * d**2 * n1**2 * ny * nz
                    + 2 * self.M0 * d * fx * m * n1 * nx * ny * self.w
                    - 2 * self.M0 * d * fy * m * n1 * nx**2 * self.w
                    - 2 * self.M0 * d * fy * m * n1 * nz**2 * self.w
                    + 2 * self.M0 * d * fz * m * n1 * ny * nz * self.w
                    - self.N0**2 * d**2 * n1**2 * nx**2
                    - self.N0**2 * d**2 * n1**2 * ny**2
                    + 2 * self.N0 * d * fx * m * n1 * nx * nz * self.w
                    + 2 * self.N0 * d * fy * m * n1 * ny * nz * self.w
                    - 2 * self.N0 * d * fz * m * n1 * nx**2 * self.w
                    - 2 * self.N0 * d * fz * m * n1 * ny**2 * self.w
                    + d**2 * n2c**2 * nx**2
                    + d**2 * n2c**2 * ny**2
                    + d**2 * n2c**2 * nz**2
                    - fx**2 * m**2 * ny**2 * self.w**2
                    - fx**2 * m**2 * nz**2 * self.w**2
                    + 2 * fx * fy * m**2 * nx * ny * self.w**2
                    + 2 * fx * fz * m**2 * nx * nz * self.w**2
                    - fy**2 * m**2 * nx**2 * self.w**2
                    - fy**2 * m**2 * nz**2 * self.w**2
                    + 2 * fy * fz * m**2 * ny * nz * self.w**2
                    - fz**2 * m**2 * nx**2 * self.w**2
                    - fz**2 * m**2 * ny**2 * self.w**2
                )
            ) / (d * n2c)
            self.N = -nz * be.sqrt(
                -(self.L0**2) * d**2 * n1**2 * ny**2
                - self.L0**2 * d**2 * n1**2 * nz**2
                + 2 * self.L0 * self.M0 * d**2 * n1**2 * nx * ny
                + 2 * self.L0 * self.N0 * d**2 * n1**2 * nx * nz
                - 2 * self.L0 * d * fx * m * n1 * ny**2 * self.w
                - 2 * self.L0 * d * fx * m * n1 * nz**2 * self.w
                + 2 * self.L0 * d * fy * m * n1 * nx * ny * self.w
                + 2 * self.L0 * d * fz * m * n1 * nx * nz * self.w
                - self.M0**2 * d**2 * n1**2 * nx**2
                - self.M0**2 * d**2 * n1**2 * nz**2
                + 2 * self.M0 * self.N0 * d**2 * n1**2 * ny * nz
                + 2 * self.M0 * d * fx * m * n1 * nx * ny * self.w
                - 2 * self.M0 * d * fy * m * n1 * nx**2 * self.w
                - 2 * self.M0 * d * fy * m * n1 * nz**2 * self.w
                + 2 * self.M0 * d * fz * m * n1 * ny * nz * self.w
                - self.N0**2 * d**2 * n1**2 * nx**2
                - self.N0**2 * d**2 * n1**2 * ny**2
                + 2 * self.N0 * d * fx * m * n1 * nx * nz * self.w
                + 2 * self.N0 * d * fy * m * n1 * ny * nz * self.w
                - 2 * self.N0 * d * fz * m * n1 * nx**2 * self.w
                - 2 * self.N0 * d * fz * m * n1 * ny**2 * self.w
                + d**2 * n2c**2 * nx**2
                + d**2 * n2c**2 * ny**2
                + d**2 * n2c**2 * nz**2
                - fx**2 * m**2 * ny**2 * self.w**2
                - fx**2 * m**2 * nz**2 * self.w**2
                + 2 * fx * fy * m**2 * nx * ny * self.w**2
                + 2 * fx * fz * m**2 * nx * nz * self.w**2
                - fy**2 * m**2 * nx**2 * self.w**2
                - fy**2 * m**2 * nz**2 * self.w**2
                + 2 * fy * fz * m**2 * ny * nz * self.w**2
                - fz**2 * m**2 * nx**2 * self.w**2
                - fz**2 * m**2 * ny**2 * self.w**2
            ) / (d * n2c) - (
                self.L0 * d * n1 * nx * nz
                + self.M0 * d * n1 * ny * nz
                - self.N0 * d * n1 * nx**2
                - self.N0 * d * n1 * ny**2
                + fx * m * nx * nz * self.w
                + fy * m * ny * nz * self.w
                - fz * m * nx**2 * self.w
                - fz * m * ny**2 * self.w
            ) / (d * n2c)

        else:
            sgn = 1
            n2c = n2 * sgn
            self.L = (
                self.L0 * d * n1 * ny**2
                + self.L0 * d * n1 * nz**2
                - self.M0 * d * n1 * nx * ny
                - self.N0 * d * n1 * nx * nz
                + fx * m * ny**2 * self.w
                + fx * m * nz**2 * self.w
                - fy * m * nx * ny * self.w
                - fz * m * nx * nz * self.w
                + nx
                * be.sqrt(
                    -(self.L0**2) * d**2 * n1**2 * ny**2
                    - self.L0**2 * d**2 * n1**2 * nz**2
                    + 2 * self.L0 * self.M0 * d**2 * n1**2 * nx * ny
                    + 2 * self.L0 * self.N0 * d**2 * n1**2 * nx * nz
                    - 2 * self.L0 * d * fx * m * n1 * ny**2 * self.w
                    - 2 * self.L0 * d * fx * m * n1 * nz**2 * self.w
                    + 2 * self.L0 * d * fy * m * n1 * nx * ny * self.w
                    + 2 * self.L0 * d * fz * m * n1 * nx * nz * self.w
                    - self.M0**2 * d**2 * n1**2 * nx**2
                    - self.M0**2 * d**2 * n1**2 * nz**2
                    + 2 * self.M0 * self.N0 * d**2 * n1**2 * ny * nz
                    + 2 * self.M0 * d * fx * m * n1 * nx * ny * self.w
                    - 2 * self.M0 * d * fy * m * n1 * nx**2 * self.w
                    - 2 * self.M0 * d * fy * m * n1 * nz**2 * self.w
                    + 2 * self.M0 * d * fz * m * n1 * ny * nz * self.w
                    - self.N0**2 * d**2 * n1**2 * nx**2
                    - self.N0**2 * d**2 * n1**2 * ny**2
                    + 2 * self.N0 * d * fx * m * n1 * nx * nz * self.w
                    + 2 * self.N0 * d * fy * m * n1 * ny * nz * self.w
                    - 2 * self.N0 * d * fz * m * n1 * nx**2 * self.w
                    - 2 * self.N0 * d * fz * m * n1 * ny**2 * self.w
                    + d**2 * n2c**2 * nx**2
                    + d**2 * n2c**2 * ny**2
                    + d**2 * n2c**2 * nz**2
                    - fx**2 * m**2 * ny**2 * self.w**2
                    - fx**2 * m**2 * nz**2 * self.w**2
                    + 2 * fx * fy * m**2 * nx * ny * self.w**2
                    + 2 * fx * fz * m**2 * nx * nz * self.w**2
                    - fy**2 * m**2 * nx**2 * self.w**2
                    - fy**2 * m**2 * nz**2 * self.w**2
                    + 2 * fy * fz * m**2 * ny * nz * self.w**2
                    - fz**2 * m**2 * nx**2 * self.w**2
                    - fz**2 * m**2 * ny**2 * self.w**2
                )
            ) / (d * n2c)
            self.M = (
                -self.L0 * d * n1 * nx * ny
                + self.M0 * d * n1 * nx**2
                + self.M0 * d * n1 * nz**2
                - self.N0 * d * n1 * ny * nz
                - fx * m * nx * ny * self.w
                + fy * m * nx**2 * self.w
                + fy * m * nz**2 * self.w
                - fz * m * ny * nz * self.w
                + ny
                * be.sqrt(
                    -(self.L0**2) * d**2 * n1**2 * ny**2
                    - self.L0**2 * d**2 * n1**2 * nz**2
                    + 2 * self.L0 * self.M0 * d**2 * n1**2 * nx * ny
                    + 2 * self.L0 * self.N0 * d**2 * n1**2 * nx * nz
                    - 2 * self.L0 * d * fx * m * n1 * ny**2 * self.w
                    - 2 * self.L0 * d * fx * m * n1 * nz**2 * self.w
                    + 2 * self.L0 * d * fy * m * n1 * nx * ny * self.w
                    + 2 * self.L0 * d * fz * m * n1 * nx * nz * self.w
                    - self.M0**2 * d**2 * n1**2 * nx**2
                    - self.M0**2 * d**2 * n1**2 * nz**2
                    + 2 * self.M0 * self.N0 * d**2 * n1**2 * ny * nz
                    + 2 * self.M0 * d * fx * m * n1 * nx * ny * self.w
                    - 2 * self.M0 * d * fy * m * n1 * nx**2 * self.w
                    - 2 * self.M0 * d * fy * m * n1 * nz**2 * self.w
                    + 2 * self.M0 * d * fz * m * n1 * ny * nz * self.w
                    - self.N0**2 * d**2 * n1**2 * nx**2
                    - self.N0**2 * d**2 * n1**2 * ny**2
                    + 2 * self.N0 * d * fx * m * n1 * nx * nz * self.w
                    + 2 * self.N0 * d * fy * m * n1 * ny * nz * self.w
                    - 2 * self.N0 * d * fz * m * n1 * nx**2 * self.w
                    - 2 * self.N0 * d * fz * m * n1 * ny**2 * self.w
                    + d**2 * n2c**2 * nx**2
                    + d**2 * n2c**2 * ny**2
                    + d**2 * n2c**2 * nz**2
                    - fx**2 * m**2 * ny**2 * self.w**2
                    - fx**2 * m**2 * nz**2 * self.w**2
                    + 2 * fx * fy * m**2 * nx * ny * self.w**2
                    + 2 * fx * fz * m**2 * nx * nz * self.w**2
                    - fy**2 * m**2 * nx**2 * self.w**2
                    - fy**2 * m**2 * nz**2 * self.w**2
                    + 2 * fy * fz * m**2 * ny * nz * self.w**2
                    - fz**2 * m**2 * nx**2 * self.w**2
                    - fz**2 * m**2 * ny**2 * self.w**2
                )
            ) / (d * n2c)
            self.N = nz * be.sqrt(
                -(self.L0**2) * d**2 * n1**2 * ny**2
                - self.L0**2 * d**2 * n1**2 * nz**2
                + 2 * self.L0 * self.M0 * d**2 * n1**2 * nx * ny
                + 2 * self.L0 * self.N0 * d**2 * n1**2 * nx * nz
                - 2 * self.L0 * d * fx * m * n1 * ny**2 * self.w
                - 2 * self.L0 * d * fx * m * n1 * nz**2 * self.w
                + 2 * self.L0 * d * fy * m * n1 * nx * ny * self.w
                + 2 * self.L0 * d * fz * m * n1 * nx * nz * self.w
                - self.M0**2 * d**2 * n1**2 * nx**2
                - self.M0**2 * d**2 * n1**2 * nz**2
                + 2 * self.M0 * self.N0 * d**2 * n1**2 * ny * nz
                + 2 * self.M0 * d * fx * m * n1 * nx * ny * self.w
                - 2 * self.M0 * d * fy * m * n1 * nx**2 * self.w
                - 2 * self.M0 * d * fy * m * n1 * nz**2 * self.w
                + 2 * self.M0 * d * fz * m * n1 * ny * nz * self.w
                - self.N0**2 * d**2 * n1**2 * nx**2
                - self.N0**2 * d**2 * n1**2 * ny**2
                + 2 * self.N0 * d * fx * m * n1 * nx * nz * self.w
                + 2 * self.N0 * d * fy * m * n1 * ny * nz * self.w
                - 2 * self.N0 * d * fz * m * n1 * nx**2 * self.w
                - 2 * self.N0 * d * fz * m * n1 * ny**2 * self.w
                + d**2 * n2c**2 * nx**2
                + d**2 * n2c**2 * ny**2
                + d**2 * n2c**2 * nz**2
                - fx**2 * m**2 * ny**2 * self.w**2
                - fx**2 * m**2 * nz**2 * self.w**2
                + 2 * fx * fy * m**2 * nx * ny * self.w**2
                + 2 * fx * fz * m**2 * nx * nz * self.w**2
                - fy**2 * m**2 * nx**2 * self.w**2
                - fy**2 * m**2 * nz**2 * self.w**2
                + 2 * fy * fz * m**2 * ny * nz * self.w**2
                - fz**2 * m**2 * nx**2 * self.w**2
                - fz**2 * m**2 * ny**2 * self.w**2
            ) / (d * n2c) - (
                self.L0 * d * n1 * nx * nz
                + self.M0 * d * n1 * ny * nz
                - self.N0 * d * n1 * nx**2
                - self.N0 * d * n1 * ny**2
                + fx * m * nx * nz * self.w
                + fy * m * ny * nz * self.w
                - fz * m * nx**2 * self.w
                - fz * m * ny**2 * self.w
            ) / (d * n2c)

        self.normalize()

    def update(self, jones_matrix: BEArray | None = None):
        """Update ray properties (primarily used for polarization)."""

    def normalize(self):
        """Normalize the direction vectors of the rays."""
        mag = be.sqrt(self.L**2 + self.M**2 + self.N**2)
        self.L = self.L / mag
        self.M = self.M / mag
        self.N = self.N / mag
        self.is_normalized = True

    def _align_surface_normal(
        self, nx: float, ny: float, nz: float
    ) -> tuple[float, float, float, BEArray]:
        """Align the surface normal with the incident ray vectors.

        Note:
            This is done as a convention to ensure the surface normal is
            pointing in the correct direction. This is required for consistency
            with the vector reflection and refraction equations used.

        Args:
            nx: The x-component of the surface normal.
            ny: The y-component of the surface normal.
            nz: The z-component of the surface normal.

        Returns:
            nx: The corrected x-component of the surface normal.
            ny: The corrected y-component of the surface normal.
            nz: The corrected z-component of the surface normal.
            dot: The dot product of the surface normal and the incident ray vectors.
        """

        # check if L0, M0 or N0 are None
        if self.L0 is None or self.M0 is None or self.N0 is None:
            raise ValueError(
                "Direction cosines (L0, M0, N0) must be set before aligning surface "
                "normal. Call refract(), reflect(), or gratingdiffract() first."
            )
        dot = self.L0 * nx + self.M0 * ny + self.N0 * nz

        sgn = be.sign(dot)
        nx = nx * sgn
        ny = ny * sgn
        nz = nz * sgn

        dot = be.abs(dot)
        return nx, ny, nz, dot

    def __str__(self) -> str:
        """Returns a string representation of the rays in a tabular format.
        Truncates output if the number of rays is large, showing first,
        central, and last rays.
        """

        if self.x is None or len(self.x) == 0:
            return "RealRays object (No rays)"

        num_rays = len(self.x)
        max_rays_to_print = 3
        header = (
            f"{'Ray #':>6} | {'x':>10} | {'y':>10} | {'z':>10} | "
            f"{'L':>10} | {'M':>10} | {'N':>10} | "
            f"{'Intensity':>10} | {'Wavelength':>12}\n"
        )
        separator = "-" * (len(header) + 5) + "\n"

        table = header + separator

        def format_ray(i):
            if 0 <= i < num_rays:
                x = be.to_numpy(self.x)[i]
                y = be.to_numpy(self.y)[i]
                z = be.to_numpy(self.z)[i]
                L = be.to_numpy(self.L)[i]
                M = be.to_numpy(self.M)[i]
                N = be.to_numpy(self.N)[i]
                intensity = be.to_numpy(self.i)[i]
                wavelength = be.to_numpy(self.w)[i]
                txt = (
                    f"{i:6} | {x:10.4f} | {y:10.4f} | {z:10.4f} | {L:10.6f} | "
                    f"{M:10.6f} | {N:10.6f} | "
                    f"{intensity:10.4f} | {wavelength:12.4f}\n"
                )
                return txt
            return ""

        if num_rays <= max_rays_to_print:
            indices_to_print = list(range(num_rays))
            count_shown = num_rays

            for i in indices_to_print:
                table += format_ray(i)

        else:
            num_ends = (max_rays_to_print - 1) // 2
            central_index = num_rays // 2

            indices = (
                set(range(num_ends))
                | {central_index}
                | set(range(num_rays - num_ends, num_rays))
            )

            sorted_indices = sorted(list(indices))
            count_shown = len(sorted_indices)

            for i in sorted_indices:
                table += format_ray(i)

        table += separator
        table += f"Showing {count_shown} of {num_rays} rays.\n"

        return table
