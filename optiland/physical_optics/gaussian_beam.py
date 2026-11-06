"""Paraxial Gaussian Beam

This module defines GaussianBeam, a rotationally symmetric Gaussian beam at one
plane, described by its complex beam parameter q. The real part of q is the
signed distance from the waist to the plane (negative before the waist), and
the imaginary part is the Rayleigh range. The beam is carried through optical
systems with Optiland's ray-transfer (ABCD) matrices.

The real and imaginary parts of q are stored as two real numbers, so every
result stays differentiable on the PyTorch backend. Lengths are in mm and
wavelengths in µm, as in Optic, and beam radii are 1/e² intensity radii. A beam
with M² > 1 is treated as an embedded Gaussian: a TEM00 beam at M² times the
wavelength.

References:
    1. A. E. Siegman, Lasers (University Science Books, 1986). Gaussian beams:
       chapters 16 and 17. Gaussian beams in ABCD systems: chapter 20.
       Resonator eigenmodes: chapter 21.

Andrei Utina, 2026
"""

from __future__ import annotations

import optiland.backend as be
from optiland.utils import machine_eps

# Round-off allowed in the determinant of a round-trip matrix, as a multiple
# of the machine epsilon times |A D| + |B C| (the size of the two products
# whose difference is the determinant). Real round trips, including float32
# chains with entries of order 1e4, stay within 1 eps; 64 follows the
# convention of ``optiland.utils.RCOND_EPS_MULTIPLIER``.
DETERMINANT_EPS_MULTIPLIER = 64.0


class GaussianBeam:
    """A paraxial, rotationally symmetric Gaussian beam at one plane.

    Args:
        q_real: Distance in mm from the waist to this plane, negative when the
            waist lies downstream.
        q_imag: Rayleigh range in mm. Must be positive.
        wavelength: Vacuum wavelength in µm.
        refractive_index: Refractive index of the medium. Defaults to 1.0.
        m2: Beam quality factor M². Defaults to 1.0, a TEM00 beam.

    Raises:
        ValueError: If q_imag, wavelength or refractive_index is not positive,
            or if m2 is below 1.
    """

    def __init__(
        self,
        q_real,
        q_imag,
        wavelength: float,
        refractive_index=1.0,
        m2: float = 1.0,
    ) -> None:
        self.q_real = be.array(q_real)
        self.q_imag = be.array(q_imag)
        self.wavelength = float(wavelength)
        self.refractive_index = be.array(refractive_index)
        self.m2 = float(m2)
        if not bool(be.all(self.q_imag > 0)):
            raise ValueError("q_imag (the Rayleigh range) must be positive.")
        if self.wavelength <= 0:
            raise ValueError("wavelength must be positive.")
        if not bool(be.all(self.refractive_index > 0)):
            raise ValueError("refractive_index must be positive.")
        if self.m2 < 1.0:
            raise ValueError("m2 must be at least 1.")

    @classmethod
    def from_waist(
        cls,
        waist_radius,
        waist_position=0.0,
        *,
        wavelength: float,
        refractive_index=1.0,
        m2: float = 1.0,
    ) -> GaussianBeam:
        """Create a beam from its waist.

        Args:
            waist_radius: Waist radius in mm.
            waist_position: Distance in mm from this plane to the waist,
                positive when the waist lies downstream. Defaults to 0.0, a
                waist on this plane.
            wavelength: Vacuum wavelength in µm.
            refractive_index: Refractive index of the medium.
            m2: Beam quality factor M².

        Returns:
            The beam at this plane.
        """
        lam = m2 * wavelength * 1e-3
        z_r = be.pi * refractive_index * be.array(waist_radius) ** 2 / lam
        return cls(-be.array(waist_position), z_r, wavelength, refractive_index, m2)

    @classmethod
    def from_round_trip(
        cls,
        matrix,
        *,
        wavelength: float,
        refractive_index=1.0,
        m2: float = 1.0,
    ) -> GaussianBeam:
        """Create the eigenmode of a stable resonator.

        The eigenmode is the beam that comes back unchanged after one round
        trip (Siegman, chapter 21).

        Args:
            matrix: 2x2 round-trip matrix that starts and ends at the same
                plane, in the same direction and medium. For a linear cavity
                modeled from a forward-going plane, this is
                ``optic.paraxial.ray_transfer_matrix(1, k)``.
            wavelength: Vacuum wavelength in µm.
            refractive_index: Refractive index at the starting plane.
            m2: Beam quality factor M².

        Returns:
            The eigenmode at the starting plane.

        Raises:
            ValueError: If the round trip does not start and end in the same
                medium, or if the resonator is not stable.
        """
        A, B = matrix[0][0], matrix[0][1]
        C, D = matrix[1][0], matrix[1][1]
        # The determinant is 1 up to round-off, which grows with the size of
        # the products A D and B C and with the working precision.
        det = A * D - B * C
        scale = be.abs(A * D) + be.abs(B * C)
        tolerance = DETERMINANT_EPS_MULTIPLIER * machine_eps(det) * scale
        if not bool(be.all(be.abs(det - 1) <= tolerance)):
            raise ValueError(
                "a round trip must start and end in the same medium (det = 1)."
            )
        half_trace = (A + D) / 2
        if not bool(be.all(be.abs(half_trace) < 1)):
            raise ValueError("the resonator is not stable: |A + D| must be below 2.")
        # Solve for the q that maps onto itself; dividing by |B| picks the
        # solution with a positive Rayleigh range, the confined beam.
        u = (D - A) / (2 * B)
        v = be.sqrt(1 - half_trace**2) / be.abs(B)
        norm = u**2 + v**2
        return cls(u / norm, v / norm, wavelength, refractive_index, m2)

    @property
    def _lambda(self):
        """Wavelength in mm that sets the beam size: M² times the vacuum one."""
        return self.m2 * self.wavelength * 1e-3

    @property
    def rayleigh_range(self):
        """Rayleigh range in mm."""
        return self.q_imag

    @property
    def waist_position(self):
        """Distance in mm from this plane to the waist, positive downstream."""
        return -self.q_real

    @property
    def waist_radius(self):
        """Waist radius in mm."""
        return be.sqrt(self._lambda * self.q_imag / (be.pi * self.refractive_index))

    @property
    def beam_radius(self):
        """Beam radius in mm at this plane."""
        return self.waist_radius * be.sqrt(1 + (self.q_real / self.q_imag) ** 2)

    @property
    def curvature(self):
        """Wavefront curvature (one over the radius) in 1/mm, zero at the waist.

        The curvature is returned instead of the radius because the radius is
        infinite at the waist.
        """
        x, y = self.q_real, self.q_imag
        return x / (x**2 + y**2)

    @property
    def divergence(self):
        """Far-field half-angle divergence in radians, in this medium."""
        return self.waist_radius / self.q_imag

    def propagate(self, distance) -> GaussianBeam:
        """Return the beam after ``distance`` mm of free space."""
        return GaussianBeam(
            self.q_real + distance,
            self.q_imag,
            self.wavelength,
            self.refractive_index,
            self.m2,
        )

    def gouy_phase(self, distance):
        """Gouy phase in radians gained over ``distance`` mm of free space."""
        x, y = self.q_real, self.q_imag
        return be.arctan2(x + distance, y) - be.arctan2(x, y)

    def transform(self, matrix, refractive_index=None) -> GaussianBeam:
        """Apply a ray-transfer (ABCD) matrix to the beam.

        Args:
            matrix: 2x2 matrix acting on ray height and physical slope, such as
                one returned by ``optic.paraxial.ray_transfer_matrix``.
            refractive_index: Refractive index after the matrix. Defaults to
                the current index.

        Returns:
            The transformed beam.
        """
        A, B = matrix[0][0], matrix[0][1]
        C, D = matrix[1][0], matrix[1][1]
        x, y = self.q_real, self.q_imag
        # The ABCD law for q, written out in its real and imaginary parts.
        den = (C * x + D) ** 2 + (C * y) ** 2
        q_real = ((A * x + B) * (C * x + D) + A * C * y**2) / den
        q_imag = y * (A * D - B * C) / den
        n_out = self.refractive_index if refractive_index is None else refractive_index
        return GaussianBeam(q_real, q_imag, self.wavelength, n_out, self.m2)

    def with_refractive_index(self, refractive_index) -> GaussianBeam:
        """Return a beam with the same waist in a medium of another index."""
        return GaussianBeam.from_waist(
            self.waist_radius,
            self.waist_position,
            wavelength=self.wavelength,
            refractive_index=refractive_index,
            m2=self.m2,
        )

    def overlap(self, other: GaussianBeam):
        """Fraction of power that couples from this beam into another.

        Args:
            other: The beam to couple into, at the same plane, wavelength and
                refractive index.

        Returns:
            The coupled power fraction, from 0 to 1.

        Raises:
            ValueError: If either beam has M² above 1, or if the wavelengths or
                refractive indices differ.
        """
        if self.m2 != 1.0 or other.m2 != 1.0:
            raise ValueError("overlap is defined only for TEM00 beams (m2 = 1).")
        if abs(self.wavelength - other.wavelength) > 1e-12:
            raise ValueError("the two beams must have the same wavelength.")
        if not bool(be.allclose(self.refractive_index, other.refractive_index)):
            raise ValueError("the two beams must be in the same medium.")
        dx = self.q_real - other.q_real
        sy = self.q_imag + other.q_imag
        return 4 * self.q_imag * other.q_imag / (dx**2 + sy**2)

    def __repr__(self) -> str:
        """Return a readable summary of the beam."""
        w = float(be.to_numpy(self.beam_radius))
        w0 = float(be.to_numpy(self.waist_radius))
        z = float(be.to_numpy(self.waist_position))
        return (
            f"GaussianBeam(w={w:.6g} mm, w0={w0:.6g} mm, waist at {z:+.6g} mm, "
            f"wavelength={self.wavelength} µm, m2={self.m2})"
        )
