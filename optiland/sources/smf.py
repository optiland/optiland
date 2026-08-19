"""Single-Mode Fiber (SMF) Source Module

This module implements an extended Gaussian source that generates a ray bundle
correctly representing the output of a single-mode fiber (SMF). The source
produces rays with Gaussian distributions in both the spatial and angular
domains, using quasi-random Sobol sequences for high-quality sampling.

Key features:
    - Spatial profile defined by the Mode Field Diameter (MFD).
    - Angular profile defined by the 1/e² full divergence angle.
    - Non-paraxial direction cosines computed via tangent mapping.
    - Sobol quasi-random sampling for low-discrepancy ray sets.
    - Optional point-source mode (zero spatial extent).

Note on ``num_rays``:
    Sobol sequences require the sample count to be a power of two.  The
    ``generate_rays`` method automatically rounds the requested ``num_rays``
    **up** to the nearest power of two, so the returned ``RealRays`` object
    may contain more rays than requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland import backend as be
from optiland.sources.base import BaseSource

if TYPE_CHECKING:
    from optiland.rays import RealRays


class SMFSource(BaseSource):
    """Extended Gaussian source representing a single-mode fiber output.

    This source generates rays with independent Gaussian distributions in
    both spatial (x, y) and angular (θ_x, θ_y) phase-space dimensions.
    It is designed to faithfully represent the far-field emission profile
    of a single-mode optical fiber.

    Sampling uses 4-dimensional Sobol sequences (two spatial, two angular)
    transformed to Gaussian distributions via the inverse error function.
    Direction cosines are computed non-paraxially from the sampled angles
    using the tangent mapping, ensuring physical accuracy at large
    divergence angles.

    Args:
        mfd_um (float): Mode Field Diameter in micrometers (µm).  The 1/e²
            beam diameter at the fiber facet.  The beam waist w₀ is ``mfd / 2``.
        divergence_deg_1e2 (float): Full-angle 1/e² far-field divergence in
            degrees.
        wavelength_um (float): Operating wavelength in micrometers (µm).
        total_power (float): Total optical power in Watts.  Defaults to 1.0.
        position (tuple[float, float, float]): Source position (x, y, z) in
            millimeters.  Defaults to ``(0, 0, 0)``.
        is_point_source (bool): If ``True``, spatial coordinates are set to
            zero (ideal point source).  Defaults to ``False``.

    Attributes:
        wavelength (float): Wavelength in µm.
        total_power (float): Total power in W.
        mfd_um (float): Mode Field Diameter in µm.
        divergence_deg_1e2 (float): Full 1/e² divergence in degrees.
        sigma_spatial_mm (float): Spatial sampling sigma (mm), equal to w₀/2
            converted to mm.
        sigma_angular_rad (float): Angular sampling sigma (rad), equal to
            half-angle / 2.
        is_point_source (bool): Whether to emit from a single point.

    Note:
        The actual number of rays returned by ``generate_rays`` is rounded
        up to the nearest power of two because Sobol sampling requires this.
        For example, requesting 1000 rays will generate 1024.
    """

    def __init__(
        self,
        mfd_um: float,
        wavelength_um: float,
        divergence_deg_1e2: float | None = None,
        total_power: float = 1.0,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        is_point_source: bool = False,
    ):
        super().__init__(position=position)

        self.wavelength = wavelength_um
        self.total_power = total_power
        self.mfd_um = mfd_um
        self.is_point_source = is_point_source

        # If divergence is not provided, calculate it assuming a diffraction-limited
        # Gaussian beam: theta_half = wavelength / (pi * w0)
        if divergence_deg_1e2 is None:
            import math

            w0 = mfd_um / 2.0
            theta_half_rad = wavelength_um / (math.pi * w0)
            self.divergence_deg_1e2 = 2 * math.degrees(theta_half_rad)
        else:
            self.divergence_deg_1e2 = divergence_deg_1e2

        # Spatial sigma for importance sampling
        # w₀ is the 1/e² radius = MFD / 2 (in µm), convert to mm
        w0_um = mfd_um / 2.0
        self.sigma_spatial_mm = (w0_um * 1e-3) / 2.0

        # Angular sigma for importance sampling
        # Half-angle from full divergence, then divide by 2 for sigma
        import math

        theta_rad = math.radians(self.divergence_deg_1e2 / 2.0)
        self.sigma_angular_rad = theta_rad / 2.0

    def generate_rays(self, num_rays: int) -> RealRays:
        """Generate rays from the SMF source using Sobol quasi-random sampling.

        Produces a ray bundle with Gaussian spatial and angular distributions
        appropriate for a single-mode fiber.  Direction cosines are computed
        non-paraxially.

        Because Sobol sequences require a power-of-two sample count, the
        actual number of rays returned is ``2 ** ceil(log2(num_rays))``.

        Args:
            num_rays (int): Desired number of rays.  Must be positive.

        Returns:
            RealRays: The generated ray bundle.

        Raises:
            ValueError: If ``num_rays`` is not a positive integer.
        """
        if num_rays <= 0:
            raise ValueError("num_rays must be a positive integer.")

        from optiland.rays import RealRays

        # Round up to nearest power of 2 (required for Sobol sampling)
        num_samples = 1 << (num_rays - 1).bit_length()

        # 4-D Sobol: dim 0,1 → spatial (x, y), dim 2,3 → angular (θx, θy)
        u = be.sobol_sampler(dim=4, num_samples=num_samples, scramble=True)

        sqrt2 = be.sqrt(be.array(2.0))

        # --- Spatial coordinates ---
        if self.is_point_source:
            x_start = be.zeros(num_samples)
            y_start = be.zeros(num_samples)
        else:
            x_start = self.sigma_spatial_mm * sqrt2 * be.erfinv(2 * u[:, 0] - 1)
            y_start = self.sigma_spatial_mm * sqrt2 * be.erfinv(2 * u[:, 1] - 1)

        # --- Angular coordinates (non-paraxial) ---
        theta_x = self.sigma_angular_rad * sqrt2 * be.erfinv(2 * u[:, 2] - 1)
        theta_y = self.sigma_angular_rad * sqrt2 * be.erfinv(2 * u[:, 3] - 1)

        # Convert angles to direction cosines via tangent mapping
        tau_x = be.tan(theta_x)
        tau_y = be.tan(theta_y)
        N_initial = 1.0 / be.sqrt(1.0 + tau_x**2 + tau_y**2)
        L_initial = tau_x * N_initial
        M_initial = tau_y * N_initial

        z_start = be.zeros_like(x_start)
        num_valid_rays = be.size(L_initial)

        # --- Power distribution (equal per ray, importance-sampled) ---
        power_per_ray = self.total_power / num_valid_rays
        intensity = be.full((num_valid_rays,), power_per_ray)
        wavelength_arr = be.full((num_valid_rays,), self.wavelength)

        rays = RealRays(
            x=x_start,
            y=y_start,
            z=z_start,
            L=L_initial,
            M=M_initial,
            N=N_initial,
            intensity=intensity,
            wavelength=wavelength_arr,
        )

        # Transform from local source coordinates to global coordinates
        self.cs.globalize(rays)

        return rays

    def __repr__(self) -> str:
        """Return a string representation of the SMFSource."""
        position = (float(self.cs.x), float(self.cs.y), float(self.cs.z))
        mode = "point" if self.is_point_source else "extended"
        return (
            f"SMFSource(mfd={self.mfd_um}µm, "
            f"divergence={self.divergence_deg_1e2}°, "
            f"wavelength={self.wavelength}µm, "
            f"power={self.total_power}W, "
            f"mode={mode}, "
            f"position={position})"
        )
