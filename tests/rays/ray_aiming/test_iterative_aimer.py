from __future__ import annotations

import unittest

import numpy as np

import optiland.backend as be
from optiland.materials.ideal import IdealMaterial
from optiland.optic import Optic
from optiland.rays.ray_aiming import (
    IterativeRayAimer,
    ParaxialRayAimer,
    create_ray_aimer,
)


class TestIterativeRayAimer(unittest.TestCase):
    def setUp(self):
        be.set_backend("numpy")
        self.optic = Optic()

        # Create a fast system where paraxial aiming might fail to be exact at the stop
        self.optic.surfaces.add(index=0, thickness=100)
        self.optic.surfaces.add(
            index=1, radius=50, thickness=10, material=IdealMaterial(n=1.5)
        )
        self.optic.surfaces.add(index=2, radius=-50, thickness=20)  # Surface 2
        self.optic.surfaces.add(
            index=3, is_stop=True, thickness=20, aperture=20.0
        )  # Surface 3 (Stop)
        self.optic.surfaces.add(index=4)  # Image

        self.optic.surfaces.add(index=4)  # Image

        self.optic.set_aperture("float_by_stop_size", 1.0)
        self.optic.fields.set_type("angle")
        self.optic.fields.add(y=0)
        self.optic.fields.add(y=10)  # 10 degrees
        self.optic.wavelengths.add(0.55)

    def test_creation(self):
        aimer = create_ray_aimer("iterative", self.optic)
        self.assertIsInstance(aimer, IterativeRayAimer)
        self.assertEqual(aimer.max_iter, 20)

    def test_aiming_convergence_finite_object(self):
        # Configure for Finite Object: ensure thickness is finite (set in setUp as 100)
        # S0 thickness=100 -> Object at Z=0, S1 at Z=100.
        self.optic.fields.set_type("object_height")
        self.optic.fields.fields.clear()
        self.optic.fields.add(y=0)
        self.optic.fields.add(y=5)  # 5mm height

        # Determine explicit target stop_r
        stop_index = self.optic.surfaces.stop_index
        stop_r = self.optic.surfaces[stop_index].aperture.r_max

        # Aim rays
        # Increase max_iter for safety
        aimer = IterativeRayAimer(self.optic, max_iter=50)
        fields = (0, 1)  # Normalized coordinates (Hx, Hy)
        wavelengths = 0.55
        pupil = (0, 1)  # Marginal ray

        # Run aiming
        x, y, z, L, M, N = aimer.aim_rays(fields, wavelengths, pupil)

        # Verify rays actually hit the stop near expected coords
        from optiland.rays import RealRays

        intensity = be.ones_like(x)
        rays = RealRays(x, y, z, L, M, N, intensity=intensity, wavelength=wavelengths)

        for i in range(stop_index + 1):
            self.optic.surfaces[i].trace(rays)

        stop_y = rays.y
        target_y = 1.0 * stop_r

        # We expect high accuracy
        error = np.abs(stop_y - target_y)
        self.assertTrue(np.all(error < 1e-4), f"Finite Object Error: {error}")

    def test_compare_paraxial_vs_iterative(self):
        # Create a system with strong spherical aberration or distortion
        # Ensure Infinite Object -> Set thickness to inf
        self.optic.surfaces[0].thickness = np.inf

        self.optic.fields.set_type("angle")
        self.optic.fields.fields.clear()
        self.optic.fields.add(y=0)
        self.optic.fields.add(y=5)  # 5 degrees

        self.optic.surfaces[1].geometry.radius = 200  # Weak curve

        paraxial_aimer = ParaxialRayAimer(self.optic)
        iterative_aimer = IterativeRayAimer(self.optic)

        fields = (0, 1)  # Normalized (0, 1) -> 5 degrees
        wavelengths = 0.55
        pupil = (0, 1)

        # PARAXIAL
        x0, y0, z0, L0, M0, N0 = paraxial_aimer.aim_rays(fields, wavelengths, pupil)
        rays0 = import_rays(x0, y0, z0, L0, M0, N0, wavelengths)
        trace_to_stop(self.optic, rays0)
        stop_r = self.optic.surfaces[self.optic.surfaces.stop_index].aperture.r_max
        error0 = np.abs(rays0.y - stop_r)

        # ITERATIVE
        x1, y1, z1, L1, M1, N1 = iterative_aimer.aim_rays(fields, wavelengths, pupil)
        rays1 = import_rays(x1, y1, z1, L1, M1, N1, wavelengths)
        trace_to_stop(self.optic, rays1)
        error1 = np.abs(rays1.y - stop_r)

        # Iterative should be much better
        self.assertLess(
            error1[0],
            error0[0],
            f"Iterative {error1} not better than Paraxial {error0}",
        )
        self.assertLess(error1[0], 1e-6)


def import_rays(x, y, z, L, M, N, w):
    import optiland.backend as be
    from optiland.rays import RealRays

    return RealRays(x, y, z, L, M, N, intensity=be.ones_like(x), wavelength=w)


def trace_to_stop(optic, rays):
    stop_idx = optic.surfaces.stop_index
    for i in range(stop_idx + 1):
        optic.surfaces[i].trace(rays)


if __name__ == "__main__":
    unittest.main()
