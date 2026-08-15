"""Perturbation-recovery regression tests for the optimization subsystem.

Most optimizer tests in this suite assert only ``result.success`` — they
verify that the solver runs, not that it converges to the correct
solution. The tests in this module close that gap with uniquely
invertible recovery problems: a lens parameter is perturbed, and the
optimizer is asked to restore a paraxial target that pins the parameter
to its exact original value.

Why paraxial operands (``f2`` / ``total_track``) instead of an
``rms_spot_size`` merit? The multi-field RMS-spot landscape of the
CookeTriplet sample is highly non-convex: with 4 DOF (3R + 1T), even a
3-5 % perturbation can hop between local minima whose merit values
differ by ~65 %. Convergence to *some* minimum therefore proves nothing
about recovering a known solution. Paraxial first-order quantities, by
contrast, are strictly monotonic in any single radius or thickness (all
other parameters held fixed), so "perturb one variable + pin it with one
paraxial equality" is a mathematically unique inversion that must
recover the original value to solver precision.

These tests are intended as a behavioral contract for optimizer
refactors (see discussion #641): any rewrite of the optimization
subsystem should keep them green.
"""

from __future__ import annotations

import math

from optiland.optimization import optimization
from optiland.samples.objectives import CookeTriplet

# CookeTriplet surface layout: 0 = object, 1..6 = lens surfaces, 7 = image.
# The thickness of surface 6 is the back focal distance (BFD).
BFD_SURFACE = 6
IMAGE_SURFACE = 7


def _radius(lens, surface_number):
    return float(lens.surfaces.surfaces[surface_number].geometry.radius)


def _thickness(lens, surface_number):
    return float(lens.surfaces.surfaces[surface_number].thickness)


def _optimize(problem, maxiter=200, tol=1e-12):
    optimizer = optimization.LeastSquares(problem)
    return optimizer.optimize(maxiter=maxiter, disp=False, tol=tol)


class TestImageSolveRecovery:
    def test_image_solve_recovers_paraxial_bfd_after_perturbation(
        self, set_test_backend
    ):
        """image_solve must converge back to the same paraxial BFD.

        ``image_solve`` is a one-step analytic paraxial marginal-ray
        solve, so the BFD it produces is a unique geometric quantity.
        Solve once to establish the paraxial baseline, perturb the BFD,
        solve again — the result must match the baseline, not merely
        "run without error".
        """
        lens = CookeTriplet()

        lens.updater.image_solve()
        bfd_paraxial = _thickness(lens, BFD_SURFACE)
        assert math.isfinite(bfd_paraxial) and bfd_paraxial > 0

        lens.updater.set_thickness(bfd_paraxial + 2.0, surface_number=BFD_SURFACE)
        assert abs(_thickness(lens, BFD_SURFACE) - (bfd_paraxial + 2.0)) < 1e-9

        lens.updater.image_solve()
        bfd_recovered = _thickness(lens, BFD_SURFACE)
        assert abs(bfd_recovered - bfd_paraxial) < 1e-6, (
            f"image_solve is not idempotent: paraxial BFD {bfd_paraxial:.6f}, "
            f"recovered {bfd_recovered:.6f} "
            f"(drift {bfd_recovered - bfd_paraxial:.2e})"
        )

    def test_cooke_triplet_shipped_bfd_is_not_paraxial_focus(self, set_test_backend):
        """Tripwire documenting an assumption of the tests in this module.

        The CookeTriplet sample ships with S6.thickness = 42.20778 mm,
        but its paraxial BFD is ~42.4149 mm (the sample is tuned for
        off-axis RMS-spot performance, not paraxial focus). Recovery
        tests must therefore establish their own baseline via
        ``image_solve`` instead of trusting the shipped thickness. If
        this test ever fails, the sample was re-tuned and that
        assumption should be revisited.
        """
        lens = CookeTriplet()
        bfd_shipped = _thickness(lens, BFD_SURFACE)
        lens.updater.image_solve()
        bfd_paraxial = _thickness(lens, BFD_SURFACE)

        assert abs(bfd_paraxial - bfd_shipped) > 0.05, (
            f"CookeTriplet shipped BFD ({bfd_shipped:.4f}) is now close to "
            f"the paraxial focus ({bfd_paraxial:.4f}); it used to differ by "
            "~0.21 mm. Revisit the baseline assumptions in this module."
        )


class TestUniqueInversionRecovery:
    """Perturb one parameter, pin it with one paraxial equality, recover."""

    def test_radius_recovers_when_pinned_by_f2(self, set_test_backend):
        lens = CookeTriplet()
        surface_number = 1
        r0 = _radius(lens, surface_number)
        f2_target = float(lens.paraxial.f2())

        lens.updater.set_radius(r0 + 5.0, surface_number=surface_number)
        assert abs(_radius(lens, surface_number) - (r0 + 5.0)) < 1e-9

        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=surface_number)
        problem.add_operand(
            operand_type="f2",
            target=f2_target,
            weight=1.0,
            input_data={"optic": lens},
        )
        result = _optimize(problem)
        assert result.success

        r_recovered = _radius(lens, surface_number)
        assert abs(r_recovered - r0) < 1e-3, (
            f"radius not recovered: original {r0:.6f}, recovered {r_recovered:.6f}"
        )

    def test_thickness_recovers_when_pinned_by_f2(self, set_test_backend):
        lens = CookeTriplet()
        surface_number = 2  # air gap between elements 1 and 2 (not the BFD)
        t0 = _thickness(lens, surface_number)
        f2_target = float(lens.paraxial.f2())

        lens.updater.set_thickness(t0 + 1.0, surface_number=surface_number)
        assert abs(_thickness(lens, surface_number) - (t0 + 1.0)) < 1e-9

        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "thickness", surface_number=surface_number)
        problem.add_operand(
            operand_type="f2",
            target=f2_target,
            weight=1.0,
            input_data={"optic": lens},
        )
        result = _optimize(problem)
        assert result.success

        t_recovered = _thickness(lens, surface_number)
        assert abs(t_recovered - t0) < 1e-3, (
            f"thickness not recovered: original {t0:.6f}, recovered {t_recovered:.6f}"
        )

    def test_negative_radius_recovers_when_pinned_by_f2(self, set_test_backend):
        """Diverging surface (negative curvature) exercises the
        sign-sensitive numeric path (scaling, bounds handling)."""
        lens = CookeTriplet()
        surface_number = 3  # front surface of the F2 flint element
        r0 = _radius(lens, surface_number)
        assert r0 < 0, f"S3 should have a negative radius, got {r0}"
        f2_target = float(lens.paraxial.f2())

        lens.updater.set_radius(r0 - 3.0, surface_number=surface_number)

        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=surface_number)
        problem.add_operand(
            operand_type="f2",
            target=f2_target,
            weight=1.0,
            input_data={"optic": lens},
        )
        result = _optimize(problem)
        assert result.success

        r_recovered = _radius(lens, surface_number)
        assert abs(r_recovered - r0) < 1e-3, (
            f"negative radius not recovered: original {r0:.6f}, "
            f"recovered {r_recovered:.6f}"
        )
        assert r_recovered < 0, "radius must stay negative"

    def test_radius_and_thickness_jointly_recover_under_triangular_constraints(
        self, set_test_backend
    ):
        """2 DOF (1R + 1T) pinned by 2 equalities (f2 + total_track).

        The constraint structure is triangular:

        * ``total_track`` = sum of thicknesses, so with all other
          thicknesses fixed it depends only on T2 — pinning T2 uniquely;
        * ``f2`` depends on all radii and thicknesses, but with the other
          radii fixed and T2 already pinned by ``total_track``, it pins
          R1 uniquely.

        (R1, T2) therefore has a unique inversion and a simultaneous
        perturbation of both must be recovered exactly.
        """
        lens = CookeTriplet()
        r_surf, t_surf = 1, 2
        r0 = _radius(lens, r_surf)
        t0 = _thickness(lens, t_surf)
        f2_target = float(lens.paraxial.f2())
        tt_target = float(lens.total_track)

        lens.updater.set_radius(r0 + 3.0, surface_number=r_surf)
        lens.updater.set_thickness(t0 + 0.8, surface_number=t_surf)

        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "radius", surface_number=r_surf)
        problem.add_variable(lens, "thickness", surface_number=t_surf)
        problem.add_operand(
            operand_type="f2",
            target=f2_target,
            weight=1.0,
            input_data={"optic": lens},
        )
        problem.add_operand(
            operand_type="total_track",
            target=tt_target,
            weight=1.0,
            input_data={"optic": lens},
        )
        result = _optimize(problem, maxiter=300)
        assert result.success

        t_recovered = _thickness(lens, t_surf)
        assert abs(t_recovered - t0) < 1e-3, (
            f"T2 not recovered: original {t0:.6f}, recovered {t_recovered:.6f}"
        )
        r_recovered = _radius(lens, r_surf)
        assert abs(r_recovered - r0) < 1e-3, (
            f"R1 not recovered: original {r0:.6f}, recovered {r_recovered:.6f}"
        )


class TestRealRayRecovery:
    def test_on_axis_rms_spot_recovers_bfd(self, set_test_backend):
        """BFD recovery through the real-ray path.

        Unlike the paraxial tests above, this exercises the full
        LeastSquares + real ray-trace + variable write-back chain. For
        the CookeTriplet on axis, spherical aberration is small, so the
        RMS-spot-optimal image plane must land close to (within 0.1 mm
        of) the starting BFD.
        """
        lens = CookeTriplet()
        bfd0 = _thickness(lens, BFD_SURFACE)

        lens.updater.set_thickness(bfd0 + 1.5, surface_number=BFD_SURFACE)

        problem = optimization.OptimizationProblem()
        problem.add_variable(lens, "thickness", surface_number=BFD_SURFACE)
        problem.add_operand(
            operand_type="rms_spot_size",
            target=0.0,
            weight=1.0,
            input_data={
                "optic": lens,
                "surface_number": IMAGE_SURFACE,
                "Hx": 0.0,
                "Hy": 0.0,
                "num_rays": 11,
                "wavelength": 0.55,
            },
        )
        result = _optimize(problem, tol=1e-9)
        assert result.success

        bfd_recovered = _thickness(lens, BFD_SURFACE)
        drift = abs(bfd_recovered - bfd0)
        assert drift < 0.1, (
            f"optimizer did not converge back to the BFD: "
            f"original {bfd0:.4f}, recovered {bfd_recovered:.4f} "
            f"(drift {drift:.4f} mm)"
        )
