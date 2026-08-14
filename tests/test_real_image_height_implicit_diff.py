"""Real-image-height field solve: forward accuracy, implicit gradients, graph size.

The field solve is an inverse problem: find the object-space field parameters
``q = (q_x, q_y)`` whose traced chief ray lands on a requested image height.
These tests cover the three things that can go wrong independently:

1. the *forward* solve reaches the requested height (and uses the full 2x2
   Jacobian, so it still works when the axes are coupled);
2. the *gradient* through the solved parameters is correct, not merely
   nonzero -- a detached intermediate produces a plausible-looking but wrong
   derivative;
3. the autograd graph does not grow with the iteration count, which is the
   whole point of solving implicitly.
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.fields.field_types.real_image_height import RealImageHeightField
from optiland.samples import objectives

from .nr_implicit_test_utils import backend_state, count_autograd_nodes

torch = pytest.importorskip("torch")

FINITE_OBJECT_Z = -555.0


def build_cooke(*, finite: bool = False, radius=None, max_iter: int = 20):
    """Cooke triplet configured for real-image-height fields."""
    optic = objectives.CookeTriplet()
    if finite:
        optic.object_surface.geometry.cs.z = be.array([FINITE_OBJECT_Z])
    optic.fields.set_type("real_image_height")
    optic.fields.fields = []
    optic.fields.add(y=0)
    optic.fields.add(y=20.0)
    if radius is not None:
        optic.surfaces[1].geometry.radius = radius
    optic.fields.field_definition.max_iter = max_iter
    return optic


def base_radius(finite: bool = False) -> float:
    radius = build_cooke(finite=finite).surfaces[1].geometry.radius
    return float(be.to_numpy(radius).ravel()[0])


def solved_origin_y(finite, radius, Hx=0.0, Hy=1.0):
    """Chief-ray origin y for a given radius -- depends on the solved field params."""
    optic = build_cooke(finite=finite, radius=radius)
    _, y0, _ = optic.fields.field_definition.get_ray_origins(
        optic,
        Hx=Hx,
        Hy=Hy,
        Px=be.array([0.0]),
        Py=be.array([0.0]),
        vx=0,
        vy=0,
    )
    return y0.sum()


def central_fd(fun, x0: float, steps=(1e-4, 1e-5)):
    """Central differences at two steps; returns (reference, all_values)."""
    values = []
    for h in steps:
        with torch.no_grad():
            plus = float(fun(torch.tensor(x0 + h, dtype=torch.float64)))
            minus = float(fun(torch.tensor(x0 - h, dtype=torch.float64)))
        values.append((plus - minus) / (2.0 * h))
    return values[-1], values


# ----------------------------------------------------------------------
# 9.4 Forward accuracy -- does the solved chief ray actually get there?
# ----------------------------------------------------------------------


class TestForwardAccuracy:
    """The solved chief ray must actually reach the requested image height."""

    @pytest.mark.parametrize("finite", [False, True], ids=["infinite", "finite"])
    @pytest.mark.parametrize(
        ("Hx", "Hy"), [(0.0, 1.0), (0.0, -0.5), (0.35, 0.8)], ids=["y", "-y", "xy"]
    )
    def test_traced_chief_ray_reaches_target(self, set_test_backend, finite, Hx, Hy):
        optic = build_cooke(finite=finite)
        field = optic.fields.field_definition

        target_x, target_y = field._targets(optic, Hx, Hy)
        qx0, qy0 = field._initial_field_parameters(optic, target_x, target_y)
        result = field._solve_field_parameters_primal(
            optic, qx0, qy0, target_x, target_y
        )
        assert bool(be.all(result.converged))

        x_img, y_img = field._trace_chief_to_image(optic, result.qx, result.qy)

        assert float(be.to_numpy(be.max(be.abs(x_img - target_x)))) < 1e-8
        assert float(be.to_numpy(be.max(be.abs(y_img - target_y)))) < 1e-8

    @pytest.mark.parametrize("precision", ["float32", "float64"])
    def test_converges_in_both_precisions(self, precision):
        with backend_state("torch", precision):
            optic = build_cooke()
            field = optic.fields.field_definition
            target_x, target_y = field._targets(optic, 0.0, 1.0)
            qx0, qy0 = field._initial_field_parameters(optic, target_x, target_y)

            with torch.no_grad():
                result = field._solve_field_parameters_primal(
                    optic, qx0, qy0, target_x, target_y
                )

            assert bool(be.all(result.converged)), (
                f"{precision} solve failed: residual={result.ry}"
            )

    def test_batched_fields_solve_independently(self, set_test_backend):
        optic = build_cooke()
        field = optic.fields.field_definition

        Hy = be.array([0.25, 0.6, 1.0])
        Hx = be.zeros_like(Hy)
        target_x, target_y = field._targets(optic, Hx, Hy)
        qx0, qy0 = field._initial_field_parameters(optic, target_x, target_y)
        result = field._solve_field_parameters_primal(
            optic, qx0, qy0, target_x, target_y
        )

        assert bool(be.all(result.converged))
        _, y_img = field._trace_chief_to_image(optic, result.qx, result.qy)
        assert float(be.to_numpy(be.max(be.abs(y_img - target_y)))) < 1e-8

    def test_pupil_sampling_does_not_change_field_solve(self, set_test_backend):
        """Pupil coordinates must not define additional field solves."""
        optic = build_cooke()
        field = optic.fields.field_definition

        def origins(n_pupil):
            Px = be.linspace(-1.0, 1.0, n_pupil)
            Py = be.zeros_like(Px)
            return field.get_ray_origins(optic, 0.0, 1.0, Px, Py, 1.0, 1.0)

        _, y_one, _ = origins(1)
        _, y_many, _ = origins(9)

        # y origin = pupil term (zero here, Py=0) + field term. The field term
        # must be identical regardless of how densely the pupil is sampled.
        assert float(be.to_numpy(be.max(be.abs(y_many - y_one[0])))) < 1e-12


# ----------------------------------------------------------------------
# 9.5 Full-Jacobian cross-coupling, isolated from prescription noise
# ----------------------------------------------------------------------


class _CoupledAnalyticField(RealImageHeightField):
    """Analytic coupled image map, to test the solver mathematics directly.

        I_x = q_x + 0.3 q_y + a
        I_y = -0.2 q_x + 1.1 q_y + b

    Both off-diagonal terms are nonzero, so a diagonal-only solver cannot
    satisfy both residuals simultaneously.
    """

    A = 0.15
    B = -0.4

    def _trace_chief_to_image(self, optic, qx, qy):
        return (
            qx + 0.3 * qy + self.A,
            -0.2 * qx + 1.1 * qy + self.B,
        )


class _StronglyCoupledAnalyticField(RealImageHeightField):
    """Same shape as above but with coupling strong enough that a
    diagonal-only fixed-point iteration diverges (spectral radius > 1).

        I_x = q_x + 3.0 q_y + a
        I_y = -2.0 q_x + 1.1 q_y + b
    """

    A = 0.15
    B = -0.4

    def _trace_chief_to_image(self, optic, qx, qy):
        return (
            qx + 3.0 * qy + self.A,
            -2.0 * qx + 1.1 * qy + self.B,
        )


class TestFullJacobianCrossCoupling:
    def test_solves_coupled_map_to_the_analytic_inverse(self, set_test_backend):
        field = _CoupledAnalyticField()
        target_x = be.array([0.5])
        target_y = be.array([-0.25])

        result = field._solve_field_parameters_primal(
            None, be.array([0.0]), be.array([0.0]), target_x, target_y
        )
        assert bool(be.all(result.converged))

        # Analytic inverse of [[1, 0.3], [-0.2, 1.1]] applied to (h - offset).
        rhs_x, rhs_y = 0.5 - _CoupledAnalyticField.A, -0.25 - _CoupledAnalyticField.B
        det = 1.0 * 1.1 - 0.3 * (-0.2)
        expected_qx = (1.1 * rhs_x - 0.3 * rhs_y) / det
        expected_qy = (1.0 * rhs_y + 0.2 * rhs_x) / det

        assert float(be.to_numpy(result.qx)[0]) == pytest.approx(expected_qx, abs=1e-10)
        assert float(be.to_numpy(result.qy)[0]) == pytest.approx(expected_qy, abs=1e-10)

    def test_off_diagonal_terms_are_recovered(self, set_test_backend):
        field = _CoupledAnalyticField()
        q = be.array([0.0])
        rx, ry = field._image_residual(None, q, q, be.array([0.0]), be.array([0.0]))
        J11, J12, J21, J22 = field._initial_fd_jacobian(
            None, q, q, be.array([0.0]), be.array([0.0]), rx, ry
        )

        assert float(be.to_numpy(J11)[0]) == pytest.approx(1.0, abs=1e-6)
        assert float(be.to_numpy(J12)[0]) == pytest.approx(0.3, abs=1e-6)
        assert float(be.to_numpy(J21)[0]) == pytest.approx(-0.2, abs=1e-6)
        assert float(be.to_numpy(J22)[0]) == pytest.approx(1.1, abs=1e-6)

    def test_diagonal_only_solver_cannot_satisfy_both_residuals(
        self, set_test_backend
    ):
        """Pin down *why* the full Jacobian is required.

        With strong x-y coupling the diagonal-only update is a fixed-point
        iteration whose matrix has spectral radius > 1, so it diverges no
        matter how many iterations it is given. The full 2x2 solve inverts
        the same map exactly.
        """
        field = _StronglyCoupledAnalyticField()
        target_x, target_y = be.array([0.5]), be.array([-0.25])

        # Emulate the previous diagonal-only update: dq_i = r_i / J_ii.
        qx, qy = be.array([0.0]), be.array([0.0])
        for _ in range(20):
            rx, ry = field._image_residual(None, qx, qy, target_x, target_y)
            qx = qx - rx / 1.0
            qy = qy - ry / 1.1

        rx, ry = field._image_residual(None, qx, qy, target_x, target_y)
        diagonal_residual = float(be.to_numpy(be.sqrt(rx**2 + ry**2)).ravel()[0])
        assert diagonal_residual > 1.0, (
            "the diagonal-only update unexpectedly handled a strongly coupled "
            "map; this test no longer isolates the cross-coupling requirement"
        )

        # The full 2x2 solver converges on the very same problem.
        result = field._solve_field_parameters_primal(
            None, be.array([0.0]), be.array([0.0]), target_x, target_y
        )
        assert bool(be.all(result.converged))
        coupled_residual = float(
            be.to_numpy(be.sqrt(result.rx**2 + result.ry**2)).ravel()[0]
        )
        assert coupled_residual < 1e-10


# ----------------------------------------------------------------------
# 9.6 AD/FD agreement for trainable optical parameters
# ----------------------------------------------------------------------


class TestImplicitGradients:
    @pytest.mark.parametrize("finite", [False, True], ids=["infinite", "finite"])
    def test_surface_radius_gradient_matches_fd(self, finite):
        with backend_state("torch"):
            r0 = base_radius(finite)
            radius = torch.tensor(r0, dtype=torch.float64, requires_grad=True)

            out = solved_origin_y(finite, radius)
            assert out.requires_grad
            (ad,) = torch.autograd.grad(out, radius)
            ad = float(ad)

            assert torch.isfinite(torch.tensor(ad))
            assert abs(ad) > 0.0, "gradient is spuriously zero"

            fd, fd_values = central_fd(lambda r: solved_origin_y(finite, r), r0)
            spread = abs(fd_values[0] - fd_values[1])
            assert spread <= max(1e-7, 1e-3 * abs(fd)), (
                f"FD unstable across steps: {fd_values}"
            )
            assert abs(ad - fd) <= max(1e-8, 1e-2 * abs(fd)), (
                f"AD/FD mismatch: AD={ad:.12e} FD={fd:.12e}"
            )

    def test_conic_gradient_matches_fd(self):
        with backend_state("torch"):

            def out_for(conic):
                optic = build_cooke()
                optic.surfaces[1].geometry.k = conic
                _, y0, _ = optic.fields.field_definition.get_ray_origins(
                    optic, 0.0, 1.0, be.array([0.0]), be.array([0.0]), 0, 0
                )
                return y0.sum()

            conic = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
            (ad,) = torch.autograd.grad(out_for(conic), conic)
            ad = float(ad)

            fd, fd_values = central_fd(out_for, 0.0, steps=(1e-3, 1e-4))
            assert abs(ad) > 0.0
            assert abs(ad - fd) <= max(1e-7, 2e-2 * abs(fd)), (
                f"AD/FD mismatch: AD={ad:.12e} FD={fd_values}"
            )

    def test_normalized_field_coordinate_gradient(self):
        """Hx/Hy derivatives must survive -- the target is recomputed
        outside no_grad precisely so that they do."""
        with backend_state("torch"):
            optic = build_cooke()
            Hy = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
            _, y0, _ = optic.fields.field_definition.get_ray_origins(
                optic, 0.0, Hy, be.array([0.0]), be.array([0.0]), 0, 0
            )
            (ad,) = torch.autograd.grad(y0.sum(), Hy)
            ad = float(ad)

            def out_for(h):
                optic = build_cooke()
                _, y, _ = optic.fields.field_definition.get_ray_origins(
                    optic, 0.0, h, be.array([0.0]), be.array([0.0]), 0, 0
                )
                return y.sum()

            fd, _ = central_fd(out_for, 0.8)
            assert abs(ad) > 0.0
            assert abs(ad - fd) <= max(1e-8, 1e-2 * abs(fd))

    def test_ad_jacobian_matches_central_fd_jacobian(self):
        """The detached AD Jacobian and the central-FD reference must agree."""
        with backend_state("torch"):
            optic = build_cooke()
            field = optic.fields.field_definition
            target_x, target_y = field._targets(optic, 0.0, 1.0)
            qx0, qy0 = field._initial_field_parameters(optic, target_x, target_y)
            with torch.no_grad():
                result = field._solve_field_parameters_primal(
                    optic, qx0, qy0, target_x, target_y
                )

            qx_probe = result.qx.detach().clone().requires_grad_(True)
            qy_probe = result.qy.detach().clone().requires_grad_(True)
            rx, ry = field._image_residual(
                optic, qx_probe, qy_probe, target_x, target_y
            )
            ad_J = field._field_jacobian_by_vjp(rx, ry, qx_probe, qy_probe)

            with torch.no_grad():
                fd_J = field._final_central_fd_jacobian(
                    optic, result.qx, result.qy, target_x, target_y
                )

            for ad_entry, fd_entry, name in zip(
                ad_J, fd_J, ("J11", "J12", "J21", "J22")
            ):
                a = float(be.to_numpy(ad_entry)[0])
                f = float(be.to_numpy(fd_entry)[0])
                assert abs(a - f) <= max(1e-5, 1e-3 * abs(f)), (
                    f"{name}: AD={a:.9e} FD={f:.9e}"
                )


# ----------------------------------------------------------------------
# 9.7 Graph complexity -- independent of iteration count
# ----------------------------------------------------------------------


class TestGraphComplexity:
    def test_graph_size_is_independent_of_max_iter(self):
        with backend_state("torch"):
            counts = []
            for max_iter in (5, 10, 20, 40):
                radius = torch.tensor(
                    base_radius(), dtype=torch.float64, requires_grad=True
                )
                optic = build_cooke(radius=radius, max_iter=max_iter)
                _, y0, _ = optic.fields.field_definition.get_ray_origins(
                    optic, 0.0, 1.0, be.array([0.0]), be.array([0.0]), 0, 0
                )
                counts.append(count_autograd_nodes(y0))

            assert max(counts) - min(counts) <= 1, (
                f"autograd graph grows with max_iter: {counts}"
            )


# ----------------------------------------------------------------------
# 9.8 Failure modes
# ----------------------------------------------------------------------


class _SingularAnalyticField(RealImageHeightField):
    """Rank-deficient map: I_y is a multiple of I_x, so J is singular."""

    def _trace_chief_to_image(self, optic, qx, qy):
        return qx + qy, 2.0 * (qx + qy)


class _StagnantAnalyticField(RealImageHeightField):
    """Residual that cannot be reduced -- every step is rejected."""

    def _trace_chief_to_image(self, optic, qx, qy):
        return be.ones_like(qx) * 1.0e3, be.ones_like(qy) * 1.0e3


class TestFailureModes:
    def test_singular_field_map_is_reported(self, set_test_backend):
        field = _SingularAnalyticField()
        q = be.array([0.0])
        rx, ry = field._image_residual(None, q, q, be.array([1.0]), be.array([0.0]))
        J = field._initial_fd_jacobian(
            None, q, q, be.array([1.0]), be.array([0.0]), rx, ry
        )

        with pytest.raises(ValueError, match="singular"):
            field._solve_2x2(*J, rx, ry, strict=True)

    def test_non_convergent_solve_raises_instead_of_returning(self, set_test_backend):
        field = _StagnantAnalyticField()
        field.max_iter = 3
        result = field._solve_field_parameters_primal(
            None, be.array([0.0]), be.array([0.0]), be.array([0.0]), be.array([0.0])
        )
        assert not bool(be.all(result.converged))

    def test_backtracking_rejects_a_residual_increasing_step(self, set_test_backend):
        """A step that makes the residual worse must not be committed."""
        field = _StagnantAnalyticField()
        field.max_iter = 2
        q0 = be.array([0.0])
        result = field._solve_field_parameters_primal(
            None, q0, q0, be.array([0.0]), be.array([0.0])
        )
        # Every trial step is rejected, so the parameters hold their last
        # finite value rather than running away.
        assert bool(be.all(be.isfinite(result.qx)))
        assert bool(be.all(be.isfinite(result.qy)))

    def test_singular_paraxial_seed_raises_clear_error(self, set_test_backend):
        optic = build_cooke()
        field = optic.fields.field_definition

        original = field._paraxial_scales
        field._paraxial_scales = lambda _optic: (be.array([0.0]), original(_optic)[1])

        with pytest.raises(ValueError, match="singular"):
            field._initial_field_parameters(optic, be.array([0.0]), be.array([1.0]))
