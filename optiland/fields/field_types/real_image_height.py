"""Real Image Height Field Module

Defines fields by the chief ray's *real* (traced) height at the image plane.

The field parameters ``q = (q_x, q_y)`` -- object-space ray slopes for an
infinite object, local object-surface heights for a finite one -- are found by
solving the coupled two-variable inverse problem

.. math::

    R(q, \\theta, h) = I(q, \\theta) - h = 0,

where :math:`I` is the traced chief-ray intercept at the image plane,
:math:`\\theta` collects the trainable optical parameters and :math:`h` is the
requested image height.

Two properties matter here:

1. **The problem is genuinely coupled.** The Jacobian
   :math:`\\partial R/\\partial q` is a full :math:`2\\times2` matrix. Treating
   x and y as two independent scalar problems (and assuming the same paraxial
   scale on both axes) is only valid for a centered, uncoupled, approximately
   rotationally symmetric system; tilted/decentered systems, freeforms and
   anamorphic systems all produce off-diagonal terms, and an axis response can
   even reverse sign.

2. **The solve must not be unrolled into autograd.** The primal iteration runs
   graph-free. A single grad-attached residual trace then carries the
   first-order derivative via the implicit function theorem,

   .. math::

       \\frac{dq^\\star}{d\\theta} = -J_q^{-1} R_\\theta,

   so the autograd graph size is independent of the iteration count.

Only first derivatives are supported.

Kramer Harrison, 2025
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass
from typing import Any

import optiland.backend as be
from optiland.utils import (
    RCOND_EPS_MULTIPLIER,
    globalize_coordinates,
    jacobian_2x2_condition,
    machine_eps,
    solve_2x2,
)

from .base import BaseFieldDefinition
from .paraxial_image_height import ParaxialImageHeightField

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None


# Maximum number of step halvings in the per-field backtracking line search,
# matching the budget used by ``IterativeRayAimer``.
_MAX_BACKTRACK = 8

# The scale-invariant 2x2 conditioning authority lives in optiland.utils and
# is shared with the ray-aiming Newton core; these aliases keep this module's
# historical names importable.
_RCOND_EPS_MULTIPLIER = RCOND_EPS_MULTIPLIER
_jacobian_2x2_condition = jacobian_2x2_condition

# Multiplier on the machine epsilon for the paraxial-seed singularity test.
_PARAXIAL_EPS_MULTIPLIER = 64.0


def _paraxial_singular_threshold(value) -> float:
    """Dtype-aware threshold below which a unit-chief-ray scale is singular.

    The unit chief ray is traced with unit object-space input, so its natural
    scale is O(1); a magnitude at the level of ``C * eps(dtype)`` is numerical
    noise, not a meaningful paraxial response. For float64 this lands near the
    historical ``1e-14``; for float32 it is orders of magnitude larger, as it
    must be -- a fixed ``1e-14`` is below float32 round-off.
    """
    return _PARAXIAL_EPS_MULTIPLIER * machine_eps(value)


@dataclass
class _FieldSolveResult:
    """Outcome of the graph-free primal field-parameter solve.

    Attributes:
        qx: Solved x field parameter.
        qy: Solved y field parameter.
        rx: Final x residual, ``x_img - target_x``.
        ry: Final y residual, ``y_img - target_y``.
        converged: Per-field boolean mask.
        iterations: Number of accepted Newton/Broyden updates.
        jacobian: Final ``(J11, J12, J21, J22)`` Broyden estimate. Used for
            primal convergence only -- never as the scientific reference
            Jacobian for the implicit correction.
    """

    qx: Any
    qy: Any
    rx: Any
    ry: Any
    converged: Any
    iterations: int
    jacobian: tuple


@BaseFieldDefinition.register("real_image_height")
class RealImageHeightField(BaseFieldDefinition):
    """Defines fields by the chief ray's real height at the image plane.

    Coordinate contract:
        The requested image height ``(h_x, h_y)`` is defined in the **global**
        ``x``/``y`` coordinates of the chief ray's traced intercept on the
        image surface (surfaces globalize ray coordinates after each trace).
        For a tilted or decentered image surface the solve therefore drives
        the *global* transverse coordinates to the target, not coordinates
        local to the image surface.

    Note:
        The chief ray is constructed by aiming at the *paraxial* entrance-pupil
        center. This is **not** exact stop-center aiming: in systems with
        strong pupil aberration or a tilted/decentered stop, the traced ray
        does not necessarily cross the physical stop at its center. Exact
        aiming through the local stop center for arbitrary tilted/decentered
        systems (which requires a nested field-plus-ray-aiming solve and
        accounts for pupil aberration) is deliberately out of scope here; see
        the ray-aiming machinery for that problem.
    """

    def __init__(self, max_iter: int = 20) -> None:
        """Initialize the field definition.

        Args:
            max_iter: Maximum number of primal Newton/Broyden iterations.
        """
        self.max_iter = max_iter

    # ------------------------------------------------------------------
    # Targets and initial guess
    # ------------------------------------------------------------------

    def _targets(self, optic, Hx, Hy):
        """Requested image-plane coordinates ``(h_x, h_y)``."""
        max_field = optic.fields.max_field
        return max_field * Hx, max_field * Hy

    def _paraxial_scales(self, optic):
        """Paraxial ``(image-height, object-unit)`` scales of a unit chief ray.

        The object-side unit is a slope for an infinite object and a height for
        a finite one.
        """
        paraxial_field = ParaxialImageHeightField()
        y_img_unit, _ = paraxial_field._trace_unit_chief_ray(optic, plane="image")
        y_obj_unit, u_obj_unit = paraxial_field._trace_unit_chief_ray(
            optic, plane="object"
        )
        obj_unit = u_obj_unit if optic.object_surface.is_infinite else y_obj_unit
        return y_img_unit, obj_unit

    def _initial_field_parameters(self, optic, target_x, target_y):
        """Paraxial seed for the field parameters.

        Raises:
            ValueError: If a paraxial scale is singular, which makes the seed
                meaningless. Dividing by a floored value and continuing would
                silently produce a nonsense starting point.
        """
        y_img_unit, obj_unit = self._paraxial_scales(optic)

        y_img_mag = float(be.to_numpy(be.max(be.abs(be.atleast_1d(y_img_unit)))))
        obj_mag = float(be.to_numpy(be.max(be.abs(be.atleast_1d(obj_unit)))))

        if y_img_mag < _paraxial_singular_threshold(y_img_unit):
            raise ValueError(
                "Paraxial unit chief-ray image height is singular "
                f"({y_img_mag:.3e}); cannot seed the real-image-height field "
                "solve. Check the stop definition and surface powers."
            )
        if obj_mag < _paraxial_singular_threshold(obj_unit):
            raise ValueError(
                "Paraxial unit chief-ray object-space scale is singular "
                f"({obj_mag:.3e}); cannot seed the real-image-height field "
                "solve."
            )

        scale = obj_unit / y_img_unit
        return be.atleast_1d(target_x * scale), be.atleast_1d(target_y * scale)

    # ------------------------------------------------------------------
    # Residual
    # ------------------------------------------------------------------

    def _trace_chief_to_image(self, optic, qx, qy):
        """Trace the chief ray for ``(qx, qy)`` to the image plane.

        Returns:
            tuple: The ``(x, y)`` image-plane intercept.
        """
        rays = self._generate_chief_rays(optic, qx, qy)
        optic.surfaces.trace(rays)

        last_surface = optic.surfaces[-1]
        last_surface.material_post.propagation_model.propagate(
            rays, last_surface.thickness
        )
        return rays.x, rays.y

    def _image_residual(self, optic, qx, qy, target_x, target_y):
        """Residual ``R(q) = I(q) - h`` at the image plane."""
        x_img, y_img = self._trace_chief_to_image(optic, qx, qy)
        return x_img - target_x, y_img - target_y

    # ------------------------------------------------------------------
    # Numerical policy helpers
    # ------------------------------------------------------------------

    def _fd_step(self, q):
        """Finite-difference step for the unknown ``q``.

        Uses ``eps ** (1/3) * max(1, |q|)``: the cube root is the standard
        optimal exponent for a *central* difference (it balances truncation
        against round-off), and unlike ``sqrt(eps)`` it stays comfortably above
        the noise floor a full ray trace introduces. For float64 this lands
        near ``6e-6`` -- the historical ``1e-6`` scale -- while float32 gets a
        step roughly 800x larger, as it must.
        """
        eps = machine_eps(q)
        return (eps ** (1.0 / 3.0)) * be.maximum(be.abs(q), be.ones_like(q))

    def _residual_tolerance(self, target_x, target_y, reference):
        """Dtype- and target-scale-aware convergence tolerance.

        ``tau = eps ** 0.75 * max(1, |h_x|, |h_y|)``. For float64 this is
        ~1.8e-12 (matching the previous hardcoded 1e-12) and for float32
        ~6.4e-6, which is achievable -- a fixed 1e-12 is not.
        """
        eps = machine_eps(reference)
        scale = max(
            1.0,
            float(be.to_numpy(be.max(be.abs(be.atleast_1d(target_x))))),
            float(be.to_numpy(be.max(be.abs(be.atleast_1d(target_y))))),
        )
        return (eps**0.75) * scale

    def _solve_2x2(self, J11, J12, J21, J22, rx, ry, *, strict=False):
        """Solve ``J dq = R`` per field via the shared normalized 2x2 solve.

        Delegates to :func:`optiland.utils.solve_2x2` (the single 2x2
        conditioning authority, shared with the ray-aiming Newton core):
        the Jacobian is normalized by its largest-magnitude entry, which
        makes both the singularity classification and the solve invariant
        under global scaling and avoids raw-determinant overflow/underflow.

        Args:
            J11, J12, J21, J22: Per-field Jacobian entries.
            rx, ry: Per-field residuals.
            strict: If True, raise on a singular/ill-conditioned Jacobian
                before any division. If False, singular fields receive a zero
                placeholder step and are reported through the returned mask so
                the caller can retry (e.g. with a central-FD Jacobian). The
                determinant is never clipped to fabricate a step.

        Returns:
            tuple: ``(dq_x, dq_y, singular_mask)``.

        Raises:
            ValueError: If ``strict`` and any field's Jacobian is singular.
        """
        result = solve_2x2(J11, J12, J21, J22, rx, ry)
        singular = be.logical_not(result.valid)

        if strict and bool(be.any(singular)):
            n_singular = int(be.to_numpy(be.sum(singular)))
            n_total = int(be.size(result.rcond))
            raise ValueError(
                f"Real-image-height field Jacobian is singular for "
                f"{n_singular} of {n_total} field(s): reciprocal Frobenius "
                "condition estimate at round-off level "
                f"(rho_F <= {_RCOND_EPS_MULTIPLIER:g} * eps). The field map "
                "is locally non-invertible, so the implicit derivative is "
                "undefined and the field solve is aborted. Check for a "
                "degenerate/afocal configuration or a field point beyond the "
                "usable image height."
            )

        return result.x1, result.x2, singular

    # ------------------------------------------------------------------
    # Jacobians
    # ------------------------------------------------------------------

    def _initial_fd_jacobian(self, optic, qx, qy, target_x, target_y, rx, ry):
        """Forward-difference ``2x2`` Jacobian around the paraxial seed.

        A forward difference is sufficient to *start* the iteration and costs
        two extra chief-ray traces on top of the baseline residual, which is
        reused rather than retraced.
        """
        hx = self._fd_step(qx)
        hy = self._fd_step(qy)

        rx_dx, ry_dx = self._image_residual(optic, qx + hx, qy, target_x, target_y)
        rx_dy, ry_dy = self._image_residual(optic, qx, qy + hy, target_x, target_y)

        J11 = (rx_dx - rx) / hx
        J21 = (ry_dx - ry) / hx
        J12 = (rx_dy - rx) / hy
        J22 = (ry_dy - ry) / hy
        return J11, J12, J21, J22

    def _final_central_fd_jacobian(self, optic, qx, qy, target_x, target_y, scale=1.0):
        """Central-difference ``2x2`` Jacobian at the converged root.

        Serves as the independent reference and as the fallback when the AD
        Jacobian is unavailable, non-finite or singular.

        Args:
            scale: Multiplier on the finite-difference step, used to retry with
                a larger perturbation when the first estimate is degenerate.
        """
        hx = self._fd_step(qx) * scale
        hy = self._fd_step(qy) * scale

        rx_px, ry_px = self._image_residual(optic, qx + hx, qy, target_x, target_y)
        rx_mx, ry_mx = self._image_residual(optic, qx - hx, qy, target_x, target_y)
        rx_py, ry_py = self._image_residual(optic, qx, qy + hy, target_x, target_y)
        rx_my, ry_my = self._image_residual(optic, qx, qy - hy, target_x, target_y)

        J11 = (rx_px - rx_mx) / (2.0 * hx)
        J21 = (ry_px - ry_mx) / (2.0 * hx)
        J12 = (rx_py - rx_my) / (2.0 * hy)
        J22 = (ry_py - ry_my) / (2.0 * hy)
        return J11, J12, J21, J22

    # ------------------------------------------------------------------
    # Primal solve
    # ------------------------------------------------------------------

    def _solve_field_parameters_primal(self, optic, qx, qy, target_x, target_y):
        """Graph-free damped Newton/Broyden solve for the field parameters.

        Follows the pattern established by ``IterativeRayAimer``: a full
        finite-difference initial Jacobian, an analytic 2x2 Newton step, a
        per-field backtracking line search that only accepts a finite,
        residual-reducing step, and good-Broyden rank-one updates in between.

        Returns:
            _FieldSolveResult: Solved parameters, final residuals, per-field
            convergence mask, iteration count and the final Broyden Jacobian.
        """
        tol = self._residual_tolerance(target_x, target_y, qx)

        rx, ry = self._image_residual(optic, qx, qy, target_x, target_y)
        J11, J12, J21, J22 = self._initial_fd_jacobian(
            optic, qx, qy, target_x, target_y, rx, ry
        )

        converged = (rx**2 + ry**2) < tol**2
        iterations = 0

        for i in range(self.max_iter):
            if bool(be.all(converged)):
                break

            dq_x, dq_y, singular = self._solve_2x2(J11, J12, J21, J22, rx, ry)

            # A singular Jacobian gets one retry with a coarser perturbation
            # before the field is allowed to fail.
            if bool(be.any(singular)):
                Jr = self._final_central_fd_jacobian(
                    optic, qx, qy, target_x, target_y, scale=100.0
                )
                J11 = be.where(singular, Jr[0], J11)
                J12 = be.where(singular, Jr[1], J12)
                J21 = be.where(singular, Jr[2], J21)
                J22 = be.where(singular, Jr[3], J22)
                dq_x, dq_y, _ = self._solve_2x2(J11, J12, J21, J22, rx, ry)

            # --- backtracking line search -----------------------------
            old_err_sq = rx**2 + ry**2
            alpha = be.ones_like(qx)
            acc_dq_x = be.zeros_like(qx)
            acc_dq_y = be.zeros_like(qy)
            acc_rx = be.copy(rx)
            acc_ry = be.copy(ry)
            # Converged fields take a zero step and are never re-traced into
            # a worse state.
            searching = be.logical_not(converged)

            for _ in range(_MAX_BACKTRACK):
                if not bool(be.any(searching)):
                    break

                trial_x = qx - alpha * dq_x
                trial_y = qy - alpha * dq_y
                try_rx, try_ry = self._image_residual(
                    optic, trial_x, trial_y, target_x, target_y
                )
                new_err_sq = try_rx**2 + try_ry**2

                improved = be.logical_and(
                    searching,
                    be.logical_and(
                        be.logical_not(be.isnan(new_err_sq)),
                        new_err_sq < old_err_sq,
                    ),
                )
                acc_dq_x = be.where(improved, alpha * dq_x, acc_dq_x)
                acc_dq_y = be.where(improved, alpha * dq_y, acc_dq_y)
                acc_rx = be.where(improved, try_rx, acc_rx)
                acc_ry = be.where(improved, try_ry, acc_ry)
                searching = be.logical_and(searching, be.logical_not(improved))
                alpha = alpha * 0.5

            qx_new = qx - acc_dq_x
            qy_new = qy - acc_dq_y

            # --- good-Broyden rank-one update --------------------------
            s_x = qx_new - qx
            s_y = qy_new - qy
            y_x = acc_rx - rx
            y_y = acc_ry - ry

            Js_x = J11 * s_x + J12 * s_y
            Js_y = J21 * s_x + J22 * s_y

            norm_sq = s_x**2 + s_y**2
            safe_norm_sq = be.maximum(norm_sq, be.ones_like(norm_sq) * 1e-30)
            # Skip the update where the step was rejected (s = 0); the rank-one
            # formula is undefined there.
            took_step = norm_sq > 0.0

            J11 = be.where(took_step, J11 + (y_x - Js_x) * s_x / safe_norm_sq, J11)
            J12 = be.where(took_step, J12 + (y_x - Js_x) * s_y / safe_norm_sq, J12)
            J21 = be.where(took_step, J21 + (y_y - Js_y) * s_x / safe_norm_sq, J21)
            J22 = be.where(took_step, J22 + (y_y - Js_y) * s_y / safe_norm_sq, J22)

            qx, qy = qx_new, qy_new
            rx, ry = acc_rx, acc_ry
            iterations = i + 1

            converged = (rx**2 + ry**2) < tol**2

            # A field that could not take any step and is still not converged
            # is stuck; recompute its Jacobian once from scratch.
            stuck = be.logical_and(be.logical_not(took_step), be.logical_not(converged))
            if bool(be.any(stuck)):
                Jr = self._final_central_fd_jacobian(
                    optic, qx, qy, target_x, target_y, scale=10.0
                )
                J11 = be.where(stuck, Jr[0], J11)
                J12 = be.where(stuck, Jr[1], J12)
                J21 = be.where(stuck, Jr[2], J21)
                J22 = be.where(stuck, Jr[3], J22)

        return _FieldSolveResult(
            qx=qx,
            qy=qy,
            rx=rx,
            ry=ry,
            converged=converged,
            iterations=iterations,
            jacobian=(J11, J12, J21, J22),
        )

    # ------------------------------------------------------------------
    # Implicit correction
    # ------------------------------------------------------------------

    def _field_jacobian_by_vjp(self, rx, ry, qx_probe, qy_probe):
        """Local ``2x2`` field Jacobian from two vector-Jacobian products.

        Fields are traced independently, so the VJP of the *summed* x residual
        with respect to the per-field probes yields the per-field first row of
        the Jacobian, and the y residual yields the second row. This is exact
        for the implemented chief-ray trace, unlike the accumulated Broyden
        estimate.
        """
        zero_x = be.zeros_like(qx_probe)
        zero_y = be.zeros_like(qy_probe)

        J11, J12 = torch.autograd.grad(
            rx,
            (qx_probe, qy_probe),
            grad_outputs=torch.ones_like(rx),
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        J21, J22 = torch.autograd.grad(
            ry,
            (qx_probe, qy_probe),
            grad_outputs=torch.ones_like(ry),
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )

        # ``allow_unused`` yields None only when an axis is genuinely
        # disconnected from the residual, i.e. truly uncoupled.
        J11 = zero_x if J11 is None else J11
        J12 = zero_y if J12 is None else J12
        J21 = zero_x if J21 is None else J21
        J22 = zero_y if J22 is None else J22
        return J11.detach(), J12.detach(), J21.detach(), J22.detach()

    def _validate_or_fallback_field_jacobian(
        self, optic, qx, qy, target_x, target_y, J11, J12, J21, J22
    ):
        """Fall back to a central-FD Jacobian if the AD Jacobian is unusable.

        The AD Jacobian is preferred because it is the exact local derivative
        of the implemented differentiable trace. It is replaced only when the
        shared scale-invariant condition test classifies it as unusable while
        the FD estimate passes the same test.
        """
        ad_cond = _jacobian_2x2_condition(J11, J12, J21, J22)
        if not bool(be.any(ad_cond.singular)):
            return J11, J12, J21, J22

        with torch.no_grad():
            fd = self._final_central_fd_jacobian(optic, qx, qy, target_x, target_y)

        # The fallback must satisfy the same conditioning contract as the AD
        # Jacobian -- a merely nonzero FD determinant is not evidence of a
        # usable inverse.
        fd_cond = _jacobian_2x2_condition(*fd)
        if bool(be.any(fd_cond.singular)):
            # Neither estimate is usable; let the strict 2x2 solve raise with a
            # descriptive message rather than silently returning garbage.
            return J11, J12, J21, J22

        warnings.warn(
            "Real-image-height AD field Jacobian was non-finite or "
            "ill-conditioned; falling back to a central finite-difference "
            "Jacobian for the implicit correction.",
            RuntimeWarning,
            stacklevel=2,
        )
        return fd

    def _implicit_field_correction(self, optic, result, Hx, Hy):
        """One grad-attached residual trace carrying the field derivative.

        Applies ``q_implicit = q_bar - J_bar^-1 R(q_probe, theta, h)`` with the
        Jacobian detached, so the output graph stays independent of the number
        of field iterations while the derivative
        ``dq/dtheta = -J^-1 R_theta`` is exact to first order.
        """
        qx_det = result.qx.detach()
        qy_det = result.qy.detach()

        # Probe variables exist only to expose the local field Jacobian.
        qx_probe = qx_det.clone().requires_grad_(True)
        qy_probe = qy_det.clone().requires_grad_(True)

        # Recomputed outside no_grad so derivatives to Hx/Hy survive.
        target_x_grad, target_y_grad = self._targets(optic, Hx, Hy)

        rx, ry = self._image_residual(
            optic, qx_probe, qy_probe, target_x_grad, target_y_grad
        )

        J11, J12, J21, J22 = self._field_jacobian_by_vjp(rx, ry, qx_probe, qy_probe)
        J11, J12, J21, J22 = self._validate_or_fallback_field_jacobian(
            optic, qx_det, qy_det, target_x_grad, target_y_grad, J11, J12, J21, J22
        )

        dq_x, dq_y, _ = self._solve_2x2(J11, J12, J21, J22, rx, ry, strict=True)
        return qx_det - dq_x, qy_det - dq_y

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ray_origins(self, optic, Hx, Hy, Px, Py, vx, vy):
        """Calculate the initial positions for rays originating at the object.

        The field parameters are solved once per field point -- pupil
        coordinates do not define additional solves and are broadcast against
        the converged parameters afterwards.

        Args:
            optic (Optic): The optical system.
            Hx (float): Normalized x field coordinate.
            Hy (float): Normalized y field coordinate.
            Px (float or be.ndarray): x-coordinate of the pupil point.
            Py (float or be.ndarray): y-coordinate of the pupil point.
            vx (float): Vignetting factor in the x-direction.
            vy (float): Vignetting factor in the y-direction.

        Returns:
            tuple: A tuple containing the x, y, and z coordinates of the
                object position.

        Raises:
            RuntimeError: If the field solve fails to converge, in which case
                the implicit derivative is undefined.
            ValueError: If the paraxial seed or the field Jacobian is singular.
        """
        self._reject_folded_use(optic)
        target_x, target_y = self._targets(optic, Hx, Hy)
        qx0, qy0 = self._initial_field_parameters(optic, target_x, target_y)

        use_implicit = (
            torch is not None
            and be.get_backend() == "torch"
            and torch.is_grad_enabled()
        )

        ctx = torch.no_grad() if use_implicit else contextlib.nullcontext()
        with ctx:
            result = self._solve_field_parameters_primal(
                optic, qx0, qy0, target_x, target_y
            )

        if not bool(be.all(result.converged)):
            max_residual = float(
                be.to_numpy(be.max(be.sqrt(result.rx**2 + result.ry**2)))
            )
            raise RuntimeError(
                "Real-image-height field solve failed to converge after "
                f"{result.iterations} iteration(s) (max residual "
                f"{max_residual:.3e}); implicit gradients are undefined. "
                "Consider a smaller field, or check that the requested image "
                "height is reachable."
            )

        if use_implicit:
            qx, qy = self._implicit_field_correction(optic, result, Hx, Hy)
        else:
            qx, qy = result.qx, result.qy

        return self._compute_ray_origins_from_params(optic, qx, qy, Px, Py, vx, vy)

    def _generate_chief_rays(self, optic, val_x, val_y):
        """Generate chief rays (Px=0, Py=0) for the given field parameters."""
        from optiland.rays import RealRays

        zeros = be.zeros_like(val_x)

        # We use _compute_ray_origins_from_params with Px=0, Py=0, vx=0, vy=0
        x0, y0, z0 = self._compute_ray_origins_from_params(
            optic, val_x, val_y, zeros, zeros, 0, 0
        )

        # Aim at the entrance pupil's real-space point (its 3-D location on
        # the entry line), not at (0, 0, axial-scalar): the two only agree
        # for a system on the global z axis through the origin.
        px, py, pz = optic.paraxial.entrance_pupil_point_gcs()

        # ``be.full_like(x0, pz)`` would read the pupil coordinates out as
        # scalars and detach them. The entrance-pupil location depends on
        # the surface prescription, so dropping its gradient makes
        # d(chief ray)/d(theta) -- and therefore the whole implicit field
        # derivative -- wrong. Adding to a zero array of the right shape
        # broadcasts identically while keeping the point attached.
        x1 = be.full_like(x0, 0.0) + px
        y1 = be.full_like(y0, 0.0) + py
        z1 = be.full_like(x0, 0.0) + pz

        mag = be.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        L = (x1 - x0) / mag
        M = (y1 - y0) / mag
        N = (z1 - z0) / mag

        intensity = be.ones_like(x0)
        wavelength = be.full_like(x0, optic.primary_wavelength)

        return RealRays(x0, y0, z0, L, M, N, intensity, wavelength)

    def _compute_ray_origins_from_params(self, optic, val_x, val_y, Px, Py, vx, vy):
        """Compute ray origins given field parameters and pupil coords."""
        if optic.object_surface.is_infinite:
            EPL = optic.paraxial.EPL()
            EPD = optic.paraxial.EPD()
            offset = self._get_starting_z_offset(optic)

            x = -val_x * (offset + EPL)
            y = -val_y * (offset + EPL)
            z = optic.surfaces.positions[1] - offset

            x0 = Px * EPD / 2 * vx + x
            y0 = Py * EPD / 2 * vy + y
            # Same reasoning as in _generate_chief_rays: keep z attached rather
            # than reading it out as a scalar through full_like.
            z0 = be.full_like(Px, 0.0) + z
        else:
            # val_x, val_y are object heights.
            #
            # Both the conversion and the broadcast must preserve the autograd
            # graph: ``be.array`` on a tensor copies its data and detaches it,
            # and ``be.full_like`` reads it out as a scalar. Either would sever
            # the residual's dependence on the field parameters, silently
            # zeroing every finite-conjugate field gradient.
            x_local = be.atleast_1d(
                val_x if hasattr(val_x, "shape") else be.array(val_x)
            )
            y_local = be.atleast_1d(
                val_y if hasattr(val_y, "shape") else be.array(val_y)
            )

            # Broadcast field parameters against the pupil sampling.
            ones = be.ones_like(be.atleast_1d(Px))
            if be.size(x_local) == 1:
                x_local = x_local * ones
            if be.size(y_local) == 1:
                y_local = y_local * ones

            z_local = optic.object_surface.geometry.sag(x_local, y_local)

            # Globalize the local coordinates
            x0, y0, z0 = globalize_coordinates(
                optic.object_surface, x_local, y_local, z_local
            )
        return x0, y0, z0

    def get_paraxial_object_position(self, optic, Hy, y1, EPL):
        """Calculate the position of the object in the paraxial optical system.

        Args:
            Hy (float): The normalized field height.
            y1 (ndarray): The initial y-coordinate of the ray.
            EPL (float): The entrance pupil location.

        Returns:
            tuple: A tuple containing the y and z coordinates of the object
                position.
        """
        return ParaxialImageHeightField().get_paraxial_object_position(
            optic, Hy, y1, EPL
        )

    def scale_chief_ray_for_field(self, optic, y_obj_unit, u_obj_unit, y_img_unit):
        """Calculates the scaling factor for a unit chief ray based on the field
        definition.

        Args:
            optic (Optic): The optical system.
            y_obj_unit (float): The object-space height of the unit ray.
            u_obj_unit (float): The object-space angle of the unit ray.
            y_img_unit (float): The image-space height of the unit ray.

        Returns:
            float: The scaling factor.
        """
        return ParaxialImageHeightField().scale_chief_ray_for_field(
            optic, y_obj_unit, u_obj_unit, y_img_unit
        )

    def _get_starting_z_offset(self, optic):
        z = optic.surfaces.positions[1:-1]
        offset = optic.paraxial.EPD()
        return offset - be.min(z)
