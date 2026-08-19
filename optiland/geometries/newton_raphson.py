"""Newton Raphson Geometry

The Newton Raphson geometry represents a surface utilizing the Newton-Raphson
method for ray tracing. This is an abstract base class that should be inherited
by any geometry that uses the Newton-Raphson method for ray tracing.

When the PyTorch backend is active with gradient tracking enabled, the
``distance`` method uses a DiffOptics-style one-step implicit correction
to compute correct first-order gradients through the converged intersection
point without unrolling the Newton-Raphson iterations through the autograd
graph.

The two-stage structure used here -- a graph-free primal Newton solve
followed by a single differentiable correction -- was contributed to Optiland
by Kushagra Kartik (https://github.com/Kushagra1480) in PR #550, addressing
the memory growth reported in issue #335.

See ``docs/developers_guide/implicit_differentiation.rst`` for the derivation,
the first-order-only contract, and the assumptions under which the derivative
is exact.

Kramer Harrison, 2024
"""

from __future__ import annotations

import contextlib
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.standard import StandardGeometry
from optiland.utils import machine_eps

try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None


# Conservative multiplier applied to the machine epsilon when building the
# scale-aware floor for the Newton denominator ``dF/dt``. Large enough to stay
# clear of round-off in the residual, small enough that a genuinely
# well-conditioned intersection is never regularized.
_DENOM_EPS_MULTIPLIER = 32.0

# Multiplier for the round-off floor under the user-supplied convergence
# tolerance. See :func:`_effective_tolerance`.
_CONV_EPS_MULTIPLIER = 8.0


def _nz_threshold(nz):
    """Dtype-aware validity threshold for the surface-normal z component.

    The sag slopes are reconstructed from the normalized normal as
    ``s_x = -n_x / n_z``; when ``|n_z|`` approaches round-off the surface is
    not a numerically valid single-valued height function ``z = s(x, y)`` at
    that point, and the reconstruction (and therefore the implicit derivative)
    is meaningless. Uses the same conservative multiplier as the ``dF/dt``
    threshold. For float64 this lands near the historical ``1e-14``; for
    float32 it is orders of magnitude larger, which is the point -- a fixed
    ``1e-14`` is below float32 round-off and never triggers.
    """
    return _DENOM_EPS_MULTIPLIER * machine_eps(nz)


# -- utility functions --
def _is_radius_infinite(radius):
    """Checks if the given radius represents an infinite radius (a plane).

    Args:
        radius (float or be.ndarray): The radius value to check.

    Returns:
        bool: True if the radius is effectively infinite (or all elements are
        infinite if it's an array), False otherwise.
    """
    is_inf_tensor = be.isinf(radius)
    if hasattr(is_inf_tensor, "ndim") and is_inf_tensor.ndim > 0:
        # If it's a multi-element array, check if all are infinite
        return bool(be.all(is_inf_tensor))
    # For scalars or single-element arrays that can be converted by .item()
    return (
        bool(is_inf_tensor.item())
        if hasattr(is_inf_tensor, "item")
        else bool(is_inf_tensor)
    )


def _sign_preserving_floor(value, eps=None):
    """Clamp values to a minimum absolute magnitude while preserving sign.

    Preserving the sign matters: replacing a small *negative* denominator with
    a positive constant reverses the Newton step direction.

    When ``eps`` is None, a dtype-aware floor is derived from the machine
    epsilon of ``value``. This is a numerical safeguard for the graph-free
    primal iteration only; the implicit correction rejects rays for which the
    floor would have engaged (see :meth:`_classify_final_roots`).
    """
    if eps is None:
        eps = _DENOM_EPS_MULTIPLIER * machine_eps(value)
    return be.where(
        be.abs(value) > eps,
        value,
        be.where(value >= 0, eps, -eps),
    )


def _denominator_threshold(value, scale=None, multiplier=_DENOM_EPS_MULTIPLIER):
    """Dtype- and scale-aware singularity threshold for ``dF/dt``.

    Implements ``tau = C * eps_dtype * max(1, scale)`` where ``scale`` is the
    local magnitude ``|s_x L| + |s_y M| + |N|``. For float64 this lands near
    the historical ``1e-14``; for float32 it is ~9 orders of magnitude larger,
    which is the point -- a fixed ``1e-14`` is below float32 round-off and so
    never triggers.
    """
    tau = multiplier * machine_eps(value)
    if scale is None:
        return tau
    return tau * be.maximum(scale, be.ones_like(scale))


def _regularize_signed(value, scale=None):
    """Floor ``value`` away from zero, sign-preserving and dtype-aware.

    Returns ``(regularized_value, near_singular_mask)``. The mask flags entries
    where the true sensitivity is singular (a tangent/grazing intersection) and
    the returned derivative is therefore a *regularization*, not the exact
    physics.
    """
    tau = _denominator_threshold(value, scale)
    near_singular = be.abs(value) <= tau
    floored = be.where(
        near_singular,
        be.where(value >= 0, tau * be.ones_like(value), -tau * be.ones_like(value)),
        value,
    )
    return floored, near_singular


def _effective_tolerance(tol, t):
    """Raise ``tol`` to the round-off floor of the working dtype.

    A residual can only ever be driven down to about ``eps * |t|``. The
    default ``tol=1e-10`` is comfortably reachable in float64 but sits *below*
    float32 round-off, so in float32 every ray would be classified
    non-converged -- suppressing its implicit gradient and emitting a warning
    on every call -- despite the root being as good as that dtype allows.

    In float64 the user-supplied tolerance dominates, so behavior there is
    unchanged.
    """
    scale = float(be.to_numpy(be.max(be.abs(t))))
    floor = _CONV_EPS_MULTIPLIER * machine_eps(t) * max(1.0, scale)
    return max(float(tol), floor)


@dataclass
class _DistanceSolveResult:
    """Outcome of the graph-free primal Newton-Raphson distance solve.

    Attributes:
        t: Propagation distance to the intersection.
        residual: Surface residual ``F(t) = sag(x, y) - z`` at ``t``.
        converged: Per-ray boolean mask, ``|F(t)| < tol``.
        iterations: Number of Newton updates actually performed.
    """

    t: Any
    residual: Any
    converged: Any
    iterations: int


@dataclass
class _RootClassification:
    """Regularity classification of every ray at the final primal root.

    ``regular`` rays satisfy the full implicit-function-theorem contract:
    converged, all quantities finite, ``|n_z|`` above the dtype-aware
    threshold and ``|dF/dt|`` above the dtype- and scale-aware threshold.
    Only these rays receive the exact first-order implicit derivative.

    The rejection masks are mutually exclusive diagnostic categories, in
    priority order: ``nonfinite`` (any non-finite quantity at the root),
    ``nonconverged`` (finite but did not meet the tolerance), ``nz_singular``
    (converged but the surface is not a valid local height function),
    ``near_singular`` (converged but tangent/grazing, ``|dF/dt|`` below
    threshold).

    Attributes:
        regular: Per-ray mask of rays eligible for implicit differentiation.
        df_dt: Detached, unclipped ``dF/dt`` at the final root.
        nonfinite: Rejection mask -- non-finite state at the root.
        nonconverged: Rejection mask -- residual above tolerance.
        nz_singular: Rejection mask -- ``|n_z|`` below dtype threshold.
        near_singular: Rejection mask -- ``|dF/dt|`` below threshold.
    """

    regular: Any
    df_dt: Any
    nonfinite: Any
    nonconverged: Any
    nz_singular: Any
    near_singular: Any


def _all_finite(*values):
    """Elementwise AND of ``isfinite`` over several same-shaped arrays."""
    finite = be.isfinite(values[0])
    for value in values[1:]:
        finite = be.logical_and(finite, be.isfinite(value))
    return finite


class NewtonRaphsonGeometry(StandardGeometry, ABC):
    """Represents a geometry that uses the Newton-Raphson method for ray tracing.

    Args:
        coordinate_system (CoordinateSystem): The coordinate system of the geometry.
        radius (float): The radius of curvature of the base sphere.
        conic (float, optional): The conic constant of the base sphere.
            Defaults to 0.0.
        tol (float, optional): Tolerance for Newton-Raphson iteration.
            Defaults to 1e-10.
        max_iter (int, optional): Maximum iterations for Newton-Raphson.
            Defaults to 100.

    """

    def __init__(self, coordinate_system, radius, conic=0.0, tol=1e-10, max_iter=100):
        super().__init__(coordinate_system, radius, conic)
        self.tol = tol
        self.max_iter = max_iter

    def __str__(self):
        return "Newton Raphson"  # pragma: no cover

    def flip(self):
        """Flip the geometry.

        Changes the sign of the radius of curvature.
        The conic constant remains unchanged.
        """
        self.radius = -self.radius

    @abstractmethod
    def sag(self, x=0, y=0):
        """Calculate the surface sag of the geometry.

        Args:
            x (float or be.ndarray, optional): The x-coordinate(s). Defaults to 0.
            y (float or be.ndarray, optional): The y-coordinate(s). Defaults to 0.

        Returns:
            float or be.ndarray: The surface sag of the geometry at the given
            coordinates.

        """
        # pragma: no cover

    @abstractmethod
    def _surface_normal(self, x, y):
        """Calculate the surface normal of the geometry at the given x and y
        position.

        Args:
            x (be.ndarray): The x-coordinate(s) at which to calculate the normal.
            y (be.ndarray): The y-coordinate(s) at which to calculate the normal.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The surface normal
            components (nx, ny, nz).

        """
        # pragma: no cover

    def surface_normal(self, rays):
        """Calculates the surface normal of the geometry at the given rays.

        Args:
            rays (RealRays): The rays, positioned at the surface, for which to
                calculate the surface normal.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The surface normal
            components (nx, ny, nz).

        """
        return self._surface_normal(rays.x, rays.y)

    # ------------------------------------------------------------------
    # Primal Newton-Raphson solve (no autograd graph)
    # ------------------------------------------------------------------

    def _surface_residual(self, t, rays):
        """Residual ``F(t) = sag(x0 + tL, y0 + tM) - (z0 + tN)``.

        The intersection distance ``t*`` is the root ``F(t*) = 0``.
        """
        x_int = rays.x + t * rays.L
        y_int = rays.y + t * rays.M
        z_int = rays.z + t * rays.N
        return self.sag(x_int, y_int) - z_int

    def _surface_residual_dt(self, t, rays):
        """Derivative ``dF/dt = s_x L + s_y M - N``, its scale, and validity.

        The sag slopes ``s_x, s_y`` are recovered from the surface normal.
        The returned scale ``|s_x L| + |s_y M| + |N|`` is used to build a
        scale-aware singularity threshold (see :func:`_denominator_threshold`).

        Returns:
            tuple: ``(df_dt, scale, nz_regular)``. The ``nz_regular`` mask
            flags rays for which ``|n_z|`` is safely above the dtype-aware
            threshold, i.e. the surface is a numerically valid local height
            function there. The primal solver may still take a floored step
            for irregular rays; the implicit correction must reject them.
        """
        x_int = rays.x + t * rays.L
        y_int = rays.y + t * rays.M

        nx, ny, nz = self._surface_normal(x_int, y_int)
        tau_nz = _nz_threshold(nz)
        nz_regular = be.logical_and(be.isfinite(nz), be.abs(nz) > tau_nz)
        nz_safe = _sign_preserving_floor(nz, tau_nz)
        fx = -nx / nz_safe
        fy = -ny / nz_safe

        df_dt = fx * rays.L + fy * rays.M - rays.N
        scale = be.abs(fx * rays.L) + be.abs(fy * rays.M) + be.abs(rays.N)
        return df_dt, scale, nz_regular

    def _solve_distance_primal(self, rays):
        """Run the Newton-Raphson iteration to find the intersection distance.

        This is a pure numerical solve with **no** autograd graph.  It is
        used both by the differentiable path (inside torch.no_grad) and by
        the non-differentiable path.

        Convergence is tested *before* the surface normal is evaluated, so a
        batch whose residual is already acceptable never triggers a normal
        computation (which can be singular at a grazing/tangent point) purely
        to satisfy the loop structure.

        Potential future optimization: support an optional fused
        ``eval_sag_and_grad(x, y)`` API to return ``(sag_val, fx, fy)``
        in a single pass and reduce duplicate surface computations.

        Args:
            rays: An object with attributes x, y, z, L, M, N.

        Returns:
            _DistanceSolveResult: Distance, final residual, per-ray
            convergence mask and iteration count.
        """
        # Better initial guess via base conic intersection
        t = super().distance(rays)

        tol = _effective_tolerance(self.tol, t)

        iterations = 0
        f_t = self._surface_residual(t, rays)
        converged = be.abs(f_t) < tol

        for i in range(self.max_iter):
            # 1-3. Convergence is checked before any normal evaluation.
            if be.all(converged):
                break

            # 4. Only reached while at least one ray is still unconverged.
            df_dt, scale, _ = self._surface_residual_dt(t, rays)
            safe_df_dt, _ = _regularize_signed(df_dt, scale)

            # 5. Freeze already-converged rays so a converged root is not
            # perturbed by further steps.
            step = be.where(converged, be.zeros_like(f_t), f_t / safe_df_dt)
            t = t - step
            iterations = i + 1

            f_t = self._surface_residual(t, rays)
            converged = be.abs(f_t) < tol

        return _DistanceSolveResult(
            t=t, residual=f_t, converged=converged, iterations=iterations
        )

    def _classify_final_roots(self, result, rays):
        """Classify every ray at the *final* primal root (graph-free).

        Must be called with autograd disabled. Evaluates ``dF/dt`` at the
        final root -- never reusing a value from an earlier iteration -- and
        builds the regularity mask required for the implicit derivative to be
        the exact first-order physics:

        ``regular = converged AND finite(everything) AND |n_z| > tau_nz
        AND |dF/dt| > tau_dFdt``

        Returns:
            _RootClassification: Detached masks and the detached, unclipped
            final-root ``dF/dt``.
        """
        t = result.t
        df_dt, scale, nz_regular = self._surface_residual_dt(t, rays)

        finite = _all_finite(
            t,
            result.residual,
            rays.x,
            rays.y,
            rays.z,
            rays.L,
            rays.M,
            rays.N,
            df_dt,
            scale,
        )

        tau = _denominator_threshold(df_dt, scale)
        denom_regular = be.abs(df_dt) > tau

        converged = result.converged
        regular = be.logical_and(
            be.logical_and(converged, finite),
            be.logical_and(nz_regular, denom_regular),
        )

        # Mutually exclusive diagnostic categories, in priority order.
        nonfinite = be.logical_not(finite)
        nonconverged = be.logical_and(finite, be.logical_not(converged))
        conv_finite = be.logical_and(finite, converged)
        nz_singular = be.logical_and(conv_finite, be.logical_not(nz_regular))
        near_singular = be.logical_and(
            be.logical_and(conv_finite, nz_regular), be.logical_not(denom_regular)
        )

        return _RootClassification(
            regular=regular,
            df_dt=df_dt,
            nonfinite=nonfinite,
            nonconverged=nonconverged,
            nz_singular=nz_singular,
            near_singular=near_singular,
        )

    def _surface_residual_subset(self, t, rays, mask):
        """Grad-attached residual evaluated only for the rays in ``mask``.

        Restricting the evaluation to the regular subset matters: PyTorch can
        propagate ``NaN`` through the *backward* pass from an invalid branch
        even when that branch is later discarded by ``where``. Rejected rays
        are therefore never traced through a grad-attached residual at all.
        """
        x_int = rays.x[mask] + t[mask] * rays.L[mask]
        y_int = rays.y[mask] + t[mask] * rays.M[mask]
        z_int = rays.z[mask] + t[mask] * rays.N[mask]
        return self.sag(x_int, y_int) - z_int

    def _warn_rejected_rays(self, state, result):
        """Emit one grouped ``RuntimeWarning`` for all rejected rays.

        Reports the per-category counts, the iteration count and (where
        meaningful) the worst residual, without dumping arrays.
        """

        def _count(mask) -> int:
            return int(be.to_numpy(be.sum(mask)))

        n_nonfinite = _count(state.nonfinite)
        n_nonconverged = _count(state.nonconverged)
        n_nz = _count(state.nz_singular)
        n_tangent = _count(state.near_singular)
        n_rejected = n_nonfinite + n_nonconverged + n_nz + n_tangent
        if n_rejected == 0:
            return

        parts = []
        if n_nonconverged:
            masked = be.where(state.nonconverged, be.abs(result.residual), 0.0)
            max_residual = float(be.to_numpy(be.max(masked)))
            parts.append(
                f"{n_nonconverged} non-converged "
                f"(max residual {max_residual:.3e} > tol {self.tol:.3e})"
            )
        if n_nonfinite:
            parts.append(f"{n_nonfinite} with non-finite state at the root")
        if n_tangent:
            parts.append(
                f"{n_tangent} tangent/grazing (|dF/dt| below the dtype- and "
                "scale-aware threshold)"
            )
        if n_nz:
            parts.append(
                f"{n_nz} with |n_z| below the dtype-aware threshold "
                "(surface not a valid local height function)"
            )

        n_total = int(be.size(result.t))
        warnings.warn(
            f"Newton-Raphson intersection rejected {n_rejected} of {n_total} "
            f"ray(s) from implicit differentiation after {result.iterations} "
            f"iteration(s): {'; '.join(parts)}. Rejected rays keep their "
            "detached primal forward value and contribute zero gradient.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _invalidate_cached_derived_state_for_autograd(self) -> None:
        """Invalidate derived caches that must be rebuilt grad-attached.

        Default no-op. Subclasses that cache tensors derived from trainable
        parameters may override this to force a rebuild before the
        differentiable correction. Forbes geometries instead build their
        coefficient cache under an explicit ``torch.enable_grad()``, which
        keeps the cache differentiable regardless of the caller's grad
        context and avoids rebuilding it on every differentiable trace.
        """

    # ------------------------------------------------------------------
    # Public distance method with autograd dispatch
    # ------------------------------------------------------------------

    def distance(self, rays):
        """
        Calculates the distance from the ray origin to the surface intersection
        using a robust Newton-Raphson method. This version uses the base conic
        intersection as a strong initial guess.

        **Differentiable mode (torch backend with grad enabled):**

        The primal Newton-Raphson solve runs inside ``torch.no_grad()`` so
        that the iterative loop is never recorded in the autograd graph.  A
        one step implicit correction (in DiffOptics style) is then applied:

            t_implicit = t_detached - F(t_detached) / (dF/dt)_detached

        Since F is near zero at convergence, the forward value is unchanged,
        but the gradients are correct to first order via the implicit function
        theorem.

        Note:
            This implicit correction is intended for correct first-order
            gradients. Higher order derivatives (double backward and beyond)
            are not guaranteed to match the exact unrolled Newton system.

        **Non differentiable mode (numpy backend, or torch without grad):**

        Returns the converged t directly.

        Assumptions required for the implicit derivative to be exact:

        1. the primal solve converged to the intended physical root;
        2. that root stays on the same branch under small parameter changes;
        3. ``dF/dt`` is not zero or numerically singular (no tangent/grazing
           intersection);
        4. ``|n_z|`` is above the dtype-aware threshold, so the surface is a
           numerically valid local height function;
        5. ``sag()`` and ``_surface_normal()`` describe the same surface;
        6. only first derivatives are supported.

        Rays that fail any of (1)-(4) keep their **detached** primal forward
        value and are never evaluated through a grad-attached residual, so a
        failed or singular root never carries a confident-looking but invalid
        gradient and never contaminates the gradients of valid rays in the
        same batch. A grouped ``RuntimeWarning`` reports the rejections by
        category.

        Args:
            rays (RealRays): The rays used for calculating distance.

        Returns:
            be.ndarray: An array of propagation distances 't' from each ray's
            current position to its intersection point with the geometry.
        """
        use_torch_diff = (
            torch is not None
            and be.get_backend() == "torch"
            and torch.is_grad_enabled()
        )

        # --- Phase A: graph-free primal solve and root classification ----
        ctx = torch.no_grad() if use_torch_diff else contextlib.nullcontext()

        with ctx:
            result = self._solve_distance_primal(rays)
            state = self._classify_final_roots(result, rays) if use_torch_diff else None

        if not use_torch_diff:
            return result.t

        # Give subclasses a chance to rebuild caches that must be attached to
        # the autograd graph before the differentiable correction runs.
        self._invalidate_cached_derived_state_for_autograd()

        # --- Phase B: grad-attached correction on the regular subset -----
        # DiffOptics-style one-step implicit correction,
        #
        #     t_valid = t_bar - F(t_bar, theta) / stopgrad(dF/dt),
        #
        # applied only to rays whose final root is regular. The forward value
        # is approximately unchanged (F ~ 0 at convergence) but the expression
        # carries the exact first-order gradient dt*/dtheta = -F_theta / F_t.
        # The implicit function theorem needs only the *value* of the inverse
        # Jacobian for a first derivative, so the denominator is detached --
        # and for regular rays it is used unclipped: regularity guarantees it
        # is safely above the singularity threshold.
        t_out = result.t.detach()
        regular = state.regular

        if bool(be.to_numpy(be.all(regular))):
            F = self._surface_residual(t_out, rays)
            t_out = t_out - F / state.df_dt
        elif bool(be.to_numpy(be.any(regular))):
            F_valid = self._surface_residual_subset(t_out, rays, regular)
            t_valid = t_out[regular] - F_valid / state.df_dt[regular]
            # Functional scatter: keeps t_out detached everywhere else while
            # gradients flow from the inserted values.
            t_out = t_out.masked_scatter(regular, t_valid)
        # If no ray is regular, the detached primal result is returned without
        # ever evaluating the differentiable residual.

        # --- Phase C: grouped failure reporting --------------------------
        self._warn_rejected_rays(state, result)
        return t_out

    def _intersection_plane(self, rays):
        """Calculates the intersection points of the rays with a plane (z=0).

        Args:
            rays (RealRays): The rays to calculate the intersection points for.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The x, y, and z
            coordinates of the intersection points.
        """
        # handle infinite radius: intersection with plane z=0
        t = be.full_like(rays.z, be.nan)

        # rays not parallel to the XY plane (N != 0)
        mask_N_nonzero = be.abs(rays.N) > self.tol

        t = be.where(mask_N_nonzero, -rays.z / rays.N, t)

        mask_N_zero_and_z_zero = (~mask_N_nonzero) & (be.abs(rays.z) < self.tol)
        t = be.where(mask_N_zero_and_z_zero, 0.0, t)

        x = rays.x + rays.L * t
        y = rays.y + rays.M * t
        z = rays.z + rays.N * t

        return x, y, z

    def _intersection_sphere(self, rays):
        """Calculates the intersection points of the rays with the geometry.

        Args:
            rays (RealRays): The rays to calculate the intersection points for.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The x, y, and z
            coordinates of the intersection points.

        """
        a = rays.L**2 + rays.M**2 + rays.N**2
        b = (
            2 * rays.L * rays.x
            + 2 * rays.M * rays.y
            - 2 * rays.N * self.radius
            + 2 * rays.N * rays.z
        )
        c = rays.x**2 + rays.y**2 + rays.z**2 - 2 * self.radius * rays.z

        # discriminant
        d = b**2 - 4 * a * c

        # two solutions for distance to sphere
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t1 = (-b + be.sqrt(d)) / (2 * a)
            t2 = (-b - be.sqrt(d)) / (2 * a)

        # find intersection points in z
        z1 = rays.z + t1 * rays.N
        z2 = rays.z + t2 * rays.N

        # take intersection closest to z = 0 (i.e., vertex of geometry)
        t = be.where(be.abs(z1) <= be.abs(z2), t1, t2)

        # handle case when a = 0
        cond = a == 0
        t[cond] = -c[cond] / b[cond]

        x = rays.x + rays.L * t
        y = rays.y + rays.M * t
        z = rays.z + rays.N * t

        return x, y, z

    def _intersection(self, rays):
        """Calculates the initial intersection points of the rays with the base
        geometry (sphere or plane) before Newton-Raphson iteration.

        Args:
            rays (RealRays): The rays to calculate the intersection points for.

        Returns:
            tuple[be.ndarray, be.ndarray, be.ndarray]: The x, y, and z
            coordinates of the initial intersection points.
        """
        if _is_radius_infinite(self.radius):
            return self._intersection_plane(rays)
        else:
            return self._intersection_sphere(rays)

    def to_dict(self):
        """Converts the geometry to a dictionary.

        Returns:
            dict: The dictionary representation of the geometry.

        """
        geometry_dict = super().to_dict()
        geometry_dict.update({"tol": self.tol, "max_iter": self.max_iter})
        return geometry_dict

    @classmethod
    def from_dict(cls, data):  # pragma: no cover
        """Creates a geometry from a dictionary representation.

        Args:
            data (dict): The dictionary representation of the geometry.

        Returns:
            NewtonRaphsonGeometry: An instance of a subclass of
            NewtonRaphsonGeometry, created from the dictionary data.

        """
        required_keys = {"cs", "radius"}
        if not required_keys.issubset(data):
            missing = required_keys - data.keys()
            raise ValueError(f"Missing required keys: {missing}")

        cs = CoordinateSystem.from_dict(data["cs"])
        conic = data.get("conic", 0.0)
        tol = data.get("tol", 1e-10)
        max_iter = data.get("max_iter", 100)

        return cls(cs, data["radius"], conic, tol, max_iter)
