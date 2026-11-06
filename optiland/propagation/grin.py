"""Gradient-index (GRIN) propagation model.

Traces real rays through a medium whose refractive index varies with position.
The ray equation is integrated with a fixed-step fourth-order Runge-Kutta
scheme in Sharma's form (A. Sharma, D. V. Kumar and A. K. Ghatak, "Tracing rays
through graded-index media: a new method," Appl. Opt. 21, 984-987, 1982).

Jie Cai (zhazhajust), 2025
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import optiland.backend as be
from optiland.propagation.base import BasePropagationModel

if TYPE_CHECKING:
    from optiland._types import BEArray, ScalarOrArray
    from optiland.materials.base import BaseMaterial
    from optiland.rays.real_rays import RealRays
    from optiland.surfaces.standard_surface import Surface

_Vector = tuple["BEArray", "BEArray", "BEArray"]
_State = tuple["BEArray", "BEArray", "BEArray", "BEArray", "BEArray", "BEArray"]


class SupportsIndexGradient(Protocol):
    """A medium that can be traced by :class:`GRINPropagation`."""

    def index_and_gradient(
        self, x: BEArray, y: BEArray, z: BEArray, wavelength: ScalarOrArray
    ) -> tuple[BEArray, BEArray, BEArray, BEArray]:
        """Returns ``n`` and ``dn/dx``, ``dn/dy``, ``dn/dz`` in the medium frame."""


class GRINPropagation(BasePropagationModel):
    """Propagates rays through a gradient-index medium.

    The medium is the ``material_post`` of the surface where it starts, and its
    coordinates are that surface's local frame. Rays are integrated in steps of
    about ``step_size`` along the path until they cross the next surface, then
    a straight chord lands them on it. This model accumulates the optical path
    itself and evaluates the index at each ray hit for refraction.

    Rays that do not reach the next surface within ``max_steps`` steps, or that
    leave the region where the index is valid, are vignetted.

    Args:
        step_size: Approximate geometric step length in mm.
        max_steps: Maximum number of integration steps per medium.
    """

    def __init__(self, step_size: float = 0.1, max_steps: int = 10_000):
        if step_size <= 0:
            raise ValueError("step_size must be positive.")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        self.step_size = float(step_size)
        self.max_steps = int(max_steps)

    def propagate(self, rays: RealRays, t: float) -> None:
        """Raises NotImplementedError: GRIN propagation needs the next surface.

        Args:
            rays: The rays object to be propagated.
            t: The distance to propagate.
        """
        raise NotImplementedError(
            "GRIN propagation runs between two surfaces; a fixed distance is "
            "not supported. Use the medium between surfaces of an optic."
        )

    def paraxial_transfer(
        self, y: BEArray, u: BEArray, t: BEArray
    ) -> tuple[BEArray, BEArray]:
        """Raises NotImplementedError: paraxial GRIN tracing is not supported.

        Args:
            y: Ray heights.
            u: Ray slopes.
            t: The axial distance.
        """
        raise NotImplementedError(
            "Paraxial tracing through a GRIN medium is not supported. Place the "
            "aperture stop at or before the GRIN medium and use real-ray analyses."
        )

    def propagate_to_surface(self, rays: RealRays, surface: Surface) -> None:
        """Integrates rays through the medium onto ``surface``.

        Args:
            rays: The rays, in the local frame of ``surface``. Modified
                in-place.
            surface: The surface at the end of the medium.
        """
        self._trace(rays, surface, record=False)

    def boundary_index(
        self, rays: RealRays, surface: Surface, entrance: Surface
    ) -> BEArray:
        """Returns the medium index at the ray hits on ``surface``.

        Args:
            rays: The rays, in the local frame of ``surface``.
            surface: The surface the rays are on.
            entrance: The surface where the medium starts.

        Returns:
            The refractive index for each ray.
        """
        medium = _medium(entrance)
        to_medium = _frame_map(surface, entrance)
        x, y, z = _transform(to_medium, rays.x, rays.y, rays.z)
        return medium.index_and_gradient(x, y, z, rays.w)[0]

    def display_path(self, rays: RealRays, surface: Surface) -> BEArray:
        """Returns global points along the ray paths through the medium.

        Only the given rays are integrated; analysis traces never store paths.

        Args:
            rays: The rays leaving the previous surface, in global coordinates.
                Not modified.
            surface: The surface at the end of the medium.

        Returns:
            Points of shape ``(num_samples, num_rays, 3)``, NaN after a ray
            reaches ``surface`` or is vignetted.
        """
        from optiland.rays.real_rays import RealRays

        rays = RealRays(rays.x, rays.y, rays.z, rays.L, rays.M, rays.N, rays.i, rays.w)
        surface.geometry.localize(rays)
        samples = self._trace(rays, surface, record=True)
        x, y, z = (be.stack(component) for component in zip(*samples, strict=True))
        shape = x.shape
        flat = [be.ravel(c) for c in (x, y, z)]
        zero, one = be.zeros_like(flat[0]), be.ones_like(flat[0])
        points = RealRays(*flat, zero, zero, one, one, one)
        surface.geometry.globalize(points)
        return be.stack(
            [be.reshape(c, shape) for c in (points.x, points.y, points.z)], axis=-1
        )

    def _trace(
        self, rays: RealRays, surface: Surface, record: bool
    ) -> list[_Vector] | None:
        """Integrates rays onto ``surface``; optionally returns sampled points."""
        from optiland.surfaces.standard_surface import _aperture_aware_distance

        entrance = _entrance(surface)
        medium = _medium(entrance)
        to_surface = _frame_map(entrance, surface)
        rotation = to_surface[0]
        to_medium = _frame_map(surface, entrance)

        x, y, z = _transform(to_medium, rays.x, rays.y, rays.z)
        L, M, N = _rotate(to_medium[0], rays.L, rays.M, rays.N)
        n = medium.index_and_gradient(x, y, z, rays.w)[0]
        state = (x, y, z, n * L, n * M, n * N)
        step = self.step_size / n
        opl = be.zeros_like(x)

        def residual(s: _State) -> BEArray:
            xs, ys, zs = _transform(to_surface, s[0], s[1], s[2])
            return zs - surface.geometry.sag(xs, ys)

        active = be.isfinite(n) & (n > 0)
        lost = ~active
        f_prev = residual(state)
        samples = [_transform(to_surface, *state[:3])] if record else None
        for _ in range(self.max_steps):
            if not be.any(active):
                break
            new_state, d_opl = _rk4_step(medium, state, step, rays.w)
            f_new = residual(new_state)
            valid = be.isfinite(f_new) & be.isfinite(d_opl)
            crossed = active & valid & (f_prev * f_new <= 0)
            lost = lost | (active & ~valid)
            advance = active & valid & ~crossed
            state = tuple(
                be.where(advance, new, old)
                for new, old in zip(new_state, state, strict=True)
            )
            opl = be.where(advance, opl + d_opl, opl)
            f_prev = be.where(advance, f_new, f_prev)
            active = advance
            if record:
                samples.append(_masked(advance, _transform(to_surface, *state[:3])))
        lost = lost | active

        x, y, z, Tx, Ty, Tz = state
        n_end = be.sqrt(Tx**2 + Ty**2 + Tz**2)
        rays.x, rays.y, rays.z = _transform(to_surface, x, y, z)
        rays.L, rays.M, rays.N = _rotate(rotation, Tx / n_end, Ty / n_end, Tz / n_end)
        rays.is_normalized = True

        chord = _aperture_aware_distance(surface, rays)
        rays.x = rays.x + chord * rays.L
        rays.y = rays.y + chord * rays.M
        rays.z = rays.z + chord * rays.N
        rays.opd = rays.opd + opl + chord * n_end
        rays.clip(lost)

        if record:
            samples.append(_masked(~lost, (rays.x, rays.y, rays.z)))
        return samples

    def to_dict(self) -> dict:
        """Serializes the model and its integrator settings.

        Returns:
            The dictionary representation.
        """
        data = super().to_dict()
        data.update({"step_size": self.step_size, "max_steps": self.max_steps})
        return data

    @classmethod
    def from_dict(cls, d: dict, material: BaseMaterial = None) -> GRINPropagation:
        """Creates a GRINPropagation model from a dictionary.

        Args:
            d: The dictionary representation of the model.
            material: The parent material instance. Accepted for API
                consistency; the medium is read from the traced surfaces.

        Returns:
            An instance of the GRINPropagation model.
        """
        return cls(
            step_size=d.get("step_size", 0.1), max_steps=d.get("max_steps", 10_000)
        )


def _rk4_step(
    medium: SupportsIndexGradient, state: _State, h: BEArray, wavelength: BEArray
) -> tuple[_State, BEArray]:
    """One RK4 step of ``dr/dτ = T``, ``dT/dτ = n∇n``, ``dOPL/dτ = n²``."""

    def derivative(s: _State) -> tuple[_State, BEArray]:
        n, gx, gy, gz = medium.index_and_gradient(s[0], s[1], s[2], wavelength)
        return (s[3], s[4], s[5], n * gx, n * gy, n * gz), n * n

    def shifted(s: _State, k: _State, scale: float) -> _State:
        return tuple(a + scale * h * b for a, b in zip(s, k, strict=True))

    k1, q1 = derivative(state)
    k2, q2 = derivative(shifted(state, k1, 0.5))
    k3, q3 = derivative(shifted(state, k2, 0.5))
    k4, q4 = derivative(shifted(state, k3, 1.0))
    new_state = tuple(
        s + h / 6 * (a + 2 * b + 2 * c + d)
        for s, a, b, c, d in zip(state, k1, k2, k3, k4, strict=True)
    )
    return new_state, h / 6 * (q1 + 2 * q2 + 2 * q3 + q4)


def _entrance(surface: Surface) -> Surface:
    """Returns the surface where the medium before ``surface`` starts."""
    from optiland.interactions.phase_interaction_model import PhaseInteractionModel

    previous = surface.previous_surface
    if getattr(previous, "geometry", None) is None:
        raise NotImplementedError(
            "GRIN media are supported only between two surfaces of an optic's "
            "forward sequence."
        )
    for face in (previous, surface):
        if isinstance(face.interaction_model, PhaseInteractionModel):
            raise NotImplementedError(
                "Phase surfaces bounding a GRIN medium are not supported."
            )
    return previous


def _medium(entrance: Surface) -> SupportsIndexGradient:
    """Returns the GRIN medium that starts at ``entrance``."""
    medium = entrance.material_post
    if not hasattr(medium, "index_and_gradient"):
        raise TypeError(
            f"GRINPropagation needs a medium with index_and_gradient(), such as "
            f"GradientMaterial; got {type(medium).__name__}."
        )
    return medium


def _frame_map(source: Surface, target: Surface) -> tuple[BEArray, BEArray]:
    """Returns ``(R, t)`` with ``p_target = R @ p_source + t`` for local frames."""
    from optiland.rays.real_rays import RealRays

    zero, one = [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]
    points = RealRays(
        [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
        zero, zero, one, one, one,
    )  # fmt: skip
    source.geometry.globalize(points)
    target.geometry.localize(points)
    origin = be.stack([points.x[0], points.y[0], points.z[0]])
    columns = [
        be.stack([points.x[i], points.y[i], points.z[i]]) - origin for i in (1, 2, 3)
    ]
    return be.stack(columns, axis=1), origin


def _rotate(R: BEArray, x: BEArray, y: BEArray, z: BEArray) -> _Vector:
    """Applies the 3x3 matrix ``R`` to vectors given by components."""
    return tuple(R[i, 0] * x + R[i, 1] * y + R[i, 2] * z for i in range(3))


def _transform(frame: tuple[BEArray, BEArray], x, y, z) -> _Vector:
    """Applies the rigid map ``(R, t)`` to points given by components."""
    R, t = frame
    return tuple(c + t[i] for i, c in enumerate(_rotate(R, x, y, z)))


def _masked(keep: BEArray, points: _Vector) -> _Vector:
    """Replaces points of rays not in ``keep`` with NaN."""
    return tuple(be.where(keep, c, be.full_like(c, be.nan)) for c in points)
