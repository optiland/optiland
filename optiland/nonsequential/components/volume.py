"""Volume -- a closed, outward-oriented solid built from boundary surfaces.

Medium sidedness (which material a ray is entering) is fixed geometrically
per-surface -- see ``RefractiveComponent.interact`` -- and stays the sole
source of truth for n1/n2. ``Volume`` is a separate, independent check: a
compound component's boundary surfaces are supposed to form a genuinely
closed solid, and nothing else checks that. It validates a boundary list at
construction time and raises loudly (``NonWatertightVolumeError``) if the
surfaces do not actually close up or are inconsistently oriented, rather
than letting a silent gap leak flux at trace time.

A ray-level medium stack (``NSQRayBundle.medium_stack``/``medium_depth``,
pushed/popped by ``RefractiveComponent.interact`` on every transmitted ray)
runs alongside this as a runtime cross-check: it does not feed back into
n1/n2 either, but a pop on an empty stack is counted in
``Diagnostics.medium_stack_underflows`` as a likely geometry defect. This
``Volume`` boundary list is not yet wired into that stack via ids (there is
no ``SceneIR``-level ``VolumeIR`` population), so the stack's push/pop
identity currently comes from ``NSQMaterial`` object identity
(``optiland.nonsequential.materials.nsq_material.medium_stack_id``), not
from a volume registry.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential._utils import as_float
from optiland.nonsequential.components.base import BaseComponent, _get_transform

if TYPE_CHECKING:
    from optiland.nonsequential.materials.nsq_material import NSQMaterial

# Default rim-coincidence tolerance [mm]. Proposed in the original spec as a
# starting point; the sag arithmetic in analytic geometries stays well
# within float64 precision at ordinary lens scales, so this is not tuned
# further here.
WATERTIGHT_TOL = 1e-6

_RIM_SAMPLES = 64
_PARITY_DIRECTIONS = 8
_PARITY_SEED = 0
_PARITY_MAX_BOUNCES = 64
_PARITY_EPSILON = 1e-6


class NonWatertightVolumeError(Exception):
    """A Volume's boundary surfaces do not form a closed, consistently
    outward-oriented solid.

    Raised at :class:`Volume` construction, never as a warning: a leak in
    the boundary lets rays enter or exit a solid without the medium stack
    (or, in this revamp, the per-surface geometric sidedness check)
    noticing, which is exactly the class of silent-wrong-answer failure
    this validation exists to prevent.
    """


def _rectangle_perimeter(half_width: float, half_height: float, n: int) -> np.ndarray:
    """Sample ``n`` points roughly evenly around a rectangle's perimeter.

    Args:
        half_width: Half-width along local x [mm].
        half_height: Half-height along local y [mm].
        n: Number of points to sample.

    Returns:
        (n, 3) array of local-frame points at z=0.
    """
    perim = 4.0 * (half_width + half_height)
    if perim <= 0.0:
        return np.zeros((n, 3))
    s = (np.arange(n) / n) * perim
    x = np.zeros(n)
    y = np.zeros(n)
    # Walk the perimeter starting at (+hw, -hh), going counter-clockwise.
    edges = [
        (2 * half_width, (1.0, 0.0), (-half_width, -half_height)),
        (2 * half_height, (0.0, 1.0), (half_width, -half_height)),
        (2 * half_width, (-1.0, 0.0), (half_width, half_height)),
        (2 * half_height, (0.0, -1.0), (-half_width, half_height)),
    ]
    remaining = s.copy()
    start = 0.0
    for length, (dx, dy), (ox, oy) in edges:
        on_edge = (remaining >= start) & (remaining < start + length)
        local_s = remaining[on_edge] - start
        x[on_edge] = ox + dx * local_s
        y[on_edge] = oy + dy * local_s
        start += length
    return np.stack([x, y, np.zeros(n)], axis=1)


def _rim_points(
    component: BaseComponent, n_samples: int = _RIM_SAMPLES
) -> np.ndarray | None:
    """Sample points along a component's aperture rim, in global coordinates.

    Supports the analytic geometries the compound builders actually use
    (conic, finite plane, annulus, frustum). Geometries with no finite open
    edge (an infinite plane, a full sphere, a mesh) return ``None`` -- there
    is nothing for a neighbouring surface to meet, so watertightness
    contributes nothing to check for them.

    Args:
        component: The boundary surface to sample.
        n_samples: Points per rim loop.

    Returns:
        (n_samples * num_loops, 3) global-frame points, or ``None``.
    """
    from optiland.nonsequential.components.geometry.analytic.annulus import (  # noqa: PLC0415
        AnnularPlaneGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.conic import (  # noqa: PLC0415
        ConicGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.frustum import (  # noqa: PLC0415
        CylindricalFrustumGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.plane import (  # noqa: PLC0415
        FinitePlaneGeometry,
    )
    from optiland.nonsequential.components.lens import _sag_at_rim  # noqa: PLC0415

    geom = component.geometry
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    loops: list[np.ndarray] = []

    if isinstance(geom, ConicGeometry):
        r = as_float(geom.aperture_radius)
        z = _sag_at_rim(as_float(geom.radius), as_float(geom.conic), r)
        loops.append(
            np.stack(
                [r * np.cos(theta), r * np.sin(theta), np.full(n_samples, z)], axis=1
            )
        )
    elif isinstance(geom, FinitePlaneGeometry):
        if geom.aperture_radius is not None:
            r = as_float(geom.aperture_radius)
            loops.append(
                np.stack(
                    [r * np.cos(theta), r * np.sin(theta), np.zeros(n_samples)], axis=1
                )
            )
        else:
            hw = as_float(geom.width) / 2.0
            hh = as_float(geom.height) / 2.0
            loops.append(_rectangle_perimeter(hw, hh, n_samples))
    elif isinstance(geom, AnnularPlaneGeometry):
        ri = as_float(geom.inner_radius)
        ro = as_float(geom.outer_radius)
        z = as_float(geom.z_offset)
        loops.append(
            np.stack(
                [ri * np.cos(theta), ri * np.sin(theta), np.full(n_samples, z)], axis=1
            )
        )
        loops.append(
            np.stack(
                [ro * np.cos(theta), ro * np.sin(theta), np.full(n_samples, z)], axis=1
            )
        )
    elif isinstance(geom, CylindricalFrustumGeometry):
        rf, zf = as_float(geom.r_front), as_float(geom.z_front)
        rb, zb = as_float(geom.r_back), as_float(geom.z_back)
        loops.append(
            np.stack(
                [rf * np.cos(theta), rf * np.sin(theta), np.full(n_samples, zf)], axis=1
            )
        )
        loops.append(
            np.stack(
                [rb * np.cos(theta), rb * np.sin(theta), np.full(n_samples, zb)], axis=1
            )
        )
    else:
        # Infinite plane, sphere, mesh: no finite rim supported yet.
        return None

    local_pts = np.concatenate(loops, axis=0)
    translation, rotation = _get_transform(component.cs)
    return local_pts @ rotation.T + translation


def _check_watertight(
    boundary: list[BaseComponent], tol: float = WATERTIGHT_TOL
) -> np.ndarray | None:
    """Verify every boundary surface's rim is met by a neighbour's rim.

    Args:
        boundary: The volume's boundary surfaces.
        tol: Maximum allowed gap [mm].

    Returns:
        All sampled rim points (for reuse as a centroid estimate), or
        ``None`` if no surface in ``boundary`` has a finite rim.

    Raises:
        NonWatertightVolumeError: If any rim point is farther than ``tol``
            from every other surface's rim.
    """
    rims = [(comp, pts) for comp in boundary if (pts := _rim_points(comp)) is not None]
    if not rims:
        return None
    if len(rims) == 1:
        return rims[0][1]

    for i, (comp_i, pts_i) in enumerate(rims):
        other_pts = np.concatenate(
            [p for j, (_, p) in enumerate(rims) if j != i], axis=0
        )
        # (n_i, n_other) pairwise distances -- rim samples are small (a few
        # hundred points across a handful of surfaces), so this is cheap.
        diff = pts_i[:, None, :] - other_pts[None, :, :]
        dists = np.sqrt((diff**2).sum(axis=2)).min(axis=1)
        worst = float(dists.max())
        if worst > tol:
            raise NonWatertightVolumeError(
                f"Volume boundary is not watertight: surface "
                f"'{comp_i.name or type(comp_i).__name__}' has a rim point "
                f"{worst:.3g} mm from the nearest point on any other boundary "
                f"surface (tolerance {tol:.1e} mm). Check that neighbouring "
                f"surfaces' aperture radii and rim geometry agree."
            )
    return np.concatenate([p for _, p in rims], axis=0)


class _DetachedProxy:
    """A minimal (cs, geometry) pair usable with ``BaseComponent.intersect``.

    Not a real component -- just enough duck-typed surface for
    ``intersect()`` (which only reads ``self.cs``/``self.geometry``) to
    work against a fully detached, plain-float geometry clone.
    """

    def __init__(self, cs: object, geometry: object) -> None:
        self.cs = cs
        self.geometry = geometry

    intersect = BaseComponent.intersect


def _detached_cs(cs: object) -> object:
    """Return a plain-float clone of a ``CoordinateSystem``.

    A differentiable scene may give x/y/z/rx/ry/rz live ``torch.Tensor``
    values (position/tilt are not currently differentiable NSQ parameters,
    but the ``CoordinateSystem`` type itself does not forbid it). Even a
    numpy-backend computation cannot touch a tensor that requires grad
    without detaching it first -- ``be.cos(rx)`` fails exactly like any
    other numpy ufunc would. Recurses through ``reference_cs`` chains, the
    same nesting :mod:`optiland.nonsequential.serialization` already
    detaches for JSON export.

    Args:
        cs: A live ``CoordinateSystem``.

    Returns:
        A new ``CoordinateSystem`` with every field a plain float.
    """
    from optiland.coordinate_system import CoordinateSystem  # noqa: PLC0415

    return CoordinateSystem(
        x=as_float(cs.x),
        y=as_float(cs.y),
        z=as_float(cs.z),
        rx=as_float(cs.rx),
        ry=as_float(cs.ry),
        rz=as_float(cs.rz),
        reference_cs=_detached_cs(cs.reference_cs) if cs.reference_cs else None,
    )


def _detached_geometry(geometry: object) -> object:
    """Return a plain-float clone of ``geometry`` for construction-time checks.

    The watertightness/ray-parity checks never need gradients (they run
    once, at construction, on discrete pass/fail geometry) but a
    differentiable scene may attach a ``torch.Tensor`` radius/conic/etc to
    the live geometry. Cloning with :func:`as_float` keeps that live
    component's tensor untouched while giving this check plain numpy
    arithmetic to work with, regardless of which backend is currently
    active.

    Args:
        geometry: A live ``ComponentGeometry`` instance.

    Returns:
        A new instance of the same class with every numeric parameter
        detached to a plain float, or ``geometry`` itself if its class is
        not one of the parametrized analytic geometries (nothing to
        detach).
    """
    from optiland.nonsequential.components.geometry.analytic.annulus import (  # noqa: PLC0415
        AnnularPlaneGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.conic import (  # noqa: PLC0415
        ConicGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.frustum import (  # noqa: PLC0415
        CylindricalFrustumGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.plane import (  # noqa: PLC0415
        FinitePlaneGeometry,
    )
    from optiland.nonsequential.components.geometry.analytic.sphere import (  # noqa: PLC0415
        SphereGeometry,
    )

    if isinstance(geometry, ConicGeometry):
        return ConicGeometry(
            as_float(geometry.radius),
            as_float(geometry.conic),
            as_float(geometry.aperture_radius),
        )
    if isinstance(geometry, FinitePlaneGeometry):
        ap = geometry.aperture_radius
        return FinitePlaneGeometry(
            as_float(geometry.width),
            as_float(geometry.height),
            as_float(ap) if ap is not None else None,
        )
    if isinstance(geometry, AnnularPlaneGeometry):
        return AnnularPlaneGeometry(
            as_float(geometry.inner_radius),
            as_float(geometry.outer_radius),
            as_float(geometry.z_offset),
        )
    if isinstance(geometry, CylindricalFrustumGeometry):
        return CylindricalFrustumGeometry(
            as_float(geometry.r_front),
            as_float(geometry.r_back),
            as_float(geometry.z_front),
            as_float(geometry.z_back),
        )
    if isinstance(geometry, SphereGeometry):
        ap = geometry.aperture_radius
        return SphereGeometry(
            as_float(geometry.radius), as_float(ap) if ap is not None else None
        )
    return geometry


def _count_crossings(
    boundary: list[BaseComponent],
    origin: np.ndarray,
    direction: np.ndarray,
    max_bounces: int = _PARITY_MAX_BOUNCES,
) -> int:
    """Count how many times a ray crosses the boundary before escaping.

    Reuses each component's own ``intersect()`` -- no separate "all hits
    along a ray" geometry API is needed: the ray is walked hit-by-hit,
    nudged past each crossing by a small epsilon, and re-intersected against
    every boundary surface, up to ``max_bounces`` (a bound, not an
    unbounded loop).

    Args:
        boundary: The volume's boundary surfaces.
        origin: Ray start point, shape (3,).
        direction: Unit ray direction, shape (3,).
        max_bounces: Maximum crossings to count before giving up.

    Returns:
        Number of boundary crossings before the ray escapes to infinity.
    """
    from optiland.nonsequential.ray_bundle import NSQRayBundle  # noqa: PLC0415

    proxies = [
        _DetachedProxy(_detached_cs(comp.cs), _detached_geometry(comp.geometry))
        for comp in boundary
    ]

    o = origin.astype(np.float64).copy()
    d = direction.astype(np.float64).copy()
    count = 0
    for _ in range(max_bounces):
        t_min = np.inf
        for comp in proxies:
            rays = NSQRayBundle(
                x=np.array([o[0]]),
                y=np.array([o[1]]),
                z=np.array([o[2]]),
                L=np.array([d[0]]),
                M=np.array([d[1]]),
                N=np.array([d[2]]),
                flux=np.array([1.0]),
                wavelength=np.array([0.55]),
                n_current=np.array([1.0]),
                bounce=np.array([0], dtype=np.int32),
                alive=np.array([True]),
                ray_id=np.array([0], dtype=np.int64),
            )
            t, _normals, hit, _n_geom = comp.intersect(rays)
            t0 = float(t[0])
            if bool(hit[0]) and t0 < t_min:
                t_min = t0
        if not np.isfinite(t_min):
            break
        o = o + (t_min + _PARITY_EPSILON) * d
        count += 1
    return count


def _check_normals_outward(
    boundary: list[BaseComponent],
    centroid: np.ndarray,
    num_directions: int = _PARITY_DIRECTIONS,
    seed: int = _PARITY_SEED,
) -> None:
    """Ray-parity check: every direction from ``centroid`` must exit an odd
    number of times.

    A point inside a closed, consistently outward-oriented solid crosses
    its boundary an odd number of times along any ray to infinity; an even
    count means either a gap (the ray slipped through unnoticed) or a
    surface with an inconsistent orientation.

    Args:
        boundary: The volume's boundary surfaces.
        centroid: An interior point estimate, shape (3,).
        num_directions: Number of random directions to test.
        seed: RNG seed, for a deterministic (reproducible) check.

    Raises:
        NonWatertightVolumeError: If any tested direction crosses an even
            number of times.
    """
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(num_directions, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    for d in dirs:
        count = _count_crossings(boundary, centroid, d)
        if count == 0 or count % 2 == 0:
            raise NonWatertightVolumeError(
                f"Volume boundary failed the inside/outside ray-parity check: "
                f"a ray cast from the estimated interior point "
                f"{centroid.tolist()} in direction {d.tolist()} crossed the "
                f"boundary {count} times (expected an odd, nonzero count for "
                f"a point inside a closed, consistently outward-oriented "
                f"solid). This usually means a gap in the boundary, a "
                f"surface with an unexpectedly flipped normal, or that the "
                f"estimated interior point is not actually inside the solid."
            )


@dataclass
class Volume:
    """A closed, outward-oriented solid built from boundary surfaces.

    Validated at construction: every boundary surface's rim must be met by
    a neighbour's rim (watertightness), and the boundary must enclose its
    own estimated interior point consistently from every direction
    (orientation). Both checks raise :class:`NonWatertightVolumeError`
    rather than warning -- a leaky or misoriented boundary produces
    silently wrong flux accounting, exactly the failure class this
    validation exists to catch at construction time instead of at trace
    time.

    Attributes:
        name: Human-readable label.
        boundary: Closed, outward-oriented list of boundary surfaces.
        interior: The medium inside this volume.
    """

    name: str
    boundary: list[BaseComponent]
    interior: NSQMaterial
    _skip_validation: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._skip_validation:
            return
        if not self.boundary:
            raise NonWatertightVolumeError(
                f"Volume '{self.name}' has no boundary surfaces."
            )
        rim_points = _check_watertight(self.boundary)
        if rim_points is not None and len(rim_points) > 0:
            centroid = rim_points.mean(axis=0)
        else:
            # No surface exposed a finite rim (e.g. a single closed sphere):
            # fall back to the mean of each surface's own coordinate origin.
            centroid = np.mean([_get_transform(c.cs)[0] for c in self.boundary], axis=0)
        # This check calls component.intersect(), which dispatches through
        # optiland.backend (be.*). The check is purely discrete geometry --
        # never differentiated -- so it always runs on the numpy backend,
        # regardless of which backend is active for the surrounding scene
        # (e.g. a Lens built while be.set_backend("torch") is active for a
        # gradient trace).
        import optiland.backend as be  # noqa: PLC0415

        previous_backend = be.get_backend()
        try:
            be.set_backend("numpy")
            _check_normals_outward(self.boundary, centroid)
        finally:
            be.set_backend(previous_backend)

    @staticmethod
    def union(
        *parts: Volume | list[BaseComponent] | BaseComponent,
    ) -> list[BaseComponent]:
        """Concatenate already-disjoint boundary surfaces into one list.

        This is the CSG operation this revamp implements: gluing separately
        constructed, non-overlapping boundary pieces together (the stated
        use cases -- a lens with a flat, a light pipe with a chamfer) are
        boundary concatenation, not boolean surface evaluation. The result
        is not itself validated; pass it to :class:`Volume` to check it.

        Args:
            *parts: Any mix of ``Volume`` instances, lists of components, or
                single components.

        Returns:
            The concatenated boundary list, in argument order.
        """
        boundary: list[BaseComponent] = []
        for part in parts:
            if isinstance(part, Volume):
                boundary.extend(part.boundary)
            elif isinstance(part, list):
                boundary.extend(part)
            else:
                boundary.append(part)
        return boundary

    @staticmethod
    def intersection(*parts: object) -> None:
        """Not implemented: true CSG intersection needs a boolean surface
        evaluator.

        Raises:
            NotImplementedError: Always. Construct the intersected geometry
                directly with analytic primitives, or use :meth:`union` for
                concatenating already-disjoint boundary surfaces.
        """
        raise NotImplementedError(
            "Volume.intersection() requires a true boolean surface evaluator, "
            "which this revamp does not implement (D16, measured and specced "
            "only). Use Volume.union() for concatenating already-disjoint "
            "boundary surfaces, or construct the intersected geometry "
            "directly with analytic primitives."
        )

    @staticmethod
    def difference(*parts: object) -> None:
        """Not implemented: true CSG difference needs a boolean surface
        evaluator.

        Raises:
            NotImplementedError: Always. See :meth:`intersection`.
        """
        raise NotImplementedError(
            "Volume.difference() requires a true boolean surface evaluator, "
            "which this revamp does not implement (D16, measured and specced "
            "only). Use Volume.union() for concatenating already-disjoint "
            "boundary surfaces, or construct the differenced geometry "
            "directly with analytic primitives."
        )
