"""Folded-path paraxial metadata.

This module is the single source of truth for the scalar folded paraxial
model introduced by issues #726/#728: it walks the surface chain once and
records, per surface, the vertex and local axis in global coordinates, the
incoming/outgoing beam directions, the reflection parity, the signed unfolded
axial coordinate, and the orientation sign that powered surfaces need for
their paraxial-effective radius. Every consumer of folded first-order data
(``SurfaceGroup.positions``, the paraxial ray tracer, the ray aimers, the
pupil-point helpers) builds or receives one :class:`ParaxialPath` instead of
re-deriving frames, directions and parity independently.

Supported scalar domain: piecewise-centered systems whose changes of
propagation direction are produced by plane fold mirrors, entered along an
arbitrary finite unit direction. Powered surfaces must be normal to their
local propagation segment. Anything outside that domain (oblique powered
mirrors, tilted powered refractive surfaces, refractive interfaces that steer
the nominal axis, transversely decentered vertex chains, non-object surfaces
at infinity on a folded arm) is recorded as a diagnostic and rejected with
:class:`UnsupportedParaxialGeometryError` when scalar paraxial analysis or
ray aiming is requested -- real ray tracing remains available for physically
valid geometries even when the scalar paraxial model is rejected.

Kramer Harrison, 2026
"""

from __future__ import annotations

import contextlib
import contextvars
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import optiland.backend as be

if TYPE_CHECKING:
    from optiland._types import BEArray, ScalarOrArray

# A direction this close to an axis is treated as on that axis. This is the
# float64 intent; use angular_tolerance() for a dtype-aware value.
_AXIS_TOL = 1e-10

# Machine epsilons by floating precision, used to widen tolerances when the
# backend computes in float32.
_EPS_BY_PRECISION = {"float32": 1.1920929e-07, "float64": 2.220446049250313e-16}

# Diagnostic codes (stable identifiers; error messages reference them).
DEGENERATE_ENTRY_AXIS = "DEGENERATE_ENTRY_AXIS"
NONCOLLINEAR_VERTEX_CHAIN = "NONCOLLINEAR_VERTEX_CHAIN"
TILTED_REFRACTIVE_SURFACE = "TILTED_REFRACTIVE_SURFACE"
OBLIQUE_POWERED_MIRROR = "OBLIQUE_POWERED_MIRROR"
UNSUPPORTED_PARAXIAL_INTERACTION = "UNSUPPORTED_PARAXIAL_INTERACTION"
NONOBJECT_INFINITY = "NONOBJECT_INFINITY"
AMBIGUOUS_WIDE_ANGLE_FIELD = "AMBIGUOUS_WIDE_ANGLE_FIELD"
SINGULAR_ANGLE_TANGENT = "SINGULAR_ANGLE_TANGENT"


class UnsupportedParaxialGeometryError(ValueError):
    """A geometry outside the supported scalar folded paraxial domain.

    Raised instead of silently returning plausible-but-wrong first-order
    numbers. Real ray tracing (``optic.surfaces.trace``) remains available
    for the same geometry.
    """


class ParaxialDomainWarning(UserWarning):
    """Scalar paraxial values are approximate for this geometry.

    Emitted (instead of :class:`UnsupportedParaxialGeometryError`) when the
    out-of-domain scalar paraxial engine is used only to seed a real-ray
    iterative solve -- the final aimed rays are verified against real
    traces, so the approximation never surfaces as a first-order result.
    """


# While set, out-of-domain scalar paraxial use warns instead of raising.
# The ray aimers set this around their solves: their paraxial numbers are
# only seeds for a real-ray Newton polish, so an approximate value is useful
# there and the exactness guarantee comes from the real traces. Direct
# first-order analysis (``Paraxial.f2()`` and friends) stays strict.
_SEED_SCOPE = contextvars.ContextVar("optiland_paraxial_seed_scope", default=False)


@contextlib.contextmanager
def paraxial_seed_scope():
    """Scope in which scalar paraxial values serve only as real-ray seeds."""
    token = _SEED_SCOPE.set(True)
    try:
        yield
    finally:
        _SEED_SCOPE.reset(token)


def in_paraxial_seed_scope() -> bool:
    """Whether a paraxial seed scope is currently active."""
    return _SEED_SCOPE.get()


def angular_tolerance() -> float:
    """Angular collinearity tolerance, backend/dtype aware.

    Returns the float64 intent of ``1e-10`` widened when the active backend
    computes at lower precision.
    """
    try:
        precision = str(be.get_precision())
    except Exception:
        precision = "float64"
    eps = _EPS_BY_PRECISION["float32" if "32" in precision else "float64"]
    return max(_AXIS_TOL, 100.0 * eps)


def position_tolerance(characteristic_scale: float) -> float:
    """Spatial residual tolerance scaled to the system's size.

    Args:
        characteristic_scale: A length characterizing the system, e.g. the
            largest finite vertex coordinate magnitude.
    """
    return angular_tolerance() * max(1.0, abs(characteristic_scale))


def tangent_singularity_tolerance_deg() -> float:
    """Rejection half-width (degrees) around the odd multiples of 90 degrees.

    A component field angle closer than this to ``90 + k * 180`` degrees is
    rejected before its tangent is evaluated: floating-point ``tan`` returns
    a huge finite number there instead of failing, and every quantity built
    from it (launch points, object positions, chief-ray scales) silently
    loses its meaning.

    The width is derived from the active backend precision. The tangent's
    relative conditioning error at distance ``delta`` (radians) from the
    pole is approximately ``eps * (pi / 2) / delta``; requiring it to stay
    below ``sqrt(eps)`` -- i.e. the value keeps at least half its
    significant digits -- gives ``delta >= (pi / 2) * sqrt(eps)``, which is
    ``90 * sqrt(eps)`` in degrees. For float64 this is ~1.3e-6 degrees; for
    float32 ~0.031 degrees. Valid nonsingular one-dimensional wide fields
    (89, 91, 95, 105 degrees, ...) lie far outside it.
    """
    try:
        precision = str(be.get_precision())
    except Exception:
        precision = "float64"
    eps = _EPS_BY_PRECISION["float32" if "32" in precision else "float64"]
    return 90.0 * eps**0.5


def require_nonsingular_tangent_angles(
    *components_deg, operation: str = "angle-field evaluation"
) -> None:
    """Reject field-angle components whose tangent is numerically singular.

    Call before every ``tan(angle)`` evaluation on component field angles.
    Angles within :func:`tangent_singularity_tolerance_deg` of an odd
    multiple of 90 degrees raise; everything else passes through untouched.

    Args:
        *components_deg: Angle components in degrees (scalars or arrays).
        operation: Name of the calling operation, used in the message.

    Raises:
        UnsupportedParaxialGeometryError: With code
            ``SINGULAR_ANGLE_TANGENT`` for the first offending component.
    """
    tol = tangent_singularity_tolerance_deg()
    for component in components_deg:
        values = be.to_numpy(be.atleast_1d(be.array(component))).reshape(-1)
        distance = abs(abs(values) % 180.0 - 90.0)
        bad = distance <= tol
        if bad.any():
            i = int(bad.nonzero()[0][0])
            raise UnsupportedParaxialGeometryError(
                f"[{SINGULAR_ANGLE_TANGENT}] {operation}: the field angle "
                f"component {values[i]:.10g} deg lies within {tol:.3g} deg "
                "of an odd multiple of 90 degrees, where its tangent is "
                "numerically singular (floating-point tan returns a huge "
                "finite number instead of failing). Use a field angle away "
                "from the pole; nonsingular wide angles such as 89, 91, 95 "
                "or 105 degrees remain supported."
            )


@dataclass(frozen=True)
class ParaxialPathDiagnostic:
    """One reason a path lies outside the supported scalar domain.

    Attributes:
        code: Stable diagnostic code (one of the module-level constants).
        surface_index: Index of the offending surface or leg, or ``None``.
        measured: The measured value that tripped the check, or ``None``.
        tolerance: The tolerance it was compared against, or ``None``.
        message: Concise human-readable explanation.
    """

    code: str
    surface_index: int | None
    measured: float | None
    tolerance: float | None
    message: str

    def __str__(self) -> str:
        parts = [f"[{self.code}]"]
        if self.surface_index is not None:
            parts.append(f"surface {self.surface_index}:")
        parts.append(self.message)
        if self.measured is not None:
            parts.append(f"(measured {self.measured:.6g}")
            if self.tolerance is not None:
                parts.append(f"vs tolerance {self.tolerance:.6g})")
            else:
                parts.append(")")
        return " ".join(parts)


def _to_float(value: ScalarOrArray) -> float:
    """Plain Python float from a scalar or 0-d/1-element backend array."""
    return float(be.to_numpy(be.array(value)).reshape(-1)[0])


def _dot(a: tuple, b: tuple) -> ScalarOrArray:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: tuple) -> ScalarOrArray:
    return be.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _normalize(a: tuple) -> tuple:
    norm = _norm(a)
    return (a[0] / norm, a[1] / norm, a[2] / norm)


def _cross(a: tuple, b: tuple) -> tuple:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _is_finite_vec(a: tuple) -> bool:
    return all(bool(be.all(be.isfinite(be.array(_to_float(c))))) for c in a)


def transverse_basis(direction: tuple) -> tuple[tuple, tuple]:
    """Deterministic transverse pair ``(u, v)`` completing ``direction``.

    ``v`` is global +y projected off the axis and normalized -- the sagittal
    direction of a fold in the x-z plane. Within :func:`angular_tolerance` of
    a +/-y entry that projection degenerates (the unavoidable pole of any
    deterministic rule), and ``u`` becomes global +x projected instead.
    ``(u, v, direction)`` is right-handed, and for a +z entry the pair
    reduces exactly to the global +x/+y axes, so field semantics continue
    the on-axis meaning. The gauge is deliberately a global reference --
    never the object surface's orientation -- so a tilted or rolled object
    plane cannot roll the field axes.
    """
    d = direction
    y_proj = (
        be.array(0.0) - d[1] * d[0],
        be.array(1.0) - d[1] * d[1],
        be.array(0.0) - d[1] * d[2],
    )
    norm = _norm(y_proj)
    # The pole decision is dtype-aware: angular_tolerance() widens with the
    # active backend precision, so a float32 run switches to the +x gauge
    # while the projection still carries meaningful digits.
    if bool(be.all(norm > angular_tolerance())):
        v = tuple(c / norm for c in y_proj)
        u = _cross(v, d)
        return u, v
    x_proj = (
        be.array(1.0) - d[0] * d[0],
        be.array(0.0) - d[0] * d[1],
        be.array(0.0) - d[0] * d[2],
    )
    norm = _norm(x_proj)
    u = tuple(c / norm for c in x_proj)
    v = _cross(d, u)
    return u, v


@dataclass(frozen=True)
class ParaxialPath:
    """Immutable per-operation snapshot of the folded-path metadata.

    All vectors are in global coordinates (GCS), stored as tuples of backend
    scalars. ``axial_positions`` is the signed unfolded axial coordinate the
    scalar paraxial model is written in -- not a global Cartesian z (use
    ``vertices_gcs`` for real-space points).

    Attributes:
        axial_positions: Signed unfolded axial coordinate per surface.
        vertices_gcs: Surface vertex positions in GCS.
        local_z_axes_gcs: Surface local +z axes in GCS.
        incoming_directions_gcs: Physical beam direction arriving at each
            surface.
        outgoing_directions_gcs: Physical beam direction leaving each surface.
        parity_before: Reflection parity (+1/-1) before each surface.
        parity_after: Reflection parity after each surface.
        orientation_sign: Per-surface sign ``s_k = parity_before *
            sgn(z_axis . incoming_direction)`` that maps authored radii and
            focal lengths to paraxial-effective ones.
        entry_direction: Unit beam direction on the entry leg.
        entry_u: First transverse basis vector of the entry frame.
        entry_v: Second transverse basis vector of the entry frame.
        all_legs_parallel_global_z: Every finite propagation leg is parallel
            to global +/-z (true also for retro and -z-entered systems).
        positions_are_global_z: ``axial_positions`` coincides bit-for-bit
            with the global z of each vertex (entry along +z, every mirror
            normal on the z axis). This -- not
            ``all_legs_parallel_global_z`` -- is the capability check for
            operations that write axial offsets into ``cs.z``.
        legacy_aiming_compatible: The historical global-z aiming branch is
            exactly correct: positions are global z AND the entry line runs
            through global x = y = 0 along +z.
        is_folded_or_off_axis: Negation of ``positions_are_global_z``.
        diagnostics: Reasons this path lies outside the supported scalar
            domain (empty for supported paths).
        advisories: Findings on a *supported* straight path that scalar
            first-order results silently ignore (tilted powered surfaces,
            interior decenters). Unlike ``diagnostics`` these never gate an
            operation -- the historical numbers are still returned -- they
            only make the approximation visible via
            :meth:`warn_scalar_approximations`.
        axis_alignments: Per-surface alignment ``z_axis . incoming_direction``
            between the surface's local +z axis and the physical beam
            arriving at it. ``|alignment| == 1`` (within
            :func:`angular_tolerance`) identifies a centered/collinear
            surface; anything else is genuinely oblique.
    """

    axial_positions: BEArray
    vertices_gcs: tuple
    local_z_axes_gcs: tuple
    incoming_directions_gcs: tuple
    outgoing_directions_gcs: tuple
    parity_before: tuple
    parity_after: tuple
    orientation_sign: tuple
    entry_direction: tuple
    entry_u: tuple
    entry_v: tuple
    all_legs_parallel_global_z: bool
    positions_are_global_z: bool
    legacy_aiming_compatible: bool
    is_folded_or_off_axis: bool
    diagnostics: tuple
    advisories: tuple = ()
    axis_alignments: tuple = ()

    @property
    def num_surfaces(self) -> int:
        return len(self.vertices_gcs)

    @property
    def orientation_sign_array(self) -> BEArray:
        """Orientation signs as a backend array aligned with the surfaces."""
        return be.array([float(s) for s in self.orientation_sign])

    def effective_orientation_signs(self) -> tuple:
        """Per-surface signs mapping authored powers to scalar-effective ones.

        This is the single collinear orientation policy shared by the
        explicit paraxial tracer and the ray-transfer-matrix assembly
        (they must never disagree because they selected different power
        conventions):

        - A centered/collinear surface (``|1 - |z_axis . d_in|| <=``
          :func:`angular_tolerance`) gets ``s_k = parity_before *
          sgn(z_axis . d_in)`` -- on straight and folded paths alike, so a
          physically equivalent surface authored with its local axis
          reversed is normalized to the same effective scalar power. The
          canonical default authoring always has ``s_k = +1``, preserving
          historical values bit-for-bit.
        - A genuinely oblique surface on a straight-classified path keeps
          the historical raw value (sign ``+1``); the approximation is
          surfaced via :meth:`warn_scalar_approximations`, never silently
          re-signed by a heuristic.
        - A genuinely oblique surface on a folded/off-axis path is outside
          the scalar domain (``require_scalar_paraxial`` raises). Inside a
          :func:`paraxial_seed_scope` -- where that rejection is downgraded
          to a warning because the values only seed a real-ray solve -- the
          collinear-limit sign is kept, which is the continuous limit of
          the supported geometry as the tilt goes to zero.
        """
        if self.is_folded_or_off_axis or not self.axis_alignments:
            return self.orientation_sign
        tol = angular_tolerance()
        return tuple(
            sign if abs(abs(alignment) - 1.0) <= tol else 1.0
            for sign, alignment in zip(
                self.orientation_sign, self.axis_alignments, strict=True
            )
        )

    @property
    def entry_is_positive_z(self) -> bool:
        """Whether the entry direction is global +z within tolerance."""
        tol = angular_tolerance()
        ex = abs(_to_float(self.entry_direction[0]))
        ey = abs(_to_float(self.entry_direction[1]))
        ez = _to_float(self.entry_direction[2])
        return ex <= tol and ey <= tol and ez > 0.0

    def require_scalar_paraxial(self, operation: str = "scalar paraxial analysis"):
        """Reject this path if it lies outside the supported scalar domain.

        Raises for direct first-order analysis. Inside a
        :func:`paraxial_seed_scope` (the ray aimers' solves, where scalar
        paraxial values only seed a real-ray Newton polish) the same
        finding is reported as a :class:`ParaxialDomainWarning` instead, so
        aiming for e.g. slightly tilted stop mirrors keeps working -- its
        exactness is guaranteed by the real traces, not the seed.

        Args:
            operation: Name of the requested operation, used in the message.

        Raises:
            UnsupportedParaxialGeometryError: If any diagnostic was recorded
                and no seed scope is active.
        """
        if not self.diagnostics:
            return
        details = "\n".join(f"  - {d}" for d in self.diagnostics)
        message = (
            f"{operation} is not supported for this geometry: the system "
            f"lies outside the scalar folded paraxial domain "
            f"(piecewise-centered legs joined by plane fold mirrors, powered "
            f"surfaces normal to their local beam segment).\n{details}\n"
            "Real ray tracing (optic.surfaces.trace) remains available for "
            "this geometry; a general vector paraxial model would be "
            "required for first-order analysis."
        )
        if in_paraxial_seed_scope():
            warnings.warn(
                "scalar paraxial values are approximate for this geometry "
                "and are used only to seed a real-ray solve:\n" + message,
                ParaxialDomainWarning,
                stacklevel=2,
            )
            return
        raise UnsupportedParaxialGeometryError(message)

    def warn_scalar_approximations(
        self, operation: str = "scalar paraxial analysis"
    ) -> None:
        """Surface any advisories as a :class:`ParaxialDomainWarning`.

        Straight +z systems with tilted powered surfaces or interior
        decenters have always had their scalar first-order values computed
        as if every surface were centered and normal to the axis. That
        behavior (and every returned number) is unchanged; this method only
        makes the approximation visible instead of silent. Real ray tracing
        accounts for the tilt/decenter exactly.

        Args:
            operation: Name of the requested operation, used in the message.
        """
        if not self.advisories:
            return
        details = "\n".join(f"  - {d}" for d in self.advisories)
        warnings.warn(
            f"{operation}: scalar first-order results ignore surface "
            "tilts/decenters on this system (each surface is treated as "
            "centered and normal to the axis, the historical behavior):\n"
            f"{details}\n"
            "Real ray tracing (optic.surfaces.trace) accounts for them "
            "exactly.",
            ParaxialDomainWarning,
            stacklevel=2,
        )

    def point_from_axial_offset(
        self,
        surface_index: int,
        axial_offset: ScalarOrArray,
        side: Literal["incoming", "outgoing"] = "incoming",
    ) -> tuple:
        """Map an unfolded axial offset from a surface vertex to a GCS point.

        The point is ``r_k + parity * offset * direction`` with the parity
        and physical direction of the selected side of surface ``k``, which
        is how a signed axial distance (an ``EPL``/``XPL``-style scalar)
        becomes a real-space location on the correct leg.

        Args:
            surface_index: Surface the offset is measured from (negative
                indices allowed, as in normal sequence indexing).
            axial_offset: Signed unfolded axial offset from that surface.
            side: Whether to use the incoming or outgoing beam direction
                and parity at that surface.

        Returns:
            The GCS point as an ``(x, y, z)`` tuple of backend scalars.
        """
        vertex = self.vertices_gcs[surface_index]
        if side == "incoming":
            parity = self.parity_before[surface_index]
            direction = self.incoming_directions_gcs[surface_index]
        else:
            parity = self.parity_after[surface_index]
            direction = self.outgoing_directions_gcs[surface_index]
        return tuple(vertex[i] + parity * axial_offset * direction[i] for i in range(3))

    def entry_frame(self) -> tuple:
        """Entry frame ``(anchor, axial_anchor, direction, u, v)``.

        ``anchor`` is the first physical surface's vertex, ``axial_anchor``
        its unfolded axial coordinate, ``direction`` the unit entry
        direction, and ``(u, v)`` the transverse basis completing it (see
        :func:`transverse_basis` for the gauge convention and its pole).
        """
        anchor = self.vertices_gcs[1]
        axial = self.axial_positions[1]
        return anchor, axial, self.entry_direction, self.entry_u, self.entry_v


def _entry_direction(frames: list) -> tuple[tuple, bool]:
    """Unit vector of the first leg, object vertex to first surface vertex.

    Read off the two vertices rather than off the object surface's own
    orientation: a tilted object plane does not steer the beam, so its
    normal is not the axis. An object at infinity leaves an infinite
    component in whichever axes the beam runs along, which carries the
    direction on its own once the finite components are dropped.

    Returns:
        ``(direction, degenerate)`` where ``degenerate`` is True when the
        +z default was used because there was nothing to read -- a lone
        surface, or an object sitting on top of the first surface.
    """
    default = (be.array(0.0), be.array(0.0), be.array(1.0))
    if len(frames) < 2:
        return default, True
    first, second = frames[0][0], frames[1][0]
    step = [second[k] - first[k] for k in range(3)]
    diverging = [bool(be.any(be.isinf(be.array(s)))) for s in step]
    if any(diverging):
        step = [
            be.sign(s) if is_inf else be.array(0.0)
            for s, is_inf in zip(step, diverging, strict=True)
        ]
    norm = be.sqrt(sum(s * s for s in step))
    if not bool(be.all(norm > _AXIS_TOL)):
        return default, True
    return tuple(s / norm for s in step), False


def _off_axis(vector: tuple) -> bool:
    """Whether a unit vector points anywhere but along +/-z."""
    return bool(abs(vector[0]) > _AXIS_TOL) or bool(abs(vector[1]) > _AXIS_TOL)


def _surface_is_powered(surface) -> bool:
    """Whether a surface bends paraxial rays (finite radius or explicit f)."""
    if getattr(surface, "surface_type", None) == "paraxial":
        return True
    radius = getattr(getattr(surface, "geometry", None), "radius", None)
    if radius is None:
        return False
    return bool(be.all(be.isfinite(be.array(_to_float(radius)))))


def _surface_is_refractive_boundary(surface) -> bool:
    """Whether a surface separates two different materials (and refracts)."""
    pre = getattr(surface, "material_pre", None)
    post = getattr(surface, "material_post", None)
    if pre is None or post is None:
        return False
    try:
        return pre != post
    except Exception:
        return True


def build_paraxial_path(surfaces: list) -> ParaxialPath:
    """Walk the surface chain once and assemble the shared path metadata.

    Args:
        surfaces: Ordered sequence of Surface objects (object surface first).

    Returns:
        The assembled :class:`ParaxialPath`. Unsupported-geometry findings
        are recorded as diagnostics on the path (this function never raises
        for them); call ``path.require_scalar_paraxial()`` at the boundary
        of any scalar first-order computation.
    """
    frames = [surf.geometry.cs.frame_in_gcs for surf in surfaces]
    mirrors = [
        bool(getattr(surf.interaction_model, "is_reflective", False))
        for surf in surfaces
    ]
    n = len(frames)
    vertices = tuple(frames[k][0] for k in range(n))
    axes = tuple(frames[k][1] for k in range(n))

    entry, degenerate_entry = _entry_direction(frames)

    # A finite object authored past the first surface (a virtual object on a
    # +z chain) reads as a -z first leg off the vertices, but the beam still
    # travels +z. Only that exact historical case is overridden; a chain
    # whose physical legs also descend in z is a genuinely -z-entered system.
    object_is_infinite = (
        bool(getattr(surfaces[0], "is_infinite", False)) if n else False
    )
    if (
        n >= 3
        and not object_is_infinite
        and not degenerate_entry
        and not _off_axis(entry)
        and _to_float(entry[2]) < 0.0
    ):
        first_leg = [vertices[2][k] - vertices[1][k] for k in range(3)]
        leg_norm = be.sqrt(sum(s * s for s in first_leg))
        if bool(be.all(be.isfinite(leg_norm))) and _to_float(leg_norm) > _AXIS_TOL:
            leg_dir = tuple(s / leg_norm for s in first_leg)
            if not _off_axis(leg_dir) and _to_float(leg_dir[2]) > 0.0:
                entry = (be.array(0.0), be.array(0.0), be.array(1.0))

    any_mirror_off_axis = any(
        is_mirror and _off_axis(normal)
        for (_, normal), is_mirror in zip(frames, mirrors, strict=True)
    )
    entry_off_axis = _off_axis(entry)
    entry_negative_z = (not entry_off_axis) and n >= 2 and _to_float(entry[2]) < 0.0

    positions_are_global_z = (
        not entry_off_axis and not entry_negative_z and not any_mirror_off_axis
    )
    is_folded_or_off_axis = not positions_are_global_z

    ang_tol = angular_tolerance()
    diagnostics: list[ParaxialPathDiagnostic] = []
    advisories: list[ParaxialPathDiagnostic] = []

    if is_folded_or_off_axis and degenerate_entry and any_mirror_off_axis:
        diagnostics.append(
            ParaxialPathDiagnostic(
                code=DEGENERATE_ENTRY_AXIS,
                surface_index=0,
                measured=None,
                tolerance=_AXIS_TOL,
                message=(
                    "the entry direction cannot be inferred (object vertex "
                    "coincides with the first surface vertex) and the system "
                    "is folded, so the +z fallback is not unambiguously valid"
                ),
            )
        )

    # Characteristic scale for spatial tolerances: largest finite vertex
    # coordinate magnitude.
    scale = 1.0
    for vertex in vertices:
        for component in vertex:
            value = _to_float(component)
            if abs(value) != float("inf") and abs(value) > scale:
                scale = abs(value)
    pos_tol = position_tolerance(scale)

    # --- The walk -------------------------------------------------------
    direction = entry
    parity = 1.0
    incoming: list[tuple] = []
    outgoing: list[tuple] = []
    parity_before: list[float] = []
    parity_after: list[float] = []
    orientation: list[float] = []
    alignments: list[float] = []
    axial: list = []

    if n:
        first = vertices[0]
        axial.append(
            first[2]
            if _is_finite_vec(first)
            else sum(first[k] * direction[k] for k in range(3))
        )

    for k in range(n):
        incoming.append(direction)
        parity_before.append(parity)

        axis_dot = _to_float(_dot(axes[k], direction))
        sign = 1.0 if axis_dot >= 0.0 else -1.0
        orientation.append(parity * sign)
        alignments.append(axis_dot)

        surf = surfaces[k]
        powered = _surface_is_powered(surf) if k > 0 else False
        oblique = abs(abs(axis_dot) - 1.0) > ang_tol

        if is_folded_or_off_axis and k > 0 and oblique:
            if powered and mirrors[k]:
                diagnostics.append(
                    ParaxialPathDiagnostic(
                        code=OBLIQUE_POWERED_MIRROR,
                        surface_index=k,
                        measured=axis_dot,
                        tolerance=ang_tol,
                        message=(
                            "powered mirror at oblique incidence is "
                            "astigmatic; |axis . beam| must equal 1"
                        ),
                    )
                )
            elif powered:
                diagnostics.append(
                    ParaxialPathDiagnostic(
                        code=TILTED_REFRACTIVE_SURFACE,
                        surface_index=k,
                        measured=axis_dot,
                        tolerance=ang_tol,
                        message=(
                            "tilted powered refractive surface introduces "
                            "transverse first-order coupling; |axis . beam| "
                            "must equal 1"
                        ),
                    )
                )
            elif not mirrors[k] and _surface_is_refractive_boundary(surf):
                diagnostics.append(
                    ParaxialPathDiagnostic(
                        code=TILTED_REFRACTIVE_SURFACE,
                        surface_index=k,
                        measured=axis_dot,
                        tolerance=ang_tol,
                        message=(
                            "tilted plane refractive interface steers the "
                            "nominal axis (Snell refraction), which the "
                            "scalar folded path does not model"
                        ),
                    )
                )
        elif k > 0 and oblique:
            # Straight-classified path: the numbers stay the historical
            # ones (tilt ignored); record an advisory so the approximation
            # is no longer silent.
            if powered:
                advisories.append(
                    ParaxialPathDiagnostic(
                        code=TILTED_REFRACTIVE_SURFACE,
                        surface_index=k,
                        measured=axis_dot,
                        tolerance=ang_tol,
                        message=(
                            "tilted powered surface on a straight system; "
                            "scalar first-order results treat it as normal "
                            "to the axis"
                        ),
                    )
                )
            elif not mirrors[k] and _surface_is_refractive_boundary(surf):
                advisories.append(
                    ParaxialPathDiagnostic(
                        code=TILTED_REFRACTIVE_SURFACE,
                        surface_index=k,
                        measured=axis_dot,
                        tolerance=ang_tol,
                        message=(
                            "tilted plane refractive interface on a straight "
                            "system; the real beam is steered (Snell), which "
                            "scalar first-order results ignore"
                        ),
                    )
                )

        if mirrors[k]:
            projection = _dot(direction, axes[k])
            direction = tuple(
                direction[i] - 2 * projection * axes[k][i] for i in range(3)
            )
            direction = _normalize(direction)
            parity = -parity

        outgoing.append(direction)
        parity_after.append(parity)

        if k + 1 < n:
            nxt = vertices[k + 1]
            prev = vertices[k]
            step = sum((nxt[i] - prev[i]) * direction[i] for i in range(3))
            if be.all(be.isfinite(axial[-1])) and be.all(be.isfinite(step)):
                axial.append(axial[-1] + parity * step)
                residual = tuple(
                    (nxt[i] - prev[i]) - step * direction[i] for i in range(3)
                )
                residual_norm = _to_float(_norm(residual))
                if residual_norm > pos_tol:
                    if is_folded_or_off_axis:
                        diagnostics.append(
                            ParaxialPathDiagnostic(
                                code=NONCOLLINEAR_VERTEX_CHAIN,
                                surface_index=k + 1,
                                measured=residual_norm,
                                tolerance=pos_tol,
                                message=(
                                    "vertex lies off the physical beam leg "
                                    "(transverse decenter); the scalar folded "
                                    "model has no transverse first-order "
                                    "coupling to carry it"
                                ),
                            )
                        )
                    else:
                        advisories.append(
                            ParaxialPathDiagnostic(
                                code=NONCOLLINEAR_VERTEX_CHAIN,
                                surface_index=k + 1,
                                measured=residual_norm,
                                tolerance=pos_tol,
                                message=(
                                    "decentered surface on a straight "
                                    "system; scalar first-order results "
                                    "treat it as centered on the axis"
                                ),
                            )
                        )
            else:
                # A leg to or from infinity carries no fold, so re-anchor on
                # global z instead of accumulating an inf that would cancel
                # into a nan.
                axial.append(nxt[2])
                if is_folded_or_off_axis and not _is_finite_vec(nxt):
                    diagnostics.append(
                        ParaxialPathDiagnostic(
                            code=NONOBJECT_INFINITY,
                            surface_index=k + 1,
                            measured=None,
                            tolerance=None,
                            message=(
                                "non-object surface at infinity on a folded "
                                "or off-axis path; the unfolded axial "
                                "coordinate cannot anchor it"
                            ),
                        )
                    )

    # Legacy read: while every leg runs along +/-z entered along +z, the
    # axial coordinate is exactly the global z of each vertex -- returned
    # bit-for-bit unchanged (the walk above would agree to round-off, but
    # the canonical path must be exact).
    if positions_are_global_z:
        axial_positions = be.array([origin[2] for origin, _ in frames])
    else:
        axial_positions = be.array(axial) if axial else be.array([])

    entry_u, entry_v = transverse_basis(entry)

    all_parallel = not entry_off_axis and not any_mirror_off_axis

    anchor_on_origin = False
    if n >= 2:
        ax = abs(_to_float(vertices[1][0]))
        ay = abs(_to_float(vertices[1][1]))
        anchor_on_origin = ax <= pos_tol and ay <= pos_tol
    legacy_aiming_compatible = positions_are_global_z and (n < 2 or anchor_on_origin)

    return ParaxialPath(
        axial_positions=axial_positions,
        vertices_gcs=vertices,
        local_z_axes_gcs=axes,
        incoming_directions_gcs=tuple(incoming),
        outgoing_directions_gcs=tuple(outgoing),
        parity_before=tuple(parity_before),
        parity_after=tuple(parity_after),
        orientation_sign=tuple(orientation),
        entry_direction=entry,
        entry_u=entry_u,
        entry_v=entry_v,
        all_legs_parallel_global_z=all_parallel,
        positions_are_global_z=positions_are_global_z,
        legacy_aiming_compatible=legacy_aiming_compatible,
        is_folded_or_off_axis=is_folded_or_off_axis,
        diagnostics=tuple(diagnostics),
        advisories=tuple(advisories),
        axis_alignments=tuple(alignments),
    )


def require_global_z_geometry(surfaces, operation: str) -> None:
    """Guard for operations that write axial offsets into global ``cs.z``.

    Args:
        surfaces: The ``SurfaceGroup`` (or surface sequence) to check.
        operation: Human-readable name of the gated operation.

    Raises:
        UnsupportedParaxialGeometryError: If the unfolded axial coordinate
            does not coincide with global z, so a scalar z-offset mutation
            would move surfaces off their physical legs.
    """
    surface_list = list(surfaces)
    path = build_paraxial_path(surface_list)
    if path.positions_are_global_z:
        return
    raise UnsupportedParaxialGeometryError(
        f"{operation} mutates surface positions along global z only, but "
        "this system's beam path is folded off the +z axis (or entered "
        "along a different direction), so its unfolded axial coordinate is "
        "not global z. Applying the offset would move surfaces off their "
        "physical legs. A future implementation must translate downstream "
        "geometry by delta_r = parity * delta_q * leg_direction instead. "
        "No geometry was modified."
    )
