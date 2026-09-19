"""Adapter registries and the closed refusal-reason set for the fused trace.

WP0 ships the skeleton: the ``FusedTraceSkip`` enum (the closed set of counted
refusal reasons), the structural/feature split, the three adapter dataclasses
and three empty registries keyed by *exact* type.  WP2 fills the registries.

Keying by exact type (``type(obj) is cls``, never ``isinstance``) is
deliberate: a subclass of ``StandardGeometry`` that overrides ``sag`` would be
silently mis-mirrored by an ``isinstance`` lookup, so an unknown subclass must
fall through to a counted refusal instead.

Nothing here imports torch: the module is importable from the NumPy path.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = [
    "APERTURE_ADAPTERS",
    "ApertureAdapter",
    "FEATURE_REASONS",
    "FusedTraceSkip",
    "GEOMETRY_ADAPTERS",
    "GeometryAdapter",
    "INTERACTION_ADAPTERS",
    "InteractionAdapter",
    "STRUCTURAL_REASONS",
    "register",
]


class FusedTraceSkip(str, enum.Enum):  # noqa: UP042 - signature frozen by plan 3.3
    """Closed set of reasons the fused trace declines a bundle.

    Every value is also a counter suffix: ``fused_trace_skip:<value>``.

    Structural reasons describe a bundle that could never be a kernel
    candidate; they are counted and never raise, even under
    ``OPTILAND_METAL_FUSED_TRACE=require``.  Feature reasons describe a
    candidate bundle the kernel does not support yet; under ``require`` they
    raise ``MetalFallbackError`` (plan 1.3).
    """

    # -- structural (never raise under ``require``) -------------------------
    GROUP_TYPE = "group_type"
    RAYS_TYPE = "rays_type"
    RAYS_SHAPE = "rays_shape"
    HOST_RESIDENT = "host_resident"
    REQUIRES_GRAD = "requires_grad"
    SKIP = "skip"

    # -- feature (raise under ``require``) ---------------------------------
    SURFACE_TYPE = "surface_type"
    SURFACE0_NOT_OBJECT = "surface0_not_object"
    TOO_MANY_RAYS = "too_many_rays"
    MIN_RAYS = "min_rays"
    GEOMETRY_TYPE = "geometry_type"
    REFERENCE_CS = "reference_cs"
    POSE_NONFINITE = "pose_nonfinite"
    INTERACTION_TYPE = "interaction_type"
    COATING = "coating"
    BSDF = "bsdf"
    POLARIZATION = "polarization"
    PROPAGATION_MODEL = "propagation_model"
    APERTURE_TYPE = "aperture_type"
    APERTURE_PARAMS = "aperture_params"
    NEWTON_PARAMS = "newton_params"
    NONFINITE_INDEX = "nonfinite_index"
    MEMORY = "memory"
    MIXED_WAVELENGTH = "mixed_wavelength"
    MIRROR_DRIFT = "mirror_drift"
    UNAVAILABLE = "unavailable"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: The six structural reasons (plan 1.3, 3.3).  Exactly these never raise.
STRUCTURAL_REASONS: frozenset[FusedTraceSkip] = frozenset(
    {
        FusedTraceSkip.GROUP_TYPE,
        FusedTraceSkip.RAYS_TYPE,
        FusedTraceSkip.RAYS_SHAPE,
        FusedTraceSkip.HOST_RESIDENT,
        FusedTraceSkip.REQUIRES_GRAD,
        FusedTraceSkip.SKIP,
    }
)

#: Every other reason: a candidate bundle refused for a feature the kernel
#: does not implement yet.  ``require`` raises on these.
FEATURE_REASONS: frozenset[FusedTraceSkip] = (
    frozenset(FusedTraceSkip) - STRUCTURAL_REASONS
)


# ``fill`` signature: (obj, row_int, row_real, row_coef, ctx) -> None
FillFn = Callable[[Any, "np.ndarray", "np.ndarray", "np.ndarray", dict], None]
CheckFn = Callable[[Any], "FusedTraceSkip | None"]
GradParamsFn = Callable[[Any], list]


@dataclass(frozen=True)
class GeometryAdapter:
    """Maps one exact geometry class onto a ``surf_int``/``surf_real`` row.

    Attributes:
        code: the ``GEOM_*`` code written to ``SI_GEOM``.
        cls: the exact class this adapter handles.
        check: returns ``None`` when the instance is supported, otherwise the
            ``FusedTraceSkip`` reason.
        fill: writes the integer, real and coefficient slots for the instance.
            Accepts tensor entries in coefficient lists (plan 3.9).
        grad_params: tensors to test for ``requires_grad`` (including tensor
            coefficients).
        fixtures: zero-argument builders used by the conformance meta-test; an
            adapter with no fixtures fails that test (design 8.5).
    """

    code: int
    cls: type
    check: CheckFn
    fill: FillFn
    grad_params: GradParamsFn
    fixtures: tuple[Callable[[], Any], ...] = ()


@dataclass(frozen=True)
class ApertureAdapter:
    """Maps one exact aperture class onto ``SI_APCODE`` + ``SR_AP0..SR_AP3``."""

    code: int
    cls: type
    check: CheckFn
    fill: FillFn
    grad_params: GradParamsFn
    fixtures: tuple[Callable[[], Any], ...] = ()


@dataclass(frozen=True)
class InteractionAdapter:
    """Maps one exact interaction-model class onto the per-surface flags."""

    code: int
    cls: type
    check: CheckFn
    fill: FillFn
    grad_params: GradParamsFn
    fixtures: tuple[Callable[[], Any], ...] = ()


#: Exact-type registries.  Empty at WP0; WP2 fills them.
GEOMETRY_ADAPTERS: dict[type, GeometryAdapter] = {}
APERTURE_ADAPTERS: dict[type, ApertureAdapter] = {}
INTERACTION_ADAPTERS: dict[type, InteractionAdapter] = {}

_REGISTRIES: dict[type, dict] = {
    GeometryAdapter: GEOMETRY_ADAPTERS,
    ApertureAdapter: APERTURE_ADAPTERS,
    InteractionAdapter: INTERACTION_ADAPTERS,
}


def register(adapter: GeometryAdapter | ApertureAdapter | InteractionAdapter) -> None:
    """Add ``adapter`` to the registry for its kind, keyed by its exact class."""
    try:
        registry = _REGISTRIES[type(adapter)]
    except KeyError:  # pragma: no cover - programming error
        raise TypeError(f"not an adapter: {adapter!r}") from None
    if adapter.cls in registry:
        raise ValueError(f"{adapter.cls.__name__} is already registered")
    registry[adapter.cls] = adapter


# ---------------------------------------------------------------------------
# WP2: the v1 adapters
# ---------------------------------------------------------------------------
# Imported here rather than at the top so the module keeps its "no torch at
# module scope" property for the NumPy path: ``optiland.backend`` decides the
# backend, and these classes are plain Python.
import numpy as np  # noqa: E402

import optiland.backend as be  # noqa: E402
from optiland.geometries.even_asphere import EvenAsphere  # noqa: E402
from optiland.geometries.odd_asphere import OddAsphere  # noqa: E402
from optiland.geometries.plane import Plane  # noqa: E402
from optiland.geometries.standard import (  # noqa: E402
    StandardGeometry,
    _is_radius_infinite,
)
from optiland.interactions.refractive_reflective_model import (  # noqa: E402
    RefractiveReflectiveModel,
)
from optiland.physical_apertures.elliptical import EllipticalAperture  # noqa: E402
from optiland.physical_apertures.offset_radial import (  # noqa: E402
    OffsetRadialAperture,
)
from optiland.physical_apertures.radial import RadialAperture  # noqa: E402
from optiland.physical_apertures.rectangular import (  # noqa: E402
    RectangularAperture,
)

from . import trace_layout as L  # noqa: E402

__all__ += ["attach_fixtures", "is_backend_scalar"]


def _is_scalar_like(value: Any) -> bool:
    """True for a plain real number or a 0-d backend array."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    ndim = getattr(value, "ndim", None)
    if ndim is None:
        return False
    if ndim == 0:
        return True
    # A 1-element array is scalar-like only when it is genuinely 1-element.
    shape = tuple(getattr(value, "shape", ()))
    return shape == (1,)


def host_plain(value: Any) -> Any:
    """The plain CPU float64 tensor behind a host-resident emulated scalar.

    A ``MetalFloat64`` that lives on the host runs every op through two
    dispatch layers (about 25 us each) and rebuilds a wrapper for the result;
    the value of ``be.cos(x)`` on it is ``torch.cos(x._host)`` bit for bit,
    because that is the op the host path executes. The record compiler
    therefore evaluates its scalar slots on the plain tensor. Anything else
    (Python numbers, NumPy scalars, GPU-resident emulated tensors, plain
    tensors) is returned unchanged.
    """
    if type(value).__name__ != "MetalFloat64":
        return value
    host = getattr(value, "_host", None)
    return value if host is None else host


def _as_float(value: Any) -> float:
    """``float(value)`` for a scalar, a 0-d array or a 1-element array.

    The material properties come back shaped like the 1-element wavelength that
    was handed to them, which NumPy 2 refuses to convert with ``float()``.
    """
    value = host_plain(value)
    ndim = getattr(value, "ndim", 0)
    if ndim:
        value = value.reshape(-1)[0]
    return float(value)


def is_backend_scalar(value: Any) -> bool:
    """True for a BACKEND array scalar, False for a Python or NumPy scalar.

    The two forms are not interchangeable in df64: the per-op path launches a
    different KERNEL VARIANT for each, and df64's ``mul`` is not commutative
    (it adds ``a.hi*b.lo`` before ``a.lo*b.hi``), so the low word moves.  Two
    places in this package depend on the distinction:

    * ``EvenAsphere.sag``'s ``Ci * r2 ** (i + 1)`` is a host-scalar op for a
      Python/NumPy ``Ci`` and an array-array op for a backend tensor.  The
      record can only carry ``float(Ci)``, so the kernel always mirrors the
      host-scalar form and ``trace_record._check_surface`` refuses the other
      one in df64 (finding R1-V1-05).
    * ``(1 + self.k) * r2`` and ``self.radius * (1 + sqrt(..))`` swap operand
      SIDES with the form of ``geometry.k`` / ``geometry.radius``.  Those two
      the kernel mirrors instead of refusing, through ``FL_K1_ON_RIGHT`` and
      ``FL_R_ON_RIGHT`` (finding R2-V1-03), because ``OpticUpdater.set_conic``
      assigns the raw value and so every conic ``Variable.update`` produces the
      Python-number form.
    """
    if isinstance(value, (bool, int, float, np.floating, np.integer)):
        return False
    return getattr(value, "ndim", None) is not None


def _tensors(*values: Any) -> list:
    """The subset of ``values`` that carry ``requires_grad``."""
    return [v for v in values if hasattr(v, "requires_grad")]


def _material_grad_params(material: Any) -> list:
    """Every tensor attribute of a material (index/extinction parameters)."""
    if material is None:
        return []
    return [v for v in vars(material).values() if hasattr(v, "requires_grad")]


# -- geometries -------------------------------------------------------------
def _check_plane(geometry: Any) -> FusedTraceSkip | None:
    return None


def _fill_plane(obj, row_int, row_real, row_coef, ctx) -> None:
    """``Plane``: ``t = -z / N`` with no guard, normal ``(0, 0, +1)``."""
    row_int[L.SI_GEOM] = L.GEOM_PLANE
    row_int[L.SI_NCOEFF] = 0
    row_int[L.SI_MAXITER] = 0
    row_int[L.SI_FLAGS] |= L.FL_RADIUS_INF
    row_real[L.SR_R] = _as_float(obj.radius)
    row_real[L.SR_K] = 0.0
    row_real[L.SR_K1] = 1.0
    row_real[L.SR_R2] = _as_float(obj.radius) ** 2
    row_real[L.SR_TOL] = 0.0


def _check_conic(geometry: Any) -> FusedTraceSkip | None:
    if not _is_scalar_like(geometry.radius) or not _is_scalar_like(geometry.k):
        return FusedTraceSkip.GEOMETRY_TYPE
    return None


def _fill_conic(obj, row_int, row_real, row_coef, ctx) -> None:
    """``StandardGeometry``: code 2 or 3 by the ``_is_radius_infinite`` test."""
    infinite = _is_radius_infinite(obj.radius)
    row_int[L.SI_GEOM] = L.GEOM_STD_INF if infinite else L.GEOM_CONIC
    row_int[L.SI_NCOEFF] = 0
    row_int[L.SI_MAXITER] = 0
    if infinite:
        row_int[L.SI_FLAGS] |= L.FL_RADIUS_INF
    _fill_conic_scalars(obj, row_int, row_real)
    row_real[L.SR_TOL] = 0.0


def _fill_conic_scalars(obj, row_int, row_real) -> None:
    """Slots 18-21, each through the host-scalar op the per-op path runs.

    ``SI_FLAGS`` also records which SIDE of the two conic products the per-op
    path puts the slot on (finding R2-V1-03): a backend-array ``k``/``radius``
    keeps it on the left, a Python/NumPy number swaps it to the right because
    the reflected dunder launches the mirrored kernel variant.
    """
    radius = host_plain(obj.radius)
    k = host_plain(obj.k)
    row_real[L.SR_R] = _as_float(radius)
    row_real[L.SR_K] = _as_float(k)
    row_real[L.SR_K1] = _as_float(1 + k)
    row_real[L.SR_R2] = _as_float(radius**2)
    if not is_backend_scalar(obj.k):
        row_int[L.SI_FLAGS] |= L.FL_K1_ON_RIGHT
    if not is_backend_scalar(obj.radius):
        row_int[L.SI_FLAGS] |= L.FL_R_ON_RIGHT


def _grad_conic(obj) -> list:
    return _tensors(obj.radius, obj.k)


def _check_asphere(geometry: Any) -> FusedTraceSkip | None:
    reason = _check_conic(geometry)
    if reason is not None:
        return reason
    for coefficient in geometry.coefficients:
        if not _is_scalar_like(coefficient):
            return FusedTraceSkip.GEOMETRY_TYPE
    return None


def _fill_asphere(code: int):
    def fill(obj, row_int, row_real, row_coef, ctx) -> None:
        """``EvenAsphere`` / ``OddAsphere``: base conic plus the coefficients."""
        row_int[L.SI_GEOM] = code
        row_int[L.SI_NCOEFF] = len(obj.coefficients)
        row_int[L.SI_MAXITER] = int(obj.max_iter)
        if _is_radius_infinite(obj.radius):
            row_int[L.SI_FLAGS] |= L.FL_RADIUS_INF
        _fill_conic_scalars(obj, row_int, row_real)
        row_real[L.SR_TOL] = _as_float(obj.tol)
        for i, coefficient in enumerate(obj.coefficients):
            # After ``Variable.update`` a coefficient is a 0-d tensor
            # (``optic_updater.py`` stores the raw value); both forms read the
            # same way (plan 3.9).
            row_coef[i] = _as_float(coefficient)

    return fill


def _grad_asphere(obj) -> list:
    return _tensors(obj.radius, obj.k, *obj.coefficients)


# -- apertures --------------------------------------------------------------
def _check_aperture(*names: str):
    def check(aperture: Any) -> FusedTraceSkip | None:
        for name in names:
            if not _is_scalar_like(getattr(aperture, name)):
                return FusedTraceSkip.APERTURE_PARAMS
        return None

    return check


def _fill_radial(obj, row_int, row_real, row_coef, ctx) -> None:
    """``radius2 <= r_max**2`` and ``radius2 >= r_min**2`` (radial.py:69)."""
    row_int[L.SI_APCODE] = L.AP_RADIAL
    row_real[L.SR_AP0] = _as_float(obj.r_max**2)
    row_real[L.SR_AP1] = _as_float(obj.r_min**2)


def _fill_offset_radial(obj, row_int, row_real, row_coef, ctx) -> None:
    """The radial test about ``(offset_x, offset_y)`` (offset_radial.py:60)."""
    row_int[L.SI_APCODE] = L.AP_OFFSET_RADIAL
    row_real[L.SR_AP0] = _as_float(obj.r_max**2)
    row_real[L.SR_AP1] = _as_float(obj.r_min**2)
    row_real[L.SR_AP2] = _as_float(obj.offset_x)
    row_real[L.SR_AP3] = _as_float(obj.offset_y)


def _fill_rect(obj, row_int, row_real, row_coef, ctx) -> None:
    """``x_min <= x <= x_max`` and ``y_min <= y <= y_max`` (rectangular.py:54)."""
    row_int[L.SI_APCODE] = L.AP_RECT
    row_real[L.SR_AP0] = _as_float(obj.x_min)
    row_real[L.SR_AP1] = _as_float(obj.x_max)
    row_real[L.SR_AP2] = _as_float(obj.y_min)
    row_real[L.SR_AP3] = _as_float(obj.y_max)


def _fill_ellipse(obj, row_int, row_real, row_coef, ctx) -> None:
    """``x**2 / a**2 + y**2 / b**2 <= 1`` about the offset (elliptical.py:61)."""
    row_int[L.SI_APCODE] = L.AP_ELLIPSE
    row_real[L.SR_AP0] = _as_float(obj.a**2)
    row_real[L.SR_AP1] = _as_float(obj.b**2)
    row_real[L.SR_AP2] = _as_float(obj.offset_x)
    row_real[L.SR_AP3] = _as_float(obj.offset_y)


def _grad_aperture(*names: str):
    def grad_params(aperture: Any) -> list:
        return _tensors(*(getattr(aperture, name) for name in names))

    return grad_params


# -- interaction ------------------------------------------------------------
def _check_interaction(model: Any) -> FusedTraceSkip | None:
    if model.coating is not None:
        return FusedTraceSkip.COATING
    if model.bsdf is not None:
        return FusedTraceSkip.BSDF
    return None


def _fill_interaction(obj, row_int, row_real, row_coef, ctx) -> None:
    """Slots 22-26 and the reflective / absorbing flags.

    Every material scalar is evaluated at ``w0`` through a 1-element
    ``be.array``, which is exactly what the per-op path evaluates for a uniform
    wavelength above ``_MAX_VALUE_KEY_ARRAY_SIZE`` (``_uniform_representative``,
    ``materials/base.py``).  ``u`` and ``u**2`` mirror ``real_rays.refract``'s
    host scalar ops; ``alpha`` mirrors ``homogeneous.propagate`` up to the
    ``/ rays.w`` division, which stays a per-ray GPU op in the kernel.
    """
    if obj.is_reflective:
        row_int[L.SI_FLAGS] |= L.FL_REFLECTIVE

    n_pre, k_pre = _material_scalars(obj.material_pre, ctx)
    n_post, _ = _material_scalars(obj.material_post, ctx)

    row_real[L.SR_NPRE] = _as_float(n_pre)
    row_real[L.SR_NPOST] = _as_float(n_post)
    u = n_pre / n_post
    row_real[L.SR_U] = _as_float(u)
    row_real[L.SR_U2] = _as_float(u**2)
    alpha = (4 * be.pi) * k_pre
    row_real[L.SR_ALPHA] = _as_float(alpha)
    if _as_float(k_pre) > 0:
        row_int[L.SI_FLAGS] |= L.FL_ABSORBING


def _material_scalars(material: Any, ctx: dict) -> tuple[Any, Any]:
    """``(n, k)`` of ``material`` at ``ctx["w0"]`` as plain 1-element tensors.

    Evaluated exactly as the per-op path does for a uniform wavelength
    (``material.n`` on a 1-element ``be.array``, ``_uniform_representative``),
    then memoized in ``ctx["materials"]`` under the material's identity for
    the lifetime of the compile context (one ``compile_records`` call, or one
    ``trace_batch`` call). Optiland's own APIs change a surface's material by
    replacing the object (``OpticUpdater.set_index`` / ``set_material``, which
    every ``Variable`` and tolerancing perturbation goes through), so a new
    design that changes a glass never hits a stale entry; the batch API's
    tier-1 canary recompiles designs through a fresh context and would expose
    one if it ever did. A batch therefore evaluates each glass once instead of
    twice per row per design, and skips the material cache's own per-call
    state fingerprint. The plain host tensors keep the arithmetic in
    ``_fill_interaction`` bit-identical to the emulated host path while
    skipping its dispatch cost.
    """
    # ``_fill_row`` hands adapters a shallow copy of the context, so the memo
    # dict must be created by the context's owner (``compile_records``,
    # ``_compile_designs``) to persist across rows; a missing one is local.
    memo = ctx.get("materials")
    if memo is None:
        memo = {}
    key = (id(material), ctx["w0"])
    hit = memo.get(key)
    if hit is not None and hit[0] is material:
        return hit[1], hit[2]
    wavelength = memo.get(("wavelength", ctx["w0"]))
    if wavelength is None:
        wavelength = memo[("wavelength", ctx["w0"])] = be.array([ctx["w0"]])
    n = host_plain(material.n(wavelength))
    k = host_plain(material.k(wavelength))
    memo[key] = (material, n, k)
    return n, k


def _grad_interaction(model: Any) -> list:
    return _material_grad_params(model.material_pre) + _material_grad_params(
        model.material_post
    )


# -- registration -----------------------------------------------------------
register(
    GeometryAdapter(
        code=L.GEOM_PLANE,
        cls=Plane,
        check=_check_plane,
        fill=_fill_plane,
        grad_params=lambda obj: [],
    )
)
register(
    GeometryAdapter(
        code=L.GEOM_CONIC,
        cls=StandardGeometry,
        check=_check_conic,
        fill=_fill_conic,
        grad_params=_grad_conic,
    )
)
register(
    GeometryAdapter(
        code=L.GEOM_EVEN,
        cls=EvenAsphere,
        check=_check_asphere,
        fill=_fill_asphere(L.GEOM_EVEN),
        grad_params=_grad_asphere,
    )
)
register(
    GeometryAdapter(
        code=L.GEOM_ODD,
        cls=OddAsphere,
        check=_check_asphere,
        fill=_fill_asphere(L.GEOM_ODD),
        grad_params=_grad_asphere,
    )
)

register(
    ApertureAdapter(
        code=L.AP_RADIAL,
        cls=RadialAperture,
        check=_check_aperture("r_max", "r_min"),
        fill=_fill_radial,
        grad_params=_grad_aperture("r_max", "r_min"),
    )
)
register(
    ApertureAdapter(
        code=L.AP_OFFSET_RADIAL,
        cls=OffsetRadialAperture,
        check=_check_aperture("r_max", "r_min", "offset_x", "offset_y"),
        fill=_fill_offset_radial,
        grad_params=_grad_aperture("r_max", "r_min", "offset_x", "offset_y"),
    )
)
register(
    ApertureAdapter(
        code=L.AP_RECT,
        cls=RectangularAperture,
        check=_check_aperture("x_min", "x_max", "y_min", "y_max"),
        fill=_fill_rect,
        grad_params=_grad_aperture("x_min", "x_max", "y_min", "y_max"),
    )
)
register(
    ApertureAdapter(
        code=L.AP_ELLIPSE,
        cls=EllipticalAperture,
        check=_check_aperture("a", "b", "offset_x", "offset_y"),
        fill=_fill_ellipse,
        grad_params=_grad_aperture("a", "b", "offset_x", "offset_y"),
    )
)

register(
    InteractionAdapter(
        code=0,
        cls=RefractiveReflectiveModel,
        check=_check_interaction,
        fill=_fill_interaction,
        grad_params=_grad_interaction,
    )
)


def attach_fixtures(cls: type, fixtures: Any) -> None:
    """Give the adapter registered for ``cls`` its conformance fixtures.

    The adapter dataclasses are frozen (plan 3.3 pins their fields), so WP5
    attaches its builders through this helper instead of editing the registry
    entries in place.

    Args:
        cls: the exact class the adapter is registered for.
        fixtures: an iterable of zero-argument builders.

    Raises:
        KeyError: when no adapter is registered for ``cls``.
    """
    import dataclasses

    for registry in _REGISTRIES.values():
        adapter = registry.get(cls)
        if adapter is not None:
            registry[cls] = dataclasses.replace(adapter, fixtures=tuple(fixtures))
            return
    raise KeyError(f"no adapter registered for {cls!r}")
