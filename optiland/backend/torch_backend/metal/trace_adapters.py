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
