"""SceneIR -- the top-level data-only scene description.

Kramer Harrison, 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal

import numpy as np

from optiland.backend.utils import is_torch_tensor
from optiland.nonsequential.ir.bsdf_ir import BsdfIR
from optiland.nonsequential.ir.medium_ir import MediumIR

# Primitive geometry kinds this revamp's lowering knows how to produce.
# "mesh" is included for completeness but requires the optional trimesh
# dependency to round-trip back into a live MeshGeometry.
PrimitiveKind = Literal["conic", "plane", "annulus", "frustum", "sphere", "mesh"]

# Which physical interaction a primitive's hit dispatches to, independent of
# any BsdfIR scatter overlay. A future BSDF-lobe rework could fold
# refractive/reflective/absorbing into BsdfIR itself, at which point this
# field would become redundant -- it has not been folded in yet.
ComponentKind = Literal["refractive", "reflective", "absorbing"]

EmitterKind = Literal["point", "collimated", "extended"]

SensorKind = Literal["irradiance", "spectral", "far_field", "ray_database"]


@dataclass(frozen=True)
class PrimitiveIR:
    """One surface, as plain data.

    Attributes:
        id: Index into ``SceneIR.primitives``; also referenced by
            ``interior_medium_id``/``exterior_medium_id`` of neighbouring
            volumes once ``Volume`` objects are wired into the IR from
            these ids (``Volume`` itself already exists in
            ``components/volume.py``; the IR-level wiring does not yet).
        kind: Geometry family; ``params`` is interpreted according to it.
        to_world: ``(4, 4)`` homogeneous local -> global transform.
        params: Kind-specific geometry parameters, e.g. for ``"conic"``:
            ``{"radius": ..., "conic": ..., "aperture_radius": ...}``. Values
            may be plain floats or backend arrays (a differentiable
            ``torch.Tensor`` stays attached; the translatability checklist
            only forbids closures and stateful objects, not autograd-carrying
            arrays).
        bsdf: Attached scatter model, if any (``BsdfIR(kind="none")`` when
            the surface is bare specular/refractive/absorbing).
        interior_medium_id: Index into ``SceneIR.media`` for the medium this
            surface bounds on its back side. Descriptive metadata only: the
            authoritative sidedness determination lives in each geometry's
            ``n_geom`` (see ``ComponentGeometry.ray_intersect`` and
            ``RefractiveComponent.interact``), not in these ids -- the
            interpreter still reads the medium directly off the live
            component. No volume topology is wired through the IR yet.
        exterior_medium_id: As above, for the front side.
        volume_id: Index into ``SceneIR.volumes``, or ``None`` when this
            primitive is not (yet) a volume boundary. Always ``None`` until
            volumes are wired into the IR.
        component_kind: Which physical interaction this primitive's hit
            dispatches to (see :data:`ComponentKind`).
        scatter_fraction: Probability that a hit ray is routed through
            ``bsdf`` rather than the specular/refractive path.
        name: Human-readable label, for diagnostics.
    """

    id: int
    kind: PrimitiveKind
    to_world: np.ndarray
    params: dict[str, Any]
    bsdf: BsdfIR
    interior_medium_id: int
    exterior_medium_id: int
    volume_id: int | None
    component_kind: ComponentKind
    scatter_fraction: float
    name: str = ""


@dataclass(frozen=True)
class VolumeIR:
    """A closed, outward-oriented set of boundary primitives (reserved).

    Not populated by :func:`~optiland.nonsequential.ir.lower.lower` --
    ``SceneIR.volumes`` is always ``()``. Medium sidedness does not need a
    ``Volume`` registry to be correct: each geometry's ``n_geom`` fixes the
    front/back determination directly (see ``RefractiveComponent.interact``).
    ``Volume`` itself -- watertightness validation, CSG composition, and
    ``Lens``/``Doublet``/``Mirror`` built on top of it -- already exists in
    ``optiland.nonsequential.components.volume``; only the IR-level wiring
    (populating this dataclass from a live scene's volumes) remains. This
    dataclass is defined now so the IR's shape will not need to change again
    when that wiring is added.

    Attributes:
        id: Index into ``SceneIR.volumes``.
        name: Human-readable label.
        boundary_primitive_ids: Ids into ``SceneIR.primitives`` forming the
            closed boundary.
        interior_medium_id: Index into ``SceneIR.media`` for the volume's
            interior.
    """

    id: int
    name: str
    boundary_primitive_ids: tuple[int, ...]
    interior_medium_id: int


@dataclass(frozen=True)
class EmitterIR:
    """A ray source, as plain data.

    Attributes:
        id: Index into ``SceneIR.emitters``.
        kind: Source family; ``params`` is interpreted according to it.
        to_world: ``(4, 4)`` homogeneous local -> global transform.
        params: Kind-specific parameters, including ``total_flux`` and the
            ``spectrum`` dict (``{"wavelengths": [...], "weights": [...]}``).
        medium_id: Index into ``SceneIR.media`` for the medium the source is
            embedded in, or ``None`` for vacuum.
        name: Human-readable label.
    """

    id: int
    kind: EmitterKind
    to_world: np.ndarray
    params: dict[str, Any]
    medium_id: int | None
    name: str = ""


@dataclass(frozen=True)
class SensorIR:
    """A detector, as plain data.

    Attributes:
        id: Index into ``SceneIR.sensors``.
        kind: Detector family; ``params`` is interpreted according to it.
        to_world: ``(4, 4)`` homogeneous local -> global transform.
        params: Kind-specific parameters (extents, pixel counts, splat, ...).
        primitive_id: Index into ``SceneIR.primitives``, reserved for a
            future unification of detectors into the primitive list itself.
            Always ``None`` -- detectors are dispatched by
            :mod:`optiland.nonsequential.detectors.dispatch`, a single
            nearest-hit routine shared by both reference backends (PR10
            deleted the two near-duplicate ``_intersect_detectors``
            implementations that previously lived on
            ``ArrayBackend``/``TorchBackend`` and had diverged in their
            grad-attachment semantics; D-10).
        absorb: Whether a hit terminates the ray. ``False`` => transmissive,
            mid-system sampling: the hit is recorded and the ray continues
            unchanged. Implemented as of PR10; mirrors the live detector's
            ``BaseDetector.absorb``.
        name: Human-readable label.
    """

    id: int
    kind: SensorKind
    to_world: np.ndarray
    params: dict[str, Any]
    primitive_id: int | None = None
    absorb: bool = True
    name: str = ""


@dataclass(frozen=True)
class RngContract:
    """Which RNG algorithm the scene's random draws are contracted to.

    Not a trace seed -- ``trace(seed=...)`` stays a per-call argument. This
    documents the *algorithm* every conforming backend must implement:
    PCG32, keyed by ``(seed, ray_id, bounce, event_slot)``. See
    :mod:`optiland.nonsequential.rng`.

    Attributes:
        algorithm: RNG algorithm identifier.
        version: Key-layout version, bumped if the ``(seed, ray_id, bounce,
            event_slot)`` mixing scheme in :mod:`optiland.nonsequential.rng`
            ever changes incompatibly.
    """

    algorithm: str = "pcg32"
    version: int = 1


@dataclass(frozen=True)
class SamplingPolicy:
    """Rare-path sampling policy.

    Every default reproduces the engine's pre-PR11 forward behaviour
    exactly: ``reflect_prob="fresnel"`` is the unconditional
    Fresnel-probability branch that always existed, and ``split_depth=0``
    means "never split," which was the only mode that existed before PR11.
    Set on a scene via ``NSQScene.sampling_policy``.

    Attributes:
        reflect_prob: Importance-sampling probability for the reflect
            branch (see
            :func:`optiland.nonsequential.sampling.resolve_reflect_prob`).
            ``"fresnel"`` uses the Fresnel reflectance itself (today's
            behaviour); ``"auto"`` clamps it into ``[0.25, 0.75]``; an
            explicit float fixes the probability. Works on both backends
            and under autograd -- the branch decision is always drawn from
            a detached probability with a compensating attached weight, so
            only the *variance* changes, never the expectation.
        split_depth: NumPy forward engine only; bounded bounce-splitting
            depth (see
            :mod:`optiland.nonsequential.backends.array_backend`). ``0`` =
            never split (the only mode the Torch backend supports -- it
            forces ``split_depth=0`` and warns if the scene sets a nonzero
            value, since fixed tensor shapes are required for the autograd
            graph).
        split_budget: Cap on live rays during splitting, as a multiple of
            ``batch_size``. Unused while ``split_depth=0``. Rays spawned
            beyond the cap are Russian-rouletted, not dropped.
        rr_start_flux: Russian-roulette threshold, as a fraction of
            per-ray initial flux (see
            :func:`optiland.nonsequential.sampling.russian_roulette`).
            Replaces the old biased hard kill below ``min_flux`` on
            both backends.
    """

    reflect_prob: float | Literal["fresnel", "auto"] = "fresnel"
    split_depth: int = 0
    split_budget: float = 4.0
    rr_start_flux: float = 1e-3


@dataclass(frozen=True)
class SceneIR:
    """The complete, backend-portable scene description.

    Attributes:
        primitives: All surfaces (from every compound component's flat
            ``.surfaces`` list), in ``scene.surfaces`` order.
        volumes: Always ``()`` (see :class:`VolumeIR`).
        media: Every distinct medium referenced by a primitive or emitter,
            deduplicated by catalog name (or the single shared vacuum
            entry).
        emitters: All sources, in ``scene.sources`` order.
        sensors: All detectors, in ``scene.detectors`` order.
        rng: RNG algorithm contract (see :class:`RngContract`).
        sampling: Rare-path sampling policy (see :class:`SamplingPolicy`).
    """

    primitives: tuple[PrimitiveIR, ...]
    volumes: tuple[VolumeIR, ...]
    media: tuple[MediumIR, ...]
    emitters: tuple[EmitterIR, ...]
    sensors: tuple[SensorIR, ...]
    rng: RngContract = field(default_factory=RngContract)
    sampling: SamplingPolicy = field(default_factory=SamplingPolicy)


# ---------------------------------------------------------------------------
# JSON round-trip (translatability checklist rule 5)
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Recursively convert IR field values to plain JSON-safe Python.

    Detaches ``torch.Tensor`` and ``numpy.ndarray`` values to nested lists
    (matching the detach convention already used by
    :mod:`optiland.nonsequential.serialization`), and dataclasses to plain
    dicts tagged with their class name for :func:`_from_jsonable` to reverse.

    Args:
        value: Any IR field value.

    Returns:
        A value built only from ``dict``, ``list``, ``str``, ``float``,
        ``int``, ``bool``, and ``None``.
    """
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if is_torch_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if is_dataclass(value):
        return {
            "__ir_type__": type(value).__name__,
            "fields": {
                f.name: _jsonable(getattr(value, f.name)) for f in fields(value)
            },
        }
    raise TypeError(
        f"Cannot serialize {value!r} of type {type(value).__name__} to JSON: "
        "SceneIR values must be dataclasses, dicts, lists/tuples, arrays, or "
        "plain scalars (translatability checklist rule 2)."
    )


_IR_TYPES: dict[str, type] = {
    "SceneIR": SceneIR,
    "PrimitiveIR": PrimitiveIR,
    "VolumeIR": VolumeIR,
    "EmitterIR": EmitterIR,
    "SensorIR": SensorIR,
    "MediumIR": MediumIR,
    "BsdfIR": BsdfIR,
    "RngContract": RngContract,
    "SamplingPolicy": SamplingPolicy,
}

# Fields whose JSON-list value must be restored as a numpy array rather than
# left as a plain Python list (everything else round-trips as list/dict/
# scalar unchanged).
_ARRAY_FIELDS = {"to_world"}


def _from_jsonable(value: Any) -> Any:
    """Inverse of :func:`_jsonable`.

    Args:
        value: A value previously produced by :func:`_jsonable`.

    Returns:
        The reconstructed dataclass / dict / list / scalar tree.
    """
    if isinstance(value, dict) and "__ir_type__" in value:
        cls = _IR_TYPES[value["__ir_type__"]]
        kwargs = {}
        for name, raw in value["fields"].items():
            restored = _from_jsonable(raw)
            if name in _ARRAY_FIELDS and isinstance(restored, list):
                restored = np.array(restored, dtype=np.float64)
            elif isinstance(restored, list) and name in (
                "primitives",
                "volumes",
                "media",
                "emitters",
                "sensors",
                "boundary_primitive_ids",
            ):
                restored = tuple(restored)
            kwargs[name] = restored
        return cls(**kwargs)
    if isinstance(value, dict):
        return {k: _from_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_jsonable(v) for v in value]
    return value


def scene_ir_to_dict(scene_ir: SceneIR) -> dict:
    """Serialize a :class:`SceneIR` to a JSON-safe dict, losslessly.

    Args:
        scene_ir: The IR to serialize.

    Returns:
        A dict built only from JSON-safe primitives, restorable via
        :func:`scene_ir_from_dict`.
    """
    return _jsonable(scene_ir)


def scene_ir_from_dict(d: dict) -> SceneIR:
    """Reconstruct a :class:`SceneIR` from a dict produced by
    :func:`scene_ir_to_dict`.

    Args:
        d: Dict previously produced by :func:`scene_ir_to_dict`.

    Returns:
        The reconstructed :class:`SceneIR`.
    """
    return _from_jsonable(d)
