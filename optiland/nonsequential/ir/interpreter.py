"""Reference IR interpreter -- NumPy/Torch backends drive their per-bounce
interaction loop from :class:`~optiland.nonsequential.ir.scene_ir.SceneIR`
data instead of iterating live ``scene.surfaces`` and branching on Python
class identity.

Design note: why this still calls into live component objects
----------------------------------------------------------------
A ``PrimitiveIR`` is pure data (translatability checklist, PR3): no
callbacks, no live objects. That is what a non-Python backend (Mitsuba,
OptiX) would need -- it never runs this module and never calls
``BaseComponent.interact()``; it reads ``PrimitiveIR.kind``/``params``/
``bsdf`` and writes its own kernel.

The NumPy/Torch *reference* interpreters are a different concern: they are
Python code, already have the live component objects in hand, and those
objects' ``intersect()``/``interact()``/``BaseBSDF.sample()`` methods are the
validated, gradient-checked implementation of the physics. Re-deriving that
same math a second time as free functions over raw ``PrimitiveIR.params``
would duplicate several hundred lines of numerically delicate code (conic
root selection, Harvey-Shack's cached inverse-CDF table, TIR handling, ...)
for no behavioural difference, at real risk of the two implementations
silently diverging. So the reference interpreters keep delegating the
*math* to the live objects, and this module supplies the part that is
genuinely new in PR4: the *dispatch* is driven by IR data (``PrimitiveIR
.component_kind``, ``PrimitiveIR.bsdf.kind``) rather than by
``isinstance()``/Python class identity, with a hard consistency check
(:func:`assert_component_kind_matches`, :func:`assert_bsdf_matches`) that
fires if ``lower()``'s mapping and a component's actual type ever disagree
-- the drift guard translatability checklist requires.

Kramer Harrison, 2026
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be

if TYPE_CHECKING:
    from optiland.nonsequential.components.base import BaseComponent
    from optiland.nonsequential.ir.scene_ir import BsdfIR, PrimitiveIR, SceneIR
    from optiland.nonsequential.ray_bundle import NSQRayBundle
    from optiland.nonsequential.rng import NSQRng

LogHitFn = Callable[["NSQRayBundle", "np.ndarray", str, object], None]
RayIdAllocator = Callable[[int], "np.ndarray"]


def _component_kind_of(component: BaseComponent) -> str:
    """Return the :data:`~.scene_ir.ComponentKind` of a live component.

    Reuses :func:`optiland.nonsequential.ir.lower._component_kind` so there
    is exactly one place that maps a live component type to its IR kind.

    Args:
        component: A scene surface.

    Returns:
        One of ``"refractive"``, ``"reflective"``, ``"absorbing"``.
    """
    from optiland.nonsequential.ir.lower import _component_kind  # noqa: PLC0415

    return _component_kind(component)


def _bsdf_kind_of(bsdf: object | None) -> str:
    """Return the :data:`~.bsdf_ir.BsdfKind` of a live BSDF (or ``None``).

    Reuses :func:`optiland.nonsequential.ir.lower._lower_bsdf` for the same
    reason as :func:`_component_kind_of`.

    Args:
        bsdf: A BSDF instance, or ``None``.

    Returns:
        The BSDF's IR kind string.
    """
    from optiland.nonsequential.ir.lower import _lower_bsdf  # noqa: PLC0415

    return _lower_bsdf(bsdf).kind


def assert_component_kind_matches(
    component: BaseComponent, primitive: PrimitiveIR
) -> None:
    """Raise if a live component's type disagrees with its lowered IR kind.

    Cheap (no per-ray cost): only compares two short strings. Called once
    per hit primitive per bounce, never per ray.

    Args:
        component: The live component ``primitive`` was lowered from.
        primitive: The corresponding :class:`PrimitiveIR`.

    Raises:
        RuntimeError: If the live component's interaction type no longer
            matches what ``lower()`` recorded -- a lowering/interpreter
            drift bug, not a user configuration error.
    """
    actual = _component_kind_of(component)
    if actual != primitive.component_kind:
        raise RuntimeError(
            f"Scene-IR drift on primitive '{primitive.name}': lower() "
            f"recorded component_kind={primitive.component_kind!r}, but the "
            f"live component is now a {type(component).__name__} "
            f"(kind {actual!r}). The interpreter and lower() must agree; "
            "this indicates a bug, not a scene configuration error."
        )


def assert_bsdf_matches(bsdf: object | None, bsdf_ir: BsdfIR) -> None:
    """Raise if a live BSDF's type disagrees with its lowered ``BsdfIR``.

    Args:
        bsdf: The live BSDF ``bsdf_ir`` was lowered from (or ``None``).
        bsdf_ir: The corresponding :class:`BsdfIR`.

    Raises:
        RuntimeError: If the live BSDF's type no longer matches what
            ``lower()`` recorded.
    """
    actual = _bsdf_kind_of(bsdf)
    if actual != bsdf_ir.kind:
        raise RuntimeError(
            f"Scene-IR drift: lower() recorded BsdfIR.kind={bsdf_ir.kind!r}, "
            f"but the live BSDF is now {bsdf!r} (kind {actual!r}). The "
            "interpreter and lower() must agree; this indicates a bug, not "
            "a scene configuration error."
        )


def apply_primitive_interactions(
    rays: NSQRayBundle,
    ir: SceneIR,
    components: list[BaseComponent],
    t_min: object,
    hit_normals: object,
    hit_n_geom: object,
    comp_idx: np.ndarray,
    comp_first_np: np.ndarray,
    rng: NSQRng,
    log_hit_fn: LogHitFn | None = None,
    ray_id_allocator: RayIdAllocator | None = None,
) -> NSQRayBundle | None:
    """Apply each hit primitive's interaction to ``rays``, in-place.

    This is the shared per-bounce "which surface did each ray hit, and what
    happens" step -- previously duplicated almost verbatim between
    ``ArrayBackend.trace()`` and ``TorchBackend.trace()`` (backend-specific
    only in whether ``t_min``/``hit_normals`` are eager NumPy arrays or
    attached Torch tensors, which ``optiland.backend`` already abstracts).

    Dispatch is IR-driven: primitives are visited in ``ir.primitives`` order
    (not by iterating ``components`` and asking "is this the hit one"), and
    each hit is checked against its recorded :data:`ComponentKind`/
    :class:`BsdfIR` before the live component's ``interact()`` executes the
    physics (see the module docstring for why the physics itself still
    lives on the component).

    Args:
        rays: Ray bundle to update in-place.
        ir: The scene's lowered IR (built once per ``trace()`` call).
        components: ``scene.surfaces``, in the same order ``ir.primitives``
            was built from -- ``components[i]`` is the live object
            ``ir.primitives[i]`` was lowered from.
        t_min: Per-ray nearest-primitive hit distance, shape (N,).
        hit_normals: Per-ray nearest-primitive hit normal, shape (N, 3).
        hit_n_geom: Per-ray nearest-primitive geometric (unflipped) normal,
            shape (N, 3); see ``ComponentGeometry.ray_intersect``.
        comp_idx: Per-ray index into ``ir.primitives``/``components`` of the
            nearest-hit primitive, or -1. NumPy int array.
        comp_first_np: Per-ray mask: True where a primitive (not a
            detector) is this ray's nearest hit and should be processed
            this bounce. NumPy bool array.
        rng: Keyed PCG32 RNG.
        log_hit_fn: Optional ``(rays, mask, primitive_name, t_offset)``
            callback for path recording, matching each backend's
            ``_log_hits`` closure.
        ray_id_allocator: ``(n) -> int64 ndarray`` of ``n`` fresh, previously
            -unused ray ids. Required to enable bounded splitting (D2, PR11,
            ``ir.sampling.split_depth > 0``) -- omit (the default) on the
            Torch backend, which forces ``split_depth=0`` and never spawns
            rays (fixed tensor shapes are required for the autograd graph).

    Returns:
        A new :class:`NSQRayBundle` of transmit-branch children spawned by
        bounded splitting this bounce, or ``None`` if none were spawned
        (splitting disabled, no eligible hits, or ``ray_id_allocator`` was
        not given). The caller is responsible for merging this into the
        live bundle -- see
        :meth:`optiland.nonsequential.backends.array_backend.ArrayBackend.trace`.
    """
    spawned_chunks: list[NSQRayBundle] = []

    for i, primitive in enumerate(ir.primitives):
        mask_i_np = comp_first_np & (comp_idx == i)
        if not mask_i_np.any():
            continue

        component = components[i]
        assert_component_kind_matches(component, primitive)
        assert_bsdf_matches(component.bsdf, primitive.bsdf)

        if log_hit_fn is not None:
            log_hit_fn(rays, mask_i_np, primitive.name, t_min)

        split_eligible_np = np.zeros_like(mask_i_np)
        if (
            ray_id_allocator is not None
            and ir.sampling.split_depth > 0
            and primitive.component_kind == "refractive"
        ):
            bounce_np = np.asarray(rays.bounce)
            split_eligible_np = mask_i_np & (bounce_np < ir.sampling.split_depth)

        split_idx = np.where(split_eligible_np)[0]
        if split_idx.size == 0:
            mask_i = be.array(mask_i_np)
            component.interact(
                rays,
                t_min,
                hit_normals,
                mask_i,
                rng,
                primitive.bsdf,
                hit_n_geom,
                sampling=ir.sampling,
            )
            continue

        # Bounded splitting: a hit ray below split_depth spawns
        # *both* Fresnel children instead of drawing one stochastically.
        # 1) Snapshot the pre-interaction state of the splitting subset
        #    (fresh ray ids, so its RNG stream is independent of the sibling
        #    that keeps the original id) before either branch mutates
        #    anything.
        new_ids = ray_id_allocator(split_idx.size)
        transmit_snapshot = rays.select(split_idx, ray_id=new_ids)
        snap_t = np.asarray(t_min)[split_idx]
        snap_normals = np.asarray(hit_normals)[split_idx]
        snap_n_geom = np.asarray(hit_n_geom)[split_idx]
        snap_mask = be.array(np.ones(split_idx.size, dtype=bool))

        # 2) Non-splitting remainder of this primitive's hits (if any):
        #    normal single-branch draw.
        remainder_np = mask_i_np & ~split_eligible_np
        if remainder_np.any():
            component.interact(
                rays,
                t_min,
                hit_normals,
                be.array(remainder_np),
                rng,
                primitive.bsdf,
                hit_n_geom,
                sampling=ir.sampling,
            )

        # 3) Reflect child: force the branch in place on the original rays.
        component.interact(
            rays,
            t_min,
            hit_normals,
            be.array(split_eligible_np),
            rng,
            primitive.bsdf,
            hit_n_geom,
            sampling=ir.sampling,
            forced_branch="reflect",
        )

        # 4) Transmit child: force the other branch on the snapshot, which
        #    becomes a newly spawned ray in the live bundle.
        component.interact(
            transmit_snapshot,
            snap_t,
            snap_normals,
            snap_mask,
            rng,
            primitive.bsdf,
            snap_n_geom,
            sampling=ir.sampling,
            forced_branch="transmit",
        )
        spawned_chunks.append(transmit_snapshot)

    if not spawned_chunks:
        return None
    from optiland.nonsequential.ray_bundle import NSQRayBundle  # noqa: PLC0415

    return NSQRayBundle.concat(spawned_chunks)
