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

import optiland.backend as be

if TYPE_CHECKING:
    import numpy as np

    from optiland.nonsequential.components.base import BaseComponent
    from optiland.nonsequential.ir.scene_ir import BsdfIR, PrimitiveIR, SceneIR
    from optiland.nonsequential.ray_bundle import NSQRayBundle
    from optiland.nonsequential.rng import NSQRng

LogHitFn = Callable[["NSQRayBundle", "np.ndarray", str, object], None]


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
    comp_idx: np.ndarray,
    comp_first_np: np.ndarray,
    rng: NSQRng,
    log_hit_fn: LogHitFn | None = None,
) -> None:
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
        comp_idx: Per-ray index into ``ir.primitives``/``components`` of the
            nearest-hit primitive, or -1. NumPy int array.
        comp_first_np: Per-ray mask: True where a primitive (not a
            detector) is this ray's nearest hit and should be processed
            this bounce. NumPy bool array.
        rng: Keyed PCG32 RNG.
        log_hit_fn: Optional ``(rays, mask, primitive_name, t_offset)``
            callback for path recording, matching each backend's
            ``_log_hits`` closure.
    """
    for i, primitive in enumerate(ir.primitives):
        mask_i_np = comp_first_np & (comp_idx == i)
        if not mask_i_np.any():
            continue

        component = components[i]
        assert_component_kind_matches(component, primitive)
        assert_bsdf_matches(component.bsdf, primitive.bsdf)

        if log_hit_fn is not None:
            log_hit_fn(rays, mask_i_np, primitive.name, t_min)

        mask_i = be.array(mask_i_np)
        component.interact(rays, t_min, hit_normals, mask_i, rng, primitive.bsdf)
