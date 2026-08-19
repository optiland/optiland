"""Shared utilities for the NSQ subpackage.

Kramer Harrison, 2026
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

from optiland.backend.utils import to_numpy  # noqa: F401

if TYPE_CHECKING:
    from optiland._types import ScalarOrArrayT

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

# Default rays per trace batch. Affects speed only, not the result: the loop
# holds several temporaries per live ray, and measured CPU throughput peaks
# between roughly 10k and 20k rays per batch, falling off 2-3x on either side.
DEFAULT_BATCH_SIZE = 16_384


def is_tensor(value: Any) -> bool:
    """Return True if ``value`` is a PyTorch tensor.

    Args:
        value: Any object.

    Returns:
        True when torch is installed and ``value`` is a ``torch.Tensor``.
    """
    if not _TORCH_AVAILABLE:
        return False
    import torch  # noqa: PLC0415

    return isinstance(value, torch.Tensor)


def as_param(value: ScalarOrArrayT) -> Any:
    """Normalize a scene parameter while preserving autograd tensors.

    Scene parameters (radii, apertures, detector extents) are stored as plain
    Python floats so that NumPy tracing stays fast and picklable.  A
    ``torch.Tensor`` is passed through untouched, so a tensor carrying
    ``requires_grad=True`` stays connected to the autograd graph and receives
    gradients from ``loss.backward()``.

    Args:
        value: Scalar parameter, or a torch Tensor for differentiable use.

    Returns:
        The tensor unchanged, or ``float(value)``.
    """
    if is_tensor(value):
        return value
    return float(value)


def as_detached_param(
    value: ScalarOrArrayT,
    name: str,
    owner: str,
    reason: str = "it feeds NumPy-only sampling",
) -> float:
    """Return a parameter as a float, rejecting gradient-carrying tensors.

    Used for parameters that feed NumPy-only Monte Carlo sampling or binning
    and therefore cannot propagate gradients in this release.  Passing a
    ``requires_grad=True`` tensor raises instead of detaching silently, so a
    user never optimizes a variable that has no effect on the loss.

    Args:
        value: Scalar parameter.
        name: Parameter name, for the error message.
        owner: Owning class name, for the error message.
        reason: Why this parameter cannot carry gradients.

    Returns:
        ``float(value)``.

    Raises:
        NotImplementedError: If ``value`` is a tensor requiring grad.
    """
    if is_tensor(value) and value.requires_grad:
        raise NotImplementedError(
            f"{owner}.{name} cannot be differentiated in this release because "
            f"{reason}, so gradients would be silently dropped. Pass a plain "
            f"float for '{name}'. Differentiable parameters are: component "
            f"geometry (radius, conic, aperture), IrradianceDetector extents, "
            f"source total_flux, material index, and BSDF reflectance."
        )
    return float(value)


def distribute_ray_budget(num_rays_total: int, source_fluxes: list[float]) -> list[int]:
    """Split a total ray budget across sources proportional to flux.

    Every source gets at least one ray, and the returned counts sum to
    exactly ``num_rays_total``. A naive per-source floor of ``max(1, ...)``
    applied to every source but the last can drive ``remaining`` to zero or
    negative when there are many low-flux sources, starving the last
    (typically highest-flux) source. This instead floors every source to 1
    and distributes the remainder proportionally to flux, so the floor can
    never consume more than ``len(source_fluxes)`` rays overall.

    Args:
        num_rays_total: Total number of rays to distribute. Must be >=
            ``len(source_fluxes)`` for every source to receive at least one
            ray; if smaller, the first ``num_rays_total`` sources (by flux
            order) each get one ray and the rest get zero.
        source_fluxes: Per-source flux, in source order. Used as weights.

    Returns:
        Per-source ray counts, in source order, summing to
        ``min(num_rays_total, ...)`` such that the total equals
        ``num_rays_total`` whenever ``num_rays_total >= len(source_fluxes)``.
    """
    n_sources = len(source_fluxes)
    if n_sources == 0:
        return []
    if n_sources == 1:
        return [num_rays_total]

    total_flux = sum(source_fluxes)
    if total_flux <= 0:
        # No flux information to weight by: split as evenly as possible.
        base = num_rays_total // n_sources
        counts = [base] * n_sources
        for i in range(num_rays_total - base * n_sources):
            counts[i] += 1
        return counts

    # Floor every source at 1 ray, then distribute the remainder
    # proportionally to flux using largest-remainder rounding so the total
    # matches num_rays_total exactly.
    floor_total = min(num_rays_total, n_sources)
    counts = [1 if i < floor_total else 0 for i in range(n_sources)]
    remaining = num_rays_total - floor_total
    if remaining <= 0:
        return counts

    shares = [remaining * (f / total_flux) for f in source_fluxes]
    base_extra = [int(s) for s in shares]
    allocated = sum(base_extra)
    leftover = remaining - allocated

    # Assign leftover rays (from integer truncation) to the sources with the
    # largest fractional remainder, breaking ties by flux.
    remainders = sorted(
        range(n_sources),
        key=lambda i: (shares[i] - base_extra[i], source_fluxes[i]),
        reverse=True,
    )
    for i in range(n_sources):
        counts[i] += base_extra[i]
    for i in remainders[:leftover]:
        counts[i] += 1

    return counts


def estimate_bounding_scale(scene: Any) -> float:
    """Estimate a reasonable length to extend escaped rays past the scene.

    Args:
        scene: NSQScene instance (or anything exposing ``surfaces`` with a
            ``bounding_box`` attribute per surface).

    Returns:
        The scene's bounding-box diagonal [mm], or ``100.0`` if the scene has
        no surfaces or the diagonal is degenerately small.
    """
    boxes = [comp.bounding_box for comp in scene.surfaces]
    if not boxes:
        return 100.0

    xmin = min(b.xmin for b in boxes)
    xmax = max(b.xmax for b in boxes)
    ymin = min(b.ymin for b in boxes)
    ymax = max(b.ymax for b in boxes)
    zmin = min(b.zmin for b in boxes)
    zmax = max(b.zmax for b in boxes)

    extent = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) ** 0.5
    return extent if extent > 1.0 else 100.0


def get_detector_names(scene: Any) -> list[str]:
    """Extract registry names for a scene's detectors.

    Args:
        scene: NSQScene instance.

    Returns:
        Ordered list of detector names, or ``[]`` if the scene has no
        detector registry.
    """
    try:
        return list(scene.detector_registry._registry.keys())  # type: ignore[attr-defined]
    except AttributeError:
        return []


def as_float(value: ScalarOrArrayT) -> float:
    """Return a parameter as a detached Python float.

    Used where a value feeds NumPy-only bookkeeping (bin edges, bounding
    boxes, plot extents) that cannot carry gradients regardless.

    Args:
        value: Scalar parameter or torch Tensor.

    Returns:
        Detached float value.
    """
    if is_tensor(value):
        return float(to_numpy(value))
    return float(value)
