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
