"""Checks if a derivative with respect to a value can be requested.

Some geometries leave out the terms of zero coefficients, because these terms
add nothing to the sag. But the derivative of a term with respect to its
coefficient is not zero when the coefficient is zero. A term can therefore be
left out only if no derivative with respect to its coefficient can be
requested. With Torch, a caller can request a derivative in three ways:

- reverse mode: the tensor requires grad and gradient recording is on. This
  includes ``torch.func.grad``, ``vjp`` and ``jacrev``.
- ``torch.func`` transforms such as ``jvp``, ``jacfwd`` and ``vmap``. They wrap
  the tensor, and the wrapped tensor has no values that the host can read.
- forward mode with ``torch.autograd.forward_ad``: the tensor has a tangent.

NumPy arrays and Python numbers never carry a derivative. The checks read no
tensor values, so they never wait for a GPU.
"""

from __future__ import annotations

import sys


def derivative_requested(value) -> bool:
    """Return True if a derivative with respect to ``value`` can be requested now.

    Args:
        value: A tensor, an array or a number.

    Returns:
        bool: False for NumPy arrays, numbers, and tensors in a context that
        records no derivative (for example inside ``torch.no_grad``).
    """
    if not hasattr(value, "requires_grad"):
        return False
    torch = sys.modules["torch"]
    if value.requires_grad and torch.is_grad_enabled():
        return True
    return _is_wrapped(value, torch) or _has_tangent(value, torch)


def reverse_mode_only() -> bool:
    """Return True if a derivative can be requested only in reverse mode now.

    That is the case outside ``torch.func`` transforms and outside a
    forward-mode dual level. Then a custom autograd function without rules
    for forward mode and transforms can be used.

    Returns:
        bool: False inside a transform or a dual level, and if Torch has no
        check for transforms.
    """
    torch = sys.modules["torch"]
    active = getattr(torch._C, "_are_functorch_transforms_active", None)
    if active is None or active():
        return False
    return getattr(torch.autograd.forward_ad, "_current_level", 0) < 0


def is_transformed(value) -> bool:
    """Return True if ``value`` is a tensor wrapped by a ``torch.func`` transform.

    The host cannot read the values of such a tensor.

    Args:
        value: A tensor, an array or a number.

    Returns:
        bool: True for a wrapped tensor.
    """
    if not hasattr(value, "requires_grad"):
        return False
    return _is_wrapped(value, sys.modules["torch"])


def _is_wrapped(value, torch) -> bool:
    functorch = getattr(torch._C, "_functorch", None)
    check = getattr(functorch, "is_functorch_wrapped_tensor", None)
    # Without the check (a future Torch version), act as before this check
    # existed: treat the tensor as not wrapped.
    return check is not None and bool(check(value))


def _has_tangent(value, torch) -> bool:
    forward_ad = torch.autograd.forward_ad
    # The level is -1 when no dual level is active: then no tensor has a
    # tangent, and the check costs one attribute read.
    if getattr(forward_ad, "_current_level", 0) < 0:
        return False
    return forward_ad.unpack_dual(value).tangent is not None
