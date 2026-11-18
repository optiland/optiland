"""One device-to-host copy of the values that a CUDA trace checks in Python.

The trace checks some tensor values in Python with ``bool()`` or
``math.isinf()``: the rotation angles of each coordinate system, whether a
radius is infinite, and whether a medium absorbs (``any(k > 0)``). On a CUDA
device, each check copies the value to the host, and the copy waits until
the device has done all queued work (a CUDA synchronization).

``SurfaceGroup`` opens a host-value scope (``host_scope``) around the
generic trace when the rays are on a CUDA device. The scope copies these
values to the host in one device-to-host copy, and the checks read the
copies (``host`` and ``absorbs``). The checks take the same branches, so the
results and the gradients do not change.

The copies are taken when the scope opens and are not updated. Code in the
scope must not change these tensors in place. The Optiland trace does not.

The scope copies any(k > 0) only inside an epoch and when the wavelengths do
not require gradients. Otherwise each absorption check still waits.

This module calls torch directly, but only for CUDA tensors: ``_copy``
copies only CUDA tensors and returns None for other arrays, and it imports
torch only when it has a CUDA tensor to copy. For a value without a copy,
for example a NumPy array, ``host`` and ``absorbs`` return None, and the
caller checks the value as before.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import TypeAlias

    # (kind, id(value)) -> (value, host copy).
    _Memo: TypeAlias = dict[tuple[int, int], tuple[object, float | bool]]

# The memo of the open scope. Each entry keeps its tensor alive, so no other
# object can get its id while the scope is open. A context variable, like
# the evaluation epoch, so other threads do not see the scope.
_HOST: contextvars.ContextVar[_Memo | None] = contextvars.ContextVar(
    "optiland_host_values", default=None
)
# Private switch for the equivalence tests.
_ENABLED = True
# The device types whose values the scope copies. A copy from the CPU does
# not wait, so the CPU is not in the set. The tests add it, so that the
# scope runs without a GPU.
_DEVICE_TYPES = frozenset({"cuda"})

# The kinds of copies: the first entry of a memo key.
_SCALAR = 0
_ABSORBS = 1


@contextlib.contextmanager
def host_scope(
    scalars: Iterable[object] = (), absorption: Iterable[object] = ()
) -> Iterator[None]:
    """Open a scope in which ``host`` and ``absorbs`` read host copies.

    Args:
        scalars: Values that the trace checks with bool() or math.isinf().
            The scope copies each one-element real tensor on the device of
            the first CUDA tensor and ignores the rest.
        absorption: Extinction coefficients k that the trace checks with
            any(k > 0). The scope copies any(k > 0) of each real tensor on
            the device of the first CUDA tensor and ignores the rest.

    Yields:
        None. In the ``with`` block, ``host`` and ``absorbs`` return the
        copies. The copies of an outer scope stay visible.
    """
    memo = _copy(scalars, absorption) if _ENABLED else None
    if memo is None:
        yield
        return
    outer = _HOST.get()
    if outer:
        memo = {**outer, **memo}
    token = _HOST.set(memo)
    try:
        yield
    finally:
        _HOST.reset(token)


def on_scope_device(value: object) -> bool:
    """Return True if ``value`` is on a device whose values the scope copies.

    The check reads ``value.device.type``, so it does not import torch.
    """
    device = getattr(value, "device", None)
    return getattr(device, "type", None) in _DEVICE_TYPES


def host(value: object) -> float | None:
    """Return the host copy of a value that the trace checks.

    Args:
        value: A value that the trace checks with bool() or math.isinf().

    Returns:
        The host copy, or None if the scope has no copy of ``value``.
    """
    memo = _HOST.get()
    return None if memo is None else _entry(memo, _SCALAR, value)


def absorbs(k: object) -> bool | None:
    """Return any(k > 0) from the host copy.

    Args:
        k: The extinction coefficients of a medium.

    Returns:
        The host copy, or None if the scope has no copy of ``k``.
    """
    memo = _HOST.get()
    return None if memo is None else _entry(memo, _ABSORBS, k)


def _entry(memo: _Memo, kind: int, value: object) -> float | bool | None:
    """Return the copy of ``value`` of this kind in ``memo``, or None."""
    entry = memo.get((kind, id(value)))
    if entry is None or entry[0] is not value:
        return None
    return entry[1]


def _copy(scalars: Iterable[object], absorption: Iterable[object]) -> _Memo | None:
    """One device-to-host copy for all values.

    torch.stack needs one device, so tensors on another device are not
    copied and keep their own check. The device is that of the first CUDA
    tensor of ``scalars``, or of ``absorption`` if ``scalars`` has none.

    Args:
        scalars: The values of ``host_scope``.
        absorption: The extinction coefficients of ``host_scope``.

    Returns:
        The memo of the scope, or None if no value has a copy.
    """
    scalars = [value for value in scalars if on_scope_device(value)]
    absorption = [k for k in absorption if on_scope_device(k)]
    if not scalars and not absorption:
        return None
    import torch  # local: this module must import without torch

    device = (scalars or absorption)[0].device
    # (kind, id(value)) -> (value, 0-d tensor to copy). A value that is in
    # the lists twice gets one entry. detach(): the copies are not part of
    # the autograd graph.
    pending: dict[tuple[int, int], tuple[object, torch.Tensor]] = {}
    for value in scalars:
        if (
            isinstance(value, torch.Tensor)
            and value.device == device
            and value.numel() == 1
            and not value.is_complex()
        ):
            copy = value.detach().reshape(())
            pending.setdefault((_SCALAR, id(value)), (value, copy))
    for k in absorption:
        if isinstance(k, torch.Tensor) and k.device == device and not k.is_complex():
            copy = torch.any(k.detach() > 0)
            pending.setdefault((_ABSORBS, id(k)), (k, copy))
    if not pending:
        return None
    # float64 holds each float32 and float64 value exactly, so each copy has
    # the truth value, the sign and the infinity of its tensor.
    stacked = torch.stack([copy.to(torch.float64) for _, copy in pending.values()])
    numbers = stacked.cpu().tolist()  # the one device-to-host copy
    memo: _Memo = {}
    for (key, (value, _)), number in zip(pending.items(), numbers, strict=True):
        memo[key] = (value, number if key[0] == _SCALAR else bool(number))
    return memo
