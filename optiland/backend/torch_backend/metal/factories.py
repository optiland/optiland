"""Host-side constructors returning ``MetalFloat64`` tensors.

Torch refuses ``float64`` on ``mps`` at every entry point (``torch.tensor``,
``torch.zeros``, ``.to``), so the torch backend never asks for it: whenever the
backend is configured for ``device='mps'`` and ``precision='float64'`` its
creation methods come here instead. Every function returns a
:class:`~optiland.backend.torch_backend.metal.tensor.MetalFloat64` in the
representation selected by :func:`optiland.backend.torch_backend.metal.get_mode`
(``mode=None``) or in an explicit ``mode``.

Values are produced exactly: sequences and arrays are converted in NumPy
float64 and encoded (``arange``/``linspace`` are evaluated by NumPy so the
endpoints are exact), constants are split into components on the host, and
plain torch tensors of any dtype are promoted exactly through the nucleus
:func:`~optiland.backend.torch_backend.metal.tensor.coerce`.

Residency follows the nucleus rule (NOTES/02-design.md section 4): a created
tensor with at most ``HOST_THRESHOLD`` elements is *host-resident* (its
authoritative value is a CPU float64 tensor, encoded onto the GPU lazily), a
larger one is *GPU-resident* (component tensors on ``mps``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from optiland.backend.torch_backend.metal import encode, get_mode
from optiland.backend.torch_backend.metal import tensor as _tensor
from optiland.backend.torch_backend.metal.tensor import (
    MetalFloat64,
    coerce,
    is_metal,
    wrap,
    wrap_host,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DEVICE = "mps"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _mode(mode: str | None) -> str:
    return mode or get_mode()


def _shape(shape: Any) -> tuple[int, ...]:
    """Normalize ``shape`` (int, 0-d, sequence, torch.Size) to a tuple of ints."""
    if isinstance(shape, torch.Size):
        return tuple(shape)
    if isinstance(shape, (list, tuple)):
        return tuple(int(s) for s in shape)
    if isinstance(shape, np.ndarray):
        return tuple(int(s) for s in shape.reshape(-1))
    return (int(shape),)


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def host_threshold() -> int:
    """The nucleus' residency threshold (read live; it can change at runtime)."""
    return int(_tensor.HOST_THRESHOLD)


def host_resident(shape: Any) -> bool:
    """True when a *created* tensor of ``shape`` lives on the host (nucleus rule)."""
    return _numel(_shape(shape)) <= host_threshold()


def _scalar(value: Any) -> float:
    """Python float of a scalar (number, 0-d array or tensor, incl. MetalFloat64)."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(()).item())
    return float(np.asarray(value, dtype=np.float64).reshape(()))


def to_host(data: Any) -> Any:
    """Recursively replace tensors inside ``data`` by host NumPy arrays / scalars.

    ``np.asarray`` cannot build an array from a list of 0-d ``MetalFloat64``
    tensors (their ``__array__`` returns a 0-d array numpy rejects in that
    position), so tensor leaves are decoded here first. The result is safe to
    hand to ``np.asarray(..., dtype=np.float64)``.

    Args:
        data: Nested lists/tuples, arrays, numbers or tensors.

    Returns:
        The same structure with every tensor turned into a NumPy array.

    Raises:
        TypeError: If ``data`` contains ``None`` (``torch.tensor`` rejects it
            too, whereas NumPy would silently produce NaN).
    """
    if data is None:
        raise TypeError("must be real number, not NoneType")
    if is_metal(data):
        return data.to_numpy()
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, (list, tuple)):
        return [to_host(d) for d in data]
    return data


def constant_components(
    shape: Sequence[int], value: float, mode: str | None = None, device: str = DEVICE
) -> tuple[torch.Tensor, ...]:
    """Contiguous GPU component tensors holding ``value`` everywhere.

    Args:
        shape: Shape of the result.
        value: The constant, encoded exactly (``lo = value - float32(value)`` for
            df64, the binary64 bit pattern for sf64; non-finite values get ``lo=0``).
        mode: Representation (default: current mode).
        device: Target device.

    Returns:
        tuple[torch.Tensor, ...]: ``(hi, lo)`` float32 tensors or ``(bits,)`` int64.
    """
    m = _mode(mode)
    shape = _shape(shape)
    if m == "df64":
        hi, lo = encode.df64_scalar(float(value))
        return (
            torch.full(shape, hi, dtype=torch.float32, device=device),
            torch.full(shape, lo, dtype=torch.float32, device=device),
        )
    bits = encode.sf64_scalar(float(value))
    return (torch.full(shape, bits, dtype=torch.int64, device=device),)


def empty_components(
    shape: Sequence[int], mode: str | None = None, device: str = DEVICE
) -> tuple[torch.Tensor, ...]:
    """Uninitialized contiguous GPU component tensors of ``shape``."""
    m = _mode(mode)
    shape = _shape(shape)
    if m == "df64":
        return (
            torch.empty(shape, dtype=torch.float32, device=device),
            torch.empty(shape, dtype=torch.float32, device=device),
        )
    return (torch.empty(shape, dtype=torch.int64, device=device),)


def _clone_metal(data: MetalFloat64, mode: str) -> MetalFloat64:
    """Detached copy of a MetalFloat64 in ``mode`` (residency by size)."""
    if data.mode != mode or (
        not data.is_host_resident and data.numel() <= host_threshold()
    ):
        return from_numpy(data.to_numpy(), mode=mode)
    if data.is_host_resident:
        return wrap_host(data.to_cpu_float64().clone(), mode)
    return wrap(tuple(c.clone() for c in data.components), mode)


# ---------------------------------------------------------------------------
# constructors
# ---------------------------------------------------------------------------
def from_numpy(
    a: Any, requires_grad: bool = False, mode: str | None = None
) -> MetalFloat64:
    """Encode a float64 array-like (any shape, incl. 0-d) as a new ``MetalFloat64``."""
    if a is None:
        raise TypeError("must be real number, not NoneType")
    return MetalFloat64.from_numpy(
        np.asarray(a, dtype=np.float64),
        _mode(mode),
        device=DEVICE,
        requires_grad=requires_grad,
    )


def tensor(
    data: Any, requires_grad: bool = False, mode: str | None = None
) -> MetalFloat64:
    """``torch.tensor`` for the emulated dtype: always a fresh, detached copy.

    Args:
        data: Number, (nested) sequence, NumPy array, or any torch tensor
            (plain CPU / mps tensors of any dtype, ``MetalFloat64``); tensors
            nested in sequences are decoded on the host first.
        requires_grad: Make the result an autograd leaf.
        mode: Representation (default: current mode).

    Returns:
        MetalFloat64: The encoded values.
    """
    m = _mode(mode)
    if is_metal(data):
        out = _clone_metal(data.detach(), m)
    elif isinstance(data, torch.Tensor):
        t = data.detach()
        if t.device.type != "mps" or t.numel() <= host_threshold():
            out = from_numpy(t.cpu().to(torch.float64).numpy(), mode=m)
        else:
            # ``coerce`` may alias the input (float32 mps -> hi): clone so the
            # result owns its memory like torch.tensor.
            out = wrap(tuple(c.clone() for c in coerce(t, m).components), m)
    else:
        out = from_numpy(to_host(data), mode=m)
    if requires_grad:
        out.requires_grad_(True)
    return out


def as_tensor(data: Any, mode: str | None = None) -> MetalFloat64:
    """``torch.as_tensor`` for the emulated dtype: no copy for a ``MetalFloat64``.

    A ``MetalFloat64`` already in ``mode`` is returned as is (autograd history
    intact); plain tensors are promoted exactly (large mps tensors stay on the
    GPU, everything else becomes host- or GPU-resident by size); everything
    else is encoded.
    """
    m = _mode(mode)
    if is_metal(data):
        return data if data.mode == m else from_numpy(data.to_numpy(), mode=m)
    if isinstance(data, torch.Tensor):
        if data.requires_grad and torch.is_grad_enabled() and m == get_mode():
            # A plain leaf that requires grad (a ``torch.nn.Parameter`` held
            # by a material, the CPU float64 inputs of a geometry test) stays
            # differentiable: ``_to_copy`` to the emulated dtype is served by
            # the factory intercept and its backward decodes the gradient
            # back to the plain leaf.
            _tensor.enable_factory_intercept()
            return data.to(device=DEVICE, dtype=torch.float64)
        t = data.detach()
        if t.device.type == "mps" and t.numel() > host_threshold():
            return coerce(t, m)
        return from_numpy(t.cpu().to(torch.float64).numpy(), mode=m)
    return from_numpy(to_host(data), mode=m)


def scalar(
    value: Any, requires_grad: bool = False, mode: str | None = None
) -> MetalFloat64:
    """0-d tensor holding ``value`` (exactly, up to the representation)."""
    return full((), _scalar(value), requires_grad=requires_grad, mode=mode)


def full(
    shape: Any, fill_value: Any, requires_grad: bool = False, mode: str | None = None
) -> MetalFloat64:
    """Tensor of ``shape`` filled with ``fill_value`` (encoded exactly on the host)."""
    m = _mode(mode)
    shape = _shape(shape)
    value = _scalar(fill_value)
    if host_resident(shape):
        return wrap_host(
            torch.full(shape, value, dtype=torch.float64), m, requires_grad
        )
    return wrap(constant_components(shape, value, m), m, requires_grad)


def zeros(
    shape: Any, requires_grad: bool = False, mode: str | None = None
) -> MetalFloat64:
    """Zero-filled tensor of ``shape``."""
    return full(shape, 0.0, requires_grad=requires_grad, mode=mode)


def ones(
    shape: Any, requires_grad: bool = False, mode: str | None = None
) -> MetalFloat64:
    """One-filled tensor of ``shape``."""
    return full(shape, 1.0, requires_grad=requires_grad, mode=mode)


def empty(
    shape: Any, requires_grad: bool = False, mode: str | None = None
) -> MetalFloat64:
    """Uninitialized tensor of ``shape`` (memory is not zeroed)."""
    m = _mode(mode)
    shape = _shape(shape)
    if host_resident(shape):
        return wrap_host(torch.empty(shape, dtype=torch.float64), m, requires_grad)
    return wrap(empty_components(shape, m), m, requires_grad)


def arange(
    start: Any,
    end: Any = None,
    step: Any = 1,
    requires_grad: bool = False,
    mode: str | None = None,
) -> MetalFloat64:
    """``numpy.arange`` evaluated in float64 on the host, then encoded.

    Args:
        start: Start of the interval (or ``end`` when ``end`` is None).
        end: End of the interval (exclusive).
        step: Spacing between values.
        requires_grad: Make the result an autograd leaf.
        mode: Representation (default: current mode).

    Returns:
        MetalFloat64: 1-D tensor.
    """
    if end is None:
        start, end = 0.0, start
    values = np.arange(_scalar(start), _scalar(end), _scalar(step), dtype=np.float64)
    return from_numpy(values, requires_grad=requires_grad, mode=mode)


def linspace(
    start: Any,
    end: Any,
    num: int = 50,
    requires_grad: bool = False,
    mode: str | None = None,
) -> MetalFloat64:
    """``numpy.linspace`` in float64 on the host (exact endpoints), then encoded."""
    values = np.linspace(_scalar(start), _scalar(end), int(num), dtype=np.float64)
    return from_numpy(values, requires_grad=requires_grad, mode=mode)


def eye(
    n: int, m: int | None = None, requires_grad: bool = False, mode: str | None = None
) -> MetalFloat64:
    """Identity matrix of shape ``(n, m or n)``."""
    return from_numpy(
        np.eye(int(n), None if m is None else int(m)), requires_grad, mode
    )


__all__ = [
    "DEVICE",
    "arange",
    "as_tensor",
    "constant_components",
    "empty",
    "empty_components",
    "eye",
    "from_numpy",
    "full",
    "host_resident",
    "host_threshold",
    "linspace",
    "ones",
    "scalar",
    "tensor",
    "to_host",
    "zeros",
]
