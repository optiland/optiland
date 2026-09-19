"""MetalFloat64 dispatch handlers: creation ops (registered via tensor.implements).

Two families reach ``__torch_dispatch__`` with a ``MetalFloat64`` operand:

* ``*_like`` and ``new_*`` ops (``zeros_like``, ``full_like``, ``new_zeros``,
  ``new_empty_strided``, ...), which autograd's derivative formulas and
  Optiland's backend call constantly. They honour the ``dtype`` / ``device``
  keyword arguments: a float64 result on ``mps`` is a fresh ``MetalFloat64``
  (``lo = 0`` for df64 constants, exact bit patterns for sf64), any other dtype
  is a plain mps tensor of that dtype, and another device gets a plain tensor
  there. Residency follows the nucleus rule: the pre-dispatch in
  ``MetalFloat64.__torch_dispatch__`` already runs these ops on the host when
  every Metal operand is small, so the handlers here see GPU-resident (large)
  operands; a *result* with at most ``HOST_THRESHOLD`` elements is still
  created host-resident (it is a new tensor, aliasing nothing).
* plain factories (``empty``, ``zeros``, ``full``, ``arange``, ``linspace``,
  ``eye``, ``scalar_tensor``, ...). These normally never dispatch here (they
  have no tensor operand, so torch rejects ``float64`` on ``mps`` before
  dispatch and the backend builds through :mod:`.factories` instead); the
  handlers exist for the rare paths that do arrive with a ``MetalFloat64``
  somewhere in the arguments.
"""

from __future__ import annotations

from typing import Any

import torch

from optiland.backend.torch_backend.metal import factories
from optiland.backend.torch_backend.metal.tensor import (
    apply_to_components,
    aten,
    implements,
    is_metal,
    scalar_value,
    wrap,
    wrap_host,
)

_HOST_KIND = {
    "empty": torch.empty,
    "zeros": torch.zeros,
    "ones": torch.ones,
}


def _target(x: Any, kwargs: dict[str, Any]) -> tuple[torch.dtype, torch.device]:
    """Resolve the requested (dtype, device) of a creation op relative to ``x``."""
    dtype = kwargs.get("dtype")
    device = kwargs.get("device")
    dtype = torch.float64 if dtype is None else dtype
    device = x.device if device is None else torch.device(device)
    return dtype, device


def _fill_value(v: Any) -> float:
    s = scalar_value(v)
    if s is None:
        s = factories._scalar(v)
    return s


def _components_like(
    comps: tuple[torch.Tensor, ...], kind: str, values: tuple[float, ...] | None
) -> tuple[torch.Tensor, ...]:
    """``kind``-like components with the same shape/strides as ``comps``.

    ``values`` supplies one fill per component (``(hi, lo)`` or ``(bits,)``);
    it is None for ``empty``.
    """
    if kind == "empty":
        return tuple(torch.empty_like(c) for c in comps)
    assert values is not None
    return tuple(torch.full_like(c, v) for c, v in zip(comps, values, strict=True))


def _created(
    kind: str, shape: tuple[int, ...], mode: str, value: float | None = None
) -> Any:
    """A new contiguous MetalFloat64 of ``shape``: host-resident when small."""
    if factories.host_resident(shape):
        if kind == "full":
            host = torch.full(shape, value, dtype=torch.float64)
        else:
            host = _HOST_KIND[kind](shape, dtype=torch.float64)
        return wrap_host(host, mode)
    if kind == "empty":
        return wrap(factories.empty_components(shape, mode), mode)
    fill = {"zeros": 0.0, "ones": 1.0}.get(kind, value)
    return wrap(factories.constant_components(shape, fill, mode), mode)


def _fill_components(mode: str, value: float) -> tuple[float, ...]:
    """Per-component fill values encoding ``value`` exactly."""
    from optiland.backend.torch_backend.metal import encode

    if mode == "df64":
        hi, lo = encode.df64_scalar(value)
        return (hi, lo)
    return (encode.sf64_scalar(value),)


def _plain(
    kind: str,
    shape: Any,
    dtype: torch.dtype,
    device: torch.device,
    value: float | None,
    stride: Any = None,
) -> torch.Tensor:
    """A plain (non-emulated) tensor for non-float64 dtypes or other devices."""
    if (
        dtype == torch.float64 and device.type == "mps"
    ):  # pragma: no cover - guarded by callers
        raise TypeError("float64 on mps must be a MetalFloat64")
    if kind == "empty":
        if stride is not None:
            return torch.empty_strided(shape, stride, dtype=dtype, device=device)
        return torch.empty(shape, dtype=dtype, device=device)
    if kind == "full":
        # Rounding a float64 constant to float32/half/int happens exactly as torch does.
        return torch.full(shape, value, dtype=dtype, device=device)
    return {"zeros": torch.zeros, "ones": torch.ones}[kind](
        shape, dtype=dtype, device=device
    )


# ---------------------------------------------------------------------------
# *_like
# ---------------------------------------------------------------------------
@implements(aten.empty_like.default, aten.zeros_like.default, aten.ones_like.default)
def _like(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    kind = func.overloadpacket.__name__.removesuffix("_like")
    dtype, device = _target(x, kwargs)
    if dtype != torch.float64 or device.type != "mps":
        return _plain(kind, tuple(x.shape), dtype, device, None)
    if x.is_host_resident or factories.host_resident(tuple(x.shape)):
        return _created(kind, tuple(x.shape), x.mode)
    if kind == "empty":
        return wrap(_components_like(x.components, "empty", None), x.mode)
    values = _fill_components(x.mode, 1.0 if kind == "ones" else 0.0)
    return wrap(_components_like(x.components, kind, values), x.mode)


@implements(aten.full_like.default)
def _full_like(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    value = _fill_value(args[1] if len(args) > 1 else kwargs["fill_value"])
    dtype, device = _target(x, kwargs)
    if dtype != torch.float64 or device.type != "mps":
        return _plain("full", tuple(x.shape), dtype, device, value)
    if x.is_host_resident or factories.host_resident(tuple(x.shape)):
        return _created("full", tuple(x.shape), x.mode, value)
    return wrap(
        _components_like(x.components, "full", _fill_components(x.mode, value)), x.mode
    )


# ---------------------------------------------------------------------------
# new_*
# ---------------------------------------------------------------------------
@implements(aten.new_empty.default, aten.new_zeros.default, aten.new_ones.default)
def _new(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    size = tuple(args[1]) if len(args) > 1 else tuple(kwargs["size"])
    kind = func.overloadpacket.__name__.removeprefix("new_")
    dtype, device = _target(x, kwargs)
    if dtype != torch.float64 or device.type != "mps":
        return _plain(kind, size, dtype, device, None)
    return _created(kind, size, x.mode)


@implements(aten.new_full.default)
def _new_full(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    size = tuple(args[1]) if len(args) > 1 else tuple(kwargs["size"])
    value = _fill_value(args[2] if len(args) > 2 else kwargs["fill_value"])
    dtype, device = _target(x, kwargs)
    if dtype != torch.float64 or device.type != "mps":
        return _plain("full", size, dtype, device, value)
    return _created("full", size, x.mode, value)


@implements(aten.new_empty_strided.default)
def _new_empty_strided(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    size = tuple(args[1]) if len(args) > 1 else tuple(kwargs["size"])
    stride = tuple(args[2]) if len(args) > 2 else tuple(kwargs["stride"])
    dtype, device = _target(x, kwargs)
    if dtype != torch.float64 or device.type != "mps":
        return _plain("empty", size, dtype, device, None, stride=stride)
    if factories.host_resident(size):
        return wrap_host(torch.empty_strided(size, stride, dtype=torch.float64), x.mode)
    n = 2 if x.mode == "df64" else 1
    cdtype = torch.float32 if x.mode == "df64" else torch.int64
    comps = tuple(
        torch.empty_strided(size, stride, dtype=cdtype, device="mps") for _ in range(n)
    )
    return wrap(comps, x.mode)


# ---------------------------------------------------------------------------
# constant padding (the pad value is a constant that must be encoded exactly)
# ---------------------------------------------------------------------------
@implements(aten.constant_pad_nd.default)
def _constant_pad_nd(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    pad = list(args[1] if len(args) > 1 else kwargs["pad"])
    value = _fill_value(args[2] if len(args) > 2 else kwargs.get("value", 0))
    values = _fill_components(x.mode, value)
    comps = tuple(
        torch.constant_pad_nd(c, pad, v)
        for c, v in zip(x.components, values, strict=True)
    )
    return wrap(comps, x.mode)


# ---------------------------------------------------------------------------
# identity-ish
# ---------------------------------------------------------------------------
@implements(aten.lift_fresh.default)
def _lift_fresh(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return args[0]


@implements(aten.lift_fresh_copy.default)
def _lift_fresh_copy(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return apply_to_components(func, args, kwargs)


# ---------------------------------------------------------------------------
# plain factories (rarely dispatched; see module docstring)
# ---------------------------------------------------------------------------
def _decode_args(args: Any, kwargs: Any) -> tuple[Any, Any]:
    """Replace MetalFloat64 leaves by host values so torch factories accept them."""
    from torch.utils._pytree import tree_map

    def d(a: Any) -> Any:
        if is_metal(a):
            return float(a.item()) if a.dim() == 0 else a.to_cpu_float64()
        return a

    return tree_map(d, args), tree_map(d, kwargs)


def _wants_emulation(kwargs: Any) -> bool:
    dtype = kwargs.get("dtype")
    device = kwargs.get("device")
    return (
        dtype == torch.float64
        and device is not None
        and torch.device(device).type == "mps"
    )


def _factory(builder: Any) -> Any:
    def handler(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        args, kwargs = _decode_args(args, kwargs)
        if _wants_emulation(kwargs):
            return builder(func, args, kwargs)
        return func(*args, **kwargs)

    return handler


def _fv(args: Any, kwargs: Any, index: int, name: str) -> float:
    return _fill_value(args[index] if len(args) > index else kwargs[name])


implements(aten.empty.memory_format)(
    _factory(lambda f, a, k: factories.empty(a[0] if a else k["size"]))
)


def _empty_strided(size: Any, stride: Any) -> Any:
    mode = factories.get_mode()
    cdtype = torch.float32 if mode == "df64" else torch.int64
    n = 2 if mode == "df64" else 1
    comps = tuple(
        torch.empty_strided(tuple(size), tuple(stride), dtype=cdtype, device="mps")
        for _ in range(n)
    )
    return wrap(comps, mode)


implements(aten.empty_strided.default)(
    _factory(
        lambda f, a, k: _empty_strided(
            a[0] if a else k["size"], a[1] if len(a) > 1 else k["stride"]
        )
    )
)
implements(aten.zeros.default)(
    _factory(lambda f, a, k: factories.zeros(a[0] if a else k["size"]))
)
implements(aten.ones.default)(
    _factory(lambda f, a, k: factories.ones(a[0] if a else k["size"]))
)
implements(aten.full.default)(
    _factory(
        lambda f, a, k: factories.full(
            a[0] if a else k["size"], _fv(a, k, 1, "fill_value")
        )
    )
)
implements(aten.scalar_tensor.default)(
    _factory(lambda f, a, k: factories.scalar(_fv(a, k, 0, "s")))
)
implements(aten.arange.default)(
    _factory(lambda f, a, k: factories.arange(_fv(a, k, 0, "end")))
)
implements(aten.arange.start)(
    _factory(
        lambda f, a, k: factories.arange(_fv(a, k, 0, "start"), _fv(a, k, 1, "end"))
    )
)
implements(aten.arange.start_step)(
    _factory(
        lambda f, a, k: factories.arange(
            _fv(a, k, 0, "start"), _fv(a, k, 1, "end"), _fv(a, k, 2, "step")
        )
    )
)
implements(aten.linspace.default)(
    _factory(
        lambda f, a, k: factories.linspace(
            _fv(a, k, 0, "start"),
            _fv(a, k, 1, "end"),
            int(a[2] if len(a) > 2 else k["steps"]),
        )
    )
)
implements(aten.eye.default)(
    _factory(lambda f, a, k: factories.eye(int(a[0] if a else k["n"])))
)
implements(aten.eye.m)(
    _factory(
        lambda f, a, k: factories.eye(
            int(a[0] if a else k["n"]), int(a[1] if len(a) > 1 else k["m"])
        )
    )
)
