"""``MetalFloat64``: emulated float64 tensors on Apple GPUs (torch MPS).

A ``MetalFloat64`` is a ``torch.Tensor`` wrapper subclass that reports
``dtype=torch.float64`` and ``device='mps'`` while storing its values as either

* ``df64``: two float32 component tensors ``(hi, lo)`` (double-single, ~48-bit
  significand, float32 exponent range), the default, or
* ``sf64``: one int64 component tensor holding IEEE binary64 bit patterns
  (software float, exact, slower).

Arithmetic reaches Metal kernels (``library.MetalLibrary``) through
``__torch_dispatch__``; structural aten ops (views, indexing, concatenation) are
applied to the component tensors unchanged; anything not implemented on the GPU
either falls back to CPU float64 explicitly (counted in :func:`stats`) or raises
``NotImplementedError`` when ``OPTILAND_METAL_STRICT=1``.

Autograd needs no custom backward: the engine records the aten op above this
dispatch layer and its derivative formula (itself aten ops) lands here again.

Handler modules register implementations with :func:`implements`; see
``ops_structural.py``, ``ops_elementwise.py``, ``ops_reduce.py``, ``ops_linalg.py``
and ``ops_creation.py`` (imported at the bottom of this module).

Representation mixing: GPU-resident tensors fix their representation and two
different ones in one op raise ``TypeError``. Two kinds of tensor are
*representation-agnostic* and are re-tagged to the fixed operand's mode when
they meet one: host-resident tensors (their truth is a CPU float64 tensor) and
tensors created by :class:`MetalFactoryMode`, i.e. the tensor-less float64
factories (``zeros``, ``scalar_tensor``, ``_efficientzerotensor``, ...) that
C++ autograd formulas emit in the *global* mode, together with everything
derived from them alone (views such as ``zeros(()).expand(sizes)``, arithmetic
on the constant, ``plain_mps.to(float64)``: :func:`mark_derived` flags every
handler result that no fixed operand produced). They are re-encoded from
their current value (exact for the constants those formulas build) when they
meet a fixed tensor; a constant stops being agnostic once an in-place op has
written data of a fixed representation into it. So ``where``, ``index_put``,
``put``, the ``*_scatter`` family, ``trace``, ``quantile`` and partially used
``split``/``unbind``/``chunk`` outputs differentiate correctly whatever
``metal.get_mode()`` is.
"""

from __future__ import annotations

import os
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from torch.utils._pytree import tree_map

from optiland.backend.torch_backend.metal import encode
from optiland.backend.torch_backend.metal.library import MetalLibrary, get_library

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

aten = torch.ops.aten

MODES = ("df64", "sf64")
DEFAULT_MODE = "df64"
MACHINE_EPS = {"df64": 2.0**-48, "sf64": 2.0**-53}
MPS0 = torch.device("mps", 0)

# Dual residency: tensors with at most this many elements are created host-resident
# (a CPU float64 tensor) and ops among them run on the CPU exactly, with no kernel
# launch and no device sync. See NOTES/02-design.md section 4. The default is 256
# (the design's value, measured in section 4.1); the OPTILAND_METAL_HOST_THRESHOLD
# environment variable overrides it at import and ``set_host_threshold`` at run
# time. Zero disables the host path (every tensor GPU-resident).
DEFAULT_HOST_THRESHOLD = 256
HOST_THRESHOLD = int(
    os.environ.get("OPTILAND_METAL_HOST_THRESHOLD", str(DEFAULT_HOST_THRESHOLD))
)


def set_host_threshold(n: int) -> None:
    """Set the element count below which tensors are host-resident (0 disables)."""
    global HOST_THRESHOLD
    HOST_THRESHOLD = int(n)


def get_host_threshold() -> int:
    """Return the current host-residency threshold."""
    return HOST_THRESHOLD


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------
_STATS: Counter[str] = Counter()


def stats() -> dict[str, int]:
    """Return a copy of the per-op counters.

    Keys: ``gpu:<kernel>`` (Metal launches), ``host:<op>`` (dual-residency CPU
    path), ``cpu_fallback:<op>`` (decode / CPU float64 / re-encode) and
    ``cpu_complex:<op>`` (ops with a complex operand, which live on the CPU).
    """
    return dict(_STATS)


def reset_stats() -> None:
    """Zero the counters."""
    _STATS.clear()


def strict_mode() -> bool:
    """True when CPU fallbacks must raise (``OPTILAND_METAL_STRICT=1``)."""
    return os.environ.get("OPTILAND_METAL_STRICT", "0") == "1"


class MetalFallbackError(NotImplementedError):
    """Raised in strict mode when an op would fall back to the CPU."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_HANDLERS: dict[Any, Callable[..., Any]] = {}


def implements(*ops: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register ``fn`` as the ``__torch_dispatch__`` handler for the given aten ops.

    ``ops`` may be ``OpOverload`` objects (``aten.add.Tensor``) or
    ``OpOverloadPacket`` objects (``aten.add``); an exact overload match wins
    over a packet match. Handlers are called as ``fn(func, types, args, kwargs)``.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        for op in ops:
            _HANDLERS[op] = fn
        return fn

    return decorator


def handler_for(func: Any) -> Callable[..., Any] | None:
    """Look up the registered handler for an ``OpOverload``."""
    h = _HANDLERS.get(func)
    if h is None:
        h = _HANDLERS.get(getattr(func, "overloadpacket", None))
    return h


class _HostCell:
    """GPU-encoding cache shared by a host-resident tensor and all its host views.

    Plain-tensor view ops executed inside ``__torch_dispatch__`` run below the
    autograd keys, so they carry no ``_base``/version-counter bookkeeping; the
    cell is the explicit alias group used to invalidate the encoding when any
    alias is mutated on the host.

    ``comps`` caches the encoding of the alias group's base tensor; ``views``
    caches encodings of host views by their geometry over the base storage
    (storage address, sizes, strides, offset), so a view rebuilt every trace
    (``coeffs[i]``, ``points.unsqueeze(0)``) is copied to the GPU once, not
    once per use. Both are dropped together on any host-side mutation.
    """

    __slots__ = ("comps", "views")

    #: Largest number of distinct view encodings kept per alias group.
    MAX_VIEWS = 64

    def __init__(self) -> None:
        self.comps: tuple[torch.Tensor, ...] | None = None
        self.views: dict[tuple[Any, ...], tuple[torch.Tensor, ...]] | None = None

    def clear(self) -> None:
        self.comps = None
        self.views = None


# ---------------------------------------------------------------------------
# The tensor subclass
# ---------------------------------------------------------------------------
class MetalFloat64(torch.Tensor):
    """Emulated float64 tensor on the Apple GPU (see module docstring).

    Residency: ``_host`` (CPU float64 tensor) is set for host-resident tensors;
    ``_comps_cache`` holds the GPU component tensors, which for host-resident
    tensors is a lazily built encoding (validated against ``_host._version``).
    """

    _comps_cache: tuple[torch.Tensor, ...] | None
    _host: torch.Tensor | None
    _cell: _HostCell | None
    _mode: str
    _agnostic: bool

    @staticmethod
    @torch._dynamo.disable
    def __new__(  # noqa: PYI034 - wrapper subclass construction
        cls,
        comps: Sequence[torch.Tensor] | None = None,
        mode: str = DEFAULT_MODE,
        requires_grad: bool = False,
        host: torch.Tensor | None = None,
        cell: _HostCell | None = None,
    ) -> MetalFloat64:
        if host is not None:
            if host.dtype != torch.float64 or host.device.type != "cpu":
                raise TypeError("host tensor must be a CPU float64 tensor")
            if isinstance(host, MetalFloat64):
                raise TypeError("host tensor must be a plain tensor")
            base = host
            comps_t = tuple(comps) if comps is not None else None
        else:
            if comps is None:
                raise TypeError("either comps or host is required")
            comps_t = tuple(comps)
            _validate_components(comps_t, mode)
            base = comps_t[0]
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            base.shape,
            strides=base.stride(),
            storage_offset=base.storage_offset(),
            dtype=torch.float64,
            device=MPS0,
            layout=base.layout,
            requires_grad=requires_grad,
        )
        r._host = host
        r._comps_cache = comps_t
        r._cell = (cell or _HostCell()) if host is not None else None
        r._mode = mode
        r._agnostic = False
        if _FACTORY_MODE is None and _AUTO_FACTORY_INTERCEPT:
            enable_factory_intercept()
        return r

    @property
    def _comps(self) -> tuple[torch.Tensor, ...]:
        """GPU component tensors, materialized (and cached) from the host copy."""
        if self._host is None:
            assert self._comps_cache is not None
            return self._comps_cache
        cell = self._cell
        assert cell is not None
        host = self._host
        # A view's cell is its base's; ``cell.comps`` holds the *base* encoding,
        # views are cached by their geometry over the shared storage.
        if host._base is None and not _is_host_view(self):
            if cell.comps is None:
                cell.comps = _encode_host(host, self._mode)
            return cell.comps
        key = (
            host.untyped_storage().data_ptr(),
            tuple(host.shape),
            tuple(host.stride()),
            host.storage_offset(),
            self._mode,
        )
        views = cell.views
        if views is None or len(views) >= _HostCell.MAX_VIEWS:
            views = cell.views = {}
        comps = views.get(key)
        if comps is None:
            comps = views[key] = _encode_host(host, self._mode)
        return comps

    @_comps.setter
    def _comps(self, value: tuple[torch.Tensor, ...]) -> None:
        self._comps_cache = tuple(value)

    def invalidate_host_cache(self) -> None:
        """Drop the cached GPU encodings after a host-side mutation of any alias."""
        if self._cell is not None:
            self._cell.clear()

    @property
    def is_host_resident(self) -> bool:
        """True when the authoritative value lives in a CPU float64 tensor."""
        return self._host is not None

    # -- torch integration -------------------------------------------------
    @classmethod
    def __torch_function__(
        cls, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        # Nearly everything goes straight to the aten dispatch below. The exceptions
        # are torch-level composites that materialize Python scalars as float64
        # tensors on the wrapper's device (mps), which MPS rejects before our
        # __torch_dispatch__ can run; those get their scalars wrapped first.
        kwargs = kwargs or {}
        fixer = _TORCH_FUNCTION_FIXES.get(func)
        if fixer is not None:
            args, kwargs = fixer(args, kwargs)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(
        cls, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        kwargs = kwargs or {}
        leaves = _flatten(args) + (_flatten(kwargs) if kwargs else [])
        absorbing = _prepare_writes(func, args, kwargs, leaves)
        if _has_complex(args):
            # Complex is never emulated: counted CPU path, whatever the residency.
            result = complex_on_cpu(func, args, kwargs)
        elif (
            HOST_THRESHOLD > 0
            and func not in _HOST_EXCLUDED
            and _host_eligible(func, leaves)
        ):
            result = _run_on_host(func, args, kwargs, leaves)
        else:
            h = handler_for(func)
            if h is None:
                return unimplemented(func, args, kwargs)
            result = mark_derived(h(func, types, args, kwargs), args, kwargs)
        if absorbing:
            # A constant that absorbed data of a fixed representation (it was
            # re-tagged to that mode before the write) is no longer agnostic.
            for t in absorbing:
                _end_agnostic(t)
        return result

    def __tensor_flatten__(self) -> tuple[list[str], Any]:
        return ["_comps"], (self._mode, self.requires_grad)

    @staticmethod
    def __tensor_unflatten__(
        inner: dict[str, Any], meta: Any, outer_size: Any, outer_stride: Any
    ) -> MetalFloat64:
        mode, requires_grad = meta
        return MetalFloat64(inner["_comps"], mode, requires_grad=requires_grad)

    def _stable_hash_for_caching(self) -> str:
        return f"MetalFloat64:{self._mode}:{self.requires_grad}:{tuple(self.shape)}"

    # -- properties --------------------------------------------------------
    @property
    def mode(self) -> str:
        """Representation: ``'df64'`` or ``'sf64'``."""
        return self._mode

    @property
    def components(self) -> tuple[torch.Tensor, ...]:
        """The underlying plain MPS tensors (``(hi, lo)`` or ``(bits,)``)."""
        return self._comps

    @property
    def machine_eps(self) -> float:
        """Relative precision actually carried by this representation."""
        return MACHINE_EPS[self._mode]

    # -- host conversion ---------------------------------------------------
    @classmethod
    def from_numpy(
        cls,
        a: Any,
        mode: str = DEFAULT_MODE,
        device: str = "mps",
        requires_grad: bool = False,
        host: bool | None = None,
    ) -> MetalFloat64:
        """Encode a float64 array (any shape, incl. 0-d) into a ``MetalFloat64``.

        Args:
            a: Array-like data.
            mode: Representation.
            device: Target device for GPU components.
            requires_grad: Autograd flag of the wrapper.
            host: Force host residency (True), GPU residency (False) or decide
                by size (None: host when ``a.size <= HOST_THRESHOLD``).
        """
        a = np.array(a, dtype=np.float64, copy=True, order="C")
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if host is None:
            host = a.size <= HOST_THRESHOLD
        if host:
            return cls(
                None, mode, requires_grad=requires_grad, host=torch.from_numpy(a)
            )
        return cls(_encode_array(a, mode, device), mode, requires_grad=requires_grad)

    def to_numpy(self) -> np.ndarray:
        """Decode to a float64 NumPy array (host copy)."""
        if self._host is not None:
            return self._host.detach().numpy().copy()
        comps = [c.detach().contiguous().cpu().numpy() for c in self._comps_cache or ()]
        if self._mode == "df64":
            out = np.asarray(encode.decode_df64(comps[0], comps[1]), dtype=np.float64)
        else:
            out = np.asarray(encode.decode_sf64(comps[0]), dtype=np.float64)
        return out.reshape(tuple(self.shape))

    def to_cpu_float64(self) -> torch.Tensor:
        """Decode to a plain CPU float64 tensor (a view of the host copy if any)."""
        if self._host is not None:
            return self._host.detach()
        return torch.from_numpy(
            np.ascontiguousarray(self.to_numpy()).reshape(tuple(self.shape))
        )

    def __setitem__(self, index: Any, value: Any) -> None:  # noqa: D105
        # torch's C++ indexing materializes Python scalars as float64 tensors on
        # the wrapper's device (mps), which MPS rejects; wrap them first so the
        # assignment reaches our dispatch handlers as a tensor value.
        # A plain tensor value is left to torch's indexing (``copy_`` /
        # ``index_put_``), whose handlers promote it *below* autograd, so a
        # value that requires grad keeps its graph (coercing it here would
        # detach it silently).
        if not isinstance(value, torch.Tensor):
            value = MetalFloat64.from_numpy(
                np.asarray(value, dtype=np.float64), self._mode
            )
        torch.Tensor.__setitem__(self, index, value)

    # -- Python protocol overrides the stock subclass paths reject --------
    def __repr__(self, *, tensor_contents: Any = None) -> str:  # noqa: D105
        # Values are printed the way torch prints a float64 tensor (print
        # options, 4 decimals by default), so the repr is as stable as the
        # torch CPU repr under sub-repr parameter changes.
        grad = ", requires_grad=True" if self.requires_grad else ""
        values = torch._tensor_str._tensor_str(self.to_cpu_float64(), 13)
        return f"MetalFloat64({values}, mode={self._mode!r}{grad})"

    def __format__(self, spec: str) -> str:  # noqa: D105
        if self.dim() == 0:
            return format(float(self.to_numpy()), spec)
        return object.__format__(self, spec)

    def tolist(self) -> Any:  # noqa: D102
        return self.to_numpy().tolist()

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:  # noqa: D105
        a = self.to_numpy()
        return a.astype(dtype) if dtype is not None else a

    def numpy(self, *, force: bool = False) -> np.ndarray:  # noqa: D102
        if not force:
            raise TypeError(
                "can't convert a MetalFloat64 (mps) tensor to numpy; "
                "use .cpu().numpy() or .to_numpy()"
            )
        return self.to_numpy()


# ---------------------------------------------------------------------------
# Helpers for handler modules
# ---------------------------------------------------------------------------
def is_metal(x: Any) -> bool:
    """True for ``MetalFloat64`` instances."""
    return isinstance(x, MetalFloat64)


def _validate_components(comps: tuple[torch.Tensor, ...], mode: str) -> None:
    if mode == "df64":
        if len(comps) != 2 or any(c.dtype != torch.float32 for c in comps):
            raise TypeError("df64 needs two float32 component tensors (hi, lo)")
    elif mode == "sf64":
        if len(comps) != 1 or comps[0].dtype != torch.int64:
            raise TypeError("sf64 needs one int64 component tensor")
    else:
        raise ValueError(f"unknown mode {mode!r}")
    base = comps[0]
    for c in comps[1:]:
        if (
            c.shape != base.shape
            or c.stride() != base.stride()
            or c.storage_offset() != base.storage_offset()
        ):
            raise ValueError(
                "component tensors must share shape, strides and storage offset"
            )
    if base.device.type != "mps":
        raise ValueError(f"components must live on mps, got {base.device}")
    if any(isinstance(c, MetalFloat64) for c in comps):
        raise TypeError("components must be plain tensors")


def wrap(
    comps: Sequence[torch.Tensor], mode: str, requires_grad: bool = False
) -> MetalFloat64:
    """Wrap plain component tensors into a ``MetalFloat64``."""
    return MetalFloat64(tuple(comps), mode, requires_grad=requires_grad)


def wrap_host(
    host: torch.Tensor,
    mode: str,
    requires_grad: bool = False,
    cell: _HostCell | None = None,
    is_view: bool = False,
) -> MetalFloat64:
    """Wrap a CPU float64 tensor into a host-resident ``MetalFloat64``.

    ``cell`` links a view to its base's alias group; ``is_view`` marks that the
    wrapper aliases another wrapper's host tensor.
    """
    r = MetalFloat64(None, mode, requires_grad=requires_grad, host=host, cell=cell)
    r._host_is_view = is_view
    return r


def _is_host_view(x: MetalFloat64) -> bool:
    return bool(getattr(x, "_host_is_view", False))


def _returns_alias(func: Any) -> bool:
    schema = getattr(func, "_schema", None)
    if schema is None:
        return False
    return any(r.alias_info is not None for r in schema.returns)


def _device_component(arr: np.ndarray, shape: tuple[int, ...], device: Any) -> Any:
    """A contiguous device tensor holding ``arr`` reshaped to ``shape``.

    ``np.ascontiguousarray`` would promote 0-d arrays to shape ``(1,)``, hence
    the reshape.
    """
    return torch.from_numpy(np.ascontiguousarray(arr).reshape(shape)).to(device)


def _encode_array(
    a: np.ndarray, mode: str, device: Any = MPS0
) -> tuple[torch.Tensor, ...]:
    """Encode a C-contiguous float64 array into device components of ``mode``."""
    shape = tuple(a.shape)
    if mode == "df64":
        hi, lo = encode.encode_df64(a)
        return (
            _device_component(hi, shape, device),
            _device_component(lo, shape, device),
        )
    if mode == "sf64":
        return (_device_component(encode.encode_sf64(a), shape, device),)
    raise ValueError(f"unknown mode {mode!r}")


def _encode_host(host: torch.Tensor, mode: str) -> tuple[torch.Tensor, ...]:
    if host.numel() == 0:
        # Keep the host tensor's strides (a host view such as ``e.view(0)`` or
        # ``e.t()`` of an empty tensor); numpy would report zero strides.
        dtype = torch.float32 if mode == "df64" else torch.int64
        n = 2 if mode == "df64" else 1
        return tuple(
            torch.empty_strided(
                tuple(host.shape), tuple(host.stride()), dtype=dtype, device=MPS0
            )
            for _ in range(n)
        )
    return _encode_array(host.detach().contiguous().numpy(), mode)


# Ops with dedicated handlers that understand residency themselves.
_HOST_EXCLUDED: set[Any] = {
    aten._to_copy.default,
    aten.copy_.default,
    aten._local_scalar_dense.default,
}


def _is_mutating(func: Any) -> bool:
    schema = getattr(func, "_schema", None)
    if schema is None:
        return False
    return any(
        a.alias_info is not None and a.alias_info.is_write for a in schema.arguments
    )


_MUTATES_FIRST: dict[Any, bool] = {}


def _mutates_first(func: Any) -> bool:
    """True when ``func`` writes its first argument (cached per overload)."""
    m = _MUTATES_FIRST.get(func)
    if m is None:
        schema = getattr(func, "_schema", None)
        info = schema.arguments[0].alias_info if schema and schema.arguments else None
        m = _MUTATES_FIRST[func] = bool(info is not None and info.is_write)
    return m


_WRITE_ARGS: dict[Any, tuple[tuple[int, str], ...]] = {}
_RETURNS_ALIAS: dict[Any, bool] = {}


def _write_targets(func: Any, args: Any, kwargs: Any) -> list[Any]:
    """The operands ``func`` writes (``self`` of in-place ops, ``out=`` tensors)."""
    info = _WRITE_ARGS.get(func)
    if info is None:
        schema = getattr(func, "_schema", None)
        info = _WRITE_ARGS[func] = tuple(
            (i, a.name)
            for i, a in enumerate(schema.arguments if schema else ())
            if a.alias_info is not None and a.alias_info.is_write
        )
    if not info:
        return []
    targets = []
    for i, name in info:
        if i < len(args):
            targets.append(args[i])
        elif name in kwargs:
            targets.append(kwargs[name])
    return targets


def _returns_alias_cached(func: Any) -> bool:
    r = _RETURNS_ALIAS.get(func)
    if r is None:
        r = _RETURNS_ALIAS[func] = _returns_alias(func)
    return r


def _prepare_writes(
    func: Any, args: Any, kwargs: Any, leaves: list[Any]
) -> list[MetalFloat64]:
    """Ready the written operands of ``func`` before its handler runs.

    * A lazily expanded host result (:func:`materialize_host`) gets its own
      storage before it is written or viewed, so the write lands where torch's
      would and a view aliases real memory.
    * A representation-agnostic target (a factory constant, module docstring)
      that is about to absorb data of a fixed representation is re-tagged to
      that representation first, value-preservingly, so the handler sees one
      representation; the tensors returned stop being agnostic once the write
      has succeeded (a failed write leaves the tagging as it was).
    """
    targets = _write_targets(func, args, kwargs)
    if not targets:
        if (
            args
            and getattr(args[0], "_host_lazy", False)
            and _returns_alias_cached(func)
        ):
            materialize_host(args[0])
        return []
    absorbing: list[MetalFloat64] = []
    fixed_mode: str | None = None
    for t in targets:
        if not isinstance(t, MetalFloat64):
            continue
        if getattr(t, "_host_lazy", False):
            materialize_host(t)
        if not t._agnostic:
            continue
        if fixed_mode is None:
            fixed_mode = next(
                (a._mode for a in leaves if is_metal(a) and _is_fixed(a)), None
            )
            if fixed_mode is None:
                return []
        if t._mode != fixed_mode:
            _retag(t, fixed_mode)
        absorbing.append(t)
    return absorbing


def _shares_storage(a: MetalFloat64, b: MetalFloat64) -> bool:
    ca, cb = a._comps_cache, b._comps_cache
    return (
        ca is not None
        and cb is not None
        and ca[0].untyped_storage().data_ptr() == cb[0].untyped_storage().data_ptr()
    )


def _end_agnostic(x: MetalFloat64) -> None:
    """``x`` (and the base it is a view of) now holds data of a fixed representation."""
    x._agnostic = False
    if x._is_view():
        base = x._base
        if (
            is_metal(base)
            and base._agnostic
            and base._host is None
            and _shares_storage(x, base)
        ):
            base._agnostic = False


def materialize_host(x: MetalFloat64) -> None:
    """Give a lazily expanded host result its own contiguous storage (in place).

    ``ops_elementwise`` evaluates an op on two scalar-like operands once on the
    host and returns the value as a stride-0 host-resident expansion so that it
    stays a kernel scalar for the next op (``_host_lazy``). torch returns a
    fresh contiguous tensor for such an op, so before the result is written
    (in place, ``out=``, ``__setitem__``) or viewed it is materialized here:
    the wrapper reports contiguous strides from then on and a write reaches
    exactly the elements torch's would.
    """
    if not getattr(x, "_host_lazy", False):
        return
    host = x._host
    assert host is not None
    x._host = host.clone(memory_format=torch.contiguous_format)
    x._host_lazy = False
    x.invalidate_host_cache()
    _set_wrapper_geometry(x, x._host)


def _host_eligible(func: Any, leaves: list[Any]) -> bool:
    """Decide whether ``func`` runs on the CPU in float64 (see module docstring)."""
    saw_metal = False
    for a in leaves:
        if isinstance(a, MetalFloat64):
            saw_metal = True
            if a.numel() > HOST_THRESHOLD:
                return False
        elif isinstance(a, torch.Tensor):
            if a.device.type == "mps" and a.numel() > HOST_THRESHOLD:
                return False
    if not saw_metal:
        return False
    if _returns_alias(func):
        # A view must alias its base: views of GPU-resident tensors stay on the GPU.
        for a in leaves:
            if isinstance(a, MetalFloat64) and a._host is None:
                return False
    if _is_mutating(func):
        # The mutated operand is the first argument; it must be host-resident so
        # that GPU aliases (views of large tensors) are never bypassed.
        target = leaves[0]
        if not (isinstance(target, MetalFloat64) and target._host is not None):
            return False
    return True


def _run_on_host(
    func: Any, args: Any, kwargs: Any, leaves: list[Any] | None = None
) -> Any:
    if leaves is None:
        leaves = _flatten(args) + (_flatten(kwargs) if kwargs else [])
    metal = [a for a in leaves if isinstance(a, MetalFloat64)]
    mode = metal[0]._mode

    def to_host(a: Any) -> Any:
        if isinstance(a, MetalFloat64):
            if a._host is not None:
                return a._host
            return a.to_cpu_float64()  # small GPU-resident operand: one decode
        if isinstance(a, torch.Tensor) and a.device.type == "mps":
            return a.cpu()
        if isinstance(a, torch.device) and a.type == "mps":
            return torch.device("cpu")
        return a

    h_args, h_kwargs = _map_args(to_host, args, kwargs)
    out = func(*h_args, **h_kwargs)
    _STATS["host:" + _op_name(func)] += 1
    mutating = _is_mutating(func)
    aliasing = _returns_alias(func)
    base = metal[0]
    # ``zeros_like(x, device="cpu")``: an explicit non-mps device asks for a
    # plain tensor there, not for an emulated one.
    requested = kwargs.get("device") if kwargs else None
    plain = requested is not None and torch.device(requested).type != "mps"

    def back(o: Any) -> Any:
        if plain:
            return o
        if isinstance(o, torch.Tensor) and not isinstance(o, MetalFloat64):
            if o.dtype == torch.float64:
                if aliasing and base._host is not None:
                    return wrap_host(o, mode, cell=base._cell, is_view=True)
                return wrap_host(o, mode)
            if o.dtype.is_floating_point or o.dtype.is_complex:
                # A native-dtype result (``zeros_like(x, dtype=float32)``) lives
                # where a plain torch program would put it: on the device.
                return o.to(MPS0)
            return o  # bool / int64 results stay on the CPU (see design note)
        return o

    wrapped = tree_map(back, out)
    if mutating:
        for a in metal:
            a.invalidate_host_cache()
    return return_and_correct_aliasing(func, args, kwargs, wrapped)


def _is_fixed(x: MetalFloat64) -> bool:
    return x._host is None and not x._agnostic


def _retag(x: MetalFloat64, mode: str) -> None:
    """Re-tag a representation-agnostic tensor to ``mode`` (see module docstring)."""
    if x._host is not None:
        x._mode = mode
        x.invalidate_host_cache()
        return
    base = x._base if x._is_view() else None
    if (
        is_metal(base)
        and base is not x
        and base._host is None
        and base._comps_cache is not None
        and base._comps_cache[0].is_contiguous()
        and base._comps_cache[0].storage_offset() == 0
    ):
        # A view of a factory constant (``buf[0]`` in ``buf[0] = a[0]``): the
        # base is re-encoded (same geometry, it is contiguous) and the view is
        # re-laid over the new components, so writes through it reach the
        # base. A view left behind by an earlier re-tag of its base (its
        # components alias the old encoding) is re-attached the same way once
        # the base is in the requested representation.
        if base._mode != mode:
            if not (base._agnostic and _shares_storage(x, base)):
                raise TypeError(
                    f"cannot mix representations {base._mode} and {mode} (view of "
                    "a tensor that already holds the other representation)"
                )
            _retag(base, mode)
        x._comps_cache = tuple(
            c.as_strided(tuple(x.shape), tuple(x.stride()), x.storage_offset())
            for c in base._comps_cache
        )
        x._mode = mode
        return
    fresh = MetalFloat64.from_numpy(x.to_numpy(), mode, host=False)
    x._comps_cache = fresh._comps_cache
    x._mode = mode
    if not x.is_contiguous():
        # A view of a factory constant (``zeros(()).expand(sizes)`` in
        # unbind_backward): the re-encoding is contiguous and owns its memory,
        # so the wrapper geometry follows it (the constant's base is untouched).
        _set_wrapper_geometry(x, x._comps_cache[0])


def mode_of(*xs: Any) -> str:
    """Representation shared by the ``MetalFloat64`` operands.

    GPU-resident operands fix the representation (mixing two is an error);
    host-resident and factory-made (``_agnostic``) operands are
    representation-agnostic and are re-tagged to match (module docstring).
    """
    modes = {x._mode for x in xs if is_metal(x) and _is_fixed(x)}
    if len(modes) > 1:
        raise TypeError(f"cannot mix representations {sorted(modes)}")
    if modes:
        mode = modes.pop()
    else:
        mode = next((x._mode for x in xs if is_metal(x)), DEFAULT_MODE)
    for x in xs:
        if is_metal(x) and x._mode != mode:
            _retag(x, mode)
    return mode


def mark_derived(result: Any, args: Any, kwargs: Any = None) -> Any:
    """Flag the GPU-resident results representation-agnostic when no operand fixed it.

    Applied to every handler result by ``__torch_dispatch__``: a value computed
    only from factory constants, host-resident tensors, scalars and plain
    tensors (``ranks - ranks_below`` in quantile, ``zeros(()).expand(sizes)``
    in unbind_backward, ``1 - w`` in lerp) carries no representation of its
    own, so a later fixed operand of the other representation re-encodes it
    (exactly, from its value) instead of raising the mixing error.
    """
    leaves = _flatten(args) + (_flatten(kwargs) if kwargs else [])
    if any(is_metal(a) and _is_fixed(a) for a in leaves):
        return result
    outs = result if isinstance(result, (list, tuple)) else (result,)
    for o in outs:
        if is_metal(o) and o._host is None:
            o._agnostic = True
    return result


def library(mode: str) -> MetalLibrary:
    """The compiled kernel library for ``mode``."""
    return get_library(mode)


def scalar_value(x: Any) -> float | None:
    """Return ``x`` as a Python float if it is a scalar operand, else None.

    Scalar operands reach dispatch as Python numbers, as wrapped 0-d CPU
    tensors, or as 0-d *host-resident* ``MetalFloat64`` tensors (surface
    radii, thicknesses, indices, ...); all are read exactly (never through
    float32) and without any device sync, so the elementwise bridges can pass
    them to the ``scalar=`` kernel variants instead of encoding and copying
    them to the GPU. A 0-d GPU-resident tensor is not a scalar here (reading
    it would sync).
    """
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, torch.Tensor) and x.dim() == 0:
        if is_metal(x):
            host = x._host
            return None if host is None else float(host.item())
        if x.device.type == "cpu":
            return float(x.item())
    return None


def scalar_like(x: Any, shape: Sequence[int]) -> float | None:
    """Return ``x`` as a Python float when it acts as a scalar for a result ``shape``.

    Extends :func:`scalar_value` to *single-valued* host-resident tensors: one
    element of any rank (a ``(1,)`` coefficient slice) or a stride-0 expansion
    of one element (``n.reshape(1).expand(rays)``), provided broadcasting the
    tensor against ``shape`` leaves ``shape`` unchanged, so the value can go
    to a scalar kernel variant without changing the result's shape.
    """
    s = scalar_value(x)
    if s is not None or not is_metal(x):
        return s
    host = x._host
    if host is None:
        return None
    n = host.numel()
    if n == 0:
        return None
    if n > 1 and any(
        st != 0 for sz, st in zip(host.shape, host.stride(), strict=True) if sz > 1
    ):
        return None
    shape = tuple(shape)
    if x.dim() > 0 and tuple(torch.broadcast_shapes(tuple(x.shape), shape)) != shape:
        return None
    return float(host[(0,) * host.dim()].item())


def coerce(
    x: Any, mode: str, device: torch.device | str = "mps", allow_cpu: bool = False
) -> MetalFloat64:
    """Convert a scalar / plain tensor / MetalFloat64 operand to a ``MetalFloat64``.

    Plain floating tensors are promoted exactly (float32 -> hi with lo = 0 for
    df64, exact conversion for sf64); integer and bool tensors likewise (int32
    values beyond 2^24 are split into an exact ``(hi, lo)`` pair). A CPU
    floating-point tensor with ``dim() > 0`` raises torch's device error
    unless ``allow_cpu`` is set (``copy_`` is the one op torch lets cross
    devices).
    """
    if is_metal(x):
        if x._mode != mode:
            if _is_fixed(x):
                raise TypeError(f"cannot mix representations {x._mode} and {mode}")
            _retag(x, mode)
        return x
    s = scalar_value(x)
    if s is not None:
        return MetalFloat64.from_numpy(
            np.array(s, dtype=np.float64), mode, device=str(device)
        )
    if isinstance(x, torch.Tensor):
        if x.is_complex():
            raise TypeError("cannot coerce a complex tensor to MetalFloat64")
        if x.device.type == "cpu":
            if x.dim() > 0 and x.dtype.is_floating_point and not allow_cpu:
                # Only 0-d CPU tensors act as scalar operands of a device tensor
                # (torch's rule); bool / integer CPU tensors are accepted because
                # the dual-residency host path produces its masks and indices there.
                raise RuntimeError(
                    "Expected all tensors to be on the same device, but found at "
                    "least two devices, mps:0 and cpu!"
                )
            return MetalFloat64.from_numpy(
                x.detach().to(torch.float64).numpy(), mode, device=str(device)
            )
        t = x.detach()
        if t.dtype == torch.bool:
            t = t.to(torch.float32)
        if mode == "df64":
            if t.dtype in (torch.float16, torch.bfloat16, torch.float32):
                hi = _fresh_f32(t)
                if t.dtype == torch.float32:
                    # Denormal float32 words are flushed to a signed zero,
                    # exactly as ``encode.encode_df64`` does (GPU FTZ: the
                    # kernels would treat them as zero while the host decode
                    # would not), so the tensor is canonical.
                    hi = _flush_denormals(hi)
                return wrap((hi, torch.zeros_like(hi)), mode)
            if t.dtype in (torch.int8, torch.uint8, torch.int16):
                hi = _fresh_f32(t)  # exact: |x| < 2^24
                return wrap((hi, torch.zeros_like(hi)), mode)
            if t.dtype == torch.int32:
                # hi = nearest float32, lo = the (< 2^8, exact) integer residual.
                hi = _fresh_f32(t)
                lo = (t.to(torch.int64) - hi.to(torch.int64)).to(torch.float32)
                return wrap((hi, lo.contiguous()), mode)
            # int64 may exceed float32: go through the host to keep exactness.
            return MetalFloat64.from_numpy(
                t.cpu().to(torch.float64).numpy(), mode, device=str(device)
            )
        # sf64: exact conversion on the host (cheap for the rare mixed case).
        return MetalFloat64.from_numpy(
            t.cpu().to(torch.float64).numpy(), mode, device=str(device)
        )
    if isinstance(x, (np.ndarray, list, tuple)):
        return MetalFloat64.from_numpy(
            np.asarray(x, dtype=np.float64), mode, device=str(device)
        )
    raise TypeError(f"cannot coerce {type(x).__name__} to MetalFloat64")


def _flush_denormals(hi: torch.Tensor) -> torch.Tensor:
    """Replace sub-``FLT_MIN`` words of a contiguous float32 tensor by signed zeros."""
    tiny = hi.abs() < encode.DF64_FLT_MIN
    return torch.where(tiny, torch.copysign(torch.zeros_like(hi), hi), hi)


def _fresh_f32(t: torch.Tensor) -> torch.Tensor:
    """``t`` as a contiguous float32 tensor at storage offset 0 (always a copy).

    ``.contiguous()`` returns a contiguous *view* unchanged, including one with
    a non-zero storage offset (``p[1:]``, ``p[i, j]``), whose ``zeros_like``
    partner would then disagree on the offset; the pair must share geometry.
    """
    return t.to(torch.float32).clone(memory_format=torch.contiguous_format)


def unwrap_components(x: MetalFloat64) -> tuple[torch.Tensor, ...]:
    """Component tensors of ``x`` (no copy)."""
    return x._comps


def comps_arg(x: MetalFloat64) -> Any:
    """Component operand in the form ``MetalLibrary.launch`` expects."""
    return x._comps if x._mode == "df64" else x._comps[0]


def result_from_launch(res: Any, mode: str) -> MetalFloat64 | torch.Tensor:
    """Wrap a ``launch`` result: value results become MetalFloat64, bools stay plain."""
    if isinstance(res, torch.Tensor):
        if res.dtype == torch.bool:
            return res
        return wrap((res,), mode)
    return wrap(tuple(res), mode)


def apply_to_components(func: Any, args: Any, kwargs: Any) -> Any:
    """Run ``func`` once per component (structural ops) and rewrap the results.

    ``MetalFloat64`` leaves anywhere in ``args``/``kwargs`` are replaced by their
    k-th component for the k-th call; every output tensor is rewrapped from the
    per-call outputs. Uses ``return_and_correct_aliasing`` so views alias and
    in-place ops return the input object.
    """
    leaves = _flatten(args) + (_flatten(kwargs) if kwargs else [])
    mode = DEFAULT_MODE
    for a in leaves:
        if is_metal(a):
            mode = a._mode
            break
    n = 2 if mode == "df64" else 1
    outs = []
    for k in range(n):

        def pick(a: Any, _k: int = k) -> Any:
            return a._comps[_k] if is_metal(a) else a

        p_args, p_kwargs = _map_args(pick, args, kwargs)
        outs.append(func(*p_args, **p_kwargs))
    if isinstance(outs[0], torch.Tensor):
        out = wrap(tuple(outs), mode)
    elif isinstance(outs[0], (list, tuple)):
        out = type(outs[0])(
            wrap(tuple(o[i] for o in outs), mode)
            if isinstance(outs[0][i], torch.Tensor)
            else outs[0][i]
            for i in range(len(outs[0]))
        )
    else:
        out = outs[0]
    return return_and_correct_aliasing(func, args, kwargs, out)


def _flatten(x: Any) -> list[Any]:
    """Leaves of an args tuple / kwargs dict (fast path: flat or one level of lists)."""
    leaves: list[Any] = []
    items = x.values() if isinstance(x, dict) else x
    for a in items:
        if isinstance(a, (list, tuple)):
            for b in a:
                if isinstance(b, (list, tuple, dict)):
                    return _flatten_pytree(x)
                leaves.append(b)
        elif isinstance(a, dict):
            return _flatten_pytree(x)
        else:
            leaves.append(a)
    return leaves


def _flatten_pytree(x: Any) -> list[Any]:
    leaves: list[Any] = []

    def visit(a: Any) -> Any:
        leaves.append(a)
        return a

    tree_map(visit, x)
    return leaves


def _map_args(fn: Callable[[Any], Any], args: Any, kwargs: Any) -> tuple[Any, Any]:
    """Apply ``fn`` to every leaf of args/kwargs (fast path: flat / one-level lists)."""

    def conv(a: Any) -> Any:
        if isinstance(a, list):
            if any(isinstance(b, (list, tuple, dict)) for b in a):
                return tree_map(fn, a)
            return [fn(b) for b in a]
        if isinstance(a, tuple):
            if any(isinstance(b, (list, tuple, dict)) for b in a):
                return tree_map(fn, a)
            return tuple(fn(b) for b in a)
        if isinstance(a, dict):
            return tree_map(fn, a)
        return fn(a)

    new_args = tuple(conv(a) for a in args)
    new_kwargs = {k: conv(v) for k, v in kwargs.items()} if kwargs else kwargs
    return new_args, new_kwargs


def _has_complex(args: Any) -> bool:
    """True when a positional operand is a Python complex or a complex tensor."""
    for a in args:
        if isinstance(a, complex):
            return True
        if isinstance(a, torch.Tensor):
            if a.is_complex():
                return True
        elif isinstance(a, (list, tuple)):
            for b in a:
                if isinstance(b, complex) or (
                    isinstance(b, torch.Tensor) and b.is_complex()
                ):
                    return True
    return False


def complex_on_cpu(func: Any, args: Any, kwargs: Any) -> Any:
    """Run an op with a complex operand on the CPU (complex is not emulated).

    MetalFloat64 operands are decoded to CPU float64, plain mps tensors are
    moved to the CPU, and the op runs there; complex results stay on the CPU
    (the documented policy for polarization / FFT paths), real float64
    results are re-encoded. Counted under ``cpu_complex:<op>``; never a strict
    mode error because there is no GPU alternative by design.
    """
    _STATS[f"cpu_complex:{_op_name(func)}"] += 1
    metal_args = [a for a in _flatten(args) + _flatten(kwargs) if is_metal(a)]
    mode = mode_of(*metal_args)

    def to_cpu(a: Any) -> Any:
        if is_metal(a):
            return a.to_cpu_float64()
        if isinstance(a, torch.Tensor) and a.device.type == "mps":
            return a.cpu()
        if isinstance(a, torch.device) and a.type == "mps":
            return torch.device("cpu")
        return a

    c_args, c_kwargs = _map_args(to_cpu, args, kwargs)
    out = func(*c_args, **c_kwargs)

    def back(o: Any) -> Any:
        if isinstance(o, torch.Tensor) and o.dtype in (torch.float64, torch.float32):
            return MetalFloat64.from_numpy(o.detach().to(torch.float64).numpy(), mode)
        return o

    return tree_map(back, out)


def cpu_fallback(func: Any, args: Any, kwargs: Any, label: str | None = None) -> Any:
    """Run ``func`` on CPU float64 copies of every MetalFloat64 operand and re-encode.

    Counted under ``cpu_fallback:<op>``; raises :class:`MetalFallbackError` in
    strict mode. Only for ops with no GPU implementation (dense linear algebra
    factorizations, FFTs, ...).
    """
    name = label or _op_name(func)
    if strict_mode():
        raise MetalFallbackError(
            f"{name} would fall back to the CPU (OPTILAND_METAL_STRICT=1)"
        )
    _STATS[f"cpu_fallback:{name}"] += 1
    metal_args = [a for a in _flatten(args) + _flatten(kwargs) if is_metal(a)]
    mode = mode_of(*metal_args)

    def to_cpu(a: Any) -> Any:
        if is_metal(a):
            return a.to_cpu_float64()
        if isinstance(a, torch.Tensor) and a.device.type == "mps":
            return a.cpu()
        if isinstance(a, torch.device) and a.type == "mps":
            return torch.device("cpu")
        return a

    c_args, c_kwargs = _map_args(to_cpu, args, kwargs)
    out = func(*c_args, **c_kwargs)

    def back(o: Any) -> Any:
        if isinstance(o, torch.Tensor):
            if o.dtype in (torch.float64, torch.float32):
                return MetalFloat64.from_numpy(
                    o.detach().to(torch.float64).numpy(), mode
                )
            return o.to("mps")
        return o

    return tree_map(back, out)


def cpu_fallback_autograd(fn: Any, args: Any, kwargs: Any, label: str) -> Any:
    """Run ``fn`` on CPU float64 *above* autograd, keeping the graph.

    For backend-level functions (``be.grid_sample``, ``be.polyfit``,
    ``be.fftconvolve``) called from Python rather than from inside
    ``__torch_dispatch__``: MetalFloat64 operands are moved to the CPU with
    ``Tensor.to`` (a differentiable ``_to_copy``), ``fn`` runs in CPU float64
    under torch autograd, and floating results come back through
    ``.to(mps, float64)``, which the factory intercept turns into a
    MetalFloat64 with its ``ToCopyBackward`` intact. Counted and strict-mode
    checked exactly like :func:`cpu_fallback`.
    """
    if strict_mode():
        raise MetalFallbackError(
            f"{label} would fall back to the CPU (OPTILAND_METAL_STRICT=1)"
        )
    _STATS[f"cpu_fallback:{label}"] += 1
    enable_factory_intercept()

    def to_cpu(a: Any) -> Any:
        if is_metal(a):
            return a.to("cpu")
        if isinstance(a, torch.Tensor) and a.device.type == "mps":
            return a.cpu()
        if isinstance(a, torch.device) and a.type == "mps":
            return torch.device("cpu")
        return a

    c_args, c_kwargs = _map_args(to_cpu, args, kwargs)
    out = fn(*c_args, **c_kwargs)

    def back(o: Any) -> Any:
        if isinstance(o, torch.Tensor) and not is_metal(o):
            if o.dtype in (torch.float64, torch.float32):
                return o.to(device=MPS0, dtype=torch.float64)
            if o.dtype.is_complex:
                return o  # complex lives on the CPU by policy
            return o.to(MPS0)
        return o

    return tree_map(back, out)


def unimplemented(func: Any, args: Any, kwargs: Any) -> Any:
    """Fail loudly for ops without a handler (never run them on the components)."""
    raise NotImplementedError(
        f"MetalFloat64: no handler for {_op_name(func)} "
        f"(args types: {[type(a).__name__ for a in args]}). "
        "Register one with @implements."
    )


def _op_name(func: Any) -> str:
    return str(
        getattr(func, "_schema", None)
        and func._schema.name
        or getattr(func, "__name__", func)
    )


def count_gpu(name: str) -> None:
    """Record one GPU execution of ``name``."""
    _STATS[f"gpu:{name}"] += 1


def count_host(name: str) -> None:
    """Record one CPU float64 execution of ``name`` on the dual-residency path."""
    _STATS[f"host:{name}"] += 1


def count_event(key: str, n: int = 1) -> None:
    """Add ``n`` to the counter ``key`` verbatim (no namespace prefix).

    The fused trace driver (``metal/trace.py``) uses this for its
    ``fused_trace:*`` and ``fused_trace_skip:*`` counters, whose key set is
    closed by :class:`~...trace_adapters.FusedTraceSkip`; launches keep going
    through :func:`count_gpu` so ``gpu:*`` stays the launch namespace.
    """
    _STATS[key] += n


# ---------------------------------------------------------------------------
# Nucleus handlers (identity / conversion); everything else lives in ops_*.py
# ---------------------------------------------------------------------------
@implements(
    aten.detach.default, aten.alias.default, aten.clone.default, aten.contiguous.default
)
def _identity_like(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    if is_metal(x) and x._host is not None:
        # A host-resident tensor larger than the threshold (the stride-0
        # expansion of a per-surface constant, a host ``cat`` result) keeps
        # its residency through detach/alias/clone/contiguous.
        return _run_on_host(func, args, kwargs)
    return apply_to_components(func, args, kwargs)


@implements(aten._local_scalar_dense.default)
def _item(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    if x._host is not None:
        return x._host.item()
    return float(x.to_numpy().reshape(()))


@implements(aten._to_copy.default)
def _to_copy(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    dtype = kwargs.get("dtype")
    device = kwargs.get("device")
    dev = torch.device(device) if device is not None else x.device
    if dev.type != "mps":
        t = x.to_cpu_float64().clone()
        if dtype is not None and dtype != torch.float64:
            t = t.to(dtype)
        return t.to(dev) if dev.type != "cpu" else t
    if dtype is None or dtype == torch.float64:
        if x._host is not None:
            return wrap_host(x._host.clone(), x._mode)
        comps = tuple(c.clone() for c in x._comps)
        return wrap(comps, x._mode)
    if dtype == torch.float32:
        if x._mode == "df64" and x._host is None:
            return x._comps[0].clone()
        return torch.from_numpy(x.to_numpy().astype(np.float32)).to("mps")
    if dtype in (torch.complex64, torch.complex128):
        # MPS has no complex128 and complex data is not an emulated type: complex
        # results live on the CPU (FFT/polarization paths), counted as a fallback.
        _STATS["cpu_fallback:_to_copy(complex)"] += 1
        return torch.from_numpy(x.to_numpy()).to(dtype)
    # bool/int targets go through the host for exact semantics
    return torch.from_numpy(x.to_numpy()).to(dtype).to("mps")


@implements(aten.copy_.default)
def _copy_(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    dst, src = args[0], args[1]
    if not is_metal(dst):
        # plain destination (e.g. float32 mps or cpu float64): decode
        dst.copy_(
            src.to_cpu_float64().to(dst.device, dst.dtype)
            if dst.device.type != "mps"
            else src._comps[0]
            if src._mode == "df64" and dst.dtype == torch.float32
            else torch.from_numpy(src.to_numpy()).to(dst.device, dst.dtype)
        )
        return dst
    if dst._host is not None:
        dst._host.copy_(
            src.to_cpu_float64() if is_metal(src) else src.to("cpu", torch.float64)
        )
        dst.invalidate_host_cache()
        return dst
    if is_metal(src):
        mode_of(dst, src)  # an agnostic side follows the fixed one
    s = coerce(src, dst._mode, allow_cpu=True)
    for d, c in zip(dst._comps, s._comps, strict=True):
        d.copy_(c)
    return dst


def _set_wrapper_geometry(x: MetalFloat64, like: torch.Tensor) -> None:
    """Give the wrapper ``x`` the sizes/strides/offset of ``like`` (in place).

    A wrapper subclass owns a placeholder storage sized at construction, so a
    resize needs two steps: swap in a fresh placeholder of the right size (the
    same unchecked swap ``return_and_correct_aliasing`` uses for views) and
    then run ``as_strided_`` on the meta key, which updates the metadata
    without dispatching anywhere. Nothing is allocated on the GPU. The
    metadata update runs under ``no_grad`` so a tensor that carries a
    ``grad_fn`` keeps it (autograd would otherwise record an ``as_strided_``).
    """
    fresh = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
        MetalFloat64,
        like.shape,
        strides=like.stride(),
        storage_offset=like.storage_offset(),
        dtype=torch.float64,
        device=MPS0,
        layout=like.layout,
        requires_grad=False,
    )
    torch._functionalize_unsafe_set(x, fresh)
    with torch.no_grad(), torch.utils._mode_utils.no_dispatch():
        meta_in_tls = torch._C._meta_in_tls_dispatch_include()
        torch._C._set_meta_in_tls_dispatch_include(True)
        try:
            aten.as_strided_.default(
                x, list(like.shape), list(like.stride()), like.storage_offset()
            )
        finally:
            torch._C._set_meta_in_tls_dispatch_include(meta_in_tls)


def resize_(x: MetalFloat64, shape: Sequence[int]) -> MetalFloat64:
    """``x.resize_(shape)`` with torch semantics (storage grows, prefix kept).

    The components (or the host tensor) are resized with the plain ``resize_``,
    so a view whose base storage is large enough is re-laid over the shared
    storage exactly as torch does; then the wrapper metadata follows.
    """
    shape = tuple(int(n) for n in shape)
    if tuple(x.shape) == shape:
        return x
    if x.requires_grad and torch.is_grad_enabled():
        raise RuntimeError("cannot resize variables that require grad")
    if x._host is not None:
        if x._host.untyped_storage().resizable():
            x._host.resize_(shape)
        else:
            # numpy-backed host storage is not resizable: reallocate, keep the prefix
            old = x._host.reshape(-1)
            new = torch.empty(shape, dtype=torch.float64)
            k = min(old.numel(), new.numel())
            new.view(-1)[:k].copy_(old[:k])
            x._host = new
        x.invalidate_host_cache()
        _set_wrapper_geometry(x, x._host)
        return x
    assert x._comps_cache is not None
    for c in x._comps_cache:
        c.resize_(shape)
    _set_wrapper_geometry(x, x._comps_cache[0])
    return x


def check_out_device(dst: torch.Tensor, op: str | None = None) -> None:
    """Raise torch's device error for an ``out=`` destination that is not on mps.

    ``op`` selects the wording: ``None`` for the elementwise / reduction form,
    the op name for the matmul family (``Expected out tensor to have device``).
    """
    if isinstance(dst, torch.Tensor) and dst.device.type != "mps":
        if op is None:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found at least "
                f"two devices, mps:0 and {dst.device}!"
            )
        raise RuntimeError(
            f"Expected out tensor to have device mps:0, but got {dst.device} instead"
        )


def check_inplace_shape(dst: torch.Tensor, shape: Sequence[int]) -> None:
    """Raise torch's error when an in-place op would have to resize ``dst``."""
    shape = tuple(int(n) for n in shape)
    if tuple(dst.shape) != shape:
        raise RuntimeError(
            f"output with shape {list(dst.shape)} doesn't match the broadcast "
            f"shape {list(shape)}"
        )


def resize_out(dst: torch.Tensor, shape: Sequence[int]) -> None:
    """Resize an ``out=`` destination (plain or MetalFloat64) with torch's warning."""
    shape = tuple(int(n) for n in shape)
    if tuple(dst.shape) == shape:
        return
    if dst.numel() != 0:
        warnings.warn(
            "An output with one or more elements was resized since it had shape "
            f"{list(dst.shape)}, which does not match the required output shape "
            f"{list(shape)}. This behavior is deprecated, and in a future PyTorch "
            "release outputs will not be resized unless they have zero elements.",
            UserWarning,
            stacklevel=3,
        )
    if is_metal(dst):
        resize_(dst, shape)
    else:
        dst.resize_(shape)


@implements(aten.resize_.default)
def _resize_(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    if "memory_format" in kwargs and kwargs["memory_format"] not in (
        None,
        torch.contiguous_format,
    ):
        raise NotImplementedError("MetalFloat64: resize_ supports contiguous_format")
    return resize_(args[0], args[1])


# ---------------------------------------------------------------------------
# torch-function level fixes (see MetalFloat64.__torch_function__)
# ---------------------------------------------------------------------------
def _fix_where(args: Any, kwargs: Any) -> tuple[Any, Any]:
    """``torch.where(cond, a, b)`` / ``x.where(cond, b)``: wrap scalar operands.

    The mode comes from any MetalFloat64 argument, including ``self`` of the
    method form (``args[0]``), so the scalar never goes through the global-mode
    factory intercept.
    """
    args = list(args)
    tensors = [a for a in args if is_metal(a)] + [
        v for v in kwargs.values() if is_metal(v)
    ]
    if not tensors:
        return tuple(args), kwargs
    mode = tensors[0]._mode
    for i in (1, 2):
        if i < len(args) and scalar_value(args[i]) is not None:
            args[i] = coerce(args[i], mode)
    for key in ("input", "other"):
        if key in kwargs and scalar_value(kwargs[key]) is not None:
            kwargs[key] = coerce(kwargs[key], mode)
    return tuple(args), kwargs


_TORCH_FUNCTION_FIXES: dict[Any, Callable[[Any, Any], tuple[Any, Any]]] = {
    torch.where: _fix_where,
    torch.Tensor.where: _fix_where,
}


# ---------------------------------------------------------------------------
# Decompositions: ops torch does not decompose for us, expressed with tensor ops on
# the wrappers (which re-enter dispatch). Handler modules may override these.
# ---------------------------------------------------------------------------
@implements(aten._euclidean_dist.default)
def _euclidean_dist(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x1, x2 = args[0], args[1]
    diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)
    return (diff * diff).sum(-1).sqrt()


@implements(aten._cdist_forward.default)
def _cdist_forward(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x1, x2, p = args[0], args[1], args[2]
    diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).abs()
    if p == 2.0:
        return (diff * diff).sum(-1).sqrt()
    if p == 1.0:
        return diff.sum(-1)
    if p == float("inf"):
        return diff.amax(-1)
    return (diff**p).sum(-1) ** (1.0 / p)


# ---------------------------------------------------------------------------
# Factory interception: C++ autograd formulas (clamp, copysign, pow, ...) create
# float64 tensors with the wrapper's options (device mps), which MPS rejects
# before any subclass dispatch. A dispatch mode is the only hook for tensor-less
# factory calls; it builds them on the CPU and wraps the result.
# ---------------------------------------------------------------------------
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402


def _current_mode() -> str:
    try:
        from optiland.backend.torch_backend import metal as _metal

        return _metal.get_mode()
    except Exception:  # pragma: no cover - during partial imports
        return DEFAULT_MODE


#: Plain-tensor ops (no MetalFloat64 operand) whose operands may sit on different
#: devices after the host path: CPU bool / int64 results of host-resident
#: predicates and index ops meet mps masks, mps index tensors and CPU complex
#: tensors (complex lives on the CPU by policy). See :func:`_unify_devices`.
_MIXED_DEVICE_OPS: set[Any] = {
    aten.bitwise_and.Tensor,
    aten.bitwise_or.Tensor,
    aten.bitwise_xor.Tensor,
    aten.bitwise_and_.Tensor,
    aten.bitwise_or_.Tensor,
    aten.bitwise_xor_.Tensor,
    aten.logical_and.default,
    aten.logical_or.default,
    aten.logical_xor.default,
    aten.logical_and_.default,
    aten.logical_or_.default,
    aten.logical_xor_.default,
    aten.where.self,
    aten.eq.Tensor,
    aten.ne.Tensor,
    aten.lt.Tensor,
    aten.le.Tensor,
    aten.gt.Tensor,
    aten.ge.Tensor,
    aten.masked_fill.Scalar,
    aten.masked_fill.Tensor,
    aten.masked_fill_.Scalar,
    aten.masked_fill_.Tensor,
    aten.index.Tensor,
    aten.index_put.default,
    aten.index_put_.default,
    aten._index_put_impl_.default,
    aten.index_select.default,
    aten.gather.default,
    aten.masked_select.default,
    aten.cat.default,
    aten.stack.default,
    aten.mul.Tensor,
    aten.add.Tensor,
    aten.sub.Tensor,
}
#: Earlier name of the table (tests / callers).
_MIXED_BOOL_OPS = _MIXED_DEVICE_OPS


def _movable_cpu(t: torch.Tensor) -> bool:
    """CPU operands the host path produces that torch will not mix with mps ones.

    Bool masks of any rank (the MPS logical kernels reject even a 0-d CPU
    operand: ``max < min`` of two host-resident bounds in clamp_backward) and
    N-d integer index tensors; a 0-d CPU integer is a scalar to torch.
    """
    if t.device.type != "cpu":
        return False
    if t.dtype == torch.bool:
        return True
    return t.dim() > 0 and not (t.dtype.is_floating_point or t.dtype.is_complex)


def _unify_devices(func: Any, args: Any) -> Any:
    """Reconcile the devices of plain operands the host path split apart.

    Host-path predicates and index ops return CPU bool / int64 tensors (so
    ``.item()`` never syncs) while kernel predicates are mps bool tensors, and
    complex data lives on the CPU by policy. Optiland combines these freely
    and torch refuses mixed devices, so, mirroring what a plain torch program
    would do:

    * a CPU *complex* operand pulls every mps operand to the CPU;
    * an in-place op on a CPU target pulls its mps operands (indices, values)
      to the CPU;
    * otherwise the CPU bool masks and N-d integer tensors move to mps.
    """
    leaves = _flatten(args)
    tensors = [a for a in leaves if isinstance(a, torch.Tensor)]
    if not any(t.device.type == "mps" for t in tensors):
        return args
    cpu = [t for t in tensors if t.device.type == "cpu"]
    if not cpu:
        return args
    if any(t.is_complex() for t in cpu) or (
        _mutates_first(func)
        and isinstance(args[0], torch.Tensor)
        and args[0].device.type == "cpu"
    ):

        def to_cpu(a: Any) -> Any:
            if isinstance(a, torch.Tensor) and a.device.type == "mps":
                return a.cpu()
            return a

        return _map_args(to_cpu, args, {})[0]
    if not any(_movable_cpu(t) for t in cpu):
        return args

    def to_mps(a: Any) -> Any:
        if isinstance(a, torch.Tensor) and _movable_cpu(a):
            return a.to(MPS0)
        return a

    return _map_args(to_mps, args, {})[0]


def _unify_bool_devices(args: Any) -> Any:
    """Move CPU bool masks to mps when an op mixes them with mps tensors.

    Kept as the documented entry point; :func:`_unify_devices` is the general
    rule :class:`MetalFactoryMode` applies.
    """
    return _unify_devices(aten.logical_and.default, args)


class MetalFactoryMode(TorchDispatchMode):
    """Redirect float64-on-mps factory calls (no MetalFloat64 operands) to the host.

    Also reconciles the devices of plain operands for the ops in
    ``_MIXED_DEVICE_OPS`` (see ``_unify_devices``) and builds complex128
    factories aimed at mps on the CPU (complex lives there by policy).
    """

    def __torch_dispatch__(
        self, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        kwargs = kwargs or {}
        if func in _MIXED_DEVICE_OPS and not any(
            isinstance(a, MetalFloat64) for a in _flatten(args)
        ):
            args = _unify_devices(func, args)
        dtype = kwargs.get("dtype")
        if dtype == torch.float64:
            device = kwargs.get("device")
            if func is aten._to_copy.default and isinstance(args[0], torch.Tensor):
                src = args[0]
                if (
                    not isinstance(src, MetalFloat64)
                    and src.device.type == "mps"
                    and (device is None or torch.device(device).type == "mps")
                ):
                    # ``plain_mps.to(float64)``, emitted by C++ autograd formulas
                    # (pow_backward_exponent, sum_backward for a float32 result):
                    # exact promotion on the GPU, agnostic like any constant.
                    if src.dtype == torch.float64:
                        return src.clone()
                    return _promote_plain(src)
            if (
                device is not None
                and torch.device(device).type == "mps"
                and not any(isinstance(a, MetalFloat64) for a in _flatten(args))
            ):
                k = dict(kwargs)
                k["device"] = torch.device("cpu")
                if func is aten._to_copy.default and isinstance(args[0], torch.Tensor):
                    k.pop("dtype", None)
                    src = args[0].detach().cpu().to(torch.float64)
                    return _wrap_new(src, k)
                out = func(*args, **k)
                return _wrap_new(out, {})
        elif dtype == torch.complex128:
            # MPS has no complex128 and complex data is not emulated: a
            # complex128 factory / conversion aimed at mps (``torch.zeros(n,
            # dtype=complex128, device=mps)`` in the Huygens PSF strategies)
            # is built on the CPU, where every complex result already lives
            # (``complex_on_cpu``, ``_to_copy`` to complex).
            device = kwargs.get("device")
            is_copy = func is aten._to_copy.default and isinstance(
                args[0], torch.Tensor
            )
            if device is None and is_copy:
                device = args[0].device  # ``mps_tensor.to(complex128)``
            if (
                device is not None
                and torch.device(device).type == "mps"
                and not any(isinstance(a, MetalFloat64) for a in _flatten(args))
            ):
                k = dict(kwargs)
                k["device"] = torch.device("cpu")
                _STATS[f"cpu_complex:{_op_name(func)}"] += 1
                if is_copy:
                    return func(args[0].cpu(), *args[1:], **k)
                return func(*args, **k)
        return func(*args, **kwargs)


def _promote_plain(src: torch.Tensor) -> MetalFloat64:
    """Exact float64 promotion of a plain mps tensor (representation-agnostic)."""
    if src.numel() <= HOST_THRESHOLD and HOST_THRESHOLD > 0:
        return wrap_host(
            src.detach().cpu().to(torch.float64).contiguous(), _current_mode()
        )
    r = coerce(src, _current_mode())
    r._agnostic = True
    return r


def _wrap_new(out: torch.Tensor, k: dict[str, Any]) -> Any:
    """Wrap a factory result built on the CPU (representation-agnostic, see above)."""
    mode = _current_mode()
    if out._is_zerotensor():
        # sgn_backward & co. build an efficient ZeroTensor, which has no storage.
        out = torch.zeros(tuple(out.shape), dtype=out.dtype)
    if out.numel() <= HOST_THRESHOLD and HOST_THRESHOLD > 0:
        return wrap_host(out.contiguous(), mode)
    r = MetalFloat64.from_numpy(out.numpy(), mode, host=False)
    r._agnostic = True
    return r


_FACTORY_MODE: MetalFactoryMode | None = None
# Arm the intercept on first MetalFloat64 creation (set False to opt out).
_AUTO_FACTORY_INTERCEPT = os.environ.get("OPTILAND_METAL_FACTORY_INTERCEPT", "1") == "1"


def enable_factory_intercept() -> None:
    """Install the factory-intercept dispatch mode (idempotent)."""
    global _FACTORY_MODE
    if _FACTORY_MODE is None:
        _FACTORY_MODE = MetalFactoryMode()
        _FACTORY_MODE.__enter__()


def disable_factory_intercept() -> None:
    """Remove the factory-intercept dispatch mode if installed."""
    global _FACTORY_MODE
    if _FACTORY_MODE is not None:
        _FACTORY_MODE.__exit__(None, None, None)
        _FACTORY_MODE = None


def factory_intercept_enabled() -> bool:
    """True while the factory-intercept mode is installed."""
    return _FACTORY_MODE is not None


@implements(aten.allclose.default)
def _allclose(func: Any, types: Any, args: Any, kwargs: Any) -> bool:
    a, b = args[0], args[1]
    rtol = kwargs.get("rtol", args[2] if len(args) > 2 else 1e-5)
    atol = kwargs.get("atol", args[3] if len(args) > 3 else 1e-8)
    equal_nan = kwargs.get("equal_nan", args[4] if len(args) > 4 else False)
    mode = mode_of(a, b)
    a = coerce(a, mode)
    b = coerce(b, mode)
    close = (a - b).abs() <= (atol + rtol * b.abs())
    if equal_nan:
        close = close | (torch.isnan(a) & torch.isnan(b))
    return bool(close.all())


@implements(aten.equal.default)
def _equal(func: Any, types: Any, args: Any, kwargs: Any) -> bool:
    a, b = args[0], args[1]
    if tuple(a.shape) != tuple(b.shape):
        return False
    mode = mode_of(a, b)
    return bool((coerce(a, mode) == coerce(b, mode)).all())


@implements(aten.isclose.default)
def _isclose(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    a, b = args[0], args[1]
    rtol = kwargs.get("rtol", args[2] if len(args) > 2 else 1e-5)
    atol = kwargs.get("atol", args[3] if len(args) > 3 else 1e-8)
    equal_nan = kwargs.get("equal_nan", args[4] if len(args) > 4 else False)
    mode = mode_of(a, b)
    a = coerce(a, mode)
    b = coerce(b, mode)
    close = (a - b).abs() <= (atol + rtol * b.abs())
    if equal_nan:
        close = close | (torch.isnan(a) & torch.isnan(b))
    return close


# Register the handler modules (order matters only for overrides).
from optiland.backend.torch_backend.metal import (  # noqa: E402,F401
    ops_creation,
    ops_elementwise,
    ops_linalg,
    ops_reduce,
    ops_structural,
)
