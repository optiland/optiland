"""MetalFloat64 dispatch handlers: reductions (registered via ``tensor.implements``).

Every reduction maps onto the ``[outer, n, inner]`` layout of ``kernels/reduce.metal``:
the reduced dims are moved to the middle (a permute + ``contiguous`` on the plain
component tensors when they are not already a consecutive block of a contiguous
tensor), one kernel launch reduces the middle axis, and the result is reshaped to
the torch output shape. Long axes use the threadgroup tree kernels with the
multi-pass partial scheme documented in ``reduce.metal``; short axes (or very many
lines) use the sequential per-line kernels.

Accuracy of the df64 kernels: ``sum`` / ``mean`` / ``var`` stay within ``1e-13``
relative on both the tree and the sequential path (the tree is used from
``TREE_FORCE_N = 1024`` elements whatever the line count; the sequential chain
accumulates ``~sqrt(n) u^2`` and reached ``1.2e-13`` at ``n = 4095``). ``prod``
over long axes is a representation limit: every df64 multiplication carries
``u^2 = 3.6e-15`` and a product of ``n`` factors accumulates ``~sqrt(n)`` of
them on both paths (``1.6e-13`` at ``n = 4095``, ``4.4e-14`` at ``n = 127``,
against torch CPU's ``2e-15``); sf64 ``prod`` is exact to ``5.5e-15``.

Derived reductions (``mean``, ``var``/``std``, ``linalg_vector_norm``, ``logsumexp``)
are composed from the reduction kernels and the elementwise kernels of the
:class:`~optiland.backend.torch_backend.metal.library.MetalLibrary` on the GPU;
nothing is decoded to the host. ``cumprod`` is the only CPU fallback in this module.

The reduction and matmul kernels (``reduce.metal``, ``matmul.metal``) are compiled
into the same ``torch.mps`` library as the elementwise kernels by ``MetalLibrary``;
:func:`reduce_kernels` returns that object (or compiles the two files on top of the
core headers when a library was built without them).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from optiland.backend.torch_backend.metal import compile as _compile
from optiland.backend.torch_backend.metal.library import broadcast_contiguous
from optiland.backend.torch_backend.metal.tensor import (
    MetalFloat64,
    aten,
    check_out_device,
    coerce,
    count_gpu,
    cpu_fallback,
    implements,
    library,
    scalar_value,
    wrap,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Threadgroup size of the tree kernels (``OPTILAND_REDUCE_TG`` in reduce.metal).
TG = 256
#: Upper bound on the number of threadgroups per line in one tree pass.
MAX_GROUPS = 1024
#: Elements per threadgroup that one tree pass aims for.
ELEMS_PER_GROUP = 4096
#: Axis length from which the tree kernels are preferred over the sequential ones.
TREE_MIN_N = 128
#: Above this many (outer, inner) lines the sequential kernels win unless the axis
#: is very long (the tree kernels launch 256 threads per line and group).
TREE_MAX_LINES = 16384
#: Axis length from which the tree kernels are used regardless of the line count:
#: a sequential df64 chain accumulates ~sqrt(n) u^2 (1.2e-13 relative at n = 4095
#: on positive data), the tree's depth is log2(n); 1024 keeps the sequential
#: path below ~6e-14 (fix round 2).
TREE_FORCE_N = 1024
#: Largest thread count issued in one dispatch.
MAX_THREADS = 1 << 30

#: Second-pass kernel per first-pass reduction (see the header of reduce.metal).
SECOND_PASS = {
    "sum": "sum",
    "nansum": "sum",
    "prod": "prod",
    "max": "max",
    "min": "min",
    "nanmax": "nanmax",
    "nanmin": "nanmin",
}

_HEADER_STACK = {
    "df64": ("df64_core.h",),
    "sf64": ("vendor/softfloat64.metal", "df64_core.h", "sf64_core.h"),
}
_KERNEL_FILES = ("reduce.metal", "matmul.metal")
_LIBS: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Kernel library access
# ---------------------------------------------------------------------------
def reduce_kernels(mode: str) -> Any:
    """Return the compiled ``torch.mps`` library holding the reduce/matmul kernels.

    ``MetalLibrary`` amalgamates ``reduce.metal`` and ``matmul.metal`` after its
    header stack, so the kernels are attributes of ``library(mode).lib`` (which
    has passed the compile-mode self-test). Should a library have been built
    without them, they are compiled once from the core headers instead.

    Args:
        mode: ``"df64"`` or ``"sf64"``.

    Returns:
        The compiled library object; kernels are attributes (``sum_df64``, ...).
    """
    lib = _LIBS.get(mode)
    if lib is None:
        lib = library(mode).lib
        if not hasattr(lib, f"sum_{mode}"):
            # The df64 elementwise library has validated the compile mode of this
            # process (EFT probes); the sf64 stack shares the same settings.
            library("df64")
            source = _compile.kernel_source(*_HEADER_STACK[mode], *_KERNEL_FILES)
            lib = _compile.compile_library(source)
        _LIBS[mode] = lib
    return lib


def _alloc(mode: str, numel: int) -> tuple[torch.Tensor, ...]:
    if mode == "df64":
        hi = torch.empty(numel, dtype=torch.float32, device="mps")
        return hi, torch.empty_like(hi)
    return (torch.empty(numel, dtype=torch.int64, device="mps"),)


def _choose_groups(n: int, lines: int) -> int:
    groups = max(1, min(MAX_GROUPS, -(-n // ELEMS_PER_GROUP)))
    cap = max(1, MAX_THREADS // max(1, lines * TG))
    return min(groups, cap)


def _use_tree(n: int, lines: int) -> bool:
    return n >= TREE_FORCE_N or (n >= TREE_MIN_N and lines <= TREE_MAX_LINES)


# ---------------------------------------------------------------------------
# Elementwise helper (df64/sf64 kernels of the MetalLibrary on component tuples)
# ---------------------------------------------------------------------------
def ew(
    mode: str,
    op: str,
    *operands: Any,
    scalar: float | None = None,
    scalar_side: str | None = None,
) -> Any:
    """Run one elementwise kernel on component tuples with host-side broadcasting.

    Args:
        mode: Representation of the operands.
        op: Elementwise op name (``codegen.OPS`` key).
        *operands: Component tuples (``(hi, lo)`` / ``(bits,)``) or, for ``where``,
            a plain bool tensor as the first operand.
        scalar: Optional Python float for the scalar variant of a binary op.
        scalar_side: ``"right"`` (default) or ``"left"`` when ``scalar`` is given.

    Returns:
        A component tuple for value results, or a bool tensor for predicates.

    Raises:
        KeyError: If ``op`` is not compiled into the library (header missing).
    """
    lib = library(mode)
    if not lib.has(op):
        raise KeyError(f"elementwise op {op!r} is not available in the {mode} library")
    flat: list[torch.Tensor] = []
    sizes: list[int] = []
    for o in operands:
        if isinstance(o, torch.Tensor):
            flat.append(o)
            sizes.append(1)
        else:
            flat.extend(o)
            sizes.append(len(o))
    b = broadcast_contiguous(*flat)
    launch_args: list[Any] = []
    pos = 0
    for o, k in zip(operands, sizes, strict=True):
        chunk = b[pos : pos + k]
        pos += k
        if isinstance(o, torch.Tensor) or mode == "sf64":
            launch_args.append(chunk[0])
        else:
            launch_args.append(tuple(chunk))
    res = lib.launch(op, *launch_args, scalar=scalar, scalar_side=scalar_side)
    count_gpu(op)
    if isinstance(res, torch.Tensor):
        return res if res.dtype == torch.bool else (res,)
    return tuple(res)


def has_op(mode: str, op: str) -> bool:
    """Whether the elementwise kernel ``op`` exists for ``mode``."""
    return library(mode).has(op)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def normalize_dims(dims: Any, ndim: int) -> tuple[int, ...]:
    """Canonicalize a torch ``dim`` argument to a sorted tuple of non-negative dims.

    ``None`` and ``[]`` mean every dim; negative dims wrap; duplicates are an
    error (as in torch). A 0-d tensor accepts dim 0 or -1 and yields ``()``.

    Args:
        dims: ``None``, an int, or a sequence of ints.
        ndim: Number of dims of the tensor.

    Returns:
        tuple[int, ...]: The sorted reduced dims.
    """
    if dims is None:
        return tuple(range(ndim))
    if isinstance(dims, int):
        dims = [dims]
    dims = list(dims)
    if not dims:
        return tuple(range(ndim))
    out: list[int] = []
    for d in dims:
        d = int(d)
        if ndim == 0:
            if d not in (0, -1):
                raise IndexError(
                    "Dimension out of range (expected to be in range of [-1, 0], "
                    f"but got {d})"
                )
            continue
        if d < -ndim or d >= ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of "
                f"[{-ndim}, {ndim - 1}], but got {d})"
            )
        d = d + ndim if d < 0 else d
        if d in out:
            raise RuntimeError(f"dim {d} appears multiple times in the list of dims")
        out.append(d)
    return tuple(sorted(out))


def out_shape(
    shape: Sequence[int], dims: Sequence[int], keepdim: bool
) -> tuple[int, ...]:
    """Shape of a reduction result."""
    dset = set(dims)
    if keepdim:
        return tuple(1 if i in dset else s for i, s in enumerate(shape))
    return tuple(s for i, s in enumerate(shape) if i not in dset)


def layout(
    comps: Sequence[torch.Tensor], dims: Sequence[int]
) -> tuple[tuple[torch.Tensor, ...], int, int, int]:
    """Bring component tensors into the contiguous ``[outer, n, inner]`` layout.

    A consecutive block of reduced dims in a contiguous tensor needs no copy;
    otherwise the kept dims are permuted in front of the reduced dims and the
    result is made contiguous (``inner = 1``).

    Args:
        comps: Plain component tensors (any strides).
        dims: Reduced dims (canonical, sorted).

    Returns:
        ``(flat_comps, outer, n, inner)``.
    """
    shape = tuple(comps[0].shape)
    ndim = len(shape)
    dset = set(dims)
    keep = [i for i in range(ndim) if i not in dset]
    red = [i for i in range(ndim) if i in dset]
    base = comps[0]
    if red and base.is_contiguous() and red == list(range(red[0], red[-1] + 1)):
        outer = math.prod(shape[: red[0]])
        n = math.prod(shape[red[0] : red[-1] + 1])
        inner = math.prod(shape[red[-1] + 1 :])
        return tuple(c.contiguous() for c in comps), outer, n, inner
    perm = keep + red
    flat = tuple(c.permute(perm).contiguous() for c in comps)
    outer = math.prod(shape[i] for i in keep)
    n = math.prod(shape[i] for i in red)
    return flat, outer, n, 1


# ---------------------------------------------------------------------------
# Kernel drivers
# ---------------------------------------------------------------------------
def run_reduce(
    mode: str, op: str, flat: Sequence[torch.Tensor], outer: int, n: int, inner: int
) -> tuple[torch.Tensor, ...]:
    """Reduce the middle axis of contiguous ``[outer, n, inner]`` components.

    Args:
        mode: Representation.
        op: First-pass kernel family (``sum``, ``nansum``, ``prod``, ``max``, ``min``,
            ``nanmax``, ``nanmin``).
        flat: Contiguous component tensors with ``outer * n * inner`` elements.
        outer: Product of the kept leading dims.
        n: Reduced axis length.
        inner: Product of the kept trailing dims.

    Returns:
        Component tensors with ``outer * inner`` elements (flat).
    """
    lines = outer * inner
    if lines == 0:
        return _alloc(mode, 0)
    lib = reduce_kernels(mode)
    if not _use_tree(n, lines):
        out = _alloc(mode, lines)
        getattr(lib, f"{op}_seq_{mode}")(
            *flat, *out, outer, n, inner, threads=[lines, 1, 1]
        )
        return out
    name = op
    cur: Sequence[torch.Tensor] = flat
    while True:
        groups = _choose_groups(n, lines)
        out = _alloc(mode, lines * groups)
        getattr(lib, f"{name}_{mode}")(
            *cur,
            *out,
            outer,
            n,
            inner,
            groups,
            threads=[lines * groups * TG, 1, 1],
            group_size=[TG, 1, 1],
        )
        if groups == 1:
            return out
        cur, n, name = out, groups, SECOND_PASS[op]


def run_arg_reduce(
    mode: str, op: str, flat: Sequence[torch.Tensor], outer: int, n: int, inner: int
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Index reduction (``argmax``/``argmin``) of the middle axis.

    Returns:
        ``(value_components, indices)`` with ``outer * inner`` elements each; the
        indices are int64 and refer to positions along the reduced axis.
    """
    lines = outer * inner
    if lines == 0:
        return _alloc(mode, 0), torch.empty(0, dtype=torch.int64, device="mps")
    lib = reduce_kernels(mode)
    dummy = torch.zeros(1, dtype=torch.int64, device="mps")
    if not _use_tree(n, lines):
        out = _alloc(mode, lines)
        out_idx = torch.empty(lines, dtype=torch.int64, device="mps")
        getattr(lib, f"{op}_seq_{mode}")(
            *flat, dummy, *out, out_idx, outer, n, inner, 0, threads=[lines, 1, 1]
        )
        return out, out_idx
    cur: Sequence[torch.Tensor] = flat
    idx = dummy
    has_idx = 0
    while True:
        groups = _choose_groups(n, lines)
        out = _alloc(mode, lines * groups)
        out_idx = torch.empty(lines * groups, dtype=torch.int64, device="mps")
        getattr(lib, f"{op}_{mode}")(
            *cur,
            idx,
            *out,
            out_idx,
            outer,
            n,
            inner,
            groups,
            has_idx,
            threads=[lines * groups * TG, 1, 1],
            group_size=[TG, 1, 1],
        )
        if groups == 1:
            return out, out_idx
        cur, idx, n, has_idx = out, out_idx, groups, 1


def reduce_comps(
    x: MetalFloat64, op: str, dims: Sequence[int], keepdim: bool
) -> tuple[torch.Tensor, ...]:
    """Reduce ``x`` over ``dims`` with kernel family ``op`` (components, out shape)."""
    flat, outer, n, inner = layout(x._comps, dims)
    res = run_reduce(x._mode, op, flat, outer, n, inner)
    count_gpu(op)
    shape = out_shape(x.shape, dims, keepdim)
    return tuple(c.reshape(shape) for c in res)


def reduce_value(
    x: MetalFloat64, op: str, dims: Any, keepdim: bool = False
) -> MetalFloat64:
    """Reduce ``x`` (any ``dim`` argument form) and wrap the result."""
    d = normalize_dims(dims, x.dim())
    return wrap(reduce_comps(x, op, d, keepdim), x._mode)


def reduced_count(x: MetalFloat64, dims: Sequence[int]) -> int:
    """Number of elements folded into each output element."""
    return math.prod(x.shape[d] for d in dims) if x.dim() else 1


# ---------------------------------------------------------------------------
# Argument plumbing
# ---------------------------------------------------------------------------
def _arg(args: Any, kwargs: Any, index: int, name: str, default: Any = None) -> Any:
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def named_args(func: Any, args: Any, kwargs: Any) -> dict[str, Any]:
    """Map every argument of the overload's schema to the value it received.

    Positional arguments are matched by position, keyword-only ones by name;
    arguments not passed take their schema default (``None`` when it has none).
    """
    out: dict[str, Any] = {}
    for i, a in enumerate(func._schema.arguments):
        if a.name in kwargs:
            out[a.name] = kwargs[a.name]
        elif not a.kwarg_only and i < len(args):
            out[a.name] = args[i]
        else:
            out[a.name] = a.default_value if a.has_default_value() else None
    return out


def out_tensors(func: Any, kwargs: Any) -> list[Any]:
    """The ``out=`` destination tensors of an overload, in schema order.

    torch names them per overload (``out`` for ``sum.IntList_out``, ``max`` and
    ``max_values`` for ``max.dim_max``, ...), so they are located through the
    schema's write-aliased keyword arguments rather than by a fixed name.
    """
    outs = []
    for a in func._schema.arguments:
        if a.alias_info is not None and a.alias_info.is_write and a.kwarg_only:
            t = kwargs.get(a.name)
            if t is not None:
                outs.append(t)
    return outs


def _check_float_dtype(op: str, dtype: Any, narrowing_ok: bool) -> None:
    """torch's ``dtype=`` rules for mean / vector_norm / norm.

    ``mean`` accepts any floating (or complex) dtype; ``linalg.vector_norm``
    and ``norm`` additionally refuse a narrowing conversion of the double input.
    """
    if dtype is None or dtype == torch.float64:
        return
    name = _DTYPE_NAMES.get(dtype, str(dtype).removeprefix("torch.").capitalize())
    if not (dtype.is_floating_point or dtype.is_complex):
        if op == "mean":
            raise RuntimeError(
                "mean(): could not infer output dtype. Optional dtype must be either "
                f"a floating point or complex dtype. Got: {name}"
            )
        raise RuntimeError(
            f"linalg.vector_norm: dtype should be floating point or complex, but got "
            f"{name}"
        )
    if not narrowing_ok and dtype != torch.complex128:
        raise RuntimeError(
            "linalg.vector_norm: the dtype of the input (Double) should be "
            f"convertible without narrowing to the specified dtype ({name})"
        )


_DTYPE_NAMES = {
    torch.bool: "Bool",
    torch.uint8: "Byte",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float32: "Float",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}


def _finish(result: Any, func: Any, kwargs: Any, dtype: Any = None) -> Any:
    """Apply a ``dtype`` request and the ``out=`` destinations to a handler result."""
    if dtype is not None and dtype != torch.float64:
        # Composite ops do not decompose below autograd, so call _to_copy directly.
        def cast(r: Any) -> Any:
            if isinstance(r, torch.Tensor) and r.dtype == torch.float64:
                return aten._to_copy.default(r, dtype=dtype)
            return r

        result = (
            tuple(cast(r) for r in result)
            if isinstance(result, tuple)
            else cast(result)
        )
    outs = out_tensors(func, kwargs)
    if not outs:
        return result
    results = result if isinstance(result, tuple) else (result,)
    for o, r in zip(outs, results, strict=True):
        check_out_device(o)
        if tuple(o.shape) != tuple(r.shape):
            o.resize_(r.shape)
        o.copy_(r)
    return tuple(outs) if isinstance(result, tuple) else outs[0]


def _check_nonempty(op: str, x: MetalFloat64, dims: Sequence[int]) -> None:
    for d in dims:
        if x.shape[d] == 0:
            raise RuntimeError(
                f"{op}(): Expected reduction dim {d} to have non-zero size."
            )


# ---------------------------------------------------------------------------
# sum / nansum / mean / prod
# ---------------------------------------------------------------------------
@implements(aten.sum.default, aten.sum.dim_IntList, aten.sum.IntList_out)
def _sum(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    dims = _arg(args, kwargs, 1, "dim")
    keepdim = bool(_arg(args, kwargs, 2, "keepdim", False))
    return _finish(
        reduce_value(x, "sum", dims, keepdim), func, kwargs, kwargs.get("dtype")
    )


@implements(aten.nansum.default, aten.nansum.out)
def _nansum(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    dims = _arg(args, kwargs, 1, "dim")
    keepdim = bool(_arg(args, kwargs, 2, "keepdim", False))
    return _finish(
        reduce_value(x, "nansum", dims, keepdim), func, kwargs, kwargs.get("dtype")
    )


def _mean_comps(
    x: MetalFloat64, dims: Sequence[int], keepdim: bool
) -> tuple[torch.Tensor, ...]:
    s = reduce_comps(x, "sum", dims, keepdim)
    return ew(x._mode, "div", s, scalar=float(reduced_count(x, dims)))


@implements(aten.mean.default, aten.mean.dim, aten.mean.out, aten.mean.dtype_out)
def _mean(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    _check_float_dtype("mean", kwargs.get("dtype"), narrowing_ok=True)
    dims = normalize_dims(_arg(args, kwargs, 1, "dim"), x.dim())
    keepdim = bool(_arg(args, kwargs, 2, "keepdim", False))
    return _finish(
        wrap(_mean_comps(x, dims, keepdim), x._mode), func, kwargs, kwargs.get("dtype")
    )


@implements(aten.prod.default, aten.prod.dim_int, aten.prod.int_out, aten.prod.out)
def _prod(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    dims = _arg(args, kwargs, 1, "dim")
    keepdim = bool(_arg(args, kwargs, 2, "keepdim", False))
    return _finish(
        reduce_value(x, "prod", dims, keepdim), func, kwargs, kwargs.get("dtype")
    )


@implements(aten.sum_to_size.default)
def _sum_to_size(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    size = tuple(int(s) for s in _arg(args, kwargs, 1, "size"))
    lead = x.dim() - len(size)
    if lead < 0:
        raise RuntimeError(f"size {size} is not expandable to size {tuple(x.shape)}")
    dims = list(range(lead)) + [
        lead + i for i, s in enumerate(size) if s == 1 and x.shape[lead + i] != 1
    ]
    comps = (
        reduce_comps(x, "sum", dims, True)
        if dims
        else tuple(c.clone() for c in x._comps)
    )
    return wrap(tuple(c.reshape(size) for c in comps), x._mode)


@implements(aten.trace.default, aten.trace.out)
def _trace(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    if x.dim() != 2:
        raise RuntimeError("trace: expected a matrix")
    diag = tuple(torch.diagonal(c).contiguous() for c in x._comps)
    res = run_reduce(x._mode, "sum", diag, 1, diag[0].numel(), 1)
    count_gpu("sum")
    return _finish(wrap(tuple(c.reshape(()) for c in res), x._mode), func, kwargs)


# ---------------------------------------------------------------------------
# amax / amin / max / min / argmax / argmin
# ---------------------------------------------------------------------------
@implements(aten.amax.default, aten.amax.out)
def _amax(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _amax_amin(func, "amax", "max", args, kwargs)


@implements(aten.amin.default, aten.amin.out)
def _amin(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _amax_amin(func, "amin", "min", args, kwargs)


def _amax_amin(func: Any, name: str, op: str, args: Any, kwargs: Any) -> Any:
    x = args[0]
    dims = normalize_dims(_arg(args, kwargs, 1, "dim", []), x.dim())
    keepdim = bool(_arg(args, kwargs, 2, "keepdim", False))
    _check_nonempty(name, x, dims)
    return _finish(wrap(reduce_comps(x, op, dims, keepdim), x._mode), func, kwargs)


@implements(aten.max.default, aten.max.unary_out)
def _max_all(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(_full_extremum("max", args[0]), func, kwargs)


@implements(aten.min.default, aten.min.unary_out)
def _min_all(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(_full_extremum("min", args[0]), func, kwargs)


def _full_extremum(op: str, x: MetalFloat64) -> MetalFloat64:
    if x.numel() == 0:
        raise RuntimeError(
            f"{op}(): Expected reduction dim to be specified for input.numel() == 0. "
            "Specify the reduction dim with the 'dim' argument."
        )
    return reduce_value(x, op, None, False)


@implements(aten.max.dim, aten.max.dim_max)
def _max_dim(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _extremum_dim(func, "max", "argmax", args, kwargs)


@implements(aten.min.dim, aten.min.dim_min)
def _min_dim(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _extremum_dim(func, "min", "argmin", args, kwargs)


def _extremum_dim(func: Any, name: str, op: str, args: Any, kwargs: Any) -> Any:
    x = args[0]
    dims = normalize_dims(int(_arg(args, kwargs, 1, "dim")), x.dim())
    keepdim = bool(_arg(args, kwargs, 2, "keepdim", False))
    _check_nonempty(name, x, dims)
    values, idx = _arg_reduce(x, op, dims, keepdim)
    return _finish((values, idx), func, kwargs)


def _arg_reduce(
    x: MetalFloat64, op: str, dims: Sequence[int], keepdim: bool
) -> tuple[MetalFloat64, torch.Tensor]:
    flat, outer, n, inner = layout(x._comps, dims)
    vals, idx = run_arg_reduce(x._mode, op, flat, outer, n, inner)
    count_gpu(op)
    shape = out_shape(x.shape, dims, keepdim)
    return wrap(tuple(c.reshape(shape) for c in vals), x._mode), idx.reshape(shape)


@implements(aten.argmax.default, aten.argmax.out)
def _argmax(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _argext(func, "argmax", args, kwargs)


@implements(aten.argmin.default, aten.argmin.out)
def _argmin(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _argext(func, "argmin", args, kwargs)


def _argext(func: Any, op: str, args: Any, kwargs: Any) -> Any:
    x = args[0]
    dim = _arg(args, kwargs, 1, "dim")
    keepdim = bool(_arg(args, kwargs, 2, "keepdim", False))
    if dim is None:
        if x.numel() == 0:
            raise RuntimeError(
                f"{op}(): Expected reduction dim to be specified for "
                "input.numel() == 0."
            )
        dims = normalize_dims(None, x.dim())
    else:
        # torch: only the *reduced* dim must be non-empty; an empty tensor whose
        # other dims are empty yields an empty index tensor.
        dims = normalize_dims(int(dim), x.dim())
        _check_nonempty(op, x, dims)
    _, idx = _arg_reduce(x, op, dims, keepdim)
    return _finish(idx, func, kwargs)


# ---------------------------------------------------------------------------
# all / any / count_nonzero
# ---------------------------------------------------------------------------
def nonzero_mask(x: MetalFloat64) -> torch.Tensor:
    """Bool tensor marking the elements whose value is not (+/-) zero (NaN counts).

    Evaluated on the bit patterns (sign bit masked off) rather than with a float
    comparison, so that a denormal float32 component, which the GPU's float
    compare would flush to zero, still counts as non-zero as it does in float64.
    """
    if x._mode == "df64":
        hi, lo = x._comps
        mag = torch.tensor(0x7FFFFFFF, dtype=torch.int32, device=hi.device)
        return ((hi.view(torch.int32) & mag) != 0) | ((lo.view(torch.int32) & mag) != 0)
    return (x._comps[0] << 1) != 0


def _bool_reduce(mask: torch.Tensor, fn: str, dims: Any, keepdim: bool) -> torch.Tensor:
    if isinstance(dims, (list, tuple)) and len(dims) == 0:
        # ``all.dims`` / ``any.dims`` with ``dim=[]`` reduce nothing in torch
        # (unlike ``sum``, whose empty list means every dim).
        return mask
    d = normalize_dims(dims, mask.dim())
    if not d:
        return getattr(torch, fn)(mask).reshape(mask.shape if keepdim else ())
    r = mask
    for dim in sorted(d, reverse=True):
        r = getattr(torch, fn)(r, dim=dim, keepdim=True)
    if not keepdim:
        for dim in sorted(d, reverse=True):
            r = r.squeeze(dim)
    return r


@implements(
    aten.all.default,
    aten.all.dim,
    aten.all.dims,
    aten.all.out,
    aten.all.dims_out,
    aten.all.all_out,
)
def _all(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    dims = _arg(args, kwargs, 1, "dim")
    keepdim = bool(_arg(args, kwargs, 2, "keepdim", False))
    return _finish(_bool_reduce(nonzero_mask(x), "all", dims, keepdim), func, kwargs)


@implements(
    aten.any.default,
    aten.any.dim,
    aten.any.dims,
    aten.any.out,
    aten.any.dims_out,
    aten.any.all_out,
)
def _any(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    dims = _arg(args, kwargs, 1, "dim")
    keepdim = bool(_arg(args, kwargs, 2, "keepdim", False))
    return _finish(_bool_reduce(nonzero_mask(x), "any", dims, keepdim), func, kwargs)


@implements(
    aten.count_nonzero.default,
    aten.count_nonzero.dim_IntList,
    aten.count_nonzero.out,
    aten.count_nonzero.dim_IntList_out,
)
def _count_nonzero(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    dims = _arg(args, kwargs, 1, "dim")
    mask = nonzero_mask(x).to(torch.int64)
    d = normalize_dims(dims, x.dim())
    res = mask.sum() if not d else mask.sum(dim=list(d))
    return _finish(res, func, kwargs)


# ---------------------------------------------------------------------------
# var / std / var_mean / std_mean
# ---------------------------------------------------------------------------
def _var_args(func: Any, args: Any, kwargs: Any) -> tuple[Any, float, bool]:
    """``(dim, correction, keepdim)`` of any ``var``/``std``/``*_mean`` overload.

    The legacy overloads carry ``unbiased`` (correction 1 or 0); the ``correction``
    overloads carry an optional scalar correction (``None`` means 1).
    """
    named = named_args(func, args, kwargs)
    if "correction" in named:
        c = named["correction"]
        correction = 1.0 if c is None else float(scalar_value(c))
    else:
        unbiased = named.get("unbiased")
        correction = 0.0 if unbiased is False else 1.0
    return named.get("dim"), correction, bool(named.get("keepdim") or False)


def var_comps(
    x: MetalFloat64, dims: Sequence[int], correction: float, keepdim: bool
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Variance and mean of ``x`` over ``dims`` (components, output shape).

    Computed on the GPU as ``sum((x - mean)^2) / max(n - correction, 0)`` with the
    mean kept in ``keepdim`` form for the broadcast subtraction.
    """
    mode = x._mode
    n = reduced_count(x, dims)
    mean_k = _mean_comps(x, dims, True)
    dev = ew(mode, "sub", x._comps, mean_k)
    sq = ew(mode, "mul", dev, dev)
    flat, outer, nn, inner = layout(sq, dims)
    ss = run_reduce(mode, "sum", flat, outer, nn, inner)
    count_gpu("sum")
    dof = max(float(n) - correction, 0.0)
    var = ew(mode, "div", ss, scalar=dof)
    shape = out_shape(x.shape, dims, keepdim)
    return tuple(c.reshape(shape) for c in var), tuple(c.reshape(shape) for c in mean_k)


@implements(
    aten.var.correction,
    aten.var.dim,
    aten.var.default,
    aten.var.correction_out,
    aten.var.out,
)
def _var(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    dims_raw, correction, keepdim = _var_args(func, args, kwargs)
    var, _ = var_comps(x, normalize_dims(dims_raw, x.dim()), correction, keepdim)
    return _finish(wrap(var, x._mode), func, kwargs)


@implements(
    aten.std.correction,
    aten.std.dim,
    aten.std.default,
    aten.std.correction_out,
    aten.std.out,
)
def _std(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    dims_raw, correction, keepdim = _var_args(func, args, kwargs)
    var, _ = var_comps(x, normalize_dims(dims_raw, x.dim()), correction, keepdim)
    return _finish(wrap(ew(x._mode, "sqrt", var), x._mode), func, kwargs)


@implements(
    aten.var_mean.correction,
    aten.var_mean.dim,
    aten.var_mean.default,
    aten.var_mean.correction_out,
)
def _var_mean(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    dims_raw, correction, keepdim = _var_args(func, args, kwargs)
    var, mean = var_comps(x, normalize_dims(dims_raw, x.dim()), correction, keepdim)
    return _finish((wrap(var, x._mode), wrap(mean, x._mode)), func, kwargs)


@implements(
    aten.std_mean.correction,
    aten.std_mean.dim,
    aten.std_mean.default,
    aten.std_mean.correction_out,
)
def _std_mean(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    dims_raw, correction, keepdim = _var_args(func, args, kwargs)
    var, mean = var_comps(x, normalize_dims(dims_raw, x.dim()), correction, keepdim)
    std = wrap(ew(x._mode, "sqrt", var), x._mode)
    return _finish((std, wrap(mean, x._mode)), func, kwargs)


# ---------------------------------------------------------------------------
# norms
# ---------------------------------------------------------------------------
def vector_norm_comps(
    x: MetalFloat64, ord: float, dims: Sequence[int], keepdim: bool
) -> tuple[torch.Tensor, ...]:
    """``linalg.vector_norm`` of ``x`` over ``dims`` as components in output shape.

    ``ord`` 2 is ``sqrt(sum(x*x))`` (no scaling, like torch CPU; in df64 the squares
    overflow float32 beyond |x| ~ 1.8e19), ``inf``/``-inf`` are NaN-propagating
    max/min of |x|, 0 counts non-zeros, 1 sums |x| and any other ``p`` uses the
    ``pow`` kernel.
    """
    mode = x._mode
    shape = out_shape(x.shape, dims, keepdim)

    def reduce_over(
        comps: tuple[torch.Tensor, ...], op: str
    ) -> tuple[torch.Tensor, ...]:
        flat, outer, n, inner = layout(comps, dims)
        r = run_reduce(mode, op, flat, outer, n, inner)
        count_gpu(op)
        return tuple(c.reshape(shape) for c in r)

    if ord == 2.0:
        return ew(mode, "sqrt", reduce_over(ew(mode, "mul", x._comps, x._comps), "sum"))
    a = ew(mode, "abs", x._comps)
    if math.isinf(ord):
        if x.numel() == 0:
            raise RuntimeError(
                "linalg.vector_norm cannot compute the "
                f"{'inf' if ord > 0 else '-inf'} norm on an empty tensor because "
                "the operation does not have an identity"
            )
        return reduce_over(a, "max" if ord > 0 else "min")
    if ord == 1.0:
        return reduce_over(a, "sum")
    if ord == 0.0:
        cnt = nonzero_mask(x).to(torch.int64)
        d = list(dims)
        cnt = cnt.sum(dim=d, keepdim=True) if d else cnt
        return tuple(c.reshape(shape) for c in coerce(cnt, mode)._comps)
    p = ew(mode, "pow", a, scalar=ord)
    return ew(mode, "pow", reduce_over(p, "sum"), scalar=1.0 / ord)


@implements(aten.linalg_vector_norm.default, aten.linalg_vector_norm.out)
def _linalg_vector_norm(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    _check_float_dtype("linalg_vector_norm", kwargs.get("dtype"), narrowing_ok=False)
    ord = scalar_value(_arg(args, kwargs, 1, "ord", 2))
    dims = normalize_dims(_arg(args, kwargs, 2, "dim"), x.dim())
    keepdim = bool(_arg(args, kwargs, 3, "keepdim", False))
    if (
        ord not in (2.0, 1.0, 0.0)
        and not math.isinf(ord)
        and not has_op(x._mode, "pow")
    ):
        return cpu_fallback(
            func, args, kwargs, label="linalg_vector_norm(pow kernel missing)"
        )
    return _finish(
        wrap(vector_norm_comps(x, ord, dims, keepdim), x._mode),
        func,
        kwargs,
        kwargs.get("dtype"),
    )


@implements(
    aten.norm.Scalar,
    aten.norm.ScalarOpt_dim,
    aten.norm.ScalarOpt_dtype,
    aten.norm.ScalarOpt_dim_dtype,
    aten.norm.Scalar_out,
    aten.norm.out,
    aten.norm.dtype_out,
    aten.norm.ScalarOpt_dtype_out,
)
def _norm(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    named = named_args(func, args, kwargs)
    p = named.get("p")
    ord = 2.0 if p is None else scalar_value(p)
    dims = normalize_dims(named.get("dim"), x.dim())
    keepdim = bool(named.get("keepdim") or False)
    dtype = named.get("dtype")
    _check_float_dtype("norm", dtype, narrowing_ok=False)
    if (
        ord not in (2.0, 1.0, 0.0)
        and not math.isinf(ord)
        and not has_op(x._mode, "pow")
    ):
        return cpu_fallback(func, args, kwargs, label="norm(pow kernel missing)")
    return _finish(
        wrap(vector_norm_comps(x, ord, dims, keepdim), x._mode), func, kwargs, dtype
    )


@implements(aten.logsumexp.default, aten.logsumexp.out)
def _logsumexp(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    mode = x._mode
    dims = normalize_dims(_arg(args, kwargs, 1, "dim"), x.dim())
    keepdim = bool(_arg(args, kwargs, 2, "keepdim", False))
    if not (has_op(mode, "exp") and has_op(mode, "log")):
        return cpu_fallback(
            func, args, kwargs, label="logsumexp(exp/log kernels missing)"
        )
    shape = out_shape(x.shape, dims, keepdim)
    if reduced_count(x, dims) == 0:
        res = MetalFloat64.from_numpy(np.full(shape, -math.inf), mode, host=False)
        return _finish(res, func, kwargs)
    m = reduce_comps(x, "max", dims, True)
    zeros = tuple(torch.zeros_like(c) for c in m)
    m = ew(mode, "where", ew(mode, "isinf", m), zeros, m)
    e = ew(mode, "exp", ew(mode, "sub", x._comps, m))
    flat, outer, n, inner = layout(e, dims)
    s = run_reduce(mode, "sum", flat, outer, n, inner)
    count_gpu("sum")
    s = tuple(c.reshape(m[0].shape) for c in s)
    r = ew(mode, "add", ew(mode, "log", s), m)
    return _finish(wrap(tuple(c.reshape(shape) for c in r), mode), func, kwargs)


# ---------------------------------------------------------------------------
# cumsum / cumprod
# ---------------------------------------------------------------------------
@implements(aten.cumsum.default, aten.cumsum.out)
def _cumsum(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    mode = x._mode
    dim = int(_arg(args, kwargs, 1, "dim"))
    dtype = kwargs.get("dtype")
    if x.dim() == 0:
        normalize_dims(dim, 0)
        return _finish(
            wrap(tuple(c.clone() for c in x._comps), mode), func, kwargs, dtype
        )
    (d,) = normalize_dims(dim, x.dim())
    shape = tuple(x.shape)
    outer = math.prod(shape[:d])
    n = shape[d]
    inner = math.prod(shape[d + 1 :])
    flat = tuple(c.contiguous() for c in x._comps)
    out = _alloc(mode, x.numel())
    if outer * inner > 0:
        getattr(reduce_kernels(mode), f"cumsum_{mode}")(
            *flat, *out, outer, n, inner, threads=[outer * inner, 1, 1]
        )
    count_gpu("cumsum")
    return _finish(
        wrap(tuple(c.reshape(shape) for c in out), mode), func, kwargs, dtype
    )


@implements(aten.cumprod.default, aten.cumprod.out)
def _cumprod(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return cpu_fallback(func, args, kwargs, label="cumprod")


def _inplace_cumulative(functional: Any) -> Any:
    """Handler for ``cumsum_`` / ``cumprod_``: the functional op copied into self."""

    def handler(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        x = args[0]
        dim = _arg(args, kwargs, 1, "dim")
        dtype = kwargs.get("dtype")
        if dtype is not None and dtype != torch.float64:
            raise RuntimeError(
                "provided dtype must match the dtype of self tensor in "
                f"{func._schema.name}"
            )
        res = functional(x, dim)
        for d, c in zip(x._comps, res._comps, strict=True):
            d.copy_(c)
        return x

    return handler


implements(aten.cumsum_.default)(_inplace_cumulative(aten.cumsum.default))
implements(aten.cumprod_.default)(_inplace_cumulative(aten.cumprod.default))


__all__ = [
    "ELEMS_PER_GROUP",
    "MAX_GROUPS",
    "SECOND_PASS",
    "TG",
    "ew",
    "has_op",
    "layout",
    "named_args",
    "nonzero_mask",
    "normalize_dims",
    "out_shape",
    "out_tensors",
    "reduce_comps",
    "reduce_kernels",
    "reduce_value",
    "reduced_count",
    "run_arg_reduce",
    "run_reduce",
    "var_comps",
    "vector_norm_comps",
]
