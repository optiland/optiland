"""MetalFloat64 dispatch handlers: structural ops (registered via tensor.implements).

Structural ops never look at the numeric value of an element, so they are exact
by construction: every view, selection, concatenation, permutation or copy is
applied to the plain component tensors (``(hi, lo)`` for df64, ``(bits,)`` for
sf64) and the results are rewrapped. Three families need more than that:

* **Ordering** (``sort``, ``argsort``, ``topk``, ``searchsorted``, ``bucketize``):
  a value is mapped to an int64 *sort key* that is monotone in the represented
  value (lexicographic ``(hi, lo)`` on sortable-int views of the float32 words
  for df64, the sign-folded bit pattern for sf64; ``-0 == +0`` and every NaN is
  the largest key, matching torch's CPU sort order). Sorting or searching the
  keys with native MPS int64 kernels is then exact, and the values are gathered
  by the resulting indices. ``searchsorted`` / ``bucketize`` map a NaN in the
  *sorted sequence* to the smallest key instead: torch's binary search compares
  with ``!(mid >= val)``, so a NaN boundary sends every probe to its right, and
  the MPS int64 search follows the same probe path (a NaN *value* keeps the
  largest key and lands at the end, as on the CPU).
  Keys order the *represented* df64 value: a canonical ``(hi, lo)`` pair can
  carry bits below float64's 2^-53 (``exp(x)`` for ``|x| ~ 1e-12`` yields
  ``(1.0, x)``), so two tensors that decode to the same float64 may still sort,
  search and compare (``eq``/``lt`` kernels) as distinct on the GPU. No strict
  inversion is possible; only ties are broken differently from a CPU oracle run
  on the decoded values.
* **Scalar-valued fills** (``fill_``, ``masked_fill``, ``scatter.value``,
  ``index_fill``): the Python scalar is encoded on the host into one exact
  scalar per component and the op runs once per component with its own scalar.
* **Accumulating writes** (``index_put(accumulate=True)``, ``scatter_add``,
  ``index_add``): float32 ``index_add_`` on the components would add ``hi`` and
  ``lo`` separately and destroy the pair. Instead the destinations are turned
  into linear indices, duplicates are summed with the elementwise ``add``
  kernel by a pairwise tree (``log2(max multiplicity)`` rounds), the existing
  values are gathered, added with the same kernel, and written back with a
  plain non-accumulating ``index_put``. Every addition is correctly rounded
  in the representation, but the result is *not bit-identical to torch CPU*:
  torch adds duplicates sequentially into the destination (a different
  association; a tree keeps the launch count logarithmic in the multiplicity)
  and ``index_add_(alpha=)`` scales the source before the additions where
  torch's CPU loop fuses ``alpha * src + self`` in one rounding, so sf64
  results can differ from torch CPU by one ulp (df64 by its usual ``u^2``).
  Non-accumulating writes with
  duplicate integer indices (``index_put``, ``put_``, ``scatter``) are
  deduplicated to *last write wins* (torch CPU order) so that both components
  always take the same winner; ``index_copy`` uses the same path for the same
  reason. Index tensors are validated as torch does before any write: a
  ``MetalFloat64`` in an index / mask / sorter / condition slot raises (its
  components are never used as indices), and ``scatter`` / ``scatter_add`` /
  ``index_add`` / ``index_copy`` / ``index_fill`` / ``put`` reject out-of-range
  and mis-shaped indices with torch's messages instead of writing elsewhere.

Lenient operands (deliberate, documented here): the structural writes and
concatenations (``cat``, ``stack``, ``where``, ``fill_``, ``masked_fill``,
``masked_scatter``, ``index_put``, ``scatter``, ``scatter_add``, ``index_add``,
``index_copy``, ``put``, ``__setitem__``) accept plain operands of any device
and dtype and promote them exactly into the target's representation, where
torch raises its dtype-mismatch (``Index put requires the source and
destination dtypes match``) or device errors; a MetalFloat64 source written
into a *plain* float32 / float16 self is demoted (rounded) to that dtype where
torch refuses the mixed dtypes. Index, mask, sorter and ``repeats`` tensors
of any op may live on the CPU (``scatter``, ``index_add``, ``index_copy``,
``put``, ``searchsorted(sorter=)``, ``index_select``, ``gather``,
``masked_fill``, ``index_fill``, ...): they are moved to mps, where torch
raises a device mismatch for some of them, because the dual-residency host
path returns its indices and predicates on the CPU by design. Index dtypes
are checked as torch does (``take`` / ``put_`` / ``index_copy_`` /
``index_fill_`` long only, the rest int32 or int64), but three shape checks
are looser than torch's: ``index_add_`` into a 0-d self accepts a 1-element
source of any rank, ``index_put_((tensor(True),), v)`` on a 0-d self writes
the element (torch: ``too many indices``), and the ``out=`` of
``searchsorted`` / ``bucketize`` may be int32 (written by ``copy_``) where
torch demands the output dtype to match ``out_int32``.

Overlap: in-place writers into an internally overlapping target (an
``expand``-ed, stride-0 tensor) raise torch's ``more than one element of the
written-to tensor refers to a single memory location`` (``_check_no_overlap``;
the per-component plain ops check it themselves). A *source* that partially
overlaps the written tensor (``x.index_copy_(0, i, x[0:2])``, ``x.put_(i,
x[5])``, ``x.index_add_(0, i, x)``, ``x.masked_scatter_(m, x.flip(0))``) and an
``out=`` that aliases an input (``torch.index_select(x, 0, i, out=x)``,
``torch.cat([x, x], out=x)``) are accepted with *snapshot* semantics: the
source is read (cloned when it shares storage with the target, or the
functional result computed) before anything is written, so the result is
what torch computes from a copy of the source. torch CPU and plain mps raise
``unsupported operation: some elements of the input tensor and the
written-to tensor refer to a single memory location`` for these, and for a
partially overlapping ``scatter_add_`` source (which torch accepts) the CPU
reads the already-modified source sequentially, so its values differ from
the snapshot result. A 0-d bool index (``index_put((tensor(True),), v)``)
follows torch: ``True`` selects everything at its position, ``False``
nothing; the raw MPS kernels abort the process on that form, so it never
reaches them. A uint8 ``where`` condition is accepted with torch's
deprecation warning. An empty ``scatter`` / ``scatter_add`` index of any rank
writes nothing, as in torch.

The backward helpers torch emits for these ops (``select_backward``,
``slice_backward``, ``diagonal_backward``, ``masked_select_backward``,
``masked_scatter_backward``, ``value_selecting_reduction_backward``,
``index_select_backward``, ``unfold_backward``) are implemented here with
locally built zero components so that autograd through views works without the
creation lane.

``out=`` overloads (``sort``, ``topk``, ``searchsorted``, ``bucketize``,
``index_select``, ``gather``, ``masked_select``, ``index_add``, ``scatter``,
``scatter_add``, ``index_copy``, ``take``, ``cat``, ``stack``, ``where``,
``nonzero``) run the functional op and copy into ``out``, resizing it to the
result shape first (a MetalFloat64 ``out`` is resized in place by the nucleus).

In-place metadata view ops (``squeeze_``, ``unsqueeze_``, ``t_``, ``transpose_``,
``as_strided_``) are applied per component too; ``return_and_correct_aliasing``
propagates the new sizes/strides to the wrapper.

Note on the backward of ``where`` (scalar branch), ``index_put``, ``put``, the
``*_scatter`` family and partially used ``split`` / ``unbind`` / ``chunk``
outputs: autograd materializes zeros / scalar constants with tensor-less
factories (``aten.zeros`` / ``scalar_tensor`` with ``dtype=float64,
device=mps``) that carry no MetalFloat64 operand, so they are served by the
nucleus's ``MetalFactoryMode`` in the *global* representation. Those tensors
are representation-agnostic (``tensor.py`` module docstring) and are re-tagged
to the gradient's mode when they meet it here, so differentiating a tensor of
the non-global representation works.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from torch.utils._pytree import tree_leaves, tree_map

from optiland.backend.torch_backend.metal import encode
from optiland.backend.torch_backend.metal import tensor as _nucleus
from optiland.backend.torch_backend.metal.tensor import (
    MetalFloat64,
    apply_to_components,
    aten,
    count_gpu,
    implements,
    is_metal,
    library,
    mode_of,
    scalar_like,
    scalar_value,
    wrap,
)
from optiland.backend.torch_backend.metal.tensor import coerce as _coerce

_INT32_MAX = 0x7FFFFFFF
_INT64_MAX = 0x7FFFFFFFFFFFFFFF
_INT64_MIN = -(1 << 63)
_F64_INF_BITS = 0x7FF0000000000000
_MPS = torch.device("mps")

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def coerce(x: Any, mode: str) -> MetalFloat64:
    """Promote a value operand of a structural op (CPU tensors accepted).

    Structural writes and concatenations take plain operands of any device
    and dtype and promote them exactly (module docstring, "lenient operands");
    only the elementwise / reduction / linalg lanes raise torch's device error
    for a CPU tensor with ``dim() > 0``.
    """
    return _coerce(x, mode, allow_cpu=True)


def _bind(func: Any, args: Any, kwargs: Any) -> dict[str, Any]:
    """Bind ``args``/``kwargs`` to the parameter names of ``func``'s schema."""
    bound: dict[str, Any] = {}
    for i, arg in enumerate(func._schema.arguments):
        if not arg.kwarg_only and i < len(args):
            bound[arg.name] = args[i]
        elif arg.name in kwargs:
            bound[arg.name] = kwargs[arg.name]
        else:
            bound[arg.name] = arg.default_value
    return bound


def _mode(*xs: Any) -> str:
    return mode_of(*[x for x in tree_leaves(list(xs)) if is_metal(x)])


def _ncomps(mode: str) -> int:
    return 2 if mode == "df64" else 1


def _scalar_comps(value: Any, mode: str) -> tuple[Any, ...]:
    """Encode a scalar operand into one exact Python scalar per component."""
    s = scalar_value(value)
    if s is None:
        if is_metal(value) and value.dim() == 0:
            s = float(value.to_numpy().reshape(()))
        elif isinstance(value, torch.Tensor) and value.dim() == 0:
            s = float(value.detach().cpu().to(torch.float64).item())
        else:
            raise TypeError(f"expected a scalar operand, got {type(value).__name__}")
    if mode == "df64":
        return tuple(encode.df64_scalar(s))
    return (encode.sf64_scalar(s),)


def _scalar_fill(
    shape: tuple[int, ...], value: float, mode: str
) -> tuple[torch.Tensor, ...]:
    """Contiguous device components of ``shape`` filled with ``value`` exactly."""
    dtype = torch.float32 if mode == "df64" else torch.int64
    return tuple(
        torch.full(shape, v, dtype=dtype, device=_MPS)
        for v in _scalar_comps(value, mode)
    )


def _rewrap(outs: list[Any], mode: str) -> Any:
    first = outs[0]
    if isinstance(first, torch.Tensor):
        return wrap(tuple(outs), mode)
    if isinstance(first, (list, tuple)):
        return type(first)(
            wrap(tuple(o[i] for o in outs), mode)
            if isinstance(first[i], torch.Tensor)
            else first[i]
            for i in range(len(first))
        )
    return first


def _apply(
    func: Any,
    args: Any,
    kwargs: Any,
    scalar_positions: tuple[int, ...] = (),
    mode: str | None = None,
) -> Any:
    """Like :func:`apply_to_components` but with per-component scalar operands.

    Args:
        func: The aten overload to run on the components.
        args: Positional arguments (MetalFloat64 leaves are unwrapped per call).
        kwargs: Keyword arguments (same treatment).
        scalar_positions: Positions in ``args`` holding a Scalar operand that must
            be encoded separately for every component (``fill_``, ``masked_fill``).
        mode: Representation, inferred from the MetalFloat64 operands by default.

    Returns:
        The rewrapped result with aliasing corrected for views and in-place ops.
    """
    mode = mode or _mode(args, kwargs)
    args = tuple(args)
    scalars = {p: _scalar_comps(args[p], mode) for p in scalar_positions}
    outs = []
    for k in range(_ncomps(mode)):

        def pick(a: Any, _k: int = k) -> Any:
            return a._comps[_k] if is_metal(a) else a

        call_args = [
            scalars[i][k] if i in scalars else tree_map(pick, a)
            for i, a in enumerate(args)
        ]
        outs.append(func(*call_args, **tree_map(pick, kwargs)))
    return return_and_correct_aliasing(func, args, kwargs, _rewrap(outs, mode))


def _writes_first_arg(func: Any) -> bool:
    schema = getattr(func, "_schema", None)
    if schema is None or not schema.arguments:
        return False
    info = schema.arguments[0].alias_info
    return info is not None and info.is_write


def _host_mutation(func: Any, args: Any, kwargs: Any) -> Any:
    """Run an in-place op whose target is host-resident on the host, else return None.

    The nucleus routes an in-place op to the host only when *every* operand is
    small; a host-resident target combined with a large operand (a long index
    tensor, a big ``values`` tensor) reaches the GPU handlers, where mutating
    the target's lazily encoded components would be lost. Delegating to the
    nucleus's host path keeps the host copy authoritative.
    """
    if not args or not _writes_first_arg(func):
        return None
    x = args[0]
    if not (is_metal(x) and getattr(x, "_host", None) is not None):
        return None
    run = getattr(_nucleus, "_run_on_host", None)
    if run is None:  # pragma: no cover - nucleus without residency support
        raise RuntimeError("host-resident target in a GPU handler and no host path")
    return run(func, args, kwargs)


def _zeros(shape: Any, mode: str) -> MetalFloat64:
    """Zero-valued MetalFloat64 built from plain components (no creation lane)."""
    if mode == "df64":
        hi = torch.zeros(tuple(shape), dtype=torch.float32, device=_MPS)
        return wrap((hi, torch.zeros_like(hi)), mode)
    return wrap((torch.zeros(tuple(shape), dtype=torch.int64, device=_MPS),), mode)


def _demote(x: Any, like: torch.Tensor) -> Any:
    """Convert a MetalFloat64 to a plain tensor with ``like``'s dtype and device."""
    if not is_metal(x):
        return x
    return x.to_cpu_float64().to(like.device, like.dtype)


def _same_storage(a: MetalFloat64, b: MetalFloat64) -> bool:
    return (
        a._comps[0].untyped_storage().data_ptr()
        == b._comps[0].untyped_storage().data_ptr()
    )


def _launch(
    mode: str, op: str, *operands: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, ...]:
    """Run an elementwise kernel on contiguous component tuples."""
    lib = library(mode)
    args = [c if mode == "df64" else c[0] for c in operands]
    res = lib.launch(op, *args)
    count_gpu(op)
    return tuple(res) if isinstance(res, (tuple, list)) else (res,)


def _contig(comps: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(c.contiguous() for c in comps)


def _with_out(result: Any, out: Any) -> Any:
    """Write a functional result into an ``out=`` tensor (plain or MetalFloat64).

    ``out`` is resized to the result shape first, as torch does.
    """
    _nucleus.check_out_device(out)
    _nucleus.resize_out(out, tuple(result.shape))
    out.copy_(result)
    return out


def _out_variant(functional: Any, *out_names: str) -> Any:
    """Handler for an ``out=`` overload: run ``functional`` and copy into the outs."""

    def handler(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        b = _bind(func, args, kwargs)
        outs = [b.pop(n) for n in out_names]
        positional = [
            b[a.name] for a in functional._schema.arguments if not a.kwarg_only
        ]
        keyword = {
            a.name: b[a.name] for a in functional._schema.arguments if a.kwarg_only
        }
        res = functional(*positional, **keyword)
        if len(out_names) == 1:
            return _with_out(res, outs[0])
        return tuple(_with_out(r, o) for r, o in zip(res, outs, strict=True))

    return handler


# ---------------------------------------------------------------------------
# Index / mask validation (torch's checks, before anything touches components)
# ---------------------------------------------------------------------------
_INT_INDEX = (torch.int64, torch.int32)


def _dtype_name(t: torch.Tensor) -> str:
    return "Double" if is_metal(t) else str(t.dtype).removeprefix("torch.").capitalize()


def _check_index(
    index: Any, message: str, exc: type = RuntimeError, long_only: bool = False
) -> None:
    """Raise ``exc(message)`` unless ``index`` is a plain int32/int64 tensor.

    ``long_only`` restricts it to int64 (``take``, ``put_``, ``index_copy_``,
    ``index_fill_``, which torch limits to long indices).
    """
    allowed = (torch.int64,) if long_only else _INT_INDEX
    if is_metal(index) or index.dtype not in allowed:
        raise exc(message.format(dtype=_dtype_name(index)))


def _on_device(index: Any) -> Any:
    """A plain index / mask tensor on mps (the host path produces them on the CPU)."""
    if isinstance(index, torch.Tensor) and index.device.type != "mps":
        return index.to(_MPS)
    return index


def _check_mask(mask: Any, message: str) -> None:
    if is_metal(mask) or mask.dtype != torch.bool:
        raise RuntimeError(message.format(dtype=_dtype_name(mask)))


def _check_range(index: torch.Tensor, size: int, dim: int, wrap: bool = False) -> None:
    """Raise like torch when an index is outside ``[0, size)`` (``[-size, size)``)."""
    if index.numel() == 0:
        return
    lo, hi = torch.stack(torch.aminmax(index)).tolist()
    bad = None
    if hi >= size:
        bad = hi
    elif lo < (-size if wrap else 0):
        bad = lo
    if bad is not None:
        raise RuntimeError(
            f"index {bad} is out of bounds for dimension {dim} with size {size}"
        )


# ---------------------------------------------------------------------------
# Views and shape ops: exact, applied per component
# ---------------------------------------------------------------------------
_VIEW_OPS = (
    aten.view.default,
    aten._unsafe_view.default,
    aten.reshape.default,
    aten.as_strided.default,
    aten.select.int,
    aten.slice.Tensor,
    aten.permute.default,
    aten.transpose.int,
    aten.t.default,
    aten.unsqueeze.default,
    aten.squeeze.default,
    aten.squeeze.dim,
    aten.squeeze.dims,
    aten.expand.default,
    aten.expand_as.default,
    aten.broadcast_to.default,
    aten.narrow.default,
    aten.split.Tensor,
    aten.split_with_sizes.default,
    aten.unsafe_split.Tensor,
    aten.unsafe_split_with_sizes.default,
    aten.unsafe_chunk.default,
    aten.unbind.int,
    aten.chunk.default,
    aten.flip.default,
    aten.roll.default,
    aten.repeat.default,
    aten.repeat_interleave.self_int,
    aten.repeat_interleave.self_Tensor,
    aten.tile.default,
    aten.diagonal.default,
    aten.movedim.int,
    aten.movedim.intlist,
    aten.unfold.default,
    aten.index_select.default,
    aten.gather.default,
    aten.masked_select.default,
    aten.zero_.default,
    # In-place metadata mutations: ``return_and_correct_aliasing`` copies the
    # mutated component metadata back onto the wrapper (verified for torch 2.14).
    aten.squeeze_.default,
    aten.squeeze_.dim,
    aten.squeeze_.dims,
    aten.unsqueeze_.default,
    aten.t_.default,
    aten.transpose_.default,
    aten.swapaxes_.default,
    aten.swapdims_.default,
    aten.as_strided_.default,
    # Zero-filling structural ops used by matmul's 1-D decomposition and the
    # derivative formulas of det / slogdet / cholesky / qr / householder_product.
    aten.diag_embed.default,
    aten.tril.default,
    aten.triu.default,
    aten.tril_.default,
    aten.triu_.default,
    # Copying variants of the views (``torch.diag`` of a matrix decomposes to
    # ``diagonal_copy``; functionalization emits the others).
    aten.diagonal_copy.default,
    aten.narrow_copy.default,
    aten.select_copy.int,
    aten.slice_copy.Tensor,
    aten.expand_copy.default,
    aten.view_copy.default,
    aten.permute_copy.default,
    aten.t_copy.default,
    aten.transpose_copy.int,
    aten.squeeze_copy.default,
    aten.squeeze_copy.dim,
    aten.squeeze_copy.dims,
    aten.unsqueeze_copy.default,
    aten.alias_copy.default,
    aten.detach_copy.default,
    aten.as_strided_copy.default,
    aten.unbind_copy.int,
    aten.split_copy.Tensor,
    aten.split_with_sizes_copy.default,
    aten.unfold_copy.default,
    # Compositions of flips / transposes / zero padding (exact per component).
    aten.rot90.default,
    aten.block_diag.default,
)


_INDEX_MESSAGE = "tensors used as indices must be long, int, byte or bool tensors"


def _check_indices(indices: Any) -> None:
    """``aten.index`` / ``index_put`` index list: plain int / byte / bool only."""
    for ind in indices:
        if ind is None:
            continue
        if is_metal(ind) or ind.dtype not in (*_INT_INDEX, torch.uint8, torch.bool):
            raise IndexError(_INDEX_MESSAGE)


#: Position of the index / mask operand the per-component op needs on mps.
_DEVICE_INDEX_POS = {
    aten.index_select.default: 2,
    aten.gather.default: 2,
    aten.masked_select.default: 1,
    aten.repeat_interleave.self_Tensor: 1,
}


@implements(*_VIEW_OPS)
def _structural(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    if func is aten.index_select.default:
        _check_index(args[2], "index_select(): Expected dtype int32 or int64 for index")
    elif func is aten.gather.default:
        _check_index(args[2], "gather(): Expected dtype int32/int64 for index")
    elif func is aten.masked_select.default:
        _check_mask(args[1], "masked_select: expected BoolTensor for mask")
    elif func is aten.repeat_interleave.self_Tensor:
        _check_index(args[1], "repeats has to be Long or Int tensor")
    if (host := _host_mutation(func, args, kwargs)) is not None:
        return host
    x = args[0] if args else None
    if (
        is_metal(x)
        and getattr(x, "_host", None) is not None
        and _nucleus._returns_alias(func)
    ):
        # A view of a host-resident tensor larger than the residency threshold
        # (a host ``cat`` result, an expanded per-surface constant) must alias
        # the host tensor, not its lazily encoded GPU copy: a write through a
        # view of the encoding cache would be lost.
        return _nucleus._run_on_host(func, args, kwargs)
    pos = _DEVICE_INDEX_POS.get(func)
    if pos is not None and len(args) > pos:
        # torch (CPU and plain mps) accepts a CPU index / mask here, and the
        # dual-residency host path returns its indices on the CPU.
        args = (*args[:pos], _on_device(args[pos]), *args[pos + 1 :])
    return apply_to_components(func, args, kwargs)


@implements(aten.index.Tensor, aten._unsafe_index.Tensor)
def _index(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    _check_indices(args[1])
    x, indices = args[0], _prep_indices(args[1])
    if any(
        isinstance(i, torch.Tensor) and i.dtype == torch.bool and i.dim() == 0
        for i in indices
    ):
        # 0-d True selects everything at its position under a new leading
        # dimension, 0-d False selects nothing (an empty leading dimension).
        kept = _drop_scalar_bools(indices)
        sel = [
            i
            for i in indices
            if not (
                isinstance(i, torch.Tensor) and i.dtype == torch.bool and i.dim() == 0
            )
        ]
        res = apply_to_components(func, (x, sel), kwargs) if sel else x
        comps = tuple(c.unsqueeze(0).clone() for c in res._comps)
        if kept is None:
            comps = tuple(c[:0] for c in comps)
        return wrap(comps, res._mode)
    return apply_to_components(func, (x, indices), kwargs)


@implements(
    aten.slice_scatter.default,
    aten.select_scatter.default,
    aten.diagonal_scatter.default,
    aten.as_strided_scatter.default,
)
def _view_scatter(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    """``*_scatter``: a copy of ``self`` with ``src`` written into the named view."""
    x, src = args[0], args[1]
    if not is_metal(x):
        return func(x, _demote(src, x), *args[2:], **kwargs)
    return apply_to_components(func, (x, coerce(src, _mode(x, src)), *args[2:]), kwargs)


@implements(aten.take.default)
def _take(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    """``take`` has no MPS kernel: gather from the flattened components instead."""
    x, index = args
    _check_index(
        index,
        "take(): Expected a long tensor for index, but got {dtype}",
        long_only=True,
    )
    if index.device.type != "mps":
        index = index.to(_MPS)
    return wrap(tuple(c.reshape(-1)[index] for c in x._comps), x._mode)


# ---------------------------------------------------------------------------
# cat / stack (mixed plain operands are promoted exactly first)
# ---------------------------------------------------------------------------
@implements(aten.cat.default, aten.stack.default)
def _cat(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    tensors = list(args[0])
    if tensors and all(is_metal(t) and t._host is not None for t in tensors):
        # Host-only inputs whose total exceeds the residency threshold (a
        # pupil distribution assembled ring by ring): concatenate on the host
        # and encode the result once when it first meets a GPU operand,
        # instead of encoding every piece.
        return _nucleus._run_on_host(func, args, kwargs)
    mode = _mode(tensors)
    coerced = [coerce(t, mode) for t in tensors]
    return apply_to_components(func, (coerced, *args[1:]), kwargs)


@implements(aten.cat.out, aten.stack.out)
def _cat_out(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    kwargs = dict(kwargs)
    out = kwargs.pop("out")
    functional = aten.cat.default if func is aten.cat.out else aten.stack.default
    return _with_out(functional(*args, **kwargs), out)


# ---------------------------------------------------------------------------
# where / masked ops / fills
# ---------------------------------------------------------------------------
@implements(
    aten.where.self, aten.where.ScalarSelf, aten.where.ScalarOther, aten.where.Scalar
)
def _where(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    cond, a, b = args
    if (
        isinstance(cond, torch.Tensor)
        and not is_metal(cond)
        and cond.dtype == torch.uint8
    ):
        warnings.warn(
            "where received a uint8 condition tensor. This behavior is deprecated "
            "and will be removed in a future version of PyTorch. Use a boolean "
            "condition instead.",
            UserWarning,
            stacklevel=2,
        )
        cond = cond.to(torch.bool)
    _check_mask(
        cond,
        "where expected condition to be a boolean tensor, but got a tensor with "
        "dtype {dtype}",
    )
    mode = _mode(a, b)
    if isinstance(cond, torch.Tensor) and cond.device.type != "mps":
        cond = cond.to(_MPS)
    shape = torch.broadcast_shapes(
        *(tuple(v.shape) for v in (cond, a, b) if isinstance(v, torch.Tensor))
    )
    sa, sb = scalar_like(a, shape), scalar_like(b, shape)
    if sa is not None and sb is not None and isinstance(cond, torch.Tensor):
        # Two scalar sides over a device mask (``where(mask, 1.0, 0.0)``): fill
        # one side on the device, then select the other as a scalar below.
        a = wrap(_scalar_fill(tuple(shape), sa, mode), mode)
        sa = None
    if (sa is None) != (sb is None):
        # One scalar side (Python number, wrapped 0-d CPU tensor or 0-d
        # host-resident MetalFloat64): pass its exact per-component encoding
        # as the scalar of the plain ``where`` (float32-representable words /
        # the int64 bit pattern), so nothing is copied to the GPU.
        if sa is None:
            tensor, scalars = coerce(a, mode), _scalar_comps(sb, mode)
            comps = tuple(
                torch.where(cond, x, v)
                for x, v in zip(tensor._comps, scalars, strict=True)
            )
        else:
            tensor, scalars = coerce(b, mode), _scalar_comps(sa, mode)
            comps = tuple(
                torch.where(cond, v, y)
                for v, y in zip(scalars, tensor._comps, strict=True)
            )
        return wrap(tuple(c.contiguous() for c in comps), mode)
    ac, bc = coerce(a, mode), coerce(b, mode)
    comps = tuple(
        torch.where(cond, x, y) for x, y in zip(ac._comps, bc._comps, strict=True)
    )
    return wrap(comps, mode)


_MASKED_FILL_MESSAGE = (
    "masked_fill_ only supports boolean masks, but got mask with dtype {dtype}"
)


@implements(aten.masked_fill.Scalar, aten.masked_fill_.Scalar)
def _masked_fill_scalar(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    if (host := _host_mutation(func, args, kwargs)) is not None:
        return host
    x = args[0]
    _check_mask(args[1], _MASKED_FILL_MESSAGE)
    if not is_metal(x):
        return func(x, *args[1:], **kwargs)
    # A CPU mask (torch accepts one; a 0-d CPU bool comes from a scalar-scalar
    # comparison on the host path, ``self == 0`` in copysign's backward).
    args = (x, _on_device(args[1]), *args[2:])
    return _apply(func, args, kwargs, scalar_positions=(2,))


@implements(aten.masked_fill.Tensor, aten.masked_fill_.Tensor)
def _masked_fill_tensor(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    if (host := _host_mutation(func, args, kwargs)) is not None:
        return host
    x, mask, value = args
    _check_mask(mask, _MASKED_FILL_MESSAGE)
    if not is_metal(x):
        return func(x, mask, _demote(value, x), **kwargs)
    mask = _on_device(mask)
    return apply_to_components(func, (x, mask, coerce(value, _mode(x, value))), kwargs)


@implements(aten.masked_scatter.default, aten.masked_scatter_.default)
def _masked_scatter(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    if (host := _host_mutation(func, args, kwargs)) is not None:
        return host
    x, mask, source = args
    _check_mask(
        mask,
        "masked_scatter_ only supports boolean masks, but got mask with dtype {dtype}",
    )
    if not is_metal(x):
        return func(x, mask, _demote(source, x), **kwargs)
    return apply_to_components(
        func, (x, mask, coerce(source, _mode(x, source))), kwargs
    )


@implements(aten.fill_.Scalar, aten.fill.Scalar)
def _fill_scalar(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    if (host := _host_mutation(func, args, kwargs)) is not None:
        return host
    return _apply(func, args, kwargs, scalar_positions=(1,))


@implements(aten.fill_.Tensor, aten.fill.Tensor)
def _fill_tensor(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    if (host := _host_mutation(func, args, kwargs)) is not None:
        return host
    x, value = args
    if not is_metal(x):
        return func(x, _demote(value, x), **kwargs)
    return apply_to_components(func, (x, coerce(value, _mode(x, value))), kwargs)


def _check_index_fill(x: Any, dim: int, index: Any) -> None:
    """torch's ``index_fill_`` checks (MPS silently ignores out-of-range indices)."""
    _check_index(
        index, "index_fill_(): Expected dtype int64 for index.", IndexError, True
    )
    if index.dim() > 1:
        raise RuntimeError("Index has to be a vector/scalar")
    nd = max(int(x.dim()), 1)
    d = int(dim) % nd
    size = int(x.shape[d]) if x.dim() else 1
    try:
        _check_range(index, size, d, wrap=True)
    except RuntimeError as e:
        raise IndexError(str(e)) from None


@implements(aten.index_fill.int_Scalar, aten.index_fill_.int_Scalar)
def _index_fill_scalar(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    _check_index_fill(args[0], args[1], args[2])
    if (host := _host_mutation(func, args, kwargs)) is not None:
        return host
    args = (args[0], args[1], _on_device(args[2]), *args[3:])
    return _apply(func, args, kwargs, scalar_positions=(3,))


@implements(aten.index_fill.int_Tensor, aten.index_fill_.int_Tensor)
def _index_fill_tensor(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    _check_index_fill(args[0], args[1], args[2])
    if (host := _host_mutation(func, args, kwargs)) is not None:
        return host
    x, dim, index, value = args
    if not is_metal(x):
        # plain self with a 0-d MetalFloat64 value: cast like torch does
        return func(x, dim, index, _demote(value, x), **kwargs)
    return apply_to_components(
        func, (x, dim, _on_device(index), coerce(value, _mode(x, value))), kwargs
    )


def _check_dim_source(
    name: str, x: Any, dim: int, index: torch.Tensor, source: Any
) -> int:
    """torch's ``index_add_`` / ``index_copy_`` shape checks; returns ``dim``.

    ``index`` must be a vector (or scalar), ``source`` must match ``self`` in
    every dimension but ``dim``, where it has ``index.numel()`` entries, and
    every index must lie in ``[0, self.size(dim))``.
    """
    if index.dim() > 1:
        raise IndexError(
            f"{name}(): Index is supposed to be a vector, but got dim: {index.dim()} "
            f"with type: {_dtype_name(index)} and size: {list(index.shape)}"
        )
    nd = int(x.dim())
    d = int(dim) % nd if nd else 0
    n = int(index.numel())
    if source.dim() == 0:
        if n != 1:
            raise IndexError(
                f"{name}(): When source is scalar, index should have one element "
                f"(got {n})"
            )
    elif nd and source.dim() != nd:
        raise IndexError(
            f"{name}(): When source and destination are not scalars, their "
            f"dimensionality must match. Source dimensionality ({source.dim()}), "
            f"destination dimensionality ({nd})"
        )
    elif nd:
        if int(source.shape[d]) != n:
            raise IndexError(
                f"{name}(): Number of indices ({n}) should be equal to "
                f"source.size(dim) ({int(source.shape[d])})"
            )
        for k in range(nd):
            if k != d and int(source.shape[k]) != int(x.shape[k]):
                raise IndexError(
                    f"{name}(): Source/destination tensor must have same slice "
                    f"shapes. Destination slice shape: {int(x.shape[k])} at "
                    f"dimension {k} and source slice shape: {int(source.shape[k])} "
                    f"at dimension {k}."
                )
    try:
        _check_range(index, int(x.shape[d]) if nd else 1, d)
    except RuntimeError:
        raise IndexError("index out of range in self") from None
    return d


def _dim_linear_index(
    x: MetalFloat64, dim: int, index: torch.Tensor, sshape: tuple[int, ...]
) -> torch.Tensor:
    """Row-major positions of ``x`` written by ``index_add`` / ``index_copy``.

    Position ``p`` of a source of shape ``sshape`` goes to ``x`` at the same
    coordinates except along ``dim``, where it goes to ``index[p_dim]``.
    """
    if index.device.type != "mps":
        index = index.to(_MPS)
    if x.dim() == 0:
        return torch.zeros(max(index.numel(), 1), dtype=torch.int64, device=_MPS)
    strides = _contiguous_strides(x.shape)
    lin = torch.zeros(sshape, dtype=torch.int64, device=_MPS)
    for k in range(len(sshape)):
        view = [1] * len(sshape)
        view[k] = sshape[k]
        if k == dim:
            coord = index.reshape(-1).view(view)
        else:
            coord = torch.arange(sshape[k], dtype=torch.int64, device=_MPS).view(view)
        lin = lin + coord * strides[k]
    return lin.reshape(-1)


@implements(aten.index_copy.default, aten.index_copy_.default)
def _index_copy(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    """``index_copy`` through the linear-index path so duplicates keep both words.

    Running ``index_copy_`` once per component lets the two MPS launches pick
    different winners for a duplicated destination and split the (hi, lo)
    pair; the shared *last write wins* dedupe keeps them together.
    """
    x, dim, index, source = args
    _check_index(
        index,
        "index_copy_(): Expected a long tensor for index, but got {dtype}",
        long_only=True,
    )
    if (host := _host_mutation(func, args, kwargs)) is not None:
        return host
    if not is_metal(x):
        return func(x, dim, index, _demote(source, x), **kwargs)
    d = _check_dim_source("index_copy_", x, dim, index, source)
    s = coerce(source, _mode(x, source))
    if _same_storage(s, x):
        s = wrap(tuple(c.clone() for c in s._comps), x._mode)
    if func is aten.index_copy.default:
        x = wrap(tuple(c.clone() for c in x._comps), x._mode)
    _check_no_overlap(x)
    sshape = tuple(s.shape) if s.dim() else tuple(1 for _ in x.shape)
    lin = _dim_linear_index(x, d, index, sshape)
    vals = tuple(c.reshape(-1) for c in s._comps)
    lin_u, vals_u = _dedupe_last(lin, vals)
    _put_linear(x._comps, lin_u, vals_u)
    return x


# ---------------------------------------------------------------------------
# scatter (src / value); reduce= variants are not supported
# ---------------------------------------------------------------------------
def _scatter_into(
    x: MetalFloat64, dim: int, index: torch.Tensor, vals: tuple[torch.Tensor, ...]
) -> None:
    """``x[..., index[p], ...] = vals[p]`` with *last write wins* for duplicates.

    ``vals`` are flat component tensors aligned with ``index.reshape(-1)``.
    Running ``scatter_`` once per component would let the two launches pick
    different winners for a duplicated destination and split the (hi, lo) pair.
    """
    _check_no_overlap(x)
    lin = _scatter_linear_index(x, dim, index).reshape(-1)
    lin_u, vals_u = _dedupe_last(lin, vals)
    _put_linear(x._comps, lin_u, vals_u)


@implements(aten.scatter.src, aten.scatter_.src)
def _scatter_src(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    if (host := _host_mutation(func, args, kwargs)) is not None:
        return host
    x, dim, index, src = args
    _check_index(index, "scatter(): Expected dtype int32/int64 for index")
    if not is_metal(x):
        return func(x, dim, index, _demote(src, x), **kwargs)
    _check_scatter(x, dim, index, src)
    s = coerce(src, _mode(x, src))
    if _same_storage(s, x):
        s = wrap(tuple(c.clone() for c in s._comps), x._mode)
    if func is aten.scatter.src:
        x = wrap(tuple(c.clone() for c in x._comps), x._mode)
    if index.numel() == 0:
        return x  # torch: an empty index (of any rank) writes nothing
    window = tuple(slice(0, n) for n in index.shape)
    _scatter_into(x, dim, index, tuple(c[window].reshape(-1) for c in s._comps))
    return x


@implements(aten.scatter.value, aten.scatter_.value)
def _scatter_value(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    if (host := _host_mutation(func, args, kwargs)) is not None:
        return host
    x, dim, index, value = args
    _check_index(index, "scatter(): Expected dtype int32/int64 for index")
    if not is_metal(x):
        return func(x, dim, index, value, **kwargs)
    _check_scatter(x, dim, index, None)
    if func is aten.scatter.value:
        x = wrap(tuple(c.clone() for c in x._comps), x._mode)
    n = index.numel()
    if n == 0:
        return x
    vals = tuple(
        torch.full((n,), sc, dtype=c.dtype, device=_MPS)
        for sc, c in zip(_scalar_comps(value, x._mode), x._comps, strict=True)
    )
    _scatter_into(x, dim, index, vals)
    return x


@implements(
    aten.scatter.reduce,
    aten.scatter_.reduce,
    aten.scatter.value_reduce,
    aten.scatter_.value_reduce,
)
def _scatter_reduce(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    raise NotImplementedError(
        "MetalFloat64: scatter(..., reduce=) is not supported; use scatter_add "
        "(exact) or index_put(accumulate=True)"
    )


# ---------------------------------------------------------------------------
# Linear-index machinery for exact accumulation
# ---------------------------------------------------------------------------
def _check_no_overlap(x: MetalFloat64) -> None:
    """torch's ``assert_no_internal_overlap`` for the linear-index writers.

    The per-component plain ops (``fill_``, ``copy_``, ...) check this
    themselves; ``_put_linear`` writes through ``index_put_`` on the raw
    component, which would silently write an expanded (stride-0) tensor in
    an order-dependent way.
    """
    if torch._debug_has_internal_overlap(x._comps[0]) == 1:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to "
            "tensor refers to a single memory location. Please clone() the "
            "tensor before performing the operation."
        )


def _drop_scalar_bools(indices: list[Any]) -> list[Any] | None:
    """Remove 0-d bool indices (torch's ``index_put`` semantics for them).

    A 0-d ``True`` selects everything at its position and is dropped; a 0-d
    ``False`` selects nothing (``None`` is returned). The MPS ``index`` /
    ``index_put`` kernels abort the process on a raw 0-d bool index.
    """
    out = []
    for ind in indices:
        if isinstance(ind, torch.Tensor) and ind.dtype == torch.bool and ind.dim() == 0:
            if not bool(ind.item()):
                return None
            continue
        out.append(ind)
    return out


def _prep_indices(indices: Any) -> list[Any]:
    """Move index tensors to the GPU; keep ``None`` entries."""
    out = []
    for ind in indices:
        if isinstance(ind, torch.Tensor) and ind.device.type != "mps":
            ind = ind.to(_MPS)
        out.append(ind)
    return out


def _has_int_index(indices: list[Any]) -> bool:
    return any(
        isinstance(i, torch.Tensor) and i.dtype not in (torch.bool, torch.uint8)
        for i in indices
    )


def _linear_index(shape: Any, indices: list[Any]) -> torch.Tensor:
    """Linear (row-major) positions selected by ``indices`` in a tensor of ``shape``."""
    numel = 1
    for s in shape:
        numel *= int(s)
    pos = torch.arange(numel, dtype=torch.int64, device=_MPS).view(tuple(shape))
    return aten.index.Tensor(pos, indices)


def _contiguous_strides(shape: Any) -> list[int]:
    strides = [1] * len(shape)
    for k in range(len(shape) - 2, -1, -1):
        strides[k] = strides[k + 1] * int(shape[k + 1])
    return strides


def _gather_linear(
    comps: tuple[torch.Tensor, ...], lin: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    return tuple(c.reshape(-1)[lin] for c in comps)


def _put_linear(
    comps: tuple[torch.Tensor, ...], lin: torch.Tensor, vals: tuple[torch.Tensor, ...]
) -> None:
    """Write ``vals`` at linear positions ``lin`` of every component (any strides)."""
    for d, v in zip(comps, vals, strict=True):
        if d.is_contiguous():
            d.view(-1).index_put_((lin,), v)
        else:
            d.index_put_(tuple(torch.unravel_index(lin, tuple(d.shape))), v)


def _dedupe_last(
    lin: torch.Tensor, vals: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Keep the last write for every duplicated linear index."""
    n = lin.numel()
    if n <= 1:
        return lin, vals
    sorted_lin, order = torch.sort(lin, stable=True)
    last = torch.ones(n, dtype=torch.bool, device=lin.device)
    last[:-1] = sorted_lin[1:] != sorted_lin[:-1]
    sel = order[last]
    return lin[sel], tuple(v[sel] for v in vals)


def _segment_sum(
    lin: torch.Tensor, vals: tuple[torch.Tensor, ...], mode: str
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Sum the values sharing a linear index, exactly, with the ``add`` kernel.

    The entries are stably sorted by index; in every round each entry of odd
    rank within its group is added into its left neighbour (even rank) and
    dropped, so the group sizes halve per round.

    Returns:
        ``(unique_lin, summed_vals)`` with one entry per distinct index.
    """
    n = lin.numel()
    if n == 0:
        return lin, vals
    lin_s, order = torch.sort(lin, stable=True)
    vals_s = tuple(v[order] for v in vals)
    while lin_s.numel() > 1:
        m = lin_s.numel()
        change = torch.ones(m, dtype=torch.bool, device=lin.device)
        change[1:] = lin_s[1:] != lin_s[:-1]
        seg = torch.cumsum(change.to(torch.int64), 0) - 1
        starts = torch.nonzero(change).squeeze(1)
        rank = torch.arange(m, dtype=torch.int64, device=lin.device) - starts[seg]
        odd = (rank & 1) == 1
        src = torch.nonzero(odd).squeeze(1)
        if src.numel() == 0:
            break
        dst = src - 1
        summed = _launch(
            mode,
            "add",
            tuple(v[dst] for v in vals_s),
            tuple(v[src] for v in vals_s),
        )
        for v, s in zip(vals_s, summed, strict=True):
            v.index_put_((dst,), s)
        keep = ~odd
        lin_s = lin_s[keep]
        vals_s = tuple(v[keep] for v in vals_s)
    return lin_s, vals_s


def _accumulate_linear(
    comps: tuple[torch.Tensor, ...],
    lin: torch.Tensor,
    vals: tuple[torch.Tensor, ...],
    mode: str,
) -> None:
    """``comps[lin] += vals`` with exact df64/sf64 addition (duplicates allowed)."""
    lin_u, vals_u = _segment_sum(lin, vals, mode)
    if lin_u.numel() == 0:
        return
    current = _gather_linear(comps, lin_u)
    summed = _launch(mode, "add", _contig(current), _contig(vals_u))
    _put_linear(comps, lin_u, summed)


def _index_put_into(
    x: MetalFloat64, indices: Any, values: Any, accumulate: bool
) -> None:
    """Mutate the components of ``x`` with ``index_put`` semantics."""
    mode = _mode(x, values)
    _check_no_overlap(x)
    indices = _drop_scalar_bools(_prep_indices(indices))
    if indices is None:
        return  # a 0-d False index selects nothing
    v = coerce(values, mode)
    if _same_storage(v, x):
        v = wrap(tuple(c.clone() for c in v._comps), mode)
    if not indices:
        # Only 0-d True indices: every element is written (MPS aborts on an
        # empty index list).
        vb = tuple(c.expand(tuple(x.shape)) for c in v._comps)
        if accumulate:
            vb = _launch(mode, "add", _contig(x._comps), _contig(vb))
        for d, c in zip(x._comps, vb, strict=True):
            d.copy_(c)
        return
    if not _has_int_index(indices):
        # Boolean masks / None only: every selected position is unique.
        if not accumulate:
            for d, c in zip(x._comps, v._comps, strict=True):
                aten.index_put_.default(d, indices, c)
            return
        current = tuple(aten.index.Tensor(c, indices) for c in x._comps)
        shape = tuple(current[0].shape)
        vb = tuple(c.expand(shape).contiguous() for c in v._comps)
        summed = _launch(mode, "add", _contig(current), vb)
        for d, c in zip(x._comps, summed, strict=True):
            aten.index_put_.default(d, indices, c)
        return
    lin = _linear_index(x.shape, indices)
    shape = tuple(lin.shape)
    lin = lin.reshape(-1)
    vals = tuple(c.expand(shape).reshape(-1) for c in v._comps)
    if accumulate:
        _accumulate_linear(x._comps, lin, vals, mode)
    else:
        lin_u, vals_u = _dedupe_last(lin, vals)
        _put_linear(x._comps, lin_u, vals_u)


@implements(aten.index_put_.default, aten._index_put_impl_.default)
def _index_put_(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    if (host := _host_mutation(func, args, kwargs)) is not None:
        return host
    b = _bind(func, args, kwargs)
    _check_indices(b["indices"])
    x = b["self"]
    if not is_metal(x):
        return func(
            x, _prep_indices(b["indices"]), _demote(b["values"], x), b["accumulate"]
        )
    _index_put_into(x, b["indices"], b["values"], bool(b["accumulate"]))
    return x


@implements(aten.index_put.default, aten._unsafe_index_put.default)
def _index_put(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    b = _bind(func, args, kwargs)
    _check_indices(b["indices"])
    x = b["self"]
    if not is_metal(x):
        return func(
            x, _prep_indices(b["indices"]), _demote(b["values"], x), b["accumulate"]
        )
    out = wrap(tuple(c.clone() for c in x._comps), x._mode)
    _index_put_into(out, b["indices"], b["values"], bool(b["accumulate"]))
    return out


def _put_into(
    x: MetalFloat64, index: torch.Tensor, source: Any, accumulate: bool
) -> None:
    """``x.view(-1)[index] (+)= source.view(-1)`` (``put_`` semantics, any strides)."""
    mode = _mode(x, source)
    _check_no_overlap(x)
    s = coerce(source, mode)
    if _same_storage(s, x):
        s = wrap(tuple(c.clone() for c in s._comps), mode)
    if index.device.type != "mps":
        index = index.to(_MPS)
    numel = x.numel()
    if index.numel() != s.numel():
        raise IndexError(
            "put_(): Expected source and index to have the same number of "
            f"elements, but got source.numel() = {s.numel()}, index.numel() = "
            f"{index.numel()}"
        )
    lin = index.reshape(-1)
    if lin.numel():
        lo, hi = torch.stack(torch.aminmax(lin)).tolist()
        bad = hi if hi >= numel else lo if lo < -numel else None
        if bad is not None:
            raise IndexError(
                f"out of range: tried to access index {bad} on a tensor of {numel} "
                "elements."
            )
    if numel:
        lin = torch.where(lin < 0, lin + numel, lin)
    vals = tuple(c.reshape(-1) for c in s._comps)
    if accumulate:
        _accumulate_linear(x._comps, lin, vals, mode)
    else:
        lin_u, vals_u = _dedupe_last(lin, vals)
        _put_linear(x._comps, lin_u, vals_u)


@implements(aten.put.default, aten.put_.default)
def _put(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    if (host := _host_mutation(func, args, kwargs)) is not None:
        return host
    b = _bind(func, args, kwargs)
    _check_index(
        b["index"],
        "put_(): Expected a long tensor for index, but got {dtype}",
        long_only=True,
    )
    x = b["self"]
    if not is_metal(x):
        return func(x, b["index"], _demote(b["source"], x), b["accumulate"])
    if func is aten.put.default:
        x = wrap(tuple(c.clone() for c in x._comps), x._mode)
    _put_into(x, b["index"], b["source"], bool(b["accumulate"]))
    return x


def _check_scatter(x: Any, dim: int, index: torch.Tensor, src: Any) -> None:
    """torch's ``scatter_shape_check`` plus the index range check."""
    nd = int(x.dim())
    d = int(dim) % nd if nd else 0
    if nd and index.numel() and index.dim() != nd:
        raise RuntimeError(
            "Index tensor must have the same number of dimensions as self tensor"
        )
    if index.numel() == 0:
        return
    if nd:
        for k in range(nd):
            if k != d and int(index.shape[k]) > int(x.shape[k]):
                raise RuntimeError(
                    f"Expected index {list(index.shape)} to be no larger than self "
                    f"{list(x.shape)} apart from dimension {d}"
                )
        if src is not None and src.dim() != 0:
            if src.dim() != nd:
                raise RuntimeError(
                    "Index tensor must have the same number of dimensions as src tensor"
                )
            for k in range(nd):
                if int(index.shape[k]) > int(src.shape[k]):
                    raise RuntimeError(
                        f"Expected index {list(index.shape)} to be no larger than "
                        f"src {list(src.shape)}"
                    )
    _check_range(index, int(x.shape[d]) if nd else 1, d)


def _scatter_linear_index(
    x: MetalFloat64, dim: int, index: torch.Tensor
) -> torch.Tensor:
    """Row-major positions in ``x`` addressed by ``scatter``'s ``index`` on ``dim``."""
    if index.device.type != "mps":
        index = index.to(_MPS)
    nd = x.dim()
    dim = dim % nd if nd else 0
    if nd == 0:
        return torch.zeros(tuple(index.shape), dtype=torch.int64, device=_MPS)
    ishape = tuple(index.shape)
    strides = _contiguous_strides(x.shape)
    lin = torch.zeros(ishape, dtype=torch.int64, device=_MPS)
    for k in range(len(ishape)):
        if k == dim:
            coord = index
        else:
            view = [1] * len(ishape)
            view[k] = ishape[k]
            coord = torch.arange(ishape[k], dtype=torch.int64, device=_MPS).view(view)
        lin = lin + coord * strides[k]
    return lin


def _scatter_add_into(x: MetalFloat64, dim: int, index: torch.Tensor, src: Any) -> None:
    """``x[..., index[p], ...] += src[p]`` for every position ``p`` of ``index``."""
    mode = _mode(x, src)
    _check_no_overlap(x)
    s = coerce(src, mode)
    if _same_storage(s, x):
        s = wrap(tuple(c.clone() for c in s._comps), mode)
    if index.numel() == 0:
        return
    lin = _scatter_linear_index(x, dim, index)
    window = tuple(slice(0, n) for n in index.shape)
    vals = tuple(c[window].reshape(-1) for c in s._comps)
    _accumulate_linear(x._comps, lin.reshape(-1), vals, mode)


@implements(aten.scatter_add.default, aten.scatter_add_.default)
def _scatter_add(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    if (host := _host_mutation(func, args, kwargs)) is not None:
        return host
    x, dim, index, src = args
    _check_index(index, "scatter(): Expected dtype int32/int64 for index")
    if not is_metal(x):
        return func(x, dim, index, _demote(src, x), **kwargs)
    _check_scatter(x, dim, index, src)
    if func is aten.scatter_add.default:
        x = wrap(tuple(c.clone() for c in x._comps), x._mode)
    _scatter_add_into(x, dim, index, src)
    return x


def _index_add_into(
    x: MetalFloat64, dim: int, index: torch.Tensor, source: Any, alpha: Any
) -> None:
    mode = _mode(x, source)
    _check_no_overlap(x)
    s = coerce(source, mode)
    if _same_storage(s, x):
        s = wrap(tuple(c.clone() for c in s._comps), mode)
    a = scalar_value(alpha)
    if a is None:
        raise TypeError("index_add: alpha must be a scalar")
    if a != 1.0:
        lib = library(mode)
        res = lib.launch(
            "mul",
            _contig(s._comps) if mode == "df64" else s._comps[0].contiguous(),
            scalar=a,
        )
        count_gpu("mul")
        s = wrap(tuple(res) if isinstance(res, (tuple, list)) else (res,), mode)
    d = _check_dim_source("index_add_", x, dim, index, source)
    sshape = tuple(s.shape) if s.dim() else tuple(1 for _ in x.shape)
    lin = _dim_linear_index(x, d, index, sshape)
    vals = tuple(c.reshape(-1) for c in s._comps)
    _accumulate_linear(x._comps, lin, vals, mode)


@implements(aten.index_add.default, aten.index_add_.default)
def _index_add(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    if (host := _host_mutation(func, args, kwargs)) is not None:
        return host
    b = _bind(func, args, kwargs)
    _check_index(
        b["index"],
        "index_add_(): Expected dtype int32/int64 for index but got: {dtype}",
    )
    x = b["self"]
    if not is_metal(x):
        return func(x, b["dim"], b["index"], _demote(b["source"], x), alpha=b["alpha"])
    if func is aten.index_add.default:
        x = wrap(tuple(c.clone() for c in x._comps), x._mode)
    _index_add_into(x, b["dim"], b["index"], b["source"], b["alpha"])
    return x


# ---------------------------------------------------------------------------
# Ordering: sort keys, sort / argsort / topk / searchsorted / bucketize
# ---------------------------------------------------------------------------
def _ordered_f32(t: torch.Tensor) -> torch.Tensor:
    """Map a float32 tensor to int32 keys monotone in value (-0 == +0, NaN max)."""
    b = t.contiguous().view(torch.int32)
    o = torch.where(b < 0, b ^ _INT32_MAX, b)
    o = torch.where(t == 0, torch.zeros_like(o), o)
    return torch.where(torch.isnan(t), torch.full_like(o, _INT32_MAX), o)


def sort_key(x: MetalFloat64, nan_key: int = _INT64_MAX) -> torch.Tensor:
    """Int64 keys with the same order (and the same ties) as the values of ``x``.

    Args:
        x: A MetalFloat64 tensor of either representation.
        nan_key: Key assigned to every NaN; the largest key by default (NaN
            sorts last, as in torch's CPU sort), ``_INT64_MIN`` for the sorted
            sequence of ``searchsorted`` (module docstring).

    Returns:
        torch.Tensor: int64 MPS tensor with the shape of ``x``. Signed zeros tie
        and all NaNs tie.
    """
    if x._mode == "df64":
        hi, lo = x._comps
        key = _ordered_f32(hi).to(torch.int64) * (1 << 32) + (
            _ordered_f32(lo).to(torch.int64) + (1 << 31)
        )
        return torch.where(torch.isnan(hi), torch.full_like(key, nan_key), key)
    bits = x._comps[0].contiguous()
    mag = bits & _INT64_MAX
    key = torch.where(bits < 0, bits ^ _INT64_MAX, bits)
    key = torch.where(mag == 0, torch.zeros_like(key), key)
    return torch.where(mag > _F64_INF_BITS, torch.full_like(key, nan_key), key)


def _gather_values(x: MetalFloat64, dim: int, idx: torch.Tensor) -> MetalFloat64:
    if x.dim() == 0:
        return wrap(tuple(c.clone() for c in x._comps), x._mode)
    return wrap(tuple(c.gather(dim, idx) for c in x._comps), x._mode)


@implements(aten.sort.default, aten.sort.stable)
def _sort(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    b = _bind(func, args, kwargs)
    x, dim, descending = b["self"], int(b["dim"]), bool(b["descending"])
    key = sort_key(x)
    idx = torch.sort(key, dim=dim, descending=descending, stable=True).indices
    return _gather_values(x, dim, idx), idx


@implements(aten.argsort.default, aten.argsort.stable)
def _argsort(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    b = _bind(func, args, kwargs)
    key = sort_key(b["self"])
    return torch.sort(
        key, dim=int(b["dim"]), descending=bool(b["descending"]), stable=True
    ).indices


@implements(aten.topk.default)
def _topk(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    b = _bind(func, args, kwargs)
    x, dim = b["self"], int(b["dim"])
    key = sort_key(x)
    idx = torch.topk(
        key, int(b["k"]), dim=dim, largest=bool(b["largest"]), sorted=bool(b["sorted"])
    ).indices
    return _gather_values(x, dim, idx), idx


@implements(aten.searchsorted.Tensor, aten.searchsorted.Scalar)
def _searchsorted(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    b = _bind(func, args, kwargs)
    seq, values = b["sorted_sequence"], b["self"]
    mode = _mode(seq, values)
    kseq = sort_key(coerce(seq, mode), nan_key=_INT64_MIN)
    kval = sort_key(coerce(values, mode))
    sorter = b["sorter"]
    if isinstance(sorter, torch.Tensor):
        if is_metal(sorter) or sorter.dtype != torch.int64:
            raise RuntimeError(
                "torch.searchsorted(): sorter must be a tensor of long dtype but "
                f"got dtype {_dtype_name(sorter)}"
            )
        if sorter.device.type != "mps":
            sorter = sorter.to(_MPS)
    return torch.searchsorted(
        kseq,
        kval,
        out_int32=bool(b["out_int32"]),
        right=bool(b["right"]),
        side=b["side"],
        sorter=sorter,
    )


@implements(aten.bucketize.Tensor, aten.bucketize.Scalar)
def _bucketize(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    b = _bind(func, args, kwargs)
    values, bounds = b["self"], b["boundaries"]
    mode = _mode(values, bounds)
    return torch.bucketize(
        sort_key(coerce(values, mode)),
        sort_key(coerce(bounds, mode), nan_key=_INT64_MIN),
        out_int32=bool(b["out_int32"]),
        right=bool(b["right"]),
    )


@implements(aten.nonzero.default)
def _nonzero(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    if x._mode == "df64":
        hi, lo = x._comps
        mask = (hi != 0) | (lo != 0)
    else:
        mask = (x._comps[0] & _INT64_MAX) != 0
    return torch.nonzero(mask)


@implements(aten.nonzero.out)
def _nonzero_out(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _with_out(aten.nonzero.default(args[0]), kwargs["out"])


# ---------------------------------------------------------------------------
# reflection / replication padding (``be.pad(mode="reflect" | "replicate")``)
# ---------------------------------------------------------------------------
_PAD_OPS = (
    aten.reflection_pad1d.default,
    aten.reflection_pad2d.default,
    aten.reflection_pad3d.default,
    aten.replication_pad1d.default,
    aten.replication_pad2d.default,
    aten.replication_pad3d.default,
)
_PAD_BACKWARD_OPS = (
    aten.reflection_pad1d_backward.default,
    aten.reflection_pad2d_backward.default,
    aten.reflection_pad3d_backward.default,
    aten.replication_pad1d_backward.default,
    aten.replication_pad2d_backward.default,
    aten.replication_pad3d_backward.default,
)


@implements(*_PAD_OPS)
def _pad(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    """Padding only copies elements: exact per component (native MPS kernels)."""
    return apply_to_components(func, args, kwargs)


@implements(*_PAD_BACKWARD_OPS)
def _pad_backward(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    """The backward sums the gradients of aliased elements: CPU float64 (counted).

    Summing ``hi`` and ``lo`` words separately would lose the compensation of
    the pair, so the accumulation runs on the host through ``cpu_fallback``.
    """
    from optiland.backend.torch_backend.metal.tensor import cpu_fallback

    return cpu_fallback(func, args, kwargs, label=func._schema.name.split("::")[-1])


# ---------------------------------------------------------------------------
# Backward helpers emitted by autograd for the ops above
# ---------------------------------------------------------------------------
@implements(aten.select_backward.default)
def _select_backward(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    b = _bind(func, args, kwargs)
    grad = b["grad_output"]
    z = _zeros(b["input_sizes"], grad._mode)
    for zc, gc in zip(z._comps, grad._comps, strict=True):
        zc.select(int(b["dim"]), int(b["index"])).copy_(gc)
    return z


@implements(aten.slice_backward.default)
def _slice_backward(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    b = _bind(func, args, kwargs)
    grad = b["grad_output"]
    z = _zeros(b["input_sizes"], grad._mode)
    for zc, gc in zip(z._comps, grad._comps, strict=True):
        aten.slice.Tensor(
            zc, int(b["dim"]), int(b["start"]), int(b["end"]), int(b["step"])
        ).copy_(gc)
    return z


@implements(aten.diagonal_backward.default)
def _diagonal_backward(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    b = _bind(func, args, kwargs)
    grad = b["grad_output"]
    z = _zeros(b["input_sizes"], grad._mode)
    for zc, gc in zip(z._comps, grad._comps, strict=True):
        zc.diagonal(int(b["offset"]), int(b["dim1"]), int(b["dim2"])).copy_(gc)
    return z


@implements(aten.masked_select_backward.default)
def _masked_select_backward(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    grad, inp, mask = args
    z = _zeros(inp.shape, grad._mode)
    for zc, gc in zip(z._comps, grad._comps, strict=True):
        zc.masked_scatter_(mask, gc)
    return z


@implements(aten.masked_scatter_backward.default)
def _masked_scatter_backward(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    """Gradient of ``masked_scatter`` w.r.t. ``source``: the selected entries first."""
    grad, mask, sizes = args
    sizes = [int(n) for n in sizes]
    numel = 1
    for n in sizes:
        numel *= n
    z = _zeros((numel,), grad._mode)
    for zc, gc in zip(z._comps, grad._comps, strict=True):
        sel = gc.masked_select(mask)
        zc[: sel.numel()].copy_(sel)
    return wrap(tuple(c.view(sizes) for c in z._comps), grad._mode)


@implements(aten.value_selecting_reduction_backward.default)
def _value_selecting_reduction_backward(
    func: Any, types: Any, args: Any, kwargs: Any
) -> Any:
    b = _bind(func, args, kwargs)
    grad, dim, indices = b["grad"], int(b["dim"]), b["indices"]
    sizes = list(b["sizes"])
    if not b["keepdim"] and sizes:
        grad = grad.unsqueeze(dim)
        indices = indices.unsqueeze(dim)
    z = _zeros(sizes, grad._mode)
    for zc, gc in zip(z._comps, grad._comps, strict=True):
        zc.scatter_(dim, indices, gc)
    return z


@implements(aten.index_select_backward.default)
def _index_select_backward(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    b = _bind(func, args, kwargs)
    grad = b["grad"]
    z = _zeros(b["self_sizes"], grad._mode)
    _index_add_into(z, int(b["dim"]), b["index"], grad, 1)
    return z


@implements(aten.unfold_backward.default)
def _unfold_backward(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    """Scatter-add the window gradients back; overlapping windows add exactly."""
    b = _bind(func, args, kwargs)
    grad = b["grad_in"]
    sizes = [int(n) for n in b["input_sizes"]]
    z = _zeros(sizes, grad._mode)
    numel = z.numel()
    pos = torch.arange(numel, dtype=torch.int64, device=_MPS).view(sizes)
    lin = pos.unfold(int(b["dim"]), int(b["size"]), int(b["step"])).reshape(-1)
    vals = tuple(c.reshape(-1) for c in grad._comps)
    _accumulate_linear(z._comps, lin, vals, grad._mode)
    return z


# ---------------------------------------------------------------------------
# out= overloads: functional result copied into out (resized first)
# ---------------------------------------------------------------------------
for _out_op, _fn_op, _names in (
    (aten.sort.values, aten.sort.default, ("values", "indices")),
    (aten.sort.values_stable, aten.sort.stable, ("values", "indices")),
    (aten.topk.values, aten.topk.default, ("values", "indices")),
    (aten.argsort.stable_out, aten.argsort.stable, ("out",)),
    (aten.searchsorted.Tensor_out, aten.searchsorted.Tensor, ("out",)),
    (aten.searchsorted.Scalar_out, aten.searchsorted.Scalar, ("out",)),
    (aten.bucketize.Tensor_out, aten.bucketize.Tensor, ("out",)),
    (aten.bucketize.Scalar_out, aten.bucketize.Scalar, ("out",)),
    (aten.index_select.out, aten.index_select.default, ("out",)),
    (aten.gather.out, aten.gather.default, ("out",)),
    (aten.masked_select.out, aten.masked_select.default, ("out",)),
    (aten.index_add.out, aten.index_add.default, ("out",)),
    (aten.index_copy.out, aten.index_copy.default, ("out",)),
    (aten.index_fill.int_Scalar_out, aten.index_fill.int_Scalar, ("out",)),
    (aten.index_fill.int_Tensor_out, aten.index_fill.int_Tensor, ("out",)),
    (aten.scatter.src_out, aten.scatter.src, ("out",)),
    (aten.scatter.value_out, aten.scatter.value, ("out",)),
    (aten.scatter_add.out, aten.scatter_add.default, ("out",)),
    (aten.take.out, aten.take.default, ("out",)),
    (aten.index.Tensor_out, aten.index.Tensor, ("out",)),
    (aten.index_put.out, aten.index_put.default, ("out",)),
    (aten.put.out, aten.put.default, ("out",)),
    (aten.masked_fill.Scalar_out, aten.masked_fill.Scalar, ("out",)),
    (aten.masked_fill.Tensor_out, aten.masked_fill.Tensor, ("out",)),
    (aten.masked_scatter.out, aten.masked_scatter.default, ("out",)),
    (aten.where.self_out, aten.where.self, ("out",)),
):
    implements(_out_op)(_out_variant(_fn_op, *_names))
