"""MetalFloat64 dispatch handlers: elementwise ops (registered via tensor.implements).

Every handler in this module is a thin layer over three bridges,
:func:`_unary`, :func:`_binary` and :func:`_ternary`, which

1. determine the representation with :func:`mode_of`,
2. route scalar operands (Python numbers, wrapped 0-d CPU tensors and 0-d
   *host-resident* ``MetalFloat64`` tensors such as surface radii and
   thicknesses) through the ``scalar=`` kernel variants: the library encodes
   them exactly as ``constant`` kernel arguments, so no host-to-GPU copy and
   no sync is involved (scalars never pass through float32). Ternary ops,
   which have no scalar kernels, are decomposed into the binary kernels they
   are built from (bit-identical results) whenever a scalar operand is present,
3. coerce every other operand exactly with :func:`coerce`, broadcast the
   component tensors on the host (:func:`broadcast_contiguous`) and launch one
   flat kernel from :data:`codegen.OPS`,
4. count the launch (``gpu:<op>``) and wrap the result (values become
   ``MetalFloat64``; predicates stay plain bool MPS tensors).

Both libraries compile every ``codegen.OPS`` kernel: the transcendentals run on
the GPU in ``sf64`` as well (through the 48-bit df64 bridge of ``sf64_math.h``,
counted ``gpu:<op>``; their accuracy is documented in that header). Only an op
whose kernel is absent from a library (``lib.has(op)`` False) goes through
:func:`cpu_fallback` with a ``<op>:missing_kernel`` label, so ``stats()``
shows exactly what runs on the CPU; today that route is inactive.

Variant plumbing (:func:`_finish`) is uniform: in-place ops copy the computed
result into the components of ``self`` (``copy_`` writes through views) and
return ``self``, and never resize it: a broadcast result larger than ``self``
raises torch's ``output with shape [...] doesn't match the broadcast shape``;
``out=`` variants copy into ``out`` (resizing it to the result shape first, as
torch does; a plain floating ``out`` receives the same-category downcast, an
integer or bool ``out`` raises, a CPU ``out`` raises torch's device error);
everything else returns the fresh tensor. In-place comparison / logical
variants (``eq_`` ... ``logical_xor_``) write ``0.0`` / ``1.0`` into ``self``.

Operand devices follow torch: a CPU *floating-point* tensor with ``dim() > 0``
raises ``Expected all tensors to be on the same device`` (only 0-d CPU tensors
are scalar operands); CPU bool and integer tensors are accepted because the
dual-residency host path produces its predicates and indices on the CPU.

Fused products: ``add``/``sub`` with ``alpha`` and ``lerp`` are evaluated as
torch's CPU kernels do, ``fma(alpha, b, a)`` and ``fma(coeff, end - start,
base)`` (``lerp_vec``), for tensor and scalar ``b`` alike (torch contracts
the wrapped scalar too). In sf64 that is the single-rounding softfloat
``fma`` kernel, so the results are bit-exact with torch's vectorized path
(torch's own scalar tail path for ``lerp`` is not fused, so torch is not
self-consistent there); in df64 the product is rounded before the add
(``df::mul_add`` is two roundings), one extra ``u^2 = 2^-96`` rounding.
``addcmul`` / ``addcdiv`` round ``value * t1``, the product / quotient and the
sum separately, like torch's *vectorized* ``self + value * t1 * t2``; torch's
scalar tail (tensors of fewer than 8 elements and the last ``n % 8``
elements) is fma-contracted on arm64, so torch is size-inconsistent there
and sf64 ``addcmul`` can differ from it by one ulp in those tail elements.

``out=`` into a MetalFloat64 of the *other* fixed representation raises the
representation-mixing ``TypeError`` (the destination is treated like any
other operand); a representation-agnostic factory ``out`` (``torch.empty(...,
dtype=float64, device="mps")``) is re-tagged to the result's representation
first. In-place ops whose ``other`` partially overlaps ``self`` (``x[1:].add_(
x[:-1])``, ``x.t().add_(x)``) succeed with compute-then-copy semantics (the
result is computed from a snapshot of the operands, then stored), where torch
raises its partial-overlap error; internally overlapping *targets* raise as
in torch.

Known deviations from torch CPU float64 (deliberate, documented here):

* A 0-d MetalFloat64 combined with an N-d float32 / float16 / bfloat16 MPS
  tensor returns a MetalFloat64 (the emulated float64 wins); torch's
  dimension-based promotion would give the N-d tensor's dtype. Every other
  mixed case (N-d MetalFloat64 with plain tensors or Python scalars) promotes
  to MetalFloat64 as torch promotes to float64.
* ``lerp`` for ``|weight| >= 0.5`` takes the sign of a zero result from
  torch's vectorized path (``fma(w - 1, end - start, end)``:
  ``lerp(1.0, -0.0, 1.0) == -0.0``); torch's scalar path (tensors of fewer
  than 8 elements, and the tail of a vectorized loop) gives ``+0.0``.
* ``round(decimals=d)`` in df64 scales by ``10^d`` in the representation
  (48 bits), so a value whose scaled form lies within ``~2^-48`` relative of a
  half-integer can round to the other neighbour (``-4.994999999999997`` with
  ``decimals=2`` gives ``-5.0``; torch and sf64 give ``-4.99``), and random
  data differs from torch by one float64 ulp in most elements. ``decimals=0``
  rounding (``round``, ``floor``, ``ceil``, ``trunc``) is exact in both modes.
* ``pow``'s ``sqrt`` / ``rsqrt`` fast paths (``-0.0 -> -0.0``, ``-inf -> NaN``)
  apply to the ``Tensor_Scalar`` overloads only, as in torch; a 0-d tensor
  exponent (``pow.Tensor_Tensor``) uses C ``pow`` semantics (``+0.0``, ``+inf``).
* ``lerp`` / ``lerp_`` accept a plain non-float64 ``start`` or ``end``
  (float32 / float16 / int / bool, N-d or 0-d, CPU or mps) and promote it
  exactly into the op's representation, and a plain float32 ``self.lerp_(
  metal, w)`` demotes the result; torch raises ``expected dtype double for
  `end``` for them. Only the weight's dtype is validated as torch does.

* ``maximum`` / ``minimum`` / ``fmax`` / ``fmin`` and everything built on them
  (``clamp``, ``clamp_min``, ``clamp_max``, ``relu``) break a ``+0 / -0`` tie
  the IEEE 754-2019 way (``maximum -> +0``, ``minimum -> -0``), which is also
  what torch's *vectorized* CPU path returns; torch's scalar path (tensors of
  fewer than 8 elements) returns the first operand instead, so torch CPU itself
  is not self-consistent there.
* ``nan_to_num`` defaults ``posinf`` / ``neginf`` to the representation's
  largest finite value (``float32`` max, ``3.4e38``, in df64; float64 max in
  sf64), not to float64 max, because df64 has the float32 exponent range.
* ``digamma`` / ``polygamma`` (the backward of ``lgamma``) have no kernel and
  run through the counted CPU fallback (``cpu_fallback:digamma``).
* Bool operands: like torch, ``x - True`` and a boolean ``alpha`` raise.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from optiland.backend.torch_backend.metal import encode
from optiland.backend.torch_backend.metal.library import broadcast_contiguous
from optiland.backend.torch_backend.metal.tensor import (
    MetalFloat64,
    aten,
    coerce,
    comps_arg,
    count_gpu,
    count_host,
    cpu_fallback,
    implements,
    is_metal,
    library,
    mode_of,
    result_from_launch,
    scalar_like,
    scalar_value,
    wrap,
    wrap_host,
)
from optiland.backend.torch_backend.metal.tensor import (
    check_inplace_shape as _check_inplace_shape,
)
from optiland.backend.torch_backend.metal.tensor import (
    check_out_device as _check_out_device,
)
from optiland.backend.torch_backend.metal.tensor import resize_out as _resize_out

#: Largest finite value representable per mode (``nan_to_num`` defaults).
_MAX_FINITE = {
    "df64": float(np.finfo(np.float32).max),
    "sf64": float(np.finfo(np.float64).max),
}


# ---------------------------------------------------------------------------
# Launch bridges
# ---------------------------------------------------------------------------
def _contig(x: MetalFloat64) -> MetalFloat64:
    """Return ``x`` with contiguous components (kernels ignore strides)."""
    if all(c.is_contiguous() for c in x._comps):
        return x
    return wrap(tuple(c.contiguous() for c in x._comps), x._mode)


def _fresh(x: MetalFloat64) -> MetalFloat64:
    """A contiguous copy of ``x`` that shares no storage with it."""
    return wrap(
        tuple(c.clone(memory_format=torch.contiguous_format) for c in x._comps), x._mode
    )


def _const_comps(
    shape: tuple[int, ...], value: float, mode: str
) -> tuple[torch.Tensor, ...]:
    """Contiguous GPU components of ``shape`` holding ``value`` exactly.

    Filled on the device (``torch.full``) from the exact scalar encoding; no
    host tensor is created and nothing is copied from the host.
    """
    if mode == "df64":
        hi, lo = encode.df64_scalar(value)
        return (
            torch.full(shape, hi, dtype=torch.float32, device="mps"),
            torch.full(shape, lo, dtype=torch.float32, device="mps"),
        )
    bits = encode.sf64_scalar(value)
    return (torch.full(shape, bits, dtype=torch.int64, device="mps"),)


def _full_like(x: MetalFloat64, value: float) -> MetalFloat64:
    """A contiguous tensor of ``x``'s shape filled exactly with ``value``."""
    return wrap(_const_comps(tuple(x.shape), float(value), x._mode), x._mode)


def _split(comps: tuple[torch.Tensor, ...], mode: str, n: int) -> list[MetalFloat64]:
    """Regroup a flat tuple of broadcast components into ``n`` MetalFloat64s."""
    k = 2 if mode == "df64" else 1
    return [wrap(comps[i * k : (i + 1) * k], mode) for i in range(n)]


def _launch(
    op: str,
    mode: str,
    *operands: Any,
    scalar: float | None = None,
    scalar_side: str | None = None,
) -> Any:
    """Launch ``op`` over prepared (contiguous, broadcast) operands and wrap."""
    lib = library(mode)
    ins = [
        o if isinstance(o, torch.Tensor) and not is_metal(o) else comps_arg(o)
        for o in operands
    ]
    res = lib.launch(op, *ins, scalar=scalar, scalar_side=scalar_side)
    count_gpu(op)
    return result_from_launch(res, mode)


def _missing(op: str, mode: str) -> bool:
    return not library(mode).has(op)


def _fallback_label(op: str) -> str:
    return f"{op}:missing_kernel"


def _unary(op: str, x: Any, fallback: Any = None) -> Any:
    """Run a unary kernel on ``x`` (MetalFloat64), or fall back to the CPU."""
    mode = mode_of(x)
    cx = coerce(x, mode)
    if _missing(op, mode):
        if fallback is None:
            raise NotImplementedError(f"MetalFloat64: kernel {op!r} missing in {mode}")
        return cpu_fallback(fallback, (cx,), {}, label=_fallback_label(op))
    return _launch(op, mode, _contig(cx))


#: Binary kernels evaluated on the host when *both* operands are scalars (a
#: Python number, a 0-d host tensor or a single-valued expansion of one): the
#: plain CPU float64 op, exactly as the nucleus host path would run it.
_HOST_SCALAR_OPS: dict[str, Any] = {
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": torch.div,
    "pow": torch.pow,
    "fmod": torch.fmod,
    "copysign": torch.copysign,
    "minimum": torch.minimum,
    "maximum": torch.maximum,
    "fmin": torch.fmin,
    "fmax": torch.fmax,
    "atan2": torch.atan2,
    "hypot": torch.hypot,
    "eq": torch.eq,
    "ne": torch.ne,
    "lt": torch.lt,
    "le": torch.le,
    "gt": torch.gt,
    "ge": torch.ge,
}


def _shape_of(x: Any) -> tuple[int, ...]:
    return tuple(x.shape) if isinstance(x, torch.Tensor) else ()


def _host_scalar_op(op: str, a: Any, b: Any, sa: float, sb: float, mode: str) -> Any:
    """``op`` on two scalars, on the host, expanded to the broadcast shape.

    ``n1 / n2`` with both the stride-0 expansion of one material constant
    over the ray bundle is scalar arithmetic: it is evaluated once in CPU
    float64 and returned as a host-resident stride-0 view, so no encoding,
    launch or sync happens and the result stays a kernel scalar for the next
    op. A comparison yields a 0-d CPU bool (which torch combines freely with
    device tensors) or, for a shaped result, an mps bool filled on the device:
    a shaped CPU bool could not be combined with the bundle's other masks.
    """
    fn = _HOST_SCALAR_OPS[op]
    r = fn(torch.tensor(sa, dtype=torch.float64), torch.tensor(sb, dtype=torch.float64))
    count_host(op)
    shape = torch.broadcast_shapes(_shape_of(a), _shape_of(b))
    if r.dtype == torch.bool:
        if len(shape):
            return torch.full(shape, bool(r), dtype=torch.bool, device="mps")
        return r
    if len(shape):
        # A stride-0 expansion of the one value; materialized (own contiguous
        # storage, torch's fresh-result semantics) by the nucleus before it is
        # written or viewed (``tensor.materialize_host``).
        out = wrap_host(r.reshape((1,) * len(shape)).expand(shape), mode)
        out._host_lazy = True
        return out
    return wrap_host(r, mode)


def _binary(op: str, a: Any, b: Any, fallback: Any = None) -> Any:
    """Run a binary kernel with torch broadcasting and scalar fast paths.

    ``a``/``b`` may each be a MetalFloat64, a plain tensor (promoted exactly),
    a Python number, a wrapped 0-d CPU tensor or a 0-d host-resident
    MetalFloat64 (exact scalar variant); a single-valued host tensor that
    broadcasts to the other operand's shape is a scalar as well, and two
    scalars are evaluated on the host (:func:`_host_scalar_op`).
    """
    mode = mode_of(a, b)
    sa, sb = scalar_value(a), scalar_value(b)
    if op in _HOST_SCALAR_OPS:
        ha = sa if sa is not None else scalar_like(a, _shape_of(a))
        hb = sb if sb is not None else scalar_like(b, _shape_of(b))
        if ha is not None and hb is not None:
            return _host_scalar_op(op, a, b, ha, hb, mode)
    if sa is None and sb is None:
        # single-valued host tensors that broadcast to the other operand's
        # shape (coefficient slices, expanded per-surface constants)
        sb = scalar_like(b, a.shape)
        if sb is None:
            sa = scalar_like(a, b.shape)
    if _missing(op, mode):
        if fallback is None:
            raise NotImplementedError(f"MetalFloat64: kernel {op!r} missing in {mode}")
        return cpu_fallback(
            fallback, (coerce(a, mode), coerce(b, mode)), {}, label=_fallback_label(op)
        )
    if sb is not None and sa is None:
        return _launch(
            op, mode, _contig(coerce(a, mode)), scalar=sb, scalar_side="right"
        )
    if sa is not None and sb is None:
        return _launch(
            op, mode, _contig(coerce(b, mode)), scalar=sa, scalar_side="left"
        )
    ca, cb = coerce(a, mode), coerce(b, mode)
    comps = broadcast_contiguous(*ca._comps, *cb._comps)
    return _launch(op, mode, *_split(comps, mode, 2))


#: Ternary kernels as the binary kernels they are composed of (``codegen.OPS``
#: expressions); used when an operand is a scalar, since only binary ops have
#: scalar kernel variants. Each binary kernel stores its (hi, lo) result
#: unchanged, so the decomposition is bit-identical to the fused kernel.
_TERNARY_DECOMPOSED: dict[str, Any] = {
    "lerp": lambda a, b, c: _binary("add", a, _binary("mul", c, _binary("sub", b, a))),
    "addcmul": lambda a, b, c: _binary("add", a, _binary("mul", b, c)),
    "addcdiv": lambda a, b, c: _binary("add", a, _binary("div", b, c)),
    "clamp": lambda a, b, c: _binary("minimum", _binary("maximum", a, b), c),
}


def _ternary(op: str, a: Any, b: Any, c: Any) -> Any:
    """Run a ternary value kernel with broadcasting.

    No scalar kernel variants exist for ternary ops; when an operand is a
    scalar (:func:`scalar_value`), the op is evaluated as the binary kernels
    it is composed of, so the scalar still travels as a kernel constant.
    """
    mode = mode_of(a, b, c)
    decomposed = _TERNARY_DECOMPOSED.get(op)
    if decomposed is not None:
        shape = torch.broadcast_shapes(
            *(tuple(v.shape) for v in (a, b, c) if isinstance(v, torch.Tensor))
        )
        if any(scalar_like(v, shape) is not None for v in (a, b, c)):
            return decomposed(a, b, c)
    ca, cb, cc = coerce(a, mode), coerce(b, mode), coerce(c, mode)
    comps = broadcast_contiguous(*ca._comps, *cb._comps, *cc._comps)
    return _launch(op, mode, *_split(comps, mode, 3))


def _host_fma(a: float, b: float, c: float) -> float:
    """``a * b + c`` with a single rounding on the host (exact rational sum)."""
    from fractions import Fraction

    try:
        return float(Fraction(a) * Fraction(b) + Fraction(c))
    except (ValueError, OverflowError):  # inf / nan operands
        return a * b + c


def _mul_add(a: Any, b: Any, c: Any) -> Any:
    """``a * b + c`` as torch's CPU kernels compute it.

    sf64: the softfloat ``fma`` kernel (one rounding), so ``add(alpha=)`` and
    ``lerp`` are bit-exact with torch's fused CPU paths; the scalar operands
    travel as 0-d device constants because ternary kernels have no scalar
    variants, and three scalars are fused exactly on the host. df64: ``mul``
    then ``add`` (the ``df::mul_add`` of the kernel library is the same two
    roundings), one extra ``u^2`` rounding compared with a fused op.
    """
    mode = mode_of(a, b, c)
    sa, sb, sc = scalar_value(a), scalar_value(b), scalar_value(c)
    if sa is not None and sb is not None and sc is not None:
        count_host("fma")
        return wrap_host(torch.tensor(_host_fma(sa, sb, sc), dtype=torch.float64), mode)
    if mode != "sf64" or _missing("fma", mode):
        return _binary("add", _binary("mul", a, b), c)
    ops = [
        wrap(_const_comps((), s, mode), mode) if s is not None else coerce(v, mode)
        for v, s in ((a, sa), (b, sb), (c, sc))
    ]
    comps = broadcast_contiguous(*ops[0]._comps, *ops[1]._comps, *ops[2]._comps)
    return _launch("fma", mode, *_split(comps, mode, 3))


def _where(cond: torch.Tensor, a: Any, b: Any) -> MetalFloat64:
    """``where(cond, a, b)`` through the ``where`` kernel (``cond``: a bool tensor)."""
    mode = mode_of(a, b)
    ca, cb = coerce(a, mode), coerce(b, mode)
    if cond.device.type != "mps":
        # A 0-d CPU bool from a scalar-scalar comparison (``b == 0`` in
        # floor_divide with a host-resident divisor).
        cond = cond.to("mps")
    comps = broadcast_contiguous(cond.to(torch.bool), *ca._comps, *cb._comps)
    return _launch("where", mode, comps[0], *_split(comps[1:], mode, 2))


# ---------------------------------------------------------------------------
# Variant plumbing (in-place / out=)
# ---------------------------------------------------------------------------
def _is_inplace(func: Any) -> bool:
    return func._schema.name.endswith("_")


def _store(dst: Any, src: Any, inplace: bool = False) -> None:
    """Copy ``src`` into ``dst`` (components for MetalFloat64, ``copy_`` otherwise).

    An ``out=`` destination is resized to ``src``'s shape first, like torch's
    ``out=``; an in-place ``self`` (``inplace=True``) is never resized: a
    broadcast result larger than ``self`` raises torch's shape error (resizing
    would grow a view's shared storage and overwrite its base). A plain
    floating-point ``dst`` receives the (same-category) downcast of a
    MetalFloat64 result; an integer or bool ``dst`` raises as torch does; a CPU
    ``dst`` raises torch's device error.
    """
    if inplace:
        _check_inplace_shape(dst, tuple(src.shape))
    else:
        _check_out_device(dst)
    if is_metal(dst):
        if is_metal(src):
            # An agnostic side follows the fixed one; two *fixed* representations
            # raise (an ``out=`` of the other representation, module docstring).
            mode_of(dst, src)
        src = coerce(src, dst._mode)
        _resize_out(dst, tuple(src.shape))
        host = getattr(dst, "_host", None)
        if host is not None:
            # host-resident destination: the CPU float64 copy is authoritative
            host.copy_(src.to_cpu_float64())
            dst.invalidate_host_cache()
            return
        for d, s in zip(dst._comps, src._comps, strict=True):
            d.copy_(s)
        return
    if is_metal(src) and not dst.dtype.is_floating_point:
        raise RuntimeError(
            "result type Double can't be cast to the desired output type "
            f"{_DTYPE_NAMES.get(dst.dtype, dst.dtype)}"
        )
    _resize_out(dst, tuple(src.shape))
    dst.copy_(src)


_DTYPE_NAMES = {
    torch.bool: "Bool",
    torch.uint8: "Byte",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
}


def _finish(func: Any, args: Any, kwargs: Any, result: Any) -> Any:
    """Route a fresh out-of-place ``result`` through the in-place / out= variants."""
    if _is_inplace(func):
        _store(args[0], result, inplace=True)
        return return_and_correct_aliasing(func, args, kwargs, args[0])
    out = kwargs.get("out")
    if out is not None:
        _store(out, result)
        return return_and_correct_aliasing(func, args, kwargs, out)
    return result


def _arg(args: Any, kwargs: Any, index: int, name: str, default: Any = None) -> Any:
    """Fetch a positional-or-keyword argument."""
    if len(args) > index:
        return args[index]
    return kwargs.get(name, default)


def _register_unary(op: str, *packets: Any) -> None:
    fallback = packets[0].default

    @implements(*packets)
    def _handler(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        return _finish(func, args, kwargs, _unary(op, args[0], fallback))


def _register_binary(op: str, *packets: Any) -> None:
    fallback = (
        packets[0].default if hasattr(packets[0], "default") else packets[0].Tensor
    )

    @implements(*packets)
    def _handler(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        return _finish(func, args, kwargs, _binary(op, args[0], args[1], fallback))


# ---------------------------------------------------------------------------
# Unary value ops (kernel name -> aten packets)
# ---------------------------------------------------------------------------
_UNARY_OPS: dict[str, tuple[Any, ...]] = {
    "neg": (aten.neg, aten.neg_),
    "abs": (aten.abs, aten.abs_),
    "sqrt": (aten.sqrt, aten.sqrt_),
    "rsqrt": (aten.rsqrt, aten.rsqrt_),
    "recip": (aten.reciprocal, aten.reciprocal_),
    "floor": (aten.floor, aten.floor_),
    "ceil": (aten.ceil, aten.ceil_),
    "trunc": (aten.trunc, aten.trunc_),
    "exp": (aten.exp, aten.exp_),
    "exp2": (aten.exp2, aten.exp2_),
    "expm1": (aten.expm1, aten.expm1_),
    "log": (aten.log, aten.log_),
    "log1p": (aten.log1p, aten.log1p_),
    "log2": (aten.log2, aten.log2_),
    "log10": (aten.log10, aten.log10_),
    "sin": (aten.sin, aten.sin_),
    "cos": (aten.cos, aten.cos_),
    "tan": (aten.tan, aten.tan_),
    "asin": (aten.asin, aten.asin_),
    "acos": (aten.acos, aten.acos_),
    "atan": (aten.atan, aten.atan_),
    "sinh": (aten.sinh, aten.sinh_),
    "cosh": (aten.cosh, aten.cosh_),
    "tanh": (aten.tanh, aten.tanh_),
    "asinh": (aten.asinh, aten.asinh_),
    "acosh": (aten.acosh, aten.acosh_),
    "atanh": (aten.atanh, aten.atanh_),
    "erf": (aten.erf, aten.erf_),
    "erfc": (aten.erfc, aten.erfc_),
    "erfinv": (aten.erfinv, aten.erfinv_),
    "lgamma": (aten.lgamma, aten.lgamma_),
}
for _op, _packets in _UNARY_OPS.items():
    _register_unary(_op, *_packets)


@implements(aten.sign, aten.sign_, aten.sgn, aten.sgn_)
def _sign(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    """``sign`` with torch semantics: ``sign(NaN) == 0`` and ``sign(-0.0) == +0.0``.

    The kernel returns NaN for NaN and preserves the sign of zero (numpy); torch
    computes ``(0 < x) - (x < 0)``, which is ``+0.0`` for both.
    """
    x = args[0]
    s = _unary("sign", x)
    zero = _unary("isnan", x) | _binary("eq", x, 0.0)
    # +0.0 is the all-zero bit pattern in both representations: exact fill.
    r = wrap(tuple(c.masked_fill(zero, 0) for c in s._comps), s._mode)
    return _finish(func, args, kwargs, r)


#: aten packet -> kernel op, for tests and reports.
UNARY_KERNEL_OPS: dict[str, str] = {
    p.__name__: op for op, ps in _UNARY_OPS.items() for p in ps
}
UNARY_KERNEL_OPS.update(
    {"sign": "sign", "sign_": "sign", "sgn": "sign", "sgn_": "sign"}
)

# ---------------------------------------------------------------------------
# Binary value ops without extra arguments
# ---------------------------------------------------------------------------
_BINARY_OPS: dict[str, tuple[Any, ...]] = {
    "mul": (aten.mul, aten.mul_, aten.multiply, aten.multiply_),
    "fmod": (aten.fmod, aten.fmod_),
    "copysign": (aten.copysign, aten.copysign_),
    "minimum": (aten.minimum, aten.min.other, aten.min.out),
    "maximum": (aten.maximum, aten.max.other, aten.max.out),
    "fmin": (aten.fmin,),
    "fmax": (aten.fmax,),
    "atan2": (aten.atan2, aten.atan2_),
    "hypot": (aten.hypot, aten.hypot_),
}
for _op, _packets in _BINARY_OPS.items():
    _register_binary(_op, *_packets)


@implements(aten.remainder, aten.remainder_)
def _remainder(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    """Python-style remainder whose zero results carry torch's sign.

    The ``remainder_py`` kernel follows Python (``copysign(0, b)`` for a zero
    result); torch keeps ``fmod``'s zero, i.e. the sign of the dividend.
    """
    a, b = args[0], args[1]
    r = _binary("remainder_py", a, b)
    zero_a = _binary("copysign", _full_like(r, 0.0), a)
    return _finish(func, args, kwargs, _where(_binary("eq", r, 0.0), zero_a, r))


# ---------------------------------------------------------------------------
# add / sub / rsub (alpha), div (rounding_mode), true_divide, floor_divide
# ---------------------------------------------------------------------------
def _is_bool(x: Any) -> bool:
    return isinstance(x, bool) or (
        isinstance(x, torch.Tensor) and not is_metal(x) and x.dtype == torch.bool
    )


def _alpha_value(alpha: Any) -> float:
    """The ``alpha`` scalar of add/sub; a boolean alpha raises as in torch."""
    if _is_bool(alpha):
        raise RuntimeError("Boolean alpha only supported for Boolean results.")
    a = scalar_value(alpha)
    if a is None:
        raise TypeError("alpha must be a Python scalar")
    return a


def _add_scaled(a: Any, b: Any, alpha: float) -> Any:
    """``a + alpha * b`` for the ``alpha`` argument of add/sub.

    Always :func:`_mul_add`, i.e. torch's fused ``fma(alpha, b, a)`` in sf64,
    for a tensor *and* a scalar ``b``: torch's CPU kernel contracts the
    wrapped scalar as well (the vectorized loop and its scalar tail alike on
    arm64), so pre-rounding ``alpha * b`` on the host would differ by up to
    hundreds of ulps under cancellation. ``alpha == 1`` is the plain ``add``.
    """
    if alpha == 1.0:
        return _binary("add", a, b)
    return _mul_add(b, alpha, a)


def _add_alpha(a: Any, b: Any, alpha: Any) -> Any:
    """``a + alpha * b``."""
    return _add_scaled(a, b, _alpha_value(alpha))


def _sub_alpha(a: Any, b: Any, alpha: Any) -> Any:
    """``a - alpha * b`` computed as torch does: ``add(a, b, -alpha)``.

    Negating ``alpha`` on the Python side reproduces torch's signed zeros
    (``sub(-0.0, b, alpha=0)`` is ``+0.0`` because integer ``-0`` is ``0``);
    ``alpha == 1`` keeps the single ``sub`` launch.
    """
    if _is_bool(a) or _is_bool(b):
        raise NotImplementedError(
            "Subtraction, the `-` operator, with a bool tensor is not supported. "
            "If you are trying to invert a mask, use the `~` or `logical_not()` "
            "operator instead."
        )
    if _is_bool(alpha):
        raise RuntimeError("Boolean alpha only supported for Boolean results.")
    if scalar_value(alpha) == 1.0:
        return _binary("sub", a, b)
    return _add_scaled(a, b, _alpha_value(-alpha))


@implements(aten.add, aten.add_)
def _add(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    alpha = _arg(args, kwargs, 2, "alpha", 1)
    return _finish(func, args, kwargs, _add_alpha(args[0], args[1], alpha))


@implements(aten.sub, aten.sub_, aten.subtract, aten.subtract_)
def _sub(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    alpha = _arg(args, kwargs, 2, "alpha", 1)
    return _finish(func, args, kwargs, _sub_alpha(args[0], args[1], alpha))


@implements(aten.rsub)
def _rsub(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    alpha = _arg(args, kwargs, 2, "alpha", 1)
    return _finish(func, args, kwargs, _sub_alpha(args[1], args[0], alpha))


def _as_operands(a: Any, b: Any) -> tuple[MetalFloat64, MetalFloat64]:
    """Both operands as MetalFloat64 (scalars become 0-d tensors that broadcast)."""
    mode = mode_of(a, b)
    return coerce(a, mode), coerce(b, mode)


def _floor_div(a: Any, b: Any) -> Any:
    """``a // b`` with torch's exact fmod-based algorithm (``div_floor_floating``).

    ``floor(a / b)`` is off by one whenever the rounded quotient lands on an
    integer (``1.0 // 0.1``); torch instead takes ``mod = fmod(a, b)`` (exact),
    ``div = (a - mod) / b``, shifts by one when ``mod`` and ``b`` disagree in
    sign, floors, and corrects a ``div - floor(div) > 0.5`` rounding. A zero
    quotient carries the sign of ``a / b``; ``b == 0`` returns ``a / b``.
    """
    a, b = _as_operands(a, b)
    q = _binary("div", a, b)
    mod = _binary("fmod", a, b)
    d = _binary("div", _binary("sub", a, mod), b)
    adjust = _binary("ne", mod, 0.0) & (_binary("lt", b, 0.0) ^ _binary("lt", mod, 0.0))
    d = _where(adjust, _binary("sub", d, 1.0), d)
    fl = _unary("floor", d)
    fl = _where(_binary("gt", _binary("sub", d, fl), 0.5), _binary("add", fl, 1.0), fl)
    zero_q = _binary("copysign", _full_like(q, 0.0), q)
    r = _where(_binary("ne", d, 0.0), fl, zero_q)
    return _where(_binary("eq", b, 0.0), q, r)


def _div(a: Any, b: Any, rounding_mode: str | None) -> Any:
    if rounding_mode is None:
        return _binary("div", a, b)
    if rounding_mode == "trunc":
        return _unary("trunc", _binary("div", a, b))
    if rounding_mode == "floor":
        return _floor_div(a, b)
    raise ValueError(
        "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
        f"but found {rounding_mode!r}"
    )


@implements(aten.div, aten.div_, aten.divide, aten.divide_)
def _div_handler(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    mode = (
        _arg(args, kwargs, 2, "rounding_mode")
        if "mode" in func._schema.overload_name
        else None
    )
    return _finish(func, args, kwargs, _div(args[0], args[1], mode))


@implements(aten.true_divide, aten.true_divide_)
def _true_divide(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(func, args, kwargs, _binary("div", args[0], args[1]))


@implements(aten.floor_divide, aten.floor_divide_)
def _floor_divide(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(func, args, kwargs, _div(args[0], args[1], "floor"))


# ---------------------------------------------------------------------------
# square / pow / float_power
# ---------------------------------------------------------------------------
def _pow_tensor_scalar(x: MetalFloat64, e: float) -> MetalFloat64:
    """``x ** e`` with the fast paths torch uses for common exponents.

    Like torch's ``pow_tensor_scalar_kernel``, ``0.5`` maps to ``sqrt`` and
    ``-0.5`` to ``rsqrt``; both differ from C ``pow`` only for ``x = -0.0``
    (``-0.0`` / ``-inf`` instead of ``+0.0`` / ``+inf``) and ``x = -inf``
    (``-inf`` / ``NaN`` instead of ``+inf`` / ``+0.0``), exactly as torch does.
    """
    if e == 0.0:
        return _full_like(x, 1.0)
    if e == 1.0:
        return _fresh(x)
    if e == 2.0:
        return _binary("mul", x, x)
    if e == 3.0:
        return _binary("mul", _binary("mul", x, x), x)
    if e == 0.5:
        return _unary("sqrt", x)
    if e == -1.0:
        return _unary("recip", x)
    if e == -2.0:
        return _unary("recip", _binary("mul", x, x))
    if e == -0.5:
        return _unary("rsqrt", x)
    return _binary("pow", x, e, aten.pow.Tensor_Scalar)


def _pow(a: Any, b: Any, scalar_exponent: bool) -> Any:
    """``a ** b``; the fast paths apply to the ``Scalar`` exponent overloads only.

    torch's ``pow.Tensor_Tensor`` uses C ``pow`` even for a 0-d exponent
    (``pow(-0.0, 0.5) == +0.0``, ``pow(-inf, -0.5) == +0.0``), while the
    ``Tensor_Scalar`` overload maps ``0.5`` / ``-0.5`` to ``sqrt`` / ``rsqrt``.
    """
    sb = scalar_value(b)
    if sb is not None and is_metal(a) and scalar_exponent:
        return _pow_tensor_scalar(a, sb)
    return _binary("pow", a, b, aten.pow.Tensor_Tensor)


@implements(aten.pow, aten.pow_, aten.float_power, aten.float_power_)
def _pow_handler(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    name = func._schema.overload_name
    scalar_exponent = name == "Tensor_Scalar" or (
        name == "Scalar" and _is_inplace(func)
    )
    return _finish(func, args, kwargs, _pow(args[0], args[1], scalar_exponent))


@implements(aten.square, aten.square_)
def _square(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(func, args, kwargs, _binary("mul", args[0], args[0]))


# ---------------------------------------------------------------------------
# round / frac / deg2rad / rad2deg
# ---------------------------------------------------------------------------
def _round(x: Any, decimals: int) -> Any:
    if decimals == 0:
        return _unary("rint", x)
    # torch: nearbyint(x * 10^d) / 10^d  (d > 0);  nearbyint(x / 10^-d) * 10^-d  (d < 0)
    if decimals > 0:
        ten_pow = 10.0**decimals
        return _binary("div", _unary("rint", _binary("mul", x, ten_pow)), ten_pow)
    ten_pow = 10.0 ** (-decimals)
    return _binary("mul", _unary("rint", _binary("div", x, ten_pow)), ten_pow)


@implements(aten.round, aten.round_)
def _round_handler(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    decimals = (
        kwargs.get("decimals", 0) if "decimals" in func._schema.overload_name else 0
    )
    return _finish(func, args, kwargs, _round(args[0], int(decimals)))


@implements(aten.frac, aten.frac_)
def _frac(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    return _finish(func, args, kwargs, _binary("sub", x, _unary("trunc", x)))


@implements(aten.deg2rad, aten.deg2rad_)
def _deg2rad(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(func, args, kwargs, _binary("mul", args[0], math.pi / 180.0))


@implements(aten.rad2deg, aten.rad2deg_)
def _rad2deg(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(func, args, kwargs, _binary("mul", args[0], 180.0 / math.pi))


# ---------------------------------------------------------------------------
# clamp family
# ---------------------------------------------------------------------------
def _clamp(x: Any, mn: Any, mx: Any) -> Any:
    if mn is None and mx is None:
        raise RuntimeError(
            "torch.clamp: At least one of 'min' or 'max' must not be None"
        )
    if mn is None:
        return _binary("minimum", x, mx)
    if mx is None:
        return _binary("maximum", x, mn)
    if scalar_value(mn) is not None or scalar_value(mx) is not None:
        return _binary("minimum", _binary("maximum", x, mn), mx)
    return _ternary("clamp", x, mn, mx)


@implements(aten.clamp, aten.clamp_)
def _clamp_handler(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    mn = _arg(args, kwargs, 1, "min")
    mx = _arg(args, kwargs, 2, "max")
    return _finish(func, args, kwargs, _clamp(args[0], mn, mx))


@implements(aten.clamp_min, aten.clamp_min_)
def _clamp_min(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(
        func, args, kwargs, _binary("maximum", args[0], _arg(args, kwargs, 1, "min"))
    )


@implements(aten.clamp_max, aten.clamp_max_)
def _clamp_max(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(
        func, args, kwargs, _binary("minimum", args[0], _arg(args, kwargs, 1, "max"))
    )


# ---------------------------------------------------------------------------
# ternary: lerp / addcmul / addcdiv
# ---------------------------------------------------------------------------
def _lerp_value(start: Any, end: Any, weight: Any) -> Any:
    """``lerp`` as torch's vectorized CPU kernel computes it.

    ``fma(coeff, end - start, base)`` with ``coeff = w, base = start`` for
    ``|w| < 0.5`` and ``coeff = w - 1, base = end`` otherwise (``lerp_vec``),
    so ``w == 1`` returns ``end`` exactly, infinities land on the expected
    side and the sign of a zero result is the vectorized path's. The fused
    step is :func:`_mul_add` (single rounding in sf64). A tensor weight is
    promoted into the op's representation first (never into the global one).
    """
    mode = mode_of(start, end, weight)
    w = scalar_value(weight)
    if scalar_value(start) is None:
        start = coerce(start, mode)  # plain start / end: into the op's mode
    if scalar_value(end) is None:
        end = coerce(end, mode)
    diff = _binary("sub", end, start)
    if w is not None:
        small = abs(w) < 0.5
        return _mul_add(w if small else w - 1.0, diff, start if small else end)
    weight = coerce(weight, mode)
    small = _binary("lt", _unary("abs", weight), 0.5)
    coeff = _where(small, weight, _binary("sub", weight, 1.0))
    base = _where(small, start, end)
    return _mul_add(coeff, diff, base)


_CPP_DTYPE_NAMES = {
    torch.float32: "float",
    torch.float16: "c10::Half",
    torch.bfloat16: "c10::BFloat16",
    torch.int64: "long long",
    torch.int32: "int",
    torch.int16: "short",
    torch.int8: "signed char",
    torch.uint8: "unsigned char",
    torch.bool: "bool",
}


@implements(aten.lerp, aten.lerp_)
def _lerp(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    weight = _arg(args, kwargs, 2, "weight")
    if (
        isinstance(weight, torch.Tensor)
        and not is_metal(weight)
        and weight.dim() > 0
        and weight.dtype != torch.float64
    ):
        raise RuntimeError(
            "expected dtype double for `weight` but got dtype "
            f"{_CPP_DTYPE_NAMES.get(weight.dtype, weight.dtype)}"
        )
    return _finish(func, args, kwargs, _lerp_value(args[0], args[1], weight))


def _addc(op: str, args: Any, kwargs: Any) -> Any:
    value = scalar_value(_arg(args, kwargs, 3, "value", 1))
    mode = mode_of(args[0], args[1], args[2])
    t1 = args[1] if value == 1.0 else _binary("mul", coerce(args[1], mode), value)
    return _ternary(op, args[0], t1, args[2])


@implements(aten.addcmul, aten.addcmul_)
def _addcmul(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(func, args, kwargs, _addc("addcmul", args, kwargs))


@implements(aten.addcdiv, aten.addcdiv_)
def _addcdiv(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(func, args, kwargs, _addc("addcdiv", args, kwargs))


# ---------------------------------------------------------------------------
# predicates and comparisons (bool MPS results)
# ---------------------------------------------------------------------------
for _op, _packet in (
    ("isnan", aten.isnan),
    ("isinf", aten.isinf),
    ("isfinite", aten.isfinite),
):
    _register_unary(_op, _packet)

for _op, _packets in (
    ("eq", (aten.eq, aten.eq_)),
    ("ne", (aten.ne, aten.not_equal, aten.ne_, aten.not_equal_)),
    ("lt", (aten.lt, aten.less, aten.lt_, aten.less_)),
    ("le", (aten.le, aten.less_equal, aten.le_, aten.less_equal_)),
    ("gt", (aten.gt, aten.greater, aten.gt_, aten.greater_)),
    ("ge", (aten.ge, aten.greater_equal, aten.ge_, aten.greater_equal_)),
):
    _register_binary(_op, *_packets)


def _signbit(x: MetalFloat64) -> torch.Tensor:
    """Sign bit from the components (exact: hi carries the sign of the pair)."""
    if x._mode == "df64":
        return torch.signbit(x._comps[0])
    return x._comps[0] < 0


@implements(aten.signbit)
def _signbit_handler(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(func, args, kwargs, _signbit(args[0]).contiguous())


@implements(aten.isposinf)
def _isposinf(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    return _finish(func, args, kwargs, _unary("isinf", x) & ~_signbit(x))


@implements(aten.isneginf)
def _isneginf(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    return _finish(func, args, kwargs, _unary("isinf", x) & _signbit(x))


def _truthy(x: Any) -> torch.Tensor:
    """``x != 0`` as a bool MPS tensor for MetalFloat64 or plain operands."""
    if is_metal(x):
        return _binary("ne", x, 0.0)
    s = scalar_value(x)
    if s is not None:
        return torch.tensor(s != 0.0, device="mps")
    if x.device.type != "mps":
        if x.dtype.is_floating_point or x.dtype.is_complex:
            # torch's rule (as ``coerce``): only 0-d CPU floats are scalars.
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found at "
                "least two devices, mps:0 and cpu!"
            )
        x = x.to("mps")  # CPU bool / integer masks: accepted like ``add`` does
    return x.ne(0)


@implements(aten.logical_not, aten.logical_not_)
def _logical_not(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(func, args, kwargs, _binary("eq", args[0], 0.0))


@implements(aten.logical_and, aten.logical_and_)
def _logical_and(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(func, args, kwargs, _truthy(args[0]) & _truthy(args[1]))


@implements(aten.logical_or, aten.logical_or_)
def _logical_or(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(func, args, kwargs, _truthy(args[0]) | _truthy(args[1]))


@implements(aten.logical_xor, aten.logical_xor_)
def _logical_xor(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(func, args, kwargs, _truthy(args[0]) ^ _truthy(args[1]))


# ---------------------------------------------------------------------------
# nan_to_num, sigmoid, softplus
# ---------------------------------------------------------------------------
@implements(aten.nan_to_num, aten.nan_to_num_)
def _nan_to_num(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = _contig(args[0])
    big = _MAX_FINITE[x._mode]
    nan = _arg(args, kwargs, 1, "nan")
    posinf = _arg(args, kwargs, 2, "posinf")
    neginf = _arg(args, kwargs, 3, "neginf")
    nan = 0.0 if nan is None else float(nan)
    posinf = big if posinf is None else float(posinf)
    neginf = -big if neginf is None else float(neginf)
    isinf = _unary("isinf", x)
    sb = _signbit(x)
    r = _where(_unary("isnan", x), _full_like(x, nan), x)
    r = _where(isinf & ~sb, _full_like(x, posinf), r)
    r = _where(isinf & sb, _full_like(x, neginf), r)
    return _finish(func, args, kwargs, r)


def _sigmoid(x: Any) -> Any:
    """``1 / (1 + exp(-x))``; saturates correctly through exp overflow."""
    return _unary(
        "recip", _binary("add", _unary("exp", _unary("neg", x), aten.exp.default), 1.0)
    )


@implements(aten.sigmoid, aten.sigmoid_)
def _sigmoid_handler(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(func, args, kwargs, _sigmoid(args[0]))


@implements(aten.sigmoid_backward)
def _sigmoid_backward(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    grad, out = args[0], args[1]
    r = _binary("mul", grad, _binary("mul", out, _binary("sub", 1.0, out)))
    return _finish(func, args, kwargs, r)


def _softplus_parts(
    x: Any, beta: Any, threshold: Any
) -> tuple[Any, torch.Tensor, float]:
    beta = scalar_value(beta)
    threshold = scalar_value(threshold)
    z = x if beta == 1.0 else _binary("mul", x, beta)
    return z, _binary("gt", z, threshold), beta


@implements(aten.softplus)
def _softplus(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    z, linear, beta = _softplus_parts(
        x, _arg(args, kwargs, 1, "beta", 1), _arg(args, kwargs, 2, "threshold", 20)
    )
    y = _unary("log1p", _unary("exp", z, aten.exp.default), aten.log1p.default)
    if beta != 1.0:
        y = _binary("div", y, beta)
    return _finish(func, args, kwargs, _where(linear, x, y))


@implements(aten.softplus_backward)
def _softplus_backward(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    grad, x = args[0], args[1]
    z, linear, _beta = _softplus_parts(
        x, _arg(args, kwargs, 2, "beta", 1), _arg(args, kwargs, 3, "threshold", 20)
    )
    ez = _unary("exp", z, aten.exp.default)
    r = _binary("mul", grad, _binary("div", ez, _binary("add", ez, 1.0)))
    return _finish(func, args, kwargs, _where(linear, grad, r))


# ---------------------------------------------------------------------------
# elementwise backward kernels autograd calls by name, and relu
# ---------------------------------------------------------------------------
@implements(aten.tanh_backward)
def _tanh_backward(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    grad, out = args[0], args[1]
    r = _binary("mul", grad, _binary("sub", 1.0, _binary("mul", out, out)))
    return _finish(func, args, kwargs, r)


@implements(aten.threshold_backward)
def _threshold_backward(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    grad, x, threshold = args[0], args[1], _arg(args, kwargs, 2, "threshold")
    keep = _binary("gt", x, threshold)
    return _finish(
        func,
        args,
        kwargs,
        _where(keep, grad, _full_like(coerce(grad, mode_of(grad, x)), 0.0)),
    )


@implements(aten.relu, aten.relu_)
def _relu(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    return _finish(func, args, kwargs, _binary("maximum", args[0], 0.0))


# ---------------------------------------------------------------------------
# digamma / polygamma: no kernel; needed by lgamma's backward (counted fallback)
# ---------------------------------------------------------------------------
@implements(aten.digamma, aten.digamma_)
def _digamma(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    x = args[0]
    r = cpu_fallback(aten.digamma.default, (coerce(x, mode_of(x)),), {}, "digamma")
    return _finish(func, args, kwargs, r)


@implements(aten.polygamma, aten.polygamma_)
def _polygamma(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    if func.overloadpacket is aten.polygamma_:
        x, n = args[0], args[1]
    else:
        n, x = args[0], args[1]
    r = cpu_fallback(
        aten.polygamma.default, (int(n), coerce(x, mode_of(x))), {}, "polygamma"
    )
    return _finish(func, args, kwargs, r)


__all__ = ["UNARY_KERNEL_OPS"]
