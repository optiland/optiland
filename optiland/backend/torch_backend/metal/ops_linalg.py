"""MetalFloat64 dispatch handlers: linear algebra (registered via ``implements``).

Products (``mm``, ``bmm``, ``addmm``, ``baddbmm``, ``addbmm``, ``mv``, ``dot``,
``vdot``, ``matmul``) run the batched GEMM / dot kernels of ``kernels/matmul.metal``
(compiled by :func:`ops_reduce.reduce_kernels`) on contiguous component tensors.
``outer``/``ger`` and ``cross`` are elementwise products of broadcast components.
The dot / GEMM accumulators start at ``+0``, so a product whose partial
products are all ``-0.0`` returns ``+0.0`` where torch CPU returns ``-0.0``
(sign of zero only; documented deviation). ``cdist`` has a GPU forward
(nucleus decomposition) and ``_cdist_backward`` here; ``dist`` is
``vector_norm(a - b)``.

Dense factorizations and solvers (``lstsq``, ``solve``, ``inv``, ``det``, ``slogdet``,
``eigh``, ``svd``, ``qr``, ``cholesky``, ``pinv``, ``lu``, ``lu_solve``, ``ldl``,
``cholesky_solve``, triangular solves, ``matrix_exp``, Householder products) are
explicit, counted CPU float64 fallbacks (``cpu_fallback:<label>`` in
``tensor.stats``); their autograd formulas run through the same fallbacks and the
structural ops registered below. The FFT primitives (``_fft_r2c``, ``_fft_c2c``) and
``linalg_eig``/``eigvals`` also run on the CPU but return their complex128 results
as CPU tensors: complex data is not an emulated type yet, so the caller (the
backend's ``to_complex``/FFT wrappers) is responsible for the host side of those
paths. ``irfft``-style ``_fft_c2r`` calls receive CPU complex input and never reach
this dispatch; their real output is a plain CPU float64 tensor.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from optiland.backend.torch_backend.metal.ops_reduce import (
    TG,
    _alloc,
    _choose_groups,
    ew,
    out_tensors,
    reduce_kernels,
    run_reduce,
)
from optiland.backend.torch_backend.metal.tensor import (
    MetalFloat64,
    aten,
    check_out_device,
    coerce,
    count_gpu,
    cpu_fallback,
    implements,
    mode_of,
    scalar_value,
    wrap,
)

#: Largest thread count issued to the matmul kernel in one dispatch.
MAX_THREADS = 1 << 30


# ---------------------------------------------------------------------------
# Kernel drivers
# ---------------------------------------------------------------------------
def matmul_comps(
    mode: str,
    a: tuple[torch.Tensor, ...],
    b: tuple[torch.Tensor, ...],
    batch: int,
    m: int,
    k: int,
    n: int,
) -> tuple[torch.Tensor, ...]:
    """``C[b] = A[b] @ B[b]`` on contiguous ``[batch, M, K]``, ``[batch, K, N]`` inputs.

    Returns:
        Flat component tensors with ``batch * M * N`` elements.
    """
    numel = batch * m * n
    out = _alloc(mode, numel)
    if numel == 0:
        return out
    if numel > MAX_THREADS:
        raise RuntimeError(
            f"matmul output of {numel} elements exceeds one dispatch; split the batch"
        )
    kernel = getattr(reduce_kernels(mode), f"matmul_{mode}")
    kernel(*a, *b, *out, batch, m, k, n, threads=[numel, 1, 1])
    count_gpu("matmul")
    return out


def dot_comps(
    mode: str, x: tuple[torch.Tensor, ...], y: tuple[torch.Tensor, ...], n: int
) -> tuple[torch.Tensor, ...]:
    """Dot product of two contiguous length-``n`` component vectors (1 element)."""
    groups = _choose_groups(n, 1)
    partial = _alloc(mode, groups)
    if n == 0:
        for c in partial:
            c.zero_()
        return partial
    kernel = getattr(reduce_kernels(mode), f"dot_{mode}")
    kernel(
        *x, *y, *partial, n, groups, threads=[groups * TG, 1, 1], group_size=[TG, 1, 1]
    )
    count_gpu("dot")
    if groups == 1:
        return partial
    return run_reduce(mode, "sum", partial, 1, groups, 1)


def _contig(x: MetalFloat64) -> tuple[torch.Tensor, ...]:
    return tuple(c.contiguous() for c in x._comps)


def _operands(*xs: Any) -> tuple[str, list[MetalFloat64]]:
    mode = mode_of(*xs)
    return mode, [coerce(x, mode) for x in xs]


def _scale(
    mode: str, comps: tuple[torch.Tensor, ...], factor: float
) -> tuple[torch.Tensor, ...]:
    return comps if factor == 1.0 else ew(mode, "mul", comps, scalar=factor)


def _affine(
    mode: str,
    bias: MetalFloat64,
    prod: tuple[torch.Tensor, ...],
    beta: float,
    alpha: float,
    shape: tuple[int, ...],
) -> MetalFloat64:
    """``beta * bias + alpha * prod`` with torch's ``beta == 0`` rule (bias ignored).

    The bias must still be expandable to the product shape (torch checks it
    before looking at ``beta``); the plain ``expand`` raises torch's message.
    """
    prod = _scale(mode, prod, alpha)
    if beta == 0.0:
        bias._comps[0].expand(shape)
        return wrap(tuple(c.reshape(shape) for c in prod), mode)
    res = ew(mode, "add", _scale(mode, bias._comps, beta), prod)
    if tuple(res[0].shape) != shape:
        raise RuntimeError(
            f"output with shape {tuple(res[0].shape)} doesn't match the broadcast "
            f"shape {shape}"
        )
    return wrap(res, mode)


def _finish(result: Any, func: Any, kwargs: Any) -> Any:
    """Write ``result`` into the overload's ``out=`` tensor (if any) and return it."""
    outs = out_tensors(func, kwargs)
    if not outs:
        return result
    (out,) = outs
    check_out_device(out, func._schema.name)
    if tuple(out.shape) != tuple(result.shape):
        out.resize_(result.shape)
    out.copy_(result)
    return out


# ---------------------------------------------------------------------------
# mm / bmm / addmm / baddbmm / addbmm / mv / dot / vdot
# ---------------------------------------------------------------------------
def _mm_comps(
    mode: str, a: MetalFloat64, b: MetalFloat64
) -> tuple[tuple[torch.Tensor, ...], tuple[int, int]]:
    if a.dim() != 2 or b.dim() != 2:
        raise RuntimeError(f"mm: expected 2-D tensors, got {a.dim()}-D and {b.dim()}-D")
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise RuntimeError(
            f"mat1 and mat2 shapes cannot be multiplied ({m}x{k} and {k2}x{n})"
        )
    return matmul_comps(mode, _contig(a), _contig(b), 1, m, k, n), (m, n)


@implements(aten.mm.default, aten.mm.out)
def _mm(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    mode, (a, b) = _operands(args[0], args[1])
    res, (m, n) = _mm_comps(mode, a, b)
    return _finish(wrap(tuple(c.reshape(m, n) for c in res), mode), func, kwargs)


def _bmm_comps(
    mode: str, a: MetalFloat64, b: MetalFloat64
) -> tuple[tuple[torch.Tensor, ...], tuple[int, int, int]]:
    if a.dim() != 3 or b.dim() != 3:
        raise RuntimeError(
            f"bmm: expected 3-D tensors, got {a.dim()}-D and {b.dim()}-D"
        )
    bs, m, k = a.shape
    bs2, k2, n = b.shape
    if bs != bs2 or k != k2:
        raise RuntimeError(
            f"bmm: shapes {tuple(a.shape)} and {tuple(b.shape)} cannot be multiplied"
        )
    return matmul_comps(mode, _contig(a), _contig(b), bs, m, k, n), (bs, m, n)


@implements(aten.bmm.default, aten.bmm.out)
def _bmm(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    mode, (a, b) = _operands(args[0], args[1])
    res, shape = _bmm_comps(mode, a, b)
    return _finish(wrap(tuple(c.reshape(shape) for c in res), mode), func, kwargs)


def _beta_alpha(kwargs: Any) -> tuple[float, float]:
    beta = scalar_value(kwargs.get("beta", 1))
    alpha = scalar_value(kwargs.get("alpha", 1))
    return float(beta), float(alpha)


@implements(aten.addmm.default, aten.addmm.out)
def _addmm(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    mode, (bias, a, b) = _operands(args[0], args[1], args[2])
    beta, alpha = _beta_alpha(kwargs)
    res, (m, n) = _mm_comps(mode, a, b)
    prod = tuple(c.reshape(m, n) for c in res)
    return _finish(_affine(mode, bias, prod, beta, alpha, (m, n)), func, kwargs)


@implements(aten.baddbmm.default, aten.baddbmm.out)
def _baddbmm(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    mode, (bias, a, b) = _operands(args[0], args[1], args[2])
    beta, alpha = _beta_alpha(kwargs)
    res, shape = _bmm_comps(mode, a, b)
    prod = tuple(c.reshape(shape) for c in res)
    return _finish(_affine(mode, bias, prod, beta, alpha, shape), func, kwargs)


@implements(aten.addbmm.default, aten.addbmm.out)
def _addbmm(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    mode, (bias, a, b) = _operands(args[0], args[1], args[2])
    beta, alpha = _beta_alpha(kwargs)
    res, (bs, m, n) = _bmm_comps(mode, a, b)
    summed = run_reduce(mode, "sum", res, 1, bs, m * n)
    count_gpu("sum")
    prod = tuple(c.reshape(m, n) for c in summed)
    return _finish(_affine(mode, bias, prod, beta, alpha, (m, n)), func, kwargs)


@implements(aten.mv.default, aten.mv.out)
def _mv(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    mode, (a, v) = _operands(args[0], args[1])
    if a.dim() != 2 or v.dim() != 1:
        raise RuntimeError("mv: expected a matrix and a vector")
    m, k = a.shape
    if v.shape[0] != k:
        raise RuntimeError(
            f"mv: size mismatch, got {tuple(a.shape)} and {tuple(v.shape)}"
        )
    res = matmul_comps(mode, _contig(a), _contig(v), 1, m, k, 1)
    return _finish(wrap(tuple(c.reshape(m) for c in res), mode), func, kwargs)


@implements(aten.dot.default, aten.dot.out, aten.vdot.default, aten.vdot.out)
def _dot(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    mode, (x, y) = _operands(args[0], args[1])
    if x.dim() != 1 or y.dim() != 1:
        raise RuntimeError(
            f"1D tensors expected, but got {x.dim()}D and {y.dim()}D tensors"
        )
    if x.shape[0] != y.shape[0]:
        raise RuntimeError(
            f"inconsistent tensor size, expected tensor [{x.shape[0]}] and src "
            f"[{y.shape[0]}] to have the same number of elements, but got "
            f"{x.shape[0]} and {y.shape[0]} elements respectively"
        )
    res = dot_comps(mode, _contig(x), _contig(y), x.shape[0])
    return _finish(wrap(tuple(c.reshape(()) for c in res), mode), func, kwargs)


@implements(aten.outer.default, aten.outer.out, aten.ger.default, aten.ger.out)
def _outer(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    mode, (x, y) = _operands(args[0], args[1])
    if x.dim() != 1 or y.dim() != 1:
        raise RuntimeError("outer: expected 1-D tensors")
    xc = tuple(c.reshape(-1, 1) for c in x._comps)
    yc = tuple(c.reshape(1, -1) for c in y._comps)
    return _finish(wrap(ew(mode, "mul", xc, yc), mode), func, kwargs)


@implements(aten.matmul.default, aten.matmul.out)
def _matmul(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    mode, (a, b) = _operands(args[0], args[1])
    da, db = a.dim(), b.dim()
    if da == 0 or db == 0:
        raise RuntimeError("both arguments to matmul need to be at least 1D")
    if da == 1 and db == 1:
        return _finish(_dot(aten.dot.default, types, (a, b), {}), func, kwargs)
    if da == 2 and db == 2:
        return _finish(_mm(aten.mm.default, types, (a, b), {}), func, kwargs)
    if da == 2 and db == 1:
        return _finish(_mv(aten.mv.default, types, (a, b), {}), func, kwargs)
    # general case on components: unsqueeze vectors, broadcast batch dims, bmm.
    ac = tuple(c.unsqueeze(0) if da == 1 else c for c in a._comps)
    bc = tuple(c.unsqueeze(-1) if db == 1 else c for c in b._comps)
    m, k = ac[0].shape[-2], ac[0].shape[-1]
    k2, n = bc[0].shape[-2], bc[0].shape[-1]
    if k != k2:
        raise RuntimeError(
            f"matmul: shapes {tuple(a.shape)} and {tuple(b.shape)} cannot be multiplied"
        )
    batch_shape = torch.broadcast_shapes(ac[0].shape[:-2], bc[0].shape[:-2])
    bs = math.prod(batch_shape)
    ae = tuple(c.expand(*batch_shape, m, k).reshape(bs, m, k).contiguous() for c in ac)
    be = tuple(c.expand(*batch_shape, k, n).reshape(bs, k, n).contiguous() for c in bc)
    res = matmul_comps(mode, ae, be, bs, m, k, n)
    shape = tuple(batch_shape) + (() if da == 1 else (m,)) + (() if db == 1 else (n,))
    return _finish(wrap(tuple(c.reshape(shape) for c in res), mode), func, kwargs)


# ---------------------------------------------------------------------------
# cdist backward / dist (the forward decompositions live in the nucleus)
# ---------------------------------------------------------------------------
@implements(aten._cdist_backward.default)
def _cdist_backward(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    """Gradient of ``cdist(x1, x2, p)`` w.r.t. ``x1`` (torch's formulas).

    ``p = 2``: ``grad * diff / dist``, ``p = 1``: ``grad * sign(diff)``,
    ``p = inf``: ``grad * sign(diff) * (|diff| == dist)``, otherwise
    ``grad * sign(diff) * |diff|^(p-1) / dist^(p-1)``; a zero distance
    contributes nothing. Computed with the wrapper's own ops (which re-enter
    dispatch), then summed over the ``x2`` rows.
    """
    grad, x1, x2, p, dist = args[0], args[1], args[2], float(args[3]), args[4]
    diff = x1.unsqueeze(-2) - x2.unsqueeze(-3)  # [..., n1, n2, d]
    g = grad.unsqueeze(-1)
    d = dist.unsqueeze(-1)
    if p == 2.0:
        safe = torch.where(d == 0, torch.ones_like(d), d)
        res = torch.where(d == 0, torch.zeros_like(diff), g * diff / safe)
    elif p == 1.0:
        res = g * torch.sign(diff)
    elif p == float("inf"):
        res = g * torch.sign(diff) * (diff.abs() == d)
    elif p == 0.0:
        res = torch.zeros_like(diff)
    else:
        safe = torch.where(d == 0, torch.ones_like(d), d)
        scaled = torch.sign(diff) * diff.abs() ** (p - 1.0) / safe ** (p - 1.0)
        res = torch.where(d == 0, torch.zeros_like(diff), g * scaled)
    return res.sum(-2)


@implements(aten.dist.default, aten.dist.out)
def _dist(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    """``dist(a, b, p)`` = ``linalg.vector_norm(a - b, p)`` (through dispatch)."""
    a, b = args[0], args[1]
    p = scalar_value(kwargs.get("p", args[2] if len(args) > 2 else 2))
    return _finish(torch.linalg.vector_norm(a - b, float(p)), func, kwargs)


# ---------------------------------------------------------------------------
# cross
# ---------------------------------------------------------------------------
def _cross_comps(
    mode: str, a: MetalFloat64, b: MetalFloat64, dim: int
) -> tuple[torch.Tensor, ...]:
    shape = torch.broadcast_shapes(a.shape, b.shape)
    ac = tuple(c.expand(shape) for c in a._comps)
    bc = tuple(c.expand(shape) for c in b._comps)

    def part(t: tuple[torch.Tensor, ...], i: int) -> tuple[torch.Tensor, ...]:
        return tuple(c.select(dim, i) for c in t)

    a0, a1, a2 = (part(ac, i) for i in range(3))
    b0, b1, b2 = (part(bc, i) for i in range(3))
    # The df64 product is not bit-commutative (the cross term of DWTimesDW3
    # is ``fma(xh, yl, xl * yh)``), so the second product of every component
    # takes its operands in the same (a_i, b_j) -> (b_j, a_i) order as the
    # first: ``cross(a, a)`` is then exactly zero, as in IEEE arithmetic
    # (``PolarizedRays.get_local_basis`` relies on ``mag == 0`` for
    # undeviated rays).
    r0 = ew(mode, "sub", ew(mode, "mul", a1, b2), ew(mode, "mul", b1, a2))
    r1 = ew(mode, "sub", ew(mode, "mul", a2, b0), ew(mode, "mul", b2, a0))
    r2 = ew(mode, "sub", ew(mode, "mul", a0, b1), ew(mode, "mul", b0, a1))
    return tuple(
        torch.stack([r0[i], r1[i], r2[i]], dim=dim).contiguous() for i in range(len(r0))
    )


@implements(aten.linalg_cross.default, aten.linalg_cross.out)
def _linalg_cross(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    mode, (a, b) = _operands(args[0], args[1])
    dim = int(kwargs.get("dim", args[2] if len(args) > 2 else -1))
    if a.dim() != b.dim():
        raise RuntimeError(
            "linalg.cross: inputs must have the same number of dimensions."
        )
    shape = torch.broadcast_shapes(a.shape, b.shape)
    d = dim + len(shape) if dim < 0 else dim
    if not 0 <= d < len(shape) or shape[d] != 3:
        raise RuntimeError(
            f"linalg.cross: inputs dimension {dim} must have length 3. "
            f"Got {shape[d] if 0 <= d < len(shape) else 'none'}"
        )
    return _finish(wrap(_cross_comps(mode, a, b, d), mode), func, kwargs)


@implements(aten.cross.default, aten.cross.out)
def _cross(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
    mode, (a, b) = _operands(args[0], args[1])
    dim = kwargs.get("dim", args[2] if len(args) > 2 else None)
    shape = torch.broadcast_shapes(a.shape, b.shape)
    if dim is None:
        candidates = [i for i, s in enumerate(shape) if s == 3]
        if not candidates:
            raise RuntimeError("no dimension of size 3 in input")
        d = candidates[0]
    else:
        d = int(dim) + len(shape) if int(dim) < 0 else int(dim)
        if not 0 <= d < len(shape) or shape[d] != 3:
            raise RuntimeError(f"cross: dimension {dim} does not have size 3")
    return _finish(wrap(_cross_comps(mode, a, b, d), mode), func, kwargs)


# ---------------------------------------------------------------------------
# CPU float64 fallbacks (dense factorizations, solvers, FFT)
# ---------------------------------------------------------------------------
_CPU_FALLBACKS: dict[str, tuple[Any, ...]] = {
    "linalg_lstsq": (aten.linalg_lstsq,),
    "linalg_solve": (aten._linalg_solve_ex,),
    "linalg_inv": (aten.linalg_inv_ex,),
    "linalg_det": (aten._linalg_det,),
    "linalg_slogdet": (aten._linalg_slogdet,),
    "linalg_eigh": (aten._linalg_eigh,),
    "linalg_svd": (aten._linalg_svd,),
    "linalg_qr": (aten.linalg_qr,),
    "linalg_cholesky": (aten.linalg_cholesky_ex,),
    "linalg_pinv": (aten.linalg_pinv,),
    "linalg_lu": (aten.linalg_lu,),
    "linalg_lu_factor": (aten.linalg_lu_factor_ex,),
    "linalg_lu_solve": (aten.linalg_lu_solve, aten.lu_unpack),
    "linalg_ldl": (aten.linalg_ldl_factor_ex, aten.linalg_ldl_solve),
    "cholesky_solve": (aten.cholesky_solve, aten.cholesky_inverse),
    "triangular_solve": (aten.triangular_solve,),
    "linalg_solve_triangular": (aten.linalg_solve_triangular,),
    "linalg_matrix_exp": (aten.linalg_matrix_exp,),
    "linalg_householder_product": (
        aten.linalg_householder_product,
        aten.geqrf,
        aten.ormqr,
    ),
    "_fft_c2r": (aten._fft_c2r,),
}


def _make_fallback(label: str) -> Any:
    def handler(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        return cpu_fallback(func, args, kwargs, label=label)

    handler.__name__ = f"_fallback_{label}"
    return handler


for _label, _ops in _CPU_FALLBACKS.items():
    implements(*_ops)(_make_fallback(_label))


class _CpuResult:
    """Opaque holder so ``cpu_fallback`` leaves CPU (complex) results untouched."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


def cpu_only_fallback(func: Any, args: Any, kwargs: Any, label: str) -> Any:
    """Run ``func`` on decoded CPU float64 operands; results stay CPU tensors.

    Counted and strict-mode checked like :func:`cpu_fallback`, but the outputs are
    not re-encoded: used for ops whose results are complex (no emulated type yet).
    """

    def run(*a: Any, **k: Any) -> _CpuResult:
        return _CpuResult(func(*a, **k))

    return cpu_fallback(run, args, kwargs, label=label).value


_CPU_ONLY = {
    "_fft_r2c": (aten._fft_r2c,),
    "_fft_c2c": (aten._fft_c2c,),
    "linalg_eig": (aten.linalg_eig, aten._linalg_eigvals, aten.linalg_eigvals),
}


def _make_cpu_only(label: str) -> Any:
    def handler(func: Any, types: Any, args: Any, kwargs: Any) -> Any:
        return cpu_only_fallback(func, args, kwargs, label=label)

    handler.__name__ = f"_cpu_only_{label}"
    return handler


for _label, _ops in _CPU_ONLY.items():
    implements(*_ops)(_make_cpu_only(_label))


__all__ = [
    "MAX_THREADS",
    "cpu_only_fallback",
    "dot_comps",
    "matmul_comps",
]
