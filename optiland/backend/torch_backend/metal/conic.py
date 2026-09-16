"""Fused finite-conic intersection for ``MetalFloat64`` rays.

``conic_intersection`` on the torch backend evaluates the shared
``optiland.backend._conic._conic_candidates`` arithmetic with ~40 elementwise
tensor ops. On the emulated-float64 Metal path every one of those is a kernel
launch, so this module runs the identical arithmetic in ONE launch
(``kernels/conic.metal``) and applies only the aperture preference and final
selection with tensor ops. The result is bit-for-bit the same selection logic
(masks are computed by the same formulas in the same precision) and the same
implicit-derivative backward as ``_ConicCPU`` in ``torch_backend/conic.py``.
"""

from __future__ import annotations

from typing import Any

import torch

from optiland.backend._conic import _Candidates, _select_distance
from optiland.backend.torch_backend.metal import compile as _compile
from optiland.backend.torch_backend.metal import encode
from optiland.backend.torch_backend.metal.tensor import (
    MetalFloat64,
    coerce,
    count_gpu,
    is_metal,
    wrap,
)

_LIBS: dict[str, Any] = {}


def _kernel_library(mode: str) -> Any:
    """Compile (once per process) the representation's headers plus ``conic.metal``."""
    lib = _LIBS.get(mode)
    if lib is None:
        from optiland.backend.torch_backend.metal.library import HEADERS

        names = [h for h in HEADERS[mode] if (_compile._KERNEL_DIR / h).is_file()]
        lib = _compile.compile_library(_compile.kernel_source(*names, "conic.metal"))
        _LIBS[mode] = lib
    return lib


def _scalar_float(v: Any) -> float:
    if is_metal(v):
        return float(v.to_numpy().reshape(()))
    if isinstance(v, torch.Tensor):
        return float(v.detach().reshape(()).item())
    return float(v)


def can_fuse_metal(values: tuple, radius: Any, conic: Any) -> bool:
    """True when the six ray arrays are same-shape MetalFloat64 and R/k are scalars."""
    first = values[0]
    if not is_metal(first):
        return False
    mode = first.mode
    if not all(
        is_metal(v) and v.mode == mode and v.shape == first.shape for v in values
    ):
        return False
    for p in (radius, conic):
        if is_metal(p) or isinstance(p, torch.Tensor):
            if p.numel() != 1:
                return False
        elif not isinstance(p, (int, float)):
            return False
    return True


def conic_candidates(
    x: MetalFloat64,
    y: MetalFloat64,
    z: MetalFloat64,
    L: MetalFloat64,
    M: MetalFloat64,
    N: MetalFloat64,
    radius: Any,
    conic: Any,
) -> _Candidates:
    """Run the fused kernel: ``_Candidates`` with MetalFloat64 roots and bool masks."""
    mode = x.mode
    lib = _kernel_library(mode)
    n = x.numel()
    shape = x.shape
    comps = [
        c.contiguous().reshape(-1)
        for v in (x, y, z, L, M, N)
        for c in v.detach().components
    ]
    r, k = _scalar_float(radius), _scalar_float(conic)
    flags = torch.empty(n, dtype=torch.uint8, device="mps")
    if mode == "df64":
        outs = [torch.empty(n, dtype=torch.float32, device="mps") for _ in range(4)]
        if n:
            lib.conic_candidates_df64(
                *comps,
                encode.df64_scalar(r),
                encode.df64_scalar(k),
                encode.df64_scalar(2.0**-48),
                *outs,
                flags,
                threads=[n, 1, 1],
            )
        t1 = wrap((outs[0].reshape(shape), outs[1].reshape(shape)), mode)
        t2 = wrap((outs[2].reshape(shape), outs[3].reshape(shape)), mode)
    else:
        outs = [torch.empty(n, dtype=torch.int64, device="mps") for _ in range(2)]
        if n:
            lib.conic_candidates_sf64(
                *comps,
                encode.sf64_scalar(r),
                encode.sf64_scalar(k),
                encode.sf64_scalar(2.0**-53),
                *outs,
                flags,
                threads=[n, 1, 1],
            )
        t1 = wrap((outs[0].reshape(shape),), mode)
        t2 = wrap((outs[1].reshape(shape),), mode)
    count_gpu("conic_candidates")
    f = flags.reshape(shape)
    bit = lambda b: (f & (1 << b)) != 0  # noqa: E731
    return _Candidates(t1, t2, bit(0), bit(1), bit(2), bit(3), bit(4))


def _partials(inputs: tuple, distance: Any, regular: Any) -> tuple:
    """Implicit derivatives of the selected root (same formulas as ``_ConicCPU``)."""
    from optiland.backend.torch_backend.conic import _partials as cpu_partials

    return cpu_partials(inputs, distance, regular)


class _ConicMetal(torch.autograd.Function):
    """Fused forward on the GPU, exact implicit derivative in the backward pass."""

    @staticmethod
    def forward(contains: Any, *inputs: Any) -> tuple[Any, Any]:
        x, y, z, L, M, N, radius, conic = inputs
        roots = conic_candidates(x, y, z, L, M, N, radius, conic)
        distance = _select_distance(roots, inputs[:6], contains, torch.where)
        return distance, roots.regular

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: tuple) -> None:
        inputs = inputs[1:]
        distance, regular = output
        ctx.mark_non_differentiable(regular)
        ctx.save_for_backward(*inputs, distance, regular)

    @staticmethod
    def backward(ctx: Any, gradient: Any, unused: Any) -> tuple:
        *inputs, distance, regular = ctx.saved_tensors
        return (
            None,
            *(
                (gradient * partial).sum_to_size(value.shape)
                for value, partial in zip(
                    inputs, _partials(inputs, distance, regular), strict=True
                )
            ),
        )


def conic_intersection_metal(
    x: Any,
    y: Any,
    z: Any,
    L: Any,
    M: Any,
    N: Any,
    radius: Any,
    conic: Any,
    contains: Any = None,
) -> Any:
    """Drop-in for ``TorchBackend.conic_intersection`` on MetalFloat64 rays.

    ``radius`` / ``conic`` become (host-resident) MetalFloat64 scalars so the
    backward's ``torch.where(regular, value, 0.0)`` over the mps ``regular``
    mask stays on the dispatch path (a plain CPU float64 parameter would be
    promoted to float64 *on mps* by torch and rejected).
    """
    mode = x.mode
    radius_t = radius if is_metal(radius) else coerce(radius, mode)
    conic_t = conic if is_metal(conic) else coerce(conic, mode)
    distance, _ = _ConicMetal.apply(contains, x, y, z, L, M, N, radius_t, conic_t)
    return distance


__all__ = ["can_fuse_metal", "conic_candidates", "conic_intersection_metal"]
