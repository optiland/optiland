"""Scalar math for the compiled float64 loops of the NumPy backend.

The compiled fast paths of the NumPy backend (``_trace_kernel.py`` and
``_aim.py``) repeat NumPy expressions and must give their bits. They share
the functions of this module, which give the bits of their NumPy
counterparts for float64 values.

Manuel Fragata Mendes, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit, types
from numba.extending import intrinsic

if TYPE_CHECKING:
    from collections.abc import Callable

    from numba.core.typing.templates import Signature


@intrinsic
def _llvm_sqrt(
    typingctx: object, value: types.Type
) -> tuple[Signature, Callable] | None:
    """Square root as the LLVM intrinsic ``llvm.sqrt.f64``."""
    if value != types.float64:
        return None

    def codegen(context, builder, signature, args):
        function = builder.module.declare_intrinsic("llvm.sqrt", [args[0].type])
        return builder.call(function, args)

    return types.float64(types.float64), codegen


@njit(inline="always", error_model="numpy", cache=True)
def _sqrt(value: float) -> float:
    """Correctly rounded square root (IEEE 754), as ``numpy.sqrt``.

    Numba lowers ``math.sqrt`` to a call of the C library, which can set
    errno, so a loop that contains it does not vectorize. The intrinsic gives
    the same value, and NaN for a negative value.
    """
    return _llvm_sqrt(value)
