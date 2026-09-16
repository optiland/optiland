"""Host-side encoding between float64 and the two emulated representations.

``df64`` (double-single) stores a float64 value ``x`` as an unevaluated sum of two
float32 words, ``hi = float32(x)`` and ``lo = float32(x - hi)``, renormalized with
one float32 Fast2Sum so that the pair is in the RNE-canonical form the kernels
produce (a ``lo`` of exactly ``+-ulp(hi)/2`` next to an odd ``hi`` becomes the
even-``hi`` form; the value is unchanged). The pair carries about 48 significant
bits; the encoding error is at most ``2^-48`` relative. ``lo`` is forced to zero
wherever ``hi`` is not finite so that ``inf`` never carries a ``NaN`` low word
(``inf - inf``) into the kernels. The df64 overflow threshold is the float32
RNE tie ``FLT_MAX + 2^103``: values whose 48-bit rounding reaches it (``|x| >=
FLT_MAX + 2^103 - 2^78``) encode as a clean ``+-inf`` (matching the kernels'
overflow rule, see ``df64_core.h``).

The GPU flushes float32 denormals to zero in arithmetic *and* comparisons while
loads and stores keep them (NOTES/01-environment-findings.md), so a denormal word
would compare equal to zero yet decode as nonzero. The encoder therefore flushes
denormal ``hi`` words (``1.4e-45 <= |x| < 1.18e-38``) to a signed zero and
denormal ``lo`` words to ``+0``: the decoded pair is then exactly what every
kernel computes with. Values that need that range must use ``sf64``.

``sf64`` (software binary64) stores the exact IEEE-754 bit pattern of ``x`` in a
signed 64-bit integer, which is what ``torch`` can hold on ``mps`` and what the
Metal ``long`` type binds to.

Scalars for kernel arguments follow the same rules: PyTorch passes Python
floats to ``torch.mps.compile_shader`` kernels as 32-bit values, so a df64
scalar is passed as a ``[hi, lo]`` list (bound to ``constant float2&``) and an
sf64 scalar as a Python ``int`` holding the signed two's-complement bit pattern
(bound to ``constant long&``).

Input validation (fix round 3): the encoders take real-valued array-likes only.
``None``, strings, bytes, complex values and object arrays raise ``TypeError``
(``np.asarray(..., float64)`` would otherwise turn ``None`` into NaN, parse
``"1.5"`` and drop imaginary parts). The sf64 encoders additionally reject
integer and bool inputs, because an int64 array is exactly what the sf64
pipeline produces (bit patterns): re-encoding one by value would corrupt it
silently (use :func:`decode_sf64` to get the values back). The df64 decoders
require float32 components of equal shape, matching what the kernels produce
and what ``MetalLibrary`` accepts; a float64 array passed as a component (an
already decoded value, say) would otherwise be rounded to 24 bits without
notice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

ArrayLike = Any

#: Unit roundoff squared for df64: the relative precision of the representation.
DF64_U2 = 2.0**-48
#: Smallest normal float32; words below it are flushed by the encoder (GPU FTZ).
DF64_FLT_MIN = 2.0**-126

_REJECTED_SCALARS = (str, bytes, complex)


def _as_float64(x: ArrayLike, what: str, allow_integer: bool) -> np.ndarray:
    """Validate a real-valued array-like and return it as a float64 array.

    Args:
        x: The input.
        what: Name of the calling encoder for the error message.
        allow_integer: Whether integer / bool inputs are acceptable (they are
            for df64, where they cannot be mistaken for encoded components).

    Raises:
        TypeError: For ``None``, strings, bytes, complex values, object arrays
            and (when ``allow_integer`` is False) integer or bool inputs.
    """
    if x is None or isinstance(x, _REJECTED_SCALARS):
        raise TypeError(f"{what}: expected real float64 values, got {type(x).__name__}")
    arr = np.asarray(x)
    kind = arr.dtype.kind
    if kind in "biu":
        if not allow_integer:
            raise TypeError(
                f"{what}: expected float64 values, got {arr.dtype} (an int64 array "
                "is an sf64 bit-pattern component; use decode_sf64 to get values)"
            )
    elif kind != "f":
        raise TypeError(f"{what}: expected real float64 values, got dtype {arr.dtype}")
    return arr.astype(np.float64, copy=False)


def _as_float32_component(x: ArrayLike, what: str) -> np.ndarray:
    """Return ``x`` as a float32 array, rejecting any other dtype."""
    arr = np.asarray(x)
    if arr.dtype != np.float32:
        raise TypeError(
            f"{what}: expected float32 components, got {arr.dtype} (the kernels "
            "produce float32 hi/lo words; a float64 value here would be rounded)"
        )
    return arr


# ---------------------------------------------------------------------------
# df64 <-> float64
# ---------------------------------------------------------------------------
def encode_df64(x: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Split float64 values into normalized ``(hi, lo)`` float32 components.

    Args:
        x: Array-like of float64 values (any shape, scalars allowed; integer and
            bool inputs are converted exactly, everything else raises).

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(hi, lo)`` float32 arrays with the shape of
        ``x``. ``lo`` is zero wherever ``hi`` is ``inf`` or ``NaN``; ``hi`` keeps
        the sign of zero.

    Raises:
        TypeError: For ``None``, str, bytes, complex or object inputs.
    """
    x64 = _as_float64(x, "encode_df64", allow_integer=True)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        hi = x64.astype(np.float32)
        finite = np.isfinite(hi)
        lo = np.where(finite, x64 - hi.astype(np.float64), 0.0).astype(np.float32)
        # Denormal words are flushed (see the module docstring); the sign of a
        # flushed hi word is kept so that -1e-40 encodes as -0.0.
        hi = np.where(np.abs(hi) < DF64_FLT_MIN, np.copysign(np.float32(0), hi), hi)
        lo = np.where(np.abs(lo) < DF64_FLT_MIN, np.float32(0), lo)
        # Float32 Fast2Sum: canonical even-hi form for tie-valued lo words (exact,
        # |lo| <= ulp(hi)/2 already holds); non-finite hi keeps lo = 0 and a zero
        # keeps its sign (-0 + +0 would be +0). When the Fast2Sum itself overflows
        # (hi = +-FLT_MAX with lo rounded to +-2^103, i.e. |x| within 2^-50 of the
        # float32 overflow tie) the value is at the df64 overflow threshold and
        # becomes a clean +-inf with lo = 0 (never the (inf, -inf) pair that the
        # error term would produce, which decodes to NaN).
        s = (hi + lo).astype(np.float32)
        e = (lo - (s - hi)).astype(np.float32)
        overflow = finite & ~np.isfinite(s)
        keep = finite & (s != 0) & ~overflow
        hi = np.where(keep, s, hi)
        lo = np.where(keep, e, np.float32(0))
        hi = np.where(overflow, np.copysign(np.float32(np.inf), hi), hi)
    # ``x - hi`` may produce -0.0 for negative x; the value is 0 either way, but
    # keep lo's zero positive so the pair is canonical.
    lo = (lo + np.float32(0.0)).astype(np.float32)
    return np.asarray(hi, dtype=np.float32), np.asarray(lo, dtype=np.float32)


def decode_df64(hi: ArrayLike, lo: ArrayLike) -> np.ndarray:
    """Reconstruct float64 values from ``(hi, lo)`` components.

    The sum ``hi + lo`` is computed in float64 (exact for normalized pairs). A
    zero result takes its sign from ``hi``, so ``-0.0`` survives a round trip.

    Args:
        hi: High float32 components.
        lo: Low float32 components (same shape as ``hi``).

    Returns:
        np.ndarray: float64 array with the shape of ``hi``.

    Raises:
        TypeError: If a component is not float32.
        ValueError: If ``hi`` and ``lo`` have different shapes.
    """
    hi32 = _as_float32_component(hi, "decode_df64 hi")
    lo32 = _as_float32_component(lo, "decode_df64 lo")
    if hi32.shape != lo32.shape:
        raise ValueError(f"decode_df64: hi shape {hi32.shape} != lo shape {lo32.shape}")
    hi64 = hi32.astype(np.float64)
    lo64 = lo32.astype(np.float64)
    with np.errstate(invalid="ignore"):
        out = hi64 + lo64
    zero = (hi64 == 0.0) & (lo64 == 0.0)
    if np.any(zero):
        out = np.where(zero, hi64, out)
    return out


def df64_scalar(x: float) -> list[float]:
    """Encode one float64 scalar as a ``[hi, lo]`` list for ``constant float2&``.

    Args:
        x: The scalar value (a real number; bool, None, str, complex raise).

    Returns:
        list[float]: Two Python floats, each exactly representable in float32.
    """
    hi, lo = encode_df64(_scalar_float64(x, "df64_scalar"))
    return [float(hi), float(lo)]


def _scalar_float64(x: Any, what: str) -> np.float64:
    if x is None or isinstance(x, (bool, *_REJECTED_SCALARS)):
        raise TypeError(f"{what}: expected a real number, got {type(x).__name__}")
    arr = np.asarray(x)
    if arr.ndim != 0 or arr.dtype.kind not in "fiu":
        raise TypeError(
            f"{what}: expected a real scalar, got {type(x).__name__} of dtype "
            f"{arr.dtype} and shape {arr.shape}"
        )
    return np.float64(arr)


# ---------------------------------------------------------------------------
# sf64 <-> float64
# ---------------------------------------------------------------------------
def encode_sf64(x: ArrayLike) -> np.ndarray:
    """Return the IEEE binary64 bit patterns of ``x`` as an int64 array.

    Args:
        x: Array-like of float64 values.

    Returns:
        np.ndarray: int64 array (a bit-for-bit view, so NaN payloads and signed
        zeros are preserved).

    Raises:
        TypeError: For integer / bool inputs (an int64 array is an sf64 component,
            see the module docstring) and for None, str, bytes, complex or
            object inputs.
    """
    x64 = np.ascontiguousarray(_as_float64(x, "encode_sf64", allow_integer=False))
    return x64.view(np.int64)


def decode_sf64(bits: ArrayLike) -> np.ndarray:
    """Reinterpret int64 bit patterns as float64 values.

    Args:
        bits: Array-like of int64 bit patterns.

    Returns:
        np.ndarray: float64 array with the shape of ``bits``.

    Raises:
        TypeError: If ``bits`` is not an integer array (a float64 array here is
            most likely an already decoded value).
    """
    arr = np.asarray(bits)
    if arr.dtype.kind not in "iu" or arr.dtype.itemsize != 8:
        raise TypeError(
            f"decode_sf64: expected int64 bit patterns, got dtype {arr.dtype}"
        )
    b = np.ascontiguousarray(arr.astype(np.int64, copy=False))
    return b.view(np.float64)


def sf64_scalar(x: float) -> int:
    """Encode one float64 scalar as a signed int for ``constant long&``.

    Args:
        x: The scalar value.

    Returns:
        int: The two's-complement interpretation of the binary64 bit pattern, in
        ``[-2^63, 2^63)``.
    """
    return int(_scalar_float64(x, "sf64_scalar").view(np.int64))


# ---------------------------------------------------------------------------
# torch helpers
# ---------------------------------------------------------------------------
def _torch() -> Any:
    import torch

    return torch


def to_mps_df64(x: ArrayLike, device: str = "mps") -> tuple[torch.Tensor, torch.Tensor]:
    """Encode float64 values and move the components to the GPU.

    Args:
        x: Array-like of float64 values (a CPU ``torch.Tensor`` is accepted too).
        device: Target device (default ``"mps"``).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Contiguous float32 ``(hi, lo)`` tensors.
    """
    torch = _torch()
    if isinstance(x, torch.Tensor):
        if x.is_complex():
            raise TypeError("to_mps_df64: complex tensors are not supported")
        x = x.detach().cpu().numpy()
    hi, lo = encode_df64(x)
    return (
        torch.from_numpy(np.ascontiguousarray(hi)).to(device),
        torch.from_numpy(np.ascontiguousarray(lo)).to(device),
    )


def from_mps_df64(hi: torch.Tensor, lo: torch.Tensor) -> np.ndarray:
    """Copy ``(hi, lo)`` component tensors to the host and decode to float64.

    Args:
        hi: float32 tensor (any device).
        lo: float32 tensor with the same shape.

    Returns:
        np.ndarray: float64 array.

    Raises:
        TypeError: If a component tensor is not float32.
        ValueError: If the shapes differ.
    """
    return decode_df64(hi.detach().cpu().numpy(), lo.detach().cpu().numpy())


def to_mps_sf64(x: ArrayLike, device: str = "mps") -> torch.Tensor:
    """Encode float64 values as binary64 bit patterns and move them to the GPU.

    Args:
        x: Array-like of float64 values (a CPU ``torch.Tensor`` is accepted too).
        device: Target device (default ``"mps"``).

    Returns:
        torch.Tensor: Contiguous int64 tensor.
    """
    torch = _torch()
    if isinstance(x, torch.Tensor):
        if x.is_complex():
            raise TypeError("to_mps_sf64: complex tensors are not supported")
        if not x.is_floating_point():
            raise TypeError(
                f"to_mps_sf64: expected a floating tensor, got {x.dtype} (an "
                "int64 tensor is an sf64 component; use from_mps_sf64 for values)"
            )
        x = x.detach().cpu().numpy()
    return torch.from_numpy(np.ascontiguousarray(encode_sf64(x))).to(device)


def from_mps_sf64(bits: torch.Tensor) -> np.ndarray:
    """Copy an int64 bit-pattern tensor to the host and decode to float64.

    Args:
        bits: int64 tensor (any device).

    Returns:
        np.ndarray: float64 array.
    """
    return decode_sf64(bits.detach().cpu().numpy())


__all__ = [
    "DF64_FLT_MIN",
    "DF64_U2",
    "decode_df64",
    "decode_sf64",
    "df64_scalar",
    "encode_df64",
    "encode_sf64",
    "from_mps_df64",
    "from_mps_sf64",
    "sf64_scalar",
    "to_mps_df64",
    "to_mps_sf64",
]
