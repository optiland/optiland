"""Compile Metal shader sources for the emulated-float64 kernels.

PyTorch reads ``PYTORCH_MPS_FAST_MATH`` once, at the first shader compilation
in a process, and caches the choice. Any value other than ``"0"`` selects
Metal's relaxed/fast math mode, which may reassociate or contract floating
point expressions and silently destroys the TwoSum/TwoProd error-free
transformations that double-single arithmetic depends on. This module sets
the variable to ``"0"`` when it is unset and refuses to run if it was set to
anything else, so a process can never mix fast-math kernels with ours.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

_FAST_MATH_VAR = "PYTORCH_MPS_FAST_MATH"
_KERNEL_DIR = Path(__file__).with_name("kernels")


class MetalMathModeError(RuntimeError):
    """Raised when PyTorch would compile Metal shaders with fast math enabled."""


def enforce_safe_math() -> None:
    """Pin PyTorch's Metal compile mode to safe/precise before the first compile.

    Raises:
        MetalMathModeError: If ``PYTORCH_MPS_FAST_MATH`` is set to a value
            other than ``"0"``.
    """
    value = os.environ.get(_FAST_MATH_VAR)
    if value is None:
        os.environ[_FAST_MATH_VAR] = "0"
    elif value != "0":
        raise MetalMathModeError(
            f"{_FAST_MATH_VAR}={value!r}: the Metal float64 emulation requires "
            f"{_FAST_MATH_VAR}=0 before the first shader compilation in this process."
        )


def kernel_source(*names: str) -> str:
    """Concatenate kernel source files from ``kernels/`` in the given order.

    Args:
        *names: File names inside the ``kernels`` directory.

    Returns:
        str: The amalgamated Metal source.
    """
    parts = [
        f"// ---- {name} ----\n" + (_KERNEL_DIR / name).read_text() for name in names
    ]
    return "\n".join(parts)


def source_sha256(source: str) -> str:
    """Return the SHA-256 of a shader source string (for provenance records)."""
    return hashlib.sha256(source.encode()).hexdigest()


_LIBRARIES: dict[str, Any] = {}


def compile_library(source: str) -> Any:
    """Compile ``source`` with ``torch.mps.compile_shader``, cached by content hash.

    Args:
        source: Complete Metal source text.

    Returns:
        The compiled library object whose attributes are the kernels.
    """
    enforce_safe_math()
    key = source_sha256(source)
    lib = _LIBRARIES.get(key)
    if lib is None:
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "Metal GPU (torch MPS) is not available in this process."
            )
        lib = torch.mps.compile_shader(source)
        _LIBRARIES[key] = lib
    return lib
