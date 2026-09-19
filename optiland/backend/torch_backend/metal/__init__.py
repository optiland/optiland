"""Metal (Apple GPU) float64 emulation for the torch backend.

Apple GPUs have no native binary64. This package supplies:

- ``compile``: the single place Metal shader source is compiled through
  ``torch.mps.compile_shader``, enforcing safe math (no fast-math) so that the
  error-free transformations the arithmetic relies on are preserved.
- ``kernels/``: Metal Shading Language sources for double-single (``df64``,
  hi/lo float32 pair) and software binary64 (``sf64``, metal-softfloat) arithmetic.
- ``tensor``: the ``MetalFloat64`` tensor subclass that reports
  ``dtype=torch.float64`` on ``device=mps`` and dispatches aten ops to those kernels.
- ``factories``: host-side constructors (``tensor``, ``zeros``, ``arange``, ...)
  returning ``MetalFloat64`` tensors in the process-wide *mode* set here.

Process-wide settings live in this module and are deliberately light to import
(no torch import at package load): :func:`set_mode` / :func:`get_mode` choose
the representation new tensors use (``'df64'`` by default, ``'sf64'`` for the
exact software-float mode; the ``OPTILAND_METAL_MODE`` environment variable
sets the initial value), :func:`enable` / :func:`disable` switch the backend
routing on and off, :func:`is_available` reports whether the GPU and the kernel
library are usable, :func:`set_host_threshold` / :func:`get_host_threshold`
control dual residency (tensors of at most that many elements live on the host
and their ops run exactly on the CPU; default 256, ``0`` keeps everything on
the GPU; ``OPTILAND_METAL_HOST_THRESHOLD`` sets the initial value), and
:func:`stats` / :func:`reset_stats` re-export the GPU / host / CPU-fallback
counters of :mod:`.tensor`.

See NOTES/02-design.md in the project root for the architecture.
"""

from __future__ import annotations

import os
from typing import Any

_MODES: tuple[str, ...] = ("df64", "sf64")


def _initial_mode() -> str:
    """Representation at import: ``OPTILAND_METAL_MODE`` or ``'df64'``."""
    mode = os.environ.get("OPTILAND_METAL_MODE", "df64")
    if mode not in _MODES:
        raise ValueError(f"OPTILAND_METAL_MODE must be one of {_MODES}, got {mode!r}")
    return mode


_mode: str = _initial_mode()
_enabled: bool = True
_available: bool | None = None
_unavailable_reason: str | None = None


def set_mode(mode: str) -> None:
    """Choose the representation used by every tensor created from now on.

    Args:
        mode: ``'df64'`` (double-single, default) or ``'sf64'`` (software
            binary64, exact but slower).

    Raises:
        ValueError: If ``mode`` is not one of the known representations.
    """
    global _mode, _available, _unavailable_reason
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if mode != _mode:
        # The availability probe compiles the library of the current mode.
        _available = None
        _unavailable_reason = None
    _mode = mode


def get_mode() -> str:
    """Return the representation new tensors use (``'df64'`` or ``'sf64'``)."""
    return _mode


def enable() -> None:
    """Route float64-on-mps creation through the emulation (the default)."""
    global _enabled
    _enabled = True
    from optiland.backend.torch_backend.metal import tensor as _tensor

    _tensor._AUTO_FACTORY_INTERCEPT = True


def disable() -> None:
    """Stop routing float64-on-mps creation through the emulation.

    With the emulation disabled the torch backend asks torch for real float64
    tensors on ``mps``, which torch rejects with ``TypeError``; this switch
    exists for audits and tests, not for production use. The factory-intercept
    dispatch mode armed by the first ``MetalFloat64`` is removed as well.
    """
    global _enabled
    _enabled = False
    from optiland.backend.torch_backend.metal import tensor as _tensor

    _tensor.disable_factory_intercept()
    _tensor._AUTO_FACTORY_INTERCEPT = False


def is_enabled() -> bool:
    """Return whether float64-on-mps creation is routed through the emulation."""
    return _enabled


def is_available() -> bool:
    """Return whether the emulation can run (MPS present, library self-test passes).

    The first call compiles the kernel library for the current mode (a few
    hundred milliseconds) and runs its self-test; the verdict is cached until
    :func:`set_mode` changes the representation. A negative verdict's cause is
    available from :func:`unavailable_reason`.
    """
    global _available, _unavailable_reason
    if _available is None:
        try:
            import torch

            if not torch.backends.mps.is_available():
                raise RuntimeError("torch MPS (Metal GPU) is not available")
            from optiland.backend.torch_backend.metal.library import get_library

            get_library(_mode)  # compiles and self-tests; raises on any problem
        except Exception as exc:  # noqa: BLE001 - the reason is reported, not hidden
            _available = False
            _unavailable_reason = f"{type(exc).__name__}: {exc}"
        else:
            _available = True
            _unavailable_reason = None
    return _available


def unavailable_reason() -> str | None:
    """Return why :func:`is_available` is False (None when available or unprobed)."""
    return _unavailable_reason


def stats() -> dict[str, int]:
    """Return the per-op counters (``gpu:<op>``, ``cpu_fallback:<op>``)."""
    from optiland.backend.torch_backend.metal import tensor as _tensor

    return _tensor.stats()


def reset_stats() -> None:
    """Zero the per-op counters."""
    from optiland.backend.torch_backend.metal import tensor as _tensor

    _tensor.reset_stats()


def set_host_threshold(n: int) -> None:
    """Set the element count up to which new tensors are host-resident.

    Args:
        n: Threshold (``0`` disables the host path: every tensor GPU-resident).
            The factories and the dispatch layer read it live.
    """
    from optiland.backend.torch_backend.metal import tensor as _tensor

    _tensor.set_host_threshold(n)


def get_host_threshold() -> int:
    """Return the current dual-residency threshold (see :func:`set_host_threshold`)."""
    from optiland.backend.torch_backend.metal import tensor as _tensor

    return _tensor.get_host_threshold()


def machine_eps(mode: str | None = None) -> float:
    """Relative precision carried by ``mode`` (default: the current mode)."""
    return {"df64": 2.0**-48, "sf64": 2.0**-53}[mode or _mode]


def __getattr__(name: str) -> Any:
    """Lazily expose ``MetalFloat64`` and ``is_metal`` (no eager torch import)."""
    if name in ("MetalFloat64", "is_metal"):
        from optiland.backend.torch_backend.metal import tensor as _tensor

        return getattr(_tensor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "disable",
    "enable",
    "get_host_threshold",
    "get_mode",
    "is_available",
    "is_enabled",
    "machine_eps",
    "reset_stats",
    "set_host_threshold",
    "set_mode",
    "stats",
    "unavailable_reason",
]
