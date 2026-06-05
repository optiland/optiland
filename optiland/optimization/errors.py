"""Optimization Errors Module

Kramer Harrison, 2026
"""

from __future__ import annotations


class ConfigurationError(ValueError):
    """Raised when an incompatible optimizer / backend / strategy combination
    is requested before any iteration begins."""


class OptimizationFailure(RuntimeError):
    """Raised when a merit/residual evaluation fails (e.g. a ray-trace error or
    non-finite result) **and** the active failure policy is ``on_failure="raise"``.

    Under the default ``on_failure="reject"`` policy this exception is *not*
    raised; the failing evaluation is reported as a non-finite value instead so
    the stepped controller can reject the trial.  See
    :mod:`optiland.optimization.failure`.
    """
