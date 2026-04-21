from __future__ import annotations

from contextlib import contextmanager

import optiland.backend as be
from optiland.coordinate_system import CoordinateSystem
from optiland.geometries.even_asphere import EvenAsphere
from optiland.rays import RealRays


def _precision_name(precision_bits: int) -> str:
    return "float64" if precision_bits == 64 else "float32"


@contextmanager
def backend_state(backend: str, precision: str = "float64"):
    """Temporarily switch backend/precision and restore previous state."""
    old_backend = be.get_backend()
    old_precision = be.get_precision()

    be.set_backend(backend)
    if backend == "torch":
        be.set_device("cpu")
    be.set_precision(precision)

    try:
        yield
    finally:
        be.set_backend(old_backend)
        if old_backend == "torch":
            be.set_device("cpu")
        be.set_precision(_precision_name(old_precision))


def build_reference_even_asphere() -> EvenAsphere:
    """Reference NR geometry used by implicit-diff tests."""
    return EvenAsphere(
        CoordinateSystem(),
        radius=20.0,
        conic=-0.35,
        tol=1e-12,
        max_iter=80,
        coefficients=[-2.248851e-4, -4.690412e-6],
    )


def build_reference_rays(
    *,
    off_axis: bool = False,
    x_override=None,
    y_override=None,
    L_override=None,
    M_override=None,
):
    """Build a deterministic ray for direct distance() testing."""
    x = 0.0 if not off_axis else 0.35
    y = 0.0 if not off_axis else -0.27
    L = 0.02 if not off_axis else 0.06
    M = -0.015 if not off_axis else -0.035

    if x_override is not None:
        x = x_override
    if y_override is not None:
        y = y_override
    if L_override is not None:
        L = L_override
    if M_override is not None:
        M = M_override

    N = be.sqrt(1.0 - L * L - M * M)

    return RealRays(
        x=x,
        y=y,
        z=-5.0,
        L=L,
        M=M,
        N=N,
        intensity=1.0,
        wavelength=0.587,
    )


def central_difference_scalar(fun, x0: float, eps: float) -> float:
    """Central finite-difference derivative for scalar-valued functions."""
    return (fun(x0 + eps) - fun(x0 - eps)) / (2.0 * eps)


def finite_diff_reference(fun, x0: float, epsilons: tuple[float, ...]):
    """Compute central-difference derivatives at multiple epsilons."""
    values = [central_difference_scalar(fun, x0, eps) for eps in epsilons]
    return values[-1], values


def assert_fd_is_stable(
    fd_values: list[float],
    *,
    abs_tol: float,
    rel_tol: float,
):
    """Check that finite-difference estimates are not overly epsilon-sensitive."""
    spread = max(fd_values) - min(fd_values)
    scale = max(max(abs(v) for v in fd_values), 1e-12)
    assert spread <= max(abs_tol, rel_tol * scale), (
        f"FD estimates are unstable across epsilons: {fd_values}"
    )


def count_autograd_nodes(tensor) -> int:
    """Count autograd Function nodes reachable from tensor.grad_fn.

    This helper tracks Function objects directly. Counting by ``id`` can
    undercount in Python due to object-id reuse during traversal.
    """
    grad_fn = getattr(tensor, "grad_fn", None)
    if grad_fn is None:
        return 0

    seen = set()
    stack = [grad_fn]

    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)

        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                stack.append(next_fn)

    return len(seen)
