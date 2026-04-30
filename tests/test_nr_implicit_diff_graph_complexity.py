"""Graph-complexity regressions for Newton-Raphson implicit differentiation.

Scientific purpose:
- verify first-order gradients remain available,
- while autograd graph size stays O(1) in ``max_iter`` for the implicit path.
"""

from __future__ import annotations

import pytest

from optiland.geometries.newton_raphson import _sign_preserving_floor
from optiland.geometries.standard import StandardGeometry

from .nr_implicit_test_utils import (
    backend_state,
    build_reference_even_asphere,
    build_reference_rays,
    count_autograd_nodes,
)

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _torch_backend_float64_cpu():
    with backend_state("torch", precision="float64"):
        yield


def _distance_unrolled_with_graph(geometry, rays):
    """Differentiable Newton loop baseline used only in this test module."""
    t = StandardGeometry.distance(geometry, rays)
    for _ in range(geometry.max_iter):
        x_int = rays.x + t * rays.L
        y_int = rays.y + t * rays.M
        z_int = rays.z + t * rays.N

        f_t = geometry.sag(x_int, y_int) - z_int

        nx, ny, nz = geometry._surface_normal(x_int, y_int)
        nz_safe = _sign_preserving_floor(nz)
        fx = -nx / nz_safe
        fy = -ny / nz_safe
        df_dt = fx * rays.L + fy * rays.M - rays.N

        t = t - f_t / _sign_preserving_floor(df_dt)

    return t


def _evaluate_case(max_iter: int, mode: str):
    geometry = build_reference_even_asphere()
    geometry.max_iter = max_iter

    # Force a fixed iteration budget so graph-scaling is actually measurable.
    # If tol allows early exit, max_iter changes may not affect loop work.
    geometry.tol = -1.0

    radius_param = torch.nn.Parameter(torch.tensor(20.0, dtype=torch.float64))
    geometry.radius = radius_param
    rays = build_reference_rays(off_axis=True)

    if mode == "implicit":
        scalar = geometry.distance(rays).sum()
    elif mode == "unrolled":
        scalar = _distance_unrolled_with_graph(geometry, rays).sum()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    node_count = count_autograd_nodes(scalar)
    scalar.backward()

    grad = radius_param.grad
    return {
        "scalar": scalar,
        "node_count": node_count,
        "grad": grad,
    }


def test_implicit_graph_size_is_flat_vs_max_iter():
    iter_values = [5, 10, 20, 40]

    node_counts = []
    for max_iter in iter_values:
        result = _evaluate_case(max_iter=max_iter, mode="implicit")

        assert torch.isfinite(result["scalar"]), f"Non-finite scalar at max_iter={max_iter}"
        assert result["grad"] is not None, f"Missing gradient at max_iter={max_iter}"
        assert torch.isfinite(result["grad"]), f"Non-finite gradient at max_iter={max_iter}"

        node_counts.append(result["node_count"])

    spread = max(node_counts) - min(node_counts)
    assert spread <= 1, (
        "Implicit-path graph size should remain effectively constant vs max_iter; "
        f"counts={node_counts}, spread={spread}"
    )


def test_unrolled_baseline_graph_size_grows_vs_max_iter():
    iter_values = [5, 10, 20, 40]

    node_counts = [
        _evaluate_case(max_iter=max_iter, mode="unrolled")["node_count"]
        for max_iter in iter_values
    ]

    assert node_counts[-1] > node_counts[0], (
        "Unrolled baseline should show graph growth with max_iter; "
        f"counts={node_counts}"
    )
    assert node_counts[-1] >= 2 * node_counts[0], (
        "Expected clear graph-size growth for unrolled Newton baseline; "
        f"counts={node_counts}"
    )


def test_implicit_graph_is_much_smaller_than_unrolled_at_high_iter():
    max_iter = 40

    implicit_nodes = _evaluate_case(max_iter=max_iter, mode="implicit")["node_count"]
    unrolled_nodes = _evaluate_case(max_iter=max_iter, mode="unrolled")["node_count"]

    assert implicit_nodes < unrolled_nodes, (
        "Implicit graph should be smaller than unrolled baseline; "
        f"implicit={implicit_nodes}, unrolled={unrolled_nodes}"
    )
