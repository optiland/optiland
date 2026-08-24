"""Graph-complexity regressions for Newton-Raphson implicit differentiation.

Scientific purpose:
- verify first-order gradients remain available,
- while autograd graph size stays O(1) in the primal iteration count.

The iteration count is controlled by an *instrumented* geometry that runs a
known, explicitly counted number of additional graph-free Newton refinement
passes after the production solve converges. This proves the short and long
configurations really executed different primal iteration counts -- a
negative-tolerance sentinel cannot, because ``_effective_tolerance`` raises
any tolerance to a positive dtype-aware floor.
"""

from __future__ import annotations

import pytest

from optiland.geometries.newton_raphson import (
    _DistanceSolveResult,
    _regularize_signed,
    _sign_preserving_floor,
)
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


def _instrument_geometry(geometry, extra_iterations: int):
    """Wrap the primal solve with counted extra graph-free Newton passes.

    The production solve converges first; each extra pass then re-evaluates
    the residual and takes a (vanishingly small) Newton step from the already
    converged root, so the returned root remains the same valid converged
    solution and the normal production implicit-correction path still runs.
    Every executed pass increments ``geometry.executed_primal_iterations``.
    """
    geometry.extra_primal_iterations = extra_iterations
    geometry.executed_primal_iterations = 0
    original_solve = geometry._solve_distance_primal

    def instrumented_solve(rays, aperture=None):
        result = original_solve(rays, aperture=aperture)
        t = result.t
        f_t = result.residual
        for _ in range(geometry.extra_primal_iterations):
            df_dt, scale, _ = geometry._surface_residual_dt(t, rays)
            safe_df_dt, _ = _regularize_signed(df_dt, scale)
            t = t - f_t / safe_df_dt
            f_t = geometry._surface_residual(t, rays)
            geometry.executed_primal_iterations += 1
        return _DistanceSolveResult(
            t=t,
            residual=f_t,
            converged=result.converged,
            iterations=result.iterations + geometry.executed_primal_iterations,
        )

    geometry._solve_distance_primal = instrumented_solve
    return geometry


def _distance_unrolled_with_graph(geometry, rays, n_iterations: int):
    """Differentiable Newton loop baseline used only in this test module."""
    t = StandardGeometry.distance(geometry, rays)
    for _ in range(n_iterations):
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


def _evaluate_implicit_case(extra_iterations: int):
    geometry = _instrument_geometry(
        build_reference_even_asphere(), extra_iterations
    )
    radius_param = torch.nn.Parameter(torch.tensor(20.0, dtype=torch.float64))
    geometry.radius = radius_param
    rays = build_reference_rays(off_axis=True)

    scalar = geometry.distance(rays).sum()
    node_count = count_autograd_nodes(scalar)
    scalar.backward()

    return {
        "scalar": scalar,
        "node_count": node_count,
        "grad": radius_param.grad,
        "executed": geometry.executed_primal_iterations,
    }


def _evaluate_unrolled_case(n_iterations: int):
    geometry = build_reference_even_asphere()
    radius_param = torch.nn.Parameter(torch.tensor(20.0, dtype=torch.float64))
    geometry.radius = radius_param
    rays = build_reference_rays(off_axis=True)

    scalar = _distance_unrolled_with_graph(geometry, rays, n_iterations).sum()
    node_count = count_autograd_nodes(scalar)
    scalar.backward()
    return {"scalar": scalar, "node_count": node_count, "grad": radius_param.grad}


def test_implicit_graph_size_is_flat_vs_executed_iterations():
    extra_iteration_values = [0, 5, 20, 40]

    node_counts = []
    executed_counts = []
    for extra in extra_iteration_values:
        result = _evaluate_implicit_case(extra)

        assert torch.isfinite(result["scalar"]), f"Non-finite scalar at extra={extra}"
        assert result["grad"] is not None, f"Missing gradient at extra={extra}"
        assert torch.isfinite(result["grad"]), f"Non-finite gradient at extra={extra}"

        node_counts.append(result["node_count"])
        executed_counts.append(result["executed"])

    # The configurations genuinely executed different primal iteration counts.
    assert executed_counts == extra_iteration_values

    # ... while the autograd graph did not grow with them.
    spread = max(node_counts) - min(node_counts)
    assert spread <= 1, (
        "Implicit-path graph size should remain effectively constant vs primal "
        f"iterations; executed={executed_counts}, counts={node_counts}"
    )


def test_instrumented_solve_returns_the_same_converged_root():
    geometry = build_reference_even_asphere()
    rays = build_reference_rays(off_axis=True)
    with torch.no_grad():
        reference = geometry._solve_distance_primal(rays)

    instrumented = _instrument_geometry(build_reference_even_asphere(), 25)
    rays = build_reference_rays(off_axis=True)
    with torch.no_grad():
        result = instrumented._solve_distance_primal(rays)

    assert instrumented.executed_primal_iterations == 25
    assert bool(result.converged.all())
    assert float((result.t - reference.t).abs().max()) < 1e-12


def test_unrolled_baseline_graph_size_grows_vs_iterations():
    iteration_values = [5, 10, 20, 40]

    node_counts = [
        _evaluate_unrolled_case(n)["node_count"] for n in iteration_values
    ]

    assert node_counts[-1] > node_counts[0], (
        "Unrolled baseline should show graph growth with iteration count; "
        f"counts={node_counts}"
    )
    assert node_counts[-1] >= 2 * node_counts[0], (
        "Expected clear graph-size growth for unrolled Newton baseline; "
        f"counts={node_counts}"
    )


def test_implicit_graph_is_much_smaller_than_unrolled_at_high_iter():
    implicit_nodes = _evaluate_implicit_case(40)["node_count"]
    unrolled_nodes = _evaluate_unrolled_case(40)["node_count"]

    assert implicit_nodes < unrolled_nodes, (
        "Implicit graph should be smaller than unrolled baseline; "
        f"implicit={implicit_nodes}, unrolled={unrolled_nodes}"
    )
