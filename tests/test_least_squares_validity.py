"""Validity tests for the legacy SciPy ``LeastSquares`` wrapper (opt v2 PR1.1)."""

from __future__ import annotations

import numpy as np

import optiland.backend as be
from optiland.optimization import optimization
from optiland.samples.objectives import CookeTriplet

from .utils import assert_allclose


def _problem(effective_weights=None, monkeypatch=None):
    """Three-operand problem; optionally force per-operand effective weights.

    Field/wavelength weights are only picked up through ``field``/``wavelength``
    index entries, which the ray operands do not accept, so the effective weight
    is pinned directly.
    """
    lens = CookeTriplet()
    problem = optimization.OptimizationProblem()
    for surface in (1, 2, 3):
        problem.add_variable(lens, "radius", surface_number=surface)
    for hy in (0.0, 0.7, 1.0):
        problem.add_operand(
            operand_type="rms_spot_size",
            target=0.0,
            weight=1.0,
            input_data={
                "optic": lens,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": hy,
                "num_rays": 3,
                "wavelength": 0.55,
                "distribution": "hexapolar",
            },
        )
    if effective_weights is not None:
        for op, w in zip(problem.operands, effective_weights, strict=True):
            monkeypatch.setattr(op, "effective_weight", lambda optic=None, w=w: w)
    return problem


def _x(problem):
    return np.array([float(be.to_numpy(v.value)) for v in problem.variables])


def test_residuals_use_effective_weight(set_test_backend, monkeypatch):
    problem = _problem([1.0, 2.0, 3.0], monkeypatch)
    optimizer = optimization.LeastSquares(problem)
    r = optimizer._compute_residuals_vector(_x(problem))
    unweighted = np.array([float(be.to_numpy(op.delta())) for op in problem.operands])
    assert_allclose(r, unweighted * np.array([1.0, 2.0, 3.0]))
    assert_allclose(r, problem.residual_vector())
    assert_allclose(float(np.sum(r**2)), problem.sum_squared())


def test_objective_matches_sum_squared_at_solution(set_test_backend, monkeypatch):
    problem = _problem([1.0, 2.0, 3.0], monkeypatch)
    result = optimization.LeastSquares(problem).optimize(maxiter=5, tol=1e-6)
    assert_allclose(2.0 * float(result.cost), problem.sum_squared(), rtol=1e-6)


def test_partial_nonfinite_rows_are_replaced(set_test_backend, monkeypatch):
    problem = _problem()
    optimizer = optimization.LeastSquares(problem)
    x = _x(problem)
    base = optimizer._compute_residuals_vector(x)
    f_last = float(np.linalg.norm(base))

    def bad():
        r = base.copy()
        r[1] = np.nan
        return be.array(r)

    monkeypatch.setattr(problem, "residual_vector", bad)
    r = optimizer._compute_residuals_vector(x)
    assert np.all(np.isfinite(r))
    assert_allclose(r[0], base[0])
    assert_allclose(r[2], base[2])
    assert_allclose(r[1], 10 * max(f_last, 1.0) / np.sqrt(base.size))


def test_exception_returns_full_penalty_vector(set_test_backend, monkeypatch):
    problem = _problem()
    optimizer = optimization.LeastSquares(problem)
    x = _x(problem)
    optimizer._compute_residuals_vector(x)

    def boom():
        raise RuntimeError("ray failure")

    monkeypatch.setattr(problem, "residual_vector", boom)
    r = optimizer._compute_residuals_vector(x)
    assert r.shape == (len(problem.operands),)
    assert np.all(np.isfinite(r)) and np.all(r > 0)


def test_forced_ray_failure_mid_run_completes(set_test_backend, monkeypatch):
    problem = _problem()
    real = problem.residual_vector
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] in (6, 7, 8):
            raise RuntimeError("ray failure")
        if calls["n"] == 9:
            r = be.to_numpy(real()).copy()
            r[0] = np.nan
            return be.array(r)
        return real()

    monkeypatch.setattr(problem, "residual_vector", flaky)
    result = optimization.LeastSquares(problem).optimize(maxiter=15, tol=1e-6)
    assert np.all(np.isfinite(result.x))
    monkeypatch.setattr(problem, "residual_vector", real)
    assert np.isfinite(float(be.to_numpy(problem.sum_squared())))


def test_final_guard_restores_best_finite_x(set_test_backend, monkeypatch):
    problem = _problem()
    real = problem.residual_vector
    nfev = {"n": 0}

    def late_failure():
        # Everything after the x0 seed evaluation is non-finite
        nfev["n"] += 1
        if nfev["n"] == 1:
            return real()
        return be.array([np.nan] * len(problem.operands))

    x0 = _x(problem)
    monkeypatch.setattr(problem, "residual_vector", late_failure)
    optimization.LeastSquares(problem).optimize(maxiter=5, tol=1e-6)
    assert_allclose(_x(problem), x0)


def test_double_gauss_does_not_raise():
    """Baseline from the opt v2 spec: SciPy used to see NaN in its FD Jacobian."""
    from optiland.samples.objectives import DoubleGauss

    lens = DoubleGauss()
    problem = optimization.OptimizationProblem()
    for s in range(1, lens.surfaces.num_surfaces - 1):
        if np.isfinite(float(be.to_numpy(lens.surfaces.radii)[s])):
            problem.add_variable(lens, "radius", surface_number=s)
        problem.add_variable(lens, "thickness", surface_number=s)
    problem.add_operand(
        operand_type="f2", target=50.0, weight=1.0, input_data={"optic": lens}
    )
    for hy in (0.0, 0.7, 1.0):
        for wl in lens.wavelengths.get_wavelengths():
            problem.add_operand(
                operand_type="rms_spot_size",
                target=0.0,
                weight=1.0,
                input_data={
                    "optic": lens,
                    "surface_number": -1,
                    "Hx": 0.0,
                    "Hy": hy,
                    "num_rays": 5,
                    "wavelength": wl,
                    "distribution": "hexapolar",
                },
            )
    result = optimization.LeastSquares(problem).optimize(
        maxiter=3, method_choice="trf", tol=1e-6
    )
    assert np.all(np.isfinite(result.x))
    assert np.isfinite(float(be.to_numpy(problem.sum_squared())))
