"""Derivative and storage contracts of compiled CPU tensor intersections."""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.backend._conic import (
    _conic_candidates,
    _select_distance,
)
from optiland.utils import machine_eps

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def cpu_torch():
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision("float64")
    be.grad_mode.disable()
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _inputs(size=3):
    y = torch.linspace(0.1, 0.5, size, dtype=torch.float64)
    return (
        y * 0 + 0.1, y, y * 0 - 1,
        y * 0 + 0.01, y * 0 + 0.02, y * 0 + 0.99975,
        torch.tensor(2.0, dtype=y.dtype), torch.tensor(0.3, dtype=y.dtype),
    )


def _solve(*values, native=False, aperture=None):
    if native:
        roots = _conic_candidates(
            *values, torch.where, torch.sqrt, torch.copysign, machine_eps
        )
        distance = _select_distance(roots, values[:6], aperture.contains if aperture else None, torch.where)
        return torch.where(roots.regular, distance, distance.detach())
    # Test the finite kernel directly. The existing plane/conic wrapper uses
    # scalar control flow and does not support vmap over the radius itself.
    return be.conic_intersection(*values, contains=aperture.contains if aperture else None)


@pytest.mark.parametrize("size", [0, 1, 17])
def test_cpu_values_and_gradients_match_native_tensor_path(size):
    inputs = tuple(value.requires_grad_() for value in _inputs(size))
    actual = _solve(*inputs)
    expected = _solve(*inputs, native=True)
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
    actual_grad = torch.autograd.grad(actual.sum(), inputs)
    expected_grad = torch.autograd.grad(expected.sum(), inputs)
    for got, want in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(got, want, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("with_aperture", [False, True])
def test_forward_mode_and_jacobians_match_native_path(with_aperture):
    from optiland.physical_apertures import RadialAperture

    aperture = RadialAperture(0.3) if with_aperture else None

    def solve(*values, native=False):
        return _solve(*values, native=native, aperture=aperture)

    inputs = _inputs()
    tangents = tuple(torch.full_like(value, 0.03) for value in inputs)
    actual = torch.func.jvp(solve, inputs, tangents)
    expected = torch.func.jvp(
        lambda *values: solve(*values, native=True), inputs, tangents
    )
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=2e-13, atol=2e-13)
    for jacobian in (torch.func.jacfwd, torch.func.jacrev):
        got = jacobian(lambda radius: solve(*inputs[:6], radius, inputs[7]))(
            inputs[6]
        )
        want = jacobian(
            lambda radius: solve(*inputs[:6], radius, inputs[7], native=True)
        )(inputs[6])
        torch.testing.assert_close(got, want, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("materialize", [False, True])
@pytest.mark.parametrize("dual_input", [0, 6], ids=["ray-x", "radius"])
def test_single_input_jvp_matches_analytic_sphere_derivative(
    materialize, dual_input
):
    from optiland.backend.torch_backend.conic import _ConicCPU

    received_tangents = []

    class TangentProbe(_ConicCPU):
        @staticmethod
        def setup_context(ctx, inputs, output):
            _ConicCPU.setup_context(ctx, inputs, output)
            # Torch normally replaces absent tangents with zero tensors. Use
            # its supported option to exercise the callback's None contract
            # without changing the production function's default behavior.
            ctx.set_materialize_grads(materialize)

        @staticmethod
        def jvp(ctx, aperture_tangent, *tangents):
            received_tangents.append(tangents)
            return _ConicCPU.jvp(ctx, aperture_tangent, *tangents)

    inputs = list(_inputs())
    inputs[3] = torch.zeros_like(inputs[0])
    inputs[4] = torch.zeros_like(inputs[0])
    inputs[5] = torch.ones_like(inputs[0])
    inputs[7] = torch.zeros_like(inputs[7])
    x, y, z, _, _, _, radius, _ = inputs
    root = torch.sqrt(radius.square() - x.square() - y.square())
    expected_value = radius - root - z
    expected_tangent = x / root if dual_input == 0 else 1 - radius / root

    with torch.autograd.forward_ad.dual_level():
        inputs[dual_input] = torch.autograd.forward_ad.make_dual(
            inputs[dual_input], torch.ones_like(inputs[dual_input])
        )
        value, tangent = torch.autograd.forward_ad.unpack_dual(
            TangentProbe.apply(None, *inputs)[0]
        )
        torch.testing.assert_close(value, expected_value, rtol=2e-14, atol=2e-14)
        torch.testing.assert_close(
            tangent, expected_tangent, rtol=2e-13, atol=2e-13
        )

    assert len(received_tangents) == 1
    for index, tangent in enumerate(received_tangents[0]):
        if index == dual_input:
            assert torch.equal(tangent, torch.ones_like(inputs[index]))
        elif materialize:
            assert torch.equal(tangent, torch.zeros_like(inputs[index]))
        else:
            assert tangent is None


@pytest.mark.parametrize("batch", [0, 1, 4])
def test_vmap_matches_native_path(batch):
    inputs = _inputs()
    stacked = tuple(value.unsqueeze(0).expand(batch, *value.shape) for value in inputs)
    actual = torch.vmap(_solve)(*stacked)
    if batch == 0:
        assert actual.shape == (0, inputs[0].numel())
        return
    expected = torch.vmap(lambda *values: _solve(*values, native=True))(*stacked)
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("parameter", [6, 7], ids=["radius", "conic"])
def test_empty_parameter_batch_preserves_unbatched_ray_shape(parameter):
    inputs = _inputs()
    original_values = tuple(value.clone() for value in inputs)
    dimensions = tuple(0 if index == parameter else None for index in range(8))

    def batched_values(size):
        return tuple(
            value.expand(size) if index == parameter else value
            for index, value in enumerate(inputs)
        )

    # First verify the batching contract on nonempty inputs against the native
    # tensor calculation, then require an empty result with the same ray axis.
    mapped = torch.vmap(_solve, in_dims=dimensions)
    reference = torch.vmap(
        lambda *values: _solve(*values, native=True), in_dims=dimensions
    )(*batched_values(2))
    torch.testing.assert_close(
        mapped(*batched_values(2)), reference, rtol=2e-14, atol=2e-14
    )
    actual = mapped(*batched_values(0))
    expected = reference[:0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (0, inputs[0].numel())
    for value, original in zip(inputs, original_values, strict=True):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


def test_vmap_of_gradient_matches_native_path():
    inputs = _inputs()
    radii = torch.tensor([1.9, 2.0, 2.1], dtype=torch.float64)

    def loss(radius, native=False):
        return _solve(*inputs[:6], radius, inputs[7], native=native).square().sum()

    actual = torch.vmap(torch.func.grad(loss))(radii)
    expected = torch.vmap(torch.func.grad(lambda r: loss(r, native=True)))(radii)
    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)


def test_cpu_invalid_leaf_lanes_do_not_pollute_parameter_gradients():
    values = list(_inputs(4))
    values[0] = torch.tensor([0.1, 5.0, float("nan"), float("inf")])
    values = tuple(v.to(dtype=torch.float64).requires_grad_() for v in values)
    result = _solve(*values)
    assert torch.isnan(result[1:]).all()
    gradients = torch.autograd.grad(result[0], values)
    for gradient in gradients:
        assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("with_aperture", [False, True])
def test_strided_cpu_inputs_reuse_storage_and_remain_unchanged(
    monkeypatch, with_aperture
):
    import optiland.backend.torch_backend.conic as cpu
    from optiland.physical_apertures import RadialAperture

    aperture = RadialAperture(0.3) if with_aperture else None
    inputs = tuple(v[::2] if v.ndim else v for v in _inputs(12))
    originals = tuple(value.clone() for value in inputs)
    kernel = "_numpy_conic_candidates" if with_aperture else "_numpy_conic_distance"
    original_kernel = getattr(cpu, kernel)
    calls = []

    def check_storage(*values):
        for actual, tensor in zip(values[:6], inputs[:6], strict=True):
            assert np.shares_memory(actual, tensor.numpy())
        calls.append(True)
        return original_kernel(*values)

    monkeypatch.setattr(cpu, kernel, check_storage)
    actual = _solve(*inputs, aperture=aperture)
    expected = _solve(*inputs, native=True, aperture=aperture)
    assert calls
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
    for current, original in zip(inputs, originals, strict=True):
        torch.testing.assert_close(current, original, rtol=0, atol=0)


@pytest.mark.parametrize("virtual", [False, True])
def test_cpu_derivatives_cover_linear_and_virtual_intersections(virtual):
    # A virtual sphere root stays on one branch under parameter perturbations.
    # An axial parabola behind the origin does not: a perturbed direction can
    # introduce a remote forward root and discontinuously change selection.
    values = [
        0.1, 0.2, 5.0 if virtual else -1.0, 0.0, 0.0, 1.0,
        2.0, 0.0 if virtual else -1.0,
    ]
    inputs = tuple(
        torch.tensor([value], dtype=torch.float64, requires_grad=True)
        for value in values
    )
    actual = _solve(*inputs)
    assert (actual < 0).item() == virtual
    assert torch.autograd.gradcheck(_solve, inputs)
    assert torch.autograd.gradgradcheck(_solve, inputs)
