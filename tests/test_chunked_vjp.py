"""Tests for optiland.ml.chunked_vjp.

The tests compare chunked_vjp with plain autograd, which computes the same
result in one graph, without batches. A bug in the chunking cannot then hide
by being present on both sides of the comparison.
"""

from __future__ import annotations

import pytest

import optiland.backend as be
from optiland.distribution import RandomDistribution
from optiland.materials import Material
from optiland.ml.chunked_vjp import chunked_vjp
from optiland.optic import Optic

from .nr_implicit_test_utils import backend_state

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _torch_backend_float64_cpu():
    with backend_state("torch", precision="float64"):
        yield


# ---------------------------------------------------------------------------
# Helpers for a synthetic reduction
# ---------------------------------------------------------------------------


# The following tests use a reduction with no optical system in it.
# chunked_vjp is not specific to ray tracing. A small reduction also makes a
# failure easy to locate.

N_SAMPLES = 60
OUTPUT_SIZE = 4


def _make_samples():
    """Return a fixed set of samples that stands in for pre-generated rays."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(N_SAMPLES, OUTPUT_SIZE, dtype=torch.float64, generator=generator)


def _make_theta():
    """Return a new tensor for theta, the parameter whose gradient the tests compare.

    Each run needs its own tensor, because .grad accumulates over calls to
    backward().
    """
    return torch.full((OUTPUT_SIZE,), 0.7, dtype=torch.float64, requires_grad=True)


def _reduce(samples, theta):
    """Sum a function of each sample over the samples.

    The sum over samples makes the reduction additive, as chunked_vjp
    requires. tanh makes it nonlinear in theta, so the gradient depends on
    the value of theta.
    """
    return torch.tanh(samples * theta).sum(dim=0)


def _loss(total):
    """Return a loss that is nonlinear in the total.

    The gradient of this loss with respect to the total is 2 * total. It
    depends on the total, so a forward pass that computed the wrong total
    would change the gradient. It also differs between elements, so a
    backward() that ignored grad_output would change it too.
    """
    return (total**2).sum()


def _batches(num_samples, batch_size):
    """Return slices that split num_samples samples into batches.

    The last batch is shorter when batch_size does not divide num_samples.
    """
    return [slice(i, i + batch_size) for i in range(0, num_samples, batch_size)]


def _run_plain_autograd(samples):
    """Return the total and theta.grad, computed with plain autograd.

    Plain autograd builds one graph over all the samples at once.
    """
    theta = _make_theta()
    total = _reduce(samples, theta)
    _loss(total).backward()
    return total.detach(), theta.grad


def _run_chunked_vjp(samples, batch_size):
    """Return the total and theta.grad, computed with chunked_vjp."""
    theta = _make_theta()
    total = chunked_vjp(
        lambda batch: _reduce(samples[batch], theta),
        _batches(len(samples), batch_size),
        params=[theta],
    )
    _loss(total).backward()
    return total.detach(), theta.grad


# ---------------------------------------------------------------------------
# Test: gradient matches non-chunked autograd
# ---------------------------------------------------------------------------


def test_gradient_matches_non_chunked_autograd():
    """The gradient from chunked_vjp matches the gradient from plain autograd."""
    samples = _make_samples()
    _, grad_plain = _run_plain_autograd(samples)
    # With 60 samples in batches of 8, the last batch holds only 4.
    _, grad_chunked = _run_chunked_vjp(samples, batch_size=8)

    # The two differ only in the order of summation, which in float64 changes
    # the result by far less than this tolerance.
    torch.testing.assert_close(grad_chunked, grad_plain, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Test: result is invariant to batch size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 7, 12, N_SAMPLES])
def test_result_is_invariant_to_batch_size(batch_size):
    """The total and the gradient do not depend on how the samples are split.

    A batch size of 1 gives 60 batches of one sample each. A batch size of 7
    gives 9 batches, the last of which holds only 4 samples. A batch size of
    12 gives 5 batches of equal size. A batch size of 60 gives a single batch,
    so nothing is split. Each batch size is compared with plain autograd, so
    a failure shows which batch size gives the wrong result.
    """
    samples = _make_samples()
    total_plain, grad_plain = _run_plain_autograd(samples)
    total_chunked, grad_chunked = _run_chunked_vjp(samples, batch_size)

    # A wrong total points at the forward pass. A correct total with a wrong
    # gradient points at the backward pass.
    torch.testing.assert_close(total_chunked, total_plain, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad_chunked, grad_plain, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Helpers for a real trace
# ---------------------------------------------------------------------------


# The following test traces rays through a singlet whose front surface is an
# even asphere. optiland finds each ray's intersection with an asphere by
# Newton-Raphson iteration and differentiates it implicitly, so the test also
# checks that chunked_vjp works with that implicit differentiation.


def _make_aspheric_singlet():
    """Return the singlet and its front radius, which requires grad."""
    lens = Optic()
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(
        index=1,
        surface_type="even_asphere",
        radius=50.0,
        thickness=10.0,
        material=Material("N-BK7"),
        is_stop=True,
        conic=-1.5,
        coefficients=[0.0, 1e-5],
    )
    lens.surfaces.add(index=2, radius=-50.0, thickness=66.5)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=25.0)
    lens.fields.set_type("angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=0.55, is_primary=True)
    radius = lens.surfaces.surfaces[1].geometry.radius
    radius.requires_grad_(True)
    return lens, radius


def _make_pupil_points(num_rays):
    """Return fixed pupil coordinates for the rays.

    Every batch then traces a subset of the same rays.
    """
    distribution = RandomDistribution(seed=42)
    distribution.generate_points(num_rays)
    return distribution.x, distribution.y


def _spot_sums(lens, px, py):
    """Return the sums of x, y and x**2 + y**2 over the rays at the image plane.

    Each is a sum over rays, so the reduction is additive over rays, as
    chunked_vjp requires.
    """
    rays = lens.trace_generic(Hx=0.0, Hy=0.0, Px=px, Py=py, wavelength=0.55)
    return torch.stack([rays.x.sum(), rays.y.sum(), (rays.x**2 + rays.y**2).sum()])


def _rms_spot_radius(total, num_rays):
    """Return the RMS spot radius about the centroid, computed from the sums.

    The centroid is a mean over all rays, so the RMS radius cannot be computed
    batch by batch. The sums it is computed from are additive, so chunked_vjp
    accumulates the sums, and the RMS radius is computed from the total.
    """
    mean_x, mean_y, mean_r2 = total / num_rays
    return torch.sqrt(mean_r2 - mean_x**2 - mean_y**2)


# ---------------------------------------------------------------------------
# Test: gradient matches non-chunked autograd through a trace
# ---------------------------------------------------------------------------


def test_gradient_matches_non_chunked_autograd_through_a_trace():
    """The gradient through a real optical trace matches plain autograd.

    The test traces 256 rays through an aspheric singlet in 4 batches of 64,
    and uses the RMS spot radius as the loss. It also checks the total, with a
    looser tolerance that the comment before the checks explains.
    """
    num_rays = 256
    px, py = _make_pupil_points(num_rays)

    with be.grad_mode.temporary_enable():
        lens_plain, radius_plain = _make_aspheric_singlet()
        total_plain = _spot_sums(lens_plain, px, py)
        _rms_spot_radius(total_plain, num_rays).backward()

        lens_chunked, radius_chunked = _make_aspheric_singlet()
        total_chunked = chunked_vjp(
            lambda batch: _spot_sums(lens_chunked, px[batch], py[batch]),
            _batches(num_rays, 64),
            params=[radius_chunked],
        )
        # backward() traces every batch again, so be.grad_mode must still be
        # enabled.
        _rms_spot_radius(total_chunked, num_rays).backward()

    # The total has a looser tolerance than the gradient: 1e-6 against 1e-12.
    # The front surface of the singlet is an even asphere, and optiland finds
    # each ray's intersection with it by Newton-Raphson iteration. The
    # iteration stops once the solver has converged to its tolerance. When
    # gradients are recorded, optiland then applies one more Newton iteration
    # to the converged result. This iteration is called the implicit
    # correction, and autograd computes the gradient through it. Inside
    # torch.no_grad(), gradients are not recorded, so optiland skips the
    # implicit correction. chunked_vjp computes the total inside
    # torch.no_grad(), while plain autograd records gradients, so the two
    # totals differ by up to 3.5e-10. A bug that drops or repeats a batch would
    # change the total by that batch's whole contribution. Each batch of 64
    # rays adds more than 1600 to the sum of x**2 + y**2, so a tolerance of
    # 1e-6 accepts the 3.5e-10 difference and rejects such a bug. The backward
    # pass of chunked_vjp records gradients, so it includes the implicit
    # correction, and the gradient matches plain autograd to rounding.
    torch.testing.assert_close(
        total_chunked.detach(), total_plain.detach(), rtol=0.0, atol=1e-6
    )
    torch.testing.assert_close(
        radius_chunked.grad, radius_plain.grad, rtol=1e-12, atol=1e-12
    )
