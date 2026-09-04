"""Beer-Lambert extinction coefficient (k) gradient, FD validated.

``k`` is documented as differentiable (the list: "material index, ...,
k, ..."): a ``torch.Tensor`` ``k`` on a custom material should carry a
correct ``d(flux)/dk`` through the Beer-Lambert bulk-absorption term
(D-13, ``alpha = 4*pi*k/wavelength``) all the way to a detector's flux.

Kramer Harrison, 2026
"""

from __future__ import annotations

import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip(
    "torch", reason="Torch not available -- skip gradient tests"
)

# ruff: noqa: E402
import optiland.backend as be
from optiland.nonsequential import (
    VACUUM,
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    NSQMaterial,
    NSQScene,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential.backends.torch_backend import TorchBackend
from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry

_NUMPY = "numpy"
_TORCH = "torch"


class _ConstantMaterial:
    """Bare stand-in exposing only .n()/.k() -- k may be a torch Tensor."""

    def __init__(self, n: float, k) -> None:
        self._n = n
        self._k = k
        self.name = "const"

    def n(self, wavelength_um):
        return self._n

    def k(self, wavelength_um):
        return self._k


def _slab_flux(k_value, thickness: float = 5.0):
    material = NSQMaterial(optiland_material=_ConstantMaterial(1.001, k_value))
    spec = Spectrum.monochromatic(0.55)
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=3.0),
    )
    entry = RefractiveComponent(
        CoordinateSystem(z=10.0),
        PlaneGeometry(),
        material_front=VACUUM,
        material_back=material,
    )
    exit_ = RefractiveComponent(
        CoordinateSystem(z=10.0 + thickness),
        PlaneGeometry(),
        material_front=material,
        material_back=VACUUM,
    )
    scene.add_component("entry", entry)
    scene.add_component("exit", exit_)
    scene.add_detector(
        "D",
        CoordinateSystem(z=10.0 + thickness + 0.5),
        IrradianceDetectorConfig(width=200, height=200, num_pixels_x=4, num_pixels_y=4),
    )
    result = scene.trace(
        num_rays=50_000, seed=1, max_depth=4, backend=TorchBackend(seed=1)
    )
    return result.detectors["D"].total_flux


def test_k_gradient_matches_finite_difference():
    be.set_backend(_TORCH)
    try:
        k0 = 1e-5  # partial-transmission regime for a 5 mm path at 0.55 um
        k = torch.tensor(k0, requires_grad=True)
        flux = _slab_flux(k)
        flux.backward()
        autograd_grad = float(k.grad)

        h = 1e-7
        flux_plus = float(_slab_flux(k0 + h))
        flux_minus = float(_slab_flux(k0 - h))
        fd_grad = (flux_plus - flux_minus) / (2 * h)

        assert autograd_grad < 0.0  # more absorption -> less transmitted flux
        assert autograd_grad == pytest.approx(fd_grad, rel=0.02)
    finally:
        be.set_backend(_NUMPY)
