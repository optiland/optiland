"""scatter_fraction gradient (D-5), finite-difference validated.

D-5 named ``scatter_fraction`` as carrying zero gradient in the pre-revamp
engine: the reflect/specular-vs-scatter branch was drawn from a detached
probability with no compensating attached weight, so
``d(flux)/d(scatter_fraction)`` was silently zero even though the branch
draw used ``self.scatter_fraction`` directly.

Writing this validation test is what caught that the PR9 commit that
claimed to close D-5 never actually did: ``BaseComponent.__init__`` still
forced ``scatter_fraction`` through a bare ``float()`` (detaching any
tensor immediately), and neither ``RefractiveComponent`` nor
``ReflectiveComponent`` applied a compensating weight to either branch.
Both are fixed alongside this test (components/base.py, refractive.py,
reflective.py) using the same detached-sample / attached-weight estimator
already used for the Fresnel branch.

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
    ReflectiveComponent,
    RefractiveComponent,
    Spectrum,
)
from optiland.nonsequential.backends.torch_backend import TorchBackend
from optiland.nonsequential.bsdf.lambertian import LambertianBSDF
from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry

_NUMPY = "numpy"
_TORCH = "torch"


def _reset_backend() -> None:
    be.set_backend(_NUMPY)


def _refractive_scene(scatter_fraction) -> NSQScene:
    glass = NSQMaterial.from_glass("N-BK7")
    bsdf = LambertianBSDF(reflectance_value=0.0, transmissive_fraction=0.9)
    spec = Spectrum.monochromatic(0.55)
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=3.0),
    )
    comp = RefractiveComponent(
        CoordinateSystem(z=10.0),
        PlaneGeometry(),
        material_front=VACUUM,
        material_back=glass,
        bsdf=bsdf,
        scatter_fraction=scatter_fraction,
    )
    scene.add_component("IF", comp)
    scene.add_detector(
        "D",
        CoordinateSystem(z=200.0),
        IrradianceDetectorConfig(width=400, height=400, num_pixels_x=8, num_pixels_y=8),
    )
    return scene


def _reflective_scene(scatter_fraction) -> NSQScene:
    bsdf = LambertianBSDF(reflectance_value=1.0, transmissive_fraction=0.0)
    spec = Spectrum.monochromatic(0.55)
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=3.0),
    )
    comp = ReflectiveComponent(
        CoordinateSystem(z=10.0),
        PlaneGeometry(),
        reflectance=1.0,
        bsdf=bsdf,
        scatter_fraction=scatter_fraction,
    )
    scene.add_component("M", comp)
    scene.add_detector(
        "D",
        CoordinateSystem(z=0.0, rx=3.14159265),
        IrradianceDetectorConfig(width=400, height=400, num_pixels_x=8, num_pixels_y=8),
    )
    return scene


@pytest.mark.parametrize("build_scene", [_refractive_scene, _reflective_scene])
def test_scatter_fraction_gradient_matches_finite_difference(build_scene):
    be.set_backend(_TORCH)
    try:
        sf_value = 0.3
        sf = torch.tensor(sf_value, requires_grad=True)
        scene = build_scene(sf)
        result = scene.trace(num_rays=20_000, seed=1, backend=TorchBackend(seed=1))
        loss = result.detectors["D"].total_flux
        loss.backward()
        autograd_grad = float(sf.grad)

        h = 1e-3
        flux_plus = float(
            build_scene(sf_value + h)
            .trace(num_rays=20_000, seed=1, backend=TorchBackend(seed=1))
            .detectors["D"]
            .total_flux
        )
        flux_minus = float(
            build_scene(sf_value - h)
            .trace(num_rays=20_000, seed=1, backend=TorchBackend(seed=1))
            .detectors["D"]
            .total_flux
        )
        fd_grad = (flux_plus - flux_minus) / (2 * h)

        assert autograd_grad != 0.0
        assert autograd_grad == pytest.approx(fd_grad, rel=0.15, abs=0.05)
    finally:
        _reset_backend()
