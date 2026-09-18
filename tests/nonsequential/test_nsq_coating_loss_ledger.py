"""Mirror, coating and BSDF lobe loss have a destination in the ledger.

A mirror below unit reflectance multiplies a ray's flux by ``R``; a coating
with ``R + T < 1`` multiplies it by a weight whose expectation is ``R + T``; a
BSDF lobe returns a weight that is a fraction of the incident flux. Before
these tests none of the three had a bin in the conservation identity, so
``flux_conservation_error`` equalled the loss exactly on a scene the engine
traced correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.coatings import SimpleCoating
from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    VACUUM,
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    MirrorConfig,
    NSQMaterial,
    NSQScene,
    RefractiveComponent,
    Spectrum,
    SurfaceConfig,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.backends.torch_backend import TorchBackend
from optiland.nonsequential.bsdf.lambertian import LambertianBSDF
from optiland.nonsequential.components.geometry.analytic.plane import PlaneGeometry

GREEN = 0.55


def _backend_for_current_array_library(seed: int):
    return (
        NumpyBackend(seed=seed)
        if be.get_backend() == "numpy"
        else TorchBackend(seed=seed)
    )


def _mirror_scene(reflectance: object) -> NSQScene:
    """1 W collimated beam onto a flat mirror, detector behind the source."""
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(z=-10.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(GREEN),
            total_flux=1.0,
            aperture_radius=1.0,
        ),
    )
    scene.add_mirror(
        "M",
        CoordinateSystem(z=0.0),
        MirrorConfig(radius=np.inf, reflectance=reflectance, aperture_radius=10.0),
    )
    scene.add_detector(
        "R",
        CoordinateSystem(z=-20.0),
        IrradianceDetectorConfig(
            width=10, height=10, num_pixels_x=16, num_pixels_y=16, splat="hard"
        ),
    )
    return scene


@pytest.mark.parametrize("reflectance", [0.5, 0.8, 1.0])
def test_mirror_loss_is_booked_and_the_identity_closes(
    set_test_backend, reflectance
):
    """``1 - R`` is a destination, so the ledger closes for any ``R``."""
    scene = _mirror_scene(reflectance)
    result = scene.trace(
        num_rays=20_000, seed=3, backend=_backend_for_current_array_library(3)
    )

    assert result.total_flux_detected == pytest.approx(reflectance, abs=1e-9)
    assert result.total_flux_coating == pytest.approx(1.0 - reflectance, abs=1e-9)
    assert result.flux_conservation_error == pytest.approx(0.0, abs=1e-12)


def test_perfect_mirror_books_nothing(set_test_backend):
    """``R = 1`` removes no flux, so the new bin is exactly zero."""
    scene = _mirror_scene(1.0)
    result = scene.trace(
        num_rays=5_000, seed=3, backend=_backend_for_current_array_library(3)
    )
    assert result.total_flux_coating == 0.0


def test_coating_object_on_a_mirror_books_its_loss(set_test_backend):
    """A ``SimpleCoating`` reflectance is booked like a constant one."""
    scene = _mirror_scene(SimpleCoating(transmittance=0.0, reflectance=0.8))
    result = scene.trace(
        num_rays=20_000, seed=3, backend=_backend_for_current_array_library(3)
    )
    assert result.total_flux_coating == pytest.approx(0.2, abs=1e-9)
    assert result.flux_conservation_error == pytest.approx(0.0, abs=1e-12)


def test_loss_is_available_per_surface(set_test_backend):
    """The bin is booked on the surface that removed the flux."""
    scene = _mirror_scene(0.5)
    result = scene.trace(
        num_rays=20_000, seed=3, backend=_backend_for_current_array_library(3)
    )
    per_surface = [
        comp.coating_loss
        for comp in scene.surfaces
        if hasattr(comp, "coating_loss")
    ]
    assert sum(per_surface) == pytest.approx(result.total_flux_coating, abs=1e-12)
    assert max(per_surface) == pytest.approx(0.5, abs=1e-9)


def test_diagnostics_report_the_coating_fraction(set_test_backend):
    """The report gains one line and keeps every warning it had."""
    scene = _mirror_scene(0.5)
    result = scene.trace(
        num_rays=20_000, seed=3, backend=_backend_for_current_array_library(3)
    )
    assert result.diagnostics.coating_loss_flux_fraction == pytest.approx(
        0.5, abs=1e-9
    )
    assert "coating_loss_flux_fraction" in result.diagnostics.report()


def test_lossless_lens_books_exactly_zero(set_test_backend):
    """A bare Fresnel interface has ``T = 1 - R``, so it books nothing.

    This is the bit-identity check: on a scene with no mirror and no
    coating the new term is exactly ``0.0`` and the identity is the
    arithmetic it was before.
    """
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(GREEN),
            total_flux=1.0,
            aperture_radius=1.0,
        ),
    )
    scene.add_lens(
        "L",
        CoordinateSystem(z=10.0),
        LensConfig(
            r1=50.0,
            r2=-50.0,
            thickness=4.0,
            material="N-BK7",
            front_aperture_radius=5.0,
        ),
    )
    scene.add_detector(
        "D",
        CoordinateSystem(z=60.0),
        IrradianceDetectorConfig(
            width=20, height=20, num_pixels_x=32, num_pixels_y=32, splat="hard"
        ),
    )
    result = scene.trace(
        num_rays=20_000, seed=7, backend=_backend_for_current_array_library(7)
    )
    assert result.total_flux_coating == 0.0


def _coated_interface_scene(coating) -> NSQScene:
    """Collimated beam at normal incidence on a single coated interface."""
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(z=-5.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(GREEN),
            total_flux=1.0,
            aperture_radius=1.0,
        ),
    )
    scene.add_component(
        "I",
        RefractiveComponent(
            cs=CoordinateSystem(z=0.0),
            geometry=PlaneGeometry(),
            material_front=VACUUM,
            material_back=NSQMaterial.from_glass("N-BK7"),
            coating=coating,
            name="I",
        ),
    )
    for name, z in (("R", -15.0), ("T", 15.0)):
        scene.add_detector(
            name,
            CoordinateSystem(z=z),
            IrradianceDetectorConfig(
                width=10, height=10, num_pixels_x=8, num_pixels_y=8
            ),
        )
    return scene


def test_lossy_coating_absorptance_is_booked(set_test_backend):
    """``1 - R - T`` on a coated interface is a destination.

    The branch itself is a stochastic draw, so the identity closes in
    expectation rather than exactly; what is left after the booking is the
    estimator's sampling residual, which shrinks with ray count.
    """
    scene = _coated_interface_scene(
        SimpleCoating(transmittance=0.85, reflectance=0.10)
    )
    coarse = scene.trace(
        num_rays=20_000, seed=7, backend=_backend_for_current_array_library(7)
    )
    fine = scene.trace(
        num_rays=200_000, seed=7, backend=_backend_for_current_array_library(7)
    )

    assert coarse.total_flux_coating == pytest.approx(0.05, abs=1e-9)
    assert fine.total_flux_coating == pytest.approx(0.05, abs=1e-9)
    # 0.0501 before the booking, at either ray count.
    assert coarse.flux_conservation_error < 1e-3
    assert fine.flux_conservation_error < coarse.flux_conservation_error


def test_uncoated_interface_books_exactly_zero(set_test_backend):
    """A bare Fresnel interface has ``T = 1 - R`` and removes nothing."""
    scene = _coated_interface_scene(coating=None)
    result = scene.trace(
        num_rays=20_000, seed=7, backend=_backend_for_current_array_library(7)
    )
    assert result.total_flux_coating == 0.0


def test_bsdf_lobe_weight_below_one_is_booked(set_test_backend):
    """What a lobe does not return was removed at the surface."""
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(z=-10.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(GREEN),
            total_flux=1.0,
            aperture_radius=1.0,
        ),
    )
    scene.add_mirror(
        "M",
        CoordinateSystem(z=0.0),
        MirrorConfig(
            radius=np.inf,
            reflectance=1.0,
            aperture_radius=10.0,
            surface=SurfaceConfig(bsdf=LambertianBSDF(reflectance_value=0.35)),
        ),
    )
    scene.add_detector(
        "D",
        CoordinateSystem(z=-20.0),
        IrradianceDetectorConfig(
            width=200, height=200, num_pixels_x=8, num_pixels_y=8, splat="hard"
        ),
    )
    result = scene.trace(
        num_rays=20_000, seed=11, backend=_backend_for_current_array_library(11)
    )
    # A perfect mirror carrying a lobe of reflectance 0.35: the whole loss is
    # the lobe's. The scatter branch's probability is clamped to 1 - 1e-6, so
    # its compensating weight leaves a relative excess of that order.
    assert result.total_flux_coating == pytest.approx(0.65, rel=1e-5)
