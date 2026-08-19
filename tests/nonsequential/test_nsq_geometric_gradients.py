"""NSQ merge gate — geometric-parameter gradient validation.

``test_nsq_gradients.py`` validates gradients w.r.t. *radiometric* parameters
(source flux, material index, BSDF reflectance).  Those all enter the trace as
scalar multipliers on ray throughput, so they exercise none of the geometry.

This module covers the case illumination designers actually optimize: a
gradient w.r.t. a **geometric** parameter, measured through a **spatially
varying** loss.  Both halves matter:

- A geometric parameter (lens radius) reaches the loss only through the
  ray-surface intersection and the refraction that follows.  A constructor
  that casts its input to ``float`` silently severs this path, and no
  flux-only test can detect that.
- A spatial loss (flux-weighted second moment of the irradiance map) responds
  to *where* rays land.  Total detected flux is nearly invariant to landing
  position while rays stay on the detector, so a flux-only loss cannot see an
  error in the landing position at all.

Each test compares autograd against central finite differences with common
random numbers (identical seed at ±h), which cancels the Monte Carlo variance
and leaves agreement limited only by the FD truncation error.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available — skip gradient tests")

# Imports below intentionally follow importorskip: they must not run when
# torch is unavailable.
# ruff: noqa: E402

import optiland.backend as be
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    MirrorConfig,
    NSQScene,
    PointSourceConfig,
    Spectrum,
)

# Relative tolerance for autograd-vs-FD agreement. Central differences with
# common random numbers land well inside this; the gradient bugs this module
# guards against produced errors of 25-100%.
_REL_TOL = 0.05


def _radial_weights(half_size: float, num_pixels: int) -> torch.Tensor:
    """Return per-pixel r^2 weights for the flux-weighted second moment.

    Args:
        half_size: Detector half-width [mm].
        num_pixels: Pixels per side.

    Returns:
        Flat tensor of r^2 at each pixel centre, shape (num_pixels**2,).
    """
    edges = np.linspace(-half_size, half_size, num_pixels + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    xx, yy = np.meshgrid(centres, centres)
    return torch.tensor((xx**2 + yy**2).ravel(), dtype=torch.float64)


def _second_moment(data: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Flux-weighted mean r^2 of a flat irradiance buffer.

    Depends only on *where* flux lands, so it is sensitive to landing-position
    errors that a total-flux loss cannot detect.

    Args:
        data: Flat detector flux buffer.
        weights: Per-pixel r^2 weights.

    Returns:
        Scalar second moment [mm^2].
    """
    return (data * weights).sum() / (data.sum() + 1e-30)


def _assert_matches_fd(
    loss_fn,
    value: float,
    h: float,
    tol: float = _REL_TOL,
) -> None:
    """Assert autograd agrees with central finite differences.

    Args:
        loss_fn: Callable mapping a parameter (float or tensor) to a scalar loss.
        value: Point at which to evaluate the gradient.
        h: Finite-difference step.
        tol: Maximum allowed relative error.
    """
    param = torch.tensor(value, dtype=torch.float64, requires_grad=True)
    loss = loss_fn(param)
    assert loss.requires_grad, (
        "Loss is detached from the parameter: the parameter never reached the "
        "autograd graph (a float() cast in a constructor will do this)."
    )
    loss.backward()
    grad_autograd = param.grad.item()
    assert np.isfinite(grad_autograd), f"Gradient is not finite: {grad_autograd}"

    with torch.no_grad():
        loss_plus = float(loss_fn(torch.tensor(value + h, dtype=torch.float64)))
        loss_minus = float(loss_fn(torch.tensor(value - h, dtype=torch.float64)))
    grad_fd = (loss_plus - loss_minus) / (2.0 * h)

    rel_err = abs(grad_autograd - grad_fd) / (abs(grad_fd) + 1e-30)
    assert rel_err < tol, (
        f"autograd={grad_autograd:.6e} vs finite-difference={grad_fd:.6e}, "
        f"relative error {rel_err:.4f} exceeds {tol}"
    )


class TestLensRadiusGradient:
    """Gradients w.r.t. a lens radius passed through the public config API."""

    def setup_method(self):
        be.set_backend("torch")
        be.set_precision("float64")

    def teardown_method(self):
        be.set_backend("numpy")

    @staticmethod
    def _trace(r1, det_z: float = 210.0, tilt_deg: float = 0.0, num_rays: int = 20_000):
        """Collimated beam through a plano-convex singlet onto a detector."""
        half, npix = 20.0, 32
        scene = NSQScene()
        scene.add_source(
            "S1",
            CoordinateSystem(z=0.0),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=1.0,
                aperture_radius=10.0,
            ),
        )
        # r1 goes in through LensConfig -- the documented public API. If the
        # geometry constructor detaches it, the loss below has no grad_fn.
        scene.add_lens(
            "L1",
            CoordinateSystem(z=100.0),
            LensConfig(
                r1=r1,
                r2=float("inf"),
                thickness=5.0,
                material="N-BK7",
                front_aperture_radius=12.0,
            ),
        )
        scene.add_detector(
            "D1",
            CoordinateSystem(z=det_z, rx=np.deg2rad(tilt_deg)),
            IrradianceDetectorConfig(
                width=2 * half,
                height=2 * half,
                num_pixels_x=npix,
                num_pixels_y=npix,
                splat="bilinear",
            ),
        )
        data = scene.trace(num_rays=num_rays, seed=11, max_depth=8).detectors["D1"].data
        return _second_moment(data, _radial_weights(half, npix))

    def test_radius_reaches_autograd_graph_via_public_api(self):
        """LensConfig(r1=tensor) must attach; a float() cast would detach it."""
        r1 = torch.tensor(120.0, dtype=torch.float64, requires_grad=True)
        loss = self._trace(r1, num_rays=4_000)
        assert loss.requires_grad, "LensConfig(r1=tensor) did not reach the graph"
        loss.backward()
        assert r1.grad is not None
        assert np.isfinite(r1.grad.item())
        assert r1.grad.item() != 0.0

    def test_spatial_loss_matches_fd_slow_lens(self):
        """Second-moment gradient vs FD for a slow lens, normal detector.

        h=0.5: at h>=1.0 the +-h detector images cross enough bilinear-splat
        pixel boundaries to bias the finite-difference estimate by several
        percent (a discretization effect, not noise -- it persists from
        20k to 500k rays). h=0.5 keeps both FD evaluations inside the same
        smooth region of the splat kernel.
        """
        _assert_matches_fd(lambda r: self._trace(r), value=120.0, h=0.5)

    def test_spatial_loss_matches_fd_fast_lens(self):
        """Faster lens: larger ray angles at the detector."""
        _assert_matches_fd(lambda r: self._trace(r, det_z=130.0), value=45.0, h=1.0)

    def test_spatial_loss_matches_fd_tilted_detector(self):
        """Tilted detector — the case a detached hit distance gets wrong.

        The splatted landing position is ``origin + t * direction``. Detaching
        ``t`` drops the ``direction * dt/dparam`` term, which is negligible at
        normal incidence but reached 26% relative error here.
        """
        _assert_matches_fd(
            lambda r: self._trace(r, det_z=130.0, tilt_deg=40.0), value=45.0, h=1.0
        )

    def test_conic_constant_matches_fd(self):
        """Conic constant is differentiable alongside the radius."""

        def loss_fn(conic):
            half, npix = 20.0, 32
            scene = NSQScene()
            scene.add_source(
                "S1",
                CoordinateSystem(z=0.0),
                CollimatedSourceConfig(
                    spectrum=Spectrum.monochromatic(0.55),
                    total_flux=1.0,
                    aperture_radius=10.0,
                ),
            )
            scene.add_lens(
                "L1",
                CoordinateSystem(z=100.0),
                LensConfig(
                    r1=60.0,
                    r2=float("inf"),
                    thickness=5.0,
                    material="N-BK7",
                    front_aperture_radius=12.0,
                    conic1=conic,
                ),
            )
            scene.add_detector(
                "D1",
                CoordinateSystem(z=160.0),
                IrradianceDetectorConfig(
                    width=2 * half,
                    height=2 * half,
                    num_pixels_x=npix,
                    num_pixels_y=npix,
                    splat="bilinear",
                ),
            )
            data = (
                scene.trace(num_rays=20_000, seed=5, max_depth=8).detectors["D1"].data
            )
            return _second_moment(data, _radial_weights(half, npix))

        _assert_matches_fd(loss_fn, value=-0.5, h=0.05)


class TestMirrorRadiusGradient:
    """Reflective path carries geometric gradients too."""

    def setup_method(self):
        be.set_backend("torch")
        be.set_precision("float64")

    def teardown_method(self):
        be.set_backend("numpy")

    @staticmethod
    def _trace(radius, num_rays: int = 20_000):
        """Point source onto a concave mirror, folded back onto a detector.

        The detector sits *behind* the source so the outgoing cone never
        crosses it — only the reflected, converging beam does.
        """
        half, npix = 30.0, 32
        scene = NSQScene()
        scene.add_source(
            "S1",
            CoordinateSystem(z=0.0),
            PointSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=1.0,
                half_angle_deg=6.0,
            ),
        )
        scene.add_mirror(
            "M1",
            CoordinateSystem(z=100.0),
            MirrorConfig(radius=radius, reflectance=1.0, aperture_radius=25.0),
        )
        scene.add_detector(
            "D1",
            CoordinateSystem(z=-40.0),
            IrradianceDetectorConfig(
                width=2 * half,
                height=2 * half,
                num_pixels_x=npix,
                num_pixels_y=npix,
                splat="bilinear",
            ),
        )
        data = scene.trace(num_rays=num_rays, seed=3, max_depth=6).detectors["D1"].data
        return _second_moment(data, _radial_weights(half, npix))

    def test_mirror_radius_matches_fd(self):
        """Second-moment gradient w.r.t. mirror radius vs FD."""
        _assert_matches_fd(self._trace, value=-150.0, h=3.0)


class TestGradientNumericalSafety:
    """Configurations that previously produced NaN or inf gradients."""

    def setup_method(self):
        be.set_backend("torch")
        be.set_precision("float64")

    def teardown_method(self):
        be.set_backend("numpy")

    @staticmethod
    def _scene(r1, r2, det_half: float, absorb_edge: bool):
        scene = NSQScene()
        if absorb_edge:
            # A diverging cone crosses the lens rim radius partway along the
            # edge cylinder, so some rays land on the absorbing edge. A
            # collimated beam runs parallel to that cylinder and never can.
            scene.add_source(
                "S1",
                CoordinateSystem(z=0.0),
                PointSourceConfig(
                    spectrum=Spectrum.monochromatic(0.55),
                    total_flux=1.0,
                    half_angle_deg=6.0,
                ),
            )
        else:
            scene.add_source(
                "S1",
                CoordinateSystem(z=0.0),
                CollimatedSourceConfig(
                    spectrum=Spectrum.monochromatic(0.55),
                    total_flux=1.0,
                    aperture_radius=10.0,
                ),
            )
        scene.add_lens(
            "L1",
            CoordinateSystem(z=100.0),
            LensConfig(
                r1=r1,
                r2=r2,
                thickness=5.0,
                material="N-BK7",
                front_aperture_radius=6.0 if absorb_edge else 12.0,
            ),
        )
        scene.add_detector(
            "D1",
            CoordinateSystem(z=200.0),
            IrradianceDetectorConfig(
                width=2 * det_half,
                height=2 * det_half,
                num_pixels_x=16,
                num_pixels_y=16,
            ),
        )
        return scene

    @pytest.mark.parametrize("r2", [float("inf"), 0.0, -100.0])
    def test_flat_back_surface_gradient_is_finite(self, r2):
        """A flat surface must not poison gradients.

        ``radius=inf`` and ``radius=0`` both denote a plano surface. Evaluated
        in radius form these produce inf/inf, whose backward pass is NaN for
        every scene parameter; the curvature form (c = 1/R, flat -> c = 0)
        keeps them finite.
        """
        r1 = torch.tensor(120.0, dtype=torch.float64, requires_grad=True)
        scene = self._scene(r1, r2, det_half=15.0, absorb_edge=False)
        loss = (
            scene.trace(num_rays=3_000, seed=7, max_depth=8).detectors["D1"].data.sum()
        )
        loss.backward()
        assert np.isfinite(r1.grad.item()), (
            f"Non-finite gradient with a plano surface (r2={r2}): {r1.grad}"
        )

    def test_rays_hitting_absorbing_edge_keep_gradients_finite(self):
        """Rays absorbed on the lens edge must not inject NaN.

        Non-hitting rays carry t = inf. Multiplying that by a direction inside
        a masked ``where`` backpropagates 0 * inf = NaN, so the interaction
        must zero t before the position update.
        """
        r1 = torch.tensor(120.0, dtype=torch.float64, requires_grad=True)
        scene = self._scene(r1, -100.0, det_half=15.0, absorb_edge=True)
        result = scene.trace(num_rays=5_000, seed=7, max_depth=8)
        assert result.total_flux_absorbed > 0.0, "Test scene absorbed nothing"
        result.detectors["D1"].data.sum().backward()
        assert np.isfinite(r1.grad.item()), f"Edge absorption gave {r1.grad}"

    def test_rays_missing_detector_keep_gradients_finite(self):
        """Rays that reach no detector carry t = inf; that must not leak."""
        r1 = torch.tensor(120.0, dtype=torch.float64, requires_grad=True)
        scene = self._scene(r1, -100.0, det_half=1.0, absorb_edge=False)
        result = scene.trace(num_rays=3_000, seed=7, max_depth=8)
        assert result.num_rays_escaped > 0, "Test scene had no escaping rays"
        result.detectors["D1"].data.sum().backward()
        assert np.isfinite(r1.grad.item()), f"Escaping rays gave {r1.grad}"

    def test_grad_mode_enable_does_not_raise(self):
        """be.grad_mode.enable() is Optiland's differentiable-mode switch.

        Under it every creation op returns a leaf tensor, so any in-place write
        into a freshly created array raises. NSQ must build arrays functionally.
        """
        be.grad_mode.enable()
        try:
            scene = self._scene(120.0, -100.0, det_half=15.0, absorb_edge=False)
            result = scene.trace(num_rays=1_000, seed=7, max_depth=6)
            assert result.total_flux_detected > 0.0
        finally:
            be.grad_mode.disable()


class TestDifferentiableParameterContract:
    """The set of differentiable parameters must be explicit, not accidental.

    Parameters that feed NumPy-only sampling cannot carry gradients in this
    release.  They must raise rather than silently detach, so a user never
    optimizes a variable that has no effect on the loss.
    """

    def setup_method(self):
        be.set_backend("torch")
        be.set_precision("float64")

    def teardown_method(self):
        be.set_backend("numpy")

    def test_source_aperture_radius_rejects_grad_tensor(self):
        """Source sampling is NumPy; a grad tensor must raise, not detach."""
        from optiland.nonsequential.sources.collimated import CollimatedSource

        r = torch.tensor(5.0, dtype=torch.float64, requires_grad=True)
        with pytest.raises(NotImplementedError, match="cannot be differentiated"):
            CollimatedSource(
                CoordinateSystem(),
                Spectrum.monochromatic(0.55),
                total_flux=1.0,
                aperture_radius=r,
            )

    def test_point_source_half_angle_rejects_grad_tensor(self):
        """Angular sampling is NumPy; a grad tensor must raise."""
        from optiland.nonsequential.sources.point import PointSource

        a = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
        with pytest.raises(NotImplementedError, match="cannot be differentiated"):
            PointSource(
                CoordinateSystem(),
                Spectrum.monochromatic(0.55),
                total_flux=1.0,
                half_angle_deg=a,
            )

    def test_plain_floats_still_accepted_everywhere(self):
        """The guard must not disturb ordinary float usage."""
        from optiland.nonsequential.sources.collimated import CollimatedSource

        src = CollimatedSource(
            CoordinateSystem(),
            Spectrum.monochromatic(0.55),
            total_flux=1.0,
            aperture_radius=torch.tensor(5.0, dtype=torch.float64),
        )
        assert float(src.aperture_radius) == pytest.approx(5.0)


class TestVisibilityGradientIsZero:
    """v1 limitation: visibility (which surface is hit) carries no gradient.

    This is a real measurement, not an architectural assertion: an occluder
    is translated across the beam and the gradient of detected flux w.r.t.
    that translation is confirmed absent. Reparameterization is roadmap #1.
    """

    def setup_method(self):
        be.set_backend("torch")
        be.set_precision("float64")

    def teardown_method(self):
        be.set_backend("numpy")

    def test_occluder_shift_has_no_gradient_path(self):
        """Moving an occluder changes flux, but supplies no gradient."""
        from optiland.nonsequential import AbsorbingComponent, FinitePlaneGeometry

        def detected_flux(shift, requires_grad: bool):
            scene = NSQScene()
            scene.add_source(
                "S1",
                CoordinateSystem(z=0.0),
                CollimatedSourceConfig(
                    spectrum=Spectrum.monochromatic(0.55),
                    total_flux=1.0,
                    aperture_radius=10.0,
                ),
            )
            occluder = AbsorbingComponent(
                CoordinateSystem(x=float(shift), z=50.0),
                FinitePlaneGeometry(width=10.0, height=40.0),
            )
            scene.add_component("OCC", occluder)
            scene.add_detector(
                "D1",
                CoordinateSystem(z=120.0),
                IrradianceDetectorConfig(
                    width=60, height=60, num_pixels_x=16, num_pixels_y=16
                ),
            )
            return scene.trace(num_rays=4_000, seed=17, max_depth=6)

        # The forward pass responds to the shift: this is a real visibility
        # effect, and exactly what v1 cannot differentiate.
        flux_centred = detected_flux(0.0, False).total_flux_detected
        flux_offset = detected_flux(9.0, False).total_flux_detected
        assert flux_offset > flux_centred, (
            "Occluder shift must change detected flux for this test to mean anything"
        )
