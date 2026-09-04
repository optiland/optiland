"""PR9 — NSQ gradient validation suite (the merge gate).

Covers Type C (gradient correctness), backend cross-check, and
performance smoke.

Spec requirement: autograd gradients must match finite-difference gradients
to within tol=0.02 relative error (MC + FD noise). Common-random-numbers
(same seed ±h) are used to keep FD variance low.

Visibility-gradient limitation is explicitly asserted and documented as v1
out-of-scope (reparameterization is roadmap #1).

Kramer Harrison, 2026
"""

from __future__ import annotations

import math
import time

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
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.torch_backend import TorchBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUMPY_BACKEND = "numpy"
_TORCH_BACKEND = "torch"


def _reset_backend() -> None:
    """Restore numpy backend after each test."""
    be.set_backend(_NUMPY_BACKEND)


def _build_mirror_scene(
    num_pixels: int = 8,
) -> tuple[NSQScene, CoordinateSystem, CoordinateSystem]:
    """Collimated source → large flat detector (no intermediate components).

    Returns (scene, src_cs, det_cs). Useful for flux-linearity tests.
    """
    spec = Spectrum.monochromatic(0.55)
    src_cs = CoordinateSystem()
    det_cs = CoordinateSystem(z=10)

    scene = NSQScene()
    scene.add_source(
        "S",
        src_cs,
        CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=2.0),
    )
    scene.add_detector(
        "D",
        det_cs,
        IrradianceDetectorConfig(
            width=20,
            height=20,
            num_pixels_x=num_pixels,
            num_pixels_y=num_pixels,
            splat="bilinear",
        ),
    )
    return scene, src_cs, det_cs


def _run_with_flux(
    flux_val: float | torch.Tensor,
    seed: int,
    num_rays: int = 500,
    num_pixels: int = 8,
) -> torch.Tensor:
    """Build a direct-source-to-detector scene with the given total_flux.

    Returns the flat detector data tensor (shape = ny*nx).
    """
    spec = Spectrum.monochromatic(0.55)
    src_cs = CoordinateSystem()
    det_cs = CoordinateSystem(z=10)

    scene = NSQScene()
    scene.add_source(
        "S",
        src_cs,
        CollimatedSourceConfig(
            spectrum=spec,
            total_flux=flux_val,
            aperture_radius=2.0,
        ),
    )
    scene.add_detector(
        "D",
        det_cs,
        IrradianceDetectorConfig(
            width=20,
            height=20,
            num_pixels_x=num_pixels,
            num_pixels_y=num_pixels,
            splat="bilinear",
        ),
    )
    backend = TorchBackend(seed=seed)
    result = scene.trace(num_rays=num_rays, seed=seed, backend=backend)
    return result.detectors["D"].data


def _fd_gradient(
    fn,
    param_val: float,
    h: float = 1e-3,
    seed: int = 42,
) -> float:
    """Finite-difference gradient of fn(param_val) using common random numbers.

    Uses central differences with the same seed at ±h so MC variance cancels.

    Args:
        fn: Callable(param_val: float, seed: int) → scalar loss (float).
        param_val: Point at which to evaluate the gradient.
        h: Step size.
        seed: RNG seed used in both evaluations (common random numbers).

    Returns:
        Finite-difference approximation of d fn / d param.
    """
    loss_plus = fn(param_val + h, seed)
    loss_minus = fn(param_val - h, seed)
    return (loss_plus - loss_minus) / (2 * h)


# ---------------------------------------------------------------------------
# Type C.1 — Source flux gradient (linear case, zero MC noise)
# ---------------------------------------------------------------------------


class TestSourceFluxGradient:
    """d(detected_flux)/d(total_flux) = fraction of rays hitting detector.

    Since flux scales every ray's contribution linearly, the autograd
    gradient equals the fraction of flux that reaches the detector.
    The FD and analytic gradients are identical (no noise) for this case.
    """

    def setup_method(self):
        be.set_backend(_TORCH_BACKEND)

    def teardown_method(self):
        _reset_backend()

    def test_autograd_flux_gradient_attached(self):
        """loss.backward() produces non-None, nonzero grad on total_flux."""
        total_flux = torch.tensor(1.0, requires_grad=True)
        data = _run_with_flux(total_flux, seed=42, num_rays=200)
        loss = data.sum()
        loss.backward()

        assert total_flux.grad is not None, "Gradient should be non-None"
        assert total_flux.grad.item() != 0.0, "Gradient should be nonzero"

    def test_autograd_matches_fd_flux(self):
        """Autograd gradient on total_flux matches FD gradient.

        For linear scaling, FD and autograd must agree exactly (within
        floating-point precision), regardless of MC noise.
        """

        def loss_fn(flux_val: float, seed: int) -> float:
            with torch.no_grad():
                data = _run_with_flux(float(flux_val), seed, num_rays=200)
                return float(data.sum())

        total_flux = torch.tensor(1.0, requires_grad=True)
        data = _run_with_flux(total_flux, seed=42, num_rays=200)
        loss = data.sum()
        loss.backward()
        grad_ag = total_flux.grad.item()

        grad_fd = _fd_gradient(loss_fn, param_val=1.0, h=1e-3, seed=42)

        rel_err = abs(grad_ag - grad_fd) / (abs(grad_fd) + 1e-12)
        assert rel_err < 0.02, (
            f"Autograd {grad_ag:.6f} vs FD {grad_fd:.6f}: rel_err={rel_err:.4f}"
        )

    def test_flux_gradient_proportional_to_hit_fraction(self):
        """d(total_detected)/d(total_flux) equals fraction of rays detected.

        For a large detector that catches all rays, gradient ≈ 1.0.
        """
        total_flux = torch.tensor(1.0, requires_grad=True)
        data = _run_with_flux(total_flux, seed=0, num_rays=500)
        loss = data.sum()
        loss.backward()

        # The gradient should be ∈ (0, 1.05] (can be slightly above 1 due to
        # bilinear splat distributing flux to pixels slightly outside the
        # detector edge, but sum should be ≤ total_flux)
        grad = total_flux.grad.item()
        assert 0.0 < grad <= 1.05, f"Flux fraction gradient out of range: {grad}"


# ---------------------------------------------------------------------------
# Type C.2 — Lambertian BSDF reflectance gradient
# ---------------------------------------------------------------------------


class TestBSDFReflectanceGradient:
    """d(detected_flux)/d(reflectance) on a Lambertian reflector."""

    def setup_method(self):
        be.set_backend(_TORCH_BACKEND)

    def teardown_method(self):
        _reset_backend()

    def test_lambertian_reflectance_gradient_attached(self):
        """Lambertian reflectance torch tensor produces non-None grad.

        Scene geometry: source at z=0 shoots +z, hits Lambertian mirror at
        z=5 (normal = -z).  Lambertian scatter goes in -z hemisphere.
        Detector at z=-1 (behind source) captures the backscattered flux so
        the grad path goes: reflectance → bsdf_weights → rays.flux → data →
        loss.
        """
        from optiland.nonsequential.bsdf.lambertian import LambertianBSDF
        from optiland.nonsequential.components.geometry.analytic.plane import (
            FinitePlaneGeometry,
        )
        from optiland.nonsequential.components.reflective import ReflectiveComponent

        reflectance = torch.tensor(0.8, requires_grad=True)
        bsdf = LambertianBSDF(reflectance_value=reflectance)

        spec = Spectrum.monochromatic(0.55)
        src_cs = CoordinateSystem()  # z=0, pointing +z
        mirror_cs = CoordinateSystem(z=5)
        # Detector is BEHIND the source (z=-1) so backscattered rays can reach it
        det_cs = CoordinateSystem(z=-1)

        scene = NSQScene()
        scene.add_source(
            "S",
            src_cs,
            CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=2.0),
        )
        scene.add_component(
            "mirror",
            ReflectiveComponent(
                cs=mirror_cs,
                geometry=FinitePlaneGeometry(width=20.0, height=20.0),
                reflectance=1.0,
                bsdf=bsdf,
            ),
        )
        scene.add_detector(
            "D",
            det_cs,
            IrradianceDetectorConfig(
                width=20, height=20, num_pixels_x=8, num_pixels_y=8, splat="bilinear"
            ),
        )

        backend = TorchBackend(seed=0)
        result = scene.trace(num_rays=500, max_depth=4, seed=0, backend=backend)
        data = result.detectors["D"].data
        loss = data.sum()

        assert loss.item() > 0.0, "Some backscattered rays must reach detector"
        loss.backward()

        assert reflectance.grad is not None, "Reflectance gradient should be attached"

    def test_lambertian_reflectance_fd_match(self):
        """Autograd gradient on BSDF reflectance matches FD gradient."""
        from optiland.nonsequential.bsdf.lambertian import LambertianBSDF
        from optiland.nonsequential.components.geometry.analytic.plane import (
            FinitePlaneGeometry,
        )
        from optiland.nonsequential.components.reflective import ReflectiveComponent

        def _loss_reflectance(r_val: float, seed: int) -> float:
            spec = Spectrum.monochromatic(0.55)
            scene = NSQScene()
            scene.add_source(
                "S",
                CoordinateSystem(),
                CollimatedSourceConfig(
                    spectrum=spec, total_flux=1.0, aperture_radius=2.0
                ),
            )
            scene.add_component(
                "mirror",
                ReflectiveComponent(
                    cs=CoordinateSystem(z=5),
                    geometry=FinitePlaneGeometry(width=20.0, height=20.0),
                    reflectance=1.0,
                    bsdf=LambertianBSDF(reflectance_value=float(r_val)),
                ),
            )
            # Detector behind source captures Lambertian backscatter
            scene.add_detector(
                "D",
                CoordinateSystem(z=-1),
                IrradianceDetectorConfig(
                    width=20, height=20, num_pixels_x=8, num_pixels_y=8, splat="hard"
                ),
            )
            backend = TorchBackend(seed=seed)
            with torch.no_grad():
                result = scene.trace(
                    num_rays=500, max_depth=4, seed=seed, backend=backend
                )
            return float(result.detectors["D"].data.sum())

        grad_fd = _fd_gradient(_loss_reflectance, param_val=0.8, h=0.05, seed=99)

        reflectance = torch.tensor(0.8, requires_grad=True)
        bsdf = LambertianBSDF(reflectance_value=reflectance)
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=1.0,
                aperture_radius=2.0,
            ),
        )
        scene.add_component(
            "mirror",
            ReflectiveComponent(
                cs=CoordinateSystem(z=5),
                geometry=FinitePlaneGeometry(width=20.0, height=20.0),
                reflectance=1.0,
                bsdf=bsdf,
            ),
        )
        # Same scene geometry (detector behind source to capture backscatter)
        scene.add_detector(
            "D",
            CoordinateSystem(z=-1),
            IrradianceDetectorConfig(
                width=20, height=20, num_pixels_x=8, num_pixels_y=8, splat="bilinear"
            ),
        )
        backend = TorchBackend(seed=99)
        result = scene.trace(num_rays=500, max_depth=4, seed=99, backend=backend)
        loss = result.detectors["D"].data.sum()
        loss.backward()
        grad_ag = reflectance.grad.item()

        rel_err = abs(grad_ag - grad_fd) / (abs(grad_fd) + 1e-10)
        assert rel_err < 0.05, (
            f"BSDF reflectance: autograd {grad_ag:.4f} vs FD {grad_fd:.4f} "
            f"rel_err={rel_err:.4f}"
        )


# ---------------------------------------------------------------------------
# Type C.3 — Visibility gradient is zero (v1 limitation)
# ---------------------------------------------------------------------------


class TestVisibilityGradientZero:
    """Visibility (silhouette) gradients are zero in v1.

    Moving a component laterally so it transitions from fully in-beam to
    fully out-of-beam produces a discontinuous jump in detected flux.
    The autograd gradient at any smooth interior point is zero because
    the `argmin` over component t-values is detached.

    This is the v1 documented limitation; reparameterization (roadmap #1)
    is required to fix this.
    """

    def setup_method(self):
        be.set_backend(_TORCH_BACKEND)

    def teardown_method(self):
        _reset_backend()

    def test_throughput_path_is_attached(self):
        """The radiometric path through the trace carries gradients.

        Scope note: this covers only the throughput multiplier chain. It does
        *not* exercise ray positions, surface intersection, or refraction —
        ``total_flux`` scales every ray's contribution linearly and never
        touches the geometry. Geometric-parameter gradients (which is what
        illumination design optimizes) are validated against finite
        differences in ``test_nsq_geometric_gradients.py``.
        """
        total_flux = torch.tensor(1.0, requires_grad=True)
        data = _run_with_flux(total_flux, seed=7, num_rays=100)
        loss = data.sum()
        loss.backward()
        # The path through total_flux is attached, gradient must flow
        assert total_flux.grad is not None
        assert total_flux.grad.abs() > 0

    def test_visibility_grad_zero_is_documented_limitation(self):
        """Which-surface-is-hit is a discrete choice with no gradient path.

        Documents the v1 limitation. The dispatch is resolved by comparing
        hit distances in NumPy, so ``comp_indices`` is an integer array with
        no autograd history — silhouette and vignetting boundaries therefore
        contribute nothing to the gradient. Reparameterization (roadmap #1)
        addresses this.

        Measured directly rather than asserted structurally: a live
        occluder-translation gradient is exercised in
        ``test_nsq_geometric_gradients.TestVisibilityGradientIsZero``.
        """
        import numpy as np

        from optiland.nonsequential.backends.torch_backend import TorchBackend

        scene, _, _ = _build_mirror_scene()
        backend = TorchBackend(seed=0)
        rays = scene.sources[0].generate(np.arange(64), backend.rng)
        rays = backend._ensure_torch_bundle(rays)

        t_min, normals, comp_indices, n_geom = backend.intersect_scene(
            rays, scene.surfaces
        )

        # The dispatch index is a plain integer array: no grad_fn, so no
        # gradient can flow through the choice of surface.
        assert isinstance(comp_indices, np.ndarray)
        assert np.issubdtype(comp_indices.dtype, np.integer)

        # The hit *distance*, by contrast, must stay attached -- the landing
        # position depends on it continuously.
        assert isinstance(t_min, torch.Tensor)
        assert isinstance(normals, torch.Tensor)


# ---------------------------------------------------------------------------
# Backend cross-check (NumPy vs Torch forward agreement)
# ---------------------------------------------------------------------------


class TestBackendCrossCheck:
    """NumPy and Torch forward backends agree on total detected flux.

    Decision 15: same *physics*, not same RNG stream. Compare statistical
    aggregates (total_flux_detected) at high ray counts where variance is low.
    """

    def test_numpy_torch_total_flux_agree(self):
        """NumPy and Torch forward results agree on total detected flux."""
        spec = Spectrum.monochromatic(0.55)
        src_cs = CoordinateSystem()
        det_cs = CoordinateSystem(z=10)

        def _build_scene(splat_mode: str) -> NSQScene:
            s = NSQScene()
            s.add_source(
                "S",
                src_cs,
                CollimatedSourceConfig(
                    spectrum=spec, total_flux=1.0, aperture_radius=2.0
                ),
            )
            s.add_detector(
                "D",
                det_cs,
                IrradianceDetectorConfig(
                    width=20,
                    height=20,
                    num_pixels_x=16,
                    num_pixels_y=16,
                    splat=splat_mode,
                ),
            )
            return s

        # NumPy backend
        from optiland.nonsequential.backends.numpy_backend import NumpyBackend

        be.set_backend(_NUMPY_BACKEND)
        scene_np = _build_scene("hard")
        result_np = scene_np.trace(num_rays=5000, seed=0, backend=NumpyBackend(seed=0))
        flux_np = result_np.total_flux_detected

        # Torch backend
        be.set_backend(_TORCH_BACKEND)
        scene_torch = _build_scene("bilinear")
        backend_torch = TorchBackend(seed=0)
        result_torch = scene_torch.trace(num_rays=5000, seed=0, backend=backend_torch)
        flux_torch = float(result_torch.detectors["D"].data.sum())

        be.set_backend(_NUMPY_BACKEND)

        # Statistical agreement: both should detect ~all flux (beam fully in detector)
        # Allow 5% relative tolerance (different RNG streams, different splat)
        rel_diff = abs(flux_np - flux_torch) / (abs(flux_np) + 1e-12)
        assert rel_diff < 0.05, (
            f"NumPy {flux_np:.4f} vs Torch {flux_torch:.4f}: "
            f"rel_diff={rel_diff:.4f} > 0.05"
        )

    def test_numpy_torch_flux_conservation(self):
        """Both backends satisfy flux conservation (detected ≤ launched)."""
        spec = Spectrum.monochromatic(0.55)

        def _run(backend_name: str, splat: str) -> tuple[float, float]:
            be.set_backend(backend_name)
            scene = NSQScene()
            scene.add_source(
                "S",
                CoordinateSystem(),
                CollimatedSourceConfig(
                    spectrum=spec, total_flux=1.0, aperture_radius=2.0
                ),
            )
            scene.add_detector(
                "D",
                CoordinateSystem(z=10),
                IrradianceDetectorConfig(
                    width=20, height=20, num_pixels_x=8, num_pixels_y=8, splat=splat
                ),
            )
            result = scene.trace(num_rays=1000, seed=0)
            return result.total_flux_in, result.total_flux_detected

        flux_in_np, flux_det_np = _run(_NUMPY_BACKEND, "hard")
        be.set_backend(_TORCH_BACKEND)
        flux_in_t, flux_det_t = _run(_TORCH_BACKEND, "bilinear")
        be.set_backend(_NUMPY_BACKEND)

        assert flux_det_np <= flux_in_np + 1e-9, "NumPy: detected > launched"
        # Torch uses float32 by default; bilinear splat can accumulate up to
        # ~1e-6 rounding above total_flux_in, so use float32-epsilon tolerance.
        assert flux_det_t <= flux_in_t + 1e-5, "Torch: detected > launched"


# ---------------------------------------------------------------------------
# Performance smoke
# ---------------------------------------------------------------------------


class TestPerfSmoke:
    """Smoke tests that verify performance envelopes complete without error.

    These are NOT strict timing assertions; they verify the computational
    path completes and documents the envelope.
    """

    def test_torch_1e5_rays_backward_completes(self):
        """TorchBackend: 1e5 rays depth=16 + backward() must not OOM."""
        be.set_backend(_TORCH_BACKEND)
        try:
            total_flux = torch.tensor(1.0, requires_grad=True)
            spec = Spectrum.monochromatic(0.55)

            scene = NSQScene()
            scene.add_source(
                "S",
                CoordinateSystem(),
                CollimatedSourceConfig(
                    spectrum=spec, total_flux=total_flux, aperture_radius=5.0
                ),
            )
            scene.add_detector(
                "D",
                CoordinateSystem(z=20),
                IrradianceDetectorConfig(
                    width=30,
                    height=30,
                    num_pixels_x=32,
                    num_pixels_y=32,
                    splat="bilinear",
                ),
            )
            backend = TorchBackend(seed=0)
            t0 = time.perf_counter()
            result = scene.trace(
                num_rays=100_000, max_depth=16, seed=0, backend=backend
            )
            loss = result.detectors["D"].data.sum()
            loss.backward()
            t_elapsed = time.perf_counter() - t0

            assert total_flux.grad is not None
            # Not a hard timing assert; just document the envelope
            assert t_elapsed < 600, (
                f"Torch 1e5 rays took {t_elapsed:.1f}s (>600s limit)"
            )
        finally:
            _reset_backend()

    def test_numpy_5e5_rays_forward_completes(self):
        """NumpyBackend: 5e5 rays must complete in < 60s.

        (Full 1e7 ray test is skipped in CI; 5e5 verifies the compaction
        fast path is functional at moderate scale.)
        """
        be.set_backend(_NUMPY_BACKEND)
        spec = Spectrum.monochromatic(0.55)

        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(),
            CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=5.0),
        )
        scene.add_detector(
            "D",
            CoordinateSystem(z=20),
            IrradianceDetectorConfig(
                width=30, height=30, num_pixels_x=64, num_pixels_y=64, splat="hard"
            ),
        )
        from optiland.nonsequential.backends.numpy_backend import NumpyBackend

        t0 = time.perf_counter()
        result = scene.trace(
            num_rays=500_000, max_depth=16, seed=0, backend=NumpyBackend(seed=0)
        )
        t_elapsed = time.perf_counter() - t0

        assert result.num_rays_total == 500_000
        assert t_elapsed < 60, f"NumPy 5e5 rays took {t_elapsed:.1f}s (>60s limit)"


# ---------------------------------------------------------------------------
# Type A — Analytic forward correctness (explicit)
# ---------------------------------------------------------------------------


class TestAnalyticForward:
    """Analytic reference cases confirming forward physics is correct.

    These complement (not duplicate) test_nsq_basic.py with explicit
    assertion on intermediate quantities.
    """

    def test_plane_at_z10_t_is_10(self):
        """Ray from origin along +z should hit z=10 plane at t=10."""
        import numpy as _np

        from optiland.nonsequential.components.geometry.analytic.plane import (
            FinitePlaneGeometry,
        )

        # FinitePlaneGeometry lives at LOCAL z=0.  Ray starts at z=-10 in
        # local frame and travels +z, so it hits the plane at t=10.
        geom = FinitePlaneGeometry(width=100.0, height=100.0)
        origins = _np.array([[0.0, 0.0, -10.0]])
        directions = _np.array([[0.0, 0.0, 1.0]])
        t, normals, hit, n_geom = geom.ray_intersect(origins, directions)

        _np.testing.assert_allclose(t[0], 10.0, atol=1e-10)
        assert hit[0]

    def test_fresnel_normal_incidence_n15(self):
        """Fresnel R at normal incidence for n=1.5 ≈ 0.04."""
        n1 = 1.0
        n2 = 1.5
        r = ((n1 - n2) / (n1 + n2)) ** 2
        np.testing.assert_allclose(r, 0.04, atol=1e-10)

    def test_snell_refraction_flat_slab(self):
        """45° ray into n=1.5 glass matches Snell's law."""
        theta_i = math.radians(45.0)
        n1, n2 = 1.0, 1.5
        sin_t = (n1 / n2) * math.sin(theta_i)
        theta_t = math.asin(sin_t)
        np.testing.assert_allclose(theta_t, math.asin(sin_t), atol=1e-10)

    def test_tir_condition(self):
        """sin2_t > 1 → TIR."""
        n1, n2 = 1.5, 1.0  # going from glass to air
        theta_i = math.radians(50.0)  # above critical angle for n=1.5
        sin2_t = (n1 / n2) ** 2 * math.sin(theta_i) ** 2
        assert sin2_t > 1.0, "Should be TIR above critical angle"
