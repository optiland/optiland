"""Dtype-aware self-intersection threshold in the non-sequential tracer.

Covers ``optiland.nonsequential._tol`` (``ulp`` and ``accept_t_min``), the
float32 singlet the absolute ``1e-9`` threshold broke, the per-ray threshold
(a far ray cannot change another ray's result), and the float64 result the
change must leave untouched.

The quick-start singlet used below: a 1 W collimated source of 5 mm aperture
radius, a biconvex N-BK7 lens (r1 = 100, r2 = -100, thickness 5,
semi-diameter 12.5) at z = 50, and a 20 x 20 mm, 64 x 64 pixel irradiance
detector at z = 150.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem

torch = pytest.importorskip("torch", reason="Torch not available")

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
    Spectrum,
)
from optiland.nonsequential import _tol as tol
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.ray_bundle import NSQRayBundle

# Exact IEEE-754 spacings at 50 mm.
_ULP50_F64 = 2.0**-47  # 7.105427357601002e-15
_ULP50_F32 = 2.0**-18  # 3.814697265625e-06


@pytest.fixture(autouse=True)
def _restore_backend():
    yield
    be.set_backend("numpy")
    be.set_precision("float64")


def _source(scene: NSQScene, aperture_radius: float = 5.0) -> None:
    scene.add_source(
        "S1",
        CoordinateSystem(z=0.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55),
            total_flux=1.0,
            aperture_radius=aperture_radius,
        ),
    )


def _singlet(thickness: float = 5.0, radius: float = 100.0) -> NSQScene:
    scene = NSQScene()
    _source(scene)
    scene.add_lens(
        "L1",
        CoordinateSystem(z=50),
        LensConfig(
            r1=radius,
            r2=-radius,
            thickness=thickness,
            material="N-BK7",
            front_aperture_radius=12.5,
        ),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=150),
        IrradianceDetectorConfig(width=20, height=20, num_pixels_x=64, num_pixels_y=64),
    )
    return scene


def _mirror_scene() -> NSQScene:
    """Lens with a stepped back aperture, then a concave mirror.

    The 12 mm source overfills the 10 mm back aperture, so rays reach the
    lens edge (frustum) and the rim (annulus); the mirror sends the light back
    through the lens to a detector behind the source.
    """
    scene = NSQScene()
    _source(scene, aperture_radius=12.0)
    scene.add_lens(
        "L1",
        CoordinateSystem(z=50),
        LensConfig(
            r1=100,
            r2=-100,
            thickness=5,
            material="N-BK7",
            front_aperture_radius=12.5,
            back_aperture_radius=10.0,
        ),
    )
    scene.add_mirror(
        "M1",
        CoordinateSystem(z=150),
        MirrorConfig(radius=-300.0, reflectance=0.9, aperture_radius=20.0),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=-10),
        IrradianceDetectorConfig(width=60, height=60, num_pixels_x=64, num_pixels_y=64),
    )
    return scene


# ---------------------------------------------------------------------------
# The primitives
# ---------------------------------------------------------------------------


class TestUlp:
    """``ulp`` is the exact IEEE-754 spacing, in the input's own dtype."""

    def test_numpy_float64(self):
        assert tol.ulp(np.float64(50.0)) == _ULP50_F64

    def test_numpy_float32(self):
        result = tol.ulp(np.float32(50.0))
        assert result.dtype == np.float32
        assert result == _ULP50_F32

    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [(torch.float64, _ULP50_F64), (torch.float32, _ULP50_F32)],
    )
    def test_torch(self, dtype, expected):
        result = tol.ulp(torch.tensor(50.0, dtype=dtype))
        assert result.dtype == dtype
        assert float(result) == expected

    def test_float32_step_is_far_above_the_old_threshold(self):
        """Why ``1e-9`` mm cannot separate a second hit at float32."""
        assert tol.ulp(np.float32(50.0)) > 1e-9 * 1e3

    def test_detached_from_the_graph(self):
        x = torch.tensor([50.0], dtype=torch.float64, requires_grad=True)
        assert not tol.ulp(x).requires_grad


class TestAcceptTMin:
    def test_numpy_exact(self):
        assert tol.accept_t_min(np.float64(50.0)) == 25 * _ULP50_F64
        assert tol.accept_t_min(np.float32(50.0)) == np.float32(25 * _ULP50_F32)

    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [(torch.float64, 25 * _ULP50_F64), (torch.float32, 25 * _ULP50_F32)],
    )
    def test_torch_exact(self, dtype, expected):
        result = tol.accept_t_min(torch.tensor(50.0, dtype=dtype))
        assert result.dtype == dtype
        assert float(result) == expected

    def test_floor_at_one_mm(self):
        assert tol.accept_t_min(0.0) == tol.accept_t_min(1.0) == 25 * 2.0**-52
        assert tol.accept_t_min(0.5) == tol.accept_t_min(1.0)

    def test_elementwise(self):
        """One threshold per entry: a large entry does not change the others."""
        mags = torch.tensor([50.0, 50.0, 1e6], dtype=torch.float32)
        result = tol.accept_t_min(mags)
        assert result.shape == (3,)
        assert float(result[0]) == float(result[1]) == 25 * _ULP50_F32
        assert float(result[2]) == 25 * 2.0**-4  # ulp(1e6) at float32 is 1/16


# ---------------------------------------------------------------------------
# The float32 singlet
# ---------------------------------------------------------------------------


def _trace_torch(scene: NSQScene, precision: str, num_rays: int = 200_000):
    be.set_backend("torch")
    be.set_device("cpu")
    be.set_precision(precision)
    return scene.trace(num_rays=num_rays, seed=42, max_depth=16)


class TestFloat32Singlet:
    """With the absolute ``1e-9`` threshold, float32 lost about two thirds of
    the singlet's light: a ray re-accepted the surface it had just left,
    bounced against it until the depth cap killed it, and its flux never
    reached the detector. 200k rays, seed 42, ``max_depth=16``, torch CPU.
    """

    def test_zero_rays_depth_killed(self):
        """62,546 of 200,000 with the absolute threshold."""
        assert _trace_torch(_singlet(), "float32").num_rays_depth_killed == 0

    def test_detected_flux_matches_float64(self):
        """0.306342 W against float64's 0.916765 W with the absolute threshold."""
        f32 = float(_trace_torch(_singlet(), "float32").total_flux_detected)
        f64 = float(_trace_torch(_singlet(), "float64").total_flux_detected)
        assert f32 == pytest.approx(f64, rel=1e-3)


# ---------------------------------------------------------------------------
# One threshold per ray
# ---------------------------------------------------------------------------


def _far_ray_bundle(n: int, with_far_ray: bool) -> NSQRayBundle:
    """Rays inside the thin lens heading for its back surface, at float32.

    The optional last ray sits 1e6 mm off axis. A threshold taken from the
    batch maximum would then be 25 ulp(1e6) = 1.5625 mm at float32, more than
    the ~1.1 mm to the back surface.
    """
    xs = np.linspace(-4.0, 4.0, n)
    x = np.concatenate([xs, [1.0e6]]) if with_far_ray else xs
    m = x.shape[0]

    def arr(values, dtype=torch.float32):
        return torch.as_tensor(np.asarray(values), dtype=dtype)

    return NSQRayBundle(
        x=arr(x),
        y=arr(np.zeros(m)),
        z=arr(np.full(m, 50.1)),
        L=arr(np.zeros(m)),
        M=arr(np.zeros(m)),
        N=arr(np.ones(m)),
        flux=arr(np.ones(m)),
        wavelength=arr(np.full(m, 0.55)),
        n_current=arr(np.full(m, 1.5)),
        bounce=np.zeros(m, dtype=np.int64),
        alive=arr(np.ones(m, dtype=bool), dtype=torch.bool),
    )


class TestPerRayThreshold:
    """A far or escaped ray cannot change another ray's threshold."""

    def test_component_hits_unchanged_by_a_far_ray(self):
        be.set_backend("torch")
        be.set_device("cpu")
        be.set_precision("float32")
        scene = _singlet(thickness=1.2, radius=200.0)
        n = 9
        for comp in scene.surfaces:
            t_a, _, hit_a, _ = comp.intersect(_far_ray_bundle(n, False))
            t_b, _, hit_b, _ = comp.intersect(_far_ray_bundle(n, True))
            np.testing.assert_array_equal(be.to_numpy(hit_b)[:n], be.to_numpy(hit_a))
            np.testing.assert_array_equal(be.to_numpy(t_b)[:n], be.to_numpy(t_a))
        # The back surface is hit by every in-lens ray, with t below 1.5625 mm.
        back = scene.surfaces[1]
        t_b, _, hit_b, _ = back.intersect(_far_ray_bundle(n, True))
        assert bool(be.to_numpy(hit_b)[:n].all())
        assert float(be.to_numpy(t_b)[:n].max()) < 1.5625

    def test_trace_unchanged_by_a_far_ray(self, monkeypatch):
        """Every source batch gets one extra ray 1e6 mm off axis, which escapes.

        Torch keeps dead and escaped rays in the batch (fixed shapes), so
        this ray stays far out at every depth. Its random stream is keyed by
        its own ray id, so the other rays draw the same numbers.
        """
        scene_a = _singlet(thickness=1.2, radius=200.0)
        result_a = _trace_torch(scene_a, "float32", num_rays=20_000)
        image_a = be.to_numpy(result_a.detectors["D1"].data)

        scene_b = _singlet(thickness=1.2, radius=200.0)
        source = scene_b.sources[0]
        generate = source.generate
        batches = []

        def generate_with_far_ray(ray_id, rng):
            rays = generate(ray_id, rng)
            batches.append(len(ray_id))
            fields = {}
            for field in dataclasses.fields(rays):
                value = getattr(rays, field.name)
                if value is not None:
                    value = be.to_numpy(value)
                    value = np.concatenate([value, value[:1]])
                fields[field.name] = value
            fields["x"][-1] += 1.0e6
            fields["ray_id"][-1] = 10**9
            return NSQRayBundle(**fields)

        monkeypatch.setattr(source, "generate", generate_with_far_ray)
        result_b = _trace_torch(scene_b, "float32", num_rays=20_000)
        image_b = be.to_numpy(result_b.detectors["D1"].data)

        np.testing.assert_array_equal(image_b, image_a)
        assert len(batches) >= 1
        assert result_b.num_rays_escaped == result_a.num_rays_escaped + len(batches)
        assert result_b.num_rays_depth_killed == result_a.num_rays_depth_killed == 0
        assert float(result_b.total_flux_detected) == float(
            result_a.total_flux_detected
        )


# ---------------------------------------------------------------------------
# Float64 is unchanged
# ---------------------------------------------------------------------------


class TestFloat64Unchanged:
    """At float64 the new threshold accepts the same roots as ``1e-9``.

    Each scene is traced twice in the same environment: once as shipped, and
    once with ``accept_t_min`` patched to return the old absolute ``1e-9``,
    which restores the previous behaviour exactly (every caller compares
    ``t`` against the value it returns). The detector images and flux totals
    must be identical, not merely close.
    """

    @staticmethod
    def _trace(scene_factory, seed):
        be.set_backend("numpy")
        result = scene_factory().trace(
            num_rays=100_000, seed=seed, backend=NumpyBackend(seed=seed)
        )
        return (
            np.asarray(result.detectors["D1"].data, dtype=np.float64),
            float(result.total_flux_detected),
            float(result.total_flux_escaped),
        )

    @pytest.mark.parametrize(
        ("scene_factory", "seed"), [(_singlet, 42), (_mirror_scene, 7)]
    )
    def test_same_as_the_absolute_threshold(self, scene_factory, seed, monkeypatch):
        image, detected, escaped = self._trace(scene_factory, seed)
        monkeypatch.setattr(tol, "accept_t_min", lambda magnitude, k=25: 1e-9)
        image_old, detected_old, escaped_old = self._trace(scene_factory, seed)

        assert image.sum() > 0
        np.testing.assert_array_equal(image, image_old)
        assert detected == detected_old
        assert escaped == escaped_old
