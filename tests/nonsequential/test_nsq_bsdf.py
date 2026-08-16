"""BSDF sampling correctness for the NSQ engine.

The scattering BSDFs (Lambertian, Harvey-Shack, tabulated) all build a local
tangent frame around the surface normal via ``_orthonormal_basis``. That
helper is evaluated for *every* ray in the bundle, including rays that hit
nothing and therefore carry a zero normal, so it has to stay finite for
degenerate input as well as produce a genuinely orthonormal frame.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

import optiland.backend as be
from optiland.nonsequential import HarveyShackBSDF, LambertianBSDF
from optiland.nonsequential.bsdf.lambertian import _orthonormal_basis


class TestOrthonormalBasis:
    """The local scatter frame must be orthonormal and never degenerate."""

    @staticmethod
    def _random_unit(n: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(n, 3))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def test_frame_is_orthonormal(self):
        """t and b are unit vectors, mutually perpendicular and normal to n."""
        n = self._random_unit(5000)
        t, b = _orthonormal_basis(n)

        assert np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-12)
        assert np.allclose(np.linalg.norm(b, axis=1), 1.0, atol=1e-12)
        assert np.allclose((t * n).sum(axis=1), 0.0, atol=1e-12)
        assert np.allclose((b * n).sum(axis=1), 0.0, atol=1e-12)
        assert np.allclose((t * b).sum(axis=1), 0.0, atol=1e-12)

    @pytest.mark.parametrize(
        "normal",
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],  # sign flip in the branchless construction
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
    )
    def test_axis_aligned_normals_are_finite(self, normal):
        """Axis-aligned normals must not degenerate.

        The previous construction crossed n with a fixed reference axis, which
        collapses to the zero vector when the two are parallel.
        """
        n = np.array([normal], dtype=float)
        t, b = _orthonormal_basis(n)

        assert np.isfinite(t).all() and np.isfinite(b).all()
        assert np.allclose(np.linalg.norm(t, axis=1), 1.0)
        assert np.allclose((t * n).sum(axis=1), 0.0, atol=1e-12)

    def test_zero_normal_is_finite(self):
        """Rays that hit nothing carry a zero normal; output must stay finite.

        BSDF sampling runs over the whole bundle, so a NaN here contaminates
        the warning stream and, on the torch backend, the gradients.
        """
        t, b = _orthonormal_basis(np.zeros((4, 3)))
        assert np.isfinite(t).all(), "Zero normal produced a non-finite tangent"
        assert np.isfinite(b).all(), "Zero normal produced a non-finite bitangent"


class TestLambertianSampling:
    """Lambertian scatter must be cosine-weighted about the surface normal."""

    def setup_method(self):
        be.set_backend("numpy")

    def test_no_warning_for_zero_normals(self):
        """Sampling with zero normals must not emit a RuntimeWarning."""
        bsdf = LambertianBSDF(reflectance_value=1.0)
        n = 256
        with np.errstate(all="raise"):
            dirs, _ = bsdf.sample(
                n,
                np.tile([0.0, 0.0, 1.0], (n, 1)),
                np.zeros((n, 3)),
                np.full(n, 0.55),
                np.random.default_rng(0),
            )
        assert np.isfinite(np.asarray(dirs)).all()

    def test_directions_are_unit_and_in_hemisphere(self):
        """Every scattered ray is a unit vector in the normal's hemisphere."""
        bsdf = LambertianBSDF(reflectance_value=1.0)
        n_rays = 20_000
        normal = np.array([0.3, -0.5, 0.8])
        normal /= np.linalg.norm(normal)

        dirs, _ = bsdf.sample(
            n_rays,
            np.tile([0.0, 0.0, 1.0], (n_rays, 1)),
            np.tile(normal, (n_rays, 1)),
            np.full(n_rays, 0.55),
            np.random.default_rng(1),
        )
        dirs = np.asarray(dirs)

        assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-12)
        assert (dirs @ normal > -1e-12).all(), "Ray scattered below the surface"

    def test_angular_distribution_is_cosine_weighted(self):
        """p(theta) = sin(2*theta), i.e. <cos(theta)> = 2/3.

        A uniform hemisphere would instead give p(theta) = sin(theta) and
        <cos(theta)> = 1/2, so this distinguishes the two.
        """
        bsdf = LambertianBSDF(reflectance_value=1.0)
        n_rays = 200_000
        normal = np.array([0.3, -0.5, 0.8])
        normal /= np.linalg.norm(normal)

        dirs, _ = bsdf.sample(
            n_rays,
            np.tile([0.0, 0.0, 1.0], (n_rays, 1)),
            np.tile(normal, (n_rays, 1)),
            np.full(n_rays, 0.55),
            np.random.default_rng(2),
        )
        cos_theta = np.asarray(dirs) @ normal

        assert cos_theta.mean() == pytest.approx(2.0 / 3.0, abs=5e-3)

        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        density, edges = np.histogram(
            theta, bins=30, range=(0.0, np.pi / 2), density=True
        )
        centres = 0.5 * (edges[:-1] + edges[1:])
        assert np.abs(density - np.sin(2.0 * centres)).max() < 0.05

    def test_reflectance_scales_flux(self):
        """The weight returned is the hemispherical reflectance."""
        bsdf = LambertianBSDF(reflectance_value=0.35)
        n_rays = 128
        _, weights = bsdf.sample(
            n_rays,
            np.tile([0.0, 0.0, 1.0], (n_rays, 1)),
            np.tile([0.0, 0.0, 1.0], (n_rays, 1)),
            np.full(n_rays, 0.55),
            np.random.default_rng(3),
        )
        assert np.allclose(np.asarray(weights), 0.35)


class TestHarveyShackSampling:
    """Harvey-Shack must sample the ABg lobe about the specular direction."""

    def setup_method(self):
        be.set_backend("numpy")

    @staticmethod
    def _sample(bsdf, n_rays=200_000, seed=4):
        """Reflect +z off a -z facing plane, so specular is -z."""
        dirs, weights = bsdf.sample(
            n_rays,
            np.tile([0.0, 0.0, 1.0], (n_rays, 1)),
            np.tile([0.0, 0.0, -1.0], (n_rays, 1)),
            np.full(n_rays, 0.55),
            np.random.default_rng(seed),
        )
        return np.asarray(dirs), np.asarray(weights)

    def test_finite_for_zero_normals_and_directions(self):
        """Degenerate rows must not produce NaN directions or weights."""
        bsdf = HarveyShackBSDF(b0=1e-3, l0=0.01, s=2.0)
        n = 128
        dirs, weights = bsdf.sample(
            n,
            np.zeros((n, 3)),
            np.zeros((n, 3)),
            np.full(n, 0.55),
            np.random.default_rng(0),
        )
        assert np.isfinite(np.asarray(dirs)).all()
        assert np.isfinite(np.asarray(weights)).all()

    def test_polished_surface_stays_near_specular(self):
        """A small break frequency keeps scatter tight about the specular ray.

        The previous sampler drew from a cosine-weighted hemisphere, so the
        median scatter angle was ~45 degrees no matter how polished the
        surface was.
        """
        dirs, _ = self._sample(HarveyShackBSDF(b0=1e-4, l0=0.01, s=2.0))
        angle = np.degrees(np.arccos(np.clip(-dirs[:, 2], -1.0, 1.0)))
        assert np.median(angle) < 10.0, (
            f"Polished surface scatters too widely: median {np.median(angle):.1f} deg"
        )

    def test_rougher_surface_scatters_more_widely(self):
        """Increasing the break frequency widens the lobe."""
        polished, _ = self._sample(HarveyShackBSDF(b0=1e-4, l0=0.01, s=2.0))
        rough, _ = self._sample(HarveyShackBSDF(b0=1e-2, l0=0.05, s=1.5))

        def median_angle(d):
            return np.median(np.degrees(np.arccos(np.clip(-d[:, 2], -1.0, 1.0))))

        assert median_angle(rough) > median_angle(polished)

    def test_radial_distribution_matches_abg(self):
        """The sampled |beta| density must follow BSDF(beta) * 2 * pi * beta."""
        bsdf = HarveyShackBSDF(b0=1e-2, l0=0.05, s=1.5)
        dirs, weights = self._sample(bsdf)
        keep = weights > 0
        # Specular is -z, so beta0 = 0 and |beta| is the transverse magnitude.
        beta = np.hypot(dirs[keep, 0], dirs[keep, 1])

        density, edges = np.histogram(beta, bins=40, range=(0.0, 1.0), density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        theory = bsdf._abg(centres) * 2.0 * np.pi * centres
        theory /= np.trapezoid(theory, centres)

        assert np.abs(density - theory).max() / theory.max() < 0.10

    def test_sampler_conserves_energy(self):
        """Rays keep full flux: the lobe redirects energy, it does not remove it.

        The physical scatter level is applied through ``scatter_fraction``,
        for which ``total_integrated_scatter`` is the natural value.
        """
        bsdf = HarveyShackBSDF(b0=1e-2, l0=0.05, s=1.5)
        _, weights = self._sample(bsdf, n_rays=20_000)
        assert weights.max() == pytest.approx(1.0)
        # Only unreachable samples (outside the hemisphere) are dropped.
        assert weights.mean() > 0.5

    def test_total_integrated_scatter_is_a_fraction(self):
        """TIS is in [0, 1] and grows with the scatter amplitude."""
        faint = HarveyShackBSDF(b0=1e-4, l0=0.01, s=2.0)
        strong = HarveyShackBSDF(b0=1e-2, l0=0.05, s=1.5)

        assert 0.0 <= faint.total_integrated_scatter <= 1.0
        assert 0.0 <= strong.total_integrated_scatter <= 1.0
        assert strong.total_integrated_scatter > faint.total_integrated_scatter


class TestScatterFraction:
    """``SurfaceConfig.scatter_fraction`` mixes scatter with the specular path."""

    def setup_method(self):
        be.set_backend("numpy")

    @staticmethod
    def _trace(fraction):
        """Collimated beam through a lens with a diffuse back face.

        Returns ``(forward_flux, backward_flux)``.
        """
        from optiland.coordinate_system import CoordinateSystem
        from optiland.nonsequential import (
            CollimatedSourceConfig,
            IrradianceDetectorConfig,
            LensConfig,
            NSQScene,
            Spectrum,
            SurfaceConfig,
        )

        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=-80),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=1.0,
                aperture_radius=10.0,
            ),
        )
        scene.add_lens(
            "L",
            CoordinateSystem(z=0),
            LensConfig(
                r1=50,
                r2=-50,
                thickness=5,
                material="N-BK7",
                front_aperture_radius=12.5,
                back=SurfaceConfig(
                    bsdf=LambertianBSDF(reflectance_value=0.9),
                    scatter_fraction=fraction,
                ),
            ),
        )
        scene.add_detector(
            "D_fwd",
            CoordinateSystem(z=100),
            IrradianceDetectorConfig(
                width=60, height=60, num_pixels_x=64, num_pixels_y=64
            ),
        )
        scene.add_detector(
            "D_bwd",
            CoordinateSystem(z=-90),
            IrradianceDetectorConfig(
                width=200, height=200, num_pixels_x=64, num_pixels_y=64
            ),
        )
        result = scene.trace(num_rays=20_000, seed=1)
        return (
            result.detectors["D_fwd"].total_flux,
            result.detectors["D_bwd"].total_flux,
        )

    def test_zero_fraction_leaves_beam_untouched(self):
        """With no scatter the beam transmits as if the BSDF were absent."""
        forward, _ = self._trace(0.0)
        assert forward > 0.85, f"Expected near-full transmission, got {forward:.3f}"

    def test_unit_fraction_is_a_pure_diffuser(self):
        """The default of 1.0 keeps the original behaviour: everything scatters."""
        forward, backward = self._trace(1.0)
        assert forward == pytest.approx(0.0, abs=1e-9)
        assert backward > 0.5

    def test_partial_fraction_splits_the_beam(self):
        """Forward flux falls and back-scatter rises monotonically."""
        results = [self._trace(f) for f in (0.0, 0.2, 0.5, 1.0)]
        forward = [r[0] for r in results]
        backward = [r[1] for r in results]

        assert forward == sorted(forward, reverse=True), forward
        assert backward == sorted(backward), backward
        # A tenth of the way in, most light still gets through.
        assert 0.6 < forward[1] < 0.85
