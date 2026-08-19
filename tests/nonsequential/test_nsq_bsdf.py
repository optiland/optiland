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
from optiland.nonsequential import (
    HarveyShackBSDF,
    LambertianBSDF,
    NSQRng,
    SpecularBRDF,
    TabulatedBSDF,
)
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
            dirs, _, _ = bsdf.sample(
                n,
                np.tile([0.0, 0.0, 1.0], (n, 1)),
                np.zeros((n, 3)),
                np.full(n, 0.55),
                NSQRng(0),
                np.arange(n),
                np.zeros(n, dtype=np.int32),
            )
        assert np.isfinite(np.asarray(dirs)).all()

    def test_directions_are_unit_and_in_hemisphere(self):
        """Every scattered ray is a unit vector in the normal's hemisphere."""
        bsdf = LambertianBSDF(reflectance_value=1.0)
        n_rays = 20_000
        normal = np.array([0.3, -0.5, 0.8])
        normal /= np.linalg.norm(normal)

        dirs, _, _ = bsdf.sample(
            n_rays,
            np.tile([0.0, 0.0, 1.0], (n_rays, 1)),
            np.tile(normal, (n_rays, 1)),
            np.full(n_rays, 0.55),
            NSQRng(1),
            np.arange(n_rays),
            np.zeros(n_rays, dtype=np.int32),
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

        dirs, _, _ = bsdf.sample(
            n_rays,
            np.tile([0.0, 0.0, 1.0], (n_rays, 1)),
            np.tile(normal, (n_rays, 1)),
            np.full(n_rays, 0.55),
            NSQRng(2),
            np.arange(n_rays),
            np.zeros(n_rays, dtype=np.int32),
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
        _, weights, _ = bsdf.sample(
            n_rays,
            np.tile([0.0, 0.0, 1.0], (n_rays, 1)),
            np.tile([0.0, 0.0, 1.0], (n_rays, 1)),
            np.full(n_rays, 0.55),
            NSQRng(3),
            np.arange(n_rays),
            np.zeros(n_rays, dtype=np.int32),
        )
        assert np.allclose(np.asarray(weights), 0.35)


class TestHarveyShackSampling:
    """Harvey-Shack must sample the ABg lobe about the specular direction."""

    def setup_method(self):
        be.set_backend("numpy")

    @staticmethod
    def _sample(bsdf, n_rays=200_000, seed=4):
        """Reflect +z off a -z facing plane, so specular is -z."""
        dirs, weights, _ = bsdf.sample(
            n_rays,
            np.tile([0.0, 0.0, 1.0], (n_rays, 1)),
            np.tile([0.0, 0.0, -1.0], (n_rays, 1)),
            np.full(n_rays, 0.55),
            NSQRng(seed),
            np.arange(n_rays),
            np.zeros(n_rays, dtype=np.int32),
        )
        return np.asarray(dirs), np.asarray(weights)

    def test_finite_for_zero_normals_and_directions(self):
        """Degenerate rows must not produce NaN directions or weights."""
        bsdf = HarveyShackBSDF(b0=1e-3, l0=0.01, s=2.0)
        n = 128
        dirs, weights, _ = bsdf.sample(
            n,
            np.zeros((n, 3)),
            np.zeros((n, 3)),
            np.full(n, 0.55),
            NSQRng(0),
            np.arange(n),
            np.zeros(n, dtype=np.int32),
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
        """The default of 1.0 keeps the original behaviour: everything scatters.

        The exact backward fraction shifted down with the D-1 geometric
        sidedness fix: index-proximity sidedness happened to keep a
        back-scattered ray's bookkept medium at glass (n1 == n2, since
        n_current already equalled material_front exactly), so the front
        face saw a ray "in glass" and mostly refracted it straight out.
        Geometric sidedness instead determines the front face's n1/n2 from
        direction and n_geom alone, which correctly identifies many steeply
        Lambertian-scattered rays as exceeding the front face's critical
        angle and traps them by TIR -- a real effect the index heuristic
        was masking. This depends on D-4 (BSDF-vs-medium-state disagreement)
        being fixed: a scattered ray at the back face must reach the front
        face bookkept as "in glass" every time, not depending on whichever
        way the back face's own, unrelated Fresnel branch die roll happened
        to land (see TestBsdfLobeMediumTracking for a direct test of that).
        """
        forward, backward = self._trace(1.0)
        assert forward == pytest.approx(0.0, abs=1e-9)
        assert backward > 0.3

    def test_partial_fraction_splits_the_beam(self):
        """Forward flux falls and back-scatter rises monotonically."""
        results = [self._trace(f) for f in (0.0, 0.2, 0.5, 1.0)]
        forward = [r[0] for r in results]
        backward = [r[1] for r in results]

        assert forward == sorted(forward, reverse=True), forward
        assert backward == sorted(backward), backward
        # A tenth of the way in, most light still gets through.
        assert 0.6 < forward[1] < 0.85


class TestTransmissiveLobes:
    """D-5: BSDF lobes can sample the transmissive (far-side) hemisphere."""

    def setup_method(self):
        be.set_backend("numpy")

    def test_zero_fraction_is_all_reflective(self):
        bsdf = LambertianBSDF(reflectance_value=1.0, transmissive_fraction=0.0)
        n_rays = 5_000
        _, _, transmitted = bsdf.sample(
            n_rays,
            np.tile([0.0, 0.0, -1.0], (n_rays, 1)),
            np.tile([0.0, 0.0, 1.0], (n_rays, 1)),
            np.full(n_rays, 0.55),
            NSQRng(0),
            np.arange(n_rays),
            np.zeros(n_rays, dtype=np.int32),
        )
        assert not np.any(np.asarray(transmitted))

    def test_lambertian_transmissive_fraction_splits_hemispheres(self):
        """~half the rays land in each hemisphere at fraction=0.5, and the
        returned ``transmitted`` mask agrees with which hemisphere each
        scattered ray actually landed in."""
        bsdf = LambertianBSDF(reflectance_value=1.0, transmissive_fraction=0.5)
        n_rays = 20_000
        normal = np.array([0.0, 0.0, 1.0])
        dirs, _, transmitted = bsdf.sample(
            n_rays,
            np.tile([0.0, 0.0, -1.0], (n_rays, 1)),
            np.tile(normal, (n_rays, 1)),
            np.full(n_rays, 0.55),
            NSQRng(1),
            np.arange(n_rays),
            np.zeros(n_rays, dtype=np.int32),
        )
        dirs = np.asarray(dirs)
        transmitted = np.asarray(transmitted)

        frac = transmitted.mean()
        assert frac == pytest.approx(0.5, abs=0.02)
        # transmitted rays are in the -normal hemisphere, reflected in +normal
        assert np.all(dirs[transmitted] @ normal < 1e-9)
        assert np.all(dirs[~transmitted] @ normal > -1e-9)

    def test_harvey_shack_transmissive_lobe_blurs_the_straight_through_ray(self):
        """At transmissive_fraction=1.0, the lobe blurs the undeviated
        (straight-through) ray rather than the specular reflection."""
        bsdf = HarveyShackBSDF(b0=1e-4, l0=0.01, s=2.0, transmissive_fraction=1.0)
        n_rays = 2_000
        incident = np.tile([0.0, 0.0, 1.0], (n_rays, 1))  # travelling +z
        normal = np.tile([0.0, 0.0, -1.0], (n_rays, 1))  # facing the ray
        dirs, weights, transmitted = bsdf.sample(
            n_rays,
            incident,
            normal,
            np.full(n_rays, 0.55),
            NSQRng(4),
            np.arange(n_rays),
            np.zeros(n_rays, dtype=np.int32),
        )
        dirs = np.asarray(dirs)
        assert np.all(np.asarray(transmitted))
        # Blurred around +z (straight through), not -z (specular reflection).
        assert np.median(dirs[:, 2]) > 0.9

    def test_specular_brdf_never_transmits(self):
        bsdf = SpecularBRDF()
        n_rays = 100
        _, _, transmitted = bsdf.sample(
            n_rays,
            np.tile([0.0, 0.0, -1.0], (n_rays, 1)),
            np.tile([0.0, 0.0, 1.0], (n_rays, 1)),
            np.full(n_rays, 0.55),
        )
        assert not np.any(np.asarray(transmitted))

    def test_tabulated_transmissive_fraction_splits_hemispheres(self, tmp_path):
        data = tmp_path / "scatter.csv"
        data.write_text(
            "0,0,0.3\n0,45,0.2\n0,90,0.05\n"
            "45,0,0.25\n45,45,0.2\n45,90,0.05\n"
            "90,0,0.1\n90,45,0.08\n90,90,0.02\n"
        )
        bsdf = TabulatedBSDF(data, transmissive_fraction=0.5)
        n_rays = 20_000
        normal = np.array([0.0, 0.0, 1.0])
        dirs, _, transmitted = bsdf.sample(
            n_rays,
            np.tile([0.0, 0.0, -1.0], (n_rays, 1)),
            np.tile(normal, (n_rays, 1)),
            np.full(n_rays, 0.55),
            NSQRng(2),
            np.arange(n_rays),
            np.zeros(n_rays, dtype=np.int32),
        )
        dirs = np.asarray(dirs)
        transmitted = np.asarray(transmitted)
        assert transmitted.mean() == pytest.approx(0.5, abs=0.02)
        assert np.all(dirs[transmitted] @ normal < 1e-9)
        assert np.all(dirs[~transmitted] @ normal > -1e-9)


class TestBsdfLobeMediumTracking:
    """D-4: a scattered ray's medium is decided by the BSDF lobe's own
    reflect/transmit side, not by the independent Fresnel branch draw."""

    def setup_method(self):
        be.set_backend("numpy")

    @staticmethod
    def _hit_glass_interface(bsdf):
        """A flat VACUUM|N-BK7 interface hit by a normal-incidence beam.

        Every ray scatters (scatter_fraction=1.0), so with D-4 fixed
        ``rays.n_current`` after ``interact()`` is fully determined by the
        BSDF's own reflect/transmit draw, never by the Fresnel branch.
        """
        from optiland.coordinate_system import CoordinateSystem
        from optiland.nonsequential import VACUUM, NSQMaterial, RefractiveComponent
        from optiland.nonsequential.components.geometry.analytic.plane import (
            PlaneGeometry,
        )
        from optiland.nonsequential.ir.bsdf_ir import BsdfIR
        from optiland.nonsequential.ray_bundle import NSQRayBundle

        glass = NSQMaterial.from_glass("N-BK7")
        comp = RefractiveComponent(
            cs=CoordinateSystem(z=0.0),
            geometry=PlaneGeometry(),
            material_front=VACUUM,
            material_back=glass,
            bsdf=bsdf,
            scatter_fraction=1.0,
            name="I",
        )
        n = 4_000
        rays = NSQRayBundle(
            x=np.zeros(n),
            y=np.zeros(n),
            z=np.full(n, -1.0),
            L=np.zeros(n),
            M=np.zeros(n),
            N=np.ones(n),
            flux=np.ones(n),
            wavelength=np.full(n, 0.55),
            n_current=np.ones(n),
            bounce=np.zeros(n, dtype=np.int32),
            alive=np.ones(n, dtype=bool),
            ray_id=np.arange(n, dtype=np.int64),
        )
        t = np.ones(n)
        normals = np.tile([0.0, 0.0, -1.0], (n, 1))
        n_geom = np.tile([0.0, 0.0, 1.0], (n, 1))
        hit_mask = np.ones(n, dtype=bool)
        bsdf_ir = BsdfIR(kind="lambertian", params={})
        comp.interact(rays, t, normals, hit_mask, NSQRng(5), bsdf_ir, n_geom)
        glass_n = float(np.asarray(glass.n(np.array([0.55]))).ravel()[0])
        return rays, glass_n

    def test_pure_reflective_lobe_always_stays_in_incident_medium(self):
        """Every ray scatters into the reflective (incident-side) hemisphere,
        so n_current must be vacuum for every single one -- before D-4, the
        independent Fresnel branch could leave roughly half of them
        incorrectly bookkept as having entered the glass."""
        bsdf = LambertianBSDF(reflectance_value=1.0, transmissive_fraction=0.0)
        rays, _glass_n = self._hit_glass_interface(bsdf)
        np.testing.assert_allclose(rays.n_current, 1.0)
        np.testing.assert_allclose(rays.k_current, 0.0)

    def test_transmissive_lobe_updates_medium_to_the_far_side(self):
        """A mixed reflect/transmit lobe: n_current must exactly match each
        ray's own lobe choice, not an independent coin flip."""
        bsdf = LambertianBSDF(reflectance_value=1.0, transmissive_fraction=0.5)
        rays, glass_n = self._hit_glass_interface(bsdf)

        entered_glass = rays.n_current > 1.0 + 1e-9
        frac = entered_glass.mean()
        assert 0.3 < frac < 0.7, "Expected a genuine ~50/50 split"
        np.testing.assert_allclose(rays.n_current[entered_glass], glass_n)
        np.testing.assert_allclose(rays.n_current[~entered_glass], 1.0)
