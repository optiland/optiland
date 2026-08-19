"""Tests for PR13: self-diagnosing SimulationResult and the
ignored-config audit.

Kramer Harrison, 2026
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    AbsorbingComponent,
    CollimatedSourceConfig,
    DoubletConfig,
    FarFieldDetectorConfig,
    IrradianceDetectorConfig,
    LensConfig,
    MirrorConfig,
    NSQScene,
    RayDatabaseConfig,
    SpectralDetectorConfig,
    Spectrum,
    SurfaceConfig,
)
from optiland.nonsequential.backends.numpy_backend import NumpyBackend
from optiland.nonsequential.components.doublet import Doublet
from optiland.nonsequential.components.geometry.analytic.plane import (
    FinitePlaneGeometry,
)
from optiland.nonsequential.components.lens import (
    Lens,
    _make_surface,
    _resolve_interaction,
)
from optiland.nonsequential.components.mirror import Mirror
from optiland.nonsequential.detectors.irradiance import IrradianceDetector
from optiland.nonsequential.detectors.spectral import SpectralDetector
from optiland.nonsequential.diagnostics import DetectorDiagnostic, Diagnostics
from optiland.nonsequential.scene import (
    _build_detector,
    _build_source,
    _resolve_total_flux,
)
from optiland.nonsequential.sources.configs import (
    ExtendedSourceConfig,
    PointSourceConfig,
)

# ---------------------------------------------------------------------------
# Ignored-config audit: every config dataclass field must be
# consumed somewhere in the live lowering path, or the audit fails.
# ---------------------------------------------------------------------------

# (config class, [consumer callables whose source is searched])
_CONFIG_CONSUMERS: list[tuple[type, list[object]]] = [
    (SurfaceConfig, [_make_surface, _resolve_interaction]),
    (LensConfig, [Lens._build]),
    (DoubletConfig, [Doublet._build]),
    (MirrorConfig, [Mirror._build]),
    (PointSourceConfig, [_build_source, _resolve_total_flux]),
    (CollimatedSourceConfig, [_build_source, _resolve_total_flux]),
    (ExtendedSourceConfig, [_build_source, _resolve_total_flux]),
    (IrradianceDetectorConfig, [_build_detector, IrradianceDetector.record]),
    (SpectralDetectorConfig, [_build_detector, SpectralDetector.record]),
    (FarFieldDetectorConfig, [_build_detector]),
    (RayDatabaseConfig, [_build_detector]),
]


@pytest.mark.parametrize(
    "config_cls,consumers",
    _CONFIG_CONSUMERS,
    ids=[c.__name__ for c, _ in _CONFIG_CONSUMERS],
)
def test_every_config_field_is_consumed(config_cls, consumers):
    """Every field of a config dataclass must be read somewhere in its
    lowering path -- a field that is accepted and silently ignored
    is exactly the D-2 class of defect (SurfaceConfig.coating was accepted
    and never read) this audit exists to catch before it ships again.
    """
    combined_source = "\n".join(inspect.getsource(fn) for fn in consumers)
    # A field is "consumed" if it's read via attribute access (`.field`) or
    # via getattr(config, "field", ...) (used for an Optional field so a
    # plain construction call still works) -- either is a real read.
    unconsumed = [
        f.name
        for f in dataclasses.fields(config_cls)
        if f".{f.name}" not in combined_source
        and f'"{f.name}"' not in combined_source
        and f"'{f.name}'" not in combined_source
    ]
    assert not unconsumed, (
        f"{config_cls.__name__} field(s) {unconsumed} are never referenced "
        f"(as '.{{field}}') in {[c.__name__ for c in consumers]} -- accepted "
        "but silently ignored."
    )


# ---------------------------------------------------------------------------
# The generic audit above is satisfied once a field is forwarded into a
# constructor kwarg (e.g. `_build_detector`'s `splat=cfg.splat`) -- exactly
# the shape D-11 (SpectralDetector.splat silently falling back to hard
# binning) shipped in. This checks the stronger claim directly: the field
# must be read somewhere in the detector class's own behaviour, not just
# passed through construction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "detector_cls", [IrradianceDetector, SpectralDetector], ids=lambda c: c.__name__
)
def test_detector_splat_is_read_in_record(detector_cls):
    assert "self.splat" in inspect.getsource(detector_cls.record)


# ---------------------------------------------------------------------------
# Diagnostics -- pure unit tests
# ---------------------------------------------------------------------------


class TestDiagnosticsWarnings:
    def test_clean_trace_has_no_warnings(self):
        diag = Diagnostics(
            depth_truncated_flux_fraction=0.0,
            rr_killed_flux_fraction=0.0,
            flux_conservation_error=0.0,
            unreached_geometry=(),
            detectors=(
                DetectorDiagnostic(
                    name="D1",
                    num_rays_hit=100_000,
                    num_pixels=1024,
                    mean_hits_per_pixel=97.6,
                    undersampled=False,
                    rays_needed_for_5pct=None,
                ),
            ),
            medium_stack_underflows=0,
            split_budget_saturated=False,
        )
        assert diag.warnings() == []
        assert "No warnings." in diag.report()

    def test_depth_truncation_warns_above_threshold(self):
        diag = Diagnostics(depth_truncated_flux_fraction=0.5)
        warns = diag.warnings()
        assert len(warns) == 1
        assert "max_depth" in warns[0]

    def test_depth_truncation_silent_below_threshold(self):
        diag = Diagnostics(depth_truncated_flux_fraction=1e-6)
        assert diag.warnings() == []

    def test_rr_killed_warns_above_threshold(self):
        diag = Diagnostics(rr_killed_flux_fraction=0.5)
        assert any("roulette" in w for w in diag.warnings())

    def test_flux_conservation_warns_above_threshold(self):
        diag = Diagnostics(flux_conservation_error=0.5)
        assert any("flux_conservation_error" in w for w in diag.warnings())

    def test_unreached_geometry_warns(self):
        diag = Diagnostics(unreached_geometry=("L1.edge", "M1.surface"))
        warns = diag.warnings()
        assert len(warns) == 1
        assert "L1.edge" in warns[0]
        assert "M1.surface" in warns[0]

    def test_undersampled_detector_warns(self):
        det = DetectorDiagnostic(
            name="D1",
            num_rays_hit=5,
            num_pixels=1024,
            mean_hits_per_pixel=5 / 1024,
            undersampled=True,
            rays_needed_for_5pct=1_000_000,
        )
        diag = Diagnostics(detectors=(det,))
        warns = diag.warnings()
        assert len(warns) == 1
        assert "D1" in warns[0]
        assert "undersampled" in warns[0]

    def test_split_budget_saturated_warns(self):
        diag = Diagnostics(split_budget_saturated=True)
        assert any("split_budget" in w for w in diag.warnings())

    def test_medium_stack_underflow_warns(self):
        diag = Diagnostics(medium_stack_underflows=3)
        assert any("underflow" in w for w in diag.warnings())

    def test_report_lists_all_warnings(self):
        diag = Diagnostics(
            depth_truncated_flux_fraction=0.5,
            unreached_geometry=("X",),
        )
        report = diag.report()
        assert "Warnings:" in report
        assert report.count("  - ") == 2


class TestRayDatabaseConfigMaxRays:
    """Regression test for a defect the audit above caught live: max_rays
    was accepted by RayDatabaseConfig and never forwarded to
    RayDatabaseDetector, so the circular-buffer limit silently never took
    effect (PR13).
    """

    def test_max_rays_is_forwarded_and_enforced(self):
        scene = NSQScene()
        spec = Spectrum.monochromatic(0.55)
        scene.add_source(
            "S",
            CoordinateSystem(),
            CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=8.0),
        )
        scene.add_detector(
            "RD",
            CoordinateSystem(z=10.0),
            RayDatabaseConfig(width=40, height=40, max_rays=50),
        )
        result = scene.trace(num_rays=2000, seed=1, backend=NumpyBackend(seed=1))
        db = result.detectors["RD"]
        assert len(db.x) <= 50

    def test_max_rays_zero_means_unlimited(self):
        scene = NSQScene()
        spec = Spectrum.monochromatic(0.55)
        scene.add_source(
            "S",
            CoordinateSystem(),
            CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=8.0),
        )
        scene.add_detector(
            "RD",
            CoordinateSystem(z=10.0),
            RayDatabaseConfig(width=40, height=40, max_rays=0),
        )
        result = scene.trace(num_rays=2000, seed=1, backend=NumpyBackend(seed=1))
        db = result.detectors["RD"]
        assert len(db.x) > 50


class TestBuildDiagnostics:
    def _scene(self):
        spec = Spectrum.monochromatic(0.55)
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(),
            CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=8.0),
        )
        scene.add_lens(
            "L1",
            CoordinateSystem(z=50.0),
            LensConfig(
                r1=60.0,
                r2=-60.0,
                thickness=6.0,
                material="N-BK7",
                front_aperture_radius=12.0,
            ),
        )
        scene.add_detector(
            "D1",
            CoordinateSystem(z=200.0),
            IrradianceDetectorConfig(
                width=40, height=40, num_pixels_x=32, num_pixels_y=32
            ),
        )
        return scene

    def test_unreached_geometry_detects_never_hit_component(self):
        scene = self._scene()
        scene.add_component(
            "baffle",
            AbsorbingComponent(
                cs=CoordinateSystem(x=1000, z=50),
                geometry=FinitePlaneGeometry(width=5, height=5),
            ),
        )
        result = scene.trace(num_rays=3000, seed=1, backend=NumpyBackend(seed=1))
        assert any(
            "baffle" in name or "component" in name
            for name in result.diagnostics.unreached_geometry
        )

    def test_depth_truncated_flux_fraction_is_high_at_max_depth_1(self):
        scene = self._scene()
        result = scene.trace(
            num_rays=3000, seed=1, max_depth=1, backend=NumpyBackend(seed=1)
        )
        assert result.diagnostics.depth_truncated_flux_fraction > 0.5

    def test_depth_truncated_flux_fraction_is_zero_at_generous_max_depth(self):
        scene = self._scene()
        result = scene.trace(
            num_rays=3000, seed=1, max_depth=32, backend=NumpyBackend(seed=1)
        )
        assert result.diagnostics.depth_truncated_flux_fraction == pytest.approx(0.0)

    def test_detector_undersampled_at_low_ray_count(self):
        scene = self._scene()
        result = scene.trace(num_rays=200, seed=1, backend=NumpyBackend(seed=1))
        d1 = next(d for d in result.diagnostics.detectors if d.name == "D1")
        assert d1.undersampled

    def test_detector_well_sampled_at_high_ray_count(self):
        scene = self._scene()
        result = scene.trace(num_rays=200_000, seed=1, backend=NumpyBackend(seed=1))
        d1 = next(d for d in result.diagnostics.detectors if d.name == "D1")
        assert not d1.undersampled
        assert d1.mean_hits_per_pixel > 10

    def test_rays_needed_for_5pct_scales_inversely_with_hits(self):
        scene = self._scene()
        result = scene.trace(num_rays=2000, seed=1, backend=NumpyBackend(seed=1))
        d1 = next(d for d in result.diagnostics.detectors if d.name == "D1")
        if d1.mean_hits_per_pixel and d1.mean_hits_per_pixel > 0:
            assert d1.rays_needed_for_5pct > 2000

    def test_diagnostics_flux_conservation_error_matches_result(self):
        scene = self._scene()
        result = scene.trace(num_rays=5000, seed=1, backend=NumpyBackend(seed=1))
        assert result.diagnostics.flux_conservation_error == pytest.approx(
            result.flux_conservation_error
        )

    def test_medium_stack_underflows_zero_for_well_formed_lens(self):
        scene = self._scene()
        result = scene.trace(num_rays=1000, seed=1, backend=NumpyBackend(seed=1))
        assert result.diagnostics.medium_stack_underflows == 0

    def test_medium_stack_underflows_zero_for_cemented_doublet(self):
        """A doublet's cement interface pushes a second, non-nested medium
        onto the stack; exiting to ambient must still fully unwind it."""
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=1.0,
                aperture_radius=8.0,
            ),
        )
        scene.add_doublet(
            "D",
            CoordinateSystem(z=50.0),
            DoubletConfig(
                r1=60.0,
                r2=-60.0,
                r3=-200.0,
                thickness1=6.0,
                thickness2=3.0,
                material1="N-BK7",
                material2="N-SF5",
                aperture_radius=12.0,
            ),
        )
        scene.add_detector(
            "Det",
            CoordinateSystem(z=200.0),
            IrradianceDetectorConfig(
                width=40, height=40, num_pixels_x=32, num_pixels_y=32
            ),
        )
        result = scene.trace(num_rays=1000, seed=1, backend=NumpyBackend(seed=1))
        assert result.diagnostics.medium_stack_underflows == 0

    def test_medium_stack_underflow_detects_leak(self):
        """A bare refractive surface hit from its interior side first (no
        prior entry) is exactly the leak this diagnostic exists to catch.
        """
        import numpy as np

        from optiland.nonsequential.components.geometry.analytic.plane import (
            FinitePlaneGeometry,
        )
        from optiland.nonsequential.components.refractive import (
            RefractiveComponent,
        )
        from optiland.nonsequential.materials.nsq_material import (
            VACUUM,
            NSQMaterial,
        )

        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=10.0, rx=np.pi),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=1.0,
                aperture_radius=4.0,
            ),
        )
        scene.add_component(
            "plate",
            RefractiveComponent(
                cs=CoordinateSystem(z=0.0),
                geometry=FinitePlaneGeometry(width=20, height=20),
                material_front=VACUUM,
                material_back=NSQMaterial.from_glass("N-BK7"),
            ),
        )
        scene.add_detector(
            "Det",
            CoordinateSystem(z=-50.0),
            IrradianceDetectorConfig(
                width=40, height=40, num_pixels_x=16, num_pixels_y=16
            ),
        )
        result = scene.trace(num_rays=1000, seed=1, backend=NumpyBackend(seed=1))
        assert result.diagnostics.medium_stack_underflows > 0

    def test_split_budget_saturated_true_under_tight_budget(self):
        from optiland.nonsequential.ir.scene_ir import SamplingPolicy

        scene = self._scene()
        scene.sampling_policy = SamplingPolicy(split_depth=3, split_budget=1.05)
        result = scene.trace(num_rays=20_000, seed=1, backend=NumpyBackend(seed=1))
        assert result.diagnostics.split_budget_saturated is True

    def test_split_budget_not_saturated_by_default(self):
        scene = self._scene()
        result = scene.trace(num_rays=5000, seed=1, backend=NumpyBackend(seed=1))
        assert result.diagnostics.split_budget_saturated is False


# ---------------------------------------------------------------------------
# SimulationResult.report() / __repr__
# ---------------------------------------------------------------------------


class TestSimulationResultReport:
    def _scene(self):
        spec = Spectrum.monochromatic(0.55)
        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(),
            CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=8.0),
        )
        scene.add_detector(
            "D1",
            CoordinateSystem(z=50.0),
            IrradianceDetectorConfig(
                width=20, height=20, num_pixels_x=16, num_pixels_y=16
            ),
        )
        return scene

    def test_report_returns_string_matching_diagnostics_report(self):
        scene = self._scene()
        result = scene.trace(num_rays=2000, seed=1, backend=NumpyBackend(seed=1))
        assert result.report() == result.diagnostics.report()

    def test_repr_is_concise_and_mentions_warning_count(self):
        scene = self._scene()
        result = scene.trace(
            num_rays=200, seed=1, max_depth=1, backend=NumpyBackend(seed=1)
        )
        r = repr(result)
        assert "SimulationResult(" in r
        assert "diagnostic warning" in r
        # Concise: must not dump full detector objects or arrays.
        assert "IrradianceMap" not in r

    def test_repr_omits_warning_note_when_clean(self):
        scene = self._scene()
        result = scene.trace(num_rays=200_000, seed=1, backend=NumpyBackend(seed=1))
        assert "diagnostic warning" not in repr(result)
