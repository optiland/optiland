"""Tests for NSQ scene round-trip serialization (PR10).

Covers:
- Lens + CollimatedSource + IrradianceDetector round-trip with trace comparison
- Mirror + PointSource + FarFieldDetector round-trip structural check
- Schema version written to JSON
- Tensors stored as plain floats (no requires_grad)
- Unknown schema version raises ValueError with actionable message
- Missing schema version key raises ValueError

Kramer Harrison, 2026
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    FarFieldDetectorConfig,
    IrradianceDetectorConfig,
    LensConfig,
    MirrorConfig,
    NSQScene,
    PointSourceConfig,
    Spectrum,
)
from optiland.nonsequential.serialization import (
    NSQ_SCHEMA_VERSION,
    scene_from_dict,
    scene_to_dict,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lens_scene() -> NSQScene:
    """Build a minimal scene: collimated source -> plano-convex lens -> detector."""
    scene = NSQScene()

    # Collimated source along +z, 5 mm aperture, 550 nm monochromatic
    src_cs = CoordinateSystem(x=0, y=0, z=0)
    spec = Spectrum.monochromatic(0.55)
    scene.add_source(
        "S1",
        src_cs,
        CollimatedSourceConfig(spectrum=spec, total_flux=1.0, aperture_radius=5.0),
    )

    # Plano-convex N-BK7 lens at z=20 mm, r1=50 mm, plano back, 5 mm thick
    lens_cs = CoordinateSystem(x=0, y=0, z=20.0)
    scene.add_lens(
        "L1",
        lens_cs,
        LensConfig(
            r1=50.0,
            r2=0.0,
            thickness=5.0,
            material="N-BK7",
            front_aperture_radius=6.0,
        ),
    )

    # Irradiance detector at z=120 mm (near focus)
    det_cs = CoordinateSystem(x=0, y=0, z=120.0)
    scene.add_detector(
        "D1",
        det_cs,
        IrradianceDetectorConfig(
            width=20.0,
            height=20.0,
            num_pixels_x=64,
            num_pixels_y=64,
            splat="hard",
        ),
    )
    return scene


def _make_mirror_scene() -> NSQScene:
    """Build a scene: point source -> concave mirror -> far-field detector."""
    scene = NSQScene()

    src_cs = CoordinateSystem(x=0, y=0, z=0)
    spec = Spectrum.monochromatic(0.55)
    scene.add_source(
        "PS1",
        src_cs,
        PointSourceConfig(spectrum=spec, total_flux=1.0, half_angle_deg=10.0),
    )

    mirror_cs = CoordinateSystem(x=0, y=0, z=50.0)
    scene.add_mirror(
        "M1",
        mirror_cs,
        MirrorConfig(radius=-100.0, reflectance=1.0, conic=-1.0, aperture_radius=20.0),
    )

    ff_cs = CoordinateSystem(x=0, y=0, z=0)
    scene.add_detector(
        "FF1",
        ff_cs,
        FarFieldDetectorConfig(num_theta=45, num_phi=180),
    )
    return scene


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSchemaVersion:
    def test_schema_version_in_json(self, tmp_path):
        """nsq_schema_version must be present and equal to NSQ_SCHEMA_VERSION."""
        scene = _make_lens_scene()
        path = tmp_path / "scene.json"
        scene.to_json(path)

        with path.open() as f:
            d = json.load(f)

        assert "nsq_schema_version" in d
        assert d["nsq_schema_version"] == NSQ_SCHEMA_VERSION

    def test_unknown_schema_version_raises(self):
        """Loading a dict with a future schema version must raise ValueError."""
        d = scene_to_dict(_make_lens_scene())
        d["nsq_schema_version"] = NSQ_SCHEMA_VERSION + 99

        with pytest.raises(ValueError, match="schema version"):
            scene_from_dict(d)

    def test_missing_schema_version_raises(self):
        """Loading a dict without nsq_schema_version must raise ValueError."""
        d = scene_to_dict(_make_lens_scene())
        del d["nsq_schema_version"]

        with pytest.raises(ValueError, match="nsq_schema_version"):
            scene_from_dict(d)

    def test_file_not_found_raises(self, tmp_path):
        """from_json on a non-existent path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            NSQScene.from_json(tmp_path / "does_not_exist.json")


class TestTensorSerialization:
    def test_values_stored_as_plain_floats(self, tmp_path):
        """All numeric values in the JSON must be plain floats, not tensors."""
        scene = _make_lens_scene()
        path = tmp_path / "scene.json"
        scene.to_json(path)

        with path.open() as f:
            raw = f.read()

        # Sanity: the JSON should not contain tensor-specific strings
        assert "requires_grad" not in raw
        assert "torch" not in raw

        d = json.load(path.open())
        # Check a component config value is a plain float
        lens_cfg = d["components"][0]["config"]
        assert isinstance(lens_cfg["r1"], float)
        assert isinstance(lens_cfg["thickness"], float)

        # Check source total_flux is a plain float
        src = d["sources"][0]
        assert isinstance(src["total_flux"], float)


class TestLensSceneRoundTrip:
    def test_structure_preserved(self, tmp_path):
        """Registry names and counts must be preserved after round-trip."""
        scene = _make_lens_scene()
        path = tmp_path / "lens_scene.json"
        scene.to_json(path)
        loaded = NSQScene.from_json(path)

        assert "L1" in loaded.component_registry
        assert "S1" in loaded.source_registry
        assert "D1" in loaded.detector_registry

        assert len(loaded.component_registry.compounds) == 1
        assert len(loaded.source_registry.sources) == 1
        assert len(loaded.detector_registry.detectors) == 1

    def test_component_config_values(self, tmp_path):
        """Lens config values must survive round-trip with float precision."""
        scene = _make_lens_scene()
        path = tmp_path / "lens_scene.json"
        scene.to_json(path)
        loaded = NSQScene.from_json(path)

        orig_comp = scene.component_registry.get("L1")
        load_comp = loaded.component_registry.get("L1")

        assert math.isclose(orig_comp._config.r1, load_comp._config.r1)
        assert math.isclose(orig_comp._config.thickness, load_comp._config.thickness)
        assert orig_comp._config.material == load_comp._config.material

    def test_coordinate_system_values(self, tmp_path):
        """Coordinate system values must survive round-trip."""
        scene = _make_lens_scene()
        path = tmp_path / "lens_scene.json"
        scene.to_json(path)
        loaded = NSQScene.from_json(path)

        orig_cs = scene.component_registry.get("L1")._cs
        load_cs = loaded.component_registry.get("L1")._cs

        assert math.isclose(float(orig_cs.z), float(load_cs.z))

    def test_detector_config_values(self, tmp_path):
        """Detector pixel counts and dimensions must survive round-trip."""
        scene = _make_lens_scene()
        path = tmp_path / "lens_scene.json"
        scene.to_json(path)
        loaded = NSQScene.from_json(path)

        from optiland.nonsequential.detectors.irradiance import (  # noqa: PLC0415
            IrradianceDetector,
        )

        det = loaded.detector_registry.get("D1")
        assert isinstance(det, IrradianceDetector)
        assert det.num_pixels_x == 64
        assert det.num_pixels_y == 64
        assert math.isclose(det.width, 20.0)

    def test_trace_flux_close_after_roundtrip(self, tmp_path):
        """Total detected flux must agree within 5% relative after round-trip.

        Uses seed=42 for reproducibility.  Monte Carlo noise is bounded by
        sqrt(N)/N ~ 1/sqrt(N), which for N=50_000 is ~0.45%, well within 5%.
        """
        scene = _make_lens_scene()

        result_orig = scene.trace(num_rays=50_000, seed=42)

        path = tmp_path / "lens_scene.json"
        scene.to_json(path)
        loaded = NSQScene.from_json(path)

        result_loaded = loaded.trace(num_rays=50_000, seed=42)

        flux_orig = result_orig.detectors["D1"].total_flux
        flux_loaded = result_loaded.detectors["D1"].total_flux

        if flux_orig > 0:
            rel_diff = abs(flux_orig - flux_loaded) / flux_orig
            assert rel_diff < 0.05, (
                f"Relative flux difference {rel_diff:.3%} exceeds 5% tolerance. "
                f"orig={flux_orig:.4f}, loaded={flux_loaded:.4f}"
            )


class TestMirrorSceneRoundTrip:
    def test_structure_preserved(self, tmp_path):
        """Mirror + PointSource + FarFieldDetector must round-trip correctly."""
        scene = _make_mirror_scene()
        path = tmp_path / "mirror_scene.json"
        scene.to_json(path)
        loaded = NSQScene.from_json(path)

        assert "M1" in loaded.component_registry
        assert "PS1" in loaded.source_registry
        assert "FF1" in loaded.detector_registry

    def test_mirror_config_values(self, tmp_path):
        """Mirror radius and conic must survive round-trip."""
        scene = _make_mirror_scene()
        path = tmp_path / "mirror_scene.json"
        scene.to_json(path)
        loaded = NSQScene.from_json(path)

        orig_cfg = scene.component_registry.get("M1")._config
        load_cfg = loaded.component_registry.get("M1")._config

        assert math.isclose(orig_cfg.radius, load_cfg.radius)
        assert math.isclose(orig_cfg.conic, load_cfg.conic)
        assert math.isclose(orig_cfg.aperture_radius, load_cfg.aperture_radius)

    def test_source_config_values(self, tmp_path):
        """PointSource half_angle_deg must survive round-trip."""
        scene = _make_mirror_scene()
        path = tmp_path / "mirror_scene.json"
        scene.to_json(path)
        loaded = NSQScene.from_json(path)

        from optiland.nonsequential.sources.point import PointSource  # noqa: PLC0415

        src = loaded.source_registry.get("PS1")
        assert isinstance(src, PointSource)
        assert math.isclose(src.half_angle_deg, 10.0)

    def test_far_field_detector_bins(self, tmp_path):
        """FarFieldDetector bin counts must survive round-trip."""
        scene = _make_mirror_scene()
        path = tmp_path / "mirror_scene.json"
        scene.to_json(path)
        loaded = NSQScene.from_json(path)

        from optiland.nonsequential.detectors.far_field import (  # noqa: PLC0415
            FarFieldDetector,
        )

        det = loaded.detector_registry.get("FF1")
        assert isinstance(det, FarFieldDetector)
        assert det.num_bins_theta == 45
        assert det.num_bins_phi == 180


class TestSceneToDictFromDict:
    def test_roundtrip_via_dict(self):
        """scene_to_dict -> scene_from_dict must reconstruct correctly."""
        scene = _make_lens_scene()
        d = scene_to_dict(scene)
        loaded = scene_from_dict(d)

        assert "L1" in loaded.component_registry
        assert "S1" in loaded.source_registry
        assert "D1" in loaded.detector_registry

    def test_dict_has_expected_keys(self):
        """scene_to_dict output must have top-level structure keys."""
        d = scene_to_dict(_make_lens_scene())
        assert "nsq_schema_version" in d
        assert "components" in d
        assert "sources" in d
        assert "detectors" in d

    def test_spectrum_wavelengths_match(self):
        """Spectrum wavelengths must survive the dict round-trip."""
        scene = _make_lens_scene()
        d = scene_to_dict(scene)
        loaded = scene_from_dict(d)

        orig_wls = scene.source_registry.get("S1").spectrum.wavelengths
        load_wls = loaded.source_registry.get("S1").spectrum.wavelengths

        np.testing.assert_allclose(orig_wls, load_wls)

    def test_empty_components_scene(self, tmp_path):
        """A scene with no components but sources+detectors must round-trip."""
        scene = NSQScene()
        src_cs = CoordinateSystem()
        spec = Spectrum.monochromatic(0.55)
        scene.add_source("S1", src_cs, PointSourceConfig(spectrum=spec))
        det_cs = CoordinateSystem(z=100.0)
        scene.add_detector(
            "D1",
            det_cs,
            IrradianceDetectorConfig(width=10.0, height=10.0),
        )

        path = tmp_path / "min_scene.json"
        scene.to_json(path)
        loaded = NSQScene.from_json(path)

        assert len(loaded.component_registry.compounds) == 0
        assert "S1" in loaded.source_registry
        assert "D1" in loaded.detector_registry
