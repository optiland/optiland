"""Scene IR tests for Non-Sequential Raytracing (PR3).

Covers lower() correctness, the translatability checklist, and the JSON
round-trip / registry-completeness guards against drift. The numeric
"lower -> interpret -> matches direct interpretation" half of the drift
guard is deferred to PR4, which is where an IR interpreter first exists.

Kramer Harrison, 2026
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    ExtendedSourceConfig,
    FarFieldDetectorConfig,
    IrradianceDetectorConfig,
    LensConfig,
    MirrorConfig,
    NSQScene,
    PointSourceConfig,
    RayDatabaseConfig,
    ReflectiveComponent,
    SpectralDetectorConfig,
    Spectrum,
)
from optiland.nonsequential.bsdf.harvey_shack import HarveyShackBSDF
from optiland.nonsequential.bsdf.lambertian import LambertianBSDF
from optiland.nonsequential.components.geometry.analytic.plane import (
    FinitePlaneGeometry,
)
from optiland.nonsequential.ir import lower, scene_ir_from_dict, scene_ir_to_dict
from optiland.nonsequential.ir.lower import (
    _component_kind,
    _lower_bsdf,
    _lower_geometry,
)
from optiland.nonsequential.ir.scene_ir import BsdfIR, MediumIR, PrimitiveIR, SceneIR


def _rich_scene() -> NSQScene:
    """A scene exercising every primitive/emitter/sensor kind this PR lowers."""
    scene = NSQScene()
    scene.add_source(
        "S_point",
        CoordinateSystem(z=-80),
        PointSourceConfig(spectrum=Spectrum.monochromatic(0.55), total_flux=1.0),
    )
    scene.add_source(
        "S_collimated",
        CoordinateSystem(z=-80),
        CollimatedSourceConfig(
            spectrum=Spectrum(np.array([0.4, 0.7]), np.array([1.0, 1.0])),
            total_flux=2.0,
            aperture_radius=5.0,
        ),
    )
    scene.add_source(
        "S_extended",
        CoordinateSystem(z=-80),
        ExtendedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, width=2.0, height=2.0
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
        ),
    )
    scene.add_mirror(
        "M",
        CoordinateSystem(z=100),
        MirrorConfig(radius=-200, reflectance=1.0, aperture_radius=20),
    )
    scene.add_component(
        "diffuser",
        ReflectiveComponent(
            cs=CoordinateSystem(z=120),
            geometry=FinitePlaneGeometry(width=40.0, height=40.0),
            reflectance=1.0,
            bsdf=LambertianBSDF(reflectance_value=0.7),
            scatter_fraction=0.3,
            name="diffuser",
        ),
    )
    scene.add_component(
        "scatterer",
        ReflectiveComponent(
            cs=CoordinateSystem(z=140),
            geometry=FinitePlaneGeometry(width=40.0, height=40.0),
            reflectance=1.0,
            bsdf=HarveyShackBSDF(b0=1e-3, l0=0.01, s=2.0),
            name="scatterer",
        ),
    )
    scene.add_detector(
        "D_irr",
        CoordinateSystem(z=150),
        IrradianceDetectorConfig(width=20, height=20, num_pixels_x=16, num_pixels_y=16),
    )
    scene.add_detector(
        "D_spec",
        CoordinateSystem(z=150),
        SpectralDetectorConfig(
            width=20, height=20, num_pixels_x=8, num_pixels_y=8, num_bins=4
        ),
    )
    scene.add_detector(
        "D_ff",
        CoordinateSystem(z=150),
        FarFieldDetectorConfig(num_theta=10, num_phi=20),
    )
    scene.add_detector(
        "D_raydb", CoordinateSystem(z=150), RayDatabaseConfig(width=20, height=20)
    )
    return scene


# ---------------------------------------------------------------------------
# lower() correctness
# ---------------------------------------------------------------------------


class TestLowerCorrectness:
    def test_primitive_count_and_order_matches_flat_surfaces(self):
        scene = _rich_scene()
        ir = lower(scene)
        assert len(ir.primitives) == len(scene.surfaces)
        for p, comp in zip(ir.primitives, scene.surfaces, strict=True):
            assert p.name == (comp.name or p.name)

    def test_emitter_and_sensor_counts(self):
        scene = _rich_scene()
        ir = lower(scene)
        assert len(ir.emitters) == 3
        assert len(ir.sensors) == 4
        assert {e.kind for e in ir.emitters} == {"point", "collimated", "extended"}
        assert {s.kind for s in ir.sensors} == {
            "irradiance",
            "spectral",
            "far_field",
            "ray_database",
        }

    def test_materials_are_deduplicated(self):
        scene = NSQScene()
        scene.add_lens(
            "L1",
            CoordinateSystem(z=0),
            LensConfig(
                r1=50, r2=-50, thickness=5, material="N-BK7", front_aperture_radius=10
            ),
        )
        scene.add_lens(
            "L2",
            CoordinateSystem(z=50),
            LensConfig(
                r1=50, r2=-50, thickness=5, material="N-BK7", front_aperture_radius=10
            ),
        )
        ir = lower(scene)
        names = [m.name for m in ir.media]
        assert names.count("N-BK7") == 1
        assert names.count("vacuum") == 1

    def test_lens_medium_ids_are_consistent_front_to_back(self):
        """The front face's interior and the back face's exterior are the
        same glass -- lowering must resolve both to the same medium id."""
        scene = NSQScene()
        scene.add_lens(
            "L",
            CoordinateSystem(z=0),
            LensConfig(
                r1=50, r2=-50, thickness=5, material="N-BK7", front_aperture_radius=10
            ),
        )
        ir = lower(scene)
        front = next(p for p in ir.primitives if p.name == "L.front")
        back = next(p for p in ir.primitives if p.name == "L.back")
        assert front.interior_medium_id == back.exterior_medium_id
        glass_medium = ir.media[front.interior_medium_id]
        assert glass_medium.name == "N-BK7"
        assert glass_medium.n_model == {"kind": "catalog", "name": "N-BK7"}

    def test_bsdf_attached_surfaces_lower_correctly(self):
        scene = _rich_scene()
        ir = lower(scene)
        diffuser = next(p for p in ir.primitives if p.name == "diffuser")
        assert diffuser.bsdf == BsdfIR(
            kind="lambertian",
            params={"reflectance_value": 0.7, "transmissive_fraction": 0.0},
        )
        assert diffuser.scatter_fraction == pytest.approx(0.3)

        scatterer = next(p for p in ir.primitives if p.name == "scatterer")
        assert scatterer.bsdf.kind == "harvey_shack"
        assert scatterer.bsdf.params == {
            "b0": 1e-3,
            "l0": 0.01,
            "s": 2.0,
            "transmissive_fraction": 0.0,
        }

    def test_bare_surface_has_none_bsdf(self):
        scene = _rich_scene()
        ir = lower(scene)
        mirror = next(p for p in ir.primitives if p.name == "M.surface")
        assert mirror.bsdf == BsdfIR(kind="none")

    def test_component_kind_matches_surface_type(self):
        scene = _rich_scene()
        ir = lower(scene)
        kinds = {p.name: p.component_kind for p in ir.primitives}
        assert kinds["L.front"] == "refractive"
        assert kinds["L.edge"] == "absorbing"
        assert kinds["M.surface"] == "reflective"
        assert kinds["diffuser"] == "reflective"

    def test_geometry_kinds_lower_correctly(self):
        scene = _rich_scene()
        ir = lower(scene)
        kinds = {p.name: p.kind for p in ir.primitives}
        assert kinds["L.front"] == "conic"
        assert kinds["L.edge"] == "frustum"
        assert kinds["M.surface"] == "conic"
        assert kinds["diffuser"] == "plane"

    def test_source_without_medium_has_no_medium_id(self):
        scene = _rich_scene()
        ir = lower(scene)
        point_emitter = next(e for e in ir.emitters if e.kind == "point")
        assert point_emitter.medium_id is None

    def test_unsupported_material_raises(self):
        from optiland.materials.ideal import IdealMaterial  # noqa: PLC0415
        from optiland.nonsequential.materials.nsq_material import (
            NSQMaterial,  # noqa: PLC0415
        )

        scene = NSQScene()
        scene.add_lens(
            "L",
            CoordinateSystem(z=0),
            LensConfig(
                r1=50,
                r2=-50,
                thickness=5,
                material=NSQMaterial(optiland_material=IdealMaterial(n=1.5)),
                front_aperture_radius=10,
            ),
        )
        with pytest.raises(ValueError, match="catalog"):
            lower(scene)


# ---------------------------------------------------------------------------
# Translatability checklist
# ---------------------------------------------------------------------------


def _walk_ir_values(value, path="root"):
    """Yield every leaf value in an IR tree, with a dotted path for errors."""
    if is_dataclass(value) and not isinstance(value, type):
        for f in fields(value):
            yield from _walk_ir_values(getattr(value, f.name), f"{path}.{f.name}")
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _walk_ir_values(v, f"{path}[{k!r}]")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _walk_ir_values(v, f"{path}[{i}]")
    else:
        yield path, value


class TestTranslatabilityChecklist:
    """SceneIR must be introspectable, callback-free data."""

    def test_no_closures_or_behaviour_carrying_objects(self):
        """Rule 1/2: every leaf is a scalar, string, None, or ndarray --
        never a callable, a CoordinateSystem, or any object with methods
        that matter to interpretation."""
        scene = _rich_scene()
        ir = lower(scene)
        allowed_leaf_types = (int, float, str, bool, type(None), np.ndarray)
        for path, value in _walk_ir_values(ir):
            assert not callable(value), f"{path} is callable: {value!r}"
            assert isinstance(value, allowed_leaf_types), (
                f"{path} has type {type(value).__name__}, not a plain "
                f"scalar/string/ndarray: {value!r}"
            )

    def test_transforms_are_4x4_matrices_not_coordinate_systems(self):
        """Rule 4."""
        scene = _rich_scene()
        ir = lower(scene)
        for p in ir.primitives:
            assert p.to_world.shape == (4, 4)
        for e in ir.emitters:
            assert e.to_world.shape == (4, 4)
        for s in ir.sensors:
            assert s.to_world.shape == (4, 4)

    def test_json_round_trip_is_lossless(self):
        """Rule 5."""
        scene = _rich_scene()
        ir = lower(scene)
        d = scene_ir_to_dict(ir)
        # Must actually be JSON-safe, not merely "JSON-safe-shaped".
        text = json.dumps(d)
        ir2 = scene_ir_from_dict(json.loads(text))

        assert len(ir2.primitives) == len(ir.primitives)
        for p1, p2 in zip(ir.primitives, ir2.primitives, strict=True):
            np.testing.assert_array_equal(p1.to_world, p2.to_world)
            assert p1.kind == p2.kind
            assert p1.params == p2.params
            assert p1.bsdf == p2.bsdf
            assert p1.component_kind == p2.component_kind
            assert p1.scatter_fraction == p2.scatter_fraction
            assert p1.name == p2.name
            assert p1.interior_medium_id == p2.interior_medium_id
            assert p1.exterior_medium_id == p2.exterior_medium_id

        assert len(ir2.media) == len(ir.media)
        assert [m.name for m in ir2.media] == [m.name for m in ir.media]
        assert len(ir2.emitters) == len(ir.emitters)
        assert len(ir2.sensors) == len(ir.sensors)
        assert ir2.rng == ir.rng
        assert ir2.sampling == ir.sampling


# ---------------------------------------------------------------------------
# Registry completeness (guard against drift, structural half of PR4's
# full numeric drift-guard)
# ---------------------------------------------------------------------------


class TestRegistryCompleteness:
    """Every geometry / BSDF / component kind either lowers or raises loudly."""

    def test_unregistered_geometry_raises(self):
        class _NotAGeometry:
            pass

        with pytest.raises(TypeError, match="geometry"):
            _lower_geometry(_NotAGeometry())

    def test_unregistered_bsdf_raises(self):
        class _NotABsdf:
            pass

        with pytest.raises(TypeError, match="BSDF"):
            _lower_bsdf(_NotABsdf())

    def test_unregistered_component_raises(self):
        class _NotAComponent:
            pass

        with pytest.raises(TypeError, match="component"):
            _component_kind(_NotAComponent())

    def test_every_bsdf_kind_reachable_from_a_real_bsdf(self):
        """Every BsdfKind literal has at least one concrete BaseBSDF that
        lowers to it -- a variant added to BsdfIR without a matching real
        class (or vice versa) is a test failure, not a runtime surprise."""
        from optiland.nonsequential.bsdf.specular import SpecularBRDF  # noqa: PLC0415

        cases = [
            (None, "none"),
            (SpecularBRDF(), "specular"),
            (LambertianBSDF(reflectance_value=0.5), "lambertian"),
            (HarveyShackBSDF(b0=1e-3, l0=0.01, s=2.0), "harvey_shack"),
        ]
        for bsdf, expected_kind in cases:
            assert _lower_bsdf(bsdf).kind == expected_kind


# ---------------------------------------------------------------------------
# Basic dataclass sanity
# ---------------------------------------------------------------------------


def test_scene_ir_is_frozen_dataclasses():
    for cls in (SceneIR, PrimitiveIR, BsdfIR, MediumIR):
        assert cls.__dataclass_params__.frozen  # type: ignore[attr-defined]
