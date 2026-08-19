"""IR interpreter tests for Non-Sequential Raytracing (PR4).

Covers: the per-bounce interaction loop is genuinely driven by SceneIR data
(component_kind / BsdfIR.kind), not by ``self.bsdf is not None`` or Python
class identity; the drift guard fires on a lower()/live-component mismatch;
and both backends build and use one IR per trace() call.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential import (
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    LensConfig,
    MirrorConfig,
    NSQScene,
    ReflectiveComponent,
    Spectrum,
)
from optiland.nonsequential.bsdf.lambertian import LambertianBSDF
from optiland.nonsequential.components.geometry.analytic.plane import (
    FinitePlaneGeometry,
)
from optiland.nonsequential.ir import lower
from optiland.nonsequential.ir.bsdf_ir import BsdfIR
from optiland.nonsequential.ir.interpreter import (
    apply_primitive_interactions,
    assert_bsdf_matches,
    assert_component_kind_matches,
)
from optiland.nonsequential.rng import NSQRng


def _lens_mirror_scene() -> NSQScene:
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(z=-80),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55), total_flux=1.0, aperture_radius=5.0
        ),
    )
    scene.add_lens(
        "L",
        CoordinateSystem(z=0),
        LensConfig(
            r1=50, r2=-50, thickness=5, material="N-BK7", front_aperture_radius=12.5
        ),
    )
    scene.add_mirror(
        "M",
        CoordinateSystem(z=100),
        MirrorConfig(radius=-200, reflectance=1.0, aperture_radius=20),
    )
    scene.add_detector(
        "D",
        CoordinateSystem(z=-90, rx=np.pi),
        IrradianceDetectorConfig(width=60, height=60, num_pixels_x=32, num_pixels_y=32),
    )
    return scene


class TestDriftGuard:
    """assert_component_kind_matches / assert_bsdf_matches fire on real drift."""

    def test_component_kind_match_passes_for_a_correctly_lowered_scene(self):
        scene = _lens_mirror_scene()
        ir = lower(scene)
        for primitive, component in zip(ir.primitives, scene.surfaces, strict=True):
            assert_component_kind_matches(component, primitive)  # must not raise

    def test_component_kind_mismatch_raises(self):
        scene = _lens_mirror_scene()
        ir = lower(scene)
        # Deliberately pair a primitive with the WRONG live component (a
        # refractive front face's IR paired with the reflective mirror).
        front = next(p for p in ir.primitives if p.name == "L.front")
        mirror = next(c for c in scene.surfaces if c.name == "M.surface")
        with pytest.raises(RuntimeError, match="drift"):
            assert_component_kind_matches(mirror, front)

    def test_bsdf_match_passes_for_a_correctly_lowered_scene(self):
        scene = _lens_mirror_scene()
        ir = lower(scene)
        for primitive, component in zip(ir.primitives, scene.surfaces, strict=True):
            assert_bsdf_matches(component.bsdf, primitive.bsdf)  # must not raise

    def test_bsdf_mismatch_raises(self):
        bsdf = LambertianBSDF(reflectance_value=0.5)
        with pytest.raises(RuntimeError, match="drift"):
            assert_bsdf_matches(bsdf, BsdfIR(kind="none"))

    def test_apply_primitive_interactions_propagates_component_kind_drift(self):
        """A caller that passes components out of step with ir.primitives
        gets a loud RuntimeError, not silently-wrong physics."""
        scene = _lens_mirror_scene()
        ir = lower(scene)
        rng = NSQRng(0)
        n = 4
        from optiland.nonsequential.ray_bundle import NSQRayBundle  # noqa: PLC0415

        rays = NSQRayBundle(
            x=np.zeros(n),
            y=np.zeros(n),
            z=np.zeros(n),
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
        t_min = np.full(n, 5.0)
        hit_normals = np.tile([0.0, 0.0, -1.0], (n, 1))
        hit_n_geom = np.tile([0.0, 0.0, 1.0], (n, 1))
        comp_idx = np.zeros(n, dtype=np.int32)  # every ray "hits" primitive 0
        comp_first = np.ones(n, dtype=bool)

        # Reorder components relative to ir.primitives -> primitive 0
        # ("L.front", refractive) is now paired with scene.surfaces[-1]
        # ("M.surface", reflective).
        reordered = [scene.surfaces[-1], *scene.surfaces[:-1]]

        with pytest.raises(RuntimeError, match="drift"):
            apply_primitive_interactions(
                rays,
                ir,
                reordered,
                t_min,
                hit_normals,
                hit_n_geom,
                comp_idx,
                comp_first,
                rng,
            )


class TestBsdfDispatchIsIrDriven:
    """The scatter branch is gated by bsdf_ir.kind, not self.bsdf truthiness."""

    def test_reflective_component_skips_scatter_when_bsdf_ir_says_none(self):
        """A component with a live BSDF attached must NOT scatter if the
        caller passes BsdfIR(kind='none') -- proves the interpreter's IR
        data, not the live self.bsdf attribute, controls the branch."""
        cs = CoordinateSystem(z=10)
        geom = FinitePlaneGeometry(width=20.0, height=20.0)
        mirror = ReflectiveComponent(
            cs=cs,
            geometry=geom,
            reflectance=1.0,
            bsdf=LambertianBSDF(reflectance_value=1.0),
        )

        n = 2000
        from optiland.nonsequential.ray_bundle import NSQRayBundle  # noqa: PLC0415

        rays = NSQRayBundle(
            x=np.zeros(n),
            y=np.zeros(n),
            z=np.zeros(n),
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
        t = np.full(n, 10.0)
        normals = np.tile([0.0, 0.0, -1.0], (n, 1))
        hit_mask = np.ones(n, dtype=bool)
        rng = NSQRng(0)

        n_geom = np.tile([0.0, 0.0, 1.0], (n, 1))
        mirror.interact(rays, t, normals, hit_mask, rng, BsdfIR(kind="none"), n_geom)

        # A +z beam hitting a z=-1-facing-normal mirror reflects to -z.
        # If the (attached) Lambertian lobe had fired, directions would be
        # scattered across the hemisphere instead of uniformly -z.
        np.testing.assert_allclose(rays.N, -1.0, atol=1e-9)
        np.testing.assert_allclose(rays.L, 0.0, atol=1e-9)
        np.testing.assert_allclose(rays.M, 0.0, atol=1e-9)

    def test_reflective_component_scatters_when_bsdf_ir_says_lambertian(self):
        """Same component, this time with a matching BsdfIR -- the lobe
        fires and directions spread across the hemisphere."""
        cs = CoordinateSystem(z=10)
        geom = FinitePlaneGeometry(width=20.0, height=20.0)
        bsdf = LambertianBSDF(reflectance_value=1.0)
        mirror = ReflectiveComponent(cs=cs, geometry=geom, reflectance=1.0, bsdf=bsdf)

        n = 2000
        from optiland.nonsequential.ray_bundle import NSQRayBundle  # noqa: PLC0415

        rays = NSQRayBundle(
            x=np.zeros(n),
            y=np.zeros(n),
            z=np.zeros(n),
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
        t = np.full(n, 10.0)
        normals = np.tile([0.0, 0.0, -1.0], (n, 1))
        hit_mask = np.ones(n, dtype=bool)
        rng = NSQRng(0)

        n_geom = np.tile([0.0, 0.0, 1.0], (n, 1))
        mirror.interact(
            rays,
            t,
            normals,
            hit_mask,
            rng,
            BsdfIR(kind="lambertian", params={"reflectance_value": 1.0}),
            n_geom,
        )

        assert not np.allclose(rays.N, -1.0, atol=1e-3)


class TestBackendsUseTheIr:
    """The live trace loop's component naming comes from the lowered IR."""

    def test_record_paths_component_names_match_ir_primitive_names(self):
        scene = _lens_mirror_scene()
        ir = lower(scene)
        expected_names = {p.name for p in ir.primitives}

        result = scene.trace(num_rays=2_000, seed=3, record_paths=True)
        hit_names = {
            row["component_name"]
            for row in result.ray_paths["events"]
            if row["event_type"] == "hit"
        }
        # Detector hits share the same _log_hits() path under the
        # detector's own (unrelated) name, and no ray reaches the lens edge
        # (its aperture is well inside the beam), so this is neither a
        # subset nor an equality check -- what matters is that the
        # *primitive* names the IR assigned are the ones actually showing
        # up in the event log for the surfaces rays do hit.
        assert {"L.front", "L.back", "M.surface"} <= hit_names
        assert hit_names <= expected_names | {"", "D"}
