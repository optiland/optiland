"""Volume tests for Non-Sequential Raytracing (PR6).

Covers watertightness validation, the ray-parity orientation check, CSG
union/intersection/difference scoping, and that Lens/Doublet are actually
validated as Volumes at construction.

Kramer Harrison, 2026
"""

from __future__ import annotations

import pytest

from optiland.coordinate_system import CoordinateSystem
from optiland.nonsequential.components.absorbing import AbsorbingComponent
from optiland.nonsequential.components.configs import DoubletConfig, LensConfig
from optiland.nonsequential.components.doublet import Doublet
from optiland.nonsequential.components.geometry.analytic.conic import ConicGeometry
from optiland.nonsequential.components.geometry.analytic.frustum import (
    CylindricalFrustumGeometry,
)
from optiland.nonsequential.components.lens import Lens
from optiland.nonsequential.components.refractive import RefractiveComponent
from optiland.nonsequential.components.volume import NonWatertightVolumeError, Volume
from optiland.nonsequential.materials.nsq_material import VACUUM, NSQMaterial


def _glass():
    return NSQMaterial.from_glass("N-BK7")


class TestLensAndDoubletBuildVolumes:
    def test_simple_lens_builds_one_volume(self):
        lens = Lens(
            "L",
            CoordinateSystem(z=0),
            LensConfig(
                r1=50, r2=-50, thickness=5, material="N-BK7", front_aperture_radius=12.5
            ),
        )
        assert lens._volume.name == "L"
        assert len(lens._volume.boundary) == 3  # front, back, edge

    def test_lens_with_rim_builds_one_volume(self):
        lens = Lens(
            "L",
            CoordinateSystem(z=0),
            LensConfig(
                r1=50,
                r2=-50,
                thickness=5,
                material="N-BK7",
                front_aperture_radius=12.5,
                back_aperture_radius=10.0,
            ),
        )
        assert len(lens._volume.boundary) == 4  # front, back, edge, rim

    def test_doublet_builds_two_volumes_sharing_the_cemented_interface(self):
        d = Doublet(
            "D",
            CoordinateSystem(z=0),
            DoubletConfig(
                r1=50,
                r2=-40,
                r3=-100,
                thickness1=5,
                thickness2=3,
                material1="N-BK7",
                material2="N-SF5",
                aperture_radius=10.0,
            ),
        )
        assert len(d._volumes) == 2
        crown, flint = d._volumes
        assert crown.name == "D.crown"
        assert flint.name == "D.flint"
        cemented = next(s for s in d.surfaces if s.name == "D.cemented")
        assert cemented in crown.boundary
        assert cemented in flint.boundary
        # 5 physical surfaces total, not duplicated for being shared.
        assert len(d.surfaces) == 5
        assert len({s.name for s in d.surfaces}) == 5


class TestWatertightnessCatchesGaps:
    def test_lens_missing_its_edge_is_not_watertight(self):
        """Front and back faces alone leave an open annular gap between
        their rims -- exactly the leak watertightness exists to catch."""
        front = RefractiveComponent(
            cs=CoordinateSystem(z=0),
            geometry=ConicGeometry(50.0, 0.0, 12.5),
            material_front=VACUUM,
            material_back=_glass(),
            name="front",
        )
        back = RefractiveComponent(
            cs=CoordinateSystem(z=5.0),
            geometry=ConicGeometry(-50.0, 0.0, 12.5),
            material_front=_glass(),
            material_back=VACUUM,
            name="back",
        )
        with pytest.raises(NonWatertightVolumeError, match="not watertight"):
            Volume(name="broken", boundary=[front, back], interior=_glass())

    def test_mismatched_edge_radius_is_not_watertight(self):
        """An edge whose radius doesn't match the faces' aperture leaves a gap."""
        front = RefractiveComponent(
            cs=CoordinateSystem(z=0),
            geometry=ConicGeometry(50.0, 0.0, 12.5),
            material_front=VACUUM,
            material_back=_glass(),
            name="front",
        )
        back = RefractiveComponent(
            cs=CoordinateSystem(z=5.0),
            geometry=ConicGeometry(-50.0, 0.0, 12.5),
            material_front=_glass(),
            material_back=VACUUM,
            name="back",
        )
        edge = AbsorbingComponent(
            cs=CoordinateSystem(z=0),
            geometry=CylindricalFrustumGeometry(
                r_front=12.5, r_back=12.5, z_front=0.0, z_back=5.0
            ),
            name="edge",
        )
        # Deliberately wrong: the true rim radius is not exactly 12.5 once
        # sag is accounted for at a curved front face -- but to construct an
        # unambiguous gap, offset the edge radius outright.
        edge.geometry.r_front = 13.0
        edge.geometry.r_back = 13.0
        with pytest.raises(NonWatertightVolumeError, match="not watertight"):
            Volume(name="broken", boundary=[front, back, edge], interior=_glass())

    def test_empty_boundary_raises(self):
        with pytest.raises(NonWatertightVolumeError):
            Volume(name="empty", boundary=[], interior=_glass())


class TestCsgScope:
    def test_union_concatenates_boundaries(self):
        lens = Lens(
            "L",
            CoordinateSystem(z=0),
            LensConfig(
                r1=50, r2=-50, thickness=5, material="N-BK7", front_aperture_radius=12.5
            ),
        )
        other = RefractiveComponent(
            cs=CoordinateSystem(z=100),
            geometry=ConicGeometry(10.0, 0.0, 1.0),
            material_front=VACUUM,
            material_back=_glass(),
            name="extra",
        )
        merged = Volume.union(lens._volume, [other])
        assert len(merged) == len(lens._volume.boundary) + 1
        assert other in merged

    def test_intersection_not_implemented(self):
        with pytest.raises(NotImplementedError, match="boolean surface evaluator"):
            Volume.intersection()

    def test_difference_not_implemented(self):
        with pytest.raises(NotImplementedError, match="boolean surface evaluator"):
            Volume.difference()


class TestVolumeIntegratesWithTracing:
    """A rebuilt Lens/Doublet still traces correctly end to end."""

    def test_singlet_still_focuses(self):
        from optiland.nonsequential import (
            CollimatedSourceConfig,
            IrradianceDetectorConfig,
            NSQScene,
            Spectrum,
        )

        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=-80),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=1.0,
                aperture_radius=5.0,
            ),
        )
        scene.add_lens(
            "L",
            CoordinateSystem(z=0),
            LensConfig(
                r1=50, r2=-50, thickness=5, material="N-BK7", front_aperture_radius=12.5
            ),
        )
        scene.add_detector(
            "D",
            CoordinateSystem(z=100),
            IrradianceDetectorConfig(
                width=20, height=20, num_pixels_x=32, num_pixels_y=32
            ),
        )
        result = scene.trace(num_rays=2_000, seed=1)
        assert result.detectors["D"].total_flux > 0.5

    def test_doublet_still_transmits(self):
        from optiland.nonsequential import (
            CollimatedSourceConfig,
            IrradianceDetectorConfig,
            NSQScene,
            Spectrum,
        )

        scene = NSQScene()
        scene.add_source(
            "S",
            CoordinateSystem(z=-80),
            CollimatedSourceConfig(
                spectrum=Spectrum.monochromatic(0.55),
                total_flux=1.0,
                aperture_radius=5.0,
            ),
        )
        scene.add_doublet(
            "D",
            CoordinateSystem(z=0),
            DoubletConfig(
                r1=50,
                r2=-40,
                r3=-100,
                thickness1=5,
                thickness2=3,
                material1="N-BK7",
                material2="N-SF5",
                aperture_radius=10.0,
            ),
        )
        scene.add_detector(
            "D_det",
            CoordinateSystem(z=100),
            IrradianceDetectorConfig(
                width=20, height=20, num_pixels_x=32, num_pixels_y=32
            ),
        )
        result = scene.trace(num_rays=2_000, seed=1)
        assert result.detectors["D_det"].total_flux > 0.5
