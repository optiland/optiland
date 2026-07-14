from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import optiland.backend as be
from optiland import physical_apertures
from optiland.optic import Optic
from optiland.visualization.system.optic_viewer import OpticViewer
from optiland.visualization.system.system import OpticalSystem


def _build_optic_with_offset_aperture() -> Optic:
    """A single parabolic mirror with an aperture offset far from its vertex.

    Regression setup for the bug where an off-axis parabolic mirror
    (e.g. an off-axis paraboloid, OAP) with an OffsetRadialAperture was not
    rendered at all in the 2D system plot, because the plotted sag line was
    sampled over a symmetric window sized from the *un-signed* aperture
    extent, which for a large offset collapses to a window that misses the
    aperture entirely.
    """
    offset_y = -33.49
    r_max = 15.0

    optic = Optic()
    optic.surfaces.add(index=0, radius=be.inf, z=-be.inf)
    aperture = physical_apertures.OffsetRadialAperture(
        r_max=r_max, r_min=0, offset_y=offset_y
    )
    optic.surfaces.add(
        index=1,
        radius=-250.0,
        z=0.0,
        conic=-1.0,
        rx=-np.pi / 2 + np.deg2rad(15),
        aperture=aperture,
        material="mirror",
        is_stop=True,
    )
    optic.surfaces.add(index=2, z=-100.0)
    optic.set_aperture(aperture_type="EPD", value=20)
    optic.fields.set_type(field_type="angle")
    optic.fields.add(y=0)
    optic.wavelengths.add(value=0.55, is_primary=True)
    return optic


def test_offset_aperture_surface_extent_covers_offset_region(set_test_backend):
    optic = _build_optic_with_offset_aperture()
    aperture = optic.surfaces[1].aperture

    from optiland.visualization.system.surface import Surface2D

    surf2d = Surface2D(optic.surfaces[1], ray_extent=1.0)

    # The plotting extent must be large enough to cover the full offset
    # aperture bounds, not just the aperture radius.
    x_min, x_max, y_min, y_max = aperture.extent
    assert surf2d.extent >= max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))


def test_offset_aperture_surface_is_plotted(set_test_backend):
    """The mirror surface line must not be entirely NaN'd out by clipping."""
    optic = _build_optic_with_offset_aperture()

    viewer = OpticViewer(optic)
    fig, ax, _ = viewer.view(fields=[(0, 0)], num_rays=3, projection="YZ")

    surface_lines = [
        line
        for line in ax.get_lines()
        if line.get_label().startswith("Surface")
    ]
    assert surface_lines, "no surface lines were plotted"

    mirror_line = surface_lines[1]  # index 0 = object surface, index 1 = mirror
    y_data = mirror_line.get_ydata()
    assert np.any(~np.isnan(y_data)), (
        "offset-aperture mirror surface was not rendered anywhere"
    )
    plt.close(fig)


def test_aperture_indicator_follows_surface_sag(set_test_backend):
    """The aperture-edge indicator line must track the tilted surface's sag.

    Regression test: OpticalSystem._plot_apertures used to build the
    indicator line assuming the aperture rim sits at local z = 0 (the
    surface vertex plane), rather than at the surface's actual sag. For a
    steeply curved, strongly tilted surface (e.g. an off-axis parabola) that
    mismatch, rotated by the tilt, produced an indicator line offset from
    the mirror by several mm instead of tracing its edge.
    """
    optic = _build_optic_with_offset_aperture()
    surface = optic.surfaces[1]
    aperture = surface.aperture

    class DummyRays:
        def __init__(self, optic):
            self.r_extent = [15] * optic.surfaces.num_surfaces

    optical_system = OpticalSystem(optic, DummyRays(optic), projection="2d")

    fig, ax = plt.subplots()
    optical_system._plot_apertures(ax, projection="YZ")

    # There should be exactly one aperture-edge line (plus the two arrow
    # annotations, which are not Line2D artists).
    lines = ax.get_lines()
    assert len(lines) == 1
    z_data, y_data = lines[0].get_data()

    _, _, y_min, y_max = aperture.extent
    expected_sag = [
        be.to_numpy(surface.geometry.sag(0.0, y_min)).item(),
        be.to_numpy(surface.geometry.sag(0.0, y_max)).item(),
    ]

    # Recompute the expected global (z, y) of the two aperture-rim points
    # using the true sag, and compare against what was actually plotted.
    from optiland.visualization.system.utils import transform

    x_local = be.array([0.0, 0.0])
    y_local = be.array([y_min, y_max])
    z_local = be.array(expected_sag)
    _, y_expected, z_expected = transform(
        x_local, y_local, z_local, surface, is_global=False
    )

    np.testing.assert_allclose(sorted(z_data), sorted(be.to_numpy(z_expected)))
    np.testing.assert_allclose(sorted(y_data), sorted(be.to_numpy(y_expected)))
    plt.close(fig)
