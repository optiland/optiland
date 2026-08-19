from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from optiland.optic import Optic
from optiland.samples import CookeTriplet
from optiland.visualization.system.optic_viewer import OpticViewer


@pytest.fixture
def optic() -> Optic:
    """A fixture that returns a CookeTriplet optic."""
    return CookeTriplet()


def test_optic_viewer_view_default_projection(optic: Optic):
    """Test that the default projection is YZ."""
    viewer = OpticViewer(optic)
    fig, ax, _ = viewer.view()
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == "Z [mm]"
    assert ax.get_ylabel() == "Y [mm]"
    plt.close(fig)


@pytest.mark.parametrize(
    "projection, expected_xlabel, expected_ylabel",
    [
        ("XY", "X [mm]", "Y [mm]"),
        ("XZ", "Z [mm]", "X [mm]"),
        ("YZ", "Z [mm]", "Y [mm]"),
    ],
)
def test_optic_viewer_view_valid_projections(
    optic: Optic, projection: str, expected_xlabel: str, expected_ylabel: str
):
    """Test that valid projections set the correct axis labels and plot data."""
    viewer = OpticViewer(optic)
    fig, ax, _ = viewer.view(projection=projection, fields=[(0, 1)], num_rays=5)
    assert isinstance(ax, Axes)
    assert ax.get_xlabel() == expected_xlabel
    assert ax.get_ylabel() == expected_ylabel

    # Verify plotted data for lenses and surfaces
    if projection == "XY":
        # Check if circles are plotted for lenses and surfaces
        from matplotlib.patches import Circle

        assert any(isinstance(p, Circle) for p in ax.patches)
        # Check that the circle for the first lens is centered correctly
        first_lens_patch = next(p for p in ax.patches if isinstance(p, Circle))
        center = first_lens_patch.center
        assert np.isclose(center[0], optic.surfaces[1].geometry.cs.x)
        assert np.isclose(center[1], optic.surfaces[1].geometry.cs.y)

    else:  # XZ, YZ
        from matplotlib.patches import Polygon

        assert any(isinstance(p, Polygon) for p in ax.patches)
        # Check that the polygon for the first lens has the correct orientation
        first_lens_patch = next(p for p in ax.patches if isinstance(p, Polygon))
        vertices = first_lens_patch.get_xy()
        if projection == "XZ":
            # In XZ, y-coordinates of vertices should contain both positive and negative values
            assert np.any(vertices[:, 1] > 0) and np.any(vertices[:, 1] < 0)
        else:  # YZ
            # In YZ, y-coordinates of vertices should contain both positive and negative values
            assert np.any(vertices[:, 1] > 0) and np.any(vertices[:, 1] < 0)

    plt.close(fig)


def test_optic_viewer_view_invalid_projection(optic: Optic):
    """Test that an invalid projection raises a ValueError."""
    viewer = OpticViewer(optic)
    with pytest.raises(
        ValueError, match="Invalid projection type. Must be 'XY', 'XZ', or 'YZ'."
    ):
        viewer.view(projection="invalid")
