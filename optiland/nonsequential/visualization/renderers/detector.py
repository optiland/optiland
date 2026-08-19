"""2D and 3D renderers for detector surfaces.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential.visualization.renderers.lens import _projection_indices

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from optiland.nonsequential.detectors.base import BaseDetector


class DetectorRenderer2D:
    """Renders a detector as a line segment in 2D."""

    def render(
        self,
        detector: BaseDetector,
        ax: Axes,
        theme=None,
        projection: str = "YZ",
    ) -> None:
        """Draw the detector footprint onto *ax*.

        Args:
            detector: Detector to render.
            ax: Matplotlib axes.
            theme: Optional theme.
            projection: Projection plane.
        """
        from optiland.nonsequential.components.base import (
            _get_transform,  # noqa: PLC0415
        )
        from optiland.nonsequential.detectors.irradiance import (
            IrradianceDetector,  # noqa: PLC0415
        )

        if not isinstance(detector, IrradianceDetector):
            return

        translation, rot = _get_transform(detector.cs)
        h_idx, v_idx = _projection_indices(projection)

        w = detector.width / 2.0
        h = detector.height / 2.0

        # Four corners of the detector in local frame
        corners_local = np.array(
            [[-w, -h, 0.0], [w, -h, 0.0], [w, h, 0.0], [-w, h, 0.0], [-w, -h, 0.0]]
        )
        corners_global = corners_local @ rot.T + translation

        ch = corners_global[:, h_idx]
        cv = corners_global[:, v_idx]

        color = "green" if theme is None else getattr(theme, "detector_color", "green")
        ax.plot(ch, cv, color=color, linewidth=1.5, linestyle="--", zorder=4)


class DetectorRenderer3D:
    """Renders a detector as a flat rectangular patch in 3D VTK."""

    def render(
        self,
        detector: BaseDetector,
        renderer,
        theme=None,
    ) -> None:
        """Add a VTK actor for the detector to *renderer*.

        Args:
            detector: Detector to render.
            renderer: VTK renderer.
            theme: Optional theme.
        """
        from optiland.nonsequential.components.base import (
            _get_transform,  # noqa: PLC0415
        )
        from optiland.nonsequential.detectors.irradiance import (
            IrradianceDetector,  # noqa: PLC0415
        )

        if not isinstance(detector, IrradianceDetector):
            return

        try:
            import vtk  # type: ignore[import]  # noqa: PLC0415
        except ImportError:
            return

        translation, rot = _get_transform(detector.cs)
        w = detector.width / 2.0
        h = detector.height / 2.0

        corners_local = np.array(
            [[-w, -h, 0.0], [w, -h, 0.0], [w, h, 0.0], [-w, h, 0.0]]
        )
        corners_global = corners_local @ rot.T + translation

        points = vtk.vtkPoints()
        for pt in corners_global:
            points.InsertNextPoint(*pt)

        quad = vtk.vtkQuad()
        for i in range(4):
            quad.GetPointIds().SetId(i, i)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(quad)

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.SetPolys(cells)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.0, 0.8, 0.0)
        actor.GetProperty().SetOpacity(0.5)

        renderer.AddActor(actor)
