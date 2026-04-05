"""2D and 3D renderers for Mirror compound components.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential.visualization.renderers.base import (
    ComponentRenderer2D,
    ComponentRenderer3D,
)
from optiland.nonsequential.visualization.renderers.lens import (
    _projection_indices,
    _sag_array,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from optiland.nonsequential.components.compound import CompoundComponent


class MirrorRenderer2D(ComponentRenderer2D):
    """Renders a Mirror as a thick arc/curve in 2D."""

    def render(
        self,
        component: CompoundComponent,
        ax: Axes,
        theme=None,
        projection: str = "YZ",
    ) -> None:
        """Draw the mirror profile onto *ax*.

        Args:
            component: A Mirror CompoundComponent.
            ax: Matplotlib axes.
            theme: Optional theme.
            projection: Projection plane.
        """
        from optiland.nonsequential.components.mirror import Mirror  # noqa: PLC0415

        if not isinstance(component, Mirror):
            return

        from optiland.nonsequential.components.base import (
            _get_transform,  # noqa: PLC0415
        )

        cfg = component._config
        translation, rot = _get_transform(component._cs)
        h_idx, v_idx = _projection_indices(projection)

        n_pts = 128
        y = np.linspace(-cfg.aperture_radius, cfg.aperture_radius, n_pts)
        z = _sag_array(cfg.radius, cfg.conic, y)

        pts_local = np.stack([np.zeros_like(y), y, z], axis=1)
        pts_global = pts_local @ rot.T + translation

        ph = pts_global[:, h_idx]
        pv = pts_global[:, v_idx]

        color = (0.5, 0.5, 0.5)
        if theme is not None:
            color = theme.parameters.get("axes.edgecolor", color)

        ax.plot(ph, pv, color=color, linewidth=2.0, zorder=3)


class MirrorRenderer3D(ComponentRenderer3D):
    """Renders a Mirror as a revolved 3D surface in VTK."""

    def render(
        self,
        component: CompoundComponent,
        renderer,
        theme=None,
    ) -> None:
        """Add VTK actors for the mirror to *renderer*.

        Args:
            component: A Mirror CompoundComponent.
            renderer: VTK renderer.
            theme: Optional theme.
        """
        from optiland.nonsequential.components.mirror import Mirror  # noqa: PLC0415

        if not isinstance(component, Mirror):
            return

        try:
            from optiland.visualization.system.utils import (
                revolve_contour,  # noqa: PLC0415
            )
        except ImportError:
            return

        from optiland.nonsequential.components.base import (
            _get_transform,  # noqa: PLC0415
        )

        cfg = component._config
        translation, rot = _get_transform(component._cs)

        n_pts = 128
        r = np.linspace(0.0, cfg.aperture_radius, n_pts)
        z = _sag_array(cfg.radius, cfg.conic, r)

        pts_local = np.stack([np.zeros_like(r), r, z], axis=1)
        pts_global = pts_local @ rot.T + translation

        actor = revolve_contour(pts_global[:, 0], pts_global[:, 1], pts_global[:, 2])

        # Style matching Sequential Mirror3D
        color = (0.75, 0.75, 0.75)
        if theme is not None:
            from matplotlib.colors import to_rgb  # noqa: PLC0415

            color_hex = theme.parameters.get("axes.edgecolor", "#BFBFBF")
            color = to_rgb(color_hex)

        prop = actor.GetProperty()
        prop.SetColor(color)
        prop.SetSpecular(1.0)
        prop.SetSpecularPower(100.0)  # Metallic
        prop.SetOpacity(1.0)

        renderer.AddActor(actor)
