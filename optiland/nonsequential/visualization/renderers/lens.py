"""2D and 3D renderers for Lens compound components.

Kramer Harrison, 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optiland.nonsequential.visualization.renderers.base import (
    ComponentRenderer2D,
    ComponentRenderer3D,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from optiland.nonsequential.components.compound import CompoundComponent


class LensRenderer2D(ComponentRenderer2D):
    """Renders a Lens as a filled cross-section polygon in 2D.

    Reuses the sag-profile approach from the sequential visualizer:
    compute front- and back-face sag profiles in global coordinates, extend
    the narrower face with a rim line to match the wider aperture, close the
    contour, and fill it.
    """

    def render(
        self,
        component: CompoundComponent,
        ax: Axes,
        theme=None,
        projection: str = "YZ",
    ) -> None:
        """Draw the lens cross-section onto *ax*.

        Args:
            component: A Lens CompoundComponent.
            ax: Matplotlib axes.
            theme: Optional theme (colour, alpha, etc.).
            projection: Projection plane.
        """
        from optiland.nonsequential.components.lens import Lens  # noqa: PLC0415

        if not isinstance(component, Lens):
            return

        cfg = component._config
        cs = component._cs

        # Determine projection indices
        h_idx, v_idx = _projection_indices(projection)

        # Retrieve global origin and rotation for this CS
        from optiland.nonsequential.components.base import (
            _get_transform,  # noqa: PLC0415
        )

        translation, rot = _get_transform(cs)

        back_r = cfg.back_aperture_radius or cfg.front_aperture_radius
        front_r = cfg.front_aperture_radius
        n_pts = 64

        front_y = np.linspace(-front_r, front_r, n_pts)
        back_y = np.linspace(-back_r, back_r, n_pts)

        # Sag values in local frame
        front_z_local = _sag_array(cfg.r1, cfg.conic1, front_y)
        back_z_local = _sag_array(cfg.r2, cfg.conic2, back_y) + cfg.thickness

        # Transform to global: apply rotation + translation (row-vector form)
        def local_to_global(
            y_arr: np.ndarray, z_arr: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            pts_local = np.stack([np.zeros_like(y_arr), y_arr, z_arr], axis=1)
            pts_global = pts_local @ rot.T + translation
            return pts_global[:, h_idx], pts_global[:, v_idx]

        fh, fv = local_to_global(front_y, front_z_local)
        bh, bv = local_to_global(back_y, back_z_local)

        # For the rim: connect front top rim to back top rim, and front
        # bottom rim to back bottom rim using straight lines
        def pts2d(local_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            g = local_pts @ rot.T + translation
            return g[:, h_idx], g[:, v_idx]

        rts_h, rts_v = pts2d(np.array([[0.0, front_r, front_z_local[-1]]]))
        rte_h, rte_v = pts2d(np.array([[0.0, back_r, back_z_local[-1]]]))
        rbs_h, rbs_v = pts2d(np.array([[0.0, -back_r, back_z_local[0]]]))
        rbe_h, rbe_v = pts2d(np.array([[0.0, -front_r, front_z_local[0]]]))

        # Contour: front (bottom→top) → rim top → back (top→bottom) → rim bottom
        contour_h = np.concatenate([fh, rts_h, rte_h, bh[::-1], rbs_h, rbe_h])
        contour_v = np.concatenate([fv, rts_v, rte_v, bv[::-1], rbs_v, rbe_v])

        face_color = (
            "lightblue" if theme is None else getattr(theme, "lens_color", "lightblue")
        )
        edge_color = (
            "steelblue"
            if theme is None
            else getattr(theme, "lens_edge_color", "steelblue")
        )

        ax.fill(contour_h, contour_v, color=face_color, alpha=0.5, zorder=2)
        ax.plot(contour_h, contour_v, color=edge_color, linewidth=1.0, zorder=3)


class LensRenderer3D(ComponentRenderer3D):
    """Renders a Lens as a revolved 3D surface in VTK."""

    def render(
        self,
        component: CompoundComponent,
        renderer,
        theme=None,
    ) -> None:
        """Add VTK actors for the lens to *renderer*.

        For rotationally symmetric lenses, revolves the 2D YZ cross-section
        contour using :func:`~optiland.visualization.system.utils.revolve_contour`.

        Args:
            component: A Lens CompoundComponent.
            renderer: VTK renderer.
            theme: Optional theme.
        """
        from optiland.nonsequential.components.lens import Lens  # noqa: PLC0415

        if not isinstance(component, Lens):
            return

        try:
            from optiland.visualization.system.utils import (  # noqa: PLC0415
                revolve_contour,
            )
        except ImportError:
            return

        cfg = component._config
        cs = component._cs

        from optiland.nonsequential.components.base import (
            _get_transform,  # noqa: PLC0415
        )

        translation, rot = _get_transform(cs)

        back_r = cfg.back_aperture_radius or cfg.front_aperture_radius
        front_r = cfg.front_aperture_radius
        n_pts = 64

        r_front = np.linspace(0.0, front_r, n_pts)
        r_back = np.linspace(0.0, back_r, n_pts)

        z_front = _sag_array(cfg.r1, cfg.conic1, r_front)
        z_back = _sag_array(cfg.r2, cfg.conic2, r_back) + cfg.thickness

        # Build contour for revolution: r along +y axis, z along optical axis
        # Contour: front face + rim + back face (reversed) + axis segment
        y_contour = np.concatenate([r_front, [back_r], r_back[::-1], [0.0]])
        z_contour = np.concatenate([z_front, [z_back[-1]], z_back[::-1], [0.0]])
        x_contour = np.zeros_like(y_contour)

        # Transform contour points to global
        pts_local = np.stack([x_contour, y_contour, z_contour], axis=1)
        pts_global = pts_local @ rot.T + translation
        xg = pts_global[:, 0]
        yg = pts_global[:, 1]
        zg = pts_global[:, 2]

        actor = revolve_contour(xg, yg, zg)
        renderer.AddActor(actor)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sag_array(radius: float, conic: float, r: np.ndarray) -> np.ndarray:
    """Compute conic sag for an array of radial positions.

    Args:
        radius: Vertex radius of curvature [mm].
        conic: Conic constant.
        r: Radial positions [mm], shape (N,).

    Returns:
        Sag values [mm], shape (N,).
    """
    if radius == 0.0:
        return np.zeros_like(r)
    r2 = r * r
    R = radius
    K = conic
    under_root = np.maximum(1.0 - (1.0 + K) * r2 / (R * R), 0.0)
    return r2 / (R * (1.0 + np.sqrt(under_root)))


def _projection_indices(projection: str) -> tuple[int, int]:
    """Return (horizontal_index, vertical_index) for the given projection.

    Args:
        projection: One of ``'YZ'``, ``'XZ'``, ``'XY'``.

    Returns:
        Tuple of array column indices for the horizontal and vertical axes.
    """
    _map = {"YZ": (2, 1), "XZ": (2, 0), "XY": (0, 1)}
    return _map.get(projection.upper(), (2, 1))
