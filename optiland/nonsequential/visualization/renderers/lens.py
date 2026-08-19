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

        wider_r = max(front_r, back_r)

        # Determine edge and annulus paths based on which face is wider
        if front_r > back_r:
            rts_h, rts_v = pts2d(
                np.array(
                    [
                        [0.0, wider_r, front_z_local[-1]],
                        [0.0, wider_r, back_z_local[-1]],
                        [0.0, back_r, back_z_local[-1]],
                    ]
                )
            )
            rbs_h, rbs_v = pts2d(
                np.array(
                    [
                        [0.0, -back_r, back_z_local[0]],
                        [0.0, -wider_r, back_z_local[0]],
                        [0.0, -wider_r, front_z_local[0]],
                    ]
                )
            )
        elif back_r > front_r:
            rts_h, rts_v = pts2d(
                np.array(
                    [
                        [0.0, front_r, front_z_local[-1]],
                        [0.0, wider_r, front_z_local[-1]],
                        [0.0, wider_r, back_z_local[-1]],
                    ]
                )
            )
            rbs_h, rbs_v = pts2d(
                np.array(
                    [
                        [0.0, -wider_r, back_z_local[0]],
                        [0.0, -wider_r, front_z_local[0]],
                        [0.0, -front_r, front_z_local[0]],
                    ]
                )
            )
        else:
            rts_h, rts_v = pts2d(
                np.array(
                    [[0.0, front_r, front_z_local[-1]], [0.0, back_r, back_z_local[-1]]]
                )
            )
            rbs_h, rbs_v = pts2d(
                np.array(
                    [[0.0, -back_r, back_z_local[0]], [0.0, -front_r, front_z_local[0]]]
                )
            )

        # Contour: front (bottom->top) -> rim top -> back (top->bottom) -> rim bottom
        contour_h = np.concatenate([fh, rts_h, bh[::-1], rbs_h])
        contour_v = np.concatenate([fv, rts_v, bv[::-1], rbs_v])

        face_color = (0.8, 0.8, 0.8, 0.6)
        if theme is not None:
            face_color = theme.parameters.get("lens.color", face_color)

        edge_color = (0.5, 0.5, 0.5)
        if theme is not None:
            edge_color = theme.parameters.get("axes.edgecolor", edge_color)

        ax.fill(contour_h, contour_v, facecolor=face_color, zorder=2)
        ax.plot(contour_h, contour_v, color=edge_color, linewidth=1.0, zorder=3)


class DoubletRenderer2D(ComponentRenderer2D):
    """Renders a cemented Doublet as two filled cross-sections in 2D."""

    def render(
        self,
        component: CompoundComponent,
        ax: Axes,
        theme=None,
        projection: str = "YZ",
    ) -> None:
        """Draw the doublet cross-section onto *ax*.

        Args:
            component: A Doublet CompoundComponent.
            ax: Matplotlib axes.
            theme: Optional theme.
            projection: Projection plane.
        """
        from optiland.nonsequential.components.doublet import Doublet  # noqa: PLC0415

        if not isinstance(component, Doublet):
            return

        cfg = component._config
        cs = component._cs
        h_idx, v_idx = _projection_indices(projection)

        from optiland.nonsequential.components.base import (
            _get_transform,  # noqa: PLC0415
        )

        translation, rot = _get_transform(cs)

        r = cfg.aperture_radius
        n_pts = 64
        y = np.linspace(-r, r, n_pts)

        # Local sags
        z1 = _sag_array(cfg.r1, cfg.conic1, y)
        z2 = _sag_array(cfg.r2, cfg.conic2, y) + cfg.thickness1
        z3 = _sag_array(cfg.r3, cfg.conic3, y) + cfg.thickness1 + cfg.thickness2

        def to_global(y_arr, z_arr):
            pts_local = np.stack([np.zeros_like(y_arr), y_arr, z_arr], axis=1)
            pts_global = pts_local @ rot.T + translation
            return pts_global[:, h_idx], pts_global[:, v_idx]

        h1, v1 = to_global(y, z1)
        h2, v2 = to_global(y, z2)
        h3, v3 = to_global(y, z3)

        # Edges
        def pts2d(y_val, z_val):
            pts_local = np.array([[0.0, y_val, z_val]])
            pts_global = pts_local @ rot.T + translation
            return pts_global[:, h_idx], pts_global[:, v_idx]

        # Element 1: Front -> Top Edge -> Interface -> Bottom Edge
        e1t_h, e1t_v = pts2d(r, z2[-1])  # Top interface rim
        e1b_h, e1b_v = pts2d(-r, z1[0])  # Bottom front rim

        c1h = np.concatenate([h1, e1t_h, h2[::-1], e1b_h])
        c1v = np.concatenate([v1, e1t_v, v2[::-1], e1b_v])

        # Element 2: Interface -> Top Edge -> Back -> Bottom Edge
        e2t_h, e2t_v = pts2d(r, z3[-1])  # Top back rim
        e2b_h, e2b_v = pts2d(-r, z2[0])  # Bottom interface rim

        c2h = np.concatenate([h2, e2t_h, h3[::-1], e2b_h])
        c2v = np.concatenate([v2, e2t_v, v3[::-1], e2b_v])

        face_color = (0.8, 0.8, 0.8, 0.6)
        if theme is not None:
            face_color = theme.parameters.get("lens.color", face_color)

        edge_color = (0.5, 0.5, 0.5)
        if theme is not None:
            edge_color = theme.parameters.get("axes.edgecolor", edge_color)

        ax.fill(c1h, c1v, facecolor=face_color, zorder=2)
        ax.fill(c2h, c2v, facecolor=face_color, zorder=2)
        ax.plot(c1h, c1v, color=edge_color, linewidth=1.0, zorder=3)
        ax.plot(c2h, c2v, color=edge_color, linewidth=1.0, zorder=3)


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
        wider_r = max(front_r, back_r)

        if front_r > back_r:
            # Annulus at the back
            y_edge = [wider_r, wider_r, back_r]
            z_edge = [z_front[-1], z_back[-1], z_back[-1]]
        elif back_r > front_r:
            # Annulus at the front
            y_edge = [front_r, wider_r, wider_r]
            z_edge = [z_front[-1], z_front[-1], z_back[-1]]
        else:
            y_edge = [wider_r]
            z_edge = [z_back[-1]]

        y_contour = np.concatenate([r_front, y_edge, r_back[::-1], [0.0]])
        z_contour = np.concatenate([z_front, z_edge, z_back[::-1], [0.0]])
        x_contour = np.zeros_like(y_contour)

        # Transform contour points to global
        pts_local = np.stack([x_contour, y_contour, z_contour], axis=1)
        pts_global = pts_local @ rot.T + translation
        xg = pts_global[:, 0]
        yg = pts_global[:, 1]
        zg = pts_global[:, 2]

        actor = revolve_contour(xg, yg, zg)

        # Configure material matching Sequential Lens3D
        color = (0.9, 0.9, 1.0)
        if theme is not None:
            from matplotlib.colors import to_rgb

            color_hex = theme.parameters.get("lens.color", "#E5E5FF")
            color = to_rgb(color_hex)
        prop = actor.GetProperty()
        prop.SetOpacity(0.5)
        prop.SetColor(color)
        prop.SetSpecular(1.0)
        prop.SetSpecularPower(50.0)

        renderer.AddActor(actor)


class DoubletRenderer3D(ComponentRenderer3D):
    """Renders a Doublet as two revolved 3D surfaces in VTK."""

    def render(
        self,
        component: CompoundComponent,
        renderer,
        theme=None,
    ) -> None:
        """Add VTK actors for the doublet to *renderer*."""
        from optiland.nonsequential.components.doublet import Doublet  # noqa: PLC0415

        if not isinstance(component, Doublet):
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

        r = cfg.aperture_radius
        n_pts = 64
        rs = np.linspace(0.0, r, n_pts)

        z1 = _sag_array(cfg.r1, cfg.conic1, rs)
        z2 = _sag_array(cfg.r2, cfg.conic2, rs) + cfg.thickness1
        z3 = _sag_array(cfg.r3, cfg.conic3, rs) + cfg.thickness1 + cfg.thickness2

        # Element 1 contour
        y_c1 = np.concatenate([rs, [r], rs[::-1], [0.0]])
        z_c1 = np.concatenate([z1, [z2[-1]], z2[::-1], [0.0]])

        # Element 2 contour
        y_c2 = np.concatenate([rs, [r], rs[::-1], [0.0]])
        z_c2 = np.concatenate([z2, [z3[-1]], z3[::-1], [0.0]])

        def build_actor(y_c, z_c):
            pts_local = np.stack([np.zeros_like(y_c), y_c, z_c], axis=1)
            pts_global = pts_local @ rot.T + translation
            return revolve_contour(pts_global[:, 0], pts_global[:, 1], pts_global[:, 2])

        actor1 = build_actor(y_c1, z_c1)
        actor2 = build_actor(y_c2, z_c2)

        # Style matching Sequential Lens3D
        color = (0.9, 0.9, 1.0)
        if theme is not None:
            from matplotlib.colors import to_rgb  # noqa: PLC0415

            color_hex = theme.parameters.get("lens.color", "#E5E5FF")
            color = to_rgb(color_hex)

        for actor in [actor1, actor2]:
            prop = actor.GetProperty()
            prop.SetOpacity(0.5)
            prop.SetColor(color)
            prop.SetSpecular(1.0)
            prop.SetSpecularPower(50.0)
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
