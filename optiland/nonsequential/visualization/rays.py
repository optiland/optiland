"""Non-Sequential Rays Visualization Module

This module contains classes for visualizing rays in a non-sequential
optical system.

Kramer Harrison, 2026
"""

from __future__ import annotations

import contextlib

import numpy as np

with contextlib.suppress(ImportError):
    import vtk

from optiland.nonsequential.tracer import NSQTracer
from optiland.visualization.system.utils import project_rays


class NSQRays2D:
    """A class to represent and visualize 2D rays in an NSQ system.

    Args:
        scene (NSQScene): The non-sequential scene to be visualized.
    """

    def __init__(self, scene):
        self.scene = scene
        self.recorded_paths = None

    def plot(
        self,
        ax,
        num_rays=100,
        theme=None,
        projection="YZ",
        rng_seed=42,
        color_by: str = "source",
    ):
        """Plots the rays.

        Args:
            ax: The matplotlib axis to plot on.
            num_rays: Number of rays to trace.
            theme (Theme, optional): The theme to apply.
            projection (str): Projection plane.
            rng_seed (int): Seed for reproducibility.
            color_by (str): How to color rays ("source" or "bounce" or "segment").
        """
        self._trace(num_rays, rng_seed)
        if not self.recorded_paths:
            return

        self._plot_lines(ax, theme, projection, color_by)

    def _trace(self, num_rays, seed):
        tracer = NSQTracer(self.scene)
        res = tracer.trace(num_rays=num_rays, seed=seed, record_paths=True)
        self.recorded_paths = res.ray_paths

    def _plot_lines(self, ax, theme=None, projection="YZ", color_by="source"):
        ray_cycle = theme.parameters.get("ray_cycle") if theme else None

        if ray_cycle is None:
            color = "C0"
            ray_cycle = [color]
        else:
            color = ray_cycle[0]

        xs = self.recorded_paths["x"]
        ys = self.recorded_paths["y"]
        zs = self.recorded_paths["z"]

        n_bounces = len(xs)
        if n_bounces < 2:
            return

        batch_size = xs[0].shape[0]

        for k in range(batch_size):
            path_x = [xs[b][k] for b in range(n_bounces)]
            path_y = [ys[b][k] for b in range(n_bounces)]
            path_z = [zs[b][k] for b in range(n_bounces)]

            px = np.array(path_x)
            py = np.array(path_y)
            pz = np.array(path_z)

            mask = ~np.isnan(px)
            px = px[mask]
            py = py[mask]
            pz = pz[mask]

            if len(px) < 2:
                continue

            if color_by in ("bounce", "segment"):
                for b_idx in range(1, len(px)):
                    horiz, vert = project_rays(
                        px[b_idx - 1 : b_idx + 1],
                        py[b_idx - 1 : b_idx + 1],
                        pz[b_idx - 1 : b_idx + 1],
                        projection,
                    )
                    c = ray_cycle[b_idx % len(ray_cycle)]
                    ax.plot(horiz, vert, color=c, linewidth=1, alpha=0.5)
            else:
                horiz, vert = project_rays(px, py, pz, projection)
                ax.plot(horiz, vert, color=color, linewidth=1, alpha=0.5)


class NSQRays3D(NSQRays2D):
    """A class to represent 3D rays for visualization using VTK."""

    def __init__(self, scene):
        super().__init__(scene)
        self._rgb_colors = [
            (0.122, 0.467, 0.706),
        ]

    def plot(self, ax, num_rays=100, theme=None, rng_seed=42, color_by: str = "source"):
        """Plots the rays in 3D.

        Args:
            ax: The VTK renderer to plot on.
            num_rays: Number of rays.
            theme: The theme.
            rng_seed: Seed.
            color_by (str): How to color rays ("source" or "bounce" or "segment").
        """
        self._trace(num_rays, rng_seed)
        if not self.recorded_paths:
            return

        self._plot_lines(ax, theme, color_by=color_by)

    def _plot_lines(self, renderer, theme=None, projection=None, color_by="source"):
        ray_cycle = theme.parameters.get("ray_cycle") if theme else None

        if ray_cycle is None:
            color = self._rgb_colors[0]
            ray_cycle = [color]
        else:
            from matplotlib.colors import to_rgb

            ray_cycle = [to_rgb(rc) for rc in ray_cycle]
            color = ray_cycle[0]

        xs = self.recorded_paths["x"]
        ys = self.recorded_paths["y"]
        zs = self.recorded_paths["z"]

        n_bounces = len(xs)
        if n_bounces < 2:
            return

        batch_size = xs[0].shape[0]

        for k in range(batch_size):
            path_x = [xs[b][k] for b in range(n_bounces)]
            path_y = [ys[b][k] for b in range(n_bounces)]
            path_z = [zs[b][k] for b in range(n_bounces)]

            px = np.array(path_x)
            py = np.array(path_y)
            pz = np.array(path_z)

            mask = ~np.isnan(px)
            if not np.any(mask):
                continue

            px = px[mask]
            py = py[mask]
            pz = pz[mask]

            for b in range(1, len(px)):
                p0 = [px[b - 1], py[b - 1], pz[b - 1]]
                p1 = [px[b], py[b], pz[b]]

                line_source = vtk.vtkLineSource()
                line_source.SetPoint1(p0)
                line_source.SetPoint2(p1)

                line_mapper = vtk.vtkPolyDataMapper()
                line_mapper.SetInputConnection(line_source.GetOutputPort())
                line_actor = vtk.vtkActor()
                line_actor.SetMapper(line_mapper)
                line_actor.GetProperty().SetLineWidth(1)

                c = (
                    ray_cycle[b % len(ray_cycle)]
                    if color_by in ("bounce", "segment")
                    else color
                )
                line_actor.GetProperty().SetColor(c)
                line_actor.GetProperty().SetOpacity(0.5)

                renderer.AddActor(line_actor)
