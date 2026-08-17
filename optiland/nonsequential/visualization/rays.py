"""Non-Sequential Rays Visualization Module.

Provides :class:`NSQRays2D` and :class:`NSQRays3D` for drawing ray paths
onto Matplotlib axes and VTK renderers respectively.

When a pre-existing :class:`~optiland.nonsequential.tracer.SimulationResult`
with ``ray_paths`` is available, pass it via the ``ray_paths`` argument to
avoid a redundant trace.  If ``ray_paths`` is ``None`` the helpers run a
fresh Monte Carlo trace with ``record_paths=True``.

Kramer Harrison, 2026
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np

with contextlib.suppress(ImportError):
    import vtk

from optiland.nonsequential.tracer import NSQTracer
from optiland.visualization.system.utils import project_rays

if TYPE_CHECKING:
    from optiland.nonsequential.scene import NSQScene

_EVENT_ORDER = {"birth": 0, "hit": 1, "death": 2}


def _sort_ray_events(ev: np.ndarray) -> np.ndarray:
    """Order one ray's events from birth, through its hits, to its death.

    Every ``hit`` shares the same event-type rank, so the tie is broken on
    ``bounce`` and then on original log order. Without both tiebreakers a
    multi-bounce path is drawn with its segments shuffled.

    Args:
        ev: Structured event records for a single ray.

    Returns:
        The same records, sorted into path order.
    """
    type_rank = np.array(
        [_EVENT_ORDER.get(str(e), 1) for e in ev["event_type"]], dtype=np.int64
    )
    # lexsort takes keys least-significant first.
    order = np.lexsort((np.arange(len(ev)), ev["bounce"], type_rank))
    return ev[order]


def _paths_from_events(events: np.ndarray, num_rays: int) -> list[np.ndarray]:
    """Split an event log into up to ``num_rays`` per-ray paths, in path order.

    Grouping is done with a single sort rather than one boolean mask per ray
    id. Masking per id costs O(rays x events), which for a trace recorded with
    ``record_paths=True`` over a few hundred thousand rays takes longer than
    the trace itself.

    Args:
        events: Structured event array with a ``ray_id`` field.
        num_rays: Maximum number of distinct rays to return. Rays are taken
            evenly across the recorded set so the sample spans the beam.

    Returns:
        List of per-ray event arrays, each sorted into path order.
    """
    if len(events) == 0:
        return []

    ray_ids = np.unique(events["ray_id"])
    if 0 < num_rays < len(ray_ids):
        keep = ray_ids[np.linspace(0, len(ray_ids) - 1, num_rays).astype(int)]
        events = events[np.isin(events["ray_id"], keep)]

    # One sort groups every ray's events together; boundaries fall where the
    # ray id changes.
    events = events[np.argsort(events["ray_id"], kind="stable")]
    split_at = np.flatnonzero(np.diff(events["ray_id"])) + 1

    return [
        _sort_ray_events(group)
        for group in np.split(events, split_at)
        if len(group) >= 2
    ]


class NSQRays2D:
    """Visualize 2D ray paths for a non-sequential scene.

    Attributes:
        scene: The NSQScene being visualized.
        recorded_paths: Ray-path dict populated by :meth:`_trace` or
            supplied directly; ``None`` until populated.
    """

    def __init__(self, scene: NSQScene) -> None:
        """Initialize NSQRays2D.

        Args:
            scene: The non-sequential scene to be visualized.
        """
        self.scene = scene
        self.recorded_paths: dict | None = None

    def plot(
        self,
        ax,
        num_rays: int = 100,
        theme=None,
        projection: str = "YZ",
        rng_seed: int = 42,
        color_by: str = "source",
        ray_paths: dict | None = None,
    ) -> None:
        """Plot ray paths onto a Matplotlib axis.

        If *ray_paths* is provided it is used directly and no new trace is
        run.  Otherwise a fresh Monte Carlo trace with ``record_paths=True``
        is performed using *num_rays* and *rng_seed*.

        Args:
            ax: Matplotlib axis to plot on.
            num_rays: Number of ray paths to draw. When *ray_paths* is
                ``None`` this is also the number of rays traced; when a
                recorded log is supplied, that many paths are sampled evenly
                from it. Drawing every path of a large recorded trace is slow
                and renders as a solid block, so keep this small.
            theme: Optional theme (colour cycle, etc.).
            projection: Projection plane (``'YZ'``, ``'XZ'``, or ``'XY'``).
            rng_seed: RNG seed for the fresh trace (ignored when
                *ray_paths* is supplied).
            color_by: Ray colouring mode: ``'source'``, ``'bounce'``, or
                ``'segment'``.
            ray_paths: Pre-existing ray-path dict from a
                :class:`~optiland.nonsequential.tracer.SimulationResult`.
                When supplied no new trace is run.
        """
        if ray_paths is not None:
            self.recorded_paths = ray_paths
        else:
            self._trace(num_rays, rng_seed)
        if not self.recorded_paths:
            return

        self._plot_lines(ax, theme, projection, color_by, num_rays)

    def _trace(self, num_rays: int, seed: int) -> None:
        """Run a fresh trace and store the resulting ray paths.

        Args:
            num_rays: Number of rays to launch.
            seed: RNG seed.
        """
        tracer = NSQTracer(self.scene)
        res = tracer.trace(num_rays=num_rays, seed=seed, record_paths=True)
        self.recorded_paths = res.ray_paths

    def _plot_lines(
        self, ax, theme=None, projection="YZ", color_by="source", num_rays=0
    ):
        ray_cycle = theme.parameters.get("ray_cycle") if theme else None

        if ray_cycle is None:
            color = "C0"
            ray_cycle = [color]
        else:
            color = ray_cycle[0]

        # Support new event-based ray_paths format {"events": structured_array}
        if "events" in self.recorded_paths:
            self._plot_from_events(
                ax, theme, projection, color_by, ray_cycle, color, num_rays
            )
            return

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

    def _plot_from_events(
        self, ax, theme, projection, color_by, ray_cycle, color, num_rays=0
    ) -> None:
        """Plot rays from the new structured event-log format."""
        for ev in _paths_from_events(self.recorded_paths["events"], num_rays):
            px = ev["x"]
            py = ev["y"]
            pz = ev["z"]

            if color_by in ("bounce", "segment"):
                for b_idx in range(1, len(px)):
                    seg_x = px[b_idx - 1 : b_idx + 1]
                    seg_y = py[b_idx - 1 : b_idx + 1]
                    seg_z = pz[b_idx - 1 : b_idx + 1]
                    horiz, vert = project_rays(seg_x, seg_y, seg_z, projection)
                    c = ray_cycle[(b_idx - 1) % len(ray_cycle)]
                    ax.plot(horiz, vert, color=c, linewidth=1, alpha=0.5)
            else:
                horiz, vert = project_rays(px, py, pz, projection)
                ax.plot(horiz, vert, color=color, linewidth=1, alpha=0.5)


class NSQRays3D(NSQRays2D):
    """Visualize 3D ray paths for a non-sequential scene using VTK.

    Inherits :meth:`_trace` and the event-path logic from
    :class:`NSQRays2D`; overrides :meth:`plot` and :meth:`_plot_lines` to
    use VTK line actors instead of Matplotlib lines.
    """

    def __init__(self, scene: NSQScene) -> None:
        """Initialize NSQRays3D.

        Args:
            scene: The non-sequential scene to be visualized.
        """
        super().__init__(scene)
        self._rgb_colors = [
            (0.122, 0.467, 0.706),
        ]

    def plot(
        self,
        ax,
        num_rays: int = 100,
        theme=None,
        rng_seed: int = 42,
        color_by: str = "source",
        ray_paths: dict | None = None,
    ) -> None:
        """Plot ray paths onto a VTK renderer.

        If *ray_paths* is provided it is used directly and no new trace is
        run.  Otherwise a fresh Monte Carlo trace is performed.

        Args:
            ax: VTK renderer to add line actors to.
            num_rays: Number of ray paths to draw; also the number of rays
                traced when *ray_paths* is ``None``.
            theme: Optional theme object.
            rng_seed: RNG seed for the fresh trace (ignored when
                *ray_paths* is supplied).
            color_by: Ray colouring mode: ``'source'``, ``'bounce'``, or
                ``'segment'``.
            ray_paths: Pre-existing ray-path dict from a
                :class:`~optiland.nonsequential.tracer.SimulationResult`.
                When supplied no new trace is run.
        """
        if ray_paths is not None:
            self.recorded_paths = ray_paths
        else:
            self._trace(num_rays, rng_seed)
        if not self.recorded_paths:
            return

        self._plot_lines(ax, theme, color_by=color_by, num_rays=num_rays)

    def _plot_lines(
        self, renderer, theme=None, projection=None, color_by="source", num_rays=0
    ):
        ray_cycle = theme.parameters.get("ray_cycle") if theme else None

        if ray_cycle is None:
            color = self._rgb_colors[0]
            ray_cycle = [color]
        else:
            from matplotlib.colors import to_rgb

            ray_cycle = [to_rgb(rc) for rc in ray_cycle]
            color = ray_cycle[0]

        # Support new event-based ray_paths format
        if "events" in self.recorded_paths:
            self._plot_from_events_3d(renderer, ray_cycle, color, color_by, num_rays)
            return

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

    def _plot_from_events_3d(
        self, renderer, ray_cycle, color, color_by, num_rays=0
    ) -> None:
        """Plot rays from the new structured event-log format (3D)."""
        for ev in _paths_from_events(self.recorded_paths["events"], num_rays):
            for b in range(1, len(ev)):
                p0 = [ev["x"][b - 1], ev["y"][b - 1], ev["z"][b - 1]]
                p1 = [ev["x"][b], ev["y"][b], ev["z"][b]]

                line_source = vtk.vtkLineSource()
                line_source.SetPoint1(p0)
                line_source.SetPoint2(p1)

                line_mapper = vtk.vtkPolyDataMapper()
                line_mapper.SetInputConnection(line_source.GetOutputPort())
                line_actor = vtk.vtkActor()
                line_actor.SetMapper(line_mapper)
                line_actor.GetProperty().SetLineWidth(1)

                if color_by in ("bounce", "segment"):
                    c = ray_cycle[(b - 1) % len(ray_cycle)]
                else:
                    c = color
                line_actor.GetProperty().SetColor(c)
                line_actor.GetProperty().SetOpacity(0.5)
                renderer.AddActor(line_actor)
