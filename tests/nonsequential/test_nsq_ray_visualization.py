"""Tests for NSQ ray-path rendering.

Covers event ordering within a ray path and per-segment colouring, both of
which decide whether a multi-bounce path is drawn correctly.

Kramer Harrison, 2026
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from optiland.coordinate_system import CoordinateSystem  # noqa: E402
from optiland.nonsequential import (  # noqa: E402
    CollimatedSourceConfig,
    IrradianceDetectorConfig,
    MirrorConfig,
    NSQScene,
    Spectrum,
)
from optiland.nonsequential.backends.array_backend import _EVENT_DTYPE  # noqa: E402
from optiland.nonsequential.visualization import NSQViewer2D  # noqa: E402
from optiland.nonsequential.visualization.rays import (  # noqa: E402
    NSQRays2D,
    _paths_from_events,
    _sort_ray_events,
)


def _events(rows: list[tuple]) -> np.ndarray:
    """Build an event array from (event_type, z, bounce) triples."""
    arr = np.zeros(len(rows), dtype=_EVENT_DTYPE)
    for i, (event_type, z, bounce) in enumerate(rows):
        arr[i]["ray_id"] = 0
        arr[i]["event_type"] = event_type
        arr[i]["z"] = z
        arr[i]["bounce"] = bounce
    return arr


def _folded_scene() -> NSQScene:
    """Collimated beam onto a 45-degree fold mirror, detector off to the side."""
    scene = NSQScene()
    scene.add_source(
        "S",
        CoordinateSystem(z=-40.0),
        CollimatedSourceConfig(
            spectrum=Spectrum.monochromatic(0.55),
            total_flux=1.0,
            aperture_radius=8.0,
        ),
    )
    scene.add_mirror(
        "M",
        CoordinateSystem(z=0.0, rx=np.radians(45.0)),
        MirrorConfig(radius=np.inf, reflectance=1.0, aperture_radius=20.0),
    )
    scene.add_detector(
        "D",
        CoordinateSystem(y=40.0, z=0.0, rx=np.radians(90.0)),
        IrradianceDetectorConfig(
            width=30.0, height=30.0, num_pixels_x=16, num_pixels_y=16
        ),
    )
    return scene


def test_sort_ray_events_orders_hits_by_bounce():
    """Hits share an event-type rank, so bounce has to break the tie."""
    shuffled = _events(
        [
            ("hit", 30.0, 2),
            ("birth", 0.0, 0),
            ("hit", 10.0, 0),
            ("death", 40.0, 3),
            ("hit", 20.0, 1),
        ]
    )
    ordered = _sort_ray_events(shuffled)

    assert [str(e) for e in ordered["event_type"]] == [
        "birth",
        "hit",
        "hit",
        "hit",
        "death",
    ]
    # z increases monotonically along the path, so ordering is verifiable.
    assert ordered["z"].tolist() == [0.0, 10.0, 20.0, 30.0, 40.0]


def test_sort_ray_events_is_stable_within_a_bounce():
    """Equal ranks keep their original log order rather than being shuffled."""
    same_bounce = _events([("hit", 1.0, 0), ("hit", 2.0, 0), ("hit", 3.0, 0)])
    assert _sort_ray_events(same_bounce)["z"].tolist() == [1.0, 2.0, 3.0]


def test_each_path_segment_gets_its_own_colour():
    """A fold mirror gives two segments; they must not be drawn identically."""
    scene = _folded_scene()
    result = scene.trace(num_rays=200, seed=1, record_paths=True)

    fig, ax = plt.subplots()
    theme = type(
        "T", (), {"parameters": {"ray_cycle": ["#111111", "#222222", "#333333"]}}
    )()
    NSQRays2D(scene).plot(
        ax, theme=theme, color_by="bounce", ray_paths=result.ray_paths
    )

    colours = [line.get_color() for line in ax.get_lines()]
    assert "#111111" in colours, "incoming leg should use the first cycle colour"
    assert "#222222" in colours, "reflected leg should use the second cycle colour"
    plt.close(fig)


def test_num_rays_limits_paths_drawn_from_a_recorded_trace():
    """num_rays must subsample a supplied log, not draw all of it.

    Drawing every path of a large recorded trace produces an unreadable solid
    block, and grouping the events per ray id used to cost O(rays x events) --
    minutes for a trace of a few hundred thousand rays.
    """
    scene = _folded_scene()
    result = scene.trace(num_rays=2_000, seed=1, record_paths=True)

    fig, ax = plt.subplots()
    NSQRays2D(scene).plot(
        ax, num_rays=25, color_by="source", ray_paths=result.ray_paths
    )
    assert len(ax.get_lines()) == 25
    plt.close(fig)


def test_paths_from_events_groups_every_ray_when_unlimited():
    scene = _folded_scene()
    result = scene.trace(num_rays=50, seed=1, record_paths=True)
    events = result.ray_paths["events"]

    paths = _paths_from_events(events, 0)

    assert len(paths) == len(np.unique(events["ray_id"]))
    for path in paths:
        # One ray per path, and each path starts where the ray was born.
        assert len(np.unique(path["ray_id"])) == 1
        assert str(path["event_type"][0]) == "birth"


def test_viewer_renders_every_color_by_mode():
    scene = _folded_scene()
    result = scene.trace(num_rays=200, seed=1, record_paths=True)

    for color_by in ("source", "bounce", "segment"):
        fig, ax = NSQViewer2D(scene).view(result, num_rays=25, color_by=color_by)
        assert len(ax.get_lines()) > 0
        plt.close(fig)
