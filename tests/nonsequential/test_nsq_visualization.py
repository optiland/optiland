"""Tests for NSQ visualization refactor — BaseViewer2D / BaseViewer3D hierarchy.

Kramer Harrison, 2026
"""

from __future__ import annotations

import pytest

from optiland.visualization.base import BaseViewer2D, BaseViewer3D

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_minimal_scene():
    """Return a minimal NSQScene with one lens, one source, one detector."""
    from optiland.coordinate_system import CoordinateSystem
    from optiland.nonsequential import (
        CollimatedSourceConfig,
        IrradianceDetectorConfig,
        LensConfig,
        NSQScene,
        Spectrum,
    )

    scene = NSQScene()
    spectrum = Spectrum.monochromatic(0.55)

    scene.add_source(
        "S1",
        CoordinateSystem(z=-15.0),
        CollimatedSourceConfig(spectrum=spectrum, total_flux=1.0, aperture_radius=5.0),
    )
    scene.add_lens(
        "L1",
        CoordinateSystem(z=0.0),
        LensConfig(
            r1=50.0,
            r2=-50.0,
            thickness=5.0,
            material="N-BK7",
            front_aperture_radius=10.0,
        ),
    )
    scene.add_detector(
        "D1",
        CoordinateSystem(z=60.0),
        IrradianceDetectorConfig(
            width=5.0, height=5.0, num_pixels_x=64, num_pixels_y=64
        ),
    )
    return scene


@pytest.fixture
def minimal_scene():
    return _make_minimal_scene()


# ---------------------------------------------------------------------------
# BaseViewer2D helper tests
# ---------------------------------------------------------------------------


class _ConcreteViewer2D(BaseViewer2D):
    """Minimal concrete subclass for testing shared helpers."""

    def view(self, **kwargs):
        return None


class _ConcreteViewer3D(BaseViewer3D):
    """Minimal concrete subclass for testing shared helpers."""

    def view(self, **kwargs):
        return None


def test_base_viewer_2d_resolve_theme_none():
    """_resolve_theme(None) must return a real Theme, not None."""
    from optiland.visualization.themes import Theme

    viewer = _ConcreteViewer2D(None)
    result = viewer._resolve_theme(None)
    assert result is not None
    assert isinstance(result, Theme)


def test_base_viewer_2d_resolve_theme_passthrough():
    """_resolve_theme(my_theme) must return the same object unchanged."""
    from optiland.visualization.themes import get_active_theme

    viewer = _ConcreteViewer2D(None)
    my_theme = get_active_theme()
    result = viewer._resolve_theme(my_theme)
    assert result is my_theme


def test_base_viewer_2d_make_figure_returns_fig_ax():
    """_make_figure(theme) must return (Figure, Axes)."""
    import matplotlib.figure
    from matplotlib.axes import Axes

    from optiland.visualization.themes import get_active_theme

    viewer = _ConcreteViewer2D(None)
    theme = get_active_theme()
    fig, ax = viewer._make_figure(theme)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, Axes)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_base_viewer_2d_make_figure_reuses_ax():
    """When ax is passed to _make_figure, it must reuse the existing figure."""
    import matplotlib.pyplot as plt

    from optiland.visualization.themes import get_active_theme

    viewer = _ConcreteViewer2D(None)
    theme = get_active_theme()

    fig0, ax0 = plt.subplots()
    fig_result, ax_result = viewer._make_figure(theme, ax=ax0)
    assert fig_result is fig0
    assert ax_result is ax0
    plt.close(fig0)


def test_base_viewer_2d_apply_axes_style_labels():
    """_apply_axes_style sets the correct axis labels for each projection."""
    import matplotlib.pyplot as plt

    from optiland.visualization.themes import get_active_theme

    viewer = _ConcreteViewer2D(None)
    theme = get_active_theme()

    expected = {
        "YZ": ("Z [mm]", "Y [mm]"),
        "XZ": ("Z [mm]", "X [mm]"),
        "XY": ("X [mm]", "Y [mm]"),
    }

    for proj, (expected_x, expected_y) in expected.items():
        fig, ax = plt.subplots()
        viewer._apply_axes_style(ax, proj, theme)
        assert ax.get_xlabel() == expected_x, f"Projection {proj}: xlabel mismatch"
        assert ax.get_ylabel() == expected_y, f"Projection {proj}: ylabel mismatch"
        plt.close(fig)


# ---------------------------------------------------------------------------
# NSQViewer2D tests
# ---------------------------------------------------------------------------


def test_nsq_viewer_2d_view_returns_fig_ax(minimal_scene):
    """NSQViewer2D.view() must return a (fig, ax) 2-tuple."""
    import matplotlib.pyplot as plt

    from optiland.nonsequential.visualization import NSQViewer2D

    viewer = NSQViewer2D(minimal_scene)
    result = viewer.view(num_rays=0)
    assert len(result) == 2
    fig, ax = result
    import matplotlib.figure

    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close(fig)


def test_nsq_viewer_2d_view_accepts_figsize(minimal_scene):
    """NSQViewer2D.view() must accept figsize kwarg."""
    import matplotlib.pyplot as plt

    from optiland.nonsequential.visualization import NSQViewer2D

    viewer = NSQViewer2D(minimal_scene)
    fig, ax = viewer.view(num_rays=0, figsize=(8, 4))
    plt.close(fig)


def test_nsq_viewer_2d_view_accepts_title(minimal_scene):
    """NSQViewer2D.view() must apply a custom title."""
    import matplotlib.pyplot as plt

    from optiland.nonsequential.visualization import NSQViewer2D

    viewer = NSQViewer2D(minimal_scene)
    fig, ax = viewer.view(num_rays=0, title="Custom Title")
    assert ax.get_title() == "Custom Title"
    plt.close(fig)


def test_nsq_viewer_2d_default_title(minimal_scene):
    """NSQViewer2D.view() uses a default title when none is specified."""
    import matplotlib.pyplot as plt

    from optiland.nonsequential.visualization import NSQViewer2D

    viewer = NSQViewer2D(minimal_scene)
    fig, ax = viewer.view(num_rays=0)
    assert "NSQ" in ax.get_title() or ax.get_title() != ""
    plt.close(fig)


# ---------------------------------------------------------------------------
# ray_paths pass-through test
# ---------------------------------------------------------------------------


def test_nsq_viewer_2d_uses_existing_ray_paths(minimal_scene):
    """When a result with ray_paths=None is passed, NSQViewer2D must not crash.

    This exercises the result.ray_paths pass-through path: if ray_paths is
    None on the result, NSQRays2D falls through to tracing fresh rays
    (num_rays > 0) -- we just verify no AttributeError is raised.
    """
    import matplotlib.pyplot as plt

    from optiland.nonsequential.tracer import SimulationResult
    from optiland.nonsequential.visualization import NSQViewer2D

    # A result without ray_paths (the common case before record_paths=True)
    result = SimulationResult(ray_paths=None)
    viewer = NSQViewer2D(minimal_scene)
    # Should not raise; num_rays=0 skips ray drawing
    fig, ax = viewer.view(result=result, num_rays=0)
    plt.close(fig)


# ---------------------------------------------------------------------------
# OpticViewer backward-compat test
# ---------------------------------------------------------------------------


def _make_simple_optic():
    """Return a minimal sequential Optic for backward-compat testing."""
    from optiland.optic import Optic

    optic = Optic()
    optic.add_surface(index=0, thickness=float("inf"))
    optic.add_surface(
        index=1, radius=50.0, thickness=5.0, material="N-BK7", is_stop=True
    )
    optic.add_surface(index=2, radius=-50.0, thickness=50.0)
    optic.add_surface(index=3)
    optic.set_aperture(aperture_type="EPD", value=10.0)
    optic.set_field_type(field_type="angle")
    optic.add_field(y=0.0)
    optic.add_wavelength(value=0.55, is_primary=True)
    return optic


def test_optic_viewer_backward_compat():
    """OpticViewer.view() must still return a 3-tuple (fig, ax, interaction_manager)."""
    import matplotlib.pyplot as plt

    from optiland.visualization.system.optic_viewer import OpticViewer

    optic = _make_simple_optic()
    viewer = OpticViewer(optic)
    result = viewer.view()
    assert len(result) == 3
    fig, ax, im = result
    plt.close(fig)
