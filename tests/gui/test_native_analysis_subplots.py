"""Detached subplot grids remain readable and interactive in a compact Qt dock."""

from __future__ import annotations

import io
import itertools
import threading

import pytest
from matplotlib.figure import Figure
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest

from optiland.samples.objectives import CookeTriplet, DoubleGauss
from optiland_gui.analysis_panel import AnalysisPanel, CustomMatplotlibToolbar
from optiland_gui.services.analysis_plots import (
    capture_analysis_plot,
    draw_analysis_plot,
)
from optiland_gui.services.analysis_worker import prepare_analysis
from optiland_gui.services.job_records import OpticSnapshot
from optiland_gui.widgets.analysis_canvas_viewport import AnalysisCanvasViewport
from optiland_gui.widgets.retained_analysis_canvas import RetainedAnalysisCanvas
from tests.gui.analysis_job_fakes import connector_for, result_data
from tests.gui.test_calculation_jobs import wait_for


@pytest.fixture(scope="module")
def optical_plots():
    plots = {}
    for name, optic in (
        ("Ray Fan", CookeTriplet),
        ("Best-Fit Ray Fan", CookeTriplet),
        ("Pupil Aberration", CookeTriplet),
        ("Spot Diagram", DoubleGauss),
        ("Through-Focus Spot", DoubleGauss),
    ):
        plots[name] = prepare_analysis(
            OpticSnapshot.capture(optic()),
            dict(name=name, constructor_args={}, view_args={}, theme="dark"),
            lambda *_: None,
            threading.Event(),
        )["plot"]
    return plots


def settle(app, canvas):
    previous = None

    def ready():
        nonlocal previous
        renderer = getattr(canvas, "renderer", None)
        geometry = (canvas.width(), canvas.height(), id(renderer))
        stable = geometry == previous
        previous = geometry
        return (
            stable
            and renderer is not None
            and not canvas._draw_pending
            and not canvas._resize_quiet.isActive()
            and not canvas._resize_progress.isActive()
            and not canvas._resize_followup.isActive()
            and abs(renderer.width - canvas.width() * canvas.device_pixel_ratio) < 1e-6
            and abs(renderer.height - canvas.height() * canvas.device_pixel_ratio)
            < 1e-6
        )

    wait_for(app, ready)


def assert_readable(canvas):
    figure = canvas.figure
    artists = list(figure.legends)
    for ax in figure.axes:
        artists.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
        for axis in (ax.xaxis, ax.yaxis):
            artists.append(axis.get_offset_text())
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks():
                if lo - 1e-8 <= tick.get_loc() <= hi + 1e-8:
                    artists.extend((tick.label1, tick.label2))
    boxes = [
        (a, a.get_window_extent(canvas.get_renderer()))
        for a in artists
        if a.get_visible() and (not hasattr(a, "get_text") or a.get_text())
    ]
    for artist, b in boxes:
        assert b.x0 >= -0.5 and b.y0 >= -0.5, (artist, b.bounds)
        assert b.x1 <= figure.bbox.width + 0.5 and b.y1 <= figure.bbox.height + 0.5, (
            artist,
            b.bounds,
        )
    for (a, left), (b, right) in itertools.combinations(boxes, 2):
        assert (
            min(left.x1, right.x1) - max(left.x0, right.x0) <= 0.5
            or min(left.y1, right.y1) - max(left.y0, right.y0) <= 0.5
        ), (a, b)


@pytest.fixture
def make_view(qapp, optical_plots):
    views = []

    def create(name="Ray Fan", theme="dark"):
        data = optical_plots[name]
        canvas = RetainedAnalysisCanvas(Figure(dpi=100))
        draw_analysis_plot(canvas.figure, data, theme)
        grid = data["subplot_grid"]
        view = AnalysisCanvasViewport(canvas, grid["rows"], grid["columns"])
        views.append(view)
        view.resize(430, 164)
        view.show()
        settle(qapp, canvas)
        return view, canvas

    yield create
    for view in views:
        view.close()
        view.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.mark.parametrize(
    "name",
    [
        "Ray Fan",
        "Best-Fit Ray Fan",
        "Pupil Aberration",
        "Spot Diagram",
        "Through-Focus Spot",
    ],
)
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_optical_grid_fits_full_canvas(make_view, name, theme, qapp):
    view, canvas = make_view(name, theme)
    for w, h in [(430, 164), (220, 164), (573, 240), (921, 620)]:
        view.resize(w, h)
        settle(qapp, canvas)
        assert_readable(canvas)
        for ax in canvas.figure.axes:
            if ax.get_aspect() != "auto":
                assert ax.bbox.height / ax.bbox.width == pytest.approx(
                    ax.get_aspect() * ax.get_data_ratio()
                )


@pytest.mark.parametrize(
    "kind",
    [
        "manual",
        "moved",
        "nested",
        "missing",
        "spanning",
        "ratios",
        "grid margins",
        "zero spacing",
        "separate",
        "duplicate cell",
        "figure text",
        "anchored legend",
        "two legends",
        "hidden",
        "invisible axes",
        "distant legend",
    ],
)
def test_unsupported_geometry_keeps_original_rectangles(kind):
    figure = Figure()
    grid = figure.add_gridspec(1, 3 if kind in ("missing", "spanning") else 2)
    if kind == "nested":
        grid = grid[0].subgridspec(1, 2)
    a = figure.add_subplot(grid[0])
    b = figure.add_subplot(grid[1:] if kind == "spanning" else grid[1])
    if kind == "manual":
        figure.add_axes((0.1, 0.2, 0.3, 0.2))
    elif kind == "moved":
        a.set_position((0.1, 0.2, 0.3, 0.2))
    elif kind == "ratios":
        grid.set_width_ratios([1, 2])
    elif kind == "grid margins":
        grid.update(left=0.2, right=0.7)
    elif kind == "zero spacing":
        grid.update(wspace=0)
    elif kind == "separate":
        b.set_subplotspec(figure.add_gridspec(1, 2)[1])
    elif kind == "duplicate cell":
        b.set_subplotspec(grid[0])
    elif kind == "figure text":
        figure.text(0.3, 0.4, "Optical reference")
    elif kind == "anchored legend":
        figure.legend([], [], loc="upper right")
    elif kind == "two legends":
        for _ in range(2):
            figure.legend([], [])
    elif kind == "hidden":
        b.set_axis_off()
    elif kind == "invisible axes":
        b.set_visible(False)
    elif kind == "distant legend":
        figure.legend([], [], loc="lower center", bbox_to_anchor=(0.5, -0.6))
    data = capture_analysis_plot(figure)
    assert data["subplot_grid"] is None
    restored = Figure()
    draw_analysis_plot(restored, data)
    for ax, original in zip(restored.axes, data["axes"], strict=True):
        assert ax.get_position().bounds == pytest.approx(original["position"])
    assert restored.axes[1].get_visible() == b.get_visible()
    assert restored.axes[1].axison == b.axison
    if kind == "distant legend":
        actual = (
            restored.legends[0]
            .get_bbox_to_anchor()
            .transformed(restored.transFigure.inverted())
        )
        assert actual.bounds == pytest.approx((0.5, -0.6, 0, 0))


def test_subplot_capture_preserves_cell_order_and_releases_manual_layout():
    original = Figure()
    grid = original.add_gridspec(1, 2)
    for i in (1, 0):
        original.add_subplot(grid[i])
    restored = Figure()
    payload = capture_analysis_plot(original)
    assert payload["subplot_grid"]["cells"] == [1, 0]
    draw_analysis_plot(restored, payload)
    assert [ax.get_subplotspec().num1 for ax in restored.axes] == [1, 0]
    original.axes[0].set_position((0.15, 0.25, 0.25, 0.3))
    draw_analysis_plot(restored, capture_analysis_plot(original))
    assert not restored.get_constrained_layout()
    assert all(ax.get_subplotspec() is None for ax in restored.axes)


def test_analysis_page_owns_scrollable_canvas_and_replaces_it(
    minimal_optic, qapp, optical_plots
):
    from shiboken6 import isValid

    panel = AnalysisPanel(connector_for(minimal_optic))
    try:
        page = panel._execute_analysis(
            panel._analysis_class_map["Ray Fan"], "Ray Fan", {}, {}
        )
        panel.switch_plot_page(0)
        result = result_data()
        result["plot"] = optical_plots["Ray Fan"]
        panel.connector.calculation_jobs.complete(page["page_id"], result)
        canvas = panel.active_mpl_canvas_widget
        view = panel.findChild(AnalysisCanvasViewport)
        assert (
            view.widget() is canvas and panel.active_mpl_toolbar_widget.canvas is canvas
        )
        panel.switch_plot_page(0)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert not isValid(view) and not isValid(canvas)
        assert (
            panel.findChild(AnalysisCanvasViewport).widget()
            is panel.active_mpl_canvas_widget
        )
    finally:
        panel.close()
        panel.deleteLater()


def test_wheel_zoom_and_manual_position_survive_resizing(make_view, qapp):
    from matplotlib.backend_bases import MouseEvent

    from optiland_gui.gui_plot_utils import handle_matplotlib_scroll_zoom

    view, canvas = make_view()
    ax, other = canvas.figure.axes[:2]
    before, independent = ax.get_xlim(), other.get_xlim()
    canvas.mpl_connect("scroll_event", handle_matplotlib_scroll_zoom)
    x, y = ax.transData.transform((0, 0))
    canvas.callbacks.process(
        "scroll_event", MouseEvent("scroll_event", canvas, x, y, button="up", step=1)
    )
    assert ax.get_xlim() != before and other.get_xlim() == independent
    assert view.verticalScrollBar().value() == 0
    position = (0.1, 0.2, 0.2, 0.3)
    ax.set_position(position)
    legend = canvas.figure.legends[0]
    legend.set_bbox_to_anchor((0.5, 0.4), canvas.figure.transFigure)
    view.resize(921, 620)
    settle(qapp, canvas)
    assert ax.get_position().bounds == pytest.approx(position)
    assert legend.get_bbox_to_anchor().transformed(
        canvas.figure.transFigure.inverted()
    ).bounds == pytest.approx((0.5, 0.4, 0, 0))


@pytest.mark.parametrize("ext", ["png", "svg", "pdf"])
def test_navigation_and_full_canvas_export(make_view, qapp, ext):
    view, canvas = make_view()
    toolbar = CustomMatplotlibToolbar(canvas, view)
    ax = canvas.figure.axes[0]
    home = ax.get_xlim()
    toolbar.push_current()
    ax.set_xlim(-0.5, 0.5)
    toolbar.push_current()
    toolbar.back()
    assert ax.get_xlim() == home
    toolbar.forward()
    assert ax.get_xlim() == (-0.5, 0.5)
    toolbar.home()
    assert ax.get_xlim() == home
    settle(qapp, canvas)
    view.setFocus()
    QTest.keyClick(view, Qt.Key.Key_PageDown)
    position = view.verticalScrollBar().value()
    assert position > 0
    output = io.BytesIO()
    size = canvas.figure.get_size_inches().copy()
    canvas.print_figure(output, format=ext, dpi=200)
    if ext == "png":
        import struct

        assert struct.unpack(">II", output.getvalue()[16:24]) == pytest.approx(
            size * 200, abs=1
        )
    else:
        assert output.getvalue().startswith(b"<?xml" if ext == "svg" else b"%PDF")
    settle(qapp, canvas)
    assert view.verticalScrollBar().value() == position
    assert_readable(canvas)


@pytest.mark.parametrize("deferred", [False, True])
def test_reopen_draws_at_current_size(make_view, qapp, deferred):
    view, canvas = make_view()
    canvas._EXPENSIVE_DRAW_MS = 0 if deferred else float("inf")
    draws = []
    canvas.mpl_connect("draw_event", lambda event: draws.append(canvas.isVisible()))
    view.resize(canvas.width() + 100, canvas.height() + 100)
    assert canvas._resize_quiet.isActive() if deferred else canvas._draw_pending
    assert view.close()
    QTest.qWait(300)
    assert draws == []
    view.show()
    settle(qapp, canvas)
    assert draws and all(draws)
    assert_readable(canvas)
