"""Resizing retained analysis results must not schedule duplicate repaint work."""

from __future__ import annotations

import hashlib
import pickle
import threading
from math import ceil

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtTest import QTest
from shiboken6 import isValid

from optiland_gui.analysis_panel import AnalysisPanel
from optiland_gui.services.analysis_worker import prepare_analysis
from optiland_gui.services.job_records import OpticSnapshot
from optiland_gui.widgets.analysis_figure_layout import AnalysisFigureLayout
from tests.gui.analysis_job_fakes import connector_for
from tests.gui.test_calculation_jobs import wait_for


@pytest.fixture
def retained_panel(qapp, minimal_optic):
    connector = connector_for(minimal_optic)
    data = prepare_analysis(
        OpticSnapshot.capture(minimal_optic),
        {
            "name": "Ray Fan",
            "constructor_args": {"num_points": 16},
            "view_args": {},
            "theme": "dark",
        },
        lambda *_args, **_kwargs: None,
        threading.Event(),
    )
    panel = AnalysisPanel(connector)
    page = panel._execute_analysis(
        panel._analysis_class_map["Ray Fan"], "Ray Fan", {"num_points": 16}, {}
    )
    panel.switch_plot_page(0)
    connector.calculation_jobs.complete(page["page_id"], data)
    panel.resize(900, 640)
    panel.show()
    QTest.qWait(350)  # Initial result fade and native canvas resize settle.
    yield panel, connector, page
    panel.close()
    panel.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def digest(data):
    return hashlib.sha256(pickle.dumps(data, protocol=5)).hexdigest()


def assert_final_pixels(canvas):
    physical_size = (
        np.array([canvas.width(), canvas.height()]) * canvas.device_pixel_ratio
    )
    np.testing.assert_allclose(
        canvas.figure.get_size_inches() * canvas.figure.dpi,
        physical_size,
        rtol=0,
        atol=1e-8,
    )
    width, height = canvas.get_width_height(physical=True)
    # Fractional DPR geometry can be rounded to raster bounds by the renderer.
    assert (int(canvas.renderer.width), int(canvas.renderer.height)) == (width, height)
    assert canvas.buffer_rgba().shape[:2] == (height, width)


def wait_for_completed_resize(qapp, canvas, draws):
    def final_state():
        physical_size = (
            np.array([canvas.width(), canvas.height()]) * canvas.device_pixel_ratio
        )
        engine = canvas.figure.get_layout_engine()
        minimum = (0, 0)
        if isinstance(engine, AnalysisFigureLayout):
            scale = canvas.figure.dpi / canvas.device_pixel_ratio
            minimum = (
                ceil(engine.minimum_width_inches * scale),
                ceil(engine.minimum_height_inches * scale),
            )
        return (
            bool(draws)
            and hasattr(canvas, "renderer")
            and (canvas.minimumWidth(), canvas.minimumHeight()) == minimum
            and canvas._resize_frame is None
            and not canvas._resize_quiet.isActive()
            and not canvas._resize_progress.isActive()
            and np.allclose(
                canvas.figure.get_size_inches() * canvas.figure.dpi,
                physical_size,
                rtol=0,
                atol=1e-8,
            )
            and (
                int(canvas.renderer.width),
                int(canvas.renderer.height),
            )
            == canvas.get_width_height(physical=True)
        )

    def completed():
        # A first draw can queue a viewport minimum change and a second resize.
        # Dispatch that geometry before deciding whether its pixels are final.
        qapp.processEvents()
        if not final_state():
            return False
        qapp.processEvents()
        return final_state()

    wait_for(qapp, completed)
    assert_final_pixels(canvas)


def test_completed_native_resize_does_not_repaint_again_after_idle(
    qapp, retained_panel
):
    panel, connector, page = retained_panel
    canvas = panel.active_mpl_canvas_widget
    draws = []
    canvas.mpl_connect(
        "draw_event", lambda _event: draws.append(canvas.get_width_height())
    )
    before_requests = len(connector.calculation_jobs.submissions)
    before_data = digest(page["prepared"])

    panel.resize(panel.width() + 60, panel.height() + 30)
    wait_for_completed_resize(qapp, canvas, draws)
    completed = len(draws)
    QTest.qWait(180)
    assert len(draws) == completed
    assert panel.active_mpl_canvas_widget is canvas
    assert len(connector.calculation_jobs.submissions) == before_requests
    assert digest(page["prepared"]) == before_data


def test_rapid_resize_coalesces_and_keeps_final_results(qapp, retained_panel):
    panel, connector, page = retained_panel
    canvas = panel.active_mpl_canvas_widget
    draws = []
    canvas.mpl_connect(
        "draw_event", lambda _event: draws.append(canvas.get_width_height())
    )
    before_requests = len(connector.calculation_jobs.submissions)
    before_data = digest(page["prepared"])
    for index in range(12):
        panel.resize(940 + index * 4, 650 + index * 3)
    wait_for_completed_resize(qapp, canvas, draws)
    assert len(draws) < 12
    completed = len(draws)
    QTest.qWait(180)
    assert len(draws) == completed
    assert_final_pixels(canvas)
    assert panel.active_mpl_canvas_widget is canvas
    assert len(connector.calculation_jobs.submissions) == before_requests
    assert digest(page["prepared"]) == before_data


def test_hidden_resize_reopens_at_final_size_without_reanalysis(qapp, retained_panel):
    panel, connector, page = retained_panel
    canvas = panel.active_mpl_canvas_widget
    before_requests = len(connector.calculation_jobs.submissions)
    before_data = digest(page["prepared"])
    panel.hide()
    panel.resize(1080, 760)
    QTest.qWait(180)
    draws = []
    canvas.mpl_connect(
        "draw_event", lambda _event: draws.append(canvas.get_width_height())
    )
    panel.show()
    wait_for_completed_resize(qapp, canvas, draws)
    assert panel.active_mpl_canvas_widget is canvas
    assert len(connector.calculation_jobs.submissions) == before_requests
    assert digest(page["prepared"]) == before_data


def test_pending_resize_can_be_destroyed_without_late_panel_callback(
    qapp, retained_panel
):
    panel, connector, _ = retained_panel
    canvas = panel.active_mpl_canvas_widget
    before_requests = len(connector.calculation_jobs.submissions)
    panel.resize(1000, 700)
    panel.hide()
    # Closing/replacing an analysis canvas must not leave a delayed panel
    # callback dereferencing its destroyed Qt owner.
    panel._cleanup_plot_area()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QTest.qWait(180)
    assert panel.active_mpl_canvas_widget is None
    assert not isValid(canvas)
    assert len(connector.calculation_jobs.submissions) == before_requests
