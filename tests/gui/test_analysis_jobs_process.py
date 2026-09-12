"""Actual isolated numerical work while Qt keeps processing interactions."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from unittest.mock import Mock

from PySide6.QtCore import QTimer

from optiland_gui.analysis_panel import AnalysisPanel
from optiland_gui.services.analysis_runner import AnalysisRunner
from optiland_gui.services.calculation_jobs import CalculationJobs, DocumentState


def wait_for(qapp, condition, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qapp.processEvents()
        if condition():
            return
        time.sleep(0.002)
    raise AssertionError("Analysis job did not reach its expected state")


def test_real_huygens_stop_and_fft_result_keep_qt_alive(
    qapp, minimal_optic, record_property
):
    document = DocumentState()
    jobs = CalculationJobs(document, cancel_grace_ms=300)
    connector = SimpleNamespace(
        document_state=document,
        calculation_jobs=jobs,
        get_optic=lambda: minimal_optic,
        toast_manager=Mock(),
        get_field_options=lambda: [("All", "all")],
        get_wavelength_options=lambda: [("Primary", "primary")],
    )
    connector._analysis_runner = AnalysisRunner(connector)
    panel = AnalysisPanel(connector)
    panel.resize(1000, 750)
    panel.show()
    beats = []
    timer = QTimer()
    timer.setInterval(10)
    timer.timeout.connect(lambda: beats.append(time.monotonic()))
    timer.start()
    terminal = []
    stages = []
    jobs.finished.connect(terminal.append)
    jobs.progress.connect(lambda request, data: stages.append(data["stage"]))
    try:
        page = panel._execute_analysis(
            None, "Huygens PSF", {"num_rays": 128, "image_size": 512}, {}
        )
        panel.switch_plot_page(0)
        assert page is not None
        wait_for(qapp, lambda: "Calculating Huygens PSF" in stages)
        before_stop = time.monotonic()
        panel.btnStop.click()
        assert time.monotonic() - before_stop < 0.1
        wait_for(qapp, lambda: terminal, timeout=5)
        assert terminal[0].status == "cancelled"
        record_property("cancel_seconds", time.monotonic() - before_stop)
        assert time.monotonic() - before_stop < 3
        assert page["state"] == "cancelled"
        fft = panel._execute_analysis(
            None, "FFT PSF", {"num_rays": 32, "grid_size": 64}, {}
        )
        panel.switch_plot_page(1)
        wait_for(qapp, lambda: fft.get("prepared"))
        assert terminal[-1].status == "succeeded"
        assert len(terminal) == 2
        assert len(beats) > 20
        assert max(b - a for a, b in zip(beats, beats[1:], strict=False)) < 0.75
        record_property(
            "max_heartbeat_gap_ms",
            1000 * max(b - a for a, b in zip(beats, beats[1:], strict=False)),
        )
        record_property("heartbeat_ticks", len(beats))
        # Theme and page operations operate on retained arrays and remain usable.
        panel.update_theme("light")
        panel.update_theme("dark")
        panel._clone_analysis_page(1)
        assert len(terminal) == 2
        capture = os.environ.get("OPTILAND_GUI005_CAPTURE")
        if capture:
            qapp.processEvents()
            assert panel.grab().save(capture)
    finally:
        timer.stop()
        panel.close()
        jobs.shutdown()
        wait_for(qapp, lambda: jobs._process is None, timeout=5)
        panel.deleteLater()
        qapp.processEvents()
