"""Analysis page identity, cancellation, stale results and presentation isolation."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from optiland_gui.analysis_panel import AnalysisPanel
from tests.gui.analysis_job_fakes import connector_for, result_data


@pytest.fixture()
def panel(minimal_optic, qapp):
    connector = connector_for(minimal_optic)
    panel = AnalysisPanel(connector)
    yield panel
    panel.close()
    panel.deleteLater()
    qapp.processEvents()


def run_page(panel, name="Ray Fan"):
    page = panel._execute_analysis(
        panel._analysis_class_map[name], name, {"num_points": 16}, {}
    )
    assert page is not None
    panel.switch_plot_page(len(panel.analysis_results_pages) - 1)
    return page


def test_pending_page_immediate_and_late_result_does_not_steal_focus(panel):
    first = run_page(panel)
    second = run_page(panel)
    assert panel.current_plot_page_index == 1
    panel.connector.calculation_jobs.complete(first["page_id"], result_data())
    assert panel.current_plot_page_index == 1
    assert first["prepared"]
    assert second["prepared"] is None


def test_completion_preserves_keyboard_focus_in_settings(panel, qapp):
    page = run_page(panel)
    panel.show()
    panel.settings_area_widget.show()
    panel.activateWindow()
    qapp.processEvents()
    setting = panel.current_settings_widgets["num_points"]
    setting.setFocus()
    qapp.processEvents()
    assert setting.hasFocus()
    panel.connector.calculation_jobs.complete(page["page_id"], result_data())
    qapp.processEvents()
    assert setting.hasFocus()


def test_retains_result_during_rerun_cancel_and_error(panel):
    page = run_page(panel)
    jobs = panel.connector.calculation_jobs
    jobs.complete(page["page_id"], result_data())
    prepared = page["prepared"]
    panel._execute_analysis(None, page["name"], {"num_points": 32}, {}, page=page)
    assert page["prepared"] is prepared
    panel.stop_analysis_slot()
    assert page["state"] == "cancelled"
    assert page["prepared"] is prepared
    panel._execute_analysis(None, page["name"], {"num_points": 64}, {}, page=page)
    jobs.complete(page["page_id"], status="failed", error="Bad optical system")
    assert page["prepared"] is prepared
    panel.connector.toast_manager.notify.assert_called()


def test_outdated_result_labeled_and_theme_never_recalculates(panel):
    page = run_page(panel)
    panel.connector.document_state.change()
    panel.connector.calculation_jobs.complete(page["page_id"], result_data())
    assert "Out of date" in panel.dataInfoLabel.text()
    with patch.object(panel.runner, "run", side_effect=AssertionError("recalculation")):
        panel.update_theme("light")
        panel.update_theme("dark")
        panel.switch_plot_page(0)
    assert "Out of date" in panel.dataInfoLabel.text()


def test_clone_shares_completed_data_and_has_independent_identity(panel):
    page = run_page(panel)
    panel.connector.calculation_jobs.complete(page["page_id"], result_data())
    panel._clone_analysis_page(0)
    cloned = panel.analysis_results_pages[1]
    assert cloned["page_id"] != page["page_id"]
    assert cloned["prepared"] is page["prepared"]
    cloned["constructor_args_used"]["num_points"] = 32
    assert page["constructor_args_used"]["num_points"] == 16


def test_remove_active_page_revokes_target_before_completion(panel):
    page = run_page(panel)
    panel._remove_analysis_page(0)
    assert page["page_id"] in panel.connector.calculation_jobs.cancelled
    assert panel.analysis_results_pages == []
    assert not panel.btnRunAll.isEnabled()


def test_result_retention_is_bounded_without_dropping_previous_data(panel):
    page = run_page(panel)
    jobs = panel.connector.calculation_jobs
    jobs.complete(page["page_id"], result_data())
    retained = page["prepared"]
    panel._execute_analysis(None, page["name"], {"num_points": 32}, {}, page=page)
    oversized = result_data()
    oversized["size_bytes"] = panel._result_budget + 1
    jobs.complete(page["page_id"], oversized)
    assert page["state"] == "failed"
    assert page["prepared"] is retained


def test_empty_system_uses_toast(panel):
    from optiland.optic import Optic

    assert not panel._validate_system_for_analysis(Optic())
    panel.connector.toast_manager.notify.assert_called()


def test_optional_fft_settings_remain_auto_on_reopen(panel):
    page = panel._execute_analysis(None, "FFT PSF", {"num_rays": 32}, {})
    panel.switch_plot_page(0)
    grid = panel.current_settings_widgets["grid_size"]
    assert grid.specialValueText() == "Auto"
    assert panel._get_value_from_spinbox(grid) is None
    panel._apply_settings_and_rerun_analysis_slot()
    assert page["constructor_args_used"]["grid_size"] is None


def test_dirty_settings_survive_completion_and_theme_change(panel):
    page = run_page(panel)
    widget = panel.current_settings_widgets["num_points"]
    widget.setValue(48)
    assert page["settings_dirty"]
    panel.connector.calculation_jobs.complete(page["page_id"], result_data())
    assert widget.value() == 48
    assert "Settings changed" in panel.dataInfoLabel.text()
    panel.update_theme("light")
    assert widget.value() == 48


def test_intentional_stop_does_not_show_worker_termination_error(panel):
    page = run_page(panel)
    panel.connector.calculation_jobs.complete(
        page["page_id"],
        status="cancelled",
        error="Calculation worker exited unexpectedly.",
    )
    assert page["error"] == ""
    panel.connector.toast_manager.notify.assert_not_called()


def test_backend_change_labels_retained_result_outdated(panel):
    from optiland_gui.services.job_records import BackendConfig

    page = run_page(panel)
    panel.connector.calculation_jobs.complete(page["page_id"], result_data())
    with patch.object(BackendConfig, "capture", return_value=BackendConfig("torch")):
        panel._update_page_status(page)
        assert "Out of date" in panel.dataInfoLabel.text()


def test_settings_draft_survives_switch_and_closing_another_page(panel):
    first, second = run_page(panel), run_page(panel)
    jobs = panel.connector.calculation_jobs
    jobs.complete(first["page_id"], result_data())
    jobs.complete(second["page_id"], result_data())
    panel.switch_plot_page(0)
    panel.current_settings_widgets["num_points"].setValue(48)
    panel.switch_plot_page(1)
    panel.switch_plot_page(0)
    assert panel.current_settings_widgets["num_points"].value() == 48
    panel.current_settings_widgets["num_points"].setValue(64)
    panel._remove_analysis_page(1)
    assert panel.current_settings_widgets["num_points"].value() == 64
    panel.analysisTypeCombo.setCurrentText("Spot Diagram")
    panel.analysisTypeCombo.setCurrentText("Ray Fan")
    assert panel.current_settings_widgets["num_points"].value() == 64
    assert first["settings_dirty"]
    assert first["constructor_args_used"]["num_points"] == 16
    assert first["result_settings"]["constructor_args"]["num_points"] == 16
    assert len(jobs.submissions) == 2


def test_clone_copies_pending_draft_without_recalculating_or_sharing_edits(panel):
    page = run_page(panel)
    jobs = panel.connector.calculation_jobs
    jobs.complete(page["page_id"], result_data())
    panel.current_settings_widgets["num_points"].setValue(48)
    panel._clone_analysis_page(0)
    clone = panel.analysis_results_pages[1]
    assert panel.current_settings_widgets["num_points"].value() == 48
    assert clone["prepared"] is page["prepared"]
    assert clone["draft_settings"] is not page["draft_settings"]
    assert clone["settings_dirty"]
    panel.current_settings_widgets["num_points"].setValue(64)
    panel.switch_plot_page(0)
    assert panel.current_settings_widgets["num_points"].value() == 48
    panel.switch_plot_page(1)
    assert panel.current_settings_widgets["num_points"].value() == 64
    assert len(jobs.submissions) == 1


def test_apply_submits_restored_draft_but_completion_keeps_later_edits(panel):
    first, second = run_page(panel), run_page(panel)
    jobs = panel.connector.calculation_jobs
    jobs.complete(first["page_id"], result_data())
    jobs.complete(second["page_id"], result_data())
    panel.switch_plot_page(0)
    panel.current_settings_widgets["num_points"].setValue(48)
    panel.switch_plot_page(1)
    panel.switch_plot_page(0)
    panel._apply_settings_and_rerun_analysis_slot()
    assert jobs.submissions[-1].parameters["constructor_args"]["num_points"] == 48
    assert first["constructor_args_used"]["num_points"] == 48
    assert first["draft_settings"] is None
    assert not first["settings_dirty"]
    panel.current_settings_widgets["num_points"].setValue(64)
    jobs.complete(first["page_id"], result_data())
    panel.switch_plot_page(1)
    panel.switch_plot_page(0)
    assert panel.current_settings_widgets["num_points"].value() == 64
    assert first["settings_dirty"]
    assert first["result_settings"]["constructor_args"]["num_points"] == 48
    assert len(jobs.submissions) == 3


def test_incomplete_text_and_control_drafts_survive_navigation(panel):
    page = panel._execute_analysis(None, "FFT PSF", {"num_rays": 32}, {})
    run_page(panel)
    panel.switch_plot_page(0)
    field = panel.current_settings_widgets["field"]
    field.setText("0,")
    field.textEdited.emit("0,")
    panel.current_settings_widgets["remove_tilt"].setChecked(True)
    panel.current_settings_widgets["strategy"].setCurrentText("centroid")
    panel.current_settings_widgets["grid_size"].setValue(0)
    panel.switch_plot_page(1)
    panel.switch_plot_page(0)
    assert panel.current_settings_widgets["field"].text() == "0,"
    assert panel.current_settings_widgets["remove_tilt"].isChecked()
    assert panel.current_settings_widgets["strategy"].currentText() == "centroid"
    assert (
        panel._get_value_from_spinbox(panel.current_settings_widgets["grid_size"])
        is None
    )
    assert page["settings_dirty"]
    panel._apply_settings_and_rerun_analysis_slot()
    assert panel.current_settings_widgets["field"].text() == "0,"
    assert page["settings_dirty"]
    assert len(panel.connector.calculation_jobs.submissions) == 2


def test_numeric_wavelength_survives_page_restoration(panel):
    panel.connector.get_wavelength_options = lambda: [
        ("all", "'all'"),
        ("primary", "'primary'"),
        ("0.4500 µm", "[0.45]"),
    ]
    page = panel._execute_analysis(
        None, "FFT PSF", {"wavelength": 0.45, "num_rays": 32}, {}
    )
    panel.switch_plot_page(0)
    assert panel._collect_current_settings()[0]["wavelength"] == 0.45
    panel._apply_settings_and_rerun_analysis_slot()
    assert page["constructor_args_used"]["wavelength"] == 0.45


def test_selected_field_survives_json_settings_round_trip(panel):
    panel.connector.get_field_options = lambda: [
        ("all", "'all'"),
        ("Field 2", "[(0.0, 0.5)]"),
    ]
    page = run_page(panel)
    panel.current_settings_widgets["fields"].setCurrentIndex(1)
    args, view = panel._collect_current_settings()
    saved = json.loads(
        json.dumps(
            {"analysis_name": "Ray Fan", "constructor_args": args, "view_args": view}
        )
    )
    panel.current_settings_widgets["fields"].setCurrentIndex(0)
    panel._apply_loaded_settings_to_ui(saved)
    assert panel._collect_current_settings()[0]["fields"] == [(0.0, 0.5)]
    panel._apply_settings_and_rerun_analysis_slot()
    assert page["constructor_args_used"]["fields"] == [(0.0, 0.5)]


def test_loaded_json_settings_restore_coordinates_choices_and_pending_draft(
    panel, tmp_path, monkeypatch
):
    from PySide6.QtWidgets import QFileDialog

    page = panel._execute_analysis(None, "FFT PSF", {"num_rays": 32}, {})
    panel.switch_plot_page(0)
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "analysis_name": "FFT PSF",
                "constructor_args": {
                    "field": [0.2, 0.3],
                    "strategy": "centroid",
                    "grid_size": None,
                    "remove_tilt": True,
                    "num_rays": 48,
                },
                "view_args": {},
            }
        )
    )
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
    panel._load_analysis_settings_slot()
    assert panel._validate_all_inputs() == (True, "")
    args, _ = panel._collect_current_settings()
    assert args["field"] == (0.2, 0.3)
    assert args["strategy"] == "centroid" and args["remove_tilt"]
    assert args["num_rays"] == 48
    assert "grid_size" not in args
    assert page["settings_dirty"]
    panel.switch_plot_page(0)
    assert panel.current_settings_widgets["field"].text() == "0.2, 0.3"
    assert len(panel.connector.calculation_jobs.submissions) == 1


def test_loading_unknown_analysis_keeps_current_settings(panel):
    run_page(panel)
    with pytest.raises(ValueError, match="supported analysis"):
        panel._apply_loaded_settings_to_ui({"analysis_name": "Unknown analysis"})
    assert panel.current_settings_widgets["num_points"].value() == 16
