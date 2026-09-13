"""Numerical preparation, rendering boundary and visible-view scheduling tests."""

from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np
import pytest

import optiland.backend as be
from optiland_gui.services.job_records import BackendConfig, OpticSnapshot
from optiland_gui.services.layout_tasks import prepare_2d, prepare_3d, prepare_sag
from tests.gui.test_calculation_jobs import wait_for


def progress(*args):
    pass


def assert_plain_data(data):
    if isinstance(data, dict):
        for key, value in data.items():
            assert isinstance(key, str | int)
            assert_plain_data(value)
    elif isinstance(data, tuple | list):
        for value in data:
            assert_plain_data(value)
    else:
        assert isinstance(data, str | int | float | np.ndarray | type(None))


def test_2d_payload_matches_core_plot_and_keeps_ownership(minimal_optic):
    from matplotlib.figure import Figure

    from optiland.visualization.system.rays import Rays2D
    from optiland.visualization.system.system import OpticalSystem

    snapshot = OpticSnapshot.capture(minimal_optic)
    data = prepare_2d(
        snapshot, {"num_rays": 5, "distribution": "line_y"}, progress, threading.Event()
    )
    assert_plain_data(data)
    optic = snapshot.restore()
    axes = Figure().add_subplot()
    rays = Rays2D(optic)
    rays.plot(axes, num_rays=5)
    OpticalSystem(optic, rays).plot(axes)
    lines = [p for p in data["primitives"] if p["kind"] == "line"]
    polygons = [p for p in data["primitives"] if p["kind"] == "polygon"]
    assert len(lines) == len(axes.lines)
    assert len(polygons) == len(axes.patches)
    for expected, actual in zip(axes.lines, lines, strict=True):
        np.testing.assert_allclose(actual["xy"], expected.get_xydata())
    for expected, actual in zip(axes.patches, polygons, strict=True):
        np.testing.assert_allclose(actual["xy"], expected.get_xy())
        assert actual["surfaces"] == (1, 2)
    assert set(data["boundaries"]) == {1, 2}


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_prepared_sag_and_mesh_have_owned_backend_independent_arrays(
    minimal_optic, backend
):
    from optiland.physical_apertures import RectangularAperture

    if backend == "torch":
        pytest.importorskip("torch")
    minimal_optic.surfaces[1].aperture = RectangularAperture(-4, 4, -3, 3)
    snapshot = replace(
        OpticSnapshot.capture(minimal_optic), backend=BackendConfig(backend)
    )
    try:
        mesh = prepare_3d(snapshot, {}, progress, threading.Event())
        sag = prepare_sag(
            snapshot,
            {
                "surface_index": 1,
                "max_extent": 5,
                "x_cross_section": 0.2,
                "y_cross_section": 0.3,
            },
            progress,
            threading.Event(),
        )
        assert_plain_data(mesh)
        assert_plain_data(sag)
        assert len(mesh["meshes"]) > 0
        assert sag["sag"].shape == (50, 50)
        optic = snapshot.restore()
        x, y = be.meshgrid(be.array(sag["coordinates"]), be.array(sag["coordinates"]))
        np.testing.assert_allclose(
            sag["sag"], be.to_numpy(optic.surfaces[1].geometry.sag(x, y))
        )
    finally:
        be.set_backend("numpy")


def test_snapshot_preserves_coating_pickup_solve_and_transform(minimal_optic):
    from optiland.coatings import PolarizerCoating
    from optiland.rays import PolarizationState
    from optiland.solves import MarginalRayHeightThicknessSolve

    minimal_optic.polarization = PolarizationState(
        is_polarized=True, Ex=1, Ey=1, phase_x=0.2, phase_y=0.3
    )
    surface = minimal_optic.surfaces[1]
    surface.interaction_model.coating = PolarizerCoating()
    surface.geometry.cs.rx = 0.1
    surface.geometry.cs.y = 0.2
    minimal_optic.pickups.add(1, "conic", 2, scale=1, offset=0.2)
    minimal_optic.solves.solves.append(
        MarginalRayHeightThicknessSolve(minimal_optic, 2, 0)
    )
    restored = OpticSnapshot.capture(minimal_optic).restore()
    assert restored.pickups.to_dict() == minimal_optic.pickups.to_dict()
    assert restored.solves.to_dict() == minimal_optic.solves.to_dict()
    assert restored.pickups.pickups[0].optic is restored
    assert restored.solves.solves[0].optic is restored
    assert (
        restored.surfaces[1].interaction_model.coating.to_dict()
        == surface.interaction_model.coating.to_dict()
    )
    np.testing.assert_allclose(
        restored.surfaces[1].geometry.cs.rx, surface.geometry.cs.rx
    )
    np.testing.assert_allclose(
        restored.surfaces[1].geometry.cs.y, surface.geometry.cs.y
    )
    np.testing.assert_allclose(restored.polarization.Ex, minimal_optic.polarization.Ex)
    np.testing.assert_allclose(
        restored.polarization.phase_y, minimal_optic.polarization.phase_y
    )


def test_ray_actor_batching_preserves_every_original_polyline(
    minimal_optic, monkeypatch
):
    import optiland_gui.services.layout_tasks as module

    batch = module._batch_ray_meshes
    unbatched = []

    def capture(meshes):
        unbatched.extend(meshes)
        return batch(meshes)

    monkeypatch.setattr(module, "_batch_ray_meshes", capture)
    data = prepare_3d(
        OpticSnapshot.capture(minimal_optic), {}, progress, threading.Event()
    )
    original = [mesh for mesh in unbatched if mesh["role"] == "ray"]
    prepared = [mesh for mesh in data["meshes"] if mesh["role"] == "ray"]
    assert len(prepared) == 1 < len(original)

    def paths(meshes):
        for mesh in meshes:
            offsets, cells = mesh["cells"]["lines"]
            for start, end in zip(offsets[:-1], offsets[1:], strict=True):
                yield mesh["points"][cells[start:end]]

    for before, after in zip(paths(original), paths(prepared), strict=True):
        np.testing.assert_array_equal(before, after)


def test_runtime_custom_geometry_is_rejected_before_data_loss(minimal_optic):
    base = type(minimal_optic.surfaces[1].geometry)

    class ScriptedGeometry(base):
        pass

    minimal_optic.surfaces[1].geometry.__class__ = ScriptedGeometry
    with pytest.raises(ValueError, match="snapshot adapter"):
        OpticSnapshot.capture(minimal_optic)


@pytest.mark.parametrize("aperture_kind", ["rectangle", "ellipse", "none"])
def test_bulk_surface_mesh_preserves_vertices_and_clipped_quad_topology(
    minimal_optic, aperture_kind
):
    from vtk.util.numpy_support import vtk_to_numpy

    from optiland.physical_apertures import EllipticalAperture, RectangularAperture
    from optiland.visualization.system.surface import Surface3D

    surface = minimal_optic.surfaces[1]
    if aperture_kind == "rectangle":
        surface.aperture = RectangularAperture(-4, 3, -2, 5)
    elif aperture_kind == "ellipse":
        surface.aperture = EllipticalAperture(4, 3)
    view = Surface3D(surface, 4)
    x, y, z = (be.to_numpy(a) for a in view._compute_sag_3d())
    actor = view._get_asymmetric_surface()
    data = actor.GetMapper().GetInput()
    np.testing.assert_array_equal(
        vtk_to_numpy(data.GetPoints().GetData()),
        np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32),
    )
    mask = surface.aperture.contains(x, y) if surface.aperture else np.hypot(x, y) <= 4
    # Independent small per-cell construction mirrors VTK quad semantics and
    # catches transposed rows, winding changes, or accidental edge inclusion.
    nrows, ncols = x.shape
    expected = []
    for i in range(nrows - 1):
        for j in range(ncols - 1):
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if all(mask[row, col] for row, col in corners):
                expected.extend(row * ncols + col for row, col in corners)
    cells = data.GetPolys()
    np.testing.assert_array_equal(vtk_to_numpy(cells.GetConnectivityArray()), expected)
    np.testing.assert_array_equal(
        vtk_to_numpy(cells.GetOffsetsArray()), np.arange(0, len(expected) + 1, 4)
    )


def test_only_visible_layout_submits_and_theme_reuses_data(
    qapp, minimal_optic, monkeypatch
):
    import optiland_gui.viewer_panel as module
    from optiland_gui.optiland_connector import OptilandConnector

    # No native OpenGL context is needed to verify 2D/Sag visibility scheduling.
    monkeypatch.setattr(module, "VTK_AVAILABLE", False)
    connector = OptilandConnector()
    connector._optic = minimal_optic
    connector.opticLoaded.emit()
    panel = module.ViewerPanel(connector)
    assert not connector.calculation_jobs.running
    panel.resize(700, 600)
    panel.show()
    try:
        wait_for(qapp, lambda: panel.viewer2D.layout_job.data is not None, timeout=30)
        assert panel.sagViewer.layout_job.data is None
        serial = connector.calculation_jobs._serial
        panel.update_theme("light")
        qapp.processEvents()
        assert connector.calculation_jobs._serial == serial
        panel.tabWidget.setCurrentWidget(panel.sagViewer)
        wait_for(qapp, lambda: panel.sagViewer.layout_job.data is not None, timeout=30)
        serial = connector.calculation_jobs._serial
        panel.update_theme("dark")
        assert panel.viewer2D.layout_job._restyle_pending
        panel.tabWidget.setCurrentIndex(0)
        qapp.processEvents()
        assert connector.calculation_jobs._serial == serial
        assert not panel.viewer2D.layout_job._restyle_pending
        connector.opticChanged.emit()
        wait_for(qapp, lambda: not connector.calculation_jobs.running, timeout=30)
        assert panel.sagViewer.layout_job._completed_key is None
    finally:
        panel.hide()
        connector.calculation_jobs.shutdown()
        wait_for(qapp, lambda: connector.calculation_jobs._process is None)
        panel.deleteLater()


def test_window_close_keeps_qt_owners_until_worker_is_reaped(qapp):
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    from PySide6.QtWidgets import QWidget

    from optiland_gui.main_window import MainWindow
    from optiland_gui.services.calculation_jobs import CalculationJobs, DocumentState

    service = CalculationJobs(
        DocumentState(),
        cancel_grace_ms=40,
        worker_command=[
            sys.executable,
            "-u",
            str(Path(__file__).with_name("calculation_worker_fixture.py")),
        ],
    )
    closed = []

    class Window(QWidget):
        closeEvent = MainWindow.closeEvent
        _calculations_stopped = MainWindow._calculations_stopped

    window = Window()
    window.connector = SimpleNamespace(calculation_jobs=service)
    window.panel_manager = SimpleNamespace(
        python_terminal=SimpleNamespace(shutdown_kernel=lambda: closed.append(True))
    )
    window.show()
    service.submit("layout", "unused", None, {"delay": 10})
    wait_for(qapp, lambda: service.active_request is not None)
    try:
        assert not window.close()
        assert window.isVisible()
        assert not closed
        wait_for(qapp, lambda: not window.isVisible())
        assert service._process is None
        assert closed == [True]
    finally:
        service.shutdown()
        wait_for(qapp, lambda: service._process is None)
        window.deleteLater()
