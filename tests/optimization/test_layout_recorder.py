"""LayoutRecorder: recording during a run and the animation afterwards."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from matplotlib import animation
from PIL import Image

import optiland.backend as be
from optiland import _instrument
from optiland.optimization import (
    HistoryObserver,
    MaxIter,
    OptimizationProblem,
    minimize,
)
from optiland.optimization.observers import LayoutRecorder, Observer, RecordedFrame
from optiland.optimization.state import OptimizationState
from optiland.samples.objectives import CookeTriplet
from optiland.visualization.system.optic_viewer import OpticViewer

pytestmark = pytest.mark.filterwarnings("ignore")


def _problem():
    optic = CookeTriplet()
    problem = OptimizationProblem()
    problem.add_operand("f2", target=50.0, weight=1.0, input_data={"optic": optic})
    for hy in (0.0, 0.7, 1.0):
        problem.add_operand(
            "rms_spot_size",
            target=0.0,
            weight=10.0,
            input_data={
                "optic": optic,
                "surface_number": -1,
                "Hx": 0.0,
                "Hy": hy,
                "num_rays": 4,
                "wavelength": optic.primary_wavelength,
                "distribution": "hexapolar",
            },
        )
    for s in (1, 2, 3, 4):
        problem.add_variable(optic, "radius", surface_number=s)
    return optic, problem


def _x(problem):
    return np.array([float(be.to_numpy(v.value)) for v in problem.variables])


def _radii(optic):
    return np.array([float(be.to_numpy(r)) for r in optic.surfaces.radii[1:5]])


def _state(iteration, value=1.0):
    return OptimizationState(
        x=np.full(2, float(iteration)), value=value, iteration=iteration
    )


def _recorded_run(steps=3, method="dls"):
    optic, problem = _problem()
    recorder = LayoutRecorder(problem)
    minimize(problem, method, tol=1e-14, stop=MaxIter(steps), observers=[recorder])
    return optic, problem, recorder


def test_is_an_observer():
    assert isinstance(LayoutRecorder(mock.Mock()), Observer)


def test_recording_never_touches_the_problem():
    problem = mock.Mock()
    recorder = LayoutRecorder(problem)
    state = _state(0, value=4.0)
    recorder.on_start(state)
    for i in range(1, 4):
        state.iteration, state.value = i, 4.0 - i
        recorder.on_step(state)
    recorder.on_end(state, mock.Mock())
    assert problem.mock_calls == []
    assert [f.iteration for f in recorder.frames] == [0, 1, 2, 3]
    assert [f.value for f in recorder.frames] == [4.0, 3.0, 2.0, 1.0]


def test_on_step_does_no_ray_tracing():
    _, problem = _problem()
    recorder = LayoutRecorder(problem)
    step_counts = []

    class Probe:
        def on_start(self, state):
            recorder.on_start(state)

        def on_step(self, state):
            before = dict(counts)
            recorder.on_step(state)
            step_counts.append({k: counts[k] - before[k] for k in counts})

        def on_end(self, state, result):
            recorder.on_end(state, result)

    with _instrument.counting() as counts:
        minimize(problem, "dls", tol=1e-14, stop=MaxIter(3), observers=[Probe()])
    assert counts["trace"] + counts["trace_generic"] > 0  # the run itself traces
    assert len(step_counts) == 3
    assert all(not any(c.values()) for c in step_counts)


def test_frames_copy_the_state():
    recorder = LayoutRecorder(mock.Mock())
    state = _state(0)
    recorder.on_start(state)
    state.x[:] = 99.0
    frame = recorder.frames[0]
    assert isinstance(frame, RecordedFrame)
    np.testing.assert_array_equal(frame.x, [0.0, 0.0])
    assert frame.x is not state.x


def test_every_keeps_the_first_and_last_design():
    recorder = LayoutRecorder(mock.Mock(), every=3)
    recorder.on_start(_state(0))
    for i in range(1, 8):
        recorder.on_step(_state(i))
    recorder.on_end(_state(7), mock.Mock())
    assert [f.iteration for f in recorder.frames] == [0, 3, 6, 7]


def test_on_end_replaces_a_frame_of_the_same_iteration():
    recorder = LayoutRecorder(mock.Mock(), every=2)
    recorder.on_start(_state(0))
    for i in (1, 2):
        recorder.on_step(_state(i, value=5.0))
    final = OptimizationState(x=np.array([7.0, 7.0]), value=0.5, iteration=2)
    recorder.on_end(final, mock.Mock())
    assert [f.iteration for f in recorder.frames] == [0, 2]
    assert recorder.frames[-1].value == 0.5
    np.testing.assert_array_equal(recorder.frames[-1].x, [7.0, 7.0])


def test_on_start_clears_an_earlier_run():
    recorder = LayoutRecorder(mock.Mock())
    recorder.on_start(_state(0))
    recorder.on_step(_state(1))
    recorder.on_start(_state(0, value=2.0))
    assert [(f.iteration, f.value) for f in recorder.frames] == [(0, 2.0)]


@pytest.mark.parametrize("method", ["dls", "l-bfgs-b"])
def test_minimize_records_every_step(method):
    _, problem = _problem()
    recorder = LayoutRecorder(problem)
    history = HistoryObserver()
    result = minimize(
        problem, method, tol=1e-14, stop=MaxIter(3), observers=[history, recorder]
    )
    assert [f.iteration for f in recorder.frames] == [0, 1, 2, 3]
    np.testing.assert_allclose(
        [f.value for f in recorder.frames],
        [r["value"] for r in history.records],
        rtol=1e-12,
    )
    np.testing.assert_allclose(recorder.frames[-1].x, result.x, rtol=1e-12)
    np.testing.assert_allclose(recorder.frames[-1].x, _x(problem), rtol=1e-12)


def test_frames_are_numpy_on_both_backends(set_test_backend):
    _, _, recorder = _recorded_run(steps=2)
    assert len(recorder.frames) == 3
    for frame in recorder.frames:
        assert type(frame.x) is np.ndarray
        assert frame.x.dtype == np.float64
        assert isinstance(frame.value, float)


def test_frames_are_numpy_with_a_torch_method():
    pytest.importorskip("torch")
    be.set_backend("torch")
    be.set_device("cpu")
    be.grad_mode.enable()
    be.set_precision("float64")
    try:
        _, problem = _problem()
        recorder = LayoutRecorder(problem)
        result = minimize(problem, "adam", maxiter=3, lr=1e-6, observers=[recorder])
    finally:
        be.set_backend("numpy")
    assert [f.iteration for f in recorder.frames] == [0, 1, 2, 3]
    assert all(type(f.x) is np.ndarray for f in recorder.frames)
    np.testing.assert_allclose(recorder.frames[-1].x, result.x, rtol=1e-12)


def test_animate_writes_one_gif_frame_per_record(set_test_backend, tmp_path):
    optic, problem, recorder = _recorded_run(steps=3)
    x_before, radii_before = _x(problem), _radii(optic)
    path = recorder.animate(tmp_path / "run.gif", fps=4)
    assert path == tmp_path / "run.gif"
    assert isinstance(path, Path)
    with Image.open(path) as image:
        assert image.n_frames == len(recorder.frames) == 4
        assert image.info["duration"] == 250
    np.testing.assert_array_equal(_x(problem), x_before)
    np.testing.assert_array_equal(_radii(optic), radii_before)


def test_animate_restores_the_variables_after_an_error(tmp_path):
    optic, problem, recorder = _recorded_run(steps=3)
    radii_before = _radii(optic)
    recorder.frames[0] = RecordedFrame(0, 1.0, recorder.frames[0].x * 1.1)
    drawn = []
    original_view = OpticViewer.view

    def failing_view(viewer, *args, **kwargs):
        drawn.append(_radii(viewer.optic))
        if len(drawn) == 3:
            raise RuntimeError("drawing failed")
        return original_view(viewer, *args, **kwargs)

    with (
        mock.patch.object(OpticViewer, "view", failing_view),
        pytest.raises(RuntimeError, match="drawing failed"),
    ):
        recorder.animate(tmp_path / "run.gif")
    assert not np.allclose(drawn[0], radii_before)  # a recorded design was drawn
    np.testing.assert_array_equal(_radii(optic), radii_before)


def test_animate_uses_one_axis_range_for_all_frames(tmp_path):
    _, _, recorder = _recorded_run(steps=3)
    limits = []
    original_view = OpticViewer.view

    def spy(viewer, *args, **kwargs):
        limits.append((kwargs.get("xlim"), kwargs.get("ylim")))
        return original_view(viewer, *args, **kwargs)

    with mock.patch.object(OpticViewer, "view", spy):
        recorder.animate(tmp_path / "run.gif")
    frame_limits = limits[2:]  # the first two draws measure the range
    assert len(frame_limits) == len(recorder.frames)
    assert frame_limits[0][0] is not None
    assert frame_limits[0][1] is not None
    assert all(lim == frame_limits[0] for lim in frame_limits)


def test_animate_keeps_given_limits_and_draw_kwargs(tmp_path):
    _, _, recorder = _recorded_run(steps=1)
    calls = []
    original_view = OpticViewer.view

    def spy(viewer, *args, **kwargs):
        calls.append(kwargs)
        return original_view(viewer, *args, **kwargs)

    draw_kwargs = {"xlim": (-20.0, 80.0), "ylim": (-25.0, 25.0), "num_rays": 2}
    with mock.patch.object(OpticViewer, "view", spy):
        recorder.animate(tmp_path / "run.gif", draw_kwargs=draw_kwargs)
    assert len(calls) == len(recorder.frames)  # no extra draws for the range
    for kwargs in calls:
        assert kwargs["xlim"] == (-20.0, 80.0)
        assert kwargs["ylim"] == (-25.0, 25.0)
        assert kwargs["num_rays"] == 2
        assert kwargs["show"] is False
    assert draw_kwargs == {"xlim": (-20.0, 80.0), "ylim": (-25.0, 25.0), "num_rays": 2}


def test_animate_needs_frames_and_a_known_format(tmp_path):
    _, problem = _problem()
    with pytest.raises(ValueError, match="No frames"):
        LayoutRecorder(problem).animate(tmp_path / "run.gif")
    _, _, recorder = _recorded_run(steps=1)
    with pytest.raises(ValueError, match="'.png'"):
        recorder.animate(tmp_path / "run.png")
    assert not (tmp_path / "run.png").exists()


def test_animate_mp4_without_ffmpeg_raises(tmp_path):
    _, _, recorder = _recorded_run(steps=1)
    with (
        mock.patch.object(animation.FFMpegWriter, "isAvailable", return_value=False),
        pytest.raises(RuntimeError, match="ffmpeg"),
    ):
        recorder.animate(tmp_path / "run.mp4")


@pytest.mark.skipif(
    not animation.FFMpegWriter.isAvailable(), reason="ffmpeg is not installed"
)
def test_animate_mp4(tmp_path):
    _, _, recorder = _recorded_run(steps=1)
    path = recorder.animate(tmp_path / "run.mp4")
    assert path.stat().st_size > 0
