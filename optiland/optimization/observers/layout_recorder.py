"""LayoutRecorder: record a run, then animate the layout and the merit.

Experimental: part of the optimization facade, subject to change.

Vanshaj Poonia, 2026
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import optiland.backend as be

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from optiland.optimization.problem import OptimizationProblem
    from optiland.optimization.state import OptimizationResult, OptimizationState


@dataclass(frozen=True, eq=False)
class RecordedFrame:
    """One recorded design of a run. *Experimental.*

    Attributes:
        iteration: Accepted steps before this design (0 is the start).
        value: Merit of the design.
        x: Copy of the scaled variable vector, always a NumPy array.
    """

    iteration: int
    value: float
    x: np.ndarray


class LayoutRecorder:
    """Records the design during a run and animates it afterwards.

    Recording is cheap: ``on_start``, ``on_step`` and ``on_end`` only copy the
    iteration, the merit and ``state.x``, every ``every`` accepted steps. The
    first and the last design are always kept. Nothing is traced or drawn
    until :meth:`animate`. *Experimental.*

    Args:
        problem: The problem being optimized. :meth:`animate` writes the
            recorded designs into its variables.
        every: Record every Nth accepted step.

    Attributes:
        frames: The recorded designs, in run order.
    """

    def __init__(self, problem: OptimizationProblem, every: int = 1) -> None:
        self.problem = problem
        self._every = max(int(every), 1)
        self.frames: list[RecordedFrame] = []

    def on_start(self, state: OptimizationState) -> None:
        """Clear the frames and record the starting design."""
        self.frames.clear()
        self.frames.append(_frame(state))

    def on_step(self, state: OptimizationState) -> None:
        """Record the design on every ``every``-th step."""
        if state.iteration % self._every == 0:
            self.frames.append(_frame(state))

    def on_end(self, state: OptimizationState, result: OptimizationResult) -> None:  # noqa: ARG002
        """Record the final design, replacing a frame of the same iteration."""
        frame = _frame(state)
        if self.frames and self.frames[-1].iteration == frame.iteration:
            self.frames[-1] = frame
        else:
            self.frames.append(frame)

    def animate(
        self,
        path: str | Path,
        fps: int = 5,
        draw_kwargs: dict | None = None,
        figsize: tuple[float, float] = (10, 6),
    ) -> Path:
        """Write an animation of the recorded designs, one frame per record.

        Each frame shows the layout (:meth:`OpticViewer.view
        <optiland.visualization.system.optic_viewer.OpticViewer.view>`) on the
        left and the merit history on a log scale on the right. All layout
        frames share the axis range of the first and the last design. The
        variables are restored afterwards, also when drawing fails.

        Args:
            path: Output file. ``.gif`` is written with Pillow, ``.mp4`` with
                ffmpeg.
            fps: Frames per second.
            draw_kwargs: Keyword arguments for ``OpticViewer.view``. ``ax``
                and ``show`` are set by the recorder.
            figsize: Figure size in inches.

        Returns:
            The path of the written file.

        Raises:
            ValueError: If nothing was recorded or the file type is not
                ``.gif`` or ``.mp4``.
            RuntimeError: For ``.mp4`` when ffmpeg is not available.
        """
        from matplotlib import animation
        from matplotlib.figure import Figure
        from matplotlib.ticker import MaxNLocator

        path = Path(path)
        if not self.frames:
            raise ValueError(
                "No frames recorded. Pass the recorder to minimize() first."
            )
        suffix = path.suffix.lower()
        if suffix == ".gif":
            writer = animation.PillowWriter(fps=fps)
        elif suffix == ".mp4":
            if not animation.FFMpegWriter.isAvailable():
                raise RuntimeError(
                    "Writing .mp4 needs ffmpeg, which was not found. "
                    "Install ffmpeg or write a .gif instead."
                )
            writer = animation.FFMpegWriter(fps=fps)
        else:
            raise ValueError(
                f"Cannot write {path.suffix!r} files; use '.gif' or '.mp4'."
            )

        saved = [float(be.to_numpy(var.value)) for var in self.problem.variables]
        fig = Figure(figsize=figsize, layout="constrained")
        ax_layout, ax_merit = fig.subplots(1, 2, gridspec_kw={"width_ratios": (3, 2)})
        try:
            options = dict(draw_kwargs or {})
            if options.get("xlim") is None or options.get("ylim") is None:
                xlim, ylim = self._layout_limits(ax_layout, options)
                options["xlim"] = options.get("xlim") or xlim
                options["ylim"] = options.get("ylim") or ylim

            iterations = [frame.iteration for frame in self.frames]
            values = [frame.value for frame in self.frames]
            ax_merit.plot(iterations, values, color="0.8")
            (done,) = ax_merit.plot([], [], color="C0")
            (current,) = ax_merit.plot([], [], "o", color="C3")
            ax_merit.set_yscale("log")
            ax_merit.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax_merit.set_xlabel("Iteration")
            ax_merit.set_ylabel("Merit function value")
            ax_merit.grid(alpha=0.25)

            writer.setup(fig, str(path), dpi=fig.dpi)
            try:
                for i, frame in enumerate(self.frames):
                    self._draw_layout(ax_layout, frame, options)
                    done.set_data(iterations[: i + 1], values[: i + 1])
                    current.set_data([frame.iteration], [frame.value])
                    ax_merit.set_title(f"Merit: {frame.value:.4g}")
                    writer.grab_frame()
            except BaseException:
                # Close the writer, but raise the drawing error, not its own.
                with contextlib.suppress(Exception):
                    writer.finish()
                raise
            writer.finish()
        finally:
            self._apply(saved)
        return path

    def _apply(self, x: np.ndarray | list[float]) -> None:
        """Write a scaled variable vector into the problem's optics."""
        for var, value in zip(self.problem.variables, x, strict=True):
            var.update(float(value))
        self.problem.update_optics()

    def _draw_layout(self, ax: Axes, frame: RecordedFrame, options: dict) -> None:
        """Draw the layout of one recorded design into ``ax``."""
        from optiland.visualization.system.optic_viewer import OpticViewer

        self._apply(frame.x)
        ax.clear()
        kwargs = {"title": f"Iteration {frame.iteration}", **options}
        kwargs.update(ax=ax, show=False)
        optic = self.problem.variables[0].optic
        _, _, interaction = OpticViewer(optic).view(**kwargs)
        interaction.disconnect()

    def _layout_limits(
        self, ax: Axes, options: dict
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Axis range that holds the layouts of the first and last design."""
        xs: list[float] = []
        ys: list[float] = []
        for frame in (self.frames[0], self.frames[-1]):
            self._draw_layout(ax, frame, options)
            xs.extend(ax.get_xlim())
            ys.extend(ax.get_ylim())
        ax.clear()
        return (min(xs), max(xs)), (min(ys), max(ys))


def _frame(state: OptimizationState) -> RecordedFrame:
    """Copy what a frame needs from the state, without touching the state."""
    x = np.array(be.to_numpy(state.x), dtype=float).reshape(-1)
    return RecordedFrame(int(state.iteration), float(state.value), x)
