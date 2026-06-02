from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock, call, patch

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import optiland.optimization.optimizer.live_plotter as live_plotter_module
from optiland.optimization.optimizer.live_plotter import LiveOptimizationPlotter


class FakeOptic:
    """Minimal optic fake recording draw calls."""

    def __init__(self) -> None:
        self.draw_calls: list[tuple[str, Axes]] = []

    def draw(self, *, title: str, ax: Axes) -> None:
        """Record the draw target and add visible content to the axes."""
        self.draw_calls.append((title, ax))
        ax.plot([0.0, 1.0], [0.0, 1.0])


@dataclass
class FakeVariable:
    """Minimal optimization variable fake."""

    optic: FakeOptic


class FakeProblem:
    """Minimal optimization problem fake returning controlled merit values."""

    def __init__(self, optic: FakeOptic, merit_values: list[Any]) -> None:
        self.variables = [FakeVariable(optic)]
        self._merit_values = iter(merit_values)

    def sum_squared(self) -> Any:
        """Return the next merit value."""
        return next(self._merit_values)


@dataclass
class FakeOptimizer:
    """Minimal optimizer fake exposing a problem attribute."""

    problem: FakeProblem


def build_optimizer(
    merit_values: list[Any] | None = None,
) -> tuple[FakeOptimizer, FakeOptic]:
    """Build a fake optimizer and return it with its optic fake."""
    optic = FakeOptic()
    problem = FakeProblem(optic, merit_values or [1.0])
    return FakeOptimizer(problem), optic


@pytest.fixture(autouse=True)
def clean_matplotlib_state() -> Iterator[None]:
    """Keep tests isolated from open figures and interactive mode."""
    plt.close("all")
    plt.ioff()
    yield
    plt.close("all")
    plt.ioff()


class TestLiveOptimizationPlotter:
    def test_initialize_creates_figures_with_placeholder_content(self) -> None:
        """Initialize persistent figures once with placeholder content."""
        optimizer, _ = build_optimizer()
        plotter = LiveOptimizationPlotter(optimizer)
        plotter._is_inline_backend = False

        assert plotter._initialized is False
        assert plotter._system_fig is None
        assert plotter._merit_fig is None

        with (
            patch.object(Figure, "show") as show_mock,
            patch.object(plotter, "_position_windows") as position_mock,
            patch.object(plotter, "_force_render") as render_mock,
        ):
            plotter.initialize()

            assert plotter._initialized is True
            assert isinstance(plotter._system_fig, Figure)
            assert isinstance(plotter._system_ax, Axes)
            assert isinstance(plotter._merit_fig, Figure)
            assert isinstance(plotter._merit_ax, Axes)
            assert plotter._merit_line is not None

            assert plotter._system_ax.get_title() == "Lens optimization"
            assert plotter._merit_ax.get_title() == "Merit function"
            assert plotter._merit_ax.get_xlabel() == "Iteration"
            assert plotter._merit_ax.get_ylabel() == "Merit function value"
            assert plotter._merit_ax.get_yscale() == "log"

            assert show_mock.call_count == 2
            position_mock.assert_called_once()
            render_mock.assert_called_once()

            system_fig = plotter._system_fig
            merit_fig = plotter._merit_fig

            plotter.initialize()

            assert plotter._system_fig is system_fig
            assert plotter._merit_fig is merit_fig
            assert show_mock.call_count == 2
            position_mock.assert_called_once()
            render_mock.assert_called_once()

    def test_update_redraws_optic_and_appends_merit_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Update redraws the optic and keeps merit history in sync."""
        optimizer, optic = build_optimizer([np.array(10.0), 5.0])

        monkeypatch.setattr(
            live_plotter_module.be,
            "to_numpy",
            lambda value: value,
        )

        plotter = LiveOptimizationPlotter(optimizer)
        plotter._is_inline_backend = True

        with patch.object(plotter, "_render") as render_mock:
            plotter.update()
            plotter.update()

        assert plotter._initialized is True

        assert len(optic.draw_calls) == 2
        assert [title for title, _ in optic.draw_calls] == [
            "Optimization",
            "Optimization",
        ]
        assert all(ax is plotter._system_ax for _, ax in optic.draw_calls)

        assert plotter.history == [10.0, 5.0]

        x_data, y_data = plotter._merit_line.get_data()
        assert list(x_data) == [0, 1]
        assert list(y_data) == [10.0, 5.0]

        assert plotter._merit_ax.get_yscale() == "log"
        assert plotter._merit_ax.get_title() == "Merit function"
        assert render_mock.call_count == 2

    def test_inline_render_creates_display_handles_then_updates_them(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Inline render creates display handles once, then reuses them."""
        optimizer, _ = build_optimizer()
        plotter = LiveOptimizationPlotter(optimizer)
        plotter._is_inline_backend = True
        plotter.initialize()

        system_handle = Mock()
        merit_handle = Mock()
        display_mock = Mock(side_effect=[system_handle, merit_handle])
        collect_mock = Mock()

        monkeypatch.setattr(live_plotter_module, "display", display_mock)
        monkeypatch.setattr(live_plotter_module.gc, "collect", collect_mock)

        plotter._render()

        assert plotter._optical_system_handle is system_handle
        assert plotter._merit_function_handle is merit_handle
        display_mock.assert_has_calls(
            [
                call(plotter._system_fig, display_id=True),
                call(plotter._merit_fig, display_id=True),
            ]
        )

        plotter._render()

        assert display_mock.call_count == 2
        system_handle.update.assert_called_once_with(plotter._system_fig)
        merit_handle.update.assert_called_once_with(plotter._merit_fig)
        assert collect_mock.call_count == 2

    def test_inline_finalize_closes_figures(self) -> None:
        """Inline finalize closes both persistent figures."""
        optimizer, _ = build_optimizer()
        plotter = LiveOptimizationPlotter(optimizer)
        plotter._is_inline_backend = True
        plotter.initialize()

        assert plotter._system_fig is not None
        assert plotter._merit_fig is not None

        system_fig_number = plotter._system_fig.number
        merit_fig_number = plotter._merit_fig.number

        assert plt.fignum_exists(system_fig_number)
        assert plt.fignum_exists(merit_fig_number)

        plotter.finalize()

        assert not plt.fignum_exists(system_fig_number)
        assert not plt.fignum_exists(merit_fig_number)

    def test_non_inline_finalize_disables_interactive_mode_and_shows_figures(
        self,
    ) -> None:
        """Non-inline finalize disables interactive mode and calls show."""
        optimizer, _ = build_optimizer()
        plotter = LiveOptimizationPlotter(optimizer)
        plotter._is_inline_backend = False

        with (
            patch.object(Figure, "show"),
            patch.object(plotter, "_position_windows"),
            patch.object(plotter, "_force_render"),
        ):
            plotter.initialize()

        with (
            patch("matplotlib.pyplot.ioff") as ioff_mock,
            patch("matplotlib.pyplot.show") as show_mock,
        ):
            plotter.finalize()

        ioff_mock.assert_called_once()
        show_mock.assert_called_once()