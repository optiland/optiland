"""Multi-Configuration Analysis Module

Holds the results of one analysis run on several configurations of a
multi-configuration system and views them side by side.

Kramer Harrison, 2026
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from collections.abc import Iterator

    from optiland.analysis.base import BaseAnalysis


def _accepts_show(view: Any) -> bool:
    """Return True if a ``view`` method has an explicit ``show`` argument."""
    try:
        return "show" in inspect.signature(view).parameters
    except (TypeError, ValueError):
        return False


def _figure_of(output: Any) -> Figure | None:
    """Return the figure in a ``view()`` output (a Figure or a tuple)."""
    if isinstance(output, Figure):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], Figure):
        return output[0]
    return None


class MultiConfigAnalysis:
    """Results of one analysis, run once per configuration.

    Created by :meth:`MultiConfiguration.analyze`. Each result is an ordinary
    analysis object bound to its configuration's optic, so its data and
    methods are used as for a single optic.

    Args:
        results: One analysis object per configuration, in the order of
            ``configurations``.
        configurations: The configuration index of each result.

    Attributes:
        results (list[BaseAnalysis]): The analysis objects.
        configurations (list[int]): The configuration index of each result.
    """

    def __init__(self, results: list[BaseAnalysis], configurations: list[int]):
        if len(results) != len(configurations):
            raise ValueError(
                f"Got {len(results)} results for {len(configurations)} "
                "configurations; pass one result per configuration."
            )
        self.results = list(results)
        self.configurations = list(configurations)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator[BaseAnalysis]:
        return iter(self.results)

    def view(self, *args: Any, show: bool = True, **kwargs: Any) -> list[Any]:
        """View every result, one figure per configuration.

        Analyses draw into a figure of their own layout (and most clear a
        figure passed to them), so the results are not merged into one
        figure. Each result's ``view(*args, **kwargs)`` makes its own figure,
        whose title (``suptitle``) is prefixed with ``"Configuration i"``.
        ``show`` is handled here: it is passed as ``show=False`` to analyses
        whose ``view`` accepts it, and ``plt.show()`` is called once after all
        figures are drawn when ``show`` is True.

        Args:
            *args: Positional arguments forwarded to each result's ``view``.
            show: If True (default), call ``plt.show()`` once at the end. Set
                False for headless use.
            **kwargs: Keyword arguments forwarded to each result's ``view``.

        Returns:
            The output of each result's ``view`` (usually a
            ``(figure, axes)`` tuple), in configuration order.
        """
        outputs = []
        for config_idx, result in zip(self.configurations, self.results, strict=True):
            view_kwargs = dict(kwargs)
            if _accepts_show(result.view):
                view_kwargs["show"] = False
            output = result.view(*args, **view_kwargs)
            fig = _figure_of(output)
            if fig is not None:
                label = f"Configuration {config_idx}"
                existing = fig.get_suptitle()
                fig.suptitle(f"{label}: {existing}" if existing else label)
            outputs.append(output)
        if show:
            plt.show()
        return outputs
