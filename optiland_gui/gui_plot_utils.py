"""Provides utility functions for GUI plotting and analysis parameter inspection.

This module contains helper functions for configuring Matplotlib styles for GUI
embedding and for dynamically inspecting the parameters of analysis classes to
auto-generate settings interfaces.

Author: Manuel Fragata Mendes, 2025
"""

from __future__ import annotations

import inspect
import logging

import matplotlib

from optiland.visualization.themes import get_active_theme, set_theme

logger = logging.getLogger(__name__)


def apply_gui_matplotlib_styles(theme: str = "light") -> None:
    """Apply Matplotlib rcParams for GUI embedding with theme awareness.

    Configures Matplotlib's runtime configuration parameters (``rcParams``)
    so that plots embedded in the GUI have a consistent and readable style.
    Adjusts colours and sizes for either a *light* or *dark* theme.

    Args:
        theme: The theme to apply, either ``"light"`` or ``"dark"``.
    """
    set_theme(theme)
    active_theme = get_active_theme()
    valid_params = {
        k: v for k, v in active_theme.parameters.items() if k in matplotlib.rcParams
    }
    matplotlib.rcParams.update(valid_params)


_SKIP_PARAMS = frozenset({"self", "cls", "optic", "optical_system"})
_VARIADIC_KINDS = frozenset(
    {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
)


def _sig_has_only_variadic(sig: inspect.Signature) -> bool:
    """Return ``True`` when every non-self parameter is ``*args`` or ``**kwargs``."""
    return all(
        p.kind in _VARIADIC_KINDS
        for name, p in sig.parameters.items()
        if name not in _SKIP_PARAMS
    )


def get_analysis_parameters(analysis_class: type) -> dict:
    """Inspect an analysis class's constructor to find its configurable parameters.

    Uses Python's :mod:`inspect` module to determine the parameters of the
    constructor of a given analysis class.  Standard parameters like ``self``
    and ``optic`` are filtered out.

    For *factory dispatch* classes (e.g. ``FFTPSF``, ``HuygensPSF``) whose
    ``__init__`` contains only ``*args`` / ``**kwargs``, the function falls
    back to inspecting ``__new__``, where the real parameter list lives.

    Args:
        analysis_class: The analysis class to inspect.

    Returns:
        A dict whose keys are parameter names and values are dicts with
        ``"default"`` and ``"annotation"`` entries.  Returns ``{}`` if
        inspection fails.
    """
    if not analysis_class:
        return {}

    params: dict = {}
    try:
        sig = inspect.signature(analysis_class.__init__)
        if _sig_has_only_variadic(sig) and hasattr(analysis_class, "__new__"):
            # Factory dispatch class — real params live in __new__
            sig = inspect.signature(analysis_class.__new__)

        for param_name, param_obj in sig.parameters.items():
            if param_name in _SKIP_PARAMS:
                continue
            if param_obj.kind in _VARIADIC_KINDS:
                continue
            params[param_name] = {
                "default": (
                    param_obj.default
                    if param_obj.default is not inspect.Parameter.empty
                    else None
                ),
                "annotation": (
                    param_obj.annotation
                    if param_obj.annotation is not inspect.Parameter.empty
                    else None
                ),
            }
    except (ValueError, TypeError) as exc:
        logger.warning(
            "Could not inspect parameters for %s: %s",
            getattr(analysis_class, "__name__", repr(analysis_class)),
            exc,
        )
    return params


def handle_matplotlib_scroll_zoom(event: object) -> None:
    """Handle mouse wheel scrolling for zooming on a Matplotlib axes.

    Args:
        event: A Matplotlib scroll event.  Must have ``inaxes``, ``step``,
            ``xdata``, and ``ydata`` attributes.
    """
    if not event.inaxes:
        return

    ax = event.inaxes
    scale_factor = 1.1 if event.step < 0 else 1 / 1.1

    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()

    xdata, ydata = event.xdata, event.ydata

    new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

    rel_x = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
    rel_y = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

    ax.set_xlim([xdata - new_width * (1 - rel_x), xdata + new_width * rel_x])
    ax.set_ylim([ydata - new_height * (1 - rel_y), ydata + new_height * rel_y])
    ax.figure.canvas.draw_idle()
