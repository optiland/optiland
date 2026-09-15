"""Worker-only preparation of every analysis exposed in the GUI registry."""

from __future__ import annotations

import importlib
import inspect
import pickle
import warnings
from typing import TYPE_CHECKING

from optiland_gui.registry import ANALYSIS_REGISTRY
from optiland_gui.services.job_records import OpticSnapshot, check_cancelled

if TYPE_CHECKING:
    from collections.abc import Callable
    from threading import Event

MAX_RESULT_BYTES = 64 * 1024 * 1024
MAX_WORKING_BYTES = 512 * 1024 * 1024


def resolve_analysis(name: str) -> type:
    """Resolve a known registry name, never a user-provided import path."""
    path = next((p for _, n, p in ANALYSIS_REGISTRY if n == name), None)
    if path is None:
        raise ValueError(f"Unknown analysis: {name}")
    module, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)


def constructor_parameters(cls: type) -> dict[str, inspect.Parameter]:
    """Resolve factory signatures without constructing an optical calculation."""
    params = inspect.signature(cls.__init__).parameters
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    if all(p.kind in variadic for k, p in params.items() if k != "self"):
        params = inspect.signature(cls.__new__).parameters
    return {
        k: p
        for k, p in params.items()
        if k not in ("self", "cls", "optic") and p.kind not in variadic
    }


def normalized_parameters(name: str, supplied: dict) -> dict:
    """Preserve supported core defaults and required field/wavelength inputs."""
    params = constructor_parameters(resolve_analysis(name))
    unknown = set(supplied) - set(params)
    if unknown:
        raise ValueError(f"Unsupported {name} settings: {', '.join(sorted(unknown))}")
    result = {
        k: p.default
        for k, p in params.items()
        if p.default is not inspect.Parameter.empty
    }
    result.update(supplied)
    for key, value in {"field": (0.0, 0.0), "wavelength": "primary"}.items():
        if key in params and key not in result:
            result[key] = value
    return result


def validate_workload(
    name: str, params: dict, view: dict, surfaces: int, fields: int, wavelengths: int
) -> int:
    """Reject unsafe GUI allocations using the parameters' actual dimensions.

    This is a conservative working-set estimate, not a promise of peak memory.
    No sampling settings are silently reduced; larger jobs can be run explicitly
    through the scripting API. The isolated worker remains cancellable.
    """
    for key in (
        "num_rays",
        "num_points",
        "num_rays_for_fit",
        "num_rings",
        "num_fields",
        "num_steps",
        "num_terms",
        "grid_size",
        "image_size",
    ):
        value = params.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{key} must be a positive integer.")
    if view.get("projection", "2d") != "2d":
        raise ValueError("Analysis pages currently support 2D projections only.")
    n = params.get("num_rays", params.get("num_points", 1))
    rings = params.get("num_rings")
    distribution = params.get("distribution", "grid")
    if rings is not None:
        rays = 1 + 3 * rings * (rings + 1)
    elif name in ("OPD Fan", "Ray Fan", "Best-Fit Ray Fan"):
        rays = n
    elif "PSF" in name or name == "FFT MTF" or distribution in ("grid", "hexapolar"):
        rays = 4 * n * n
    else:
        rays = n
    num_fields = params.get("num_fields", fields)
    steps = params.get("num_steps", 1)
    if (
        name in ("Spot Diagram", "Ray Fan", "Best-Fit Ray Fan", "OPD Fan")
        and num_fields > 32
    ):
        raise ValueError("This analysis display supports at most 32 fields per page.")
    if name == "Through-Focus Spot" and num_fields * steps > 64:
        raise ValueError(
            "Through-focus display is limited to 64 field/focus plots; "
            "reduce num_steps or fields."
        )
    grid = params.get("grid_size") or 2 * n
    image = params.get("image_size") or grid
    ray_bytes = rays * max(2, surfaces) * 12 * 8
    retained_bytes = rays * max(1, num_fields) * max(1, wavelengths) * steps * 6 * 8
    fit_rays = params.get("num_rays_for_fit", 0)
    ray_bytes += 4 * fit_rays * fit_rays * max(2, surfaces) * 12 * 8
    retained_bytes += rays * params.get("num_terms", 0) * 8
    plot_points = view.get("num_points", 256)
    if (
        isinstance(plot_points, bool)
        or not isinstance(plot_points, int)
        or plot_points < 1
    ):
        raise ValueError("Plot num_points must be a positive integer.")
    if (
        name == "MMDFT PSF"
        and params.get("pixel_pitch") is not None
        and params.get("image_size") is None
    ):
        raise ValueError(
            "Set an explicit image_size when specifying MMDFT pixel_pitch; "
            "automatic sizing is unbounded."
        )
    image_bytes = 0
    if "PSF" in name or name == "FFT MTF":
        image_bytes = (grid * grid + image * image) * 16 * 12
    if "PSF" in name or name in ("OPD", "Zernike OPD"):
        image_bytes += plot_points * plot_points * 8 * 8
    estimate = ray_bytes + retained_bytes + image_bytes
    if estimate > MAX_WORKING_BYTES:
        raise ValueError(
            f"{name} settings require an estimated {estimate / 2**20:.0f} MiB "
            "working set (GUI limit 512 MiB). Reduce pupil sampling, image/grid "
            "size, fields or focus steps; no settings have been changed."
        )
    return estimate


def prepare_analysis(
    snapshot: OpticSnapshot,
    parameters: dict,
    progress: Callable[..., None],
    cancelled: Event,
) -> dict:
    """Prepare detached data and preserve numerical warnings for the result page."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", DeprecationWarning)
        result = _prepare_analysis(snapshot, parameters, progress, cancelled)
    result["warnings"] = list(dict.fromkeys(str(w.message) for w in recorded))
    return result


def _prepare_analysis(
    snapshot: OpticSnapshot,
    parameters: dict,
    progress: Callable[..., None],
    cancelled: Event,
) -> dict:
    """Calculate and prepare plots entirely inside the shared isolated process."""
    import matplotlib.pyplot as plt

    from optiland_gui.gui_plot_utils import apply_gui_matplotlib_styles
    from optiland_gui.services.analysis_plots import capture_analysis_plot

    name = parameters["name"]
    args = normalized_parameters(name, parameters["constructor_args"])
    view = dict(parameters["view_args"])
    progress("Restoring analysis snapshot")
    optic = snapshot.restore()
    validate_workload(
        name,
        args,
        view,
        optic.surfaces.num_surfaces,
        optic.fields.num_fields,
        optic.wavelengths.num_wavelengths,
    )
    check_cancelled(cancelled)
    progress(f"Calculating {name}")
    calculation_args = dict(args)
    if name == "MMDFT PSF" and calculation_args.get("wavelength") == "primary":
        calculation_args["wavelength"] = optic.wavelengths.primary_wavelength.value
    analysis = resolve_analysis(name)(optic=optic, **calculation_args)
    check_cancelled(cancelled)
    progress("Preparing analysis plot")
    apply_gui_matplotlib_styles(parameters.get("theme", "dark"))
    figure = plt.figure(figsize=(7, 5.5))
    try:
        view_parameters = inspect.signature(analysis.view).parameters
        unknown = set(view) - set(view_parameters)
        if unknown:
            raise ValueError(f"Unsupported plot settings: {', '.join(sorted(unknown))}")
        if "fig_to_plot_on" not in view_parameters:
            raise ValueError(f"{name} does not support prepared embedded plots.")
        if "show" in view_parameters:
            view["show"] = False
        analysis.view(fig_to_plot_on=figure, **view)
        check_cancelled(cancelled)
        if len(figure.axes) > 64:
            raise ValueError("Prepared analysis exceeds the 64-axis page limit.")
        summary = (
            analysis.get_summary_text() if hasattr(analysis, "get_summary_text") else ""
        )
        prepared = capture_analysis_plot(figure)
        result = {
            "plot": prepared,
            "summary": summary,
            "constructor_args": args,
            "view_args": parameters["view_args"],
        }
        size = len(pickle.dumps(result, protocol=5))
        if size > MAX_RESULT_BYTES:
            raise ValueError(
                "Prepared analysis exceeds the 64 MiB page limit; reduce plot sampling."
            )
        result["size_bytes"] = size
        return result
    finally:
        plt.close(figure)
