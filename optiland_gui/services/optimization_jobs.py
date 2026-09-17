"""Process-only optimization over a detached candidate and plain definitions."""

from __future__ import annotations

import importlib
import inspect
import json
import math
import time
from numbers import Real
from typing import TYPE_CHECKING, Any

import optiland.backend as be
from optiland.optimization import OptimizationProblem

from .job_records import OpticSnapshot, check_cancelled

if TYPE_CHECKING:
    from collections.abc import Callable
    from threading import Event

    from optiland.optic import Optic


def resolve_wavelength(optic: Optic, value: Any) -> float:
    """Resolve a dialog selector to one positive wavelength, without execution.

    The shared wavelength menu also serves analyses that accept multiple values;
    optimization variables and ray operands require one scalar wavelength.
    """
    message = "Select one positive wavelength or Primary for this optimization input."
    if isinstance(value, str):
        value = value.strip()
        if value in {"primary", "'primary'", '"primary"'}:
            value = optic.wavelengths.primary_wavelength.value
        else:
            try:
                value = json.loads(value)
            except (ValueError, TypeError) as error:
                raise ValueError(message) from error
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(message)
        value = value[0]
    if isinstance(value, str) and value == "primary":
        value = optic.wavelengths.primary_wavelength.value
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(message)
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(message)
    return value


def variable_arguments(optic: Optic, definition: dict) -> dict:
    """Share all configured variable inputs between display and computation."""
    values = {
        key: value
        for key, value in definition.items()
        if key not in {"type", "optic"} and value is not None
    }
    if "wavelength" in values:
        values["wavelength"] = resolve_wavelength(optic, values["wavelength"])
    return values


def build_problem(
    optic: Optic, variables: list[dict], operands: list[dict], operand_metadata: dict
) -> OptimizationProblem:
    """Fail setup explicitly instead of silently optimizing an incomplete problem."""
    problem = OptimizationProblem()
    for definition in variables:
        problem.add_variable(
            optic, definition["type"], **variable_arguments(optic, definition)
        )
    for definition in operands:
        values = definition.get("input_data")
        values = (
            dict(values)
            if isinstance(values, dict)
            else json.loads(definition.get("input_data_str") or "{}")
        )
        if "wavelength" in values or "wavelength" in operand_metadata.get(
            definition["type"], {}
        ):
            values["wavelength"] = resolve_wavelength(
                optic, values.get("wavelength", "primary")
            )
        values["optic"] = optic
        problem.add_operand(
            operand_type=definition["type"],
            target=definition.get("target"),
            min_val=definition.get("min_val"),
            max_val=definition.get("max_val"),
            weight=definition.get("weight", 1.0),
            input_data=values,
        )
    if not len(problem.variables) or not len(problem.operands):
        raise ValueError("Optimization needs at least one variable and one operand.")
    return problem


def optimize(
    snapshot: OpticSnapshot,
    parameters: dict,
    progress: Callable[..., None],
    cancelled: Event,
) -> dict:
    """Build, run and validate one candidate; never access the live document."""
    started = time.monotonic()
    check_cancelled(cancelled)
    progress("Preparing optimization")
    optic = snapshot.restore()
    problem = build_problem(
        optic,
        parameters["variables"],
        parameters["operands"],
        parameters["operand_metadata"],
    )
    original_update = problem.update_optics

    def update():
        check_cancelled(cancelled)
        original_update()
        check_cancelled(cancelled)

    problem.update_optics = update
    update()
    initial_merit = float(problem.rss())
    module, name = parameters["optimizer"].split(":", 1)
    optimizer_type = importlib.import_module(module)
    for component in name.split("."):
        optimizer_type = getattr(optimizer_type, component)
    optimizer = optimizer_type(problem)
    counters = {"evaluations": 0, "callbacks": 0}
    last_report = -math.inf
    last_preview = -math.inf
    latest_merit = initial_merit

    def report(force=False):
        nonlocal last_report, last_preview
        check_cancelled(cancelled)
        now = time.monotonic()
        if not force and now - last_report < 0.2:
            return
        last_report = now
        details = {
            **counters,
            "merit": latest_merit,
            "elapsed": now - started,
            "variables": [
                float(variable.variable.get_value()) for variable in problem.variables
            ],
        }
        preview = parameters.get("preview", {})
        frequency = max(int(preview.get("frequency", 10)), 1)
        if (
            preview.get("enabled")
            and counters["evaluations"] >= frequency
            and now - last_preview >= 1.0
        ):
            from .layout_tasks import prepare_2d

            try:
                details["preview"] = prepare_2d(
                    OpticSnapshot.capture(optic),
                    {"num_rays": 5, "distribution": "line_y"},
                    lambda *args, **kwargs: check_cancelled(cancelled),
                    cancelled,
                )
            except Exception as exc:
                check_cancelled(cancelled)
                details["preview_error"] = str(exc)
            last_preview = time.monotonic()
        progress("Optimizing", details=details)

    def wrap_objective(function, residuals=False, squared=False):
        def evaluate(*args, **kwargs):
            nonlocal latest_merit
            check_cancelled(cancelled)
            value = function(*args, **kwargs)
            check_cancelled(cancelled)
            counters["evaluations"] += 1
            if residuals:
                latest_merit = float(be.to_numpy(be.linalg.norm(be.array(value))))
            else:
                latest_merit = (
                    math.sqrt(max(float(value), 0.0)) if squared else float(value)
                )
            report()
            return value

        return evaluate

    # Wrap the outer evaluation boundary: least-squares catches exceptions from
    # individual operands, so cancellation inside op.fun alone would be swallowed.
    if hasattr(optimizer, "_compute_residuals_vector"):
        optimizer._compute_residuals_vector = wrap_objective(
            optimizer._compute_residuals_vector, residuals=True
        )
    elif optimizer_type.__name__ == "OrthogonalDescent":
        problem.rss = wrap_objective(problem.rss)
    else:
        optimizer._fun = wrap_objective(optimizer._fun, squared=True)

    def callback(*args, **kwargs):
        check_cancelled(cancelled)
        counters["callbacks"] += 1
        report()

    kwargs = dict(parameters["optimizer_kwargs"])
    if optimizer_type.__name__ == "SHGO":
        if "maxiter" in kwargs:
            kwargs["iters"] = kwargs.pop("maxiter")
        if "disp" in kwargs:
            kwargs["options"] = {
                **kwargs.get("options", {}),
                "disp": kwargs.pop("disp"),
            }
    elif optimizer_type.__name__ == "BasinHopping" and "maxiter" in kwargs:
        kwargs["niter"] = kwargs.pop("maxiter")
    signature = inspect.signature(optimizer.optimize)
    if "workers" in signature.parameters:
        # The shared executor already isolates this owned problem. Nested pools
        # would copy its state and escape the coordinator's cancellation policy.
        kwargs["workers"] = 1
    if "callback" in signature.parameters:
        kwargs["callback"] = callback
    if "plot" in signature.parameters:
        kwargs["plot"] = False
    report(force=True)
    result = optimizer.optimize(**kwargs)
    check_cancelled(cancelled)
    update()
    final_merit = float(problem.rss())
    if not math.isfinite(final_merit):
        raise ValueError("Optimization produced a non-finite final merit.")
    # Some public algorithms return no convergence result. Preserve their finite
    # candidate for explicit review instead of inventing a convergence outcome.
    converged = bool(getattr(result, "success", False))
    message = str(
        getattr(result, "message", "Algorithm returned no convergence status.")
    )
    # Several exposed SciPy methods ignore bounds. Their success flag alone
    # cannot authorize committing a prescription outside the user's constraints.
    violations = []
    for index, (variable, definition) in enumerate(
        zip(problem.variables, parameters["variables"], strict=True)
    ):
        value = float(variable.variable.get_value())
        if not math.isfinite(value):
            raise ValueError(f"Variable {index} has a non-finite final value.")
        lower, upper = definition.get("min_val"), definition.get("max_val")
        tolerance = 1e-9 * max(1.0, abs(value))
        if (lower is not None and value < lower - tolerance) or (
            upper is not None and value > upper + tolerance
        ):
            violations.append(str(index))
    if violations:
        converged = False
        message = (
            "Candidate violates configured bounds for variable(s) "
            + ", ".join(violations)
            + ". "
            + message
        )
    return {
        "candidate": OpticSnapshot.capture(optic),
        "initial_merit": initial_merit,
        "final_merit": final_merit,
        "converged": converged,
        "message": message,
        "iterations": getattr(result, "nit", None),
        "evaluations": int(getattr(result, "nfev", counters["evaluations"])),
        "elapsed": time.monotonic() - started,
    }
