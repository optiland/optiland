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
from optiland.optimization.observers import CancelToken

from .job_records import OpticSnapshot, check_cancelled

if TYPE_CHECKING:
    from collections.abc import Callable
    from threading import Event

    from optiland.optic import Optic
    from optiland.optimization.observers import Observer
    from optiland.optimization.state import OptimizationResult, OptimizationState

#: Minimum seconds between progress messages, so at most 10 per second.
PROGRESS_INTERVAL = 0.1


class EventCancelToken(CancelToken):
    """A :class:`CancelToken` that reads the job's own cancellation event.

    The calculation process signals cancellation through one event per job.
    This token exposes that event to the native solvers, which then stop
    cleanly at the next accepted step.

    Args:
        event: The job's cancellation event.
    """

    def __init__(self, event: Event) -> None:
        super().__init__()
        self._job_event = event

    def cancel(self) -> None:
        """Request cancellation of the job."""
        self._job_event.set()

    def reset(self) -> None:
        """Clear the job's cancellation request."""
        self._job_event.clear()

    @property
    def cancelled(self) -> bool:
        """True once the job has been cancelled."""
        return self._job_event.is_set()


class StepObserver:
    """Forward each accepted step as ``(iteration, evaluations, merit, x)``.

    ``merit`` is the RSS, ``sqrt(sum(r ** 2))``, as in the job's other
    reports; ``x`` is the scaled variable vector.

    Args:
        publish: Receives one call per accepted step.
    """

    def __init__(self, publish: Callable[[int, int, float, list[float]], None]) -> None:
        self._publish = publish

    def on_start(self, state: OptimizationState) -> None:
        """Do nothing; the job reports the starting design itself."""

    def on_step(self, state: OptimizationState) -> None:
        """Publish the accepted step."""
        self._publish(
            int(state.iteration),
            int(state.n_evals),
            math.sqrt(max(float(state.value), 0.0)),
            [float(value) for value in be.to_numpy(state.x).ravel()],
        )

    def on_end(self, state: OptimizationState, result: OptimizationResult) -> None:
        """Do nothing; the job reports the final design itself."""


class DampedLeastSquares:
    """Optiland's native damped least squares for the GUI. *Experimental.*

    Runs ``minimize(problem, "dls")``, which takes step observers and a cancel
    token; the legacy optimizer classes take neither.

    Args:
        problem: The problem to solve. Its optic holds the best accepted
            design on return.
    """

    def __init__(self, problem: OptimizationProblem) -> None:
        self.problem = problem

    def optimize(
        self,
        maxiter: int = 1000,
        tol: float = 1e-3,
        scheme: str = "forward",
        observers: list[Observer] | None = None,
        cancel_token: CancelToken | None = None,
    ) -> OptimizationResult:
        """Run the native solver.

        Args:
            maxiter: Maximum residual evaluations, Jacobian columns included.
            tol: Relative merit decrease below which the run converges.
            scheme: Finite-difference scheme, ``"forward"`` or ``"central"``.
            observers: Notified after every accepted step.
            cancel_token: Stops the run at the next accepted step once set.

        Returns:
            The solver result.
        """
        from optiland.optimization import minimize

        return minimize(
            self.problem,
            "dls",
            maxiter=maxiter,
            tol=tol,
            observers=observers,
            cancel_token=cancel_token,
            scheme=scheme,
        )


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
    if isinstance(value, list | tuple):
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


def operand_arguments(optic, definition, operand_metadata):
    """Resolve one operand's inputs against the explicitly owned optic."""
    values = definition.get("input_data")
    values = (
        dict(values)
        if isinstance(values, dict)
        else json.loads(definition.get("input_data_str") or "{}")
    )
    if not isinstance(values, dict):
        raise ValueError("Operand parameters must be a JSON object.")
    if "wavelength" in values or "wavelength" in operand_metadata.get(
        definition["type"], {}
    ):
        values["wavelength"] = resolve_wavelength(
            optic, values.get("wavelength", "primary")
        )
    values["optic"] = optic
    return values


def add_operands(
    problem: OptimizationProblem,
    optic: Optic,
    operands: list[dict],
    operand_metadata: dict,
) -> None:
    """Add operand definitions to a problem, in list order."""
    for definition in operands:
        problem.add_operand(
            operand_type=definition["type"],
            target=definition.get("target"),
            min_val=definition.get("min_val"),
            max_val=definition.get("max_val"),
            weight=definition.get("weight", 1.0),
            input_data=operand_arguments(optic, definition, operand_metadata),
        )


def build_problem(
    optic: Optic, variables: list[dict], operands: list[dict], operand_metadata: dict
) -> OptimizationProblem:
    """Fail setup explicitly instead of silently optimizing an incomplete problem."""
    problem = OptimizationProblem()
    for definition in variables:
        problem.add_variable(
            optic, definition["type"], **variable_arguments(optic, definition)
        )
    add_operands(problem, optic, operands, operand_metadata)
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

    def report(force=False, x=None):
        nonlocal last_report, last_preview
        now = time.monotonic()
        if not force and now - last_report < PROGRESS_INTERVAL:
            return
        last_report = now
        if x is None:
            x = [float(be.to_numpy(variable.value)) for variable in problem.variables]
        # Only the scaled vector crosses to the GUI, which applies it to its
        # own copy of the optic.
        details = {
            **counters,
            "iteration": counters["callbacks"],
            "merit": latest_merit,
            "elapsed": now - started,
            "x": x,
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

    def callback(*args, **kwargs):
        check_cancelled(cancelled)
        counters["callbacks"] += 1
        report()

    def step(iteration, evaluations, merit, x):
        nonlocal latest_merit
        counters["callbacks"] = iteration
        counters["evaluations"] = evaluations
        latest_merit = merit
        report(x=x)

    kwargs = dict(parameters["optimizer_kwargs"])
    signature = inspect.signature(optimizer.optimize)
    if "observers" in signature.parameters:
        # Native solvers report accepted steps and stop cleanly through the
        # token. A raise inside their evaluator would only reject one trial.
        kwargs["observers"] = [StepObserver(step)]
        kwargs["cancel_token"] = EventCancelToken(cancelled)
    else:
        problem.update_optics = update
        # Wrap the outer evaluation boundary: least-squares catches exceptions
        # from individual operands, so cancellation inside op.fun alone would
        # be swallowed.
        if hasattr(optimizer, "_compute_residuals_vector"):
            optimizer._compute_residuals_vector = wrap_objective(
                optimizer._compute_residuals_vector, residuals=True
            )
        elif optimizer_type.__name__ == "OrthogonalDescent":
            problem.rss = wrap_objective(problem.rss)
        else:
            optimizer._fun = wrap_objective(optimizer._fun, squared=True)

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
    if "workers" in signature.parameters:
        # The shared executor already isolates this owned problem. Nested pools
        # would copy its state and escape the coordinator's cancellation policy.
        kwargs["workers"] = 1
    if "callback" in signature.parameters:
        kwargs["callback"] = callback
    if "plot" in signature.parameters:
        kwargs["plot"] = False
    check_cancelled(cancelled)
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
