"""Full OptimizationService for the Optiland GUI.

Manages variable/operand lists, builds ``OptimizationProblem`` objects, and
runs the optimizer in a ``QThread`` so the main thread stays responsive.
"""

from __future__ import annotations

import json
import logging

from PySide6.QtCore import QObject, QThread, Signal, Slot

logger = logging.getLogger(__name__)


class _OptimizationWorker(QObject):
    """Runs ``optimizer.optimize()`` on a background ``QThread``.

    Signals:
        finished (str): Summary string emitted when optimisation completes.
        error (str): Error message emitted if ``optimize()`` raises.
        progress (int): Iteration counter emitted via the callback (where
            the optimizer supports one).
    """

    finished = Signal(str)
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, optimizer: object, kwargs: dict) -> None:
        super().__init__()
        self._optimizer = optimizer
        self._kwargs = kwargs
        self._iteration_count = 0
        self._cancelled = False

    def request_cancel(self) -> None:
        """Signal that the run should be aborted at the next callback."""
        self._cancelled = True

    def _make_callback(self):
        """Return a callback function that forwards progress signals."""

        def callback(*args, **kwargs):  # noqa: ANN002, ANN003
            if self._cancelled:
                return True  # Truthy return stops some scipy optimisers
            self._iteration_count += 1
            self.progress.emit(self._iteration_count)

        return callback

    @Slot()
    def run(self) -> None:
        """Execute the optimiser.  Called by QThread.started."""
        import inspect

        try:
            initial_merit = float(self._optimizer.problem.rss())
        except Exception:
            initial_merit = float("nan")

        try:
            run_kwargs = dict(self._kwargs)
            sig = inspect.signature(self._optimizer.optimize)
            if "callback" in sig.parameters:
                run_kwargs.setdefault("callback", self._make_callback())

            self._optimizer.optimize(**run_kwargs)

            try:
                final_merit = float(self._optimizer.problem.rss())
            except Exception:
                final_merit = float("nan")

            summary = (
                f"Optimization complete.\n"
                f"Iterations: {self._iteration_count}\n"
                f"Initial merit: {initial_merit:.6f}\n"
                f"Final merit:   {final_merit:.6f}"
            )
            self.finished.emit(summary)
        except Exception as exc:
            logger.exception("Optimization worker error")
            self.error.emit(str(exc))


class OptimizationService:
    """Manages optimization variables, operands, and threaded execution.

    Args:
        connector: The :class:`~optiland_gui.optiland_connector.OptilandConnector`
            instance that owns this service.
    """

    # ------------------------------------------------------------------
    # Catalog constants
    # ------------------------------------------------------------------

    OPERAND_CATEGORIES: dict[str, list[str]] = {
        "Paraxial": [
            "f1",
            "f2",
            "F1",
            "F2",
            "P1",
            "P2",
            "N1",
            "N2",
            "EPD",
            "EPL",
            "XPD",
            "XPL",
            "magnification",
            "total_track",
        ],
        "Aberration": [
            "seidel",
            "TSC",
            "SC",
            "CC",
            "TCC",
            "TAC",
            "AC",
            "TPC",
            "PC",
            "DC",
            "TAchC",
            "LchC",
            "TchC",
            "TSC_sum",
            "SC_sum",
            "CC_sum",
            "TCC_sum",
            "TAC_sum",
            "AC_sum",
            "TPC_sum",
            "PC_sum",
            "DC_sum",
            "TAchC_sum",
            "LchC_sum",
            "TchC_sum",
        ],
        "Ray": [
            "real_x_intercept",
            "real_y_intercept",
            "real_z_intercept",
            "real_x_intercept_lcs",
            "real_y_intercept_lcs",
            "real_z_intercept_lcs",
            "clearance",
            "real_L",
            "real_M",
            "real_N",
            "rms_spot_size",
            "OPD_difference",
            "AOI",
        ],
        "Lens": ["edge_thickness"],
    }

    COMMON_VARIABLE_TYPES: list[tuple[str, str]] = [
        ("Radius", "radius"),
        ("Thickness", "thickness"),
        ("Conic", "conic"),
        ("Asphere Coeff", "asphere_coeff"),
        ("Index", "index"),
        ("Tilt", "tilt"),
        ("Decenter", "decenter"),
    ]

    # Required extra keys (beyond 'optic') per operand type.
    # Deprecated: use OPERAND_METADATA instead.
    _REQUIRED_KEYS: dict[str, list[str]] = {
        "clearance": ["surface_number"],
        "edge_thickness": ["surface_number"],
        "real_x_intercept": ["surface_number"],
        "real_y_intercept": ["surface_number"],
        "real_z_intercept": ["surface_number"],
        "real_x_intercept_lcs": ["surface_number"],
        "real_y_intercept_lcs": ["surface_number"],
        "real_z_intercept_lcs": ["surface_number"],
        "real_L": ["surface_number"],
        "real_M": ["surface_number"],
        "real_N": ["surface_number"],
        "AOI": ["surface_number"],
        "seidel": ["seidel_number", "surface_number"],
    }

    # Default extra input_data (JSON string, optic excluded) per operand type.
    # Deprecated: use OPERAND_METADATA instead.
    _DEFAULT_INPUT_DATA: dict[str, str] = {
        "clearance": '{"surface_number": 1}',
        "edge_thickness": '{"surface_number": 1}',
        "real_x_intercept": '{"surface_number": 1}',
        "real_y_intercept": '{"surface_number": 1}',
        "real_z_intercept": '{"surface_number": 1}',
        "real_x_intercept_lcs": '{"surface_number": 1}',
        "real_y_intercept_lcs": '{"surface_number": 1}',
        "real_z_intercept_lcs": '{"surface_number": 1}',
        "real_L": '{"surface_number": 1}',
        "real_M": '{"surface_number": 1}',
        "real_N": '{"surface_number": 1}',
        "AOI": '{"surface_number": 1}',
        "seidel": '{"seidel_number": 0, "surface_number": 1}',
    }

    # Graphical configuration metadata for variables.
    # Maps variable type -> dict of parameter name -> metadata.
    VARIABLE_METADATA: dict[str, dict] = {
        "radius": {},
        "thickness": {},
        "conic": {},
        "asphere_coeff": {
            "coeff_number": {"type": "int", "default": 0, "min": 0, "max": 20}
        },
        "index": {"wavelength": {"type": "wavelength", "default": "primary"}},
        "tilt": {"axis": {"type": "choice", "options": ["x", "y"], "default": "x"}},
        "decenter": {"axis": {"type": "choice", "options": ["x", "y"], "default": "x"}},
        "polynomial_coeff": {"coeff_index": {"type": "int", "default": 0}},
        "chebyshev_coeff": {"coeff_index": {"type": "int", "default": 0}},
        "zernike_coeff": {"coeff_index": {"type": "int", "default": 0}},
        "reciprocal_radius": {},
        "forbes_qbfs_coeff": {"coeff_number": {"type": "int", "default": 0}},
        "forbes_qnormalslope_coeff": {"coeff_number": {"type": "int", "default": 0}},
        "forbes_q2d_coeff": {"coeff_number": {"type": "int", "default": 0}},
        "norm_radius": {},
        "nurbs_control_point": {"coeff_index": {"type": "int", "default": 0}},
        "nurbs_weight": {"coeff_index": {"type": "int", "default": 0}},
    }

    # Graphical configuration metadata for operands.
    # Maps operand type -> dict of parameter name -> metadata.
    OPERAND_METADATA: dict[str, dict] = {}  # Populated below

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, connector: object) -> None:
        self._connector = connector
        self._variables: list[dict] = []
        self._operands: list[dict] = []
        self._thread: QThread | None = None
        self._worker: _OptimizationWorker | None = None
        self._init_operand_metadata()
        self._init_optimizer_metadata()

    def _init_operand_metadata(self) -> None:
        """Initialize the OPERAND_METADATA dictionary."""
        # Common parameter groups
        std_ray = {
            "surface_number": {"type": "int", "default": 1},
            "Hx": {"type": "float", "default": 0.0},
            "Hy": {"type": "float", "default": 0.0},
            "Px": {"type": "float", "default": 0.0},
            "Py": {"type": "float", "default": 0.0},
            "wavelength": {"type": "wavelength", "default": "primary"},
        }
        dist_ray = {
            "surface_number": {"type": "int", "default": 1},
            "Hx": {"type": "float", "default": 0.0},
            "Hy": {"type": "float", "default": 0.0},
            "num_rays": {"type": "int", "default": 6},
            "wavelength": {"type": "wavelength", "default": "primary"},
            "distribution": {
                "type": "choice",
                "options": ["hexapolar", "grid", "uniform", "random"],
                "default": "hexapolar",
            },
        }

        meta = self.OPERAND_METADATA

        # Aberrations
        for op in [
            "TSC",
            "SC",
            "CC",
            "TCC",
            "TAC",
            "AC",
            "TPC",
            "PC",
            "DC",
            "TAchC",
            "LchC",
            "TchC",
        ]:
            meta[op] = {"surface_number": {"type": "int", "default": 1}}

        meta["seidel"] = {
            "seidel_number": {"type": "int", "default": 1, "min": 1, "max": 5},
            "surface_number": {"type": "int", "default": 1},
        }

        # Ray Intercepts and Cosines
        for op in [
            "real_x_intercept",
            "real_y_intercept",
            "real_z_intercept",
            "real_x_intercept_lcs",
            "real_y_intercept_lcs",
            "real_z_intercept_lcs",
            "real_L",
            "real_M",
            "real_N",
            "AOI",
        ]:
            meta[op] = std_ray.copy()

        # Others
        meta["rms_spot_size"] = dist_ray.copy()
        meta["OPD_difference"] = {
            "Hx": {"type": "float", "default": 0.0},
            "Hy": {"type": "float", "default": 0.0},
            "num_rays": {"type": "int", "default": 6},
            "wavelength": {"type": "wavelength", "default": "primary"},
            "distribution": {
                "type": "choice",
                "options": ["gaussian_quad", "hexapolar", "grid"],
                "default": "gaussian_quad",
            },
        }

        meta["edge_thickness"] = {"surface_number": {"type": "int", "default": 1}}

        # Clearance is special
        meta["clearance"] = {
            "line_ray_surface_idx": {"type": "int", "default": 1},
            "line_ray_field_coords": {"type": "tuple", "default": [0.0, 0.0]},
            "line_ray_pupil_coords": {"type": "tuple", "default": [0.0, 0.0]},
            "point_ray_surface_idx": {"type": "int", "default": 1},
            "point_ray_field_coords": {"type": "tuple", "default": [0.0, 0.0]},
            "point_ray_pupil_coords": {"type": "tuple", "default": [0.0, 0.0]},
            "wavelength": {"type": "wavelength", "default": "primary"},
        }

    def _init_optimizer_metadata(self) -> None:
        """Populate the OPTIMIZER_METADATA dictionary."""
        from optiland.optimization.optimizer.scipy import (
            SHGO,
            BasinHopping,
            DifferentialEvolution,
            DualAnnealing,
            LeastSquares,
            OptimizerGeneric,
            OrthogonalDescent,
        )

        meta = self.OPTIMIZER_METADATA

        meta[OptimizerGeneric] = {
            "method": {
                "type": "choice",
                "options": [
                    "Default",
                    "Nelder-Mead",
                    "Powell",
                    "CG",
                    "BFGS",
                    "L-BFGS-B",
                    "TNC",
                    "COBYLA",
                    "SLSQP",
                    "trust-constr",
                ],
                "default": "Default",
            },
            "maxiter": {"type": "int", "default": 1000},
            "tol": {"type": "float", "default": 1e-3, "decimals": 6},
            "disp": {"type": "bool", "default": True},
        }

        meta[LeastSquares] = {
            "maxiter": {"type": "int", "default": 1000},
            "tol": {"type": "float", "default": 1e-3, "decimals": 6},
            "method_choice": {
                "type": "choice",
                "options": ["lm", "trf", "dogbox"],
                "default": "lm",
            },
            "disp": {"type": "bool", "default": True},
        }

        meta[OrthogonalDescent] = {
            "max_iter": {"type": "int", "default": 100},
            "tol": {"type": "float", "default": 1e-4, "decimals": 6},
        }

        # Global optimizers generally share these
        global_params = {
            "maxiter": {"type": "int", "default": 1000},
            "disp": {"type": "bool", "default": True},
        }
        for cls in [DualAnnealing, DifferentialEvolution, SHGO, BasinHopping]:
            meta[cls] = global_params.copy()

    def get_optimizer_metadata(self, optimizer_cls: type) -> dict:
        """Return optimization parameter metadata for an optimizer class."""
        return self.OPTIMIZER_METADATA.get(optimizer_cls, {})

    # ------------------------------------------------------------------
    # Variable management
    # ------------------------------------------------------------------

    def add_variable(self, var_dict: dict) -> None:
        """Append a variable descriptor.

        Args:
            var_dict: Required keys: ``surface_number`` (int), ``type`` (str).
                Optional keys: ``min_val`` (float|None), ``max_val``
                (float|None), ``coeff_number`` (int|None).
        """
        self._variables.append(dict(var_dict))

    def remove_variable(self, index: int) -> None:
        """Remove a variable by its list index.

        Args:
            index: Zero-based index.
        """
        if 0 <= index < len(self._variables):
            self._variables.pop(index)

    def get_variables(self) -> list[dict]:
        """Return a shallow copy of the variable list."""
        return list(self._variables)

    def set_variable(self, index: int, var_dict: dict) -> None:
        """Replace a variable at *index* with *var_dict*."""
        if 0 <= index < len(self._variables):
            self._variables[index] = dict(var_dict)

    def get_variable_metadata(self, var_type: str) -> dict:
        """Return graphical configuration metadata for a variable type.

        Args:
            var_type: Variable type key.

        Returns:
            Dict of parameter metadata (defaulting to surface_number only).
        """
        return self.VARIABLE_METADATA.get(
            var_type, {"surface_number": {"type": "int", "default": 1}}
        )

    def clear_variables(self) -> None:
        """Remove all registered variables."""
        self._variables.clear()

    def get_variable_current_value(self, var_dict: dict) -> float | None:
        """Read the current physical value for a variable from the live optic.

        Args:
            var_dict: A variable descriptor dict.

        Returns:
            The current float value, or ``None`` if retrieval fails.
        """
        optic = self._connector._optic
        if optic is None:
            return None
        try:
            from optiland.optimization.variable.variable import Variable as _Var

            extra: dict = {}
            if var_dict.get("coeff_number") is not None:
                extra["coeff_number"] = var_dict["coeff_number"]
            v = _Var(
                optic,
                var_dict["type"],
                surface_number=var_dict["surface_number"],
                **extra,
            )
            return float(v.variable.get_value())
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Operand management
    # ------------------------------------------------------------------

    def add_operand(self, op_dict: dict) -> None:
        """Append an operand descriptor.

        Args:
            op_dict: Required keys: ``type`` (str), ``category`` (str).
                Optional keys: ``target`` (float|None), ``min_val``
                (float|None), ``max_val`` (float|None), ``weight`` (float),
                ``input_data_str`` (str — JSON without optic).
        """
        self._operands.append(dict(op_dict))

    def remove_operand(self, index: int) -> None:
        """Remove an operand by its list index.

        Args:
            index: Zero-based index.
        """
        if 0 <= index < len(self._operands):
            self._operands.pop(index)

    def get_operands(self) -> list[dict]:
        """Return a shallow copy of the operand list."""
        return list(self._operands)

    def set_operand(self, index: int, op_dict: dict) -> None:
        """Replace an operand at *index* with *op_dict*."""
        if 0 <= index < len(self._operands):
            self._operands[index] = dict(op_dict)

    def _resolve_wavelength(self, wave_val: str | float) -> float | str:
        """Resolve 'primary' to the actual wavelength float."""
        if wave_val == "primary":
            optic = self._connector._optic
            if optic is not None:
                return float(optic.primary_wavelength)
        return wave_val

    def get_operand_current_value(self, op_dict: dict) -> float | None:
        """Read the current physical value for an operand from the live optic."""
        optic = self._connector._optic
        if optic is None:
            return None
        try:
            from optiland.optimization.operand.operand import Operand

            input_data_val = op_dict.get("input_data")
            if isinstance(input_data_val, dict):
                extra_data = dict(input_data_val)
            else:
                try:
                    extra_data = json.loads(op_dict.get("input_data_str") or "{}")
                except json.JSONDecodeError:
                    extra_data = {}

            op_dict.get("category", "General")
            op_meta = self.OPERAND_METADATA.get(op_dict["type"], {})
            if (
                "wavelength" in op_meta
                and extra_data.get("wavelength", "primary") == "primary"
            ):
                extra_data["wavelength"] = self._resolve_wavelength("primary")

            input_data = {"optic": optic, **extra_data}
            op_inst = Operand(operand_type=op_dict["type"], input_data=input_data)
            return float(op_inst.value)
        except Exception:
            return None

    def get_operand_metadata(self, op_type: str) -> dict:
        """Return graphical configuration metadata for an operand type.

        Args:
            op_type: Operand type key.

        Returns:
            Dict of parameter metadata (empty if none required).
        """
        return self.OPERAND_METADATA.get(op_type, {})

    def clear_operands(self) -> None:
        """Remove all registered operands."""
        self._operands.clear()

    def get_default_input_data_str(self, op_type: str) -> str:
        """Return the default extra-parameter JSON string for an operand type.

        Args:
            op_type: Operand type key.

        Returns:
            A JSON string (optic excluded); defaults to ``"{}"``.
        """
        return self._DEFAULT_INPUT_DATA.get(op_type, "{}")

    def validate_operand_input_data(
        self, op_type: str, input_data_str_or_dict: str | dict | None
    ) -> str | None:
        """Check that *input_data* contains all required keys for *op_type*.

        Args:
            op_type: The operand type key.
            input_data_str_or_dict: JSON string or dict of extra parameters.

        Returns:
            An error message string if validation fails, or ``None`` if valid.
        """
        required = self._REQUIRED_KEYS.get(op_type, [])
        if not required:
            return None

        if isinstance(input_data_str_or_dict, dict):
            data = input_data_str_or_dict
        else:
            try:
                data = json.loads(input_data_str_or_dict or "{}")
            except json.JSONDecodeError:
                return f"Invalid JSON in parameters for '{op_type}'"

        missing = [k for k in required if k not in data]
        if missing:
            return f"'{op_type}' requires parameter(s): {', '.join(missing)}"
        return None

    # ------------------------------------------------------------------
    # Problem construction
    # ------------------------------------------------------------------

    def build_problem(self, optic: object) -> object:
        """Build an :class:`~optiland.optimization.OptimizationProblem`.

        Injects the live optic into each operand's ``input_data`` before adding
        to the problem.

        Args:
            optic: The :class:`~optiland.optic.Optic` instance to optimise.

        Returns:
            A configured ``OptimizationProblem``.
        """
        from optiland.optimization import OptimizationProblem

        problem = OptimizationProblem()

        for vd in self._variables:
            extra: dict = {}
            if vd.get("coeff_number") is not None:
                extra["coeff_number"] = vd["coeff_number"]
            try:
                problem.add_variable(
                    optic,
                    vd["type"],
                    surface_number=vd["surface_number"],
                    min_val=vd.get("min_val"),
                    max_val=vd.get("max_val"),
                    **extra,
                )
            except Exception as exc:
                logger.warning(
                    "OptimizationService: skipping variable %s surface %s: %s",
                    vd.get("type"),
                    vd.get("surface_number"),
                    exc,
                )

        for od in self._operands:
            input_data_val = od.get("input_data")
            if isinstance(input_data_val, dict):
                extra_data = dict(input_data_val)
            else:
                try:
                    extra_data = json.loads(od.get("input_data_str") or "{}")
                except json.JSONDecodeError:
                    extra_data = {}

            # Resolve 'primary' wavelength if it's explicitly 'primary' or
            # if it's missing but the operand expects it (defaulting to primary).
            od.get("category", "General")
            op_meta = self.OPERAND_METADATA.get(od["type"], {})
            if (
                "wavelength" in op_meta
                and extra_data.get("wavelength", "primary") == "primary"
            ):
                extra_data["wavelength"] = self._resolve_wavelength("primary")

            input_data = {"optic": optic, **extra_data}
            try:
                problem.add_operand(
                    operand_type=od["type"],
                    target=od.get("target"),
                    min_val=od.get("min_val"),
                    max_val=od.get("max_val"),
                    weight=od.get("weight", 1.0),
                    input_data=input_data,
                )
            except Exception as exc:
                logger.warning(
                    "OptimizationService: skipping operand %s: %s",
                    od.get("type"),
                    exc,
                )
                tm = getattr(self._connector, "toast_manager", None)
                if tm:
                    tm.notify(f"Operand '{od.get('type')}' skipped: {exc}", "warning")

        return problem

    # ------------------------------------------------------------------
    # Optimizer catalog
    # ------------------------------------------------------------------

    # bounds_mode values: "none" | "required" | "rejected"
    # "required" = all variables must have bounds set
    # "rejected" = no variables may have bounds set
    # "none"     = bounds are optional / ignored
    _BOUNDS_REQUIREMENTS: dict[str, str] = {}  # populated by get_optimizer_groups()

    @staticmethod
    def _build_scipy_method_cls(method: str, base_cls: type) -> type:
        """Create a ScipyMethod subclass locked to *method*."""

        class ScipyMethod(base_cls):  # type: ignore[valid-type]
            def optimize(self, maxiter=1000, disp=True, tol=1e-3, callback=None):
                return super().optimize(
                    method=None if method == "Default" else method,
                    maxiter=maxiter,
                    disp=disp,
                    tol=tol,
                    callback=callback,
                )

        ScipyMethod.__name__ = f"ScipyMethod_{method.replace('-', '_')}"
        return ScipyMethod

    @staticmethod
    def get_optimizer_groups() -> dict[str, list[tuple[str, type, str]]]:
        """Return optimisers organised into ``"Local"`` and ``"Global"`` groups.

        Each entry is a ``(display_name, cls, bounds_mode)`` tuple where
        ``bounds_mode`` is one of ``"none"``, ``"required"``, or ``"rejected"``.

        Returns:
            Ordered dict mapping group name → list of (name, class, bounds_mode).
        """
        from optiland.optimization.optimizer.scipy import (
            SHGO,
            BasinHopping,
            DifferentialEvolution,
            DualAnnealing,
            LeastSquares,
            OptimizerGeneric,
            OrthogonalDescent,
        )

        local: list[tuple[str, type, str]] = [
            ("Generic (scipy.minimize)", OptimizerGeneric, "none"),
            ("Least Squares", LeastSquares, "none"),
            ("Orthogonal Descent", OrthogonalDescent, "none"),
        ]

        global_: list[tuple[str, type, str]] = [
            ("Dual Annealing [bounds req.]", DualAnnealing, "required"),
            ("Differential Evolution [bounds req.]", DifferentialEvolution, "required"),
            ("SHGO [bounds req.]", SHGO, "required"),
            ("Basin Hopping [no bounds]", BasinHopping, "rejected"),
        ]

        return {"Local": local, "Global": global_}

    # Configuration metadata for optimizers: parameters for .optimize()
    OPTIMIZER_METADATA: dict[type, dict] = {}  # Populated in __init__

    @staticmethod
    def get_optimizer_catalog() -> list[tuple[str, type]]:
        """Return all optimisers as a flat ``(display_name, cls)`` list.

        Returns:
            A list of ``(name, class)`` pairs for all available optimisers.
        """
        catalog: list[tuple[str, type]] = []
        for entries in OptimizationService.get_optimizer_groups().values():
            for name, cls, _ in entries:
                catalog.append((name, cls))
        return catalog

    def validate_bounds_for_optimizer(self, optimizer_cls: type) -> str | None:
        """Check that variable bounds match *optimizer_cls* requirements.

        Args:
            optimizer_cls: The optimizer class to validate against.

        Returns:
            An error message string if validation fails, or ``None`` if valid.
        """
        # Look up bounds_mode from the groups catalog
        bounds_mode = "none"
        for entries in self.get_optimizer_groups().values():
            for _name, cls, mode in entries:
                if cls is optimizer_cls:
                    bounds_mode = mode
                    break

        if bounds_mode == "none":
            return None

        has_all_bounds = all(
            v.get("min_val") is not None and v.get("max_val") is not None
            for v in self._variables
        )
        has_any_bounds = any(
            v.get("min_val") is not None or v.get("max_val") is not None
            for v in self._variables
        )

        if bounds_mode == "required" and not has_all_bounds:
            return (
                f"{optimizer_cls.__name__} requires bounds on all variables. "
                "Set Min/Max for each variable."
            )
        if bounds_mode == "rejected" and has_any_bounds:
            return (
                f"{optimizer_cls.__name__} does not accept bounds. "
                "Remove Min/Max from all variables."
            )
        return None

    # ------------------------------------------------------------------
    # Threaded execution
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``True`` while an optimisation run is in progress."""
        return self._thread is not None and self._thread.isRunning()

    def run(
        self,
        optimizer_cls: type,
        optimizer_kwargs: dict,
        on_progress: object | None = None,
        on_finished: object | None = None,
        on_error: object | None = None,
    ) -> None:
        """Capture undo state, build problem, then start the optimizer thread.

        Args:
            optimizer_cls: Optimizer class (e.g., ``LeastSquares``).
            optimizer_kwargs: Forwarded to ``optimizer.optimize()``.
            on_progress: Optional ``(iteration_count: int) -> None`` callback.
            on_finished: Optional ``(summary: str) -> None`` callback.
            on_error: Optional ``(message: str) -> None`` callback.
        """
        if self.is_running:
            return

        optic = self._connector._optic
        if optic is None:
            return

        # Capture undo checkpoint before modifying optic
        self._connector._undo_redo_manager.add_state(
            self._connector._capture_optic_state()
        )

        try:
            problem = self.build_problem(optic)
            optimizer = optimizer_cls(problem)
        except Exception as exc:
            if on_error:
                on_error(f"Failed to build optimisation problem: {exc}")
            return

        self._thread = QThread()
        self._worker = _OptimizationWorker(optimizer, optimizer_kwargs)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)

        if on_progress is not None:
            self._worker.progress.connect(on_progress)
        if on_finished is not None:
            self._worker.finished.connect(on_finished)
        if on_error is not None:
            self._worker.error.connect(on_error)

        self._thread.start()

    def stop(self) -> None:
        """Request cancellation of an in-progress run."""
        if self._worker is not None:
            self._worker.request_cancel()

    @Slot()
    def _on_thread_finished(self) -> None:
        """Handle thread completion by clearing the reference."""
        self._thread = None
