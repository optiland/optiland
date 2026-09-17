"""Full OptimizationService for the Optiland GUI.

Manages variable/operand lists, builds ``OptimizationProblem`` objects, and
runs the optimizer in the shared isolated calculation process.
"""

from __future__ import annotations

import json
import logging
import pickle
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal, Slot

from .job_records import JobRequest, JobResult, OpticSnapshot

if TYPE_CHECKING:
    from optiland.optic import Optic

logger = logging.getLogger(__name__)


class OptimizationService(QObject):
    """Manages definitions and accepts owned candidates on the GUI thread.

    Args:
        connector: The :class:`~optiland_gui.optiland_connector.OptilandConnector`
            instance that owns this service.
    """

    progressChanged = Signal(dict)
    completed = Signal(str)
    failed = Signal(str)
    stateChanged = Signal(str)
    candidateAvailable = Signal(bool)

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
        super().__init__(connector if isinstance(connector, QObject) else None)
        self._connector = connector
        self._variables: list[dict] = []
        self._operands: list[dict] = []
        self._request = None
        self._stopping = False
        self._candidate = None
        self._callbacks = (None, None, None)
        self.preview_options = {"enabled": False, "frequency": 10}
        jobs = connector.calculation_jobs
        jobs.progress.connect(self._on_progress)
        jobs.finished.connect(self._on_finished)
        jobs.state_changed.connect(self._on_state_changed)
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

            from .optimization_jobs import variable_arguments

            v = _Var(
                optic,
                var_dict["type"],
                **variable_arguments(optic, var_dict),
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
    # Owned process execution
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """``True`` while an optimisation run is in progress."""
        return self._request is not None

    def run(
        self,
        optimizer_cls: type,
        optimizer_kwargs: dict,
        on_progress: object | None = None,
        on_finished: object | None = None,
        on_error: object | None = None,
    ) -> None:
        """Queue a frozen candidate; callbacks are invoked by GUI-owned slots.

        Args:
            optimizer_cls: Optimizer class (e.g., ``LeastSquares``).
            optimizer_kwargs: Forwarded to ``optimizer.optimize()``.
            on_progress: Optional ``(iteration_count: int) -> None`` callback.
            on_finished: Optional ``(summary: str) -> None`` callback.
            on_error: Optional ``(message: str) -> None`` callback.
        """
        if self.is_running:
            return

        self._callbacks = (on_progress, on_finished, on_error)
        optic = self._connector._optic
        if optic is None:
            self._emit_error("Open an optical system before running optimization.")
            return

        self._stopping = False
        self._candidate = None
        self.candidateAvailable.emit(False)
        try:
            if "<locals>" in optimizer_cls.__qualname__:
                raise ValueError(
                    "Optimizer classes must be importable by the calculation process."
                )
            snapshot = OpticSnapshot.capture(optic)
            self._request = self._connector.calculation_jobs.submit(
                "optimization",
                "optiland_gui.services.optimization_jobs:optimize",
                snapshot,
                {
                    "variables": self._variables,
                    "operands": self._operands,
                    "operand_metadata": self.OPERAND_METADATA,
                    "optimizer": (
                        f"{optimizer_cls.__module__}:{optimizer_cls.__qualname__}"
                    ),
                    "optimizer_kwargs": optimizer_kwargs,
                    "preview": self.preview_options,
                },
                cancel_on_document_change=False,
                context={"edit_token": self._connector.document_state.edit_token},
            )
            self.stateChanged.emit("queued")
        except Exception as exc:
            self._emit_error(str(exc))

    def stop(self) -> None:
        """Request cancellation of an in-progress run."""
        if self._request is not None:
            self._stopping = True
            self.stateChanged.emit("cancelling")
            self._connector.calculation_jobs.cancel_target("optimization")

    @Slot(object, str)
    def _on_state_changed(self, request: JobRequest, state: str) -> None:
        if self._request is not None and request.job_id == self._request.job_id:
            self.stateChanged.emit(state)

    @Slot(object, dict)
    def _on_progress(self, request: JobRequest, message: dict) -> None:
        if (
            self._stopping
            or self._request is None
            or request.job_id != self._request.job_id
        ):
            return
        details = message.get("details") or {}
        self.progressChanged.emit(
            {
                "stage": message["stage"],
                **details,
                "definitions_current": self._definitions_match(request),
            }
        )
        callback = self._callbacks[0]
        if callback is not None and "callbacks" in details:
            callback(details["callbacks"])

    def _emit_error(self, message: str) -> None:
        callback = self._callbacks[2]
        self._callbacks = (None, None, None)
        self._request = None
        self.failed.emit(message)
        if callback is not None:
            callback(message)

    @Slot(object)
    def _on_finished(self, result: JobResult) -> None:
        if self._request is None or result.request.job_id != self._request.job_id:
            return
        # Keep ownership through document/candidate notifications. A listener
        # must not start another run before this run's callbacks are detached.
        request = self._request
        if result.status == "failed" and not self._stopping:
            self._emit_error(result.error)
            return
        if result.status == "cancelled" or self._stopping:
            summary = "Optimization cancelled. The document was not changed."
        else:
            data = result.data
            current = (
                self._connector.calculation_jobs.is_current(request)
                and request.context["edit_token"]
                == self._connector.document_state.edit_token
                and self._definitions_match(request)
            )
            if current and data["converged"]:
                try:
                    candidate = data["candidate"].restore()
                except Exception as exc:
                    self._emit_error(
                        f"Could not restore the optimized candidate: {exc}"
                    )
                    return
                self._commit(candidate, pickle.loads(request.snapshot.data))
                outcome = "Optimization converged and was applied."
            else:
                self._candidate = data
                self.candidateAvailable.emit(True)
                outcome = (
                    "Document or optimization definitions changed; "
                    "candidate retained for separate review."
                    if not current
                    else "Optimization stopped without convergence; "
                    "candidate retained for review."
                )
            iterations = data["iterations"]
            if iterations is None:
                iterations = "not reported"
            summary = (
                f"{outcome}\nInitial merit: {data['initial_merit']:.6f}\n"
                f"Final merit: {data['final_merit']:.6f}\n"
                f"Evaluations: {data['evaluations']}; iterations: {iterations}\n"
                f"Elapsed: {data['elapsed']:.2f} s\n{data['message']}"
            )
        callback = self._callbacks[1]
        self._callbacks = (None, None, None)
        self._request = None
        self.completed.emit(summary)
        if callback is not None:
            callback(summary)

    def _commit(self, candidate: Optic, previous: dict) -> None:
        self._connector._optic = candidate
        self._connector._undo_redo_manager.add_state(previous)
        self._connector.set_modified(True)
        self._connector.notify_change("replacement")

    def _definitions_match(self, request: JobRequest) -> bool:
        return (
            self._variables == request.parameters["variables"]
            and self._operands == request.parameters["operands"]
        )

    def take_candidate(self) -> dict | None:
        """Detach the retained candidate for an explicit separate-document review."""
        candidate, self._candidate = self._candidate, None
        self.candidateAvailable.emit(False)
        return candidate
