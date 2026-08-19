"""Optimization Problem Module

This module contains the OptimizationProblem class, which represents an
optimization problem. The class allows for the addition of operands and
variables to the merit function, and provides methods to evaluate the merit
function and print information about the optimization problem.

Kramer Harrison, 2025
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd

import optiland.backend as be
from optiland.optimization.operand import OperandManager
from optiland.optimization.variable import VariableManager

if TYPE_CHECKING:
    from optiland.optimization.batched_evaluator import BatchedRayEvaluator
    from optiland.optimization.scaling.base import Scaler


class OptimizationProblem:
    """Represents an optimization problem.

    Attributes:
        operands (list): List of operands in the merit function.
        variables (list): List of variables in the merit function.
        initial_value (float): Initial value of the merit function.

    Methods:
        add_operand: Add an operand to the merit function.
        add_variable: Add a variable to the merit function.
        fun_array: Array of operand weighted deltas squared, where the delta
            is the difference between the current and target value.
        sum_squared: Sum of squared operand weighted deltas.
        rss: Root Sum of Squares (RSS) of the current merit function.
        operand_info: Print information about the operands in the merit
            function.
        variable_info: Print information about the variables in the merit
            function.
        info: Print information about the merit function, including operand
            and variable info.

    """

    def __init__(self, batching: bool = True):
        """Initialize an optimization problem.

        Args:
            batching: If ``True`` (default), batched ray evaluation is enabled.
                Set to ``False`` to opt out and use per-operand evaluation.
        """
        self.operands = OperandManager()
        self.variables = VariableManager()
        self.initial_value = 0.0
        self._batched_evaluator: BatchedRayEvaluator | None = None

        # Enable gradient tracking for PyTorch
        if be.get_backend() == "torch" and not be.grad_mode.requires_grad:
            warnings.warn("Gradient tracking is enabled for PyTorch.", stacklevel=2)
            be.grad_mode.enable()

        if batching:
            self.enable_batching()

    @staticmethod
    def _to_item(x):
        """
        Convert a single-element backend array to a Python scalar.
        This is a utility for printing and string formatting.
        """
        if x is None:
            return None
        if hasattr(x, "item"):
            return x.item()
        return x

    def enable_batching(self):
        """Enable batched ray evaluation for faster optimization.

        When batching is enabled, operands that require ray tracing are
        grouped by optic and wavelength so that redundant traces are
        eliminated. This can dramatically speed up merit-function
        evaluation for problems with many ray operands.

        The evaluator is re-created whenever this method is called, so
        it always reflects the current set of operands.
        """
        from optiland.optimization.batched_evaluator import BatchedRayEvaluator

        self._batched_evaluator = BatchedRayEvaluator(self)

    def disable_batching(self):
        """Disable batched ray evaluation and use standard per-operand
        evaluation."""
        self._batched_evaluator = None

    @property
    def batching_enabled(self) -> bool:
        """Whether batched evaluation is currently active."""
        return self._batched_evaluator is not None

    def add_operand(
        self,
        operand_type=None,
        target=None,
        min_val=None,
        max_val=None,
        weight=1,
        input_data=None,
    ):
        """Add an operand to the merit function"""
        if input_data is None:
            input_data = {}
        self.operands.add(operand_type, target, min_val, max_val, weight, input_data)
        # Invalidate batch plan when operands change
        if self._batched_evaluator is not None:
            self._batched_evaluator.refresh()

    def add_variable(self, optic, variable_type, scaler: Scaler = None, **kwargs):
        """Add a variable to the merit function"""
        self.variables.add(optic, variable_type, scaler=scaler, **kwargs)

    def clear_operands(self):
        """Clear all operands from the merit function"""
        self.initial_value = 0.0
        self.operands.clear()
        if self._batched_evaluator is not None:
            self._batched_evaluator.refresh()

    def clear_variables(self):
        """Clear all variables from the merit function"""
        self.initial_value = 0.0
        self.variables.clear()

    def fun_array(self):
        """Array of operand contribution terms for the merit function.

        Each term is computed as::

            (effective_weight(op) * op.delta()) ** 2

        where ``effective_weight = operand.weight * field_weight * wl_weight``.
        Field and wavelength weights are read from the optic stored in each
        operand's ``input_data``. Operands with an effective weight of zero are
        excluded from the result.

        When batching is enabled, delegates to the
        :class:`~optiland.optimization.batched_evaluator.BatchedRayEvaluator`
        which minimises redundant ray traces.

        Returns:
            be.ndarray: 1-D array of per-operand contribution values. Returns
            ``[0.0]`` when there are no active operands.
        """
        if self._batched_evaluator is not None:
            return self._batched_evaluator.fun_array()

        terms = []
        for op in self.operands:
            ew = op.effective_weight()
            if ew == 0.0:
                continue
            terms.append((ew * op.delta()) ** 2)
        if not terms:
            return be.array([0.0])
        return be.stack(terms)

    def residual_vector(self):
        """Vector of weighted operand deltas (unsquared).

        Returns a 1-D array whose *i*-th element is
        ``weight_i * delta_i`` for each operand. This is the residual
        vector **r** needed by least-squares algorithms such as the
        Damped Least-Squares (Levenberg-Marquardt) optimizer.

        Unlike :meth:`fun_array`, the values are *not* squared, so the
        merit function equals ``sum(residual_vector() ** 2)``.

        When batching is enabled, delegates to the
        :class:`~optiland.optimization.batched_evaluator.BatchedRayEvaluator`
        which minimises redundant ray traces.

        Returns:
            be.ndarray: A 1-D array of length ``len(self.operands)``.
        """
        if self._batched_evaluator is not None:
            return self._batched_evaluator.residual_vector()
        terms = []
        for op in self.operands:
            ew = op.effective_weight()
            if ew == 0.0:
                terms.append(be.array(0.0))
            else:
                terms.append(ew * op.delta())
        if not terms:
            return be.array([])
        return be.stack(terms)

    def sum_squared(self):
        """Calculate the sum of squared operand weighted deltas.

        When batching is enabled, delegates to the
        :class:`~optiland.optimization.batched_evaluator.BatchedRayEvaluator`
        which minimises redundant ray traces.
        """
        if self._batched_evaluator is not None:
            return self._batched_evaluator.sum_squared()
        return be.sum(self.fun_array())

    def rss(self):
        """RSS of current merit function"""
        return be.sqrt(self.sum_squared())

    def update_optics(self):
        """Update all optics considered in the optimization problem"""
        unique_optics = set()
        for var in self.variables:
            unique_optics.add(var.optic)
        for optic in unique_optics:
            optic.updater.update()

    def operand_info(self):
        """Print information about the operands in the merit function"""
        data = {
            "Operand Type": [op.operand_type.replace("_", " ") for op in self.operands],
            "Target": [
                f"{self._to_item(op.target):+.3f}" if op.target is not None else ""
                for op in self.operands
            ],
            "Min. Bound": [
                self._to_item(op.min_val) if op.min_val is not None else ""
                for op in self.operands
            ],
            "Max. Bound": [
                self._to_item(op.max_val) if op.max_val is not None else ""
                for op in self.operands
            ],
            "Weight": [self._to_item(op.weight) for op in self.operands],
            "Eff. Weight": [
                self._to_item(op.effective_weight()) for op in self.operands
            ],
            "Value": [f"{self._to_item(op.value):+.3f}" for op in self.operands],
            "Delta": [f"{self._to_item(op.delta()):+.3f}" for op in self.operands],
        }

        df = pd.DataFrame(data)

        # Contribution uses (effective_weight × delta)² per operand.
        ew_list = [op.effective_weight() for op in self.operands]
        contrib_values = []
        for op, ew in zip(self.operands, ew_list, strict=False):
            if ew == 0.0:
                contrib_values.append(be.array(0.0))
            else:
                contrib_values.append((be.array(ew) * op.delta()) ** 2)

        total = sum(self._to_item(v) for v in contrib_values)

        if total == 0.0:
            df["Contrib. [%]"] = 0.0
        else:
            df["Contrib. [%]"] = [
                round(self._to_item(v) / total * 100, 2) for v in contrib_values
            ]

        print(df.to_markdown(headers="keys", tablefmt="fancy_outline"))

    def variable_info(self):
        """Print information about the variables in the merit function."""
        data = {
            "Variable Type": [var.type for var in self.variables],
            "Surface": [var.surface_number for var in self.variables],
            "Value": [
                self._to_item(var.variable.inverse_scale(var.value))
                for var in self.variables
            ],
            "Min. Bound": [self._to_item(var.min_val) for var in self.variables],
            "Max. Bound": [self._to_item(var.max_val) for var in self.variables],
        }

        df = pd.DataFrame(data)
        print(df.to_markdown(headers="keys", tablefmt="fancy_outline"))

    def merit_info(self):
        """Print information about the merit function."""
        current_value = self.sum_squared()

        # Convert tensor to a Python scalar for calculations and printing
        printable_current_value = self._to_item(current_value)

        if self.initial_value == 0.0:
            improve_percent = 0.0
        else:
            improve_percent = (
                (self.initial_value - printable_current_value)
                / self.initial_value
                * 100
            )

        data = {
            "Merit Function Value": [printable_current_value],
            "Improvement (%)": improve_percent,
        }
        df = pd.DataFrame(data)
        print(df.to_markdown(headers="keys", tablefmt="fancy_outline"))

    def weight_breakdown(self) -> list[dict]:
        """Return a list of dicts describing each operand's effective weight.

        The effective weight is the product of the operand's own weight, the
        field weight (looked up from the optic via the operand's ``input_data``),
        and the wavelength weight.  The formula used in the merit function is::

            (effective_weight × delta) ** 2

        Each returned dict contains:

        * ``operand_type`` (str): The operand type string.
        * ``field``: The field index or coordinate from ``input_data`` (or None).
        * ``wavelength``: The wavelength index or value from ``input_data``
          (or None).
        * ``operand_weight`` (float): The user-set ``Operand.weight``.
        * ``field_weight`` (float): The field's weight from the optic (1.0 if
          not resolvable).
        * ``wl_weight`` (float): The wavelength's weight from the optic (1.0 if
          not resolvable).
        * ``effective_weight`` (float): Product of the three weights above.

        Returns:
            list[dict]: One dict per operand in ``self.operands``.
        """
        import contextlib

        rows = []
        for op in self.operands:
            optic = op.input_data.get("optic") if op.input_data else None
            field_idx = op.input_data.get("field") if op.input_data else None
            wl_idx = op.input_data.get("wavelength") if op.input_data else None

            field_w = 1.0
            wl_w = 1.0
            if optic is not None:
                if field_idx is not None and isinstance(field_idx, int):
                    with contextlib.suppress(IndexError):
                        field_w = optic.fields.fields[field_idx].weight
                if wl_idx is not None and isinstance(wl_idx, int):
                    with contextlib.suppress(IndexError):
                        wl_w = optic.wavelengths.wavelengths[wl_idx].weight

            rows.append(
                {
                    "operand_type": op.operand_type,
                    "field": field_idx,
                    "wavelength": wl_idx,
                    "operand_weight": op.weight,
                    "field_weight": field_w,
                    "wl_weight": wl_w,
                    "effective_weight": op.weight * field_w * wl_w,
                }
            )
        return rows

    def info(self):
        """Print information about the optimization problem."""
        self.merit_info()
        self.operand_info()
        self.variable_info()
