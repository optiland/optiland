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

import numpy as np
import pandas as pd

import optiland.backend as be
from optiland.optimization.block import (
    DefaultMeritBlock,
    ResidualBlock,
    direct_context,
    evaluate_blocks,
)
from optiland.optimization.constraint import ConstraintManager
from optiland.optimization.constraint.physical import (
    PHYSICAL_MODES,
    ThicknessLimits,
    thickness_specs,
)
from optiland.optimization.failure import ConfigurationError
from optiland.optimization.operand import OperandManager
from optiland.optimization.variable import VariableManager

if TYPE_CHECKING:
    from collections.abc import Sequence

    from optiland.optic.optic import Optic
    from optiland.optimization.batched_evaluator import BatchedRayEvaluator
    from optiland.optimization.scaling.base import Scaler


class OptimizationProblem:
    """Represents an optimization problem.

    Attributes:
        operands (list): List of operands in the merit function.
        variables (list): List of variables in the merit function.
        constraints (ConstraintManager): Hard constraints (experimental).
        residual_blocks (list): Residual blocks of the merit (experimental).
        initial_value (float): Initial value of the merit function.

    Methods:
        add_operand: Add an operand to the merit function.
        add_variable: Add a variable to the merit function.
        add_constraint: Add a hard constraint (experimental).
        clear_constraints: Remove all hard constraints (experimental).
        constraint_info: Print the hard constraints (experimental).
        add_residual_block: Add a residual block to the merit (experimental).
        clear_residual_blocks: Remove all residual blocks (experimental).
        add_default_merit: Add a per-ray default merit (experimental).
        add_physical_constraints: Add thickness limits (experimental).
        block_info: Print the residual blocks (experimental).
        fun_array: Array of operand weighted deltas squared, where the delta
            is the difference between the current and target value, then the
            squared rows of the residual blocks.
        sum_squared: Sum of squared operand weighted deltas and block rows.
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
        self.constraints = ConstraintManager()
        self.residual_blocks: list[ResidualBlock] = []
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

    def add_constraint(
        self,
        operand_type=None,
        target=None,
        min_val=None,
        max_val=None,
        weight=1,
        input_data=None,
        tol=1e-8,
    ):
        """Add a hard constraint. *Experimental.*

        Uses the operand spec of :meth:`add_operand`: ``target`` makes an
        equality, ``min_val`` and/or ``max_val`` an inequality. A constraint
        is not a merit term: it stays out of ``fun_array``, ``residual_vector``
        and ``sum_squared``. Only the native damped least squares enforces
        constraints; the other optimizers raise ``ConfigurationError``.

        Args:
            operand_type: Registered operand name.
            target: Equality target.
            min_val: Lower bound of the operand value.
            max_val: Upper bound of the operand value.
            weight: Positive scale of the rows, ``effective_weight * delta``.
            input_data: Operand input data.
            tol: Feasibility tolerance of each row, in row units.

        Returns:
            The new :class:`~optiland.optimization.constraint.Constraint`.

        Raises:
            ConfigurationError: If the specification is invalid.
        """
        constraint = self.constraints.add(
            operand_type, target, min_val, max_val, weight, input_data, tol
        )
        if self._batched_evaluator is not None:
            self._batched_evaluator.refresh()
        return constraint

    def clear_constraints(self):
        """Remove all hard constraints. *Experimental.*"""
        self.constraints.clear()
        if self._batched_evaluator is not None:
            self._batched_evaluator.refresh()

    def add_residual_block(self, block: ResidualBlock) -> ResidualBlock:
        """Add a residual block to the merit. *Experimental.*

        The block's rows follow the operands in ``residual_vector``,
        ``fun_array`` and ``sum_squared``, so every optimizer sees them. With
        batching, its rays share the operands' bundles.

        Args:
            block: The block.

        Returns:
            The block.

        Raises:
            ConfigurationError: If ``block`` is not a ``ResidualBlock``.
        """
        if not isinstance(block, ResidualBlock):
            raise ConfigurationError(f"Expected a ResidualBlock, got {block!r}.")
        self.initial_value = 0.0
        self.residual_blocks.append(block)
        if self._batched_evaluator is not None:
            self._batched_evaluator.refresh()
        return block

    def clear_residual_blocks(self):
        """Remove all residual blocks from the merit. *Experimental.*"""
        self.initial_value = 0.0
        self.residual_blocks = []
        if self._batched_evaluator is not None:
            self._batched_evaluator.refresh()

    def add_default_merit(
        self,
        optic: Optic,
        kind: str = "spot",
        rings: int = 3,
        arms: int = 6,
        reference: str = "centroid",
        fields: str | Sequence[int] = "all",
        wavelengths: str | Sequence[int] = "all",
        weight: float = 1.0,
    ) -> DefaultMeritBlock:
        """Add a per-ray default merit on a Gaussian pupil. *Experimental.*

        Adds one :class:`~optiland.optimization.block.DefaultMeritBlock`, no
        operands. Each residual is transverse (``dx``, ``dy``) or OPD per ray,
        times ``sqrt(quadrature_w * field_w * wavelength_w) * weight``, so
        ``sum(r ** 2)`` estimates the weighted RMS squared.

        Args:
            optic: The optic.
            kind: ``"spot"`` or ``"wavefront"``.
            rings: Quadrature rings.
            arms: Quadrature arms.
            reference: ``"centroid"`` or ``"chief"``.
            fields: ``"all"`` or field indices.
            wavelengths: ``"all"`` or wavelength indices.
            weight: Scale of every row.

        Returns:
            The new block.

        Raises:
            ConfigurationError: If an argument is invalid.
        """
        block = DefaultMeritBlock(
            optic, kind, rings, arms, reference, fields, wavelengths, weight
        )
        return self.add_residual_block(block)

    def add_physical_constraints(
        self,
        optic: Optic,
        min_center_glass: float | None = None,
        min_edge_glass: float | None = None,
        min_center_air: float | None = None,
        min_edge_air: float | None = None,
        max_center: float | None = None,
        *,
        max_center_glass: float | None = None,
        max_center_air: float | None = None,
        max_edge_glass: float | None = None,
        max_edge_air: float | None = None,
        mode: str = "constraint",
        weight: float = 1.0,
        tol: float = 1e-8,
    ) -> list:
        """Add thickness limits between lens surfaces. *Experimental.*

        Opt-in: Optiland never applies these limits by itself. A limit left as
        None is not applied. The gaps from the object and to the image are not
        limited. Limits are magnitudes, also after a mirror.

        With ``mode="constraint"`` the limits are hard constraints, which only
        the native damped least squares enforces. With ``mode="penalty"`` they
        are merit operands with ``min_val``/``max_val``, ``center_thickness``
        and ``edge_thickness``, which work with every optimizer but may be
        violated at the optimum.

        Args:
            optic: The optic.
            min_center_glass: Smallest center thickness in glass.
            min_edge_glass: Smallest edge thickness in glass.
            min_center_air: Smallest center thickness in air.
            min_edge_air: Smallest edge thickness in air.
            max_center: Largest center thickness in glass and air, unless
                ``max_center_glass`` or ``max_center_air`` is given.
            max_center_glass: Largest center thickness in glass.
            max_center_air: Largest center thickness in air.
            max_edge_glass: Largest edge thickness in glass.
            max_edge_air: Largest edge thickness in air.
            mode: ``"constraint"`` or ``"penalty"``.
            weight: Weight of each row or operand.
            tol: Feasibility tolerance of each row, for ``"constraint"``.

        Returns:
            The new :class:`~optiland.optimization.constraint.Constraint` or
            :class:`~optiland.optimization.operand.Operand` objects.

        Raises:
            ConfigurationError: If ``mode`` is unknown, no limit is set or the
                limits are inconsistent.
        """
        if mode not in PHYSICAL_MODES:
            raise ConfigurationError(
                f"Unknown mode {mode!r}; expected one of {list(PHYSICAL_MODES)}."
            )
        specs = thickness_specs(
            optic,
            ThicknessLimits(
                min_center_glass,
                max_center if max_center_glass is None else max_center_glass,
                min_edge_glass,
                max_edge_glass,
            ),
            ThicknessLimits(
                min_center_air,
                max_center if max_center_air is None else max_center_air,
                min_edge_air,
                max_edge_air,
            ),
        )
        added = []
        for operand_type, min_val, max_val, input_data in specs:
            if mode == "constraint":
                added.append(
                    self.add_constraint(
                        operand_type,
                        min_val=min_val,
                        max_val=max_val,
                        weight=weight,
                        input_data=input_data,
                        tol=tol,
                    )
                )
            else:
                self.add_operand(
                    operand_type,
                    min_val=min_val,
                    max_val=max_val,
                    weight=weight,
                    input_data=input_data,
                )
                added.append(self.operands.operands[-1])
        return added

    @property
    def num_residuals(self) -> int:
        """Length of :meth:`residual_vector`: operands plus block rows."""
        return len(self.operands) + sum(b.size() for b in self.residual_blocks)

    def block_spans(self) -> list[tuple[int, int, ResidualBlock]]:
        """``(first row, end row, block)`` of each block in the residual vector."""
        spans = []
        start = len(self.operands)
        for block in self.residual_blocks:
            stop = start + block.size()
            spans.append((start, stop, block))
            start = stop
        return spans

    def drop_block_rows(self, rows: Sequence[int] | np.ndarray) -> int:
        """Drop rows of residual blocks from the ray set. *Experimental.*

        For native drivers, ray-set re-validation. Operand rows
        are never dropped, and neither are a block's last rows.

        Args:
            rows: Indices into the residual vector.

        Returns:
            The number of rows dropped.
        """
        rows = np.asarray(rows, dtype=int)
        dropped = 0
        for start, stop, block in self.block_spans():
            new = np.setdiff1d(
                rows[(rows >= start) & (rows < stop)] - start, block.dropped_rows
            )
            if new.size and new.size + block.dropped_rows.size < stop - start:
                block.drop_rows(new)
                dropped += new.size
        return dropped

    def restore_block_rows(self) -> None:
        """Bring back all dropped block rows. *Experimental.*"""
        for block in self.residual_blocks:
            block.restore_rows()

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

        Residual block rows follow the operands, squared.

        Returns:
            be.ndarray: 1-D array of per-operand contribution values. Returns
            ``[0.0]`` when there are no active operands or block rows.
        """
        if self._batched_evaluator is not None:
            return self._batched_evaluator.fun_array()

        terms = []
        for op in self.operands:
            ew = op.effective_weight()
            if ew == 0.0:
                continue
            terms.append((ew * op.delta()) ** 2)
        if self.residual_blocks:
            rows = self._direct_block_rows() ** 2
            return be.concatenate([be.stack(terms), rows]) if terms else rows
        if not terms:
            return be.array([0.0])
        return be.stack(terms)

    def _direct_block_rows(self):
        """Block rows from one trace per request, for unbatched evaluation."""
        context = direct_context(self.residual_blocks)
        return evaluate_blocks(self.residual_blocks, context)

    def residual_vector(self):
        """Vector of weighted operand deltas (unsquared).

        Returns a 1-D array whose *i*-th element is
        ``weight_i * delta_i`` for each operand. This is the residual
        vector **r** needed by least-squares algorithms such as the
        Damped Least-Squares (Levenberg-Marquardt) optimizer.

        Unlike :meth:`fun_array`, the values are *not* squared, so the
        merit function equals ``sum(residual_vector() ** 2)``.

        Residual block rows follow the operands, already weighted.

        When batching is enabled, delegates to the
        :class:`~optiland.optimization.batched_evaluator.BatchedRayEvaluator`
        which minimises redundant ray traces.

        Returns:
            be.ndarray: A 1-D array of length :attr:`num_residuals`.
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
        if self.residual_blocks:
            rows = self._direct_block_rows()
            return be.concatenate([be.stack(terms), rows]) if terms else rows
        if not terms:
            return be.array([])
        return be.stack(terms)

    def constraint_vector(self):
        """Signed constraint rows at the current state. *Experimental.*

        Each row is ``effective_weight * delta``, signed so that an equality
        is feasible at 0 and an inequality at ``<= 0``.

        Returns:
            be.ndarray: A 1-D array with one entry per row.
        """
        if self._batched_evaluator is not None:
            full = self._batched_evaluator.residuals_and_constraints()
            return full[self.num_residuals :]
        return self.constraints.vector()

    def residuals_and_constraints(self):
        """Residuals followed by constraint rows, from one evaluation.

        With batching, constraint rows share the merit's ray bundles.
        *Experimental.*

        Returns:
            be.ndarray: :meth:`residual_vector` then :meth:`constraint_vector`.
        """
        if self._batched_evaluator is not None:
            return self._batched_evaluator.residuals_and_constraints()
        return be.concatenate(
            [be.ravel(self.residual_vector()), be.ravel(self.constraints.vector())]
        )

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
        total += sum(self._block_merits())

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

    def constraint_info(self):
        """Print the hard constraint rows at the current state. *Experimental.*"""
        rows = self.constraints.rows
        report = self.constraints.report(self.constraint_vector())
        data = {
            "Constraint": [item["label"] for item in report],
            "Kind": [item["kind"] for item in report],
            "Eff. Weight": [
                self._to_item(row.operand.effective_weight()) for row in rows
            ],
            "Residual": [f"{item['residual']:+.3e}" for item in report],
            "Violation": [f"{item['violation']:.3e}" for item in report],
            "Feasible": [item["feasible"] for item in report],
        }
        df = pd.DataFrame(data)
        print(df.to_markdown(headers="keys", tablefmt="fancy_outline"))

    def _block_merits(self) -> list[float]:
        """Sum of the squared rows of each block."""
        spans = self.block_spans()
        if not spans:
            return []
        r = self.residual_vector()
        return [self._to_item(be.sum(r[a:b] ** 2)) for a, b, _ in spans]

    def block_info(self):
        """Print one summary row per residual block. *Experimental.*"""
        merits = self._block_merits()
        total = self._to_item(self.sum_squared())
        rows = []
        for (start, stop, block), merit in zip(self.block_spans(), merits, strict=True):
            rows.append(
                {
                    "Block": type(block).__name__,
                    **block.describe(),
                    "Rows": stop - start,
                    "Dropped": block.dropped_rows.size,
                    "Merit": f"{merit:.6g}",
                    "Contrib. [%]": round(merit / total * 100, 2) if total else 0.0,
                }
            )
        df = pd.DataFrame(rows).fillna("")
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
        if self.residual_blocks:
            self.block_info()
        self.variable_info()
        if len(self.constraints):
            self.constraint_info()
