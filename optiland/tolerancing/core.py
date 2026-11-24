"""Core Tolerancing Module

This module contains the core classes for the tolerancing module. The
Tolerancing class is the main class for defining a tolerancing problem. It
allows the user to add operands, perturbations, and compensators to an optical
system.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from optiland.optimization.operand import Operand
from optiland.tolerancing import types as _types  # noqa: F401 (registers types)
from optiland.tolerancing.compensator import (
    CompensatorOptimizer,
    QuickFocusCompensator,
)
from optiland.tolerancing.perturbation import BaseSampler, Perturbation
from optiland.tolerancing.presets import add_default_tolerances
from optiland.tolerancing.registry import tolerance_registry
from optiland.tolerancing.tolerance import Tolerance

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from optiland.iso10110 import DrawingSpec
    from optiland.tolerancing.base import BasePerturbation
    from optiland.tolerancing.distributions import Distribution


class OperandEvaluator:
    """Evaluates a set of named operands against an Optic instance.

    Args:
        operands: List of Operand objects to evaluate.
    """

    def __init__(self, operands: list[Operand]) -> None:
        self._operands = operands

    def evaluate(self) -> list[float]:
        """Return evaluated values for the current optic state.

        Returns:
            List of operand values in the same order as the operand list.
        """
        return [float(op.value) for op in self._operands]


class Tolerancing:
    """A class representing a tolerancing problem. This class is the core of the
    tolerancing module. It allows the user to define a tolerancing problem by
    adding operands (metrics), perturbations, and compensators to an optical
    system.

    Args:
        optic: The optic object to be toleranced.
        method: The method to use for optimization. Defaults to 'generic'.
        tol: The tolerance for the compensator optimizer. Defaults to 1e-5.

    Attributes:
        optic: The optic object to be toleranced.
        method: The method to use for optimization. Defaults to 'generic'.
        tol: The tolerance for the compensator optimizer. Defaults to 1e-5.
        _optic_nominal: A deep copy of the optic object representing its
            nominal state.
        operands: A list of operands in the tolerancing problem.
        perturbations: A list of perturbations applied to the optic.
        compensator: An optimizer for compensating the perturbations.

    Methods:
        __init__(self, optic): Initializes a Tolerancing object with the given
            optic.
        add_operand(self, operand_type, input_data={}): Adds an operand to the
            tolerancing problem.
        add_perturbation(self, variable_type, perturbation, **kwargs): Adds a
            perturbation to the optic.
        add_compensator(self, variable_type, **kwargs): Adds a compensator
            variable to the optimizer.
        apply_compensators(self): Applies compensators to the optic.
        evaluate(self): Evaluates the operands in the tolerancing problem.
        reset(self): Resets the optic to its nominal state.

    """

    def __init__(self, optic, method="generic", tol=1e-5):
        self.optic = optic
        self.method = method
        self.tol = tol
        self.operands = []
        self.perturbations = []
        self.compensator = CompensatorOptimizer(method=method, tol=tol)
        self.quick_focus: QuickFocusCompensator | None = None

    def add_operand(
        self,
        operand_type: str,
        input_data: dict = None,
        target: float = None,
        weight: float = 1.0,
        min_val: float = None,
        max_val: float = None,
    ):
        """Add an operand to the tolerancing problem.

        Args:
            operand_type (str): The type of the operand.
            target (float): The target value of the operand (equality operand).
                If target and both bounds are omitted, the current operand
                value is captured as the target. When bounds are provided,
                target remains None so compensation uses the allowed range.
            min_val (float): The operand should stay above this
                value (inequality operand).
            max_val (float): The operand should stay below this
                value (inequality operand).
            weight (float): The weight of the operand.
            input_data (dict): Additional input data for the operand.

        """
        if input_data is None:
            input_data = {}
        new_operand = Operand(
            operand_type,
            target,
            min_val,
            max_val,
            weight,
            input_data,
        )
        self.operands.append(new_operand)

    def add_perturbation(self, variable_type: str, sampler: BaseSampler, **kwargs):
        """Add a perturbation to the optic.

        Args:
            variable_type: The type of the variable to be perturbed, such as
                "radius", "thickness", etc. See the variable module for more
                information.
            sampler: The sampler object used to generate perturbation values.
            **kwargs: Additional keyword arguments for the variable.

        """
        perturbation = Perturbation(self.optic, variable_type, sampler, **kwargs)
        self.perturbations.append(perturbation)

    def add_tolerance(
        self,
        tolerance_type: str,
        limit: float | None = None,
        *,
        minus: float | None = None,
        plus: float | None = None,
        distribution: str | Distribution = "normal",
        **kwargs,
    ) -> Tolerance:
        """Add a tolerance: a signed deviation from nominal within limits.

        Unlike ``add_perturbation``, which sets absolute values of an
        optimization variable, a tolerance is stated relative to the nominal
        design, e.g. ``limit=0.05`` for ±0.05. Both can be used together.

        Args:
            tolerance_type: The registered tolerance type: ``"radius"``,
                ``"fringes"``, ``"irregularity"``, ``"conic"``,
                ``"thickness"``, ``"surface_decenter"``, ``"surface_tilt"``,
                ``"element_decenter"``, ``"element_tilt"``, ``"element_roll"``,
                ``"index"`` or ``"abbe"``, or a type added to
                ``tolerance_registry``.
            limit: Symmetric limit; the tolerance spans ``[-limit, +limit]``.
            minus: Lower limit, for asymmetric tolerances. Requires ``plus``.
            plus: Upper limit, for asymmetric tolerances. Requires ``minus``.
            distribution: The distribution used by Monte Carlo analyses:
                ``"normal"`` (default, truncated with the limits at ±2σ),
                ``"uniform"``, ``"parabolic"``, ``"end_point"``, or a
                ``Distribution`` instance.
            **kwargs: Arguments for the tolerance type, such as
                ``surface_number`` or ``axis``; element types take ``first``,
                ``last`` and, for tilts and rolls, ``pivot``.

        Returns:
            The new tolerance.

        Raises:
            ValueError: If the limits are missing, given twice, or reversed,
                or if the tolerance type is unknown.
        """
        if limit is not None:
            if minus is not None or plus is not None:
                raise ValueError("Give either limit or minus and plus, not both.")
            if limit < 0:
                raise ValueError(f"limit must not be negative, got {limit}.")
            minus, plus = -limit, limit
        elif minus is None or plus is None:
            raise ValueError("Give either limit, or both minus and plus.")

        tolerance_cls = tolerance_registry.get(tolerance_type)
        tolerance = Tolerance(
            tolerance_cls(self.optic, **kwargs), minus, plus, distribution
        )
        self.perturbations.append(tolerance)
        return tolerance

    def add_default_tolerances(
        self,
        table: Mapping[str, Any],
        surfaces: Iterable[int] | None = None,
    ) -> list[Tolerance]:
        """Add the tolerances of a table to every matching part of the design.

        For example ``{"fringes": 2, "thickness": 0.05, "surface_tilt": 0.02}``
        adds ±2 fringes to every refracting surface, ±0.05 to every gap and a
        ±0.02° tilt about x and y to every surface. Optiland ships no grade
        tables; pass your shop's or vendor's values. See
        ``optiland.tolerancing.presets.add_default_tolerances`` for where each
        type is placed.

        Args:
            table: Tolerance type to size: a number (symmetric limit) or a dict
                of ``add_tolerance`` arguments.
            surfaces: Restrict to these surface numbers. Defaults to all.

        Returns:
            The added tolerances.
        """
        return add_default_tolerances(self, table, surfaces)

    def add_tolerances_from_drawing(
        self, drawing: DrawingSpec, **kwargs
    ) -> list[Tolerance]:
        """Add the tolerances stated by an ISO 10110 drawing specification.

        Delegates to
        :func:`optiland.tolerancing.iso10110_bridge.tolerances_from_drawing`,
        which documents the field mapping and unit conversions.

        Args:
            drawing: The ISO 10110 drawing specification.
            **kwargs: Options of ``tolerances_from_drawing``, such as
                ``distribution``, ``centering`` or ``strict``.

        Returns:
            The new tolerances.
        """
        from optiland.tolerancing.iso10110_bridge import tolerances_from_drawing

        return tolerances_from_drawing(self, drawing, **kwargs)

    def ordered_perturbations(self) -> list[BasePerturbation]:
        """Return the perturbations in the order they are applied.

        Perturbations are sorted by phase (structure, material, form, then
        rigid body); within a phase they keep insertion order.

        Returns:
            The perturbations in application order.
        """
        return sorted(self.perturbations, key=lambda p: p.phase)

    def add_compensator(self, variable_type: str, **kwargs):
        """Add a compensator variable to the optimizer.

        Args:
            variable_type: The type of the variable to be used for
                compensation, such as "thickness", etc. See the variable
                module for more information. ``"quick_focus"`` instead moves
                the image surface to the minimum RMS spot position on every
                trial, without an optimizer; see ``QuickFocusCompensator``
                for its keyword arguments. It runs before the optimizer
                compensators. Adding it again replaces it.
            **kwargs: Additional keyword arguments for the variable.

        """
        if variable_type == "quick_focus":
            self.quick_focus = QuickFocusCompensator(self.optic, **kwargs)
            return
        self.compensator.add_variable(self.optic, variable_type, **kwargs)

    def apply_compensators(self):
        """Apply compensators to the optic."""
        result = {}
        if self.quick_focus is not None:
            result.update(self.quick_focus.apply())

        if self.compensator.has_variables:
            # add operands to the optimization problem (for compensation)
            self.compensator.operands = self.operands

            # run optimizer for compensating the perturbations
            self.compensator.run()

            # undo scaling and record the optimized values
            for i, var in enumerate(self.compensator.variables):
                result[f"C{i}: {var!s}"] = float(var.variable.inverse_scale(var.value))

        return result

    def evaluate(self) -> list[float]:
        """Evaluate the operands."""
        return OperandEvaluator(self.operands).evaluate()

    def reset(self):
        """Reset the optic to its initial state."""
        # quick focus first: resetting a thickness moves the image surface
        if self.quick_focus is not None:
            self.quick_focus.reset()

        # reverse application order, so later phases are undone first
        for perturbation in reversed(self.ordered_perturbations()):
            perturbation.reset()

        for compensator in self.compensator.variables:
            compensator.reset()
