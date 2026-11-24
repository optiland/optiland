"""Core Tolerancing Module

This module contains the core classes for the tolerancing module. The
Tolerancing class is the main class for defining a tolerancing problem. It
allows the user to add operands, perturbations, and compensators to an optical
system.

Kramer Harrison, 2024
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optiland.optimization.operand import Operand
from optiland.tolerancing import types as _types  # noqa: F401 (registers types)
from optiland.tolerancing.compensator import CompensatorOptimizer
from optiland.tolerancing.perturbation import BaseSampler, Perturbation
from optiland.tolerancing.registry import tolerance_registry
from optiland.tolerancing.tolerance import Tolerance

if TYPE_CHECKING:
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
                ``"fringes"``, ``"conic"``, ``"thickness"``,
                ``"surface_decenter"``, ``"surface_tilt"``,
                ``"element_decenter"``, ``"element_tilt"`` or
                ``"element_roll"``, or a type added to ``tolerance_registry``.
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
                module for more information.
            **kwargs: Additional keyword arguments for the variable.

        """
        self.compensator.add_variable(self.optic, variable_type, **kwargs)

    def apply_compensators(self):
        """Apply compensators to the optic."""
        result = {}
        if self.compensator.has_variables:
            # add operands to the optimization problem (for compensation)
            self.compensator.operands = self.operands

            # run optimizer for compensating the perturbations
            self.compensator.run()

            # undo scaling and record the optimized values
            result = {}
            for i, var in enumerate(self.compensator.variables):
                result[f"C{i}: {var!s}"] = float(var.variable.inverse_scale(var.value))

        return result

    def evaluate(self) -> list[float]:
        """Evaluate the operands."""
        return OperandEvaluator(self.operands).evaluate()

    def reset(self):
        """Reset the optic to its initial state."""
        # reverse application order, so later phases are undone first
        for perturbation in reversed(self.ordered_perturbations()):
            perturbation.reset()

        for compensator in self.compensator.variables:
            compensator.reset()
