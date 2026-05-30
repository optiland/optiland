"""Operand Module

This module defines various operands to be used during lens optimization. These
include paraxial, real ray, aberrations, wavefront, spot size, and many other
operand types.

In general, to use one of these operands in optimization, you
can simply do the following:
    1. Identify the key in the METRIC_DICT variable, or add your new operand.
    2. Identify the input data dictionary that is required for the calculation
       of this operand.
    3. Add the operand to your optimization.OptimizationProblem instance using
       the add_operand method. Include the operand type, target, weight, and
       the input data (in dict format). See examples.

Kramer Harrison, 2024
"""

from __future__ import annotations

from dataclasses import dataclass

from optiland.optimization.operand.aberration import AberrationOperand
from optiland.optimization.operand.lens import LensOperand
from optiland.optimization.operand.paraxial import ParaxialOperand
from optiland.optimization.operand.ray import RayOperand

# Dictionary of operands and their associated functions
METRIC_DICT = {
    "f1": ParaxialOperand.f1,
    "f2": ParaxialOperand.f2,
    "F1": ParaxialOperand.F1,
    "F2": ParaxialOperand.F2,
    "P1": ParaxialOperand.P1,
    "P2": ParaxialOperand.P2,
    "N1": ParaxialOperand.N1,
    "N2": ParaxialOperand.N2,
    "EPD": ParaxialOperand.EPD,
    "EPL": ParaxialOperand.EPL,
    "XPD": ParaxialOperand.XPD,
    "XPL": ParaxialOperand.XPL,
    "magnification": ParaxialOperand.magnification,
    "total_track": ParaxialOperand.total_track,
    "seidel": AberrationOperand.seidels,
    "TSC": AberrationOperand.TSC,
    "SC": AberrationOperand.SC,
    "CC": AberrationOperand.CC,
    "TCC": AberrationOperand.TCC,
    "TAC": AberrationOperand.TAC,
    "AC": AberrationOperand.AC,
    "TPC": AberrationOperand.TPC,
    "PC": AberrationOperand.PC,
    "DC": AberrationOperand.DC,
    "TAchC": AberrationOperand.TAchC,
    "LchC": AberrationOperand.LchC,
    "TchC": AberrationOperand.TchC,
    "TSC_sum": AberrationOperand.TSC_sum,
    "SC_sum": AberrationOperand.SC_sum,
    "CC_sum": AberrationOperand.CC_sum,
    "TCC_sum": AberrationOperand.TCC_sum,
    "TAC_sum": AberrationOperand.TAC_sum,
    "AC_sum": AberrationOperand.AC_sum,
    "TPC_sum": AberrationOperand.TPC_sum,
    "PC_sum": AberrationOperand.PC_sum,
    "DC_sum": AberrationOperand.DC_sum,
    "TAchC_sum": AberrationOperand.TAchC_sum,
    "LchC_sum": AberrationOperand.LchC_sum,
    "TchC_sum": AberrationOperand.TchC_sum,
    "real_x_intercept": RayOperand.x_intercept,
    "real_y_intercept": RayOperand.y_intercept,
    "real_z_intercept": RayOperand.z_intercept,
    "real_x_intercept_lcs": RayOperand.x_intercept_lcs,
    "real_y_intercept_lcs": RayOperand.y_intercept_lcs,
    "real_z_intercept_lcs": RayOperand.z_intercept_lcs,
    "clearance": RayOperand.clearance,
    "real_L": RayOperand.L,
    "real_M": RayOperand.M,
    "real_N": RayOperand.N,
    "rms_spot_size": RayOperand.rms_spot_size,
    "OPD_difference": RayOperand.OPD_difference,
    "edge_thickness": LensOperand.edge_thickness,
    "AOI": RayOperand.AOI,
}


class OperandRegistry:
    """A registry to manage operand functions.
    This class allows you to register functions with specific operand names,
    retrieve them, and check if an operand name is registered.

    Attributes:
        _registry (dict): A dictionary to store operand names and their
            associated functions.

    Methods:
        register(name, func):
            Register a function with a specified operand name.
        get(name):
            Retrieve the function associated with an operand name.
        __contains__(name):
            Check if an operand name is registered.
        __repr__():
            Return a string representation of the OperandRegistry.

    """

    def __init__(self):
        self._registry = {}

    def register(self, name, func, overwrite=False):
        """Register a function with a specified operand name.

        Args:
            name (str): The name of the operand.
            func (function): The function to be registered.
            overwrite (bool): Whether to overwrite an existing registration.

        Example:
            >>> def my_operand(optic, surface_number):
            ...     return optic.surfaces[surface_number].geometry.radius
            >>> operand_registry.register("my_radius", my_operand)
            >>> # Now usable in OptimizationProblem:
            >>> # problem.add_operand("my_radius", target=10.0, input_data={...})

        """
        if name in self._registry and not overwrite:
            raise ValueError(f'Operand "{name}" is already registered.')
        self._registry[name] = func

    def get(self, name):
        """Retrieve the function associated with an operand name.

        Args:
            name (str): The name of the operand.

        """
        return self._registry.get(name)

    def __contains__(self, name):
        """Check if an operand name is registered.

        Args:
            name (str): The name of the operand.

        """
        return name in self._registry

    def __repr__(self):
        return f"OperandRegistry({list(self._registry.keys())})"


# Create the global operand registry
operand_registry = OperandRegistry()


# Add all operands to the registry
for name, func in METRIC_DICT.items():
    operand_registry.register(name, func)


@dataclass
class Operand:
    """Represents an operand used in optimization calculations.
    If no target is specified, a default is created at the current value.

    Attributes:
        operand_type (str): The type of the operand.
        target (float): The target value of the operand (equality operand).
        min_val (float): The operand should stay above this
            value (inequality operand).
        max_val (float): The operand should stay below this
            value (inequality operand).
        weight (float): The weight of the operand.
        input_data (dict): Additional input data for the operand.

    Methods:
        value(): Get the current value of the operand.
        delta_target(): Calculate the difference between the value and target.
        delta_ineq(): Calculate the difference between the value and targets.
        fun(): Calculate the objective function value.

    """

    operand_type: str = None
    target: float = None
    min_val: float = None
    max_val: float = None
    weight: float = None
    input_data: dict = None

    def __post_init__(self):
        if (
            self.min_val is not None
            and self.max_val is not None
            and self.min_val > self.max_val
        ):
            raise ValueError(
                f"{self.operand_type} operand: min_val is higher than max_val",
            )
        if self.target is not None and (
            self.min_val is not None or self.max_val is not None
        ):
            raise ValueError(
                f"{self.operand_type} operand cannot accept both"
                f" equality and inequality targets",
            )
        if all(x is None for x in [self.target, self.min_val, self.max_val]):
            # No target has been defined, default it to the current value
            self.target = self.value

    @property
    def value(self):
        """Get current value of the operand"""
        metric_function = operand_registry.get(self.operand_type)
        if metric_function:
            return metric_function(**self.input_data)
        raise ValueError(f"Unknown operand type: {self.operand_type}")

    def delta_target(self):
        """Calculate the difference between the value and target"""
        return self.value - self.target

    def delta_ineq(self):
        """Calculate the difference between the value and bounds.

        If the value is within the bound(s), then this operand simply is zero.
        Otherwise, it is the distance to the closest bound.
        """
        value = self.value  # Calculate the value only once
        lower_penalty = max(0, self.min_val - value) if self.min_val is not None else 0
        upper_penalty = max(0, value - self.max_val) if self.max_val is not None else 0
        return lower_penalty + upper_penalty

    def delta(self):
        """Calculate the difference to target"""
        if self.target is not None:
            return self.delta_target()
        if self.min_val is not None or self.max_val is not None:
            return self.delta_ineq()
        raise ValueError(f"{self.operand_type} operand cannot compute delta")

    def effective_weight(self, optic=None) -> float:
        """Return the total scaling factor this operand contributes.

        This is ``operand.weight × field_weight × wavelength_weight``,
        where field and wavelength weights come from the optic's current
        field/wavelength definitions. If the operand references a field or
        wavelength index, the corresponding weight is looked up from the
        optic. User-specified raw coordinates default to weight=1.0.

        The optic is resolved in the following order:

            1. The ``optic`` argument (if provided).
            2. ``self.input_data["optic"]`` (the standard operand convention).
            3. Falls back to field_w = wl_w = 1.0 when no optic is available.

        Args:
            optic: The optic this operand will be evaluated against.  If
                None, the optic is read from ``self.input_data``.

        Returns:
            The effective scalar multiplier for this operand's contribution.
        """
        import contextlib

        if optic is None and self.input_data:
            optic = self.input_data.get("optic")

        field_w = 1.0
        wl_w = 1.0

        if optic is not None:
            field_idx = self.input_data.get("field") if self.input_data else None
            wl_idx = self.input_data.get("wavelength") if self.input_data else None

            if field_idx is not None and isinstance(field_idx, int):
                with contextlib.suppress(IndexError):
                    field_w = optic.fields.fields[field_idx].weight

            if wl_idx is not None and isinstance(wl_idx, int):
                with contextlib.suppress(IndexError):
                    wl_w = optic.wavelengths.wavelengths[wl_idx].weight

        return self.weight * field_w * wl_w

    def fun(self):
        """Calculate the objective function value"""
        return self.weight * self.delta()

    def __str__(self):
        """Return a string representation of the operand"""
        return self.operand_type.replace("_", " ")
